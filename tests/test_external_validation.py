"""Regression and integration tests for the external empirical validation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from urban_resilience.data.external_loader import (
    ExternalDataLoader,
    ExternalEventMeta,
    ObservedEventDataset,
)
from urban_resilience.validation.external_evaluator import (
    ExternalEventEvaluator,
    SpatialMetrics,
    TemporalMetrics,
    UncertaintyMetrics,
)
from run_external_validation import validate_holdout_disjointness


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "external" / "minimal_event"


def _recompute_provenance(event_dir: Path) -> None:
    """Helper to recalculate checksums in PROVENANCE.json after intentional file changes."""
    prov_file = event_dir / "PROVENANCE.json"
    with open(prov_file, "r", encoding="utf-8") as f:
        prov = json.load(f)

    for fname in list(prov.get("checksums", {}).keys()):
        fpath = event_dir / fname
        if fpath.is_file():
            h = hashlib.sha256()
            with open(fpath, "rb") as fp:
                for chunk in iter(lambda: fp.read(65536), b""):
                    h.update(chunk)
            prov["checksums"][fname] = h.hexdigest()

    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)


def test_fixture_end_to_end_dry_run(tmp_path: Path):
    """End-to-end dry-run execution on minimal fixture generating all expected artifacts."""
    loader = ExternalDataLoader()
    dataset = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    assert dataset.meta.event_id == "SZ_20230907_MINIMAL_FIXTURE"
    assert dataset.meta.data_kind == "fixture"
    assert len(dataset.target_zones) == 12

    eval_out = tmp_path / "external_run"
    evaluator = ExternalEventEvaluator(output_dir=eval_out)
    report, paths = evaluator.evaluate_event(
        dataset=dataset,
        dry_run=True,
        monte_carlo_samples=30,
        random_seed=42,
    )

    # 1. Assert artifact existence
    assert paths["report_md"].is_file()
    assert paths["metrics_json"].is_file()
    assert paths["predictions_csv"].is_file()
    assert paths["spatial_csv"].is_file()
    assert paths["plot_temporal"].is_file()
    assert paths["plot_spatial"].is_file()

    # 2. Assert metric ranges and types
    assert isinstance(report.temporal_metrics, TemporalMetrics)
    assert report.temporal_metrics.rmse >= 0.0
    assert report.temporal_metrics.mae >= 0.0
    assert 0.0 <= report.temporal_metrics.max_activity_loss_pred <= 1.0
    assert 0.0 <= report.temporal_metrics.max_activity_loss_obs <= 1.0

    assert isinstance(report.spatial_metrics, SpatialMetrics)
    assert -1.0 <= report.spatial_metrics.spearman_rho_loss <= 1.0
    assert report.spatial_metrics.spearman_rho_impact is not None
    assert -1.0 <= report.spatial_metrics.spearman_rho_impact <= 1.0

    assert isinstance(report.uncertainty_metrics, UncertaintyMetrics)
    assert 0.0 <= report.uncertainty_metrics.picp_95 <= 1.0
    assert report.uncertainty_metrics.mpiw > 0.0
    assert report.uncertainty_metrics.winkler_score > 0.0

    assert len(report.baseline_comparisons) == 4

    # 3. Assert fixture warning banner in report
    report_text = paths["report_md"].read_text(encoding="utf-8")
    assert "FIXTURE / DRY-RUN 验证标记" in report_text
    assert "不代表真实历史观测的泛化性能" in report_text


def test_fixture_rejection_without_dry_run(tmp_path: Path):
    """Enforce data policy: fixtures cannot be evaluated as empirical data without --dry-run."""
    loader = ExternalDataLoader()
    dataset = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path)
    with pytest.raises(ValueError, match="Data policy violation.*--dry-run"):
        evaluator.evaluate_event(dataset=dataset, dry_run=False)


def test_provenance_checksum_mismatch(tmp_path: Path):
    """Detects unauthorized data corruption or mutation against PROVENANCE.json hashes."""
    corrupted_dir = tmp_path / "corrupted_event"
    shutil.copytree(FIXTURE_PATH, corrupted_dir)

    # Tamper with activity.csv without updating PROVENANCE.json
    act_file = corrupted_dir / "activity.csv"
    with open(act_file, "a", encoding="utf-8") as f:
        f.write("\n# Tampered record")

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="Checksum mismatch for 'activity.csv'"):
        loader.load_event(corrupted_dir, verify_checksums=True)


def test_timeline_gap_rejected(tmp_path: Path):
    """Rejects activity sequence with missing hourly intervals."""
    test_dir = tmp_path / "gap_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    df = pd.read_csv(test_dir / "activity.csv")
    # Drop hour 5 for zone 'SZ_FT_CBD'
    dt_drop = df[df["zone_id"] == "SZ_FT_CBD"].iloc[5]["timestamp"]
    df = df[~((df["zone_id"] == "SZ_FT_CBD") & (df["timestamp"] == dt_drop))]
    df.to_csv(test_dir / "activity.csv", index=False)
    _recompute_provenance(test_dir)

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="hourly records, expected total_hours=48"):
        loader.load_event(test_dir, verify_checksums=True)


def test_timeline_duplicate_timestamp_rejected(tmp_path: Path):
    """Rejects duplicate timestamps in activity observation."""
    test_dir = tmp_path / "dup_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    df = pd.read_csv(test_dir / "activity.csv")
    # Replace hour 6 timestamp with hour 5 timestamp
    dup_ts = df[df["zone_id"] == "SZ_FT_CBD"].iloc[5]["timestamp"]
    idx_to_replace = df[df["zone_id"] == "SZ_FT_CBD"].index[6]
    df.loc[idx_to_replace, "timestamp"] = dup_ts
    df.to_csv(test_dir / "activity.csv", index=False)
    _recompute_provenance(test_dir)

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="non-continuous or contains duplicate/gap"):
        loader.load_event(test_dir, verify_checksums=True)


def test_invalid_model_zone_rejected(tmp_path: Path):
    """Rejects zone mappings referencing undefined model zones."""
    test_dir = tmp_path / "bad_zone_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    zones_df = pd.read_csv(test_dir / "zones.csv")
    zones_df.loc[0, "model_zone_id"] = "SZ_INVALID_ZONE_999"
    zones_df.to_csv(test_dir / "zones.csv", index=False)
    _recompute_provenance(test_dir)

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="zones.csv contains unknown model zone IDs"):
        loader.load_event(test_dir, verify_checksums=True)


def test_ex_ante_causality_timing_violation(tmp_path: Path):
    """Rejects ex-ante forecast if forecast_available_time is not prior to start_time - lead_time."""
    test_dir = tmp_path / "causality_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set forecast_available_time only 2 hours before start_time when lead_time_hours is 12
    data["forecast_available_time"] = "2023-09-07T10:00:00+08:00"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="Causality violation in ex_ante_forecast"):
        loader.load_event(test_dir, verify_checksums=False)


def test_hindcast_mode_notice_in_report(tmp_path: Path):
    """Verifies that observed_hindcast mode bypasses ex-ante check and adds clear notice in report."""
    test_dir = tmp_path / "hindcast_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["input_mode"] = "observed_hindcast"
    # Even if forecast_available_time is post-event start, hindcast permits it
    data["forecast_available_time"] = "2023-09-07T14:00:00+08:00"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    loader = ExternalDataLoader()
    ds = loader.load_event(test_dir, verify_checksums=False)
    assert ds.meta.input_mode == "observed_hindcast"

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "hindcast_out")
    report, paths = evaluator.evaluate_event(ds, dry_run=True, monte_carlo_samples=20)
    report_text = paths["report_md"].read_text(encoding="utf-8")
    assert "后报验证 (HINDCAST) 标记" in report_text or "FIXTURE" in report_text


def test_holdout_disjointness_validation():
    """Asserts that test and calibration event sets must not share any event IDs."""
    base_meta = ExternalEventMeta(
        event_id="EVT_001",
        city_name="Shenzhen",
        hazard_type="rainstorm",
        start_time=datetime(2023, 9, 7, 12, 0, tzinfo=timezone.utc),
        end_time=datetime(2023, 9, 8, 0, 0, tzinfo=timezone.utc),
        forecast_available_time=datetime(2023, 9, 7, 0, 0, tzinfo=timezone.utc),
        lead_time_hours=12,
        duration_hours=12,
        total_hours=48,
        event_start_hour=12,
        peak_intensity=0.8,
        role="test",
        data_kind="fixture",
        input_mode="ex_ante_forecast",
    )

    ds_test = ObservedEventDataset(
        meta=base_meta,
        zone_mapping=pd.DataFrame(),
        hourly_zone_activity=pd.DataFrame(),
        city_total_activity=np.ones(48),
        baseline_mean=100.0,
        normalized_city_activity=np.ones(48),
        normalized_zone_activity=pd.DataFrame(),
    )

    calib_meta = base_meta.model_copy(update={"role": "calibration"})
    ds_calib = ObservedEventDataset(
        meta=calib_meta,
        zone_mapping=pd.DataFrame(),
        hourly_zone_activity=pd.DataFrame(),
        city_total_activity=np.ones(48),
        baseline_mean=100.0,
        normalized_city_activity=np.ones(48),
        normalized_zone_activity=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="Event holdout violation.*EVT_001"):
        validate_holdout_disjointness([ds_test, ds_calib])


def test_numerical_sanity_perfect_forecast(tmp_path: Path):
    """Verifies that identical predicted and observed series yield theoretical metric values."""
    evaluator = ExternalEventEvaluator(output_dir=tmp_path)
    seq = np.sin(np.linspace(0, np.pi, 48)) + 1.0

    # 1. Temporal metrics under identical inputs
    tm = evaluator._compute_temporal_metrics(
        observed=seq,
        predicted=seq,
        baseline=np.ones(48),
        eval_start_hour=12,
        duration_hours=12,
        recovery_threshold=0.95,
    )
    assert tm.rmse == pytest.approx(0.0, abs=1e-7)
    assert tm.mae == pytest.approx(0.0, abs=1e-7)
    assert tm.r2_score == pytest.approx(1.0, abs=1e-7)
    assert tm.max_activity_loss_error == pytest.approx(0.0, abs=1e-7)
    assert tm.peak_loss_hour_error == 0

    # 2. Uncertainty metrics with full coverage
    um = evaluator._compute_uncertainty_metrics(
        observed=seq,
        lower_bound=seq - 0.1,
        upper_bound=seq + 0.1,
        eval_start_hour=12,
    )
    assert um.picp_95 == pytest.approx(1.0, abs=1e-7)
    assert um.ece == pytest.approx(0.05, abs=1e-7)  # |1.0 - 0.95|


def test_network_isolation_guarantee(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Enforces zero-network footprint: external evaluation must operate completely offline."""
    def guarded_connect(*args, **kwargs):
        raise RuntimeError("Network connection prohibited in offline evaluation pipeline!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    loader = ExternalDataLoader()
    ds = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path)
    report, paths = evaluator.evaluate_event(ds, dry_run=True, monte_carlo_samples=20)
    assert paths["report_md"].is_file()
    assert report.event_id == "SZ_20230907_MINIMAL_FIXTURE"


def test_cli_subprocess_dry_run_and_rejection(tmp_path: Path):
    """Tests CLI exit codes: 2 when run without --dry-run on fixture, 0 with --dry-run."""
    root = Path(__file__).resolve().parent.parent

    # 1. Without --dry-run: exit code 2
    res_no_dry = subprocess.run(
        [
            sys.executable,
            str(root / "run_external_validation.py"),
            "--data-root",
            str(FIXTURE_PATH.parent),
            "--event-id",
            FIXTURE_PATH.name,
            "--output-dir",
            str(tmp_path / "out1"),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(root),
    )
    assert res_no_dry.returncode == 2
    output_no_dry = (res_no_dry.stdout or "") + (res_no_dry.stderr or "")
    assert "Refusing to execute fixture as real empirical validation without '--dry-run'" in output_no_dry

    # 2. With --dry-run: exit code 0
    res_dry = subprocess.run(
        [
            sys.executable,
            str(root / "run_external_validation.py"),
            "--data-root",
            str(FIXTURE_PATH.parent),
            "--event-id",
            FIXTURE_PATH.name,
            "--output-dir",
            str(tmp_path / "out2"),
            "--dry-run",
            "--monte-carlo",
            "20",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(root),
    )
    assert res_dry.returncode == 0
    assert (tmp_path / "out2" / "SZ_20230907_MINIMAL_FIXTURE" / "report.md").is_file()


def test_unverified_observed_data_rejected_without_verified_provenance(tmp_path: Path):
    """Enforce strict gate: data_kind='observed' must have provenance_status='verified' with full schema."""
    test_dir = tmp_path / "fake_observed"
    shutil.copytree(FIXTURE_PATH, test_dir)

    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["data_kind"] = "observed"
    data["provenance_status"] = "unverified"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    loader = ExternalDataLoader()
    with pytest.raises(ValueError, match="Data policy violation: Event .* is marked with data_kind='observed'"):
        loader.load_event(test_dir, verify_checksums=False)


def test_synthetic_demo_requires_allow_unverified(tmp_path: Path):
    """Enforce policy: synthetic demo requires explicit --allow-unverified or dry_run=True."""
    test_dir = tmp_path / "synthetic_demo"
    shutil.copytree(FIXTURE_PATH, test_dir)

    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["data_kind"] = "synthetic"
    data["provenance_status"] = "unverified"
    data["hazard_distribution"] = None
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    prov_path = test_dir / "PROVENANCE.json"
    with open(prov_path, "r", encoding="utf-8") as f:
        pdata = json.load(f)
    pdata["data_kind"] = "synthetic"
    pdata["provenance_status"] = "unverified"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(pdata, f, indent=2)

    _recompute_provenance(test_dir)

    loader = ExternalDataLoader()
    ds = loader.load_event(test_dir, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "demo_out")
    # Without allow_unverified or dry_run -> ValueError
    with pytest.raises(ValueError, match="Data policy violation: Event .* has data_kind='synthetic'"):
        evaluator.evaluate_event(ds, dry_run=False, allow_unverified=False)

    # With allow_unverified=True -> Succeeds
    report, paths = evaluator.evaluate_event(ds, dry_run=False, allow_unverified=True, monte_carlo_samples=10)
    assert paths["report_md"].is_file()
    assert report.verdicts is not None
    assert report.verdicts.provenance_verdict == "FAILED"


def test_time_alignment_diurnal_phase():
    """Confirms that start_datetime properly sets start_hour_of_day and aligns diurnal curves."""
    from urban_resilience.data.mobility_loader import MobilityNetworkLoader
    from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader

    urban_context = UrbanSpatialDataLoader(city_name="Shenzhen").load_context()
    m_loader = MobilityNetworkLoader(urban_context, random_seed=42)

    # Monday 00:00 start
    dt_mon_00 = datetime(2023, 9, 4, 0, 0, tzinfo=timezone.utc)
    net_00 = m_loader.generate_mobility_dataset(total_hours=24, start_datetime=dt_mon_00)
    tot_00 = net_00.hourly_time_series.sum(axis=1).values

    # Thursday 18:00 start
    dt_thu_18 = datetime(2023, 9, 7, 18, 0, tzinfo=timezone.utc)
    net_18 = m_loader.generate_mobility_dataset(total_hours=24, start_datetime=dt_thu_18)
    tot_18 = net_18.hourly_time_series.sum(axis=1).values

    # Hour 0 of net_18 is 18:00 (evening peak), while hour 0 of net_00 is 00:00 (night trough)
    assert tot_18[0] > tot_00[0]
    # Hour 6 of net_18 is 00:00 (night trough), while hour 6 of net_00 is 06:00 (morning rise)
    assert tot_18[6] < tot_18[0]


def test_baselines_use_observed_pre_event_series(tmp_path: Path):
    """Ensures Historical baseline matches pre-event observed activity and projects unshocked cycle post-event."""
    loader = ExternalDataLoader()
    ds = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "base_eval")
    report, paths = evaluator.evaluate_event(ds, dry_run=True, monte_carlo_samples=10)

    pred_df = pd.read_csv(paths["predictions_csv"])
    eval_start = ds.meta.event_start_hour

    # 1. In pre-event hours (0 to eval_start - 1), historical_baseline must equal observed_normalized_activity
    obs_pre = pred_df["observed_normalized_activity"].values[:eval_start]
    hist_pre = pred_df["historical_baseline"].values[:eval_start]
    np.testing.assert_allclose(obs_pre, hist_pre, atol=1e-4)

    # 2. Post-event: historical baseline must be a projection of unshocked pre-event cycle, NOT post-event observations
    # At hour 24 (lag = 24 - 24 = 0), baseline must equal hour 0 pre-event observation
    assert np.isclose(pred_df["historical_baseline"].values[24], obs_pre[0], atol=1e-4)
    # The actual observation at hour 24 is shocked/impacted, so baseline must distinctly differ from it
    obs_post_24 = pred_df["observed_normalized_activity"].values[24]
    assert abs(pred_df["historical_baseline"].values[24] - obs_post_24) > 0.05

    # 3. Autoregressive and Ridge baselines are derived from unshocked baseline, not directly copying post-event observations
    ar_24 = pred_df["autoregressive_pred"].values[24]
    assert np.isclose(ar_24, 0.85 * obs_pre[0], atol=1e-4)
    assert not np.isclose(ar_24, obs_post_24, atol=1e-4)

    # 4. Anti-leakage invariant: post-event historical baseline cannot be identical to post-event observations
    obs_post = pred_df["observed_normalized_activity"].values[eval_start:]
    hist_post = pred_df["historical_baseline"].values[eval_start:]
    assert not np.allclose(hist_post, obs_post, atol=1e-3)


def test_baseline_fails_if_post_event_observed_leaked():
    """Negative counter-example test: if post-event baseline naively leaks shocked observations, validation must fail."""
    leaky_df = pd.DataFrame({
        "observed_normalized_activity": [1.0, 1.0, 0.35, 0.30],
        "historical_baseline": [1.0, 1.0, 0.35, 0.30],  # Leaked post-event shocked observations!
    })
    eval_start = 2

    with pytest.raises(AssertionError, match="Baseline leakage detected"):
        obs_post = leaky_df["observed_normalized_activity"].values[eval_start:]
        hist_post = leaky_df["historical_baseline"].values[eval_start:]
        if np.allclose(hist_post, obs_post, atol=1e-3):
            raise AssertionError("Baseline leakage detected: post-event baseline matches shocked observations")


def test_impacts_csv_does_not_affect_model_predictions(tmp_path: Path):
    """Ensures impacts.csv is strictly for external evaluation and never influences predictions."""
    test_dir_1 = tmp_path / "impacts_orig"
    test_dir_2 = tmp_path / "impacts_modified"
    shutil.copytree(FIXTURE_PATH, test_dir_1)
    shutil.copytree(FIXTURE_PATH, test_dir_2)

    # In test_dir_2, alter impacts.csv radically by inverting real impact_score values
    imp_file = test_dir_2 / "impacts.csv"
    imp_df = pd.read_csv(imp_file)
    imp_df["impact_score"] = 1.0 - imp_df["impact_score"].values
    imp_df.to_csv(imp_file, index=False)
    _recompute_provenance(test_dir_2)

    loader = ExternalDataLoader()
    ds1 = loader.load_event(test_dir_1, verify_checksums=True)
    ds2 = loader.load_event(test_dir_2, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "imp_test")
    report1, paths1 = evaluator.evaluate_event(ds1, dry_run=True, monte_carlo_samples=10, random_seed=42)
    report2, paths2 = evaluator.evaluate_event(ds2, dry_run=True, monte_carlo_samples=10, random_seed=42)

    pred1 = pd.read_csv(paths1["predictions_csv"])
    pred2 = pd.read_csv(paths2["predictions_csv"])

    np.testing.assert_allclose(
        pred1["multi_agent_pred_mean"].values,
        pred2["multi_agent_pred_mean"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pred1["multi_agent_lower_95"].values,
        pred2["multi_agent_lower_95"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pred1["multi_agent_upper_95"].values,
        pred2["multi_agent_upper_95"].values,
        atol=1e-6,
    )
    assert report1.spatial_metrics.spearman_rho_loss == report2.spatial_metrics.spearman_rho_loss
    assert report1.spatial_metrics.spearman_rho_impact != report2.spatial_metrics.spearman_rho_impact


def test_unverified_hazard_distribution_discarded(tmp_path: Path):
    """Custom hazard_distribution without ex-ante forecast source and timestamp is discarded."""
    test_dir = tmp_path / "hazard_dist_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # hazard_distribution is defined, but hazard_forecast_source and hazard_forecast_available_time are None
    data["hazard_distribution"] = {"SZ_FT_CBD": 0.9, "SZ_LH_COMM": 0.8}
    data["hazard_forecast_source"] = None
    data["hazard_forecast_available_time"] = None
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    loader = ExternalDataLoader()
    ds = loader.load_event(test_dir, verify_checksums=False)

    # hazard_distribution should be discarded to avoid leakage
    assert ds.meta.hazard_distribution is None


def test_automatic_verdicts_failure_on_unverified_demo(tmp_path: Path):
    """Validates that unverified synthetic demo automatically triggers FAILED quality gate."""
    loader = ExternalDataLoader()
    ds = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "verdict_test")
    report, paths = evaluator.evaluate_event(ds, dry_run=True, monte_carlo_samples=10)

    assert report.verdicts is not None
    assert report.verdicts.provenance_verdict == "FAILED"
    assert report.verdicts.overall_verdict == "FAILED"

    report_text = paths["report_md"].read_text(encoding="utf-8")
    assert "外部验证未通过 (VALIDATION FAILED)" in report_text
    assert "质量门禁与验证判定" in report_text


def test_report_text_conditional_headers_and_provenance(tmp_path: Path):
    """Verifies unverified synthetic/fixture demo uses conditioned headings and includes provenance fields."""
    loader = ExternalDataLoader()
    ds = loader.load_event(FIXTURE_PATH, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "cond_test")
    report, paths = evaluator.evaluate_event(ds, dry_run=True, monte_carlo_samples=10)

    report_text = paths["report_md"].read_text(encoding="utf-8")

    # Synthetic / Unverified titles and banners
    assert "# 外部验证演示报告（合成/未验证数据）" in report_text
    assert "SYNTHETIC / UNVERIFIED — NOT REAL-WORLD VALIDATION" in report_text
    assert "合成目标值 (Synthetic Target)" in report_text
    assert "合成灾情指数（未验证）" in report_text
    assert "合成损失排名" in report_text

    # Provenance and coverage fields
    assert "数据来源机构 (Source)" in report_text
    assert "授权协议 (License)" in report_text
    assert "数据抓取/生成时间 (Retrieved At)" in report_text
    assert "关键文件校验和 (activity.csv SHA-256)" in report_text
    assert "数据覆盖度 (Coverage)" in report_text

    # Negative assertions: must NOT use verified observed terms
    assert "外部真实事件与移动数据事前验证报告" not in report_text
    assert "外部真实事件后报验证报告" not in report_text
    assert "实测值 (Observed)" not in report_text
    assert "实测灾情指数" not in report_text
    assert "实测损失率" not in report_text
    assert "实测损失排名" not in report_text


def test_report_text_verified_observed_headers(tmp_path: Path):
    """Verifies that when data is verified observed, report switches to observed headers and omits synthetic banner."""
    test_dir = tmp_path / "verified_event"
    shutil.copytree(FIXTURE_PATH, test_dir)

    # Modify event.json to simulate an audited observed event
    event_path = test_dir / "event.json"
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["data_kind"] = "observed"
    data["provenance_status"] = "verified"
    data["event_start_hour"] = 24
    data["start_time"] = "2023-09-08T00:00:00+08:00"
    data["end_time"] = "2023-09-08T12:00:00+08:00"
    data["forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    data["lead_time_hours"] = 24
    data["lead_time_source"] = "Shenzhen Meteorological Bureau"
    data["input_mode"] = "ex_ante_forecast"
    data["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    data["hazard_forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Modify PROVENANCE.json
    prov_path = test_dir / "PROVENANCE.json"
    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["data_kind"] = "observed"
    prov["provenance_status"] = "verified"
    prov["is_verified_observed"] = True
    prov["source_name"] = "Shenzhen Open Data Portal"
    prov["license"] = "CC-BY-4.0"
    prov["retrieved_at"] = "2023-09-08T00:00:00Z"
    prov["source_url"] = "https://opendata.sz.gov.cn/dataset/empirical_20230907"
    prov["raw_file_checksums"] = {"raw_activity.csv": "abc1234567890abcdef12345"}
    prov["aggregation_method"] = "Hourly spatial aggregation by administrative disaster management zones"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    _recompute_provenance(test_dir)

    loader = ExternalDataLoader()
    ds = loader.load_event(test_dir, verify_checksums=True)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "obs_test")
    report, paths = evaluator.evaluate_event(ds, dry_run=False, monte_carlo_samples=10)

    report_text = paths["report_md"].read_text(encoding="utf-8")

    # Must contain verified observed headings
    assert "# 外部真实事件与移动数据事前验证报告" in report_text
    assert "实测值 (Observed)" in report_text
    assert "实测灾情指数" in report_text
    assert "实测损失排名" in report_text
    assert "严格事前因果验证 (EX-ANTE VALIDATION)" in report_text

    # Negative assertions: must NOT contain unverified banners or synthetic target
    assert "SYNTHETIC / UNVERIFIED — NOT REAL-WORLD VALIDATION" not in report_text
    assert "合成目标值 (Synthetic Target)" not in report_text
    assert "合成灾情指数（未验证）" not in report_text


def test_cli_quality_gate_failed_output(tmp_path: Path):
    """Validates that CLI execution outputs QUALITY GATE FAILED and never 'completed successfully' when failed.

    Uses tmp_path as --output-dir to guarantee complete test isolation, preventing modification
    of repository outputs/external/ logs or artifacts.
    """
    repo_root = Path(__file__).resolve().parent.parent
    real_log = repo_root / "outputs" / "external" / "external_validation.log"
    log_stat_before = (real_log.stat().st_size, real_log.stat().st_mtime) if real_log.exists() else None

    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        str(repo_root / "run_external_validation.py"),
        "--event-id",
        "SZ_20230907_HAIKUI_SYNTHETIC_DEMO",
        "--allow-unverified",
        "--output-dir",
        str(out_dir),
        "--monte-carlo",
        "20",
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode == 0
    stdout_normalized = " ".join(res.stdout.split())
    assert "QUALITY GATE FAILED — see report." in stdout_normalized
    assert "completed successfully" not in res.stdout

    # Assert outputs were generated in tmp_path, not repo outputs/
    assert (out_dir / "external_validation.log").exists()
    assert (out_dir / "SZ_20230907_HAIKUI_SYNTHETIC_DEMO" / "report.md").exists()
    assert (out_dir / "SZ_20230907_HAIKUI_SYNTHETIC_DEMO" / "metrics.json").exists()

    # Real repo external_validation.log must be completely untouched
    if log_stat_before is not None:
        log_stat_after = (real_log.stat().st_size, real_log.stat().st_mtime)
        assert log_stat_before == log_stat_after, (
            f"Repository log was modified during test: before={log_stat_before}, after={log_stat_after}"
        )


def test_no_unverified_observed_datasets_in_repository():
    """Enforces repository hygiene: no dataset may claim data_kind == 'observed' without verified provenance."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "external"
    if not data_dir.is_dir():
        return

    for event_dir in data_dir.iterdir():
        if not event_dir.is_dir():
            continue
        event_file = event_dir / "event.json"
        if not event_file.is_file():
            continue
        with open(event_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        prov_file = event_dir / "PROVENANCE.json"
        assert prov_file.is_file(), f"Missing PROVENANCE.json in {event_dir}"
        with open(prov_file, "r", encoding="utf-8") as f:
            prov = json.load(f)

        if meta.get("data_kind") == "observed":
            assert meta.get("provenance_status") == "verified", (
                f"Event {event_dir.name} claims 'observed' but provenance_status is not 'verified'"
            )
            assert prov.get("is_verified_observed") is True, (
                f"Event {event_dir.name} claims 'observed' but is_verified_observed is not True"
            )


def test_readme_report_paths_exist():
    """Validates that all external validation reports referenced in README.md exist on disk and use relative paths."""
    repo_root = Path(__file__).resolve().parent.parent
    readme_path = repo_root / "README.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")

    # FIX-3: No file:/// or hardcoded Windows absolute paths
    assert "file:///" not in text, "README.md contains absolute file:/// links"
    assert "d:/" not in text.lower(), "README.md contains hardcoded D:/ drive links"

    expected_paths = [
        repo_root / "shenzhen_historical_disaster_empirical_analysis.md",
        repo_root / "outputs" / "reports" / "shenzhen_historical_disaster_empirical_analysis.md",
        repo_root / "outputs" / "external" / "SZ_20230907_HAIKUI_SYNTHETIC_DEMO" / "report.md",
        repo_root / "data" / "external" / "README.md",
    ]
    for p in expected_paths:
        assert p.is_file(), f"Referenced report file does not exist: {p}"


def test_baseline_smape_untraceable_figures_removed():
    """Ensures empirical analysis reports do not contain untraceable sMAPE figures and are synchronized."""
    repo_root = Path(__file__).resolve().parent.parent
    root_report = repo_root / "shenzhen_historical_disaster_empirical_analysis.md"
    outputs_report = repo_root / "outputs" / "reports" / "shenzhen_historical_disaster_empirical_analysis.md"

    assert root_report.is_file()
    assert outputs_report.is_file()

    content_root = root_report.read_text(encoding="utf-8")
    content_outputs = outputs_report.read_text(encoding="utf-8")

    # Must be synchronized
    assert content_root == content_outputs, "Root and outputs copies of empirical analysis report are out of sync"

    # Must NOT contain the untraceable figures
    assert "23.15%" not in content_root, "Untraceable figure 23.15% found in report"
    assert "27.60%" not in content_root, "Untraceable figure 27.60% found in report"
    assert "34.42%" not in content_root, "Untraceable figure 34.42% found in report"

    # Must NOT have sMAPE column in baseline table
    assert "| sMAPE |" not in content_root
    # But Multi-Agent's sMAPE (60.81%) is recorded in note
    assert "60.81%" in content_root


def test_hindcast_banner_conditioned_on_data_kind(tmp_path: Path):
    """Verifies that HINDCAST note is conditioned on data_kind: synthetic vs verified observed."""
    # 1. Synthetic hindcast event
    synth_dir = tmp_path / "hindcast_synth"
    shutil.copytree(FIXTURE_PATH, synth_dir)
    with open(synth_dir / "event.json", "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["input_mode"] = "observed_hindcast"
    meta_dict["data_kind"] = "synthetic"
    meta_dict["provenance_status"] = "unverified"
    with open(synth_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    with open(synth_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov_dict = json.load(f)
    prov_dict["data_kind"] = "synthetic"
    prov_dict["provenance_status"] = "unverified"
    prov_dict["is_verified_observed"] = False
    with open(synth_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov_dict, f, indent=2)

    _recompute_provenance(synth_dir)

    loader = ExternalDataLoader()
    ds_synth = loader.load_event(synth_dir, verify_checksums=True)
    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "eval_synth")
    report_synth, paths_synth = evaluator.evaluate_event(ds_synth, allow_unverified=True, monte_carlo_samples=10)

    text_synth = paths_synth["report_md"].read_text(encoding="utf-8")
    assert "合成/演示后报模式 (DEMO HINDCAST) 标记" in text_synth
    assert "本验证输入为合成事件序列，并非真实气象观测" in text_synth
    assert "事后实际降雨或气象观测数据作为强迫" not in text_synth

    # 2. Verified observed hindcast event
    obs_dir = tmp_path / "hindcast_obs"
    shutil.copytree(FIXTURE_PATH, obs_dir)
    with open(obs_dir / "event.json", "r", encoding="utf-8") as f:
        meta_obs = json.load(f)
    meta_obs["input_mode"] = "observed_hindcast"
    meta_obs["data_kind"] = "observed"
    meta_obs["provenance_status"] = "verified"
    meta_obs["event_start_hour"] = 24
    meta_obs["start_time"] = "2023-09-08T00:00:00+08:00"
    meta_obs["end_time"] = "2023-09-08T12:00:00+08:00"
    meta_obs["forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    meta_obs["lead_time_hours"] = 24
    meta_obs["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    meta_obs["hazard_forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    with open(obs_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta_obs, f, indent=2)

    with open(obs_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov_obs = json.load(f)
    prov_obs["data_kind"] = "observed"
    prov_obs["provenance_status"] = "verified"
    prov_obs["is_verified_observed"] = True
    prov_obs["source_name"] = "Shenzhen Open Data Portal"
    prov_obs["license"] = "CC-BY-4.0"
    prov_obs["retrieved_at"] = "2023-09-08T00:00:00Z"
    prov_obs["source_url"] = "https://opendata.sz.gov.cn/dataset/hindcast_test"
    prov_obs["raw_file_checksums"] = {"raw_activity.csv": "hash1234567890abcdef"}
    prov_obs["aggregation_method"] = "Hourly spatial aggregation by zones"
    with open(obs_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov_obs, f, indent=2)

    _recompute_provenance(obs_dir)

    ds_obs = loader.load_event(obs_dir, verify_checksums=True)
    evaluator_obs = ExternalEventEvaluator(output_dir=tmp_path / "eval_obs")
    report_obs, paths_obs = evaluator_obs.evaluate_event(ds_obs, dry_run=False, monte_carlo_samples=10)

    text_obs = paths_obs["report_md"].read_text(encoding="utf-8")
    assert "后报验证 (HINDCAST) 标记" in text_obs
    assert "事后实际降雨或气象观测数据作为强迫" in text_obs


def test_time_alignment_diurnal_phase_shift_degrades_correlation():
    """Asserts that shifting start_datetime by 12 hours causes significant diurnal phase mismatch."""
    from scipy.stats import pearsonr
    from urban_resilience.data.mobility_loader import MobilityNetworkLoader
    from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader

    urban_context = UrbanSpatialDataLoader(city_name="Shenzhen").load_context()
    m_loader = MobilityNetworkLoader(urban_context, random_seed=42)

    # Reference starting at 00:00 (Monday)
    dt_ref = datetime(2023, 9, 4, 0, 0, tzinfo=timezone.utc)
    net_ref = m_loader.generate_mobility_dataset(total_hours=24, start_datetime=dt_ref)
    ref_curve = net_ref.hourly_time_series.sum(axis=1).values

    # Inverted 12h phase starting at 12:00 (Monday)
    dt_shifted = datetime(2023, 9, 4, 12, 0, tzinfo=timezone.utc)
    net_shifted = m_loader.generate_mobility_dataset(total_hours=24, start_datetime=dt_shifted)
    shifted_curve = net_shifted.hourly_time_series.sum(axis=1).values

    corr, _ = pearsonr(ref_curve, shifted_curve)
    assert corr < -0.3, f"Expected 12h diurnal shift to produce negative correlation, got {corr}"


def test_observed_provenance_contract_enforcement(tmp_path: Path):
    """Enforces that data_kind='observed' strictly requires verified provenance and all contract fields."""
    loader = ExternalDataLoader()

    # Base valid verified event template
    base_dir = tmp_path / "valid_verified"
    shutil.copytree(FIXTURE_PATH, base_dir)

    with open(base_dir / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["data_kind"] = "observed"
    meta["provenance_status"] = "verified"
    meta["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    meta["hazard_forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    with open(base_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(base_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["data_kind"] = "observed"
    prov["provenance_status"] = "verified"
    prov["is_verified_observed"] = True
    prov["source_name"] = "Shenzhen Metro Open Data"
    prov["license"] = "CC-BY-4.0"
    prov["retrieved_at"] = "2023-09-08T00:00:00Z"
    prov["source_url"] = "https://opendata.sz.gov.cn/dataset/metro_ridership"
    prov["raw_file_checksums"] = {"raw_afc.csv": "abc1234567890abcdef123456"}
    prov["aggregation_method"] = "Hourly station aggregation mapped to 12 zones"
    with open(base_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    _recompute_provenance(base_dir)

    # 1. Valid verified observed event loads successfully
    ds = loader.load_event(base_dir, verify_checksums=True)
    assert ds.meta.data_kind == "observed"
    assert ds.meta.provenance_status == "verified"

    # 2. Rejection: observed with unverified provenance_status in event.json
    case2_dir = tmp_path / "unverified_observed"
    shutil.copytree(base_dir, case2_dir)
    with open(case2_dir / "event.json", "r", encoding="utf-8") as f:
        m2 = json.load(f)
    m2["provenance_status"] = "unverified"
    with open(case2_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(m2, f, indent=2)
    with pytest.raises(ValueError, match="Data policy violation.*observed.*provenance_status"):
        loader.load_event(case2_dir, verify_checksums=False)

    # 3. Rejection: verified provenance missing source_url and dataset_id
    case3_dir = tmp_path / "missing_url"
    shutil.copytree(base_dir, case3_dir)
    with open(case3_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)
    del p3["source_url"]
    with open(case3_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(p3, f, indent=2)
    _recompute_provenance(case3_dir)
    with pytest.raises(ValueError, match="Verified provenance requires a verifiable 'source_url' or 'dataset_id'"):
        loader.load_event(case3_dir, verify_checksums=True)

    # 4. Rejection: verified provenance missing raw_file_checksums
    case4_dir = tmp_path / "missing_raw_hash"
    shutil.copytree(base_dir, case4_dir)
    with open(case4_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        p4 = json.load(f)
    p4["raw_file_checksums"] = {}
    with open(case4_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(p4, f, indent=2)
    _recompute_provenance(case4_dir)
    with pytest.raises(ValueError, match="Verified provenance requires non-empty 'raw_file_checksums'"):
        loader.load_event(case4_dir, verify_checksums=True)

    # 5. Rejection: verified provenance missing aggregation_method
    case5_dir = tmp_path / "missing_agg"
    shutil.copytree(base_dir, case5_dir)
    with open(case5_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        p5 = json.load(f)
    p5["aggregation_method"] = ""
    with open(case5_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(p5, f, indent=2)
    _recompute_provenance(case5_dir)
    with pytest.raises(ValueError, match="Verified provenance requires an explicit 'aggregation_method'"):
        loader.load_event(case5_dir, verify_checksums=True)


def test_observed_mobility_adapter_anti_leakage(tmp_path: Path):
    """Verifies that ObservedMobilityAdapter enforces strict ex-ante causality and prevents data leakage.

    Asserts:
    1. Adapter pre-event network contains zero observations with timestamp >= event_start.
    2. Adapter forecast baseline repeats pre-event 24h cycle, not observed event values.
    3. Modifying event-period or recovery-period observations leaves model predictions and evidences invariant.
    4. Adapter rejects pre-event windows shorter than 24 hours.
    """
    from urban_resilience.data.external_loader import ExternalDataLoader
    from urban_resilience.data.observed_mobility import ObservedMobilityAdapter
    from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
    from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    urban_context = spatial_loader.load_context()
    loader = ExternalDataLoader()

    # Build verified observed event fixture
    base_dir = tmp_path / "verified_event"
    shutil.copytree(FIXTURE_PATH, base_dir)

    with open(base_dir / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["data_kind"] = "observed"
    meta["provenance_status"] = "verified"
    meta["event_start_hour"] = 24
    meta["total_hours"] = 48
    meta["duration_hours"] = 12
    meta["start_time"] = "2023-09-08T00:00:00+08:00"
    meta["end_time"] = "2023-09-08T12:00:00+08:00"
    meta["forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    meta["lead_time_hours"] = 24
    meta["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    meta["hazard_forecast_available_time"] = "2023-09-07T00:00:00+08:00"
    with open(base_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(base_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["data_kind"] = "observed"
    prov["provenance_status"] = "verified"
    prov["is_verified_observed"] = True
    prov["source_name"] = "Shenzhen Metro Open Data"
    prov["source_url"] = "https://opendata.sz.gov.cn/dataset/metro_ridership"
    prov["raw_file_checksums"] = {"raw_afc.parquet": "abcdef1234567890"}
    prov["aggregation_method"] = "Hourly station aggregation mapped to 12 zones"
    with open(base_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    _recompute_provenance(base_dir)

    dataset1 = loader.load_event(base_dir, verify_checksums=True)

    # 1. Adapter unit checks
    adapter = ObservedMobilityAdapter(dataset1, urban_context)
    pre_net = adapter.build_pre_event_network()
    assert len(pre_net.hourly_time_series) == 24
    # All pre-event labels are strictly earlier than start_time
    start_dt = pd.to_datetime(dataset1.meta.start_time)
    for lbl in pre_net.time_labels:
        assert pd.to_datetime(lbl) < start_dt

    ext_net = adapter.build_extended_network(total_hours=48)
    assert len(ext_net.hourly_time_series) == 48
    cols = ext_net.hourly_time_series.columns
    # For h < 24: matches pre-event observation exactly
    for h in range(24):
        np.testing.assert_allclose(
            ext_net.hourly_time_series.iloc[h].values,
            dataset1.hourly_zone_activity.loc[h, cols].values,
        )
    # For h >= 24: matches pre-event diurnal cycle (h % 24), NOT any post-event observation
    for h in range(24, 48):
        cycle_h = (h - 24) % 24
        np.testing.assert_allclose(
            ext_net.hourly_time_series.iloc[h].values,
            dataset1.hourly_zone_activity.loc[cycle_h, cols].values,
        )

    # 2. Evaluate original dataset
    evaluator1 = ExternalEventEvaluator(output_dir=tmp_path / "out1")
    rep1, art1 = evaluator1.evaluate_event(dataset1, allow_unverified=False)
    assert rep1.model_input_mobility == "observed_pre_event_extended"
    assert rep1.pre_event_window_hours == 24
    assert rep1.od_source == "synthetic_fallback"
    assert rep1.leakage_guard == "post_event_activity_excluded"
    assert rep1.predictability_source == "neutral_fallback"
    assert rep1.profile_source == "reference_default"
    assert rep1.feature_sources["predictability"] == "neutral_fallback"
    assert rep1.feature_sources["daily_profiles"] == "reference_default"

    rep1_text = (art1["report_md"]).read_text(encoding="utf-8")
    assert "可预测性特征降级告警 (Predictability Feature Degraded)" in rep1_text
    assert "当前灾前观测窗口（24 小时）不足 48 小时" in rep1_text
    assert "混杂验证说明 (Hybrid Validation Limitation)" in rep1_text
    assert "`predictability` (可预测性): `neutral_fallback`" in rep1_text
    assert "`spatial` (空间拓扑): `hybrid_observed_with_gravity_od`" in rep1_text
    assert "- **非纯观测/混杂组件 (Hybrid Components)**:" in rep1_text
    assert "`predictability`" in rep1_text
    assert "`spatial`" in rep1_text
    assert "`daily_profiles`" in rep1_text

    # Hybrid components assertions (24h has neutral predictability fallback)
    assert "predictability" in rep1.hybrid_components
    assert "spatial" in rep1.hybrid_components
    assert "functional" in rep1.hybrid_components
    assert "daily_profiles" in rep1.hybrid_components
    assert "weekly_profiles" in rep1.hybrid_components
    assert "od_flows" in rep1.hybrid_components
    assert "temporal" not in rep1.hybrid_components
    assert "flexibility" not in rep1.hybrid_components

    pred_csv1 = pd.read_csv(art1["predictions_csv"])

    # 3. Modify event and recovery period observations in a second dataset
    dir2 = tmp_path / "verified_event_modified_post"
    shutil.copytree(base_dir, dir2)

    act2_df = pd.read_csv(dir2 / "activity.csv")
    # Identify records with hour >= 24 (timestamp >= start_time - lead_time_hours + 24)
    act2_df["dt"] = pd.to_datetime(act2_df["timestamp"])
    start_time_dt = pd.to_datetime(meta["start_time"])
    post_mask = act2_df["dt"] >= start_time_dt

    # Drastically modify post-event activity (multiply by 5.0)
    act2_df.loc[post_mask, "activity_metric"] = act2_df.loc[post_mask, "activity_metric"] * 5.0
    act2_df.drop(columns=["dt"]).to_csv(dir2 / "activity.csv", index=False)
    _recompute_provenance(dir2)

    dataset2 = loader.load_event(dir2, verify_checksums=True)

    evaluator2 = ExternalEventEvaluator(output_dir=tmp_path / "out2")
    rep2, art2 = evaluator2.evaluate_event(dataset2, allow_unverified=False)

    pred_csv2 = pd.read_csv(art2["predictions_csv"])

    # 4. Strict assertions: model predictions must be 100% IDENTICAL despite post-event data change
    np.testing.assert_allclose(
        pred_csv1["multi_agent_pred_mean"].values,
        pred_csv2["multi_agent_pred_mean"].values,
        rtol=1e-6,
        err_msg="Model predictions leaked post-event observations!",
    )
    np.testing.assert_allclose(
        pred_csv1["multi_agent_lower_95"].values,
        pred_csv2["multi_agent_lower_95"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        pred_csv1["multi_agent_upper_95"].values,
        pred_csv2["multi_agent_upper_95"].values,
        rtol=1e-6,
    )

    # Manifest traces should match
    with open(art1["metrics_json"], "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(art2["metrics_json"], "r", encoding="utf-8") as f:
        m2 = json.load(f)
    assert m1["model_input_mobility"] == "observed_pre_event_extended"
    assert m2["model_input_mobility"] == "observed_pre_event_extended"
    assert m1["predictability_source"] == "neutral_fallback"
    assert m1["profile_source"] == "reference_default"
    assert m2["predictability_source"] == "neutral_fallback"
    assert m2["profile_source"] == "reference_default"

    # Evaluation metrics must change because ground truth targets changed
    assert rep1.temporal_metrics.rmse != rep2.temporal_metrics.rmse

    # 5. Assert short pre-event window rejection (< 24 hours)
    dir_short = tmp_path / "short_pre_event"
    shutil.copytree(base_dir, dir_short)
    with open(dir_short / "event.json", "r", encoding="utf-8") as f:
        m_short = json.load(f)
    m_short["event_start_hour"] = 12
    with open(dir_short / "event.json", "w", encoding="utf-8") as f:
        json.dump(m_short, f, indent=2)
    _recompute_provenance(dir_short)
    ds_short = loader.load_event(dir_short, verify_checksums=True)
    with pytest.raises(ValueError, match="Insufficient pre-event observation window.*At least 24"):
        ObservedMobilityAdapter(ds_short, urban_context)


def test_observed_mobility_adapter_predictability_sources(tmp_path: Path):
    """Verifies that ObservedMobilityAdapter accurately marks predictability_source based on window length."""
    from urban_resilience.data.external_loader import ExternalDataLoader
    from urban_resilience.data.observed_mobility import ObservedMobilityAdapter
    from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
    from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    urban_context = spatial_loader.load_context()
    loader = ExternalDataLoader()

    # Build 48h pre-event observed fixture
    dir48 = tmp_path / "event_48h"
    shutil.copytree(FIXTURE_PATH, dir48)

    # Need activity data covering 72 hours (48h pre-event + 12h duration + 12h recovery)
    base_act = pd.read_csv(FIXTURE_PATH / "activity.csv")
    zones = base_act["zone_id"].unique()
    dfs = []
    base_time = pd.to_datetime("2023-09-06T00:00:00+08:00")
    for h in range(72):
        ts = (base_time + pd.Timedelta(hours=h)).isoformat()
        cycle_h = h % 24
        sub = base_act[base_act["timestamp"].str.endswith(f"T{cycle_h:02d}:00:00+08:00")].copy() if not base_act.empty else pd.DataFrame()
        if len(sub) != len(zones):
            sub = pd.DataFrame({
                "timestamp": [ts] * len(zones),
                "zone_id": zones,
                "activity_metric": [1000.0] * len(zones),
                "metric_type": ["metro_ridership"] * len(zones),
            })
        else:
            sub["timestamp"] = ts
        dfs.append(sub)
    act48_df = pd.concat(dfs, ignore_index=True)
    act48_df.to_csv(dir48 / "activity.csv", index=False)

    with open(dir48 / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["data_kind"] = "observed"
    meta["provenance_status"] = "verified"
    meta["event_start_hour"] = 48
    meta["total_hours"] = 72
    meta["duration_hours"] = 12
    meta["start_time"] = "2023-09-08T00:00:00+08:00"
    meta["end_time"] = "2023-09-08T12:00:00+08:00"
    meta["forecast_available_time"] = "2023-09-06T00:00:00+08:00"
    meta["lead_time_hours"] = 48
    meta["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    meta["hazard_forecast_available_time"] = "2023-09-06T00:00:00+08:00"
    with open(dir48 / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(dir48 / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["data_kind"] = "observed"
    prov["provenance_status"] = "verified"
    prov["is_verified_observed"] = True
    prov["source_name"] = "Shenzhen Metro Open Data"
    prov["source_url"] = "https://opendata.sz.gov.cn/dataset/metro_ridership"
    prov["raw_file_checksums"] = {"raw_afc.parquet": "hash1234567890"}
    prov["aggregation_method"] = "Hourly station aggregation mapped to 12 zones"
    with open(dir48 / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    _recompute_provenance(dir48)
    ds48 = loader.load_event(dir48, verify_checksums=True)

    adapter48 = ObservedMobilityAdapter(ds48, urban_context)
    assert adapter48.event_start_hour == 48
    assert adapter48.predictability_source == "observed_pre_event"
    assert adapter48.profile_source == "reference_default"
    assert adapter48.feature_sources["predictability"] == "observed_pre_event"

    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "out48")
    rep48, art48 = evaluator.evaluate_event(ds48, allow_unverified=False, monte_carlo_samples=10)
    assert rep48.predictability_source == "observed_pre_event"
    assert rep48.profile_source == "reference_default"
    with open(art48["metrics_json"], "r", encoding="utf-8") as f:
        m48 = json.load(f)
    assert m48["predictability_source"] == "observed_pre_event"
    assert m48["profile_source"] == "reference_default"

    # Under 48h pre-event, predictability is pure observed, but spatial/OD/profiles remain hybrid
    assert "predictability" not in rep48.hybrid_components
    assert "spatial" in rep48.hybrid_components
    assert "functional" in rep48.hybrid_components
    assert "daily_profiles" in rep48.hybrid_components
    assert "weekly_profiles" in rep48.hybrid_components
    assert "od_flows" in rep48.hybrid_components

    rep48_text = (art48["report_md"]).read_text(encoding="utf-8")
    assert "可预测性特征降级告警" not in rep48_text
    assert "混杂验证说明 (Hybrid Validation Limitation)" in rep48_text
    assert "`predictability` (可预测性): `observed_pre_event`" in rep48_text
    assert "`spatial` (空间拓扑): `hybrid_observed_with_gravity_od`" in rep48_text
    assert "- **非纯观测/混杂组件 (Hybrid Components)**:" in rep48_text
    assert "`predictability`" not in rep48_text.split("非纯观测/混杂组件 (Hybrid Components)")[1].split("- **OD 流量来源**")[0]
    assert "`spatial`" in rep48_text.split("非纯观测/混杂组件 (Hybrid Components)")[1].split("- **OD 流量来源**")[0]


def test_verified_observed_loader_48h_pre_event_pass_path(tmp_path: Path):
    """Validates the standard pass path for ExternalDataLoader with verified observed data and >=48h pre-event window."""
    from urban_resilience.data.external_loader import ExternalDataLoader
    from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

    event_dir = tmp_path / "verified_48h_event"
    shutil.copytree(FIXTURE_PATH, event_dir)

    # 1. Construct 72h continuous hourly activity data with 48h pre-event
    base_act = pd.read_csv(FIXTURE_PATH / "activity.csv")
    zones = base_act["zone_id"].unique()
    dfs = []
    base_time = pd.to_datetime("2023-09-06T00:00:00+08:00")
    for h in range(72):
        ts = (base_time + pd.Timedelta(hours=h)).isoformat()
        cycle_h = h % 24
        sub = base_act[base_act["timestamp"].str.endswith(f"T{cycle_h:02d}:00:00+08:00")].copy() if not base_act.empty else pd.DataFrame()
        if len(sub) != len(zones):
            sub = pd.DataFrame({
                "timestamp": [ts] * len(zones),
                "zone_id": zones,
                "activity_metric": [1200.0] * len(zones),
                "metric_type": ["metro_ridership"] * len(zones),
            })
        else:
            sub["timestamp"] = ts
        dfs.append(sub)
    act_df = pd.concat(dfs, ignore_index=True)
    act_df.to_csv(event_dir / "activity.csv", index=False)

    # 2. Construct verified observed event.json with 48h pre-event and causal forecast available time
    with open(event_dir / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["data_kind"] = "observed"
    meta["provenance_status"] = "verified"
    meta["event_start_hour"] = 48
    meta["total_hours"] = 72
    meta["duration_hours"] = 12
    meta["start_time"] = "2023-09-08T00:00:00+08:00"
    meta["end_time"] = "2023-09-08T12:00:00+08:00"
    meta["forecast_available_time"] = "2023-09-06T00:00:00+08:00"
    meta["lead_time_hours"] = 48
    meta["lead_time_source"] = "Shenzhen Meteorological Bureau"
    meta["input_mode"] = "ex_ante_forecast"
    meta["hazard_forecast_source"] = "Shenzhen Meteorological Bureau"
    meta["hazard_forecast_available_time"] = "2023-09-06T00:00:00+08:00"
    with open(event_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Construct compliant PROVENANCE.json with raw_file_checksums
    with open(event_dir / "PROVENANCE.json", "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["data_kind"] = "observed"
    prov["provenance_status"] = "verified"
    prov["is_verified_observed"] = True
    prov["source_name"] = "Shenzhen Open Data Portal / Metro Ridership"
    prov["source_url"] = "https://opendata.sz.gov.cn/dataset/sz_metro_20230907"
    prov["license"] = "CC-BY-4.0"
    prov["retrieved_at"] = "2023-09-10T12:00:00Z"
    prov["raw_file_checksums"] = {"raw_afc_202309.parquet": "9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c"}
    prov["aggregation_method"] = "Hourly station exit aggregation weighted to model resilience zones"
    with open(event_dir / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    _recompute_provenance(event_dir)

    # 4. ExternalDataLoader must load without error
    loader = ExternalDataLoader()
    ds = loader.load_event(event_dir, verify_checksums=True)
    assert ds.meta.data_kind == "observed"
    assert ds.meta.provenance_status == "verified"
    assert ds.meta.event_start_hour == 48
    assert ds.provenance.get("provenance_status") == "verified"

    # 5. ExternalEventEvaluator evaluates successfully without requiring allow_unverified
    evaluator = ExternalEventEvaluator(output_dir=tmp_path / "eval_out")
    report, artifacts = evaluator.evaluate_event(ds, allow_unverified=False, dry_run=False, monte_carlo_samples=10)
    assert report.data_kind == "observed"
    assert report.model_input_mobility == "observed_pre_event_extended"
    assert report.predictability_source == "observed_pre_event"
    assert report.verdicts.provenance_verdict == "PASS"
    assert artifacts["report_md"].exists()
    assert artifacts["metrics_json"].exists()







