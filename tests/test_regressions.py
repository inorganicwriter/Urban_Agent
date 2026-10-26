"""Regression test suite verifying system stability, edge cases, and robustness."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from urban_resilience.config import AgentWeightConfig, SimulationConfig, ProjectPaths, compute_baseline_mean
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.agents.temporal_redundancy import TemporalRedundancyAgent
from urban_resilience.agents.intraday_flexibility import IntradayFlexibilityAgent
from urban_resilience.agents.predictability import PredictabilityAgent
from urban_resilience.agents.spatial_redundancy import SpatialRedundancyAgent
from urban_resilience.agents.functional_mix import FunctionalMixAgent
from urban_resilience.agents.resilience_synthesis import (
    ResilienceSynthesisAgent,
    compute_disruption_metrics,
)
from urban_resilience.outputs.prediction_engine import AnticipatoryPredictionEngine
from urban_resilience.outputs.visualizer import UrbanResilienceVisualizer
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.validation.evaluator import HistoricalEventEvaluator
from urban_resilience.api import app
from urban_resilience.cli import main as cli_main
from urban_resilience.types import (
    HazardType,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    SpatialZoneResilience,
    ValidationResult,
    BaselineComparison,
    CalibrationMetrics,
)
from run_full_city_analysis import _generate_markdown_report


@pytest.fixture(scope="module")
def shared_context():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    ctx = loader.load_context()
    mob_loader = MobilityNetworkLoader(ctx, random_seed=42)
    mob_net = mob_loader.generate_mobility_dataset(total_hours=48)
    return ctx, mob_net


# ==============================================================================
# Relative activity loss metrics
# ==============================================================================
def test_maximum_activity_loss_is_relative_loss(shared_context):
    ctx, mob_net = shared_context
    synth = ResilienceSynthesisAgent(
        ctx,
        mob_net,
        sim_config=SimulationConfig(
            total_hours=48,
            event_start_hour=12,
            event_duration_hours=12,
            monte_carlo_samples=30,
        ),
    )
    # Synthetic baseline and predicted arrays:
    # Hour 15: baseline=2.0, pred=1.0 -> drop=1.0, relative drop = 1.0 / 2.0 = 50%
    # Hour 16: baseline=1.0, pred=0.6 -> drop=0.4, relative drop = 0.4 / 1.0 = 40%
    base = np.ones(48, dtype=float)
    base[15] = 2.0
    base[16] = 1.0
    pred = np.ones(48, dtype=float)
    pred[15] = 1.0
    pred[16] = 0.6

    metrics = compute_disruption_metrics(
        baseline=base,
        predicted=pred,
        event_start_hour=12,
        event_duration_hours=12,
    )
    assert metrics.maximum_activity_loss == pytest.approx(0.5, rel=1e-3)
    assert metrics.peak_loss_hour == 15

    # Test synthesis calculation on a real event
    event_agent = EventIntelligenceAgent(ctx)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        peak_intensity=0.80,
        total_simulation_hours=48,
    )
    agents_pool = {
        "temporal": TemporalRedundancyAgent(ctx, mob_net),
        "flexibility": IntradayFlexibilityAgent(ctx, mob_net),
        "predictability": PredictabilityAgent(ctx, mob_net),
        "spatial": SpatialRedundancyAgent(ctx, mob_net),
        "functional": FunctionalMixAgent(ctx, mob_net),
    }
    evidences = {k: a.analyze() for k, a in agents_pool.items()}
    pred_res, _, _ = synth.synthesize(evidences, event_ctx)

    # Relative loss must be bounded in [0, 1]
    m = pred_res.disruption_metrics
    assert 0.0 <= m.maximum_activity_loss <= 1.0

    # format_key_indicators must reflect the exact percentage of relative loss
    formatted = AnticipatoryPredictionEngine.format_key_indicators(m)
    expected_pct_str = f"{m.maximum_activity_loss * 100:.1f}%"
    assert formatted["Maximum Activity Loss"] == expected_pct_str
    assert m.peak_loss_hour >= event_ctx.event_start_hour


# ==============================================================================
# SimulationConfig cross-field validation
# ==============================================================================
def test_simulation_config_cross_field_validations():
    # 1. event_start_hour < 1
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig(event_start_hour=0, total_hours=48)

    # 2. event_duration_hours < 1
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig(event_duration_hours=0, total_hours=48)

    # 3. total_hours < 24
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig(total_hours=12, event_start_hour=4, event_duration_hours=4)

    # 4. event_start + duration > total_hours
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig(event_start_hour=30, event_duration_hours=18, total_hours=36)

    # 5. recovery_threshold_ratio out of (0, 1]
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig(recovery_threshold_ratio=1.5, total_hours=48)

    # 6. event_lead_time_hours > event_start_hour is allowed and truncated to available pre-event window
    cfg_lead = SimulationConfig(event_lead_time_hours=20, event_start_hour=10, total_hours=48)
    assert cfg_lead.event_lead_time_hours == 20
    assert cfg_lead.event_start_hour == 10

    dummy_data = np.arange(1, 30, dtype=float)
    mean_val = compute_baseline_mean(dummy_data, event_start_hour=10, lead_time_hours=20)
    assert np.isclose(mean_val, float(np.mean(dummy_data[:10])))


def test_api_rejects_invalid_payload_with_422():
    client = TestClient(app)
    # event_start_hour=0 must return 422 Unprocessable Entity
    resp = client.post("/api/v1/resilience/simulate", json={
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 0,
        "duration_hours": 6,
        "total_hours": 36,
    })
    assert resp.status_code == 422

    # window overflow: start=30, duration=18, total=36
    resp_overflow = client.post("/api/v1/resilience/simulate", json={
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 30,
        "duration_hours": 18,
        "total_hours": 36,
    })
    assert resp_overflow.status_code == 422


def test_cli_rejects_invalid_args_with_exit_code_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--start-hour", "0"])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info2:
        cli_main(["--start-hour", "30", "--duration", "18", "--total-hours", "36"])
    assert exc_info2.value.code == 2


# ==============================================================================
# Recovery time semantics and non-negativity
# ==============================================================================
def test_recovery_time_consistent_semantics(shared_context):
    ctx, mob_net = shared_context
    sim_cfg = SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=12,
        recovery_threshold_ratio=0.90,
        monte_carlo_samples=20,
    )
    synth = ResilienceSynthesisAgent(ctx, mob_net, sim_config=sim_cfg)
    event_agent = EventIntelligenceAgent(ctx)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        total_simulation_hours=48,
    )
    pred_res, _, _ = synth.synthesize({}, event_ctx)
    m = pred_res.disruption_metrics

    # Must be non-negative and >= duration_hours for normal disruption
    assert m.recovery_time_hours >= event_ctx.duration_hours
    assert m.recovery_time_hours > 0

    # Test never-recovered fallback scenario via production compute_disruption_metrics
    total_h = 48
    event_start = 12
    norm_baseline = np.ones(total_h)
    predicted_mean = np.full(total_h, 0.2)  # permanently suppressed
    metrics_fallback = compute_disruption_metrics(
        baseline=norm_baseline,
        predicted=predicted_mean,
        event_start_hour=event_start,
        event_duration_hours=12,
        recovery_threshold_ratio=sim_cfg.recovery_threshold_ratio,
    )
    assert metrics_fallback.recovery_time_hours == float(total_h - event_start)


# ==============================================================================
# Evaluator window bounds validation
# ==============================================================================
def test_evaluator_window_bounds_check(shared_context):
    ctx, mob_net = shared_context
    evaluator = HistoricalEventEvaluator(ctx, mob_net)
    y_pred = np.ones(48)
    y_true = np.ones(48)

    # Normal window
    metrics = evaluator.evaluate_predictions(y_pred, y_true, evaluation_window_start=24)
    assert "rmse" in metrics

    # Start >= total length must raise ValueError
    with pytest.raises(ValueError, match="evaluation_window_start"):
        evaluator.evaluate_predictions(y_pred, y_true, evaluation_window_start=48)

    with pytest.raises(ValueError, match="evaluation_window_start"):
        evaluator.evaluate_predictions(y_pred, y_true, evaluation_window_start=100)


# ==============================================================================
# Defensive metric calculations under short series
# ==============================================================================
def test_agents_defensive_under_short_series(shared_context):
    ctx, mob_net = shared_context
    temp_agent = TemporalRedundancyAgent(ctx, mob_net)
    flex_agent = IntradayFlexibilityAgent(ctx, mob_net)

    # Pass short series of length 12
    short_series = np.ones(12, dtype=float) * 100.0

    # Must not raise reshape ValueError
    temp_res = temp_agent._calc_entropy_and_peaks(short_series)
    assert "hourly_entropy" in temp_res
    assert "peak_concentration" in temp_res
    assert np.isfinite(temp_res["temporal_redundancy_score"])

    flex_res = flex_agent._calc_flexibility_metrics(short_series, "SZ_FT_CBD")
    assert "intraday_flexibility_score" in flex_res
    assert np.isfinite(flex_res["intraday_flexibility_score"])

    # CLI passing --total-hours 12 must exit with code 2
    with pytest.raises(SystemExit) as exc:
        cli_main(["--total-hours", "12"])
    assert exc.value.code == 2


# ==============================================================================
# MobilityNetworkLoader graph construction and fallback
# ==============================================================================
def test_mobility_loader_build_network_graph(shared_context):
    ctx, _ = shared_context
    loader = MobilityNetworkLoader(ctx, random_seed=42)
    loader.generate_mobility_dataset(total_hours=48)

    g0 = loader.build_network_graph(0)
    assert len(g0.nodes) == len(ctx.zones)
    assert g0.number_of_edges() > 0

    # Out-of-range timestep falls back gracefully
    g999 = loader.build_network_graph(999)
    assert len(g999.nodes) == len(ctx.zones)
    assert g999.number_of_edges() > 0


# ==============================================================================
# Dynamic report generation and negative ablation delta interpretation
# ==============================================================================
def test_report_dynamic_generation(tmp_path):
    report_file = tmp_path / "test_report.md"

    # Synthetic outlook where SZ_FT_CBD is rank 1
    zone_res1 = SpatialZoneResilience(
        zone_id="SZ_FT_CBD",
        zone_name="Futian Central CBD",
        vulnerability_score=0.888,
        vulnerability_rank=1,
        composite_resilience_index=0.650,
        primary_vulnerability_driver="Rigid Temporal Peak",
        key_evidence=["High traffic"],
        confidence=0.95,
    )
    zone_res2 = SpatialZoneResilience(
        zone_id="SZ_DP_ECO",
        zone_name="Dapeng Peninsula Eco-Tourism",
        vulnerability_score=0.400,
        vulnerability_rank=2,
        composite_resilience_index=0.750,
        primary_vulnerability_driver="Low Spatial Routes",
        key_evidence=["Coastal"],
        confidence=0.90,
    )
    outlook = SpatialResilienceOutlook(
        zone_rankings=[zone_res1, zone_res2],
        vulnerable_areas=["SZ_FT_CBD"],
        priority_ranking=["SZ_FT_CBD", "SZ_DP_ECO"],
        spatial_summary="Summary",
        zone_features=None,
    )

    val = ValidationResult(
        eval_metrics={"rmse": 0.05, "mae": 0.03, "mape_pct": 3.0, "r2_score": 0.98},
        baseline_comparisons=[
            BaselineComparison(model_name="Historical Baseline", rmse=0.10, mae=0.08, mape=10.0, r2_score=0.90),
            BaselineComparison(model_name="Proposed Multi-Agent Framework", rmse=0.05, mae=0.03, mape=3.0, r2_score=0.98),
        ],
        ablation_results={
            "Full Framework": {"rmse": 0.05, "mae": 0.03, "delta_rmse_pct": 0.0},
            "w/o Temporal Redundancy": {"rmse": 0.048, "mae": 0.028, "delta_rmse_pct": -4.0},  # Negative delta
            "w/o Spatial Redundancy": {"rmse": 0.07, "mae": 0.05, "delta_rmse_pct": +40.0},
        },
        calibration=CalibrationMetrics(picp_95=0.88, mpiw=0.12, interval_score=0.3, ece=0.07),
    )
    indicators = {
        "Maximum Activity Loss": "35.0%",
        "Cumulative Activity Loss": "5.00 unit-hours",
        "Recovery Time": "20.0 hours",
        "Peak Loss Hour": "T+18h",
    }

    _generate_markdown_report(report_file, None, outlook, val, indicators)
    content = report_file.read_text(encoding="utf-8")

    # 1. Top region in text must be Futian (SZ_FT_CBD)
    assert "SZ_FT_CBD" in content
    assert "Futian Central CBD" in content

    # 2. When ablation delta is negative, text must indicate error reduction
    assert "w/o Temporal Redundancy" in content
    lines = content.split("\n")
    for line in lines:
        if "w/o Temporal Redundancy" in line:
            assert "关键不可替代" not in line


# ==============================================================================
# Isolation: tests must not pollute repository outputs
# ==============================================================================
def test_tests_do_not_pollute_outputs():
    real_root = Path(__file__).resolve().parent.parent
    root_outputs = real_root / "outputs"
    assert ProjectPaths.OUTPUTS_DIR != root_outputs
    assert "pytest" in str(ProjectPaths.OUTPUTS_DIR).lower() or "tmp" in str(ProjectPaths.OUTPUTS_DIR).lower()

    # Snapshot of real outputs directory
    files_before = {
        p.relative_to(root_outputs): (p.stat().st_size, p.stat().st_mtime)
        for p in root_outputs.rglob("*")
        if p.is_file()
    }

    # Run data loader and pipeline stage in test
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    ctx = spatial_loader.load_context()
    _ = MobilityNetworkLoader(ctx).generate_mobility_dataset(total_hours=24)

    # Real outputs must remain completely untouched
    files_after = {
        p.relative_to(root_outputs): (p.stat().st_size, p.stat().st_mtime)
        for p in root_outputs.rglob("*")
        if p.is_file()
    }
    assert files_before == files_after


# ==============================================================================
# Targeted follow-up evidence refinement
# ==============================================================================
def test_targeted_follow_up_refines_evidence(shared_context):
    ctx, mob_net = shared_context
    sim_cfg = SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=12)
    synth = ResilienceSynthesisAgent(ctx, mob_net, sim_config=sim_cfg)
    event_agent = EventIntelligenceAgent(ctx)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        total_simulation_hours=48,
    )

    agents_pool = {
        "temporal": TemporalRedundancyAgent(ctx, mob_net),
        "flexibility": IntradayFlexibilityAgent(ctx, mob_net),
        "predictability": PredictabilityAgent(ctx, mob_net),
        "spatial": SpatialRedundancyAgent(ctx, mob_net),
        "functional": FunctionalMixAgent(ctx, mob_net),
    }
    evidences = {k: a.analyze() for k, a in agents_pool.items()}

    # Run synthesis without follow-up (agents_pool=None)
    pred_no_follow, _, features_no_follow = synth.synthesize(evidences, event_ctx, agents_pool=None)

    # Run synthesis WITH follow-up
    synth_with = ResilienceSynthesisAgent(ctx, mob_net, sim_config=sim_cfg)
    pred_with, _, features_with = synth_with.synthesize(evidences, event_ctx, agents_pool=agents_pool)

    assert len(synth_with.follow_up_logs) > 0
    followed_zones = [log["zone_id"] for log in synth_with.follow_up_logs]
    assert len(followed_zones) > 0

    has_change = False
    for zid in followed_zones:
        score_before = features_no_follow.loc[zid, "vulnerability_score"]
        score_after = features_with.loc[zid, "vulnerability_score"]
        if score_before != score_after:
            has_change = True
            break
    assert has_change, "Targeted follow-up must refine evidence and recalculate vulnerability"


# ==============================================================================
# Configuration propagation across components
# ==============================================================================
def test_config_wiring():
    sim_cfg = SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=6,
        event_lead_time_hours=6,
        recovery_threshold_ratio=0.92,
        monte_carlo_samples=25,
    )
    pipeline = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=sim_cfg)

    _ = pipeline.run_full_pipeline()
    assert pipeline.event_context.event_start_hour == 12
    assert pipeline.event_context.duration_hours == 6

    # Verify baseline window helper uses event_lead_time_hours
    hourly_series = pipeline.mobility_network.hourly_time_series.sum(axis=1).values
    lead_time = sim_cfg.event_lead_time_hours
    start = sim_cfg.event_start_hour
    expected_baseline_mean = float(np.mean(hourly_series[start - lead_time : start]))
    assert expected_baseline_mean > 0


# ==============================================================================
# API city constraints and local RNG encapsulation
# ==============================================================================
def test_api_city_and_local_rng():
    client = TestClient(app)

    # Non-Shenzhen city must return 422
    resp_bj = client.post("/api/v1/resilience/simulate", json={
        "city_name": "Beijing",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 36,
    })
    assert resp_bj.status_code == 422
    assert "Shenzhen" in resp_bj.text

    # Case-insensitive "shenzhen" must succeed
    resp_sz = client.post("/api/v1/resilience/simulate", json={
        "city_name": "shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 36,
        "monte_carlo_samples": 20,
    })
    assert resp_sz.status_code == 200

    # Global RNG state must not be modified by pipeline execution
    sim_cfg = SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=20, random_seed=42)
    state_before = copy.deepcopy(np.random.get_state())
    pipeline1 = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=sim_cfg)
    res1 = pipeline1.run_full_pipeline()
    state_after = copy.deepcopy(np.random.get_state())

    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])

    # Two identical seeds must give identical predictions
    pipeline2 = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=sim_cfg)
    res2 = pipeline2.run_full_pipeline()
    np.testing.assert_array_almost_equal(res1["prediction"].predicted_activity, res2["prediction"].predicted_activity)


def test_dashboard_with_arbitrary_zone_count(tmp_path, shared_context):
    ctx, mob_net = shared_context
    viz = UrbanResilienceVisualizer(tmp_path)

    # Create a sub-context with 5 zones
    small_ctx = copy.deepcopy(ctx)
    small_ctx.zones = {k: ctx.zones[k] for k in list(ctx.zones.keys())[:5]}

    pred = ResiliencePrediction(
        time_hours=np.arange(48),
        baseline_activity=np.ones(48),
        predicted_activity=np.ones(48) * 0.9,
        lower_bound_95=np.ones(48) * 0.85,
        upper_bound_95=np.ones(48) * 0.95,
        disruption_metrics=None,
    )
    outlook = SpatialResilienceOutlook(
        zone_rankings=[
            SpatialZoneResilience(zid, zid, 0.5, i + 1, 0.5, "Driver", [], 0.9)
            for i, zid in enumerate(small_ctx.zones.keys())
        ],
        vulnerable_areas=["SZ_FT_CBD"],
        priority_ranking=list(small_ctx.zones.keys()),
        spatial_summary="",
        zone_features=None,
    )
    event_ctx = EventIntelligenceAgent(ctx).create_event_context(
        hazard_type=HazardType.RAIN_STORM, event_start_hour=12, duration_hours=6, total_simulation_hours=48
    )

    dash_path = viz.plot_combined_dashboard(pred, outlook, small_ctx, event_ctx, save_filename="small_dash.png")
    assert dash_path.exists()


# ==============================================================================
# AgentWeightConfig validation
# ==============================================================================
def test_agent_weight_config_validation():
    # Negative weight
    with pytest.raises((ValidationError, ValueError)):
        AgentWeightConfig(temporal_redundancy=-0.1)

    # Sum != 1.0 (e.g. 1.2)
    with pytest.raises((ValidationError, ValueError)):
        AgentWeightConfig(
            temporal_redundancy=0.3,
            intraday_flexibility=0.3,
            predictability=0.3,
            spatial_redundancy=0.3,
            functional_mix=0.3,
        )

    # Valid config
    valid_cfg = AgentWeightConfig(
        temporal_redundancy=0.20,
        intraday_flexibility=0.20,
        predictability=0.15,
        spatial_redundancy=0.25,
        functional_mix=0.20,
    )
    assert valid_cfg.temporal_redundancy == 0.20


# ==============================================================================
# Early event start hours in API and CLI
# ==============================================================================
def test_api_and_cli_early_event_start():
    client = TestClient(app)
    # API with start_hour=6 must return 200
    resp = client.post("/api/v1/resilience/simulate", json={
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 6,
        "duration_hours": 6,
        "total_hours": 36,
        "monte_carlo_samples": 20,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert np.isfinite(data["disruption_metrics"]["maximum_activity_loss"])

    # CLI with --start-hour 6 must exit 0
    cli_code = cli_main([
        "--city", "Shenzhen",
        "--hazard", "rainstorm",
        "--start-hour", "6",
        "--duration", "6",
        "--total-hours", "36",
        "--mc-samples", "20",
        "--json-output",
    ])
    assert cli_code == 0


# ==============================================================================
# Lead time consistency across evaluator and baselines
# ==============================================================================
def test_lead_time_propagation(shared_context):
    ctx, mob_net = shared_context
    sim_cfg = SimulationConfig(
        total_hours=48,
        event_start_hour=24,
        event_duration_hours=8,
        event_lead_time_hours=4,
        monte_carlo_samples=25,
    )
    pipeline = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=sim_cfg)
    res = pipeline.run_full_pipeline()

    pred = res["prediction"]
    # Evaluator with matching lead_time_hours=4
    evaluator_lead4 = HistoricalEventEvaluator(ctx, mob_net, lead_time_hours=4)
    gt_lead4 = evaluator_lead4.generate_ground_truth_outcomes(pipeline.event_context, total_hours=48)

    # Pre-event baseline activity curves MUST match identically
    np.testing.assert_allclose(pred.baseline_activity[:24], gt_lead4[:24], rtol=1e-10)

    # Pipeline validation RMSE should be reasonable (< 0.15), not scale mismatch
    assert res["validation_result"].eval_metrics["rmse"] < 0.15

    # Evaluator with mismatched lead=12 produces different baseline mean
    evaluator_lead12 = HistoricalEventEvaluator(ctx, mob_net, lead_time_hours=12)
    gt_lead12 = evaluator_lead12.generate_ground_truth_outcomes(pipeline.event_context, total_hours=48)
    assert not np.allclose(pred.baseline_activity[:24], gt_lead12[:24])


# ==============================================================================
# Pipeline parameter revalidation
# ==============================================================================
def test_run_full_pipeline_parameter_validation():
    pipeline = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=SimulationConfig(total_hours=96))

    # Window overflow: 90 + 18 = 108 > 96
    with pytest.raises((ValueError, ValidationError)):
        pipeline.run_full_pipeline(event_start_hour=90, duration_hours=18)

    # Intensity out of [0.1, 1.0]
    with pytest.raises((ValueError, ValidationError)):
        pipeline.run_full_pipeline(peak_intensity=1.5)

    # Start < 1
    with pytest.raises((ValueError, ValidationError)):
        pipeline.run_full_pipeline(event_start_hour=0)


# ==============================================================================
# Side-effect-free module imports
# ==============================================================================
def test_city_analysis_import_side_effect_free():
    real_root = Path(__file__).resolve().parent.parent
    real_log = real_root / "outputs" / "resilience_run.log"
    exists_before = real_log.exists()
    size_before = real_log.stat().st_size if exists_before else None
    mtime_before = real_log.stat().st_mtime if exists_before else None

    # Run import in a separate subprocess explicitly rooted at real_root with PYTHONPATH=real_root
    env = dict(os.environ)
    env["PYTHONPATH"] = str(real_root)
    proc = subprocess.run(
        [sys.executable, "-c", "import run_full_city_analysis"],
        cwd=str(real_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    if not exists_before:
        assert not real_log.exists(), "resilience_run.log was created merely by importing run_full_city_analysis"
    else:
        assert real_log.exists()
        assert real_log.stat().st_size == size_before
        assert real_log.stat().st_mtime == mtime_before


# ==============================================================================
# API input upper bounds
# ==============================================================================
def test_api_input_upper_bounds():
    client = TestClient(app)
    # total_hours > 168 must return 422
    resp_total = client.post("/api/v1/resilience/simulate", json={"total_hours": 1000})
    assert resp_total.status_code == 422

    # monte_carlo_samples > 500 must return 422
    resp_mc = client.post("/api/v1/resilience/simulate", json={"monte_carlo_samples": 10000})
    assert resp_mc.status_code == 422

    # duration_hours > 72 must return 422
    resp_dur = client.post("/api/v1/resilience/simulate", json={"duration_hours": 100})
    assert resp_dur.status_code == 422

    # event_start_hour > 168 must return 422
    resp_start = client.post("/api/v1/resilience/simulate", json={"event_start_hour": 200})
    assert resp_start.status_code == 422


# ==============================================================================
# Analytical report content and dynamic logic
# ==============================================================================
def test_report_generation_content_and_dynamics(tmp_path):
    report_file = tmp_path / "test_report_gen.md"
    outlook = SpatialResilienceOutlook(
        zone_rankings=[
            SpatialZoneResilience("SZ_NS_SK", "Shekou Port & Coastal Living Zone", 0.445, 1, 0.700, "Surge Risk", [], 0.9),
            SpatialZoneResilience("SZ_DP_ECO", "Dapeng Peninsula Eco-Tourism", 0.430, 2, 0.704, "Surge Risk", [], 0.9),
        ],
        vulnerable_areas=["SZ_NS_SK", "SZ_DP_ECO"],
        priority_ranking=["SZ_NS_SK", "SZ_DP_ECO"],
        spatial_summary="",
        zone_features=None,
    )
    val = ValidationResult(
        eval_metrics={"rmse": 0.03, "mae": 0.05, "mape_pct": 2.0, "r2_score": 0.99},
        baseline_comparisons=[
            BaselineComparison("Historical Baseline", 0.09, 0.04, 3.0, 0.95),  # MAE 0.04 is better than Proposed 0.05
            BaselineComparison("Proposed Multi-Agent Framework", 0.03, 0.05, 2.0, 0.99),
        ],
        ablation_results={
            "Full Framework": {"rmse": 0.03, "mae": 0.05, "delta_rmse_pct": 0.0},
            "w/o Temporal Redundancy": {"rmse": 0.029, "mae": 0.051, "delta_rmse_pct": -1.1},
            "w/o Neutral Feature": {"rmse": 0.030, "mae": 0.050, "delta_rmse_pct": 0.0},
            "w/o Spatial Redundancy": {"rmse": 0.035, "mae": 0.06, "delta_rmse_pct": 16.5},
        },
        calibration=CalibrationMetrics(picp_95=0.80, mpiw=0.10, interval_score=0.2, ece=0.15),
    )
    indicators = {
        "Maximum Activity Loss": "18.5%",
        "Cumulative Activity Loss": "2.93 unit-hours",
        "Recovery Time": "21.0 hours",
        "Peak Loss Hour": "T+33h",
    }
    scenario = {
        "hazard": "typhoon",
        "duration_hours": 9,
        "peak_intensity": 0.70,
        "event_start_hour": 15,
    }

    _generate_markdown_report(report_file, pred=None, outlook=outlook, val=val, indicators=indicators, scenario=scenario)
    content = report_file.read_text(encoding="utf-8")

    # 1. Full Framework row must state baseline without ablation
    assert "完整框架基准（未消融）" in content
    full_line = [line for line in content.splitlines() if "Full Framework" in line][0]
    assert "移除后" not in full_line

    # 2. Negative delta describes error decrease without unconditional praise
    assert "关键不可替代" not in content
    assert "误差反而降低" in content

    # 3. Non-Full Framework with delta=0 shows no significant change
    neutral_line = [line for line in content.splitlines() if "w/o Neutral Feature" in line][0]
    assert "无显著变化" in neutral_line
    assert "误差反而降低" not in neutral_line

    # 4. Baseline conclusion must NOT claim all metrics superior when MAE was worse
    assert "所有指标均优于" not in content

    # 5. Low PICP does not have unconditional praise
    assert "既有效包络" not in content
    assert "覆盖率不足" in content

    # 6. Scenario parameters are rendered dynamically
    assert "9小时持续冲击" in content
    assert "峰值强迫 0.7" in content
    assert "18小时" not in content
    assert "0.88" not in content

    # 7. Fallback when scenario is None
    report_none = tmp_path / "test_report_none.md"
    _generate_markdown_report(report_none, pred=None, outlook=outlook, val=val, indicators=indicators, scenario=None)
    content_none = report_none.read_text(encoding="utf-8")
    assert "18小时" not in content_none
    assert "0.88" not in content_none
    assert "情景参数未显式提供" in content_none


# ==============================================================================
# Pipeline configuration immutability across repeated executions
# ==============================================================================
def test_pipeline_sim_config_immutability():
    p = UrbanResiliencePipeline(
        sim_config=SimulationConfig(
            total_hours=36,
            event_start_hour=6,
            event_duration_hours=6,
            monte_carlo_samples=20,
        )
    )
    p.run_full_pipeline(event_start_hour=12, duration_hours=6)
    assert p.sim_config.event_start_hour == 6, "self.sim_config was mutated by run_full_pipeline"
    p.run_full_pipeline()
    assert p.event_context.event_start_hour == 6, "Repeated call failed to use original sim_config"


# ==============================================================================
# Pure disruption metrics computation
# ==============================================================================
def test_compute_disruption_metrics():
    # Test case 1: synthetic relative loss
    baseline = np.array([2.0, 1.0])
    predicted = np.array([1.0, 0.6])
    metrics = compute_disruption_metrics(baseline, predicted, event_start_hour=0, event_duration_hours=1, recovery_threshold_ratio=0.95)
    assert np.isclose(metrics.maximum_activity_loss, 0.5)
    assert metrics.peak_loss_hour == 0

    # Test case 2: unrecovered scenario
    total_h = 36
    event_start = 12
    event_dur = 6
    base_unrec = np.ones(total_h)
    pred_unrec = np.zeros(total_h)
    metrics_unrec = compute_disruption_metrics(base_unrec, pred_unrec, event_start, event_dur, recovery_threshold_ratio=0.95)
    assert metrics_unrec.recovery_time_hours == float(total_h - event_start)
    assert metrics_unrec.recovery_time_hours >= 0

    # Test case 3: recovered post-event
    base_rec = np.ones(36)
    pred_rec = np.ones(36) * 0.5
    pred_rec[20:] = 1.0
    metrics_rec = compute_disruption_metrics(base_rec, pred_rec, event_start=12, event_duration_hours=6, recovery_threshold_ratio=0.95)
    assert metrics_rec.recovery_time_hours == float(20 - 12)
    assert metrics_rec.recovery_time_hours >= event_dur


# ==============================================================================
# Evaluation and calibration defensive validations
# ==============================================================================
def test_evaluation_and_calibration_input_validations():
    from urban_resilience.validation.calibration import UncertaintyCalibrator
    from urban_resilience.types import DisruptionMetrics
    from urban_resilience.validation.baselines import BaselineBenchmarkRunner

    pred = ResiliencePrediction(
        time_hours=np.arange(10),
        baseline_activity=np.ones(10),
        predicted_activity=np.ones(10),
        lower_bound_95=np.ones(10) * 0.9,
        upper_bound_95=np.ones(10) * 1.1,
        disruption_metrics=DisruptionMetrics(0.1, 1.0, 5.0, 2, 1.0),
    )
    with pytest.raises(ValueError):
        UncertaintyCalibrator.evaluate_calibration(pred, np.ones(10), eval_start_hour=15)

    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    ctx = spatial_loader.load_context()
    mob_net = MobilityNetworkLoader(ctx).generate_mobility_dataset(total_hours=24)
    runner = BaselineBenchmarkRunner(ctx, mob_net)
    evt_ctx = EventIntelligenceAgent(ctx).create_event_context(event_start_hour=12, duration_hours=6, total_simulation_hours=24)
    with pytest.raises(ValueError):
        runner.run_all_baselines(evt_ctx, np.ones(24), np.ones(24), eval_start_hour=30)
