"""Tests for prediction tabular assembler, ML predictor, calibration, and data snapshots."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from urban_resilience.data.grid_adapter import MicroGridCell, SpatialGridAdapter
from urban_resilience.data.snapshots import DataSnapshotRegistry, SnapshotMetadata
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.prediction.calibration import ConformalIntervalCalibrator
from urban_resilience.prediction.features import (
    CANONICAL_FEATURE_COLUMNS,
    FeatureAssembler,
)
from urban_resilience.prediction.predictor import ResiliencePredictor
from urban_resilience.prediction.train import EventLevelSplitter
from urban_resilience.types import AgentEvidence


@pytest.fixture(scope="module")
def urban_context():
    return UrbanSpatialDataLoader(city_name="Shenzhen").load_context()


def test_feature_assembler(urban_context):
    assembler = FeatureAssembler(urban_context)
    evidences = {
        "temporal": AgentEvidence(
            agent_name="temporal",
            metrics={"hourly_entropy": 0.8},
            zone_metrics={"SZ_FT_CBD": {"hourly_entropy": 0.8}},
            traces=[],
            confidence=0.9,
            qualitative_inference="Test temporal",
        ),
        "spatial": AgentEvidence(
            agent_name="spatial",
            metrics={"destination_diversity": 0.7},
            zone_metrics={"SZ_FT_CBD": {"destination_diversity": 0.7}},
            traces=[],
            confidence=0.9,
            qualitative_inference="Test spatial",
        ),
    }

    # Explicitly test non-strict assembly when incomplete evidences are supplied
    df = assembler.assemble_zone_features(evidences, hazard_intensity={"SZ_FT_CBD": 0.9}, strict=False)
    assert len(df) == len(urban_context.zone_ids)
    assert list(df.columns) == CANONICAL_FEATURE_COLUMNS
    assert df.loc["SZ_FT_CBD", "feat_temporal_hourly_entropy"] == 0.8
    assert df.loc["SZ_FT_CBD", "feat_context_hazard_intensity"] == 0.9

    # Default strict=True raises ValueError on incomplete evidences
    with pytest.raises(ValueError) as excinfo:
        assembler.assemble_zone_features(evidences, hazard_intensity={"SZ_FT_CBD": 0.9})
    assert "missing required features under strict mode" in str(excinfo.value)


def test_resilience_predictor_train_and_predict(tmp_path: Path):
    n_samples = 40
    data_X = {col: np.random.uniform(0.1, 0.9, size=n_samples) for col in CANONICAL_FEATURE_COLUMNS}
    X = pd.DataFrame(data_X, index=[f"sample_{i}" for i in range(n_samples)])

    y_loss = 0.3 * X["feat_context_hazard_intensity"] + 0.1 * np.random.normal(0, 0.05, size=n_samples)
    y_rec = 12.0 + 20.0 * X["feat_context_hazard_intensity"] + np.random.normal(0, 1.0, size=n_samples)
    y_cum = y_loss * y_rec
    y = pd.DataFrame({
        "target_maximum_activity_loss": np.clip(y_loss, 0.05, 0.95),
        "target_recovery_time_hours": np.clip(y_rec, 2.0, 72.0),
        "target_cumulative_activity_loss": y_cum,
    }, index=X.index)

    predictor = ResiliencePredictor(model_type="hist_gbdt")
    predictor.fit(X, y)
    assert predictor.is_fitted is True

    preds = predictor.predict(X[:5])
    assert "target_maximum_activity_loss" in preds
    assert len(preds["target_maximum_activity_loss"]) == 5

    structured = predictor.predict_structured(X[:5])
    assert structured.city_max_loss > 0.0
    assert len(structured.zone_predictions) == 5

    # Test save and load
    model_file = tmp_path / "model.joblib"
    predictor.save(model_file)
    assert model_file.exists()

    loaded = ResiliencePredictor.load(model_file)
    assert loaded.is_fitted is True
    preds_loaded = loaded.predict(X[:5])
    np.testing.assert_allclose(preds["target_maximum_activity_loss"], preds_loaded["target_maximum_activity_loss"])


def test_event_level_splitter_zero_leakage():
    df = pd.DataFrame({
        "event_id": ["E1"] * 10 + ["E2"] * 10 + ["E3"] * 10 + ["E4"] * 10,
        "feat": np.random.rand(40),
    })
    splitter = EventLevelSplitter(n_splits=3)
    for t_idx, v_idx in splitter.split(df, event_col="event_id"):
        train_events = set(df.iloc[t_idx]["event_id"])
        val_events = set(df.iloc[v_idx]["event_id"])
        assert len(train_events.intersection(val_events)) == 0


def test_conformal_interval_calibrator():
    calibrator = ConformalIntervalCalibrator(target_coverage=0.90)
    y_true = pd.DataFrame({
        "target_maximum_activity_loss": np.array([0.2, 0.3, 0.25, 0.4, 0.35]),
    })
    y_pred = {
        "target_maximum_activity_loss": np.array([0.22, 0.28, 0.24, 0.42, 0.33]),
    }

    calibrator.fit_residuals(y_true, y_pred)
    assert "target_maximum_activity_loss" in calibrator.nonconformity_quantiles

    intervals = calibrator.calibrate_intervals(y_pred)
    lower, upper = intervals["target_maximum_activity_loss"]
    assert len(lower) == 5
    assert np.all(lower <= upper)

    eval_res = calibrator.evaluate_coverage(y_true, intervals)
    assert eval_res["target_maximum_activity_loss"].empirical_coverage >= 0.80


def test_data_snapshot_integrity(tmp_path: Path):
    test_file = tmp_path / "sample_data.csv"
    test_file.write_text("zone_id,flow\nSZ_FT_CBD,100\n", encoding="utf-8")

    registry = DataSnapshotRegistry()
    checksum = registry.compute_file_checksum(test_file)

    meta = SnapshotMetadata(
        snapshot_id="snap_test_01",
        cutoff_timestamp="2023-09-07T00:00:00Z",
        sha256_checksum=checksum,
        is_verified=True,
    )
    registry.register_snapshot(meta)
    assert registry.verify_snapshot_integrity("snap_test_01", test_file) is True


def test_spatial_grid_adapter(urban_context):
    adapter = SpatialGridAdapter(urban_context)
    cells = [
        MicroGridCell(cell_id="c1", latitude=22.5, longitude=114.0, zone_id="SZ_FT_CBD", population_weight=2.0),
        MicroGridCell(cell_id="c2", latitude=22.5, longitude=114.1, zone_id="SZ_FT_CBD", population_weight=3.0),
    ]
    adapter.register_cells(cells)

    cell_df = pd.DataFrame({"c1": [10.0, 20.0], "c2": [15.0, 25.0]})
    zone_df = adapter.aggregate_cell_series_to_zones(cell_df)
    assert "SZ_FT_CBD" in zone_df.columns
    assert zone_df.loc[0, "SZ_FT_CBD"] == 25.0

    cell_vals = adapter.disaggregate_zone_metric_to_cells({"SZ_FT_CBD": 0.5})
    assert len(cell_vals) == 2
    assert "c1" in cell_vals
