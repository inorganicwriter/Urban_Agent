"""Tests for FIX-5: Predictor integration, untrained fallback, conformal intervals, and feature assembler safety."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.orchestration.supervisor import SupervisorAgent
from urban_resilience.prediction.calibration import ConformalIntervalCalibrator
from urban_resilience.prediction.features import CANONICAL_FEATURE_COLUMNS, FeatureAssembler
from urban_resilience.prediction.predictor import ResiliencePredictor
from urban_resilience.types import AgentEvidence, HazardType


@pytest.fixture(scope="module")
def city_env():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = loader.load_context()
    mob_loader = MobilityNetworkLoader(context, random_seed=42)
    mobility = mob_loader.generate_mobility_dataset(total_hours=48)
    event_agent = EventIntelligenceAgent(context)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=6,
        peak_intensity=0.8,
        total_simulation_hours=48,
    )
    return context, mobility, event_ctx


def test_feature_assembler_missing_features_behavior(city_env):
    """Verifies that FeatureAssembler raises ValueError in strict mode and flags insufficient in non-strict mode."""
    context, _, _ = city_env
    assembler = FeatureAssembler(context)

    # Incomplete evidence: missing 4 domains
    sparse_evidences = {
        "temporal": AgentEvidence(
            agent_name="temporal",
            metrics={"hourly_entropy": 0.8},
            zone_metrics={"SZ_FT_CBD": {"hourly_entropy": 0.8}},
            traces=[],
            confidence=0.9,
            qualitative_inference="Test",
        )
    }

    # 1. Non-strict mode: doesn't crash, but marks status as 'insufficient'
    df_non_strict = assembler.assemble_zone_features(sparse_evidences, strict=False)
    assert df_non_strict.attrs["status"] == "insufficient"
    assert len(df_non_strict.attrs["missing_features"]) > 0
    assert assembler.assembly_status == "insufficient"

    # 2. Strict mode: raises ValueError
    with pytest.raises(ValueError) as exc_info:
        assembler.assemble_zone_features(sparse_evidences, strict=True)
    assert "missing required features under strict mode" in str(exc_info.value)

    # 3. Default strict mode (strict=True by default) raises ValueError
    with pytest.raises(ValueError) as exc_info_def:
        assembler.assemble_zone_features(sparse_evidences)
    assert "missing required features under strict mode" in str(exc_info_def.value)


def test_predictor_with_calibrator_generates_conformal_intervals():
    """Predictor with calibrator returns valid conformal prediction intervals without hardcoding."""
    n_samples = 30
    X = pd.DataFrame(
        np.random.uniform(0.1, 0.9, (n_samples, len(CANONICAL_FEATURE_COLUMNS))),
        columns=CANONICAL_FEATURE_COLUMNS,
        index=[f"zone_{i}" for i in range(n_samples)],
    )
    y = pd.DataFrame({
        "target_maximum_activity_loss": np.random.uniform(0.1, 0.5, n_samples),
        "target_recovery_time_hours": np.random.uniform(10.0, 40.0, n_samples),
        "target_cumulative_activity_loss": np.random.uniform(1.0, 10.0, n_samples),
    }, index=X.index)

    predictor = ResiliencePredictor(model_type="hist_gbdt")
    predictor.fit(X, y)

    # 1. Without calibrator: intervals empty, uncalibrated=True
    uncalibrated_result = predictor.predict_structured(X[:5])
    assert uncalibrated_result.uncalibrated is True
    first_zone_uncal = uncalibrated_result.zone_predictions["zone_0"]
    assert first_zone_uncal.uncalibrated is True
    assert len(first_zone_uncal.intervals) == 0  # No hardcoded ±0.05!

    # 2. With calibrator: intervals populated from conformal quantiles
    calibrator = ConformalIntervalCalibrator(target_coverage=0.95)
    preds = predictor.predict(X)
    calibrator.fit_residuals(y, preds)

    calibrated_result = predictor.predict_structured(X[:5], calibrator=calibrator)
    assert calibrated_result.uncalibrated is False
    first_zone_cal = calibrated_result.zone_predictions["zone_0"]
    assert first_zone_cal.uncalibrated is False
    assert "maximum_activity_loss" in first_zone_cal.intervals
    interval = first_zone_cal.intervals["maximum_activity_loss"]
    assert interval.lower_95 <= interval.point_estimate <= interval.upper_95


def test_supervisor_untrained_fallback_manifest_recording(tmp_path: Path, city_env):
    """When trained model weights file does not exist, supervisor records untrained_fallback in manifest."""
    context, mobility, event_ctx = city_env

    # Point output_dir to empty tmp_path where outputs/models/resilience_predictor.pt does NOT exist
    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="deterministic",
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    assert manifest.predictor_status.get("model_status") == "untrained_fallback"
    assert manifest.predictor_status.get("engine") == "monte_carlo_physics"
    assert manifest.metadata.get("predictor_status", {}).get("model_status") == "untrained_fallback"
    assert pred.provenance.summary_metric_source == "monte_carlo_physics"
    assert pred.provenance.curve_source == "monte_carlo_physics"
    assert pred.provenance.curve_metric_source == "monte_carlo_physics"
    assert pred.provenance.external_temporal_metric_source == "monte_carlo_physics"
    assert pred.provenance.metric_sources["maximum_activity_loss"] == "monte_carlo_physics"
    assert pred.provenance.metric_sources["peak_loss_hour"] == "monte_carlo_physics"
    assert pred.provenance.model_status == "untrained_fallback"


def test_supervisor_dummy_or_corrupt_model_rejected(tmp_path: Path, city_env):
    """When a dummy or corrupted model file exists, supervisor rejects it and records untrained_fallback."""
    context, mobility, event_ctx = city_env

    # Create dummy model file
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fake_model = model_dir / "resilience_predictor.joblib"
    fake_model.write_text("dummy model weights")

    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="deterministic",
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    assert manifest.predictor_status.get("model_status") == "untrained_fallback"
    assert manifest.predictor_status.get("engine") == "monte_carlo_physics"
    assert "artifact_hash" in manifest.predictor_status
    assert "error" in manifest.predictor_status
    assert pred.provenance.summary_metric_source == "monte_carlo_physics"
    assert pred.provenance.model_status == "untrained_fallback"


def test_supervisor_missing_features_records_insufficient(tmp_path: Path, city_env):
    """When a fitted model exists but features are incomplete under strict mode, records insufficient_features."""
    context, mobility, event_ctx = city_env
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "resilience_predictor.joblib"

    # Train a valid predictor on synthetic sample matrix
    n_samples = 20
    X = pd.DataFrame(
        np.random.uniform(0.1, 0.9, (n_samples, len(CANONICAL_FEATURE_COLUMNS))),
        columns=CANONICAL_FEATURE_COLUMNS,
        index=[f"zone_{i}" for i in range(n_samples)],
    )
    y = pd.DataFrame({
        "target_maximum_activity_loss": np.random.uniform(0.1, 0.5, n_samples),
        "target_recovery_time_hours": np.random.uniform(10.0, 40.0, n_samples),
        "target_cumulative_activity_loss": np.random.uniform(1.0, 10.0, n_samples),
    }, index=X.index)

    predictor = ResiliencePredictor(model_type="ridge")
    predictor.fit(X, y)
    predictor.save(model_file)

    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="deterministic",
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )

    # Deliberately remove drainage_capacity on a zone to test strict mode feature check
    old_drainage = context.zones["SZ_FT_CBD"].drainage_capacity
    context.zones["SZ_FT_CBD"].drainage_capacity = None
    try:
        pred, outlook, zone_features, manifest = supervisor.run(
            event_context=event_ctx,
            save_manifest=True,
        )

        assert manifest.predictor_status.get("model_status") == "insufficient_features"
        assert manifest.predictor_status.get("engine") == "monte_carlo_physics"
        assert manifest.predictor_status.get("model_version") == "1.0.0"
        assert manifest.predictor_status.get("feature_schema_version") == "1.0.0"
        assert "artifact_hash" in manifest.predictor_status
        assert pred.provenance.summary_metric_source == "monte_carlo_physics"
        assert pred.provenance.model_status == "insufficient_features"
    finally:
        context.zones["SZ_FT_CBD"].drainage_capacity = old_drainage


def test_supervisor_fitted_artifact_with_complete_features(tmp_path: Path, city_env):
    """When a valid fitted model and complete features exist, supervisor predicts and records trained_model."""
    context, mobility, event_ctx = city_env

    # Configure drainage capacity on all zones so feature assembly succeeds under strict mode
    for z in context.zones.values():
        z.drainage_capacity = 50.0

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "resilience_predictor.joblib"

    n_samples = 20
    X = pd.DataFrame(
        np.random.uniform(0.1, 0.9, (n_samples, len(CANONICAL_FEATURE_COLUMNS))),
        columns=CANONICAL_FEATURE_COLUMNS,
        index=[f"zone_{i}" for i in range(n_samples)],
    )
    y = pd.DataFrame({
        "target_maximum_activity_loss": np.random.uniform(0.1, 0.5, n_samples),
        "target_recovery_time_hours": np.random.uniform(10.0, 40.0, n_samples),
        "target_cumulative_activity_loss": np.random.uniform(1.0, 10.0, n_samples),
    }, index=X.index)

    predictor = ResiliencePredictor(model_type="ridge")
    predictor.fit(X, y)

    calibrator = ConformalIntervalCalibrator(target_coverage=0.95)
    preds = predictor.predict(X)
    calibrator.fit_residuals(y, preds)

    predictor.save(model_file, calibrator=calibrator)

    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="deterministic",
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    assert manifest.predictor_status.get("model_status") == "trained_model"
    assert manifest.predictor_status.get("model_version") == "1.0.0"
    assert manifest.predictor_status.get("feature_schema_version") == "1.0.0"
    assert len(manifest.predictor_status.get("artifact_hash", "")) == 64
    assert manifest.predictor_status.get("calibrator") == "conformal_quantile"
    assert "city_max_loss" in manifest.predictor_status
    assert "city_recovery_time_hours" in manifest.predictor_status

    # FIX-1 Provenance assertions
    assert pred.provenance.summary_metric_source == "trained_model"
    assert pred.provenance.curve_source == "monte_carlo_physics"
    assert pred.provenance.curve_metric_source == "monte_carlo_physics"
    assert pred.provenance.external_temporal_metric_source == "monte_carlo_physics"
    assert pred.provenance.metric_sources["maximum_activity_loss"] == "trained_model"
    assert pred.provenance.metric_sources["recovery_time_hours"] == "trained_model"
    assert pred.provenance.metric_sources["cumulative_activity_loss"] == "trained_model"
    assert pred.provenance.metric_sources["peak_loss_hour"] == "monte_carlo_physics"
    assert pred.provenance.metric_sources["resilience_triangle_area"] == "monte_carlo_physics"
    assert pred.provenance.interval_source == "conformal"
    assert pred.provenance.model_status == "trained_model"
    assert pred.disruption_metrics.maximum_activity_loss == manifest.predictor_status["city_max_loss"]
    assert pred.disruption_metrics.recovery_time_hours == manifest.predictor_status["city_recovery_time_hours"]
    assert pred.disruption_metrics.cumulative_activity_loss == manifest.predictor_status["city_cumulative_loss"]
    assert manifest.metadata["prediction_provenance"]["summary_metric_source"] == "trained_model"
    assert manifest.metadata["prediction_provenance"]["metric_sources"]["maximum_activity_loss"] == "trained_model"


def test_api_responses_include_prediction_provenance():
    """Validates that API simulation responses expose prediction_provenance and predictor_status."""
    from fastapi.testclient import TestClient
    from urban_resilience.api import app

    client = TestClient(app)
    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "peak_intensity": 0.75,
        "total_hours": 36,
        "monte_carlo_samples": 5,
    }
    response = client.post("/api/v1/resilience/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_provenance" in data
    assert "predictor_status" in data
    assert "metric_sources" in data
    assert data["prediction_provenance"]["curve_source"] == "monte_carlo_physics"
    assert data["prediction_provenance"]["curve_metric_source"] == "monte_carlo_physics"
    assert data["prediction_provenance"]["external_temporal_metric_source"] == "monte_carlo_physics"
    assert data["prediction_provenance"]["metric_sources"]["peak_loss_hour"] == "monte_carlo_physics"
    assert data["prediction_provenance"]["summary_metric_source"] in ("trained_model", "monte_carlo_physics")
