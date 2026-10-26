"""Tests for FIX-3: Synthetic train/calibration/test protocol with strict event-level separation."""

from pathlib import Path
import pytest

from urban_resilience.config import SimulationConfig
from urban_resilience.experiments.runner import run_four_group_experiments
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.prediction.synthetic_protocol import (
    build_synthetic_dataset,
    train_and_calibrate_synthetic_predictor,
)


def test_synthetic_protocol_disjointness_validation(tmp_path: Path):
    """Verifies that overlapping seeds between train, calibration, and test splits are rejected."""
    # 1. Overlap between train and calibration
    with pytest.raises(ValueError) as excinfo1:
        build_synthetic_dataset(
            train_seeds=[101, 102],
            calibration_seeds=[102, 103],
            test_seeds=[104, 105],
        )
    assert "overlap" in str(excinfo1.value).lower()

    # 2. Overlap between train and test
    with pytest.raises(ValueError) as excinfo2:
        build_synthetic_dataset(
            train_seeds=[101, 102],
            calibration_seeds=[103, 104],
            test_seeds=[101, 105],
        )
    assert "overlap" in str(excinfo2.value).lower()

    # 3. Overlap between calibration and test
    with pytest.raises(ValueError) as excinfo3:
        build_synthetic_dataset(
            train_seeds=[101, 102],
            calibration_seeds=[103, 104],
            test_seeds=[104, 105],
        )
    assert "overlap" in str(excinfo3.value).lower()


def test_synthetic_dataset_builder_and_training_pipeline(tmp_path: Path):
    """Verifies end-to-end dataset building, disjoint event splitting, training, conformal calibration, and holdout test."""
    cfg = SimulationConfig(
        total_hours=24,
        event_start_hour=6,
        event_duration_hours=4,
        monte_carlo_samples=3,
    )

    train_seeds = [201, 202]
    cal_seeds = [203]
    test_seeds = [204]

    # Build disjoint synthetic dataset
    bundle = build_synthetic_dataset(
        train_seeds=train_seeds,
        calibration_seeds=cal_seeds,
        test_seeds=test_seeds,
        sim_config=cfg,
        engineering_only=True,
    )

    # 1. Verify strict event-level disjointness
    bundle.verify_disjointness()

    train_evts = set(bundle.train_split.event_ids)
    cal_evts = set(bundle.calibration_split.event_ids)
    test_evts = set(bundle.test_split.event_ids)

    assert len(train_evts) == 2
    assert len(cal_evts) == 1
    assert len(test_evts) == 1

    assert train_evts.isdisjoint(cal_evts)
    assert train_evts.isdisjoint(test_evts)
    assert cal_evts.isdisjoint(test_evts)

    # 2. Verify sample counts (12 zones per event)
    assert len(bundle.train_split.X) == 24
    assert len(bundle.calibration_split.X) == 12
    assert len(bundle.test_split.X) == 12

    # 3. Verify label source and scientific tagging
    assert bundle.train_split.label_source == "synthetic_simulator"
    assert bundle.calibration_split.label_source == "synthetic_simulator"
    assert bundle.test_split.label_source == "synthetic_simulator"
    assert bundle.engineering_only is True

    # 4. Train predictor on train split only and calibrate on calibration split only
    model_artifact = tmp_path / "models" / "synthetic_predictor.joblib"
    predictor, calibrator, test_metrics = train_and_calibrate_synthetic_predictor(
        dataset_bundle=bundle,
        model_type="ridge",
        output_path=model_artifact,
        target_coverage=0.95,
    )

    assert model_artifact.exists()
    assert predictor.is_fitted is True
    assert len(calibrator.nonconformity_quantiles) > 0
    assert test_metrics["engineering_only"] is True
    assert test_metrics["label_source"] == "synthetic_simulator"
    assert "target_maximum_activity_loss_rmse" in test_metrics
    assert predictor.training_metadata.get("label_source") == "synthetic_simulator"
    assert predictor.training_metadata.get("engineering_only") is True
    assert predictor.training_metadata.get("training_event_ids") == bundle.train_split.event_ids

    # Verify loaded predictor preserves training metadata
    from urban_resilience.prediction.predictor import ResiliencePredictor
    loaded = ResiliencePredictor.load(model_artifact)
    assert loaded.training_metadata.get("label_source") == "synthetic_simulator"
    assert loaded.training_metadata.get("engineering_only") is True

    # 5. Use the fitted predictor/calibrator across four-group experiments on test seed
    mock_llm = MockLLMClient(auto_fallback=True)
    suite = run_four_group_experiments(
        event_seeds=test_seeds,
        llm_client=mock_llm,
        model_artifact_path=model_artifact,
        output_dir=tmp_path / "experiments_with_predictor",
        sim_config=cfg,
        engineering_only=True,
    )

    assert len(suite.summary_df) == 4
    assert suite.engineering_only is True

    # Verify that all 4 groups used the trained predictor artifact
    for g, res in suite.results.items():
        assert res.details.get("predictor_status", {}).get("model_status") == "trained_model"
        assert res.details.get("predictor_status", {}).get("training_label_source") == "synthetic_simulator"
        assert res.details.get("predictor_status", {}).get("engineering_only") is True
        assert res.engineering_only is True
