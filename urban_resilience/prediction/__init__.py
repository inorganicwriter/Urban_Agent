"""Machine-learning prediction and calibration modules for urban resilience."""

from urban_resilience.prediction.calibration import CalibrationMetrics, ConformalIntervalCalibrator
from urban_resilience.prediction.features import (
    CANONICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    FeatureAssembler,
    FeatureSchemaMetadata,
)
from urban_resilience.prediction.predictor import (
    CityPredictionResult,
    PredictionInterval,
    ResiliencePredictor,
    ZonePredictionResult,
)
from urban_resilience.prediction.provenance import (
    build_provenance_warnings,
    get_provenance_warning,
    get_synthetic_warning,
    is_synthetic_trained,
    is_untrained_fallback,
    validate_verified_data_provenance,
)
from urban_resilience.prediction.synthetic_protocol import (
    SyntheticDatasetBundle,
    SyntheticDatasetSplit,
    build_synthetic_dataset,
    train_and_calibrate_synthetic_predictor,
)
from urban_resilience.prediction.train import EventLevelSplitter, evaluate_event_holdout, train_resilience_model

__all__ = [
    "CANONICAL_FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "FeatureSchemaMetadata",
    "FeatureAssembler",
    "PredictionInterval",
    "ZonePredictionResult",
    "CityPredictionResult",
    "ResiliencePredictor",
    "CalibrationMetrics",
    "ConformalIntervalCalibrator",
    "EventLevelSplitter",
    "evaluate_event_holdout",
    "train_resilience_model",
    "SyntheticDatasetSplit",
    "SyntheticDatasetBundle",
    "build_synthetic_dataset",
    "train_and_calibrate_synthetic_predictor",
    "is_synthetic_trained",
    "is_untrained_fallback",
    "build_provenance_warnings",
    "get_synthetic_warning",
    "get_provenance_warning",
    "validate_verified_data_provenance",
]
