"""Multi-target ML predictor for urban disaster disruption and recovery curves.

Wraps Scikit-Learn gradient boosting / ridge estimators to predict maximum activity loss,
recovery time hours, and cumulative disruption from tabular agent feature matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from urban_resilience.prediction.features import CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS

logger = logging.getLogger(__name__)


class PredictionInterval(BaseModel):
    """Lower and upper bounds for an interval prediction."""

    lower_95: float
    point_estimate: float
    upper_95: float


class ZonePredictionResult(BaseModel):
    """Prediction outputs for a single urban zone."""

    zone_id: str
    maximum_activity_loss: float
    recovery_time_hours: float
    cumulative_activity_loss: float
    intervals: Dict[str, PredictionInterval] = Field(default_factory=dict)
    uncalibrated: bool = False


class CityPredictionResult(BaseModel):
    """Aggregated city-wide prediction results."""

    model_name: str
    model_version: str
    zone_predictions: Dict[str, ZonePredictionResult]
    city_max_loss: float
    city_recovery_time_hours: float
    city_cumulative_loss: float
    uncalibrated: bool = False


class ResiliencePredictor:
    """Predictor estimating disruption and recovery metrics from feature matrices."""

    def __init__(
        self,
        model_type: str = "hist_gbdt",
        model_version: str = "1.0.0",
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.model_version = model_version
        self.random_state = random_state
        self.is_fitted = False
        self.models: Dict[str, Any] = {}
        self.training_metadata: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self) -> None:
        """Initializes separate estimators for each target metric."""
        for target in TARGET_COLUMNS:
            if self.model_type == "hist_gbdt":
                self.models[target] = HistGradientBoostingRegressor(
                    max_iter=100,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    random_state=self.random_state,
                )
            elif self.model_type == "random_forest":
                self.models[target] = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                )
            elif self.model_type == "ridge":
                self.models[target] = Ridge(alpha=1.0)
            else:
                raise ValueError(f"Unsupported model_type: '{self.model_type}'")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> ResiliencePredictor:
        """Fits the predictor on feature matrix X and multi-target DataFrame y.

        Args:
            X: DataFrame with columns matching CANONICAL_FEATURE_COLUMNS.
            y: DataFrame with columns matching TARGET_COLUMNS.
        """
        # Validate features
        missing_cols = [c for c in CANONICAL_FEATURE_COLUMNS if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Feature matrix is missing required columns: {missing_cols}")

        X_mat = X[CANONICAL_FEATURE_COLUMNS].values
        for target in TARGET_COLUMNS:
            if target not in y.columns:
                raise ValueError(f"Target DataFrame missing target column: '{target}'")
            y_vec = y[target].values
            self.models[target].fit(X_mat, y_vec)

        self.is_fitted = True
        logger.info("ResiliencePredictor (%s v%s) fitted successfully on %d samples.", self.model_type, self.model_version, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generates point predictions for all targets.

        Returns:
            Dictionary mapping target name to 1D numpy array of predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("ResiliencePredictor must be fitted before predict().")

        missing_cols = [c for c in CANONICAL_FEATURE_COLUMNS if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Feature matrix is missing required columns: {missing_cols}")

        X_mat = X[CANONICAL_FEATURE_COLUMNS].values
        predictions = {}
        for target in TARGET_COLUMNS:
            preds = self.models[target].predict(X_mat)
            if "loss" in target:
                preds = np.clip(preds, 0.0, 1.0)
            elif "recovery" in target:
                preds = np.clip(preds, 1.0, 168.0)
            predictions[target] = preds

        return predictions

    def predict_structured(
        self,
        X: pd.DataFrame,
        calibrator: Optional[Any] = None,
    ) -> CityPredictionResult:
        """Generates structured CityPredictionResult from input feature matrix."""
        preds = self.predict(X)
        zone_ids = list(X.index)

        zone_preds: Dict[str, ZonePredictionResult] = {}
        max_losses = preds["target_maximum_activity_loss"]
        rec_times = preds["target_recovery_time_hours"]
        cum_losses = preds["target_cumulative_activity_loss"]

        calibrated_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        has_calibrator = calibrator is not None and bool(getattr(calibrator, "nonconformity_quantiles", None))
        if has_calibrator:
            calibrated_intervals = calibrator.calibrate_intervals(preds)

        for i, zid in enumerate(zone_ids):
            loss = float(max_losses[i])
            rec = float(rec_times[i])
            cum = float(cum_losses[i])

            intervals: Dict[str, PredictionInterval] = {}
            if has_calibrator:
                if "target_maximum_activity_loss" in calibrated_intervals:
                    intervals["maximum_activity_loss"] = PredictionInterval(
                        lower_95=float(calibrated_intervals["target_maximum_activity_loss"][0][i]),
                        point_estimate=loss,
                        upper_95=float(calibrated_intervals["target_maximum_activity_loss"][1][i]),
                    )
                if "target_recovery_time_hours" in calibrated_intervals:
                    intervals["recovery_time_hours"] = PredictionInterval(
                        lower_95=float(calibrated_intervals["target_recovery_time_hours"][0][i]),
                        point_estimate=rec,
                        upper_95=float(calibrated_intervals["target_recovery_time_hours"][1][i]),
                    )

            zone_preds[zid] = ZonePredictionResult(
                zone_id=str(zid),
                maximum_activity_loss=loss,
                recovery_time_hours=rec,
                cumulative_activity_loss=cum,
                intervals=intervals,
                uncalibrated=not has_calibrator,
            )

        return CityPredictionResult(
            model_name=f"ResiliencePredictor_{self.model_type}",
            model_version=self.model_version,
            zone_predictions=zone_preds,
            city_max_loss=float(np.mean(max_losses)),
            city_recovery_time_hours=float(np.max(rec_times)),
            city_cumulative_loss=float(np.sum(cum_losses)),
            uncalibrated=not has_calibrator,
        )

    def save(
        self,
        filepath: Union[str, Path],
        calibrator: Optional[Any] = None,
        feature_schema_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serializes predictor weights, optional calibrator, schema metadata, and training provenance to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(metadata) if metadata is not None else dict(self.training_metadata)
        self.training_metadata = meta
        joblib.dump(
            {
                "model_type": self.model_type,
                "model_version": self.model_version,
                "feature_schema_version": feature_schema_version,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted,
                "models": self.models,
                "calibrator": calibrator,
                "training_metadata": meta,
            },
            path,
        )
        logger.info("ResiliencePredictor saved to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ResiliencePredictor:
        """Deserializes predictor from disk with strict schema, version, and fitted checks."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        try:
            state = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"Failed to deserialize model artifact at {path}: {exc}") from exc

        if not isinstance(state, dict):
            raise ValueError(f"Model artifact at {path} is not a valid dictionary structure.")

        required_keys = {"model_type", "model_version", "is_fitted", "models"}
        missing = required_keys - set(state.keys())
        if missing:
            raise ValueError(f"Model artifact missing required keys: {sorted(list(missing))}")

        if not state.get("is_fitted", False):
            raise ValueError(f"Model artifact at {path} is not fitted.")

        models = state.get("models", {})
        if not isinstance(models, dict):
            raise ValueError("Model artifact 'models' field is not a dictionary.")

        for target in TARGET_COLUMNS:
            if target not in models:
                raise ValueError(f"Model artifact missing estimator for target '{target}'.")
            est = models[target]
            if not hasattr(est, "predict"):
                raise ValueError(f"Estimator for target '{target}' does not implement predict().")

        predictor = cls(
            model_type=state["model_type"],
            model_version=state["model_version"],
            random_state=state.get("random_state", 42),
        )
        predictor.is_fitted = True
        predictor.models = models
        predictor.feature_schema_version = state.get("feature_schema_version", "1.0.0")
        predictor.calibrator = state.get("calibrator", None)
        predictor.training_metadata = dict(state.get("training_metadata", {}))
        return predictor
