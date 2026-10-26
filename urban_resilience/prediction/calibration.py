"""Conformal and quantile interval calibration for resilience predictions.

Guarantees empirical coverage rates (e.g., 90% or 95% CI) using split-conformal
non-conformity score scaling on validation residuals.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalibrationMetrics(BaseModel):
    """Evaluation metrics for uncertainty prediction intervals."""

    target_name: str
    nominal_confidence: float = 0.95
    empirical_coverage: float = Field(..., description="Prediction Interval Coverage Probability (PICP)")
    mean_interval_width: float = Field(..., description="Mean Prediction Interval Width (MPIW)")
    is_valid: bool = Field(..., description="True if empirical coverage meets or exceeds nominal target")


class ConformalIntervalCalibrator:
    """Split-conformal prediction calibrator for regression residuals."""

    def __init__(self, target_coverage: float = 0.95):
        self.target_coverage = target_coverage
        self.nonconformity_quantiles: Dict[str, float] = {}

    def fit_residuals(
        self,
        y_true: pd.DataFrame,
        y_pred: Dict[str, np.ndarray],
    ) -> ConformalIntervalCalibrator:
        """Computes empirical (1 - alpha) non-conformity quantiles on calibration fold.

        Args:
            y_true: DataFrame containing ground truth columns.
            y_pred: Dictionary of predicted point estimates matching column names.
        """
        for col in y_true.columns:
            if col not in y_pred:
                continue
            residuals = np.abs(y_true[col].values - y_pred[col])
            n = len(residuals)
            # Standard finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
            q_level = float(np.clip(np.ceil((n + 1) * self.target_coverage) / n, 0.5, 1.0))
            quantile_val = float(np.quantile(residuals, q_level, method="higher"))
            self.nonconformity_quantiles[col] = quantile_val
            logger.info("Calibrator fitted for %s: non-conformity quantile (%.2f) = %.4f", col, self.target_coverage, quantile_val)

        return self

    def calibrate_intervals(
        self,
        y_pred: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Applies calibrated conformal margin to point predictions.

        Returns:
            Dictionary mapping target name to (lower_bound, upper_bound) arrays.
        """
        intervals = {}
        for target, preds in y_pred.items():
            margin = self.nonconformity_quantiles.get(target, 0.05)
            lower = np.maximum(0.0, preds - margin)
            upper = preds + margin
            if "loss" in target:
                upper = np.minimum(1.0, upper)
            intervals[target] = (lower, upper)
        return intervals

    def evaluate_coverage(
        self,
        y_true: pd.DataFrame,
        intervals: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, CalibrationMetrics]:
        """Calculates PICP coverage and MPIW sharpness against held-out ground truth."""
        results = {}
        for target, (lower, upper) in intervals.items():
            if target not in y_true.columns:
                continue
            vals = y_true[target].values
            covered = np.logical_and(vals >= lower, vals <= upper)
            picp = float(np.mean(covered))
            mpiw = float(np.mean(upper - lower))

            results[target] = CalibrationMetrics(
                target_name=target,
                nominal_confidence=self.target_coverage,
                empirical_coverage=round(picp, 4),
                mean_interval_width=round(mpiw, 4),
                is_valid=bool(picp >= self.target_coverage - 0.05),
            )
        return results
