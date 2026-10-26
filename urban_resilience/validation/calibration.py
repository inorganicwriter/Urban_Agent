"""Uncertainty calibration module computing PICP, MPIW, and Winkler interval scores."""

from __future__ import annotations

import logging
import numpy as np

from urban_resilience.types import CalibrationMetrics, ResiliencePrediction

logger = logging.getLogger(__name__)


class UncertaintyCalibrator:
    """Evaluates the calibration quality, empirical coverage, and sharpness of prediction intervals."""

    @staticmethod
    def evaluate_calibration(
        prediction: ResiliencePrediction,
        ground_truth: np.ndarray,
        nominal_coverage: float = 0.95,
        eval_start_hour: int = 24,
    ) -> CalibrationMetrics:
        if eval_start_hour < 0 or eval_start_hour >= len(ground_truth):
            raise ValueError(
                f"eval_start_hour ({eval_start_hour}) must be >= 0 and < len(ground_truth) ({len(ground_truth)})"
            )
        if len(prediction.lower_bound_95) != len(ground_truth) or len(prediction.upper_bound_95) != len(ground_truth):
            raise ValueError("prediction bounds and ground_truth must have identical lengths")

        y_true = ground_truth[eval_start_hour:]
        lower = prediction.lower_bound_95[eval_start_hour:]
        upper = prediction.upper_bound_95[eval_start_hour:]

        # 1. Prediction Interval Coverage Probability (PICP)
        covered = (y_true >= lower) & (y_true <= upper)
        picp = float(np.mean(covered))

        # 2. Mean Prediction Interval Width (MPIW)
        widths = upper - lower
        mpiw = float(np.mean(widths))

        # 3. Winkler Score (alpha = 1 - nominal_coverage = 0.05)
        alpha = 1.0 - nominal_coverage
        winkler_penalties = widths.copy()
        below = y_true < lower
        above = y_true > upper
        winkler_penalties[below] += (2.0 / alpha) * (lower[below] - y_true[below])
        winkler_penalties[above] += (2.0 / alpha) * (y_true[above] - upper[above])
        winkler_score = float(np.mean(winkler_penalties))

        # 4. Expected Calibration Error (ECE) for interval coverage
        ece = float(np.abs(picp - nominal_coverage))

        logger.info("Calibration evaluated: PICP=%.3f (nominal=%.2f), MPIW=%.3f, ECE=%.3f",
                    picp, nominal_coverage, mpiw, ece)

        return CalibrationMetrics(
            picp_95=picp,
            mpiw=mpiw,
            interval_score=winkler_score,
            ece=ece,
        )
