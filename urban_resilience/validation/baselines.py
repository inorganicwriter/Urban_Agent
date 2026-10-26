"""Baseline benchmark models for comparative evaluation against the multi-agent system."""

from __future__ import annotations

import logging
from typing import List
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urban_resilience.config import compute_baseline_mean
from urban_resilience.types import BaselineComparison, EventContext, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class BaselineBenchmarkRunner:
    """Executes standard benchmark algorithms to contrast performance against the Multi-Agent framework."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork, lead_time_hours: int = 12):
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.lead_time_hours = lead_time_hours

    def run_all_baselines(
        self,
        event_context: EventContext,
        ground_truth: np.ndarray,
        multi_agent_pred: np.ndarray,
        eval_start_hour: int = 24,
    ) -> List[BaselineComparison]:
        """Runs benchmarks and compiles structured comparison metrics."""
        if eval_start_hour >= len(ground_truth):
            raise ValueError(
                f"eval_start_hour ({eval_start_hour}) must be < len(ground_truth) ({len(ground_truth)})"
            )

        y_true = ground_truth[eval_start_hour:]
        results = []

        # 1. Historical Mean (no event adaptation)
        baseline_raw = self.mobility_network.hourly_time_series.sum(axis=1).values
        b_mean = compute_baseline_mean(
            baseline_raw,
            event_start_hour=event_context.event_start_hour,
            lead_time_hours=self.lead_time_hours,
        )
        baseline_norm = baseline_raw / b_mean
        pred_hist_mean = baseline_norm[: len(ground_truth)][eval_start_hour:]
        results.append(self._score_model("Historical Baseline (No Shock)", y_true, pred_hist_mean))

        # 2. Autoregressive / Persistence Model
        pred_ar = self._run_autoregressive(baseline_norm, event_context, len(ground_truth))[eval_start_hour:]
        results.append(self._score_model("Autoregressive (AR Extrapolation)", y_true, pred_ar))

        # 3. Static Linear Regression (Ridge)
        pred_ridge = self._run_static_ridge(ground_truth, event_context, len(ground_truth))[eval_start_hour:]
        results.append(self._score_model("Static Ridge Regression", y_true, pred_ridge))

        # 4. Proposed Multi-Agent Cognitive Framework
        pred_agent = multi_agent_pred[eval_start_hour:]
        results.append(self._score_model("Proposed Multi-Agent Framework", y_true, pred_agent))

        return results

    def _score_model(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> BaselineComparison:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))) * 100.0
        r2 = float(r2_score(y_true, y_pred))
        return BaselineComparison(
            model_name=name,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2_score=r2,
        )

    def _run_autoregressive(self, baseline: np.ndarray, event_context: EventContext, total_h: int) -> np.ndarray:
        """Autoregressive projection based on pre-event lag-24 inertia."""
        ar_pred = baseline[:total_h].copy()
        start = event_context.event_start_hour
        # Linear decay heuristic without domain-specific resilience mechanisms
        for t in range(start, total_h):
            lag = t - 24
            ar_pred[t] = 0.85 * baseline[lag] if lag >= 0 else baseline[t]
        return ar_pred

    def _run_static_ridge(self, ground_truth: np.ndarray, event_context: EventContext, total_h: int) -> np.ndarray:
        """Static feature regression using basic POI counts and population."""
        X = []
        y = []
        for zid, zone in self.urban_context.zones.items():
            feat = [zone.population / 100000.0, zone.road_density_km_per_sqkm, sum(zone.poi_counts.values()) / 1000.0]
            X.append(feat)
            y.append(event_context.zone_intensity_map.get(zid, 0.5))

        X = np.array(X)
        y = np.array(y)
        reg = Ridge(alpha=1.0).fit(X, y)
        avg_shock_pred = float(np.mean(reg.predict(X)))

        # Static uniform reduction
        baseline = self.mobility_network.hourly_time_series.sum(axis=1).values[:total_h]
        b_mean = compute_baseline_mean(
            baseline,
            event_start_hour=event_context.event_start_hour,
            lead_time_hours=self.lead_time_hours,
        )
        baseline_norm = baseline / b_mean

        pred = baseline_norm.copy()
        start = event_context.event_start_hour
        dur = event_context.duration_hours
        pred[start : start + dur] *= (1.0 - 0.40 * avg_shock_pred)
        return pred
