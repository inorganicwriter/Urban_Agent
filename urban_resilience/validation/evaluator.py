"""Evaluator computing post-event ground-truth loss and recovery labels across spatial/temporal holdouts."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urban_resilience.config import compute_baseline_mean
from urban_resilience.types import (
    EventContext,
    MobilityNetwork,
    UrbanContext,
)

logger = logging.getLogger(__name__)


class HistoricalEventEvaluator:
    """Simulates/loads observed post-event outcomes to evaluate ex-ante predictions against ground truth."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork, lead_time_hours: int = 12):
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.lead_time_hours = lead_time_hours

    def generate_ground_truth_outcomes(
        self, event_context: EventContext, total_hours: int = 96, noise_scale: float = 0.03
    ) -> np.ndarray:
        """Generates observed post-event ground-truth activity curves incorporating realistic shock physics."""
        rng = np.random.default_rng(101)  # Distinct seed for empirical reality
        baseline = self.mobility_network.hourly_time_series.sum(axis=1).values[:total_hours]
        start = event_context.event_start_hour
        dur = event_context.duration_hours

        baseline_mean = compute_baseline_mean(baseline, event_start_hour=start, lead_time_hours=self.lead_time_hours)
        if baseline_mean <= 0 or not np.isfinite(baseline_mean):
            raise ValueError(f"Invalid baseline mean: {baseline_mean}. Baseline series must be positive and non-empty.")
        baseline_norm = baseline / baseline_mean

        hourly_shock = event_context.hourly_forecast[:total_hours]

        # Calculate empirical ground-truth trajectory by integrating zone-level impacts
        total_baseline = self.mobility_network.hourly_time_series.sum(axis=1).values[:total_hours]
        zone_ground_truth = np.zeros(total_hours, dtype=float)

        for zid, zone in self.urban_context.zones.items():
            base_z = self.mobility_network.hourly_time_series[zid].values[:total_hours]
            weight = np.mean(base_z) / np.mean(total_baseline)

            # Empirical physical factors (elevation, road density, POI mix)
            elev_vuln = max(0.0, (40.0 - zone.elevation_m) / 40.0)
            conn_vuln = max(0.0, (15.0 - zone.road_density_km_per_sqkm) / 15.0)
            poi_div = min(1.0, len(zone.poi_counts) / 7.0)
            hazard = event_context.zone_intensity_map.get(zid, 0.5)

            # Ground truth resilience aligns with physical infrastructure and geography
            emp_resilience = float(np.clip(0.35 * (1.0 - conn_vuln) + 0.35 * poi_div + 0.30 * (1.0 - elev_vuln), 0.1, 0.95))
            emp_vuln = float(np.clip(hazard * (1.10 - emp_resilience), 0.05, 0.95))

            loss_peak = float(np.clip(emp_vuln * 0.75, 0.05, 0.90))
            rec_speed = float(0.08 + 0.14 * emp_resilience)

            z_act = np.ones(total_hours, dtype=float)
            for t in range(total_hours):
                if start <= t < start + dur:
                    frac = hourly_shock[t] / (event_context.peak_intensity + 1e-6)
                    loss = loss_peak * frac + rng.normal(0, noise_scale)
                    z_act[t] = max(0.1, 1.0 - loss)
                elif t >= start + dur:
                    h_post = t - (start + dur)
                    residual = loss_peak * np.exp(-rec_speed * h_post) + rng.normal(0, noise_scale)
                    z_act[t] = min(1.05, max(0.1, 1.0 - max(0.0, residual)))

            zone_ground_truth += weight * z_act

        observed = baseline_norm * zone_ground_truth
        return observed

    def evaluate_predictions(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        evaluation_window_start: int = 24,
    ) -> Dict[str, float]:
        """Calculates evaluation metrics (RMSE, MAE, MAPE, R2) strictly over the event & recovery window."""
        if evaluation_window_start >= len(ground_truth):
            raise ValueError(
                f"evaluation_window_start cannot exceed or equal sequence length: {evaluation_window_start} >= {len(ground_truth)}"
            )

        y_true = ground_truth[evaluation_window_start:]
        y_pred = predicted[evaluation_window_start:]

        if len(y_true) == 0:
            raise ValueError("Evaluation window contains no data points.")

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))) * 100.0
        r2 = float(r2_score(y_true, y_pred))

        return {
            "rmse": rmse,
            "mae": mae,
            "mape_pct": mape,
            "r2_score": r2,
        }

    def split_spatial_temporal_holdouts(
        self, holdout_zone_ratio: float = 0.25
    ) -> Tuple[List[str], List[str]]:
        """Partitions zones into training and spatial holdout validation sets."""
        zone_ids = self.urban_context.zone_ids
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(zone_ids)
        n_holdout = max(1, int(len(zone_ids) * holdout_zone_ratio))
        holdout_zones = list(shuffled[:n_holdout])
        train_zones = list(shuffled[n_holdout:])
        return train_zones, holdout_zones
