"""Temporal Redundancy Agent evaluating hourly entropy and peak concentration."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.types import AgentEvidence, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class TemporalRedundancyAgent(BaseReActAgent):
    """Examines temporal buffer capacity via hourly activity entropy and peak concentration."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        super().__init__("TemporalRedundancyAgent", urban_context, mobility_network)
        self._register_tool("calc_entropy_and_peaks", self._calc_entropy_and_peaks)

    def _calc_entropy_and_peaks(self, series: np.ndarray) -> Dict[str, float]:
        """Calculates normalized Shannon entropy, peak ratio, and Gini concentration.

        If series length is less than 24 hours, returns neutral fallback metrics (0.5)
        to prevent reshaping errors while preserving all metric keys.
        """
        if len(series) < 24:
            return {
                "hourly_entropy": 0.5,
                "peak_concentration": 0.5,
                "gini_index": 0.5,
                "peak_ratio": 0.5,
                "temporal_redundancy_score": 0.5,
            }

        total = np.sum(series)
        if total <= 1e-6:
            return {
                "hourly_entropy": 0.0,
                "peak_concentration": 1.0,
                "gini_index": 1.0,
                "peak_ratio": 1.0,
                "temporal_redundancy_score": 0.0,
            }

        # 24-hour average profile
        n_hours = len(series)
        n_days = max(1, n_hours // 24)
        reshaped = series[: n_days * 24].reshape(n_days, 24)
        avg_24h = np.mean(reshaped, axis=0)

        probs = avg_24h / np.sum(avg_24h)
        # Normalized Shannon entropy [0, 1]
        entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(24.0)
        entropy = float(np.clip(entropy, 0.0, 1.0))

        # Peak concentration: ratio of morning (7-9) + evening (17-19) to total
        peak_hours = [7, 8, 9, 17, 18, 19]
        peak_vol = np.sum(avg_24h[peak_hours])
        peak_ratio = float(peak_vol / np.sum(avg_24h))

        # Gini coefficient of hourly distribution
        sorted_vals = np.sort(avg_24h)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals) + 1e-12))
        gini = float(np.clip(gini, 0.0, 1.0))

        peak_concentration = float(0.5 * peak_ratio + 0.5 * gini)

        # Redundancy score: higher entropy and lower peak concentration means higher temporal redundancy
        redundancy_score = float(np.clip(0.6 * entropy + 0.4 * (1.0 - peak_concentration), 0.0, 1.0))

        return {
            "hourly_entropy": entropy,
            "peak_concentration": peak_concentration,
            "gini_index": gini,
            "peak_ratio": peak_ratio,
            "temporal_redundancy_score": redundancy_score,
        }

    def analyze(self) -> AgentEvidence:
        """Executes ReAct analysis across all city zones."""
        self.traces.clear()

        # Step 1: Reason and Query
        self.execute_react_step(
            thought="Query baseline hourly mobility curves across all urban zones to evaluate temporal buffering.",
            action_type="query_data",
            action_input={"query_type": "hourly_series"},
        )

        # Step 2: Compute Metrics per Zone
        zone_metrics: Dict[str, Dict[str, float]] = {}
        for zid in self.urban_context.zone_ids:
            series = self.mobility_network.hourly_time_series[zid].values
            res = self._calc_entropy_and_peaks(series)
            zone_metrics[zid] = res

        # Step 3: Check Evidence
        check_bounds = {"hourly_entropy": (0.0, 1.0), "peak_concentration": (0.0, 1.0)}
        avg_entropy = float(np.mean([m["hourly_entropy"] for m in zone_metrics.values()]))
        avg_peak_conc = float(np.mean([m["peak_concentration"] for m in zone_metrics.values()]))
        avg_redundancy = float(np.mean([m["temporal_redundancy_score"] for m in zone_metrics.values()]))

        self.execute_react_step(
            thought="Check whether aggregated metrics fall within theoretical bounded ranges.",
            action_type="check_evidence",
            action_input={
                "metrics": {"hourly_entropy": avg_entropy, "peak_concentration": avg_peak_conc},
                "bounds": check_bounds,
            },
        )

        # Qualitative inference
        inference = (
            f"Mean temporal entropy is {avg_entropy:.3f} with peak concentration {avg_peak_conc:.3f}. "
            f"Zones with high temporal redundancy can absorb shock by spreading activities into off-peak windows."
        )

        anomalies = []
        for zid, m in zone_metrics.items():
            if m["peak_concentration"] > 0.65:
                anomalies.append(f"Zone {zid} has extreme peak concentration ({m['peak_concentration']:.2f}), vulnerable to peak-hour shocks.")

        return AgentEvidence(
            agent_name=self.agent_name,
            metrics={
                "mean_hourly_entropy": avg_entropy,
                "mean_peak_concentration": avg_peak_conc,
                "mean_temporal_redundancy": avg_redundancy,
            },
            zone_metrics=zone_metrics,
            traces=list(self.traces),
            confidence=0.92,
            qualitative_inference=inference,
            potential_anomalies=anomalies,
        )
