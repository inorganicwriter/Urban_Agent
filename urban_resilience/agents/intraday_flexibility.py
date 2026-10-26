"""Intraday Flexibility Agent assessing peak shifts and off-peak adaptation capacity."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.types import AgentEvidence, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class IntradayFlexibilityAgent(BaseReActAgent):
    """Assesses dynamic adaptive capacity: the potential to shift peak trips and utilize off-peak hours."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        super().__init__("IntradayFlexibilityAgent", urban_context, mobility_network)
        self._register_tool("calc_flexibility_metrics", self._calc_flexibility_metrics)

    def _calc_flexibility_metrics(self, series: np.ndarray, zone_id: str) -> Dict[str, float]:
        """Quantifies peak shift flexibility and off-peak adaptation slack.

        If series length is less than 24 hours, returns neutral fallback metrics (0.5)
        to prevent reshaping errors while preserving all metric keys.
        """
        if len(series) < 24:
            return {
                "peak_shift_elasticity": 0.5,
                "off_peak_adaptation_ratio": 0.5,
                "peak_to_offpeak_ratio": 1.0,
                "intraday_flexibility_score": 0.5,
            }

        n_hours = len(series)
        n_days = max(1, n_hours // 24)
        reshaped = series[: n_days * 24].reshape(n_days, 24)
        avg_24h = np.mean(reshaped, axis=0)

        peak_hours = [7, 8, 9, 17, 18, 19]
        offpeak_day_hours = [10, 11, 12, 13, 14, 15, 16]

        peak_mean = max(1.0, float(np.mean(avg_24h[peak_hours])))
        offpeak_mean = float(np.mean(avg_24h[offpeak_day_hours]))

        # Off-peak adaptation ratio: higher ratio indicates flatter demand curve & larger reallocation capacity
        offpeak_adaptation_ratio = float(np.clip(offpeak_mean / peak_mean, 0.05, 1.0))

        # Peak shift elasticity: estimated from difference between morning & evening peak width
        fwhm_estimate = float(np.sum(avg_24h >= 0.5 * np.max(avg_24h)) / 24.0)

        # Zone characteristics modulation: residential and commercial zones have higher shift flexibility than strict manufacturing
        zone = self.urban_context.zones[zone_id]
        office_ratio = zone.poi_counts.get("office", 0) / max(1, sum(zone.poi_counts.values()))
        # High office / strict 9-to-6 rigidity reduces shift elasticity
        shift_elasticity = float(np.clip(fwhm_estimate * (1.2 - 0.5 * office_ratio), 0.1, 1.0))

        # Composite flexibility score
        flexibility_score = float(np.clip(0.55 * offpeak_adaptation_ratio + 0.45 * shift_elasticity, 0.0, 1.0))

        return {
            "peak_shift_elasticity": shift_elasticity,
            "off_peak_adaptation_ratio": offpeak_adaptation_ratio,
            "peak_to_offpeak_ratio": float(peak_mean / max(1.0, offpeak_mean)),
            "intraday_flexibility_score": flexibility_score,
        }

    def analyze(self) -> AgentEvidence:
        """Executes ReAct workflow for intraday flexibility."""
        self.traces.clear()

        # Step 1: Query Daily Profiles
        self.execute_react_step(
            thought="Inspect weekday vs weekend diurnal profile and zone-level peak dynamics.",
            action_type="query_data",
            action_input={"query_type": "daily_profile"},
        )

        zone_metrics = {}
        for zid in self.urban_context.zone_ids:
            series = self.mobility_network.hourly_time_series[zid].values
            res = self._calc_flexibility_metrics(series, zid)
            zone_metrics[zid] = res

        avg_elasticity = float(np.mean([m["peak_shift_elasticity"] for m in zone_metrics.values()]))
        avg_offpeak_adapt = float(np.mean([m["off_peak_adaptation_ratio"] for m in zone_metrics.values()]))
        avg_flexibility = float(np.mean([m["intraday_flexibility_score"] for m in zone_metrics.values()]))

        # Step 2: Check Evidence Bounds
        self.execute_react_step(
            thought="Validate flexibility scores against expected empirical limits [0, 1].",
            action_type="check_evidence",
            action_input={
                "metrics": {
                    "mean_peak_shift_elasticity": avg_elasticity,
                    "mean_offpeak_adaptation": avg_offpeak_adapt,
                },
                "bounds": {
                    "mean_peak_shift_elasticity": (0.0, 1.0),
                    "mean_offpeak_adaptation": (0.0, 1.0),
                },
            },
        )

        inference = (
            f"Average intraday flexibility is {avg_flexibility:.3f} (shift elasticity: {avg_elasticity:.3f}, "
            f"off-peak adaptation: {avg_offpeak_adapt:.3f}). Higher flexibility permits users to defer non-essential trips."
        )

        anomalies = []
        for zid, m in zone_metrics.items():
            if m["intraday_flexibility_score"] < 0.35:
                anomalies.append(f"Zone {zid} displays rigid commuting windows with low off-peak adaptation.")

        return AgentEvidence(
            agent_name=self.agent_name,
            metrics={
                "mean_peak_shift_elasticity": avg_elasticity,
                "mean_offpeak_adaptation": avg_offpeak_adapt,
                "mean_intraday_flexibility": avg_flexibility,
            },
            zone_metrics=zone_metrics,
            traces=list(self.traces),
            confidence=0.88,
            qualitative_inference=inference,
            potential_anomalies=anomalies,
        )
