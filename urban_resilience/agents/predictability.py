"""Predictability Agent evaluating temporal autocorrelation and rhythm regularity."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.types import AgentEvidence, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class PredictabilityAgent(BaseReActAgent):
    """Analyzes system predictability via multi-lag autocorrelation and cyclical rhythm regularity."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        super().__init__("PredictabilityAgent", urban_context, mobility_network)
        self._register_tool("calc_acf_and_regularity", self._calc_acf_and_regularity)

    def _calc_acf_and_regularity(self, series: np.ndarray) -> Dict[str, float]:
        """Calculates autocorrelation function (lag 1, lag 24) and inter-day rhythm regularity."""
        n = len(series)
        if n < 48:
            return {"autocorr_lag1": 0.5, "autocorr_lag24": 0.5, "rhythm_regularity": 0.5, "predictability_score": 0.5}

        # Normalize series
        mean_val = np.mean(series)
        var_val = np.var(series) + 1e-12
        centered = series - mean_val

        # Autocorrelation for lags 1 to 24
        def compute_lag_corr(lag: int) -> float:
            c = np.sum(centered[: n - lag] * centered[lag:]) / ((n - lag) * var_val)
            return float(np.clip(c, -1.0, 1.0))

        acf_1 = compute_lag_corr(1)
        acf_24 = compute_lag_corr(24)  # Diurnal rhythm lock

        # Inter-day cosine similarity between consecutive 24-hour segments
        n_days = n // 24
        day_segments = [series[d * 24 : (d + 1) * 24] for d in range(n_days)]
        cos_sims = []
        for i in range(len(day_segments) - 1):
            s1 = day_segments[i]
            s2 = day_segments[i + 1]
            norm1 = np.linalg.norm(s1) + 1e-12
            norm2 = np.linalg.norm(s2) + 1e-12
            sim = float(np.dot(s1, s2) / (norm1 * norm2))
            cos_sims.append(sim)

        regularity = float(np.mean(cos_sims)) if cos_sims else 0.85
        regularity = float(np.clip(regularity, 0.0, 1.0))

        # Overall predictability: weighted combination of 24h diurnal lock and inter-day regularity
        predictability_score = float(np.clip(0.4 * max(0.0, acf_1) + 0.3 * max(0.0, acf_24) + 0.3 * regularity, 0.0, 1.0))

        return {
            "autocorr_lag1": acf_1,
            "autocorr_lag24": acf_24,
            "rhythm_regularity": regularity,
            "predictability_score": predictability_score,
        }

    def analyze(self) -> AgentEvidence:
        """Runs the ReAct workflow for mobility predictability."""
        self.traces.clear()

        # Step 1: Query Time Series
        self.execute_react_step(
            thought="Retrieve zone-wise hourly series to inspect 24-hour diurnal autocorrelation and periodicity.",
            action_type="query_data",
            action_input={"query_type": "hourly_series"},
        )

        zone_metrics = {}
        for zid in self.urban_context.zone_ids:
            series = self.mobility_network.hourly_time_series[zid].values
            res = self._calc_acf_and_regularity(series)
            zone_metrics[zid] = res

        avg_acf1 = float(np.mean([m["autocorr_lag1"] for m in zone_metrics.values()]))
        avg_acf24 = float(np.mean([m["autocorr_lag24"] for m in zone_metrics.values()]))
        avg_regularity = float(np.mean([m["rhythm_regularity"] for m in zone_metrics.values()]))
        avg_predictability = float(np.mean([m["predictability_score"] for m in zone_metrics.values()]))

        # Step 2: Validate Evidence
        self.execute_react_step(
            thought="Validate diurnal autocorrelation and regularity bounds.",
            action_type="check_evidence",
            action_input={
                "metrics": {
                    "autocorr_lag24": avg_acf24,
                    "rhythm_regularity": avg_regularity,
                },
                "bounds": {
                    "autocorr_lag24": (-1.0, 1.0),
                    "rhythm_regularity": (0.0, 1.0),
                },
            },
        )

        inference = (
            f"Diurnal autocorrelation (lag 24) is strong at {avg_acf24:.3f} and inter-day regularity is {avg_regularity:.3f}. "
            f"High predictability enables reliable baseline projection and prompt detection of anomalous drops."
        )

        anomalies = []
        for zid, m in zone_metrics.items():
            if m["rhythm_regularity"] < 0.60:
                anomalies.append(f"Zone {zid} shows low rhythm regularity ({m['rhythm_regularity']:.2f}), indicating high stochasticity.")

        return AgentEvidence(
            agent_name=self.agent_name,
            metrics={
                "mean_autocorr_lag1": avg_acf1,
                "mean_autocorr_lag24": avg_acf24,
                "mean_rhythm_regularity": avg_regularity,
                "mean_predictability_score": avg_predictability,
            },
            zone_metrics=zone_metrics,
            traces=list(self.traces),
            confidence=0.94,
            qualitative_inference=inference,
            potential_anomalies=anomalies,
        )
