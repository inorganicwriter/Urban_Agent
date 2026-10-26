"""Functional Mix Agent evaluating POI diversity and activity composition balance."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.types import AgentEvidence, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class FunctionalMixAgent(BaseReActAgent):
    """Evaluates local functional self-sufficiency via POI diversity and land-use activity composition."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        super().__init__("FunctionalMixAgent", urban_context, mobility_network)
        self._register_tool("calc_poi_diversity_and_mix", self._calc_poi_diversity_and_mix)

    def _calc_poi_diversity_and_mix(self, zone_id: str) -> Dict[str, float]:
        """Calculates normalized Shannon POI diversity and activity balance index."""
        zone = self.urban_context.zones[zone_id]
        poi_counts = zone.poi_counts
        total_poi = sum(poi_counts.values())

        if total_poi <= 0:
            return {"poi_entropy": 0.0, "simpson_diversity": 0.0, "activity_balance": 0.0, "functional_mix_score": 0.0}

        counts = np.array(list(poi_counts.values()), dtype=float)
        probs = counts / total_poi
        k = len(counts)

        # Shannon entropy
        shannon = -np.sum(probs * np.log(probs + 1e-12)) / np.log(max(2, k))
        shannon = float(np.clip(shannon, 0.0, 1.0))

        # Simpson diversity index 1 - sum(p^2)
        simpson = float(1.0 - np.sum(probs ** 2))

        # Activity composition balance: entropy of land use
        land_use = zone.land_use_mix
        lu_vals = np.array(list(land_use.values()), dtype=float)
        lu_probs = lu_vals / (np.sum(lu_vals) + 1e-12)
        lu_entropy = -np.sum(lu_probs * np.log(lu_probs + 1e-12)) / np.log(max(2, len(lu_vals)))
        lu_balance = float(np.clip(lu_entropy, 0.0, 1.0))

        # Composite functional mix score
        mix_score = float(np.clip(0.40 * shannon + 0.30 * simpson + 0.30 * lu_balance, 0.0, 1.0))

        return {
            "poi_entropy": shannon,
            "simpson_diversity": simpson,
            "activity_balance": lu_balance,
            "functional_mix_score": mix_score,
            "total_poi_count": float(total_poi),
        }

    def analyze(self) -> AgentEvidence:
        """Executes ReAct analysis for functional mix."""
        self.traces.clear()

        # Step 1: Query Zone Context
        self.execute_react_step(
            thought="Retrieve POI category inventories and land-use fractions to evaluate local self-containment.",
            action_type="query_data",
            action_input={"query_type": "zone_info", "zone_id": self.urban_context.zone_ids[0]},
        )

        zone_metrics = {}
        for zid in self.urban_context.zone_ids:
            res = self._calc_poi_diversity_and_mix(zid)
            zone_metrics[zid] = res

        avg_poi_ent = float(np.mean([m["poi_entropy"] for m in zone_metrics.values()]))
        avg_balance = float(np.mean([m["activity_balance"] for m in zone_metrics.values()]))
        avg_mix = float(np.mean([m["functional_mix_score"] for m in zone_metrics.values()]))

        # Step 2: Validate Evidence
        self.execute_react_step(
            thought="Check bounded validity of POI entropy and activity balance metrics.",
            action_type="check_evidence",
            action_input={
                "metrics": {"mean_poi_entropy": avg_poi_ent, "mean_activity_balance": avg_balance},
                "bounds": {"mean_poi_entropy": (0.0, 1.0), "mean_activity_balance": (0.0, 1.0)},
            },
        )

        inference = (
            f"Mean functional mix score is {avg_mix:.3f} (POI Shannon entropy: {avg_poi_ent:.3f}). "
            f"Mixed-use zones support essential life-safety and sustenance trips locally, mitigating reliance on external transit corridors."
        )

        anomalies = []
        for zid, m in zone_metrics.items():
            if m["functional_mix_score"] < 0.40:
                anomalies.append(f"Zone {zid} has single-function land use (score: {m['functional_mix_score']:.2f}), making it dependent on distant zones.")

        return AgentEvidence(
            agent_name=self.agent_name,
            metrics={
                "mean_poi_entropy": avg_poi_ent,
                "mean_activity_balance": avg_balance,
                "mean_functional_mix_score": avg_mix,
            },
            zone_metrics=zone_metrics,
            traces=list(self.traces),
            confidence=0.90,
            qualitative_inference=inference,
            potential_anomalies=anomalies,
        )
