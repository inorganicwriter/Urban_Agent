"""Spatial Redundancy Agent evaluating alternative destinations and transport network routes."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np
import networkx as nx

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.types import AgentEvidence, MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class SpatialRedundancyAgent(BaseReActAgent):
    """Evaluates spatial diversification, alternative destination choices, and network route redundancy."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        super().__init__("SpatialRedundancyAgent", urban_context, mobility_network)
        self.road_graph = self._build_road_graph()
        self._register_tool("calc_destination_diversity", self._calc_destination_diversity)
        self._register_tool("calc_network_route_redundancy", self._calc_network_route_redundancy)

    def _build_road_graph(self) -> nx.Graph:
        """Constructs an undirected topology graph of physical road connections."""
        G = nx.Graph()
        for zid in self.urban_context.zone_ids:
            G.add_node(zid)
        for u, neighbors in self.urban_context.road_adjacency.items():
            for v in neighbors:
                G.add_edge(u, v)
        return G

    def _calc_destination_diversity(self, zone_id: str) -> float:
        """Calculates normalized Shannon entropy of trips distributed across destinations."""
        # Average OD flow from zone_id across all hours
        n_hours = len(self.mobility_network.od_matrices)
        out_flows = np.zeros(len(self.urban_context.zone_ids), dtype=float)

        for h in range(n_hours):
            od = self.mobility_network.od_matrices[h]
            if zone_id in od.index:
                out_flows += od.loc[zone_id].values

        total = np.sum(out_flows)
        if total <= 1e-6:
            return 0.0

        probs = out_flows / total
        n_zones = len(self.urban_context.zone_ids)
        entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(n_zones)
        return float(np.clip(entropy, 0.0, 1.0))

    def _calc_network_route_redundancy(self, zone_id: str) -> Dict[str, float]:
        """Calculates topological degree connectivity and edge-independent route redundancy."""
        degree = self.road_graph.degree(zone_id)
        max_degree = max(dict(self.road_graph.degree()).values())
        norm_degree = float(degree / max(1, max_degree))

        # Check node betweenness centrality
        betweenness = nx.betweenness_centrality(self.road_graph)
        zone_betweenness = float(betweenness.get(zone_id, 0.0))

        # Transit redundancy: count of transit lines serving this zone
        transit_count = sum(1 for line in self.urban_context.transit_lines if zone_id in line["zones"])
        transit_redundancy = float(np.clip(transit_count / 3.0, 0.0, 1.0))

        # Spatial route redundancy score
        route_redundancy = float(np.clip(0.45 * norm_degree + 0.25 * zone_betweenness + 0.30 * transit_redundancy, 0.0, 1.0))

        return {
            "degree_connectivity": float(degree),
            "betweenness_centrality": zone_betweenness,
            "transit_redundancy": transit_redundancy,
            "route_redundancy_score": route_redundancy,
        }

    def analyze(self) -> AgentEvidence:
        """Executes ReAct spatial redundancy assessment."""
        self.traces.clear()

        # Step 1: Query Road and Transit Topology
        self.execute_react_step(
            thought="Query spatial road adjacency graph and OD matrix to determine route and destination options.",
            action_type="query_data",
            action_input={"query_type": "od_matrix", "time_step": 0},
        )

        zone_metrics = {}
        for zid in self.urban_context.zone_ids:
            dest_diversity = self._calc_destination_diversity(zid)
            route_res = self._calc_network_route_redundancy(zid)
            route_score = route_res["route_redundancy_score"]

            composite_spatial = float(0.50 * dest_diversity + 0.50 * route_score)
            zone_metrics[zid] = {
                "destination_diversity": dest_diversity,
                "degree_connectivity": route_res["degree_connectivity"],
                "betweenness_centrality": route_res["betweenness_centrality"],
                "transit_redundancy": route_res["transit_redundancy"],
                "route_redundancy_score": route_score,
                "spatial_redundancy_score": composite_spatial,
            }

        avg_dest_div = float(np.mean([m["destination_diversity"] for m in zone_metrics.values()]))
        avg_route_red = float(np.mean([m["route_redundancy_score"] for m in zone_metrics.values()]))
        avg_spatial = float(np.mean([m["spatial_redundancy_score"] for m in zone_metrics.values()]))

        # Step 2: Check Evidence
        self.execute_react_step(
            thought="Verify spatial redundancy bounds across city topology.",
            action_type="check_evidence",
            action_input={
                "metrics": {
                    "mean_destination_diversity": avg_dest_div,
                    "mean_route_redundancy": avg_route_red,
                },
                "bounds": {
                    "mean_destination_diversity": (0.0, 1.0),
                    "mean_route_redundancy": (0.0, 1.0),
                },
            },
        )

        inference = (
            f"Mean spatial redundancy score is {avg_spatial:.3f}. Zones with dense road grids and multiple transit lines "
            f"possess alternative bypass paths when primary arterial corridors are inundated."
        )

        anomalies = []
        for zid, m in zone_metrics.items():
            if m["spatial_redundancy_score"] < 0.25:
                anomalies.append(f"Zone {zid} has critically low spatial redundancy (score: {m['spatial_redundancy_score']:.2f}), risk of isolation!")

        return AgentEvidence(
            agent_name=self.agent_name,
            metrics={
                "mean_destination_diversity": avg_dest_div,
                "mean_route_redundancy": avg_route_red,
                "mean_spatial_redundancy": avg_spatial,
            },
            zone_metrics=zone_metrics,
            traces=list(self.traces),
            confidence=0.91,
            qualitative_inference=inference,
            potential_anomalies=anomalies,
        )
