"""Domain-specific computation tools for Spatial Redundancy Analysis.

Provides typed, sandboxed tools calculating destination diversity entropy, road network
degree/betweenness centrality, and transit line coverage redundancy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class SpatialRegionArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier (e.g. 'SZ_FT_CBD')")


def _get_or_build_road_graph(context: Dict[str, Any]) -> nx.Graph:
    """Helper to retrieve or build the road adjacency graph in context."""
    if "_road_graph" in context:
        return context["_road_graph"]

    urban_context: UrbanContext = context["urban_context"]
    G = nx.Graph()
    for zid in urban_context.zone_ids:
        G.add_node(zid)
    for u, neighbors in urban_context.road_adjacency.items():
        for v in neighbors:
            G.add_edge(u, v)

    context["_road_graph"] = G
    return G


def compute_destination_diversity(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates normalized Shannon entropy of trips distributed across destinations from region_id."""
    mobility: MobilityNetwork = context["mobility_network"]
    urban_context: UrbanContext = context["urban_context"]

    n_hours = len(mobility.od_matrices)
    out_flows = np.zeros(len(urban_context.zone_ids), dtype=float)

    for h in range(n_hours):
        od = mobility.od_matrices[h]
        if region_id in od.index:
            out_flows += od.loc[region_id].values

    total = np.sum(out_flows)
    if total <= 1e-6:
        return {
            "region_id": region_id,
            "destination_diversity": 0.0,
            "total_outflow": 0.0,
        }

    probs = out_flows / total
    n_zones = len(urban_context.zone_ids)
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(n_zones)
    diversity = float(np.clip(entropy, 0.0, 1.0))

    return {
        "region_id": region_id,
        "destination_diversity": round(diversity, 4),
        "total_outflow": round(float(total), 2),
        "active_destination_count": int(np.sum(out_flows > 1e-6)),
    }


def compute_network_route_redundancy(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates topological degree connectivity and betweenness centrality redundancy on road graph."""
    urban_context: UrbanContext = context["urban_context"]
    G = _get_or_build_road_graph(context)

    degree = G.degree(region_id) if region_id in G else 0
    max_degree = max(dict(G.degree()).values()) if G.nodes else 1
    norm_degree = float(degree / max(1, max_degree))

    betweenness = nx.betweenness_centrality(G)
    zone_betweenness = float(betweenness.get(region_id, 0.0))

    # Transit redundancy: count of transit lines serving this zone
    transit_count = sum(1 for line in urban_context.transit_lines if region_id in line["zones"])
    transit_redundancy = float(np.clip(transit_count / 3.0, 0.0, 1.0))

    route_redundancy = float(
        np.clip(
            0.45 * norm_degree + 0.25 * zone_betweenness + 0.30 * transit_redundancy,
            0.0,
            1.0,
        )
    )

    return {
        "region_id": region_id,
        "route_redundancy_score": round(route_redundancy, 4),
        "degree_connectivity": float(degree),
        "normalized_degree": round(norm_degree, 4),
        "betweenness_centrality": round(zone_betweenness, 4),
        "transit_lines_count": transit_count,
        "transit_redundancy": round(transit_redundancy, 4),
    }


def compute_composite_spatial_redundancy_score(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes overall spatial redundancy score combining destination diversity and route redundancy."""
    div_res = compute_destination_diversity(region_id, context)
    route_res = compute_network_route_redundancy(region_id, context)

    diversity = div_res.get("destination_diversity", 0.0)
    route = route_res.get("route_redundancy_score", 0.0)
    composite = float(np.clip(0.40 * diversity + 0.60 * route, 0.0, 1.0))

    return {
        "region_id": region_id,
        "spatial_redundancy_score": round(composite, 4),
        "destination_diversity": diversity,
        "route_redundancy_score": route,
    }


def build_spatial_tool_registry(
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    snapshot_id: str = "snapshot_spatial_v1",
) -> ToolRegistry:
    """Builds and returns a ToolRegistry populated with spatial redundancy tools."""
    allowed = list(urban_context.zone_ids)
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="compute_destination_diversity",
            description="Calculates Shannon entropy of trip distribution across destinations.",
            args_schema=SpatialRegionArgs,
            fn=lambda region_id, context: compute_destination_diversity(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_network_route_redundancy",
            description="Calculates topological degree connectivity, betweenness centrality, and transit redundancy.",
            args_schema=SpatialRegionArgs,
            fn=lambda region_id, context: compute_network_route_redundancy(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_composite_spatial_redundancy_score",
            description="Computes composite spatial redundancy combining destination diversity and physical route connectivity.",
            args_schema=SpatialRegionArgs,
            fn=lambda region_id, context: compute_composite_spatial_redundancy_score(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    return registry
