"""Domain-specific computation tools for Functional Mix Analysis.

Provides typed, sandboxed tools calculating POI category Shannon entropy, Simpson diversity,
and land-use activity composition balance with strict parameter validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class FunctionalRegionArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier (e.g. 'SZ_FT_CBD')")


def compute_poi_diversity_metrics(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates normalized Shannon POI entropy and Simpson diversity index for region_id."""
    urban_context: UrbanContext = context["urban_context"]
    zone = urban_context.zones[region_id]
    poi_counts = zone.poi_counts
    total_poi = sum(poi_counts.values())

    if total_poi <= 0:
        return {
            "region_id": region_id,
            "poi_entropy": 0.0,
            "simpson_diversity": 0.0,
            "total_poi_count": 0,
        }

    counts = np.array(list(poi_counts.values()), dtype=float)
    probs = counts / total_poi
    k = len(counts)

    shannon = -np.sum(probs * np.log(probs + 1e-12)) / np.log(max(2, k))
    shannon = float(np.clip(shannon, 0.0, 1.0))
    simpson = float(1.0 - np.sum(probs**2))

    return {
        "region_id": region_id,
        "poi_entropy": round(shannon, 4),
        "simpson_diversity": round(simpson, 4),
        "total_poi_count": total_poi,
        "categories_present": int(np.sum(counts > 0)),
    }


def compute_land_use_balance(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates land-use activity composition balance entropy across land use types."""
    urban_context: UrbanContext = context["urban_context"]
    zone = urban_context.zones[region_id]
    land_use = zone.land_use_mix
    lu_vals = np.array(list(land_use.values()), dtype=float)
    total_lu = np.sum(lu_vals)

    if total_lu <= 0:
        return {
            "region_id": region_id,
            "activity_balance": 0.0,
        }

    lu_probs = lu_vals / (total_lu + 1e-12)
    lu_entropy = -np.sum(lu_probs * np.log(lu_probs + 1e-12)) / np.log(max(2, len(lu_vals)))
    lu_balance = float(np.clip(lu_entropy, 0.0, 1.0))

    return {
        "region_id": region_id,
        "activity_balance": round(lu_balance, 4),
        "residential_fraction": round(float(land_use.get("residential", 0.0)), 4),
        "commercial_fraction": round(float(land_use.get("commercial", 0.0)), 4),
        "industrial_fraction": round(float(land_use.get("industrial", 0.0)), 4),
    }


def compute_composite_functional_mix_score(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes overall composite functional mix score combining POI entropy, Simpson, and land use balance."""
    poi_res = compute_poi_diversity_metrics(region_id, context)
    lu_res = compute_land_use_balance(region_id, context)

    shannon = poi_res.get("poi_entropy", 0.0)
    simpson = poi_res.get("simpson_diversity", 0.0)
    lu_balance = lu_res.get("activity_balance", 0.0)

    mix_score = float(np.clip(0.40 * shannon + 0.30 * simpson + 0.30 * lu_balance, 0.0, 1.0))

    return {
        "region_id": region_id,
        "functional_mix_score": round(mix_score, 4),
        "poi_entropy": shannon,
        "simpson_diversity": simpson,
        "activity_balance": lu_balance,
        "total_poi_count": poi_res.get("total_poi_count", 0),
    }


def build_functional_tool_registry(
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    snapshot_id: str = "snapshot_functional_v1",
) -> ToolRegistry:
    """Builds and returns a ToolRegistry populated with functional mix tools."""
    allowed = list(urban_context.zone_ids)
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="compute_poi_diversity_metrics",
            description="Calculates Shannon entropy and Simpson index across POI categories.",
            args_schema=FunctionalRegionArgs,
            fn=lambda region_id, context: compute_poi_diversity_metrics(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_land_use_balance",
            description="Calculates land-use activity composition balance entropy.",
            args_schema=FunctionalRegionArgs,
            fn=lambda region_id, context: compute_land_use_balance(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_composite_functional_mix_score",
            description="Computes composite functional mix score combining POI entropy, Simpson index, and land use balance.",
            args_schema=FunctionalRegionArgs,
            fn=lambda region_id, context: compute_composite_functional_mix_score(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    return registry
