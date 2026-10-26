"""General data quality and snapshot auditing tools.

Provides tools to inspect temporal series completeness, coverage, stationarity,
and snapshot provenance without domain-specific logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class SeriesWindowArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier")
    start_hour: int = Field(default=0, ge=0, description="Start hour offset")
    window_hours: int = Field(default=24, ge=1, le=168, description="Length of observation window")


class DataQualityCheckArgs(BaseModel):
    region_id: str = Field(default="all", description="Target region ID or 'all'")


class RefinedEvidenceArgs(BaseModel):
    region_id: str = Field(description="Target urban zone identifier, e.g. 'SZ_FT_CBD'")
    domain: Optional[str] = Field(default=None, description="Resilience domain category")
    query_context: Optional[Dict[str, Any]] = Field(default=None, description="Query context dictionary with refined window parameters")
    reason: Optional[str] = Field(default="", description="Reason for targeted inquiry")


def inspect_series_window(
    region_id: str,
    context: Dict[str, Any],
    start_hour: int = 0,
    window_hours: int = 24,
) -> Dict[str, Any]:
    """Inspects statistical properties of a sliced temporal window for a region."""
    mobility: MobilityNetwork = context["mobility_network"]
    if region_id not in mobility.hourly_time_series.columns:
        return {
            "valid": False,
            "region_id": region_id,
            "error": f"Region '{region_id}' not found in mobility data.",
        }

    series = mobility.hourly_time_series[region_id].values
    total_hours = len(series)
    end_hour = min(total_hours, start_hour + window_hours)
    slice_data = series[start_hour:end_hour]

    if len(slice_data) == 0:
        return {
            "valid": False,
            "region_id": region_id,
            "error": "Window slice is empty.",
        }

    mean_v = float(np.mean(slice_data))
    std_v = float(np.std(slice_data))
    min_v = float(np.min(slice_data))
    max_v = float(np.max(slice_data))
    zero_ratio = float(np.sum(slice_data <= 1e-6) / len(slice_data))

    return {
        "valid": True,
        "region_id": region_id,
        "start_hour": start_hour,
        "actual_hours": len(slice_data),
        "mean": round(mean_v, 2),
        "std": round(std_v, 2),
        "min": round(min_v, 2),
        "max": round(max_v, 2),
        "zero_fraction": round(zero_ratio, 4),
    }


def check_citywide_data_completeness(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Audits city-wide temporal series completeness, null counts, and negative anomalies."""
    mobility: MobilityNetwork = context["mobility_network"]
    urban_context: UrbanContext = context["urban_context"]

    target_zones = [region_id] if region_id != "all" and region_id in urban_context.zone_ids else list(urban_context.zone_ids)
    missing_by_zone = {}
    negative_by_zone = {}

    for zid in target_zones:
        if zid in mobility.hourly_time_series.columns:
            vals = mobility.hourly_time_series[zid].values
            missing_by_zone[zid] = int(np.sum(np.isnan(vals)))
            negative_by_zone[zid] = int(np.sum(vals < 0))
        else:
            missing_by_zone[zid] = -1

    all_complete = all(cnt == 0 for cnt in missing_by_zone.values())
    no_negatives = all(cnt == 0 for cnt in negative_by_zone.values())

    return {
        "target_zone_count": len(target_zones),
        "all_zones_complete": all_complete,
        "all_zones_non_negative": no_negatives,
        "missing_hours_summary": missing_by_zone,
    }


def query_refined_evidence(
    region_id: str,
    context: Dict[str, Any],
    domain: Optional[str] = None,
    query_context: Optional[Dict[str, Any]] = None,
    reason: str = "",
) -> Dict[str, Any]:
    """Queries secondary observation window or independent refined source for follow-up verification."""
    from urban_resilience.orchestration.follow_up_data import SyntheticFollowUpDataSource

    active_domain = domain or context.get("active_domain", "temporal")
    data_source = context.get("follow_up_data_source") or SyntheticFollowUpDataSource()
    q_ctx = query_context or context.get("query_context") or {}
    sim_no_delta = bool(context.get("simulate_no_delta", False) or q_ctx.get("simulate_no_delta", False))

    obs = data_source.fetch(
        zone_id=region_id,
        domain=active_domain,
        reason=reason,
        context={**context, "query_context": q_ctx, "simulate_no_delta": sim_no_delta},
    )

    out = {
        "region_id": region_id,
        "domain": active_domain,
        "source": obs.source,
        **obs.metrics,
    }
    return out


def create_refined_evidence_tool_spec(
    allowed_regions: Optional[list[str]] = None,
    snapshot_id: str = "snapshot_data_v1",
) -> ToolSpec:
    """Builds the standardized ToolSpec for the query_refined_evidence tool."""
    return ToolSpec(
        name="query_refined_evidence",
        description="Queries secondary observation window or distinct data source for refined follow-up evidence.",
        args_schema=RefinedEvidenceArgs,
        fn=lambda region_id, context, domain=None, query_context=None, reason="": query_refined_evidence(
            region_id=region_id, context=context, domain=domain, query_context=query_context, reason=reason
        ),
        allowed_regions=allowed_regions,
        data_snapshot=snapshot_id,
        tool_version="1.0.0",
    )


def build_data_tool_registry(
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    snapshot_id: str = "snapshot_data_v1",
) -> ToolRegistry:
    """Builds and returns a ToolRegistry populated with generic data auditing tools."""
    allowed = list(urban_context.zone_ids) + ["all"]
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="inspect_series_window",
            description="Inspects windowed mobility series slice statistics for a specific region.",
            args_schema=SeriesWindowArgs,
            fn=lambda region_id, context, start_hour=0, window_hours=24: inspect_series_window(region_id, context, start_hour, window_hours),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="check_citywide_data_completeness",
            description="Audits city-wide temporal series completeness, null counts, and negative anomalies.",
            args_schema=DataQualityCheckArgs,
            fn=lambda region_id, context: check_citywide_data_completeness(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(create_refined_evidence_tool_spec(allowed_regions=allowed, snapshot_id=snapshot_id))

    return registry
