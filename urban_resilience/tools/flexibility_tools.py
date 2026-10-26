"""Domain-specific computation tools for Intraday Flexibility Analysis.

Provides typed, sandboxed tools calculating peak-shift potential, off-peak adaptation,
and peak-to-offpeak demand ratios with strict parameter validation and execution tracing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class FlexibilityRegionArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier (e.g. 'SZ_FT_CBD')")


class FlexibilityQualityArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier")
    min_hours: int = Field(default=24, description="Minimum hours required for valid diurnal estimation")


def inspect_flexibility_data(
    region_id: str,
    context: Dict[str, Any],
    min_hours: int = 24,
) -> Dict[str, Any]:
    """Inspects mobility time series quality and duration for flexibility evaluation."""
    mobility: MobilityNetwork = context["mobility_network"]
    if region_id not in mobility.hourly_time_series.columns:
        return {
            "valid": False,
            "region_id": region_id,
            "error": f"Region '{region_id}' not found in mobility hourly time series.",
        }

    series = mobility.hourly_time_series[region_id].values
    n_hours = len(series)
    missing_count = int(np.sum(np.isnan(series)))
    zero_count = int(np.sum(series <= 1e-6))

    return {
        "valid": bool(n_hours >= min_hours and missing_count == 0),
        "region_id": region_id,
        "total_hours": n_hours,
        "missing_hours": missing_count,
        "zero_activity_hours": zero_count,
        "mean_activity": float(np.nanmean(series)),
    }


def compute_peak_shift_elasticity(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes peak shift elasticity modulated by local land use and office concentration."""
    mobility: MobilityNetwork = context["mobility_network"]
    urban_context: UrbanContext = context["urban_context"]

    series = mobility.hourly_time_series[region_id].values
    if len(series) < 24:
        return {
            "region_id": region_id,
            "peak_shift_elasticity": 0.5,
            "status": "insufficient_data_fallback",
        }

    n_days = max(1, len(series) // 24)
    reshaped = series[: n_days * 24].reshape(n_days, 24)
    avg_24h = np.mean(reshaped, axis=0)

    fwhm_estimate = float(np.sum(avg_24h >= 0.5 * np.max(avg_24h)) / 24.0)
    zone = urban_context.zones[region_id]
    office_ratio = zone.poi_counts.get("office", 0) / max(1, sum(zone.poi_counts.values()))
    shift_elasticity = float(np.clip(fwhm_estimate * (1.2 - 0.5 * office_ratio), 0.1, 1.0))

    return {
        "region_id": region_id,
        "peak_shift_elasticity": round(shift_elasticity, 4),
        "fwhm_estimate": round(fwhm_estimate, 4),
        "office_ratio": round(office_ratio, 4),
    }


def compute_offpeak_adaptation(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes off-peak adaptation ratio and peak-to-offpeak reallocation slack."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values
    if len(series) < 24:
        return {
            "region_id": region_id,
            "off_peak_adaptation_ratio": 0.5,
            "peak_to_offpeak_ratio": 1.0,
            "status": "insufficient_data_fallback",
        }

    n_days = max(1, len(series) // 24)
    reshaped = series[: n_days * 24].reshape(n_days, 24)
    avg_24h = np.mean(reshaped, axis=0)

    peak_hours = [7, 8, 9, 17, 18, 19]
    offpeak_day_hours = [10, 11, 12, 13, 14, 15, 16]

    peak_mean = max(1.0, float(np.mean(avg_24h[peak_hours])))
    offpeak_mean = float(np.mean(avg_24h[offpeak_day_hours]))
    offpeak_adaptation_ratio = float(np.clip(offpeak_mean / peak_mean, 0.05, 1.0))
    peak_to_offpeak_ratio = float(peak_mean / max(1.0, offpeak_mean))

    return {
        "region_id": region_id,
        "off_peak_adaptation_ratio": round(offpeak_adaptation_ratio, 4),
        "peak_to_offpeak_ratio": round(peak_to_offpeak_ratio, 4),
        "peak_mean_volume": round(peak_mean, 2),
        "offpeak_mean_volume": round(offpeak_mean, 2),
    }


def compute_composite_flexibility_score(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes overall composite intraday flexibility score (0.0 to 1.0)."""
    res_elasticity = compute_peak_shift_elasticity(region_id, context)
    res_offpeak = compute_offpeak_adaptation(region_id, context)

    elasticity = res_elasticity.get("peak_shift_elasticity", 0.5)
    offpeak = res_offpeak.get("off_peak_adaptation_ratio", 0.5)
    flexibility_score = float(np.clip(0.55 * offpeak + 0.45 * elasticity, 0.0, 1.0))

    return {
        "region_id": region_id,
        "intraday_flexibility_score": round(flexibility_score, 4),
        "peak_shift_elasticity": elasticity,
        "off_peak_adaptation_ratio": offpeak,
    }


def build_flexibility_tool_registry(
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    snapshot_id: str = "snapshot_flexibility_v1",
) -> ToolRegistry:
    """Builds and returns a ToolRegistry populated with flexibility tools."""
    allowed = list(urban_context.zone_ids)
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="inspect_flexibility_data",
            description="Inspects quality and length of series for flexibility evaluation.",
            args_schema=FlexibilityQualityArgs,
            fn=lambda region_id, context, min_hours=24: inspect_flexibility_data(region_id, context, min_hours),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_peak_shift_elasticity",
            description="Calculates peak shift elasticity based on peak width and office concentration.",
            args_schema=FlexibilityRegionArgs,
            fn=lambda region_id, context: compute_peak_shift_elasticity(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_offpeak_adaptation",
            description="Calculates off-peak adaptation ratio and peak-to-offpeak volume ratio.",
            args_schema=FlexibilityRegionArgs,
            fn=lambda region_id, context: compute_offpeak_adaptation(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_composite_flexibility_score",
            description="Calculates composite intraday flexibility score combining shift elasticity and offpeak slack.",
            args_schema=FlexibilityRegionArgs,
            fn=lambda region_id, context: compute_composite_flexibility_score(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    return registry
