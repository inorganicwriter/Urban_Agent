"""Domain-specific computation tools for Predictability Analysis.

Provides typed, sandboxed tools calculating autocorrelation profiles (Lag 1, Lag 24),
inter-day cosine regularity, and cyclical rhythm stability with strict parameter validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class PredictabilityRegionArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier (e.g. 'SZ_FT_CBD')")


class PredictabilityQualityArgs(BaseModel):
    region_id: str = Field(..., description="Target urban zone / region identifier")
    min_hours: int = Field(default=48, description="Minimum hours required for multi-lag autocorrelation")


def inspect_predictability_data(
    region_id: str,
    context: Dict[str, Any],
    min_hours: int = 48,
) -> Dict[str, Any]:
    """Inspects series length and stationarity conditions for autocorrelation estimation."""
    mobility: MobilityNetwork = context["mobility_network"]
    if region_id not in mobility.hourly_time_series.columns:
        return {
            "valid": False,
            "region_id": region_id,
            "error": f"Region '{region_id}' not found in mobility hourly time series.",
        }

    series = mobility.hourly_time_series[region_id].values
    n = len(series)
    missing = int(np.sum(np.isnan(series)))

    return {
        "valid": bool(n >= min_hours and missing == 0),
        "region_id": region_id,
        "total_hours": n,
        "missing_hours": missing,
        "variance": float(np.var(series)) if n > 0 else 0.0,
    }


def compute_autocorr_lags(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes autocorrelation at lag 1 (short-term persistence) and lag 24 (diurnal lock)."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values
    n = len(series)

    if n < 48:
        return {
            "region_id": region_id,
            "autocorr_lag1": 0.5,
            "autocorr_lag24": 0.5,
            "status": "insufficient_data_fallback",
        }

    mean_val = np.mean(series)
    var_val = np.var(series) + 1e-12
    centered = series - mean_val

    def compute_lag_corr(lag: int) -> float:
        c = np.sum(centered[: n - lag] * centered[lag:]) / ((n - lag) * var_val)
        return float(np.clip(c, -1.0, 1.0))

    acf_1 = compute_lag_corr(1)
    acf_24 = compute_lag_corr(24)

    return {
        "region_id": region_id,
        "autocorr_lag1": round(acf_1, 4),
        "autocorr_lag24": round(acf_24, 4),
        "diurnal_lock_strength": "strong" if acf_24 > 0.6 else "moderate" if acf_24 > 0.3 else "weak",
    }


def compute_rhythm_regularity(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes inter-day cosine similarity between consecutive 24-hour diurnal segments."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values
    n = len(series)

    if n < 48:
        return {
            "region_id": region_id,
            "rhythm_regularity": 0.5,
            "status": "insufficient_data_fallback",
        }

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

    return {
        "region_id": region_id,
        "rhythm_regularity": round(regularity, 4),
        "days_compared": len(cos_sims),
        "min_daily_similarity": round(float(np.min(cos_sims)), 4) if cos_sims else 0.85,
    }


def compute_composite_predictability_score(
    region_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes overall predictability score combining lag autocorrelations and regularity."""
    acf_res = compute_autocorr_lags(region_id, context)
    reg_res = compute_rhythm_regularity(region_id, context)

    acf_1 = acf_res.get("autocorr_lag1", 0.5)
    acf_24 = acf_res.get("autocorr_lag24", 0.5)
    regularity = reg_res.get("rhythm_regularity", 0.5)

    score = float(
        np.clip(
            0.4 * max(0.0, acf_1) + 0.3 * max(0.0, acf_24) + 0.3 * regularity,
            0.0,
            1.0,
        )
    )

    return {
        "region_id": region_id,
        "predictability_score": round(score, 4),
        "autocorr_lag1": acf_1,
        "autocorr_lag24": acf_24,
        "rhythm_regularity": regularity,
    }


def build_predictability_tool_registry(
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    snapshot_id: str = "snapshot_predictability_v1",
) -> ToolRegistry:
    """Builds and returns a ToolRegistry populated with predictability tools."""
    allowed = list(urban_context.zone_ids)
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="inspect_predictability_data",
            description="Inspects length and quality of series for autocorrelation analysis.",
            args_schema=PredictabilityQualityArgs,
            fn=lambda region_id, context, min_hours=48: inspect_predictability_data(region_id, context, min_hours),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_autocorr_lags",
            description="Computes lag-1 autocorrelation and lag-24 diurnal lock.",
            args_schema=PredictabilityRegionArgs,
            fn=lambda region_id, context: compute_autocorr_lags(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_rhythm_regularity",
            description="Computes inter-day cosine similarity and cyclic rhythm stability.",
            args_schema=PredictabilityRegionArgs,
            fn=lambda region_id, context: compute_rhythm_regularity(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    registry.register(
        ToolSpec(
            name="compute_composite_predictability_score",
            description="Computes composite predictability score combining persistence, diurnal lock, and regularity.",
            args_schema=PredictabilityRegionArgs,
            fn=lambda region_id, context: compute_composite_predictability_score(region_id, context),
            allowed_regions=allowed,
            data_snapshot=snapshot_id,
            tool_version="1.0.0",
        )
    )

    return registry
