"""Deterministic computational tools for Temporal Buffer Redundancy analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.types import MobilityNetwork


class RegionArgs(BaseModel):
    """Arguments requiring a target region ID."""

    region_id: str = Field(description="Target urban zone / region identifier, e.g. 'SZ_FT_CBD'")


class DayTypeArgs(BaseModel):
    """Arguments for comparing weekday vs weekend or cyclic day types."""

    region_id: str = Field(description="Target urban zone identifier")
    weekday_count: int = Field(default=5, ge=1, le=14, description="Number of baseline weekdays in window")


def inspect_data_quality(region_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects raw mobility time series data quality, missing rate, and variance for a zone."""
    mobility: MobilityNetwork = context["mobility_network"]
    if region_id not in mobility.hourly_time_series.columns:
        raise KeyError(f"Region '{region_id}' not found in mobility network time series.")

    series = mobility.hourly_time_series[region_id].values
    total_hours = len(series)
    nan_mask = np.isnan(series)
    missing_count = int(np.sum(nan_mask))
    valid_series = series[~nan_mask]

    if len(valid_series) == 0:
        return {
            "region_id": region_id,
            "total_hours": total_hours,
            "valid_hours": 0,
            "missing_rate": 1.0,
            "zero_rate": 0.0,
            "mean_activity": 0.0,
            "std_activity": 0.0,
            "variance": 0.0,
            "min_activity": 0.0,
            "max_activity": 0.0,
            "quality_grade": "F",
        }

    zero_count = int(np.sum(valid_series <= 1e-6))
    zero_rate = float(zero_count / len(valid_series))
    mean_val = float(np.mean(valid_series))
    var_val = float(np.var(valid_series))
    std_val = float(np.std(valid_series))
    min_val = float(np.min(valid_series))
    max_val = float(np.max(valid_series))

    missing_rate = float(missing_count / total_hours)
    quality_grade = "A" if missing_rate == 0.0 and zero_rate < 0.2 else ("B" if missing_rate < 0.05 else "C")

    return {
        "region_id": region_id,
        "total_hours": total_hours,
        "valid_hours": len(valid_series),
        "missing_rate": missing_rate,
        "zero_rate": zero_rate,
        "mean_activity": mean_val,
        "std_activity": std_val,
        "variance": var_val,
        "min_activity": min_val,
        "max_activity": max_val,
        "quality_grade": quality_grade,
    }


def compute_hourly_entropy(region_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Computes normalized diurnal Shannon entropy of human mobility in a zone."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values

    if len(series) < 24:
        return {
            "region_id": region_id,
            "hourly_entropy": 0.5,
            "uniformity_ratio": 0.5,
            "active_hours_count": 12,
        }

    total = np.sum(series)
    if total <= 1e-6:
        return {
            "region_id": region_id,
            "hourly_entropy": 0.0,
            "uniformity_ratio": 0.0,
            "active_hours_count": 0,
        }

    n_days = max(1, len(series) // 24)
    reshaped = series[: n_days * 24].reshape(n_days, 24)
    avg_24h = np.mean(reshaped, axis=0)

    probs = avg_24h / np.sum(avg_24h)
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(24.0)
    entropy = float(np.clip(entropy, 0.0, 1.0))

    uniformity = float(np.min(avg_24h) / max(np.max(avg_24h), 1e-6))
    active_hours = int(np.sum(avg_24h > 0.1 * np.mean(avg_24h)))

    return {
        "region_id": region_id,
        "hourly_entropy": entropy,
        "uniformity_ratio": uniformity,
        "active_hours_count": active_hours,
    }


def compute_peak_concentration(region_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates peak hour activity ratio, Gini concentration index, and composite score."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values

    if len(series) < 24:
        return {
            "region_id": region_id,
            "peak_concentration": 0.5,
            "peak_ratio": 0.5,
            "gini_index": 0.5,
            "morning_peak_ratio": 0.25,
            "evening_peak_ratio": 0.25,
        }

    n_days = max(1, len(series) // 24)
    avg_24h = np.mean(series[: n_days * 24].reshape(n_days, 24), axis=0)
    total_avg = np.sum(avg_24h)
    if total_avg <= 1e-6:
        return {
            "region_id": region_id,
            "peak_concentration": 1.0,
            "peak_ratio": 1.0,
            "gini_index": 1.0,
            "morning_peak_ratio": 0.5,
            "evening_peak_ratio": 0.5,
        }

    morning_hours = [7, 8, 9]
    evening_hours = [17, 18, 19]
    peak_hours = morning_hours + evening_hours

    morning_ratio = float(np.sum(avg_24h[morning_hours]) / total_avg)
    evening_ratio = float(np.sum(avg_24h[evening_hours]) / total_avg)
    peak_ratio = float(np.sum(avg_24h[peak_hours]) / total_avg)

    sorted_vals = np.sort(avg_24h)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals) + 1e-12))
    gini = float(np.clip(gini, 0.0, 1.0))

    peak_concentration = float(0.5 * peak_ratio + 0.5 * gini)

    return {
        "region_id": region_id,
        "peak_concentration": peak_concentration,
        "peak_ratio": peak_ratio,
        "gini_index": gini,
        "morning_peak_ratio": morning_ratio,
        "evening_peak_ratio": evening_ratio,
    }


def compare_day_types(region_id: str, context: Dict[str, Any], weekday_count: int = 5) -> Dict[str, Any]:
    """Compares weekday vs weekend diurnal profile dispersion and correlation."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values
    total_hours = len(series)
    n_days = total_hours // 24

    if n_days < 2:
        return {
            "region_id": region_id,
            "day_type_correlation": 0.95,
            "weekday_weekend_ratio": 1.0,
            "temporal_dispersion_index": 0.5,
        }

    reshaped = series[: n_days * 24].reshape(n_days, 24)
    # Split weekdays vs weekend (or first half vs second half if fewer than 7 days)
    if n_days >= 7:
        wd_days = reshaped[:weekday_count]
        we_days = reshaped[weekday_count:7]
    else:
        split = max(1, int(n_days * 0.7))
        wd_days = reshaped[:split]
        we_days = reshaped[split:]

    wd_profile = np.mean(wd_days, axis=0)
    we_profile = np.mean(we_days, axis=0) if len(we_days) > 0 else wd_profile

    # Pearson correlation of profiles
    std_prod = np.std(wd_profile) * np.std(we_profile)
    if std_prod > 1e-6:
        corr = float(np.corrcoef(wd_profile, we_profile)[0, 1])
    else:
        corr = 1.0

    wd_mean = float(np.mean(wd_profile))
    we_mean = float(np.mean(we_profile))
    ratio = float(wd_mean / max(we_mean, 1e-6))

    # Dispersion: 1 - |corr| normalized
    dispersion = float(np.clip(1.0 - max(0.0, corr), 0.0, 1.0))

    return {
        "region_id": region_id,
        "day_type_correlation": corr,
        "weekday_weekend_ratio": ratio,
        "temporal_dispersion_index": dispersion,
    }


def multi_peak_detection(region_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Detects diurnal peak modes (unimodal vs bimodal vs multimodal) and prominence."""
    mobility: MobilityNetwork = context["mobility_network"]
    series = mobility.hourly_time_series[region_id].values

    if len(series) < 24:
        return {
            "region_id": region_id,
            "peak_count": 2,
            "bimodal_index": 0.8,
            "primary_peak_hour": 8,
            "secondary_peak_hour": 18,
            "prominence_score": 0.7,
        }

    n_days = max(1, len(series) // 24)
    avg_24h = np.mean(series[: n_days * 24].reshape(n_days, 24), axis=0)

    # Simple local maxima detection with periodic boundary wrapping
    peaks = []
    mean_val = np.mean(avg_24h)
    for i in range(24):
        prev_val = avg_24h[(i - 1) % 24]
        curr_val = avg_24h[i]
        next_val = avg_24h[(i + 1) % 24]
        if curr_val > prev_val and curr_val > next_val and curr_val > mean_val:
            peaks.append((i, curr_val))

    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
    peak_count = len(peaks_sorted)
    primary_peak = peaks_sorted[0][0] if peak_count > 0 else 8
    secondary_peak = peaks_sorted[1][0] if peak_count > 1 else (primary_peak + 10) % 24

    # Bimodal index: how pronounced the top two peaks are relative to minimum between them
    if peak_count >= 2:
        h1, v1 = peaks_sorted[0]
        h2, v2 = peaks_sorted[1]
        mid_h = (h1 + h2) // 2
        mid_val = avg_24h[mid_h]
        trough_depth = max(0.0, (min(v1, v2) - mid_val) / max(v1, 1e-6))
        bimodal_index = float(np.clip(trough_depth, 0.0, 1.0))
        prominence = float(np.clip((v1 - np.min(avg_24h)) / max(v1, 1e-6), 0.0, 1.0))
    else:
        bimodal_index = 0.0
        prominence = float(np.clip((avg_24h[primary_peak] - np.min(avg_24h)) / max(avg_24h[primary_peak], 1e-6), 0.0, 1.0))

    return {
        "region_id": region_id,
        "peak_count": peak_count,
        "bimodal_index": bimodal_index,
        "primary_peak_hour": int(primary_peak),
        "secondary_peak_hour": int(secondary_peak),
        "prominence_score": prominence,
    }


def build_temporal_tool_registry(
    mobility_network: MobilityNetwork,
    allowed_regions: Optional[List[str]] = None,
) -> ToolRegistry:
    """Builds a ToolRegistry populated with the 5 Temporal computational tools."""
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="inspect_data_quality",
            description="Inspects missing rate, zero rate, variance, and sample completeness for a zone's mobility series.",
            fn=lambda region_id, context: inspect_data_quality(region_id, context),
            args_schema=RegionArgs,
            allowed_regions=allowed_regions,
            read_only=True,
            timeout=5.0,
            tool_version="1.0.0",
            units={"missing_rate": "ratio", "mean_activity": "persons/h", "variance": "activity^2"},
        )
    )

    registry.register(
        ToolSpec(
            name="compute_hourly_entropy",
            description="Calculates normalized diurnal Shannon entropy [0, 1] characterizing temporal spreading.",
            fn=lambda region_id, context: compute_hourly_entropy(region_id, context),
            args_schema=RegionArgs,
            allowed_regions=allowed_regions,
            read_only=True,
            timeout=5.0,
            tool_version="1.0.0",
            units={"hourly_entropy": "dimensionless_entropy_ratio"},
        )
    )

    registry.register(
        ToolSpec(
            name="compute_peak_concentration",
            description="Calculates rush-hour peak concentration, peak ratio, and Gini index of the 24h curve.",
            fn=lambda region_id, context: compute_peak_concentration(region_id, context),
            args_schema=RegionArgs,
            allowed_regions=allowed_regions,
            read_only=True,
            timeout=5.0,
            tool_version="1.0.0",
            units={"peak_concentration": "ratio", "gini_index": "index"},
        )
    )

    registry.register(
        ToolSpec(
            name="compare_day_types",
            description="Compares weekday vs weekend profile correlation, ratio, and temporal dispersion.",
            fn=lambda region_id, context, weekday_count=5: compare_day_types(region_id, context, weekday_count),
            args_schema=DayTypeArgs,
            allowed_regions=allowed_regions,
            read_only=True,
            timeout=5.0,
            tool_version="1.0.0",
            units={"day_type_correlation": "pearson_r", "temporal_dispersion_index": "index"},
        )
    )

    registry.register(
        ToolSpec(
            name="multi_peak_detection",
            description="Detects local peak modes, peak hours, and bimodal commuter peak strength.",
            fn=lambda region_id, context: multi_peak_detection(region_id, context),
            args_schema=RegionArgs,
            allowed_regions=allowed_regions,
            read_only=True,
            timeout=5.0,
            tool_version="1.0.0",
            units={"peak_count": "count", "bimodal_index": "index"},
        )
    )

    return registry
