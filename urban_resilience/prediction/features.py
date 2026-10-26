"""Fixed feature schema definition and tabular assembler for multi-agent resilience evidence.

Converts structured multi-agent outputs (FeatureReport collections, AgentEvidence,
and zone spatial attributes) into fixed-column tabular representations for training
and inference, enforcing deterministic column ordering, type validation, and explicit imputation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from urban_resilience.types import AgentEvidence, UrbanContext

logger = logging.getLogger(__name__)

# Standard canonical feature column schema
CANONICAL_FEATURE_COLUMNS = [
    # Temporal domain
    "feat_temporal_hourly_entropy",
    "feat_temporal_peak_concentration",
    "feat_temporal_score",
    # Flexibility domain
    "feat_flexibility_peak_shift_elasticity",
    "feat_flexibility_offpeak_adaptation",
    "feat_flexibility_peak_to_offpeak_ratio",
    "feat_flexibility_score",
    # Predictability domain
    "feat_predictability_autocorr_lag1",
    "feat_predictability_autocorr_lag24",
    "feat_predictability_rhythm_regularity",
    "feat_predictability_score",
    # Spatial domain
    "feat_spatial_destination_diversity",
    "feat_spatial_route_redundancy",
    "feat_spatial_score",
    # Functional domain
    "feat_functional_poi_entropy",
    "feat_functional_simpson_diversity",
    "feat_functional_activity_balance",
    "feat_functional_score",
    # Context / Exposure attributes
    "feat_context_population",
    "feat_context_elevation_mean",
    "feat_context_drainage_capacity",
    "feat_context_hazard_intensity",
]

TARGET_COLUMNS = [
    "target_maximum_activity_loss",
    "target_recovery_time_hours",
    "target_cumulative_activity_loss",
]


class FeatureSchemaMetadata(BaseModel):
    """Metadata describing the feature schema and versioning."""

    schema_version: str = Field(default="1.0.0")
    feature_count: int = Field(default=len(CANONICAL_FEATURE_COLUMNS))
    feature_columns: List[str] = Field(default_factory=lambda: list(CANONICAL_FEATURE_COLUMNS))
    target_columns: List[str] = Field(default_factory=lambda: list(TARGET_COLUMNS))
    imputation_strategy: str = Field(default="neutral_baseline_median")


class FeatureAssembler:
    """Assembles tabular training / prediction matrices from multi-agent evidences."""

    def __init__(self, urban_context: UrbanContext):
        self.urban_context = urban_context
        self.metadata = FeatureSchemaMetadata()
        self.assembly_status: str = "complete"
        self.last_missing_features: List[str] = []

    def assemble_zone_features(
        self,
        evidences: Dict[str, AgentEvidence],
        hazard_intensity: Optional[Dict[str, float]] = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Assembles a fixed-column DataFrame of zone features from multi-agent evidences.

        Args:
            evidences: Dictionary mapping domain name (e.g., 'temporal', 'spatial')
                       or agent name to AgentEvidence.
            hazard_intensity: Optional mapping of zone_id to normalized hazard intensity [0, 1].
            strict: If True, raises ValueError when any domain or feature is missing.
                    If False, marks status as 'insufficient' in DataFrame attrs and metadata.

        Returns:
            pd.DataFrame indexed by zone_id with columns matching CANONICAL_FEATURE_COLUMNS.
        """
        rows = []
        intensity_map = hazard_intensity or {}
        missing_features: List[str] = []

        # Resolve evidences by domain key
        resolved_evidences: Dict[str, AgentEvidence] = {}
        for k, ev in evidences.items():
            name_lower = k.lower()
            if "temporal" in name_lower:
                resolved_evidences["temporal"] = ev
            elif "flexibility" in name_lower:
                resolved_evidences["flexibility"] = ev
            elif "predictability" in name_lower:
                resolved_evidences["predictability"] = ev
            elif "spatial" in name_lower:
                resolved_evidences["spatial"] = ev
            elif "functional" in name_lower:
                resolved_evidences["functional"] = ev

        ev_temp = resolved_evidences.get("temporal")
        ev_flex = resolved_evidences.get("flexibility")
        ev_pred = resolved_evidences.get("predictability")
        ev_spat = resolved_evidences.get("spatial")
        ev_func = resolved_evidences.get("functional")

        for d_name, ev in [
            ("temporal", ev_temp),
            ("flexibility", ev_flex),
            ("predictability", ev_pred),
            ("spatial", ev_spat),
            ("functional", ev_func),
        ]:
            if ev is None:
                missing_features.append(f"domain_{d_name}_missing")

        for zid in self.urban_context.zone_ids:
            zone = self.urban_context.zones[zid]
            row: Dict[str, Any] = {"zone_id": zid}

            def _get_metric(zm: Dict[str, Any], metric_name: str, canonical_name: str) -> float:
                if metric_name not in zm or zm[metric_name] is None:
                    missing_features.append(f"{zid}:{canonical_name}")
                    return float("nan")
                val = zm[metric_name]
                if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val):
                    missing_features.append(f"{zid}:{canonical_name}")
                    return float("nan")
                return float(val)

            # 1. Temporal
            zm_temp = ev_temp.zone_metrics.get(zid, {}) if ev_temp and hasattr(ev_temp, "zone_metrics") else {}
            row["feat_temporal_hourly_entropy"] = _get_metric(zm_temp, "hourly_entropy", "feat_temporal_hourly_entropy")
            row["feat_temporal_peak_concentration"] = _get_metric(zm_temp, "peak_concentration", "feat_temporal_peak_concentration")
            row["feat_temporal_score"] = _get_metric(zm_temp, "temporal_redundancy_score", "feat_temporal_score")

            # 2. Flexibility
            zm_flex = ev_flex.zone_metrics.get(zid, {}) if ev_flex and hasattr(ev_flex, "zone_metrics") else {}
            row["feat_flexibility_peak_shift_elasticity"] = _get_metric(zm_flex, "peak_shift_elasticity", "feat_flexibility_peak_shift_elasticity")
            row["feat_flexibility_offpeak_adaptation"] = _get_metric(zm_flex, "off_peak_adaptation_ratio", "feat_flexibility_offpeak_adaptation")
            row["feat_flexibility_peak_to_offpeak_ratio"] = _get_metric(zm_flex, "peak_to_offpeak_ratio", "feat_flexibility_peak_to_offpeak_ratio")
            row["feat_flexibility_score"] = _get_metric(zm_flex, "intraday_flexibility_score", "feat_flexibility_score")

            # 3. Predictability
            zm_pred = ev_pred.zone_metrics.get(zid, {}) if ev_pred and hasattr(ev_pred, "zone_metrics") else {}
            row["feat_predictability_autocorr_lag1"] = _get_metric(zm_pred, "autocorr_lag1", "feat_predictability_autocorr_lag1")
            row["feat_predictability_autocorr_lag24"] = _get_metric(zm_pred, "autocorr_lag24", "feat_predictability_autocorr_lag24")
            row["feat_predictability_rhythm_regularity"] = _get_metric(zm_pred, "rhythm_regularity", "feat_predictability_rhythm_regularity")
            row["feat_predictability_score"] = _get_metric(zm_pred, "predictability_score", "feat_predictability_score")

            # 4. Spatial
            zm_spat = ev_spat.zone_metrics.get(zid, {}) if ev_spat and hasattr(ev_spat, "zone_metrics") else {}
            row["feat_spatial_destination_diversity"] = _get_metric(zm_spat, "destination_diversity", "feat_spatial_destination_diversity")
            row["feat_spatial_route_redundancy"] = _get_metric(zm_spat, "route_redundancy_score", "feat_spatial_route_redundancy")
            row["feat_spatial_score"] = _get_metric(zm_spat, "spatial_redundancy_score", "feat_spatial_score")

            # 5. Functional
            zm_func = ev_func.zone_metrics.get(zid, {}) if ev_func and hasattr(ev_func, "zone_metrics") else {}
            row["feat_functional_poi_entropy"] = _get_metric(zm_func, "poi_entropy", "feat_functional_poi_entropy")
            row["feat_functional_simpson_diversity"] = _get_metric(zm_func, "simpson_diversity", "feat_functional_simpson_diversity")
            row["feat_functional_activity_balance"] = _get_metric(zm_func, "activity_balance", "feat_functional_activity_balance")
            row["feat_functional_score"] = _get_metric(zm_func, "functional_mix_score", "feat_functional_score")

            # 6. Physical Context & Hazard
            row["feat_context_population"] = float(zone.population)
            row["feat_context_elevation_mean"] = float(getattr(zone, "elevation_m", getattr(zone, "elevation_mean", 10.0)))

            drainage = getattr(zone, "drainage_capacity", None)
            if drainage is None or not isinstance(drainage, (int, float)) or not np.isfinite(drainage):
                missing_features.append(f"{zid}:feat_context_drainage_capacity")
                row["feat_context_drainage_capacity"] = float("nan")
            else:
                row["feat_context_drainage_capacity"] = float(drainage)

            if zid not in intensity_map:
                missing_features.append(f"{zid}:feat_context_hazard_intensity")
                row["feat_context_hazard_intensity"] = float("nan")
            else:
                row["feat_context_hazard_intensity"] = float(intensity_map[zid])

            rows.append(row)

        if missing_features and strict:
            raise ValueError(
                f"FeatureAssembler: missing required features under strict mode: {missing_features[:5]}... "
                f"({len(missing_features)} total missing)"
            )

        df = pd.DataFrame(rows).set_index("zone_id")
        out_df = df[CANONICAL_FEATURE_COLUMNS]
        self.last_missing_features = missing_features

        if missing_features:
            self.metadata.imputation_strategy = "neutral_baseline_median (insufficient_features)"
            out_df.attrs["status"] = "insufficient"
            out_df.attrs["missing_features"] = missing_features
            self.assembly_status = "insufficient"
        else:
            out_df.attrs["status"] = "complete"
            out_df.attrs["missing_features"] = []
            self.assembly_status = "complete"

        return out_df
