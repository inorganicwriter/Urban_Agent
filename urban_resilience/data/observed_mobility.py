"""Pre-event observed mobility adapter ensuring strict ex-ante causality."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from urban_resilience.data.external_loader import ObservedEventDataset
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class ObservedMobilityAdapter:
    """Adapts empirical pre-event observations into strictly ex-ante MobilityNetworks.

    Guarantees that agents and baseline forecast models only observe activity prior to
    event_start (t < event_start). Post-event observations are strictly excluded from
    model inputs to prevent causal leakage.
    """

    def __init__(
        self,
        dataset: ObservedEventDataset,
        urban_context: UrbanContext,
    ):
        self.dataset = dataset
        self.urban_context = urban_context
        self.meta = dataset.meta
        self.event_start_hour = int(self.meta.event_start_hour)
        self.zone_ids = self.urban_context.zone_ids

        # Minimum pre-event observation window validation
        if self.event_start_hour < 24:
            raise ValueError(
                f"Insufficient pre-event observation window: got event_start_hour={self.event_start_hour}. "
                "At least 24 continuous pre-event hours (timestamp < event_start) are required "
                "to infer diurnal patterns under ex-ante causality."
            )

        # Strictly extract only pre-event observations (t < event_start)
        raw_pre = self.dataset.hourly_zone_activity.iloc[: self.event_start_hour].copy()

        # Ensure all model zones exist and columns align
        self.pre_event_activity = pd.DataFrame(index=np.arange(self.event_start_hour), columns=self.zone_ids, dtype=float)
        for zid in self.zone_ids:
            if zid in raw_pre.columns:
                self.pre_event_activity[zid] = raw_pre[zid].values[: self.event_start_hour]
            else:
                self.pre_event_activity[zid] = 0.0

        # Build base timestamps for time labels
        base_time = pd.to_datetime(self.meta.start_time) - pd.Timedelta(hours=self.event_start_hour)
        self.base_time = base_time

        # Metadata tags
        self.model_input_mobility = "observed_pre_event_extended"
        self.pre_event_window_hours = self.event_start_hour
        self.od_source = "synthetic_fallback"
        self.leakage_guard = "post_event_activity_excluded"
        self.predictability_source = "observed_pre_event" if self.event_start_hour >= 48 else "neutral_fallback"
        self.profile_source = "reference_default"
        self.feature_sources: Dict[str, str] = {
            "temporal": "observed_pre_event",
            "flexibility": "observed_pre_event",
            "spatial": "hybrid_observed_with_gravity_od",
            "functional": "static_urban_context",
            "predictability": self.predictability_source,
            "daily_profiles": self.profile_source,
            "weekly_profiles": self.profile_source,
            "od_flows": self.od_source,
        }
        self.hybrid_components: List[str] = [
            k for k, src in self.feature_sources.items()
            if src != "observed_pre_event"
        ]

    def _build_gravity_od_matrices(self, hourly_series: pd.DataFrame, num_hours: int) -> Dict[int, pd.DataFrame]:
        """Calculates OD flow matrices from zone totals using the economic gravity fallback."""
        od_dict: Dict[int, pd.DataFrame] = {}
        dist_df = self.urban_context.zone_distance_matrix

        for h in range(num_hours):
            od_mat = np.zeros((len(self.zone_ids), len(self.zone_ids)), dtype=float)
            for i, origin in enumerate(self.zone_ids):
                total_out = float(hourly_series.loc[h, origin])
                attractions = []
                for dest in self.zone_ids:
                    d = max(1.0, float(dist_df.loc[origin, dest])) if not dist_df.empty else 1.0
                    dest_zone = self.urban_context.zones[dest]
                    dest_weight = (
                        dest_zone.poi_counts.get("office", 10) * 1.5
                        + dest_zone.poi_counts.get("commercial", 10) * 1.2
                        + dest_zone.poi_counts.get("residential", 10) * 0.8
                    )
                    gravity = dest_weight / (d ** 1.35)
                    attractions.append(gravity)

                attractions_arr = np.array(attractions)
                denom = np.sum(attractions_arr)
                probs = attractions_arr / denom if denom > 0 else np.ones(len(self.zone_ids)) / len(self.zone_ids)
                od_mat[i, :] = total_out * probs

            od_dict[h] = pd.DataFrame(od_mat, index=self.zone_ids, columns=self.zone_ids)

        return od_dict

    def _build_profiles(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Derives 24-hour diurnal and weekly profiles from pre-event data or standard reference."""
        # Standard baseline profiles from reference loader
        ref_loader = MobilityNetworkLoader(self.urban_context)
        daily_profiles = ref_loader._build_daily_profiles()
        weekly_profiles = ref_loader._build_weekly_profiles()
        return daily_profiles, weekly_profiles

    def build_pre_event_network(self) -> MobilityNetwork:
        """Constructs MobilityNetwork containing strictly pre-event observations (len = event_start_hour).

        Used by domain subagents to extract pre-disaster redundancy, flexibility, and predictability.
        """
        daily_profiles, weekly_profiles = self._build_profiles()
        od_matrices = self._build_gravity_od_matrices(self.pre_event_activity, self.event_start_hour)

        time_labels = [
            (self.base_time + pd.Timedelta(hours=h)).isoformat()
            for h in range(self.event_start_hour)
        ]

        return MobilityNetwork(
            od_matrices=od_matrices,
            daily_profiles=daily_profiles,
            weekly_profiles=weekly_profiles,
            hourly_time_series=self.pre_event_activity.copy(),
            time_labels=time_labels,
        )

    def build_extended_network(self, total_hours: Optional[int] = None) -> MobilityNetwork:
        """Constructs extended forecast baseline MobilityNetwork over total_hours.

        - Hours 0 <= h < event_start_hour use authentic pre-event observations.
        - Hours h >= event_start_hour repeat the last 24h pre-event diurnal cycle.
        - Zero post-event observations (event or recovery period) enter this network.
        """
        target_hours = total_hours if total_hours is not None else int(self.meta.total_hours)
        daily_profiles, weekly_profiles = self._build_profiles()

        # Build pre-event OD matrices first
        pre_od = self._build_gravity_od_matrices(self.pre_event_activity, self.event_start_hour)

        # Extend hourly series
        extended_df = pd.DataFrame(index=np.arange(target_hours), columns=self.zone_ids, dtype=float)
        extended_od: Dict[int, pd.DataFrame] = {}

        # Last 24-hour cycle of pre-event window
        cycle_start = self.event_start_hour - 24

        for h in range(target_hours):
            if h < self.event_start_hour:
                extended_df.iloc[h] = self.pre_event_activity.iloc[h].values
                extended_od[h] = pre_od[h]
            else:
                cycle_offset = (h - self.event_start_hour) % 24
                source_h = cycle_start + cycle_offset
                extended_df.iloc[h] = self.pre_event_activity.iloc[source_h].values
                extended_od[h] = pre_od[source_h].copy()

        time_labels = [
            (self.base_time + pd.Timedelta(hours=h)).isoformat()
            for h in range(target_hours)
        ]

        return MobilityNetwork(
            od_matrices=extended_od,
            daily_profiles=daily_profiles,
            weekly_profiles=weekly_profiles,
            hourly_time_series=extended_df,
            time_labels=time_labels,
        )
