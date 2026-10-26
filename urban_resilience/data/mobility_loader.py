"""Mobility network loader simulating historical OD flows, network structures, and temporal rhythms."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import networkx as nx

from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class MobilityNetworkLoader:
    """Generates and manages multi-temporal OD flows, graph networks, and mobility rhythms."""

    def __init__(self, urban_context: UrbanContext, random_seed: int = 42):
        self.urban_context = urban_context
        self.random_seed = random_seed
        self.zone_ids = urban_context.zone_ids
        self.rng = np.random.default_rng(self.random_seed)
        self.od_matrices: Dict[int, pd.DataFrame] = {}

    def generate_mobility_dataset(
        self,
        total_hours: int = 96,
        start_hour_of_day: int = 0,
        start_day_of_week: int = 0,
        start_datetime: Optional[datetime] = None,
    ) -> MobilityNetwork:
        """Constructs complete historical mobility networks, OD matrices, and rhythms."""
        if start_datetime is not None:
            start_hour_of_day = start_datetime.hour
            start_day_of_week = start_datetime.weekday()

        logger.info(
            "Generating historical mobility networks for %d zones over %d hours (start_hour=%d, start_day=%d)...",
            len(self.zone_ids),
            total_hours,
            start_hour_of_day,
            start_day_of_week,
        )

        daily_profiles = self._build_daily_profiles()
        weekly_profiles = self._build_weekly_profiles()
        hourly_series = self._build_hourly_series(
            daily_profiles,
            weekly_profiles,
            total_hours,
            start_hour_of_day=start_hour_of_day,
            start_day_of_week=start_day_of_week,
        )
        od_matrices = self._build_od_matrices(hourly_series, total_hours)
        self.od_matrices = od_matrices
        time_labels = [f"T+{h:03d}h" for h in range(total_hours)]

        return MobilityNetwork(
            od_matrices=od_matrices,
            daily_profiles=daily_profiles,
            weekly_profiles=weekly_profiles,
            hourly_time_series=hourly_series,
            time_labels=time_labels,
        )

    def _build_daily_profiles(self) -> pd.DataFrame:
        """Generates realistic 24-hour diurnal activity curves for weekday and weekend."""
        hours = np.arange(24)
        # Weekday: morning peak (8am) and evening peak (18pm)
        weekday_morning = np.exp(-((hours - 8.2) ** 2) / (2 * (1.6 ** 2)))
        weekday_evening = np.exp(-((hours - 18.0) ** 2) / (2 * (1.8 ** 2)))
        weekday_midday = 0.45 * np.exp(-((hours - 12.5) ** 2) / (2 * (2.5 ** 2)))
        weekday_night = 0.08 + 0.05 * np.sin(np.pi * (hours - 2) / 12)
        weekday_base = 0.10 + 0.70 * weekday_morning + 0.85 * weekday_evening + weekday_midday + weekday_night

        # Weekend: leisurely midday/evening peak (11am - 19pm)
        weekend_afternoon = 0.80 * np.exp(-((hours - 14.5) ** 2) / (2 * (4.0 ** 2)))
        weekend_evening = 0.75 * np.exp(-((hours - 19.5) ** 2) / (2 * (2.5 ** 2)))
        weekend_base = 0.12 + weekend_afternoon + weekend_evening

        df = pd.DataFrame({
            "hour": hours,
            "weekday_ratio": weekday_base / np.mean(weekday_base),
            "weekend_ratio": weekend_base / np.mean(weekend_base),
        })
        return df

    def _build_weekly_profiles(self) -> pd.DataFrame:
        """Weekly modulation factor across days of the week (Monday=0 ... Sunday=6)."""
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        factors = [1.02, 1.05, 1.04, 1.03, 1.15, 0.92, 0.85]
        return pd.DataFrame({"day": days, "factor": factors})

    def _build_hourly_series(
        self,
        daily_profiles: pd.DataFrame,
        weekly_profiles: pd.DataFrame,
        total_hours: int,
        start_hour_of_day: int = 0,
        start_day_of_week: int = 0,
    ) -> pd.DataFrame:
        """Constructs baseline continuous volume for every zone aligned with initial time offset."""
        records = []

        weekday_curve = daily_profiles["weekday_ratio"].values
        weekend_curve = daily_profiles["weekend_ratio"].values
        day_factors = weekly_profiles["factor"].values

        for h in range(total_hours):
            current_h = start_hour_of_day + h
            hour_of_day = current_h % 24
            days_elapsed = current_h // 24
            day_idx = (start_day_of_week + days_elapsed) % 7
            is_weekend = day_idx in [5, 6]
            curve = weekend_curve if is_weekend else weekday_curve

            base_val = curve[hour_of_day] * day_factors[day_idx]

            row = {"hour": h}
            for zid in self.zone_ids:
                zone = self.urban_context.zones[zid]
                # Scale baseline volume by zone population and commercial/office weight
                attractiveness = (zone.population / 100000.0) * (
                    1.2 * zone.poi_counts.get("office", 100) / 1000.0
                    + 1.0 * zone.poi_counts.get("commercial", 100) / 1000.0
                    + 0.8 * zone.poi_counts.get("residential", 100) / 1000.0
                )
                noise = self.rng.normal(1.0, 0.03)  # subtle stochastic noise
                row[zid] = float(max(10.0, base_val * attractiveness * 1500.0 * noise))
            records.append(row)

        df = pd.DataFrame(records)
        df.set_index("hour", inplace=True)
        return df

    def _build_od_matrices(self, hourly_series: pd.DataFrame, total_hours: int) -> Dict[int, pd.DataFrame]:
        """Calculates origin-destination matrix for representative time steps using an economic gravity model."""
        od_dict = {}
        dist_df = self.urban_context.zone_distance_matrix

        # Generate OD matrices for all hours
        for h in range(total_hours):
            od_mat = np.zeros((len(self.zone_ids), len(self.zone_ids)), dtype=float)
            for i, origin in enumerate(self.zone_ids):
                total_out = hourly_series.loc[h, origin]
                # Gravitational attraction to destinations
                attractions = []
                for j, dest in enumerate(self.zone_ids):
                    d = max(1.0, dist_df.loc[origin, dest])
                    dest_zone = self.urban_context.zones[dest]
                    # Job / activity attraction
                    dest_weight = (dest_zone.poi_counts.get("office", 10) * 1.5 +
                                   dest_zone.poi_counts.get("commercial", 10) * 1.2 +
                                   dest_zone.poi_counts.get("residential", 10) * 0.8)
                    gravity = dest_weight / (d ** 1.35)
                    attractions.append(gravity)

                attractions = np.array(attractions)
                probs = attractions / np.sum(attractions)
                od_mat[i, :] = total_out * probs

            od_dict[h] = pd.DataFrame(od_mat, index=self.zone_ids, columns=self.zone_ids)

        return od_dict

    def build_network_graph(self, time_step: int) -> nx.DiGraph:
        """Converts an OD matrix at a given time step into a directed NetworkX graph."""
        if not self.od_matrices:
            raise ValueError("No OD matrices available. Run generate_mobility_dataset first.")

        if time_step in self.od_matrices:
            od = self.od_matrices[time_step]
        else:
            first_key = sorted(self.od_matrices.keys())[0]
            logger.warning("time_step %d not found in od_matrices, falling back to %d", time_step, first_key)
            od = self.od_matrices[first_key]

        G = nx.DiGraph()

        for z in self.zone_ids:
            zone = self.urban_context.zones[z]
            G.add_node(z, name=zone.name, lat=zone.latitude, lon=zone.longitude)

        for origin in self.zone_ids:
            for dest in self.zone_ids:
                if origin != dest:
                    flow = float(od.loc[origin, dest])
                    if flow > 20.0:  # significant flow threshold
                        G.add_edge(origin, dest, weight=flow)
        return G
