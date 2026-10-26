"""Spatial data loader providing real-world urban context and spatial topology.

Focuses on Shenzhen Metropolitan Area as a representative mega-city case study,
capturing heterogeneous functional zones, multi-modal transport infrastructure,
POI distributions, land-use composition, and topography.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from urban_resilience.types import UrbanContext, UrbanZone

logger = logging.getLogger(__name__)


class UrbanSpatialDataLoader:
    """Loads and synthesizes multi-source spatial urban context."""

    def __init__(self, city_name: str = "Shenzhen"):
        self.city_name = city_name

    def load_context(self) -> UrbanContext:
        """Constructs and returns the comprehensive UrbanContext."""
        logger.info("Loading urban spatial context for %s...", self.city_name)
        zones = self._build_zones()
        road_adjacency = self._build_road_adjacency()
        transit_lines = self._build_transit_lines()
        distance_matrix = self._compute_distance_matrix(zones)

        return UrbanContext(
            city_name=self.city_name,
            zones=zones,
            road_adjacency=road_adjacency,
            transit_lines=transit_lines,
            zone_distance_matrix=distance_matrix,
        )

    def _build_zones(self) -> Dict[str, UrbanZone]:
        """Builds realistic, multi-attribute functional zones in Shenzhen."""
        # Drainage capacity (mm/h) based on synthetic municipal stormwater engineering assumptions
        zone_definitions = [
            {
                "zone_id": "SZ_FT_CBD",
                "name": "Futian Central CBD",
                "latitude": 22.5385,
                "longitude": 114.0558,
                "area_sqkm": 6.8,
                "population": 185000,
                "gdp_per_capita": 280000,
                "road_density": 12.8,
                "transit_stops": 38,
                "elevation_m": 12.5,
                "drainage_capacity": 65.0,
                "poi": {"office": 1420, "commercial": 980, "food": 1650, "residential": 310, "transit": 120, "medical": 28, "education": 22},
                "land_use": {"commercial": 0.45, "office": 0.30, "residential": 0.15, "infrastructure": 0.10},
            },
            {
                "zone_id": "SZ_NS_TECH",
                "name": "Nanshan High-Tech Park (Yuehai)",
                "latitude": 22.5401,
                "longitude": 113.9530,
                "area_sqkm": 11.5,
                "population": 290000,
                "gdp_per_capita": 340000,
                "road_density": 11.2,
                "transit_stops": 42,
                "elevation_m": 15.0,
                "drainage_capacity": 60.0,
                "poi": {"office": 2350, "commercial": 620, "food": 1820, "residential": 450, "transit": 95, "medical": 32, "education": 45},
                "land_use": {"office": 0.52, "commercial": 0.18, "residential": 0.18, "infrastructure": 0.12},
            },
            {
                "zone_id": "SZ_NS_QH",
                "name": "Qianhai Modern Service Zone",
                "latitude": 22.5186,
                "longitude": 113.8992,
                "area_sqkm": 15.0,
                "population": 95000,
                "gdp_per_capita": 310000,
                "road_density": 8.5,
                "transit_stops": 26,
                "elevation_m": 4.5,  # Low coastal elevation, flood vulnerable
                "drainage_capacity": 55.0,
                "poi": {"office": 1100, "commercial": 410, "food": 580, "residential": 210, "transit": 65, "medical": 12, "education": 15},
                "land_use": {"office": 0.40, "commercial": 0.25, "infrastructure": 0.20, "residential": 0.15},
            },
            {
                "zone_id": "SZ_NS_SK",
                "name": "Shekou Port & Coastal Living Zone",
                "latitude": 22.4820,
                "longitude": 113.9168,
                "area_sqkm": 10.8,
                "population": 140000,
                "gdp_per_capita": 240000,
                "road_density": 9.1,
                "transit_stops": 28,
                "elevation_m": 3.8,  # Coastal vulnerable
                "drainage_capacity": 50.0,
                "poi": {"commercial": 580, "residential": 720, "food": 950, "office": 420, "transit": 55, "medical": 18, "education": 30},
                "land_use": {"residential": 0.42, "commercial": 0.28, "port_industrial": 0.20, "infrastructure": 0.10},
            },
            {
                "zone_id": "SZ_LH_COMM",
                "name": "Luohu Dongmen-Guomao Commercial",
                "latitude": 22.5460,
                "longitude": 114.1189,
                "area_sqkm": 8.2,
                "population": 260000,
                "gdp_per_capita": 195000,
                "road_density": 13.5,
                "transit_stops": 45,
                "elevation_m": 8.2,  # Low elevation, old urban drainage
                "drainage_capacity": 45.0,
                "poi": {"commercial": 1850, "food": 2100, "residential": 890, "office": 780, "transit": 110, "medical": 35, "education": 28},
                "land_use": {"commercial": 0.48, "residential": 0.32, "office": 0.12, "infrastructure": 0.08},
            },
            {
                "zone_id": "SZ_BA_CTR",
                "name": "Baoan Central District",
                "latitude": 22.5532,
                "longitude": 113.8830,
                "area_sqkm": 14.6,
                "population": 380000,
                "gdp_per_capita": 180000,
                "road_density": 9.8,
                "transit_stops": 36,
                "elevation_m": 5.2,
                "drainage_capacity": 50.0,
                "poi": {"residential": 1450, "commercial": 890, "food": 1420, "office": 650, "transit": 80, "medical": 42, "education": 60},
                "land_use": {"residential": 0.45, "commercial": 0.25, "office": 0.18, "infrastructure": 0.12},
            },
            {
                "zone_id": "SZ_LG_BT",
                "name": "Longgang Bantian ICT Base",
                "latitude": 22.6285,
                "longitude": 114.0662,
                "area_sqkm": 28.5,
                "population": 460000,
                "gdp_per_capita": 260000,
                "road_density": 8.2,
                "transit_stops": 32,
                "elevation_m": 42.0,  # Inland higher terrain
                "drainage_capacity": 45.0,
                "poi": {"office": 1890, "residential": 1680, "food": 1540, "commercial": 720, "transit": 70, "medical": 38, "education": 52},
                "land_use": {"industrial_office": 0.46, "residential": 0.34, "commercial": 0.12, "infrastructure": 0.08},
            },
            {
                "zone_id": "SZ_LH_RES",
                "name": "Longhua Minzhi / North Railway Hub",
                "latitude": 22.6105,
                "longitude": 114.0298,
                "area_sqkm": 18.2,
                "population": 520000,
                "gdp_per_capita": 165000,
                "road_density": 10.4,
                "transit_stops": 52,
                "elevation_m": 35.0,
                "drainage_capacity": 45.0,
                "poi": {"residential": 2400, "transit": 160, "food": 1850, "commercial": 920, "office": 510, "medical": 45, "education": 75},
                "land_use": {"residential": 0.58, "infrastructure_transit": 0.22, "commercial": 0.14, "office": 0.06},
            },
            {
                "zone_id": "SZ_YT_PORT",
                "name": "Yantian Deep-water Port & Logistics",
                "latitude": 22.5810,
                "longitude": 114.2625,
                "area_sqkm": 21.0,
                "population": 115000,
                "gdp_per_capita": 210000,
                "road_density": 6.4,  # Mountainous and bottleneck access
                "transit_stops": 18,
                "elevation_m": 6.5,
                "drainage_capacity": 50.0,
                "poi": {"port_logistics": 850, "residential": 320, "food": 380, "commercial": 210, "transit": 30, "office": 220, "medical": 14},
                "land_use": {"port_industrial": 0.55, "residential": 0.20, "nature_mountain": 0.18, "commercial": 0.07},
            },
            {
                "zone_id": "SZ_GM_SCI",
                "name": "Guangming Science City",
                "latitude": 22.7538,
                "longitude": 113.9160,
                "area_sqkm": 35.0,
                "population": 220000,
                "gdp_per_capita": 175000,
                "road_density": 7.1,
                "transit_stops": 22,
                "elevation_m": 48.0,
                "drainage_capacity": 45.0,
                "poi": {"office": 650, "education": 95, "residential": 850, "food": 720, "commercial": 340, "transit": 35, "medical": 20},
                "land_use": {"office_research": 0.40, "residential": 0.30, "ecological": 0.20, "commercial": 0.10},
            },
            {
                "zone_id": "SZ_PS_IND",
                "name": "Pingshan Biomedical & EV Hub",
                "latitude": 22.6982,
                "longitude": 114.3412,
                "area_sqkm": 42.0,
                "population": 280000,
                "gdp_per_capita": 160000,
                "road_density": 5.8,  # Lower network redundancy
                "transit_stops": 20,
                "elevation_m": 55.0,
                "drainage_capacity": 40.0,
                "poi": {"office": 920, "residential": 980, "food": 680, "commercial": 310, "transit": 28, "medical": 22, "education": 35},
                "land_use": {"industrial": 0.58, "residential": 0.26, "commercial": 0.08, "infrastructure": 0.08},
            },
            {
                "zone_id": "SZ_DP_ECO",
                "name": "Dapeng Peninsula Eco-Tourism",
                "latitude": 22.5950,
                "longitude": 114.4820,
                "area_sqkm": 68.0,
                "population": 75000,
                "gdp_per_capita": 140000,
                "road_density": 3.2,  # Single-corridor vulnerability
                "transit_stops": 12,
                "elevation_m": 8.0,  # Coastal surge vulnerable
                "drainage_capacity": 30.0,
                "poi": {"commercial": 240, "food": 450, "residential": 280, "transit": 15, "office": 60, "medical": 8, "education": 12},
                "land_use": {"ecological_tourism": 0.75, "residential": 0.15, "commercial": 0.07, "infrastructure": 0.03},
            },
        ]

        zones = {}
        for zd in zone_definitions:
            zone = UrbanZone(
                zone_id=zd["zone_id"],
                name=zd["name"],
                latitude=zd["latitude"],
                longitude=zd["longitude"],
                area_sqkm=zd["area_sqkm"],
                population=zd["population"],
                gdp_per_capita=zd["gdp_per_capita"],
                road_density_km_per_sqkm=zd["road_density"],
                transit_stop_count=zd["transit_stops"],
                elevation_m=zd["elevation_m"],
                poi_counts=zd["poi"],
                land_use_mix=zd["land_use"],
                drainage_capacity=zd.get("drainage_capacity", 50.0),
            )
            zones[zone.zone_id] = zone
        return zones

    def _build_road_adjacency(self) -> Dict[str, List[str]]:
        """Physical arterial road connections and expressway corridors between zones."""
        return {
            "SZ_FT_CBD": ["SZ_NS_TECH", "SZ_LH_COMM", "SZ_LH_RES", "SZ_LG_BT"],
            "SZ_NS_TECH": ["SZ_FT_CBD", "SZ_NS_QH", "SZ_NS_SK", "SZ_BA_CTR", "SZ_LH_RES"],
            "SZ_NS_QH": ["SZ_NS_TECH", "SZ_NS_SK", "SZ_BA_CTR"],
            "SZ_NS_SK": ["SZ_NS_TECH", "SZ_NS_QH"],
            "SZ_LH_COMM": ["SZ_FT_CBD", "SZ_LG_BT", "SZ_YT_PORT"],
            "SZ_BA_CTR": ["SZ_NS_TECH", "SZ_NS_QH", "SZ_GM_SCI"],
            "SZ_LG_BT": ["SZ_FT_CBD", "SZ_LH_COMM", "SZ_LH_RES", "SZ_PS_IND"],
            "SZ_LH_RES": ["SZ_FT_CBD", "SZ_NS_TECH", "SZ_LG_BT", "SZ_GM_SCI"],
            "SZ_YT_PORT": ["SZ_LH_COMM", "SZ_DP_ECO", "SZ_PS_IND"],
            "SZ_GM_SCI": ["SZ_BA_CTR", "SZ_LH_RES"],
            "SZ_PS_IND": ["SZ_LG_BT", "SZ_YT_PORT", "SZ_DP_ECO"],
            "SZ_DP_ECO": ["SZ_YT_PORT", "SZ_PS_IND"],  # Coastal cul-de-sac
        }

    def _build_transit_lines(self) -> List[Dict[str, Any]]:
        """Key metro rail trunk lines connecting zones."""
        return [
            {"line_name": "Line 1 (Luobao)", "zones": ["SZ_LH_COMM", "SZ_FT_CBD", "SZ_NS_TECH", "SZ_NS_QH", "SZ_BA_CTR"]},
            {"line_name": "Line 2/8 (Shekou-Yantian)", "zones": ["SZ_NS_SK", "SZ_NS_TECH", "SZ_FT_CBD", "SZ_LH_COMM", "SZ_YT_PORT"]},
            {"line_name": "Line 4 (Longhua-Futian)", "zones": ["SZ_FT_CBD", "SZ_LH_RES"]},
            {"line_name": "Line 5 (Huanzhong Loop)", "zones": ["SZ_NS_QH", "SZ_BA_CTR", "SZ_LH_RES", "SZ_LG_BT", "SZ_LH_COMM"]},
            {"line_name": "Line 11 (Airport Rapid)", "zones": ["SZ_FT_CBD", "SZ_NS_TECH", "SZ_NS_QH", "SZ_BA_CTR"]},
            {"line_name": "Line 14 (Pingshan Express)", "zones": ["SZ_FT_CBD", "SZ_LH_COMM", "SZ_PS_IND"]},
        ]

    def _compute_distance_matrix(self, zones: Dict[str, UrbanZone]) -> pd.DataFrame:
        """Computes pairwise great-circle distance matrix (km) between all zones."""
        zone_ids = list(zones.keys())
        n = len(zone_ids)
        dist = np.zeros((n, n), dtype=float)

        def haversine_km(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        for i, zid1 in enumerate(zone_ids):
            z1 = zones[zid1]
            for j, zid2 in enumerate(zone_ids):
                if i == j:
                    dist[i, j] = 0.5 * np.sqrt(z1.area_sqkm / np.pi)  # Internal intra-zone travel radius
                else:
                    z2 = zones[zid2]
                    dist[i, j] = haversine_km(z1.latitude, z1.longitude, z2.latitude, z2.longitude)

        return pd.DataFrame(dist, index=zone_ids, columns=zone_ids)
