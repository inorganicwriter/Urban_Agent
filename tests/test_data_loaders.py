"""Unit tests for spatial, mobility, and event intelligence data loaders."""

import numpy as np

from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.types import HazardType


def test_spatial_data_loader():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = loader.load_context()

    assert context.city_name == "Shenzhen"
    assert len(context.zones) >= 12
    assert "SZ_FT_CBD" in context.zones
    assert "SZ_NS_TECH" in context.zones

    # Verify road adjacency
    assert len(context.road_adjacency) == len(context.zones)
    for zid, neighbors in context.road_adjacency.items():
        assert len(neighbors) > 0

    # Distance matrix symmetry and diagonal
    dist_df = context.zone_distance_matrix
    assert dist_df.shape == (len(context.zones), len(context.zones))
    assert np.all(np.diag(dist_df.values) >= 0.0)


def test_mobility_network_loader():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()

    mobility_loader = MobilityNetworkLoader(context, random_seed=42)
    mob_net = mobility_loader.generate_mobility_dataset(total_hours=48)

    assert mob_net.hourly_time_series.shape[0] == 48
    assert mob_net.hourly_time_series.shape[1] == len(context.zones)
    assert len(mob_net.od_matrices) == 48

    # Check OD matrix dimensions
    od_0 = mob_net.od_matrices[0]
    assert od_0.shape == (len(context.zones), len(context.zones))
    assert np.all(od_0.values >= 0.0)


def test_event_intelligence_agent():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()

    event_agent = EventIntelligenceAgent(context)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.TYPHOON,
        event_start_hour=24,
        duration_hours=18,
        peak_intensity=0.90,
        total_simulation_hours=72,
    )

    assert event_ctx.event_start_hour == 24
    assert event_ctx.duration_hours == 18
    assert len(event_ctx.hourly_forecast) == 72
    assert len(event_ctx.affected_zones) > 0
    assert event_ctx.peak_intensity == 0.90
    # Before event start, shock forecast must be zero
    assert event_ctx.hourly_forecast[0] == 0.0
    assert event_ctx.hourly_forecast[20] == 0.0
    assert event_ctx.hourly_forecast[33] > 0.0
