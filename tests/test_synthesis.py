"""Unit tests for ResilienceSynthesisAgent, targeted follow-up, and uncertainty quantification."""

import numpy as np

from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.agents.temporal_redundancy import TemporalRedundancyAgent
from urban_resilience.agents.intraday_flexibility import IntradayFlexibilityAgent
from urban_resilience.agents.predictability import PredictabilityAgent
from urban_resilience.agents.spatial_redundancy import SpatialRedundancyAgent
from urban_resilience.agents.functional_mix import FunctionalMixAgent
from urban_resilience.agents.resilience_synthesis import ResilienceSynthesisAgent
from urban_resilience.types import HazardType


def test_resilience_synthesis_and_follow_up():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()
    mobility_loader = MobilityNetworkLoader(context, random_seed=42)
    mob_net = mobility_loader.generate_mobility_dataset(total_hours=72)

    event_agent = EventIntelligenceAgent(context)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=24,
        duration_hours=12,
        peak_intensity=0.80,
        total_simulation_hours=72,
    )

    agents_pool = {
        "temporal": TemporalRedundancyAgent(context, mob_net),
        "flexibility": IntradayFlexibilityAgent(context, mob_net),
        "predictability": PredictabilityAgent(context, mob_net),
        "spatial": SpatialRedundancyAgent(context, mob_net),
        "functional": FunctionalMixAgent(context, mob_net),
    }

    evidences = {k: agent.analyze() for k, agent in agents_pool.items()}

    synthesis_agent = ResilienceSynthesisAgent(context, mob_net)
    prediction, spatial_outlook, zone_features = synthesis_agent.synthesize(
        evidences=evidences,
        event_context=event_ctx,
        agents_pool=agents_pool,
    )

    # Validate prediction properties
    assert len(prediction.time_hours) == 72
    assert len(prediction.predicted_activity) == 72
    assert np.all(prediction.lower_bound_95 <= prediction.predicted_activity + 1e-5)
    assert np.all(prediction.predicted_activity <= prediction.upper_bound_95 + 1e-5)

    # Disruption metrics
    m = prediction.disruption_metrics
    assert 0.0 < m.maximum_activity_loss < 1.0
    assert m.cumulative_activity_loss > 0.0
    assert m.recovery_time_hours > 0.0

    # Spatial outlook rankings
    assert len(spatial_outlook.zone_rankings) == len(context.zones)
    assert len(spatial_outlook.priority_ranking) == len(context.zones)
    assert len(spatial_outlook.vulnerable_areas) > 0
    # Check that rank 1 has highest vulnerability
    assert spatial_outlook.zone_rankings[0].vulnerability_rank == 1
    assert spatial_outlook.zone_rankings[0].vulnerability_score >= spatial_outlook.zone_rankings[-1].vulnerability_score

    # Check that follow-up traces executed
    assert len(synthesis_agent.follow_up_logs) > 0
