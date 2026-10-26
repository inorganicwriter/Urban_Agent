"""Unit tests for domain-expert agents under the shared ReAct workflow."""

import pytest

from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.agents.temporal_redundancy import TemporalRedundancyAgent
from urban_resilience.agents.intraday_flexibility import IntradayFlexibilityAgent
from urban_resilience.agents.predictability import PredictabilityAgent
from urban_resilience.agents.spatial_redundancy import SpatialRedundancyAgent
from urban_resilience.agents.functional_mix import FunctionalMixAgent


@pytest.fixture(scope="module")
def common_context():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()
    mobility_loader = MobilityNetworkLoader(context, random_seed=42)
    mob_net = mobility_loader.generate_mobility_dataset(total_hours=72)
    return context, mob_net


def test_temporal_redundancy_agent(common_context):
    context, mob_net = common_context
    agent = TemporalRedundancyAgent(context, mob_net)
    evidence = agent.analyze()

    assert evidence.agent_name == "TemporalRedundancyAgent"
    assert 0.0 <= evidence.metrics["mean_hourly_entropy"] <= 1.0
    assert 0.0 <= evidence.metrics["mean_peak_concentration"] <= 1.0
    assert len(evidence.zone_metrics) == len(context.zones)
    assert len(evidence.traces) >= 2  # Has ReAct traces
    assert evidence.confidence > 0.8


def test_intraday_flexibility_agent(common_context):
    context, mob_net = common_context
    agent = IntradayFlexibilityAgent(context, mob_net)
    evidence = agent.analyze()

    assert evidence.agent_name == "IntradayFlexibilityAgent"
    assert 0.0 <= evidence.metrics["mean_intraday_flexibility"] <= 1.0
    assert len(evidence.zone_metrics) == len(context.zones)
    assert len(evidence.traces) >= 2


def test_predictability_agent(common_context):
    context, mob_net = common_context
    agent = PredictabilityAgent(context, mob_net)
    evidence = agent.analyze()

    assert evidence.agent_name == "PredictabilityAgent"
    assert -1.0 <= evidence.metrics["mean_autocorr_lag24"] <= 1.0
    assert 0.0 <= evidence.metrics["mean_rhythm_regularity"] <= 1.0
    assert len(evidence.zone_metrics) == len(context.zones)


def test_spatial_redundancy_agent(common_context):
    context, mob_net = common_context
    agent = SpatialRedundancyAgent(context, mob_net)
    evidence = agent.analyze()

    assert evidence.agent_name == "SpatialRedundancyAgent"
    assert 0.0 <= evidence.metrics["mean_spatial_redundancy"] <= 1.0
    assert len(evidence.zone_metrics) == len(context.zones)


def test_functional_mix_agent(common_context):
    context, mob_net = common_context
    agent = FunctionalMixAgent(context, mob_net)
    evidence = agent.analyze()

    assert evidence.agent_name == "FunctionalMixAgent"
    assert 0.0 <= evidence.metrics["mean_functional_mix_score"] <= 1.0
    assert len(evidence.zone_metrics) == len(context.zones)
