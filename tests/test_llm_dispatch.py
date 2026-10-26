"""Tests for FIX-1: LLM Subagent domain dispatch, exact name matching, and ambiguous fallback errors."""

import pytest

from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.orchestration.contracts import OrchestrationError, Task
from urban_resilience.orchestration.react_subagent import (
    FlexibilitySubagentLLM,
    FunctionalSubagentLLM,
    PredictabilitySubagentLLM,
    SpatialSubagentLLM,
    TemporalSubagentLLM,
)
from urban_resilience.orchestration.subagent import BaseSubagent, SubagentRegistry


@pytest.fixture(scope="module")
def city_env():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = loader.load_context()
    mob_loader = MobilityNetworkLoader(context, random_seed=42)
    mobility = mob_loader.generate_mobility_dataset(total_hours=48)
    return context, mobility


def test_llm_subagents_canonical_naming(city_env):
    """Verifies that all 5 LLM subagents are named canonically."""
    context, mobility = city_env
    mock_llm = MockLLMClient()

    temp = TemporalSubagentLLM(context, mobility, mock_llm)
    flex = FlexibilitySubagentLLM(context, mobility, mock_llm)
    pred = PredictabilitySubagentLLM(context, mobility, mock_llm)
    spat = SpatialSubagentLLM(context, mobility, mock_llm)
    func = FunctionalSubagentLLM(context, mobility, mock_llm)

    assert temp.name == "temporal"
    assert flex.name == "flexibility"
    assert pred.name == "predictability"
    assert spat.name == "spatial"
    assert func.name == "functional"


def test_registry_dispatch_all_five_domains(city_env):
    """Verifies that 5 planner tasks resolve to their respective LLM Subagents."""
    context, mobility = city_env
    mock_llm = MockLLMClient()

    registry = SubagentRegistry()
    registry.register(TemporalSubagentLLM(context, mobility, mock_llm))
    registry.register(FlexibilitySubagentLLM(context, mobility, mock_llm))
    registry.register(PredictabilitySubagentLLM(context, mobility, mock_llm))
    registry.register(SpatialSubagentLLM(context, mobility, mock_llm))
    registry.register(FunctionalSubagentLLM(context, mobility, mock_llm))

    for domain, expected_cls in [
        ("temporal", TemporalSubagentLLM),
        ("flexibility", FlexibilitySubagentLLM),
        ("predictability", PredictabilitySubagentLLM),
        ("spatial", SpatialSubagentLLM),
        ("functional", FunctionalSubagentLLM),
    ]:
        task = Task(task_id=f"t_{domain}", agent_name=domain, action=f"{domain}_analysis")
        matched = registry.find_for_task(task)
        assert isinstance(matched, expected_cls)
        assert matched.name == domain


def test_registry_ambiguous_capability_fallback_raises_error():
    """Verifies that ambiguous capability fallback raises OrchestrationError instead of silently returning the first."""
    registry = SubagentRegistry()

    class SubagentA(BaseSubagent):
        def __init__(self):
            super().__init__(name="subagent_a", capabilities=["shared_capability"])

        def execute(self, task):
            pass

    class SubagentB(BaseSubagent):
        def __init__(self):
            super().__init__(name="subagent_b", capabilities=["shared_capability"])

        def execute(self, task):
            pass

    registry.register(SubagentA())
    registry.register(SubagentB())

    # Task has unknown agent_name, but action matches shared_capability
    task = Task(task_id="t_ambig", agent_name="unknown_agent", action="shared_capability")

    with pytest.raises(OrchestrationError) as exc_info:
        registry.find_for_task(task)

    assert "Ambiguous subagent resolution" in str(exc_info.value)
    assert "subagent_a" in str(exc_info.value)
    assert "subagent_b" in str(exc_info.value)
