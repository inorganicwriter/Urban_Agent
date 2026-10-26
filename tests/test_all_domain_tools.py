"""Tests for all 5 domain tool suites and corresponding LLM ReAct subagents."""

import pytest
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.orchestration.contracts import Task, TaskStatus
from urban_resilience.orchestration.react_subagent import (
    FlexibilitySubagentLLM,
    SpatialSubagentLLM,
)
from urban_resilience.tools.data_tools import build_data_tool_registry
from urban_resilience.tools.flexibility_tools import build_flexibility_tool_registry
from urban_resilience.tools.functional_tools import build_functional_tool_registry
from urban_resilience.tools.predictability_tools import build_predictability_tool_registry
from urban_resilience.tools.spatial_tools import build_spatial_tool_registry


@pytest.fixture(scope="module")
def city_env():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    urban_context = spatial_loader.load_context()
    mobility_gen = MobilityNetworkLoader(urban_context=urban_context, random_seed=42)
    mobility = mobility_gen.generate_mobility_dataset(total_hours=48)
    return {
        "urban_context": urban_context,
        "mobility_network": mobility,
    }


def test_flexibility_tools_execution(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    registry = build_flexibility_tool_registry(urban_context, mobility)

    ctx = {"urban_context": urban_context, "mobility_network": mobility}
    res_inspect = registry.execute("inspect_flexibility_data", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_inspect.status == "success"
    assert res_inspect.data["valid"] is True
    assert res_inspect.data["total_hours"] == 48

    res_score = registry.execute("compute_composite_flexibility_score", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_score.status == "success"
    assert 0.0 <= res_score.data["intraday_flexibility_score"] <= 1.0


def test_predictability_tools_execution(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    registry = build_predictability_tool_registry(urban_context, mobility)

    ctx = {"urban_context": urban_context, "mobility_network": mobility}
    res_acf = registry.execute("compute_autocorr_lags", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_acf.status == "success"
    assert -1.0 <= res_acf.data["autocorr_lag1"] <= 1.0
    assert -1.0 <= res_acf.data["autocorr_lag24"] <= 1.0

    res_reg = registry.execute("compute_rhythm_regularity", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_reg.status == "success"
    assert 0.0 <= res_reg.data["rhythm_regularity"] <= 1.0


def test_spatial_tools_execution(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    registry = build_spatial_tool_registry(urban_context, mobility)

    ctx = {"urban_context": urban_context, "mobility_network": mobility}
    res_div = registry.execute("compute_destination_diversity", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_div.status == "success"
    assert 0.0 <= res_div.data["destination_diversity"] <= 1.0

    res_route = registry.execute("compute_network_route_redundancy", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_route.status == "success"
    assert res_route.data["degree_connectivity"] >= 1


def test_functional_tools_execution(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    registry = build_functional_tool_registry(urban_context, mobility)

    ctx = {"urban_context": urban_context, "mobility_network": mobility}
    res_poi = registry.execute("compute_poi_diversity_metrics", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_poi.status == "success"
    assert 0.0 <= res_poi.data["poi_entropy"] <= 1.0
    assert res_poi.data["total_poi_count"] > 0

    res_lu = registry.execute("compute_land_use_balance", {"region_id": "SZ_FT_CBD"}, context=ctx)
    assert res_lu.status == "success"
    assert 0.0 <= res_lu.data["activity_balance"] <= 1.0


def test_data_tools_execution(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    registry = build_data_tool_registry(urban_context, mobility)

    ctx = {"urban_context": urban_context, "mobility_network": mobility}
    res_win = registry.execute("inspect_series_window", {"region_id": "SZ_FT_CBD", "window_hours": 24}, context=ctx)
    assert res_win.status == "success"
    assert res_win.data["actual_hours"] == 24

    res_comp = registry.execute("check_citywide_data_completeness", {"region_id": "all"}, context=ctx)
    assert res_comp.status == "success"
    assert res_comp.data["all_zones_complete"] is True


def test_flexibility_subagent_llm(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient()

    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Check flexibility series",
        "tool_call": {"call_id": "c1", "tool": "inspect_flexibility_data", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute shift elasticity",
        "tool_call": {"call_id": "c2", "tool": "compute_peak_shift_elasticity", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute offpeak adaptation",
        "tool_call": {"call_id": "c3", "tool": "compute_offpeak_adaptation", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "final",
        "reasoning": "Flexibility assessment complete",
        "feature_report": None,
    })

    subagent = FlexibilitySubagentLLM(urban_context, mobility, mock_llm)
    task = Task(task_id="t_flex", agent_name="flexibility", action="targeted_inquiry", payload={"region_id": "SZ_FT_CBD"})
    res = subagent.execute(task)

    assert res.status == TaskStatus.SUCCESS
    assert "SZ_FT_CBD" in res.output.zone_metrics
    assert "peak_shift_elasticity" in res.output.zone_metrics["SZ_FT_CBD"]


def test_spatial_subagent_llm(city_env):
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient()

    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute destination diversity",
        "tool_call": {"call_id": "s1", "tool": "compute_destination_diversity", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute route redundancy",
        "tool_call": {"call_id": "s2", "tool": "compute_network_route_redundancy", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "final",
        "reasoning": "Spatial assessment complete",
        "feature_report": None,
    })

    subagent = SpatialSubagentLLM(urban_context, mobility, mock_llm)
    task = Task(task_id="t_spat", agent_name="spatial", action="targeted_inquiry", payload={"region_id": "SZ_FT_CBD"})
    res = subagent.execute(task)

    assert res.status == TaskStatus.SUCCESS
    assert "SZ_FT_CBD" in res.output.zone_metrics
    assert "route_redundancy_score" in res.output.zone_metrics["SZ_FT_CBD"]
