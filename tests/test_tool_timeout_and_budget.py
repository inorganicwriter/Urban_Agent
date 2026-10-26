"""Tests for FIX-6: Non-blocking tool timeout, snapshot_id validation, and token_budget enforcement."""

import time
import pytest
from pydantic import BaseModel

from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.orchestration.contracts import Task
from urban_resilience.orchestration.react_subagent import TemporalSubagentLLM
from urban_resilience.tools.registry import ToolRegistry, ToolSpec


class EmptyArgs(BaseModel):
    pass


def slow_sleep_tool(context=None):
    """Tool that sleeps for 10 seconds."""
    time.sleep(10.0)
    return {"status": "done"}


def fast_tool(context=None):
    return {"result": 42}


@pytest.fixture(scope="module")
def city_env():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = loader.load_context()
    mob_loader = MobilityNetworkLoader(context, random_seed=42)
    mobility = mob_loader.generate_mobility_dataset(total_hours=48)
    return context, mobility


def test_tool_timeout_non_blocking():
    """A tool sleeping for 10s with timeout=1.0s must return within 2.0s without blocking."""
    registry = ToolRegistry(default_timeout_seconds=1.0)
    registry.register(
        ToolSpec(
            name="slow_sleep_tool",
            description="Sleeps 10s",
            args_schema=EmptyArgs,
            fn=slow_sleep_tool,
            timeout=1.0,
        )
    )

    t0 = time.perf_counter()
    res = registry.execute("slow_sleep_tool", {}, call_id="c_sleep")
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.5, f"Execution blocked for {elapsed:.2f}s instead of timing out promptly!"
    assert res.status == "timeout"
    assert "timed out" in res.error


def test_tool_snapshot_id_validation():
    """Verifies that ToolRegistry checks snapshot_id consistency."""
    registry = ToolRegistry(expected_snapshot_id="snapshot_2023_09_07")
    registry.register(
        ToolSpec(
            name="fast_tool",
            description="Fast tool",
            args_schema=EmptyArgs,
            fn=fast_tool,
            data_snapshot="snapshot_2023_09_07",
        )
    )

    # Correct snapshot_id executes
    res_ok = registry.execute("fast_tool", {}, snapshot_id="snapshot_2023_09_07")
    assert res_ok.status == "success"

    # Mismatched snapshot_id is rejected
    res_bad = registry.execute("fast_tool", {}, snapshot_id="invalid_wrong_snapshot")
    assert res_bad.status == "error"
    assert "Data snapshot mismatch" in res_bad.error


def test_subagent_token_budget_exhaustion_degradation(city_env):
    """Verifies that subagent gracefully degrades to insufficient when token_budget is exceeded."""
    context, mobility = city_env
    mock_llm = MockLLMClient(tokens_per_call=500)

    def loop_tool_calls(messages, schema):
        return {
            "action": "call_tool",
            "reasoning": "Still computing",
            "tool_call": {"call_id": "c1", "tool": "inspect_data_quality", "args": {"region_id": "SZ_FT_CBD"}},
        }

    mock_llm.set_response_generator(loop_tool_calls)

    subagent = TemporalSubagentLLM(
        urban_context=context,
        mobility_network=mobility,
        llm_client=mock_llm,
        max_tool_calls=10,
        token_budget=1000,  # Tight token budget: will exceed after 2 calls
    )

    task = Task(
        task_id="t_budget_check",
        agent_name="temporal",
        action="targeted_inquiry",
        payload={"region_id": "SZ_FT_CBD"},
    )

    res = subagent.execute(task)
    report = res.metadata.get("feature_report", {})
    assert report.get("status") == "insufficient"
    assert any("Token budget" in lim for lim in report.get("limitations", []))
