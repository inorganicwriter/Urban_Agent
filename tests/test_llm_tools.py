"""Unit tests for LLM client protocols, mock client, ToolRegistry, and temporal tools."""

from __future__ import annotations

import time
import pytest
from pydantic import BaseModel, Field, ValidationError

from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import EnvironmentConfiguredLLMClient, MockLLMClient
from urban_resilience.llm.schemas import (
    FeatureReport,
    FeatureValue,
)
from urban_resilience.orchestration.react_subagent import feature_reports_to_agent_evidence
from urban_resilience.tools.registry import ToolRegistry, ToolSpec
from urban_resilience.tools.temporal_tools import (
    build_temporal_tool_registry,
    compare_day_types,
    compute_hourly_entropy,
    compute_peak_concentration,
    inspect_data_quality,
    multi_peak_detection,
)


# ---------------------------------------------------------------------------
# 1. MockLLMClient & EnvironmentConfiguredLLMClient Tests
# ---------------------------------------------------------------------------

def test_mock_llm_client_scripted_queue():
    """Validates FIFO queue execution, token tracking, and latency in MockLLMClient."""
    client = MockLLMClient(model="mock-gpt4", default_latency=0.001)

    client.enqueue_response({"action": "call_tool", "tool": "test_tool"})
    client.enqueue_response('{"action": "final", "reasoning": "done"}')

    res1 = client.complete_json([{"role": "user", "content": "hi"}])
    assert res1["action"] == "call_tool"
    assert client.call_count == 1
    assert client.token_usage["total_tokens"] > 0

    res2 = client.complete_json([{"role": "user", "content": "next"}])
    assert res2["action"] == "final"
    assert client.call_count == 2
    assert len(client.history) == 2

    # Queue exhausted
    with pytest.raises(RuntimeError, match="MockLLMClient queue is empty"):
        client.complete_json([{"role": "user", "content": "more"}])


def test_mock_llm_client_generator_and_invalid_json():
    """Validates generator callback and error handling on invalid JSON strings."""
    client = MockLLMClient()

    def dynamic_gen(messages, schema):
        last = messages[-1]["content"]
        return {"action": "echo", "received": last}

    client.set_response_generator(dynamic_gen)
    res = client.complete_json([{"role": "user", "content": "ping"}])
    assert res["action"] == "echo"
    assert res["received"] == "ping"

    # Test invalid JSON string
    client.reset()
    client.enqueue_response("{invalid-json-string")
    with pytest.raises(ValueError, match="MockLLMClient produced invalid JSON string"):
        client.complete_json([{"role": "user", "content": "bad"}])


def test_environment_configured_llm_client_guards(monkeypatch):
    """Validates that EnvironmentConfiguredLLMClient strictly guards against missing API keys."""
    monkeypatch.delenv("URBAN_LLM_API_KEY", raising=False)
    client = EnvironmentConfiguredLLMClient()

    with pytest.raises(RuntimeError, match="URBAN_LLM_API_KEY.*is not configured"):
        client.complete_json([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# 2. ToolRegistry Security & Execution Tests
# ---------------------------------------------------------------------------

class SampleArgs(BaseModel):
    region_id: str
    factor: float = Field(ge=0.0, le=10.0)


def test_tool_registry_whitelist_and_args_validation():
    """Validates that ToolRegistry rejects unregistered tools and malformed arguments."""
    registry = ToolRegistry()

    # 1. Unregistered tool rejection
    unregistered_res = registry.execute("unknown_tool", args={"x": 1})
    assert unregistered_res.status == "error"
    assert "Unregistered tool 'unknown_tool'" in unregistered_res.error

    # Register sample tool
    spec = ToolSpec(
        name="sample_calc",
        description="Multiplies factor by 2",
        fn=lambda region_id, factor: {"computed": factor * 2.0},
        args_schema=SampleArgs,
        allowed_regions=["SZ_FT_CBD", "SZ_NS_TECH"],
        timeout=2.0,
    )
    registry.register(spec)

    # 2. Valid invocation
    valid_res = registry.execute("sample_calc", args={"region_id": "SZ_FT_CBD", "factor": 3.0})
    assert valid_res.status == "success"
    assert valid_res.data["computed"] == 6.0
    assert valid_res.call_id.startswith("call_")

    # 3. Invalid arguments (factor exceeds upper bound 10.0)
    invalid_res = registry.execute("sample_calc", args={"region_id": "SZ_FT_CBD", "factor": 99.0})
    assert invalid_res.status == "error"
    assert "Invalid arguments for tool 'sample_calc'" in invalid_res.error

    # 4. Unauthorized region rejection
    unauthorized_res = registry.execute("sample_calc", args={"region_id": "UNAUTHORIZED_ZONE", "factor": 1.0})
    assert unauthorized_res.status == "error"
    assert "Region 'UNAUTHORIZED_ZONE' is unauthorized" in unauthorized_res.error


def test_tool_registry_timeout():
    """Validates that slow-running tools are interrupted by timeout guard."""
    registry = ToolRegistry()

    def slow_fn(region_id: str):
        time.sleep(0.5)
        return {"done": True}

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="Simulates slow calculation",
            fn=slow_fn,
            timeout=0.05,  # 50ms timeout
        )
    )

    res = registry.execute("slow_tool", args={"region_id": "SZ_FT_CBD"})
    assert res.status == "timeout"
    assert "timed out" in res.error


# ---------------------------------------------------------------------------
# 3. Temporal Tools Execution & Numerical Correctness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_context():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    urban_context = spatial_loader.load_context()
    mobility_loader = MobilityNetworkLoader(urban_context=urban_context, random_seed=42)
    mobility = mobility_loader.generate_mobility_dataset(total_hours=48)
    return {"urban_context": urban_context, "mobility_network": mobility}


def test_temporal_tools_suite(sample_context):
    """Validates that all 5 Temporal tools execute correctly and return structured summaries."""
    zone_id = "SZ_FT_CBD"
    mobility = sample_context["mobility_network"]
    ctx = sample_context

    # 1. inspect_data_quality
    dq = inspect_data_quality(zone_id, ctx)
    assert dq["region_id"] == zone_id
    assert dq["total_hours"] == 48
    assert dq["missing_rate"] == 0.0
    assert dq["quality_grade"] == "A"
    assert dq["mean_activity"] > 0

    # 2. compute_hourly_entropy
    ent = compute_hourly_entropy(zone_id, ctx)
    assert 0.0 <= ent["hourly_entropy"] <= 1.0
    assert 0.0 <= ent["uniformity_ratio"] <= 1.0

    # 3. compute_peak_concentration
    pc = compute_peak_concentration(zone_id, ctx)
    assert 0.0 <= pc["peak_concentration"] <= 1.0
    assert 0.0 <= pc["gini_index"] <= 1.0

    # 4. compare_day_types
    dt = compare_day_types(zone_id, ctx, weekday_count=5)
    assert -1.0 <= dt["day_type_correlation"] <= 1.0
    assert 0.0 <= dt["temporal_dispersion_index"] <= 1.0

    # 5. multi_peak_detection
    mp = multi_peak_detection(zone_id, ctx)
    assert mp["peak_count"] >= 1
    assert 0 <= mp["primary_peak_hour"] <= 23

    # 6. Test build_temporal_tool_registry
    registry = build_temporal_tool_registry(mobility, allowed_regions=[zone_id])
    assert len(registry.list_tools()) == 5
    exec_res = registry.execute("compute_hourly_entropy", args={"region_id": zone_id}, context=ctx)
    assert exec_res.status == "success"
    assert "hourly_entropy" in exec_res.data


# ---------------------------------------------------------------------------
# 4. FeatureReport Evidence References Enforcement
# ---------------------------------------------------------------------------

def test_feature_report_evidence_refs_validation():
    """Validates that FeatureReport strictly requires evidence_refs for complete reports."""
    # 1. Valid report with evidence_refs
    valid_report = FeatureReport(
        agent="TemporalRedundancyAgent",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(
                name="hourly_entropy",
                value=0.88,
                unit="ratio",
                evidence_refs=["call_tool_001"],
            )
        },
    )
    assert valid_report.status == "complete"

    # 2. Invalid report without evidence_refs -> must raise ValidationError
    with pytest.raises(ValidationError, match="empty evidence_refs"):
        FeatureReport(
            agent="TemporalRedundancyAgent",
            region_id="SZ_FT_CBD",
            status="complete",
            features={
                "hourly_entropy": FeatureValue(
                    name="hourly_entropy",
                    value=0.88,
                    unit="ratio",
                    evidence_refs=[],  # Missing references!
                )
            },
        )

    # 3. Status 'insufficient' is permitted with empty features
    insufficient_report = FeatureReport(
        agent="TemporalRedundancyAgent",
        region_id="SZ_FT_CBD",
        status="insufficient",
        features={},
        limitations=["Data series too short"],
    )
    assert insufficient_report.status == "insufficient"


def test_feature_reports_to_agent_evidence():
    """Validates conversion of zone FeatureReports to standard AgentEvidence."""
    rep_cbd = FeatureReport(
        agent="TemporalRedundancyAgent",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(name="hourly_entropy", value=0.85, evidence_refs=["call_1"]),
            "peak_concentration": FeatureValue(name="peak_concentration", value=0.40, evidence_refs=["call_2"]),
        },
    )
    rep_tech = FeatureReport(
        agent="TemporalRedundancyAgent",
        region_id="SZ_NS_TECH",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(name="hourly_entropy", value=0.75, evidence_refs=["call_1"]),
            "peak_concentration": FeatureValue(name="peak_concentration", value=0.50, evidence_refs=["call_2"]),
        },
    )

    evidence = feature_reports_to_agent_evidence(
        agent_name="TemporalRedundancyAgent",
        reports={"SZ_FT_CBD": rep_cbd, "SZ_NS_TECH": rep_tech},
    )

    assert evidence.agent_name == "TemporalRedundancyAgent"
    assert "SZ_FT_CBD" in evidence.zone_metrics
    assert "SZ_NS_TECH" in evidence.zone_metrics
    assert "temporal_redundancy_score" in evidence.zone_metrics["SZ_FT_CBD"]
    assert 0.0 <= evidence.metrics["temporal_redundancy_score"] <= 1.0
