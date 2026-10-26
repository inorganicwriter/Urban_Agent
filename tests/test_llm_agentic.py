"""Integration tests for LLM ReAct Temporal Subagent, Supervisor Reviewer, and hybrid orchestration."""

from __future__ import annotations

from pathlib import Path
import pytest

from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.orchestration.contracts import Task, TaskStatus, replay_manifest_execution
from urban_resilience.orchestration.llm_reviewer import LLMSupervisorReviewer
from urban_resilience.orchestration.react_subagent import TemporalSubagentLLM
from urban_resilience.orchestration.supervisor import SupervisorAgent
from urban_resilience.types import HazardType


@pytest.fixture(scope="module")
def city_env():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    urban_context = spatial_loader.load_context()
    mobility_loader = MobilityNetworkLoader(urban_context=urban_context, random_seed=42)
    mobility = mobility_loader.generate_mobility_dataset(total_hours=48)
    event_agent = EventIntelligenceAgent(urban_context=urban_context)
    event_context = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=18,
        duration_hours=12,
        peak_intensity=0.8,
        total_simulation_hours=48,
    )
    return {
        "urban_context": urban_context,
        "mobility_network": mobility,
        "event_context": event_context,
    }


# ---------------------------------------------------------------------------
# 1. TemporalSubagentLLM ReAct Loop Tests
# ---------------------------------------------------------------------------

def test_temporal_subagent_llm_scripted_react_cycle(city_env):
    """Validates complete ReAct tool invocation loop where all metrics originate strictly from tools."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient(model="mock-gpt4")

    # Script responses for zone inquiry:
    # Step 1: call inspect_data_quality
    # Step 2: call compute_hourly_entropy
    # Step 3: call compute_peak_concentration
    # Step 4: final with FeatureReport referencing all calls
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "First inspect data quality",
        "tool_call": {"call_id": "call_01", "tool": "inspect_data_quality", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute diurnal Shannon entropy",
        "tool_call": {"call_id": "call_02", "tool": "compute_hourly_entropy", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "call_tool",
        "reasoning": "Compute peak concentration",
        "tool_call": {"call_id": "call_03", "tool": "compute_peak_concentration", "args": {"region_id": "SZ_FT_CBD"}},
    })
    mock_llm.enqueue_response({
        "action": "final",
        "reasoning": "Extracted verified features with evidence references",
        "feature_report": {
            "agent": "temporal",
            "region_id": "SZ_FT_CBD",
            "status": "complete",
            "features": {
                "hourly_entropy": {
                    "name": "hourly_entropy",
                    "value": 0.9516,
                    "evidence_refs": ["call_02"],
                    "quality": {"coverage": 1.0, "valid_days": 2, "missing_rate": 0.0},
                },
                "peak_concentration": {
                    "name": "peak_concentration",
                    "value": 0.3545,
                    "evidence_refs": ["call_03"],
                    "quality": {"coverage": 1.0, "valid_days": 2, "missing_rate": 0.0},
                },
            },
        },
    })

    subagent = TemporalSubagentLLM(
        urban_context=urban_context,
        mobility_network=mobility,
        llm_client=mock_llm,
    )

    task = Task(
        task_id="task_test_temporal_inquiry",
        agent_name="temporal",
        action="targeted_inquiry",
        payload={"region_id": "SZ_FT_CBD"},
    )

    result = subagent.execute(task)
    assert result.status == TaskStatus.SUCCESS
    assert result.output is not None
    assert "SZ_FT_CBD" in result.output.zone_metrics
    cbd_metrics = result.output.zone_metrics["SZ_FT_CBD"]
    assert cbd_metrics["hourly_entropy"] == pytest.approx(0.9516, abs=1e-3)
    assert cbd_metrics["peak_concentration"] == pytest.approx(0.3545, abs=1e-3)
    assert len(subagent.tool_call_records) == 3
    assert subagent.tool_call_records[0]["call_id"] == "call_01"
    assert subagent.tool_call_records[1]["call_id"] == "call_02"
    assert subagent.tool_call_records[2]["call_id"] == "call_03"


def test_temporal_subagent_llm_max_tool_calls_budget_exceeded(city_env):
    """Validates that exceeding max_tool_calls gracefully degrades to status='insufficient'."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient()

    # Infinite tool loop without reaching 'final'
    def infinite_loop(messages, schema):
        return {
            "action": "call_tool",
            "reasoning": "Still exploring",
            "tool_call": {"call_id": "loop_call", "tool": "inspect_data_quality", "args": {"region_id": "SZ_FT_CBD"}},
        }

    mock_llm.set_response_generator(infinite_loop)

    subagent = TemporalSubagentLLM(
        urban_context=urban_context,
        mobility_network=mobility,
        llm_client=mock_llm,
        max_tool_calls=3,  # Strict cap of 3 tool calls
    )

    task = Task(
        task_id="task_test_budget",
        agent_name="temporal",
        action="targeted_inquiry",
        payload={"region_id": "SZ_FT_CBD"},
    )

    result = subagent.execute(task)
    assert result.status == TaskStatus.DEGRADED
    metadata = result.metadata or {}
    report = metadata.get("feature_report", {})
    assert report.get("status") == "insufficient"
    assert any("max_tool_calls (3) reached" in lim for lim in report.get("limitations", []))


def test_temporal_subagent_llm_insufficient_decision(city_env):
    """Validates that explicit action='insufficient' produces degraded task result."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient()

    mock_llm.enqueue_response({
        "action": "insufficient",
        "reasoning": "Sensor signal corrupted or high missing rate",
        "limitations": ["Severe packet loss in CBD gateway"],
    })

    subagent = TemporalSubagentLLM(
        urban_context=urban_context,
        mobility_network=mobility,
        llm_client=mock_llm,
    )

    task = Task(
        task_id="task_test_insufficient",
        agent_name="temporal",
        action="targeted_inquiry",
        payload={"region_id": "SZ_FT_CBD"},
    )

    result = subagent.execute(task)
    assert result.status == TaskStatus.DEGRADED
    report = result.metadata.get("feature_report", {})
    assert report.get("status") == "insufficient"
    assert "Severe packet loss in CBD gateway" in report.get("limitations", [])


# ---------------------------------------------------------------------------
# 2. LLMSupervisorReviewer Governance & Safeguard Tests
# ---------------------------------------------------------------------------

def test_llm_supervisor_reviewer_delegate_and_whitelist():
    """Validates that LLMSupervisorReviewer enforces agent and action whitelisting."""
    mock_llm = MockLLMClient()
    reviewer = LLMSupervisorReviewer(llm_client=mock_llm, max_rounds=2)

    # Decision proposing 1 valid task and 1 unauthorized task
    mock_llm.enqueue_response({
        "action": "delegate",
        "reasoning": "Inspect CBD rush hour vulnerability",
        "tasks": [
            {
                "agent_name": "temporal",
                "action": "targeted_inquiry",
                "objective": "Check morning rush hour",
                "payload": {"region_id": "SZ_FT_CBD"},
            },
            {
                "agent_name": "UNAUTHORIZED_AGENT",  # Must be rejected!
                "action": "arbitrary_custom_action",
            },
        ],
    })

    import pandas as pd
    dummy_features = pd.DataFrame({"temporal_redundancy_score": [0.8]}, index=["SZ_FT_CBD"])

    decision = reviewer.review_and_decide(
        round_num=1,
        evidence_summary={"temporal": {"status": "complete"}},
        conflicts=[],
        zone_features=dummy_features,
        tokens_spent=100,
    )

    assert decision.action == "delegate"
    assert len(decision.tasks) == 1
    assert decision.tasks[0]["agent_name"] == "temporal"
    assert decision.tasks[0]["action"] == "targeted_inquiry"


def test_llm_supervisor_reviewer_round_and_budget_limits():
    """Validates that max_rounds and budget_token_limit automatically halt follow-up review."""
    mock_llm = MockLLMClient()
    reviewer = LLMSupervisorReviewer(llm_client=mock_llm, max_rounds=2, budget_token_limit=1000)

    import pandas as pd
    dummy_features = pd.DataFrame({"temporal_redundancy_score": [0.8]}, index=["SZ_FT_CBD"])

    # 1. Round 3 (> max_rounds) automatically forces action='stop' without querying LLM
    dec_round3 = reviewer.review_and_decide(
        round_num=3,
        evidence_summary={},
        conflicts=[],
        zone_features=dummy_features,
        tokens_spent=200,
    )
    assert dec_round3.action == "stop"
    assert "Round limit reached" in dec_round3.reasoning
    assert mock_llm.call_count == 0  # Zero LLM call made

    # 2. Budget exhausted forces action='stop'
    dec_budget = reviewer.review_and_decide(
        round_num=1,
        evidence_summary={},
        conflicts=[],
        zone_features=dummy_features,
        tokens_spent=1500,  # Exceeds 1000 budget!
    )
    assert dec_budget.action == "stop"
    assert "budget exhausted" in dec_budget.reasoning.lower()


# ---------------------------------------------------------------------------
# 3. SupervisorAgent in llm_hybrid Mode & Manifest Replay
# ---------------------------------------------------------------------------

def test_supervisor_hybrid_llm_mode_end_to_end_and_replay(tmp_path: Path, city_env):
    """Validates full end-to-end execution of SupervisorAgent in llm_hybrid mode with manifest audit & replay."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    event_context = city_env["event_context"]

    mock_llm = MockLLMClient(model="mock-agentic-v1")

    # Dynamic generator for all zone ReAct calls: inspect -> entropy -> final
    def agent_and_reviewer_gen(messages, schema):
        user_msg = messages[-1]["content"]

        # Supervisor Reviewer call
        if "Round #1 Review:" in user_msg:
            return {
                "action": "delegate",
                "reasoning": "Request targeted check on CBD anomaly",
                "tasks": [
                    {
                        "agent_name": "temporal",
                        "action": "targeted_inquiry",
                        "objective": "Verify CBD profile",
                        "payload": {"region_id": "SZ_FT_CBD"},
                    }
                ],
            }

        # Subagent ReAct calls for all 5 domains
        system_content = messages[0]["content"] if messages else ""
        if "temporal" in system_content.lower():
            tool_1, tool_2 = "inspect_data_quality", "compute_hourly_entropy"
        elif "flexibility" in system_content.lower():
            tool_1, tool_2 = "inspect_flexibility_data", "compute_peak_shift_elasticity"
        elif "predictability" in system_content.lower():
            tool_1, tool_2 = "inspect_predictability_data", "compute_autocorr_lags"
        elif "spatial" in system_content.lower():
            tool_1, tool_2 = "compute_destination_diversity", "compute_network_route_redundancy"
        elif "functional" in system_content.lower():
            tool_1, tool_2 = "compute_poi_diversity_metrics", "compute_land_use_balance"
        else:
            tool_1, tool_2 = "inspect_data_quality", "compute_hourly_entropy"

        content = user_msg
        if "Begin ReAct assessment" in content:
            return {
                "action": "call_tool",
                "reasoning": f"First call {tool_1}",
                "tool_call": {"call_id": "c1", "tool": tool_1, "args": {}},
            }
        elif "Tool observation (call_id='c1'" in content:
            return {
                "action": "call_tool",
                "reasoning": f"Second call {tool_2}",
                "tool_call": {"call_id": "c2", "tool": tool_2, "args": {}},
            }
        elif "Tool observation (call_id='c2'" in content:
            return {
                "action": "final",
                "reasoning": "Complete features extracted",
                "feature_report": None,
            }
        return {"action": "final", "feature_report": None}

    mock_llm.set_response_generator(agent_and_reviewer_gen)

    manifest_path = tmp_path / "hybrid_manifest.json"

    supervisor = SupervisorAgent(
        urban_context=urban_context,
        mobility_network=mobility,
        forecast_mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        output_dir=tmp_path,
        strict=True,
    )

    pred, outlook, zone_feats, manifest = supervisor.run(
        event_context=event_context,
        output_manifest_path=manifest_path,
        save_manifest=True,
    )

    # 1. Output structures validity
    assert pred is not None
    assert outlook is not None
    assert len(zone_feats) >= 12
    assert manifest.status == TaskStatus.SUCCESS
    assert manifest.orchestration_mode == "llm_hybrid"
    assert manifest.llm_model == "mock-agentic-v1"
    assert manifest.prompt_version == "v1.0.0"

    # 2. Telemetry and tool calls recording in manifest
    assert manifest.token_usage.get("total_tokens", 0) > 0
    assert len(manifest.tool_calls) > 0
    assert len(manifest.follow_up_decisions) >= 1
    assert manifest_path.exists()

    # 3. Replay verification
    replay_summary = replay_manifest_execution(manifest)
    assert replay_summary["is_valid"] is True
    assert replay_summary["total_tasks"] > 0
    assert replay_summary["executed_tasks_count"] > 0
    assert replay_summary["tool_calls_count"] > 0
    assert replay_summary["follow_up_decisions_count"] >= 1


def test_deterministic_mode_remains_baseline(city_env):
    """Validates that orchestration_mode='deterministic' uses standard rule-based controller."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    event_context = city_env["event_context"]

    supervisor = SupervisorAgent(
        urban_context=urban_context,
        mobility_network=mobility,
        forecast_mobility_network=mobility,
        orchestration_mode="deterministic",
        llm_client=None,
    )

    pred, outlook, zone_feats, manifest = supervisor.run(
        event_context=event_context,
        save_manifest=False,
    )

    assert manifest.orchestration_mode == "deterministic"
    assert manifest.llm_model is None
    assert len(manifest.follow_up_decisions) == 0
    assert manifest.status == TaskStatus.SUCCESS


def test_supervisor_reviewer_max_rounds_unified(city_env):
    """Validates that supervisor review initializes LLMSupervisorReviewer with max_rounds=2."""
    urban_context = city_env["urban_context"]
    mobility = city_env["mobility_network"]
    mock_llm = MockLLMClient()

    supervisor = SupervisorAgent(
        urban_context=urban_context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
    )
    assert supervisor.llm_reviewer is not None
    assert supervisor.llm_reviewer.max_rounds == 2
