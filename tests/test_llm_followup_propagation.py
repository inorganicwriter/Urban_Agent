"""Tests for FIX-3: Follow-up impact propagation, multi-round review loop, and stop reason tracking."""

from pathlib import Path
import pytest

from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.orchestration.supervisor import SupervisorAgent
from urban_resilience.types import HazardType


@pytest.fixture(scope="module")
def city_env():
    loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = loader.load_context()
    mob_loader = MobilityNetworkLoader(context, random_seed=42)
    mobility = mob_loader.generate_mobility_dataset(total_hours=48)
    event_agent = EventIntelligenceAgent(context)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=6,
        peak_intensity=0.8,
        total_simulation_hours=48,
    )
    return context, mobility, event_ctx


def make_mock_agent_responder(reviewer_func):
    """Helper creating a responder that dispatches domain-correct tools for subagents and delegates review."""
    def responder(messages, schema):
        user_msg = messages[-1]["content"] if messages else ""
        if "Review:" in user_msg:
            return reviewer_func(user_msg)

        system_content = messages[0]["content"] if messages else ""
        if "temporal" in system_content.lower():
            tool_name = "compute_hourly_entropy"
        elif "flexibility" in system_content.lower():
            tool_name = "compute_peak_shift_elasticity"
        elif "predictability" in system_content.lower():
            tool_name = "compute_autocorr_lags"
        elif "spatial" in system_content.lower():
            tool_name = "compute_destination_diversity"
        elif "functional" in system_content.lower():
            tool_name = "compute_poi_diversity_metrics"
        else:
            tool_name = "compute_hourly_entropy"

        if "Begin ReAct assessment" in user_msg:
            return {
                "action": "call_tool",
                "reasoning": f"Calling tool {tool_name}",
                "tool_call": {"call_id": "c1", "tool": tool_name, "args": {}},
            }
        elif "Tool observation (call_id='c1'" in user_msg:
            return {
                "action": "final",
                "reasoning": "Observations complete",
                "feature_report": None,
            }
        return {"action": "final", "feature_report": None}

    return responder


def test_followup_propagates_to_evidence_bus_and_features(tmp_path: Path, city_env):
    """Verifies that follow-up delegate execution updates EvidenceBus and zone_features."""
    context, mobility, event_ctx = city_env
    mock_llm = MockLLMClient()

    review_calls = []

    def reviewer(user_msg):
        if "Round #1 Review:" in user_msg:
            review_calls.append(1)
            return {
                "action": "delegate",
                "reasoning": "Need follow-up check on SZ_FT_CBD temporal redundancy",
                "tasks": [
                    {
                        "agent_name": "temporal",
                        "action": "targeted_inquiry",
                        "objective": "CBD targeted check",
                        "payload": {"region_id": "SZ_FT_CBD"},
                    }
                ],
            }
        elif "Round #2 Review:" in user_msg:
            review_calls.append(2)
            return {
                "action": "stop",
                "reasoning": "Review complete: CBD follow-up verified",
                "tasks": [],
            }
        return {"action": "stop", "reasoning": "Done"}

    mock_llm.set_response_generator(make_mock_agent_responder(reviewer))

    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
        allow_degraded=True,
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    # 1. Reviewer was called for round 1 and round 2
    assert len(review_calls) == 2
    # 2. Stop reason recorded in manifest
    assert manifest.stop_reason == "Review complete: CBD follow-up verified"
    assert manifest.metadata.get("stop_reason") == "Review complete: CBD follow-up verified"
    # 3. Follow-up tasks recorded in follow_up_summary
    assert len(manifest.follow_up_summary) >= 1
    assert manifest.follow_up_summary[0]["agent_name"] == "temporal"
    # 4. EvidenceBus was updated with the follow-up output
    temporal_ev = supervisor.evidence_bus.get_evidence("temporal")
    assert temporal_ev is not None
    assert "SZ_FT_CBD" in temporal_ev.zone_metrics


def test_supervisor_review_two_rounds_max_termination(tmp_path: Path, city_env):
    """Verifies that review loop halts at max_rounds=2 even if reviewer attempts infinite delegation."""
    context, mobility, event_ctx = city_env
    mock_llm = MockLLMClient()

    def reviewer(user_msg):
        return {
            "action": "delegate",
            "reasoning": "Keep investigating indefinitely",
            "tasks": [
                {
                    "agent_name": "spatial",
                    "action": "targeted_inquiry",
                    "objective": "Keep inquiring spatial",
                    "payload": {"region_id": "SZ_FT_CBD"},
                }
            ],
        }

    mock_llm.set_response_generator(make_mock_agent_responder(reviewer))

    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
        allow_degraded=True,
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    # Must terminate and record max_rounds_reached
    assert manifest.stop_reason == "max_rounds_reached"
    rounds = [log["round"] for log in manifest.follow_up_summary]
    assert 1 in rounds
    assert 2 in rounds
    assert 3 not in rounds  # Round 3 never ran


def test_supervisor_review_budget_exhausted_stop_reason(tmp_path: Path, city_env):
    """Verifies that existing subagent tokens do not starve review, and review-phase budget exhaustion halts review."""
    context, mobility, event_ctx = city_env

    # 1. Preset client tokens to 10000 with review budget 5000.
    # Prior subagent tokens MUST NOT starve the review phase.
    mock_llm_not_starved = MockLLMClient()
    mock_llm_not_starved.token_usage["total_tokens"] = 10000

    def reviewer_delegate(user_msg):
        return {
            "action": "delegate",
            "reasoning": "Delegating inquiry without budget starvation",
            "tasks": [{"agent_name": "temporal", "action": "targeted_inquiry", "payload": {"region_id": "SZ_FT_CBD"}}],
        }

    mock_llm_not_starved.set_response_generator(make_mock_agent_responder(reviewer_delegate))

    supervisor_not_starved = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm_not_starved,
        output_dir=tmp_path / "not_starved",
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
        allow_degraded=True,
    )
    supervisor_not_starved.llm_reviewer.budget_token_limit = 5000

    _, _, _, manifest_not_starved = supervisor_not_starved.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    # Assert that prior tokens (10000) did not starve review: round 1 follow-up executed!
    assert manifest_not_starved.stop_reason is None or "budget exhausted" not in manifest_not_starved.stop_reason.lower()
    assert len(manifest_not_starved.follow_up_summary) > 0

    # 2. When review budget is exhausted in the review phase itself (e.g. budget_token_limit = 0)
    mock_llm_exhausted = MockLLMClient()
    mock_llm_exhausted.set_response_generator(make_mock_agent_responder(reviewer_delegate))

    supervisor_exhausted = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm_exhausted,
        output_dir=tmp_path / "exhausted",
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
        allow_degraded=True,
    )
    supervisor_exhausted.llm_reviewer.budget_token_limit = 0  # Zero budget for review phase

    _, _, _, manifest_exhausted = supervisor_exhausted.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    assert "budget exhausted" in manifest_exhausted.stop_reason.lower()
    assert len(manifest_exhausted.follow_up_summary) == 0


def test_followup_updates_target_zone_metrics_and_preserves_other_zones(tmp_path: Path, city_env):
    """Verifies that follow-up updates target zone in zone_features and composite_resilience without polluting other zones."""
    import re
    context, mobility, event_ctx = city_env

    # 1. First run: baseline without follow-up (review stops immediately)
    mock_base = MockLLMClient()
    def base_reviewer(user_msg):
        return {"action": "stop", "reasoning": "No follow-up needed in baseline"}

    mock_base.set_response_generator(make_mock_agent_responder(base_reviewer))
    sup_base = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_base,
        output_dir=tmp_path / "base",
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )
    pred_base, outlook_base, feat_base, _ = sup_base.run(event_context=event_ctx)

    base_cbd_temp = feat_base.loc["SZ_FT_CBD", "temporal_redundancy"]
    base_cbd_comp = feat_base.loc["SZ_FT_CBD", "composite_resilience"]
    base_tech_temp = feat_base.loc["SZ_NS_TECH", "temporal_redundancy"]
    base_tech_comp = feat_base.loc["SZ_NS_TECH", "composite_resilience"]

    # 2. Second run: follow-up on SZ_FT_CBD that successfully updates features
    mock_fu = MockLLMClient()
    fu_inquiry_active = False

    def fu_reviewer(user_msg):
        if "Round #1 Review:" in user_msg:
            return {
                "action": "delegate",
                "reasoning": "Investigate CBD temporal redundancy",
                "tasks": [
                    {
                        "agent_name": "temporal",
                        "action": "targeted_inquiry",
                        "objective": "CBD targeted check",
                        "payload": {"region_id": "SZ_FT_CBD"},
                    }
                ],
            }
        return {"action": "stop", "reasoning": "Follow-up completed"}

    base_resp = make_mock_agent_responder(fu_reviewer)

    observed_entropy = 0.770955
    observed_peak = 0.354468

    def fu_responder(messages, schema):
        nonlocal fu_inquiry_active, observed_entropy, observed_peak
        user_msg = messages[-1]["content"] if messages else ""
        if "Review:" in user_msg:
            return fu_reviewer(user_msg)

        if "Targeted follow-up" in user_msg or "CBD targeted check" in user_msg:
            fu_inquiry_active = True

        if fu_inquiry_active:
            if "Begin ReAct assessment" in user_msg:
                return {
                    "action": "call_tool",
                    "reasoning": "Step 1: compute hourly entropy",
                    "tool_call": {"call_id": "c1", "tool": "compute_hourly_entropy", "args": {"region_id": "SZ_FT_CBD"}},
                }
            elif "Tool observation (call_id='c1'" in user_msg and "SZ_FT_CBD" in user_msg:
                match = re.search(r"'hourly_entropy':\s*([0-9\.]+)", user_msg)
                if match:
                    observed_entropy = float(match.group(1))
                return {
                    "action": "call_tool",
                    "reasoning": "Step 2: compute peak concentration",
                    "tool_call": {"call_id": "c2", "tool": "compute_peak_concentration", "args": {"region_id": "SZ_FT_CBD"}},
                }
            elif "Tool observation (call_id='c2'" in user_msg and "SZ_FT_CBD" in user_msg:
                match = re.search(r"'peak_concentration':\s*([0-9\.]+)", user_msg)
                if match:
                    observed_peak = float(match.group(1))
                return {
                    "action": "final",
                    "reasoning": "Refined multi-tool observations complete",
                    "feature_report": {
                        "agent": "temporal",
                        "region_id": "SZ_FT_CBD",
                        "status": "complete",
                        "features": {
                            "hourly_entropy": {"value": observed_entropy, "evidence_refs": ["c1"]},
                            "peak_concentration": {"value": observed_peak, "evidence_refs": ["c2"]},
                        },
                    },
                }

        return base_resp(messages, schema)

    mock_fu.set_response_generator(fu_responder)
    sup_fu = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_fu,
        output_dir=tmp_path / "fu",
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
    )
    pred_fu, outlook_fu, feat_fu, manifest_fu = sup_fu.run(event_context=event_ctx)

    # Assert target zone metrics changed
    assert feat_fu.loc["SZ_FT_CBD", "temporal_redundancy"] != base_cbd_temp
    assert feat_fu.loc["SZ_FT_CBD", "composite_resilience"] != base_cbd_comp

    # Assert other zones were NOT overwritten or altered
    assert feat_fu.loc["SZ_NS_TECH", "temporal_redundancy"] == base_tech_temp
    assert feat_fu.loc["SZ_NS_TECH", "composite_resilience"] == base_tech_comp

    # Assert EvidenceBus preserves ALL 12 zones for temporal agent (none lost)
    temporal_ev = sup_fu.evidence_bus.get_evidence("temporal")
    assert temporal_ev is not None
    assert len(temporal_ev.zone_metrics) == len(context.zone_ids)
    assert "SZ_FT_CBD" in temporal_ev.zone_metrics
    assert "SZ_NS_TECH" in temporal_ev.zone_metrics


def test_failed_followup_preserves_original_evidence_without_pollution(tmp_path: Path, city_env):
    """Verifies that when a follow-up task fails, EvidenceBus and zone_features remain unpolluted."""
    context, mobility, event_ctx = city_env
    mock_llm = MockLLMClient()

    def reviewer(user_msg):
        if "Round #1 Review:" in user_msg:
            return {
                "action": "delegate",
                "reasoning": "Schedule task that will fail",
                "tasks": [
                    {
                        "agent_name": "temporal",
                        "action": "targeted_inquiry",
                        "objective": "CBD failing check",
                        "payload": {"region_id": "SZ_FT_CBD"},
                    }
                ],
            }
        return {"action": "stop", "reasoning": "Done"}

    base_resp = make_mock_agent_responder(reviewer)

    def responder(messages, schema):
        user_msg = messages[-1]["content"] if messages else ""
        if "Review:" in user_msg:
            return reviewer(user_msg)

        if "CBD failing check" in user_msg:
            # Deliberately raise or fail the tool step so TaskStatus becomes FAILED
            return {
                "action": "call_tool",
                "tool_call": {"call_id": "c1", "tool": "invalid_nonexistent_tool", "args": {}},
            }

        return base_resp(messages, schema)

    mock_llm.set_response_generator(responder)
    supervisor = SupervisorAgent(
        urban_context=context,
        mobility_network=mobility,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        output_dir=tmp_path,
        sim_config=SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6, monte_carlo_samples=10),
        allow_degraded=True,
    )

    pred, outlook, zone_features, manifest = supervisor.run(
        event_context=event_ctx,
        save_manifest=True,
    )

    # Initial evidence remains intact across all zones
    temporal_ev = supervisor.evidence_bus.get_evidence("temporal")
    assert temporal_ev is not None
    assert len(temporal_ev.zone_metrics) == len(context.zone_ids)
    # Zone features for SZ_FT_CBD was preserved and not corrupted
    assert "SZ_FT_CBD" in zone_features.index
    assert zone_features.loc["SZ_FT_CBD", "temporal_redundancy"] > 0.0
