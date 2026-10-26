"""Comprehensive tests for core code and design fixes:
- P0-1: Natural coordination and follow-up data source (without monkeypatching)
- P0-2: Four-group semantic clarity and alias compatibility
- P0-3: Predictor role downgrade to engineering baseline and disclaimers
- P0-4: Governance & compliance audit metrics
- P1-1: Strict allow_degraded enforcement
- P1-2: Reproducibility caching and replay client
- P1-3: FixtureEventSource checksum validation and normalization
"""

from pathlib import Path
import shutil
import pytest

from urban_resilience.audit import AuditMetrics, compute_audit_metrics
from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import (
    EventIntelligenceAgent,
    FixtureEventSource,
)
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.experiments.runner import (
    ResultsDict,
    run_experiment_group,
    run_four_group_experiments,
)
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.llm.replay import CachingLLMClient, ReplayLLMClient
from urban_resilience.llm.schemas import FeatureReport, FeatureValue
from urban_resilience.orchestration import (
    IncompleteEvidenceError,
    OrchestrationError,
    RunManifest,
    TaskStatus,
)
from urban_resilience.orchestration.contracts import Task, TaskResult
from urban_resilience.orchestration.evidence_bus import (
    ConflictRecord,
    ConflictTargetSelector,
    EvidenceBus,
)
from urban_resilience.orchestration.subagent import BaseSubagent, DomainSubagent
from urban_resilience.types import AgentEvidence, HazardType, PredictionProvenance


@pytest.fixture
def test_env():
    ctx = UrbanSpatialDataLoader(city_name="Shenzhen").load_context()
    mob = MobilityNetworkLoader(urban_context=ctx, random_seed=42).generate_mobility_dataset(total_hours=24)
    evt = EventIntelligenceAgent(urban_context=ctx).create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=6,
        duration_hours=4,
        peak_intensity=0.75,
        total_simulation_hours=24,
    )
    cfg = SimulationConfig(
        total_hours=24,
        event_start_hour=6,
        event_duration_hours=4,
        monte_carlo_samples=3,
        random_seed=42,
    )
    return ctx, mob, evt, cfg


def test_natural_coordination_without_monkeypatch(tmp_path: Path, test_env):
    """P0-1: Verifies natural coordination exercising through query_refined_evidence tool.

    No monkeypatching of production tools (like compute_hourly_entropy) is used.
    """
    ctx, mob, evt, cfg = test_env
    mock_llm = MockLLMClient(auto_fallback=True)

    res = run_experiment_group(
        group="multi_coordinated",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "coord",
        sim_config=cfg,
        engineering_only=True,
        synthetic_review_delegate=True,
    )

    assert res.group == "multi_coordinated"
    assert res.follow_up_executed is True
    assert res.follow_up_count > 0
    assert res.follow_up_success_count > 0
    assert res.coordination_exercised is True
    assert res.coordination_status == "exercised"
    assert res.evidence_changed is True or res.zone_features_changed is True
    assert res.predictor_role == "engineering_baseline"


def test_refined_source_no_delta(tmp_path: Path, test_env):
    """P0-1: Verifies followed_up_no_effect when follow-up data source produces identical metrics."""
    ctx, mob, evt, cfg = test_env
    client = MockLLMClient(auto_fallback=True)
    target_zone = ctx.zone_ids[0]

    def static_reviewer(messages, schema=None):
        user_msg = messages[-1].get("content", "") if messages else ""
        if "review" in user_msg.lower():
            if "round #1" in user_msg.lower() or "round 1" in user_msg.lower():
                return {
                    "action": "delegate",
                    "reasoning": "Delegation to verify no_effect behavior",
                    "tasks": [
                        {
                            "agent_name": "temporal",
                            "action": "targeted_inquiry",
                            "objective": "Targeted inquiry on static zone",
                            "payload": {
                                "region_id": target_zone,
                                "follow_up_source": "synthetic_refined_window",
                                "query_context": {"simulate_no_delta": True},
                            },
                        }
                    ],
                }
            return {"action": "stop", "reasoning": "Done", "tasks": []}
        from urban_resilience.llm.client import default_mock_generator

        return default_mock_generator(messages, schema)

    client.set_response_generator(static_reviewer)

    res = run_experiment_group(
        group="multi_coordinated",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=client,
        output_dir=tmp_path / "no_delta",
        sim_config=cfg,
        engineering_only=True,
        synthetic_review_delegate=False,
    )

    assert res.follow_up_executed is True
    assert res.coordination_status == "followed_up_no_effect"
    assert res.coordination_exercised is False


def test_four_group_semantics(tmp_path: Path, test_env):
    """P0-2: Verifies exact semantic definitions across the 4 canonical groups and alias compatibility."""
    ctx, mob, evt, cfg = test_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "suite_semantics",
        sim_config=cfg,
        engineering_only=True,
    )

    # 1. Alias lookup in ResultsDict
    assert isinstance(suite.results, ResultsDict)
    assert "single_adaptive" in suite.results
    assert "single_domain_adaptive" in suite.results
    assert suite.results["single_adaptive"] is suite.results["single_domain_adaptive"]

    # 2. Semantic properties per group
    res_fixed = suite.results["fixed_features"]
    assert res_fixed.llm_domains == []
    assert res_fixed.coordination_enabled is False
    assert res_fixed.predictor_role == "engineering_baseline"

    res_single = suite.results["single_domain_adaptive"]
    assert res_single.llm_domains == ["temporal"]
    assert res_single.coordination_enabled is False
    assert res_single.predictor_role == "engineering_baseline"

    res_uncoord = suite.results["multi_uncoordinated"]
    assert len(res_uncoord.llm_domains) == 5
    assert res_uncoord.coordination_enabled is False
    assert res_uncoord.predictor_role == "engineering_baseline"

    res_coord = suite.results["multi_coordinated"]
    assert len(res_coord.llm_domains) == 5
    assert res_coord.coordination_enabled is True
    assert res_coord.predictor_role == "engineering_baseline"


def test_strict_allow_degraded(tmp_path: Path, test_env):
    """P1-2: Verifies allow_degraded=False raises on failed/degraded tasks, while allow_degraded=True records them."""
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    ctx, mob, evt, cfg = test_env

    class InjectedFailingSubagent(BaseSubagent):
        def __init__(self):
            super().__init__(name="temporal", capabilities=["temporal_analysis", "domain_analysis"])

        def execute(self, task: Task) -> TaskResult:
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=TaskStatus.FAILED,
                error="Injected domain task failure for strict testing",
            )

    # 1. Strict mode (allow_degraded=False) with failing domain task MUST raise IncompleteEvidenceError or OrchestrationError
    sup_strict = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="deterministic",
        output_dir=tmp_path / "strict_fail",
        sim_config=cfg,
        allow_degraded=False,
    )
    sup_strict.registry.register(InjectedFailingSubagent())
    with pytest.raises((IncompleteEvidenceError, OrchestrationError)):
        sup_strict.run(evt)

    # 2. Relaxed mode (allow_degraded=True) with the same failing task does NOT raise, and records status
    sup_relaxed = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="deterministic",
        output_dir=tmp_path / "relaxed_fail",
        sim_config=cfg,
        allow_degraded=True,
    )
    sup_relaxed.registry.register(InjectedFailingSubagent())
    _pred, _outlook, _zf, manifest = sup_relaxed.run(evt)
    assert manifest.status in (TaskStatus.FAILED, TaskStatus.DEGRADED)
    assert manifest.failure_count > 0 or manifest.degraded_count > 0

    # 3. Experiment runner run_experiment_group with allow_degraded=False raises OrchestrationError on failure
    def failing_client_generator(*args, **kwargs):
        raise RuntimeError("Simulated LLM network crash")

    failing_client_strict = MockLLMClient()
    failing_client_strict.set_response_generator(failing_client_generator)
    with pytest.raises((OrchestrationError, IncompleteEvidenceError)):
        run_experiment_group(
            group="single_adaptive",
            urban_context=ctx,
            mobility_network=mob,
            event_context=evt,
            llm_client=failing_client_strict,
            output_dir=tmp_path / "group_strict_fail",
            sim_config=cfg,
            allow_degraded=False,
            engineering_only=True,
        )

    # 4. Same failing client under allow_degraded=True completes without raising, recording status
    failing_client_relaxed = MockLLMClient()
    failing_client_relaxed.set_response_generator(failing_client_generator)
    res_relaxed = run_experiment_group(
        group="single_adaptive",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=failing_client_relaxed,
        output_dir=tmp_path / "group_relaxed_fail",
        sim_config=cfg,
        allow_degraded=True,
        engineering_only=True,
    )
    assert res_relaxed.status in ("FAILED", "DEGRADED")
    assert res_relaxed.failed_count > 0 or res_relaxed.degraded_count > 0


def test_caching_and_replay_llm_client(tmp_path: Path):
    """P1-2: Verifies CachingLLMClient records to JSONL and ReplayLLMClient matches offline."""
    cache_file = tmp_path / "llm_cache.jsonl"
    mock = MockLLMClient()
    mock.enqueue_response({"action": "call_tool", "tool": "inspect_data_quality"})
    mock.enqueue_response({"action": "final", "feature_report": None})

    # 1. Record mode
    caching_client = CachingLLMClient(underlying_client=mock, cache_file_path=cache_file)
    messages_1 = [{"role": "user", "content": "Inspect region SZ_FT_CBD"}]
    resp_1 = caching_client.complete_json(messages_1)
    assert resp_1["action"] == "call_tool"

    messages_2 = [{"role": "user", "content": "Conclude evaluation"}]
    resp_2 = caching_client.complete_json(messages_2)
    assert resp_2["action"] == "final"
    assert cache_file.exists()

    # 2. Replay mode (completely offline, zero calls to mock client)
    replay_client = ReplayLLMClient(cache_file_path=cache_file)
    replay_resp_1 = replay_client.complete_json(messages_1)
    assert replay_resp_1 == resp_1

    replay_resp_2 = replay_client.complete_json(messages_2)
    assert replay_resp_2 == resp_2

    # 3. Missing query raises RuntimeError
    with pytest.raises(RuntimeError, match="No recorded response found for prompt hash"):
        replay_client.complete_json([{"role": "user", "content": "Unseen query not in cache"}])


def test_fixture_event_source(tmp_path: Path):
    """P2: Verifies FixtureEventSource checksum validation of all fixture files and tampering detection."""
    fixture_dir = Path("tests/fixtures/external/minimal_event")
    source = FixtureEventSource(fixture_dir=fixture_dir)

    # 1. Authentic fixture validates successfully
    assert source.verify_source() is True

    events = source.fetch_events()
    assert len(events) == 1
    raw_event = events[0]
    assert raw_event["event_id"] == "SZ_20230907_MINIMAL_FIXTURE"
    assert raw_event["provenance_status"] == "unverified"
    assert raw_event["source_type"] == "fixture_offline"

    prov = source.provenance()
    assert prov["source_type"] == "fixture_offline"
    assert prov["provenance_status"] == "unverified"

    # Normalize into canonical EventContext
    evt_ctx = source.normalize(raw_event, total_simulation_hours=48)
    assert evt_ctx.event_id == "SZ_20230907_MINIMAL_FIXTURE"
    assert evt_ctx.hazard_type == HazardType.RAIN_STORM
    assert evt_ctx.duration_hours == 12
    assert evt_ctx.peak_intensity == 0.85
    assert len(evt_ctx.zone_intensity_map) == 12

    # 2. Tampering activity.csv fails verification
    tampered_dir = tmp_path / "tampered_fixture"
    shutil.copytree(fixture_dir, tampered_dir)
    with open(tampered_dir / "activity.csv", "a", encoding="utf-8") as f:
        f.write("\n9999,tampered_data,0.0")
    assert FixtureEventSource(fixture_dir=tampered_dir).verify_source() is False

    # 3. Deleting zones.csv fails verification
    (tampered_dir / "zones.csv").unlink()
    assert FixtureEventSource(fixture_dir=tampered_dir).verify_source() is False

    # 4. Modifying event.json fails verification
    tampered_dir2 = tmp_path / "tampered_fixture2"
    shutil.copytree(fixture_dir, tampered_dir2)
    with open(tampered_dir2 / "event.json", "w", encoding="utf-8") as f:
        f.write('{"event_id": "MODIFIED"}')
    assert FixtureEventSource(fixture_dir=tampered_dir2).verify_source() is False


def test_audit_metrics_computation():
    """P1-1: Verifies compute_audit_metrics computes all indicators from actual records."""
    tasks = [
        Task(
            task_id="t1",
            agent_name="temporal",
            action="assess_temporal_dynamics",
            domain="temporal",
            description="task 1",
            status=TaskStatus.SUCCESS,
        ),
        Task(
            task_id="t2",
            agent_name="spatial",
            action="assess_spatial_dynamics",
            domain="spatial",
            description="task 2",
            status=TaskStatus.SUCCESS,
        ),
        Task(
            task_id="t3",
            agent_name="functional",
            action="assess_functional_dynamics",
            domain="functional",
            description="task 3",
            status=TaskStatus.DEGRADED,
        ),
    ]
    tool_calls = [
        {"tool": "compute_entropy", "status": "success"},
        {"tool": "inspect_quality", "status": "error"},
        {"tool": "query_refined", "status": "success"},
    ]
    manifest = RunManifest(
        run_id="run_audit_test",
        plan_id="plan_audit_test",
        status=TaskStatus.SUCCESS,
        tasks=[t.model_dump() for t in tasks],
        tool_calls=tool_calls,
        metadata={
            "review_tokens_used": 150,
            "feature_value_count": 10,
            "feature_with_refs_count": 8,
            "validation_attempts": 5,
            "validation_rejections": 1,
            "conflicts_detected": 4,
            "conflicts_resolved": 3,
        },
    )

    metrics = compute_audit_metrics(
        manifest=manifest,
        tokens_used=500,
        latency_seconds=12.5,
        follow_up_count=2,
        follow_up_delta_count=1,
    )

    assert isinstance(metrics, AuditMetrics)
    assert metrics.evidence_coverage == 0.8  # 8 / 10
    assert metrics.unsupported_numeric_claim_rate == 0.2  # 1 / 5
    assert metrics.anti_fabrication_rejection_count == 1
    assert round(metrics.tool_call_success_rate, 4) == round(2.0 / 3.0, 4)  # 2 success out of 3
    assert metrics.follow_up_yield == 0.5  # 1 delta out of 2 follow-ups
    assert metrics.conflict_resolution_rate == 0.75  # 3 resolved out of 4
    assert round(metrics.degraded_rate, 4) == round(1.0 / 3.0, 4)  # 1 degraded out of 3
    assert metrics.conflicts_detected == 4
    assert metrics.conflicts_resolved == 3
    assert metrics.total_tokens == 500
    assert metrics.total_latency_seconds == 12.5

    d = metrics.to_dict()
    assert d["evidence_coverage"] == 0.8
    assert d["unsupported_numeric_claim_rate"] == 0.2
    assert d["anti_fabrication_rejection_count"] == 1
    assert round(d["tool_call_success_rate"], 4) == round(2.0 / 3.0, 4)
    assert d["total_tokens"] == 500


def test_audit_metrics_none_when_empty():
    """P1-1: Verifies None when there are no tool calls, follow-ups, conflicts, or feature values."""
    manifest = RunManifest(
        run_id="run_empty_test",
        plan_id="plan_empty_test",
        status=TaskStatus.SUCCESS,
        tasks=[],
        tool_calls=[],
        metadata={
            "feature_value_count": 0,
            "feature_with_refs_count": 0,
            "validation_attempts": 0,
            "validation_rejections": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
        },
    )

    metrics = compute_audit_metrics(
        manifest=manifest,
        tokens_used=0,
        latency_seconds=0.1,
        follow_up_count=0,
        follow_up_delta_count=0,
    )

    assert metrics.evidence_coverage is None
    assert metrics.unsupported_numeric_claim_rate is None
    assert metrics.anti_fabrication_rejection_count == 0
    assert metrics.tool_call_success_rate is None
    assert metrics.follow_up_yield is None
    assert metrics.conflict_resolution_rate is None
    assert metrics.degraded_rate is None

    # to_dict retains None without error or converting to 1.0/0.0
    d = metrics.to_dict()
    assert d["evidence_coverage"] is None
    assert d["unsupported_numeric_claim_rate"] is None
    assert d["tool_call_success_rate"] is None
    assert d["follow_up_yield"] is None
    assert d["conflict_resolution_rate"] is None
    assert d["degraded_rate"] is None


def test_anti_fabrication_rejection_metrics(tmp_path: Path, test_env):
    """P1-1: Verifies fabricated claims are rejected and increment anti-fabrication counters."""
    from urban_resilience.orchestration.react_subagent import TemporalSubagentLLM

    ctx, mob, _evt, _cfg = test_env

    # Subagent that hallucinated an unverified number (99.9 vs real)
    mock = MockLLMClient()
    mock.enqueue_response({"action": "call_tool", "tool": "inspect_data_quality"})
    mock.enqueue_response({
        "action": "final",
        "feature_report": FeatureReport(
            agent="temporal",
            region_id=ctx.zone_ids[0],
            status="complete",
            features={
                "hourly_entropy": FeatureValue(
                    name="hourly_entropy",
                    value=999.99,  # Fabricated numeric value!
                    evidence_refs=["call_001"],
                )
            },
        ),
    })

    subagent = TemporalSubagentLLM(
        urban_context=ctx,
        mobility_network=mob,
        llm_client=mock,
        max_tool_calls=2,
    )

    task = Task(
        task_id="t_fab_test",
        agent_name="temporal",
        action="targeted_inquiry",
        payload={"region_id": ctx.zone_ids[0]},
    )
    result = subagent.execute(task)

    # Subagent caught the fabrication and rejected it
    assert subagent.validation_attempts >= 1
    assert subagent.validation_rejections >= 1
    assert result.metadata["validation_rejections"] >= 1


def test_conflict_detection_and_resolution(test_env):
    """P0-1 & P1-1: Verifies EvidenceBus conflict detection, structured tracking, and distinguishing conflicts vs anomalies."""
    ctx, _mob, _evt, _cfg = test_env
    bus = EvidenceBus()

    # 1. Register conflicting signals between spatial and flexibility on zone 0
    target_zone = ctx.zone_ids[0]
    ev_spatial = AgentEvidence(
        agent_name="spatial",
        metrics={},
        zone_metrics={target_zone: {"spatial_redundancy_score": 0.85}},
        traces=[],
        confidence=0.9,
        potential_anomalies=[f"Zone {target_zone}: isolated peak vulnerability anomaly"],
        qualitative_inference="High spatial redundancy",
    )
    ev_flex = AgentEvidence(
        agent_name="flexibility",
        metrics={},
        zone_metrics={target_zone: {"intraday_flexibility_score": 0.20}},
        traces=[],
        confidence=0.9,
        qualitative_inference="Rigid commuting schedules",
    )
    bus.register_evidence("spatial", ev_spatial)
    bus.register_evidence("flexibility", ev_flex)

    conflicts = bus.detect_conflicts(ctx)
    assert len(conflicts) >= 1
    assert any("Conflict detected" in c for c in conflicts)

    # Verify structured conflicts distinguishing
    assert len(bus.cross_domain_conflicts) == 1
    assert isinstance(bus.cross_domain_conflicts[0], ConflictRecord)
    assert bus.cross_domain_conflicts[0].kind == "cross_domain"
    assert bus.cross_domain_conflicts[0].zone_id == target_zone
    assert set(bus.cross_domain_conflicts[0].domains) == {"spatial", "flexibility"}
    assert bus.cross_domain_conflicts[0].targetable is True

    # Anomaly is tracked separately in anomalies and NOT in cross_domain_conflicts
    assert len(bus.anomalies) == 1
    assert bus.anomalies[0].kind == "anomaly"
    assert bus.anomalies[0].zone_id == target_zone
    assert bus.conflicts_detected_count == 1
    assert bus.anomalies_detected_count == 1
    assert bus.unlocalized_conflicts_count == 0

    # Test ConflictTargetSelector
    target = ConflictTargetSelector.select_target(bus.structured_conflicts)
    assert target is not None
    sel_zone, sel_domain, sel_reason = target
    assert sel_zone == target_zone
    assert sel_domain == "flexibility"
    assert "flexibility" in sel_reason

    # 2. Resolution: follow-up updates flexibility to 0.50 (no longer conflicting)
    ev_flex_updated = AgentEvidence(
        agent_name="flexibility",
        metrics={},
        zone_metrics={target_zone: {"intraday_flexibility_score": 0.50}},
        traces=[],
        confidence=0.9,
        qualitative_inference="Updated commuting flexibility after targeted query",
    )
    bus.register_evidence("flexibility", ev_flex_updated)
    bus.detect_conflicts(ctx)
    assert len(bus.cross_domain_conflicts) == 0
    assert bus.conflicts_detected_count == 0
    # Anomaly remains tracked in anomalies
    assert bus.anomalies_detected_count == 1


def test_supervisor_conflict_targeted_coordination_integration(test_env, tmp_path):
    """P0-4: Supervisor-level integration test of cross-domain conflict targeting and resolution."""
    from urban_resilience.llm.client import default_mock_generator
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    ctx, mob, evt, _cfg = test_env
    target_zone = ctx.zone_ids[0]
    unrelated_zone = ctx.zone_ids[1]

    # --- Case 1: Real cross-domain conflict targeted and resolved ---
    client1 = MockLLMClient(auto_fallback=True)
    sup1 = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=client1,
        output_dir=tmp_path / "run_case1",
        enable_follow_ups=True,
    )

    class ConflictingSpatialSub(DomainSubagent):
        def execute(self, task: Task) -> TaskResult:
            ev = AgentEvidence(
                agent_name="spatial",
                metrics={"spatial_redundancy_score": 0.85},
                zone_metrics={z: {"spatial_redundancy_score": 0.85 if z == target_zone else 0.5} for z in ctx.zone_ids},
                traces=[],
                confidence=0.9,
                qualitative_inference="High spatial redundancy",
            )
            return TaskResult(task_id=task.task_id, agent_name="spatial", status=TaskStatus.SUCCESS, output=ev)

    class ConflictingFlexSub(DomainSubagent):
        def execute(self, task: Task) -> TaskResult:
            if task.action == "targeted_inquiry":
                ev = AgentEvidence(
                    agent_name="flexibility",
                    metrics={"intraday_flexibility_score": 0.52},
                    zone_metrics={target_zone: {"intraday_flexibility_score": 0.52}},
                    traces=[],
                    confidence=0.9,
                    qualitative_inference="Refined flexibility via secondary window",
                )
                return TaskResult(task_id=task.task_id, agent_name="flexibility", status=TaskStatus.SUCCESS, output=ev)
            ev = AgentEvidence(
                agent_name="flexibility",
                metrics={"intraday_flexibility_score": 0.20},
                zone_metrics={z: {"intraday_flexibility_score": 0.20 if z == target_zone else 0.5} for z in ctx.zone_ids},
                traces=[],
                confidence=0.9,
                qualitative_inference="Rigid windows",
            )
            return TaskResult(task_id=task.task_id, agent_name="flexibility", status=TaskStatus.SUCCESS, output=ev)

    sup1.registry.register(ConflictingSpatialSub("spatial", ctx, capabilities=["spatial_redundancy"]))
    sup1.registry.register(ConflictingFlexSub("flexibility", ctx, capabilities=["intraday_flexibility", "targeted_inquiry"]))

    def reviewer_gen1(messages, schema=None):
        user_msg = messages[-1].get("content", "") if messages else ""
        if "review" in user_msg.lower():
            if "round #1" in user_msg.lower() or "round 1" in user_msg.lower():
                target = ConflictTargetSelector.select_target(sup1.evidence_bus.structured_conflicts)
                if target:
                    z, d, reason = target
                    return {
                        "action": "delegate",
                        "reasoning": f"conflict_targeted_follow_up: {reason}",
                        "tasks": [{
                            "agent_name": d,
                            "action": "targeted_inquiry",
                            "objective": f"Resolve conflict on {z}",
                            "payload": {"region_id": z, "zone_id": z},
                        }],
                    }
            return {"action": "stop", "reasoning": "Done", "tasks": []}
        return default_mock_generator(messages, schema)

    client1.set_response_generator(reviewer_gen1)
    _pred1, _out1, _zf1, manifest1 = sup1.run(evt, save_manifest=False)

    assert manifest1.metadata.get("conflicts_detected") >= 1
    assert manifest1.metadata.get("conflicts_resolved") >= 1
    assert manifest1.metadata.get("coordination_status") == "exercised"
    resolved_list = manifest1.metadata.get("resolved_conflicts", [])
    assert len(resolved_list) >= 1
    assert resolved_list[0]["zone_id"] == target_zone

    audit1 = compute_audit_metrics(manifest=manifest1)
    assert audit1.conflict_resolution_rate is not None
    assert audit1.conflict_resolution_rate > 0.0

    # --- Case 2: Counter-example - follow-up to unrelated zone does NOT resolve conflict ---
    client2 = MockLLMClient(auto_fallback=True)
    sup2 = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=client2,
        output_dir=tmp_path / "run_case2",
        enable_follow_ups=True,
    )
    sup2.registry.register(ConflictingSpatialSub("spatial", ctx, capabilities=["spatial_redundancy"]))
    sup2.registry.register(ConflictingFlexSub("flexibility", ctx, capabilities=["intraday_flexibility", "targeted_inquiry"]))

    def reviewer_gen2_unrelated(messages, schema=None):
        user_msg = messages[-1].get("content", "") if messages else ""
        if "review" in user_msg.lower():
            if "round #1" in user_msg.lower() or "round 1" in user_msg.lower():
                # Deliberately inquiry unrelated zone
                return {
                    "action": "delegate",
                    "reasoning": "follow_up_on_unrelated_zone",
                    "tasks": [{
                        "agent_name": "flexibility",
                        "action": "targeted_inquiry",
                        "objective": f"Inquire unrelated {unrelated_zone}",
                        "payload": {"region_id": unrelated_zone, "zone_id": unrelated_zone},
                    }],
                }
            return {"action": "stop", "reasoning": "Done", "tasks": []}
        return default_mock_generator(messages, schema)

    client2.set_response_generator(reviewer_gen2_unrelated)
    _pred2, _out2, _zf2, manifest2 = sup2.run(evt, save_manifest=False)

    assert manifest2.metadata.get("conflicts_detected") == 1
    assert manifest2.metadata.get("conflicts_resolved") == 0
    audit2 = compute_audit_metrics(manifest=manifest2)
    assert audit2.conflict_resolution_rate == 0.0

    # --- Case 3: Counter-example - anomaly only does NOT count as cross-domain conflict ---
    client3 = MockLLMClient(auto_fallback=True)
    sup3 = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=client3,
        output_dir=tmp_path / "run_case3",
        enable_follow_ups=True,
    )

    class AnomalyOnlySpatialSub(DomainSubagent):
        def execute(self, task: Task) -> TaskResult:
            ev = AgentEvidence(
                agent_name="spatial",
                metrics={"spatial_redundancy_score": 0.50},
                zone_metrics={z: {"spatial_redundancy_score": 0.50} for z in ctx.zone_ids},
                traces=[],
                confidence=0.9,
                potential_anomalies=[f"Zone {target_zone}: low redundancy anomaly"],
                qualitative_inference="Normal metrics with isolated anomaly",
            )
            return TaskResult(task_id=task.task_id, agent_name="spatial", status=TaskStatus.SUCCESS, output=ev)

    class NormalFlexSub(DomainSubagent):
        def execute(self, task: Task) -> TaskResult:
            ev = AgentEvidence(
                agent_name="flexibility",
                metrics={"intraday_flexibility_score": 0.50},
                zone_metrics={z: {"intraday_flexibility_score": 0.50} for z in ctx.zone_ids},
                traces=[],
                confidence=0.9,
                qualitative_inference="Normal flexibility",
            )
            return TaskResult(task_id=task.task_id, agent_name="flexibility", status=TaskStatus.SUCCESS, output=ev)

    sup3.registry.register(AnomalyOnlySpatialSub("spatial", ctx, capabilities=["spatial_redundancy"]))
    sup3.registry.register(NormalFlexSub("flexibility", ctx, capabilities=["intraday_flexibility", "targeted_inquiry"]))

    _pred3, _out3, _zf3, manifest3 = sup3.run(evt, save_manifest=False)

    assert manifest3.metadata.get("conflicts_detected") == 0
    assert manifest3.metadata.get("conflicts_resolved") == 0
    assert manifest3.metadata.get("anomalies_detected") >= 1
    audit3 = compute_audit_metrics(manifest=manifest3)
    assert audit3.conflict_resolution_rate is None


def test_single_domain_reference_semantics():
    """P1-2: Verifies single_domain_adaptive is documented as single-domain reference, not single-agent baseline."""
    readme_path = Path("README.md")
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    assert "single_domain_adaptive" in content
    assert "Single-domain LLM reference" in content or "single-domain reference" in content.lower()
    assert "NOT a monolithic single-agent baseline" in content
    assert "do NOT use \"single-agent\"" in content or "do not use 'single-agent'" in content.lower()


def test_predictor_role_downgrade_and_disclaimers():
    """P0-3: Verifies predictor downgrade to engineering baseline and disclaimers."""
    prov = PredictionProvenance(summary_metric_source="monte_carlo_physics")
    assert prov.predictor_role == "engineering_baseline"
    assert prov.accuracy_claim == "unverified_on_real_data"

    prov_dict = prov.to_dict()
    assert prov_dict["predictor_role"] == "engineering_baseline"
    assert prov_dict["accuracy_claim"] == "unverified_on_real_data"
