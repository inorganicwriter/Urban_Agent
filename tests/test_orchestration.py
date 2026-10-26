"""Comprehensive unit and integration tests for the hierarchical orchestration architecture."""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from urban_resilience.config import AgentWeightConfig, SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.orchestration import (
    BaseSubagent,
    DeterministicTaskPlanner,
    EvidenceBus,
    ExecutionPlan,
    FollowUpController,
    IncompleteEvidenceError,
    OrchestrationError,
    SubagentRegistry,
    SupervisorAgent,
    Task,
    TaskResult,
    TaskScheduler,
    TaskStatus,
)
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.types import AgentEvidence, HazardType, ReActTrace


@pytest.fixture
def test_context_and_network():
    """Loads a fast, reproducible context and mobility network for orchestration testing."""
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()
    mob_loader = MobilityNetworkLoader(context, random_seed=42)
    mob_net = mob_loader.generate_mobility_dataset(total_hours=48)
    return context, mob_net


@pytest.fixture
def test_event_context(test_context_and_network):
    context, _ = test_context_and_network
    event_agent = EventIntelligenceAgent(context)
    return event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        peak_intensity=0.85,
        total_simulation_hours=48,
    )


def test_planner_dag_generation_and_validation():
    """Validates planner DAG construction, task dependencies, and graph validation."""
    planner = DeterministicTaskPlanner()

    # Domain tasks DAG
    domain_plan = planner.plan_domain_tasks(plan_id="test_domain_plan")
    assert len(domain_plan.tasks) == 5
    task_ids = [t.task_id for t in domain_plan.tasks]
    assert "task_temporal" in task_ids
    assert "task_flexibility" in task_ids
    assert "task_predictability" in task_ids
    assert "task_spatial" in task_ids
    assert "task_functional" in task_ids
    for t in domain_plan.tasks:
        assert t.dependencies == []

    # Full pipeline DAG
    full_plan = planner.plan_full_pipeline(plan_id="test_full_plan")
    assert len(full_plan.tasks) == 6
    synthesis_task = full_plan.get_task("task_synthesis")
    assert synthesis_task is not None
    assert set(synthesis_task.dependencies) == set(task_ids)

    # Validation: duplicate task IDs
    dup_plan = ExecutionPlan(
        plan_id="dup_plan",
        tasks=[
            Task(task_id="t1", agent_name="temporal", action="domain_analysis"),
            Task(task_id="t1", agent_name="flexibility", action="domain_analysis"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate task IDs"):
        dup_plan.validate_dag()

    # Validation: missing dependency
    missing_dep_plan = ExecutionPlan(
        plan_id="missing_dep",
        tasks=[
            Task(task_id="t1", agent_name="temporal", action="domain_analysis", dependencies=["t_nonexistent"]),
        ],
    )
    with pytest.raises(ValueError, match="depends on non-existent task"):
        missing_dep_plan.validate_dag()

    # Validation: cyclic dependency
    cycle_plan = ExecutionPlan(
        plan_id="cycle_plan",
        tasks=[
            Task(task_id="t1", agent_name="temporal", action="domain_analysis", dependencies=["t2"]),
            Task(task_id="t2", agent_name="flexibility", action="domain_analysis", dependencies=["t1"]),
        ],
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        cycle_plan.validate_dag()


def test_scheduler_topological_execution_and_parallel_consistency(test_context_and_network):
    """Verifies that single-threaded and multi-threaded scheduler runs produce numerically identical results."""
    context, mob_net = test_context_and_network

    # Serial scheduler
    sup_serial = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        max_workers=1,
    )
    evidences_serial = sup_serial.execute_domain_analysis()

    # Parallel scheduler (4 workers)
    sup_parallel = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        max_workers=4,
    )
    evidences_parallel = sup_parallel.execute_domain_analysis()

    assert set(evidences_serial.keys()) == set(evidences_parallel.keys())
    for agent_key in evidences_serial:
        ev_s = evidences_serial[agent_key]
        ev_p = evidences_parallel[agent_key]
        assert ev_s.confidence == pytest.approx(ev_p.confidence, abs=1e-7)
        for m_k in ev_s.metrics:
            assert ev_s.metrics[m_k] == pytest.approx(ev_p.metrics[m_k], abs=1e-7)


def test_scheduler_retry_and_degraded_status():
    """Tests scheduler retry count, failure handling, and degraded state marking without swallowing errors."""
    class FlakySubagent(BaseSubagent):
        def __init__(self, fail_count: int, error_msg: str = "Temporary failure"):
            super().__init__(name="flaky", capabilities=["flaky_action"])
            self.fail_count = fail_count
            self.attempts = 0
            self.error_msg = error_msg

        def execute(self, task: Task) -> TaskResult:
            self.attempts += 1
            if self.attempts <= self.fail_count:
                raise RuntimeError(f"{self.error_msg} (attempt {self.attempts})")
            return TaskResult(task_id=task.task_id, agent_name=self.name, status=TaskStatus.SUCCESS, output="recovered")

    # 1. Successful after 1 retry
    reg1 = SubagentRegistry()
    reg1.register(FlakySubagent(fail_count=1))
    scheduler = TaskScheduler(max_retries=2)
    t1 = Task(task_id="t_flaky_recover", agent_name="flaky", action="flaky_action", max_retries=2)
    plan1 = ExecutionPlan(plan_id="p1", tasks=[t1])
    res1 = scheduler.execute_plan(plan1, reg1)
    assert res1["t_flaky_recover"].status == TaskStatus.SUCCESS
    assert res1["t_flaky_recover"].retry_count == 1
    assert res1["t_flaky_recover"].output == "recovered"

    # 2. Persistent failure without allow_degraded: must be FAILED with traceback preserved
    reg2 = SubagentRegistry()
    reg2.register(FlakySubagent(fail_count=10, error_msg="Critical failure"))
    t2 = Task(task_id="t_flaky_fail", agent_name="flaky", action="flaky_action", max_retries=1)
    plan2 = ExecutionPlan(plan_id="p2", tasks=[t2])
    res2 = scheduler.execute_plan(plan2, reg2)
    assert res2["t_flaky_fail"].status == TaskStatus.FAILED
    assert "Critical failure" in res2["t_flaky_fail"].error
    assert "RuntimeError" in res2["t_flaky_fail"].error

    # 3. Persistent failure WITH allow_degraded: marked DEGRADED
    reg3 = SubagentRegistry()
    reg3.register(FlakySubagent(fail_count=10, error_msg="Degraded fallback failure"))
    t3 = Task(
        task_id="t_flaky_degraded",
        agent_name="flaky",
        action="flaky_action",
        max_retries=1,
        metadata={"allow_degraded": True},
    )
    plan3 = ExecutionPlan(plan_id="p3", tasks=[t3])
    res3 = scheduler.execute_plan(plan3, reg3)
    assert res3["t_flaky_degraded"].status == TaskStatus.DEGRADED
    assert "Degraded fallback failure" in res3["t_flaky_degraded"].error

    # 4. Dependent task is SKIPPED when dependency fails
    t4 = Task(
        task_id="t_dependent",
        agent_name="flaky",
        action="flaky_action",
        dependencies=["t_flaky_fail"],
    )
    plan4 = ExecutionPlan(plan_id="p4", tasks=[t2, t4])
    res4 = scheduler.execute_plan(plan4, reg2)
    assert res4["t_dependent"].status == TaskStatus.SKIPPED
    assert "Skipped because dependency 't_flaky_fail' had status FAILED" in res4["t_dependent"].error


def test_follow_up_controller_creates_and_schedules_tasks(test_context_and_network):
    """Verifies that FollowUpController identifies conflicts and coordinates tasks through scheduler."""
    context, mob_net = test_context_and_network
    sup = SupervisorAgent(urban_context=context, mobility_network=mob_net)
    evidences = sup.execute_domain_analysis()

    # Prepare features via synthesis subagent
    synthesis_task = Task(
        task_id="init_features",
        agent_name="synthesis",
        action="feature_integration",
        payload={
            "evidences": evidences,
            "event_context": EventIntelligenceAgent(context).create_event_context(
                hazard_type=HazardType.RAIN_STORM,
                event_start_hour=12,
                duration_hours=12,
                total_simulation_hours=48,
            ),
        },
    )
    init_res = sup.scheduler.execute_task(synthesis_task, sup.registry)
    zone_features = init_res.output

    controller = FollowUpController(weight_config=sup.weight_config)
    notes, follow_ups_needed = controller.identify_follow_ups(zone_features)
    assert len(follow_ups_needed) > 0

    # Ensure tasks are generated with proper targets
    tasks = controller.create_follow_up_tasks(follow_ups_needed)
    assert len(tasks) == len(follow_ups_needed)
    target_agents = {t.agent_name for t in tasks}
    assert target_agents.issubset({"flexibility", "spatial"})

    # Execute follow-up tasks through scheduler
    fu_plan = ExecutionPlan(plan_id="fu_test_plan", tasks=tasks)
    results = sup.scheduler.execute_plan(fu_plan, sup.registry)
    for tid in results:
        assert results[tid].status == TaskStatus.SUCCESS

    # Apply adjustments
    updated_features, new_logs = controller.apply_follow_up_results(
        follow_ups_needed=follow_ups_needed,
        task_results=results,
        zone_features=zone_features,
    )
    assert len(new_logs) == len(follow_ups_needed)

    # Check that at least one zone had its vulnerability refined
    has_refinement = False
    for req in follow_ups_needed:
        zid = req["zone_id"]
        if zone_features.loc[zid, "vulnerability_score"] != updated_features.loc[zid, "vulnerability_score"]:
            has_refinement = True
            break
    assert has_refinement, "Targeted follow-up must refine vulnerability scores"


def test_evidence_bus_completeness_and_conflicts():
    """Verifies completeness validation, conflict detection, and summary extraction on EvidenceBus."""
    bus = EvidenceBus()

    # 1. Incomplete state
    is_complete, missing = bus.check_completeness()
    assert not is_complete
    assert len(missing) == 5

    # 2. Register mock evidence
    def make_mock_evidence(name: str, spat_score: float = 0.5, flex_score: float = 0.5, conf: float = 0.9):
        return AgentEvidence(
            agent_name=name,
            metrics={"score": 0.5},
            zone_metrics={
                "SZ_01": {
                    "spatial_redundancy_score": spat_score,
                    "intraday_flexibility_score": flex_score,
                }
            },
            traces=[ReActTrace(1, "reason", "act", {}, "obs")],
            confidence=conf,
            qualitative_inference="Test inference",
        )

    bus.register_evidence("temporal", make_mock_evidence("temporal"))
    bus.register_evidence("predictability", make_mock_evidence("predictability"))
    bus.register_evidence("functional", make_mock_evidence("functional"))

    is_complete, missing = bus.check_completeness()
    assert not is_complete
    assert set(missing) == {"flexibility", "spatial"}

    # Register conflicting evidence: spatial=0.85, flexibility=0.20
    bus.register_evidence("spatial", make_mock_evidence("spatial", spat_score=0.85))
    bus.register_evidence("flexibility", make_mock_evidence("flexibility", flex_score=0.20))

    is_complete, missing = bus.check_completeness()
    assert is_complete
    assert len(missing) == 0

    conflicts = bus.detect_conflicts()
    assert len(conflicts) >= 1
    assert any("Conflict detected between high spatial redundancy" in c for c in conflicts)

    summary = bus.get_summary()
    assert summary["is_complete"] is True
    assert len(summary["conflicts_detected"]) >= 1
    assert "spatial" in summary["agent_confidences"]


def test_supervisor_end_to_end_and_manifest_export(test_context_and_network, test_event_context, tmp_path: Path):
    """Verifies end-to-end execution of SupervisorAgent and serialization of RunManifest."""
    context, mob_net = test_context_and_network
    manifest_path = tmp_path / "test_manifest.json"

    sup = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        weight_config=AgentWeightConfig(),
        sim_config=SimulationConfig(total_hours=48, random_seed=42),
    )

    pred, outlook, features, manifest = sup.run(
        event_context=test_event_context,
        output_manifest_path=manifest_path,
        save_manifest=True,
    )

    # Validate predictions & outlook
    assert len(pred.predicted_activity) == 48
    assert len(outlook.zone_rankings) == len(context.zones)
    assert len(sup.follow_up_logs) > 0

    # Validate manifest object
    assert manifest.status == TaskStatus.SUCCESS
    assert manifest.failure_count == 0
    assert len(manifest.task_results) >= 6  # 5 domain + 1 integration + follow-ups + 1 forecast
    assert manifest.total_duration_seconds > 0.0
    assert manifest.metadata["city_name"] == context.city_name

    # Validate persisted JSON manifest
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == manifest.run_id
    assert data["status"] == "SUCCESS"
    assert "evidence_summary" in data
    assert "follow_up_summary" in data
    assert len(data["follow_up_summary"]) > 0


def test_pipeline_orchestration_regression_equivalence(tmp_path: Path):
    """Ensures UrbanResiliencePipeline runs via SupervisorAgent and preserves output schema and numerical consistency."""
    sim_cfg = SimulationConfig(
        total_hours=48,
        event_lead_time_hours=12,
        event_start_hour=12,
        event_duration_hours=12,
        peak_intensity=0.85,
        monte_carlo_samples=100,
        random_seed=42,
    )

    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen",
        sim_config=sim_cfg,
        output_dir=tmp_path,
    )

    results = pipeline.run_full_pipeline()

    # Schema integrity
    assert "prediction" in results
    assert "spatial_outlook" in results
    assert "validation_result" in results
    assert "key_indicators" in results
    assert "plot_paths" in results
    assert "priority_table" in results
    assert "manifest" in results

    manifest = results["manifest"]
    assert manifest is not None
    assert manifest.status == TaskStatus.SUCCESS

    # Check disruption metrics validity
    m = results["prediction"].disruption_metrics
    assert 0.0 < m.maximum_activity_loss < 1.0
    assert m.cumulative_activity_loss > 0.0
    assert m.recovery_time_hours > 0.0

    # Manifest file check
    manifest_file = tmp_path / "orchestration" / f"run_pipeline_{sim_cfg.random_seed}" / "manifest.json"
    assert manifest_file.exists()


def test_external_evaluator_manifest_generation(tmp_path: Path):
    """Verifies that ExternalEventEvaluator generates an orchestration manifest.json for evaluated events."""
    from urban_resilience.data.external_loader import ExternalDataLoader
    from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

    event_dir = Path("data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO")
    if not event_dir.exists():
        pytest.skip("Synthetic demo event directory not found")

    loader = ExternalDataLoader()
    dataset = loader.load_event(event_dir)

    evaluator = ExternalEventEvaluator(output_dir=tmp_path)
    report, artifact_paths = evaluator.evaluate_event(dataset, allow_unverified=True)

    manifest_file = tmp_path / "SZ_20230907_HAIKUI_SYNTHETIC_DEMO" / "manifest.json"
    assert manifest_file.is_file(), f"Expected manifest file at {manifest_file}"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["status"] in ("SUCCESS", "DEGRADED")
    assert "evidence_summary" in manifest_data
    assert "task_results" in manifest_data
    assert report.temporal_metrics.rmse > 0
    assert report.verdicts is not None
    assert report.verdicts.overall_verdict == "FAILED"


def test_scheduler_timeout_serial_and_parallel():
    """Validates unified timeout enforcement in TaskScheduler for both serial and parallel execution without hanging."""
    class SlowSubagent(BaseSubagent):
        def __init__(self):
            super().__init__(name="slow_agent")

        def execute(self, task: Task) -> TaskResult:
            time.sleep(1.0)
            return TaskResult(task_id=task.task_id, agent_name=self.name, status=TaskStatus.SUCCESS, output="done")

    registry = SubagentRegistry()
    registry.register(SlowSubagent())

    # Serial execution path (max_workers=1)
    scheduler_serial = TaskScheduler(max_workers=1, max_retries=0)
    task_serial = Task(
        task_id="task_timeout_serial",
        agent_name="slow_agent",
        action="sleep",
        timeout_seconds=0.1,
        max_retries=0,
    )
    t0 = time.perf_counter()
    res_serial = scheduler_serial.execute_task(task_serial, registry)
    duration_serial = time.perf_counter() - t0

    assert res_serial.status == TaskStatus.FAILED
    assert "timed out" in res_serial.error.lower()
    assert duration_serial < 0.8

    # Parallel execution path (max_workers=4)
    scheduler_parallel = TaskScheduler(max_workers=4, max_retries=0)
    plan_parallel = ExecutionPlan(
        plan_id="plan_timeout_parallel",
        tasks=[
            Task(
                task_id=f"task_timeout_par_{i}",
                agent_name="slow_agent",
                action="sleep",
                timeout_seconds=0.1,
                max_retries=0,
            )
            for i in range(2)
        ],
    )
    t0 = time.perf_counter()
    results_parallel = scheduler_parallel.execute_plan(plan_parallel, registry)
    duration_parallel = time.perf_counter() - t0

    assert len(results_parallel) == 2
    for r in results_parallel.values():
        assert r.status == TaskStatus.FAILED
        assert "timed out" in r.error.lower()
    assert duration_parallel < 0.8


def test_scheduler_retry_upper_bound_semantics():
    """Validates that retry count adheres to 1 + min(task.max_retries, scheduler.max_retries)."""
    attempt_counts = {"count": 0}

    class FailingSubagent(BaseSubagent):
        def __init__(self):
            super().__init__(name="failing_agent")

        def execute(self, task: Task) -> TaskResult:
            attempt_counts["count"] += 1
            raise RuntimeError("Subagent intentional failure")

    registry = SubagentRegistry()
    registry.register(FailingSubagent())

    # 1. scheduler.max_retries=0, task.max_retries=3 -> only 1 attempt executed
    attempt_counts["count"] = 0
    sched1 = TaskScheduler(max_workers=1, max_retries=0)
    task1 = Task(task_id="t1", agent_name="failing_agent", action="fail", max_retries=3)
    r1 = sched1.execute_task(task1, registry)
    assert r1.status == TaskStatus.FAILED
    assert attempt_counts["count"] == 1
    assert r1.retry_count == 0

    # 2. scheduler.max_retries=3, task.max_retries=0 -> only 1 attempt executed
    attempt_counts["count"] = 0
    sched2 = TaskScheduler(max_workers=1, max_retries=3)
    task2 = Task(task_id="t2", agent_name="failing_agent", action="fail", max_retries=0)
    r2 = sched2.execute_task(task2, registry)
    assert r2.status == TaskStatus.FAILED
    assert attempt_counts["count"] == 1
    assert r2.retry_count == 0

    # 3. scheduler.max_retries=2, task.max_retries=5 -> 3 attempts (1 initial + 2 retries)
    attempt_counts["count"] = 0
    sched3 = TaskScheduler(max_workers=1, max_retries=2)
    task3 = Task(task_id="t3", agent_name="failing_agent", action="fail", max_retries=5)
    r3 = sched3.execute_task(task3, registry)
    assert r3.status == TaskStatus.FAILED
    assert attempt_counts["count"] == 3
    assert r3.retry_count == 2

    # 4. scheduler.max_retries=None, task.max_retries=2 -> 3 attempts (1 initial + 2 retries)
    attempt_counts["count"] = 0
    sched4 = TaskScheduler(max_workers=1, max_retries=None)
    task4 = Task(task_id="t4", agent_name="failing_agent", action="fail", max_retries=2)
    r4 = sched4.execute_task(task4, registry)
    assert r4.status == TaskStatus.FAILED
    assert attempt_counts["count"] == 3
    assert r4.retry_count == 2


def test_supervisor_strict_evidence_gate_and_failure_propagation(test_context_and_network, test_event_context):
    """Verifies that under strict mode, failing domain tasks raise IncompleteEvidenceError and set manifest to FAILED."""
    context, mob_net = test_context_and_network

    sup = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        strict=True,
        allow_degraded=False,
    )

    def failing_execute(task: Task) -> TaskResult:
        raise RuntimeError("Simulated temporal domain crash")

    sup.subagents["temporal"].execute = failing_execute

    with pytest.raises(IncompleteEvidenceError) as exc_info:
        sup.run(event_context=test_event_context, save_manifest=False)

    err = exc_info.value
    assert "temporal" in err.missing_domains or any("temporal" in k for k in err.failed_tasks.keys())
    assert sup.latest_manifest is not None
    assert sup.latest_manifest.status == TaskStatus.FAILED
    assert sup.latest_manifest.error_summary is not None

    # Test with allow_degraded=True: should not raise IncompleteEvidenceError
    sup_degraded = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        strict=True,
        allow_degraded=True,
    )
    sup_degraded.subagents["temporal"].execute = failing_execute

    pred, outlook, feats, manifest = sup_degraded.run(event_context=test_event_context, save_manifest=False)
    assert manifest.status == TaskStatus.DEGRADED
    assert manifest.failure_count > 0


def test_pipeline_and_external_evaluator_strict_failure_handling(tmp_path: Path, monkeypatch):
    """Verifies pipeline and external evaluator cleanly abort and raise OrchestrationError on supervisor failure."""
    from urban_resilience.data.external_loader import ExternalDataLoader
    from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen",
        sim_config=SimulationConfig(total_hours=48, random_seed=42),
        output_dir=tmp_path,
    )

    def failing_run(*args, **kwargs):
        raise IncompleteEvidenceError("Simulated pipeline supervisor failure", missing_domains=["spatial"])

    monkeypatch.setattr(SupervisorAgent, "run", failing_run)

    with pytest.raises(OrchestrationError, match="Simulated pipeline supervisor failure"):
        pipeline.run_full_pipeline()

    event_dir = Path("data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO")
    if event_dir.exists():
        loader = ExternalDataLoader()
        dataset = loader.load_event(event_dir)
        evaluator = ExternalEventEvaluator(output_dir=tmp_path)
        with pytest.raises(OrchestrationError, match="Simulated pipeline supervisor failure"):
            evaluator.evaluate_event(dataset, allow_unverified=True)


def test_api_and_cli_orchestration_error_handling(monkeypatch):
    """Verifies API returns HTTP 503 and CLI returns exit code 1 when OrchestrationError is raised."""
    from fastapi.testclient import TestClient
    from urban_resilience.api import app
    from urban_resilience.cli import main

    def mock_run_pipeline_fail(*args, **kwargs):
        raise IncompleteEvidenceError("Critical orchestration collapse", missing_domains=["functional"])

    monkeypatch.setattr(UrbanResiliencePipeline, "run_full_pipeline", mock_run_pipeline_fail)

    # API test
    client = TestClient(app)
    response = client.post(
        "/api/v1/resilience/simulate",
        json={
            "city_name": "Shenzhen",
            "hazard_type": "rainstorm",
            "event_start_hour": 24,
            "duration_hours": 18,
            "peak_intensity": 0.85,
            "total_hours": 96,
            "monte_carlo_samples": 50,
        },
    )
    assert response.status_code == 503
    assert "Critical orchestration collapse" in response.json()["detail"]
    assert "functional" in response.json()["detail"]

    # CLI test
    exit_code = main(["--total-hours", "48", "--duration", "12"])
    assert exit_code == 1


def test_supervisor_state_reset_between_runs(test_context_and_network, test_event_context):
    """Ensures state reset between successive runs on the same SupervisorAgent instance."""
    context, mob_net = test_context_and_network

    sup = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        sim_config=SimulationConfig(total_hours=48, random_seed=42),
    )

    # Run 1
    _, _, _, manifest1 = sup.run(event_context=test_event_context, save_manifest=False)
    run_id_1 = manifest1.run_id
    tasks_1 = set(sup._active_task_results.keys())
    assert len(tasks_1) > 0

    # Run 2
    _, _, _, manifest2 = sup.run(event_context=test_event_context, save_manifest=False)
    run_id_2 = manifest2.run_id
    tasks_2 = set(sup._active_task_results.keys())

    assert run_id_1 != run_id_2
    assert len(tasks_2) == len(tasks_1)
    assert f"task_feature_integration_{run_id_1}" not in tasks_2
    assert f"task_forecast_{run_id_1}" not in tasks_2
    assert len(sup.follow_up_logs) == len(manifest2.follow_up_summary)


def test_manifest_tasks_schema_and_duration_coverage(test_context_and_network, test_event_context):
    """Verifies RunManifest schema completeness including tasks, dependencies, and total duration."""
    context, mob_net = test_context_and_network

    sup = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
        sim_config=SimulationConfig(total_hours=48, random_seed=42),
    )

    t0 = time.perf_counter()
    _, _, _, manifest = sup.run(event_context=test_event_context, save_manifest=False)
    wall_duration = time.perf_counter() - t0

    assert manifest.total_duration_seconds > 0.0
    assert manifest.total_duration_seconds <= wall_duration + 0.5
    assert len(manifest.tasks) >= 6

    for t in manifest.tasks:
        assert "task_id" in t
        assert "agent_name" in t
        assert "action" in t
        assert "objective" in t
        assert "dependencies" in t
        assert isinstance(t["dependencies"], list)
        assert "priority" in t
        assert t["status"] in ("SUCCESS", "DEGRADED", "FAILED")

    manifest_dict = manifest.to_dict()
    assert isinstance(manifest_dict["tasks"], list)
    assert len(manifest_dict["tasks"]) == len(manifest.tasks)


def test_scheduler_timeout_no_retry_by_default():
    """Verifies that timeouts do not retry by default unless retry_on_timeout=True."""
    attempt_counts = {"count": 0}

    class HangingSubagent(BaseSubagent):
        def __init__(self):
            super().__init__(name="hanging_agent")

        def execute(self, task: Task) -> TaskResult:
            attempt_counts["count"] += 1
            time.sleep(0.3)
            return TaskResult(task_id=task.task_id, agent_name=self.name, status=TaskStatus.SUCCESS, output="done")

    registry = SubagentRegistry()
    registry.register(HangingSubagent())

    # Case 1: retry_on_timeout=False (default), max_retries=3 -> only 1 attempt executed
    attempt_counts["count"] = 0
    sched = TaskScheduler(max_workers=1, max_retries=3)
    task = Task(
        task_id="t_timeout_no_retry",
        agent_name="hanging_agent",
        action="sleep",
        timeout_seconds=0.05,
        max_retries=3,
        metadata={"retry_on_timeout": False},
    )
    t0 = time.perf_counter()
    res = sched.execute_task(task, registry)
    elapsed = time.perf_counter() - t0

    assert res.status == TaskStatus.FAILED
    assert "timed out" in res.error.lower()
    assert attempt_counts["count"] == 1
    assert elapsed < 0.5

    # Case 2: retry_on_timeout=True, max_retries=2 -> 3 attempts (1 initial + 2 retries)
    attempt_counts["count"] = 0
    task_retry = Task(
        task_id="t_timeout_retry",
        agent_name="hanging_agent",
        action="sleep",
        timeout_seconds=0.05,
        max_retries=2,
        metadata={"retry_on_timeout": True},
    )
    t0 = time.perf_counter()
    res_retry = sched.execute_task(task_retry, registry)
    elapsed_retry = time.perf_counter() - t0

    assert res_retry.status == TaskStatus.FAILED
    assert "timed out" in res_retry.error.lower()
    assert attempt_counts["count"] == 3
    assert elapsed_retry < 0.9


def test_execute_domain_analysis_state_reset(test_context_and_network):
    """Verifies execute_domain_analysis with reset=True clears prior task results and evidence."""
    context, mob_net = test_context_and_network
    sup = SupervisorAgent(
        urban_context=context,
        mobility_network=mob_net,
    )

    evidences1 = sup.execute_domain_analysis(reset=True)
    assert len(evidences1) == 5
    assert len(sup._active_task_results) == 5

    evidences2 = sup.execute_domain_analysis(reset=True)
    assert len(evidences2) == 5
    # With reset=True, active task results should only contain current run's tasks
    assert len(sup._active_task_results) == 5


def test_pipeline_numerical_regression_baseline(tmp_path: Path):
    """Verifies that full pipeline reproduction matches benchmark baseline metrics within tight tolerance."""
    sim_cfg = SimulationConfig(
        total_hours=96,
        event_start_hour=24,
        event_duration_hours=18,
        random_seed=42,
        monte_carlo_samples=200,
    )
    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen",
        sim_config=sim_cfg,
        output_dir=tmp_path,
    )

    results = pipeline.run_full_pipeline()

    # 1. Disruption metrics
    metrics = results["prediction"].disruption_metrics
    assert 0.175 <= metrics.maximum_activity_loss <= 0.195, f"Expected ~0.185, got {metrics.maximum_activity_loss}"
    assert 20.5 <= metrics.recovery_time_hours <= 21.5, f"Expected ~21.0h, got {metrics.recovery_time_hours}"

    # 2. Benchmark RMSE
    eval_metrics = results["validation_result"].eval_metrics
    assert 0.0270 <= eval_metrics["rmse"] <= 0.0310, f"Expected ~0.0290, got {eval_metrics['rmse']}"

    # 3. UQ PICP
    picp_95 = results["validation_result"].calibration.picp_95
    assert 0.934 <= picp_95 <= 0.954, f"Expected ~0.944, got {picp_95}"

    # 4. Top-1 vulnerable zone
    top_zone = results["spatial_outlook"].zone_rankings[0].zone_id
    assert top_zone == "SZ_NS_SK", f"Expected SZ_NS_SK, got {top_zone}"


