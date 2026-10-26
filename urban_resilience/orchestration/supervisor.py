"""Supervisor agent orchestrating planning, scheduling, evidence validation, follow-ups, and manifests."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from urban_resilience.config import AgentWeightConfig, ProjectPaths, SimulationConfig
from urban_resilience.llm.client import LLMClient
from urban_resilience.orchestration.contracts import (
    ExecutionPlan,
    IncompleteEvidenceError,
    OrchestrationError,
    RunManifest,
    SUPPORTED_LLM_DOMAINS,
    Task,
    TaskResult,
    TaskStatus,
    is_follow_up_success_status,
)
from urban_resilience.orchestration.evidence_bus import EvidenceBus
from urban_resilience.orchestration.follow_up import FollowUpController
from urban_resilience.orchestration.llm_reviewer import LLMSupervisorReviewer
from urban_resilience.orchestration.planner import DeterministicTaskPlanner, LLMPlannerProtocol
from urban_resilience.orchestration.react_subagent import (
    FlexibilitySubagentLLM,
    FunctionalSubagentLLM,
    PredictabilitySubagentLLM,
    SpatialSubagentLLM,
    TemporalSubagentLLM,
)
from urban_resilience.orchestration.scheduler import TaskScheduler
from urban_resilience.orchestration.subagent import (
    FlexibilitySubagent,
    FunctionalSubagent,
    PredictabilitySubagent,
    SpatialSubagent,
    SubagentRegistry,
    SynthesisSubagent,
    TemporalSubagent,
)
from urban_resilience.prediction.features import FeatureAssembler
from urban_resilience.prediction.predictor import ResiliencePredictor
from urban_resilience.types import (
    AgentEvidence,
    EventContext,
    MobilityNetwork,
    PredictionProvenance,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    UrbanContext,
)

logger = logging.getLogger(__name__)


def recompute_zone_features(
    zone_features: pd.DataFrame,
    weights: Optional[AgentWeightConfig] = None,
    event_context: Optional[EventContext] = None,
) -> pd.DataFrame:
    """Recalculates composite_resilience and vulnerability_score across all zones in-place."""
    w = (weights or AgentWeightConfig()).as_dict()
    w_temp = w.get("temporal_redundancy", 0.20)
    w_flex = w.get("intraday_flexibility", 0.20)
    w_pred = w.get("predictability", 0.15)
    w_spat = w.get("spatial_redundancy", 0.25)
    w_func = w.get("functional_mix", 0.20)

    for zid in zone_features.index:
        r_temp = (
            float(zone_features.at[zid, "temporal_redundancy"])
            if "temporal_redundancy" in zone_features.columns
            else 0.5
        )
        r_flex = (
            float(zone_features.at[zid, "intraday_flexibility"])
            if "intraday_flexibility" in zone_features.columns
            else 0.5
        )
        r_pred = (
            float(zone_features.at[zid, "predictability"])
            if "predictability" in zone_features.columns
            else 0.5
        )
        r_spat = (
            float(zone_features.at[zid, "spatial_redundancy"])
            if "spatial_redundancy" in zone_features.columns
            else 0.5
        )
        r_func = (
            float(zone_features.at[zid, "functional_mix"])
            if "functional_mix" in zone_features.columns
            else 0.5
        )

        comp = float(
            w_temp * r_temp
            + w_flex * r_flex
            + w_pred * r_pred
            + w_spat * r_spat
            + w_func * r_func
        )
        comp = float(np.clip(comp, 0.05, 0.95))
        zone_features.at[zid, "composite_resilience"] = comp

        h_int = 0.5
        if event_context and zid in event_context.zone_intensity_map:
            h_int = float(event_context.zone_intensity_map[zid])
        elif "hazard_intensity" in zone_features.columns:
            h_int = float(zone_features.at[zid, "hazard_intensity"])

        vuln = float(np.clip(h_int * (1.15 - comp), 0.05, 1.0))
        zone_features.at[zid, "vulnerability_score"] = vuln

    return zone_features


class SupervisorAgent:
    """Master hierarchical orchestrator coordinating domain subagents and resilience synthesis."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        forecast_mobility_network: Optional[MobilityNetwork] = None,
        weight_config: Optional[AgentWeightConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
        planner: Optional[LLMPlannerProtocol] = None,
        max_workers: int = 4,
        max_retries: int = 1,
        output_dir: Optional[Path] = None,
        strict: bool = True,
        allow_degraded: bool = False,
        orchestration_mode: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        model_artifact_path: Optional[Union[str, Path]] = None,
        enable_follow_ups: bool = True,
        llm_domain_keys: Optional[Set[str]] = None,
        follow_up_data_source: Optional[Any] = None,
    ):
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.forecast_mobility_network = forecast_mobility_network or mobility_network
        self.weight_config = weight_config or AgentWeightConfig()
        self.sim_config = sim_config or SimulationConfig()
        self.output_dir = Path(output_dir) if output_dir else ProjectPaths.OUTPUTS_DIR
        self.model_artifact_path = Path(model_artifact_path) if model_artifact_path else None
        self.strict = strict
        self.allow_degraded = allow_degraded
        self.enable_follow_ups = enable_follow_ups
        from urban_resilience.orchestration.follow_up_data import SyntheticFollowUpDataSource
        self.follow_up_data_source = follow_up_data_source or SyntheticFollowUpDataSource()
        self.llm_domain_keys = set(llm_domain_keys) if llm_domain_keys is not None else None
        if self.llm_domain_keys is not None:
            unknown_domains = set(self.llm_domain_keys) - SUPPORTED_LLM_DOMAINS
            if unknown_domains:
                raise ValueError(
                    f"Unsupported LLM domain(s): {sorted(list(unknown_domains))}. "
                    f"Must be a subset of {sorted(list(SUPPORTED_LLM_DOMAINS))}."
                )
        self.orchestration_mode = (
            orchestration_mode
            if orchestration_mode is not None
            else os.environ.get("URBAN_ORCHESTRATION_MODE", "deterministic")
        )
        self.llm_client = llm_client

        if self.orchestration_mode == "llm_hybrid" and self.llm_client is None:
            raise ValueError("llm_hybrid orchestration mode requires a valid llm_client, got None.")

        # Initialize Registry and Subagents
        self.registry = SubagentRegistry()
        subagent_token_budget = 100000

        llm_factories = {
            "temporal": lambda: TemporalSubagentLLM(
                urban_context=self.urban_context,
                mobility_network=self.mobility_network,
                llm_client=self.llm_client,
                token_budget=subagent_token_budget,
                follow_up_data_source=self.follow_up_data_source,
            ),
            "flexibility": lambda: FlexibilitySubagentLLM(
                urban_context=self.urban_context,
                mobility_network=self.mobility_network,
                llm_client=self.llm_client,
                token_budget=subagent_token_budget,
                follow_up_data_source=self.follow_up_data_source,
            ),
            "predictability": lambda: PredictabilitySubagentLLM(
                urban_context=self.urban_context,
                mobility_network=self.mobility_network,
                llm_client=self.llm_client,
                token_budget=subagent_token_budget,
                follow_up_data_source=self.follow_up_data_source,
            ),
            "spatial": lambda: SpatialSubagentLLM(
                urban_context=self.urban_context,
                mobility_network=self.mobility_network,
                llm_client=self.llm_client,
                token_budget=subagent_token_budget,
                follow_up_data_source=self.follow_up_data_source,
            ),
            "functional": lambda: FunctionalSubagentLLM(
                urban_context=self.urban_context,
                mobility_network=self.mobility_network,
                llm_client=self.llm_client,
                token_budget=subagent_token_budget,
                follow_up_data_source=self.follow_up_data_source,
            ),
        }

        det_factories = {
            "temporal": lambda: TemporalSubagent(self.urban_context, self.mobility_network),
            "flexibility": lambda: FlexibilitySubagent(self.urban_context, self.mobility_network),
            "predictability": lambda: PredictabilitySubagent(self.urban_context, self.mobility_network),
            "spatial": lambda: SpatialSubagent(self.urban_context, self.mobility_network),
            "functional": lambda: FunctionalSubagent(self.urban_context, self.mobility_network),
        }

        self.subagents = {}
        for domain in ["temporal", "flexibility", "predictability", "spatial", "functional"]:
            use_llm = False
            if self.orchestration_mode == "llm_hybrid":
                if self.llm_domain_keys is None or domain in self.llm_domain_keys:
                    use_llm = True

            if use_llm:
                self.subagents[domain] = llm_factories[domain]()
            else:
                self.subagents[domain] = det_factories[domain]()

        for sub in self.subagents.values():
            self.registry.register(sub)

        self.synthesis_subagent = SynthesisSubagent(
            urban_context=self.urban_context,
            mobility_network=self.forecast_mobility_network,
            weight_config=self.weight_config,
            sim_config=self.sim_config,
        )
        self.registry.register(self.synthesis_subagent)

        # Orchestration components
        self.planner: LLMPlannerProtocol = planner or DeterministicTaskPlanner()
        self.scheduler = TaskScheduler(max_workers=max_workers, max_retries=max_retries)
        self.evidence_bus = EvidenceBus()
        self.follow_up_controller = FollowUpController(weight_config=self.weight_config)
        self.llm_reviewer = (
            LLMSupervisorReviewer(llm_client=self.llm_client, max_rounds=2, budget_token_limit=200000)
            if (self.orchestration_mode == "llm_hybrid" and self.llm_client is not None)
            else None
        )

        self._active_task_results: Dict[str, TaskResult] = {}
        self.latest_manifest: Optional[RunManifest] = None

    def _reset_run_state(self) -> None:
        """Resets execution state, evidence bus, follow-ups, and manifests for a fresh run."""
        self._active_task_results = {}
        self.evidence_bus = EvidenceBus()
        self.follow_up_controller = FollowUpController(weight_config=self.weight_config)
        if self.llm_reviewer:
            self.llm_reviewer.decision_history.clear()
        self.synthesis_subagent.synthesis_agent.follow_up_logs = []
        self.latest_manifest = None

    @property
    def follow_up_logs(self) -> List[Dict[str, Any]]:
        """Accesses recorded follow-up inquiry traces."""
        return self.follow_up_controller.follow_up_logs

    def execute_domain_analysis(
        self,
        plan: Optional[ExecutionPlan] = None,
        strict: Optional[bool] = None,
        allow_degraded: Optional[bool] = None,
        reset: bool = True,
    ) -> Dict[str, AgentEvidence]:
        """Plans, schedules, and executes domain analysis tasks across all 5 domain perspectives.

        Args:
            plan: Execution plan for domain tasks. Defaults to planner.plan_domain_tasks().
            strict: Strict evidence gate mode. Defaults to self.strict.
            allow_degraded: Allow degraded evidence mode. Defaults to self.allow_degraded.
            reset: If True, resets supervisor run state before executing, preventing
                inter-run state accumulation across standalone diagnostic phase calls.
        """
        if reset:
            self._reset_run_state()

        is_strict = self.strict if strict is None else strict
        can_degrade = self.allow_degraded if allow_degraded is None else allow_degraded

        logger.info("SupervisorAgent: Launching domain analysis phase...")
        active_plan = plan or self.planner.plan_domain_tasks()
        results = self.scheduler.execute_plan(active_plan, self.registry)
        self._active_task_results.update(results)

        for task in active_plan.tasks:
            res = results.get(task.task_id)
            if res and res.status in (TaskStatus.SUCCESS, TaskStatus.DEGRADED) and isinstance(res.output, AgentEvidence):
                self.evidence_bus.register_evidence(task.agent_name, res.output, task_id=task.task_id)
            elif res and res.status == TaskStatus.FAILED:
                logger.error("Domain task '%s' failed: %s", task.task_id, res.error)

        is_complete, missing = self.evidence_bus.check_completeness()
        failed_tasks = {
            tid: res.error for tid, res in results.items()
            if res.status in (TaskStatus.FAILED, TaskStatus.DEGRADED)
        }
        if (not is_complete or failed_tasks) and is_strict and not can_degrade:
            err_msg = (
                f"Domain evidence incomplete or degraded under strict mode: missing {missing}, "
                f"failed/degraded {list(failed_tasks.keys())}"
            )
            logger.error(err_msg)
            raise IncompleteEvidenceError(
                err_msg,
                missing_domains=missing,
                failed_tasks=failed_tasks,
            )
        elif not is_complete:
            logger.warning("Domain evidence incomplete; missing perspectives: %s", missing)

        self.evidence_bus.detect_conflicts(self.urban_context)
        return self.evidence_bus.get_all_evidences()

    def _compile_manifest(
        self,
        run_id: str,
        event_context: EventContext,
        total_elapsed: float,
        all_tasks: Optional[Dict[str, Task]] = None,
        output_manifest_path: Optional[Path | str] = None,
        save_manifest: bool = True,
        forced_status: Optional[TaskStatus] = None,
        error_summary: Optional[str] = None,
        can_degrade: bool = False,
        stop_reason: Optional[str] = None,
        predictor_status: Optional[Dict[str, Any]] = None,
        prediction: Optional[ResiliencePrediction] = None,
        review_tokens_used: int = 0,
        follow_up_success_count: int = 0,
        follow_up_executed: bool = False,
        evidence_changed: bool = False,
        zone_features_changed: bool = False,
        coordination_status: str = "not_attempted",
        coordination_exercised: bool = False,
        conflicts_detected: int = 0,
        conflicts_resolved: int = 0,
        conflicts_list: Optional[List[str]] = None,
        anomalies_detected: int = 0,
        unlocalized_conflicts: int = 0,
        resolved_conflicts: Optional[List[Dict[str, Any]]] = None,
    ) -> RunManifest:
        """Helper to build and optionally persist a RunManifest."""
        failures = sum(1 for r in self._active_task_results.values() if r.status == TaskStatus.FAILED)
        degraded = sum(1 for r in self._active_task_results.values() if r.status == TaskStatus.DEGRADED)

        if forced_status is not None:
            overall_status = forced_status
        elif failures > 0 or degraded > 0:
            overall_status = TaskStatus.DEGRADED if can_degrade else TaskStatus.FAILED
        else:
            overall_status = TaskStatus.SUCCESS

        tasks_list: List[Dict[str, Any]] = []
        if all_tasks:
            for task in all_tasks.values():
                res = self._active_task_results.get(task.task_id)
                deps = list(task.dependencies)
                if not deps and task.parent_task_id:
                    deps = [task.parent_task_id]
                task_dict = {
                    "task_id": task.task_id,
                    "parent_task_id": task.parent_task_id,
                    "agent_name": task.agent_name,
                    "action": task.action,
                    "objective": task.objective,
                    "dependencies": deps,
                    "priority": task.priority,
                    "status": res.status.value if res else "NOT_RUN",
                }
                tasks_list.append(task_dict)

        serialized_results = {
            tid: {
                "agent_name": r.agent_name,
                "status": r.status.value,
                "execution_time_seconds": r.execution_time_seconds,
                "retry_count": r.retry_count,
                "error": r.error,
                "trace_steps": len(r.trace),
            }
            for tid, r in self._active_task_results.items()
        }

        # Collect LLM telemetry and tool calls if available
        llm_model = getattr(self.llm_client, "model", None) if self.llm_client else None
        token_usage = getattr(self.llm_client, "token_usage", {}) if self.llm_client else {}
        llm_latency = getattr(self.llm_client, "total_latency_seconds", 0.0) if self.llm_client else 0.0
        follow_up_decs = list(self.llm_reviewer.decision_history) if self.llm_reviewer else []
        tool_calls = []
        for sub in self.subagents.values():
            if hasattr(sub, "tool_call_records"):
                tool_calls.extend(sub.tool_call_records)

        # Aggregate feature report verification metrics across tasks
        feat_val_count = 0
        feat_with_refs_count = 0
        val_attempts = 0
        val_rejections = 0
        for r in self._active_task_results.values():
            if isinstance(getattr(r, "metadata", None), dict):
                m = r.metadata
                if "feature_value_count" in m:
                    feat_val_count += int(m.get("feature_value_count", 0))
                    feat_with_refs_count += int(m.get("feature_with_refs_count", 0))
                    val_attempts += int(m.get("validation_attempts", 0))
                    val_rejections += int(m.get("validation_rejections", 0))

        if feat_val_count == 0:
            for sub in self.subagents.values():
                if hasattr(sub, "feature_value_count") and getattr(sub, "feature_value_count", 0) > 0:
                    feat_val_count += getattr(sub, "feature_value_count", 0)
                    feat_with_refs_count += getattr(sub, "feature_with_refs_count", 0)
                    val_attempts += getattr(sub, "validation_attempts", 0)
                    val_rejections += getattr(sub, "validation_rejections", 0)

        evidence_coverage_val = float(feat_with_refs_count / feat_val_count) if feat_val_count > 0 else None

        manifest_metadata = {
            "city_name": self.urban_context.city_name,
            "event_id": event_context.event_id,
            "review_tokens_used": review_tokens_used,
            "follow_up_success_count": follow_up_success_count,
            "follow_up_executed": follow_up_executed,
            "evidence_changed": evidence_changed,
            "zone_features_changed": zone_features_changed,
            "coordination_status": coordination_status,
            "coordination_exercised": coordination_exercised,
            "feature_value_count": feat_val_count,
            "feature_with_refs_count": feat_with_refs_count,
            "validation_attempts": val_attempts,
            "validation_rejections": val_rejections,
            "anti_fabrication_rejection_count": val_rejections,
            "evidence_coverage": evidence_coverage_val,
            "conflicts_detected": conflicts_detected,
            "conflicts_resolved": conflicts_resolved,
            "anomalies_detected": anomalies_detected,
            "unlocalized_conflicts": unlocalized_conflicts,
            "conflicts": list(conflicts_list or []),
            "resolved_conflicts": list(resolved_conflicts or []),
        }
        if stop_reason:
            manifest_metadata["stop_reason"] = stop_reason
        if predictor_status:
            manifest_metadata["predictor_status"] = predictor_status
        if prediction is not None and hasattr(prediction, "provenance") and prediction.provenance:
            manifest_metadata["prediction_provenance"] = prediction.provenance.to_dict()
        if self.llm_domain_keys is not None:
            manifest_metadata["llm_domain_keys"] = sorted(list(self.llm_domain_keys))

        manifest = RunManifest(
            run_id=run_id,
            plan_id=f"plan_{run_id}",
            status=overall_status,
            total_duration_seconds=total_elapsed,
            failure_count=failures,
            degraded_count=degraded,
            error_summary=error_summary,
            tasks=tasks_list,
            task_results=serialized_results,
            evidence_summary=self.evidence_bus.get_summary(),
            follow_up_summary=list(self.follow_up_controller.follow_up_logs),
            orchestration_mode=self.orchestration_mode,
            llm_model=llm_model,
            prompt_version="v1.0.0" if self.orchestration_mode == "llm_hybrid" else None,
            token_usage=token_usage,
            llm_latency_seconds=llm_latency,
            tool_calls=tool_calls,
            follow_up_decisions=follow_up_decs,
            stop_reason=stop_reason,
            predictor_status=predictor_status or {},
            metadata=manifest_metadata,
        )

        if save_manifest:
            target_path = output_manifest_path or (
                self.output_dir / "orchestration" / run_id / "manifest.json"
            )
            manifest.save_to_file(target_path)
            logger.info("SupervisorAgent: RunManifest saved to %s", target_path)

        return manifest

    def run(
        self,
        event_context: EventContext,
        output_manifest_path: Optional[Path | str] = None,
        save_manifest: bool = True,
        strict: Optional[bool] = None,
        allow_degraded: Optional[bool] = None,
    ) -> Tuple[ResiliencePrediction, SpatialResilienceOutlook, pd.DataFrame, RunManifest]:
        """Executes full end-to-end multi-agent orchestration workflow and produces a RunManifest.

        When save_manifest is True, the manifest is saved to output_manifest_path if provided,
        or defaults to ``outputs/orchestration/<run_id>/manifest.json``.
        Note:
            - Full pipeline runs specify ``outputs/orchestration/run_pipeline_<random_seed>/manifest.json``.
            - External evaluation runs specify ``outputs/external/<event_id>/manifest.json``.
            - Standalone SupervisorAgent.run() uses unique ``outputs/orchestration/<run_id>/manifest.json``.
        """
        is_strict = self.strict if strict is None else strict
        can_degrade = self.allow_degraded if allow_degraded is None else allow_degraded

        self._reset_run_state()
        start_time = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        logger.info("SupervisorAgent: Beginning orchestrated execution (run_id: %s)...", run_id)

        all_tasks: Dict[str, Task] = {}

        # 1. Domain Analysis Phase
        domain_plan = self.planner.plan_domain_tasks(plan_id=f"plan_domains_{run_id}")
        for t in domain_plan.tasks:
            all_tasks[t.task_id] = t
        domain_results = self.scheduler.execute_plan(domain_plan, self.registry)
        self._active_task_results.update(domain_results)

        for task in domain_plan.tasks:
            res = domain_results.get(task.task_id)
            if res and res.status in (TaskStatus.SUCCESS, TaskStatus.DEGRADED) and isinstance(res.output, AgentEvidence):
                self.evidence_bus.register_evidence(task.agent_name, res.output, task_id=task.task_id)

        is_complete, missing = self.evidence_bus.check_completeness()
        failed_domain_tasks = {
            tid: res.error for tid, res in domain_results.items()
            if res.status in (TaskStatus.FAILED, TaskStatus.DEGRADED)
        }

        if (not is_complete or failed_domain_tasks) and is_strict and not can_degrade:
            total_elapsed = time.perf_counter() - start_time
            err_summary = (
                f"Domain evidence incomplete or degraded under strict mode. Missing: {missing}; "
                f"Failed/degraded tasks: {list(failed_domain_tasks.keys())}"
            )
            logger.error(err_summary)
            manifest = self._compile_manifest(
                run_id=run_id,
                event_context=event_context,
                total_elapsed=total_elapsed,
                all_tasks=all_tasks,
                output_manifest_path=output_manifest_path,
                save_manifest=save_manifest,
                forced_status=TaskStatus.FAILED,
                error_summary=err_summary,
                can_degrade=can_degrade,
            )
            self.latest_manifest = manifest
            raise IncompleteEvidenceError(
                err_summary,
                missing_domains=missing,
                failed_tasks=failed_domain_tasks,
            )
        elif not is_complete:
            logger.warning("Domain evidence incomplete (missing: %s); continuing in degraded mode.", missing)

        evidences = self.evidence_bus.get_all_evidences()
        self.evidence_bus.detect_conflicts(self.urban_context)
        initial_cross_domain_conflicts = list(self.evidence_bus.cross_domain_conflicts)
        conflicts_detected = len(initial_cross_domain_conflicts)
        anomalies_detected = len(self.evidence_bus.anomalies)
        unlocalized_conflicts = len(self.evidence_bus.unlocalized_conflicts)

        # 2. Synthesis & Targeted Follow-Up Phase
        # Feature integration
        task_integrate = Task(
            task_id=f"task_feature_integration_{run_id}",
            parent_task_id=f"plan_domains_{run_id}",
            agent_name="synthesis",
            action="feature_integration",
            objective="Integrate multi-perspective domain evidences into zone feature matrix",
            dependencies=[t.task_id for t in domain_plan.tasks],
            payload={"evidences": evidences, "event_context": event_context},
        )
        all_tasks[task_integrate.task_id] = task_integrate
        res_integrate = self.scheduler.execute_task(task_integrate, self.registry)
        self._active_task_results[task_integrate.task_id] = res_integrate

        if res_integrate.status not in (TaskStatus.SUCCESS, TaskStatus.DEGRADED):
            total_elapsed = time.perf_counter() - start_time
            manifest = self._compile_manifest(
                run_id=run_id,
                event_context=event_context,
                total_elapsed=total_elapsed,
                all_tasks=all_tasks,
                output_manifest_path=output_manifest_path,
                save_manifest=save_manifest,
                forced_status=TaskStatus.FAILED,
                error_summary=f"Feature integration failed: {res_integrate.error}",
            )
            self.latest_manifest = manifest
            raise OrchestrationError(f"Feature integration failed: {res_integrate.error}")

        zone_features: pd.DataFrame = res_integrate.output

        # Identify follow-ups: capture pre-followup canonical baseline snapshots
        zone_features_before_followup = zone_features.copy(deep=True)
        evidence_snapshot = self.evidence_bus.get_canonical_metrics_snapshot()

        fu_tasks: List[Task] = []
        llm_stop_reason: Optional[str] = None
        review_tokens_used = 0
        if not self.enable_follow_ups:
            logger.info("SupervisorAgent: Follow-up phase disabled (enable_follow_ups=False).")
        elif self.orchestration_mode == "llm_hybrid" and self.llm_reviewer is not None:
            # LLM Supervisor Review Phase (max 2 rounds, budget-guarded)
            max_rounds = 2
            current_round = 1
            review_tokens_start = int(getattr(self.llm_client, "token_usage", {}).get("total_tokens", 0))
            while current_round <= max_rounds:
                current_total = int(getattr(self.llm_client, "token_usage", {}).get("total_tokens", 0))
                tokens_spent = max(0, current_total - review_tokens_start)
                decision = self.llm_reviewer.review_and_decide(
                    round_num=current_round,
                    evidence_summary=self.evidence_bus.get_summary(),
                    conflicts=self.evidence_bus.conflicts,
                    zone_features=zone_features,
                    tokens_spent=tokens_spent,
                    structured_conflicts=self.evidence_bus.structured_conflicts,
                )
                if decision.action != "delegate" or not decision.tasks:
                    llm_stop_reason = decision.reasoning or "no_further_delegation"
                    logger.info(
                        "SupervisorAgent: LLM Reviewer decided '%s' ('%s'). Halting review loop at round %d.",
                        decision.action,
                        llm_stop_reason,
                        current_round,
                    )
                    break

                logger.info(
                    "SupervisorAgent: LLM Reviewer scheduled %d follow-up tasks in round %d...",
                    len(decision.tasks),
                    current_round,
                )
                round_fu_tasks: List[Task] = []
                for i, t_info in enumerate(decision.tasks):
                    fut = Task(
                        task_id=f"task_llm_fu_{run_id}_{current_round}_{i+1}",
                        parent_task_id=task_integrate.task_id,
                        agent_name=t_info.get("agent_name", "temporal"),
                        action=t_info.get("action", "targeted_inquiry"),
                        objective=t_info.get("objective", f"Targeted follow-up #{i+1} by LLM reviewer (round {current_round})"),
                        payload=t_info.get("payload", {}),
                    )
                    round_fu_tasks.append(fut)
                    fu_tasks.append(fut)
                    all_tasks[fut.task_id] = fut

                fu_plan = ExecutionPlan(plan_id=f"plan_llm_fu_{run_id}_{current_round}", tasks=round_fu_tasks)
                fu_results = self.scheduler.execute_plan(fu_plan, self.registry)
                self._active_task_results.update(fu_results)

                for fut in round_fu_tasks:
                    res = fu_results.get(fut.task_id)
                    status_str = res.status.value if res else "UNKNOWN"
                    self.follow_up_controller.follow_up_logs.append({
                        "round": current_round,
                        "task_id": fut.task_id,
                        "agent_name": fut.agent_name,
                        "action": fut.action,
                        "status": status_str,
                        "reasoning": decision.reasoning,
                    })

                    # If task failed or output is missing: keep original evidence without pollution
                    if not res or res.status == TaskStatus.FAILED or res.output is None:
                        logger.warning("Follow-up task '%s' failed; keeping original evidence intact.", fut.task_id)
                        continue

                    # If task succeeded or degraded with valid output, merge EvidenceBus & update zone_features
                    if res.status in (TaskStatus.SUCCESS, TaskStatus.DEGRADED) and isinstance(res.output, AgentEvidence):
                        # 1. Merge into EvidenceBus
                        self.evidence_bus.merge_zone_evidence(fut.agent_name, res.output, task_id=fut.task_id)
                        merged_ev = self.evidence_bus.get_evidence(fut.agent_name)
                        if merged_ev is not None:
                            evidences[fut.agent_name] = merged_ev

                        # 2. Update zone_features for target zone
                        target_zone = fut.payload.get("region_id") or fut.payload.get("zone_id")
                        domain = fut.agent_name

                        domain_to_col = {
                            "temporal": "temporal_redundancy",
                            "flexibility": "intraday_flexibility",
                            "predictability": "predictability",
                            "spatial": "spatial_redundancy",
                            "functional": "functional_mix",
                        }
                        domain_to_score_key = {
                            "temporal": "temporal_redundancy_score",
                            "flexibility": "intraday_flexibility_score",
                            "predictability": "predictability_score",
                            "spatial": "spatial_redundancy_score",
                            "functional": "functional_mix_score",
                        }

                        col_name = domain_to_col.get(domain)
                        score_key = domain_to_score_key.get(domain)

                        if col_name and hasattr(res.output, "zone_metrics") and res.output.zone_metrics:
                            for zid, zmetrics in res.output.zone_metrics.items():
                                if target_zone and zid != target_zone:
                                    continue
                                if zid in zone_features.index:
                                    new_val = zmetrics.get(score_key, zmetrics.get(col_name))
                                    if new_val is not None and isinstance(new_val, (int, float)):
                                        zone_features.at[zid, col_name] = float(new_val)

                # Re-compute composite_resilience and vulnerability_score across zone_features
                recompute_zone_features(zone_features, self.weight_config, event_context)

                # 3. Re-evaluate conflicts
                self.evidence_bus.detect_conflicts(self.urban_context)

                current_round += 1
                if current_round > max_rounds and llm_stop_reason is None:
                    llm_stop_reason = "max_rounds_reached"

            current_total_end = int(getattr(self.llm_client, "token_usage", {}).get("total_tokens", 0))
            review_tokens_used = max(0, current_total_end - review_tokens_start)
        else:
            notes, follow_ups_needed = self.follow_up_controller.identify_follow_ups(zone_features)
            if follow_ups_needed:
                logger.info("SupervisorAgent: Scheduling %d follow-up tasks...", len(follow_ups_needed))
                fu_tasks = self.follow_up_controller.create_follow_up_tasks(follow_ups_needed)
                for fut in fu_tasks:
                    fut.parent_task_id = task_integrate.task_id
                    all_tasks[fut.task_id] = fut
                fu_plan = ExecutionPlan(plan_id=f"plan_fu_{run_id}", tasks=fu_tasks)
                fu_results = self.scheduler.execute_plan(fu_plan, self.registry)
                all_task_results = dict(self._active_task_results)
                all_task_results.update(fu_results)
                self._active_task_results = all_task_results

                zone_features, _ = self.follow_up_controller.apply_follow_up_results(
                    follow_ups_needed=follow_ups_needed,
                    task_results=fu_results,
                    zone_features=zone_features,
                    weights=self.weight_config,
                )
                self.synthesis_subagent.synthesis_agent.follow_up_logs = list(self.follow_up_controller.follow_up_logs)

        # Determine follow-up and coordination deltas
        current_evidence_snapshot = self.evidence_bus.get_canonical_metrics_snapshot()
        evidence_changed = (evidence_snapshot != current_evidence_snapshot)
        num_before = zone_features_before_followup.select_dtypes(include=np.number)
        num_after = zone_features.select_dtypes(include=np.number)
        zone_features_changed = bool(
            not zone_features_before_followup.index.equals(zone_features.index)
            or not np.allclose(num_before.values, num_after.values, atol=1e-3, rtol=1e-3)
        )

        follow_up_logs = list(self.follow_up_controller.follow_up_logs)
        follow_up_success_count = sum(
            1 for log in follow_up_logs
            if is_follow_up_success_status(log.get("status"))
        )
        follow_up_executed = len(fu_tasks) > 0 or len(follow_up_logs) > 0

        if not follow_up_executed:
            coordination_status = "not_attempted"
            coordination_exercised = False
        elif follow_up_success_count == 0:
            coordination_status = "follow_up_failed"
            coordination_exercised = False
        elif evidence_changed or zone_features_changed:
            coordination_status = "exercised"
            coordination_exercised = True
        else:
            coordination_status = "followed_up_no_effect"
            coordination_exercised = False

        # Calculate conflicts resolved: strictly for localized cross-domain conflicts
        self.evidence_bus.detect_conflicts(self.urban_context)
        remaining_cross_conflicts = list(self.evidence_bus.cross_domain_conflicts)
        remaining_conflict_keys = {
            (c.zone_id, tuple(sorted(c.domains)))
            for c in remaining_cross_conflicts
            if c.zone_id
        }
        remaining_conflict_msgs = {c.message for c in remaining_cross_conflicts}

        successful_targeted_zones = set()
        for log in follow_up_logs:
            if is_follow_up_success_status(log.get("status")) and log.get("target_zone"):
                successful_targeted_zones.add(log.get("target_zone"))
        for fut in fu_tasks:
            z = fut.payload.get("region_id") or fut.payload.get("zone_id")
            if z:
                fut_res = self._active_task_results.get(fut.task_id)
                if fut_res and is_follow_up_success_status(fut_res.status):
                    successful_targeted_zones.add(z)

        conflicts_resolved = 0
        resolved_conflicts_records: List[Dict[str, Any]] = []
        if conflicts_detected > 0:
            for c in initial_cross_domain_conflicts:
                c_zone = c.zone_id
                c_domains_key = tuple(sorted(c.domains))
                if (
                    c_zone in successful_targeted_zones
                    and (c_zone, c_domains_key) not in remaining_conflict_keys
                    and c.message not in remaining_conflict_msgs
                ):
                    c.resolved = True
                    conflicts_resolved += 1
                    resolved_conflicts_records.append(c.to_dict())
        else:
            conflicts_resolved = 0

        # Predictor status: check if trained ML predictor model weights exist and validate
        model_file = self.model_artifact_path
        if model_file is None or not model_file.exists():
            for candidate in [
                self.output_dir / "models" / "resilience_predictor.joblib",
                self.output_dir / "models" / "resilience_predictor.pt",
            ]:
                if candidate.exists():
                    model_file = candidate
                    break

        predictor_status: Dict[str, Any] = {}
        predictor_used: bool = False
        calibrator_used: bool = False
        city_pred_result: Optional[Any] = None

        if model_file is None or not model_file.exists():
            predictor_status = {
                "model_status": "untrained_fallback",
                "engine": "monte_carlo_physics",
                "summary_metric_source": "monte_carlo_physics",
                "curve_source": "monte_carlo_physics",
                "curve_metric_source": "monte_carlo_physics",
                "external_temporal_metric_source": "monte_carlo_physics",
                "metric_sources": {
                    "maximum_activity_loss": "monte_carlo_physics",
                    "recovery_time_hours": "monte_carlo_physics",
                    "cumulative_activity_loss": "monte_carlo_physics",
                    "peak_loss_hour": "monte_carlo_physics",
                    "resilience_triangle_area": "monte_carlo_physics",
                },
                "interval_source": "monte_carlo_quantile",
            }
        else:
            artifact_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
            try:
                predictor = ResiliencePredictor.load(model_file)
            except Exception as exc:
                logger.warning(
                    "Model artifact at %s failed validation (%s); falling back to physics simulation.",
                    model_file,
                    exc,
                )
                predictor_status = {
                    "model_status": "untrained_fallback",
                    "engine": "monte_carlo_physics",
                    "summary_metric_source": "monte_carlo_physics",
                    "curve_source": "monte_carlo_physics",
                    "curve_metric_source": "monte_carlo_physics",
                    "external_temporal_metric_source": "monte_carlo_physics",
                    "metric_sources": {
                        "maximum_activity_loss": "monte_carlo_physics",
                        "recovery_time_hours": "monte_carlo_physics",
                        "cumulative_activity_loss": "monte_carlo_physics",
                        "peak_loss_hour": "monte_carlo_physics",
                        "resilience_triangle_area": "monte_carlo_physics",
                    },
                    "interval_source": "monte_carlo_quantile",
                    "artifact_hash": artifact_hash,
                    "error": f"Invalid model artifact: {exc}",
                }
                predictor = None

            if predictor is not None:
                schema_ver = getattr(predictor, "feature_schema_version", "1.0.0")
                if predictor.model_version != "1.0.0" or schema_ver != "1.0.0":
                    predictor_status = {
                        "model_status": "untrained_fallback",
                        "engine": "monte_carlo_physics",
                        "summary_metric_source": "monte_carlo_physics",
                        "curve_source": "monte_carlo_physics",
                        "curve_metric_source": "monte_carlo_physics",
                        "external_temporal_metric_source": "monte_carlo_physics",
                        "metric_sources": {
                            "maximum_activity_loss": "monte_carlo_physics",
                            "recovery_time_hours": "monte_carlo_physics",
                            "cumulative_activity_loss": "monte_carlo_physics",
                            "peak_loss_hour": "monte_carlo_physics",
                            "resilience_triangle_area": "monte_carlo_physics",
                        },
                        "interval_source": "monte_carlo_quantile",
                        "artifact_hash": artifact_hash,
                        "error": f"Incompatible model_version '{predictor.model_version}' or schema_version '{schema_ver}'",
                    }
                else:
                    # Assemble features under strict mode
                    assembler = FeatureAssembler(self.urban_context)
                    intensity_map = event_context.zone_intensity_map if event_context else {}
                    try:
                        X_features = assembler.assemble_zone_features(
                            evidences,
                            hazard_intensity=intensity_map,
                            strict=True,
                        )
                    except Exception as exc:
                        logger.warning("Feature assembly failed under strict mode: %s; falling back.", exc)
                        training_meta = getattr(predictor, "training_metadata", {}) or {}
                        predictor_status = {
                            "model_status": "insufficient_features",
                            "engine": "monte_carlo_physics",
                            "summary_metric_source": "monte_carlo_physics",
                            "curve_source": "monte_carlo_physics",
                            "curve_metric_source": "monte_carlo_physics",
                            "external_temporal_metric_source": "monte_carlo_physics",
                            "metric_sources": {
                                "maximum_activity_loss": "monte_carlo_physics",
                                "recovery_time_hours": "monte_carlo_physics",
                                "cumulative_activity_loss": "monte_carlo_physics",
                                "peak_loss_hour": "monte_carlo_physics",
                                "resilience_triangle_area": "monte_carlo_physics",
                            },
                            "interval_source": "monte_carlo_quantile",
                            "model_version": predictor.model_version,
                            "feature_schema_version": schema_ver,
                            "artifact_hash": artifact_hash,
                            "training_metadata": training_meta,
                            "training_label_source": training_meta.get("label_source", "unknown"),
                            "engineering_only": bool(training_meta.get("engineering_only", False)),
                            "training_event_ids": list(training_meta.get("training_event_ids", [])),
                            "error": str(exc),
                        }
                        X_features = None

                    if X_features is not None:
                        calibrator = getattr(predictor, "calibrator", None)
                        city_pred_result = predictor.predict_structured(X_features, calibrator=calibrator)
                        has_cal = calibrator is not None and bool(getattr(calibrator, "nonconformity_quantiles", None))
                        predictor_used = True
                        calibrator_used = has_cal
                        interval_source_str = "conformal" if has_cal else "uncalibrated"
                        trained_metric_sources = {
                            "maximum_activity_loss": "trained_model",
                            "recovery_time_hours": "trained_model",
                            "cumulative_activity_loss": "trained_model",
                            "peak_loss_hour": "monte_carlo_physics",
                            "resilience_triangle_area": "monte_carlo_physics",
                        }
                        training_meta = getattr(predictor, "training_metadata", {}) or {}
                        training_label_source = training_meta.get("label_source", "unknown")
                        training_engineering_only = bool(training_meta.get("engineering_only", False))
                        training_event_ids = list(training_meta.get("training_event_ids", []))

                        predictor_status = {
                            "model_status": "trained_model",
                            "engine": "hist_gradient_boosting",
                            "summary_metric_source": "trained_model",
                            "curve_source": "monte_carlo_physics",
                            "curve_metric_source": "monte_carlo_physics",
                            "external_temporal_metric_source": "monte_carlo_physics",
                            "metric_sources": trained_metric_sources,
                            "interval_source": interval_source_str,
                            "model_type": predictor.model_type,
                            "model_version": predictor.model_version,
                            "feature_schema_version": schema_ver,
                            "artifact_hash": artifact_hash,
                            "model_path": str(model_file),
                            "calibrator": "conformal_quantile" if has_cal else "uncalibrated",
                            "city_max_loss": float(city_pred_result.city_max_loss),
                            "city_recovery_time_hours": float(city_pred_result.city_recovery_time_hours),
                            "city_cumulative_loss": float(city_pred_result.city_cumulative_loss),
                            "training_metadata": training_meta,
                            "training_label_source": training_label_source,
                            "engineering_only": training_engineering_only,
                            "training_event_ids": training_event_ids,
                        }

        # Forecast
        fu_dep_ids = [t.task_id for t in fu_tasks]
        task_forecast = Task(
            task_id=f"task_forecast_{run_id}",
            parent_task_id=f"plan_full_{run_id}",
            agent_name="synthesis",
            action="forecast",
            objective="Generate anticipatory resilience disruption curve and spatial outlook",
            dependencies=[task_integrate.task_id] + fu_dep_ids,
            payload={
                "evidences": evidences,
                "event_context": event_context,
                "zone_features": zone_features,
            },
        )
        all_tasks[task_forecast.task_id] = task_forecast
        res_forecast = self.scheduler.execute_task(task_forecast, self.registry)
        self._active_task_results[task_forecast.task_id] = res_forecast

        if res_forecast.status not in (TaskStatus.SUCCESS, TaskStatus.DEGRADED):
            total_elapsed = time.perf_counter() - start_time
            manifest = self._compile_manifest(
                run_id=run_id,
                event_context=event_context,
                total_elapsed=total_elapsed,
                all_tasks=all_tasks,
                output_manifest_path=output_manifest_path,
                save_manifest=save_manifest,
                forced_status=TaskStatus.FAILED,
                error_summary=f"Forecasting failed: {res_forecast.error}",
                stop_reason=llm_stop_reason,
                predictor_status=predictor_status,
            )
            self.latest_manifest = manifest
            raise OrchestrationError(f"Forecasting failed: {res_forecast.error}")

        prediction: ResiliencePrediction = res_forecast.output["prediction"]
        spatial_outlook: SpatialResilienceOutlook = res_forecast.output["spatial_outlook"]

        # Apply predictor summary metrics and provenance
        if predictor_used and city_pred_result is not None:
            prediction.disruption_metrics.maximum_activity_loss = float(city_pred_result.city_max_loss)
            prediction.disruption_metrics.recovery_time_hours = float(city_pred_result.city_recovery_time_hours)
            prediction.disruption_metrics.cumulative_activity_loss = float(city_pred_result.city_cumulative_loss)
            trained_metric_sources = {
                "maximum_activity_loss": "trained_model",
                "recovery_time_hours": "trained_model",
                "cumulative_activity_loss": "trained_model",
                "peak_loss_hour": "monte_carlo_physics",
                "resilience_triangle_area": "monte_carlo_physics",
            }
            prediction.provenance = PredictionProvenance(
                summary_metric_source="trained_model",
                curve_source="monte_carlo_physics",
                curve_metric_source="monte_carlo_physics",
                external_temporal_metric_source="monte_carlo_physics",
                metric_sources=trained_metric_sources,
                model_status="trained_model",
                model_version=predictor_status.get("model_version"),
                feature_schema_version=predictor_status.get("feature_schema_version"),
                artifact_hash=predictor_status.get("artifact_hash"),
                training_label_source=predictor_status.get("training_label_source", "unknown"),
                engineering_only=bool(predictor_status.get("engineering_only", False)),
                training_event_ids=list(predictor_status.get("training_event_ids", [])),
                interval_source="conformal" if calibrator_used else "uncalibrated",
                notes=[
                    "summary metrics from trained predictor; curve from monte_carlo_physics",
                    "resilience_triangle_area maintains curve integration value",
                ],
            )
            predictor_status["metric_sources"] = trained_metric_sources
            predictor_status["curve_metric_source"] = "monte_carlo_physics"
            predictor_status["external_temporal_metric_source"] = "monte_carlo_physics"
            predictor_status["predictor_role"] = "engineering_baseline"
            predictor_status["accuracy_claim"] = "unverified_on_real_data"
        else:
            err_msg = predictor_status.get("error")
            fallback_notes = [err_msg] if err_msg else ["Default physics-based Monte Carlo baseline; no trained ML weights"]
            fallback_metric_sources = {
                "maximum_activity_loss": "monte_carlo_physics",
                "recovery_time_hours": "monte_carlo_physics",
                "cumulative_activity_loss": "monte_carlo_physics",
                "peak_loss_hour": "monte_carlo_physics",
                "resilience_triangle_area": "monte_carlo_physics",
            }
            predictor_status["training_label_source"] = "none"
            predictor_status["engineering_only"] = True
            predictor_status["training_event_ids"] = []
            prediction.provenance = PredictionProvenance(
                summary_metric_source="monte_carlo_physics",
                curve_source="monte_carlo_physics",
                curve_metric_source="monte_carlo_physics",
                external_temporal_metric_source="monte_carlo_physics",
                metric_sources=fallback_metric_sources,
                model_status=predictor_status.get("model_status", "untrained_fallback"),
                model_version=predictor_status.get("model_version"),
                feature_schema_version=predictor_status.get("feature_schema_version"),
                artifact_hash=predictor_status.get("artifact_hash"),
                training_label_source="none",
                engineering_only=True,
                training_event_ids=[],
                interval_source="monte_carlo_quantile",
                notes=fallback_notes,
            )
            predictor_status["metric_sources"] = fallback_metric_sources
            predictor_status["curve_metric_source"] = "monte_carlo_physics"
            predictor_status["external_temporal_metric_source"] = "monte_carlo_physics"
            predictor_status["predictor_role"] = "engineering_baseline"
            predictor_status["accuracy_claim"] = "unverified_on_real_data"

        total_elapsed = time.perf_counter() - start_time

        manifest = self._compile_manifest(
            run_id=run_id,
            event_context=event_context,
            total_elapsed=total_elapsed,
            all_tasks=all_tasks,
            output_manifest_path=output_manifest_path,
            save_manifest=save_manifest,
            can_degrade=can_degrade,
            stop_reason=llm_stop_reason,
            predictor_status=predictor_status,
            prediction=prediction,
            review_tokens_used=review_tokens_used,
            follow_up_success_count=follow_up_success_count,
            follow_up_executed=follow_up_executed,
            evidence_changed=evidence_changed,
            zone_features_changed=zone_features_changed,
            coordination_status=coordination_status,
            coordination_exercised=coordination_exercised,
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            anomalies_detected=anomalies_detected,
            unlocalized_conflicts=unlocalized_conflicts,
            resolved_conflicts=resolved_conflicts_records,
            conflicts_list=[c["message"] for c in initial_cross_domain_conflicts],
        )
        self.latest_manifest = manifest

        if (manifest.status in (TaskStatus.FAILED, TaskStatus.DEGRADED) or manifest.failure_count > 0 or manifest.degraded_count > 0) and not can_degrade:
            raise OrchestrationError(
                f"Run completed with failures or degradation under strict allow_degraded=False: {manifest.error_summary or 'task failures/degradations'}"
            )

        return prediction, spatial_outlook, zone_features, manifest
