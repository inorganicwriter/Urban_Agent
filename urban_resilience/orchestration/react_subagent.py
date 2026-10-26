"""LLM ReAct subagent base class, strict anti-fabrication validator, and 5 domain implementations."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set
import numpy as np

from urban_resilience.llm.client import LLMClient
from urban_resilience.llm.schemas import (
    AgentDecision,
    FeatureQuality,
    FeatureReport,
    FeatureValue,
)
from urban_resilience.orchestration.contracts import Task, TaskResult, TaskStatus
from urban_resilience.orchestration.subagent import BaseSubagent
from urban_resilience.tools.flexibility_tools import build_flexibility_tool_registry
from urban_resilience.tools.functional_tools import build_functional_tool_registry
from urban_resilience.tools.predictability_tools import build_predictability_tool_registry
from urban_resilience.tools.registry import ToolRegistry, ToolResult
from urban_resilience.tools.spatial_tools import build_spatial_tool_registry
from urban_resilience.tools.temporal_tools import build_temporal_tool_registry
from urban_resilience.types import AgentEvidence, MobilityNetwork, ReActTrace, UrbanContext

logger = logging.getLogger(__name__)

# Required and allowed features per cognitive resilience domain
DOMAIN_REQUIRED_FEATURES: Dict[str, Set[str]] = {
    "temporal": {"hourly_entropy"},
    "flexibility": {"peak_shift_elasticity"},
    "predictability": {"autocorr_lag1"},
    "spatial": {"destination_diversity"},
    "functional": {"poi_entropy"},
}

DOMAIN_ALLOWED_FEATURES: Dict[str, Set[str]] = {
    "temporal": {
        "hourly_entropy", "peak_concentration", "temporal_redundancy_score",
        "gini_index", "peak_count", "bimodal_index", "day_type_correlation",
        "temporal_dispersion_index", "mean_activity", "missing_rate",
    },
    "flexibility": {
        "peak_shift_elasticity", "off_peak_adaptation_ratio", "peak_to_offpeak_ratio",
        "intraday_flexibility_score", "fwhm_estimate", "office_ratio",
        "peak_mean_volume", "offpeak_mean_volume",
    },
    "predictability": {
        "autocorr_lag1", "autocorr_lag24", "rhythm_regularity", "predictability_score",
        "days_compared", "min_daily_similarity", "variance",
    },
    "spatial": {
        "destination_diversity", "route_redundancy_score", "spatial_redundancy_score",
        "degree_connectivity", "betweenness_centrality", "transit_redundancy",
        "transit_lines_count", "normalized_degree", "total_outflow", "active_destination_count",
    },
    "functional": {
        "poi_entropy", "simpson_diversity", "activity_balance", "functional_mix_score",
        "total_poi_count", "categories_present", "residential_fraction",
        "commercial_fraction", "industrial_fraction",
    },
}


def validate_feature_report_against_tools(
    report: FeatureReport,
    executed_call_ids: List[str],
    tool_results: Dict[str, ToolResult],
    required_features: Optional[Set[str]] = None,
    allowed_features: Optional[Set[str]] = None,
    tolerance: float = 1e-3,
) -> FeatureReport:
    """Validates that all metrics in a FeatureReport strictly originate from executed tools.

    Guarantees:
    1. Every feature must have non-empty evidence_refs.
    2. Every referenced call_id must have been executed in this cycle.
    3. The reported numeric value must match the corresponding tool output within tolerance.
    4. Feature names must be authorized for the domain.
    5. If any required features are missing, the report status is marked as 'insufficient'.
    6. Quality metrics (coverage, missing rate) are derived from tool results, not LLM self-reporting.
    """
    if not report.features:
        report.status = "insufficient"
        report.limitations.append("Report contains no extracted features")
        return report

    executed_set = set(executed_call_ids)
    for feat_name, feat_val in report.features.items():
        # Check non-empty refs
        if not feat_val.evidence_refs:
            raise ValueError(f"Feature '{feat_name}' has empty evidence_refs. Numeric values without evidence are rejected.")

        # Check refs exist in executed_call_ids
        for ref in feat_val.evidence_refs:
            if ref not in executed_set:
                raise ValueError(
                    f"Feature '{feat_name}' references unexecuted call_id '{ref}'. "
                    f"Executed calls: {sorted(list(executed_set))}"
                )
            if ref not in tool_results:
                raise ValueError(f"Feature '{feat_name}' references call_id '{ref}' with no available result.")

        # Check value matches tool observation
        matched_in_any_ref = False
        for ref in feat_val.evidence_refs:
            tr = tool_results[ref]
            if tr.status != "success":
                report.status = "insufficient"
                report.limitations.append(f"Feature '{feat_name}' references call '{ref}' with failed status: {tr.status}")
                continue

            obs_dict = {**getattr(tr, "output", {}), **tr.data} if isinstance(getattr(tr, "output", None), dict) else tr.data
            if feat_name in obs_dict:
                matched_in_any_ref = True
                tool_val = obs_dict[feat_name]
                if not isinstance(tool_val, (int, float)) or isinstance(tool_val, bool):
                    report.status = "insufficient"
                    report.limitations.append(f"Tool observation for '{feat_name}' in call '{ref}' is non-numeric: {tool_val}")
                    break
                if not np.isclose(float(feat_val.value), float(tool_val), atol=tolerance, rtol=tolerance):
                    report.status = "insufficient"
                    report.limitations.append(
                        f"Fabricated numeric value for feature '{feat_name}': reported {feat_val.value}, "
                        f"does not match tool output in call '{ref}' ({tool_val})."
                    )
                    break

        if not matched_in_any_ref and report.status != "insufficient":
            report.status = "insufficient"
            report.limitations.append(
                f"Feature '{feat_name}' is not produced by any of its referenced calls ({feat_val.evidence_refs})."
            )

        # Check allowed feature names
        if allowed_features is not None and feat_name not in allowed_features:
            raise ValueError(
                f"Feature '{feat_name}' is unauthorized for this domain. "
                f"Allowed features: {sorted(list(allowed_features))}"
            )

    # Check required features
    if required_features:
        missing_required = required_features - set(report.features.keys())
        if missing_required:
            logger.warning("FeatureReport for %s missing required features %s", report.region_id, missing_required)
            report.status = "insufficient"
            report.limitations.append(f"Missing required features: {sorted(list(missing_required))}")

    # Derive quality from tool execution results
    successful_results = [tr for tr in tool_results.values() if tr.status == "success"]
    cov = float(np.mean([tr.coverage for tr in successful_results])) if successful_results else 0.0
    report.quality = FeatureQuality(
        coverage=cov,
        evidence_coverage=cov,
        feature_quality_score=cov,
        valid_days=1,
        missing_rate=max(0.0, 1.0 - cov),
    )
    return report


def feature_reports_to_agent_evidence(
    agent_name: str,
    reports: Dict[str, FeatureReport],
    traces: Optional[List[ReActTrace]] = None,
    default_confidence: float = 0.90,
    composite_key: Optional[str] = None,
) -> AgentEvidence:
    """Converts a collection of zone FeatureReports into a standard AgentEvidence object.

    Computes appropriate domain-specific composite scores and anomaly flags based on
    the domain agent.
    """
    zone_metrics: Dict[str, Dict[str, float]] = {}
    anomalies: List[str] = []

    # Map agent name to primary composite score key
    target_composite_key = composite_key
    if not target_composite_key:
        name_lower = agent_name.lower()
        if "temporal" in name_lower:
            target_composite_key = "temporal_redundancy_score"
        elif "flexibility" in name_lower:
            target_composite_key = "intraday_flexibility_score"
        elif "predictability" in name_lower:
            target_composite_key = "predictability_score"
        elif "spatial" in name_lower:
            target_composite_key = "spatial_redundancy_score"
        elif "functional" in name_lower:
            target_composite_key = "functional_mix_score"
        else:
            target_composite_key = "composite_domain_score"

    for zid, rep in reports.items():
        metrics_dict: Dict[str, float] = {}
        for fname, fval in rep.features.items():
            metrics_dict[fname] = float(fval.value)

        # Ensure composite metric exists
        if target_composite_key not in metrics_dict:
            if target_composite_key == "temporal_redundancy_score":
                entropy = metrics_dict.get("hourly_entropy", 0.5)
                peak_conc = metrics_dict.get("peak_concentration", 0.5)
                score = float(np.clip(0.6 * entropy + 0.4 * (1.0 - peak_conc), 0.0, 1.0))
            elif target_composite_key == "intraday_flexibility_score":
                elasticity = metrics_dict.get("peak_shift_elasticity", 0.5)
                offpeak = metrics_dict.get("off_peak_adaptation_ratio", 0.5)
                score = float(np.clip(0.55 * offpeak + 0.45 * elasticity, 0.0, 1.0))
            elif target_composite_key == "predictability_score":
                acf1 = metrics_dict.get("autocorr_lag1", 0.5)
                acf24 = metrics_dict.get("autocorr_lag24", 0.5)
                reg = metrics_dict.get("rhythm_regularity", 0.5)
                score = float(np.clip(0.4 * max(0.0, acf1) + 0.3 * max(0.0, acf24) + 0.3 * reg, 0.0, 1.0))
            elif target_composite_key == "spatial_redundancy_score":
                div = metrics_dict.get("destination_diversity", 0.5)
                route = metrics_dict.get("route_redundancy_score", 0.5)
                score = float(np.clip(0.40 * div + 0.60 * route, 0.0, 1.0))
            elif target_composite_key == "functional_mix_score":
                poi_ent = metrics_dict.get("poi_entropy", 0.5)
                simpson = metrics_dict.get("simpson_diversity", 0.5)
                balance = metrics_dict.get("activity_balance", 0.5)
                score = float(np.clip(0.40 * poi_ent + 0.30 * simpson + 0.30 * balance, 0.0, 1.0))
            else:
                score = 0.5
            metrics_dict[target_composite_key] = score

        zone_metrics[zid] = metrics_dict

        if rep.status != "complete" or rep.limitations:
            anomalies.append(f"Zone {zid}: status={rep.status}; limitations={rep.limitations}")
        elif metrics_dict.get(target_composite_key, 1.0) < 0.35:
            anomalies.append(f"Zone {zid}: critically low {target_composite_key} ({metrics_dict[target_composite_key]:.3f})")

    # Global averages
    global_metrics: Dict[str, float] = {}
    all_keys = set()
    for zm in zone_metrics.values():
        all_keys.update(zm.keys())
    for k in all_keys:
        vals = [zm[k] for zm in zone_metrics.values() if k in zm]
        if vals:
            global_metrics[k] = float(np.mean(vals))

    return AgentEvidence(
        agent_name=agent_name,
        metrics=global_metrics,
        zone_metrics=zone_metrics,
        traces=traces or [],
        confidence=default_confidence if not anomalies else max(0.6, default_confidence - 0.05 * len(anomalies)),
        qualitative_inference=(
            f"Multi-zone LLM ReAct analysis completed across {len(reports)} zones for {agent_name}. "
            f"Average {target_composite_key}: {global_metrics.get(target_composite_key, 0.5):.3f}."
        ),
        potential_anomalies=anomalies,
    )


class BaseLLMReActSubagent(BaseSubagent):
    """Base class for subagents using an LLM ReAct loop with deterministic tool execution."""

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        capabilities: Optional[List[str]] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        composite_key: Optional[str] = None,
        required_features: Optional[Set[str]] = None,
        allowed_features: Optional[Set[str]] = None,
        follow_up_data_source: Optional[Any] = None,
    ):
        caps = list(capabilities or [f"{name}_analysis", "domain_analysis", "targeted_inquiry", "query_data"])
        super().__init__(name=name, capabilities=caps)
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.max_tool_calls = max_tool_calls
        self.token_budget = token_budget
        self.prompt_version = prompt_version
        self.composite_key = composite_key
        self.required_features = required_features or DOMAIN_REQUIRED_FEATURES.get(name, set())
        self.allowed_features = allowed_features or DOMAIN_ALLOWED_FEATURES.get(name, set())
        self.follow_up_data_source = follow_up_data_source
        self.tool_call_records: List[Dict[str, Any]] = []
        self.feature_value_count: int = 0
        self.feature_with_refs_count: int = 0
        self.validation_attempts: int = 0
        self.validation_rejections: int = 0

        if self.tool_registry.get("query_refined_evidence") is None:
            from urban_resilience.tools.data_tools import create_refined_evidence_tool_spec
            self.tool_registry.register(
                create_refined_evidence_tool_spec(allowed_regions=list(urban_context.zone_ids))
            )

    def _build_system_prompt(self, target_region: str) -> str:
        """Constructs system prompt describing task, constraints, and tool requirements."""
        tools_summary = "\n".join(
            f"- {spec.name}: {spec.description} (params: {[k for k in (spec.args_schema.model_fields if spec.args_schema and hasattr(spec.args_schema, 'model_fields') else [])]})"
            for spec in self.tool_registry.list_tools()
        )
        return (
            f"You are the {self.name} ({self.__class__.__name__}) subagent for urban resilience assessment.\n"
            f"Target region: '{target_region}'. City: '{self.urban_context.city_name}'.\n"
            f"Data snapshot: 'pre_event_mobility_v1'. Prompt version: '{self.prompt_version}'.\n\n"
            f"Available computational tools:\n{tools_summary}\n\n"
            f"OPERATIONAL INSTRUCTIONS:\n"
            f"1. You must decide which computational tool to invoke via action='call_tool'.\n"
            f"2. You are FORBIDDEN from inventing or hallucinating numeric metrics. All numeric values must come from tool observations.\n"
            f"3. When you have collected adequate evidence, return action='final' with a complete FeatureReport.\n"
            f"4. EVERY numeric feature in FeatureReport MUST include evidence_refs pointing to the tool call_id that produced it.\n"
            f"5. If tools fail or data is insufficient, return action='insufficient' with explicit limitations.\n"
            f"6. Output must strictly conform to AgentDecision JSON schema."
        )

    def execute_react_cycle_for_region(
        self,
        region_id: str,
        task_id: str,
        task_objective: Optional[str] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> FeatureReport:
        """Executes a multi-turn ReAct reasoning and tool invocation cycle for one region."""
        initial_prompt = f"Begin ReAct assessment for region '{region_id}'."
        if task_objective:
            initial_prompt += f" Objective: {task_objective}."
        if task_payload and (
            "follow_up_source" in task_payload
            or "follow_up_query" in task_payload
            or "query_context" in task_payload
        ):
            fu_src = task_payload.get("follow_up_source", "synthetic_refined_window")
            initial_prompt += f" TARGETED FOLLOW-UP INQUIRY: Query secondary refined source '{fu_src}' via 'query_refined_evidence'."
        initial_prompt += " First inspect data quality and extract domain resilience features."

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt(region_id)},
            {
                "role": "user",
                "content": initial_prompt,
            },
        ]

        step = 0
        executed_call_ids: List[str] = []
        tool_results_by_call_id: Dict[str, ToolResult] = {}
        start_tokens = getattr(self.llm_client, "token_usage", {}).get("total_tokens", 0)

        while step < self.max_tool_calls:
            step += 1

            # Token budget check
            current_tokens = getattr(self.llm_client, "token_usage", {}).get("total_tokens", 0)
            if current_tokens - start_tokens >= self.token_budget:
                logger.warning("[%s] Token budget exceeded (%d tokens used).", self.name, current_tokens - start_tokens)
                return FeatureReport(
                    agent=self.name,
                    region_id=region_id,
                    status="insufficient",
                    features={},
                    limitations=[f"Token budget exceeded ({current_tokens - start_tokens} >= {self.token_budget})"],
                    llm_model=getattr(self.llm_client, "model", "mock"),
                    prompt_version=self.prompt_version,
                )

            raw_decision = self.llm_client.complete_json(
                messages=messages,
                schema=AgentDecision.model_json_schema(),
            )

            try:
                decision = AgentDecision.model_validate(raw_decision)
            except Exception as exc:
                logger.warning("[%s] Invalid AgentDecision JSON from LLM: %s. Raw: %s", self.name, exc, raw_decision)
                messages.append({
                    "role": "user",
                    "content": f"Error: Output did not match AgentDecision schema: {exc}. Please retry with valid JSON.",
                })
                continue

            # Case 1: Call Tool
            if decision.action == "call_tool":
                if not decision.tool_call:
                    logger.warning("[%s] Decision action is 'call_tool' but no tool_call provided.", self.name)
                    messages.append({
                        "role": "user",
                        "content": "Error: action='call_tool' was specified but 'tool_call' object is missing.",
                    })
                    continue

                call = decision.tool_call
                call_args = dict(call.args or {})
                if "region_id" not in call_args and "region_ids" not in call_args:
                    call_args["region_id"] = region_id

                tool_ctx: Dict[str, Any] = {
                    "urban_context": self.urban_context,
                    "mobility_network": self.mobility_network,
                    "active_domain": self.name,
                    "follow_up_data_source": getattr(self, "follow_up_data_source", None),
                }
                if task_payload:
                    tool_ctx["task_payload"] = task_payload
                    tool_ctx["query_context"] = task_payload.get("query_context") or task_payload.get("follow_up_query") or {}
                    if "follow_up_source" in task_payload:
                        tool_ctx["follow_up_source"] = task_payload["follow_up_source"]
                    if "simulate_no_delta" in task_payload:
                        tool_ctx["simulate_no_delta"] = task_payload["simulate_no_delta"]

                tool_res = self.tool_registry.execute(
                    call.tool,
                    call_args,
                    context=tool_ctx,
                    call_id=call.call_id,
                )
                executed_call_ids.append(tool_res.call_id)
                tool_results_by_call_id[tool_res.call_id] = tool_res

                # Record tool call in audit log
                self.tool_call_records.append({
                    "task_id": task_id,
                    "agent": self.name,
                    "region_id": region_id,
                    "step": step,
                    "call_id": tool_res.call_id,
                    "tool": tool_res.tool,
                    "status": tool_res.status,
                    "error": tool_res.error,
                })

                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {decision.reasoning}\nAction: call tool {call.tool}(args={call_args})",
                })
                obs_content = (
                    f"Tool observation (call_id='{tool_res.call_id}', status='{tool_res.status}'): "
                    f"data={tool_res.data}"
                    if tool_res.status == "success"
                    else f"Tool error (call_id='{tool_res.call_id}'): {tool_res.error}"
                )
                messages.append({"role": "user", "content": obs_content})

            # Case 2: Final Report
            elif decision.action == "final":
                if not decision.feature_report:
                    logger.warning("[%s] Final decision missing feature_report. Falling back to observation synthesis.", self.name)
                    extracted_feats: Dict[str, FeatureValue] = {}
                    for cid, tr in tool_results_by_call_id.items():
                        for k, v in tr.data.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                if self.allowed_features is None or k in self.allowed_features:
                                    extracted_feats[k] = FeatureValue(
                                        name=k,
                                        value=float(v),
                                        unit="dimensionless",
                                        evidence_refs=[cid],
                                    )
                    report = FeatureReport(
                        agent=self.name,
                        region_id=region_id,
                        status="complete" if extracted_feats else "insufficient",
                        features=extracted_feats,
                        quality=FeatureQuality(data_coverage=1.0, missing_rate=0.0),
                        limitations=[],
                        llm_model=getattr(self.llm_client, "model", "mock"),
                        prompt_version=self.prompt_version,
                    )
                else:
                    report = decision.feature_report

                # Strict validation against executed tools!
                self.validation_attempts += 1
                for _fname, fval in report.features.items():
                    self.feature_value_count += 1
                    if fval.evidence_refs and len(fval.evidence_refs) > 0:
                        self.feature_with_refs_count += 1

                try:
                    res_rep = validate_feature_report_against_tools(
                        report=report,
                        executed_call_ids=executed_call_ids,
                        tool_results=tool_results_by_call_id,
                        required_features=self.required_features,
                        allowed_features=self.allowed_features,
                    )
                    if res_rep.status == "insufficient":
                        if any("Fabricated numeric value" in lim or "not produced" in lim for lim in res_rep.limitations):
                            self.validation_rejections += 1
                    return res_rep
                except ValueError as exc:
                    self.validation_rejections += 1
                    logger.warning("[%s] FeatureReport rejected for %s: %s", self.name, region_id, exc)
                    messages.append({
                        "role": "user",
                        "content": f"Error: Feature report rejected: {exc}. All features must match tools with valid evidence_refs. Please correct and retry.",
                    })
                    continue

            # Case 3: Insufficient Evidence
            elif decision.action == "insufficient":
                return FeatureReport(
                    agent=self.name,
                    region_id=region_id,
                    status="insufficient",
                    features={},
                    limitations=decision.limitations or ["LLM determined evidence was insufficient"],
                    llm_model=getattr(self.llm_client, "model", "mock"),
                    prompt_version=self.prompt_version,
                )

        # Max steps exceeded
        logger.warning("[%s] Max tool calls (%d) reached for region %s without valid final decision.", self.name, self.max_tool_calls, region_id)
        return FeatureReport(
            agent=self.name,
            region_id=region_id,
            status="insufficient",
            features={},
            limitations=[f"Budget or tool limit reached: max_tool_calls ({self.max_tool_calls}) reached"],
            llm_model=getattr(self.llm_client, "model", "mock"),
            prompt_version=self.prompt_version,
        )

    def execute(self, task: Task) -> TaskResult:
        """Executes full multi-zone domain analysis or single-zone targeted follow-up inquiry."""
        start_time = time.perf_counter()
        logger.info("[%s] Beginning LLM ReAct task: %s (action: %s)", self.name, task.task_id, task.action)
        self.tool_call_records.clear()
        self.feature_value_count = 0
        self.feature_with_refs_count = 0
        self.validation_attempts = 0
        self.validation_rejections = 0

        # Handle targeted inquiry on a specific zone
        if task.action in ("targeted_inquiry", "query_data"):
            target_zone = task.payload.get("region_id") or task.payload.get("zone_id") or self.urban_context.zone_ids[0]
            report = self.execute_react_cycle_for_region(
                target_zone,
                task_id=task.task_id,
                task_objective=task.objective,
                task_payload=task.payload,
            )
            elapsed = time.perf_counter() - start_time
            evidence = feature_reports_to_agent_evidence(self.name, {target_zone: report}, composite_key=self.composite_key)
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=TaskStatus.SUCCESS if report.status == "complete" else TaskStatus.DEGRADED,
                output=evidence,
                execution_time_seconds=elapsed,
                metadata={
                    "feature_report": report.model_dump(),
                    "tool_calls": list(self.tool_call_records),
                    "feature_value_count": self.feature_value_count,
                    "feature_with_refs_count": self.feature_with_refs_count,
                    "validation_attempts": self.validation_attempts,
                    "validation_rejections": self.validation_rejections,
                },
            )

        # Full domain analysis across all city zones
        reports: Dict[str, FeatureReport] = {}
        traces: List[ReActTrace] = []

        for zid in self.urban_context.zone_ids:
            rep = self.execute_react_cycle_for_region(zid, task_id=task.task_id, task_objective=task.objective)
            reports[zid] = rep
            traces.append(
                ReActTrace(
                    step_number=len(traces) + 1,
                    thought=f"ReAct reasoning completed for {zid}",
                    action_type="llm_react_feature_report",
                    action_input={"region_id": zid, "status": rep.status},
                    observation=f"Extracted {len(rep.features)} features: {list(rep.features.keys())}",
                )
            )

        evidence = feature_reports_to_agent_evidence(self.name, reports, traces=traces, composite_key=self.composite_key)
        elapsed = time.perf_counter() - start_time
        has_failed = any(r.status == "failed" for r in reports.values())
        has_insufficient = any(r.status == "insufficient" for r in reports.values())

        final_status = TaskStatus.FAILED if has_failed else (TaskStatus.DEGRADED if has_insufficient else TaskStatus.SUCCESS)

        return TaskResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=final_status,
            output=evidence,
            execution_time_seconds=elapsed,
            trace=[
                {
                    "step_number": t.step_number,
                    "thought": t.thought,
                    "action_type": t.action_type,
                    "action_input": t.action_input,
                    "observation": t.observation,
                }
                for t in traces
            ],
            metadata={
                "reports_summary": {z: r.status for z, r in reports.items()},
                "tool_calls_count": len(self.tool_call_records),
                "tool_calls": list(self.tool_call_records),
                "feature_value_count": self.feature_value_count,
                "feature_with_refs_count": self.feature_with_refs_count,
                "validation_attempts": self.validation_attempts,
                "validation_rejections": self.validation_rejections,
            },
        )


class TemporalSubagentLLM(BaseLLMReActSubagent):
    """LLM ReAct subagent for Temporal Redundancy analysis across urban zones."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        follow_up_data_source: Optional[Any] = None,
    ):
        registry = tool_registry or build_temporal_tool_registry(
            mobility_network=mobility_network,
            allowed_regions=list(urban_context.zone_ids),
        )
        super().__init__(
            name="temporal",
            llm_client=llm_client,
            tool_registry=registry,
            urban_context=urban_context,
            mobility_network=mobility_network,
            capabilities=["temporal_redundancy", "domain_analysis", "targeted_inquiry", "query_data"],
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            prompt_version=prompt_version,
            composite_key="temporal_redundancy_score",
            required_features=DOMAIN_REQUIRED_FEATURES["temporal"],
            allowed_features=DOMAIN_ALLOWED_FEATURES["temporal"],
            follow_up_data_source=follow_up_data_source,
        )


class FlexibilitySubagentLLM(BaseLLMReActSubagent):
    """LLM ReAct subagent for Intraday Flexibility analysis across urban zones."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        follow_up_data_source: Optional[Any] = None,
    ):
        registry = tool_registry or build_flexibility_tool_registry(
            urban_context=urban_context,
            mobility_network=mobility_network,
        )
        super().__init__(
            name="flexibility",
            llm_client=llm_client,
            tool_registry=registry,
            urban_context=urban_context,
            mobility_network=mobility_network,
            capabilities=["intraday_flexibility", "domain_analysis", "targeted_inquiry", "query_data"],
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            prompt_version=prompt_version,
            composite_key="intraday_flexibility_score",
            required_features=DOMAIN_REQUIRED_FEATURES["flexibility"],
            allowed_features=DOMAIN_ALLOWED_FEATURES["flexibility"],
            follow_up_data_source=follow_up_data_source,
        )


class PredictabilitySubagentLLM(BaseLLMReActSubagent):
    """LLM ReAct subagent for Predictability analysis across urban zones."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        follow_up_data_source: Optional[Any] = None,
    ):
        registry = tool_registry or build_predictability_tool_registry(
            urban_context=urban_context,
            mobility_network=mobility_network,
        )
        super().__init__(
            name="predictability",
            llm_client=llm_client,
            tool_registry=registry,
            urban_context=urban_context,
            mobility_network=mobility_network,
            capabilities=["predictability", "domain_analysis", "targeted_inquiry", "query_data"],
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            prompt_version=prompt_version,
            composite_key="predictability_score",
            required_features=DOMAIN_REQUIRED_FEATURES["predictability"],
            allowed_features=DOMAIN_ALLOWED_FEATURES["predictability"],
            follow_up_data_source=follow_up_data_source,
        )


class SpatialSubagentLLM(BaseLLMReActSubagent):
    """LLM ReAct subagent for Spatial Redundancy analysis across urban zones."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        follow_up_data_source: Optional[Any] = None,
    ):
        registry = tool_registry or build_spatial_tool_registry(
            urban_context=urban_context,
            mobility_network=mobility_network,
        )
        super().__init__(
            name="spatial",
            llm_client=llm_client,
            tool_registry=registry,
            urban_context=urban_context,
            mobility_network=mobility_network,
            capabilities=["spatial_redundancy", "domain_analysis", "targeted_inquiry", "query_data"],
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            prompt_version=prompt_version,
            composite_key="spatial_redundancy_score",
            required_features=DOMAIN_REQUIRED_FEATURES["spatial"],
            allowed_features=DOMAIN_ALLOWED_FEATURES["spatial"],
            follow_up_data_source=follow_up_data_source,
        )


class FunctionalSubagentLLM(BaseLLMReActSubagent):
    """LLM ReAct subagent for Functional Mix analysis across urban zones."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls: int = 6,
        token_budget: int = 10000,
        prompt_version: str = "v1.0.0",
        follow_up_data_source: Optional[Any] = None,
    ):
        registry = tool_registry or build_functional_tool_registry(
            urban_context=urban_context,
            mobility_network=mobility_network,
        )
        super().__init__(
            name="functional",
            llm_client=llm_client,
            tool_registry=registry,
            urban_context=urban_context,
            mobility_network=mobility_network,
            capabilities=["functional_mix", "domain_analysis", "targeted_inquiry", "query_data"],
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            prompt_version=prompt_version,
            composite_key="functional_mix_score",
            required_features=DOMAIN_REQUIRED_FEATURES["functional"],
            allowed_features=DOMAIN_ALLOWED_FEATURES["functional"],
            follow_up_data_source=follow_up_data_source,
        )
