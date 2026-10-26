"""Four-group ablation experiment suite for urban resilience orchestration.

This module provides a reproducible runner comparing four orchestration paradigms:
1. fixed_features: Fixed deterministic pipeline baseline without adaptive follow-up.
2. single_adaptive: Single LLM domain agent (temporal only LLM, other domains deterministic, follow-ups disabled).
3. multi_uncoordinated: Multi-agent LLM ReAct assessment across all domains without supervisor cross-round review or coordination.
4. multi_coordinated: Full multi-agent LLM orchestration with supervisor review, conflict detection, and coordinated follow-ups.

NOTE ON EMPIRICAL VALIDITY:
Unless verified historical disaster observations and damage ground truth are explicitly
provided, all experiment outputs are marked with `engineering_only = True`. This confirms
that the runs verify engineering workflow execution rather than empirical benchmark superiority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import time

import numpy as np
import pandas as pd
from scipy import stats

from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import (
    LLMClient,
    MockLLMClient,
    default_mock_generator,
    get_llm_client,
)
from urban_resilience.audit import AuditMetrics, compute_audit_metrics
from urban_resilience.orchestration.contracts import SUPPORTED_LLM_DOMAINS
from urban_resilience.orchestration.evidence_bus import ConflictTargetSelector
from urban_resilience.orchestration.supervisor import OrchestrationError, SupervisorAgent
from urban_resilience.prediction.provenance import (
    get_provenance_warning,
    get_synthetic_warning,
    is_synthetic_trained,
)
from urban_resilience.types import (
    EventContext,
    HazardType,
    MobilityNetwork,
    UrbanContext,
)

logger = logging.getLogger(__name__)

ExperimentGroup = Literal[
    "fixed_features",
    "single_domain_adaptive",
    "single_adaptive",
    "multi_uncoordinated",
    "multi_coordinated",
]


class ResultsDict(dict):
    """Dictionary supporting alias lookups between single_adaptive and single_domain_adaptive."""

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key == "single_domain_adaptive" and super().__contains__("single_adaptive"):
            return super().__getitem__("single_adaptive")
        if key == "single_adaptive" and super().__contains__("single_domain_adaptive"):
            return super().__getitem__("single_domain_adaptive")
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if key == "single_domain_adaptive" and super().__contains__("single_adaptive"):
            return True
        if key == "single_adaptive" and super().__contains__("single_domain_adaptive"):
            return True
        return False

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class ExperimentResult:
    """Outcome metrics for a single experimental group."""

    group: str
    run_id: str
    status: str
    execution_time_seconds: float
    city_max_loss: float
    city_recovery_time_hours: float
    city_resilience_score: float
    follow_up_count: int
    tokens_used: int
    orchestration_mode: str
    name: str = ""
    llm_domains: List[str] = field(default_factory=list)
    coordination_enabled: bool = False
    predictor_status: str = "untrained_fallback"
    predictor_role: str = "engineering_baseline"
    token_usage_before: int = 0
    token_usage_after: int = 0
    review_tokens_used: int = 0
    follow_up_executed: bool = False
    follow_up_success_count: int = 0
    evidence_changed: bool = False
    zone_features_changed: bool = False
    coordination_status: str = "not_attempted"
    coordination_exercised: bool = False
    stop_reason: Optional[str] = None
    city_cumulative_loss: float = 0.0
    predictor_label_source: str = "none"
    predictor_engineering_only: bool = True
    degraded_count: int = 0
    failed_count: int = 0
    audit_metrics: Optional[AuditMetrics] = None
    synthetic_artifact_warning: Optional[str] = None
    provenance_warning: Optional[str] = None
    engineering_only: bool = True
    anomalies_detected: int = 0
    unlocalized_conflicts: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "name": self.name or self.group,
            "run_id": self.run_id,
            "status": self.status,
            "orchestration_mode": self.orchestration_mode,
            "llm_domains": list(self.llm_domains),
            "coordination_enabled": self.coordination_enabled,
            "city_max_loss": round(self.city_max_loss, 4),
            "city_recovery_time_hours": round(self.city_recovery_time_hours, 2),
            "city_resilience_score": round(self.city_resilience_score, 4),
            "city_cumulative_loss": round(self.city_cumulative_loss, 2),
            "follow_up_count": self.follow_up_count,
            "follow_up_executed": self.follow_up_executed,
            "follow_up_success_count": self.follow_up_success_count,
            "evidence_changed": self.evidence_changed,
            "zone_features_changed": self.zone_features_changed,
            "coordination_status": self.coordination_status,
            "coordination_exercised": self.coordination_exercised,
            "stop_reason": self.stop_reason,
            "token_usage_before": self.token_usage_before,
            "token_usage_after": self.token_usage_after,
            "tokens_used": self.tokens_used,
            "review_tokens_used": self.review_tokens_used,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "predictor_status": self.predictor_status or self.details.get("predictor_status", {}).get("model_status", "untrained_fallback"),
            "predictor_role": self.predictor_role,
            "predictor_label_source": self.predictor_label_source,
            "predictor_engineering_only": self.predictor_engineering_only,
            "degraded_count": self.degraded_count,
            "failed_count": self.failed_count,
            "evidence_coverage": self.audit_metrics.evidence_coverage if self.audit_metrics else None,
            "unsupported_numeric_claim_rate": self.audit_metrics.unsupported_numeric_claim_rate if self.audit_metrics else None,
            "anti_fabrication_rejection_count": self.audit_metrics.anti_fabrication_rejection_count if self.audit_metrics else 0,
            "tool_call_success_rate": self.audit_metrics.tool_call_success_rate if self.audit_metrics else None,
            "follow_up_yield": self.audit_metrics.follow_up_yield if self.audit_metrics else None,
            "conflicts_detected": self.audit_metrics.conflicts_detected if self.audit_metrics else 0,
            "conflicts_resolved": self.audit_metrics.conflicts_resolved if self.audit_metrics else 0,
            "conflict_resolution_rate": self.audit_metrics.conflict_resolution_rate if self.audit_metrics else None,
            "anomalies_detected": self.audit_metrics.anomalies_detected if self.audit_metrics else self.anomalies_detected,
            "unlocalized_conflicts": self.audit_metrics.unlocalized_conflicts if self.audit_metrics else self.unlocalized_conflicts,
            "degraded_rate": self.audit_metrics.degraded_rate if self.audit_metrics else None,
            "synthetic_artifact_warning": self.synthetic_artifact_warning,
            "provenance_warning": self.provenance_warning,
            "single_adaptive_domain": self.details.get("single_adaptive_domain"),
            "engineering_only": self.engineering_only,
        }


@dataclass
class ExperimentSuiteResult:
    """Consolidated outcome of the four-group ablation experiment across single or multiple seeds."""

    results: Dict[str, ExperimentResult]
    summary_df: pd.DataFrame
    raw_df: Optional[pd.DataFrame] = None
    results_by_seed: Dict[int, Dict[str, ExperimentResult]] = field(default_factory=dict)
    engineering_only: bool = True
    event_id: str = "SYNTHETIC_EXPERIMENT"
    event_seeds: List[int] = field(default_factory=lambda: [42])
    single_adaptive_domain: str = "temporal"

    @property
    def synthetic_artifact_warning(self) -> Optional[str]:
        """Returns synthetic warning if any result used a synthetic-trained artifact."""
        for res in self.results.values():
            if res.synthetic_artifact_warning:
                return res.synthetic_artifact_warning
        if self.raw_df is not None and "synthetic_artifact_warning" in self.raw_df.columns:
            warnings = self.raw_df["synthetic_artifact_warning"].dropna()
            if not warnings.empty:
                return str(warnings.iloc[0])
        return None

    def to_markdown(self) -> str:
        """Renders comparative results table in GitHub-flavored markdown."""
        seeds_str = ", ".join(str(s) for s in self.event_seeds)
        md_lines = [
            "# Urban Resilience Orchestration - Four-Group Ablation Experiment",
            "",
        ]
        if self.synthetic_artifact_warning:
            md_lines.extend([
                "> [!WARNING]",
                "> **Synthetic Predictor Artifact**: 当前实验使用了 synthetic simulator 标签训练的预测模型，不是真实历史灾害模型。",
                "",
            ])
        md_lines.extend([
            "> [!NOTE]",
            "> **Engineering Smoke Test Status**: `engineering_only = True`. All hazard events, mobility networks,",
            "> and target labels in this experiment suite are synthetically generated for workflow and pipeline verification.",
            "> They do NOT represent empirical historical ground truth.",
            "",
            f"- **Event Seeds**: `[{seeds_str}]`",
            f"- **Single Adaptive Domain**: `{self.single_adaptive_domain}`",
            f"- **Engineering Only**: `{self.engineering_only}`",
            "",
            "## 1. Multi-Seed Aggregated Performance (Mean ± Std [95% CI])",
            "",
            "| Group | Orchestration | Status | Coordination | Stop Reason | Max Loss (Mean ± Std [95% CI]) | Recovery Time (Mean ± Std [95% CI]) | Resilience Score | Avg Follow-ups | Avg Tokens | Avg Time (s) | Training Label | Engineering Only |",
            "| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for _, row in self.summary_df.iterrows():
            loss_mean = row.get("city_max_loss_mean", row.get("city_max_loss", 0.0)) * 100.0
            loss_std = row.get("city_max_loss_std", 0.0) * 100.0
            loss_ci_low = row.get("city_max_loss_ci95_lower", row.get("city_max_loss", 0.0)) * 100.0
            loss_ci_high = row.get("city_max_loss_ci95_upper", row.get("city_max_loss", 0.0)) * 100.0

            rec_mean = row.get("city_recovery_time_hours_mean", row.get("city_recovery_time_hours", 0.0))
            rec_std = row.get("city_recovery_time_hours_std", 0.0)
            rec_ci_low = row.get("city_recovery_time_hours_ci95_lower", rec_mean)
            rec_ci_high = row.get("city_recovery_time_hours_ci95_upper", rec_mean)

            loss_str = f"{loss_mean:.1f}% ± {loss_std:.1f}% [{loss_ci_low:.1f}%, {loss_ci_high:.1f}%]"
            rec_str = f"{rec_mean:.1f}h ± {rec_std:.1f}h [{rec_ci_low:.1f}h, {rec_ci_high:.1f}h]"

            score = row.get("city_resilience_score_mean", row.get("city_resilience_score", 0.5))
            fu = row.get("follow_up_count_mean", row.get("follow_up_count", 0))
            tok = row.get("tokens_used_mean", row.get("tokens_used", 0))
            t_sec = row.get("execution_time_seconds_mean", row.get("execution_time_seconds", 0.0))
            coordination_str = str(row.get("coordination_status") or "not_attempted")
            stop_reason_str = str(row.get("stop_reason") or "none")[:40]
            label_src = str(row.get("predictor_label_source", "none"))
            eng_only = str(row.get("predictor_engineering_only", row.get("engineering_only", True)))

            time_med = row.get("execution_time_median")
            time_min = row.get("execution_time_min")
            time_max = row.get("execution_time_max")
            if time_med is not None and time_min is not None and time_max is not None and len(self.event_seeds) > 1:
                time_str = f"{time_med:.2f}s [{time_min:.2f}s, {time_max:.2f}s]"
            else:
                time_str = f"{t_sec:.2f}s"

            md_lines.append(
                f"| `{row['group']}` | `{row['orchestration_mode']}` | {row['status']} | "
                f"`{coordination_str}` | {stop_reason_str} | "
                f"{loss_str} | {rec_str} | "
                f"{score:.3f} | {fu:.1f} | "
                f"{int(tok)} | {time_str} | {label_src} | {eng_only} |"
            )

        # Table 2: Paired Seed Comparison (relative to baselines)
        if len(self.results_by_seed) >= 1:
            md_lines.extend([
                "",
                "## 2. Paired Seed Comparison (`multi_coordinated` vs Baselines)",
                "",
                "| Seed | Baseline Group | Coordinated Max Loss | Baseline Max Loss | Paired Delta Loss | Coordinated Recovery | Baseline Recovery | Paired Delta Rec (h) | Resilience Score Gain |",
                "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for s, g_dict in self.results_by_seed.items():
                coord = g_dict.get("multi_coordinated")
                if not coord:
                    continue
                for bg in ["fixed_features", "single_adaptive", "multi_uncoordinated"]:
                    base = g_dict.get(bg) or (g_dict.get("single_domain_adaptive") if bg == "single_adaptive" else None)
                    if not base:
                        continue
                    d_loss = (coord.city_max_loss - base.city_max_loss) * 100.0
                    d_rec = coord.city_recovery_time_hours - base.city_recovery_time_hours
                    d_score = coord.city_resilience_score - base.city_resilience_score
                    md_lines.append(
                        f"| {s} | `{bg}` | {coord.city_max_loss*100:.1f}% | {base.city_max_loss*100:.1f}% | "
                        f"{d_loss:+.2f}% | {coord.city_recovery_time_hours:.1f}h | {base.city_recovery_time_hours:.1f}h | "
                        f"{d_rec:+.2f}h | {d_score:+.4f} |"
                    )

        # Table 3: Harness Governance & Audit Metrics
        md_lines.extend([
            "",
            "## 3. Harness Governance & Compliance Audit Metrics",
            "",
            "| Group | Evidence Coverage | Unsupported Claims | Anti-Fab Rejections | Tool Success Rate | Follow-up Yield | Conflicts Detected | Conflicts Resolved | Conflict Resolution | Anomalies Detected | Unlocalized Conflicts | Degraded Rate | Degraded Tasks | Failed Tasks | Total Tokens |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for _, row in self.summary_df.iterrows():
            def _fmt_pct(val: Any) -> str:
                if val is None or pd.isna(val) or val == "":
                    return "N/A"
                return f"{float(val)*100:.1f}%"

            def _fmt_float(val: Any) -> str:
                if val is None or pd.isna(val) or val == "":
                    return "N/A"
                return f"{float(val):.2f}"

            cov_str = _fmt_float(row.get("evidence_coverage"))
            unsupp_str = _fmt_float(row.get("unsupported_numeric_claim_rate"))
            rejs = int(row.get("anti_fabrication_rejection_count", 0))
            tsr_str = _fmt_pct(row.get("tool_call_success_rate"))
            yield_str = _fmt_pct(row.get("follow_up_yield"))
            conf_det = int(row.get("conflicts_detected", 0))
            conf_res = int(row.get("conflicts_resolved", 0))
            conf_res_str = _fmt_pct(row.get("conflict_resolution_rate"))
            anom_det = int(row.get("anomalies_detected", 0))
            unloc_conf = int(row.get("unlocalized_conflicts", 0))
            deg_rate_str = _fmt_pct(row.get("degraded_rate"))
            deg_cnt = int(row.get("degraded_count", 0))
            fail_cnt = int(row.get("failed_count", 0))
            tok = int(row.get("tokens_used_mean", row.get("tokens_used", 0)))
            md_lines.append(
                f"| `{row['group']}` | {cov_str} | {unsupp_str} | {rejs} | {tsr_str} | {yield_str} | "
                f"{conf_det} | {conf_res} | {conf_res_str} | {anom_det} | {unloc_conf} | "
                f"{deg_rate_str} | {deg_cnt} | {fail_cnt} | {tok} |"
            )

        if self.raw_df is not None and not self.raw_df.empty:
            md_lines.extend([
                "",
                "## 4. Per-Seed Run Breakdown",
                "",
                "| Seed | Group | Run ID | Max Loss (%) | Recovery Time (h) | Resilience Score | Follow-ups | Coordination | Stop Reason | Tokens | Time (s) | Predictor Status | Training Label | Engineering Only | Status |",
                "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |",
            ])
            for _, r in self.raw_df.iterrows():
                coord_r = str(r.get("coordination_status") or "not_attempted")
                stop_r = str(r.get("stop_reason") or "none")[:35]
                label_src = str(r.get("predictor_label_source", "none"))
                eng_only = str(r.get("predictor_engineering_only", r.get("engineering_only", True)))
                md_lines.append(
                    f"| {r['seed']} | `{r['group']}` | `{r['run_id'][:12]}` | "
                    f"{r['city_max_loss']*100:.1f}% | {r['city_recovery_time_hours']:.1f}h | "
                    f"{r['city_resilience_score']:.3f} | {r['follow_up_count']} | "
                    f"`{coord_r}` | {stop_r} | "
                    f"{r['tokens_used']} | {r['execution_time_seconds']:.2f}s | "
                    f"{r.get('predictor_status', 'untrained_fallback')} | {label_src} | {eng_only} | {r['status']} |"
                )

        md_lines.extend([
            "",
            "## 5. Operational Disclaimers & Integrity Declarations",
            "- **Single-Domain Reference**: `single_domain_adaptive` evaluates single-domain LLM adaptation (default: temporal) alongside deterministic rules; it serves as a single-domain reference ablation, NOT a monolithic single-agent baseline.",
            "- **Predictor Role**: `engineering_baseline` (The multi-target predictor is trained on synthetic physical simulator labels and serves strictly as an engineering pipeline baseline; empirical accuracy is unverified and NOT claimed).",
            "- **Data Status**: `engineering_only = True` (All disaster events and mobility matrices in this suite are synthetically simulated for verification).",
            "- **Empirical Data Status**: `BLOCKED (pending authoritative empirical datasets)`",
            "- **Predictor Status**: `engineering_baseline (unverified empirical accuracy)`",
            "",
        ])

        return "\n".join(md_lines)


def run_experiment_group(
    group: ExperimentGroup,
    urban_context: UrbanContext,
    mobility_network: MobilityNetwork,
    event_context: EventContext,
    llm_client: Optional[LLMClient] = None,
    output_dir: Optional[Union[Path, str]] = None,
    sim_config: Optional[SimulationConfig] = None,
    single_adaptive_domain: str = "temporal",
    model_artifact_path: Optional[Union[str, Path]] = None,
    engineering_only: bool = True,
    synthetic_review_delegate: bool = True,
    allow_degraded: bool = False,
) -> ExperimentResult:
    """Executes a single experimental group under controlled ablation settings."""
    if not engineering_only:
        raise ValueError("Real empirical validation is BLOCKED: engineering_only=False requires a verified data loader.")

    if single_adaptive_domain not in SUPPORTED_LLM_DOMAINS:
        raise ValueError(
            f"Unsupported single_adaptive_domain '{single_adaptive_domain}'. "
            f"Must be one of {sorted(list(SUPPORTED_LLM_DOMAINS))}."
        )

    canonical_group = "single_domain_adaptive" if group in ("single_adaptive", "single_domain_adaptive") else group
    if canonical_group == "fixed_features":
        llm_domains: List[str] = []
        coordination_enabled = False
    elif canonical_group == "single_domain_adaptive":
        llm_domains = [single_adaptive_domain]
        coordination_enabled = False
    elif canonical_group == "multi_uncoordinated":
        llm_domains = ["temporal", "flexibility", "predictability", "spatial", "functional"]
        coordination_enabled = False
    elif canonical_group == "multi_coordinated":
        llm_domains = ["temporal", "flexibility", "predictability", "spatial", "functional"]
        coordination_enabled = True
    else:
        llm_domains = []
        coordination_enabled = False

    out_dir = Path(output_dir) if output_dir else Path(f"outputs/experiments/{group}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = sim_config or SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=6,
        monte_carlo_samples=10,
    )

    t0 = time.perf_counter()

    if canonical_group == "fixed_features":
        # Group 1: Deterministic baseline with follow-ups disabled
        supervisor = SupervisorAgent(
            urban_context=urban_context,
            mobility_network=mobility_network,
            orchestration_mode="deterministic",
            model_artifact_path=model_artifact_path,
            output_dir=out_dir,
            sim_config=config,
            allow_degraded=allow_degraded,
            enable_follow_ups=False,
        )
        token_usage_before = 0
        pred, outlook, zf, manifest = supervisor.run(event_context=event_context, save_manifest=True)
        token_usage_after = 0
        tokens_used = 0

    elif canonical_group == "single_domain_adaptive":
        # Group 2: Single-domain LLM reference (configured domain only, default temporal), other 4 domains deterministic, follow-ups disabled
        client = llm_client or get_llm_client(mode="mock", auto_fallback=True)
        if isinstance(client, MockLLMClient) and not client._response_queue and client._generator is None:
            client.auto_fallback = True
        supervisor = SupervisorAgent(
            urban_context=urban_context,
            mobility_network=mobility_network,
            orchestration_mode="llm_hybrid",
            llm_client=client,
            llm_domain_keys={single_adaptive_domain},
            model_artifact_path=model_artifact_path,
            output_dir=out_dir,
            sim_config=config,
            allow_degraded=allow_degraded,
            enable_follow_ups=False,
        )
        token_usage_before = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        pred, outlook, zf, manifest = supervisor.run(event_context=event_context, save_manifest=True)
        token_usage_after = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        tokens_used = max(0, token_usage_after - token_usage_before)

    elif canonical_group == "multi_uncoordinated":
        # Group 3: Multi-agent ReAct Subagents across all domains, but NO cross-domain supervisor review delegation
        client = llm_client or get_llm_client(mode="mock", auto_fallback=True)
        if isinstance(client, MockLLMClient) and not client._response_queue and client._generator is None:
            client.auto_fallback = True

        supervisor = SupervisorAgent(
            urban_context=urban_context,
            mobility_network=mobility_network,
            orchestration_mode="llm_hybrid",
            llm_client=client,
            model_artifact_path=model_artifact_path,
            output_dir=out_dir,
            sim_config=config,
            allow_degraded=allow_degraded,
            enable_follow_ups=False,
        )
        token_usage_before = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        pred, outlook, zf, manifest = supervisor.run(event_context=event_context, save_manifest=True)
        token_usage_after = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        tokens_used = max(0, token_usage_after - token_usage_before)

    elif canonical_group == "multi_coordinated":
        # Group 4: Full multi-agent orchestration with supervisor review, targeted conflict resolution or information-gain follow-ups
        client = llm_client or get_llm_client(mode="mock", auto_fallback=True)
        if isinstance(client, MockLLMClient) and not client._response_queue and client._generator is None:
            client.auto_fallback = True

        supervisor = SupervisorAgent(
            urban_context=urban_context,
            mobility_network=mobility_network,
            orchestration_mode="llm_hybrid",
            llm_client=client,
            model_artifact_path=model_artifact_path,
            output_dir=out_dir,
            sim_config=config,
            allow_degraded=allow_degraded,
            enable_follow_ups=True,
        )

        injected_mock_reviewer = False
        original_generator = getattr(client, "_generator", None)
        if (
            synthetic_review_delegate
            and isinstance(client, MockLLMClient)
            and original_generator is None
        ):
            def synthetic_review_generator(
                messages: List[Dict[str, str]],
                schema: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                user_msg = messages[-1].get("content", "") if messages else ""
                if "review" in user_msg.lower():
                    if "round #1" in user_msg.lower() or "round 1" in user_msg.lower():
                        # Select target from EvidenceBus structured conflicts if any localized cross-domain conflict exists
                        target = ConflictTargetSelector.select_target(supervisor.evidence_bus.structured_conflicts)
                        if target is not None:
                            target_zone, target_domain, target_reason = target
                            c_id = None
                            for sc in supervisor.evidence_bus.cross_domain_conflicts:
                                if sc.zone_id == target_zone:
                                    c_id = getattr(sc, "conflict_id", None) or sc.get("conflict_id")
                                    break
                            return {
                                "action": "delegate",
                                "reasoning": f"conflict_targeted_follow_up: {target_reason}",
                                "tasks": [
                                    {
                                        "agent_name": target_domain,
                                        "action": "targeted_inquiry",
                                        "objective": f"Conflict-targeted follow-up for {target_zone} ({target_domain})",
                                        "payload": {
                                            "region_id": target_zone,
                                            "zone_id": target_zone,
                                            "conflict_id": c_id,
                                            "follow_up_source": "synthetic_refined_window",
                                            "query_context": {
                                                "refined_window_hours": 72,
                                                "baseline_type": "7day_profile",
                                            },
                                            "follow_up_query": {
                                                "refined_window_hours": 72,
                                                "baseline_type": "7day_profile",
                                            },
                                        },
                                    }
                                ],
                            }
                        else:
                            # No localized cross-domain conflict found: fallback to information gain follow-up
                            target_zone = urban_context.zone_ids[0]
                            target_domain = "temporal"
                            return {
                                "action": "delegate",
                                "reasoning": f"no_localized_conflict: information_gain_follow_up on {target_zone}",
                                "tasks": [
                                    {
                                        "agent_name": target_domain,
                                        "action": "targeted_inquiry",
                                        "objective": f"Information-gain follow-up re-evaluation for {target_zone}",
                                        "payload": {
                                            "region_id": target_zone,
                                            "zone_id": target_zone,
                                            "follow_up_source": "synthetic_refined_window",
                                            "query_context": {
                                                "refined_window_hours": 72,
                                                "baseline_type": "7day_profile",
                                            },
                                            "follow_up_query": {
                                                "refined_window_hours": 72,
                                                "baseline_type": "7day_profile",
                                            },
                                        },
                                    }
                                ],
                            }
                    else:
                        return {
                            "action": "stop",
                            "reasoning": "Synthetic one-round follow-up complete",
                            "tasks": [],
                        }
                return default_mock_generator(messages, schema)

            client.set_response_generator(synthetic_review_generator)
            injected_mock_reviewer = True

        token_usage_before = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        try:
            pred, outlook, zf, manifest = supervisor.run(event_context=event_context, save_manifest=True)
        finally:
            if injected_mock_reviewer:
                client._generator = original_generator

        token_usage_after = int(getattr(client, "token_usage", {}).get("total_tokens", 0))
        tokens_used = max(0, token_usage_after - token_usage_before)

    else:
        raise ValueError(f"Unknown experiment group '{group}'.")

    elapsed = time.perf_counter() - t0
    resilience_score = float(zf["composite_resilience"].mean()) if "composite_resilience" in zf.columns else 0.5
    manifest_meta = manifest.metadata or {}
    fu_count = len(supervisor.follow_up_controller.follow_up_logs)
    review_tokens_used = int(manifest_meta.get("review_tokens_used", 0))
    follow_up_success_count = int(manifest_meta.get("follow_up_success_count", 0))
    follow_up_executed = bool(manifest_meta.get("follow_up_executed", False))
    evidence_changed = bool(manifest_meta.get("evidence_changed", False))
    zone_features_changed = bool(manifest_meta.get("zone_features_changed", False))
    coordination_status = str(manifest_meta.get("coordination_status", "not_attempted"))
    coordination_exercised = bool(manifest_meta.get("coordination_exercised", False))

    stop_reason = (
        manifest.stop_reason
        or manifest_meta.get("stop_reason")
        or ("follow_ups_disabled" if not supervisor.enable_follow_ups else None)
    )

    pred_status = manifest.predictor_status or {}
    if is_synthetic_trained(pred_status):
        synthetic_artifact_warning = get_synthetic_warning(pred_status)
        predictor_label_source = str(pred_status.get("training_label_source", "synthetic_simulator"))
        predictor_engineering_only = True
    elif pred_status.get("model_status") == "trained_model":
        synthetic_artifact_warning = get_synthetic_warning(pred_status)
        predictor_label_source = str(pred_status.get("training_label_source", "unknown"))
        predictor_engineering_only = bool(pred_status.get("engineering_only", engineering_only))
    else:
        synthetic_artifact_warning = None
        predictor_label_source = "none"
        predictor_engineering_only = bool(pred_status.get("engineering_only", engineering_only))
    provenance_warning = get_provenance_warning(pred_status)

    from urban_resilience.orchestration.contracts import TaskStatus
    degraded_count = getattr(manifest, "degraded_count", 0) or sum(
        1 for t in manifest.tasks
        if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) in ("DEGRADED", TaskStatus.DEGRADED)
        or str(t.get("status") if isinstance(t, dict) else getattr(t, "status", "")).upper() == "DEGRADED"
    )
    failed_count = getattr(manifest, "failure_count", 0) or sum(
        1 for t in manifest.tasks
        if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) in ("FAILED", TaskStatus.FAILED)
        or str(t.get("status") if isinstance(t, dict) else getattr(t, "status", "")).upper() == "FAILED"
    )

    if not allow_degraded and (manifest.status not in (TaskStatus.SUCCESS, "SUCCESS") or degraded_count > 0 or failed_count > 0):
        manifest.status = TaskStatus.FAILED
        raise OrchestrationError(
            f"Group '{group}' had {degraded_count} degraded tasks and {failed_count} failed tasks under strict allow_degraded=False."
        )

    audit_metrics = compute_audit_metrics(
        manifest=manifest,
        tokens_used=tokens_used,
        latency_seconds=elapsed,
        follow_up_count=fu_count,
        follow_up_delta_count=1 if (evidence_changed or zone_features_changed) else 0,
        anti_fabrication_rejections=manifest_meta.get("validation_rejections", manifest_meta.get("anti_fabrication_rejection_count")),
        conflicts_detected=manifest_meta.get("conflicts_detected"),
        conflicts_resolved=manifest_meta.get("conflicts_resolved"),
        anomalies_detected=manifest_meta.get("anomalies_detected"),
        unlocalized_conflicts=manifest_meta.get("unlocalized_conflicts"),
        feature_value_count=manifest_meta.get("feature_value_count"),
        feature_with_refs_count=manifest_meta.get("feature_with_refs_count"),
        validation_attempts=manifest_meta.get("validation_attempts"),
        validation_rejections=manifest_meta.get("validation_rejections"),
    )

    return ExperimentResult(
        group=group,
        name=canonical_group,
        llm_domains=llm_domains,
        coordination_enabled=coordination_enabled,
        run_id=manifest.run_id,
        status=manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status),
        execution_time_seconds=elapsed,
        city_max_loss=pred.disruption_metrics.maximum_activity_loss,
        city_recovery_time_hours=pred.disruption_metrics.recovery_time_hours,
        city_resilience_score=resilience_score,
        city_cumulative_loss=pred.disruption_metrics.cumulative_activity_loss,
        follow_up_count=fu_count,
        follow_up_executed=follow_up_executed,
        follow_up_success_count=follow_up_success_count,
        evidence_changed=evidence_changed,
        zone_features_changed=zone_features_changed,
        coordination_status=coordination_status,
        coordination_exercised=coordination_exercised,
        tokens_used=tokens_used,
        token_usage_before=token_usage_before,
        token_usage_after=token_usage_after,
        review_tokens_used=review_tokens_used,
        stop_reason=stop_reason,
        orchestration_mode=supervisor.orchestration_mode,
        predictor_status=str(pred_status.get("model_status", "untrained_fallback")),
        predictor_role="engineering_baseline",
        predictor_label_source=predictor_label_source,
        predictor_engineering_only=predictor_engineering_only,
        degraded_count=degraded_count,
        failed_count=failed_count,
        audit_metrics=audit_metrics,
        anomalies_detected=int(manifest_meta.get("anomalies_detected", 0)),
        unlocalized_conflicts=int(manifest_meta.get("unlocalized_conflicts", 0)),
        synthetic_artifact_warning=synthetic_artifact_warning,
        provenance_warning=provenance_warning,
        engineering_only=engineering_only,
        details={
            "predictor_status": manifest.predictor_status,
            "single_adaptive_domain": single_adaptive_domain if canonical_group == "single_domain_adaptive" else None,
            "error_summary": manifest.error_summary,
            "stop_reason": stop_reason,
            "follow_up_summary": manifest.follow_up_summary,
            "review_tokens_used": review_tokens_used,
            "follow_up_executed": follow_up_executed,
            "follow_up_success_count": follow_up_success_count,
            "evidence_changed": evidence_changed,
            "zone_features_changed": zone_features_changed,
            "coordination_status": coordination_status,
            "coordination_exercised": coordination_exercised,
            "degraded_count": degraded_count,
            "failed_count": failed_count,
            "audit_metrics": audit_metrics.to_dict() if audit_metrics else {},
        },
    )


def run_four_group_experiments(
    urban_context: Optional[UrbanContext] = None,
    mobility_network: Optional[MobilityNetwork] = None,
    event_context: Optional[EventContext] = None,
    event_seeds: Optional[List[int]] = None,
    llm_client: Optional[LLMClient] = None,
    llm_mode: str = "mock",
    single_adaptive_domain: str = "temporal",
    output_dir: Optional[Union[Path, str]] = None,
    sim_config: Optional[SimulationConfig] = None,
    engineering_only: bool = True,
    model_artifact_path: Optional[Union[str, Path]] = None,
    synthetic_review_delegate: bool = True,
    verified_data_provenance: Optional[Dict[str, Any]] = None,
    allow_degraded: bool = False,
) -> ExperimentSuiteResult:
    """Runs all 4 experimental groups across one or multiple random seeds, aggregating metrics and 95% CIs.

    Args:
        urban_context: Target city context. Defaults to default Shenzhen context.
        mobility_network: Pre-event mobility network. Generated per-seed if event_seeds > 1.
        event_context: Disaster hazard context. Generated per-seed if event_seeds > 1.
        event_seeds: List of random seeds for multiple event repetitions (default: [42]).
        llm_client: LLM client for hybrid groups.
        llm_mode: Mode for obtaining LLM client when llm_client is None ('mock', 'env', 'auto').
        single_adaptive_domain: Cognitive domain assigned to LLM for single_adaptive (default: 'temporal').
        output_dir: Base directory for experiment runs and artifacts.
        sim_config: Simulation configuration.
        engineering_only: Explicit flag indicating whether runs represent engineering tests.
        model_artifact_path: Optional path to serialized ResiliencePredictor artifact.
        synthetic_review_delegate: When True and in mock mode, injects synthetic 1-round reviewer delegation for multi_coordinated.
        verified_data_provenance: Explicit verified provenance required if engineering_only is False.

    Returns:
        ExperimentSuiteResult containing individual group results, raw and aggregated summary DataFrames.
    """
    if not engineering_only:
        raise ValueError("Real empirical validation is BLOCKED: engineering_only=False requires a verified data loader.")

    if single_adaptive_domain not in SUPPORTED_LLM_DOMAINS:
        raise ValueError(
            f"Unsupported single_adaptive_domain '{single_adaptive_domain}'. "
            f"Must be one of {sorted(list(SUPPORTED_LLM_DOMAINS))}."
        )

    ctx = urban_context or UrbanSpatialDataLoader(city_name="Shenzhen").load_context()
    config = sim_config or SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=6,
        monte_carlo_samples=10,
    )

    seeds = list(event_seeds) if event_seeds is not None else [42]
    base_out = Path(output_dir) if output_dir else Path("outputs/experiments")
    base_out.mkdir(parents=True, exist_ok=True)

    # Initialize LLM Client
    client: Optional[LLMClient] = None
    if llm_client is not None:
        client = llm_client
    else:
        client = get_llm_client(mode=llm_mode if llm_mode in ("mock", "env", "auto") else "mock", auto_fallback=True)

    groups: List[ExperimentGroup] = [
        "fixed_features",
        "single_adaptive",
        "multi_uncoordinated",
        "multi_coordinated",
    ]

    latest_results: Dict[str, ExperimentResult] = ResultsDict()
    results_by_seed: Dict[int, Dict[str, ExperimentResult]] = {}
    raw_records: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info("Starting ablation evaluation on seed %d...", seed)
        results_by_seed[seed] = ResultsDict()
        seed_cfg = SimulationConfig(
            total_hours=config.total_hours,
            event_start_hour=config.event_start_hour,
            event_duration_hours=config.event_duration_hours,
            monte_carlo_samples=config.monte_carlo_samples,
            random_seed=seed,
        )

        # Generate mobility and event context for this seed (unless caller passed single custom objects)
        if mobility_network is not None and len(seeds) == 1:
            mob = mobility_network
        else:
            mob_loader = MobilityNetworkLoader(urban_context=ctx, random_seed=seed)
            mob = mob_loader.generate_mobility_dataset(total_hours=seed_cfg.total_hours)

        if event_context is not None and len(seeds) == 1:
            evt = event_context
        else:
            event_agent = EventIntelligenceAgent(urban_context=ctx)
            peak_intensity = float(np.clip(0.70 + (seed % 5) * 0.05, 0.5, 0.95))
            evt = event_agent.create_event_context(
                hazard_type=HazardType.RAIN_STORM,
                event_start_hour=seed_cfg.event_start_hour,
                duration_hours=seed_cfg.event_duration_hours,
                peak_intensity=peak_intensity,
                total_simulation_hours=seed_cfg.total_hours,
            )

        for g in groups:
            g_out = base_out / f"seed_{seed}" / g
            res = run_experiment_group(
                group=g,
                urban_context=ctx,
                mobility_network=mob,
                event_context=evt,
                llm_client=client,
                output_dir=g_out,
                sim_config=seed_cfg,
                single_adaptive_domain=single_adaptive_domain,
                model_artifact_path=model_artifact_path,
                engineering_only=engineering_only,
                synthetic_review_delegate=synthetic_review_delegate,
                allow_degraded=allow_degraded,
            )
            latest_results[g] = res
            results_by_seed[seed][g] = res

            rec = res.to_dict()
            rec["seed"] = seed
            raw_records.append(rec)

    raw_df = pd.DataFrame(raw_records)

    # Compute statistical aggregation per group across seeds (mean, std, 95% CI)
    agg_rows: List[Dict[str, Any]] = []
    n_seeds = len(seeds)

    for g in groups:
        sub = raw_df[raw_df["group"] == g]
        row_dict: Dict[str, Any] = {
            "group": g,
            "orchestration_mode": sub["orchestration_mode"].iloc[0] if not sub.empty else g,
            "status": sub["status"].iloc[0] if not sub.empty else "UNKNOWN",
            "n_seeds": n_seeds,
            "engineering_only": engineering_only,
        }

        # Calculate metrics statistics
        stat_cols = [
            ("city_max_loss", "city_max_loss"),
            ("city_recovery_time_hours", "city_recovery_time_hours"),
            ("city_resilience_score", "city_resilience_score"),
            ("city_cumulative_loss", "city_cumulative_loss"),
            ("follow_up_count", "follow_up_count"),
            ("tokens_used", "tokens_used"),
            ("review_tokens_used", "review_tokens_used"),
            ("execution_time_seconds", "execution_time_seconds"),
        ]

        t_crit = stats.t.ppf(0.975, df=n_seeds - 1) if n_seeds >= 2 else 0.0

        for col_name, raw_col in stat_cols:
            vals = sub[raw_col].values if raw_col in sub.columns else np.array([0.0])
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n_seeds >= 2 else 0.0
            ci_margin = float(t_crit * (std_val / np.sqrt(n_seeds))) if n_seeds >= 2 else 0.0

            # Store detailed stats
            row_dict[f"{col_name}_mean"] = round(mean_val, 4)
            row_dict[f"{col_name}_std"] = round(std_val, 4)
            row_dict[f"{col_name}_ci95_lower"] = round(mean_val - ci_margin, 4)
            row_dict[f"{col_name}_ci95_upper"] = round(mean_val + ci_margin, 4)

            # Store backward-compatible mean directly under col_name
            row_dict[col_name] = round(mean_val, 4) if col_name not in ("tokens_used", "review_tokens_used") else int(mean_val)

        # Execution time statistics: median, min, max
        time_vals = sub["execution_time_seconds"].values if "execution_time_seconds" in sub.columns else np.array([0.0])
        row_dict["execution_time_median"] = round(float(np.median(time_vals)), 4)
        row_dict["execution_time_min"] = round(float(np.min(time_vals)), 4)
        row_dict["execution_time_max"] = round(float(np.max(time_vals)), 4)

        # Governance & Audit metric aggregations
        def _mean_or_none(col_name: str) -> Optional[float]:
            if col_name in sub.columns:
                valid_vals = sub[col_name].dropna()
                if not valid_vals.empty:
                    return round(float(valid_vals.mean()), 4)
            return None

        row_dict["evidence_coverage"] = _mean_or_none("evidence_coverage")
        row_dict["unsupported_numeric_claim_rate"] = _mean_or_none("unsupported_numeric_claim_rate")
        row_dict["anti_fabrication_rejection_count"] = int(sub["anti_fabrication_rejection_count"].sum()) if "anti_fabrication_rejection_count" in sub.columns else 0
        row_dict["tool_call_success_rate"] = _mean_or_none("tool_call_success_rate")
        row_dict["follow_up_yield"] = _mean_or_none("follow_up_yield")
        row_dict["conflicts_detected"] = int(sub["conflicts_detected"].sum()) if "conflicts_detected" in sub.columns else 0
        row_dict["conflicts_resolved"] = int(sub["conflicts_resolved"].sum()) if "conflicts_resolved" in sub.columns else 0
        row_dict["conflict_resolution_rate"] = _mean_or_none("conflict_resolution_rate")
        row_dict["anomalies_detected"] = int(sub["anomalies_detected"].sum()) if "anomalies_detected" in sub.columns else 0
        row_dict["unlocalized_conflicts"] = int(sub["unlocalized_conflicts"].sum()) if "unlocalized_conflicts" in sub.columns else 0
        row_dict["degraded_rate"] = _mean_or_none("degraded_rate")
        row_dict["degraded_count"] = int(sub["degraded_count"].sum()) if "degraded_count" in sub.columns else 0
        row_dict["failed_count"] = int(sub["failed_count"].sum()) if "failed_count" in sub.columns else 0
        row_dict["predictor_role"] = (
            sub["predictor_role"].iloc[0] if "predictor_role" in sub.columns and not sub.empty else "engineering_baseline"
        )

        row_dict["coordination_status"] = (
            sub["coordination_status"].iloc[0] if "coordination_status" in sub.columns and not sub.empty else "not_attempted"
        )
        row_dict["coordination_exercised"] = bool(
            sub["coordination_exercised"].any() if "coordination_exercised" in sub.columns else False
        )
        row_dict["follow_up_executed"] = bool(
            sub["follow_up_executed"].any() if "follow_up_executed" in sub.columns else False
        )
        row_dict["evidence_changed"] = bool(
            sub["evidence_changed"].any() if "evidence_changed" in sub.columns else False
        )
        row_dict["zone_features_changed"] = bool(
            sub["zone_features_changed"].any() if "zone_features_changed" in sub.columns else False
        )
        stop_reasons = sub["stop_reason"].dropna().tolist() if "stop_reason" in sub.columns else []
        row_dict["stop_reason"] = stop_reasons[0] if stop_reasons else None
        row_dict["predictor_label_source"] = (
            sub["predictor_label_source"].iloc[0] if "predictor_label_source" in sub.columns and not sub.empty else "none"
        )
        row_dict["predictor_engineering_only"] = bool(
            sub["predictor_engineering_only"].any() if "predictor_engineering_only" in sub.columns else False
        )
        synth_warnings = sub["synthetic_artifact_warning"].dropna() if "synthetic_artifact_warning" in sub.columns else pd.Series()
        row_dict["synthetic_artifact_warning"] = (
            synth_warnings.iloc[0] if not synth_warnings.empty else None
        )

        agg_rows.append(row_dict)

    summary_df = pd.DataFrame(agg_rows)

    # Save outputs
    raw_df.to_csv(base_out / "experiment_raw_runs.csv", index=False)
    summary_df.to_csv(base_out / "experiment_summary.csv", index=False)
    logger.info("Saved multi-seed experiment summary CSV to %s", base_out / "experiment_summary.csv")

    suite_res = ExperimentSuiteResult(
        results=latest_results,
        summary_df=summary_df,
        raw_df=raw_df,
        results_by_seed=results_by_seed,
        engineering_only=engineering_only,
        event_id=f"SYNTHETIC_EXPERIMENT_SEEDS_{'_'.join(str(s) for s in seeds)}",
        event_seeds=seeds,
        single_adaptive_domain=single_adaptive_domain,
    )

    # Write summary markdown report
    with open(base_out / "experiment_report.md", "w", encoding="utf-8") as f:
        f.write(suite_res.to_markdown())

    return suite_res
