"""Harness governance and audit metrics for agentic urban resilience simulations.

Computes 9 standard audit indicators tracking anti-fabrication enforcement, tool
call reliability, follow-up information yield, degradation frequency, and computational cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AuditMetrics:
    """Quantitative governance and compliance audit metrics for simulation runs."""

    evidence_coverage: Optional[float] = None  # Proportion of claims with valid tool evidence refs [0, 1]
    unsupported_numeric_claim_rate: Optional[float] = None  # Proportion of numeric claims lacking tool backing (target 0.0)
    anti_fabrication_rejection_count: int = 0  # Count of fabricated claims rejected at validation gate
    tool_call_success_rate: Optional[float] = None  # Proportion of successful tool executions [0, 1]
    follow_up_yield: Optional[float] = None  # Proportion of follow-ups producing measurable delta [0, 1]
    conflict_resolution_rate: Optional[float] = None  # Proportion of cross-domain conflicts addressed [0, 1]
    degraded_rate: Optional[float] = None  # Proportion of tasks executed in degraded state [0, 1]
    total_tokens: int = 0  # Total LLM tokens consumed
    total_latency_seconds: float = 0.0  # End-to-end elapsed runtime in seconds
    conflicts_detected: int = 0  # Count of cross-domain conflicts detected
    conflicts_resolved: int = 0  # Count of cross-domain conflicts resolved
    anomalies_detected: int = 0  # Count of domain anomalies detected
    unlocalized_conflicts: int = 0  # Count of cross-domain conflicts lacking zone localization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_coverage": round(float(self.evidence_coverage), 4) if self.evidence_coverage is not None else None,
            "unsupported_numeric_claim_rate": round(float(self.unsupported_numeric_claim_rate), 4) if self.unsupported_numeric_claim_rate is not None else None,
            "anti_fabrication_rejection_count": int(self.anti_fabrication_rejection_count),
            "tool_call_success_rate": round(float(self.tool_call_success_rate), 4) if self.tool_call_success_rate is not None else None,
            "follow_up_yield": round(float(self.follow_up_yield), 4) if self.follow_up_yield is not None else None,
            "conflict_resolution_rate": round(float(self.conflict_resolution_rate), 4) if self.conflict_resolution_rate is not None else None,
            "degraded_rate": round(float(self.degraded_rate), 4) if self.degraded_rate is not None else None,
            "total_tokens": int(self.total_tokens),
            "total_latency_seconds": round(float(self.total_latency_seconds), 4),
            "conflicts_detected": int(self.conflicts_detected),
            "conflicts_resolved": int(self.conflicts_resolved),
            "anomalies_detected": int(self.anomalies_detected),
            "unlocalized_conflicts": int(self.unlocalized_conflicts),
        }


def compute_audit_metrics(
    manifest: Optional[Any] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    tokens_used: int = 0,
    latency_seconds: float = 0.0,
    follow_up_count: int = 0,
    follow_up_delta_count: int = 0,
    anti_fabrication_rejections: Optional[int] = None,
    conflicts_detected: Optional[int] = None,
    conflicts_resolved: Optional[int] = None,
    anomalies_detected: Optional[int] = None,
    unlocalized_conflicts: Optional[int] = None,
    feature_value_count: Optional[int] = None,
    feature_with_refs_count: Optional[int] = None,
    validation_attempts: Optional[int] = None,
    validation_rejections: Optional[int] = None,
) -> AuditMetrics:
    """Computes standard AuditMetrics from run artifacts and execution records."""
    tasks = getattr(manifest, "tasks", []) if manifest else []
    total_tasks = len(tasks)
    degraded_tasks = sum(
        1 for t in tasks
        if (
            (isinstance(t, dict) and str(t.get("status", "")).upper() in ("DEGRADED", "TASKSTATUS.DEGRADED"))
            or getattr(t, "status", "") in ("DEGRADED", "TaskStatus.DEGRADED")
            or str(getattr(t, "status", "")).upper() in ("DEGRADED", "TASKSTATUS.DEGRADED")
        )
    )
    degraded_rate: Optional[float] = float(degraded_tasks / total_tasks) if total_tasks > 0 else None

    # Tool call success rate: None if no calls executed
    calls = tool_calls
    if calls is None and manifest:
        calls = getattr(manifest, "tool_calls", None)
        if not calls and hasattr(manifest, "metadata"):
            calls = manifest.metadata.get("tool_calls")

    if calls and len(calls) > 0:
        successful_calls = sum(1 for c in calls if c.get("status") in ("success", "SUCCESS"))
        tool_success_rate: Optional[float] = float(successful_calls / len(calls))
    else:
        tool_success_rate = None

    # Follow-up yield: None if follow_up_count is 0
    fu_count = follow_up_count
    if fu_count == 0 and manifest and hasattr(manifest, "metadata"):
        fu_count = len(manifest.metadata.get("follow_up_tasks", []))

    if fu_count > 0:
        fu_yield: Optional[float] = float(np.clip(follow_up_delta_count / fu_count, 0.0, 1.0))
    else:
        fu_yield = None

    # Conflict resolution rate: None if conflicts_detected is 0
    conf_det = conflicts_detected
    if conf_det is None and manifest and hasattr(manifest, "metadata"):
        conf_det = manifest.metadata.get("conflicts_detected")
        if conf_det is None:
            conf_det = len(manifest.metadata.get("conflicts", []))
    conf_det_int = int(conf_det or 0)

    conf_res = conflicts_resolved
    if conf_res is None and manifest and hasattr(manifest, "metadata"):
        conf_res = manifest.metadata.get("conflicts_resolved")
        if conf_res is None:
            conf_res = len(manifest.metadata.get("resolved_conflicts", []))
    conf_res_int = int(conf_res or 0)

    if conf_det_int > 0:
        conflict_res_rate: Optional[float] = float(np.clip(conf_res_int / conf_det_int, 0.0, 1.0))
    else:
        conflict_res_rate = None

    anom_det = anomalies_detected
    if anom_det is None and manifest and hasattr(manifest, "metadata"):
        anom_det = manifest.metadata.get("anomalies_detected")
    anom_det_int = int(anom_det or 0)

    unloc_conf = unlocalized_conflicts
    if unloc_conf is None and manifest and hasattr(manifest, "metadata"):
        unloc_conf = manifest.metadata.get("unlocalized_conflicts")
    unloc_conf_int = int(unloc_conf or 0)

    # Evidence coverage: feature_with_refs / feature_value_count; None if feature_value_count=0
    f_val_count = feature_value_count
    if f_val_count is None and manifest and hasattr(manifest, "metadata"):
        f_val_count = manifest.metadata.get("feature_value_count")

    f_refs_count = feature_with_refs_count
    if f_refs_count is None and manifest and hasattr(manifest, "metadata"):
        f_refs_count = manifest.metadata.get("feature_with_refs_count")

    if f_val_count is not None and f_val_count > 0:
        coverage: Optional[float] = float(np.clip((f_refs_count or 0) / f_val_count, 0.0, 1.0))
    else:
        explicit_cov = manifest.metadata.get("evidence_coverage") if manifest and hasattr(manifest, "metadata") else None
        coverage = float(explicit_cov) if explicit_cov is not None else None

    # Unsupported numeric claim rate and anti fabrication rejections
    v_attempts = validation_attempts
    if v_attempts is None and manifest and hasattr(manifest, "metadata"):
        v_attempts = manifest.metadata.get("validation_attempts")

    v_rejections = validation_rejections
    if v_rejections is None and manifest and hasattr(manifest, "metadata"):
        v_rejections = manifest.metadata.get("validation_rejections")
        if v_rejections is None:
            v_rejections = manifest.metadata.get("anti_fabrication_rejection_count")
    if v_rejections is None:
        v_rejections = anti_fabrication_rejections
    v_rejections_int = int(v_rejections or 0)

    if v_attempts is not None and v_attempts > 0:
        unsupported_rate: Optional[float] = float(np.clip(v_rejections_int / v_attempts, 0.0, 1.0))
    else:
        unsupported_rate = None

    return AuditMetrics(
        evidence_coverage=coverage,
        unsupported_numeric_claim_rate=unsupported_rate,
        anti_fabrication_rejection_count=v_rejections_int,
        tool_call_success_rate=tool_success_rate,
        follow_up_yield=fu_yield,
        conflict_resolution_rate=conflict_res_rate,
        degraded_rate=degraded_rate,
        total_tokens=tokens_used,
        total_latency_seconds=latency_seconds,
        conflicts_detected=conf_det_int,
        conflicts_resolved=conf_res_int,
        anomalies_detected=anom_det_int,
        unlocalized_conflicts=unloc_conf_int,
    )
