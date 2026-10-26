"""Evidence bus for collecting, indexing, validating completeness, and detecting conflicting domain signals."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from urban_resilience.types import AgentEvidence, UrbanContext

logger = logging.getLogger(__name__)


@dataclass
class ConflictRecord:
    """Structured record of a detected domain signal conflict, anomaly, or invalid value."""

    kind: str  # "cross_domain" | "anomaly" | "invalid"
    zone_id: Optional[str]
    domains: List[str]
    message: str
    targetable: bool
    resolved: bool = False
    conflict_id: Optional[str] = None

    def __post_init__(self):
        if not self.conflict_id:
            z = self.zone_id or "global"
            d = "-".join(sorted(self.domains))
            self.conflict_id = f"{self.kind}:{z}:{d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "zone_id": self.zone_id,
            "domains": list(self.domains),
            "message": self.message,
            "targetable": self.targetable,
            "resolved": self.resolved,
            "conflict_id": self.conflict_id,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


class ConflictTargetSelector:
    """Selects targeted follow-up domain and zone from detected conflicts."""

    @staticmethod
    def select_target(
        structured_conflicts: List[Union[ConflictRecord, Dict[str, Any]]],
    ) -> Optional[Tuple[str, str, str]]:
        """Selects (zone_id, domain_to_query, reason) for cross-domain conflicts.

        Returns None if no localized, targetable cross-domain conflict is available.
        Prioritizes kind="cross_domain" with non-empty zone_id and targetable=True.
        """
        for record in structured_conflicts:
            kind = record.get("kind") if isinstance(record, dict) else getattr(record, "kind", None)
            zone_id = record.get("zone_id") if isinstance(record, dict) else getattr(record, "zone_id", None)
            targetable = record.get("targetable") if isinstance(record, dict) else getattr(record, "targetable", False)
            domains = record.get("domains", []) if isinstance(record, dict) else getattr(record, "domains", [])
            resolved = record.get("resolved", False) if isinstance(record, dict) else getattr(record, "resolved", False)

            if kind == "cross_domain" and zone_id and targetable and not resolved:
                domains_set = set(domains)
                # Domain selection priority: spatial vs flexibility -> query flexibility
                if "spatial" in domains_set and "flexibility" in domains_set:
                    domain_to_query = "flexibility"
                elif "flexibility" in domains_set:
                    domain_to_query = "flexibility"
                elif "temporal" in domains_set:
                    domain_to_query = "temporal"
                elif domains:
                    domain_to_query = domains[0]
                else:
                    domain_to_query = "temporal"

                reason = (
                    f"Targeted conflict resolution for zone '{zone_id}' across domains {domains}: "
                    f"querying '{domain_to_query}' to reconcile discrepancy."
                )
                return zone_id, domain_to_query, reason

        return None


class EvidenceBus:
    """Central repository and validator for domain evidence produced during orchestration."""

    DEFAULT_REQUIRED_DOMAINS = ["temporal", "flexibility", "predictability", "spatial", "functional"]

    def __init__(self):
        self._evidences_by_agent: Dict[str, AgentEvidence] = {}
        self._evidences_by_task: Dict[str, AgentEvidence] = {}
        self._conflict_records: List[str] = []
        self._structured_conflicts: List[ConflictRecord] = []

    def register_evidence(
        self,
        agent_name: str,
        evidence: AgentEvidence,
        task_id: Optional[str] = None,
    ) -> None:
        """Stores evidence indexed by domain agent name and optionally task ID."""
        if agent_name in self._evidences_by_agent:
            logger.info("Overwriting existing evidence for agent '%s' on EvidenceBus", agent_name)

        self._evidences_by_agent[agent_name] = evidence
        if task_id:
            self._evidences_by_task[task_id] = evidence
        logger.debug("Evidence registered for '%s' (confidence: %.2f)", agent_name, evidence.confidence)

    def merge_zone_evidence(
        self,
        agent_name: str,
        zone_evidence: AgentEvidence,
        task_id: Optional[str] = None,
    ) -> None:
        """Merges fine-grained or single-zone follow-up evidence into existing domain evidence without overwriting other zones."""
        if agent_name not in self._evidences_by_agent:
            self.register_evidence(agent_name, zone_evidence, task_id=task_id)
            return

        existing = self._evidences_by_agent[agent_name]

        # 1. Merge zone-level metrics
        if hasattr(zone_evidence, "zone_metrics") and zone_evidence.zone_metrics:
            for zid, zm in zone_evidence.zone_metrics.items():
                if zid not in existing.zone_metrics:
                    existing.zone_metrics[zid] = dict(zm)
                else:
                    existing.zone_metrics[zid].update(zm)

        # 2. Append execution traces
        if hasattr(zone_evidence, "traces") and zone_evidence.traces:
            existing.traces.extend(zone_evidence.traces)

        # 3. Append anomalies deduplicated
        if hasattr(zone_evidence, "potential_anomalies") and zone_evidence.potential_anomalies:
            for anom in zone_evidence.potential_anomalies:
                if anom not in existing.potential_anomalies:
                    existing.potential_anomalies.append(anom)

        # 4. Recompute global metrics across all zones
        all_metric_keys = set()
        for zm in existing.zone_metrics.values():
            all_metric_keys.update(zm.keys())
        for k in all_metric_keys:
            vals = [
                zm[k]
                for zm in existing.zone_metrics.values()
                if k in zm and isinstance(zm[k], (int, float)) and not isinstance(zm[k], bool)
            ]
            if vals:
                existing.metrics[k] = float(np.mean(vals))

        if task_id:
            self._evidences_by_task[task_id] = zone_evidence

        logger.info(
            "EvidenceBus: Merged follow-up evidence for agent '%s' across %d total zones",
            agent_name,
            len(existing.zone_metrics),
        )

    def get_evidence(self, agent_name: str) -> Optional[AgentEvidence]:
        """Retrieves evidence by agent name."""
        return self._evidences_by_agent.get(agent_name)

    def get_all_evidences(self) -> Dict[str, AgentEvidence]:
        """Returns a copy of all registered evidences indexed by agent name."""
        return dict(self._evidences_by_agent)

    def check_completeness(
        self,
        required_domains: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Verifies that evidence is present for all required domain perspectives."""
        req = required_domains or self.DEFAULT_REQUIRED_DOMAINS
        missing = [domain for domain in req if domain not in self._evidences_by_agent]
        is_complete = len(missing) == 0
        if not is_complete:
            logger.warning("EvidenceBus incomplete: missing domains %s", missing)
        return is_complete, missing

    def detect_conflicts(self, urban_context: Optional[UrbanContext] = None) -> List[str]:
        """Detects conflicting domain signals, metric anomalies, or invalid values across registered evidence."""
        conflicts: List[str] = []
        structured: List[ConflictRecord] = []

        # 1. Cross-domain conflict: High spatial redundancy vs. Low intraday flexibility
        spat_ev = self._evidences_by_agent.get("spatial")
        flex_ev = self._evidences_by_agent.get("flexibility")

        if spat_ev and flex_ev:
            for zid in spat_ev.zone_metrics:
                if zid in flex_ev.zone_metrics:
                    spat_score = spat_ev.zone_metrics[zid].get("spatial_redundancy_score", 0.5)
                    flex_score = flex_ev.zone_metrics[zid].get("intraday_flexibility_score", 0.5)
                    if spat_score > 0.65 and flex_score < 0.35:
                        msg = (
                            f"Zone {zid}: Conflict detected between high spatial redundancy "
                            f"({spat_score:.2f}) and rigid commuting windows ({flex_score:.2f})"
                        )
                        conflicts.append(msg)
                        structured.append(
                            ConflictRecord(
                                kind="cross_domain",
                                zone_id=zid,
                                domains=["spatial", "flexibility"],
                                message=msg,
                                targetable=True,
                                resolved=False,
                            )
                        )

        # 2. Extreme anomalies or out-of-bounds metrics (invalid / anomaly)
        for name, ev in self._evidences_by_agent.items():
            if not (0.0 <= ev.confidence <= 1.0):
                msg = f"Agent '{name}' has out-of-range confidence: {ev.confidence}"
                conflicts.append(msg)
                structured.append(
                    ConflictRecord(
                        kind="invalid",
                        zone_id=None,
                        domains=[name],
                        message=msg,
                        targetable=False,
                        resolved=False,
                    )
                )

            for zid, metrics in ev.zone_metrics.items():
                for m_name, val in metrics.items():
                    if np.isnan(val) or np.isinf(val):
                        msg = f"Agent '{name}', Zone {zid}: Metric '{m_name}' is NaN/inf"
                        conflicts.append(msg)
                        structured.append(
                            ConflictRecord(
                                kind="invalid",
                                zone_id=zid,
                                domains=[name],
                                message=msg,
                                targetable=False,
                                resolved=False,
                            )
                        )

            for anomaly in ev.potential_anomalies:
                msg = f"Agent '{name}' reported anomaly: {anomaly}"
                conflicts.append(msg)

                # Attempt to parse localized zone_id from anomaly message
                parsed_zone: Optional[str] = None
                m_zone = re.search(r"\bZone\s+([A-Za-z0-9_]+)\b", anomaly)
                if m_zone:
                    parsed_zone = m_zone.group(1).rstrip(":,.;")
                elif urban_context:
                    for zid in urban_context.zone_ids:
                        if zid in anomaly:
                            parsed_zone = zid
                            break

                targetable = bool(parsed_zone is not None)
                structured.append(
                    ConflictRecord(
                        kind="anomaly",
                        zone_id=parsed_zone,
                        domains=[name],
                        message=msg,
                        targetable=targetable,
                        resolved=False,
                    )
                )

        self._conflict_records = conflicts
        self._structured_conflicts = structured
        if conflicts:
            logger.info("EvidenceBus detected %d cross-agent conflicts/anomalies", len(conflicts))
        return conflicts

    @property
    def conflicts(self) -> List[str]:
        """Returns the list of detected conflicts and anomalies strings for backward compatibility."""
        return list(self._conflict_records)

    @property
    def structured_conflicts(self) -> List[ConflictRecord]:
        """Returns the structured list of all detected records (cross_domain, anomaly, invalid)."""
        return list(self._structured_conflicts)

    @property
    def cross_domain_conflicts(self) -> List[ConflictRecord]:
        """Returns localized cross-domain conflicts (kind == 'cross_domain' and zone_id is not None)."""
        return [c for c in self._structured_conflicts if c.kind == "cross_domain" and c.zone_id is not None]

    @property
    def anomalies(self) -> List[ConflictRecord]:
        """Returns domain anomalies (kind == 'anomaly')."""
        return [c for c in self._structured_conflicts if c.kind == "anomaly"]

    @property
    def unlocalized_conflicts(self) -> List[ConflictRecord]:
        """Returns cross-domain conflicts lacking zone localization (kind == 'cross_domain' and zone_id is None)."""
        return [c for c in self._structured_conflicts if c.kind == "cross_domain" and c.zone_id is None]

    @property
    def conflicts_detected_count(self) -> int:
        """Returns count of localized cross-domain conflicts (anomalies excluded)."""
        return len(self.cross_domain_conflicts)

    @property
    def anomalies_detected_count(self) -> int:
        """Returns count of domain anomalies."""
        return len(self.anomalies)

    @property
    def unlocalized_conflicts_count(self) -> int:
        """Returns count of cross-domain conflicts lacking zone localization."""
        return len(self.unlocalized_conflicts)

    def get_canonical_metrics_snapshot(self) -> Dict[str, Any]:
        """Returns a canonical deep-copied snapshot of domain metrics and confidence for delta detection.

        Only compares numerical zone_metrics, global metrics, and confidence.
        Execution traces, qualitative inferences, and timestamp logs are excluded.
        """
        snapshot: Dict[str, Any] = {}
        for agent_name in sorted(self._evidences_by_agent.keys()):
            ev = self._evidences_by_agent[agent_name]
            zm_copy: Dict[str, Dict[str, float]] = {}
            if hasattr(ev, "zone_metrics") and ev.zone_metrics:
                for zid in sorted(ev.zone_metrics.keys()):
                    zm = ev.zone_metrics[zid]
                    zm_copy[zid] = {
                        k: round(float(v), 6)
                        for k, v in sorted(zm.items())
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
            m_copy: Dict[str, float] = {}
            if hasattr(ev, "metrics") and ev.metrics:
                for k, v in sorted(ev.metrics.items()):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        m_copy[k] = round(float(v), 6)
            conf = (
                round(float(ev.confidence), 6)
                if hasattr(ev, "confidence") and isinstance(ev.confidence, (int, float))
                else 0.0
            )
            snapshot[agent_name] = {
                "zone_metrics": zm_copy,
                "metrics": m_copy,
                "confidence": conf,
            }
        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """Produces a structured evidence summary suitable for inclusion in RunManifest."""
        is_complete, missing = self.check_completeness()
        return {
            "domains_registered": list(self._evidences_by_agent.keys()),
            "is_complete": is_complete,
            "missing_domains": missing,
            "conflicts_detected": [c.message for c in self.cross_domain_conflicts],
            "cross_domain_conflicts": [c.to_dict() for c in self.cross_domain_conflicts],
            "anomalies": [c.to_dict() for c in self.anomalies],
            "unlocalized_conflicts": [c.to_dict() for c in self.unlocalized_conflicts],
            "all_conflicts": list(self._conflict_records),
            "agent_confidences": {
                name: ev.confidence for name, ev in self._evidences_by_agent.items()
            },
            "reported_anomalies": {
                name: list(ev.potential_anomalies)
                for name, ev in self._evidences_by_agent.items()
                if ev.potential_anomalies
            },
        }
