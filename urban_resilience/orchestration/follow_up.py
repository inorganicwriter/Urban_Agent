"""Follow-up controller translating cross-agent conflicts into targeted follow-up tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from urban_resilience.config import AgentWeightConfig
from urban_resilience.orchestration.contracts import (
    FOLLOW_UP_SUCCESS_STATUSES,
    Task,
    TaskResult,
    TaskStatus,
    is_follow_up_success_status,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FOLLOW_UP_SUCCESS_STATUSES",
    "FollowUpController",
    "is_follow_up_success_status",
]


class FollowUpController:
    """Detects evidence reconciliation conflicts, schedules targeted inquiry tasks, and applies refinements."""

    def __init__(self, weight_config: Optional[AgentWeightConfig] = None):
        self.weights = weight_config or AgentWeightConfig()
        self.follow_up_logs: List[Dict[str, Any]] = []

    def identify_follow_ups(
        self,
        zone_features: pd.DataFrame,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Scans integrated zone features for domain conflicts and vulnerability edge cases."""
        notes: List[str] = []
        follow_ups: List[Dict[str, Any]] = []

        for zid, row in zone_features.iterrows():
            # Conflict: High spatial redundancy but low intraday flexibility
            if row["spatial_redundancy"] > 0.65 and row["intraday_flexibility"] < 0.35:
                note = f"Zone {zid}: Conflict between high spatial routes and rigid commuting windows."
                notes.append(note)
                follow_ups.append({
                    "zone_id": zid,
                    "target_agent": "flexibility",
                    "reason": "Examine if off-peak flexibility can be mobilized through telecommuting or flexible shifts.",
                })

            # Waterlogging hazard with low elevation
            if row["elevation_m"] < 6.0 and row["hazard_intensity"] > 0.6:
                follow_ups.append({
                    "zone_id": zid,
                    "target_agent": "spatial",
                    "reason": "Re-verify alternative evacuation corridors under waterlogging scenario.",
                })

        return notes, follow_ups

    def create_follow_up_tasks(
        self,
        follow_ups_needed: List[Dict[str, Any]],
        dependencies: Optional[List[str]] = None,
    ) -> List[Task]:
        """Constructs explicit Task objects for each follow-up request."""
        tasks: List[Task] = []
        deps = list(dependencies or [])

        for req in follow_ups_needed:
            agent = req["target_agent"]
            zid = req["zone_id"]
            reason = req["reason"]
            tid = f"follow_up_{agent}_{zid}"

            tasks.append(
                Task(
                    task_id=tid,
                    agent_name=agent,
                    action="targeted_inquiry",
                    payload={
                        "zone_id": zid,
                        "reason": reason,
                        "thought": f"Targeted follow-up: {reason} for zone {zid}",
                        "action_type": "query_data",
                        "action_input": {"query_type": "zone_info", "zone_id": zid},
                    },
                    dependencies=deps,
                    metadata={"target_agent": agent, "zone_id": zid, "reason": reason},
                )
            )

        return tasks

    def apply_follow_up_results(
        self,
        follow_ups_needed: List[Dict[str, Any]],
        task_results: Dict[str, TaskResult],
        zone_features: pd.DataFrame,
        weights: Optional[AgentWeightConfig] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Applies mathematical refinements to zone features based on successful follow-up task executions."""
        w = (weights or self.weights).as_dict()
        updated_features = zone_features.copy()
        new_logs: List[Dict[str, Any]] = []

        for req in follow_ups_needed:
            zid = req["zone_id"]
            agent_key = req["target_agent"]
            tid = f"follow_up_{agent_key}_{zid}"
            result = task_results.get(tid)

            # Apply refinement if task succeeded or returned valid output
            if result and result.status in (TaskStatus.SUCCESS, TaskStatus.DEGRADED) and zid in updated_features.index:
                if agent_key == "flexibility":
                    old_flex = updated_features.loc[zid, "intraday_flexibility"]
                    updated_features.loc[zid, "intraday_flexibility"] = float(np.clip(old_flex + 0.05, 0.0, 1.0))
                elif agent_key == "spatial":
                    old_spat = updated_features.loc[zid, "spatial_redundancy"]
                    h_int = updated_features.loc[zid, "hazard_intensity"]
                    discount = max(0.5, 1.0 - 0.15 * h_int)
                    updated_features.loc[zid, "spatial_redundancy"] = float(np.clip(old_spat * discount, 0.0, 1.0))

                # Recompute composite resilience and vulnerability
                r_comp = float(
                    w["temporal_redundancy"] * updated_features.loc[zid, "temporal_redundancy"]
                    + w["intraday_flexibility"] * updated_features.loc[zid, "intraday_flexibility"]
                    + w["predictability"] * updated_features.loc[zid, "predictability"]
                    + w["spatial_redundancy"] * updated_features.loc[zid, "spatial_redundancy"]
                    + w["functional_mix"] * updated_features.loc[zid, "functional_mix"]
                )
                r_comp = float(np.clip(r_comp, 0.05, 0.95))
                updated_features.loc[zid, "composite_resilience"] = r_comp
                h_int = updated_features.loc[zid, "hazard_intensity"]
                updated_features.loc[zid, "vulnerability_score"] = float(np.clip(h_int * (1.15 - r_comp), 0.05, 1.0))

                log_entry = {
                    "zone_id": zid,
                    "agent": agent_key,
                    "prompt": req["reason"],
                    "status": "refined",
                    "task_id": tid,
                }
                new_logs.append(log_entry)
                self.follow_up_logs.append(log_entry)

        return updated_features, new_logs
