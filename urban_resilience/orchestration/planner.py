"""Task planning interfaces and deterministic DAG plan generators."""

from __future__ import annotations

import abc
import uuid
from typing import Any, Dict, List, Optional

from urban_resilience.orchestration.contracts import ExecutionPlan, Task
from urban_resilience.types import EventContext


class LLMPlannerProtocol(abc.ABC):
    """Abstract protocol for task planners (supports deterministic and future LLM adapters)."""

    @abc.abstractmethod
    def plan_domain_tasks(self, plan_id: Optional[str] = None) -> ExecutionPlan:
        """Plans domain analysis tasks."""
        pass

    @abc.abstractmethod
    def plan_full_pipeline(
        self, event_context: Optional[EventContext] = None, plan_id: Optional[str] = None
    ) -> ExecutionPlan:
        """Plans complete end-to-end multi-agent resilience workflow."""
        pass

    @abc.abstractmethod
    def plan_follow_up_tasks(
        self, follow_ups_needed: List[Dict[str, Any]], dependencies: Optional[List[str]] = None
    ) -> List[Task]:
        """Generates targeted follow-up tasks to resolve conflicting domain evidence."""
        pass


class DeterministicTaskPlanner(LLMPlannerProtocol):
    """Deterministic, rule-based planner generating DAG execution plans."""

    DOMAIN_AGENTS = ["temporal", "flexibility", "predictability", "spatial", "functional"]

    def plan_domain_tasks(self, plan_id: Optional[str] = None) -> ExecutionPlan:
        """Constructs parallel domain analysis tasks for all 5 domains."""
        pid = plan_id or f"plan_domains_{uuid.uuid4().hex[:8]}"
        tasks = []

        for agent_name in self.DOMAIN_AGENTS:
            tasks.append(
                Task(
                    task_id=f"task_{agent_name}",
                    agent_name=agent_name,
                    action="domain_analysis",
                    dependencies=[],
                    metadata={"description": f"Domain resilience analysis by {agent_name} subagent"},
                )
            )

        plan = ExecutionPlan(plan_id=pid, tasks=tasks)
        plan.validate_dag()
        return plan

    def plan_full_pipeline(
        self, event_context: Optional[EventContext] = None, plan_id: Optional[str] = None
    ) -> ExecutionPlan:
        """Constructs full pipeline DAG: 5 domain tasks -> synthesis task."""
        pid = plan_id or f"plan_full_{uuid.uuid4().hex[:8]}"
        domain_plan = self.plan_domain_tasks(plan_id=pid)
        domain_task_ids = [t.task_id for t in domain_plan.tasks]

        synthesis_task = Task(
            task_id="task_synthesis",
            agent_name="synthesis",
            action="synthesize",
            dependencies=domain_task_ids,
            payload={"event_context": event_context} if event_context else {},
            metadata={"description": "Cross-agent evidence reconciliation and event-conditioned prediction"},
        )

        domain_plan.add_task(synthesis_task)
        domain_plan.validate_dag()
        return domain_plan

    def plan_follow_up_tasks(
        self, follow_ups_needed: List[Dict[str, Any]], dependencies: Optional[List[str]] = None
    ) -> List[Task]:
        """Constructs targeted follow-up tasks for specific zones and subagents."""
        tasks: List[Task] = []
        deps = list(dependencies or [])

        for item in follow_ups_needed:
            agent = item["target_agent"]
            zid = item["zone_id"]
            reason = item.get("reason", "")
            tid = f"follow_up_{agent}_{zid}"
            tasks.append(
                Task(
                    task_id=tid,
                    agent_name=agent,
                    action="targeted_inquiry",
                    payload={"zone_id": zid, "reason": reason, "query_type": "zone_info"},
                    dependencies=deps,
                    metadata={"target_agent": agent, "zone_id": zid, "reason": reason},
                )
            )

        return tasks
