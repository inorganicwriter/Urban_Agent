"""Orchestration package providing explicit Supervisor -> Domain Subagent execution."""

from __future__ import annotations

from urban_resilience.orchestration.contracts import (
    ExecutionPlan,
    IncompleteEvidenceError,
    OrchestrationError,
    RunManifest,
    SUPPORTED_LLM_DOMAINS,
    Task,
    TaskResult,
    TaskStatus,
)
from urban_resilience.orchestration.evidence_bus import (
    ConflictRecord,
    ConflictTargetSelector,
    EvidenceBus,
)
from urban_resilience.orchestration.follow_up import FollowUpController
from urban_resilience.orchestration.planner import DeterministicTaskPlanner, LLMPlannerProtocol
from urban_resilience.orchestration.scheduler import TaskScheduler
from urban_resilience.orchestration.subagent import (
    BaseSubagent,
    DomainSubagent,
    FlexibilitySubagent,
    FunctionalSubagent,
    PredictabilitySubagent,
    SpatialSubagent,
    SubagentRegistry,
    SynthesisSubagent,
    TemporalSubagent,
)
from urban_resilience.orchestration.supervisor import SupervisorAgent

__all__ = [
    "IncompleteEvidenceError",
    "OrchestrationError",
    "TaskStatus",
    "Task",
    "TaskResult",
    "ExecutionPlan",
    "RunManifest",
    "SUPPORTED_LLM_DOMAINS",
    "BaseSubagent",
    "DomainSubagent",
    "TemporalSubagent",
    "FlexibilitySubagent",
    "PredictabilitySubagent",
    "SpatialSubagent",
    "FunctionalSubagent",
    "SynthesisSubagent",
    "SubagentRegistry",
    "LLMPlannerProtocol",
    "DeterministicTaskPlanner",
    "TaskScheduler",
    "EvidenceBus",
    "ConflictRecord",
    "ConflictTargetSelector",
    "FollowUpController",
    "SupervisorAgent",
]
