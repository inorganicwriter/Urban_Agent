"""Orchestration contracts, data structures, and lifecycle states."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class OrchestrationError(Exception):
    """Base exception for orchestration and supervisor failures."""

    pass


SUPPORTED_LLM_DOMAINS: set[str] = {
    "temporal",
    "flexibility",
    "predictability",
    "spatial",
    "functional",
}


class IncompleteEvidenceError(OrchestrationError):
    """Raised when required domain perspectives are missing or failed during orchestration."""

    def __init__(
        self,
        message_or_missing: Optional[str | List[str]] = None,
        missing_domains: Optional[List[str]] = None,
        task_errors: Optional[Dict[str, str]] = None,
        failed_tasks: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ):
        actual_missing: List[str] = []
        actual_message: Optional[str] = None
        if isinstance(message_or_missing, list):
            actual_missing = list(message_or_missing)
        elif isinstance(message_or_missing, str):
            actual_message = message_or_missing

        if missing_domains is not None:
            actual_missing = list(missing_domains)
        if message is not None:
            actual_message = message

        actual_errors = dict(task_errors or failed_tasks or {})
        self.missing_domains = actual_missing
        self.task_errors = actual_errors
        self.failed_tasks = actual_errors

        if not actual_message:
            err_details = f"; task errors: {actual_errors}" if actual_errors else ""
            actual_message = f"Incomplete domain evidence: missing required perspective(s) {actual_missing}{err_details}"

        super().__init__(actual_message)


class TaskStatus(str, Enum):
    """Execution status for tasks and orchestrated plans."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"


FOLLOW_UP_SUCCESS_STATUSES: set[str] = {
    TaskStatus.SUCCESS.value,
    TaskStatus.DEGRADED.value,
    "SUCCESS",
    "DEGRADED",
    "refined",
}


def is_follow_up_success_status(status: Any) -> bool:
    """Returns True if the follow-up task execution status represents a successful outcome.

    Recognizes standard TaskStatus SUCCESS/DEGRADED as well as deterministic follow-up 'refined'.
    """
    if status is None:
        return False
    if isinstance(status, TaskStatus):
        return status in (TaskStatus.SUCCESS, TaskStatus.DEGRADED)
    return str(status) in FOLLOW_UP_SUCCESS_STATUSES


class Task(BaseModel):
    """Specification of an executable subagent work unit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    parent_task_id: Optional[str] = None
    agent_name: str
    action: str
    objective: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[float] = None
    max_retries: int = 1
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Execution result and telemetry for a single task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    agent_name: str = ""
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    retry_count: int = 0
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """Directed acyclic graph of tasks to be executed by the scheduler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_id: str
    tasks: List[Task] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        """Appends a task to the plan."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Looks up a task by its ID."""
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def validate_dag(self) -> None:
        """Validates that the task graph is a valid, acyclic DAG with known dependencies.

        Raises:
            ValueError: If duplicate task IDs, unknown dependencies, or cycles are found.
        """
        task_ids = [t.task_id for t in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            seen = set()
            duplicates = {tid for tid in task_ids if tid in seen or seen.add(tid)}
            raise ValueError(f"Duplicate task IDs found in ExecutionPlan: {duplicates}")

        id_set = set(task_ids)
        for t in self.tasks:
            for dep in t.dependencies:
                if dep not in id_set:
                    raise ValueError(f"Task '{t.task_id}' depends on non-existent task '{dep}'")

        # Cycle detection using DFS coloring: 0=unvisited, 1=visiting (on stack), 2=visited
        dep_graph: Dict[str, List[str]] = {t.task_id: list(t.dependencies) for t in self.tasks}
        color: Dict[str, int] = {tid: 0 for tid in task_ids}

        def dfs(node: str, path: List[str]):
            color[node] = 1
            for dep in dep_graph.get(node, []):
                if color.get(dep, 0) == 1:
                    cycle = " -> ".join(path + [dep])
                    raise ValueError(f"Cycle detected in execution graph: {cycle}")
                if color.get(dep, 0) == 0:
                    dfs(dep, path + [dep])
            color[node] = 2

        for tid in task_ids:
            if color[tid] == 0:
                dfs(tid, [tid])


def _serialize_safe(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_serialize_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize_safe(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _serialize_safe(obj.to_dict())
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _serialize_safe(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return {k: _serialize_safe(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


class RunManifest(BaseModel):
    """Comprehensive execution manifest capturing plan, results, telemetry, and evidence summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    plan_id: str
    status: TaskStatus
    total_duration_seconds: float = 0.0
    failure_count: int = 0
    degraded_count: int = 0
    error_summary: Optional[str] = None
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    task_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    follow_up_summary: List[Dict[str, Any]] = Field(default_factory=list)
    orchestration_mode: str = "deterministic"
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)
    llm_latency_seconds: float = 0.0
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    follow_up_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    predictor_status: Dict[str, Any] = Field(default_factory=dict)
    feature_reports: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts manifest to a clean serializable dictionary."""
        return _serialize_safe(self.model_dump())

    def save_to_file(self, path: Path | str) -> Path:
        """Persists the manifest as a formatted JSON document."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return p


def replay_manifest_execution(manifest: RunManifest | Dict[str, Any]) -> Dict[str, Any]:
    """Replays and verifies the task graph, statuses, and tool calls recorded in a RunManifest."""
    data = manifest.to_dict() if isinstance(manifest, RunManifest) else manifest
    tasks = data.get("tasks", [])
    task_results = data.get("task_results", {})
    tool_calls = data.get("tool_calls", [])

    executed_tids = [t["task_id"] for t in tasks if t.get("status") in ("SUCCESS", "DEGRADED", "FAILED")]
    missing_results = [tid for tid in executed_tids if tid not in task_results]

    return {
        "run_id": data.get("run_id"),
        "total_tasks": len(tasks),
        "executed_tasks_count": len(executed_tids),
        "missing_results": missing_results,
        "tool_calls_count": len(tool_calls),
        "follow_up_decisions_count": len(data.get("follow_up_decisions", [])),
        "is_valid": len(missing_results) == 0,
    }
