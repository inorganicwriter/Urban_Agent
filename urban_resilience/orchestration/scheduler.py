"""Task scheduler supporting topological sorting, parallel execution, timeouts, retries, and failure degradation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
import time
import traceback
from typing import Dict, List, Optional, Set

from urban_resilience.orchestration.contracts import ExecutionPlan, Task, TaskResult, TaskStatus
from urban_resilience.orchestration.subagent import SubagentRegistry

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Schedules and executes tasks based on DAG topological dependencies.

    Note:
        Python worker threads cannot be forcibly terminated asynchronously.
        Tasks that time out will have their future cancelled and the executor shut down
        with wait=False, cancel_futures=True, returning TaskStatus.FAILED immediately
        without retrying by default (unless task.metadata['retry_on_timeout'] is True)
        to prevent concurrent thread race conditions. If hard interruption is strictly
        required, process isolation (ProcessPoolExecutor) should be used.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_retries: Optional[int] = 1,
        default_timeout_seconds: Optional[float] = None,
    ):
        """Initializes the TaskScheduler.

        Args:
            max_workers: Maximum number of worker threads for parallel stages.
            max_retries: Scheduler-level upper cap on task retries. If None, task-level
                ``task.max_retries`` is respected without capping. If an integer, retries
                are capped at ``min(task.max_retries, scheduler.max_retries)``. Defaults to 1.
            default_timeout_seconds: Default task timeout in seconds.
        """
        self.max_workers = max(1, max_workers)
        self.max_retries = max_retries
        self.default_timeout_seconds = default_timeout_seconds

    def compute_execution_stages(self, plan: ExecutionPlan) -> List[List[Task]]:
        """Groups tasks into topological execution stages where tasks in the same stage can run in parallel."""
        plan.validate_dag()
        task_map = {t.task_id: t for t in plan.tasks}
        completed: Set[str] = set()
        stages: List[List[Task]] = []
        remaining = set(task_map.keys())

        while remaining:
            # Find all tasks whose dependencies are entirely satisfied
            current_stage_ids = [
                tid for tid in remaining
                if all(dep in completed for dep in task_map[tid].dependencies)
            ]
            if not current_stage_ids:
                raise ValueError(f"Unresolvable dependencies in execution plan: remaining={remaining}")

            # Sort deterministically by task_id
            current_stage_ids.sort()
            current_stage = [task_map[tid] for tid in current_stage_ids]
            stages.append(current_stage)

            for tid in current_stage_ids:
                remaining.remove(tid)
                completed.add(tid)

        return stages

    def execute_task(
        self,
        task: Task,
        registry: SubagentRegistry,
    ) -> TaskResult:
        """Executes a single task with retries, unified timeout management, and status logging.

        Applies consistently across both single-worker and multi-worker execution paths.
        """
        try:
            subagent = registry.find_for_task(task)
        except Exception as exc:
            err = f"Failed to resolve subagent for task '{task.task_id}' (agent_name='{task.agent_name}'): {exc}"
            logger.error(err)
            return TaskResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error=err,
            )

        if subagent is None:
            err = f"No subagent available to handle task '{task.task_id}' (agent_name='{task.agent_name}')"
            logger.error(err)
            return TaskResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status=TaskStatus.FAILED,
                error=err,
            )

        task.status = TaskStatus.RUNNING

        # FIX-3: Scheduler max_retries is the upper bound on allowed retries.
        if self.max_retries is not None:
            allowed_retries = min(task.max_retries, self.max_retries)
        else:
            allowed_retries = task.max_retries
        allowed_retries = max(0, allowed_retries)
        max_attempts = 1 + allowed_retries

        timeout = task.timeout_seconds if task.timeout_seconds is not None else self.default_timeout_seconds
        last_error = ""
        last_result: Optional[TaskResult] = None

        for attempt in range(max_attempts):
            task.retry_count = attempt
            start_time = time.perf_counter()

            try:
                # FIX-2: Unified timeout enforcement inside execute_task
                if timeout is not None and timeout > 0:
                    task_exec = ThreadPoolExecutor(max_workers=1)
                    fut = task_exec.submit(subagent.execute, task)
                    try:
                        result = fut.result(timeout=timeout)
                        task_exec.shutdown(wait=False)
                    except FuturesTimeoutError:
                        fut.cancel()
                        task_exec.shutdown(wait=False, cancel_futures=True)
                        elapsed = time.perf_counter() - start_time
                        last_error = f"Task timed out after {timeout} seconds"
                        last_result = TaskResult(
                            task_id=task.task_id,
                            agent_name=task.agent_name,
                            status=TaskStatus.FAILED,
                            error=last_error,
                            execution_time_seconds=elapsed,
                            retry_count=attempt,
                        )
                        logger.warning(
                            "Task '%s' attempt %d/%d timed out after %s seconds",
                            task.task_id, attempt + 1, max_attempts, timeout
                        )
                        retry_on_timeout = task.metadata.get("retry_on_timeout", False)
                        if not retry_on_timeout:
                            task.status = TaskStatus.FAILED
                            return last_result
                        continue
                    except Exception as exc:
                        task_exec.shutdown(wait=False)
                        raise exc
                else:
                    result = subagent.execute(task)

                result.retry_count = attempt
                if result.status == TaskStatus.SUCCESS:
                    task.status = TaskStatus.SUCCESS
                    return result
                else:
                    last_error = result.error or "Subagent returned failure status"
                    last_result = result
                    logger.warning(
                        "Task '%s' attempt %d/%d failed: %s",
                        task.task_id, attempt + 1, max_attempts, last_error
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                tb = traceback.format_exc()
                last_error = f"{type(exc).__name__}: {str(exc)}\n{tb}"
                logger.warning(
                    "Task '%s' attempt %d/%d encountered exception: %s",
                    task.task_id, attempt + 1, max_attempts, exc
                )
                last_result = TaskResult(
                    task_id=task.task_id,
                    agent_name=task.agent_name,
                    status=TaskStatus.FAILED,
                    error=last_error,
                    execution_time_seconds=elapsed,
                    retry_count=attempt,
                )

        # Persistent failure: evaluate whether task is allowed to degrade
        allow_degraded = task.metadata.get("allow_degraded", False)
        final_status = TaskStatus.DEGRADED if allow_degraded else TaskStatus.FAILED
        task.status = final_status

        if last_result is not None:
            last_result.status = final_status
            last_result.retry_count = allowed_retries
            return last_result

        return TaskResult(
            task_id=task.task_id,
            agent_name=task.agent_name,
            status=final_status,
            error=last_error,
            retry_count=allowed_retries,
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        registry: SubagentRegistry,
    ) -> Dict[str, TaskResult]:
        """Executes an entire ExecutionPlan stage by stage.

        Independent tasks within the same stage run concurrently using ThreadPoolExecutor
        (or sequentially when max_workers=1).
        """
        stages = self.compute_execution_stages(plan)
        results: Dict[str, TaskResult] = {}

        for stage_idx, stage_tasks in enumerate(stages):
            logger.debug("Executing stage %d with %d tasks...", stage_idx + 1, len(stage_tasks))

            # Check if any dependencies for tasks in this stage failed
            runnable_tasks: List[Task] = []
            for task in stage_tasks:
                failed_dep = None
                for dep in task.dependencies:
                    dep_res = results.get(dep)
                    if dep_res and dep_res.status not in (TaskStatus.SUCCESS, TaskStatus.DEGRADED):
                        failed_dep = dep
                        break

                if failed_dep:
                    err_msg = f"Skipped because dependency '{failed_dep}' had status {results[failed_dep].status.value}"
                    logger.warning("Task '%s' %s", task.task_id, err_msg)
                    task.status = TaskStatus.SKIPPED
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        agent_name=task.agent_name,
                        status=TaskStatus.SKIPPED,
                        error=err_msg,
                    )
                else:
                    runnable_tasks.append(task)

            if not runnable_tasks:
                continue

            if self.max_workers == 1 or len(runnable_tasks) == 1:
                # Deterministic sequential execution
                for task in runnable_tasks:
                    res = self.execute_task(task, registry)
                    results[task.task_id] = res
            else:
                # Parallel execution across worker pool without double timeout
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(runnable_tasks))) as executor:
                    futures = {
                        executor.submit(self.execute_task, task, registry): task
                        for task in runnable_tasks
                    }
                    for fut, task in futures.items():
                        try:
                            res = fut.result()
                        except Exception as exc:
                            logger.error("Unexpected worker error on task '%s': %s", task.task_id, exc)
                            res = TaskResult(
                                task_id=task.task_id,
                                agent_name=task.agent_name,
                                status=TaskStatus.FAILED,
                                error=f"Worker exception: {exc}\n{traceback.format_exc()}",
                            )
                        results[task.task_id] = res

        return results
