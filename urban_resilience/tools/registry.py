"""Tool specification, registry, and execution dispatcher with whitelist & security enforcement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type
import uuid

from pydantic import BaseModel, ValidationError

from urban_resilience.llm.schemas import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification of an executable python calculation tool."""

    name: str
    description: str
    fn: Callable[..., Dict[str, Any]]
    args_schema: Optional[Type[BaseModel]] = None
    allowed_regions: Optional[List[str]] = None
    read_only: bool = True
    timeout: float = 10.0
    data_snapshot: str = "v1"
    tool_version: str = "1.0.0"
    units: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Thread-safe registry for computational tools with validation, region checks, and timeouts."""

    def __init__(
        self,
        default_timeout_seconds: float = 5.0,
        expected_snapshot_id: Optional[str] = None,
    ):
        self._tools: Dict[str, ToolSpec] = {}
        self.default_timeout_seconds = default_timeout_seconds
        self.expected_snapshot_id = expected_snapshot_id

    def register(self, spec: ToolSpec) -> None:
        """Registers a tool specification into the registry."""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Looks up a tool specification by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        """Returns all registered tool specifications."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns JSON-compatible schema descriptions for LLM tool selection."""
        schemas = []
        for spec in self._tools.values():
            schema_info: Dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "tool_version": spec.tool_version,
                "read_only": spec.read_only,
            }
            if spec.args_schema and hasattr(spec.args_schema, "model_json_schema"):
                schema_info["parameters"] = spec.args_schema.model_json_schema()
            elif spec.args_schema:
                schema_info["parameters"] = str(spec.args_schema)
            schemas.append(schema_info)
        return schemas

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        region_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        deadline: Optional[float] = None,
        snapshot_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> ToolResult:
        """Executes a tool with whitelist, argument validation, region enforcement, and timeout guard.

        Args:
            tool_name: Name of registered tool.
            args: Input arguments.
            region_id: Target zone/region ID. If omitted, inspected from args.
            context: Shared runtime context (e.g., mobility_network, urban_context).
            deadline: Optional absolute timestamp (time.time()) deadline.
            snapshot_id: Optional data snapshot version to verify against tool spec.
            call_id: Optional tracking identifier. Auto-generated if None.

        Returns:
            ToolResult containing status, data, units, and telemetry.
        """
        active_call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        spec = self._tools.get(tool_name)

        # 1. Whitelist Check
        if spec is None:
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="error",
                error=f"Unregistered tool '{tool_name}'. Allowed tools: {list(self._tools.keys())}",
                data={},
            )

        # 2. Snapshot Check
        if snapshot_id is not None and snapshot_id != spec.data_snapshot:
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="error",
                error=f"Data snapshot mismatch for '{tool_name}': tool requires '{spec.data_snapshot}', got '{snapshot_id}'.",
                data={},
            )

        # 3. Deadline Check
        now = time.time()
        if deadline is not None and now >= deadline:
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="timeout",
                error=f"Execution deadline ({deadline:.2f}) already expired (current: {now:.2f}).",
                data={},
            )

        # Target region resolution & region_ids list check
        target_region = region_id or args.get("region_id") or args.get("zone_id")
        target_regions_list = args.get("region_ids")

        # 4. Region Authorization Check
        if spec.allowed_regions is not None:
            if target_regions_list is not None and isinstance(target_regions_list, list):
                unauthorized = [r for r in target_regions_list if r not in spec.allowed_regions]
                if unauthorized:
                    return ToolResult(
                        call_id=active_call_id,
                        tool=tool_name,
                        status="error",
                        error=(
                            f"Regions {unauthorized} in 'region_ids' are unauthorized for tool '{tool_name}'. "
                            f"Allowed regions: {spec.allowed_regions}"
                        ),
                        data={},
                    )
            elif not target_region:
                return ToolResult(
                    call_id=active_call_id,
                    tool=tool_name,
                    status="error",
                    error=f"Tool '{tool_name}' requires an authorized region, but none was provided.",
                    data={},
                )
            elif target_region not in spec.allowed_regions:
                return ToolResult(
                    call_id=active_call_id,
                    tool=tool_name,
                    status="error",
                    error=(
                        f"Region '{target_region}' is unauthorized for tool '{tool_name}'. "
                        f"Allowed regions: {spec.allowed_regions}"
                    ),
                    data={},
                )

        # 5. Argument Validation via Pydantic
        clean_args = dict(args)
        if spec.args_schema is not None:
            try:
                validated = spec.args_schema.model_validate(args)
                clean_args = validated.model_dump()
            except ValidationError as exc:
                return ToolResult(
                    call_id=active_call_id,
                    tool=tool_name,
                    status="error",
                    error=f"Invalid arguments for tool '{tool_name}': {exc}",
                    data={},
                )

        # 6. Determine Timeout
        now = time.time()
        timeout = spec.timeout
        if deadline is not None:
            remaining = max(0.01, deadline - now)
            timeout = min(timeout, remaining)

        # 7. Invocation with timeout guard (non-blocking executor)
        start_time = time.perf_counter()

        def _invoke() -> Dict[str, Any]:
            sig = inspect.signature(spec.fn)
            kwargs = dict(clean_args)
            if "context" in sig.parameters and context is not None:
                kwargs["context"] = context
            return spec.fn(**kwargs)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_invoke)
            data = future.result(timeout=timeout)
            executor.shutdown(wait=False)

            elapsed = time.perf_counter() - start_time
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="success",
                data=data,
                units=spec.units,
                coverage=1.0,
                data_snapshot=spec.data_snapshot,
                tool_version=spec.tool_version,
                execution_time_seconds=elapsed,
            )

        except FutureTimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            elapsed = time.perf_counter() - start_time
            logger.warning("Tool '%s' (call_id: %s) timed out after %.2fs", tool_name, active_call_id, elapsed)
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="timeout",
                error=f"Tool execution timed out after {timeout:.2f}s",
                data={},
                data_snapshot=spec.data_snapshot,
                tool_version=spec.tool_version,
                execution_time_seconds=elapsed,
            )
        except Exception as exc:
            executor.shutdown(wait=False)
            elapsed = time.perf_counter() - start_time
            logger.error("Tool '%s' (call_id: %s) failed: %s", tool_name, active_call_id, exc, exc_info=True)
            return ToolResult(
                call_id=active_call_id,
                tool=tool_name,
                status="error",
                error=f"Tool execution error: {type(exc).__name__}: {str(exc)}",
                data={},
                data_snapshot=spec.data_snapshot,
                tool_version=spec.tool_version,
                execution_time_seconds=elapsed,
            )
