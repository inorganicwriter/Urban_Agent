"""Base class for domain-expert agents implementing the shared ReAct workflow."""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List

from urban_resilience.types import AgentEvidence, MobilityNetwork, ReActTrace, UrbanContext

logger = logging.getLogger(__name__)


class BaseReActAgent(abc.ABC):
    """Abstract base agent providing ReAct (Reason -> Act -> Observe) capabilities."""

    def __init__(self, agent_name: str, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        self.agent_name = agent_name
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.traces: List[ReActTrace] = []
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._register_tools()

    def _register_tool(self, name: str, func: Callable[..., Any]):
        """Registers an executable tool for the agent."""
        self._tools[name] = func

    def _register_tools(self):
        """Registers default shared tools: query_data, compute_metrics, check_evidence."""
        self._register_tool("query_data", self._tool_query_data)
        self._register_tool("compute_metrics", self._tool_compute_metrics)
        self._register_tool("check_evidence", self._tool_check_evidence)

    def _tool_query_data(self, query_type: str, **kwargs) -> Any:
        """Query spatial or mobility records."""
        if query_type == "zone_info":
            zone_id = kwargs.get("zone_id")
            return self.urban_context.zones.get(zone_id)
        elif query_type == "hourly_series":
            zone_id = kwargs.get("zone_id")
            if zone_id:
                return self.mobility_network.hourly_time_series[zone_id].values
            return self.mobility_network.hourly_time_series.values
        elif query_type == "daily_profile":
            return self.mobility_network.daily_profiles
        elif query_type == "od_matrix":
            time_step = kwargs.get("time_step", 0)
            return self.mobility_network.od_matrices.get(time_step)
        return None

    def _tool_compute_metrics(self, metric_func_name: str, **kwargs) -> Any:
        """Executes an internal metric computation registered in subclasses."""
        if hasattr(self, metric_func_name):
            func = getattr(self, metric_func_name)
            return func(**kwargs)
        raise ValueError(f"Unknown metric function {metric_func_name} in {self.agent_name}")

    def _tool_check_evidence(self, metrics: Dict[str, float], bounds: Dict[str, tuple]) -> Dict[str, bool]:
        """Validates computed metrics against physical plausible bounds."""
        checks = {}
        for k, val in metrics.items():
            if k in bounds:
                low, high = bounds[k]
                checks[k] = bool(low <= val <= high)
            else:
                checks[k] = True
        return checks

    def execute_react_step(self, thought: str, action_type: str, action_input: Dict[str, Any]) -> Any:
        """Executes one step in the ReAct loop: Reason -> Act -> Observe."""
        step_idx = len(self.traces) + 1
        logger.debug("[%s] Step %d Reason: %s", self.agent_name, step_idx, thought)

        if action_type not in self._tools:
            obs = f"Error: Tool '{action_type}' not available in {self.agent_name}"
        else:
            try:
                obs = self._tools[action_type](**action_input)
            except Exception as e:
                obs = f"Execution error in {action_type}: {str(e)}"

        logger.debug("[%s] Step %d Act [%s] -> Observe: %s", self.agent_name, step_idx, action_type, str(obs)[:80])
        trace = ReActTrace(
            step_number=step_idx,
            thought=thought,
            action_type=action_type,
            action_input=action_input,
            observation=obs,
        )
        self.traces.append(trace)
        return obs

    @abc.abstractmethod
    def analyze(self) -> AgentEvidence:
        """Executes the agent's full domain analysis workflow and returns AgentEvidence."""
        pass
