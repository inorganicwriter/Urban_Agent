"""Domain and synthesis subagents wrapping existing domain models under uniform interfaces."""

from __future__ import annotations

import abc
import logging
import time
from typing import Dict, List, Optional

from urban_resilience.agents.base import BaseReActAgent
from urban_resilience.agents.functional_mix import FunctionalMixAgent
from urban_resilience.agents.intraday_flexibility import IntradayFlexibilityAgent
from urban_resilience.agents.predictability import PredictabilityAgent
from urban_resilience.agents.resilience_synthesis import ResilienceSynthesisAgent
from urban_resilience.agents.spatial_redundancy import SpatialRedundancyAgent
from urban_resilience.agents.temporal_redundancy import TemporalRedundancyAgent
from urban_resilience.config import AgentWeightConfig, SimulationConfig
from urban_resilience.orchestration.contracts import Task, TaskResult, TaskStatus
from urban_resilience.types import MobilityNetwork, UrbanContext

logger = logging.getLogger(__name__)


class BaseSubagent(abc.ABC):
    """Abstract base class for domain and synthesis subagents."""

    def __init__(self, name: str, capabilities: Optional[List[str]] = None):
        self.name = name
        self.capabilities = capabilities or []

    def can_handle(self, task: Task) -> bool:
        """Determines if this subagent can process the given task."""
        if task.agent_name == self.name:
            return True
        return task.action in self.capabilities

    @abc.abstractmethod
    def execute(self, task: Task) -> TaskResult:
        """Executes the task and returns a TaskResult."""
        pass


class DomainSubagent(BaseSubagent):
    """Subagent wrapping a ReAct domain agent."""

    def __init__(
        self,
        name: str,
        agent: BaseReActAgent,
        capabilities: Optional[List[str]] = None,
    ):
        caps = list(capabilities or [])
        default_caps = ["domain_analysis", "analyze", "query_data", "compute_metrics", "check_evidence"]
        for c in default_caps:
            if c not in caps:
                caps.append(c)
        super().__init__(name=name, capabilities=caps)
        self.agent = agent

    def execute(self, task: Task) -> TaskResult:
        """Executes domain analysis or ReAct step tool operations."""
        start_time = time.perf_counter()
        logger.debug("[%s] Executing task %s (action: %s)", self.name, task.task_id, task.action)

        try:
            if task.action in ("domain_analysis", "analyze"):
                evidence = self.agent.analyze()
                traces = [
                    {
                        "step_number": t.step_number,
                        "thought": t.thought,
                        "action_type": t.action_type,
                        "action_input": t.action_input,
                        "observation": str(t.observation)[:150],
                    }
                    for t in self.agent.traces
                ]
                elapsed = time.perf_counter() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status=TaskStatus.SUCCESS,
                    output=evidence,
                    execution_time_seconds=elapsed,
                    trace=traces,
                )

            elif task.action in ("query_data", "targeted_inquiry"):
                thought = task.payload.get("thought", f"Targeted inquiry for {self.name}")
                action_type = task.payload.get("action_type", "query_data")
                action_input = task.payload.get(
                    "action_input",
                    {k: v for k, v in task.payload.items() if k not in ("thought", "action_type")},
                )
                obs = self.agent.execute_react_step(
                    thought=thought,
                    action_type=action_type,
                    action_input=action_input,
                )
                traces = [
                    {
                        "step_number": t.step_number,
                        "thought": t.thought,
                        "action_type": t.action_type,
                        "action_input": t.action_input,
                        "observation": str(t.observation)[:150],
                    }
                    for t in self.agent.traces
                ]
                elapsed = time.perf_counter() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status=TaskStatus.SUCCESS,
                    output=obs,
                    execution_time_seconds=elapsed,
                    trace=traces,
                )
            else:
                raise ValueError(f"Unsupported action '{task.action}' for domain subagent '{self.name}'")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error("[%s] Error executing task %s: %s", self.name, task.task_id, exc, exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=TaskStatus.FAILED,
                error=str(exc),
                execution_time_seconds=elapsed,
            )


class TemporalSubagent(DomainSubagent):
    """Subagent for temporal buffer capacity and peak concentration."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        agent = TemporalRedundancyAgent(urban_context, mobility_network)
        super().__init__(name="temporal", agent=agent, capabilities=["temporal_redundancy"])


class FlexibilitySubagent(DomainSubagent):
    """Subagent for intraday commute flexibility and off-peak shifts."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        agent = IntradayFlexibilityAgent(urban_context, mobility_network)
        super().__init__(name="flexibility", agent=agent, capabilities=["intraday_flexibility", "targeted_inquiry"])


class PredictabilitySubagent(DomainSubagent):
    """Subagent for regularity and diurnal rhythm predictability."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        agent = PredictabilityAgent(urban_context, mobility_network)
        super().__init__(name="predictability", agent=agent, capabilities=["predictability"])


class SpatialSubagent(DomainSubagent):
    """Subagent for topological spatial redundancy and evacuation corridors."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        agent = SpatialRedundancyAgent(urban_context, mobility_network)
        super().__init__(name="spatial", agent=agent, capabilities=["spatial_redundancy", "targeted_inquiry"])


class FunctionalSubagent(DomainSubagent):
    """Subagent for POI diversity and land-use functional balance."""

    def __init__(self, urban_context: UrbanContext, mobility_network: MobilityNetwork):
        agent = FunctionalMixAgent(urban_context, mobility_network)
        super().__init__(name="functional", agent=agent, capabilities=["functional_mix"])


class SynthesisSubagent(BaseSubagent):
    """Subagent wrapping ResilienceSynthesisAgent for feature integration, reconciliation, and predictions."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        weight_config: Optional[AgentWeightConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
        synthesis_agent: Optional[ResilienceSynthesisAgent] = None,
    ):
        super().__init__(
            name="synthesis",
            capabilities=["synthesize", "feature_integration", "forecast", "spatial_outlook"],
        )
        self.synthesis_agent = synthesis_agent or ResilienceSynthesisAgent(
            urban_context=urban_context,
            mobility_network=mobility_network,
            weight_config=weight_config,
            sim_config=sim_config,
        )

    def execute(self, task: Task) -> TaskResult:
        """Executes feature integration, reconciliation, or event-conditioned prediction."""
        start_time = time.perf_counter()
        try:
            if task.action == "feature_integration":
                evidences = task.payload["evidences"]
                event_context = task.payload["event_context"]
                features = self.synthesis_agent._integrate_features(evidences, event_context)
                elapsed = time.perf_counter() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status=TaskStatus.SUCCESS,
                    output=features,
                    execution_time_seconds=elapsed,
                )

            elif task.action in ("synthesize", "forecast"):
                evidences = task.payload.get("evidences")
                event_context = task.payload["event_context"]
                zone_features = task.payload.get("zone_features")

                if zone_features is not None:
                    # Generate predictions using already reconciled/adjusted zone features
                    pred = self.synthesis_agent._generate_event_conditioned_prediction(zone_features, event_context)
                    outlook = self.synthesis_agent._generate_spatial_outlook(zone_features, event_context, pred)
                    output = {
                        "prediction": pred,
                        "spatial_outlook": outlook,
                        "zone_features": zone_features,
                    }
                else:
                    # Full synthesize call (with optional agents_pool)
                    agents_pool = task.payload.get("agents_pool")
                    pred, outlook, z_feat = self.synthesis_agent.synthesize(
                        evidences=evidences,
                        event_context=event_context,
                        agents_pool=agents_pool,
                    )
                    output = {
                        "prediction": pred,
                        "spatial_outlook": outlook,
                        "zone_features": z_feat,
                    }

                elapsed = time.perf_counter() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status=TaskStatus.SUCCESS,
                    output=output,
                    execution_time_seconds=elapsed,
                )

            else:
                raise ValueError(f"Unsupported action '{task.action}' for SynthesisSubagent")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error("[synthesis] Error executing task %s: %s", task.task_id, exc, exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=TaskStatus.FAILED,
                error=str(exc),
                execution_time_seconds=elapsed,
            )


DOMAIN_ALIASES: Dict[str, str] = {
    "temporalredundancyagent": "temporal",
    "intradayflexibilityagent": "flexibility",
    "predictabilityagent": "predictability",
    "spatialredundancyagent": "spatial",
    "functionalmixagent": "functional",
    "resiliencesynthesisagent": "synthesis",
}


class SubagentRegistry:
    """Registry maintaining active subagents indexed by name and capabilities."""

    def __init__(self):
        self._subagents: Dict[str, BaseSubagent] = {}

    def register(self, subagent: BaseSubagent) -> None:
        """Registers a subagent."""
        self._subagents[subagent.name] = subagent

    def get(self, name: str) -> Optional[BaseSubagent]:
        """Retrieves a subagent by name."""
        return self._subagents.get(name)

    def values(self):
        """Returns active subagents."""
        return self._subagents.values()

    def __iter__(self):
        return iter(self._subagents.values())

    def find_for_task(self, task: Task) -> Optional[BaseSubagent]:
        """Finds the best matching subagent for a given task.

        Exact name match is prioritized, followed by canonical domain alias resolution.
        If fallback by capability matches multiple subagents, an OrchestrationError is raised
        to prevent ambiguous dispatch.
        """
        # 1. Exact name match
        if task.agent_name in self._subagents:
            return self._subagents[task.agent_name]

        # 2. Canonical domain alias match
        normalized_name = task.agent_name.lower().replace("_", "").replace("-", "")
        if normalized_name in DOMAIN_ALIASES:
            canonical_name = DOMAIN_ALIASES[normalized_name]
            if canonical_name in self._subagents:
                return self._subagents[canonical_name]

        # 3. Capability match (Strict: error if ambiguous multiple matches)
        matching = [sub for sub in self._subagents.values() if sub.can_handle(task)]
        if len(matching) == 1:
            return matching[0]
        elif len(matching) > 1:
            from urban_resilience.orchestration.contracts import OrchestrationError

            matched_names = [sub.name for sub in matching]
            raise OrchestrationError(
                f"Ambiguous subagent resolution for task '{task.task_id}' (agent_name='{task.agent_name}', "
                f"action='{task.action}'): multiple capable subagents found ({matched_names}). "
                f"Explicit task.agent_name is required."
            )

        return None

    def list_subagents(self) -> List[str]:
        """Returns the list of registered subagent names."""
        return list(self._subagents.keys())
