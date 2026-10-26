"""Pydantic schemas for LLM tool invocation, decisions, and verified feature reporting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCall(BaseModel):
    """Specification of an individual tool call issued by an LLM agent."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(description="Unique identifier for this tool call, e.g. call_001")
    tool: str = Field(description="Name of the tool to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="Key-value arguments for the tool")


class ToolResult(BaseModel):
    """Structured execution output returned by a tool."""

    model_config = ConfigDict(extra="allow")

    call_id: str
    tool: str = Field(default="", description="Name of executed tool")
    status: str = Field(default="success", description="'success' | 'error' | 'timeout'")
    data: Dict[str, Any] = Field(default_factory=dict)
    units: Dict[str, str] = Field(default_factory=dict)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    data_snapshot: str = Field(default="v1")
    tool_version: str = Field(default="1.0.0")
    error: Optional[str] = None
    execution_time_seconds: float = 0.0


class FeatureQuality(BaseModel):
    """Data quality and validity indicators for an extracted feature."""

    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_days: int = Field(default=1, ge=0)
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    extra: Dict[str, Any] = Field(default_factory=dict)


class FeatureValue(BaseModel):
    """A single verified numeric feature extracted by Python tools."""

    name: str
    value: float
    unit: str = ""
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="List of tool call_ids that produced or justified this numeric value",
    )
    quality: FeatureQuality = Field(default_factory=FeatureQuality)


class FeatureReport(BaseModel):
    """Structured feature report produced by an agent for a specific region/zone."""

    model_config = ConfigDict(extra="allow")

    agent: str
    region_id: str
    round: int = 1
    status: str = Field(default="complete", description="'complete' | 'insufficient' | 'failed'")
    features: Dict[str, FeatureValue] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    data_snapshot: str = "v1"
    tool_version: str = "1.0.0"
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def validate_evidence_refs(self) -> FeatureReport:
        """Enforces that all numeric features have at least one explicit evidence reference."""
        if self.status == "complete":
            for fname, fval in self.features.items():
                if not fval.evidence_refs:
                    raise ValueError(
                        f"Feature '{fname}' in FeatureReport has empty evidence_refs. "
                        f"Every numeric value must explicitly reference the tool call_id that produced it."
                    )
        return self


class AgentDecision(BaseModel):
    """Decision output of an LLM ReAct agent at each reasoning step."""

    model_config = ConfigDict(extra="allow")

    action: str = Field(description="'call_tool' | 'final' | 'insufficient'")
    tool_call: Optional[ToolCall] = None
    feature_report: Optional[FeatureReport] = None
    reasoning: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    """Review decision output by LLM Supervisor."""

    model_config = ConfigDict(extra="allow")

    action: str = Field(description="'delegate' | 'stop' | 'insufficient'")
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: Optional[str] = None
    budget_consumed: float = 0.0
