"""Urban Resilience Anticipatory Multi-Agent Prediction Framework.

A multi-agent cognitive architecture for ex-ante urban resilience modeling,
integrating mobility dynamics, static context, ReAct-based domain agents,
evidence synthesis with targeted follow-up, and anticipatory disruption forecasting.
"""

__version__ = "1.0.0"
__author__ = "Urban Agent Team"

from urban_resilience.types import (
    UrbanContext,
    MobilityNetwork,
    EventContext,
    AgentEvidence,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    ValidationResult,
)
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.cli import main as cli_main
from urban_resilience.api import app as fastapi_app

__all__ = [
    "UrbanContext",
    "MobilityNetwork",
    "EventContext",
    "AgentEvidence",
    "ResiliencePrediction",
    "SpatialResilienceOutlook",
    "ValidationResult",
    "UrbanResiliencePipeline",
    "cli_main",
    "fastapi_app",
]
