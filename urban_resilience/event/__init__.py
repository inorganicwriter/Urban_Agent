"""Disaster event intelligence, bulletin structuring, and source verification.

STATUS: scaffold_offline (Offline bulletin parser scaffold).
External live data ingestion remains BLOCKED pending verified real disaster feeds.
"""

from urban_resilience.event.agent import EventDiscoveryAgent
from urban_resilience.event.schemas import (
    EventQuery,
    HazardBulletin,
    HazardSeverityLevel,
)
from urban_resilience.event.verifier import OfficialSourceVerifier

STATUS = "scaffold_offline"
DATA_STATUS = "BLOCKED"

__all__ = [
    "STATUS",
    "DATA_STATUS",
    "HazardSeverityLevel",
    "HazardBulletin",
    "EventQuery",
    "OfficialSourceVerifier",
    "EventDiscoveryAgent",
]
