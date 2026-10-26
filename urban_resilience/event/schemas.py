"""Schemas for disaster event bulletins, hazard severity, and queries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HazardSeverityLevel(str, Enum):
    """Official early warning severity color levels in China."""
    BLUE = "BLUE"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    UNKNOWN = "UNKNOWN"


class HazardBulletin(BaseModel):
    """Standardized hazard bulletin document representing official emergency notices."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bulletin_id: str = Field(..., description="Unique bulletin identifier")
    source_authority: str = Field(..., description="Issuing agency, e.g. 'Shenzhen Meteorological Bureau'")
    source_channel: str = Field(default="", description="Channel or URL, e.g. 'weather.sz.gov.cn'")
    published_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO timestamp of issuance")
    hazard_type: str = Field(default="rainstorm", description="Hazard category: rainstorm, typhoon, flood, etc.")
    severity_level: HazardSeverityLevel = Field(default=HazardSeverityLevel.YELLOW, description="Standardized severity level")
    affected_zones: List[str] = Field(default_factory=list, description="Targeted urban zone IDs or district names")
    peak_rainfall_mm: Optional[float] = Field(default=None, description="Reported or forecast peak hourly rainfall in mm")
    max_wind_speed_ms: Optional[float] = Field(default=None, description="Reported maximum wind speed in m/s")
    is_verified_official: bool = Field(default=False, description="Verification status by OfficialSourceVerifier")
    raw_text: str = Field(default="", description="Original bulletin notification text")


class EventQuery(BaseModel):
    """Query parameter specification for historical or real-time event discovery."""

    city: str = Field(default="Shenzhen", description="Target metropolitan municipality")
    keywords: List[str] = Field(default_factory=list, description="Search keywords, e.g. ['暴雨', '红色预警']")
    time_start: Optional[str] = Field(default=None, description="Start ISO date/datetime")
    time_end: Optional[str] = Field(default=None, description="End ISO date/datetime")
    min_severity: Optional[HazardSeverityLevel] = Field(default=None, description="Filter by minimum severity")
