"""EventDiscoveryAgent discovering, verifying, and structuring disaster bulletins."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from urban_resilience.event.schemas import EventQuery, HazardBulletin, HazardSeverityLevel
from urban_resilience.event.search_tools import search_bulletins_in_store
from urban_resilience.event.verifier import OfficialSourceVerifier

logger = logging.getLogger(__name__)


class EventDiscoveryAgent:
    """Agent discovering disaster events from verified official bulletins."""

    def __init__(
        self,
        verifier: Optional[OfficialSourceVerifier] = None,
        bulletin_store: Optional[List[HazardBulletin]] = None,
    ):
        self.verifier = verifier or OfficialSourceVerifier()
        self.bulletin_store: List[HazardBulletin] = list(bulletin_store or [])

    def register_bulletin(self, bulletin: HazardBulletin) -> None:
        """Adds a bulletin record into the agent's offline store."""
        self.bulletin_store.append(bulletin)

    def discover_events(
        self,
        query: EventQuery,
        require_verified: bool = True,
    ) -> List[HazardBulletin]:
        """Discovers event bulletins matching query, verifying source authenticity.

        Args:
            query: The event search parameters.
            require_verified: If True, only bulletins verified by OfficialSourceVerifier are returned.

        Returns:
            List of matching HazardBulletin records.
        """
        raw_matches = search_bulletins_in_store(query, self.bulletin_store)
        verified_matches: List[HazardBulletin] = []

        for b in raw_matches:
            is_valid, reason = self.verifier.verify_source(b.source_authority, b.source_channel)
            b.is_verified_official = is_valid
            if require_verified and not is_valid:
                logger.warning("EventDiscoveryAgent: filtered out unverified bulletin '%s': %s", b.bulletin_id, reason)
                continue
            verified_matches.append(b)

        return verified_matches

    def parse_bulletin_text(
        self,
        bulletin_id: str,
        raw_text: str,
        source_authority: str,
        source_channel: str = "",
        published_at: Optional[str] = None,
    ) -> HazardBulletin:
        """Parses unstructured emergency notice text into a structured HazardBulletin.

        Extracts warning level (RED, ORANGE, YELLOW, BLUE), peak rainfall if mentioned,
        and affected districts/zones.
        """
        is_verified, reason = self.verifier.verify_source(source_authority, source_channel)
        if not is_verified:
            logger.warning("Parsing bulletin from unverified authority '%s': %s", source_authority, reason)

        # Detect severity level
        text_upper = raw_text.upper()
        if "RED" in text_upper or "红色" in raw_text:
            severity = HazardSeverityLevel.RED
        elif "ORANGE" in text_upper or "橙色" in raw_text:
            severity = HazardSeverityLevel.ORANGE
        elif "YELLOW" in text_upper or "黄色" in raw_text:
            severity = HazardSeverityLevel.YELLOW
        elif "BLUE" in text_upper or "蓝色" in raw_text:
            severity = HazardSeverityLevel.BLUE
        else:
            severity = HazardSeverityLevel.UNKNOWN

        # Detect hazard type
        hazard_type = "rainstorm"
        if "台风" in raw_text or "typhoon" in raw_text.lower():
            hazard_type = "typhoon"
        elif "暴雨" in raw_text or "rain" in raw_text.lower():
            hazard_type = "rainstorm"
        elif "内涝" in raw_text or "flood" in raw_text.lower():
            hazard_type = "urban_flood"

        # Regex for peak rainfall in mm (e.g., "120.5mm", "85毫米")
        peak_rainfall = None
        rain_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)", raw_text, re.IGNORECASE)
        if rain_match:
            try:
                peak_rainfall = float(rain_match.group(1))
            except ValueError:
                pass

        # Regex for wind speed in m/s (e.g. "32.5m/s", "12级")
        max_wind = None
        wind_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|米/秒)", raw_text, re.IGNORECASE)
        if wind_match:
            try:
                max_wind = float(wind_match.group(1))
            except ValueError:
                pass

        # Detect affected districts in Shenzhen
        sz_districts = ["福田", "南山", "罗湖", "宝安", "龙岗", "龙华", "盐田", "坪山", "光明", "大鹏"]
        affected: List[str] = [d for d in sz_districts if d in raw_text]

        bulletin = HazardBulletin(
            bulletin_id=bulletin_id,
            source_authority=source_authority,
            source_channel=source_channel,
            published_at=published_at or "",
            hazard_type=hazard_type,
            severity_level=severity,
            affected_zones=affected,
            peak_rainfall_mm=peak_rainfall,
            max_wind_speed_ms=max_wind,
            is_verified_official=is_verified,
            raw_text=raw_text,
        )

        return bulletin
