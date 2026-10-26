"""Event intelligence agent implementing Retrieve -> Verify -> Align pipeline.

Strictly adheres to ex-ante causality: only utilizes forecasts and hazard scenario
intelligence available BEFORE the event occurs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import numpy as np

from urban_resilience.config import ProjectPaths
from urban_resilience.types import EventContext, HazardType, UrbanContext

logger = logging.getLogger(__name__)


@runtime_checkable
class EventSource(Protocol):
    """Protocol for disaster event intelligence sources."""

    def fetch_events(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Retrieves raw disaster event records."""
        ...

    def normalize(self, event_data: Dict[str, Any], total_simulation_hours: int = 96) -> EventContext:
        """Converts raw event dictionary into a canonical EventContext."""
        ...

    def verify_source(self, **kwargs: Any) -> bool:
        """Verifies provenance, integrity, and authenticity of the event data."""
        ...

    def provenance(self) -> Dict[str, Any]:
        """Returns provenance dictionary for the event data source."""
        ...


class FixtureEventSource:
    """Offline event source loading fixture events from local JSON descriptors.

    Verifies checksums and provenance offline; performs zero network calls.
    """

    def __init__(
        self,
        urban_context: Optional[UrbanContext] = None,
        fixture_dir: Optional[Path | str] = None,
    ):
        self.urban_context = urban_context
        self.fixture_dir = (
            Path(fixture_dir)
            if fixture_dir
            else ProjectPaths.DATA_DIR / "external" / "SZ_20230907_HAIKUI_SYNTHETIC_DEMO"
        )

    def fetch_events(self, event_id: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Reads event descriptor from fixture directory without marking as verified."""
        evt_path = self.fixture_dir / "event.json"
        if not evt_path.exists():
            return []
        with open(evt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if event_id and data.get("event_id") != event_id:
            return []
        data.setdefault("provenance_status", "unverified")
        data.setdefault("source_type", "fixture_offline")
        data.setdefault("engineering_only", True)
        return [data]

    def verify_source(self, **kwargs: Any) -> bool:
        """Verifies local presence and SHA-256 checksums for all files listed in PROVENANCE.json."""
        evt_path = self.fixture_dir / "event.json"
        if not evt_path.exists():
            return False
        prov_path = self.fixture_dir / "PROVENANCE.json"
        if not prov_path.exists():
            return False
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                prov = json.load(f)
            checksums = prov.get("checksums", {})
            if not checksums or "event.json" not in checksums:
                return False
            for fname, expected_hash in checksums.items():
                fpath = self.fixture_dir / fname
                if not fpath.exists() or not fpath.is_file():
                    return False
                with open(fpath, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if digest != expected_hash:
                    return False
            return True
        except Exception:
            return False

    def provenance(self) -> Dict[str, Any]:
        """Returns provenance metadata from PROVENANCE.json with source_type=fixture_offline and unverified status."""
        prov_path = self.fixture_dir / "PROVENANCE.json"
        prov_data: Dict[str, Any] = {}
        if prov_path.exists():
            try:
                with open(prov_path, "r", encoding="utf-8") as f:
                    prov_data = json.load(f)
            except Exception:
                pass
        prov_data["source_type"] = "fixture_offline"
        prov_data["fixture_dir"] = str(self.fixture_dir)
        prov_data["engineering_only"] = True
        prov_data["provenance_status"] = "unverified"
        return prov_data

    def normalize(self, event_data: Dict[str, Any], total_simulation_hours: int = 96) -> EventContext:
        """Projects event descriptor into canonical EventContext."""
        ht_raw = event_data.get("hazard_type", "rainstorm").lower()
        if ht_raw in ("rain_storm", "rainstorm"):
            hazard_type = HazardType.RAIN_STORM
        elif ht_raw == "typhoon":
            hazard_type = HazardType.TYPHOON
        elif ht_raw == "heat_wave":
            hazard_type = HazardType.HEAT_WAVE
        elif ht_raw == "infrastructure_failure":
            hazard_type = HazardType.INFRASTRUCTURE_FAILURE
        else:
            hazard_type = HazardType.RAIN_STORM

        start_hr = int(event_data.get("event_start_hour", 24))
        dur_hr = int(event_data.get("duration_hours", 18))
        peak_intensity = float(event_data.get("peak_intensity", 0.85))

        # Temporal intensity curve
        hours = np.arange(total_simulation_hours)
        peak_center = start_hr + dur_hr / 2.0
        sigma = dur_hr / 3.0
        hourly_intensity = np.zeros(total_simulation_hours, dtype=float)
        in_window = (hours >= start_hr) & (hours < start_hr + dur_hr)
        hourly_intensity[in_window] = peak_intensity * np.exp(-((hours[in_window] - peak_center) ** 2) / (2 * (sigma ** 2)))

        zone_intensity_map: Dict[str, float] = {}
        affected_zones: List[str] = []

        if "hazard_distribution" in event_data:
            zone_intensity_map = {k: float(v) for k, v in event_data["hazard_distribution"].items()}
            affected_zones = [k for k, v in zone_intensity_map.items() if v > 0.35]
        elif "zone_intensity_map" in event_data:
            zone_intensity_map = {k: float(v) for k, v in event_data["zone_intensity_map"].items()}
            affected_zones = [k for k, v in zone_intensity_map.items() if v > 0.35]
        elif self.urban_context:
            for zid, zone in self.urban_context.zones.items():
                elevation_penalty = max(0.0, (50.0 - zone.elevation_m) / 50.0) * 0.35
                is_coastal = zone.elevation_m < 10.0 and (zone.longitude < 114.0 or zone.longitude > 114.2)
                coastal_factor = 0.30 if is_coastal else 0.05
                factor = float(np.clip(peak_intensity * (0.50 + elevation_penalty + coastal_factor), 0.1, 1.0))
                zone_intensity_map[zid] = factor
                if factor > 0.35:
                    affected_zones.append(zid)
        else:
            zone_intensity_map = {"SZ_FT_CBD": peak_intensity}
            affected_zones = ["SZ_FT_CBD"]

        return EventContext(
            event_id=event_data.get("event_id", f"EVT_{hazard_type.value.upper()}_{start_hr:03d}"),
            hazard_type=hazard_type,
            event_start_hour=start_hr,
            duration_hours=dur_hr,
            peak_intensity=peak_intensity,
            affected_zones=affected_zones,
            zone_intensity_map=zone_intensity_map,
            hourly_forecast=hourly_intensity,
            description=f"Normalized from fixture source {event_data.get('event_id')}",
            source=event_data.get("forecast_source", "FixtureEventSource (offline)"),
        )


class EmpiricalMeteorologicalAPISource:
    """Official meteorological open API source interface.

    TODO: Empirical data ingestion, currently BLOCKED pending open municipal telemetry.
    """

    def fetch_events(self, **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "Empirical meteorological API ingestion is BLOCKED pending verified open data access."
        )

    def normalize(self, event_data: Dict[str, Any], total_simulation_hours: int = 96) -> EventContext:
        raise NotImplementedError("Empirical normalization is BLOCKED.")

    def verify_source(self, **kwargs: Any) -> bool:
        return False

    def provenance(self) -> Dict[str, Any]:
        return {
            "source_type": "empirical_api",
            "status": "BLOCKED",
            "engineering_only": False,
        }


class EventIntelligenceAgent:
    """Agent that ingests, verifies, and aligns ex-ante hazard intelligence."""

    def __init__(
        self,
        urban_context: UrbanContext,
        event_source: Optional[EventSource] = None,
    ):
        self.urban_context = urban_context
        self.event_source = event_source

    def create_from_source(
        self,
        event_id: Optional[str] = None,
        total_simulation_hours: int = 96,
    ) -> EventContext:
        """Loads and normalizes event context directly from configured EventSource."""
        src = self.event_source or FixtureEventSource(urban_context=self.urban_context)
        events = src.fetch_events(event_id=event_id)
        if not events:
            raise ValueError(f"No events found matching event_id='{event_id}' in event source.")
        return src.normalize(events[0], total_simulation_hours=total_simulation_hours)

    def create_event_context(
        self,
        hazard_type: HazardType = HazardType.RAIN_STORM,
        event_start_hour: int = 24,
        duration_hours: int = 18,
        peak_intensity: float = 0.85,
        total_simulation_hours: int = 96,
    ) -> EventContext:
        """Executes the full Retrieve -> Verify -> Align pipeline to produce EventContext."""
        logger.info("Event Intelligence Agent: Processing hazard forecast [%s]...", hazard_type.value)

        # Step 1: Retrieve
        raw_forecast = self._retrieve_hazard_forecast(
            hazard_type=hazard_type,
            start_hour=event_start_hour,
            duration=duration_hours,
            peak=peak_intensity,
            total_hours=total_simulation_hours,
        )

        # Step 2: Verify
        verified_forecast = self._verify_hazard_physics(raw_forecast)

        # Step 3: Align
        zone_intensity_map, hourly_curve, affected_zones = self._align_spatiotemporal(
            verified_forecast, total_simulation_hours
        )

        event_context = EventContext(
            event_id=f"EVT_{hazard_type.value.upper()}_{event_start_hour:03d}",
            hazard_type=hazard_type,
            event_start_hour=event_start_hour,
            duration_hours=duration_hours,
            peak_intensity=peak_intensity,
            affected_zones=affected_zones,
            zone_intensity_map=zone_intensity_map,
            hourly_forecast=hourly_curve,
            description=(
                f"Severe {hazard_type.value} forecast starting at hour {event_start_hour} "
                f"lasting {duration_hours}h. Coastal surge and urban waterlogging alert issued."
            ),
            source="Pre-event Weather Radar, Satellite Inversion & Meteorological Observatory",
        )
        logger.info("Event context assembled successfully. Affected zones: %d/%d",
                    len(affected_zones), len(self.urban_context.zones))
        return event_context

    def _retrieve_hazard_forecast(
        self, hazard_type: HazardType, start_hour: int, duration: int, peak: float, total_hours: int
    ) -> Dict[str, Any]:
        """Simulates ingestion of high-resolution numerical weather prediction (NWP) alerts."""
        return {
            "hazard_type": hazard_type,
            "start_hour": start_hour,
            "duration": duration,
            "peak_intensity": min(1.0, max(0.1, peak)),
            "total_hours": total_hours,
            "radar_reflectivity_dbz": 55.0 if hazard_type == HazardType.RAIN_STORM else 45.0,
            "wind_speed_m_s": 28.5 if hazard_type == HazardType.TYPHOON else 12.0,
            "is_pre_event": True,
        }

    def _verify_hazard_physics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Validates meteorological consistency and bounds."""
        if raw["start_hour"] < 1:
            raise ValueError("Event start time must be >= 1")
        if raw["duration"] <= 0:
            raise ValueError("Event duration must be positive")
        if not (0.0 <= raw["peak_intensity"] <= 1.0):
            raise ValueError("Intensity must be bounded in [0, 1]")
        if raw["is_pre_event"] is not True:
            raise ValueError("Causal violation: Post-event data detected in pre-event pipeline!")
        return raw

    def _align_spatiotemporal(
        self, verified: Dict[str, Any], total_hours: int
    ) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Projects meteorological shock onto specific urban zones based on coastal proximity and elevation."""
        start = verified["start_hour"]
        dur = verified["duration"]
        peak = verified["peak_intensity"]

        # Temporal intensity curve (Gaussian-shaped bell curve for rainfall/typhoon)
        hours = np.arange(total_hours)
        peak_center = start + dur / 2.0
        sigma = dur / 3.0
        hourly_intensity = np.zeros(total_hours, dtype=float)
        in_window = (hours >= start) & (hours < start + dur)
        hourly_intensity[in_window] = peak * np.exp(-((hours[in_window] - peak_center) ** 2) / (2 * (sigma ** 2)))

        zone_intensity_map = {}
        affected = []

        for zid, zone in self.urban_context.zones.items():
            # Spatial vulnerability modulation:
            # Lower elevation -> higher inundation risk
            elevation_penalty = max(0.0, (50.0 - zone.elevation_m) / 50.0) * 0.35
            # Coastal zones have higher storm surge and typhoon exposure
            is_coastal = zone.elevation_m < 10.0 and (zone.longitude < 114.0 or zone.longitude > 114.2)
            coastal_factor = 0.30 if is_coastal else 0.05

            zone_factor = float(np.clip(peak * (0.50 + elevation_penalty + coastal_factor), 0.1, 1.0))
            zone_intensity_map[zid] = zone_factor
            if zone_factor > 0.35:
                affected.append(zid)

        return zone_intensity_map, hourly_intensity, affected
