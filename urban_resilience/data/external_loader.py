"""External historical event and empirical mobility data loader with strict schema and causality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from urban_resilience.config import compute_baseline_mean

logger = logging.getLogger(__name__)

MODEL_ZONES: Set[str] = {
    "SZ_FT_CBD",
    "SZ_NS_TECH",
    "SZ_NS_QH",
    "SZ_NS_SK",
    "SZ_LH_COMM",
    "SZ_BA_CTR",
    "SZ_LG_BT",
    "SZ_LH_RES",
    "SZ_YT_PORT",
    "SZ_GM_SCI",
    "SZ_PS_IND",
    "SZ_DP_ECO",
}


class ExternalEventMeta(BaseModel):
    """Metadata schema for external historical disaster event and forecast context."""

    event_id: str
    city_name: str = "Shenzhen"
    hazard_type: str = "rainstorm"
    start_time: datetime
    end_time: datetime
    forecast_available_time: datetime
    lead_time_hours: int = Field(ge=1)
    duration_hours: int = Field(ge=1)
    total_hours: int = Field(ge=1)
    event_start_hour: int = Field(ge=0)
    peak_intensity: float = Field(ge=0.0, le=1.0)
    hazard_distribution: Optional[Dict[str, float]] = None
    hazard_forecast_source: Optional[str] = None
    hazard_forecast_available_time: Optional[datetime] = None
    forecast_source: str = "Official Meteorological Early Warning"
    role: str = "test"
    data_kind: str = "synthetic"
    provenance_status: str = "unverified"
    input_mode: str = "ex_ante_forecast"

    @field_validator("data_kind")
    @classmethod
    def validate_data_kind(cls, v: str) -> str:
        if v not in ("observed", "fixture", "synthetic"):
            raise ValueError(f"data_kind must be 'observed', 'fixture', or 'synthetic', got '{v}'")
        return v

    @field_validator("provenance_status")
    @classmethod
    def validate_provenance_status(cls, v: str) -> str:
        if v not in ("verified", "unverified"):
            raise ValueError(f"provenance_status must be 'verified' or 'unverified', got '{v}'")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("test", "calibration"):
            raise ValueError(f"role must be 'test' or 'calibration', got '{v}'")
        return v

    @field_validator("input_mode")
    @classmethod
    def validate_input_mode(cls, v: str) -> str:
        if v not in ("ex_ante_forecast", "observed_hindcast"):
            raise ValueError(f"input_mode must be 'ex_ante_forecast' or 'observed_hindcast', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_causality_and_timeline(self) -> ExternalEventMeta:
        # Strict provenance gate: observed data requires verified status
        if self.data_kind == "observed" and self.provenance_status != "verified":
            raise ValueError(
                f"Data policy violation: Event '{self.event_id}' is marked with data_kind='observed' "
                f"but has provenance_status='{self.provenance_status}'. "
                "Observed real-world datasets must be strictly verified with complete provenance "
                "(source_url/dataset_id, raw_file_checksums, aggregation_method). "
                "Use data_kind='synthetic' or 'fixture' and provenance_status='unverified' for unverified data."
            )

        # Timezone check
        for attr, dt in [
            ("start_time", self.start_time),
            ("end_time", self.end_time),
            ("forecast_available_time", self.forecast_available_time),
        ]:
            if dt.tzinfo is None:
                raise ValueError(f"Field '{attr}' must include timezone information (non-naive datetime): {dt}")

        if self.hazard_forecast_available_time is not None and self.hazard_forecast_available_time.tzinfo is None:
            raise ValueError("Field 'hazard_forecast_available_time' must include timezone information.")

        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be strictly after start_time ({self.start_time})")

        # Ex-ante causality verification: forecast must be available lead_time_hours before start_time
        if self.input_mode == "ex_ante_forecast":
            required_cutoff = self.start_time - timedelta(hours=self.lead_time_hours)
            if self.forecast_available_time > required_cutoff:
                raise ValueError(
                    f"Causality violation in ex_ante_forecast: forecast_available_time ({self.forecast_available_time}) "
                    f"must be <= start_time - lead_time_hours ({required_cutoff})"
                )

        if self.total_hours < self.event_start_hour + self.duration_hours:
            raise ValueError(
                f"total_hours ({self.total_hours}) must be >= event_start_hour ({self.event_start_hour}) "
                f"+ duration_hours ({self.duration_hours})"
            )

        # Hazard distribution validation & leakage prevention
        if self.hazard_distribution:
            # Check if verifiable forecast source & time are provided
            has_forecast_source = bool(self.hazard_forecast_source and self.hazard_forecast_source.strip())
            has_forecast_time = self.hazard_forecast_available_time is not None

            if not (has_forecast_source and has_forecast_time):
                logger.warning(
                    "Event '%s' specified custom hazard_distribution without verified "
                    "hazard_forecast_source and hazard_forecast_available_time. "
                    "Rejecting custom hazard_distribution to prevent data leakage.",
                    self.event_id,
                )
                self.hazard_distribution = None
            else:
                required_cutoff = self.start_time - timedelta(hours=self.lead_time_hours)
                if self.hazard_forecast_available_time > required_cutoff:
                    raise ValueError(
                        f"Causality violation in hazard_distribution forecast: "
                        f"hazard_forecast_available_time ({self.hazard_forecast_available_time}) "
                        f"must be <= start_time - lead_time_hours ({required_cutoff})"
                    )

                invalid_zones = set(self.hazard_distribution.keys()) - MODEL_ZONES
                if invalid_zones:
                    raise ValueError(f"hazard_distribution contains invalid model zone IDs: {invalid_zones}")
                for zid, val in self.hazard_distribution.items():
                    if not (0.0 <= val <= 1.0):
                        raise ValueError(f"hazard_distribution for zone '{zid}' must be in [0, 1], got {val}")

        return self


@dataclass
class ObservedEventDataset:
    """Complete in-memory dataset of an observed or fixture external event."""

    meta: ExternalEventMeta
    zone_mapping: pd.DataFrame
    hourly_zone_activity: pd.DataFrame  # hours x 12 model zones, raw aggregated values
    city_total_activity: np.ndarray  # total raw city activity per hour
    baseline_mean: float  # pre-event baseline mean for city total
    normalized_city_activity: np.ndarray  # dimensionless relative city activity
    normalized_zone_activity: pd.DataFrame  # hours x 12 zones, normalized by zone pre-event baselines
    zone_impacts: Dict[str, float] = field(default_factory=dict)  # model_zone_id -> observed impact
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_zones(self) -> Set[str]:
        return set(self.hourly_zone_activity.columns)


class ExternalDataLoader:
    """Loads, validates, and normalizes external event observations."""

    def __init__(self, target_zones: Optional[Set[str]] = None):
        self.target_zones = target_zones or MODEL_ZONES

    def load_event(self, event_dir: Path | str, verify_checksums: bool = True) -> ObservedEventDataset:
        """Loads and strictly validates an external event directory.

        Args:
            event_dir: Path to directory containing event.json, zones.csv, activity.csv, etc.
            verify_checksums: If True, computes and verifies SHA-256 hashes against PROVENANCE.json.

        Returns:
            ObservedEventDataset ready for external evaluation.
        """
        path = Path(event_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"External event directory does not exist: {path}")

        # 1. Load and parse PROVENANCE.json
        prov_file = path / "PROVENANCE.json"
        if not prov_file.is_file():
            raise FileNotFoundError(f"Missing required provenance file: {prov_file}")

        try:
            with open(prov_file, "r", encoding="utf-8") as f:
                provenance = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse PROVENANCE.json at {prov_file}: {e}") from e

        if verify_checksums:
            self._verify_checksums(path, provenance)

        # 2. Load and validate event.json
        event_file = path / "event.json"
        if not event_file.is_file():
            raise FileNotFoundError(f"Missing required event metadata file: {event_file}")

        try:
            with open(event_file, "r", encoding="utf-8") as f:
                event_data = json.load(f)
            meta = ExternalEventMeta(**event_data)
        except Exception as e:
            raise ValueError(f"Validation failed for event.json at {event_file}: {e}") from e

        # Ensure provenance schema and authenticity checks
        self._verify_provenance_schema(provenance, meta)

        # Ensure provenance data_kind matches meta data_kind
        prov_data_kind = provenance.get("data_kind")
        if prov_data_kind and prov_data_kind != meta.data_kind:
            raise ValueError(
                f"Mismatch in data_kind: PROVENANCE.json specifies '{prov_data_kind}', "
                f"but event.json specifies '{meta.data_kind}'"
            )

        # 3. Load zones.csv
        zones_file = path / "zones.csv"
        if not zones_file.is_file():
            raise FileNotFoundError(f"Missing required zone mapping file: {zones_file}")
        zones_df = self._load_zones(zones_file)

        # 4. Load activity.csv
        activity_file = path / "activity.csv"
        if not activity_file.is_file():
            raise FileNotFoundError(f"Missing required activity observation file: {activity_file}")

        (
            hourly_zone_activity,
            city_total_activity,
            baseline_mean,
            normalized_city_activity,
            normalized_zone_activity,
        ) = self._load_and_process_activity(activity_file, meta, zones_df)

        # 5. Optional impacts.csv
        zone_impacts: Dict[str, float] = {}
        impacts_file = path / "impacts.csv"
        if impacts_file.is_file():
            zone_impacts = self._load_impacts(impacts_file, zones_df)

        return ObservedEventDataset(
            meta=meta,
            zone_mapping=zones_df,
            hourly_zone_activity=hourly_zone_activity,
            city_total_activity=city_total_activity,
            baseline_mean=baseline_mean,
            normalized_city_activity=normalized_city_activity,
            normalized_zone_activity=normalized_zone_activity,
            zone_impacts=zone_impacts,
            provenance=provenance,
        )

    def _verify_checksums(self, event_dir: Path, provenance: Dict[str, Any]) -> None:
        """Verifies SHA-256 checksums of files listed in PROVENANCE.json."""
        checksums = provenance.get("checksums", {})
        if not checksums or not isinstance(checksums, dict):
            raise ValueError("PROVENANCE.json must contain a non-empty 'checksums' dictionary.")

        for filename, expected_hash in checksums.items():
            target_path = event_dir / filename
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"File '{filename}' declared in PROVENANCE.json does not exist at {target_path}"
                )
            actual_hash = self._compute_sha256(target_path)
            if actual_hash.lower() != expected_hash.strip().lower():
                raise ValueError(
                    f"Checksum mismatch for '{filename}': expected {expected_hash}, calculated {actual_hash}"
                )

    def _verify_provenance_schema(self, provenance: Dict[str, Any], meta: ExternalEventMeta) -> None:
        """Enforces provenance contracts, source auditing, and real-world authenticity checks."""
        if "retrieved_at" not in provenance and "accessed_date" not in provenance:
            raise ValueError("PROVENANCE.json missing required retrieval timestamp ('retrieved_at' or 'accessed_date').")
        required_keys = {"source_name", "license", "checksums"}
        missing_keys = required_keys - set(provenance.keys())
        if missing_keys:
            raise ValueError(f"PROVENANCE.json missing required contract fields: {missing_keys}")

        prov_status = provenance.get("provenance_status", "unverified")
        if prov_status not in ("verified", "unverified"):
            raise ValueError(f"PROVENANCE.json has invalid provenance_status '{prov_status}', must be 'verified' or 'unverified'")

        if prov_status != meta.provenance_status:
            raise ValueError(
                f"Mismatch in provenance_status: PROVENANCE.json specifies '{prov_status}', "
                f"but event.json specifies '{meta.provenance_status}'"
            )

        if meta.data_kind == "observed" or prov_status == "verified":
            source_url = provenance.get("source_url")
            dataset_id = provenance.get("dataset_id")
            if not (source_url or dataset_id):
                raise ValueError("Verified provenance requires a verifiable 'source_url' or 'dataset_id'.")
            raw_checksums = provenance.get("raw_file_checksums")
            if not raw_checksums or not isinstance(raw_checksums, dict):
                raise ValueError("Verified provenance requires non-empty 'raw_file_checksums' dictionary.")
            agg_method = provenance.get("aggregation_method")
            if not agg_method or not str(agg_method).strip():
                raise ValueError("Verified provenance requires an explicit 'aggregation_method' description.")

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_zones(self, zones_path: Path) -> pd.DataFrame:
        """Loads and validates zone mapping table."""
        try:
            df = pd.read_csv(zones_path)
        except Exception as e:
            raise ValueError(f"Unable to read CSV at {zones_path}: {e}") from e

        required_cols = {"source_zone_id", "model_zone_id", "weight"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"zones.csv must contain columns {required_cols}, found {list(df.columns)}")

        # Validate model_zone_id values
        invalid_model_zones = set(df["model_zone_id"].unique()) - self.target_zones
        if invalid_model_zones:
            raise ValueError(f"zones.csv contains unknown model zone IDs: {invalid_model_zones}")

        # Validate weights
        if (df["weight"] <= 0).any():
            raise ValueError("All weights in zones.csv must be strictly positive (> 0.0)")

        # Normalize weights per source_zone_id
        weight_sums = df.groupby("source_zone_id")["weight"].transform("sum")
        df["normalized_weight"] = df["weight"] / weight_sums

        # Check model zone coverage
        covered_model_zones = set(df["model_zone_id"].unique())
        missing_model_zones = self.target_zones - covered_model_zones
        if missing_model_zones:
            logger.warning("zones.csv does not map to all 12 model zones; missing: %s", missing_model_zones)

        return df

    def _load_and_process_activity(
        self, activity_path: Path, meta: ExternalEventMeta, zones_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, float, np.ndarray, pd.DataFrame]:
        """Loads activity observations, verifies hourly continuity, and maps to model zones."""
        try:
            df = pd.read_csv(activity_path)
        except Exception as e:
            raise ValueError(f"Unable to read CSV at {activity_path}: {e}") from e

        required_cols = {"timestamp", "zone_id", "activity_metric", "metric_type"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"activity.csv must contain columns {required_cols}, found {list(df.columns)}")

        if (df["activity_metric"] < 0).any():
            raise ValueError("activity_metric in activity.csv contains negative values")

        # Parse timestamps and verify timezone awareness
        try:
            df["dt"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            raise ValueError(f"Failed to parse timestamps in activity.csv: {e}") from e

        if df["dt"].dt.tz is None:
            raise ValueError("Timestamps in activity.csv must include timezone offset.")

        # Check timeline length and hourly continuity per zone
        zone_groups = df.groupby("zone_id")
        for zid, grp in zone_groups:
            sorted_grp = grp.sort_values("dt")
            dts = sorted_grp["dt"].values
            if len(dts) != meta.total_hours:
                raise ValueError(
                    f"Zone '{zid}' has {len(dts)} hourly records, expected total_hours={meta.total_hours}"
                )

            # Check continuous 1-hour increments with no duplicates
            time_diffs_sec = np.diff(dts.astype("int64")) // 10**9
            if not np.all(time_diffs_sec == 3600):
                invalid_indices = np.where(time_diffs_sec != 3600)[0]
                first_bad = invalid_indices[0]
                raise ValueError(
                    f"Timestamp sequence for zone '{zid}' is non-continuous or contains duplicate/gap at index "
                    f"{first_bad}: interval is {time_diffs_sec[first_bad]} seconds (expected 3600s)"
                )

        # Check mapping from activity zone_id to model zones
        known_source_zones = set(zones_df["source_zone_id"].unique())
        activity_zones = set(df["zone_id"].unique())

        # If activity zones are already model zones and directly in target_zones:
        direct_model_match = activity_zones.issubset(self.target_zones)
        if not direct_model_match:
            unmapped = activity_zones - known_source_zones
            if unmapped:
                raise ValueError(f"activity.csv contains zone_ids not defined in zones.csv: {unmapped}")

        # Map and aggregate activity to 12 model zones
        # Create hourly pivot table for each source zone
        pivot_source = df.pivot(index="dt", columns="zone_id", values="activity_metric").sort_index()

        # Build hours x 12 model zones matrix
        hourly_model_activity = pd.DataFrame(index=np.arange(meta.total_hours), columns=list(sorted(self.target_zones)), dtype=float)
        hourly_model_activity.fillna(0.0, inplace=True)

        if direct_model_match and len(activity_zones.intersection(self.target_zones)) == len(self.target_zones):
            # Direct mapping
            for zid in self.target_zones:
                hourly_model_activity[zid] = pivot_source[zid].values[: meta.total_hours]
        else:
            # Weighted aggregation through zones_df
            for _, row in zones_df.iterrows():
                src_z = row["source_zone_id"]
                mod_z = row["model_zone_id"]
                weight = row["normalized_weight"]
                if src_z in pivot_source.columns:
                    hourly_model_activity[mod_z] += pivot_source[src_z].values[: meta.total_hours] * weight

        # Total city activity per hour
        city_total_activity = hourly_model_activity.sum(axis=1).values

        # Baseline normalization using pre-event lead time window
        baseline_mean = compute_baseline_mean(
            city_total_activity,
            event_start_hour=meta.event_start_hour,
            lead_time_hours=meta.lead_time_hours,
        )
        if baseline_mean <= 0 or not np.isfinite(baseline_mean):
            raise ValueError(f"Computed baseline mean for observed activity is invalid: {baseline_mean}")

        normalized_city_activity = city_total_activity / baseline_mean

        # Per-zone normalization
        normalized_zone_activity = pd.DataFrame(index=hourly_model_activity.index, columns=hourly_model_activity.columns, dtype=float)
        for zid in self.target_zones:
            z_series = hourly_model_activity[zid].values
            z_base = compute_baseline_mean(
                z_series,
                event_start_hour=meta.event_start_hour,
                lead_time_hours=meta.lead_time_hours,
            )
            if z_base > 0 and np.isfinite(z_base):
                normalized_zone_activity[zid] = z_series / z_base
            else:
                normalized_zone_activity[zid] = z_series / (np.mean(z_series) + 1e-6)

        return (
            hourly_model_activity,
            city_total_activity,
            baseline_mean,
            normalized_city_activity,
            normalized_zone_activity,
        )

    def _load_impacts(self, impacts_path: Path, zones_df: pd.DataFrame) -> Dict[str, float]:
        """Loads and aggregates physical zone impact observations."""
        try:
            df = pd.read_csv(impacts_path)
        except Exception as e:
            raise ValueError(f"Unable to read CSV at {impacts_path}: {e}") from e

        if "zone_id" not in df.columns or "impact_score" not in df.columns:
            raise ValueError("impacts.csv must contain columns 'zone_id' and 'impact_score'")

        zone_impacts: Dict[str, float] = {zid: 0.0 for zid in self.target_zones}

        # If already model zones:
        impact_zones = set(df["zone_id"].unique())
        if impact_zones.issubset(self.target_zones):
            for _, row in df.iterrows():
                zid = str(row["zone_id"])
                if zid in zone_impacts:
                    zone_impacts[zid] = float(row["impact_score"])
        else:
            # Map through zones_df
            src_impacts = dict(zip(df["zone_id"], df["impact_score"]))
            for _, row in zones_df.iterrows():
                src_z = row["source_zone_id"]
                mod_z = row["model_zone_id"]
                weight = row["normalized_weight"]
                if src_z in src_impacts:
                    zone_impacts[mod_z] += float(src_impacts[src_z]) * weight

        return zone_impacts
