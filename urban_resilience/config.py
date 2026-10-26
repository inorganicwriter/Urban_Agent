"""Global configuration parameters, directory paths, and simulation settings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator


def compute_baseline_mean(
    series_or_totals: np.ndarray,
    event_start_hour: int,
    lead_time_hours: int = 12,
) -> float:
    """Computes the pre-event baseline activity mean over the lead time window.

    When lead_time_hours exceeds event_start_hour, the baseline window is automatically
    truncated to [0, event_start_hour] using all available pre-event hours.

    Raises:
        ValueError: If event_start_hour <= 0 or the observation window is empty.
    """
    if event_start_hour <= 0:
        raise ValueError(f"event_start_hour must be >= 1 to establish baseline, got {event_start_hour}")

    start_idx = max(0, event_start_hour - max(1, lead_time_hours))
    end_idx = event_start_hour
    window_data = series_or_totals[start_idx:end_idx]

    if len(window_data) == 0:
        raise ValueError("Baseline observation window is empty.")

    mean_val = float(np.mean(window_data))
    if np.isnan(mean_val) or mean_val <= 0:
        raise ValueError("Invalid non-positive or NaN baseline mean encountered.")
    return mean_val


class SimulationConfig(BaseModel):
    """Configuration for simulation time horizons, time steps, and Monte Carlo iterations."""
    total_hours: int = Field(default=96, description="Total simulation window in hours (e.g. 4 days)")
    event_lead_time_hours: int = Field(default=12, description="Pre-event baseline observation window")
    event_start_hour: int = Field(default=24, description="Hour when the shock/event occurs")
    event_duration_hours: int = Field(default=18, description="Duration of physical disturbance")
    peak_intensity: float = Field(default=0.85, description="Peak hazard shock intensity in [0.1, 1.0]")
    monte_carlo_samples: int = Field(default=200, description="Monte Carlo draws for uncertainty quantification")
    recovery_threshold_ratio: float = Field(default=0.95, description="Fraction of baseline to declare recovered")
    random_seed: int = Field(default=42, description="Random state for reproducibility")
    start_datetime: Optional[datetime] = Field(default=None, description="Real timestamp of simulation window start (index 0)")
    start_hour_of_day: int = Field(default=0, ge=0, le=23, description="Hour of day at index 0 (0-23)")
    start_day_of_week: int = Field(default=0, ge=0, le=6, description="Day of week at index 0 (0=Monday...6=Sunday)")

    @model_validator(mode="after")
    def validate_simulation_bounds(self) -> SimulationConfig:
        """Enforces cross-field consistency and validity bounds."""
        if self.start_datetime is not None:
            self.start_hour_of_day = self.start_datetime.hour
            self.start_day_of_week = self.start_datetime.weekday()
        if self.total_hours < 24:
            raise ValueError(f"total_hours must be at least 24, got {self.total_hours}")
        if self.event_start_hour < 1:
            raise ValueError(f"event_start_hour must be at least 1, got {self.event_start_hour}")
        if self.event_duration_hours < 1:
            raise ValueError(f"event_duration_hours must be at least 1, got {self.event_duration_hours}")
        if self.event_start_hour + self.event_duration_hours > self.total_hours:
            raise ValueError(
                f"Event window ({self.event_start_hour} + {self.event_duration_hours} = "
                f"{self.event_start_hour + self.event_duration_hours}) exceeds total_hours ({self.total_hours})"
            )
        if not (0.1 <= self.peak_intensity <= 1.0):
            raise ValueError(f"peak_intensity must be in [0.1, 1.0], got {self.peak_intensity}")
        if not (0.0 < self.recovery_threshold_ratio <= 1.0):
            raise ValueError(
                f"recovery_threshold_ratio must be in (0, 1], got {self.recovery_threshold_ratio}"
            )
        if self.event_lead_time_hours < 1:
            raise ValueError(f"event_lead_time_hours must be at least 1, got {self.event_lead_time_hours}")
        # Note: When event_lead_time_hours > event_start_hour, compute_baseline_mean
        # automatically truncates to all available pre-event hours [0, event_start_hour].
        if self.monte_carlo_samples < 1:
            raise ValueError(f"monte_carlo_samples must be at least 1, got {self.monte_carlo_samples}")
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {self.random_seed}")
        return self


class AgentWeightConfig(BaseModel):
    """Importance weights for synthesizing multi-perspective agent evidence."""
    temporal_redundancy: float = 0.20
    intraday_flexibility: float = 0.20
    predictability: float = 0.15
    spatial_redundancy: float = 0.25
    functional_mix: float = 0.20

    @model_validator(mode="after")
    def validate_weights(self) -> AgentWeightConfig:
        """Validates that weights are non-negative and sum to 1.0."""
        weights = [
            self.temporal_redundancy,
            self.intraday_flexibility,
            self.predictability,
            self.spatial_redundancy,
            self.functional_mix,
        ]
        for w in weights:
            if w < 0:
                raise ValueError(f"Weights must be non-negative, got {w}")
        total_w = sum(weights)
        if abs(total_w - 1.0) > 1e-5:
            raise ValueError(f"Weights must sum to 1.0 (tolerance 1e-5), got sum={total_w}")
        return self

    def as_dict(self) -> Dict[str, float]:
        return {
            "temporal_redundancy": self.temporal_redundancy,
            "intraday_flexibility": self.intraday_flexibility,
            "predictability": self.predictability,
            "spatial_redundancy": self.spatial_redundancy,
            "functional_mix": self.functional_mix,
        }


class ProjectPaths:
    """Project filesystem paths."""
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
    PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
    REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

    @classmethod
    def ensure_directories(cls):
        for p in [cls.DATA_DIR, cls.OUTPUTS_DIR, cls.PLOTS_DIR, cls.REPORTS_DIR]:
            p.mkdir(parents=True, exist_ok=True)


ProjectPaths.ensure_directories()
