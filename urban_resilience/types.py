"""Core data types and schemas for the urban resilience multi-agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class HazardType(str, Enum):
    """Types of urban disturbance and hazard scenarios."""
    RAIN_STORM = "rainstorm"
    TYPHOON = "typhoon"
    EXTREME_HEAT = "extreme_heat"
    URBAN_FLOOD = "urban_flood"
    TRANSIT_FAILURE = "transit_failure"


@dataclass
class UrbanZone:
    """Represents a spatial unit (zone / grid cell / neighborhood) in a city."""
    zone_id: str
    name: str
    latitude: float
    longitude: float
    area_sqkm: float
    population: float
    gdp_per_capita: float
    poi_counts: Dict[str, int] = field(default_factory=dict)
    land_use_mix: Dict[str, float] = field(default_factory=dict)
    road_density_km_per_sqkm: float = 0.0
    transit_stop_count: int = 0
    elevation_m: float = 10.0
    drainage_capacity: Optional[float] = None


@dataclass
class UrbanContext:
    """Static and structural urban baseline context (roads, transit, POIs, socioeconomics)."""
    city_name: str
    zones: Dict[str, UrbanZone]
    road_adjacency: Dict[str, List[str]] = field(default_factory=dict)
    transit_lines: List[Dict[str, Any]] = field(default_factory=list)
    zone_distance_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def zone_ids(self) -> List[str]:
        return list(self.zones.keys())


@dataclass
class MobilityNetwork:
    """Dynamic historical mobility data across time slices and temporal rhythms."""
    od_matrices: Dict[int, pd.DataFrame]  # time_step -> OD matrix (origin x destination)
    daily_profiles: pd.DataFrame  # hour of day (0..23) -> volume per zone
    weekly_profiles: pd.DataFrame  # day of week (0..6) -> relative mobility
    hourly_time_series: pd.DataFrame  # timestamp/hour_index -> total volume per zone
    time_labels: List[str] = field(default_factory=list)


@dataclass
class EventContext:
    """Anticipatory event intelligence available strictly *before* the disaster hits."""
    event_id: str
    hazard_type: HazardType
    event_start_hour: int
    duration_hours: int
    peak_intensity: float  # e.g., mm/h precipitation or anomaly severity in [0, 1]
    affected_zones: List[str]
    zone_intensity_map: Dict[str, float]  # zone_id -> hazard intensity factor [0, 1]
    hourly_forecast: np.ndarray  # predicted intensity curve over simulation window
    description: str = ""
    source: str = "Pre-event Weather Radar & Municipal Early Warning"


@dataclass
class ReActTrace:
    """Record of a single ReAct step: Reasoning -> Action -> Observation."""
    step_number: int
    thought: str
    action_type: str
    action_input: Dict[str, Any]
    observation: Any


@dataclass
class AgentEvidence:
    """Standard evidence package produced by a multi-perspective agent."""
    agent_name: str
    metrics: Dict[str, float]  # global or aggregated metrics
    zone_metrics: Dict[str, Dict[str, float]]  # zone_id -> {metric_name: value}
    traces: List[ReActTrace]
    confidence: float
    qualitative_inference: str
    potential_anomalies: List[str] = field(default_factory=list)


@dataclass
class DisruptionMetrics:
    """Quantitative performance metrics characterizing the disruption and recovery curve."""
    maximum_activity_loss: float  # Max relative drop from baseline: max((A0 - A_pred) / max(A0, 1e-9)), clipped to [0, 1]
    cumulative_activity_loss: float  # Total area between baseline and predicted activity (normalized activity units * hours; differs from relative loss ratio)
    recovery_time_hours: float  # Hours from event start until activity returns to >= recovery_threshold_ratio of baseline (or window truncation)
    peak_loss_hour: int  # Simulation hour when maximum relative loss occurs
    resilience_triangle_area: float  # Integral of activity loss


@dataclass
class PredictionProvenance:
    """Provenance and lineage tracking for resilience predictions."""
    summary_metric_source: str  # "trained_model" | "monte_carlo_physics"
    curve_source: str = "monte_carlo_physics"
    curve_metric_source: str = "monte_carlo_physics"
    external_temporal_metric_source: str = "monte_carlo_physics"
    metric_sources: Dict[str, str] = field(default_factory=lambda: {
        "maximum_activity_loss": "monte_carlo_physics",
        "recovery_time_hours": "monte_carlo_physics",
        "cumulative_activity_loss": "monte_carlo_physics",
        "peak_loss_hour": "monte_carlo_physics",
        "resilience_triangle_area": "monte_carlo_physics",
    })
    model_status: str = "untrained_fallback"
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    artifact_hash: Optional[str] = None
    training_label_source: Optional[str] = None
    engineering_only: bool = False
    training_event_ids: List[str] = field(default_factory=list)
    interval_source: str = "monte_carlo_quantile"  # "conformal" | "uncalibrated" | "monte_carlo_quantile"
    predictor_role: str = "engineering_baseline"
    accuracy_claim: str = "unverified_on_real_data"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_metric_source": self.summary_metric_source,
            "curve_source": self.curve_source,
            "curve_metric_source": self.curve_metric_source,
            "external_temporal_metric_source": self.external_temporal_metric_source,
            "metric_sources": dict(self.metric_sources),
            "model_status": self.model_status,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "artifact_hash": self.artifact_hash,
            "training_label_source": self.training_label_source,
            "engineering_only": self.engineering_only,
            "training_event_ids": list(self.training_event_ids),
            "interval_source": self.interval_source,
            "predictor_role": self.predictor_role,
            "accuracy_claim": self.accuracy_claim,
            "notes": list(self.notes),
        }


@dataclass
class ResiliencePrediction:
    """Anticipatory disruption and recovery trajectory across time."""
    time_hours: np.ndarray
    baseline_activity: np.ndarray
    predicted_activity: np.ndarray
    lower_bound_95: np.ndarray
    upper_bound_95: np.ndarray
    disruption_metrics: DisruptionMetrics
    zone_predictions: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    provenance: PredictionProvenance = field(
        default_factory=lambda: PredictionProvenance(summary_metric_source="monte_carlo_physics")
    )


@dataclass
class SpatialZoneResilience:
    """Spatial resilience diagnostics for an individual zone."""
    zone_id: str
    zone_name: str
    vulnerability_score: float  # [0, 1], higher means more vulnerable
    vulnerability_rank: int  # 1 = most vulnerable (highest intervention priority)
    composite_resilience_index: float  # [0, 1], higher means more resilient
    primary_vulnerability_driver: str
    key_evidence: List[str]
    confidence: float


@dataclass
class SpatialResilienceOutlook:
    """Comprehensive spatial vulnerability assessment and priority ranking."""
    zone_rankings: List[SpatialZoneResilience]
    vulnerable_areas: List[str]
    priority_ranking: List[str]
    spatial_summary: str
    zone_features: pd.DataFrame


@dataclass
class BaselineComparison:
    """Metrics comparing the multi-agent system against baseline benchmarks."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    r2_score: float


@dataclass
class CalibrationMetrics:
    """Uncertainty calibration metrics."""
    picp_95: float  # Prediction Interval Coverage Probability (nominal: 0.95)
    mpiw: float  # Mean Prediction Interval Width
    interval_score: float  # Winkler interval score
    ece: float  # Expected Calibration Error


@dataclass
class ValidationResult:
    """Validation report including post-event ground truth comparisons and ablations."""
    eval_metrics: Dict[str, float]
    baseline_comparisons: List[BaselineComparison]
    ablation_results: Dict[str, Dict[str, float]]  # Ablated agent -> {metric: delta}
    calibration: CalibrationMetrics
