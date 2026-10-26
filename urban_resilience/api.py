"""RESTful Web API for Urban Resilience Multi-Agent Anticipatory System.

Built with FastAPI, enabling cloud service deployment, digital-twin dashboard integration,
and real-time event simulation requests.

Note:
    Simulations run as synchronous CPU-intensive tasks. Each simulation creates an isolated
    directory under outputs/api_runs/<run_id>/ to prevent concurrency collisions.
    Production deployments should integrate authentication, rate limiting, and asynchronous
    task worker queues (e.g., Celery/Redis).
"""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import hmac
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from urban_resilience import __version__
from urban_resilience.config import ProjectPaths, SimulationConfig
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.llm.client import LLMClient, get_llm_client
from urban_resilience.orchestration import OrchestrationError
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.prediction.provenance import (
    get_provenance_warning,
    get_synthetic_warning,
)
from urban_resilience.types import HazardType

logger = logging.getLogger("UrbanResilienceAPI")

app = FastAPI(
    title="Urban Resilience Anticipatory Multi-Agent API",
    version=__version__,
    description="Cognitive multi-agent architecture for anticipatory urban resilience prediction and emergency dispatch.",
)

# In-memory cached spatial context
_spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
_urban_context = _spatial_loader.load_context()

class BoundedRunRegistry:
    """Thread-safe bounded in-memory registry for simulation runs with TTL eviction."""

    def __init__(self, max_entries: Optional[int] = None, ttl_seconds: Optional[float] = None):
        if max_entries is None:
            raw_max = os.environ.get("URBAN_REGISTRY_MAX_ENTRIES")
            try:
                max_entries = int(raw_max) if raw_max is not None else 100
                if max_entries < 1:
                    raise ValueError("URBAN_REGISTRY_MAX_ENTRIES must be >= 1")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid URBAN_REGISTRY_MAX_ENTRIES '%s': %s. Falling back to 100.",
                    raw_max,
                    exc,
                )
                max_entries = 100
        else:
            try:
                max_entries = int(max_entries)
                if max_entries < 1:
                    raise ValueError("max_entries must be >= 1")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid max_entries argument '%s': %s. Falling back to 100.",
                    max_entries,
                    exc,
                )
                max_entries = 100

        if ttl_seconds is None:
            raw_ttl = os.environ.get("URBAN_REGISTRY_TTL_SECONDS")
            try:
                ttl_seconds = float(raw_ttl) if raw_ttl is not None else 86400.0  # 24h default
                if ttl_seconds <= 0.0 or math.isnan(ttl_seconds) or math.isinf(ttl_seconds):
                    raise ValueError("URBAN_REGISTRY_TTL_SECONDS must be > 0 and finite")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid URBAN_REGISTRY_TTL_SECONDS '%s': %s. Falling back to 86400.0.",
                    raw_ttl,
                    exc,
                )
                ttl_seconds = 86400.0
        else:
            try:
                ttl_seconds = float(ttl_seconds)
                if ttl_seconds <= 0.0 or math.isnan(ttl_seconds) or math.isinf(ttl_seconds):
                    raise ValueError("ttl_seconds must be > 0 and finite")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid ttl_seconds argument '%s': %s. Falling back to 86400.0.",
                    ttl_seconds,
                    exc,
                )
                ttl_seconds = 86400.0

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, Dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def set(self, run_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            self._purge_expired_locked()
            if run_id in self._entries:
                del self._entries[run_id]
            elif len(self._entries) >= self.max_entries:
                # Evict oldest entry (FIFO) without deleting disk artifacts
                if self._entries:
                    self._entries.popitem(last=False)
            self._entries[run_id] = (time.time(), record)

    def get(self, run_id: str, default: Any = None) -> Any:
        with self._lock:
            if run_id not in self._entries:
                return default
            ts, record = self._entries[run_id]
            if time.time() - ts > self.ttl_seconds:
                del self._entries[run_id]
                return default
            return record

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired_keys = [k for k, (ts, _) in self._entries.items() if now - ts > self.ttl_seconds]
        for k in expired_keys:
            del self._entries[k]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    def __contains__(self, run_id: str) -> bool:
        return self.get(run_id) is not None

    def __getitem__(self, run_id: str) -> Dict[str, Any]:
        val = self.get(run_id)
        if val is None:
            raise KeyError(run_id)
        return val

    def __setitem__(self, run_id: str, record: Dict[str, Any]) -> None:
        self.set(run_id, record)

    def keys(self) -> List[str]:
        with self._lock:
            self._purge_expired_locked()
            return list(self._entries.keys())

    def values(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._purge_expired_locked()
            return [rec for _, rec in self._entries.values()]

    def items(self) -> List[tuple[str, Dict[str, Any]]]:
        with self._lock:
            self._purge_expired_locked()
            return [(k, rec) for k, (_, rec) in self._entries.items()]


class InMemoryRateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, default_limit: int = 60):
        self.default_limit = default_limit
        self._history: Dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def get_limit(self) -> int:
        val = os.environ.get("URBAN_RATE_LIMIT_PER_MINUTE")
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
        return self.default_limit

    def check(self, client_id: str) -> tuple[bool, int]:
        """Returns (is_allowed, retry_after_seconds)."""
        limit = self.get_limit()
        now = time.time()
        window = 60.0

        with self._lock:
            if client_id not in self._history:
                self._history[client_id] = deque()
            q = self._history[client_id]

            # Evict entries older than 60s
            while q and q[0] <= now - window:
                q.popleft()

            if len(q) >= limit:
                oldest = q[0]
                retry_after = max(1, math.ceil(window - (now - oldest)))
                return False, retry_after

            q.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


# Thread-safe in-memory run registry, rate limiter, and execution lock
_api_lock = threading.Lock()
_runs_registry = BoundedRunRegistry()
_rate_limiter = InMemoryRateLimiter()
_latest_run_id: Optional[str] = None


def reset_api_state(max_entries: Optional[int] = None, ttl_seconds: Optional[float] = None) -> None:
    """Resets in-memory registry and rate-limiting history for clean test isolation."""
    global _runs_registry, _latest_run_id, _injected_llm_client
    with _api_lock:
        _runs_registry = BoundedRunRegistry(max_entries=max_entries, ttl_seconds=ttl_seconds)
        _latest_run_id = None
        _rate_limiter.reset()
        _injected_llm_client = None


_injected_llm_client: Optional[LLMClient] = None


def set_injected_llm_client(client: Optional[LLMClient]) -> None:
    """Allows testing harnesses to inject mock LLMClient instances into API executions."""
    global _injected_llm_client
    _injected_llm_client = client


def resolve_llm_client_for_run(orchestration_mode: str) -> Optional[LLMClient]:
    """Resolves and validates an LLM client for the requested orchestration mode.

    - Under 'deterministic': returns None.
    - Under 'llm_hybrid':
      1. Returns injected client if set via set_injected_llm_client().
      2. If URBAN_LLM_MODE is 'mock' or 'test', returns MockLLMClient.
      3. If URBAN_LLM_API_KEY is present, returns EnvironmentConfiguredLLMClient.
      4. Otherwise raises HTTPException(422) indicating missing API key/provider.
    """
    if orchestration_mode != "llm_hybrid":
        return None

    if _injected_llm_client is not None:
        return _injected_llm_client

    mode_env = os.environ.get("URBAN_LLM_MODE", "").strip().lower()
    if mode_env in ("mock", "test"):
        return get_llm_client(mode="mock")

    api_key = os.environ.get("URBAN_LLM_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=422,
            detail="URBAN_LLM_API_KEY environment variable is required when orchestration_mode is 'llm_hybrid'.",
        )

    return get_llm_client(mode="env")


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    # 1. Rate Limiting Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, retry_after = _rate_limiter.check(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Rate limit exceeded. Try again in {retry_after} seconds."},
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Optional API Key Authentication
    # Read dynamically so monkeypatching URBAN_API_KEY takes immediate effect
    configured_key = os.environ.get("URBAN_API_KEY", "").strip()
    if configured_key:
        path = request.url.path
        if path.startswith("/api/v1/resilience/"):
            client_key = request.headers.get("x-api-key")
            if not client_key or not hmac.compare_digest(
                client_key.encode("utf-8"), configured_key.encode("utf-8")
            ):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing API key. Provide a valid 'X-API-Key' header."},
                )

    return await call_next(request)


class SimulationRequest(BaseModel):
    """Payload for triggering an anticipatory resilience simulation."""
    city_name: str = Field(default="Shenzhen", description="Name of target city")
    hazard_type: HazardType = Field(default=HazardType.RAIN_STORM, description="Hazard scenario type")
    event_start_hour: int = Field(default=24, ge=1, le=168, description="Hour when the hazard lands")
    duration_hours: int = Field(default=18, ge=1, le=72, description="Disturbance duration in hours")
    peak_intensity: float = Field(default=0.85, ge=0.1, le=1.0, description="Severity factor in [0.1, 1.0]")
    total_hours: int = Field(default=96, ge=24, le=168, description="Total simulation window")
    monte_carlo_samples: int = Field(default=100, ge=1, le=500, description="Monte Carlo draws")
    orchestration_mode: str = Field(default="deterministic", description="Supervisor orchestration mode: 'deterministic' or 'llm_hybrid'")

    @model_validator(mode="after")
    def validate_simulation_window_and_city(self) -> SimulationRequest:
        if self.city_name.strip().lower() != "shenzhen":
            raise ValueError("Only 'Shenzhen' is supported as the target city in the current release.")
        if self.orchestration_mode not in ("deterministic", "llm_hybrid"):
            raise ValueError("orchestration_mode must be 'deterministic' or 'llm_hybrid'")
        if self.event_start_hour < 1:
            raise ValueError("event_start_hour must be >= 1")
        if self.duration_hours < 1:
            raise ValueError("duration_hours must be >= 1")
        if self.total_hours < 24:
            raise ValueError("total_hours must be >= 24")
        if self.event_start_hour + self.duration_hours > self.total_hours:
            raise ValueError(
                f"event_start_hour + duration_hours ({self.event_start_hour + self.duration_hours}) "
                f"exceeds total_hours ({self.total_hours})"
            )
        if not (0.1 <= self.peak_intensity <= 1.0):
            raise ValueError("peak_intensity must be in [0.1, 1.0]")
        if self.monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be >= 1")
        return self


class ZoneResponse(BaseModel):
    """Metadata of an urban spatial zone."""
    zone_id: str
    name: str
    latitude: float
    longitude: float
    area_sqkm: float
    population: float
    elevation_m: float
    road_density: float
    poi_counts: Dict[str, int]
    drainage_capacity: Optional[float] = None


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming API readiness."""
    configured_key = os.environ.get("URBAN_API_KEY", "").strip()
    return {
        "status": "healthy",
        "service": "Urban Resilience Multi-Agent Anticipatory System",
        "version": __version__,
        "ready": True,
        "auth_mode": "authenticated" if configured_key else "development_open",
        "rate_limit_per_minute": _rate_limiter.get_limit(),
        "registry_entries": len(_runs_registry),
    }


@app.get("/api/v1/city/zones", response_model=List[ZoneResponse], tags=["Urban Context"])
def get_urban_zones():
    """Returns spatial context and infrastructure profile for all city zones."""
    records = []
    for zid, z in _urban_context.zones.items():
        records.append(
            ZoneResponse(
                zone_id=z.zone_id,
                name=z.name,
                latitude=z.latitude,
                longitude=z.longitude,
                area_sqkm=z.area_sqkm,
                population=z.population,
                elevation_m=z.elevation_m,
                road_density=z.road_density_km_per_sqkm,
                poi_counts=z.poi_counts,
                drainage_capacity=getattr(z, "drainage_capacity", None),
            )
        )
    return records


@app.post("/api/v1/resilience/simulate", tags=["Resilience Simulation"])
def run_simulation(req: SimulationRequest):
    """Triggers an end-to-end multi-agent resilience simulation under the given hazard scenario.

    Creates an isolated execution directory under outputs/api_runs/<run_id>/ to prevent
    file overwrites across concurrent or successive requests.
    """
    sim_cfg = SimulationConfig(
        total_hours=req.total_hours,
        event_start_hour=req.event_start_hour,
        event_duration_hours=req.duration_hours,
        monte_carlo_samples=req.monte_carlo_samples,
    )

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = ProjectPaths.OUTPUTS_DIR / "api_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = run_dir / "manifest.json"

    llm_client = resolve_llm_client_for_run(req.orchestration_mode)

    pipeline = UrbanResiliencePipeline(
        city_name=req.city_name,
        sim_config=sim_cfg,
        output_dir=run_dir,
        orchestration_mode=req.orchestration_mode,
        llm_client=llm_client,
    )

    try:
        results = pipeline.run_full_pipeline(
            hazard_type=req.hazard_type,
            event_start_hour=req.event_start_hour,
            duration_hours=req.duration_hours,
            peak_intensity=req.peak_intensity,
            output_manifest_path=run_manifest_path,
        )
    except OrchestrationError as exc:
        logger.error("Simulation failed due to orchestration error: %s", exc)
        detail = str(exc)
        if hasattr(exc, "missing_domains") and exc.missing_domains:
            detail = f"{detail} (missing domains: {exc.missing_domains})"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc

    outlook = results["spatial_outlook"]
    metrics = results["prediction"].disruption_metrics

    # Format response payload
    rankings = [
        {
            "rank": z.vulnerability_rank,
            "zone_id": z.zone_id,
            "zone_name": z.zone_name,
            "vulnerability_score": round(z.vulnerability_score, 3),
            "composite_resilience": round(z.composite_resilience_index, 3),
            "primary_driver": z.primary_vulnerability_driver,
        }
        for z in outlook.zone_rankings
    ]

    plot_paths = results["plot_paths"]
    plot_urls = {
        "curve": f"/api/v1/resilience/runs/{run_id}/plots/curve",
        "map": f"/api/v1/resilience/runs/{run_id}/plots/map",
        "dashboard": f"/api/v1/resilience/runs/{run_id}/plots/dashboard",
    }

    response_payload = {
        "run_id": run_id,
        "status": "success",
        "city": req.city_name,
        "hazard_type": req.hazard_type.value,
        "orchestration_mode": req.orchestration_mode,
        "stop_reason": getattr(results.get("manifest"), "stop_reason", None),
        "indicators": results["key_indicators"],
        "disruption_metrics": {
            "maximum_activity_loss": round(metrics.maximum_activity_loss, 4),
            "cumulative_activity_loss": round(metrics.cumulative_activity_loss, 2),
            "recovery_time_hours": round(metrics.recovery_time_hours, 1),
            "peak_loss_hour": metrics.peak_loss_hour,
        },
        "spatial_rankings": rankings,
        "vulnerable_areas": outlook.vulnerable_areas,
        "metric_sources": (
            dict(results["prediction"].provenance.metric_sources)
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance and results["prediction"].provenance.metric_sources
            else {
                "maximum_activity_loss": "monte_carlo_physics",
                "recovery_time_hours": "monte_carlo_physics",
                "cumulative_activity_loss": "monte_carlo_physics",
                "peak_loss_hour": "monte_carlo_physics",
                "resilience_triangle_area": "monte_carlo_physics",
            }
        ),
        "prediction_provenance": (
            results["prediction"].provenance.to_dict()
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else {
                "summary_metric_source": "monte_carlo_physics",
                "curve_source": "monte_carlo_physics",
                "curve_metric_source": "monte_carlo_physics",
                "external_temporal_metric_source": "monte_carlo_physics",
                "metric_sources": {
                    "maximum_activity_loss": "monte_carlo_physics",
                    "recovery_time_hours": "monte_carlo_physics",
                    "cumulative_activity_loss": "monte_carlo_physics",
                    "peak_loss_hour": "monte_carlo_physics",
                    "resilience_triangle_area": "monte_carlo_physics",
                },
                "model_status": "untrained_fallback",
                "training_label_source": "none",
                "engineering_only": True,
                "training_event_ids": [],
                "interval_source": "monte_carlo_quantile",
            }
        ),
        "predictor_status": getattr(results.get("manifest"), "predictor_status", {}) or {},
        "predictor_role": (
            getattr(results["prediction"].provenance, "predictor_role", "engineering_baseline")
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else "engineering_baseline"
        ),
        "accuracy_claim": (
            getattr(results["prediction"].provenance, "accuracy_claim", "unverified_on_real_data")
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else "unverified_on_real_data"
        ),
        "training_label_source": (
            results["prediction"].provenance.training_label_source
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else "none"
        ),
        "engineering_only": (
            bool(results["prediction"].provenance.engineering_only)
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else True
        ),
        "training_event_ids": (
            list(results["prediction"].provenance.training_event_ids)
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else []
        ),
        "synthetic_warning": get_synthetic_warning(
            results["prediction"].provenance
            if hasattr(results.get("prediction"), "provenance")
            else None
        ),
        "provenance_warning": get_provenance_warning(
            results["prediction"].provenance
            if hasattr(results.get("prediction"), "provenance")
            else None
        ),
        "artifacts_dir": str(run_dir),
        "manifest_path": str(run_manifest_path),
        "timeseries_csv": str(run_dir / "disruption_recovery_timeseries.csv"),
        "spatial_rankings_csv": str(run_dir / "spatial_priority_rankings.csv"),
        "plots": plot_paths,
        "plot_urls": plot_urls,
    }

    # Thread-safe in-memory cache update
    global _latest_run_id
    _runs_registry.set(
        run_id,
        {
            **response_payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    with _api_lock:
        _latest_run_id = run_id

    return response_payload


def _execute_async_simulation(run_id: str, run_dir: Path, req: SimulationRequest):
    """Worker function executing simulation in background thread."""
    try:
        _runs_registry.set(
            run_id,
            {
                "run_id": run_id,
                "status": "running",
                "city": req.city_name,
                "hazard_type": req.hazard_type.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        sim_cfg = SimulationConfig(
            total_hours=req.total_hours,
            event_start_hour=req.event_start_hour,
            event_duration_hours=req.duration_hours,
            monte_carlo_samples=req.monte_carlo_samples,
        )
        run_manifest_path = run_dir / "manifest.json"
        llm_client = resolve_llm_client_for_run(req.orchestration_mode)

        pipeline = UrbanResiliencePipeline(
            city_name=req.city_name,
            sim_config=sim_cfg,
            output_dir=run_dir,
            orchestration_mode=req.orchestration_mode,
            llm_client=llm_client,
        )
        results = pipeline.run_full_pipeline(
            hazard_type=req.hazard_type,
            event_start_hour=req.event_start_hour,
            duration_hours=req.duration_hours,
            peak_intensity=req.peak_intensity,
            output_manifest_path=run_manifest_path,
        )
        outlook = results["spatial_outlook"]
        metrics = results["prediction"].disruption_metrics
        rankings = [
            {
                "rank": z.vulnerability_rank,
                "zone_id": z.zone_id,
                "zone_name": z.zone_name,
                "vulnerability_score": round(z.vulnerability_score, 3),
                "composite_resilience": round(z.composite_resilience_index, 3),
                "primary_driver": z.primary_vulnerability_driver,
            }
            for z in outlook.zone_rankings
        ]
        plot_paths = results["plot_paths"]
        plot_urls = {
            "curve": f"/api/v1/resilience/runs/{run_id}/plots/curve",
            "map": f"/api/v1/resilience/runs/{run_id}/plots/map",
            "dashboard": f"/api/v1/resilience/runs/{run_id}/plots/dashboard",
        }
        response_payload = {
            "run_id": run_id,
            "status": "success",
            "city": req.city_name,
            "hazard_type": req.hazard_type.value,
            "indicators": results["key_indicators"],
            "disruption_metrics": {
                "maximum_activity_loss": round(metrics.maximum_activity_loss, 4),
                "cumulative_activity_loss": round(metrics.cumulative_activity_loss, 2),
                "recovery_time_hours": round(metrics.recovery_time_hours, 1),
                "peak_loss_hour": metrics.peak_loss_hour,
            },
            "spatial_rankings": rankings,
            "vulnerable_areas": outlook.vulnerable_areas,
            "metric_sources": (
                dict(results["prediction"].provenance.metric_sources)
                if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance and results["prediction"].provenance.metric_sources
                else {
                    "maximum_activity_loss": "monte_carlo_physics",
                    "recovery_time_hours": "monte_carlo_physics",
                    "cumulative_activity_loss": "monte_carlo_physics",
                    "peak_loss_hour": "monte_carlo_physics",
                    "resilience_triangle_area": "monte_carlo_physics",
                }
            ),
            "prediction_provenance": (
                results["prediction"].provenance.to_dict()
                if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
                else {
                    "summary_metric_source": "monte_carlo_physics",
                    "curve_source": "monte_carlo_physics",
                    "curve_metric_source": "monte_carlo_physics",
                    "external_temporal_metric_source": "monte_carlo_physics",
                    "metric_sources": {
                        "maximum_activity_loss": "monte_carlo_physics",
                        "recovery_time_hours": "monte_carlo_physics",
                        "cumulative_activity_loss": "monte_carlo_physics",
                        "peak_loss_hour": "monte_carlo_physics",
                        "resilience_triangle_area": "monte_carlo_physics",
                    },
                    "model_status": "untrained_fallback",
                    "training_label_source": "none",
                    "engineering_only": True,
                    "training_event_ids": [],
                    "interval_source": "monte_carlo_quantile",
                }
            ),
            "predictor_status": getattr(results.get("manifest"), "predictor_status", {}) or {},
        "predictor_role": (
            getattr(results["prediction"].provenance, "predictor_role", "engineering_baseline")
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else "engineering_baseline"
        ),
        "accuracy_claim": (
            getattr(results["prediction"].provenance, "accuracy_claim", "unverified_on_real_data")
            if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
            else "unverified_on_real_data"
        ),
            "training_label_source": (
                results["prediction"].provenance.training_label_source
                if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
                else "none"
            ),
            "engineering_only": (
                bool(results["prediction"].provenance.engineering_only)
                if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
                else True
            ),
            "training_event_ids": (
                list(results["prediction"].provenance.training_event_ids)
                if hasattr(results.get("prediction"), "provenance") and results["prediction"].provenance
                else []
            ),
            "synthetic_warning": get_synthetic_warning(
                results["prediction"].provenance
                if hasattr(results.get("prediction"), "provenance")
                else None
            ),
            "provenance_warning": get_provenance_warning(
                results["prediction"].provenance
                if hasattr(results.get("prediction"), "provenance")
                else None
            ),
            "artifacts_dir": str(run_dir),
            "manifest_path": str(run_manifest_path),
            "timeseries_csv": str(run_dir / "disruption_recovery_timeseries.csv"),
            "spatial_rankings_csv": str(run_dir / "spatial_priority_rankings.csv"),
            "plots": plot_paths,
            "plot_urls": plot_urls,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _runs_registry.set(run_id, response_payload)
        global _latest_run_id
        with _api_lock:
            _latest_run_id = run_id
    except Exception as exc:
        logger.error("Async simulation run '%s' failed: %s", run_id, exc, exc_info=True)
        _runs_registry.set(
            run_id,
            {
                "run_id": run_id,
                "status": "failed",
                "city": req.city_name,
                "error": str(exc),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


@app.post("/api/v1/resilience/submit", status_code=status.HTTP_202_ACCEPTED, tags=["Resilience Simulation"])
def submit_simulation_async(req: SimulationRequest):
    """Submits an asynchronous resilience simulation task, returning immediately with run_id."""
    # Validate LLM availability synchronously before accepting async task
    resolve_llm_client_for_run(req.orchestration_mode)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = ProjectPaths.OUTPUTS_DIR / "api_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _runs_registry.set(
        run_id,
        {
            "run_id": run_id,
            "status": "pending",
            "city": req.city_name,
            "hazard_type": req.hazard_type.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    worker_thread = threading.Thread(
        target=_execute_async_simulation,
        args=(run_id, run_dir, req),
        daemon=True,
    )
    worker_thread.start()

    return {
        "run_id": run_id,
        "status": "pending",
        "status_url": f"/api/v1/resilience/runs/{run_id}",
        "message": "Simulation task submitted successfully and executing asynchronously.",
    }


@app.get("/api/v1/resilience/runs/{run_id}", tags=["Resilience Simulation"])
def get_simulation_run(run_id: str):
    """Retrieves execution record, status, and artifact paths for a specific simulation run."""
    run_record = _runs_registry.get(run_id)
    if run_record:
        return run_record

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Simulation run '{run_id}' not found.",
    )


@app.get("/api/v1/resilience/runs/{run_id}/plots/{plot_type}", tags=["Visualizations"])
def get_run_plot_image(run_id: str, plot_type: str):
    """Streams publication-quality plots from an isolated simulation run."""
    plot_filenames = {
        "curve": "predicted_disruption_recovery.png",
        "map": "spatial_resilience_outlook.png",
        "dashboard": "urban_resilience_dashboard.png",
    }
    if plot_type not in plot_filenames:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid plot type '{plot_type}'. Valid types: {list(plot_filenames.keys())}",
        )

    run_dir = ProjectPaths.OUTPUTS_DIR / "api_runs" / run_id
    file_path = run_dir / "plots" / plot_filenames[plot_type]
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_type}' for run '{run_id}' not found at {file_path}.",
        )
    return FileResponse(file_path, media_type="image/png")


@app.get("/api/v1/resilience/plots/{plot_type}", tags=["Visualizations"], deprecated=True)
def get_plot_image(plot_type: str):
    """[DEPRECATED] Streams generated plots from the latest simulation run.

    Deprecated: Please migrate to /api/v1/resilience/runs/{run_id}/plots/{plot_type}
    to retrieve plots from an isolated execution run.
    """
    plot_filenames = {
        "curve": "predicted_disruption_recovery.png",
        "map": "spatial_resilience_outlook.png",
        "dashboard": "urban_resilience_dashboard.png",
    }
    if plot_type not in plot_filenames:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid plot type '{plot_type}'. Valid types: {list(plot_filenames.keys())}",
        )

    with _api_lock:
        latest_id = _latest_run_id

    if latest_id:
        candidate = ProjectPaths.OUTPUTS_DIR / "api_runs" / latest_id / "plots" / plot_filenames[plot_type]
        if candidate.exists():
            return FileResponse(candidate, media_type="image/png")

    file_path = ProjectPaths.PLOTS_DIR / plot_filenames[plot_type]
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_type}' has not been generated yet. Please run a simulation first.",
        )
    return FileResponse(file_path, media_type="image/png")
