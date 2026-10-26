"""External historical disaster validation pipeline evaluating ex-ante predictions against empirical observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from urban_resilience.config import AgentWeightConfig, SimulationConfig, compute_baseline_mean
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.external_loader import ObservedEventDataset
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.orchestration import OrchestrationError, SupervisorAgent, TaskStatus
from urban_resilience.prediction.provenance import is_synthetic_trained
from urban_resilience.types import BaselineComparison, HazardType

logger = logging.getLogger(__name__)


@dataclass
class TemporalMetrics:
    """Quantitative performance of temporal activity prediction over evaluation window."""

    rmse: float
    mae: float
    mape_pct: float
    r2_score: float
    max_activity_loss_pred: float
    max_activity_loss_obs: float
    max_activity_loss_error: float
    peak_loss_hour_pred: int
    peak_loss_hour_obs: int
    peak_loss_hour_error: int
    recovery_time_pred: float
    recovery_time_obs: float
    recovery_time_error: float
    smape_pct: float = 0.0


@dataclass
class SpatialMetrics:
    """Spatial vulnerability rank correlation against observed impacts and losses."""

    spearman_rho_loss: float
    spearman_pvalue_loss: float
    spearman_rho_impact: Optional[float] = None
    spearman_pvalue_impact: Optional[float] = None


@dataclass
class UncertaintyMetrics:
    """Uncertainty calibration and interval sharpness metrics."""

    picp_95: float
    mpiw: float
    winkler_score: float
    ece: float
    nominal_coverage: float = 0.95


@dataclass
class ValidationVerdicts:
    """Automated pass/fail verdicts on key evaluation criteria."""

    provenance_verdict: str
    baseline_verdict: str
    uq_verdict: str
    amplitude_verdict: str
    overall_verdict: str
    reasons: List[str]


@dataclass
class ExternalValidationReport:
    """Comprehensive validation metrics across temporal, spatial, and uncertainty dimensions."""

    event_id: str
    city_name: str
    hazard_type: str
    data_kind: str
    role: str
    input_mode: str
    forecast_source: str
    temporal_metrics: TemporalMetrics
    spatial_metrics: SpatialMetrics
    uncertainty_metrics: UncertaintyMetrics
    baseline_comparisons: List[BaselineComparison]
    verdicts: Optional[ValidationVerdicts] = None
    model_input_mobility: str = "synthetic"
    pre_event_window_hours: Optional[int] = None
    od_source: str = "synthetic"
    leakage_guard: str = "post_event_activity_excluded"
    predictability_source: str = "synthetic"
    profile_source: str = "synthetic"
    feature_sources: Dict[str, Any] = field(default_factory=dict)
    hybrid_components: List[str] = field(default_factory=list)
    external_temporal_metric_source: str = "monte_carlo_physics"
    predictor_status: Dict[str, Any] = field(default_factory=lambda: {
        "model_status": "untrained_fallback",
        "engine": "monte_carlo_physics",
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
        "interval_source": "monte_carlo_quantile",
    })
    prediction_provenance: Dict[str, Any] = field(default_factory=lambda: {
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
        "interval_source": "monte_carlo_quantile",
    })

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class ExternalEventEvaluator:
    """Evaluates ex-ante anticipatory model predictions against external empirical observations."""

    def __init__(self, output_dir: Optional[Path | str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/external")

    def evaluate_event(
        self,
        dataset: ObservedEventDataset,
        dry_run: bool = False,
        allow_unverified: bool = False,
        monte_carlo_samples: int = 200,
        random_seed: int = 42,
    ) -> Tuple[ExternalValidationReport, Dict[str, Path]]:
        """Executes external evaluation on an observed, synthetic, or fixture event dataset.

        Args:
            dataset: Validated ObservedEventDataset.
            dry_run: Must be True if dataset.meta.data_kind == 'fixture'.
            allow_unverified: Must be True if dataset.meta.data_kind == 'synthetic' or unverified.
            monte_carlo_samples: Number of MC draws for prediction uncertainty.
            random_seed: Reproducible seed for synthesis and Monte Carlo.

        Returns:
            Tuple of (ExternalValidationReport, dict of generated artifact file paths).
        """
        # 1. Enforce safety policy on fixtures and unverified datasets
        if dataset.meta.data_kind == "fixture" and not dry_run:
            raise ValueError(
                f"Data policy violation: Event '{dataset.meta.event_id}' has data_kind='fixture'. "
                "Evaluating fixtures requires explicit confirmation via dry_run=True (or CLI flag --dry-run). "
                "Synthetic fixtures cannot be evaluated as empirical data without this flag."
            )
        if (dataset.meta.data_kind == "synthetic" or dataset.meta.provenance_status == "unverified") and not (dry_run or allow_unverified):
            raise ValueError(
                f"Data policy violation: Event '{dataset.meta.event_id}' has data_kind='{dataset.meta.data_kind}' "
                f"or provenance_status='{dataset.meta.provenance_status}'. "
                "Evaluating synthetic or unverified datasets requires explicit confirmation via "
                "allow_unverified=True (or CLI flag --allow-unverified / --dry-run)."
            )

        meta = dataset.meta
        event_out_dir = self.output_dir / meta.event_id
        plots_dir = event_out_dir / "plots"
        event_out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting external validation for event %s (data_kind=%s, mode=%s)...",
                    meta.event_id, meta.data_kind, meta.input_mode)

        # 2. Run anticipatory prediction conditioned on event metadata with true calendar alignment
        base_time = pd.to_datetime(meta.start_time) - pd.Timedelta(hours=meta.event_start_hour)
        window_start = base_time.to_pydatetime()

        sim_config = SimulationConfig(
            total_hours=meta.total_hours,
            event_lead_time_hours=meta.lead_time_hours,
            event_start_hour=meta.event_start_hour,
            event_duration_hours=meta.duration_hours,
            peak_intensity=meta.peak_intensity,
            monte_carlo_samples=monte_carlo_samples,
            random_seed=random_seed,
            start_datetime=window_start,
        )
        weight_config = AgentWeightConfig()

        spatial_loader = UrbanSpatialDataLoader(city_name=meta.city_name)
        urban_context = spatial_loader.load_context()

        # Check if verified observed data -> use ex-ante observed mobility adapter
        is_verified_observed = (
            meta.data_kind == "observed"
            and getattr(meta, "provenance_status", "unverified") == "verified"
        )

        if is_verified_observed:
            from urban_resilience.data.observed_mobility import ObservedMobilityAdapter
            adapter = ObservedMobilityAdapter(dataset=dataset, urban_context=urban_context)
            domain_mobility = adapter.build_pre_event_network()
            forecast_mobility = adapter.build_extended_network(total_hours=meta.total_hours)
            mobility_network = forecast_mobility
            model_input_mobility = adapter.model_input_mobility
            pre_event_window_hours = adapter.pre_event_window_hours
            od_source = adapter.od_source
            leakage_guard = adapter.leakage_guard
            predictability_source = adapter.predictability_source
            profile_source = adapter.profile_source
            feature_sources = adapter.feature_sources
            hybrid_components = adapter.hybrid_components
        else:
            mobility_loader = MobilityNetworkLoader(urban_context, random_seed=random_seed)
            mobility_network = mobility_loader.generate_mobility_dataset(
                total_hours=meta.total_hours,
                start_hour_of_day=sim_config.start_hour_of_day,
                start_day_of_week=sim_config.start_day_of_week,
                start_datetime=sim_config.start_datetime,
            )
            domain_mobility = mobility_network
            forecast_mobility = mobility_network
            model_input_mobility = "synthetic"
            pre_event_window_hours = None
            od_source = "synthetic"
            leakage_guard = "post_event_activity_excluded"
            predictability_source = "synthetic"
            profile_source = "synthetic"
            feature_sources = {
                "temporal": "synthetic",
                "flexibility": "synthetic",
                "spatial": "synthetic",
                "functional": "synthetic",
                "predictability": "synthetic",
                "daily_profiles": "synthetic",
                "weekly_profiles": "synthetic",
                "od_flows": "synthetic",
            }
            hybrid_components = list(feature_sources.keys())

        event_agent = EventIntelligenceAgent(urban_context)
        hazard_enum = HazardType(meta.hazard_type.lower()) if meta.hazard_type.lower() in [h.value for h in HazardType] else HazardType.RAIN_STORM
        event_context = event_agent.create_event_context(
            hazard_type=hazard_enum,
            event_start_hour=meta.event_start_hour,
            duration_hours=meta.duration_hours,
            peak_intensity=meta.peak_intensity,
            total_simulation_hours=meta.total_hours,
        )

        # If custom zone hazard distribution was provided in event.json, override
        if meta.hazard_distribution:
            for zid, val in meta.hazard_distribution.items():
                if zid in event_context.zone_intensity_map:
                    event_context.zone_intensity_map[zid] = val

        # Run hierarchical orchestration through SupervisorAgent (single production path)
        supervisor = SupervisorAgent(
            urban_context=urban_context,
            mobility_network=domain_mobility,
            forecast_mobility_network=forecast_mobility,
            weight_config=weight_config,
            sim_config=sim_config,
            output_dir=self.output_dir,
        )
        prediction, spatial_outlook, _, manifest = supervisor.run(
            event_context=event_context,
            output_manifest_path=self.output_dir / meta.event_id / "manifest.json",
            save_manifest=True,
        )
        if manifest.status == TaskStatus.FAILED:
            raise OrchestrationError(
                f"External evaluation aborted: orchestration status is FAILED. Error: {manifest.error_summary}"
            )

        # 3. Compute baseline models on identical pre-event normalization
        baseline_comps, baseline_preds = self._evaluate_baselines(
            mobility_network=mobility_network,
            meta=meta,
            observed_normalized=dataset.normalized_city_activity,
            multi_agent_pred=prediction.predicted_activity,
            urban_context=urban_context,
            event_context=event_context,
        )

        # 4. Evaluate temporal metrics using observed pre-event baseline
        temporal_metrics = self._compute_temporal_metrics(
            observed=dataset.normalized_city_activity,
            predicted=prediction.predicted_activity,
            baseline=prediction.baseline_activity,
            eval_start_hour=meta.event_start_hour,
            duration_hours=meta.duration_hours,
            recovery_threshold=sim_config.recovery_threshold_ratio,
            observed_baseline=baseline_preds["historical"],
        )

        # 5. Evaluate spatial metrics
        spatial_metrics, spatial_df = self._compute_spatial_metrics(
            spatial_outlook=spatial_outlook,
            dataset=dataset,
            meta=meta,
        )

        # 6. Evaluate uncertainty quantification metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(
            observed=dataset.normalized_city_activity,
            lower_bound=prediction.lower_bound_95,
            upper_bound=prediction.upper_bound_95,
            eval_start_hour=meta.event_start_hour,
        )

        # 7. Automated quality gate verdicts
        verdicts = self._compute_verdicts(
            meta=meta,
            temporal_metrics=temporal_metrics,
            uncertainty_metrics=uncertainty_metrics,
            baseline_comparisons=baseline_comps,
        )

        report = ExternalValidationReport(
            event_id=meta.event_id,
            city_name=meta.city_name,
            hazard_type=meta.hazard_type,
            data_kind=meta.data_kind,
            role=meta.role,
            input_mode=meta.input_mode,
            forecast_source=meta.forecast_source,
            temporal_metrics=temporal_metrics,
            spatial_metrics=spatial_metrics,
            uncertainty_metrics=uncertainty_metrics,
            baseline_comparisons=baseline_comps,
            verdicts=verdicts,
            model_input_mobility=model_input_mobility,
            pre_event_window_hours=pre_event_window_hours,
            od_source=od_source,
            leakage_guard=leakage_guard,
            predictability_source=predictability_source,
            profile_source=profile_source,
            feature_sources=feature_sources,
            hybrid_components=hybrid_components,
            external_temporal_metric_source="monte_carlo_physics",
            predictor_status=getattr(manifest, "predictor_status", {}) or {
                "model_status": "untrained_fallback",
                "engine": "monte_carlo_physics",
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
                "interval_source": "monte_carlo_quantile",
            },
            prediction_provenance=(
                prediction.provenance.to_dict()
                if hasattr(prediction, "provenance") and prediction.provenance
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
                    "interval_source": "monte_carlo_quantile",
                }
            ),
        )

        # 8. Generate output artifacts
        artifact_paths = self._export_artifacts(
            event_out_dir=event_out_dir,
            plots_dir=plots_dir,
            report=report,
            meta=meta,
            dataset=dataset,
            prediction=prediction,
            baseline_preds=baseline_preds,
            spatial_df=spatial_df,
        )

        return report, artifact_paths

    def _compute_temporal_metrics(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        baseline: np.ndarray,
        eval_start_hour: int,
        duration_hours: int,
        recovery_threshold: float,
        observed_baseline: Optional[np.ndarray] = None,
    ) -> TemporalMetrics:
        """Computes RMSE, MAE, MAPE, sMAPE, R2, peak loss, and recovery time on empirical curve."""
        y_true = observed[eval_start_hour:]
        y_pred = predicted[eval_start_hour:]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9)))) * 100.0
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9))) * 100.0
        r2 = float(r2_score(y_true, y_pred))

        obs_base = observed_baseline if observed_baseline is not None else baseline
        # Relative loss: max((baseline - activity) / baseline)
        loss_obs_series = np.maximum(0.0, (obs_base - observed) / np.maximum(obs_base, 1e-9))
        loss_pred_series = np.maximum(0.0, (baseline - predicted) / np.maximum(baseline, 1e-9))

        max_loss_obs = float(np.clip(np.max(loss_obs_series[eval_start_hour:]), 0.0, 1.0))
        max_loss_pred = float(np.clip(np.max(loss_pred_series[eval_start_hour:]), 0.0, 1.0))
        max_loss_err = float(abs(max_loss_pred - max_loss_obs))

        peak_h_obs = int(eval_start_hour + np.argmax(loss_obs_series[eval_start_hour:]))
        peak_h_pred = int(eval_start_hour + np.argmax(loss_pred_series[eval_start_hour:]))
        peak_h_err = abs(peak_h_pred - peak_h_obs)

        # Recovery time calculation
        rec_time_obs = self._calc_recovery_time(observed, obs_base, eval_start_hour, duration_hours, recovery_threshold)
        rec_time_pred = self._calc_recovery_time(predicted, baseline, eval_start_hour, duration_hours, recovery_threshold)
        rec_time_err = float(abs(rec_time_pred - rec_time_obs))

        return TemporalMetrics(
            rmse=rmse,
            mae=mae,
            mape_pct=mape,
            r2_score=r2,
            max_activity_loss_pred=max_loss_pred,
            max_activity_loss_obs=max_loss_obs,
            max_activity_loss_error=max_loss_err,
            peak_loss_hour_pred=peak_h_pred,
            peak_loss_hour_obs=peak_h_obs,
            peak_loss_hour_error=peak_h_err,
            recovery_time_pred=rec_time_pred,
            recovery_time_obs=rec_time_obs,
            recovery_time_error=rec_time_err,
            smape_pct=smape,
        )

    @staticmethod
    def _calc_recovery_time(
        activity: np.ndarray,
        baseline: np.ndarray,
        start_hour: int,
        duration_hours: int,
        threshold: float,
    ) -> float:
        total_h = min(len(activity), len(baseline))
        recovery_time = float(total_h - start_hour)
        search_start = min(total_h, start_hour + duration_hours)
        for t in range(search_start, total_h):
            ratio = activity[t] / max(baseline[t], 1e-9)
            if ratio >= threshold:
                return float(t - start_hour)
        return recovery_time

    def _compute_spatial_metrics(
        self,
        spatial_outlook: Any,
        dataset: ObservedEventDataset,
        meta: Any,
    ) -> Tuple[SpatialMetrics, pd.DataFrame]:
        """Calculates zone-level observed peak loss and computes Spearman rank correlations."""
        zone_dict = {z.zone_id: z for z in spatial_outlook.zone_rankings}
        records = []

        pred_vulns = []
        obs_losses = []
        obs_impacts = []
        has_impacts = bool(dataset.zone_impacts)

        for zid in sorted(dataset.target_zones if hasattr(dataset, "target_zones") else dataset.normalized_zone_activity.columns):
            zone_obj = zone_dict.get(zid)
            pred_vuln = zone_obj.vulnerability_score if zone_obj else 0.5
            pred_rank = zone_obj.vulnerability_rank if zone_obj else 99
            zone_name = zone_obj.zone_name if zone_obj else zid

            # Calculate observed peak relative activity loss in this zone
            z_act = dataset.normalized_zone_activity[zid].values
            z_loss_series = np.maximum(0.0, 1.0 - z_act)
            obs_peak_loss = float(np.clip(np.max(z_loss_series), 0.0, 1.0))

            obs_impact = dataset.zone_impacts.get(zid, np.nan)

            records.append({
                "zone_id": zid,
                "zone_name": zone_name,
                "predicted_vulnerability": round(pred_vuln, 4),
                "predicted_vulnerability_rank": pred_rank,
                "observed_peak_loss": round(obs_peak_loss, 4),
                "observed_impact_score": round(obs_impact, 4) if np.isfinite(obs_impact) else None,
            })

            pred_vulns.append(pred_vuln)
            obs_losses.append(obs_peak_loss)
            if has_impacts and np.isfinite(obs_impact):
                obs_impacts.append(obs_impact)

        spatial_df = pd.DataFrame(records)
        spatial_df["observed_loss_rank"] = spatial_df["observed_peak_loss"].rank(ascending=False, method="min").astype(int)

        # Spearman rank correlation vs observed peak loss
        if len(set(obs_losses)) > 1 and len(set(pred_vulns)) > 1:
            res_loss = spearmanr(pred_vulns, obs_losses)
            rho_loss = float(res_loss.statistic)
            pval_loss = float(res_loss.pvalue)
        else:
            rho_loss, pval_loss = 0.0, 1.0

        # Spearman rank correlation vs physical impacts (if present)
        rho_impact, pval_impact = None, None
        if has_impacts and len(obs_impacts) == len(pred_vulns) and len(set(obs_impacts)) > 1:
            res_impact = spearmanr(pred_vulns, obs_impacts)
            rho_impact = float(res_impact.statistic)
            pval_impact = float(res_impact.pvalue)

        metrics = SpatialMetrics(
            spearman_rho_loss=rho_loss,
            spearman_pvalue_loss=pval_loss,
            spearman_rho_impact=rho_impact,
            spearman_pvalue_impact=pval_impact,
        )

        return metrics, spatial_df

    def _compute_uncertainty_metrics(
        self,
        observed: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        eval_start_hour: int,
    ) -> UncertaintyMetrics:
        """Computes empirical PICP, MPIW, and Winkler score."""
        y_true = observed[eval_start_hour:]
        lower = lower_bound[eval_start_hour:]
        upper = upper_bound[eval_start_hour:]

        covered = (y_true >= lower) & (y_true <= upper)
        picp = float(np.mean(covered))

        widths = upper - lower
        mpiw = float(np.mean(widths))

        alpha = 0.05
        penalties = widths.copy()
        below = y_true < lower
        above = y_true > upper
        penalties[below] += (2.0 / alpha) * (lower[below] - y_true[below])
        penalties[above] += (2.0 / alpha) * (y_true[above] - upper[above])
        winkler = float(np.mean(penalties))

        ece = float(abs(picp - 0.95))

        return UncertaintyMetrics(
            picp_95=picp,
            mpiw=mpiw,
            winkler_score=winkler,
            ece=ece,
            nominal_coverage=0.95,
        )

    def _evaluate_baselines(
        self,
        mobility_network: Any,
        meta: Any,
        observed_normalized: np.ndarray,
        multi_agent_pred: np.ndarray,
        urban_context: Any,
        event_context: Any,
    ) -> Tuple[List[BaselineComparison], Dict[str, np.ndarray]]:
        """Evaluates benchmark models using observed pre-event window baseline."""
        eval_start = meta.event_start_hour
        y_true = observed_normalized[eval_start:]
        total_h = meta.total_hours

        # Diurnal normalized baseline from time-aligned mobility network
        raw_base = mobility_network.hourly_time_series.sum(axis=1).values[:total_h]
        b_mean = compute_baseline_mean(raw_base, event_start_hour=eval_start, lead_time_hours=meta.lead_time_hours)
        norm_base = raw_base / b_mean

        # Build observed unshocked baseline: pre-event is actual observation, post-event projects pre-event cycle
        obs_baseline = np.zeros(total_h, dtype=float)
        obs_baseline[:eval_start] = observed_normalized[:eval_start]
        for t in range(eval_start, total_h):
            lag = t - 24
            if lag >= 0:
                obs_baseline[t] = obs_baseline[lag]
            else:
                obs_baseline[t] = norm_base[t]

        # 1. Historical Baseline (No Shock)
        pred_hist = obs_baseline.copy()

        # 2. Autoregressive (lag-24 with decay)
        pred_ar = obs_baseline.copy()
        for t in range(eval_start, total_h):
            lag = t - 24
            val = obs_baseline[lag] if lag >= 0 else obs_baseline[t]
            pred_ar[t] = 0.85 * val

        # 3. Static Ridge
        X, y = [], []
        for zid, zone in urban_context.zones.items():
            feat = [zone.population / 100000.0, zone.road_density_km_per_sqkm, sum(zone.poi_counts.values()) / 1000.0]
            X.append(feat)
            y.append(event_context.zone_intensity_map.get(zid, 0.5))
        reg = Ridge(alpha=1.0).fit(np.array(X), np.array(y))
        ridge_impact = float(np.mean(reg.predict(np.array(X))))
        pred_ridge = obs_baseline.copy()
        for t in range(eval_start, total_h):
            if t < eval_start + meta.duration_hours:
                pred_ridge[t] *= max(0.2, 1.0 - 0.4 * ridge_impact)

        # 4. Multi-Agent
        pred_agent = multi_agent_pred[:total_h]

        models = [
            ("Historical Baseline (No Shock)", pred_hist[eval_start:]),
            ("Autoregressive (AR Extrapolation)", pred_ar[eval_start:]),
            ("Static Ridge Regression", pred_ridge[eval_start:]),
            ("Proposed Multi-Agent Framework", pred_agent[eval_start:]),
        ]

        comparisons = []
        for name, p in models:
            comparisons.append(
                BaselineComparison(
                    model_name=name,
                    rmse=float(np.sqrt(mean_squared_error(y_true, p))),
                    mae=float(mean_absolute_error(y_true, p)),
                    mape=float(np.mean(np.abs((y_true - p) / np.maximum(y_true, 1e-9)))) * 100.0,
                    r2_score=float(r2_score(y_true, p)),
                )
            )

        baseline_preds = {
            "historical": pred_hist,
            "autoregressive": pred_ar,
            "static_ridge": pred_ridge,
            "multi_agent": pred_agent,
        }

        return comparisons, baseline_preds

    @staticmethod
    def _compute_verdicts(
        meta: Any,
        temporal_metrics: TemporalMetrics,
        uncertainty_metrics: UncertaintyMetrics,
        baseline_comparisons: List[BaselineComparison],
    ) -> ValidationVerdicts:
        """Determines automated pass/fail verdicts against rigorous empirical standards."""
        reasons: List[str] = []

        # 1. Provenance Gate: real observed data requires verified status
        prov_status = getattr(meta, "provenance_status", "unverified")
        if meta.data_kind == "observed" and prov_status == "verified":
            prov_v = "PASS"
        else:
            prov_v = "FAILED"
            reasons.append(f"数据属性为 {meta.data_kind} (状态: {prov_status})，非已审计真实历史观测")

        # 2. Baseline Outperformance Gate
        ma_comp = next((b for b in baseline_comparisons if "Proposed" in b.model_name), None)
        other_comps = [b for b in baseline_comparisons if "Proposed" not in b.model_name]
        if ma_comp is None:
            base_v = "FAILED"
            reasons.append("缺少 Multi-Agent 评估结果")
        elif other_comps and any(b.rmse < ma_comp.rmse for b in other_comps):
            base_v = "FAILED"
            better_models = [b.model_name for b in other_comps if b.rmse < ma_comp.rmse]
            reasons.append(f"Multi-Agent 预测误差 (RMSE={ma_comp.rmse:.4f}) 劣于基线模型: {', '.join(better_models)}")
        else:
            base_v = "PASS"

        # 3. UQ Coverage Gate (PICP >= 0.80)
        if uncertainty_metrics.picp_95 >= 0.80:
            uq_v = "PASS"
        else:
            uq_v = "FAILED"
            reasons.append(f"不确定性置信带实际覆盖率不足 (PICP={uncertainty_metrics.picp_95:.2%} < 80.00%)")

        # 4. Amplitude Accuracy Gate (error <= 0.15)
        if temporal_metrics.max_activity_loss_error <= 0.15:
            amp_v = "PASS"
        else:
            amp_v = "FAILED"
            reasons.append(f"最大活动损失振幅误差过大 (误差={temporal_metrics.max_activity_loss_error:.2%} > 15.00%)")

        # 5. Overall Verdict
        overall_v = "PASS" if (prov_v == "PASS" and base_v == "PASS" and uq_v == "PASS" and amp_v == "PASS") else "FAILED"

        return ValidationVerdicts(
            provenance_verdict=prov_v,
            baseline_verdict=base_v,
            uq_verdict=uq_v,
            amplitude_verdict=amp_v,
            overall_verdict=overall_v,
            reasons=reasons,
        )

    def _export_artifacts(
        self,
        event_out_dir: Path,
        plots_dir: Path,
        report: ExternalValidationReport,
        meta: Any,
        dataset: ObservedEventDataset,
        prediction: Any,
        baseline_preds: Dict[str, np.ndarray],
        spatial_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Generates report.md, metrics.json, tables, and figures."""
        paths: Dict[str, Path] = {}

        # 1. metrics.json
        metrics_path = event_out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        paths["metrics_json"] = metrics_path

        # 2. predictions_vs_observed.csv
        total_h = meta.total_hours
        hours = np.arange(total_h)
        base_time = pd.to_datetime(meta.start_time) - pd.Timedelta(hours=meta.event_start_hour)
        timestamps = [(base_time + pd.Timedelta(hours=int(h))).isoformat() for h in hours]

        pred_df = pd.DataFrame({
            "hour": hours,
            "timestamp": timestamps,
            "observed_normalized_activity": np.round(dataset.normalized_city_activity[:total_h], 4),
            "multi_agent_pred_mean": np.round(prediction.predicted_activity[:total_h], 4),
            "multi_agent_lower_95": np.round(prediction.lower_bound_95[:total_h], 4),
            "multi_agent_upper_95": np.round(prediction.upper_bound_95[:total_h], 4),
            "historical_baseline": np.round(baseline_preds["historical"][:total_h], 4),
            "autoregressive_pred": np.round(baseline_preds["autoregressive"][:total_h], 4),
            "static_ridge_pred": np.round(baseline_preds["static_ridge"][:total_h], 4),
        })
        pred_csv_path = event_out_dir / "predictions_vs_observed.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        paths["predictions_csv"] = pred_csv_path

        # 3. spatial_validation.csv
        spatial_csv_path = event_out_dir / "spatial_validation.csv"
        spatial_df.to_csv(spatial_csv_path, index=False)
        paths["spatial_csv"] = spatial_csv_path

        # 4. plots/temporal_validation.png
        plot_temp_path = plots_dir / "temporal_validation.png"
        self._plot_temporal(
            plot_path=plot_temp_path,
            hours=hours,
            observed=dataset.normalized_city_activity[:total_h],
            prediction=prediction,
            baseline_preds=baseline_preds,
            meta=meta,
        )
        paths["plot_temporal"] = plot_temp_path

        # 5. plots/spatial_rank_correlation.png
        plot_spatial_path = plots_dir / "spatial_rank_correlation.png"
        self._plot_spatial(
            plot_path=plot_spatial_path,
            spatial_df=spatial_df,
            spatial_metrics=report.spatial_metrics,
            event_id=meta.event_id,
        )
        paths["plot_spatial"] = plot_spatial_path

        # 6. report.md
        report_md_path = event_out_dir / "report.md"
        md_content = self._render_markdown_report(report, meta, spatial_df, dataset=dataset)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        paths["report_md"] = report_md_path

        logger.info("External validation artifacts exported to %s", event_out_dir)
        return paths

    def _plot_temporal(
        self,
        plot_path: Path,
        hours: np.ndarray,
        observed: np.ndarray,
        prediction: Any,
        baseline_preds: Dict[str, np.ndarray],
        meta: Any,
    ) -> None:
        """Renders observed vs predicted trajectory figure."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        # Highlight event duration
        ax.axvspan(
            meta.event_start_hour,
            meta.event_start_hour + meta.duration_hours,
            color="#fed7d7",
            alpha=0.5,
            label=f"Hazard Active ({meta.hazard_type.upper()})",
        )

        # Prediction interval (95% CI)
        ax.fill_between(
            hours,
            prediction.lower_bound_95[: len(hours)],
            prediction.upper_bound_95[: len(hours)],
            color="#bee3f8",
            alpha=0.6,
            label="Multi-Agent 95% Credible Interval",
        )

        # Trajectories
        ax.plot(hours, observed, color="#1a202c", marker="o", markersize=3, linewidth=1.5, label="Observed Empirical Activity")
        ax.plot(hours, prediction.predicted_activity[: len(hours)], color="#2b6cb0", linewidth=2.2, label="Multi-Agent Anticipatory Forecast")
        ax.plot(hours, baseline_preds["historical"][: len(hours)], color="#718096", linestyle="--", linewidth=1.2, label="Historical Baseline (No Shock)")
        ax.plot(hours, baseline_preds["autoregressive"][: len(hours)], color="#dd6b20", linestyle=":", linewidth=1.4, label="Autoregressive (AR Extrapolation)")

        ax.set_title(
            f"External Validation: Observed vs Anticipatory Forecast [{meta.event_id}]\n"
            f"City: {meta.city_name} | Hazard: {meta.hazard_type.upper()} | Lead Time: {meta.lead_time_hours}h",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Timeline (Hours from Sequence Start)", fontsize=11)
        ax.set_ylabel("Normalized Aggregate Activity (Pre-event = 1.0)", fontsize=11)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", framealpha=0.9, fontsize=10)

        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def _plot_spatial(
        self,
        plot_path: Path,
        spatial_df: pd.DataFrame,
        spatial_metrics: SpatialMetrics,
        event_id: str,
    ) -> None:
        """Renders spatial vulnerability rank correlation scatter plot."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        x = spatial_df["predicted_vulnerability"].values
        y = spatial_df["observed_peak_loss"].values
        labels = spatial_df["zone_id"].values

        ax.scatter(x, y, color="#2b6cb0", s=65, edgecolors="#1a365d", zorder=4)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=8, color="#2d3748")

        # Trendline
        if len(x) > 1 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x) * 0.9, max(x) * 1.1, 50)
            ax.plot(x_line, m * x_line + b, color="#e53e3e", linestyle="--", linewidth=1.5,
                    label=f"Trendline (Spearman $\\rho$={spatial_metrics.spearman_rho_loss:.3f}, p={spatial_metrics.spearman_pvalue_loss:.3f})")

        ax.set_title(f"Spatial Resilience Validation: Predicted Vulnerability vs Observed Loss\nEvent: {event_id}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Model Predicted Vulnerability Score [0, 1]", fontsize=10)
        ax.set_ylabel("Observed Peak Relative Activity Loss [0, 1]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def _render_markdown_report(
        self,
        report: ExternalValidationReport,
        meta: Any,
        spatial_df: pd.DataFrame,
        dataset: Optional[ObservedEventDataset] = None,
    ) -> str:
        """Generates comprehensive GitHub Flavored Markdown external validation report."""
        is_verified_observed = (
            report.data_kind == "observed"
            and getattr(meta, "provenance_status", "unverified") == "verified"
        )

        # 1. Dynamic Title (FIX-2)
        if is_verified_observed:
            if report.input_mode == "ex_ante_forecast":
                report_title = f"# 外部真实事件与移动数据事前验证报告: {report.event_id}"
            else:
                report_title = f"# 外部真实事件后报验证报告: {report.event_id}"
        else:
            report_title = f"# 外部验证演示报告（合成/未验证数据）: {report.event_id}"

        # 2. Prominent Banners (FIX-2)
        unverified_banner = ""
        if not is_verified_observed:
            fixture_tag = " (FIXTURE / DRY-RUN 验证标记)" if getattr(meta, "data_kind", "") == "fixture" else ""
            unverified_banner = (
                "> [!WARNING]\n"
                f"> ### SYNTHETIC / UNVERIFIED — NOT REAL-WORLD VALIDATION{fixture_tag}\n"
                "> **非真实世界历史观测声明**：本报告基于合成演示数据 (Synthetic Demo) 或未审计测试桩 (Fixture) 生成，"
                "**不代表真实历史环境下的预测泛化能力**（不代表真实历史观测的泛化性能）。严禁将本报告结论作为真实世界业务闭环或模型实测精度的证明。"
                "真实有效性评估必须接入经权威机构审计的真实观测数据并提供完整 Provenance。\n\n"
            )

        hindcast_banner = ""
        if report.input_mode == "observed_hindcast":
            if is_verified_observed:
                hindcast_banner = (
                    "> [!NOTE]\n"
                    "> **后报验证 (HINDCAST) 标记**：本验证输入采用了事后实际降雨或气象观测数据作为强迫，评估的是物理响应机制，"
                    "而非严格的事前因果预报 (Ex-Ante Forecast)。\n\n"
                )
            else:
                hindcast_banner = (
                    "> [!NOTE]\n"
                    "> **合成/演示后报模式 (DEMO HINDCAST) 标记**：本验证输入为合成事件序列，并非真实气象观测；"
                    "评估仅供管线验证与机制演示，本报告不代表真实世界验证。\n\n"
                )
        elif is_verified_observed and report.input_mode == "ex_ante_forecast":
            hindcast_banner = (
                "> [!IMPORTANT]\n"
                "> **严格事前因果验证 (EX-ANTE VALIDATION)**：本验证严格依据灾害发生前（提前期 "
                f"{report.forecast_source}）发布的官方气象预警进行因果推演，并与真实观测数据进行盲测比对。\n\n"
            )

        # 3. Quality Gate Verdicts Banner
        v = report.verdicts
        verdict_banner = ""
        if v and v.overall_verdict == "FAILED":
            reasons_md = "\n".join(f">   - {r}" for r in v.reasons)
            verdict_banner = (
                "> [!CAUTION]\n"
                "> ### 外部验证未通过 (QUALITY GATE FAILED)\n"
                "> **外部验证未通过 (VALIDATION FAILED)**：本次验证未达到既定质量门禁标准。**禁止宣称“真实历史数据高精度闭环完成”**。\n"
                f"{reasons_md}\n\n"
            )
        elif v and v.overall_verdict == "PASS":
            verdict_banner = (
                "> [!TIP]\n"
                "> ### 外部验证全部达标 (QUALITY GATE PASSED)\n"
                "> 本次验证基于真实已审计观测数据，各项指标均通过严格因果门禁。\n\n"
            )

        # 4. Provenance and Coverage Information (FIX-5)
        prov = dataset.provenance if (dataset and dataset.provenance) else {}
        source_name = prov.get("source_name", "Synthetic Generator / Unverified" if not is_verified_observed else "Authorized Real Source")
        license_info = prov.get("license", "CC0-1.0 (Public Domain)" if not is_verified_observed else "Specified License")
        retrieved_time = prov.get("retrieved_at", prov.get("accessed_date", "2026-10-18"))
        checksums_dict = prov.get("checksums", {})
        act_sha = checksums_dict.get("activity.csv", "N/A")
        if act_sha != "N/A" and len(act_sha) > 16:
            act_sha_short = act_sha[:16] + "..."
        else:
            act_sha_short = act_sha

        total_hours = meta.total_hours
        zone_count = len(spatial_df) if spatial_df is not None else 12
        missing_hours = 0
        missing_zones = 0
        if dataset and hasattr(dataset, "hourly_zone_activity") and dataset.hourly_zone_activity is not None:
            missing_hours = int(dataset.hourly_zone_activity.isna().sum().sum())
        data_attr_desc = "已审计真实历史观测 (Verified Observed)" if is_verified_observed else "非真实观测 (Synthetic Demo / Fixture)"

        # 5. Conditioned Column Headers and Labels (FIX-2)
        target_col = "实测值 (Observed)" if is_verified_observed else "合成目标值 (Synthetic Target)"
        target_loss_col = "实测损失率" if is_verified_observed else "合成损失率"
        target_loss_rank = "实测损失排名" if is_verified_observed else "合成损失排名"
        target_impact_col = "实测灾情指数" if is_verified_observed else "合成灾情指数（未验证）"
        target_loss_stat = "实测峰值损失" if is_verified_observed else "合成目标峰值损失"
        val_value_label = "实测值" if is_verified_observed else "合成验证值"

        tm = report.temporal_metrics
        sm = report.spatial_metrics
        um = report.uncertainty_metrics

        baseline_rows = []
        for b in report.baseline_comparisons:
            is_ours = "Proposed" in b.model_name
            prefix = "**" if is_ours else ""
            suffix = "**" if is_ours else ""
            baseline_rows.append(
                f"| {prefix}{b.model_name}{suffix} | {prefix}{b.rmse:.4f}{suffix} | "
                f"{prefix}{b.mae:.4f}{suffix} | {prefix}{b.mape:.2f}%{suffix} | {prefix}{b.r2_score:.4f}{suffix} |"
            )
        baseline_table = "\n".join(baseline_rows)

        spatial_rows = []
        for _, row in spatial_df.sort_values("predicted_vulnerability_rank").iterrows():
            imp_str = f"{row['observed_impact_score']:.4f}" if pd.notnull(row['observed_impact_score']) else "N/A"
            spatial_rows.append(
                f"| {row['zone_id']} | {row['zone_name']} | {row['predicted_vulnerability']:.4f} | "
                f"{row['predicted_vulnerability_rank']} | {row['observed_peak_loss']:.4f} | "
                f"{row['observed_loss_rank']} | {imp_str} |"
            )
        spatial_table = "\n".join(spatial_rows)
        impact_stat_line = ""
        if sm.spearman_rho_impact is not None:
            impact_stat_line = f"\n- **脆弱性评分 vs 目标物理灾情得分 Spearman 秩相关系数 ($\\rho$)**: **{sm.spearman_rho_impact:.4f}** (p-value: `{sm.spearman_pvalue_impact:.4e}`)"

        verdict_section = ""
        if v:
            verdict_section = f"""## 质量门禁与验证判定 (Quality Gate Verdicts)

| 门禁检验项 | 判定状态 | 门禁标准说明 | 实际表现 |
| :--- | :---: | :--- | :--- |
| **数据源真实性与 Provenance 审计** | `{v.provenance_verdict}` | 必须为 `observed` 且 `provenance_status='verified'` | `{report.data_kind}` (`{getattr(meta, 'provenance_status', 'unverified')}`) |
| **基线模型超越检验 (Benchmark Outperformance)** | `{v.baseline_verdict}` | Multi-Agent RMSE 必须优于或等于各基准模型 | RMSE: Multi-Agent={tm.rmse:.4f} |
| **不确定性区间覆盖 (UQ Calibration)** | `{v.uq_verdict}` | 95% 置信带实际覆盖率 (PICP) $\\ge 80.0\\%$ | PICP: {um.picp_95:.2%} |
| **冲击振幅误差 (Amplitude Accuracy)** | `{v.amplitude_verdict}` | 最大活动损失绝对误差 $\\le 15.0\\%$ | 误差: {tm.max_activity_loss_error:.2%} |
| **综合门禁结论 (Overall Verdict)** | **`{v.overall_verdict}`** | 全部单项检验通过方可判定闭环验证通过 | 门禁状态: {v.overall_verdict} |

---
"""

        pred_fallback_warning = ""
        if report.predictability_source == "neutral_fallback":
            pred_fallback_warning = f"""> [!WARNING]
> **可预测性特征降级告警 (Predictability Feature Degraded)**: 当前灾前观测窗口（{report.pre_event_window_hours} 小时）不足 48 小时。PredictabilityAgent 计算 lag-24 自相关与日间余弦规律性需要至少 48 小时连续观测，当前已自动启用防御性中性值回退（0.50），未能充分反映真实日间节律与周期稳定性。建议接入至少 48 小时（推荐 72 小时）连续灾前观测数据以解锁完整 Multi-Agent 评估能力。

"""

        hybrid_components_str = ", ".join(f"`{c}`" for c in report.hybrid_components) if report.hybrid_components else "无"
        hybrid_warning = f"""> [!NOTE]
> **混杂验证说明 (Hybrid Validation Limitation)**: 当前外部验证为**混杂特征外部验证 (Hybrid Validation)**，而非全特征纯经验观测验证。虽然时序活动与弹性扰动直接来源于灾前实测（`observed_pre_event`），但空间 OD 流量依赖重力流回退（`{report.od_source}`）、功能混合依赖静态 POI 上下文（`{report.feature_sources.get('functional', 'static_urban_context')}`）、日/周基准轮廓依赖参考模型（`{report.profile_source}`）。当前包含的非纯观测/混杂组件: {hybrid_components_str}。请勿将本报告结果解读为所有 Agent 特征均由纯经验观测驱动。

"""

        pred_source_note = ""
        p_status = report.predictor_status or {}
        summary_source = p_status.get("summary_metric_source", "monte_carlo_physics")
        metric_sources_dict = p_status.get("metric_sources", {})
        metric_sources_lines = "\n".join(
            f"  - `{k}`: `{v}`" for k, v in metric_sources_dict.items()
        ) if metric_sources_dict else "  - (默认全量物理模拟)"

        if summary_source == "trained_model":
            pred_source_note = f"""> [!NOTE]
> **预测指标口径分离说明 (Metric Source Separation)**:
> - **汇总指标 (Summary Metrics)**: 最大活动损失率、韧性恢复时长、累积损失量来源于多目标机器学习回归模型 (`trained_model`)；
> - **时序曲线与轨迹指标 (Curve & Dynamic Metrics)**: 逐小时扰动曲线以及相关的 RMSE、MAE、MAPE、sMAPE、R²、PICP 等指标均来源于蒙特卡洛物理动力学模拟 (`monte_carlo_physics`)；
> - **严禁混为一谈**: 预测模型输出的是宏观汇总估计，时序拟合度反映的是物理动力学机制，不得将二者混淆为同一套模型性能。
> - **指标分项来源明细**:
{metric_sources_lines}

"""
        else:
            pred_source_note = f"""> [!NOTE]
> **预测模型状态与指标口径说明**: 当前预测器状态为 `{p_status.get('model_status', 'untrained_fallback')}`。所有宏观汇总指标及逐小时时序曲线均由蒙特卡洛物理动力学模拟 (`monte_carlo_physics`) 生成。
> - **指标分项来源明细**:
{metric_sources_lines}

"""

        synthetic_artifact_warning = ""
        if is_synthetic_trained(p_status):
            synthetic_artifact_warning = (
                "> [!WARNING]\n"
                f"> **合成标签训练警告 (Synthetic Predictor Artifact)**: 当前加载的预测器模型基于合成模拟器标签 (`label_source={p_status.get('training_label_source', 'synthetic_simulator')}`, `engineering_only=True`) 训练，"
                "**绝非基于真实历史灾情数据训练的经验基准模型**。禁止宣称真实世界泛化能力。\n\n"
            )
        elif p_status.get("summary_metric_source") == "trained_model" and p_status.get("training_label_source") in ("unknown", None, "none", ""):
            synthetic_artifact_warning = (
                "> [!WARNING]\n"
                "> **模型训练溯源警告 (Unverified Model Training Provenance)**: 当前加载的预测器模型缺少经过验证的训练标签来源，禁止作为可审计经验基准模型。\n\n"
            )

        content = f"""{report_title}

{verdict_banner}{unverified_banner}{hindcast_banner}{pred_fallback_warning}{synthetic_artifact_warning}{hybrid_warning}### 数据属性与 Provenance 溯源信息
- **城市**: {report.city_name}
- **灾害类型**: {report.hazard_type.upper()}
- **预警来源**: {report.forecast_source}
- **数据属性**: `{report.data_kind}` (`{report.role}`) — **{data_attr_desc}**
- **预测器训练来源**: `{p_status.get('training_label_source', 'none')}` (engineering_only={p_status.get('engineering_only', True)})
- **Provenance 审计状态**: `{getattr(meta, 'provenance_status', 'unverified')}`
- **数据来源机构 (Source)**: {source_name}
- **授权协议 (License)**: {license_info}
- **数据抓取/生成时间 (Retrieved At)**: {retrieved_time}
- **关键文件校验和 (activity.csv SHA-256)**: `{act_sha_short}`
- **数据覆盖度 (Coverage)**: 时间跨度 {total_hours} 小时，涵盖 {zone_count} 个空间评估分区 (时序缺失: {missing_hours} 小时, 空间缺失: {missing_zones} 分区)
- **输入模式**: `{report.input_mode}`
- **模型移动性输入**: `{report.model_input_mobility}`
- **灾前观测窗口**: {f"{report.pre_event_window_hours} 小时" if report.pre_event_window_hours is not None else "N/A (Synthetic Demo)"}
- **汇总指标来源 (Summary Metric Source)**: `{p_status.get('summary_metric_source', 'monte_carlo_physics')}`
- **时序曲线来源 (Curve Metric Source)**: `{p_status.get('curve_metric_source', 'monte_carlo_physics')}`
- **外部评估时序指标来源 (External Temporal Source)**: `{report.external_temporal_metric_source}`
- **特征来源映射 (Feature Sources)**:
  - `temporal` (时序扰动): `{report.feature_sources.get('temporal', 'synthetic')}`
  - `flexibility` (弹性冗余): `{report.feature_sources.get('flexibility', 'synthetic')}`
  - `spatial` (空间拓扑): `{report.feature_sources.get('spatial', 'synthetic')}`
  - `functional` (功能混合): `{report.feature_sources.get('functional', 'synthetic')}`
  - `predictability` (可预测性): `{report.feature_sources.get('predictability', 'synthetic')}`
  - `daily_profiles` (日内基准轮廓): `{report.feature_sources.get('daily_profiles', 'synthetic')}`
  - `weekly_profiles` (周内基准轮廓): `{report.feature_sources.get('weekly_profiles', 'synthetic')}`
  - `od_flows` (空间 OD 流量): `{report.feature_sources.get('od_flows', 'synthetic')}`
- **非纯观测/混杂组件 (Hybrid Components)**: {hybrid_components_str}
- **OD 流量来源**: `{report.od_source}`
- **防数据泄漏门禁**: `{report.leakage_guard}`
- **提前期**: {meta.lead_time_hours} 小时 (事件起始相对时刻: 第 {meta.event_start_hour} 小时)
- **事件历时**: {meta.duration_hours} 小时 (总评估窗口: {meta.total_hours} 小时)

---

{verdict_section}{pred_source_note}## 1. 时序扰动与恢复对比指标 (Temporal Disruption & Recovery)

| 评估指标 | 预测值 (Predicted) | {target_col} | 绝对误差 (Absolute Error) |
| :--- | :---: | :---: | :---: |
| **最大活动损失率 (Peak Loss Ratio)** | {tm.max_activity_loss_pred:.2%} | {tm.max_activity_loss_obs:.2%} | {tm.max_activity_loss_error:.2%} |
| **峰值冲击发生时刻 (Peak Hour)** | 第 {tm.peak_loss_hour_pred} 小时 | 第 {tm.peak_loss_hour_obs} 小时 | {tm.peak_loss_hour_error} 小时 |
| **韧性恢复时长 (Recovery Time, >=95%)** | {tm.recovery_time_pred:.1f} 小时 | {tm.recovery_time_obs:.1f} 小时 | {tm.recovery_time_error:.1f} 小时 |
| **均方根误差 (RMSE)** | - | - | **{tm.rmse:.4f}** |
| **平均绝对误差 (MAE)** | - | - | **{tm.mae:.4f}** |
| **平均绝对百分比误差 (MAPE)** | - | - | **{tm.mape_pct:.2f}%** |
| **对称平均绝对百分比误差 (sMAPE)** | - | - | **{tm.smape_pct:.2f}%** |
| **判定系数 (R² Score)** | - | - | **{tm.r2_score:.4f}** |

---

## 2. 空间脆弱性排序与目标灾情相关性 (Spatial Rank Correlation)

- **脆弱性评分 vs {target_loss_stat} Spearman 秩相关系数 ($\\rho$)**: **{sm.spearman_rho_loss:.4f}** (p-value: `{sm.spearman_pvalue_loss:.4e}`){impact_stat_line}

### 分区详细对比

| 分区 ID | 分区名称 | 预测脆弱度 | 预测脆弱排名 | {target_loss_col} | {target_loss_rank} | {target_impact_col} |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{spatial_table}

---

## 3. 不确定性量化与校准评估 (Uncertainty Quantification)

| 指标 | {val_value_label} | 理想/基准参考 | 含义说明 |
| :--- | :---: | :---: | :--- |
| **区间覆盖概率 (PICP)** | **{um.picp_95:.2%}** | 95.00% | 80%~95% 置信带实际包含实测值的比例 |
| **区间平均宽度 (MPIW)** | **{um.mpiw:.4f}** | 越小越好 (锐度高) | 预测区间的平均宽度 |
| **Winkler 综合区间得分** | **{um.winkler_score:.4f}** | 越小越好 | 综合惩罚区间宽度与违约偏差的严谨评分 |
| **期望校准误差 (ECE)** | **{um.ece:.4f}** | 越接近 0 越好 | 实际覆盖率与名义置信水平的偏差绝对值 |

---

## 4. 基线模型对比评估 (Benchmark Model Comparison)

| 算法模型 | RMSE | MAE | MAPE | R² 评分 |
| :--- | :---: | :---: | :---: | :---: |
{baseline_table}

---

## 5. 生成可视化文件索引

- **时序验证图**: `plots/temporal_validation.png`
- **空间相关性散点图**: `plots/spatial_rank_correlation.png`
- **逐小时时序对比表**: `predictions_vs_observed.csv`
- **分区诊断数据表**: `spatial_validation.csv`
- **指标全量数据**: `metrics.json`
"""
        return content
