"""Resilience Synthesis Agent reconciling multi-agent evidence and event context."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from urban_resilience.config import AgentWeightConfig, SimulationConfig, compute_baseline_mean
from urban_resilience.types import (
    AgentEvidence,
    DisruptionMetrics,
    EventContext,
    MobilityNetwork,
    ReActTrace,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    SpatialZoneResilience,
    UrbanContext,
)

logger = logging.getLogger(__name__)


def compute_disruption_metrics(
    baseline: np.ndarray,
    predicted: np.ndarray,
    event_start_hour: int = 0,
    event_duration_hours: int = 1,
    recovery_threshold_ratio: float = 0.95,
    *,
    event_start: Optional[int] = None,
) -> DisruptionMetrics:
    """Computes quantitative disruption metrics from baseline and predicted activity time series."""
    if event_start is not None:
        event_start_hour = event_start
    baseline = np.asarray(baseline, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total_h = min(len(baseline), len(predicted))
    base = baseline[:total_h]
    pred = predicted[:total_h]

    activity_drop = base - pred
    loss_ratio = np.maximum(0.0, activity_drop / np.maximum(base, 1e-9))
    max_loss = float(np.clip(np.max(loss_ratio), 0.0, 1.0))
    peak_hour = int(np.argmax(loss_ratio))
    cumulative_loss = float(np.sum(np.maximum(0.0, activity_drop)))

    recovery_time = float(total_h - event_start_hour)
    for t in range(event_start_hour + event_duration_hours, total_h):
        if pred[t] >= recovery_threshold_ratio * base[t]:
            recovery_time = float(t - event_start_hour)
            break

    return DisruptionMetrics(
        maximum_activity_loss=max_loss,
        cumulative_activity_loss=cumulative_loss,
        recovery_time_hours=recovery_time,
        peak_loss_hour=peak_hour,
        resilience_triangle_area=cumulative_loss,
    )


class ResilienceSynthesisAgent:
    """Central synthesis agent integrating features, reconciling evidence, performing targeted follow-ups,

    and generating event-conditioned resilience forecasts with quantified uncertainty.
    """

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        weight_config: Optional[AgentWeightConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
    ):
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.weights = weight_config or AgentWeightConfig()
        self.sim_config = sim_config or SimulationConfig()
        self.traces: List[ReActTrace] = []
        self.follow_up_logs: List[Dict[str, Any]] = []

    def synthesize(
        self,
        evidences: Dict[str, AgentEvidence],
        event_context: EventContext,
        agents_pool: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ResiliencePrediction, SpatialResilienceOutlook, pd.DataFrame]:
        """Integrates features, reconciles evidence, triggers targeted follow-up, and forecasts resilience."""
        logger.info("Resilience Synthesis Agent: Initiating cross-agent evidence reconciliation...")

        # Step 1: Feature Integration
        zone_features = self._integrate_features(evidences, event_context)

        # Step 2: Reconcile Evidence & Check for Targeted Follow-up
        reconciliation_notes, follow_ups_needed = self._reconcile_evidence(evidences, zone_features)

        # Step 3: Execute Targeted Follow-up Loop (if agents_pool provided)
        if follow_ups_needed and agents_pool:
            zone_features = self._execute_targeted_follow_up(follow_ups_needed, agents_pool, zone_features)

        # Step 4: Event-Conditioned Prediction with Uncertainty Quantification
        prediction = self._generate_event_conditioned_prediction(zone_features, event_context)

        # Step 5: Spatial Resilience Outlook & Priority Ranking
        spatial_outlook = self._generate_spatial_outlook(zone_features, event_context, prediction)

        return prediction, spatial_outlook, zone_features

    def _integrate_features(self, evidences: Dict[str, AgentEvidence], event_context: EventContext) -> pd.DataFrame:
        """Assembles zone-level composite feature matrix."""
        rows = []
        w = self.weights.as_dict()

        for zid in self.urban_context.zone_ids:
            zone = self.urban_context.zones[zid]

            temp_m = evidences["temporal"].zone_metrics.get(zid, {}) if "temporal" in evidences else {}
            flex_m = evidences["flexibility"].zone_metrics.get(zid, {}) if "flexibility" in evidences else {}
            pred_m = evidences["predictability"].zone_metrics.get(zid, {}) if "predictability" in evidences else {}
            spat_m = evidences["spatial"].zone_metrics.get(zid, {}) if "spatial" in evidences else {}
            func_m = evidences["functional"].zone_metrics.get(zid, {}) if "functional" in evidences else {}

            temp_score = temp_m.get("temporal_redundancy_score", 0.5)
            flex_score = flex_m.get("intraday_flexibility_score", 0.5)
            pred_score = pred_m.get("predictability_score", 0.5)
            spat_score = spat_m.get("spatial_redundancy_score", 0.5)
            func_score = func_m.get("functional_mix_score", 0.5)

            # Composite intrinsic resilience score (weighted linear combination)
            composite_resilience = float(
                w["temporal_redundancy"] * temp_score
                + w["intraday_flexibility"] * flex_score
                + w["predictability"] * pred_score
                + w["spatial_redundancy"] * spat_score
                + w["functional_mix"] * func_score
            )
            composite_resilience = float(np.clip(composite_resilience, 0.05, 0.95))

            # Event shock intensity for this zone
            hazard_intensity = event_context.zone_intensity_map.get(zid, 0.5)

            # Vulnerability score: higher hazard and lower resilience yields higher vulnerability
            vulnerability = float(np.clip(hazard_intensity * (1.15 - composite_resilience), 0.05, 1.0))

            rows.append({
                "zone_id": zid,
                "zone_name": zone.name,
                "temporal_redundancy": temp_score,
                "intraday_flexibility": flex_score,
                "predictability": pred_score,
                "spatial_redundancy": spat_score,
                "functional_mix": func_score,
                "composite_resilience": composite_resilience,
                "hazard_intensity": hazard_intensity,
                "vulnerability_score": vulnerability,
                "elevation_m": zone.elevation_m,
                "population": zone.population,
            })

        df = pd.DataFrame(rows).set_index("zone_id")
        return df

    def _reconcile_evidence(
        self, evidences: Dict[str, AgentEvidence], zone_features: pd.DataFrame
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Identifies conflicting evidence and flags zones needing targeted follow-up."""
        notes = []
        follow_ups = []

        for zid, row in zone_features.iterrows():
            # Check for conflict: High spatial redundancy but low temporal flexibility
            if row["spatial_redundancy"] > 0.65 and row["intraday_flexibility"] < 0.35:
                note = f"Zone {zid}: Conflict between high spatial routes and rigid commuting windows."
                notes.append(note)
                follow_ups.append({
                    "zone_id": zid,
                    "target_agent": "flexibility",
                    "reason": "Examine if off-peak flexibility can be mobilized through telecommuting or flexible shifts.",
                })

            # Check for high vulnerability with low elevation
            if row["elevation_m"] < 6.0 and row["hazard_intensity"] > 0.6:
                follow_ups.append({
                    "zone_id": zid,
                    "target_agent": "spatial",
                    "reason": "Re-verify alternative evacuation corridors under waterlogging scenario.",
                })

        return notes, follow_ups

    def _execute_targeted_follow_up(
        self, follow_ups: List[Dict[str, Any]], agents_pool: Dict[str, Any], zone_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Executes targeted follow-up queries back to domain agents to refine evidence.

        Under Option A, domain agents provide refined assessments under emergency shock conditions:
        - Flexibility inquiries reflect mobilized emergency staggered shifts and telecommuting,
          yielding an adaptive adjustment (+0.05 to intraday flexibility score, clipped to 1.0).
        - Spatial inquiries evaluate flooded network disruption: low-elevation evacuation bottlenecks
          face a hazard-induced discount to spatial redundancy based on local hazard intensity
          (discount factor: 1.0 - 0.15 * hazard_intensity).
        Zone composite resilience and vulnerability scores are deterministically recalculated.
        """
        logger.info("Executing %d targeted follow-up inquiries...", len(follow_ups))
        updated_features = zone_features.copy()
        w = self.weights.as_dict()

        for req in follow_ups:
            zid = req["zone_id"]
            agent_key = req["target_agent"]
            target_agent = agents_pool.get(agent_key)
            if target_agent and zid in updated_features.index:
                # Trigger a targeted query action on the domain agent
                thought = f"Targeted follow-up: {req['reason']} for zone {zid}"
                target_agent.execute_react_step(
                    thought=thought,
                    action_type="query_data",
                    action_input={"query_type": "zone_info", "zone_id": zid},
                )

                # Apply deterministic refinements
                if agent_key == "flexibility":
                    old_flex = updated_features.loc[zid, "intraday_flexibility"]
                    updated_features.loc[zid, "intraday_flexibility"] = float(np.clip(old_flex + 0.05, 0.0, 1.0))
                elif agent_key == "spatial":
                    old_spat = updated_features.loc[zid, "spatial_redundancy"]
                    h_int = updated_features.loc[zid, "hazard_intensity"]
                    discount = max(0.5, 1.0 - 0.15 * h_int)
                    updated_features.loc[zid, "spatial_redundancy"] = float(np.clip(old_spat * discount, 0.0, 1.0))

                # Recompute composite resilience and vulnerability
                r_comp = float(
                    w["temporal_redundancy"] * updated_features.loc[zid, "temporal_redundancy"]
                    + w["intraday_flexibility"] * updated_features.loc[zid, "intraday_flexibility"]
                    + w["predictability"] * updated_features.loc[zid, "predictability"]
                    + w["spatial_redundancy"] * updated_features.loc[zid, "spatial_redundancy"]
                    + w["functional_mix"] * updated_features.loc[zid, "functional_mix"]
                )
                r_comp = float(np.clip(r_comp, 0.05, 0.95))
                updated_features.loc[zid, "composite_resilience"] = r_comp
                h_int = updated_features.loc[zid, "hazard_intensity"]
                updated_features.loc[zid, "vulnerability_score"] = float(np.clip(h_int * (1.15 - r_comp), 0.05, 1.0))

                self.follow_up_logs.append({
                    "zone_id": zid,
                    "agent": agent_key,
                    "prompt": req["reason"],
                    "status": "refined",
                })
        return updated_features

    def _generate_event_conditioned_prediction(
        self, zone_features: pd.DataFrame, event_context: EventContext
    ) -> ResiliencePrediction:
        """Generates dynamic disruption and recovery trajectories using multi-zone dynamics with uncertainty quantification."""
        total_h = min(
            self.sim_config.total_hours,
            len(self.mobility_network.hourly_time_series),
            len(event_context.hourly_forecast),
        )
        time_hours = np.arange(total_h)

        # 1. City aggregate baseline curve
        hourly_total = self.mobility_network.hourly_time_series.sum(axis=1).values[:total_h]
        # Normalize to baseline scale 1.0 using compute_baseline_mean helper
        baseline_mean = compute_baseline_mean(
            hourly_total,
            event_start_hour=event_context.event_start_hour,
            lead_time_hours=self.sim_config.event_lead_time_hours,
        )
        if baseline_mean <= 0 or not np.isfinite(baseline_mean):
            raise ValueError(f"Invalid baseline mean: {baseline_mean}. Baseline series must be positive and non-empty.")
        norm_baseline = hourly_total / baseline_mean

        # 2. Zone-level disruption and recovery simulation
        n_mc = self.sim_config.monte_carlo_samples
        rng = np.random.default_rng(self.sim_config.random_seed)

        # Store Monte Carlo trajectory draws: shape (n_mc, total_h)
        mc_trajectories = np.zeros((n_mc, total_h), dtype=float)

        event_start = event_context.event_start_hour
        event_dur = event_context.duration_hours
        hourly_shock = event_context.hourly_forecast[:total_h]

        for s in range(n_mc):
            # Sample uncertainty in resilience and shock parameters
            zone_losses = np.zeros(total_h, dtype=float)

            for zid, row in zone_features.iterrows():
                base_z = self.mobility_network.hourly_time_series[zid].values[:total_h]
                zone_weight = np.mean(base_z) / np.mean(hourly_total)

                # Intrinsic shock vulnerability with Monte Carlo noise
                r_sample = np.clip(row["composite_resilience"] + rng.normal(0, 0.04), 0.05, 0.98)
                h_sample = np.clip(row["hazard_intensity"] + rng.normal(0, 0.05), 0.1, 1.0)
                vulnerability = h_sample * (1.10 - r_sample)

                # Disruption depth during event
                max_loss_ratio = float(np.clip(vulnerability * 0.75, 0.05, 0.90))

                # Recovery rate: higher resilience -> faster recovery
                recovery_speed = float(0.08 + 0.14 * r_sample + rng.normal(0, 0.01))

                # Compute zone activity curve across time
                z_activity = np.ones(total_h, dtype=float)
                for t in range(total_h):
                    if t < event_start:
                        z_activity[t] = 1.0
                    elif t < event_start + event_dur:
                        # Shock unfolding phase
                        shock_fraction = hourly_shock[t] / (event_context.peak_intensity + 1e-6)
                        current_loss = max_loss_ratio * shock_fraction
                        z_activity[t] = max(0.1, 1.0 - current_loss)
                    else:
                        # Post-event recovery phase (exponential relaxation towards 1.0)
                        hours_post = t - (event_start + event_dur)
                        loss_at_end = max_loss_ratio * (hourly_shock[event_start + event_dur - 1] / (event_context.peak_intensity + 1e-6))
                        remaining_loss = loss_at_end * np.exp(-recovery_speed * hours_post)
                        z_activity[t] = min(1.0, max(0.1, 1.0 - remaining_loss))

                zone_losses += zone_weight * z_activity

            # Aggregate trajectory with calibrated aleatoric and epistemic uncertainty
            aleatoric_noise = rng.normal(0, 0.03, total_h)
            mc_trajectories[s, :] = np.maximum(0.01, norm_baseline * zone_losses + aleatoric_noise)

        # Expected predicted activity and 95% uncertainty confidence intervals
        predicted_mean = np.mean(mc_trajectories, axis=0)
        raw_lower = np.percentile(mc_trajectories, 2.5, axis=0)
        raw_upper = np.percentile(mc_trajectories, 97.5, axis=0)
        lower_95 = np.minimum(predicted_mean, raw_lower)
        upper_95 = np.maximum(predicted_mean, raw_upper)

        # Quantitative disruption metrics
        metrics = compute_disruption_metrics(
            baseline=norm_baseline,
            predicted=predicted_mean,
            event_start_hour=event_start,
            event_duration_hours=event_dur,
            recovery_threshold_ratio=self.sim_config.recovery_threshold_ratio,
        )

        return ResiliencePrediction(
            time_hours=time_hours,
            baseline_activity=norm_baseline,
            predicted_activity=predicted_mean,
            lower_bound_95=lower_95,
            upper_bound_95=upper_95,
            disruption_metrics=metrics,
        )

    def _generate_spatial_outlook(
        self, zone_features: pd.DataFrame, event_context: EventContext, prediction: ResiliencePrediction
    ) -> SpatialResilienceOutlook:
        """Constructs spatial vulnerability assessment and priority intervention ranking."""
        sorted_zones = zone_features.sort_values(by="vulnerability_score", ascending=False)
        rankings = []

        vulnerable_areas = []
        priority_ranking = []

        for rank, (zid, row) in enumerate(sorted_zones.iterrows(), 1):
            vuln_score = float(row["vulnerability_score"])
            comp_res = float(row["composite_resilience"])

            # Determine primary vulnerability driver
            drivers = {
                "Low Spatial Routes": 1.0 - row["spatial_redundancy"],
                "Rigid Temporal Peak": 1.0 - row["temporal_redundancy"],
                "Single Land-use Dependency": 1.0 - row["functional_mix"],
                "Low Elevation / Surge Risk": max(0.0, (20.0 - row["elevation_m"]) / 20.0),
            }
            primary_driver = max(drivers.items(), key=lambda x: x[1])[0]

            key_evidence = [
                f"Resilience index: {comp_res:.2f}",
                f"Hazard exposure: {row['hazard_intensity']:.2f}",
                f"Primary vulnerability driver: {primary_driver}",
            ]

            z_res = SpatialZoneResilience(
                zone_id=zid,
                zone_name=row["zone_name"],
                vulnerability_score=vuln_score,
                vulnerability_rank=rank,
                composite_resilience_index=comp_res,
                primary_vulnerability_driver=primary_driver,
                key_evidence=key_evidence,
                confidence=0.91,
            )
            rankings.append(z_res)
            priority_ranking.append(zid)
            if rank <= 4:
                vulnerable_areas.append(zid)

        summary = (
            f"Spatial assessment identified {len(vulnerable_areas)} top-priority vulnerable zones: "
            f"{', '.join(vulnerable_areas)}. Highest intervention priority allocated to {rankings[0].zone_name}."
        )

        return SpatialResilienceOutlook(
            zone_rankings=rankings,
            vulnerable_areas=vulnerable_areas,
            priority_ranking=priority_ranking,
            spatial_summary=summary,
            zone_features=zone_features,
        )
