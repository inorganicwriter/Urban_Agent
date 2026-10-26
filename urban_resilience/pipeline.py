"""Full-pipeline orchestrator coordinating data ingestion, multi-agent reasoning, synthesis, and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from urban_resilience.config import AgentWeightConfig, ProjectPaths, SimulationConfig
from urban_resilience.types import (
    AgentEvidence,
    EventContext,
    HazardType,
    MobilityNetwork,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    UrbanContext,
    ValidationResult,
)
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.orchestration import OrchestrationError, RunManifest, SupervisorAgent
from urban_resilience.outputs.prediction_engine import AnticipatoryPredictionEngine
from urban_resilience.outputs.visualizer import UrbanResilienceVisualizer
from urban_resilience.validation.evaluator import HistoricalEventEvaluator
from urban_resilience.validation.baselines import BaselineBenchmarkRunner
from urban_resilience.validation.ablation import AgentAblationStudy
from urban_resilience.validation.calibration import UncertaintyCalibrator

logger = logging.getLogger(__name__)


class UrbanResiliencePipeline:
    """Master pipeline orchestrating the entire anticipatory urban resilience prediction workflow."""

    def __init__(
        self,
        city_name: str = "Shenzhen",
        sim_config: Optional[SimulationConfig] = None,
        weight_config: Optional[AgentWeightConfig] = None,
        output_dir: Optional[Path] = None,
        orchestration_mode: str = "deterministic",
        llm_client: Optional[Any] = None,
    ):
        self.city_name = city_name
        self.sim_config = sim_config or SimulationConfig()
        self.weight_config = weight_config or AgentWeightConfig()
        self.output_dir = Path(output_dir) if output_dir else ProjectPaths.OUTPUTS_DIR
        self.plots_dir = (self.output_dir / "plots") if output_dir else ProjectPaths.PLOTS_DIR
        ProjectPaths.ensure_directories()
        self.visualizer = UrbanResilienceVisualizer(self.plots_dir)
        self.orchestration_mode = orchestration_mode
        self.llm_client = llm_client

        # Pipeline state components
        self.urban_context: Optional[UrbanContext] = None
        self.mobility_network: Optional[MobilityNetwork] = None
        self.event_context: Optional[EventContext] = None
        self.supervisor: Optional[SupervisorAgent] = None
        self.manifest: Optional[RunManifest] = None
        self.domain_evidences: Dict[str, AgentEvidence] = {}
        self.prediction: Optional[ResiliencePrediction] = None
        self.spatial_outlook: Optional[SpatialResilienceOutlook] = None
        self.validation_result: Optional[ValidationResult] = None
        self.resolved_config: Optional[SimulationConfig] = None

    def run_full_pipeline(
        self,
        hazard_type: Optional[HazardType] = None,
        event_start_hour: Optional[int] = None,
        duration_hours: Optional[int] = None,
        peak_intensity: Optional[float] = None,
        output_manifest_path: Optional[Path | str] = None,
        orchestration_mode: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes all 5 stages of the framework end-to-end."""
        active_mode = orchestration_mode if orchestration_mode is not None else self.orchestration_mode
        active_client = llm_client if llm_client is not None else self.llm_client

        hazard_type = hazard_type or HazardType.RAIN_STORM
        resolved_start = self.sim_config.event_start_hour if event_start_hour is None else event_start_hour
        resolved_duration = self.sim_config.event_duration_hours if duration_hours is None else duration_hours
        resolved_peak = self.sim_config.peak_intensity if peak_intensity is None else peak_intensity

        # Validate unified explicit parameters through resolved SimulationConfig
        resolved_config = SimulationConfig(
            **{
                **self.sim_config.model_dump(),
                "event_start_hour": resolved_start,
                "event_duration_hours": resolved_duration,
                "peak_intensity": resolved_peak,
            }
        )
        self.resolved_config = resolved_config

        logger.info("================================================================================")
        logger.info("STARTING URBAN RESILIENCE ANTICIPATORY MULTI-AGENT PIPELINE FOR %s", self.city_name.upper())
        logger.info("================================================================================")

        # -------------------------------------------------------------------------
        # STAGE 1: Multi-source inputs (Ex-ante causality)
        # -------------------------------------------------------------------------
        logger.info("STAGE 1: Ingesting multi-source inputs and pre-event intelligence...")
        spatial_loader = UrbanSpatialDataLoader(city_name=self.city_name)
        self.urban_context = spatial_loader.load_context()

        mobility_loader = MobilityNetworkLoader(self.urban_context, random_seed=resolved_config.random_seed)
        self.mobility_network = mobility_loader.generate_mobility_dataset(
            total_hours=resolved_config.total_hours,
            start_hour_of_day=resolved_config.start_hour_of_day,
            start_day_of_week=resolved_config.start_day_of_week,
            start_datetime=resolved_config.start_datetime,
        )

        event_agent = EventIntelligenceAgent(self.urban_context)
        self.event_context = event_agent.create_event_context(
            hazard_type=hazard_type,
            event_start_hour=resolved_config.event_start_hour,
            duration_hours=resolved_config.event_duration_hours,
            peak_intensity=resolved_config.peak_intensity,
            total_simulation_hours=resolved_config.total_hours,
        )

        # -------------------------------------------------------------------------
        # STAGE 2 & 3: Multi-Agent Orchestration & Synthesis (SupervisorAgent Single Path)
        # -------------------------------------------------------------------------
        logger.info("STAGE 2 & 3: Running hierarchical orchestration via SupervisorAgent...")
        self.supervisor = SupervisorAgent(
            urban_context=self.urban_context,
            mobility_network=self.mobility_network,
            weight_config=self.weight_config,
            sim_config=resolved_config,
            output_dir=self.output_dir,
            orchestration_mode=active_mode,
            llm_client=active_client,
        )
        manifest_path = (
            Path(output_manifest_path)
            if output_manifest_path
            else self.output_dir / "orchestration" / f"run_pipeline_{resolved_config.random_seed}" / "manifest.json"
        )
        try:
            self.prediction, self.spatial_outlook, zone_features, self.manifest = self.supervisor.run(
                event_context=self.event_context,
                output_manifest_path=manifest_path,
                save_manifest=True,
            )
            self.domain_evidences = self.supervisor.evidence_bus.get_all_evidences()
        except OrchestrationError as exc:
            logger.error("Pipeline aborted: SupervisorAgent run failed: %s", exc)
            self.manifest = self.supervisor.latest_manifest
            raise

        # -------------------------------------------------------------------------
        # STAGE 4: Anticipatory outputs and visualizations
        # -------------------------------------------------------------------------
        logger.info("STAGE 4: Rendering anticipatory disruption curves and spatial outlook...")
        plot_curve_path = self.visualizer.plot_disruption_and_recovery(self.prediction, self.event_context)
        plot_map_path = self.visualizer.plot_spatial_resilience_outlook(self.spatial_outlook, self.urban_context)
        plot_dash_path = self.visualizer.plot_combined_dashboard(
            self.prediction, self.spatial_outlook, self.urban_context, self.event_context
        )

        # Format tabular summaries
        df_timeseries = AnticipatoryPredictionEngine.build_summary_dataframe(self.prediction)
        df_priority = AnticipatoryPredictionEngine.export_spatial_priority_table(self.spatial_outlook)
        indicators = AnticipatoryPredictionEngine.format_key_indicators(self.prediction.disruption_metrics)

        # Save tabular summaries to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df_timeseries.to_csv(self.output_dir / "disruption_recovery_timeseries.csv", index=False)
        df_priority.to_csv(self.output_dir / "spatial_priority_rankings.csv", index=False)

        # -------------------------------------------------------------------------
        # STAGE 5: Historical-event validation, baselines & ablations
        # -------------------------------------------------------------------------
        logger.info("STAGE 5: Performing historical-event validation, benchmark tests, and ablations...")
        evaluator = HistoricalEventEvaluator(
            self.urban_context,
            self.mobility_network,
            lead_time_hours=resolved_config.event_lead_time_hours,
        )
        ground_truth = evaluator.generate_ground_truth_outcomes(self.event_context, total_hours=resolved_config.total_hours)

        eval_metrics = evaluator.evaluate_predictions(
            predicted=self.prediction.predicted_activity,
            ground_truth=ground_truth,
            evaluation_window_start=self.event_context.event_start_hour,
        )

        benchmark_runner = BaselineBenchmarkRunner(
            self.urban_context,
            self.mobility_network,
            lead_time_hours=resolved_config.event_lead_time_hours,
        )
        baseline_comps = benchmark_runner.run_all_baselines(
            event_context=self.event_context,
            ground_truth=ground_truth,
            multi_agent_pred=self.prediction.predicted_activity,
            eval_start_hour=self.event_context.event_start_hour,
        )

        ablation_study = AgentAblationStudy(self.urban_context, self.mobility_network, sim_config=resolved_config)
        ablation_results = ablation_study.run_ablation_experiments(
            evidences=self.domain_evidences,
            event_context=self.event_context,
            ground_truth=ground_truth,
            eval_start_hour=self.event_context.event_start_hour,
        )

        calibration = UncertaintyCalibrator.evaluate_calibration(
            prediction=self.prediction,
            ground_truth=ground_truth,
            nominal_coverage=0.95,
            eval_start_hour=self.event_context.event_start_hour,
        )

        self.validation_result = ValidationResult(
            eval_metrics=eval_metrics,
            baseline_comparisons=baseline_comps,
            ablation_results=ablation_results,
            calibration=calibration,
        )

        logger.info("================================================================================")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("Max Activity Loss: %s", indicators["Maximum Activity Loss"])
        logger.info("Recovery Time: %s", indicators["Recovery Time"])
        logger.info("Model RMSE vs Ground Truth: %.4f | PICP (95%% CI): %.1f%%",
                    eval_metrics["rmse"], calibration.picp_95 * 100)
        logger.info("================================================================================")

        return {
            "prediction": self.prediction,
            "spatial_outlook": self.spatial_outlook,
            "validation_result": self.validation_result,
            "key_indicators": indicators,
            "plot_paths": {
                "curve": str(plot_curve_path),
                "map": str(plot_map_path),
                "dashboard": str(plot_dash_path),
            },
            "priority_table": df_priority,
            "manifest": self.manifest,
            "output_dir": str(self.output_dir),
            "manifest_path": str(manifest_path),
        }
