"""Unit tests for validation, baselines, ablations, and calibration."""

from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.agents.temporal_redundancy import TemporalRedundancyAgent
from urban_resilience.agents.intraday_flexibility import IntradayFlexibilityAgent
from urban_resilience.agents.predictability import PredictabilityAgent
from urban_resilience.agents.spatial_redundancy import SpatialRedundancyAgent
from urban_resilience.agents.functional_mix import FunctionalMixAgent
from urban_resilience.agents.resilience_synthesis import ResilienceSynthesisAgent
from urban_resilience.validation.evaluator import HistoricalEventEvaluator
from urban_resilience.validation.baselines import BaselineBenchmarkRunner
from urban_resilience.validation.ablation import AgentAblationStudy
from urban_resilience.validation.calibration import UncertaintyCalibrator
from urban_resilience.types import HazardType


def test_validation_and_benchmarks():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    context = spatial_loader.load_context()
    mobility_loader = MobilityNetworkLoader(context, random_seed=42)
    mob_net = mobility_loader.generate_mobility_dataset(total_hours=48)

    event_agent = EventIntelligenceAgent(context)
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        peak_intensity=0.80,
        total_simulation_hours=48,
    )

    agents_pool = {
        "temporal": TemporalRedundancyAgent(context, mob_net),
        "flexibility": IntradayFlexibilityAgent(context, mob_net),
        "predictability": PredictabilityAgent(context, mob_net),
        "spatial": SpatialRedundancyAgent(context, mob_net),
        "functional": FunctionalMixAgent(context, mob_net),
    }
    evidences = {k: agent.analyze() for k, agent in agents_pool.items()}

    synthesis_agent = ResilienceSynthesisAgent(context, mob_net)
    prediction, _, _ = synthesis_agent.synthesize(evidences, event_ctx)

    evaluator = HistoricalEventEvaluator(context, mob_net)
    ground_truth = evaluator.generate_ground_truth_outcomes(event_ctx, total_hours=48)

    # 1. Evaluate metrics
    eval_m = evaluator.evaluate_predictions(prediction.predicted_activity, ground_truth, evaluation_window_start=12)
    assert "rmse" in eval_m
    assert "mae" in eval_m
    assert eval_m["rmse"] >= 0.0

    # 2. Benchmark baselines
    benchmarks = BaselineBenchmarkRunner(context, mob_net).run_all_baselines(
        event_ctx, ground_truth, prediction.predicted_activity, eval_start_hour=12
    )
    assert len(benchmarks) == 4
    # Ensure Multi-Agent framework has lower RMSE than naive historical mean
    hist_baseline = next(b for b in benchmarks if "Historical" in b.model_name)
    agent_model = next(b for b in benchmarks if "Proposed" in b.model_name)
    assert agent_model.rmse < hist_baseline.rmse

    # 3. Ablation Study
    ablation_study = AgentAblationStudy(context, mob_net)
    ablations = ablation_study.run_ablation_experiments(evidences, event_ctx, ground_truth, eval_start_hour=12)
    assert "Full Framework" in ablations
    assert "w/o Spatial Redundancy" in ablations
    assert len(ablations) == 6

    # 4. Calibration
    calib = UncertaintyCalibrator.evaluate_calibration(prediction, ground_truth, eval_start_hour=12)
    assert 0.0 <= calib.picp_95 <= 1.0
    assert calib.mpiw > 0.0
    assert calib.interval_score > 0.0
