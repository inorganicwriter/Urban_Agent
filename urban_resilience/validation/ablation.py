"""Ablation study engine evaluating the marginal contribution of each domain agent."""

from __future__ import annotations

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from urban_resilience.config import AgentWeightConfig, SimulationConfig
from urban_resilience.types import AgentEvidence, EventContext, MobilityNetwork, UrbanContext
from urban_resilience.agents.resilience_synthesis import ResilienceSynthesisAgent

logger = logging.getLogger(__name__)


class AgentAblationStudy:
    """Systematically ablates individual agents to quantify their marginal value to prediction accuracy."""

    def __init__(
        self,
        urban_context: UrbanContext,
        mobility_network: MobilityNetwork,
        sim_config: Optional[SimulationConfig] = None,
    ):
        self.urban_context = urban_context
        self.mobility_network = mobility_network
        self.sim_config = sim_config or SimulationConfig()

    def run_ablation_experiments(
        self,
        evidences: Dict[str, AgentEvidence],
        event_context: EventContext,
        ground_truth: np.ndarray,
        eval_start_hour: int = 24,
    ) -> Dict[str, Dict[str, float]]:
        """Executes full agent ablation battery: Full, w/o Temporal, w/o Flexibility, w/o Spatial, etc."""
        logger.info("Executing Agent Ablation experiments...")
        y_true = ground_truth[eval_start_hour:]
        results = {}

        # 1. Full Multi-Agent Pipeline
        full_synth = ResilienceSynthesisAgent(self.urban_context, self.mobility_network, sim_config=self.sim_config)
        pred_full, _, _ = full_synth.synthesize(evidences, event_context)
        y_full = pred_full.predicted_activity[eval_start_hour:]
        rmse_full = float(np.sqrt(mean_squared_error(y_true, y_full)))
        mae_full = float(mean_absolute_error(y_true, y_full))
        results["Full Framework"] = {"rmse": rmse_full, "mae": mae_full, "delta_rmse_pct": 0.0}

        # 2. Ablations of each domain agent
        ablation_configs = [
            ("w/o Temporal Redundancy", {"temporal_redundancy": 0.0, "intraday_flexibility": 0.25, "predictability": 0.20, "spatial_redundancy": 0.30, "functional_mix": 0.25}),
            ("w/o Intraday Flexibility", {"temporal_redundancy": 0.25, "intraday_flexibility": 0.0, "predictability": 0.20, "spatial_redundancy": 0.30, "functional_mix": 0.25}),
            ("w/o Predictability", {"temporal_redundancy": 0.25, "intraday_flexibility": 0.25, "predictability": 0.0, "spatial_redundancy": 0.28, "functional_mix": 0.22}),
            ("w/o Spatial Redundancy", {"temporal_redundancy": 0.28, "intraday_flexibility": 0.28, "predictability": 0.18, "spatial_redundancy": 0.0, "functional_mix": 0.26}),
            ("w/o Functional Mix", {"temporal_redundancy": 0.28, "intraday_flexibility": 0.28, "predictability": 0.18, "spatial_redundancy": 0.26, "functional_mix": 0.0}),
        ]

        for label, weights_dict in ablation_configs:
            cfg = AgentWeightConfig(**weights_dict)
            ablated_synth = ResilienceSynthesisAgent(
                self.urban_context, self.mobility_network, weight_config=cfg, sim_config=self.sim_config
            )
            pred_abl, _, _ = ablated_synth.synthesize(evidences, event_context)
            y_abl = pred_abl.predicted_activity[eval_start_hour:]

            rmse_abl = float(np.sqrt(mean_squared_error(y_true, y_abl)))
            mae_abl = float(mean_absolute_error(y_true, y_abl))
            delta_rmse = ((rmse_abl - rmse_full) / rmse_full) * 100.0

            results[label] = {
                "rmse": rmse_abl,
                "mae": mae_abl,
                "delta_rmse_pct": float(delta_rmse),
            }

        return results
