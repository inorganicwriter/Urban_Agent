"""Synthetic train/calibration/test protocol with strict event-level separation.

Generates disjoint multi-event synthetic data splits, labels them using the physics simulator,
trains predictors on train split only, calibrates conformal intervals on calibration split only,
and validates on test split only.
All synthetic artifacts are marked engineering_only=True and label_source='synthetic_simulator'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.orchestration.subagent import (
    FlexibilitySubagent,
    FunctionalSubagent,
    PredictabilitySubagent,
    SpatialSubagent,
    TemporalSubagent,
)
from urban_resilience.prediction.calibration import ConformalIntervalCalibrator
from urban_resilience.prediction.features import (
    TARGET_COLUMNS,
    FeatureAssembler,
)
from urban_resilience.prediction.predictor import ResiliencePredictor
from urban_resilience.prediction.train import evaluate_event_holdout
from urban_resilience.types import AgentEvidence, HazardType, UrbanContext

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDatasetSplit:
    """A partition of synthetic events with features and simulation-generated ground truth."""

    split_name: str
    seeds: List[int]
    event_ids: List[str]
    X: pd.DataFrame
    y: pd.DataFrame
    label_source: str = "synthetic_simulator"
    engineering_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "seeds": list(self.seeds),
            "event_ids": list(self.event_ids),
            "sample_count": len(self.X),
            "label_source": self.label_source,
            "engineering_only": self.engineering_only,
        }


@dataclass
class SyntheticDatasetBundle:
    """Consolidated bundle containing strictly disjoint train, calibration, and test splits."""

    train_split: SyntheticDatasetSplit
    calibration_split: SyntheticDatasetSplit
    test_split: SyntheticDatasetSplit
    engineering_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify_disjointness(self) -> bool:
        """Verifies that train, calibration, and test seeds and event IDs are strictly disjoint."""
        train_seeds = set(self.train_split.seeds)
        cal_seeds = set(self.calibration_split.seeds)
        test_seeds = set(self.test_split.seeds)

        if train_seeds & cal_seeds:
            raise RuntimeError(f"Overlap between train and calibration seeds: {train_seeds & cal_seeds}")
        if train_seeds & test_seeds:
            raise RuntimeError(f"Overlap between train and test seeds: {train_seeds & test_seeds}")
        if cal_seeds & test_seeds:
            raise RuntimeError(f"Overlap between calibration and test seeds: {cal_seeds & test_seeds}")

        train_evts = set(self.train_split.event_ids)
        cal_evts = set(self.calibration_split.event_ids)
        test_evts = set(self.test_split.event_ids)

        if train_evts & cal_evts:
            raise RuntimeError(f"Overlap between train and calibration event IDs: {train_evts & cal_evts}")
        if train_evts & test_evts:
            raise RuntimeError(f"Overlap between train and test event IDs: {train_evts & test_evts}")
        if cal_evts & test_evts:
            raise RuntimeError(f"Overlap between calibration and test event IDs: {cal_evts & test_evts}")

        return True


def _generate_synthetic_event_data(
    seed: int,
    urban_context: UrbanContext,
    sim_config: SimulationConfig,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Generates features and simulator ground truth for a single synthetic disaster event."""
    event_id = f"SYNTH_EVENT_SEED_{seed}"

    # Generate synthetic mobility
    mob_loader = MobilityNetworkLoader(urban_context=urban_context, random_seed=seed)
    mobility = mob_loader.generate_mobility_dataset(total_hours=sim_config.total_hours)

    # Generate synthetic hazard event
    event_agent = EventIntelligenceAgent(urban_context=urban_context)
    peak_intensity = float(np.clip(0.65 + (seed % 7) * 0.05, 0.4, 0.98))
    event_ctx = event_agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=sim_config.event_start_hour,
        duration_hours=sim_config.event_duration_hours,
        peak_intensity=peak_intensity,
        total_simulation_hours=sim_config.total_hours,
    )

    # Execute deterministic domain subagents to extract structured evidence
    subagents = {
        "temporal": TemporalSubagent(urban_context, mobility),
        "flexibility": FlexibilitySubagent(urban_context, mobility),
        "predictability": PredictabilitySubagent(urban_context, mobility),
        "spatial": SpatialSubagent(urban_context, mobility),
        "functional": FunctionalSubagent(urban_context, mobility),
    }
    evidences: Dict[str, AgentEvidence] = {}
    for name, sub in subagents.items():
        ev = sub.agent.analyze()
        evidences[name] = ev

    # Assemble canonical features strictly
    assembler = FeatureAssembler(urban_context)
    X_features = assembler.assemble_zone_features(
        evidences,
        hazard_intensity=event_ctx.zone_intensity_map,
        strict=True,
    )

    # Generate ground-truth labels using the physics simulator
    y_rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)

    for zid in urban_context.zone_ids:
        h_intensity = float(event_ctx.zone_intensity_map.get(zid, 0.5))

        # Composite resilience formula matching synthesis simulator
        temp_score = evidences["temporal"].zone_metrics.get(zid, {}).get("temporal_redundancy_score", 0.5)
        flex_score = evidences["flexibility"].zone_metrics.get(zid, {}).get("intraday_flexibility_score", 0.5)
        pred_score = evidences["predictability"].zone_metrics.get(zid, {}).get("predictability_score", 0.5)
        spat_score = evidences["spatial"].zone_metrics.get(zid, {}).get("spatial_redundancy_score", 0.5)
        func_score = evidences["functional"].zone_metrics.get(zid, {}).get("functional_mix_score", 0.5)

        comp_resilience = float(np.clip(
            0.20 * temp_score + 0.20 * flex_score + 0.15 * pred_score + 0.25 * spat_score + 0.20 * func_score,
            0.05, 0.95
        ))
        vulnerability = float(np.clip(h_intensity * (1.15 - comp_resilience), 0.05, 1.0))

        # Peak loss target [0.05, 0.85]
        max_loss = float(np.clip(vulnerability * 0.70 + rng.normal(0, 0.02), 0.02, 0.95))

        # Recovery time target [4, 72 hours]
        recovery_speed = float(np.clip(0.08 + 0.14 * comp_resilience + rng.normal(0, 0.01), 0.04, 0.35))
        recovery_time = float(np.clip(
            -np.log(0.05 / max(max_loss, 0.06)) / recovery_speed + (event_ctx.duration_hours / 2.0),
            2.0, 72.0
        ))

        # Cumulative activity loss
        cumulative_loss = float(max_loss * (event_ctx.duration_hours * 0.5 + recovery_time * 0.5))

        y_rows.append({
            "target_maximum_activity_loss": max_loss,
            "target_recovery_time_hours": recovery_time,
            "target_cumulative_activity_loss": cumulative_loss,
        })

    y_targets = pd.DataFrame(y_rows, index=X_features.index)[TARGET_COLUMNS]

    # Reindex with event-specific index to prevent cross-event collisions
    sample_index = [f"{event_id}_{zid}" for zid in X_features.index]
    X_features.index = sample_index
    y_targets.index = sample_index

    return event_id, X_features, y_targets


def build_synthetic_dataset(
    train_seeds: List[int],
    calibration_seeds: List[int],
    test_seeds: List[int],
    urban_context: Optional[UrbanContext] = None,
    sim_config: Optional[SimulationConfig] = None,
    engineering_only: bool = True,
) -> SyntheticDatasetBundle:
    """Builds strictly event-level disjoint synthetic train, calibration, and test datasets.

    Args:
        train_seeds: Random seeds for training events.
        calibration_seeds: Random seeds for conformal calibration events.
        test_seeds: Random seeds for unseen test events.
        urban_context: Spatial data context. Defaults to Shenzhen.
        sim_config: Simulation config.
        engineering_only: Explicit scientific attribution flag.

    Returns:
        SyntheticDatasetBundle containing the partitioned datasets.
    """
    if not engineering_only:
        raise ValueError(
            "build_synthetic_dataset generates synthetic simulator data and engineering_only MUST be True. "
            "Claiming empirical ground truth on synthetic data without engineering_only=True is strictly prohibited."
        )

    s_train = set(train_seeds)
    s_cal = set(calibration_seeds)
    s_test = set(test_seeds)

    if not s_train:
        raise ValueError("train_seeds cannot be empty.")
    if not s_cal:
        raise ValueError("calibration_seeds cannot be empty.")
    if not s_test:
        raise ValueError("test_seeds cannot be empty.")

    if s_train & s_cal:
        raise ValueError(f"train_seeds and calibration_seeds overlap: {s_train & s_cal}")
    if s_train & s_test:
        raise ValueError(f"train_seeds and test_seeds overlap: {s_train & s_test}")
    if s_cal & s_test:
        raise ValueError(f"calibration_seeds and test_seeds overlap: {s_cal & s_test}")

    ctx = urban_context or UrbanSpatialDataLoader(city_name="Shenzhen").load_context()
    cfg = sim_config or SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=6,
        monte_carlo_samples=5,
    )

    def _build_split(split_name: str, seeds: List[int]) -> SyntheticDatasetSplit:
        x_list: List[pd.DataFrame] = []
        y_list: List[pd.DataFrame] = []
        event_ids: List[str] = []

        for seed in seeds:
            eid, X_evt, y_evt = _generate_synthetic_event_data(seed, ctx, cfg)
            event_ids.append(eid)
            x_list.append(X_evt)
            y_list.append(y_evt)

        X_all = pd.concat(x_list, axis=0)
        y_all = pd.concat(y_list, axis=0)

        return SyntheticDatasetSplit(
            split_name=split_name,
            seeds=list(seeds),
            event_ids=event_ids,
            X=X_all,
            y=y_all,
            label_source="synthetic_simulator",
            engineering_only=engineering_only,
        )

    train_split = _build_split("train", train_seeds)
    cal_split = _build_split("calibration", calibration_seeds)
    test_split = _build_split("test", test_seeds)

    bundle = SyntheticDatasetBundle(
        train_split=train_split,
        calibration_split=cal_split,
        test_split=test_split,
        engineering_only=engineering_only,
        metadata={
            "train_seeds": list(train_seeds),
            "calibration_seeds": list(calibration_seeds),
            "test_seeds": list(test_seeds),
            "city_name": ctx.city_name,
        },
    )
    bundle.verify_disjointness()
    return bundle


def train_and_calibrate_synthetic_predictor(
    dataset_bundle: SyntheticDatasetBundle,
    model_type: str = "ridge",
    output_path: Optional[Union[str, Path]] = None,
    target_coverage: float = 0.95,
) -> Tuple[ResiliencePredictor, ConformalIntervalCalibrator, Dict[str, Any]]:
    """Trains predictor strictly on train_split and calibrates on calibration_split.

    Args:
        dataset_bundle: Pre-built disjoint synthetic dataset bundle.
        model_type: Base regression model type ('ridge', 'hist_gbdt', 'random_forest').
        output_path: Optional path to save serialized model bundle (.joblib).
        target_coverage: Nominal coverage probability for conformal calibration (e.g. 0.95).

    Returns:
        Tuple of (fitted ResiliencePredictor, fitted ConformalIntervalCalibrator, holdout evaluation metrics).
    """
    if not getattr(dataset_bundle, "engineering_only", False):
        raise ValueError(
            "train_and_calibrate_synthetic_predictor requires dataset_bundle.engineering_only=True. "
            "Synthetic training cannot be executed without engineering_only flag."
        )

    dataset_bundle.verify_disjointness()

    # 1. Fit predictor ONLY on train split
    predictor = ResiliencePredictor(model_type=model_type, model_version="1.0.0")
    predictor.fit(dataset_bundle.train_split.X, dataset_bundle.train_split.y)

    # 2. Calibrate conformal intervals ONLY on calibration split
    calibrator = ConformalIntervalCalibrator(target_coverage=target_coverage)
    preds_cal = predictor.predict(dataset_bundle.calibration_split.X)
    calibrator.fit_residuals(dataset_bundle.calibration_split.y, preds_cal)

    # 3. Evaluate holdout performance ONLY on test split
    holdout_metrics = evaluate_event_holdout(
        predictor,
        dataset_bundle.test_split.X,
        dataset_bundle.test_split.y,
    )
    holdout_metrics["engineering_only"] = True
    holdout_metrics["label_source"] = "synthetic_simulator"

    training_meta = {
        "label_source": "synthetic_simulator",
        "engineering_only": True,
        "training_event_ids": list(dataset_bundle.train_split.event_ids),
        "calibration_event_ids": list(dataset_bundle.calibration_split.event_ids),
        "test_event_ids": list(dataset_bundle.test_split.event_ids),
        "target_coverage": target_coverage,
        "holdout_metrics": holdout_metrics,
    }
    predictor.training_metadata = training_meta

    # 4. Save model artifact if requested
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        predictor.save(out_file, calibrator=calibrator, metadata=training_meta)
        logger.info("Saved synthetic predictor and calibrator to %s", out_file)

    return predictor, calibrator, holdout_metrics
