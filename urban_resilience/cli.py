"""Command-line interface (CLI) for running urban resilience anticipatory analysis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

from urban_resilience.config import SimulationConfig
from urban_resilience.llm.client import LLMClient, get_llm_client
from urban_resilience.orchestration import OrchestrationError
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.prediction.provenance import (
    get_provenance_warning,
    get_synthetic_warning,
)
from urban_resilience.types import HazardType

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="urban-resilience",
        description="Anticipatory Urban Resilience Multi-Agent Prediction CLI",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="Shenzhen",
        help="Target city name (default: Shenzhen)",
    )
    parser.add_argument(
        "--hazard",
        type=str,
        default="rainstorm",
        choices=[h.value for h in HazardType],
        help="Hazard type to simulate (default: rainstorm)",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=24,
        help="Simulation hour when the hazard hits (default: 24)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=18,
        help="Hazard disturbance duration in hours (default: 18)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.85,
        help="Peak hazard intensity factor in [0.1, 1.0] (default: 0.85)",
    )
    parser.add_argument(
        "--total-hours",
        type=int,
        default=96,
        help="Total simulation window in hours (default: 96)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=150,
        help="Monte Carlo draws for uncertainty quantification (default: 150)",
    )
    parser.add_argument(
        "--orchestration-mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "llm_hybrid"],
        help="Supervisor orchestration mode: 'deterministic' (default) or 'llm_hybrid'",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output summary metrics in JSON format to stdout",
    )
    return parser


def main(args=None) -> int:
    """Entry point for the CLI tool."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO if not parsed_args.json_output else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if parsed_args.start_hour < 1:
        parser.error(f"--start-hour must be >= 1, got {parsed_args.start_hour}")
    if parsed_args.duration < 1:
        parser.error(f"--duration must be >= 1, got {parsed_args.duration}")
    if parsed_args.total_hours < 24:
        parser.error(f"--total-hours must be >= 24, got {parsed_args.total_hours}")
    if parsed_args.start_hour + parsed_args.duration > parsed_args.total_hours:
        parser.error(
            f"--start-hour ({parsed_args.start_hour}) + --duration ({parsed_args.duration}) "
            f"exceeds --total-hours ({parsed_args.total_hours})"
        )
    if not (0.1 <= parsed_args.intensity <= 1.0):
        parser.error(f"--intensity must be between 0.1 and 1.0, got {parsed_args.intensity}")
    if parsed_args.mc_samples < 1:
        parser.error(f"--mc-samples must be >= 1, got {parsed_args.mc_samples}")

    try:
        sim_cfg = SimulationConfig(
            total_hours=parsed_args.total_hours,
            event_start_hour=parsed_args.start_hour,
            event_duration_hours=parsed_args.duration,
            monte_carlo_samples=parsed_args.mc_samples,
        )
    except Exception as e:
        parser.error(str(e))

    llm_client: Optional[LLMClient] = None
    if parsed_args.orchestration_mode == "llm_hybrid":
        mode_env = os.environ.get("URBAN_LLM_MODE", "").strip().lower()
        if mode_env in ("mock", "test"):
            llm_client = get_llm_client(mode="mock")
        else:
            api_key = os.environ.get("URBAN_LLM_API_KEY", "").strip()
            if not api_key:
                parser.error(
                    "URBAN_LLM_API_KEY environment variable is required when --orchestration-mode is 'llm_hybrid'"
                )
            llm_client = get_llm_client(mode="env")

    hazard = HazardType(parsed_args.hazard)
    pipeline = UrbanResiliencePipeline(
        city_name=parsed_args.city,
        sim_config=sim_cfg,
        orchestration_mode=parsed_args.orchestration_mode,
        llm_client=llm_client,
    )

    try:
        results = pipeline.run_full_pipeline(
            hazard_type=hazard,
            event_start_hour=parsed_args.start_hour,
            duration_hours=parsed_args.duration,
            peak_intensity=parsed_args.intensity,
        )
    except OrchestrationError as exc:
        logger.error("CLI run aborted due to orchestration error: %s", exc)
        print(f"Error: Multi-agent orchestration failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "city": parsed_args.city,
        "hazard": parsed_args.hazard,
        "indicators": results["key_indicators"],
        "top_vulnerable_zones": results["spatial_outlook"].vulnerable_areas[:4],
        "rmse": round(results["validation_result"].eval_metrics["rmse"], 4),
        "picp_95": round(results["validation_result"].calibration.picp_95, 3),
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
                "interval_source": "monte_carlo_quantile",
            }
        ),
        "predictor_status": getattr(results.get("manifest"), "predictor_status", {}) or {},
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
        "plot_paths": results["plot_paths"],
    }

    if parsed_args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        print("\n================ CLI RUN COMPLETED ================")
        print(f"City: {parsed_args.city} | Hazard: {parsed_args.hazard}")
        if summary.get("synthetic_warning"):
            print(f"  [NOTE] {summary['synthetic_warning']}")
        elif summary.get("provenance_warning"):
            print(f"  [WARNING] {summary['provenance_warning']}")
        for k, v in results["key_indicators"].items():
            print(f"  • {k}: {v}")
        print(f"Top Vulnerable Areas: {', '.join(summary['top_vulnerable_zones'])}")
        print("Generated Plots:")
        for k, p in results["plot_paths"].items():
            print(f"  - {k}: {p}")
        print("===================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
