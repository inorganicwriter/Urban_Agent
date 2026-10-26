"""End-to-end integration test for the entire Urban Resilience Pipeline."""

from pathlib import Path

from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.types import HazardType
from urban_resilience.config import SimulationConfig


def test_pipeline_e2e_execution(tmp_path):
    sim_cfg = SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=12,
        monte_carlo_samples=50,
        random_seed=42,
    )

    pipeline = UrbanResiliencePipeline(city_name="Shenzhen", sim_config=sim_cfg)
    result = pipeline.run_full_pipeline(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=12,
        duration_hours=12,
        peak_intensity=0.85,
    )

    assert result["prediction"] is not None
    assert result["spatial_outlook"] is not None
    assert result["validation_result"] is not None
    assert "Maximum Activity Loss" in result["key_indicators"]
    assert "Recovery Time" in result["key_indicators"]

    # Verify generated plot files exist on disk
    for plot_type, path_str in result["plot_paths"].items():
        p = Path(path_str)
        assert p.exists()
        assert p.stat().st_size > 0
