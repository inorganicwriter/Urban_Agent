"""Prediction engine formatting and structuring anticipatory output tables and summaries."""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np
import pandas as pd

from urban_resilience.types import (
    DisruptionMetrics,
    ResiliencePrediction,
    SpatialResilienceOutlook,
)

logger = logging.getLogger(__name__)


class AnticipatoryPredictionEngine:
    """Formats and exports numerical tables and diagnostic summaries for anticipatory resilience."""

    @staticmethod
    def build_summary_dataframe(prediction: ResiliencePrediction) -> pd.DataFrame:
        """Constructs a consolidated time-series dataframe of predictions, bounds, and losses."""
        df = pd.DataFrame({
            "hour": prediction.time_hours,
            "baseline_activity": prediction.baseline_activity,
            "predicted_activity": prediction.predicted_activity,
            "lower_bound_95": prediction.lower_bound_95,
            "upper_bound_95": prediction.upper_bound_95,
            "activity_loss": np.maximum(0.0, prediction.baseline_activity - prediction.predicted_activity),
            "uncertainty_width": prediction.upper_bound_95 - prediction.lower_bound_95,
        })
        return df

    @staticmethod
    def format_key_indicators(metrics: DisruptionMetrics) -> Dict[str, str]:
        """Formats the core indicators into human-readable strings."""
        return {
            "Maximum Activity Loss": f"{metrics.maximum_activity_loss * 100:.1f}%",
            "Cumulative Activity Loss": f"{metrics.cumulative_activity_loss:.2f} unit-hours",
            "Recovery Time": f"{metrics.recovery_time_hours:.1f} hours",
            "Peak Loss Hour": f"T+{metrics.peak_loss_hour:02d}h",
        }

    @staticmethod
    def export_spatial_priority_table(outlook: SpatialResilienceOutlook) -> pd.DataFrame:
        """Converts spatial zone resilience objects into an actionable priority ranking table."""
        records = []
        for z in outlook.zone_rankings:
            records.append({
                "Rank": z.vulnerability_rank,
                "Zone ID": z.zone_id,
                "Zone Name": z.zone_name,
                "Vulnerability Score": round(z.vulnerability_score, 3),
                "Composite Resilience Index": round(z.composite_resilience_index, 3),
                "Primary Driver": z.primary_vulnerability_driver,
                "Confidence": round(z.confidence, 2),
            })
        return pd.DataFrame(records)
