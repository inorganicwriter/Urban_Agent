"""Visualization module generating publication-quality figures matching the architectural diagram."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from urban_resilience.types import (
    EventContext,
    ResiliencePrediction,
    SpatialResilienceOutlook,
    UrbanContext,
)

logger = logging.getLogger(__name__)

# Configure clean aesthetic styling
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


class UrbanResilienceVisualizer:
    """Renders Disruption & Recovery curves and Spatial Resilience Outlook heatmaps."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_disruption_and_recovery(
        self,
        prediction: ResiliencePrediction,
        event_context: EventContext,
        save_filename: str = "predicted_disruption_recovery.png",
    ) -> Path:
        """Plots the exact Disruption & Recovery curve from Box 4 of the architectural diagram."""
        t = prediction.time_hours
        base = prediction.baseline_activity
        pred = prediction.predicted_activity
        lower = prediction.lower_bound_95
        upper = prediction.upper_bound_95
        metrics = prediction.disruption_metrics

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        # Baseline curve (dashed gray/navy)
        ax.plot(t, base, color="#4A5568", linestyle="--", linewidth=1.8, label="Baseline (no event)", alpha=0.85)

        # Predicted activity curve (solid deep blue)
        ax.plot(t, pred, color="#0D6EFD", linewidth=2.5, label="Predicted activity")

        # Uncertainty range (soft sky blue fill)
        ax.fill_between(t, lower, upper, color="#BEE3F8", alpha=0.55, label="Uncertainty range (95% CI)")

        # Activity loss area (soft amber/yellow fill between baseline and prediction)
        ax.fill_between(
            t, pred, base,
            where=(pred < base),
            color="#FEFCBF",
            edgecolor="#ECC94B",
            linestyle=":",
            alpha=0.75,
            label="Activity loss",
        )

        # Event shock vertical marker
        ax.axvline(
            x=event_context.event_start_hour,
            color="#E53E3E",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.text(
            event_context.event_start_hour + 0.8,
            max(base) * 0.98,
            "Event",
            color="#E53E3E",
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        # Formatting axes
        ax.set_title("Predicted Disruption & Recovery Trajectory", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Time (Hours)", fontsize=11, fontweight="medium")
        ax.set_ylabel("Activity Level", fontsize=11, fontweight="medium")
        ax.set_ylim(0.0, max(upper) * 1.15)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#CBD5E0")

        # Inset Key Metrics Annotation Box
        metrics_text = (
            f"Key Disruption Metrics:\n"
            f"• Maximum activity loss: {metrics.maximum_activity_loss * 100:.1f}%\n"
            f"• Cumulative activity loss: {metrics.cumulative_activity_loss:.1f}\n"
            f"• Recovery time: {metrics.recovery_time_hours:.1f} hours"
        )
        ax.text(
            0.03,
            0.18,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7FAFC", edgecolor="#CBD5E0", alpha=0.95),
        )

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Saved Disruption & Recovery curve to %s", save_path)
        return save_path

    def plot_spatial_resilience_outlook(
        self,
        outlook: SpatialResilienceOutlook,
        urban_context: UrbanContext,
        save_filename: str = "spatial_resilience_outlook.png",
    ) -> Path:
        """Plots the Spatial Resilience Outlook grid map matching Box 4 in the architectural diagram."""
        zones = urban_context.zones
        zone_ids = list(zones.keys())

        # Construct a 2D layout grid for the zones
        # 12 zones organized into a 3 rows x 4 cols spatial schema
        n_zones = len(zone_ids)
        cols = 4
        rows = int(np.ceil(n_zones / cols))

        grid_matrix = np.full((rows, cols), np.nan)
        zone_name_grid = [["" for _ in range(cols)] for _ in range(rows)]
        rank_grid = [["" for _ in range(cols)] for _ in range(rows)]

        rank_map = {z.zone_id: z.vulnerability_rank for z in outlook.zone_rankings}
        score_map = {z.zone_id: z.vulnerability_score for z in outlook.zone_rankings}

        for idx, zid in enumerate(zone_ids):
            r = idx // cols
            c = idx % cols
            grid_matrix[r, c] = score_map.get(zid, 0.5)
            z_short = zones[zid].name.split()[0]
            zone_name_grid[r][c] = f"{zid}\n({z_short})"
            rank_grid[r][c] = f"Rank #{rank_map.get(zid, 99)}"

        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        # Colormap: Higher vulnerability (warm amber/rust red) to Lower vulnerability (deep steel blue)
        im = ax.imshow(grid_matrix, cmap="RdYlBu_r", vmin=0.1, vmax=0.9, aspect="auto")

        # Overlay text inside each grid cell
        for r in range(rows):
            for c in range(cols):
                if not np.isnan(grid_matrix[r, c]):
                    score_val = grid_matrix[r, c]
                    text_color = "white" if (score_val > 0.70 or score_val < 0.30) else "black"
                    cell_text = f"{zone_name_grid[r][c]}\n{rank_grid[r][c]}\nVuln: {score_val:.2f}"
                    ax.text(c, r, cell_text, ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

        # Colorbar with diagram labels
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Vulnerability Scale (Higher vs Lower)", fontsize=11, fontweight="medium")
        cbar.ax.text(1.2, 0.95, "Higher\nvulnerability", transform=cbar.ax.transAxes, color="#9C1C0D", fontweight="bold")
        cbar.ax.text(1.2, 0.05, "Lower\nvulnerability", transform=cbar.ax.transAxes, color="#1D4E89", fontweight="bold")

        ax.set_title("Spatial Resilience Outlook: Vulnerable Areas & Priority Ranking", fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(range(cols))
        ax.set_xticklabels([f"West-East Col {c+1}" for c in range(cols)], fontsize=9)
        ax.set_yticks(range(rows))
        ax.set_yticklabels([f"North-South Row {r+1}" for r in range(rows)], fontsize=9)

        # Legend / footnote
        vulnerable_str = ", ".join(outlook.vulnerable_areas[:4])
        fig.text(
            0.10,
            0.02,
            f"Vulnerable Areas (Top Priorities): {vulnerable_str} | Generated with Quantified Uncertainty",
            fontsize=9.5,
            fontstyle="italic",
            color="#4A5568",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = self.output_dir / save_filename
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Saved Spatial Resilience Outlook map to %s", save_path)
        return save_path

    def plot_combined_dashboard(
        self,
        prediction: ResiliencePrediction,
        outlook: SpatialResilienceOutlook,
        urban_context: UrbanContext,
        event_context: EventContext,
        save_filename: str = "urban_resilience_dashboard.png",
    ) -> Path:
        """Combines time-series disruption and spatial outlook into an executive 2-panel dashboard."""
        fig, (ax_curve, ax_map) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

        # Panel 1: Time Series
        t = prediction.time_hours
        base = prediction.baseline_activity
        pred = prediction.predicted_activity
        lower = prediction.lower_bound_95
        upper = prediction.upper_bound_95

        ax_curve.plot(t, base, color="#4A5568", linestyle="--", linewidth=1.5, label="Baseline (no event)")
        ax_curve.plot(t, pred, color="#0D6EFD", linewidth=2.2, label="Predicted activity")
        ax_curve.fill_between(t, lower, upper, color="#BEE3F8", alpha=0.5, label="Uncertainty range (95% CI)")
        ax_curve.fill_between(t, pred, base, where=(pred < base), color="#FEFCBF", alpha=0.7, label="Activity loss")
        ax_curve.axvline(x=event_context.event_start_hour, color="#E53E3E", linestyle="-.", linewidth=1.4)
        ax_curve.text(event_context.event_start_hour + 0.6, max(base) * 0.95, "Event Shock", color="#E53E3E", fontweight="bold")
        ax_curve.set_title("Predicted Disruption & Recovery", fontsize=12, fontweight="bold")
        ax_curve.set_xlabel("Time (Hours)")
        ax_curve.set_ylabel("Activity Level")
        ax_curve.grid(True, alpha=0.4)
        ax_curve.legend(loc="lower right", fontsize=8.5)

        # Panel 2: Spatial Grid
        zones = urban_context.zones
        zone_ids = list(zones.keys())
        n_zones = len(zone_ids)
        cols = 4
        rows = max(1, int(np.ceil(n_zones / cols)))
        grid = np.full((rows, cols), np.nan)
        score_map = {z.zone_id: z.vulnerability_score for z in outlook.zone_rankings}
        rank_map = {z.zone_id: z.vulnerability_rank for z in outlook.zone_rankings}

        for idx, zid in enumerate(zone_ids):
            r, c = idx // cols, idx % cols
            grid[r, c] = score_map.get(zid, 0.5)
            val = grid[r, c]
            txt_c = "white" if (val > 0.65 or val < 0.35) else "black"
            rank_display = rank_map.get(zid, "N/A")
            ax_map.text(c, r, f"{zid}\nRank #{rank_display}\n{val:.2f}", ha="center", va="center", color=txt_c, fontsize=7.5, fontweight="bold")

        im = ax_map.imshow(grid, cmap="RdYlBu_r", vmin=0.1, vmax=0.9, aspect="auto")
        ax_map.set_title("Spatial Resilience Outlook (Vulnerability Grid)", fontsize=12, fontweight="bold")
        ax_map.set_xticks(range(cols))
        ax_map.set_yticks(range(rows))
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        plt.suptitle(f"Anticipatory Urban Resilience Assessment - {urban_context.city_name}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        save_path = self.output_dir / save_filename
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Saved combined executive dashboard to %s", save_path)
        return save_path
