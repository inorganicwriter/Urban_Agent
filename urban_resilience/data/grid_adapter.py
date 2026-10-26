"""Spatial grid adapter translating between fine-grained micro-cells and macro urban zones.

Supports multi-scale spatial aggregation (grid cells -> subdistricts -> zones) and
disaggregation (zone predictions -> raster / cell level) with area and population weighting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from urban_resilience.types import UrbanContext

logger = logging.getLogger(__name__)


class MicroGridCell(BaseModel):
    """Specification of a fine-grained spatial micro-grid cell."""

    cell_id: str
    latitude: float
    longitude: float
    zone_id: str
    population_weight: float = 1.0
    area_km2: float = 0.25


class SpatialGridAdapter:
    """Manages mapping between micro-grids and macro urban zones."""

    def __init__(self, urban_context: UrbanContext):
        self.urban_context = urban_context
        self._cells_by_id: Dict[str, MicroGridCell] = {}
        self._zone_to_cells: Dict[str, List[str]] = {z: [] for z in urban_context.zone_ids}

    def register_cells(self, cells: List[MicroGridCell]) -> None:
        """Registers a list of micro-grid cells into the spatial hierarchy."""
        for c in cells:
            self._cells_by_id[c.cell_id] = c
            if c.zone_id in self._zone_to_cells:
                self._zone_to_cells[c.zone_id].append(c.cell_id)
            else:
                self._zone_to_cells[c.zone_id] = [c.cell_id]

        logger.info("SpatialGridAdapter registered %d micro-cells across %d zones.", len(cells), len(self._zone_to_cells))

    def aggregate_cell_series_to_zones(
        self,
        cell_series_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregates micro-cell time series (columns=cell_ids) to macro zone series.

        Returns:
            pd.DataFrame indexed by time with columns matching zone_ids.
        """
        zone_dfs = {}
        for zid, cids in self._zone_to_cells.items():
            valid_cids = [cid for cid in cids if cid in cell_series_df.columns]
            if valid_cids:
                zone_dfs[zid] = cell_series_df[valid_cids].sum(axis=1)
            else:
                # Fallback to zero
                zone_dfs[zid] = pd.Series(0.0, index=cell_series_df.index)

        return pd.DataFrame(zone_dfs, index=cell_series_df.index)

    def disaggregate_zone_metric_to_cells(
        self,
        zone_metric_map: Dict[str, float],
        weight_by: str = "population",
    ) -> Dict[str, float]:
        """Disaggregates macro zone-level metrics down to individual micro-grid cells.

        Args:
            zone_metric_map: Dictionary mapping zone_id to scalar metric (e.g. vulnerability score).
            weight_by: 'population' or 'uniform'.

        Returns:
            Dictionary mapping cell_id to cell-level value.
        """
        cell_values: Dict[str, float] = {}

        for zid, cids in self._zone_to_cells.items():
            val = zone_metric_map.get(zid, 0.0)
            if not cids:
                continue

            if weight_by == "population":
                total_pop = sum(self._cells_by_id[cid].population_weight for cid in cids)
                if total_pop > 0:
                    for cid in cids:
                        w = self._cells_by_id[cid].population_weight / total_pop
                        cell_values[cid] = round(val * w * len(cids), 4)
                else:
                    for cid in cids:
                        cell_values[cid] = val
            else:
                for cid in cids:
                    cell_values[cid] = val

        return cell_values
