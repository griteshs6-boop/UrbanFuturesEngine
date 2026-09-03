"""6.9 Cadastral — Bhu-Naksha parcel polygons (state tier, via the adapter).

Per cell: ``parcel_count``, ``mean_parcel_sqm``, ``median_parcel_sqm`` and the Gini
coefficient of parcel size. Only the first two are columns of ``schemas.CELLS``; the median
and the Gini go to the ``cell_parcel_stats`` side table and the schema addition is reported
rather than made (``cells`` is ``strict=True`` and belongs to another module).

Fallback, per Section 6.9: where parcels are unavailable, fragmentation is inferred from the
building-footprint size distribution — ``mean_parcel_sqm ~ mean_footprint_sqm /
plot_coverage_ratio`` — and ``data_conf`` is marked down. Every fallback value is flagged
``inferred_from_footprints``, so Layer 0's parcel-assembly feasibility term can tell the
difference between a surveyed parcel size and an inference from roof outlines.

All areas are computed in the city's ``crs_metric``.

Genuinely complete for the parcel transform and the footprint fallback; parcel *acquisition*
from Bhu-Naksha is a portal read the adapter's ``access_terms()`` does not permit
automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed

logger = logging.getLogger(__name__)

__all__ = [
    "PARCEL_COUNT_COLUMN",
    "MEAN_PARCEL_COLUMN",
    "PARCEL_STATS_TABLE",
    "CadastralIngester",
    "gini",
    "parcels_to_cells",
    "fragmentation_from_footprints",
]

PARCEL_COUNT_COLUMN = "parcel_count"
MEAN_PARCEL_COLUMN = "mean_parcel_sqm"

#: Side table for the two statistics ``schemas.CELLS`` does not declare.
PARCEL_STATS_TABLE = "cell_parcel_stats"

_AREA = "_parcel_sqm"

#: The 2 in the Gini mean-absolute-difference identity
#: ``G = 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n``. A mathematical constant in a
#: closed-form statistic, not a model parameter (CONTRACT.md rule 1).
_GINI_NUMERATOR_FACTOR = 2


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative sample. ``nan`` for an empty sample.

    The standard mean-absolute-difference form, computed on the sorted sample so the result
    is order-independent and therefore deterministic.
    """
    sample = np.sort(np.asarray(values, dtype=float))
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    if n == 0:
        return float("nan")
    total = sample.sum()
    if total <= 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=float)
    return float(
        (_GINI_NUMERATOR_FACTOR * (ranks * sample).sum()) / (n * total) - (n + 1) / n
    )


def parcels_to_cells(
    parcels: gpd.GeoDataFrame, cells: pd.DataFrame, *, crs_metric: str, config: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(cell_columns, parcel_stats)`` from surveyed parcel polygons (Section 6.9).

    A parcel is attributed to the cell containing its centroid — a parcel is a unit of
    ownership, so splitting one across cells would invent parcels that do not exist.
    """
    min_area = float(cfg("cadastral.min_parcel_sqm", config))
    index = pd.Index(cells["h3"].astype(str), name="h3")
    empty_columns = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            PARCEL_COUNT_COLUMN: np.zeros(len(index), dtype=np.int64),
            MEAN_PARCEL_COLUMN: np.nan,
        }
    )
    empty_stats = pd.DataFrame(columns=["h3", "median_parcel_sqm", "parcel_size_gini"])
    if parcels is None or not len(parcels):
        return empty_columns, empty_stats

    layer = parcels if parcels.crs is not None else parcels.set_crs(geo.GEOGRAPHIC_CRS)
    layer = geo.to_metric(layer[[layer.geometry.name]], crs_metric)
    layer[_AREA] = layer.geometry.area
    layer = layer[layer[_AREA] >= min_area]
    if not len(layer):
        return empty_columns, empty_stats
    centroids = gpd.GeoDataFrame(
        {_AREA: layer[_AREA].to_numpy()}, geometry=layer.geometry.centroid, crs=layer.crs
    )
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    joined = gpd.sjoin(centroids, hexes, how="inner", predicate="within")
    grouped = joined.groupby("h3")[_AREA]

    columns = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            PARCEL_COUNT_COLUMN: grouped.size().reindex(index).fillna(0).astype(np.int64).to_numpy(),
            MEAN_PARCEL_COLUMN: grouped.mean().reindex(index).to_numpy(),
        }
    )
    stats = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            "median_parcel_sqm": grouped.median().reindex(index).to_numpy(),
            "parcel_size_gini": grouped.apply(lambda s: gini(s.to_numpy()))
            .reindex(index)
            .to_numpy(),
        }
    )
    no_parcels = columns[PARCEL_COUNT_COLUMN].to_numpy() == 0
    columns = mark_imputed(columns, PARCEL_COUNT_COLUMN, np.zeros(len(columns), bool), "")
    columns = mark_imputed(columns, MEAN_PARCEL_COLUMN, no_parcels, "no_parcels_in_cell")
    return columns, stats


def fragmentation_from_footprints(
    footprints: gpd.GeoDataFrame | None,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 6.9 fallback: infer parcel size from the footprint size distribution.

    ``mean_parcel_sqm = mean_footprint_sqm / plot_coverage_ratio``. Everything produced here
    is flagged ``inferred_from_footprints`` and lowers ``data_conf``; ``parcel_count`` is the
    building count, which is a lower bound on the parcel count, not the parcel count.
    """
    coverage = float(cfg("cadastral.fallback_plot_coverage_ratio", config))
    index = pd.Index(cells["h3"].astype(str), name="h3")
    columns = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            PARCEL_COUNT_COLUMN: np.zeros(len(index), dtype=np.int64),
            MEAN_PARCEL_COLUMN: np.nan,
        }
    )
    stats = pd.DataFrame(
        {"h3": index.to_numpy(), "median_parcel_sqm": np.nan, "parcel_size_gini": np.nan}
    )
    if footprints is not None and len(footprints):
        layer = (
            footprints if footprints.crs is not None else footprints.set_crs(geo.GEOGRAPHIC_CRS)
        )
        layer = geo.to_metric(layer[[layer.geometry.name]], crs_metric)
        layer[_AREA] = layer.geometry.area
        centroids = gpd.GeoDataFrame(
            {_AREA: layer[_AREA].to_numpy()}, geometry=layer.geometry.centroid, crs=layer.crs
        )
        hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
        joined = gpd.sjoin(centroids, hexes, how="inner", predicate="within")
        grouped = joined.groupby("h3")[_AREA]
        columns[PARCEL_COUNT_COLUMN] = (
            grouped.size().reindex(index).fillna(0).astype(np.int64).to_numpy()
        )
        columns[MEAN_PARCEL_COLUMN] = (grouped.mean().reindex(index) / coverage).to_numpy()
        stats["median_parcel_sqm"] = (grouped.median().reindex(index) / coverage).to_numpy()
        stats["parcel_size_gini"] = (
            grouped.apply(lambda s: gini(s.to_numpy())).reindex(index).to_numpy()
        )

    ones = np.ones(len(columns), dtype=bool)
    columns = mark_imputed(columns, PARCEL_COUNT_COLUMN, ones, "inferred_from_footprints")
    columns = mark_imputed(columns, MEAN_PARCEL_COLUMN, ones, "inferred_from_footprints")
    return columns, stats


class CadastralIngester(Ingester):
    """Bhu-Naksha parcels -> ``parcel_count``, ``mean_parcel_sqm`` (+ median, Gini)."""

    source_id = "registration_rera_ec_portals"
    tier = "state"
    fills = (PARCEL_COUNT_COLUMN, MEAN_PARCEL_COLUMN)
    spatial_res = "survey-number parcel polygons"
    temporal_res = "as maintained by the revenue department"
    notes = (
        "Reached through the state adapter. Where the state publishes no parcels, "
        "fragmentation is inferred from the building-footprint size distribution and every "
        "value is flagged inferred_from_footprints (Section 6.9)."
    )

    def __init__(
        self,
        reader: Any,
        *,
        adapter: Any = None,
        city: CityConfig | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reader, city=city, config=config)
        self.adapter = adapter
        self.side_tables: dict[str, pd.DataFrame] = {}
        #: True when the Section 6.9 fallback was used.
        self.used_fallback = False

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        from ufe.ingest.adapters.ap import KEY_PARCELS

        return (KEY_PARCELS,)

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        key = self.keys(city)[0]
        if self.reader.exists(key):
            if not force and key in self._fetched:
                return self._fetched[key]
            path = self.reader.path(key, force=force)
            self._fetched[key] = path
            return path
        return Path(f"unavailable://{key}")

    def parse(self, raw: Path) -> pd.DataFrame:
        """Parcel polygons from the adapter, or an empty frame when the state has none."""
        if self.adapter is None or self.city is None:
            raise ValueError("CadastralIngester needs a StateAdapter and a CityConfig")
        parcels = self.adapter.cadastral_parcels(self.city)
        return parcels if parcels is not None else gpd.GeoDataFrame(geometry=[])

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("CadastralIngester needs a CityConfig")
        crs = self.city.crs_metric
        if df is not None and len(df):
            self.used_fallback = False
            columns, stats = parcels_to_cells(df, cells, crs_metric=crs, config=self.config)
        else:
            self.used_fallback = True
            from ufe.ingest.buildings import KEY_FOOTPRINTS

            footprints = (
                self.reader.vector(KEY_FOOTPRINTS)
                if self.reader.exists(KEY_FOOTPRINTS)
                else None
            )
            logger.warning(
                "no cadastral parcels for %s; falling back to footprint-inferred "
                "fragmentation and marking data_conf down (Section 6.9)",
                self.city.city_id,
            )
            columns, stats = fragmentation_from_footprints(
                footprints, cells, crs_metric=crs, config=self.config
            )
        self.side_tables = {PARCEL_STATS_TABLE: stats}
        return columns
