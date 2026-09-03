"""6.3 Buildings — Google-Microsoft Open Buildings (every available vintage).

Section 6.3, step by step:

* spatial join footprints to cells, ``builtup_frac = sum(footprint_area) / area_sqm``;
* **fetch every available vintage**, not just the latest — the historical panel is required
  for backtesting, so each vintage becomes ``cells_history`` rows for its year;
* ``storeys = max(1, round(height_m / 3.2))`` where a building-height raster exists,
  otherwise ``storeys = f(zone_class)`` from a lookup;
* ``floorspace_sqm = footprint_area * storeys``;
* residential/commercial split by ``zone_class`` where no typology layer exists.

Two of those are explicitly weak and both are flagged, per Section 6.3 ("Flag imputed
values in ``data_conf``", "This is a known weak point — mark ``data_conf`` down"):

* no height raster -> ``floorspace_*`` flagged ``storeys_from_zone_class``;
* no typology layer -> ``floorspace_*`` flagged ``res_share_from_zone_class``.

Section 6.3 says the storeys-by-zone lookup lives in ``vizag.yaml``. It does not
(``config/cities/vizag.yaml`` has no such block), so this module reads
``city_config['storeys_by_zone']`` first and falls back to
``buildings.storeys_by_zone_fallback`` in ``config/ingest.yaml``, flagging every affected
cell. The gap is reported rather than papered over.

Footprint area is computed in the city's ``crs_metric`` (Section 0.3) — never in degrees.

Genuinely complete for the footprint-to-``builtup_frac`` transform and the vintage panel.
The storey and typology steps are structurally complete but run on lookups, not data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed, zonal

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_FOOTPRINTS",
    "KEY_HEIGHT",
    "BUILTUP_COLUMN",
    "FLOORSPACE_RES_COLUMN",
    "FLOORSPACE_COM_COLUMN",
    "BuildingsIngester",
    "footprint_area_per_cell",
    "builtup_frac",
    "storeys_per_cell",
    "buildings_to_cells",
    "buildings_to_history",
]

#: ``buildings/<year>`` is the per-vintage key convention; this is the latest vintage.
KEY_FOOTPRINTS = "buildings"
KEY_HEIGHT = "building_height"

BUILTUP_COLUMN = "builtup_frac"
FLOORSPACE_RES_COLUMN = "floorspace_res_sqm"
FLOORSPACE_COM_COLUMN = "floorspace_com_sqm"

_AREA = "_footprint_sqm"


def footprint_area_per_cell(
    footprints: gpd.GeoDataFrame, cells: pd.DataFrame, *, crs_metric: str
) -> pd.Series:
    """Total footprint area (m^2) intersecting each cell, computed in ``crs_metric``.

    A true intersection rather than a centroid join: a warehouse straddling two hexes
    contributes its actual share to each, which matters at res 9.
    """
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    if footprints is None or not len(footprints):
        return pd.Series(0.0, index=pd.Index(cells["h3"].astype(str), name="h3"))
    fp = footprints if footprints.crs is not None else footprints.set_crs(geo.GEOGRAPHIC_CRS)
    fp = geo.to_metric(fp[[fp.geometry.name]], crs_metric)
    pieces = gpd.overlay(
        hexes, fp.reset_index(drop=True), how="intersection", keep_geom_type=False
    )
    if not len(pieces):
        return pd.Series(0.0, index=pd.Index(cells["h3"].astype(str), name="h3"))
    pieces[_AREA] = pieces.geometry.area
    totals = pieces.groupby("h3")[_AREA].sum()
    return totals.reindex(pd.Index(cells["h3"].astype(str), name="h3")).fillna(0.0)


def builtup_frac(
    footprint_sqm: pd.Series, area_sqm: pd.Series, *, config: Any = None
) -> np.ndarray:
    """``sum(footprint_area) / area_sqm``, clipped into [0, 1] (Section 6 ACCEPTANCE)."""
    ceiling = float(cfg("buildings.builtup_frac_ceiling", config))
    area = np.asarray(area_sqm, dtype=float)
    frac = np.divide(
        np.asarray(footprint_sqm, dtype=float),
        area,
        out=np.zeros(len(area), dtype=float),
        where=area > 0,
    )
    return np.clip(frac, 0, ceiling)


def _zone_class(cells: pd.DataFrame) -> np.ndarray:
    """``cells.zone_class``, or empty strings when zoning has not been ingested yet.

    Section 20.2 runs the national tier (step 6) before the city tier (step 8), so a first
    pass legitimately has no ``zone_class``. Every value derived from the absent column is
    flagged imputed regardless, so a missing zoning layer degrades ``data_conf`` rather
    than crashing the tier.
    """
    if "zone_class" in cells.columns:
        return cells["zone_class"].astype(str).to_numpy()
    return np.full(len(cells), "", dtype=object)


def storeys_per_cell(
    cells: pd.DataFrame,
    *,
    height_raster: str | Path | None = None,
    storeys_by_zone: Mapping[str, float] | None = None,
    config: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(storeys, imputed_mask)`` per cell.

    From the height raster where it covers the cell — ``max(1, round(height_m / 3.2))``,
    Section 6.3 — and from the zone lookup otherwise, in which case the mask is True.
    """
    metres_per_storey = float(cfg("buildings.metres_per_storey", config))
    lookup = dict(storeys_by_zone or cfg("buildings.storeys_by_zone_fallback", config))
    zone_default = float(min(lookup.values())) if lookup else 1
    from_zone = np.asarray(
        [float(lookup.get(str(z), zone_default)) for z in _zone_class(cells)], dtype=float
    )

    if height_raster is None:
        return np.maximum(1, np.round(from_zone)), np.ones(len(cells), dtype=bool)

    stats = zonal(height_raster, cells, ["mean", "count"])
    stats = stats.set_index("h3").reindex(cells["h3"].astype(str))
    heights = pd.to_numeric(stats["mean"], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(stats["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    missing = ~np.isfinite(heights) | (counts <= 0)
    from_height = np.maximum(1, np.round(np.nan_to_num(heights) / metres_per_storey))
    storeys = np.where(missing, np.maximum(1, np.round(from_zone)), from_height)
    return storeys, missing


def buildings_to_cells(
    footprints: gpd.GeoDataFrame,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    height_raster: str | Path | None = None,
    storeys_by_zone: Mapping[str, float] | None = None,
    res_share_by_zone: Mapping[str, float] | None = None,
    typology: pd.DataFrame | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """``builtup_frac``, ``floorspace_res_sqm``, ``floorspace_com_sqm`` per cell."""
    h3_index = cells["h3"].astype(str)
    area = footprint_area_per_cell(footprints, cells, crs_metric=crs_metric)
    frac = builtup_frac(area, cells["area_sqm"], config=config)

    storeys, storeys_imputed = storeys_per_cell(
        cells,
        height_raster=height_raster,
        storeys_by_zone=storeys_by_zone,
        config=config,
    )
    floorspace = area.to_numpy(dtype=float) * storeys

    if typology is not None and len(typology):
        shares = (
            typology.set_index("h3")["res_share"]
            .reindex(h3_index)
            .astype(float)
        )
        res_share = shares.fillna(0.0).to_numpy(dtype=float)
        share_imputed = shares.isna().to_numpy()
    else:
        lookup = dict(res_share_by_zone or cfg("buildings.res_share_by_zone_fallback", config))
        default = float(np.mean(list(lookup.values()))) if lookup else 0
        res_share = np.asarray(
            [float(lookup.get(str(z), default)) for z in _zone_class(cells)], dtype=float
        )
        share_imputed = np.ones(len(cells), dtype=bool)

    out = pd.DataFrame(
        {
            "h3": h3_index.to_numpy(),
            BUILTUP_COLUMN: frac,
            FLOORSPACE_RES_COLUMN: floorspace * res_share,
            FLOORSPACE_COM_COLUMN: floorspace * (1 - res_share),
        }
    )
    for column in (FLOORSPACE_RES_COLUMN, FLOORSPACE_COM_COLUMN):
        out = mark_imputed(out, column, storeys_imputed, "storeys_from_zone_class")
        out = mark_imputed(out, column, share_imputed, "res_share_from_zone_class")
    # `builtup_frac` itself is observed wherever the footprint layer covers the cell.
    out = mark_imputed(out, BUILTUP_COLUMN, np.zeros(len(out), dtype=bool), "")
    return out


def buildings_to_history(
    footprints_by_year: Mapping[int, gpd.GeoDataFrame],
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    config: Any = None,
) -> pd.DataFrame:
    """One ``cells_history`` row per (cell, vintage) — the backtest panel of Section 6.3."""
    frames: list[pd.DataFrame] = []
    for year in sorted(footprints_by_year):
        area = footprint_area_per_cell(footprints_by_year[year], cells, crs_metric=crs_metric)
        frames.append(
            pd.DataFrame(
                {
                    "h3": cells["h3"].astype(str).to_numpy(),
                    "year": np.full(len(cells), int(year), dtype=np.int64),
                    BUILTUP_COLUMN: builtup_frac(area, cells["area_sqm"], config=config),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["h3", "year", BUILTUP_COLUMN])
    return pd.concat(frames, ignore_index=True)


class BuildingsIngester(Ingester):
    """Open Buildings -> ``builtup_frac``, ``floorspace_res_sqm``, ``floorspace_com_sqm``."""

    source_id = "google_microsoft_open_buildings"
    tier = "national"
    fills = (BUILTUP_COLUMN, FLOORSPACE_RES_COLUMN, FLOORSPACE_COM_COLUMN)
    spatial_res = "building footprint polygons"
    temporal_res = "per released vintage (every vintage ingested)"
    notes = (
        "builtup_frac by exact polygon intersection in crs_metric. Storeys from a GHSL "
        "building-height raster when available, else from a zone_class lookup; "
        "residential/commercial split from a zone_class lookup when no typology layer "
        "exists. Both fallbacks are flagged imputed and lower data_conf (Section 6.3)."
    )

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_FOOTPRINTS, KEY_HEIGHT)

    def vintage_key(self, year: int) -> str:
        """Reader key for one vintage. Section 6.3: "Fetch every available vintage"."""
        return f"{KEY_FOOTPRINTS}/{int(year)}"

    def available_vintages(self, candidates: "Any") -> tuple[int, ...]:
        """Which of ``candidates`` the reader can actually supply, oldest first.

        The candidate years come from the caller (the CLI passes the release years the
        dataset publishes); the reader is the authority on what is present. Nothing is
        guessed, so a missing vintage shows up as an absent history year rather than as an
        interpolated one.
        """
        return tuple(
            sorted(int(y) for y in candidates if self.reader.exists(self.vintage_key(int(y))))
        )

    def footprints_by_vintage(self, years: "Any") -> dict[int, gpd.GeoDataFrame]:
        """Load every available vintage in ``years`` for :func:`buildings_to_history`."""
        return {
            year: self.reader.vector(self.vintage_key(year))
            for year in self.available_vintages(years)
        }

    def parse(self, raw: Path) -> pd.DataFrame:
        """The footprint layer, read back through the reader that issued ``raw``."""
        return self.reader.vector(KEY_FOOTPRINTS)

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("BuildingsIngester needs a CityConfig to know crs_metric")
        height = self.reader.path(KEY_HEIGHT) if self.reader.exists(KEY_HEIGHT) else None
        return buildings_to_cells(
            df,
            cells,
            crs_metric=self.city.crs_metric,
            height_raster=height,
            storeys_by_zone=self.city.get("storeys_by_zone"),
            res_share_by_zone=self.city.get("res_share_by_zone"),
            config=self.config,
        )
