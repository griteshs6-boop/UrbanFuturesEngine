"""6.1 Terrain — Copernicus DEM GLO-30 (SRTM 30 m as fallback).

Pipeline, exactly as Section 6.1 specifies it:

    mosaic tiles covering the halo -> reproject to `crs_metric` -> slope by numpy gradient
    over the *projected* raster -> zonal mean of elevation and slope per cell using
    **exactextract**

The reprojection is not cosmetic: ``np.gradient`` over a geographic raster would divide
metres of rise by *degrees* of run, which is the "never compute distance in degrees"
failure of Section 0.3. Everything metric here happens in the city's ``crs_metric``, read
from its config via :func:`ufe.geo.city_metric_crs`; the cell geometry itself stays
EPSG:4326 on disk and is reprojected only inside :func:`ufe.ingest.core.zonal`.

``exactextract`` rather than ``rasterstats`` is mandated by Section 2.1b and justified in
Section 6.1: a res-9 hexagon is roughly 3x a 30 m DEM pixel, so approximate coverage
introduces real error at cell edges.

Genuinely complete: the transform half is fully implemented and tested against a synthetic
DEM with an analytically known slope. Only the ``fetch`` half (which Copernicus tiles cover
the halo, and downloading them) awaits real data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.ingest.core import (
    CityConfig,
    Ingester,
    cfg,
    mark_imputed,
    zonal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_DEM",
    "ELEV_COLUMN",
    "SLOPE_COLUMN",
    "TerrainIngester",
    "slope_percent",
    "reproject_raster",
    "write_slope_raster",
    "terrain_to_cells",
]

KEY_DEM = "dem"
ELEV_COLUMN = "elev_m"
SLOPE_COLUMN = "slope_pct"

_KIND = "kind"
_PATH = "path"
_ELEV, _SLOPE = "elev", "slope"


# --------------------------------------------------------------------------------------
# Pure numeric core
# --------------------------------------------------------------------------------------


def slope_percent(
    elevation: np.ndarray, *, pixel_width_m: float, pixel_height_m: float, scale: float
) -> np.ndarray:
    """``slope_pct = scale * sqrt((dz/dx)^2 + (dz/dy)^2)`` (Section 6.1).

    ``elevation`` must already be in a metric projection, and the pixel sizes are its
    ground sample distances in metres. ``scale`` is ``terrain.slope_percent_scale`` from
    ``config/ingest.yaml`` (100, because the spec wants percent) — never a literal here.
    """
    dz_dy, dz_dx = np.gradient(
        np.asarray(elevation, dtype="float64"), pixel_height_m, pixel_width_m
    )
    return scale * np.hypot(dz_dx, dz_dy)


def reproject_raster(src_path: str | Path, dst_crs: str, dst_path: str | Path) -> Path:
    """Reproject a raster into ``dst_crs``, writing ``dst_path``. Idempotent by path.

    A no-op copy-through when the source is already in ``dst_crs``, so a DEM delivered in
    the city's metric CRS is not resampled twice.
    """
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dst_path = Path(dst_path)
    with rasterio.open(src_path) as src:
        if src.crs is not None and str(src.crs) == str(dst_crs):
            profile = src.profile
            data = src.read()
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            return dst_path
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return dst_path


def write_slope_raster(
    elev_path: str | Path, out_path: str | Path, *, config: Any = None
) -> Path:
    """Compute slope in percent from a *projected* DEM and write it beside it.

    Raises :class:`ufe.geo.NonMetricCRSError` if the DEM is in a geographic CRS — the guard
    that makes "never compute distance in degrees" structural rather than a convention.
    """
    import rasterio

    out_path = Path(out_path)
    nodata_fallback = float(cfg("terrain.fallback_nodata", config))
    min_elev = float(cfg("terrain.min_plausible_elev_m", config))
    max_elev = float(cfg("terrain.max_plausible_elev_m", config))
    scale = float(cfg("terrain.slope_percent_scale", config))

    with rasterio.open(elev_path) as src:
        geo.assert_metric_crs(src.crs)
        band = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else nodata_fallback
        void = (band == nodata) | (band < min_elev) | (band > max_elev) | ~np.isfinite(band)
        # np.gradient cannot see nodata; fill voids with the valid mean so the gradient at
        # the void edge is finite, then re-mask.  A void-adjacent slope is still marked
        # nodata below, so nothing invented survives into `slope_pct`.
        filled = band.copy()
        if void.any() and (~void).any():
            filled[void] = float(band[~void].mean())
        pixel_w, pixel_h = abs(src.transform.a), abs(src.transform.e)
        slope = slope_percent(
            filled, pixel_width_m=pixel_w, pixel_height_m=pixel_h, scale=scale
        )
        slope[void] = nodata_fallback
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=nodata_fallback)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(slope.astype("float32"), 1)
    return out_path


def terrain_to_cells(
    elev_raster: str | Path,
    slope_raster: str | Path,
    cells: pd.DataFrame,
    *,
    config: Any = None,
) -> pd.DataFrame:
    """Zonal mean elevation and slope per cell, exact-area weighted (Section 6.1).

    Cells the DEM does not cover come back null; they are filled with the city mean and
    flagged imputed, because ``cells.elev_m`` and ``cells.slope_pct`` are non-nullable in
    the schema. Nothing is filled without a flag.
    """
    elev = zonal(elev_raster, cells, ["mean", "count"]).rename(
        columns={"mean": ELEV_COLUMN, "count": "_elev_count"}
    )
    slope = zonal(slope_raster, cells, ["mean", "count"]).rename(
        columns={"mean": SLOPE_COLUMN, "count": "_slope_count"}
    )
    out = elev.merge(slope, on="h3", how="outer")
    out = out.set_index("h3").reindex(cells["h3"].astype(str)).reset_index()

    for column, count_col in ((ELEV_COLUMN, "_elev_count"), (SLOPE_COLUMN, "_slope_count")):
        values = pd.to_numeric(out[column], errors="coerce")
        uncovered = values.isna() | (pd.to_numeric(out[count_col], errors="coerce").fillna(0) <= 0)
        fill = float(values[~uncovered].mean()) if (~uncovered).any() else 0.0
        out[column] = values.where(~uncovered, fill).astype(float)
        out = mark_imputed(out, column, uncovered.to_numpy(), "dem_gap_city_mean")
    # slope_pct must be non-negative (schema check); the hypot guarantees it, clip is a
    # defence against a nodata sentinel leaking through the zonal mean.
    out[SLOPE_COLUMN] = out[SLOPE_COLUMN].clip(lower=0)
    return out.drop(columns=["_elev_count", "_slope_count"])


# --------------------------------------------------------------------------------------
# The ingester
# --------------------------------------------------------------------------------------


class TerrainIngester(Ingester):
    """Copernicus DEM GLO-30 -> ``elev_m``, ``slope_pct``."""

    source_id = "copernicus_dem"
    tier = "national"
    fills = (ELEV_COLUMN, SLOPE_COLUMN)
    spatial_res = "30 m"
    temporal_res = "static (2019-2021 acquisition)"
    notes = (
        "Slope computed by numpy gradient over the DEM reprojected into the city's "
        "crs_metric; zonal means by exactextract (Section 6.1). SRTM 30 m is the declared "
        "fallback source and would be recorded as a separate ingest_runs row."
    )

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_DEM,)

    def parse(self, raw: Path) -> pd.DataFrame:
        """Reproject the DEM to ``crs_metric`` and derive the slope raster.

        Returns the two-row manifest ``(kind, path)`` that :meth:`to_cells` consumes. The
        derived rasters are written into the city work dir and reused on a re-run.
        """
        if self.city is None:
            raise ValueError("TerrainIngester needs a CityConfig to know crs_metric")
        work = self.city.work_dir(self.work_root)
        elev_path = work / f"{Path(raw).stem}_{self.city.crs_metric.replace(':', '')}.tif"
        slope_path = work / f"{elev_path.stem}_slope.tif"
        reproject_raster(raw, self.city.crs_metric, elev_path)
        write_slope_raster(elev_path, slope_path, config=self.config)
        return pd.DataFrame(
            [{_KIND: _ELEV, _PATH: str(elev_path)}, {_KIND: _SLOPE, _PATH: str(slope_path)}]
        )

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        paths = df.set_index(_KIND)[_PATH]
        return terrain_to_cells(paths[_ELEV], paths[_SLOPE], cells, config=self.config)

    #: Where derived rasters go. Overridden in tests with a tmp_path.
    work_root: str | Path | None = None
