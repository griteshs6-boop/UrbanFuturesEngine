"""6.6 Nightlights — VIIRS DNB monthly cloud-free composites, 2012-present.

Section 6.6, in order:

1. per month, zonal mean radiance per cell via **exactextract**;
2. the standard outlier mask: drop values < 0, cap at the 99.9th percentile *per month*;
3. an annual **median** per cell per year — "the median is more stable than the mean for
   this series", so the annual aggregate is a median of the (masked) monthly means;
4. write to ``cells_history``.

``cells.nightlight`` gets the base-year annual median; the whole panel goes to
``cells_history(h3, year, nightlight)``, which the backtest needs.

Genuinely complete: the monthly-mean -> mask -> annual-median chain is implemented and
tested against synthetic monthly rasters with a hand-computable answer, including a
deliberate negative value and a deliberate outlier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe.ingest.core import CityConfig, Ingester, cfg, mark_imputed, zonal

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_PREFIX",
    "NIGHTLIGHT_COLUMN",
    "NightlightsIngester",
    "monthly_key",
    "monthly_means",
    "apply_outlier_mask",
    "annual_median",
    "nightlights_to_history",
    "nightlights_to_cells",
]

KEY_PREFIX = "viirs"
NIGHTLIGHT_COLUMN = "nightlight"

_YEAR, _MONTH, _VALUE = "year", "month", "nightlight"


def monthly_key(year: int, month: int) -> str:
    """Reader key for one monthly composite, e.g. ``viirs/2019-04``."""
    return f"{KEY_PREFIX}/{int(year):04d}-{int(month):02d}"


def monthly_means(
    rasters: Mapping[tuple[int, int], str | Path], cells: pd.DataFrame
) -> pd.DataFrame:
    """Zonal mean radiance per cell per month -> ``(h3, year, month, nightlight)``."""
    frames: list[pd.DataFrame] = []
    for (year, month), raster in sorted(rasters.items()):
        stats = zonal(raster, cells, ["mean"]).rename(columns={"mean": _VALUE})
        stats[_YEAR] = int(year)
        stats[_MONTH] = int(month)
        frames.append(stats[["h3", _YEAR, _MONTH, _VALUE]])
    if not frames:
        return pd.DataFrame(columns=["h3", _YEAR, _MONTH, _VALUE])
    return pd.concat(frames, ignore_index=True)


def apply_outlier_mask(monthly: pd.DataFrame, *, config: Any = None) -> pd.DataFrame:
    """Drop radiance below the floor and cap at the per-month percentile (Section 6.6).

    The cap is applied *per month* because the DNB series is not stationary: a single
    global percentile would clip a bright recent month against a dim 2012 one.
    """
    floor = float(cfg("nightlights.min_valid_radiance", config))
    percentile = float(cfg("nightlights.cap_percentile", config))
    out = monthly.copy()
    out[_VALUE] = pd.to_numeric(out[_VALUE], errors="coerce")
    out.loc[out[_VALUE] < floor, _VALUE] = np.nan

    def _cap(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if not len(valid):
            return group
        return group.clip(upper=float(np.percentile(valid.to_numpy(dtype=float), percentile)))

    out[_VALUE] = out.groupby([_YEAR, _MONTH])[_VALUE].transform(_cap)
    return out


def annual_median(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annual median of the monthly means per cell (Section 6.6)."""
    if not len(monthly):
        return pd.DataFrame(columns=["h3", _YEAR, _VALUE])
    return (
        monthly.groupby(["h3", _YEAR], as_index=False)[_VALUE]
        .median()
        .sort_values(["h3", _YEAR])
        .reset_index(drop=True)
    )


def nightlights_to_history(
    rasters: Mapping[tuple[int, int], str | Path], cells: pd.DataFrame, *, config: Any = None
) -> pd.DataFrame:
    """The ``cells_history(h3, year, nightlight)`` panel."""
    masked = apply_outlier_mask(monthly_means(rasters, cells), config=config)
    return annual_median(masked)


def nightlights_to_cells(
    history: pd.DataFrame, cells: pd.DataFrame, *, base_year: int
) -> pd.DataFrame:
    """``cells.nightlight`` = the base-year annual median, falling back to the latest year."""
    index = pd.Index(cells["h3"].astype(str), name="h3")
    out = pd.DataFrame({"h3": index.to_numpy()})
    if not len(history):
        out[NIGHTLIGHT_COLUMN] = 0.0
        return mark_imputed(
            out, NIGHTLIGHT_COLUMN, np.ones(len(out), dtype=bool), "no_viirs_composites_zero"
        )
    wanted = history[history[_YEAR] == int(base_year)]
    fell_back = False
    if not len(wanted):
        latest = int(history[_YEAR].max())
        logger.warning(
            "no VIIRS composites for base_year %s; using %s instead", base_year, latest
        )
        wanted = history[history[_YEAR] == latest]
        fell_back = True
    values = (
        wanted.set_index("h3")[_VALUE].reindex(index).astype(float)
    )
    missing = values.isna()
    out[NIGHTLIGHT_COLUMN] = values.fillna(0.0).clip(lower=0).to_numpy()
    out = mark_imputed(out, NIGHTLIGHT_COLUMN, missing.to_numpy(), "viirs_gap_zero")
    if fell_back:
        out = mark_imputed(
            out, NIGHTLIGHT_COLUMN, np.ones(len(out), dtype=bool), "viirs_year_substituted"
        )
    return out


class NightlightsIngester(Ingester):
    """VIIRS DNB -> ``cells.nightlight`` and the ``cells_history`` nightlight panel."""

    source_id = "viirs_nightlights"
    tier = "national"
    fills = (NIGHTLIGHT_COLUMN,)
    spatial_res = "~500 m (15 arc-second)"
    temporal_res = "monthly composites, 2012-present; annual median per cell"
    notes = (
        "Monthly zonal means by exactextract, negative radiance dropped, capped at the "
        "99.9th percentile per month, then an annual median per cell (Section 6.6)."
    )

    #: ``{(year, month): raster}`` resolved by :meth:`parse` from the reader.
    def keys(self, city: CityConfig) -> tuple[str, ...]:
        months = int(cfg("nightlights.months_per_year", self.config))
        first = int(cfg("nightlights.first_available_year", self.config))
        return tuple(
            monthly_key(year, month)
            for year in range(first, int(city.base_year) + 1)
            for month in range(1, months + 1)
        )

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        """The first available monthly composite; the rest are read by key in ``parse``."""
        for key in self.keys(city):
            if self.reader.exists(key):
                if not force and key in self._fetched:
                    return self._fetched[key]
                path = self.reader.path(key, force=force)
                self._fetched[key] = path
                return path
        from ufe.ingest.core import MissingSource

        raise MissingSource(f"no VIIRS monthly composite available for {city.city_id}")

    def parse(self, raw: Path) -> pd.DataFrame:
        """Manifest of every available monthly composite: ``(year, month, path)``."""
        if self.city is None:
            raise ValueError("NightlightsIngester needs a CityConfig")
        rows = []
        for key in self.keys(self.city):
            if not self.reader.exists(key):
                continue
            stamp = key.rsplit("/", 1)[-1]
            year, month = stamp.split("-")
            rows.append(
                {
                    _YEAR: int(year),
                    _MONTH: int(month),
                    "path": str(self.reader.path(key)),
                }
            )
        return pd.DataFrame(rows, columns=[_YEAR, _MONTH, "path"])

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("NightlightsIngester needs a CityConfig")
        rasters = {
            (int(row[_YEAR]), int(row[_MONTH])): row["path"] for _, row in df.iterrows()
        }
        history = nightlights_to_history(rasters, cells, config=self.config)
        self.side_tables = {"cells_history": history.rename(columns={_VALUE: "nightlight"})}
        return nightlights_to_cells(history, cells, base_year=self.city.base_year)
