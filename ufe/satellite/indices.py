"""Spectral index computation and cloud-aware monthly compositing.

Pure functions only. Nothing here performs I/O or network access; it operates on the
`SceneAssets` records produced by `ufe.satellite.stac` (real or synthetic backend) and on
plain numpy arrays. No numeric literal outside 0/1/array-indices appears below — every
threshold is read from the `Params` object passed in.

Band naming follows Sentinel-2 L2A: B02 (blue), B03 (green), B04 (red), B08 (NIR), B11
(SWIR1), SCL (Scene Classification Layer).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from ufe.params import Params
    from ufe.satellite.stac import SceneAssets


def ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""
    return _safe_ratio(b08 - b04, b08 + b04)


def ndbi(b11: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index = (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return _safe_ratio(b11 - b08, b11 + b08)


def bsi(b02: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Bare Soil Index = ((SWIR1+Red) - (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))."""
    numer = (b11 + b04) - (b08 + b02)
    denom = (b11 + b04) + (b08 + b02)
    return _safe_ratio(numer, denom)


def brightness(b02: np.ndarray, b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """Mean of the three visible bands."""
    return np.mean(np.stack([b02, b03, b04], axis=0), axis=0)


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Elementwise numer/denom, NaN where denom == 0 (never divide by zero)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, numer / denom)
    return out


def cloud_mask(scl: np.ndarray, cloud_codes: Sequence[int]) -> np.ndarray:
    """Boolean array, True where the pixel is cloud/shadow/cirrus per the SCL codes."""
    mask = np.zeros_like(scl, dtype=bool)
    for code in cloud_codes:
        mask |= scl == code
    return mask


@dataclass(frozen=True)
class MonthlyComposite:
    """One AOI-aggregated monthly observation. `valid` is False when the month was dropped
    for excess cloud cover or insufficient clear-pixel coverage — callers must not treat a
    dropped month as a data point (no spurious state change may be derived from it)."""

    month: pd.Timestamp
    ndvi: float
    ndbi: float
    bsi: float
    brightness: float
    cloud_frac: float
    valid: bool


def build_monthly_composites(
    scenes: Sequence["SceneAssets"], params: "Params"
) -> list[MonthlyComposite]:
    """Group scenes by calendar month, cloud-mask each with the SCL band, take the per-pixel
    median across scenes in the month, then reduce to a single AOI-mean value per index.

    A month is `valid=False` (and its index values NaN) when:
      - the mean per-scene cloud fraction over the AOI exceeds `query.max_month_cloud_frac`, or
      - fewer than `query.min_valid_pixel_frac` of AOI pixels have any clear observation.

    This mirrors spec Section 18.1 steps 2-4: "drop months with >40% cloud over the AOI" —
    a fully clouded month must produce no composite, not a spurious change.
    """
    cloud_codes = params.value("scl.cloud_codes")
    max_month_cloud_frac = params.value("query.max_month_cloud_frac")
    min_valid_pixel_frac = params.value("query.min_valid_pixel_frac")

    by_month: dict[pd.Period, list["SceneAssets"]] = {}
    for scene in scenes:
        period = pd.Timestamp(scene.time).to_period("M")
        by_month.setdefault(period, []).append(scene)

    composites: list[MonthlyComposite] = []
    for period in sorted(by_month.keys()):
        month_scenes = by_month[period]
        month_ts = period.to_timestamp(how="start")

        masked_bands: dict[str, list[np.ndarray]] = {b: [] for b in ("B02", "B03", "B04", "B08", "B11")}
        any_clear = None

        for scene in month_scenes:
            scl = scene.bands["SCL"]
            mask = cloud_mask(scl, cloud_codes)
            clear = ~mask
            any_clear = clear if any_clear is None else (any_clear | clear)
            for band in masked_bands:
                arr = scene.bands[band].astype(float).copy()
                arr[mask] = np.nan
                masked_bands[band].append(arr)

        # Cloud fraction "over the AOI" for the month: the fraction of pixels that had NO
        # clear observation from ANY scene all month — i.e. the complement of coverage
        # achieved by combining every scene, not an average of individual scenes' cloud
        # cover (a pixel clear in even one scene is a usable pixel for the median).
        valid_pixel_frac = float(np.mean(any_clear)) if any_clear is not None else 0.0
        month_cloud_frac = 1.0 - valid_pixel_frac

        is_valid = (
            month_cloud_frac <= max_month_cloud_frac and valid_pixel_frac >= min_valid_pixel_frac
        )

        if not is_valid:
            composites.append(
                MonthlyComposite(
                    month=month_ts,
                    ndvi=float("nan"),
                    ndbi=float("nan"),
                    bsi=float("nan"),
                    brightness=float("nan"),
                    cloud_frac=month_cloud_frac,
                    valid=False,
                )
            )
            continue

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # A pixel that is cloud-flagged in every scene of the month is legitimately
            # all-NaN across the stack; numpy's "All-NaN slice" warning is expected noise,
            # not a bug — the resulting NaN correctly propagates into the AOI mean below.
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median_bands = {
                band: np.nanmedian(np.stack(arrs, axis=0), axis=0)
                for band, arrs in masked_bands.items()
            }

        ndvi_arr = ndvi(median_bands["B08"], median_bands["B04"])
        ndbi_arr = ndbi(median_bands["B11"], median_bands["B08"])
        bsi_arr = bsi(median_bands["B02"], median_bands["B04"], median_bands["B08"], median_bands["B11"])
        brightness_arr = brightness(median_bands["B02"], median_bands["B03"], median_bands["B04"])

        composites.append(
            MonthlyComposite(
                month=month_ts,
                ndvi=float(np.nanmean(ndvi_arr)),
                ndbi=float(np.nanmean(ndbi_arr)),
                bsi=float(np.nanmean(bsi_arr)),
                brightness=float(np.nanmean(brightness_arr)),
                cloud_frac=month_cloud_frac,
                valid=True,
            )
        )

    return composites


def composites_to_frame(composites: Sequence[MonthlyComposite]) -> pd.DataFrame:
    """Convert a list of `MonthlyComposite` into a plain, new dataframe."""
    return pd.DataFrame(
        {
            "month": [c.month for c in composites],
            "ndvi": [c.ndvi for c in composites],
            "ndbi": [c.ndbi for c in composites],
            "bsi": [c.bsi for c in composites],
            "brightness": [c.brightness for c in composites],
            "cloud_frac": [c.cloud_frac for c in composites],
            "valid": [c.valid for c in composites],
        }
    )
