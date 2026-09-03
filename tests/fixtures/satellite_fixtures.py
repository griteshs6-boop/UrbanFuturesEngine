"""Deterministic, in-memory test fixtures for the satellite monitor.

Everything here is seeded and synthetic: no network, no real rasters. Band values are
chosen by solving the NDVI/NDBI/BSI formulas backwards from target index values (see
`bands_from_targets`), so every composite's expected index value is hand-computable and the
fixtures genuinely exercise the index maths, not just a smoke test.

`SimpleParams` is a minimal stand-in for `ufe.params.Params` (dotted-path `.value()` /
`.conf()` lookups against `config/params/satellite.yaml`) used ONLY because `ufe.params`
does not exist on disk yet. Once it lands, tests should prefer `ufe.params.load_params` and
this fixture becomes redundant; kept import-guarded so a real `Params` is used automatically
if importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ufe.satellite.stac import SceneAssets

SATELLITE_YAML_PATH = Path(__file__).resolve().parents[2] / "config" / "params" / "satellite.yaml"


class SimpleParams:
    """Dotted-path lookup over a raw `{value, conf, scope, ...}` leaf tree, matching the
    subset of `ufe.params.Params` this module needs (`.value`, `.conf`)."""

    def __init__(self, tree: dict[str, Any]):
        self._tree = tree

    @classmethod
    def from_yaml(cls, path: Path = SATELLITE_YAML_PATH) -> "SimpleParams":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def _resolve(self, path: str) -> Any:
        node: Any = self._tree
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"parameter path not found: {path}")
            node = node[part]
        return node

    def value(self, path: str) -> Any:
        node = self._resolve(path)
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return node

    def conf(self, path: str) -> str:
        node = self._resolve(path)
        if isinstance(node, dict) and "conf" in node:
            return node["conf"]
        raise KeyError(f"no conf tag at: {path}")


def load_satellite_params() -> Any:
    """Prefer the real `ufe.params.Params` if it exists on disk; otherwise fall back to
    `SimpleParams` reading the same YAML file directly."""
    try:
        from ufe.params import load_params  # type: ignore

        return load_params(satellite_config_path=SATELLITE_YAML_PATH)  # pragma: no cover
    except Exception:
        return SimpleParams.from_yaml()


# Sentinel-2 SCL codes used by the fixtures (must match config/params/satellite.yaml).
SCL_VALID = 4  # vegetation
SCL_CLOUD = 8  # cloud, medium probability


def bands_from_targets(ndvi_t: float, ndbi_t: float, bsi_t: float, b08: float = 0.20) -> dict[str, float]:
    """Solve NDVI/NDBI/BSI backwards for scalar B02/B03/B04/B08/B11 reflectance values that
    hit the given target index values exactly (up to floating point). Derivation:

      NDVI = (B08-B04)/(B08+B04)  =>  B04 = B08*(1-ndvi)/(1+ndvi)
      NDBI = (B11-B08)/(B11+B08)  =>  B11 = B08*(1+ndbi)/(1-ndbi)
      BSI  = (S-T)/(S+T), S = B11+B04, T = B08+B02  =>  T = S*(1-bsi)/(1+bsi)  =>  B02 = T-B08

    B03 (green) is not used by any index computed here; set to the mean of B02 and B04 for
    a plausible-looking scene.
    """
    b04 = b08 * (1 - ndvi_t) / (1 + ndvi_t)
    b11 = b08 * (1 + ndbi_t) / (1 - ndbi_t)
    s = b11 + b04
    t = s * (1 - bsi_t) / (1 + bsi_t)
    b02 = t - b08
    b03 = (b02 + b04) / 2
    return {"B02": b02, "B03": b03, "B04": b04, "B08": b08, "B11": b11}


# Hand-verified state presets (see module docstring derivation). Baseline is "none".
STATE_TARGETS: dict[str, dict[str, float]] = {
    "none": {"ndvi": 0.70, "ndbi": -0.40, "bsi": -0.30},
    "cleared": {"ndvi": 0.50, "ndbi": -0.40, "bsi": -0.28},
    "earthworks": {"ndvi": 0.30, "ndbi": -0.35, "bsi": -0.10},
    "structure": {"ndvi": 0.10, "ndbi": 0.00, "bsi": 0.05},
}


def make_scene(
    state: str,
    time: pd.Timestamp,
    rng: np.random.Generator,
    shape: tuple[int, int] = (6, 6),
    cloud_frac: float = 0.0,
    cloud_cover_pct: float = 0.0,
) -> SceneAssets:
    """Build one synthetic scene: every clear pixel gets the exact band values for `state`
    (from `STATE_TARGETS` / `bands_from_targets`); `cloud_frac` of pixels (chosen by the
    seeded `rng`, deterministic given the generator's state) are flagged cloud in SCL and
    get a distinct, deliberately wrong band value so an un-masked pipeline would misclassify
    — proving the cloud mask is actually doing something.
    """
    targets = STATE_TARGETS[state]
    band_values = bands_from_targets(targets["ndvi"], targets["ndbi"], targets["bsi"])

    n_pixels = shape[0] * shape[1]
    n_cloud = int(round(cloud_frac * n_pixels))
    cloud_flat = np.zeros(n_pixels, dtype=bool)
    if n_cloud > 0:
        idx = rng.choice(n_pixels, size=n_cloud, replace=False)
        cloud_flat[idx] = True
    cloud_mask_2d = cloud_flat.reshape(shape)

    scl = np.where(cloud_mask_2d, SCL_CLOUD, SCL_VALID).astype(np.int16)
    bands: dict[str, np.ndarray] = {"SCL": scl}
    for band, val in band_values.items():
        arr = np.full(shape, val, dtype=float)
        # Cloudy pixels get a garbage value far outside any real reflectance range, so if
        # the cloud mask were ever skipped the composite would be visibly wrong, not just
        # subtly biased.
        arr[cloud_mask_2d] = 9.99
        bands[band] = arr

    return SceneAssets(time=time, cloud_cover_pct=cloud_cover_pct, bands=bands)


@dataclass(frozen=True)
class SyntheticSite:
    project_id: str
    aoi_bounds_4326: tuple[float, float, float, float]
    announced_date: date
    true_transition_month: pd.Timestamp | None  # first month truly at "cleared" or beyond


class ScriptedImageryBackend:
    """`ImageryBackend` implementation serving pre-built `SceneAssets`, keyed by the exact
    AOI bounds tuple each project was given (there is no project_id in the `ImageryBackend`
    protocol by design — bounds are the only stable key an AOI-scoped backend has)."""

    def __init__(self, scenes_by_aoi: dict[tuple[float, float, float, float], list[SceneAssets]]):
        self._scenes_by_aoi = scenes_by_aoi

    def fetch_scenes(self, aoi_bounds_4326, start, end, params):
        scenes = self._scenes_by_aoi.get(tuple(aoi_bounds_4326), [])
        max_cloud = params.value("query.max_scene_cloud_cover_pct")
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return [
            s for s in scenes
            if start_ts <= s.time < end_ts and s.cloud_cover_pct < max_cloud
        ]


def _monthly_timestamps(start: date, n_months: int) -> list[pd.Timestamp]:
    return [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(n_months)]


def build_labelled_sites(
    seed: int = 20260903,
) -> tuple[list[SyntheticSite], ScriptedImageryBackend]:
    """Build the 10-site labelled fixture for the Section 18 ACCEPTANCE block: known
    construction start dates, one scene per month for 3 years (12mo pre-announcement
    baseline + 24mo monitored window), deterministic from `seed`.

    Sites 0-7: clean transition to "cleared" exactly at the announced construction month
    (well inside the +/-4 month acceptance tolerance).
    Site 8: transition happens, but is masked by 5 months of >40%-cloud composites right
    at the transition, pushing first-detected-transition 5 months late — deliberately
    OUTSIDE the +/-4 month tolerance, so the acceptance test is genuinely discriminating
    rather than trivially all-pass.
    Site 9: no activity for the full 3 years (never advances past "none"); used by the
    third ACCEPTANCE bullet, included here too as a negative control.
    """
    rng = np.random.default_rng(seed)
    sites: list[SyntheticSite] = []
    scenes_by_aoi: dict[tuple[float, float, float, float], list[SceneAssets]] = {}

    n_pre = 12
    n_post = 24
    n_months = n_pre + n_post

    for i in range(10):
        lon0 = 83.0 + i * 0.05
        aoi = (lon0, 17.0, lon0 + 0.01, 17.01)
        window_start = date(2023, 1, 1)
        announced = date(2024, 1, 1)  # month index n_pre
        months = _monthly_timestamps(window_start, n_months)

        scenes: list[SceneAssets] = []
        transition_month: pd.Timestamp | None = None

        if i == 9:
            # negative control: no activity, ever.
            for m in months:
                scenes.append(make_scene("none", m, rng, cloud_frac=0.0))
        elif i == 8:
            # clean baseline, transition at n_pre, but heavily clouded for 5 months from
            # the transition month onward.
            transition_month = months[n_pre]
            for idx, m in enumerate(months):
                state = "none" if idx < n_pre else "cleared"
                cloud = 0.9 if n_pre <= idx < n_pre + 5 else 0.0
                scenes.append(make_scene(state, m, rng, cloud_frac=cloud))
        else:
            transition_month = months[n_pre]
            for idx, m in enumerate(months):
                state = "none" if idx < n_pre else "cleared"
                scenes.append(make_scene(state, m, rng, cloud_frac=0.0))

        scenes_by_aoi[aoi] = scenes
        sites.append(
            SyntheticSite(
                project_id=f"site_{i:02d}",
                aoi_bounds_4326=aoi,
                announced_date=announced,
                true_transition_month=transition_month,
            )
        )

    return sites, ScriptedImageryBackend(scenes_by_aoi)
