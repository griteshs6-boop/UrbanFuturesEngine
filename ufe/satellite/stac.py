"""STAC search and asset resolution for the satellite monitor.

Defines the injectable `ImageryBackend` interface (spec Section 18.1 / CONTRACT.md's
`TravelTimeBackend` pattern applied to imagery): the rest of the satellite pipeline
(`indices.py`, `monitor.py`) never talks to pystac-client or stackstac directly, only to this
Protocol, so tests can supply small synthetic rasters instead of hitting a real endpoint.

`StacImageryBackend` is the real implementation, built exactly per spec Section 18.1:
`pystac_client.Client.open` to search a public STAC catalogue, `stackstac.stack` to build the
lazy xarray cube. It performs network I/O and is exercised only by `@pytest.mark.needs_data`
tests, never by the default unit-test run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from ufe.params import Params


@dataclass(frozen=True)
class SceneAssets:
    """One time-slice of band arrays for an AOI, already clipped/resampled to a common grid.

    `bands` keys match `params.value("stac.assets")` (B02, B03, B04, B08, B11, SCL). Arrays are
    2D (y, x), same shape across bands and across scenes for a given project/AOI.
    """

    time: pd.Timestamp
    cloud_cover_pct: float
    bands: dict[str, np.ndarray]


class ImageryBackend(Protocol):
    """Injectable source of Sentinel-2-like scenes for an AOI over a date range.

    Implementations must not raise on an AOI/date range with zero scenes; they return an
    empty list, and callers (indices.py / monitor.py) treat "no scenes" the same as "all
    months invalid" rather than erroring.
    """

    def fetch_scenes(
        self,
        aoi_bounds_4326: tuple[float, float, float, float],
        start: date,
        end: date,
        params: "Params",
    ) -> list[SceneAssets]:
        """Return one entry per available scene intersecting the AOI in [start, end),
        already filtered by `query.max_scene_cloud_cover_pct`. Order is arbitrary — callers
        sort/group by `time` themselves.
        """
        ...


def buffer_aoi_bounds(bounds_4326: tuple[float, float, float, float], buffer_m: float) -> tuple[float, float, float, float]:
    """Buffer a lon/lat bounding box by `buffer_m` metres, done in a local UTM-ish projection
    and reprojected back to EPSG:4326, so a metre buffer is genuinely metres (CONTRACT rule 7:
    never compute distance in degrees)."""
    import pyproj
    from shapely.geometry import box
    from shapely.ops import transform

    # UTM zone/EPSG-code arithmetic below is fixed geodetic convention (a 6-degree zone
    # width, EPSG 326xx/327xx numbering), not a calibrated model coefficient — CONTRACT
    # rule 1 targets tunable parameters, not universal coordinate-system definitions.
    minx, miny, maxx, maxy = bounds_4326
    lon0 = (minx + maxx) / 2.0
    zone = int((lon0 + 180) // 6) + 1
    epsg = 32600 + zone if (miny + maxy) / 2.0 >= 0 else 32700 + zone

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform

    geom_utm = transform(to_utm, box(minx, miny, maxx, maxy))
    buffered_utm = geom_utm.buffer(buffer_m)
    buffered_wgs = transform(to_wgs, buffered_utm)
    return buffered_wgs.bounds


class StacImageryBackend:
    """Real backend: queries a public STAC catalogue and builds a `stackstac` cube.

    Network I/O happens only inside `fetch_scenes`, never at import time or in any other
    module — satisfying CONTRACT rule 3 (no network at simulation time; this is ingestion).
    """

    def fetch_scenes(
        self,
        aoi_bounds_4326: tuple[float, float, float, float],
        start: date,
        end: date,
        params: "Params",
    ) -> list[SceneAssets]:
        import pystac_client
        import stackstac

        endpoint = params.value("stac.endpoint")
        collection = params.value("stac.collection")
        assets = params.value("stac.assets")
        resolution = params.value("stac.resolution_m")
        max_cloud = params.value("query.max_scene_cloud_cover_pct")

        cat = pystac_client.Client.open(endpoint)
        search = cat.search(
            collections=[collection],
            bbox=aoi_bounds_4326,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = list(search.item_collection())
        if not items:
            return []

        cube = stackstac.stack(
            items,
            assets=assets,
            resolution=resolution,
            bounds_latlon=aoi_bounds_4326,
        )

        scenes: list[SceneAssets] = []
        computed = cube.compute()
        for i in range(computed.sizes["time"]):
            time_val = pd.Timestamp(computed["time"].values[i])
            item = items[i]
            cloud_cover = float(item.properties.get("eo:cloud_cover", float("nan")))
            band_arrays = {
                asset: np.asarray(computed.sel(band=asset).isel(time=i).values)
                for asset in assets
            }
            scenes.append(SceneAssets(time=time_val, cloud_cover_pct=cloud_cover, bands=band_arrays))
        return scenes
