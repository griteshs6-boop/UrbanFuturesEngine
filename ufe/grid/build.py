"""Module 1 -- grid construction (spec Section 5).

Builds the res-9 H3 simulation grid covering a city boundary plus its analysis halo
(Section 5.1), and the res-8 parent map used for reporting aggregation and as the
destination grid. Only the geometry columns this module owns are produced here --
``h3, h3_res8, in_city, geometry, lat, lon, area_sqm`` -- everything else in
``SCHEMAS['cells']`` is filled in by Module 2's ingest layers (Section 5.1 step 6).

Every numeric constant this module needs (the simulation resolution, the report/parent
resolution, the halo buffer distance) is read from YAML via ``ufe.params`` -- never a
Python literal (CONTRACT.md rule 1). If a path does not exist in the parameter tree,
``Params.value`` raises ``ufe.errors.MissingParameter`` -- that is the "clearly-named
lookup that fails loudly" the build brief asks for, not a fallback default coded here.

``accessibility.grid.halo_buffer_m`` (Section 5.1 step 2's "buffer it outward by 50_000 metres") is not
currently present in ``config/params/accessibility.yaml``'s ``grid:`` block, which today
only carries ``sim_resolution``, ``report_resolution``, ``dest_resolution``,
``dest_max_count`` and ``dest_min_jobs``. This is flagged in the build report; the lookup
below simply surfaces that gap loudly rather than hardcoding 50_000 in Python.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ufe import geo
from ufe.errors import UFEError

logger = logging.getLogger(__name__)

__all__ = [
    "GridSizingError",
    "SIM_RESOLUTION_PATH",
    "REPORT_RESOLUTION_PATH",
    "HALO_BUFFER_PATH",
    "load_boundary",
    "sim_resolution",
    "report_resolution",
    "halo_buffer_m",
    "cell_ids_for_boundary",
    "build_grid",
]

#: Section 5.1 step 3: `h3.polygon_to_cells(boundary_buffered, res=9)`. Read from YAML.
SIM_RESOLUTION_PATH = "accessibility.grid.sim_resolution"
#: Section 5.1 step 5: the res-8 parent map, used for reporting and as the dest grid.
REPORT_RESOLUTION_PATH = "accessibility.grid.report_resolution"
#: Section 5.1 step 2: the "analysis halo" buffer distance in metres. See module note --
#: this path does not currently resolve; `halo_buffer_m` raises `MissingParameter`.
HALO_BUFFER_PATH = "accessibility.grid.halo_buffer_m"

#: H3 v4 containment mode for `polygon_to_cells_experimental`. `"overlap"` includes every
#: cell that touches the polygon at all, which is what the Section 5 ACCEPTANCE
#: requirement "union of cell polygons contains the boundary, no gaps" needs -- the
#: default `"center"` mode (a cell counts only if its centroid falls inside the polygon)
#: can leave slivers uncovered along the boundary. This is a categorical mode name, not a
#: numeric parameter.
_CONTAIN_MODE = "overlap"


class GridSizingError(UFEError):
    """The built grid's cell count or area falls outside the Section 5.2 sanity band."""


def load_boundary(path: str | Path) -> BaseGeometry:
    """Load a city boundary polygon (Section 5.1 step 1).

    Returns the union of every feature's geometry, reprojected to EPSG:4326 if the file
    declares a different CRS.
    """
    path = Path(path)
    gdf = gpd.read_file(path)
    if gdf.crs is not None and str(gdf.crs) != str(geo.GEOGRAPHIC_CRS):
        gdf = gdf.to_crs(geo.GEOGRAPHIC_CRS)
    return unary_union(gdf.geometry.to_numpy())


def sim_resolution(params: Any) -> int:
    """The H3 resolution of the simulation grid (Section 5.1 step 3), from YAML."""
    return int(params.value(SIM_RESOLUTION_PATH))


def report_resolution(params: Any) -> int:
    """The H3 parent resolution for reporting/destinations (Section 5.1 step 5)."""
    return int(params.value(REPORT_RESOLUTION_PATH))


def halo_buffer_m(params: Any) -> float:
    """The Section 5.1 step 2 analysis-halo buffer distance, in metres, from YAML.

    Raises ``ufe.errors.MissingParameter`` today -- see the module docstring.
    """
    return float(params.value(HALO_BUFFER_PATH))


def cell_ids_for_boundary(boundary: BaseGeometry, resolution: int) -> list[str]:
    """The sorted, deduplicated set of H3 cell ids covering `boundary` at `resolution`.

    Uses the experimental `"overlap"` containment mode so the returned cell set has no
    gaps against `boundary` (Section 5 ACCEPTANCE). Sorting makes the result deterministic
    (Section 5.2 ACCEPTANCE: "re-running produces an identical cell set").
    """
    shape = h3.geo_to_h3shape(boundary)
    cells = h3.polygon_to_cells_experimental(shape, resolution, contain=_CONTAIN_MODE)
    return sorted(set(cells))


def _cell_polygon_lonlat(h3_id: str) -> shapely.Polygon:
    """A cell's boundary as a Shapely polygon in (lon, lat) order (EPSG:4326)."""
    return shapely.Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h3_id)])


def build_grid(boundary: BaseGeometry, params: Any) -> pd.DataFrame:
    """Build the Section 5.1 simulation grid for `boundary`.

    Steps (Section 5.1):
    1. `boundary` is the city boundary polygon (EPSG:4326), already loaded by the caller
       (e.g. via `load_boundary`).
    2. Buffered outward by `halo_buffer_m(params)` metres in the city's `crs_metric` --
       the analysis halo.
    3. `h3.polygon_to_cells` on the buffered polygon at `sim_resolution(params)`.
    4. Centroid (`h3.cell_to_latlng`) and true area (`h3.cell_to_boundary` projected to
       `crs_metric`) per cell.
    5. The res-`report_resolution(params)` parent of every cell.
    6. Returns geometry columns only: `h3, h3_res8, in_city, geometry, lat, lon, area_sqm`
       -- sorted by `h3` for determinism. `in_city` is set by an independent
       point-in-polygon test of each cell's centroid against the *unbuffered* `boundary`.

    All other `SCHEMAS['cells']` columns are filled by Module 2.
    """
    crs_metric = geo.city_metric_crs(params)
    res = sim_resolution(params)
    parent_res = report_resolution(params)
    distance_m = halo_buffer_m(params)

    buffered = geo.metric_buffer(boundary, distance_m, crs_metric)

    cell_ids = cell_ids_for_boundary(buffered, res)
    if not cell_ids:
        raise GridSizingError(
            "polygon_to_cells produced zero cells for the given boundary; check the "
            "boundary geometry and sim_resolution"
        )

    centroids = [h3.cell_to_latlng(cell_id) for cell_id in cell_ids]
    lats = np.array([lat for lat, _ in centroids])
    lons = np.array([lng for _, lng in centroids])

    polygons = [_cell_polygon_lonlat(cell_id) for cell_id in cell_ids]
    polygons_gs = gpd.GeoSeries(polygons, crs=geo.GEOGRAPHIC_CRS)
    areas = geo.metric_area_sqm(polygons_gs, crs_metric).to_numpy()
    geometry_wkb = shapely.to_wkb(np.asarray(polygons_gs, dtype=object))

    h3_res8 = [h3.cell_to_parent(cell_id, parent_res) for cell_id in cell_ids]

    centroid_points = gpd.points_from_xy(lons, lats)
    in_city = shapely.covers(boundary, centroid_points)

    frame = pd.DataFrame(
        {
            "h3": cell_ids,
            "h3_res8": h3_res8,
            "in_city": in_city,
            "geometry": list(geometry_wkb),
            "lat": lats,
            "lon": lons,
            "area_sqm": areas,
        }
    )
    return frame.sort_values("h3", kind="stable").reset_index(drop=True)
