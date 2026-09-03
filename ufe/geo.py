"""The CRS discipline shared by every module (spec Section 0.3).

Rule, verbatim from the spec: "Store all geometry in EPSG:4326. Do all metric computation
(distance, area, buffers) in EPSG:7755 (India TM) or the appropriate UTM zone (43N/44N for
AP). Never compute distance in degrees. There is a test for this."

This module is the *only* place that talks to `pyproj` directly. Every other module that
needs a distance, an area or a buffer imports the helpers below instead of reprojecting by
hand, so the "never compute distance in degrees" rule has exactly one enforcement point.

Public API
----------
``GEOGRAPHIC_CRS``      -- the storage CRS, ``"EPSG:4326"`` (Section 0.3).
``NonMetricCRSError``   -- raised by the guard below.
``is_geographic(crs)``  -- True if `crs` uses degree units.
``assert_metric_crs(crs)`` -- the guard: raises ``NonMetricCRSError`` on a geographic CRS,
    otherwise returns the parsed ``pyproj.CRS``.
``city_metric_crs(params)`` -- a city's ``crs_metric``, read from its config; never a
    Python literal.
``to_metric(geom, crs_metric, source_crs=GEOGRAPHIC_CRS)`` -- reproject into `crs_metric`.
``to_geographic(geom, crs_metric)`` -- reproject a `crs_metric` geometry back to 4326.
``metric_area_sqm(geom, crs_metric, source_crs=GEOGRAPHIC_CRS)`` -- area in the metric CRS.
``metric_distance_m(a, b, crs_metric, source_crs=GEOGRAPHIC_CRS)`` -- distance in the
    metric CRS.
``metric_buffer(geom, distance_m, crs_metric, source_crs=GEOGRAPHIC_CRS)`` -- buffer in the
    metric CRS, returned back in ``GEOGRAPHIC_CRS`` (geometry is always stored in 4326).

`geom` / `a` / `b` above accept a bare Shapely geometry, a ``geopandas.GeoSeries`` or a
``geopandas.GeoDataFrame`` -- whichever the caller already has.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, Union

import geopandas as gpd
import pyproj
from shapely import ops as shapely_ops
from shapely.geometry.base import BaseGeometry

from ufe.errors import MissingParameter, NonMetricCRSError
from ufe.store.schemas import GEOMETRY_CRS

logger = logging.getLogger(__name__)

__all__ = [
    "GEOGRAPHIC_CRS",
    "NonMetricCRSError",
    "is_geographic",
    "assert_metric_crs",
    "city_metric_crs",
    "to_metric",
    "to_geographic",
    "metric_area_sqm",
    "metric_distance_m",
    "metric_buffer",
]

#: Section 0.3 -- everything on disk (and everything passed into this module by default)
#: is EPSG:4326. Re-exported from `ufe.store.schemas` so there is exactly one source of
#: truth for the storage CRS string.
GEOGRAPHIC_CRS = GEOMETRY_CRS

GeomT = TypeVar("GeomT", bound=Union[BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame])


# `NonMetricCRSError` now lives in the shared hierarchy in `ufe/errors.py` (CONTRACT.md:
# "Custom exceptions live in ufe/errors.py"). It is re-exported here — and named in
# `__all__` above — so `from ufe.geo import NonMetricCRSError` keeps working.


def is_geographic(crs: Any) -> bool:
    """True when `crs` uses degree (lat/lon) units rather than metres."""
    return pyproj.CRS.from_user_input(crs).is_geographic


def assert_metric_crs(crs: Any) -> pyproj.CRS:
    """The Section 0.3 guard. Raises :class:`NonMetricCRSError` when `crs` is geographic.

    Every metric helper below calls this before doing any area/distance/buffer math, so a
    caller can never silently compute a metric quantity in degrees -- they get an exception
    naming the offending CRS instead.
    """
    crs_obj = pyproj.CRS.from_user_input(crs)
    if crs_obj.is_geographic:
        raise NonMetricCRSError(
            f"refusing a metric operation in geographic CRS {crs_obj.to_string()!r}. "
            "Reproject to the city's crs_metric (spec Section 0.3) before computing "
            "distance, area or a buffer."
        )
    return crs_obj


def city_metric_crs(params: Any) -> str:
    """A city's `crs_metric` (EPSG:7755 or the correct UTM zone), read from its config.

    Never a Python literal (CONTRACT.md rule 1): the caller is expected to hold a
    ``ufe.params.Params`` for the city, and this reads its ``city_config['crs_metric']``.
    Raises :class:`ufe.errors.MissingParameter` if the city config declares none.
    """
    city_config = params.city_config
    crs_metric = city_config.get("crs_metric")
    if not crs_metric:
        raise MissingParameter(
            f"city config for {city_config.get('city_id', '<unknown>')!r} has no "
            "'crs_metric' (spec Section 0.3); cannot do metric geometry."
        )
    return crs_metric


def _reproject(geom: GeomT, target_crs: Any, source_crs: Any) -> GeomT:
    if isinstance(geom, (gpd.GeoDataFrame, gpd.GeoSeries)):
        current = geom.crs
        if current is None:
            geom = geom.set_crs(source_crs)
        return geom.to_crs(target_crs)
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_ops.transform(transformer.transform, geom)


def to_metric(
    geom: GeomT, crs_metric: Any, *, source_crs: Any = GEOGRAPHIC_CRS
) -> GeomT:
    """Reproject `geom` from `source_crs` (default EPSG:4326) into `crs_metric`.

    Raises :class:`NonMetricCRSError` if `crs_metric` is itself geographic.
    """
    assert_metric_crs(crs_metric)
    return _reproject(geom, crs_metric, source_crs)


def to_geographic(geom: GeomT, crs_metric: Any) -> GeomT:
    """Reproject a `crs_metric` geometry back to `GEOGRAPHIC_CRS` for storage."""
    assert_metric_crs(crs_metric)
    return _reproject(geom, GEOGRAPHIC_CRS, crs_metric)


def metric_area_sqm(geom: GeomT, crs_metric: Any, *, source_crs: Any = GEOGRAPHIC_CRS) -> Any:
    """Area of `geom` in square metres, computed in `crs_metric` -- never in degrees."""
    projected = to_metric(geom, crs_metric, source_crs=source_crs)
    if isinstance(projected, gpd.GeoDataFrame):
        return projected.geometry.area
    if isinstance(projected, gpd.GeoSeries):
        return projected.area
    return projected.area


def metric_distance_m(
    a: GeomT, b: GeomT, crs_metric: Any, *, source_crs: Any = GEOGRAPHIC_CRS
) -> Any:
    """Distance between `a` and `b` in metres, computed in `crs_metric` -- never degrees."""
    pa = to_metric(a, crs_metric, source_crs=source_crs)
    pb = to_metric(b, crs_metric, source_crs=source_crs)
    return pa.distance(pb)


def metric_buffer(
    geom: GeomT, distance_m: float, crs_metric: Any, *, source_crs: Any = GEOGRAPHIC_CRS
) -> GeomT:
    """Buffer `geom` outward by `distance_m` metres in `crs_metric`.

    The result is reprojected back to `GEOGRAPHIC_CRS` before being returned, since
    geometry is always stored in EPSG:4326 (Section 0.3).
    """
    projected = to_metric(geom, crs_metric, source_crs=source_crs)
    if isinstance(projected, gpd.GeoDataFrame):
        buffered = projected.copy()
        buffered[buffered.geometry.name] = projected.geometry.buffer(distance_m)
    elif isinstance(projected, gpd.GeoSeries):
        buffered = projected.buffer(distance_m)
    else:
        buffered = projected.buffer(distance_m)
    return to_geographic(buffered, crs_metric)
