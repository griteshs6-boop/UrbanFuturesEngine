"""Tests for `ufe/geo.py` -- the shared CRS discipline (spec Section 0.3).

Section 0.3, verbatim: "Store all geometry in EPSG:4326. Do all metric computation
(distance, area, buffers) in EPSG:7755 (India TM) or the appropriate UTM zone (43N/44N for
AP). Never compute distance in degrees. There is a test for this." The tests below are
that test, plus coverage of the rest of the `ufe.geo` public surface.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from ufe import geo
from ufe.errors import MissingParameter
from ufe.params import load_params

VIZAG = load_params("vizag")

# Vizag's `crs_metric` (config/cities/vizag.yaml) -- UTM 44N, correct for Andhra Pradesh
# east of ~82.5E (Section 0.3: "the appropriate UTM zone (43N/44N for AP)").
VIZAG_CRS_METRIC = "EPSG:32644"


# ---------------------------------------------------------------- city_metric_crs / config


def test_city_metric_crs_reads_from_city_config_not_a_literal():
    assert geo.city_metric_crs(VIZAG) == VIZAG_CRS_METRIC
    assert geo.city_metric_crs(VIZAG) == VIZAG.city_config["crs_metric"]


def test_city_metric_crs_raises_missing_parameter_when_absent():
    class _NoCrsMetric:
        city_config = {"city_id": "nowhere"}

    with pytest.raises(MissingParameter):
        geo.city_metric_crs(_NoCrsMetric())


# --------------------------------------------------------------------------- is_geographic


def test_is_geographic_true_for_storage_crs():
    assert geo.is_geographic(geo.GEOGRAPHIC_CRS) is True
    assert geo.is_geographic("EPSG:4326") is True


def test_is_geographic_false_for_metric_crs():
    assert geo.is_geographic(VIZAG_CRS_METRIC) is False
    assert geo.is_geographic("EPSG:7755") is False


# ---------------------------------------------------------------------- assert_metric_crs


def test_assert_metric_crs_passes_through_a_projected_crs():
    crs_obj = geo.assert_metric_crs(VIZAG_CRS_METRIC)
    assert crs_obj.is_geographic is False


def test_assert_metric_crs_raises_on_geographic_crs():
    with pytest.raises(geo.NonMetricCRSError):
        geo.assert_metric_crs("EPSG:4326")


# ----------------------------------- ACCEPTANCE (Section 0.3): never compute distance in
# ----------------------------------- degrees -- every metric helper must raise, not warn.


@pytest.mark.acceptance
def test_metric_distance_in_geographic_crs_raises():
    a, b = Point(83.30, 17.71), Point(83.31, 17.72)
    with pytest.raises(geo.NonMetricCRSError):
        geo.metric_distance_m(a, b, geo.GEOGRAPHIC_CRS)


@pytest.mark.acceptance
def test_metric_area_in_geographic_crs_raises():
    square = Polygon([(83.30, 17.71), (83.30, 17.72), (83.31, 17.72), (83.31, 17.71)])
    with pytest.raises(geo.NonMetricCRSError):
        geo.metric_area_sqm(square, geo.GEOGRAPHIC_CRS)


@pytest.mark.acceptance
def test_metric_buffer_in_geographic_crs_raises():
    point = Point(83.30, 17.71)
    with pytest.raises(geo.NonMetricCRSError):
        geo.metric_buffer(point, 1, geo.GEOGRAPHIC_CRS)


@pytest.mark.acceptance
def test_metric_helpers_accept_a_correct_utm_zone_without_raising():
    a, b = Point(83.30, 17.71), Point(83.31, 17.72)
    # Must not raise: EPSG:32644 (UTM 44N) is projected, not geographic.
    distance = geo.metric_distance_m(a, b, VIZAG_CRS_METRIC)
    assert distance > 0


# --------------------------------------------------------------------------- correctness


def test_metric_distance_is_a_sane_metre_value_not_a_degree_value():
    # ~0.01 degrees of latitude is roughly 1.1 km, not ~0.01 (which a degree-space
    # distance call would wrongly return).
    a, b = Point(83.30, 17.71), Point(83.30, 17.72)
    distance = geo.metric_distance_m(a, b, VIZAG_CRS_METRIC)
    assert 500 < distance < 5000


def test_metric_area_of_a_known_square_is_plausible_square_metres():
    # ~0.01deg x 0.01deg near Vizag's latitude is on the order of 1 sq km, not ~1e-4
    # (a degree^2 value), confirming the area was computed after reprojection.
    square = Polygon([(83.30, 17.71), (83.30, 17.72), (83.31, 17.72), (83.31, 17.71)])
    area = geo.metric_area_sqm(square, VIZAG_CRS_METRIC)
    assert 500_000 < area < 2_000_000


def test_metric_buffer_returns_geometry_in_geographic_crs():
    point = Point(83.30, 17.71)
    buffered = geo.metric_buffer(point, 1000, VIZAG_CRS_METRIC)
    # Buffering by 1000m should produce a geometry with a small degree-scale extent,
    # confirming the result was reprojected back to EPSG:4326 for storage (Section 0.3).
    minx, miny, maxx, maxy = buffered.bounds
    assert (maxx - minx) < 1
    assert (maxy - miny) < 1
    assert buffered.contains(point)


def test_to_metric_then_to_geographic_round_trips():
    point = Point(83.30, 17.71)
    projected = geo.to_metric(point, VIZAG_CRS_METRIC)
    assert geo.is_geographic(VIZAG_CRS_METRIC) is False
    back = geo.to_geographic(projected, VIZAG_CRS_METRIC)
    assert back.x == pytest.approx(point.x, abs=1e-9)
    assert back.y == pytest.approx(point.y, abs=1e-9)


def test_metric_helpers_accept_geoseries():
    gs = gpd.GeoSeries(
        [Point(83.30, 17.71), Point(83.31, 17.72)], crs=geo.GEOGRAPHIC_CRS
    )
    projected = geo.to_metric(gs, VIZAG_CRS_METRIC)
    assert geo.is_geographic(projected.crs) is False
    buffered = geo.metric_buffer(gs, 1000, VIZAG_CRS_METRIC)
    assert str(buffered.crs) == str(geo.GEOGRAPHIC_CRS)
    assert not buffered.is_empty.any()
    # Re-derive the metric area explicitly rather than reading `.area` on the
    # geographic-CRS result (which geopandas would rightly warn about).
    assert (geo.metric_area_sqm(buffered, VIZAG_CRS_METRIC) > 0).all()
