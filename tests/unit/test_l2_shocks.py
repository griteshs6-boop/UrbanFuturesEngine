"""Tests for Layer 2, shock resolution (spec Section 9).

The Section 9 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "Resolver contains zero archetype names in code" -> ``test_acc_no_archetype_names_in_code``
* "Adding a new archetype to YAML with no code change produces effects"
  -> ``test_acc_new_archetype_from_yaml_only``
* "A 30,000-job electronics project produces `effective_households` under 2,000 and
  `dormitory_workers` over 18,000" -> ``test_acc_dormitory_typology_no_phantom_buyers``
* "The same project as `gcc` produces over 9,000 effective households"
  -> ``test_acc_office_typology_produces_households``
* "Airport wedge: a cell on the CBD bearing gets wedge=1.0; directly opposite gets 0.0"
  -> ``test_acc_airport_wedge_is_directional_not_radial``  (the Section 21
  "Radial airport model" named guard)
* "Exclusive bands: a cell at 300 m from a metro station receives 0.09, not 0.145"
  -> ``test_acc_distance_bands_are_exclusive``  (the Section 21 "Cumulative distance
  bands" named guard)
* "Field cap engages and is logged" -> ``test_acc_field_cap_engages_and_is_logged``

The Section 21 "Double-counted service jobs" named guard is
``test_acc_no_service_employment_emitted_in_layer_2``.

Two archetypes the ACCEPTANCE block names (``gcc``, and any airport archetype) are absent
from ``config/params/archetypes.yaml`` — see the module docstring of ``l2_shocks`` and the
build report.  Rather than edit shipped config, these tests load the real parameter tree
through a *temporary overlay* directory (:func:`overlay_params`) that appends the missing
archetype blocks.  That also gives the "new archetype, no code change" acceptance test its
subject.

Every expected number below is either hand-computed from the shipped YAML (the arithmetic
is written out in the test) or recomputed from ``Params`` — never a magic constant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, Point

from ufe import geo
from ufe.errors import MissingParameter
from ufe.layers import l2_shocks as L2
from ufe.layers import l4_supply as L4
from ufe.params import (
    DEFAULT_CITIES_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_PARAMS_DIR,
    load_params,
)
from ufe.store.schemas import INCOME_BANDS, SECTORS

from tests.fixtures.synthetic import (  # noqa: F401  (registers the session fixture)
    synthetic_city,
)

CITY = "vizag"

#: A year comfortably past `open_year + operational_ramp_years`, so `phase_weight == 1`
#: and `discount == 1` and therefore `activation_weight == p_completion`.  Every test that
#: wants unweighted magnitudes sets `p_completion = 1` and resolves at this year.
ANNOUNCED_YEAR = 2020
OPEN_YEAR = 2030
LATE_YEAR = 2035

#: The three archetypes actually present in the shipped YAML, referred to by role rather
#: than by name so the "zero archetype names" grep stays honest about the *implementation*
#: while the tests remain readable.
TRANSIT = "metro_rail"
DORMITORY_INDUSTRY = "electronics_assembly"

#: Archetype blocks the ACCEPTANCE block requires but the YAML does not ship.  These are
#: TEST FIXTURES, not parameters: they exist only inside a tmp_path overlay.
OVERLAY_ARCHETYPES: dict[str, dict] = {
    # Module 5 ACCEPTANCE: "The same project as `gcc` produces over 9,000 effective
    # households."  An office-sector employer with no dormitory typology.
    "gcc": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "it_office",
            "median_wage_inr_mo": {"value": 60000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {
                "value": 15000,
                "conf": "G",
                "scope": "global",
            },
        },
    },
    # Module 5 ACCEPTANCE / Section 21 "Radial airport model": an airport-shaped archetype
    # whose fields carry the Section 9.3 directional wedge flag.
    "airport_greenfield": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "mppa",
        "network_effect": {"type": "none"},
        "employment": None,
        "directional_wedge": True,
        "premium": [
            {
                "target": "residential",
                "max_m": 20000,
                "value": 0.10,
                "conf": "G",
                "scope": "global",
            }
        ],
    },
    # "Adding a new archetype to YAML with no code change produces effects."
    "_acceptance_new_archetype": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "units_per_year",
        "network_effect": {"type": "none"},
        "employment": None,
        "premium": [
            {
                "target": "residential",
                "max_m": 1000,
                "value": 0.07,
                "conf": "G",
                "scope": "global",
            }
        ],
    },
}


# --------------------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    """The real parameter tree plus :data:`OVERLAY_ARCHETYPES`, loaded from a tmp dir.

    Nothing under ``config/`` is touched.
    """
    target = tmp_path_factory.mktemp("params_overlay")
    for src in sorted(Path(DEFAULT_PARAMS_DIR).glob("*.yaml")):
        (target / src.name).write_text(src.read_text())
    archetypes_path = target / "archetypes.yaml"
    tree = yaml.safe_load(archetypes_path.read_text())
    tree.update(OVERLAY_ARCHETYPES)
    archetypes_path.write_text(yaml.safe_dump(tree, sort_keys=False))
    return load_params(
        CITY,
        params_dir=target,
        cities_dir=DEFAULT_CITIES_DIR,
        classes_file=DEFAULT_CLASSES_FILE,
    )


@pytest.fixture()
def cells(synthetic_city):  # noqa: F811
    return synthetic_city.cells


# --------------------------------------------------------------------------------------
# frame builders — cells and the project template come from tests/fixtures/synthetic.py
# --------------------------------------------------------------------------------------


def _project(
    template: pd.DataFrame,
    *,
    project_id: str,
    archetype: str,
    scale_unit: str,
    scale_value: float,
    geom: str,
    p_completion: float = 1,
    open_year: int = OPEN_YEAR,
    median_wage: float | None = None,
) -> pd.DataFrame:
    """One project row, built by overwriting a synthetic template row."""
    row = template.iloc[[0]].copy(deep=True).reset_index(drop=True)
    row["project_id"] = project_id
    row["archetype"] = archetype
    row["scale_unit"] = scale_unit
    row["scale_value"] = float(scale_value)
    row["geom"] = geom
    row["geom_type"] = "point"
    row["median_wage_inr_mo"] = median_wage
    row["p_completion"] = float(p_completion)
    row["open_year"] = float(open_year)
    row["announced_year"] = float(ANNOUNCED_YEAR)
    return row


@pytest.fixture()
def template(synthetic_city):  # noqa: F811
    return synthetic_city.projects


def _cell_point(cells: pd.DataFrame, i: int = 0) -> Point:
    return Point(float(cells["lon"].iloc[i]), float(cells["lat"].iloc[i]))


def _offset_point(origin: Point, params, *, east_m: float, north_m: float = 0) -> Point:
    """A point exactly `east_m`/`north_m` metres from `origin`, via ``ufe.geo``."""
    crs = geo.city_metric_crs(params)
    projected = geo.to_metric(origin, crs)
    moved = Point(projected.x + float(east_m), projected.y + float(north_m))
    return geo.to_geographic(moved, crs)


def _unit_step_towards(origin: Point, target: Point, params, *, metres: float) -> Point:
    """Walk `metres` from `origin` along the `origin -> target` bearing, in metric CRS."""
    crs = geo.city_metric_crs(params)
    o = geo.to_metric(origin, crs)
    t = geo.to_metric(target, crs)
    dx, dy = t.x - o.x, t.y - o.y
    norm = float(np.hypot(dx, dy))
    return geo.to_geographic(
        Point(o.x + dx / norm * float(metres), o.y + dy / norm * float(metres)), crs
    )


def _cbd_point(params) -> Point:
    cbd = params.city_config["cbd_point"]
    return Point(float(cbd["lon"]), float(cbd["lat"]))


def _cell_with_geom(cells: pd.DataFrame, point: Point, params) -> pd.DataFrame:
    """A one-row `cells` frame whose single cell sits at `point`.

    Keeps every other column from the synthetic fixture, so the frame is still schema
    shaped; only the location moves.
    """
    row = cells.iloc[[0]].copy(deep=True).reset_index(drop=True)
    row["lon"] = point.x
    row["lat"] = point.y
    return row


def _cells_at(cells: pd.DataFrame, points: list[Point]) -> pd.DataFrame:
    frame = cells.iloc[: len(points)].copy(deep=True).reset_index(drop=True)
    frame["h3"] = [f"cell-{i}" for i in range(len(points))]
    frame["lon"] = [p.x for p in points]
    frame["lat"] = [p.y for p in points]
    return frame


# --------------------------------------------------------------------------------------
# ACCEPTANCE — Section 9
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_no_archetype_names_in_code():
    """"grep -c "metro_rail\\|data_centre" ufe/layers/l2_shocks.py returns 0"."""
    source = Path(L2.__file__).read_text()
    for name in (
        TRANSIT,
        "data_centre",
        DORMITORY_INDUSTRY,
        "gcc",
        "airport_greenfield",
        "township",
        "it_park",
    ):
        assert name not in source, f"archetype name {name!r} leaked into the resolver"


@pytest.mark.acceptance
def test_acc_new_archetype_from_yaml_only(cells, template, overlay):
    """A YAML-only archetype produces effects with no code change."""
    origin = _cell_point(cells)
    near = _offset_point(origin, overlay, east_m=300)
    frame = _cells_at(cells, [near])
    projects = _project(
        template,
        project_id="p-new",
        archetype="_acceptance_new_archetype",
        scale_unit="units_per_year",
        scale_value=1,
        geom=origin.wkt,
    )

    out = L2.resolve_shocks(frame, projects, overlay, year=LATE_YEAR)
    resolution = out.attrs[L2.ATTR_KEY]

    assert resolution.fields, "the new archetype emitted no FieldEffect"
    # 0.07 log points, step decay, inside max_m=1000, activation weight 1.
    assert out[L2.COL_FIELD_RESIDENTIAL].iloc[0] == pytest.approx(0.07)


@pytest.mark.acceptance
def test_acc_dormitory_typology_no_phantom_buyers(cells, template, params):
    """30,000 dormitory-industry jobs: <2,000 households, >18,000 dormitory workers.

    Section 21 named guard: "Dormitory workers as apartment buyers".
    """
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )

    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    dorm_share = params.value(
        f"archetypes.{DORMITORY_INDUSTRY}.employment.dormitory_share"
    )
    inmigrant = params.value("behaviour.migration.inmigrant_share_by_sector.manuf_light")
    wph = params.value("behaviour.workers_per_household")
    ownership = params.value(
        f"archetypes.{DORMITORY_INDUSTRY}.housing_typology.ownership_demand_share"
    )
    expected_dorm = jobs * dorm_share
    expected_hh = jobs * inmigrant / wph * (1 - dorm_share) * ownership

    assert out[L2.COL_DORMITORY_WORKERS].sum() == pytest.approx(expected_dorm)
    assert out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum() == pytest.approx(expected_hh)

    # The acceptance thresholds themselves.
    assert out[L2.COL_DORMITORY_WORKERS].sum() > 18000
    assert out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum() < 2000


@pytest.mark.acceptance
def test_acc_office_typology_produces_households(cells, template, overlay):
    """The same 30,000 jobs as an office employer: >9,000 effective households."""
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-office",
        archetype="gcc",
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )

    out = L2.resolve_shocks(
        frame,
        projects,
        overlay,
        year=LATE_YEAR,
        # `behaviour.office_sqm_per_seat` is null in the shipped YAML (see report).
        missing_office_sqm_per_seat=L2.IGNORE,
    )

    inmigrant = overlay.value("behaviour.migration.inmigrant_share_by_sector.it_office")
    wph = overlay.value("behaviour.workers_per_household")
    expected_hh = jobs * inmigrant / wph  # no dormitory share, no ownership share

    assert out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum() == pytest.approx(expected_hh)
    assert out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum() > 9000
    assert out[L2.COL_DORMITORY_WORKERS].sum() == 0


@pytest.mark.acceptance
def test_acc_airport_wedge_is_directional_not_radial(cells, template, overlay):
    """THE WEDGE TEST — Section 21 "Radial airport model" named guard.

    A cell on the CBD bearing gets ``wedge == 1``; the cell directly opposite, at the
    *same radius*, gets ``wedge == 0``.  A radial buffer would give both the same value,
    so this test is what stops the model recommending the wrong side of the airport.
    """
    airport = _offset_point(_cbd_point(overlay), overlay, east_m=12000, north_m=9000)
    radius = 3000
    cbd = _cbd_point(overlay)

    towards = _unit_step_towards(airport, cbd, overlay, metres=radius)
    away = _unit_step_towards(
        airport,
        # reflect the CBD through the airport to get the opposite bearing
        _offset_point(
            airport,
            overlay,
            east_m=-(geo.to_metric(cbd, geo.city_metric_crs(overlay)).x
                     - geo.to_metric(airport, geo.city_metric_crs(overlay)).x),
            north_m=-(geo.to_metric(cbd, geo.city_metric_crs(overlay)).y
                      - geo.to_metric(airport, geo.city_metric_crs(overlay)).y),
        ),
        overlay,
        metres=radius,
    )
    frame = _cells_at(cells, [towards, away])

    projects = _project(
        template,
        project_id="p-air",
        archetype="airport_greenfield",
        scale_unit="mppa",
        scale_value=1,
        geom=airport.wkt,
    )
    out = L2.resolve_shocks(frame, projects, overlay, year=LATE_YEAR)

    magnitude = overlay.value("archetypes.airport_greenfield.premium.0.value")
    got = out[L2.COL_FIELD_RESIDENTIAL].to_numpy(dtype=float)

    # wedge = 0.5 * (1 + cos theta): 1.0 on the CBD bearing, 0.0 directly opposite.
    assert got[0] == pytest.approx(magnitude, rel=1e-6)
    assert got[1] == pytest.approx(0, abs=1e-9)
    # And explicitly: this is NOT a radial buffer.
    assert got[0] != pytest.approx(got[1])


@pytest.mark.acceptance
def test_acc_distance_bands_are_exclusive(cells, template, params):
    """A cell at 300 m from a transit station receives 0.09, not 0.09 + 0.055 = 0.145.

    Section 21 named guard: "Cumulative distance bands".
    """
    station = _cell_point(cells)
    near = _offset_point(station, params, east_m=300)
    frame = _cells_at(cells, [near])
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=station.wkt,
    )

    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    outer = params.value(f"archetypes.{TRANSIT}.premium.1.value")
    got = float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0])

    assert got == pytest.approx(inner)
    assert got != pytest.approx(inner + outer)


@pytest.mark.acceptance
def test_acc_field_cap_engages_and_is_logged(cells, template, params, caplog):
    """Section 9.4: overlapping fields sum in log space and clip at the YAML cap."""
    station = _cell_point(cells)
    near = _offset_point(station, params, east_m=300)
    frame = _cells_at(cells, [near])

    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    cap_high = params.value("price.fields.cap_high")
    n = int(np.ceil(cap_high / inner)) + 1  # enough overlapping projects to breach the cap
    projects = pd.concat(
        [
            _project(
                template,
                project_id=f"p-transit-{i}",
                archetype=TRANSIT,
                scale_unit="km",
                scale_value=1,
                geom=station.wkt,
            )
            for i in range(n)
        ],
        ignore_index=True,
    )

    with caplog.at_level(logging.WARNING, logger=L2.__name__):
        out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    assert n * inner > cap_high  # the uncapped sum really does breach
    assert float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(cap_high)
    assert bool(out[L2.COL_FIELD_CAP_HIT].iloc[0]) is True

    diag = out.attrs[L2.ATTR_KEY].diagnostics
    assert diag["cap_hit_share"] == pytest.approx(1)
    assert diag["cap_warn_share"] == pytest.approx(params.value("price.fields.cap_warn_share"))
    assert diag["cap_warning"] is True
    assert any("cap" in record.message for record in caplog.records)


@pytest.mark.acceptance
def test_acc_no_service_employment_emitted_in_layer_2(cells, template, params, monkeypatch):
    """Section 21 named guard: "Double-counted service jobs".

    Service employment is a Layer 5 quantity computed from resident population.  Layer 2
    must never emit it, however large the employment shock it is resolving.
    """
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )

    # Record every parameter path the layer reads, so we can prove the service-jobs
    # coefficient is never even looked at.
    read_paths: list[str] = []
    for name in ("get", "value", "sample"):
        real = getattr(params, name)

        def spy(path, *a, _real=real, **kw):
            read_paths.append(str(path))
            return _real(path, *a, **kw)

        monkeypatch.setattr(params, name, spy)

    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    resolution = out.attrs[L2.ATTR_KEY]

    service = SECTORS.index("retail_svc")
    per_sector = np.vstack(out[L2.COL_JOBS_BY_SECTOR].to_numpy())
    assert per_sector[:, service].sum() == 0

    # No EmploymentEffect anywhere in the resolution routes to the service sector, ...
    assert all(e.sector != service for e in resolution.employment)
    # ... `behaviour.service_jobs_per_resident` and `behaviour.dorm_service_factor` (the
    # two Layer 5 service-jobs coefficients) are never read, ...
    assert read_paths
    assert not any("service" in path for path in read_paths)
    # ... and no output column carries service employment.
    assert not any("service" in c for c in out.columns if c.startswith(L2.COL_PREFIX))


# --------------------------------------------------------------------------------------
# 9.3 — decay functions, hand-computed
# --------------------------------------------------------------------------------------


def test_step_decay(params):
    mag = 0.2
    max_m = 500.0
    d = np.array([0.0, 499.0, 500.0, 501.0])
    got = L2.field_decay("step", mag, d, max_m, params)
    assert got == pytest.approx([mag, mag, mag, 0.0])


def test_linear_decay(params):
    mag = 0.2
    max_m = 500.0
    d = np.array([0.0, 250.0, 500.0, 900.0])
    got = L2.field_decay("linear", mag, d, max_m, params)
    assert got == pytest.approx([0.2, 0.1, 0.0, 0.0])


def test_exponential_decay_is_about_five_percent_at_max_m(params):
    mag = 0.2
    max_m = 500.0
    k = params.value("price.fields.exponential_decay_k")
    d = np.array([0.0, 250.0, 500.0])
    got = L2.field_decay("exponential", mag, d, max_m, params)
    assert got == pytest.approx(
        [mag, mag * np.exp(-k / 2), mag * np.exp(-k)]
    )
    # Section 9.3 comment: "~5% at max_m".
    assert got[2] / mag == pytest.approx(np.exp(-k), rel=1e-9)
    assert 0.04 < got[2] / mag < 0.06


def test_unknown_decay_raises(params):
    with pytest.raises(ValueError, match="decay"):
        L2.field_decay("gaussian", 0.1, np.array([0.0]), 500.0, params)


def test_decay_uses_distance_to_geometry_not_centroid(cells, template, params):
    """Section 9.3: "for line and polygon origins, d is distance to the geometry"."""
    crs = geo.city_metric_crs(params)
    anchor = _cell_point(cells)
    # A 4 km line running east; its centroid is 2 km from the western end.
    west = geo.to_metric(anchor, crs)
    line = geo.to_geographic(LineString([(west.x, west.y), (west.x + 4000, west.y)]), crs)
    # A cell 300 m north of the WESTERN END is 300 m from the geometry but ~2 km from the
    # centroid, so centroid-based distance would give 0 instead of the 0-500 m band.
    near_end = geo.to_geographic(Point(west.x, west.y + 300), crs)
    frame = _cells_at(cells, [near_end])

    projects = _project(
        template,
        project_id="p-line",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=line.wkt,
    )
    projects["geom_type"] = "linestring"
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    # 300 m from the geometry -> inside the 0-500 m band.  Distance to the centroid is
    # ~2 km, which would fall outside every residential band and give 0.
    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    assert float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(inner)


# --------------------------------------------------------------------------------------
# 9.3 — the wedge as a function
# --------------------------------------------------------------------------------------


def test_wedge_at_ninety_degrees_is_one_half(params):
    crs = geo.city_metric_crs(params)
    origin = np.array([0.0, 0.0])
    cbd = np.array([0.0, 1000.0])
    targets = np.array([[1000.0, 0.0], [0.0, 1000.0], [0.0, -1000.0]])
    got = L2.wedge_factor(origin, targets, cbd, params)
    scale = params.value("price.fields.airport_wedge_scale")
    assert got == pytest.approx([scale, 2 * scale, 0.0])
    assert crs  # crs is read from the city config, not a literal


def test_wedge_at_the_origin_is_full(params):
    got = L2.wedge_factor(
        np.array([0.0, 0.0]), np.array([[0.0, 0.0]]), np.array([0.0, 1000.0]), params
    )
    assert got == pytest.approx([1.0])


# --------------------------------------------------------------------------------------
# 9.4 — composition and caps
# --------------------------------------------------------------------------------------


def test_overlapping_fields_compose_in_log_space(cells, template, params):
    """Two different projects hitting one cell add their log-point magnitudes."""
    station = _cell_point(cells)
    near = _offset_point(station, params, east_m=300)
    frame = _cells_at(cells, [near])
    projects = pd.concat(
        [
            _project(
                template,
                project_id="p-a",
                archetype=TRANSIT,
                scale_unit="km",
                scale_value=1,
                geom=station.wkt,
            ),
            _project(
                template,
                project_id="p-b",
                archetype=TRANSIT,
                scale_unit="km",
                scale_value=1,
                geom=station.wkt,
            ),
        ],
        ignore_index=True,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    assert float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(2 * inner)


def test_negative_cap_engages(cells, template, params):
    origin = _cell_point(cells)
    near = _offset_point(origin, params, east_m=300)
    frame = _cells_at(cells, [near])
    dis = params.value(f"archetypes.{DORMITORY_INDUSTRY}.disamenity.0.value")
    cap_low = params.value("price.fields.cap_low")
    n = int(np.ceil(abs(cap_low / dis))) + 1
    projects = pd.concat(
        [
            _project(
                template,
                project_id=f"p-{i}",
                archetype=DORMITORY_INDUSTRY,
                scale_unit="jobs",
                scale_value=1,
                geom=origin.wkt,
            )
            for i in range(n)
        ],
        ignore_index=True,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    assert float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(cap_low)
    assert bool(out[L2.COL_FIELD_CAP_HIT].iloc[0]) is True


def test_cap_is_never_exceeded_anywhere(cells, template, params):
    """Whole-frame invariant: no cell, no target, ever outside the caps."""
    origin = _cell_point(cells)
    projects = pd.concat(
        [
            _project(
                template,
                project_id=f"p-{i}",
                archetype=TRANSIT,
                scale_unit="km",
                scale_value=1,
                geom=_offset_point(origin, params, east_m=100 * i).wkt,
            )
            for i in range(12)
        ],
        ignore_index=True,
    )
    out = L2.resolve_shocks(cells, projects, params, year=LATE_YEAR)
    cap_low = params.value("price.fields.cap_low")
    cap_high = params.value("price.fields.cap_high")
    for column in (L2.COL_FIELD_RESIDENTIAL, L2.COL_FIELD_COMMERCIAL, L2.COL_FIELD_OFFICE):
        assert out[column].min() >= cap_low - 1e-12
        assert out[column].max() <= cap_high + 1e-12


def test_no_projects_gives_zero_fields(cells, template, params):
    empty = template.iloc[:0].copy()
    empty["p_completion"] = pd.Series(dtype=float)
    empty["open_year"] = pd.Series(dtype=float)
    out = L2.resolve_shocks(cells, empty, params, year=LATE_YEAR)
    assert (out[L2.COL_FIELD_RESIDENTIAL] == 0).all()
    assert out.attrs[L2.ATTR_KEY].diagnostics["cap_hit_share"] == 0


# --------------------------------------------------------------------------------------
# 9.2 — the resolution algorithm
# --------------------------------------------------------------------------------------


def test_effects_are_scaled_by_the_layer_3_activation_weight(cells, template, params):
    """Requirement: shocks are weighted by `l3_credibility.activation_weight`."""
    jobs = 10000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    half = 1 / (1 + 1)
    projects = _project(
        template,
        project_id="p-half",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
        p_completion=half,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    weights = out.attrs[L2.ATTR_KEY].weights
    assert weights["p-half"] == pytest.approx(half)
    assert out[L2.COL_JOBS_PERMANENT].sum() == pytest.approx(jobs * half)


def test_dead_project_contributes_nothing(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dead",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=30000,
        geom=origin.wkt,
        p_completion=0,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    assert out[L2.COL_JOBS_PERMANENT].sum() == 0
    assert out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum() == 0
    assert out[L2.COL_FIELD_RESIDENTIAL].abs().sum() == 0


def test_counterfactual_fails_zeroes_the_shock(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-cf",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=30000,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(
        frame,
        projects,
        params,
        year=LATE_YEAR,
        force_project_state={"p-cf": "fails"},
    )
    assert out[L2.COL_JOBS_PERMANENT].sum() == 0


def test_construction_employment_is_temporary_and_separate(cells, template, params):
    """9.2 step 1: construction employment = unit * peak * local_retention."""
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    mw = 100
    projects = _project(
        template,
        project_id="p-dc",
        archetype="data_centre",
        scale_unit="mw",
        scale_value=mw,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, missing_office_sqm_per_seat=L2.IGNORE
    )
    peak = params.value("archetypes.data_centre.employment.construction_peak_per_unit")
    retention = params.value(
        "archetypes.data_centre.employment.construction_local_retention"
    )
    years = params.value("archetypes.data_centre.employment.construction_years")
    per_unit = params.value("archetypes.data_centre.employment.permanent_per_unit")

    assert out[L2.COL_JOBS_CONSTRUCTION].sum() == pytest.approx(mw * peak * retention)
    assert out[L2.COL_JOBS_PERMANENT].sum() == pytest.approx(mw * per_unit)

    resolution = out.attrs[L2.ATTR_KEY]
    construction = [e for e in resolution.employment if e.is_construction]
    assert len(construction) == 1
    assert construction[0].duration_years == pytest.approx(years)
    assert construction[0].sector == SECTORS.index("construction")
    permanent = [e for e in resolution.employment if not e.is_construction]
    assert permanent[0].duration_years is None


def test_sterilisation_emits_a_negative_capacity_supply_effect(cells, template, params):
    """9.2 step 5: `delta_capacity_sqm = -unit * land_take_sqm_per_unit`."""
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    mw = 100
    projects = _project(
        template,
        project_id="p-dc",
        archetype="data_centre",
        scale_unit="mw",
        scale_value=mw,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, missing_office_sqm_per_seat=L2.IGNORE
    )
    land_take = params.value("archetypes.data_centre.land_take_sqm_per_unit")
    effects = out.attrs[L2.ATTR_KEY].supply
    assert len(effects) == 1
    assert effects[0].delta_capacity_sqm == pytest.approx(-mw * land_take)
    assert effects[0].delta_floorspace_sqm == 0
    assert out[L2.COL_DELTA_CAPACITY].sum() == pytest.approx(-mw * land_take)


def test_supply_effects_are_accepted_by_layer_4(cells, template, params):
    """INTERFACE COORDINATION: our `SupplyEffect` must drop straight into `apply_supply`."""
    origin = _cell_point(cells)
    mw = 1
    projects = _project(
        template,
        project_id="p-dc",
        archetype="data_centre",
        scale_unit="mw",
        scale_value=mw,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(
        cells, projects, params, year=LATE_YEAR, missing_office_sqm_per_seat=L2.IGNORE
    )
    effects = list(out.attrs[L2.ATTR_KEY].supply)
    assert effects
    # Same field names, same order, same types as l4_supply expects.
    assert [f.name for f in L2.dataclasses.fields(L2.SupplyEffect)][:4] == [
        "cell",
        "delta_floorspace_sqm",
        "delta_capacity_sqm",
        "start_year",
    ]
    for effect in effects:
        effect.start_year  # noqa: B018 - attribute presence is the contract
    supplied = [
        L4.SupplyEffect(
            cell=e.cell,
            delta_floorspace_sqm=e.delta_floorspace_sqm,
            delta_capacity_sqm=e.delta_capacity_sqm,
            start_year=e.start_year,
        )
        for e in effects
    ]
    # Section 11.3 applies an effect in its own `start_year`, which for a sterilisation
    # is the project's `open_year`.
    apply_year = effects[0].start_year
    baseline = L4.apply_supply(cells, params, year=apply_year)
    applied = L4.apply_supply(cells, params, year=apply_year, effects=supplied)
    assert applied["capacity_sqm"].sum() < baseline["capacity_sqm"].sum()


def test_network_effect_is_emitted_for_network_archetypes(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=10,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    network = out.attrs[L2.ATTR_KEY].network
    assert len(network) == 1
    assert network[0].kind == params.get(f"archetypes.{TRANSIT}.network_effect")["type"]
    assert network[0].open_year == OPEN_YEAR


def test_construction_penalty_is_bounded_by_the_build_window(cells, template, params):
    """9.2 step 4: the construction penalty is "bounded by build window".

    The window is ``[construction_start_year, open_year)`` — `end_year` is exclusive, so
    the penalty is gone in the year the thing opens.
    """
    station = _cell_point(cells)
    penalty = params.get(f"archetypes.{TRANSIT}.construction_penalty")
    inside_m = float(penalty["max_m"]) / (1 + 1)
    near = _offset_point(station, params, east_m=inside_m)
    frame = _cells_at(cells, [near])
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=station.wkt,
    )

    build_years = params.value("archetypes._defaults.construction_years")
    during = int(OPEN_YEAR - build_years) + 1

    mid = L2.resolve_shocks(frame, projects, params, year=during)
    at_open = L2.resolve_shocks(frame, projects, params, year=OPEN_YEAR)
    late = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    # The penalty targets `all`, so it lands on every field target.
    magnitude = float(penalty["value"])
    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    weight = mid.attrs[L2.ATTR_KEY].weights["p-transit"]
    assert float(mid[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(
        (inner + magnitude) * weight
    )
    # `all` also reaches the office target, which has its own single premium band.
    premiums = params.get(f"archetypes.{TRANSIT}.premium")
    office = next(
        float(e["value"]) for e in premiums if e["target"] == L2.TARGET_OFFICE
    )
    assert float(mid[L2.COL_FIELD_OFFICE].iloc[0]) == pytest.approx(
        (office + magnitude) * weight
    )

    # In `open_year` and later the penalty is gone; only the premium remains.
    for frame_out in (at_open, late):
        w = frame_out.attrs[L2.ATTR_KEY].weights["p-transit"]
        assert float(frame_out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(
            inner * w
        )

    penalties = [
        e
        for e in mid.attrs[L2.ATTR_KEY].fields
        if e.target == L2.TARGET_ALL
    ]
    assert len(penalties) == 1
    assert penalties[0].end_year == OPEN_YEAR
    assert not [
        e for e in late.attrs[L2.ATTR_KEY].fields if e.target == L2.TARGET_ALL
    ]


def test_applies_when_flag_gates_a_field(cells, template, params):
    """A field entry carrying `applies_when` only fires for a project with that flag."""
    origin = _cell_point(cells)
    entry = params.get(f"archetypes.{TRANSIT}.disamenity")[0]
    flag = str(entry["applies_when"])
    near = _offset_point(origin, params, east_m=float(entry["max_m"]) / (1 + 1))
    frame = _cells_at(cells, [near])
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )

    without = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    with_flag = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, project_flags={"p-transit": [flag]}
    )
    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    assert float(without[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(inner)
    assert float(with_flag[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(
        inner + float(entry["value"])
    )


def test_premium_multiplier_flag_scales_premiums(cells, template, params):
    """`premium_multipliers.<flag>` scales that project's premiums, nothing else."""
    origin = _cell_point(cells)
    near = _offset_point(origin, params, east_m=300)
    frame = _cells_at(cells, [near])
    flag = next(iter(params.get(f"archetypes.{TRANSIT}.premium_multipliers")))
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, project_flags={"p-transit": [flag]}
    )
    inner = params.value(f"archetypes.{TRANSIT}.premium.0.value")
    mult = params.value(f"archetypes.{TRANSIT}.premium_multipliers.{flag}")
    assert float(out[L2.COL_FIELD_RESIDENTIAL].iloc[0]) == pytest.approx(inner * mult)


def test_no_network_effect_for_none_type(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=100,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    assert out.attrs[L2.ATTR_KEY].network == ()


def test_dormitory_floorspace_demand(cells, template, params):
    """9.2 step 6: `sqm = jobs * share * sqm_per_worker`."""
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    share = params.value(f"archetypes.{DORMITORY_INDUSTRY}.employment.dormitory_share")
    per_worker = params.value(
        f"archetypes.{DORMITORY_INDUSTRY}.housing_typology.dormitory_sqm_per_worker"
    )
    demand = [e for e in out.attrs[L2.ATTR_KEY].floorspace_demand if e.use == L2.USE_DORMITORY]
    assert len(demand) == 1
    assert demand[0].sqm == pytest.approx(jobs * share * per_worker)
    assert out[L2.COL_FLOORSPACE_DEMAND].sum() == pytest.approx(jobs * share * per_worker)


def test_missing_office_sqm_per_seat_raises_by_default(cells, template, params):
    """`behaviour.office_sqm_per_seat` is null: "must raise rather than substitute"."""
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dc",
        archetype="data_centre",
        scale_unit="mw",
        scale_value=10,
        geom=origin.wkt,
    )
    with pytest.raises(MissingParameter, match="office_sqm_per_seat"):
        L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)


def test_scale_unit_mismatch_raises(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-bad",
        archetype=TRANSIT,
        scale_unit="mw",
        scale_value=1,
        geom=origin.wkt,
    )
    with pytest.raises(ValueError, match="scale_unit"):
        L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)


def test_unknown_archetype_raises_and_can_be_ignored(cells, template, params):
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-unknown",
        archetype="not_an_archetype",
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    with pytest.raises(MissingParameter, match="not_an_archetype"):
        L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    out = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, unknown_archetypes=L2.IGNORE
    )
    assert out[L2.COL_FIELD_RESIDENTIAL].iloc[0] == 0


# --------------------------------------------------------------------------------------
# 9.5 — wage band routing
# --------------------------------------------------------------------------------------


def test_wage_band_routing_hand_computed(cells, template, params):
    """household_income = wage * workers_per_household * household_wage_premium."""
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)

    wage = params.value(f"archetypes.{DORMITORY_INDUSTRY}.employment.median_wage_inr_mo")
    wph = params.value("behaviour.workers_per_household")
    premium = params.value("behaviour.wage_to_band.household_wage_premium")
    boundaries = [float(b["value"]) for b in params.get("behaviour.income_bands.boundaries_inr_mo")]
    income = wage * wph * premium
    band = int(np.digitize(income, boundaries))

    by_band = np.vstack(out[L2.COL_HOUSEHOLDS_BY_BAND].to_numpy())
    assert by_band.shape[1] == len(INCOME_BANDS)
    assert by_band[:, band].sum() == pytest.approx(out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum())
    for other in range(len(INCOME_BANDS)):
        if other != band:
            assert by_band[:, other].sum() == 0


def test_project_wage_overrides_the_archetype_wage(cells, template, params):
    """9.2: `wage = project.median_wage_inr_mo or A.employment.median_wage_inr_mo`."""
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    high_wage = 250000
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=1000,
        geom=origin.wkt,
        median_wage=high_wage,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    wph = params.value("behaviour.workers_per_household")
    premium = params.value("behaviour.wage_to_band.household_wage_premium")
    boundaries = [
        float(b["value"]) for b in params.get("behaviour.income_bands.boundaries_inr_mo")
    ]
    band = int(np.digitize(high_wage * wph * premium, boundaries))
    assert band == len(INCOME_BANDS) - 1  # the top band
    by_band = np.vstack(out[L2.COL_HOUSEHOLDS_BY_BAND].to_numpy())
    assert by_band[:, band].sum() > 0


def test_income_index_shifts_the_band_boundaries(cells, template, params):
    """Section 3.7 boundaries are "inflation-indexed by base year"."""
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=1000,
        geom=origin.wkt,
    )
    base = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    indexed = L2.resolve_shocks(
        frame, projects, params, year=LATE_YEAR, income_index=3
    )
    base_band = int(np.argmax(np.vstack(base[L2.COL_HOUSEHOLDS_BY_BAND].to_numpy())[0]))
    indexed_band = int(
        np.argmax(np.vstack(indexed[L2.COL_HOUSEHOLDS_BY_BAND].to_numpy())[0])
    )
    assert indexed_band < base_band


def test_dormitory_workers_do_not_enter_household_allocation(cells, template, params):
    """9.5: "dormitory_workers do not enter the household allocation"."""
    jobs = 30000
    origin = _cell_point(cells)
    frame = _cells_at(cells, [origin])
    projects = _project(
        template,
        project_id="p-dorm",
        archetype=DORMITORY_INDUSTRY,
        scale_unit="jobs",
        scale_value=jobs,
        geom=origin.wkt,
    )
    out = L2.resolve_shocks(frame, projects, params, year=LATE_YEAR)
    by_band = np.vstack(out[L2.COL_HOUSEHOLDS_BY_BAND].to_numpy())
    assert by_band.sum() == pytest.approx(out[L2.COL_EFFECTIVE_HOUSEHOLDS].sum())
    assert by_band.sum() < out[L2.COL_DORMITORY_WORKERS].sum()


# --------------------------------------------------------------------------------------
# purity, shape and determinism
# --------------------------------------------------------------------------------------


def test_same_index_row_count_and_no_mutation(cells, template, params):
    origin = _cell_point(cells)
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    before_cells = cells.copy(deep=True)
    before_projects = projects.copy(deep=True)
    out = L2.resolve_shocks(cells, projects, params, year=LATE_YEAR)

    assert len(out) == len(cells)
    assert out.index.equals(cells.index)
    assert set(cells.columns).issubset(out.columns)
    pd.testing.assert_frame_equal(cells, before_cells)
    pd.testing.assert_frame_equal(projects, before_projects)


def test_determinism_deterministic_mode(cells, template, params):
    origin = _cell_point(cells)
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    a = L2.resolve_shocks(cells, projects, params, year=LATE_YEAR)
    b = L2.resolve_shocks(cells, projects, params, year=LATE_YEAR)
    pd.testing.assert_frame_equal(a, b)


def test_monte_carlo_requires_an_explicit_generator(cells, template, params):
    origin = _cell_point(cells)
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    with pytest.raises(ValueError, match="rng"):
        L2.resolve_shocks(cells, projects, params, year=LATE_YEAR, monte_carlo=True)

    seed = 20240101
    a = L2.resolve_shocks(
        cells,
        projects,
        params,
        year=LATE_YEAR,
        monte_carlo=True,
        rng=np.random.default_rng(seed),
    )
    b = L2.resolve_shocks(
        cells,
        projects,
        params,
        year=LATE_YEAR,
        monte_carlo=True,
        rng=np.random.default_rng(seed),
    )
    pd.testing.assert_frame_equal(a, b)


def test_no_metric_computation_in_degrees(cells, template, params, monkeypatch):
    """Section 21 "Degrees used as metres": every distance goes through `ufe.geo`."""
    calls: list[str] = []
    real = geo.to_metric

    def spy(geom, crs_metric, **kw):
        calls.append(str(crs_metric))
        return real(geom, crs_metric, **kw)

    monkeypatch.setattr(L2.geo, "to_metric", spy)
    origin = _cell_point(cells)
    projects = _project(
        template,
        project_id="p-transit",
        archetype=TRANSIT,
        scale_unit="km",
        scale_value=1,
        geom=origin.wkt,
    )
    L2.resolve_shocks(cells, projects, params, year=LATE_YEAR)
    assert calls, "no metric reprojection happened at all"
    assert all(c == params.city_config["crs_metric"] for c in calls)


def test_no_numeric_literals_beyond_zero_and_one():
    """CONTRACT.md rule 1 / Section 0.1 rule 3, enforced on our own source."""
    import ast

    source = Path(L2.__file__).read_text()
    allowed = {0, 1}
    offenders = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value not in allowed
    ]
    assert offenders == []
