"""Tests for Layer 0, substrate assembly (spec Section 7).

The Section 7 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "No cell has ``capacity_sqm < 0`` or ``headroom_sqm < 0``"
  -> ``test_acc_capacity_and_headroom_are_non_negative``
* "``undevelopable_frac in [0,1]``"
  -> ``test_acc_undevelopable_frac_in_unit_interval``
* "a synthetic cell fully covered by two overlapping gate polygons yields exactly 1.0, not 2.0"
  -> ``test_acc_two_overlapping_gate_polygons_yield_one``
* "Elasticity classifier on a fixture of 6 hand-labelled cells reproduces the labels"
  -> ``test_acc_elasticity_classifier_reproduces_six_hand_labels``
* "``jobs_by_sector`` city total matches the grown census total +/- 1%"
  -> ``test_acc_jobs_by_sector_city_total_matches_grown_census``
* "Function is pure: calling twice on the same input returns equal frames"
  -> ``test_acc_function_is_pure``

Everything beyond those blocks tests the arithmetic of 7.1-7.5 directly: hand-computed
values, monotonicity, bounds and idempotence.  Expected values are recomputed from the
YAML through ``Params`` rather than written down, so the tests contain no model numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from shapely import wkb

from ufe.layers import l0_substrate as L0
from ufe.params import load_params
from ufe.store import schemas as S

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    build_city,
    synthetic_cells,
    synthetic_city,
)

CITY = "vizag"

#: Section 7 ACCEPTANCE: "matches the grown census total +/- 1%".
ACCEPTANCE_REL_TOL = 1 / 100


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def cells(synthetic_city):  # noqa: F811
    return synthetic_city.cells


@pytest.fixture(scope="module")
def assembled(cells, params):
    return L0.assemble_substrate(cells, params)


def _thresholds(params) -> dict[str, float]:
    return {
        "builtup_min": params.value(L0.P_DENSE_CORE_BUILTUP_MIN),
        "ratio_max": params.value(L0.P_DENSE_CORE_HEADROOM_RATIO_MAX),
        "undev_min": params.value(L0.P_CONSTRAINED_UNDEVELOPABLE_MIN),
        "open_builtup_max": params.value(L0.P_OPEN_FRINGE_BUILTUP_MAX),
    }


def _well_above(threshold: float) -> float:
    """A fraction comfortably above ``threshold`` and still inside 0..1."""
    return float(np.mean((threshold, 1.0)))


# --------------------------------------------------------------------------------------
# ACCEPTANCE — Module 3
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_capacity_and_headroom_are_non_negative(assembled):
    """"No cell has ``capacity_sqm < 0`` or ``headroom_sqm < 0``"."""
    assert (assembled["capacity_sqm"] >= 0).all()
    assert (assembled["headroom_sqm"] >= 0).all()
    assert np.isfinite(assembled["capacity_sqm"]).all()
    assert np.isfinite(assembled["headroom_sqm"]).all()


@pytest.mark.acceptance
def test_acc_undevelopable_frac_in_unit_interval(assembled):
    """"``undevelopable_frac in [0,1]``"."""
    frac = assembled["undevelopable_frac"]
    assert frac.notna().all()
    assert (frac >= 0).all()
    assert (frac <= 1).all()


@pytest.mark.acceptance
def test_acc_two_overlapping_gate_polygons_yield_one(cells, params):
    """"a synthetic cell fully covered by two overlapping gate polygons yields exactly 1.0,
    not 2.0"."""
    one = cells.iloc[[0]].copy()
    # Both gate layers cover the whole cell and overlap each other: the hexagon itself and
    # its bounding box.  Summing the two shares would give 2.0; the union gives 1.0.
    hexagon = wkb.loads(bytes(one["geometry"].iloc[0]))
    gates = {"water": [hexagon], "defence": [hexagon.envelope]}

    out = L0.assemble_substrate(one, params, gates=gates)
    assert out["undevelopable_frac"].iloc[0] == pytest.approx(1.0)


@pytest.mark.acceptance
def test_acc_elasticity_classifier_reproduces_six_hand_labels(cells, params):
    """"Elasticity classifier on a fixture of 6 hand-labelled cells reproduces the labels"."""
    fixture, labels = _hand_labelled_six(cells, params)
    out = L0.assemble_substrate(fixture, params)
    assert list(out["elasticity_class"]) == labels


@pytest.mark.acceptance
def test_acc_jobs_by_sector_city_total_matches_grown_census(cells, params):
    """"``jobs_by_sector`` city total matches the grown census total +/- 1%"."""
    ward_jobs, growth = _census_inputs(cells)
    out = L0.assemble_substrate(
        cells, params, ward_jobs_2011=ward_jobs, sector_growth=growth
    )

    grown_total = float(
        sum(ward_jobs[sector].sum() * growth[sector] for sector in S.SECTORS)
    )
    city_total = float(np.stack(out["jobs_by_sector"].to_numpy()).sum())
    assert city_total == pytest.approx(grown_total, rel=ACCEPTANCE_REL_TOL)

    # ... and sector by sector, not just in aggregate.
    by_sector = np.stack(out["jobs_by_sector"].to_numpy()).sum(axis=0)
    for index, sector in enumerate(S.SECTORS):
        expected = ward_jobs[sector].sum() * growth[sector]
        assert by_sector[index] == pytest.approx(expected, rel=ACCEPTANCE_REL_TOL)


@pytest.mark.acceptance
def test_acc_function_is_pure(cells, params):
    """"Function is pure: calling twice on the same input returns equal frames"."""
    before = cells.copy(deep=True)
    first = L0.assemble_substrate(cells, params)
    second = L0.assemble_substrate(cells, params)

    pd.testing.assert_frame_equal(first, second)
    # The input frame is untouched.
    pd.testing.assert_frame_equal(cells, before)
    assert first is not cells


# --------------------------------------------------------------------------------------
# frame contract
# --------------------------------------------------------------------------------------


def test_returns_new_frame_with_same_index_and_row_count(cells, assembled):
    assert assembled is not cells
    assert len(assembled) == len(cells)
    pd.testing.assert_index_equal(assembled.index, cells.index)
    assert list(assembled["h3"]) == list(cells["h3"])


def test_output_validates_against_the_cells_schema(assembled):
    """The schema is ``strict=True``: this fails if Layer 0 invents a column."""
    S.CELLS.validate(assembled, lazy=True)


def test_emitted_columns_are_declared_in_the_schema(assembled):
    assert set(L0.DERIVED_COLUMNS) <= set(S.CELLS.columns)
    assert set(assembled.columns) <= set(S.CELLS.columns)


def test_all_derived_columns_are_populated(assembled):
    for column in L0.DERIVED_COLUMNS:
        assert column in assembled.columns
        assert assembled[column].notna().all(), column


def test_missing_required_input_column_raises(cells, params):
    from ufe.errors import SchemaValidationError

    broken = cells.drop(columns=["permitted_far"])
    with pytest.raises(SchemaValidationError):
        L0.assemble_substrate(broken, params)


# --------------------------------------------------------------------------------------
# 7.1 undevelopable fraction
# --------------------------------------------------------------------------------------


def test_undevelopable_frac_absorbs_the_slope_gate(cells, params):
    cutoff = params.value(L0.P_SLOPE_CUTOFF_PCT)
    frame = cells.copy()
    frame["slope_pct"] = np.where(
        np.arange(len(frame)) % (1 + 1) == 0, cutoff + 1, 0.0
    )
    frame["undevelopable_frac"] = 0.0

    out = L0.assemble_substrate(frame, params)
    steep = frame["slope_pct"].to_numpy() > cutoff
    assert (out.loc[steep, "undevelopable_frac"] == 1.0).all()
    assert (out.loc[~steep, "undevelopable_frac"] == 0.0).all()


def test_undevelopable_frac_never_exceeds_one_when_gates_are_summed(cells, params):
    cutoff = params.value(L0.P_SLOPE_CUTOFF_PCT)
    frame = cells.copy()
    # An already-large ingested fraction plus a slope gate would sum past 1 without the clip.
    frame["undevelopable_frac"] = _well_above(0.0)
    frame["slope_pct"] = cutoff + 1

    out = L0.assemble_substrate(frame, params)
    assert (out["undevelopable_frac"] == 1.0).all()


def test_geometric_gates_are_unioned_not_summed(cells, params):
    """Two half-covering, fully overlapping gates give the share of one, not of both."""
    one = cells.iloc[[0]].copy()
    one["slope_pct"] = 0.0  # isolate the gate geometry from the Section 7.2 slope gate
    hexagon = wkb.loads(bytes(one["geometry"].iloc[0]))
    minx, miny, maxx, maxy = hexagon.bounds
    from shapely.geometry import box

    half = box(minx, miny, minx + (maxx - minx) / (1 + 1), maxy)
    covered = hexagon.intersection(half)

    single = L0.assemble_substrate(one, params, gates={"water": [covered]})
    doubled = L0.assemble_substrate(
        one, params, gates={"water": [covered], "wetland": [covered]}
    )
    assert doubled["undevelopable_frac"].iloc[0] == pytest.approx(
        single["undevelopable_frac"].iloc[0]
    )
    assert 0 < single["undevelopable_frac"].iloc[0] < 1


def test_empty_gate_layers_leave_the_ingested_fraction_alone(cells, params):
    frame = cells.copy()
    frame["slope_pct"] = 0.0
    out = L0.assemble_substrate(frame, params, gates={})
    np.testing.assert_allclose(
        out["undevelopable_frac"].to_numpy(), frame["undevelopable_frac"].to_numpy()
    )


# --------------------------------------------------------------------------------------
# 7.2 slope cost multiplier
# --------------------------------------------------------------------------------------


def test_slope_cost_mult_hand_computed(cells, params):
    start = params.value(L0.P_SLOPE_PENALTY_START_PCT)
    per_pct = params.value(L0.P_SLOPE_PENALTY_PER_PCT)
    cutoff = params.value(L0.P_SLOPE_CUTOFF_PCT)

    frame = cells.iloc[: (1 + 1 + 1 + 1)].copy()
    slopes = [0.0, start, (start + cutoff) / (1 + 1), cutoff]
    frame["slope_pct"] = slopes

    out = L0.assemble_substrate(frame, params)
    expected = [1 + per_pct * max(0.0, s - start) for s in slopes]
    np.testing.assert_allclose(out["slope_cost_mult"].to_numpy(), expected)
    # Flat ground and anything below the penalty start cost nothing extra.
    assert out["slope_cost_mult"].iloc[0] == 1.0
    assert out["slope_cost_mult"].iloc[1] == 1.0


def test_slope_cost_mult_is_infinite_above_the_cutoff(cells, params):
    cutoff = params.value(L0.P_SLOPE_CUTOFF_PCT)
    frame = cells.iloc[: (1 + 1)].copy()
    frame["slope_pct"] = [cutoff, cutoff + 1]

    out = L0.assemble_substrate(frame, params)
    assert math.isfinite(out["slope_cost_mult"].iloc[0])
    assert math.isinf(out["slope_cost_mult"].iloc[1])


def test_slope_cost_mult_is_monotone_in_slope(cells, params):
    cutoff = params.value(L0.P_SLOPE_CUTOFF_PCT)
    frame = cells.iloc[: (1 + 1 + 1 + 1 + 1)].copy()
    frame["slope_pct"] = np.linspace(0.0, cutoff, len(frame))

    out = L0.assemble_substrate(frame, params)
    values = out["slope_cost_mult"].to_numpy()
    assert (np.diff(values) >= 0).all()
    assert values[-1] > values[0]


# --------------------------------------------------------------------------------------
# 7.3 utility state, assembly feasibility, capacity and headroom
# --------------------------------------------------------------------------------------


def test_utility_state_precedence(cells, params):
    frame = cells.iloc[: (1 + 1 + 1 + 1 + 1)].copy()
    frame["util_water"] = [0, 0, 1, 1, 1]
    frame["util_sewer"] = [0, 1, 0, 1, 1]
    frame["util_power"] = [1, 1, 1, 0, 1]

    out = L0.assemble_substrate(frame, params)
    # "power alone without water counts as none"; so does sewer without water.
    assert list(out["utility_state"]) == [
        "none",
        "none",
        "water",
        "water_sewer",
        "water_sewer_power",
    ]
    assert set(out["utility_state"]) <= set(S.UTILITY_STATES)


def test_assembly_feasibility_is_the_yaml_step_function(cells, params):
    tiers = L0.assembly_feasibility_tiers(params)
    frame = cells.iloc[: len(tiers)].copy()
    # One cell just at each tier's minimum parcel size.
    frame["mean_parcel_sqm"] = [float(t.min_parcel_sqm) for t in tiers]
    frame["mean_parcel_sqm"] = frame["mean_parcel_sqm"].replace(0.0, 1.0)

    got = L0.assembly_multiplier(frame["mean_parcel_sqm"].to_numpy(), params)
    expected = [t.value for t in tiers]
    np.testing.assert_allclose(got, expected)
    # Bigger parcels are never harder to assemble.
    assert (np.diff(np.array(expected)[::-1]) >= 0).all()


def test_assembly_multiplier_for_unknown_parcel_size_uses_the_lowest_tier(params):
    tiers = L0.assembly_feasibility_tiers(params)
    lowest = min(tiers, key=lambda t: t.min_parcel_sqm)
    got = L0.assembly_multiplier(np.array([np.nan]), params)
    assert got[0] == pytest.approx(lowest.value)


def test_capacity_and_headroom_hand_computed(cells, params):
    frame = cells.iloc[[0]].copy()
    frame["slope_pct"] = 0.0
    frame["undevelopable_frac"] = 0.0
    frame["util_water"] = 1
    frame["util_sewer"] = 1
    frame["util_power"] = 1
    tiers = L0.assembly_feasibility_tiers(params)
    top = max(tiers, key=lambda t: t.min_parcel_sqm)
    frame["mean_parcel_sqm"] = float(top.min_parcel_sqm) + 1

    out = L0.assemble_substrate(frame, params)

    area = float(frame["area_sqm"].iloc[0])
    far = float(frame["permitted_far"].iloc[0])
    utility_mult = params.value(L0.P_UTILITY_GATE + ".water_sewer_power")
    expected_capacity = area * (1 - 0.0) * far * utility_mult * top.value
    assert out["capacity_sqm"].iloc[0] == pytest.approx(expected_capacity)

    built = float(frame["floorspace_res_sqm"].iloc[0] + frame["floorspace_com_sqm"].iloc[0])
    assert out["headroom_sqm"].iloc[0] == pytest.approx(max(0.0, expected_capacity - built))


def test_headroom_floors_at_zero_when_stock_exceeds_capacity(cells, params):
    frame = cells.iloc[[0]].copy()
    frame["permitted_far"] = 0.0
    out = L0.assemble_substrate(frame, params)
    assert out["capacity_sqm"].iloc[0] == 0.0
    assert out["headroom_sqm"].iloc[0] == 0.0


def test_more_undevelopable_means_less_headroom(cells, params):
    frame = cells.iloc[[0]].copy()
    frame["slope_pct"] = 0.0
    frame["floorspace_res_sqm"] = 0.0
    frame["floorspace_com_sqm"] = 0.0

    fracs = np.linspace(0.0, 1.0, 1 + 1 + 1 + 1 + 1)
    headrooms = []
    capacities = []
    for frac in fracs:
        one = frame.copy()
        one["undevelopable_frac"] = frac
        out = L0.assemble_substrate(one, params)
        headrooms.append(out["headroom_sqm"].iloc[0])
        capacities.append(out["capacity_sqm"].iloc[0])

    assert (np.diff(headrooms) <= 0).all()
    assert (np.diff(capacities) <= 0).all()
    assert headrooms[-1] == 0.0


def test_capacity_never_exceeds_the_gross_far_capacity(assembled):
    gross = (
        assembled["area_sqm"]
        * (1 - assembled["undevelopable_frac"])
        * assembled["permitted_far"]
    )
    assert (assembled["capacity_sqm"] <= gross + gross.max() * 0).all()


# --------------------------------------------------------------------------------------
# 7.4 supply elasticity
# --------------------------------------------------------------------------------------


def _hand_labelled_six(cells: pd.DataFrame, params) -> tuple[pd.DataFrame, list[str]]:
    """Six cells whose elasticity class is forced by hand, per the 7.4 classifier."""
    t = _thresholds(params)
    n = 1 + 1 + 1 + 1 + 1 + 1
    frame = cells.iloc[:n].copy()
    frame["slope_pct"] = 0.0
    frame["floorspace_res_sqm"] = 0.0
    frame["floorspace_com_sqm"] = 0.0
    frame["util_sewer"] = 1
    frame["util_power"] = 1
    tiers = L0.assembly_feasibility_tiers(params)
    top = max(tiers, key=lambda t_: t_.min_parcel_sqm)
    frame["mean_parcel_sqm"] = float(top.min_parcel_sqm) + 1

    #                       builtup                 undevelopable          far
    #  0 dense_core        : built up, no headroom (fully undevelopable)
    #  1 dense_core        : built up, no headroom (zero FAR)
    #  2 constrained_periph: empty, heavily gated
    #  3 open_fringe       : empty, ungated, no water
    #  4 typical_periph    : empty, ungated, has water
    #  5 typical_periph    : built up but with ample headroom
    frame["builtup_frac"] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    frame["undevelopable_frac"] = [
        1.0,
        0.0,
        _well_above(t["undev_min"]),
        0.0,
        0.0,
        0.0,
    ]
    frame["permitted_far"] = [
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        t["ratio_max"] + 1,
    ]
    frame["util_water"] = [1, 1, 1, 0, 1, 1]
    labels = [
        "dense_core",
        "dense_core",
        "constrained_periph",
        "open_fringe",
        "typical_periph",
        "typical_periph",
    ]
    return frame, labels


def test_eps_supply_matches_the_class_value_from_yaml(assembled, params):
    for klass in S.ELASTICITY_CLASSES:
        rows = assembled[assembled["elasticity_class"] == klass]
        if rows.empty:
            continue
        expected = params.value(f"{L0.P_ELASTICITY_CLASS}.{klass}")
        np.testing.assert_allclose(rows["eps_supply"].to_numpy(), expected)


def test_every_class_value_is_read_from_the_city_resolved_tree(cells, params):
    """Vizag overrides ``dense_core``; the layer must use the override, not the base value."""
    fixture, labels = _hand_labelled_six(cells, params)
    out = L0.assemble_substrate(fixture, params)
    dense = out[out["elasticity_class"] == "dense_core"]
    assert not dense.empty
    np.testing.assert_allclose(
        dense["eps_supply"].to_numpy(), params.value(f"{L0.P_ELASTICITY_CLASS}.dense_core")
    )


def test_ambiguous_cells_fall_back_to_the_city_default_class(cells, params):
    frame = cells.iloc[[0]].copy()
    frame["builtup_frac"] = np.nan
    out = L0.assemble_substrate(frame, params)
    default = params.value(L0.P_ELASTICITY_CLASS_DEFAULT)
    assert out["elasticity_class"].iloc[0] == default
    assert out["eps_supply"].iloc[0] == pytest.approx(
        params.value(f"{L0.P_ELASTICITY_CLASS}.{default}")
    )


def test_elasticity_class_is_always_one_of_the_schema_vocabulary(assembled):
    assert set(assembled["elasticity_class"]) <= set(S.ELASTICITY_CLASSES)


def test_regression_branch_applies_the_log_linear_form(cells, params):
    coefficients = {"a0": 1.0, "a1": 1.0, "a2": 0.0, "a3": 1.0, "a4": 0.0}
    frame = cells.iloc[: (1 + 1 + 1)].copy()
    frame["slope_pct"] = 0.0
    frame["regulatory_index"] = 0.0
    frame["mean_parcel_sqm"] = 1.0

    out = L0.assemble_substrate(frame, params, regression_coefficients=coefficients)

    offset = params.value(L0.P_REGRESSION_LOG_OFFSET)
    undev = out["undevelopable_frac"].to_numpy()
    builtup = frame["builtup_frac"].to_numpy()
    expected = np.exp(
        coefficients["a0"]
        + coefficients["a1"] * np.log(1 - undev + offset)
        + coefficients["a3"] * np.log(1 - builtup + offset)
    )
    np.testing.assert_allclose(out["eps_supply"].to_numpy(), expected)
    # The class label is still assigned, so downstream reporting keeps working.
    assert set(out["elasticity_class"]) <= set(S.ELASTICITY_CLASSES)


def test_regression_enabled_without_coefficients_raises(cells, params):
    from ufe.errors import MissingParameter

    with pytest.raises(MissingParameter):
        L0.assemble_substrate(cells, params, use_regression=True)


def test_fit_drops_zones_with_an_unstable_price_change(params):
    threshold = params.value(L0.P_REGRESSION_MIN_ABS_DLN_PRICE)
    offset = params.value(L0.P_REGRESSION_LOG_OFFSET)
    n = 1 + 1 + 1 + 1 + 1 + 1
    rng = np.random.default_rng(len(S.SECTORS))
    zones = pd.DataFrame(
        {
            "dln_price": np.concatenate(
                [np.full(n, threshold + 1), np.full(n, threshold / (1 + 1))]
            ),
            "dln_floorspace": rng.uniform(0.0, 1.0, n + n),
            "undevelopable_frac": rng.uniform(0.0, 1.0, n + n),
            "regulatory_index": rng.uniform(0.0, 1.0, n + n),
            "builtup_frac": rng.uniform(0.0, 1.0, n + n),
            "mean_parcel_sqm": rng.uniform(1.0, len(S.SECTORS), n + n),
        }
    )

    fit = L0.fit_elasticity_regression(zones, params)
    assert fit.n == n
    assert set(fit.coefficients) == {"a0", "a1", "a2", "a3", "a4"}
    assert np.isfinite(fit.r2)
    assert offset > 0


def test_fit_recovers_coefficients_it_generated(params):
    offset = params.value(L0.P_REGRESSION_LOG_OFFSET)
    threshold = params.value(L0.P_REGRESSION_MIN_ABS_DLN_PRICE)
    rng = np.random.default_rng(len(S.ELASTICITY_CLASSES))
    n = len(S.SECTORS) * len(S.SECTORS)
    undev = rng.uniform(0.0, 1.0, n)
    reg = rng.uniform(0.0, 1.0, n)
    builtup = rng.uniform(0.0, 1.0, n)
    parcel = rng.uniform(1.0, len(S.SECTORS), n)
    truth = {"a0": 1.0, "a1": 1.0, "a2": 0.0, "a3": 1.0, "a4": 0.0}
    ln_eps = (
        truth["a0"]
        + truth["a1"] * np.log(1 - undev + offset)
        + truth["a3"] * np.log(1 - builtup + offset)
    )
    dln_price = np.full(n, threshold + 1)
    zones = pd.DataFrame(
        {
            "dln_price": dln_price,
            "dln_floorspace": np.exp(ln_eps) * dln_price,
            "undevelopable_frac": undev,
            "regulatory_index": reg,
            "builtup_frac": builtup,
            "mean_parcel_sqm": parcel,
        }
    )

    fit = L0.fit_elasticity_regression(zones, params)
    for name, value in truth.items():
        assert fit.coefficients[name] == pytest.approx(value, abs=ACCEPTANCE_REL_TOL)
    assert fit.r2 == pytest.approx(1.0, abs=ACCEPTANCE_REL_TOL)


# --------------------------------------------------------------------------------------
# 7.5 jobs by sector
# --------------------------------------------------------------------------------------


def _census_inputs(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """A stand-in Census-2011 ward table and a per-sector growth index."""
    wards = sorted(cells[L0.DEFAULT_WARD_COL].unique())
    rng = np.random.default_rng(len(wards))
    ward_jobs = pd.DataFrame(
        rng.uniform(0.0, len(S.SECTORS), size=(len(wards), len(S.SECTORS))),
        index=pd.Index(wards, name=L0.DEFAULT_WARD_COL),
        columns=list(S.SECTORS),
    )
    growth = {
        sector: 1 + index / len(S.SECTORS) for index, sector in enumerate(S.SECTORS)
    }
    return ward_jobs, growth


def test_jobs_are_conserved_within_each_ward(cells, params):
    ward_jobs, growth = _census_inputs(cells)
    out = L0.assemble_substrate(
        cells, params, ward_jobs_2011=ward_jobs, sector_growth=growth
    )
    jobs = np.stack(out["jobs_by_sector"].to_numpy())
    frame = pd.DataFrame(jobs, index=out[L0.DEFAULT_WARD_COL].to_numpy())
    got = frame.groupby(level=0).sum()

    for ward in ward_jobs.index:
        for index, sector in enumerate(S.SECTORS):
            assert got.loc[ward, index] == pytest.approx(
                ward_jobs.loc[ward, sector] * growth[sector]
            )


def test_jobs_follow_the_dasymetric_surface(cells, params):
    """Within a ward, a cell with twice the commercial floorspace gets twice the jobs."""
    ward = cells[L0.DEFAULT_WARD_COL].value_counts().idxmax()
    frame = cells[cells[L0.DEFAULT_WARD_COL] == ward].copy()
    assert len(frame) > 1
    weights = np.arange(len(frame)) + 1.0
    frame["floorspace_com_sqm"] = weights

    ward_jobs = pd.DataFrame(
        np.ones((1, len(S.SECTORS))),
        index=pd.Index([ward], name=L0.DEFAULT_WARD_COL),
        columns=list(S.SECTORS),
    )
    growth = {sector: 1.0 for sector in S.SECTORS}
    out = L0.assemble_substrate(
        frame, params, ward_jobs_2011=ward_jobs, sector_growth=growth
    )
    jobs = np.stack(out["jobs_by_sector"].to_numpy())[:, 0]
    np.testing.assert_allclose(jobs, weights / weights.sum())


def test_jobs_spread_uniformly_where_the_surface_is_empty(cells, params):
    ward = cells[L0.DEFAULT_WARD_COL].iloc[0]
    frame = cells[cells[L0.DEFAULT_WARD_COL] == ward].copy()
    frame["floorspace_com_sqm"] = 0.0
    frame["retail_poi_count"] = 0.0

    ward_jobs = pd.DataFrame(
        np.ones((1, len(S.SECTORS))),
        index=pd.Index([ward], name=L0.DEFAULT_WARD_COL),
        columns=list(S.SECTORS),
    )
    out = L0.assemble_substrate(
        frame,
        params,
        ward_jobs_2011=ward_jobs,
        sector_growth={sector: 1.0 for sector in S.SECTORS},
    )
    jobs = np.stack(out["jobs_by_sector"].to_numpy())[:, 0]
    np.testing.assert_allclose(jobs, np.full(len(frame), 1 / len(frame)))


def test_jobs_are_left_untouched_without_census_inputs(cells, assembled):
    for got, original in zip(assembled["jobs_by_sector"], cells["jobs_by_sector"]):
        np.testing.assert_allclose(np.asarray(got), np.asarray(original))


def test_a_census_ward_absent_from_the_grid_raises(cells, params):
    from ufe.errors import SchemaValidationError

    ward_jobs, growth = _census_inputs(cells)
    extra = ward_jobs.copy()
    extra.loc["not-a-ward"] = 1.0
    with pytest.raises(SchemaValidationError):
        L0.assemble_substrate(
            cells, params, ward_jobs_2011=extra, sector_growth=growth
        )


def test_jobs_by_sector_keeps_the_schema_list_length(cells, params):
    ward_jobs, growth = _census_inputs(cells)
    out = L0.assemble_substrate(
        cells, params, ward_jobs_2011=ward_jobs, sector_growth=growth
    )
    assert out["jobs_by_sector"].map(len).eq(len(S.SECTORS)).all()
    S.CELLS.validate(out, lazy=True)


# --------------------------------------------------------------------------------------
# purity, determinism, idempotence
# --------------------------------------------------------------------------------------


def test_idempotent_on_an_already_assembled_frame(cells, params):
    once = L0.assemble_substrate(cells, params)
    twice = L0.assemble_substrate(once, params)
    pd.testing.assert_frame_equal(once, twice)


def test_idempotent_with_gates_and_census_inputs(cells, params):
    ward_jobs, growth = _census_inputs(cells)
    hexagon = wkb.loads(bytes(cells["geometry"].iloc[0]))
    kwargs = dict(
        gates={"water": [hexagon]}, ward_jobs_2011=ward_jobs, sector_growth=growth
    )
    once = L0.assemble_substrate(cells, params, **kwargs)
    twice = L0.assemble_substrate(once, params, **kwargs)
    pd.testing.assert_frame_equal(once, twice)


def test_row_order_does_not_change_per_cell_results(cells, params):
    straight = L0.assemble_substrate(cells, params).set_index("h3")
    shuffled = cells.iloc[::-1]
    reversed_out = L0.assemble_substrate(shuffled, params).set_index("h3")
    pd.testing.assert_frame_equal(
        straight, reversed_out.loc[straight.index], check_like=True
    )


def test_no_module_level_mutable_state():
    """Purity guard: the module holds no mutable globals a run could scribble on."""
    for name in dir(L0):
        if name.startswith("_"):
            continue
        value = getattr(L0, name)
        assert not isinstance(value, (list, dict, set)), name
