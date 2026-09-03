"""Tests for Layer 4, supply (spec Section 11).

The Section 11 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "``headroom_sqm`` never negative" -> ``test_acc_headroom_never_negative``
* "A cell with 30 months inventory has a damping factor below 0.5"
  -> ``test_acc_thirty_months_inventory_damps_below_half``
* "Sterilising 40 acres reduces ``capacity_sqm`` by exactly
  ``40 * 4046.86 * permitted_far``" -> ``test_acc_sterilisation_reduces_capacity_exactly``
* "Total city capacity across years is non-increasing absent explicit FAR changes"
  -> ``test_acc_capacity_non_increasing_without_far_changes``

Everything else exercises 11.1-11.3 directly: the absorption cap actually binding, supply
never exceeding headroom, carried-forward state round-tripping across a multi-year loop,
and a zero-demand no-op. No model numbers are hand-written; expected values are recomputed
from the YAML through ``Params``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.layers import l2_shocks as L2
from ufe.layers import l4_supply as L4
from ufe.params import load_params

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    build_city,
    synthetic_cells,
    synthetic_city,
)

CITY = "vizag"
BASE_YEAR = 2024


def test_supply_effect_is_the_canonical_layer2_dataclass():
    """`l4_supply` re-exports Layer 2's `SupplyEffect`; it no longer defines its own.

    `l4_supply.SupplyEffect` must stay a valid reference (existing callers use it), but it
    must be literally the same class Layer 2 emits -- not a look-alike copy.
    """
    assert L4.SupplyEffect is L2.SupplyEffect
    assert "SupplyEffect" in L4.__all__

    effect = L4.SupplyEffect(
        cell="x", delta_floorspace_sqm=0.0, delta_capacity_sqm=0.0, start_year=BASE_YEAR
    )
    assert isinstance(effect, L2.SupplyEffect)
    # `project_id` is Layer 2's extra field and defaults, so the old 4-arg call still works.
    assert effect.project_id == ""


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture()
def cells(synthetic_city):  # noqa: F811
    return synthetic_city.cells


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _first_h3(cells: pd.DataFrame) -> str:
    return str(cells["h3"].iloc[0])


# --------------------------------------------------------------------------------------
# ACCEPTANCE
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_headroom_never_negative(cells, params):
    out = L4.apply_supply(cells, params, year=BASE_YEAR)
    assert (out["headroom_sqm"] >= 0).all()

    # Push demand far above capacity everywhere and re-check.
    huge_demand = pd.Series(
        np.full(len(cells), cells["capacity_sqm"].max() * 1000), index=cells["h3"]
    )
    out2 = L4.apply_supply(cells, params, year=BASE_YEAR, demand_sqm=huge_demand)
    assert (out2["headroom_sqm"] >= 0).all()


@pytest.mark.acceptance
def test_acc_thirty_months_inventory_damps_below_half(params):
    damping = L4.inventory_damping(np.array([30.0]), params)
    assert damping[0] < 1 / 2


@pytest.mark.acceptance
def test_acc_sterilisation_reduces_capacity_exactly(cells, params):
    sqm_per_acre = params.value("supply.units.sqm_per_acre")
    acres = 40
    cell_h3 = _first_h3(cells)
    permitted_far = float(cells.loc[cells["h3"] == cell_h3, "permitted_far"].iloc[0])
    expected_delta = acres * sqm_per_acre * permitted_far

    # Give the cell ample capacity so the sterilisation is not clamped at zero by the
    # non-negativity floor -- the point of this test is the exact-delta subtraction.
    one_cell = cells.copy()
    one_cell.loc[one_cell["h3"] == cell_h3, "capacity_sqm"] = expected_delta * 10

    effect = L4.SupplyEffect(
        cell=cell_h3,
        delta_floorspace_sqm=0.0,
        delta_capacity_sqm=-expected_delta,
        start_year=BASE_YEAR,
    )
    out = L4.apply_supply(one_cell, params, year=BASE_YEAR, effects=[effect])
    before = float(one_cell.loc[one_cell["h3"] == cell_h3, "capacity_sqm"].iloc[0])
    after = float(out.loc[out["h3"] == cell_h3, "capacity_sqm"].iloc[0])
    assert before - after == pytest.approx(expected_delta)


@pytest.mark.acceptance
def test_acc_capacity_non_increasing_without_far_changes(cells, params):
    """No `SupplyEffect` at all -> capacity can only shrink (it never shrinks here, but
    it must never grow, i.e. total city capacity is non-increasing year over year)."""
    state = None
    totals = []
    current = cells
    for year in range(BASE_YEAR, BASE_YEAR + 5):
        out = L4.apply_supply(current, params, year=year, state=state)
        state = out.attrs[L4.ATTR_KEY]
        totals.append(out["capacity_sqm"].sum())
        current = out
    for earlier, later in zip(totals, totals[1:]):
        assert later <= earlier + 1e-6  # tolerance only, not a model parameter


# --------------------------------------------------------------------------------------
# 11.1 state carried forward
# --------------------------------------------------------------------------------------


def test_state_round_trips_across_multi_year_loop(cells, params):
    """Threading `.attrs[ATTR_KEY]` across years must match a from-scratch derivation."""
    state = None
    current = cells
    for year in range(BASE_YEAR, BASE_YEAR + 3):
        out = L4.apply_supply(current, params, year=year, state=state)
        state = out.attrs[L4.ATTR_KEY]
        assert isinstance(state, L4.SupplyState)
        assert state.base_year == BASE_YEAR
        assert list(state.capacity_sqm.index) == list(cells["h3"])
        current = out
    # after three no-demand years, nothing should have moved
    assert (state.built_sqm.to_numpy() == pytest.approx(
        (cells["floorspace_res_sqm"] + cells["floorspace_com_sqm"]).to_numpy()
    ))


def test_initial_state_seeded_from_cells_when_state_omitted(cells, params):
    out = L4.apply_supply(cells, params, year=BASE_YEAR, state=None)
    state = out.attrs[L4.ATTR_KEY]
    expected_built = (cells["floorspace_res_sqm"] + cells["floorspace_com_sqm"]).to_numpy()
    assert state.built_sqm.to_numpy() == pytest.approx(expected_built)
    assert state.base_year == BASE_YEAR


def test_apply_supply_returns_same_index_and_row_count(cells, params):
    out = L4.apply_supply(cells, params, year=BASE_YEAR)
    assert len(out) == len(cells)
    assert list(out.index) == list(cells.index)


def test_apply_supply_is_pure_does_not_mutate_input(cells, params):
    before = cells.copy(deep=True)
    L4.apply_supply(cells, params, year=BASE_YEAR)
    pd.testing.assert_frame_equal(cells, before)


# --------------------------------------------------------------------------------------
# 11.2 absorption cap
# --------------------------------------------------------------------------------------


def test_absorption_cap_binds_delivered_equals_cap(cells, params):
    """Construct a cell whose demand exceeds both the cap and headroom is NOT the binding
    constraint (headroom set generously large), so the cap alone determines delivery."""
    h3_id = _first_h3(cells)
    hist_absorption = 1000.0
    state = L4.SupplyState(
        built_sqm=pd.Series([0.0], index=[h3_id]),
        capacity_sqm=pd.Series([1e9], index=[h3_id]),
        headroom_sqm=pd.Series([1e9], index=[h3_id]),
        inventory_months=pd.Series([1.0], index=[h3_id]),
        hist_absorption_sqm=pd.Series([hist_absorption], index=[h3_id]),
        committed_backlog_sqm=pd.Series([0.0], index=[h3_id]),
        base_year=BASE_YEAR,
    )
    one_cell = cells[cells["h3"] == h3_id].copy()
    demand = pd.Series([hist_absorption * 100], index=[h3_id])  # demand >> cap

    out = L4.apply_supply(
        one_cell, params, year=BASE_YEAR, state=state, demand_sqm=demand
    )
    expected_cap = out.attrs["absorption_cap_sqm"].loc[h3_id]
    delivered = out.attrs["delivered_sqm"].loc[h3_id]

    assert delivered == pytest.approx(expected_cap)
    assert delivered < demand.loc[h3_id]
    # the cap itself must be strictly less than the outsized demand for this to be a
    # meaningful test of the cap binding
    assert expected_cap < demand.loc[h3_id]


def test_supply_never_exceeds_headroom(cells, params):
    """Even when the absorption cap is generous, delivered supply cannot exceed headroom."""
    h3_id = _first_h3(cells)
    small_headroom = 5.0
    state = L4.SupplyState(
        built_sqm=pd.Series([0.0], index=[h3_id]),
        capacity_sqm=pd.Series([small_headroom], index=[h3_id]),
        headroom_sqm=pd.Series([small_headroom], index=[h3_id]),
        inventory_months=pd.Series([1.0], index=[h3_id]),
        hist_absorption_sqm=pd.Series([1e9], index=[h3_id]),  # huge cap
        committed_backlog_sqm=pd.Series([0.0], index=[h3_id]),
        base_year=BASE_YEAR,
    )
    one_cell = cells[cells["h3"] == h3_id].copy()
    demand = pd.Series([1e9], index=[h3_id])

    out = L4.apply_supply(
        one_cell, params, year=BASE_YEAR, state=state, demand_sqm=demand
    )
    new_state = out.attrs[L4.ATTR_KEY]
    assert new_state.built_sqm.loc[h3_id] <= small_headroom + 1e-9
    assert out.loc[out["h3"] == h3_id, "headroom_sqm"].iloc[0] >= 0


def test_relative_attractiveness_normalised_to_city_mean_one():
    utility = np.array([0.1, 5.0, -3.0, 2.0])
    rel = L4.relative_attractiveness(utility)
    assert rel.mean() == pytest.approx(1.0)
    # the highest-utility cell must have the highest relative attractiveness
    assert np.argmax(rel) == np.argmax(utility)


def test_relative_attractiveness_uniform_utility_gives_all_ones():
    utility = np.zeros(6)
    rel = L4.relative_attractiveness(utility)
    assert rel == pytest.approx(np.ones(6))


def test_inventory_damping_monotonically_decreasing(params):
    months = np.array([1.0, 5.0, 18.0, 30.0, 60.0])
    damping = L4.inventory_damping(months, params)
    assert np.all(np.diff(damping) <= 0)
    assert (damping >= 0).all() and (damping <= 1).all()


def test_absorption_cap_grows_with_base_growth(params):
    hist = np.array([100.0])
    rel = np.array([1.0])
    damping = np.array([1.0])
    cap_t0 = L4.absorption_cap_sqm(hist, BASE_YEAR, BASE_YEAR, rel, damping, params)
    cap_t5 = L4.absorption_cap_sqm(hist, BASE_YEAR + 5, BASE_YEAR, rel, damping, params)
    base_growth = params.value("supply.absorption.base_growth")
    assert cap_t5[0] == pytest.approx(hist[0] * (1 + base_growth) ** 5)
    if base_growth > 0:
        assert cap_t5[0] > cap_t0[0]


# --------------------------------------------------------------------------------------
# 11.3 applying supply effects
# --------------------------------------------------------------------------------------


def test_township_backlog_delivers_gradually_not_instantly(cells, params):
    """A launched township (positive `delta_floorspace_sqm`) must not appear in `built_sqm`
    in one shot when the absorption cap is small relative to its size."""
    h3_id = _first_h3(cells)
    township_sqm = 100_000.0
    # Small relative to the township, but large enough (compounded over decades at the
    # configured base_growth rate) to fully deliver within the test's year horizon.
    small_cap_hist_absorption = 2_000.0

    one_cell = cells[cells["h3"] == h3_id].copy()
    one_cell.loc[:, "capacity_sqm"] = township_sqm * 10
    one_cell.loc[:, "headroom_sqm"] = township_sqm * 10
    one_cell.loc[:, "floorspace_res_sqm"] = 0.0
    one_cell.loc[:, "floorspace_com_sqm"] = 0.0
    one_cell.loc[:, "hist_absorption_sqm"] = small_cap_hist_absorption
    one_cell.loc[:, "inventory_months"] = 1.0

    effect = L4.SupplyEffect(
        cell=h3_id,
        delta_floorspace_sqm=township_sqm,
        delta_capacity_sqm=0.0,
        start_year=BASE_YEAR,
    )
    out = L4.apply_supply(one_cell, params, year=BASE_YEAR, effects=[effect])
    state = out.attrs[L4.ATTR_KEY]

    # not delivered in one shot
    assert state.built_sqm.loc[h3_id] < township_sqm
    assert state.built_sqm.loc[h3_id] > 0
    # the remainder sits in the backlog, to be delivered in later years
    assert state.committed_backlog_sqm.loc[h3_id] == pytest.approx(
        township_sqm - state.built_sqm.loc[h3_id]
    )

    # letting it run for many years eventually delivers (approximately) the whole township
    current = out
    state2 = state
    for year in range(BASE_YEAR + 1, BASE_YEAR + 50):
        current = L4.apply_supply(current, params, year=year, state=state2)
        state2 = current.attrs[L4.ATTR_KEY]
    assert state2.built_sqm.loc[h3_id] == pytest.approx(township_sqm, rel=1e-3)
    assert state2.committed_backlog_sqm.loc[h3_id] == pytest.approx(0.0, abs=1.0)


def test_sterilisation_effect_ignored_in_other_years(cells, params):
    h3_id = _first_h3(cells)
    one_cell = cells[cells["h3"] == h3_id].copy()
    before_capacity = float(one_cell["capacity_sqm"].iloc[0])
    effect = L4.SupplyEffect(
        cell=h3_id, delta_floorspace_sqm=0.0, delta_capacity_sqm=-1.0, start_year=BASE_YEAR + 1
    )
    out = L4.apply_supply(one_cell, params, year=BASE_YEAR, effects=[effect])
    assert out.loc[out["h3"] == h3_id, "capacity_sqm"].iloc[0] == pytest.approx(
        before_capacity
    )


def test_supply_effect_unknown_cell_raises(cells, params):
    effect = L4.SupplyEffect(
        cell="not-a-real-cell", delta_floorspace_sqm=1.0, delta_capacity_sqm=0.0,
        start_year=BASE_YEAR,
    )
    with pytest.raises(Exception):
        L4.apply_supply(cells, params, year=BASE_YEAR, effects=[effect])


# --------------------------------------------------------------------------------------
# purity / no-op
# --------------------------------------------------------------------------------------


def test_zero_demand_is_a_pure_no_op(cells, params):
    out = L4.apply_supply(cells, params, year=BASE_YEAR)
    state = out.attrs[L4.ATTR_KEY]

    np.testing.assert_allclose(
        out["capacity_sqm"].to_numpy(), cells["capacity_sqm"].to_numpy()
    )
    np.testing.assert_allclose(
        out["headroom_sqm"].to_numpy(), cells["headroom_sqm"].to_numpy()
    )
    np.testing.assert_allclose(
        state.built_sqm.to_numpy(),
        (cells["floorspace_res_sqm"] + cells["floorspace_com_sqm"]).to_numpy(),
    )
    assert (state.committed_backlog_sqm == 0).all()

    # calling twice with no demand gives byte-identical results (determinism)
    out2 = L4.apply_supply(cells, params, year=BASE_YEAR)
    pd.testing.assert_frame_equal(
        out[list(L4.CARRIED_FORWARD_COLUMNS)], out2[list(L4.CARRIED_FORWARD_COLUMNS)]
    )


def test_no_unseeded_randomness_same_seed_same_result(cells, params):
    out1 = L4.apply_supply(cells, params, year=BASE_YEAR)
    out2 = L4.apply_supply(cells, params, year=BASE_YEAR)
    pd.testing.assert_frame_equal(
        out1[list(L4.CARRIED_FORWARD_COLUMNS)], out2[list(L4.CARRIED_FORWARD_COLUMNS)]
    )
