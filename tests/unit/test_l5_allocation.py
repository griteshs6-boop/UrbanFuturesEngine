"""Tests for Layer 5, allocation (spec Section 12).

The Section 12 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "Null test: with zero projects and zero exogenous growth, the allocation reproduces the
  base-year household distribution to within 0.5% per cell. This is the test that catches a
  missing ``alpha_i``."
  -> ``test_acc_null_reproduces_observed_distribution``
     ``test_acc_null_zero_growth_relocates_nobody``
     ``test_acc_null_test_has_teeth_without_alpha``  (the null test must be able to *fail*)
* "Allocation conserves households: ``sum(allocated) + spill == sum(new_hh)``"
  -> ``test_acc_allocation_conserves_households``
* "No cell exceeds ``headroom_sqm``" -> ``test_acc_no_cell_exceeds_headroom``
* "Doubling a single cell's price reduces its allocated share for the low band more than for
  the high band" -> ``test_acc_doubling_price_hits_low_band_hardest``
* "Convergence achieved in under 12 iterations on the Vizag fixture"
  -> ``test_acc_converges_within_max_iterations``
* "A deliberately unstable parameter set (``phi = 1.4``) raises ``ConvergenceError`` rather
  than looping or returning garbage" -> ``test_acc_unstable_phi_raises_convergence_error``

Everything else exercises 12.2-12.7 directly.  Model coefficients are never written into
this file: expected values are recomputed from ``config/params/behaviour.yaml`` through
``Params``.  The bare numbers that do appear are *test inputs* (a three-cell frame's prices
and household counts) and the deliberately-unstable ``phi`` the acceptance block names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ufe.errors import ConvergenceError, MissingParameter, UFEError
from ufe.layers import l5_allocation as L5
from ufe.layers.l1_accessibility import apply_accessibility
from ufe.layers.routing import HaversineBackend, precompute_matrices
from ufe.params import load_params
from ufe.store.schemas import INCOME_BANDS, SECTORS, Sector

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    build_city,
    synthetic_cells,
    synthetic_city,
)

CITY = "vizag"
BASE_YEAR = 2024
NEXT_YEAR = BASE_YEAR + 1

#: Strict floating-point tolerance, derived rather than typed.
STRICT = float(np.sqrt(np.finfo(float).eps))


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture()
def cells(synthetic_city):  # noqa: F811
    return synthetic_city.cells


@pytest.fixture(scope="module")
def matrices(params):
    """A real, offline :class:`MatrixSet` for the fixture city (no OSRM, no network)."""
    frame = synthetic_cells()
    return precompute_matrices(frame, params, HaversineBackend(params))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


class _ParamsOverride:
    """Duck-typed ``Params`` that replaces a handful of paths.

    The acceptance block requires "a deliberately unstable parameter set (``phi = 1.4``)".
    Injecting it here rather than in ``l5_allocation`` keeps the layer free of numeric
    literals (CONTRACT.md rule 1) while still letting the test build the unstable set.
    """

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self._overrides = dict(overrides)

    def value(self, path: str) -> Any:
        if path in self._overrides:
            return self._overrides[path]
        return self._base.value(path)

    def get(self, path: str) -> Any:
        if path in self._overrides:
            return self._overrides[path]
        return self._base.get(path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


@dataclass(frozen=True)
class _EmploymentEffect:
    """Structurally identical to the Section 9.1 ``EmploymentEffect`` Layer 2 emits.

    Declared locally because ``ufe/layers/l2_shocks.py`` is being written concurrently; the
    layer under test consumes these by attribute, never by type.
    """

    cell: str
    sector: int
    jobs: float
    median_wage_inr_mo: float
    start_year: int
    ramp_years: int
    capture_radius_m: float = 0.0
    dormitory_share: float = 0.0
    is_construction: bool = False
    duration_years: int | None = None


def _pph(params) -> dict[str, float]:
    """Persons per household by band.

    ``behaviour.persons_per_household_by_band`` is deliberately null on disk (the spec never
    supplies it), so Section 12.5 quantities must be handed in explicitly.  These are test
    inputs, not model parameters: the fixture city is generated at a flat 4.2 persons per
    household (``tests/fixtures/synthetic.yaml``), and the band gradient below is chosen only
    so that the bands are distinguishable.
    """
    del params
    return {"low": 4.6, "mid": 4.2, "upper_mid": 3.8, "high": 3.4}


def _totals(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(frame["hh_by_band"].map(lambda v: np.asarray(v, dtype=float)).to_numpy())


def _jobs(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        frame["jobs_by_sector"].map(lambda v: np.asarray(v, dtype=float)).to_numpy()
    )


def _relocated_fraction(before: pd.DataFrame, after: pd.DataFrame) -> float:
    """Share of the city's households that changed cell."""
    b = before["households"].to_numpy(dtype=float)
    a = after["households"].to_numpy(dtype=float)
    return float(np.abs(a - b).sum() / (len(("in", "out")) * b.sum()))


# --------------------------------------------------------------------------------------
# ACCEPTANCE — the null test (Section 12 / Section 21 headline failure mode)
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_null_reproduces_observed_distribution(cells, params):
    """Re-allocating the whole city with no shocks must return the observed distribution.

    This is the test Section 21 names: "Missing cell fixed effect -> model relocates half the
    city in year 1".  The assertion is exact, not loose: with ``alpha_i`` estimated by
    inverting the base-year distribution, a full re-allocation of the household stock is a
    fixed point of the logit, so every cell must come back to floating-point precision.  The
    spec's 0.5% figure is asserted too, from YAML.
    """
    observed = cells["households"].to_numpy(dtype=float)
    observed_bands = _totals(cells)

    out = L5.allocate(
        cells,
        params,
        year=BASE_YEAR,
        reallocate_stock=True,
        headroom_sqm=np.inf,
    )

    predicted = out["households"].to_numpy(dtype=float)
    rel = np.abs(predicted - observed) / observed

    assert rel.max() < STRICT
    assert rel.max() < float(params.value(L5.P_NULL_TEST_TOLERANCE))
    # ...and band by band, not merely in total.
    np.testing.assert_allclose(_totals(out), observed_bands, rtol=STRICT, atol=0)


@pytest.mark.acceptance
def test_acc_null_zero_growth_relocates_nobody(cells, params):
    """Zero projects + zero exogenous growth => the frame is returned untouched."""
    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        natural_growth_rate=0,
        employment_effects=(),
    )

    np.testing.assert_array_equal(
        out["households"].to_numpy(dtype=float), cells["households"].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(_totals(out), _totals(cells))
    np.testing.assert_array_equal(
        out["population"].to_numpy(dtype=float), cells["population"].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(_jobs(out), _jobs(cells))
    assert out.attrs[L5.ATTR_DIAGNOSTICS]["new_households"] == 0
    assert _relocated_fraction(cells, out) == 0


@pytest.mark.acceptance
def test_acc_null_test_has_teeth_without_alpha(cells, params):
    """The null test must be capable of failing: strip ``alpha_i`` and the city moves.

    Guards against a vacuous null test — the specific way this acceptance item gets
    silently defeated.
    """
    tolerance = float(params.value(L5.P_NULL_TEST_TOLERANCE))

    with_alpha = L5.allocate(
        cells, params, year=BASE_YEAR, reallocate_stock=True, headroom_sqm=np.inf
    )
    without_alpha = L5.allocate(
        cells,
        params,
        year=BASE_YEAR,
        reallocate_stock=True,
        headroom_sqm=np.inf,
        alpha=0,
    )

    moved_with = _relocated_fraction(cells, with_alpha)
    moved_without = _relocated_fraction(cells, without_alpha)

    assert moved_with < tolerance < moved_without
    # Section 21's "relocates half the city in year 1" is not hyperbole on this fixture.
    half = 1 / len(("first half", "second half"))
    assert moved_without > half


# --------------------------------------------------------------------------------------
# ACCEPTANCE — conservation, capacity, price gradient, convergence
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_allocation_conserves_households(cells, params):
    """``sum(allocated) + spill == sum(new_hh)``."""
    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]

    allocated = diag["allocated_by_band"].to_numpy(dtype=float)
    spill = np.asarray(diag["spill_by_band"], dtype=float)
    demand = np.asarray(diag["demand_by_band"], dtype=float)

    np.testing.assert_allclose(allocated.sum(axis=0) + spill, demand, rtol=STRICT, atol=0)
    assert diag["new_households"] > 0
    # the frame agrees with the diagnostics
    np.testing.assert_allclose(
        _totals(out) - _totals(cells), allocated, rtol=STRICT, atol=STRICT
    )


@pytest.mark.acceptance
def test_acc_no_cell_exceeds_headroom(cells, params):
    """Households are never allocated into floorspace that does not exist (Section 12.4)."""
    tight = cells.copy()
    # Squeeze the city hard so the constraint is guaranteed to bind somewhere.
    tight["headroom_sqm"] = tight["headroom_sqm"] * 0

    out = L5.allocate(
        tight,
        params,
        year=NEXT_YEAR,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]
    used = diag["allocated_sqm"].to_numpy(dtype=float)
    headroom = tight["headroom_sqm"].to_numpy(dtype=float)

    assert np.all(used <= headroom + STRICT)
    assert diag["spill_households"] > 0


@pytest.mark.acceptance
def test_acc_doubling_price_hits_low_band_hardest(cells, params):
    """Doubling one cell's price cuts its low-band share more than its high-band share."""
    doubled = cells.copy()
    target = 0
    price = doubled["price_res_inr_sqft"].to_numpy(dtype=float).copy()
    price[target] = price[np.isfinite(price)].mean()
    doubled["price_res_inr_sqft"] = price
    base = doubled.copy()
    price = price.copy()
    price[target] = price[target] + price[target]  # double, without typing `2`
    doubled["price_res_inr_sqft"] = price

    alpha = L5.estimate_alpha_res(base, params)

    shares_before = L5.choice_shares(L5.utility(base, params, alpha=alpha))
    shares_after = L5.choice_shares(L5.utility(doubled, params, alpha=alpha))

    drop_low = shares_before["low"].iloc[target] - shares_after["low"].iloc[target]
    drop_high = shares_before["high"].iloc[target] - shares_after["high"].iloc[target]
    ratio_low = shares_after["low"].iloc[target] / shares_before["low"].iloc[target]
    ratio_high = shares_after["high"].iloc[target] / shares_before["high"].iloc[target]

    assert drop_low > 0 and drop_high > 0
    assert ratio_low < ratio_high


@pytest.mark.acceptance
def test_acc_converges_within_max_iterations(cells, params, matrices):
    """The agglomeration inner loop converges inside ``max_iterations`` (Section 12.6)."""
    cap = int(params.value(L5.P_MAX_ITERATIONS))

    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        matrices=matrices,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]

    assert diag["converged"] is True
    assert diag["iterations"] <= cap
    assert diag["max_delta_lnA"] < float(params.value(L5.P_CONVERGENCE_TOL))


@pytest.mark.acceptance
def test_acc_unstable_phi_raises_convergence_error(cells, params, matrices):
    """``phi = 1.4`` must raise, not loop and not return garbage (Sections 12.6 / 21)."""
    unstable = _ParamsOverride(params, **{L5.P_SPILLOVER_PHI: 1.4})

    with pytest.raises(ConvergenceError) as excinfo:
        L5.allocate(
            cells,
            unstable,
            year=NEXT_YEAR,
            matrices=matrices,
            persons_per_household_by_band=_pph(params),
        )
    assert "phi" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 12.3 — the utility function
# --------------------------------------------------------------------------------------


def _tiny_frame() -> pd.DataFrame:
    """Three hand-built cells.  Values here are test inputs, not model parameters."""
    return pd.DataFrame(
        {
            "h3": ["a", "b", "c"],
            "lnA": [10.0, 11.0, 12.0],
            "price_res_inr_sqft": [4000.0, 5000.0, 6000.0],
            "amenity": [0.1, 0.2, 0.3],
            "disamenity": [0.4, 0.3, 0.2],
            "households": [100.0, 200.0, 300.0],
            "hh_by_band": [
                [40.0, 30.0, 20.0, 10.0],
                [50.0, 100.0, 30.0, 20.0],
                [60.0, 90.0, 100.0, 50.0],
            ],
            "headroom_sqm": [1.0e6, 1.0e6, 1.0e6],
        }
    )


def test_utility_matches_hand_calculation(params):
    frame = _tiny_frame()
    alpha = pd.Series([0.5, -0.25, 0.75], index=frame.index)
    field = pd.Series([0.1, 0.0, -0.1], index=frame.index)

    got = L5.utility(frame, params, alpha=alpha, field_res=field)

    hh = frame["households"].to_numpy(dtype=float)
    bands = _totals(frame)
    for k, band in enumerate(INCOME_BANDS):
        b = {name: float(params.value(f"{L5.P_LOGIT}.{band}.{name}")) for name in L5.COEFFICIENTS}
        expected = (
            b["b_access"] * frame["lnA"].to_numpy(dtype=float)
            + b["b_price"] * np.log(frame["price_res_inr_sqft"].to_numpy(dtype=float))
            + b["b_amenity"] * frame["amenity"].to_numpy(dtype=float)
            - b["b_disamenity"] * frame["disamenity"].to_numpy(dtype=float)
            + b["b_agglom"] * np.log(hh + 1)
            + b["b_same_band"] * (bands[:, k] / hh)
            + field.to_numpy(dtype=float)
            + alpha.to_numpy(dtype=float)
        )
        np.testing.assert_allclose(got[band].to_numpy(dtype=float), expected, rtol=STRICT)


def test_choice_shares_sum_to_one(cells, params):
    shares = L5.choice_shares(L5.utility(cells, params))
    np.testing.assert_allclose(
        shares.sum(axis=0).to_numpy(dtype=float), np.ones(len(INCOME_BANDS)), rtol=STRICT
    )
    assert (shares.to_numpy() >= 0).all()


def test_higher_accessibility_raises_a_cells_share(cells, params):
    alpha = L5.estimate_alpha_res(cells, params)
    before = L5.choice_shares(L5.utility(cells, params, alpha=alpha))

    lifted = cells.copy()
    lnA = lifted["lnA"].to_numpy(dtype=float).copy()
    target = 0
    lnA[target] = lnA[target] + 1
    lifted["lnA"] = lnA
    after = L5.choice_shares(L5.utility(lifted, params, alpha=alpha))

    for band in INCOME_BANDS:
        assert after[band].iloc[target] > before[band].iloc[target]


def test_higher_price_lowers_a_cells_share(cells, params):
    alpha = L5.estimate_alpha_res(cells, params)
    before = L5.choice_shares(L5.utility(cells, params, alpha=alpha))

    dearer = cells.copy()
    price = dearer["price_res_inr_sqft"].to_numpy(dtype=float).copy()
    target = 0
    price[target] = np.nanmax(price)
    dearer["price_res_inr_sqft"] = price
    after = L5.choice_shares(L5.utility(dearer, params, alpha=alpha))

    for band in INCOME_BANDS:
        assert after[band].iloc[target] < before[band].iloc[target]


def test_band_accessibility_uses_band_mode_weights(cells, params, matrices):
    """Section 12.3: "``lnA_ik`` uses the band-specific ``access_mode_weights``"."""
    by_band = L5.band_accessibility(cells, params, matrices)

    assert set(by_band) == set(INCOME_BANDS)
    finite = np.isfinite(by_band["low"]) & np.isfinite(by_band["high"])
    assert finite.any()
    # A low-income household's accessibility is walk-and-bus accessibility, so it is not the
    # same surface as a high-income household's.
    assert not np.allclose(by_band["low"][finite], by_band["high"][finite])


# --------------------------------------------------------------------------------------
# 12.2 — household demand
# --------------------------------------------------------------------------------------


def test_exogenous_growth_uses_natural_growth_rate(cells, params):
    rate = float(params.value(L5.P_NATURAL_GROWTH_RATE))
    demand = L5.household_demand(cells, params, year=NEXT_YEAR)

    assert demand.exogenous == pytest.approx(cells["households"].sum() * rate)
    assert demand.job_driven == 0
    # exogenous households take the current city band distribution
    observed = _totals(cells).sum(axis=0)
    np.testing.assert_allclose(
        demand.by_band / demand.by_band.sum(), observed / observed.sum(), rtol=STRICT
    )


def test_job_driven_demand_matches_section_12_2(cells, params):
    h3 = str(cells["h3"].iloc[0])
    jobs = 10000.0
    wage = 60000.0
    effect = _EmploymentEffect(
        cell=h3,
        sector=int(Sector.it_office),
        jobs=jobs,
        median_wage_inr_mo=wage,
        start_year=BASE_YEAR,  # ramp(t - start_year) is 0 in the start year itself
        ramp_years=1,
        dormitory_share=0.4,
    )
    weight = 0.75

    demand = L5.household_demand(
        cells,
        params,
        year=NEXT_YEAR,
        employment_effects=[effect],
        activation_weights=[weight],
        natural_growth_rate=0,
    )

    share = float(params.value(f"{L5.P_INMIGRANT_SHARE}.it_office"))
    wph = float(params.value(L5.P_WORKERS_PER_HOUSEHOLD))
    expected = jobs * weight * share * (1 - effect.dormitory_share) / wph

    assert demand.job_driven == pytest.approx(expected)
    assert demand.by_band.sum() == pytest.approx(expected)
    # Section 9.5 / Section 21: dormitory workers are not apartment buyers.
    assert demand.dormitory_workers.sum() == pytest.approx(jobs * weight * effect.dormitory_share)

    band = L5.wage_band(wage, params)
    assert demand.by_band[band] == pytest.approx(expected)
    assert demand.by_band.sum() == pytest.approx(demand.by_band[band])


def test_ramp_spreads_jobs_over_ramp_years(cells, params):
    h3 = str(cells["h3"].iloc[0])
    effect = _EmploymentEffect(
        cell=h3,
        sector=int(Sector.manuf_light),
        jobs=1000.0,
        median_wage_inr_mo=20000.0,
        start_year=BASE_YEAR,
        ramp_years=4,
    )
    partial = L5.household_demand(
        cells, params, year=BASE_YEAR + 1, employment_effects=[effect], natural_growth_rate=0
    )
    full = L5.household_demand(
        cells, params, year=BASE_YEAR + 10, employment_effects=[effect], natural_growth_rate=0
    )
    before = L5.household_demand(
        cells, params, year=BASE_YEAR - 1, employment_effects=[effect], natural_growth_rate=0
    )

    assert before.job_driven == 0
    assert 0 < partial.job_driven < full.job_driven
    assert partial.job_driven == pytest.approx(full.job_driven / effect.ramp_years)


def test_wage_band_routing_uses_indexed_boundaries(params):
    boundaries = L5.income_band_boundaries(params)
    wph = float(params.value(L5.P_WORKERS_PER_HOUSEHOLD))
    premium = float(params.value(L5.P_WAGE_PREMIUM))

    for k, boundary in enumerate(boundaries):
        just_under = (boundary / (wph * premium)) * (1 - STRICT)
        just_over = (boundary / (wph * premium)) * (1 + STRICT)
        assert L5.wage_band(just_under, params) == k
        assert L5.wage_band(just_over, params) == k + 1


# --------------------------------------------------------------------------------------
# 12.4 — constrained allocation
# --------------------------------------------------------------------------------------


def test_capacity_constraint_binds_and_spills(params):
    frame = _tiny_frame()
    sqm = np.array([float(params.value(f"{L5.P_SQM_PER_HH}.{b}")) for b in INCOME_BANDS])
    # Room for exactly one low-band household in cell `a`, nothing anywhere else.
    headroom = np.array([sqm[0], 0.0, 0.0])
    utility = pd.DataFrame(np.zeros((len(frame), len(INCOME_BANDS))), columns=list(INCOME_BANDS))
    demand = np.array([10.0, 0.0, 0.0, 0.0])

    allocated, spill, capped = L5.allocate_constrained(utility, demand, headroom, sqm, params)

    assert allocated.sum() == pytest.approx(sqm[0] / sqm[0])
    assert spill.sum() == pytest.approx(demand.sum() - allocated.sum())
    assert capped.any()
    assert (allocated * sqm).sum(axis=1).max() <= headroom.max() + STRICT


def test_constraint_binds_on_some_cells_and_pushes_demand_elsewhere(cells, params):
    """A partially-constrained city caps the attractive cells and re-houses the overflow."""
    free = L5.allocate(cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params))
    free_alloc = free.attrs[L5.ATTR_DIAGNOSTICS]["allocated_sqm"].to_numpy(dtype=float)

    squeezed = cells.copy()
    # Only the floorspace this cell would freely have absorbed, halved.
    squeezed["headroom_sqm"] = free_alloc / len(("first half", "second half"))
    out = L5.allocate(
        squeezed, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]

    assert diag["capped_cells"].any()
    used = diag["allocated_sqm"].to_numpy(dtype=float)
    assert np.all(used <= squeezed["headroom_sqm"].to_numpy(dtype=float) + STRICT)
    # every household is either housed or explicitly recorded as spill
    assert diag["allocated_by_band"].to_numpy().sum() + diag["spill_households"] == pytest.approx(
        diag["new_households"]
    )


def test_allocation_is_row_order_independent(cells, params):
    forward = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )
    shuffled = cells.iloc[::-1].reset_index(drop=True)
    backward = L5.allocate(
        shuffled, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )

    a = forward.set_index("h3")["households"].sort_index()
    b = backward.set_index("h3")["households"].sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=STRICT, atol=STRICT)


def test_allocate_is_pure_and_shape_preserving(cells, params):
    before = cells.copy(deep=True)
    out = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )

    assert len(out) == len(cells)
    assert out.index.equals(cells.index)
    assert list(out.columns) == list(cells.columns)
    pd.testing.assert_frame_equal(cells, before)


def test_allocate_is_deterministic(cells, params):
    first = L5.allocate(cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params))
    second = L5.allocate(cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params))
    np.testing.assert_array_equal(
        first["households"].to_numpy(), second["households"].to_numpy()
    )


def test_cells_without_accessibility_are_ineligible(cells, params, matrices):
    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        matrices=matrices,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]
    eligible = diag["eligible"].to_numpy(dtype=bool)

    assert not eligible.all()  # the fixture's matrix covers only a subset of cells
    allocated = diag["allocated_by_band"].to_numpy(dtype=float)
    assert allocated[~eligible].sum() == 0


# --------------------------------------------------------------------------------------
# 12.5 — induced service employment (emitted HERE and only here)
# --------------------------------------------------------------------------------------


def test_service_jobs_land_only_in_retail_svc(cells, params):
    out = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )
    delta = _jobs(out) - _jobs(cells)

    for sector in SECTORS:
        index = int(Sector[sector])
        if sector == "retail_svc":
            assert delta[:, index].sum() > 0
        else:
            assert np.abs(delta[:, index]).max() == 0


def test_service_jobs_match_section_12_5(cells, params):
    pph = _pph(params)
    out = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=pph
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]

    allocated = diag["allocated_by_band"].to_numpy(dtype=float)
    residents = allocated @ np.array([pph[b] for b in INCOME_BANDS])
    rate = float(params.value(L5.P_SERVICE_JOBS_PER_RESIDENT))

    np.testing.assert_allclose(
        diag["new_residents"].to_numpy(dtype=float), residents, rtol=STRICT, atol=STRICT
    )
    np.testing.assert_allclose(
        diag["new_service_jobs"].to_numpy(dtype=float), rate * residents, rtol=STRICT, atol=STRICT
    )


def test_dormitory_workers_generate_service_jobs_at_a_reduced_rate(cells, params):
    h3 = str(cells["h3"].iloc[0])
    effect = _EmploymentEffect(
        cell=h3,
        sector=int(Sector.manuf_light),
        jobs=5000.0,
        median_wage_inr_mo=18000.0,
        start_year=BASE_YEAR,
        ramp_years=1,
        dormitory_share=1.0,  # every worker is a dormitory worker
    )
    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        employment_effects=[effect],
        natural_growth_rate=0,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]

    # Section 9.5 / Section 21: no phantom apartment buyers ...
    assert diag["new_households"] == 0
    # ... but the dormitory workers still buy tea.
    rate = float(params.value(L5.P_SERVICE_JOBS_PER_RESIDENT))
    factor = float(params.value(L5.P_DORM_SERVICE_FACTOR))
    row = cells.index[cells["h3"] == h3][0]
    assert diag["new_service_jobs"].loc[row] == pytest.approx(effect.jobs * rate * factor)
    assert diag["new_service_jobs"].loc[row] < effect.jobs * rate


def test_no_allocation_means_no_service_jobs(cells, params):
    out = L5.allocate(cells, params, year=NEXT_YEAR, natural_growth_rate=0)
    np.testing.assert_array_equal(_jobs(out), _jobs(cells))


@pytest.mark.acceptance
def test_acc_employment_cannot_grow_faster_than_population(cells, params):
    """Section 21: "Double-counted service jobs -> employment grows faster than population"."""
    out = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )

    pop_before = cells["population"].sum()
    pop_after = out["population"].sum()
    jobs_before = _jobs(cells).sum()
    jobs_after = _jobs(out).sum()

    assert pop_after > pop_before
    assert jobs_after > jobs_before
    assert (jobs_after / jobs_before) < (pop_after / pop_before)


def test_missing_persons_per_household_raises(cells, params):
    """``behaviour.persons_per_household_by_band`` is null on disk; do not substitute."""
    with pytest.raises(MissingParameter) as excinfo:
        L5.allocate(cells, params, year=NEXT_YEAR)
    assert "persons_per_household_by_band" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 12.6 — agglomeration damping
# --------------------------------------------------------------------------------------


def test_phi_must_be_below_one(cells, params, matrices):
    at_one = _ParamsOverride(params, **{L5.P_SPILLOVER_PHI: 1})
    with pytest.raises(ConvergenceError):
        L5.allocate(
            cells,
            at_one,
            year=NEXT_YEAR,
            matrices=matrices,
            persons_per_household_by_band=_pph(params),
        )


def test_congestion_feedback_lowers_accessibility(cells, params, matrices):
    """Rising ``builtup_frac`` must slow the network inside the inner loop (Section 12.6)."""
    base = L5.band_accessibility(cells, params, matrices)

    denser = cells.copy()
    denser["builtup_frac"] = np.ones(len(denser))
    congested = L5.band_accessibility(
        denser, params, matrices, congestion_builtup=denser["builtup_frac"], base_builtup=cells["builtup_frac"]
    )

    finite = np.isfinite(base["mid"]) & np.isfinite(congested["mid"])
    assert (congested["mid"][finite] < base["mid"][finite]).all()


def test_iteration_cap_without_convergence_raises(cells, params, matrices):
    """Hitting ``max_iterations`` raises rather than silently returning (Section 12.6)."""
    impossible = _ParamsOverride(params, **{L5.P_CONVERGENCE_TOL: 0})
    with pytest.raises(ConvergenceError):
        L5.allocate(
            cells,
            impossible,
            year=NEXT_YEAR,
            matrices=matrices,
            persons_per_household_by_band=_pph(params),
        )


def test_growing_residual_raises_before_the_cap(cells, params, matrices, monkeypatch):
    """A residual that grows instead of shrinking is divergence — raise, do not grind on.

    White-box: the accessibility surface is replaced by one that runs away, which is what a
    mis-specified spillover looks like from inside the loop.
    """
    real = L5.band_accessibility
    calls = {"n": 0}

    def runaway(frame, prm, mtx, **kw):
        calls["n"] += 1
        base = real(frame, prm, mtx)
        return {band: values + calls["n"] * calls["n"] for band, values in base.items()}

    monkeypatch.setattr(L5, "band_accessibility", runaway)
    with pytest.raises(ConvergenceError) as excinfo:
        L5.allocate(
            cells,
            params,
            year=NEXT_YEAR,
            matrices=matrices,
            persons_per_household_by_band=_pph(params),
        )
    assert "diverging" in str(excinfo.value)
    assert calls["n"] < int(params.value(L5.P_MAX_ITERATIONS))


def test_service_jobs_feed_back_into_accessibility(cells, params, matrices):
    """Step 5d: new service jobs update the opportunity surface within the year."""
    out = L5.allocate(
        cells,
        params,
        year=NEXT_YEAR,
        matrices=matrices,
        persons_per_household_by_band=_pph(params),
    )
    diag = out.attrs[L5.ATTR_DIAGNOSTICS]
    assert diag["band_accessibility"] == "band_mode_weights"
    # the surface actually moved within the year
    assert diag["max_delta_lnA"] > 0
    final = diag["lnA_by_band"]["mid"]
    base = L5.band_accessibility(cells, params, matrices)["mid"]
    finite = np.isfinite(base) & np.isfinite(final)
    assert not np.array_equal(final[finite], base[finite])


# --------------------------------------------------------------------------------------
# 12.7 — firm allocation
# --------------------------------------------------------------------------------------


def test_firm_allocation_raises_on_unfitted_coefficients(cells, params):
    with pytest.raises(MissingParameter) as excinfo:
        L5.allocate_firms(cells, params, jobs_by_sector={int(Sector.it_office): 1000.0})
    assert "firm_logit" in str(excinfo.value)


def test_firm_allocation_places_jobs_when_coefficients_exist(cells, params):
    supplied = _ParamsOverride(
        params,
        **{
            f"{L5.P_FIRM_LOGIT}.enabled": True,
            f"{L5.P_FIRM_COEFFICIENTS}.c_market": 0.8,
            f"{L5.P_FIRM_COEFFICIENTS}.c_labour": 0.5,
            f"{L5.P_FIRM_COEFFICIENTS}.c_land": -0.9,
            f"{L5.P_FIRM_COEFFICIENTS}.c_agglom": 0.3,
            f"{L5.P_FIRM_COEFFICIENTS}.c_freight": 0.2,
        },
    )
    jobs = {int(Sector.it_office): 1000.0}
    out = L5.allocate_firms(cells, supplied, jobs_by_sector=jobs)

    delta = _jobs(out) - _jobs(cells)
    assert delta[:, int(Sector.it_office)].sum() == pytest.approx(1000.0)
    assert np.abs(np.delete(delta, int(Sector.it_office), axis=1)).max() == 0


def test_firm_allocation_zoning_gate_excludes_prohibited_cells(cells, params):
    supplied = _ParamsOverride(
        params,
        **{
            f"{L5.P_FIRM_LOGIT}.enabled": True,
            f"{L5.P_FIRM_COEFFICIENTS}.c_market": 0.8,
            f"{L5.P_FIRM_COEFFICIENTS}.c_labour": 0.5,
            f"{L5.P_FIRM_COEFFICIENTS}.c_land": -0.9,
            f"{L5.P_FIRM_COEFFICIENTS}.c_agglom": 0.3,
            f"{L5.P_FIRM_COEFFICIENTS}.c_freight": 0.2,
        },
    )
    sector = int(Sector.manuf_light)
    out = L5.allocate_firms(
        cells,
        supplied,
        jobs_by_sector={sector: 500.0},
        zoning_gate={sector: ("ind",)},
    )
    delta = _jobs(out) - _jobs(cells)
    prohibited = (cells["zone_class"] != "ind").to_numpy()

    assert delta[prohibited, sector].max() == 0
    assert delta[:, sector].sum() == pytest.approx(500.0)


# --------------------------------------------------------------------------------------
# state threading and misc guards
# --------------------------------------------------------------------------------------


def test_alpha_is_estimated_once_and_threaded(cells, params):
    first = L5.allocate(
        cells, params, year=NEXT_YEAR, persons_per_household_by_band=_pph(params)
    )
    state = first.attrs[L5.ATTR_STATE]
    assert state.base_year == NEXT_YEAR

    second = L5.allocate(
        first, params, year=NEXT_YEAR + 1, state=state,
        persons_per_household_by_band=_pph(params),
    )
    # alpha is held fixed through the simulation (Section 12.3)
    np.testing.assert_array_equal(
        second.attrs[L5.ATTR_STATE].alpha.by_band.to_numpy(), state.alpha.by_band.to_numpy()
    )
    np.testing.assert_allclose(
        second["alpha_res"].to_numpy(dtype=float),
        first["alpha_res"].to_numpy(dtype=float),
        rtol=STRICT,
    )


def test_alpha_res_column_is_centred(cells, params):
    fit = L5.estimate_alpha_res(cells, params)
    assert fit.per_cell.mean() == pytest.approx(0, abs=STRICT)
    for band in INCOME_BANDS:
        assert fit.by_band[band].mean() == pytest.approx(0, abs=STRICT)


def test_unknown_keyword_raises(cells, params):
    with pytest.raises(TypeError):
        L5.allocate(cells, params, year=NEXT_YEAR, nonsense=True)


def test_missing_column_raises(cells, params):
    broken = cells.drop(columns=["households"])
    with pytest.raises(UFEError):
        L5.allocate(broken, params, year=NEXT_YEAR)


def test_layer_makes_no_llm_or_network_import():
    import ufe.layers.l5_allocation as module

    source = module.__file__
    text = open(source, encoding="utf-8").read()
    assert "ufe.ai" not in text
    assert "requests" not in text
    assert "httpx" not in text
