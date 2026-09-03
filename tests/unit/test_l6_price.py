"""Tests for Layer 6, price formation (spec Section 13).

The Section 13 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "identical demand shock applied to two cells with ``eps = 0.1`` and ``eps = 3.0``
  produces a price rise roughly 5-8x larger in the constrained cell and a quantity rise
  roughly 20x larger in the elastic one" -> ``test_acc_elasticity_test``
* "a cell with zero headroom converts the entire demand shock to price; ``d ln Q = 0``
  exactly" -> ``test_acc_capacity_zero_headroom_all_price``
* "running with no projects and ``phi_t = 0.055`` produces exactly 5.5 log points of
  appreciation in every cell and zero excess-over-trend anywhere"
  -> ``test_acc_macro_isolation``
* "Overshoot decays to under 5% of peak after 5 half-lives"
  -> ``test_acc_overshoot_decays_below_five_percent_after_five_half_lives``
* "Decomposition: ``Sigma lambda + interaction == total`` to 1e-9"
  -> ``test_acc_decomposition_reconciles``
* "Removing a factor and re-running FULL reproduces that factor's LOO run exactly"
  -> ``test_acc_loo_reproduces_full_without_factor``

Beyond the acceptance block these tests cover the brief's explicit requirements: the
overheating detector catching a deliberately explosive corridor (Section 13.5 x Section 21
"Agglomeration divergence"), market clearing either converging or raising
:class:`ufe.errors.ConvergenceError`, uncertainty bands widening in both directions, the
INR/sqft <-> sqm unit conversions, purity and determinism.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from ufe.errors import ConvergenceError, MissingParameter
from ufe.layers import l6_price as L6
from ufe.params import Params, load_params

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    synthetic_city,
)

CITY = "vizag"
BASE_YEAR = 2024
MACRO_PHI = 0.055  # the Section 13 ACCEPTANCE macro-isolation value


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture()
def cells(synthetic_city):  # noqa: F811
    return synthetic_city.cells


def _widened(params: Params, *, low: float, high: float) -> Params:
    """A copy of `params` with a wider range on the macro trend (input uncertainty)."""
    tree = copy.deepcopy(params.resolved)
    leaf = tree["price"]["macro"]["scenarios"]["base"]
    leaf["low"], leaf["high"] = low, high
    leaf["value"] = min(max(leaf["value"], low), high)
    return Params(
        city_id=params.city_id,
        city_class=params.city_class,
        resolved=tree,
        deviations=[],
        class_defaults_applied=[],
        source_files=[],
        city_config=params.city_config,
    )


# --------------------------------------------------------------------------------------
# 0.3 units — "Price | INR per square foot | the market convention; convert internally
# only for area math". A unit error here is invisible and catastrophic.
# --------------------------------------------------------------------------------------


def test_units_sqm_per_sqft_is_the_definitional_value():
    # International foot = 0.3048 m exactly (EPSG unit 9002), so 1 sqft = 0.09290304 sqm.
    assert L6.sqm_per_sqft() == pytest.approx(0.09290304, rel=1e-12)
    assert L6.metres_per_foot() == pytest.approx(0.3048, rel=1e-12)


def test_units_sqft_from_sqm_known_answer():
    assert L6.sqft_from_sqm(1.0) == pytest.approx(10.7639104167, rel=1e-9)
    assert L6.sqm_from_sqft(1.0) == pytest.approx(0.09290304, rel=1e-12)


def test_units_round_trip_is_exact_to_floating_point():
    values = np.array([1.0, 137.5, 1e6])
    np.testing.assert_allclose(
        L6.sqm_from_sqft(L6.sqft_from_sqm(values)), values, rtol=1e-12
    )


def test_units_value_inr_is_the_only_price_times_area_path():
    # 1000 INR/sqft over 100 sqm = 1000 * (100 / 0.09290304) INR.
    assert L6.value_inr(1000.0, 100.0) == pytest.approx(1000.0 * 100.0 / 0.09290304)
    # Converting the price instead of the area must give the same money.
    assert L6.value_inr(1000.0, 100.0) == pytest.approx(
        L6.price_inr_per_sqm(1000.0) * 100.0
    )


def test_units_price_conversions_are_inverse():
    assert L6.price_inr_per_sqft(L6.price_inr_per_sqm(7500.0)) == pytest.approx(7500.0)


def test_units_months_per_year_comes_from_the_calendar():
    assert L6.MONTHS_PER_YEAR == 12


def test_gross_yield_uses_monthly_rent_and_is_dimensionless():
    # Both rent and price are INR/sqft, so the yield needs no area conversion at all.
    rent = np.array([10.0])
    price = np.array([5000.0])
    np.testing.assert_allclose(
        L6.gross_yield(rent, price), rent * L6.MONTHS_PER_YEAR / price
    )


# --------------------------------------------------------------------------------------
# 13.1 / 13.2 — ACCEPTANCE
# --------------------------------------------------------------------------------------


ELASTIC_EPS = 3.0
CONSTRAINED_EPS = 0.1
SHOCK = 0.20
BIG_STOCK = 1.0e5


def _two_cell_clearing(params, *, headroom, shock=SHOCK):
    eta = params.value(L6.P_ETA)
    return (
        eta,
        L6.clear_market(
            params,
            d_ln_D_local=np.array([shock, shock]),
            d_ln_S0=np.zeros(2),
            eta=eta,
            eps=np.array([CONSTRAINED_EPS, ELASTIC_EPS]),
            quantity_sqm=np.array([BIG_STOCK, BIG_STOCK]),
            headroom_sqm=np.asarray(headroom, dtype=float),
            absorption_cap_sqm=np.full(2, np.inf),
            phi_t=0.0,
        ),
    )


@pytest.mark.acceptance
def test_acc_elasticity_test(params):
    """Section 21's "Elasticity ignored | Every zone reads 'buy'" guard.

    Spec: "identical demand shock applied to two cells with `eps = 0.1` and `eps = 3.0`
    produces a price rise roughly 5-8x larger in the constrained cell and a quantity rise
    roughly 20x larger in the elastic one."

    The 5-8x band is met on the *level* price rise with the shipped
    `price.hedonic.eta_demand_price = 0.65`.  The "roughly 20x" quantity figure is not
    reproducible from the Section 13.1 equations at any eta that also satisfies the price
    band (see the build report); the assertion below is that the elastic cell's quantity
    rise is larger by a wide margin, and the exact ratio is pinned to the closed form.
    """
    eta, result = _two_cell_clearing(params, headroom=[np.inf, np.inf])

    constrained, elastic = 0, 1
    assert not result.constrained.any()

    # closed form, recomputed from YAML rather than hand-written
    np.testing.assert_allclose(
        result.d_ln_P,
        SHOCK / (eta + np.array([CONSTRAINED_EPS, ELASTIC_EPS])),
        rtol=1e-12,
    )

    price_rise = np.expm1(result.d_ln_P)
    quantity_rise = np.expm1(result.d_ln_Q)

    price_ratio = price_rise[constrained] / price_rise[elastic]
    assert 5.0 <= price_ratio <= 8.0

    quantity_ratio = quantity_rise[elastic] / quantity_rise[constrained]
    assert quantity_ratio > 5.0
    assert quantity_ratio == pytest.approx(
        np.expm1(ELASTIC_EPS * SHOCK / (eta + ELASTIC_EPS))
        / np.expm1(CONSTRAINED_EPS * SHOCK / (eta + CONSTRAINED_EPS)),
        rel=1e-12,
    )


@pytest.mark.acceptance
def test_acc_capacity_zero_headroom_all_price(params):
    """"a cell with zero headroom converts the entire demand shock to price;
    `d ln Q = 0` exactly.\""""
    eta, result = _two_cell_clearing(params, headroom=[0.0, 0.0])
    assert result.constrained.all()
    np.testing.assert_array_equal(result.d_ln_Q, np.zeros(2))
    np.testing.assert_allclose(result.d_ln_P, np.full(2, SHOCK / eta), rtol=1e-12)
    # And the constrained price rise strictly exceeds the unconstrained one.
    _, free = _two_cell_clearing(params, headroom=[np.inf, np.inf])
    assert (result.d_ln_P > free.d_ln_P).all()


def test_absorption_cap_binds_before_headroom(params):
    eta = params.value(L6.P_ETA)
    cap = 1.0e3
    result = L6.clear_market(
        params,
        d_ln_D_local=np.array([SHOCK]),
        d_ln_S0=np.zeros(1),
        eta=eta,
        eps=np.array([ELASTIC_EPS]),
        quantity_sqm=np.array([BIG_STOCK]),
        headroom_sqm=np.array([np.inf]),
        absorption_cap_sqm=np.array([cap]),
        phi_t=0.0,
    )
    assert result.constrained.all()
    assert result.d_ln_Q[0] == pytest.approx(np.log((BIG_STOCK + cap) / BIG_STOCK))
    assert result.d_ln_P[0] == pytest.approx((SHOCK - result.d_ln_Q[0]) / eta)


@pytest.mark.acceptance
def test_acc_macro_isolation(params, cells):
    """Section 21's "Macro trend attributed to projects" guard.

    "running with no projects and `phi_t = 0.055` produces exactly 5.5 log points of
    appreciation in every cell and zero excess-over-trend anywhere."
    """
    out = L6.form_prices(cells, params, year=BASE_YEAR, phi_t=MACRO_PHI)
    np.testing.assert_array_equal(
        out["d_ln_P_fundamental"].to_numpy(), np.full(len(cells), MACRO_PHI)
    )
    np.testing.assert_array_equal(
        out["excess_over_trend"].to_numpy(), np.zeros(len(cells))
    )
    np.testing.assert_array_equal(out["d_ln_Q"].to_numpy(), np.zeros(len(cells)))
    np.testing.assert_array_equal(out["overshoot_log"].to_numpy(), np.zeros(len(cells)))


def test_macro_scenario_read_from_yaml(params, cells):
    out = L6.form_prices(cells, params, year=BASE_YEAR, scenario="bull")
    expected = params.value("price.macro.scenarios.bull")
    np.testing.assert_allclose(out["phi_t"].to_numpy(), expected)
    with pytest.raises(MissingParameter):
        L6.form_prices(cells, params, year=BASE_YEAR, scenario="melt_up")


# --------------------------------------------------------------------------------------
# 13.1 — demand shift assembly
# --------------------------------------------------------------------------------------


def test_demand_shift_assembles_the_four_terms(params, cells):
    gamma = params.value(L6.P_GAMMA_BUILT)
    n = len(cells)
    d_lnA = pd.Series(np.linspace(0.0, 0.1, n), index=cells.index)
    new_hh = pd.Series(np.full(n, 10.0), index=cells.index)
    field = pd.Series(np.full(n, 0.05), index=cells.index)

    got = L6.demand_shift(cells, params, d_lnA=d_lnA, new_hh=new_hh, field=field)
    households = cells["households"].to_numpy(dtype=float)
    expected = (
        gamma * d_lnA.to_numpy()
        + np.log((households + 10.0) / households)
        + 0.05
    )
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-12)


def test_field_is_capped_at_the_yaml_caps(params, cells):
    low = params.value(L6.P_FIELD_CAP_LOW)
    high = params.value(L6.P_FIELD_CAP_HIGH)
    field = pd.Series(
        np.linspace(low - 1.0, high + 1.0, len(cells)), index=cells.index
    )
    capped = L6.field_effect(field, params)
    assert capped.min() == pytest.approx(low)
    assert capped.max() == pytest.approx(high)


def test_land_pass_uses_gamma_land_multiple(params, cells):
    gamma = params.value(L6.P_GAMMA_BUILT)
    multiple = params.value(L6.P_GAMMA_LAND_MULTIPLE)
    d_lnA = pd.Series(np.full(len(cells), 0.1), index=cells.index)
    built = L6.demand_shift(cells, params, d_lnA=d_lnA)
    land = L6.demand_shift(cells, params, d_lnA=d_lnA, land=True)
    np.testing.assert_allclose(
        (land - built).to_numpy(), gamma * (multiple - 1.0) * 0.1, atol=1e-12
    )


def test_include_land_without_an_elasticity_reports_the_missing_path(params, cells):
    with pytest.raises(MissingParameter) as excinfo:
        L6.form_prices(cells, params, year=BASE_YEAR, include_land=True)
    assert L6.P_EPS_LAND_MULTIPLE in str(excinfo.value)


def test_include_land_with_an_explicit_elasticity(params, cells):
    out = L6.form_prices(
        cells, params, year=BASE_YEAR, include_land=True, eps_land_multiple=2.0
    )
    assert "price_land_inr_sqft_fundamental" in out.columns
    assert "d_ln_P_land" in out.columns


# --------------------------------------------------------------------------------------
# 13.1 / 13.2 — convergence guarantee (brief requirement 3)
# --------------------------------------------------------------------------------------


def _clear_one(params, **kw):
    base = dict(
        d_ln_D_local=np.array([0.1]),
        d_ln_S0=np.zeros(1),
        eta=params.value(L6.P_ETA),
        eps=np.array([1.0]),
        quantity_sqm=np.array([BIG_STOCK]),
        headroom_sqm=np.array([np.inf]),
        absorption_cap_sqm=np.array([np.inf]),
        phi_t=0.0,
    )
    base.update(kw)
    return L6.clear_market(params, **base)


def test_clearing_converges_and_records_iterations(params):
    result = _clear_one(params)
    assert result.converged
    assert result.iterations >= 1


def test_zero_combined_elasticity_raises_convergence_error(params):
    with pytest.raises(ConvergenceError):
        _clear_one(params, eta=0.0, eps=np.array([0.0]))


def test_non_finite_demand_shift_raises_convergence_error(params):
    with pytest.raises(ConvergenceError):
        _clear_one(params, d_ln_D_local=np.array([np.nan]))
    with pytest.raises(ConvergenceError):
        _clear_one(params, d_ln_D_local=np.array([np.inf]))


def test_explosive_demand_shock_raises_rather_than_returning_an_absurd_number(params):
    """Section 21: "Agglomeration divergence | Prices explode in one corridor |
    ... `ConvergenceError`". Never a silently absurd number."""
    with pytest.raises(ConvergenceError):
        _clear_one(params, d_ln_D_local=np.array([1.0e5]), headroom_sqm=np.array([0.0]))


def test_explosive_price_level_raises_in_form_prices(params, cells):
    field = pd.Series(np.zeros(len(cells)), index=cells.index)
    field.iloc[0] = np.inf
    with pytest.raises(ConvergenceError):
        L6.form_prices(cells, params, year=BASE_YEAR, field=field, cap_field=False)


def test_zero_quantity_cell_is_handled_not_divided_by(params):
    result = _clear_one(
        params, quantity_sqm=np.array([0.0]), headroom_sqm=np.array([0.0])
    )
    assert result.converged
    assert np.isfinite(result.d_ln_P).all()
    assert result.d_ln_Q[0] == 0.0


# --------------------------------------------------------------------------------------
# 13.3 — overshoot
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_overshoot_decays_below_five_percent_after_five_half_lives(params):
    half_life = params.value(L6.P_OVERSHOOT_HALF_LIFE)
    peak_factor = params.value(L6.P_OVERSHOOT_PEAK)
    trigger = params.value(L6.P_OVERSHOOT_TRIGGER)
    shock = np.array([trigger * 2.0])

    peak = L6.overshoot(shock, params, year=BASE_YEAR, announce_year=BASE_YEAR)
    assert peak[0] == pytest.approx(peak_factor * shock[0])

    five_half_lives = BASE_YEAR + 5 * half_life
    later = L6.overshoot(shock, params, year=five_half_lives, announce_year=BASE_YEAR)
    assert later[0] < 0.05 * peak[0]


def test_overshoot_below_trigger_is_zero(params):
    trigger = params.value(L6.P_OVERSHOOT_TRIGGER)
    shock = np.array([trigger, trigger * 0.5, trigger * 1.01])
    got = L6.overshoot(shock, params, year=BASE_YEAR, announce_year=BASE_YEAR)
    assert got[0] == 0.0  # strict: "if shock_i > trigger_min_shock"
    assert got[1] == 0.0
    assert got[2] > 0.0


def test_overshoot_is_zero_before_announcement(params):
    trigger = params.value(L6.P_OVERSHOOT_TRIGGER)
    shock = np.array([trigger * 3.0])
    got = L6.overshoot(shock, params, year=BASE_YEAR - 1, announce_year=BASE_YEAR)
    assert got[0] == 0.0


def test_overshoot_reported_and_fundamental_are_separable(params, cells):
    trigger = params.value(L6.P_OVERSHOOT_TRIGGER)
    shock = pd.Series(np.full(len(cells), trigger * 4.0), index=cells.index)
    out = L6.form_prices(
        cells,
        params,
        year=BASE_YEAR,
        phi_t=0.0,
        announcement_shock=shock,
        announce_year=BASE_YEAR,
    )
    np.testing.assert_allclose(
        out["d_ln_P_reported"].to_numpy(),
        out["d_ln_P_fundamental"].to_numpy() + out["overshoot_log"].to_numpy(),
        rtol=1e-12,
    )
    assert (out["overshoot_log"] > 0).all()


# --------------------------------------------------------------------------------------
# 13.4 — factor decomposition
# --------------------------------------------------------------------------------------


FACTORS = ("metro", "airport", "data_centres")
N_DECOMP_CELLS = 40


def _decomposition_run(index):
    """A toy run function: each factor adds its own log-price contribution, plus a
    deliberate interaction so `interaction_i` is not trivially zero."""
    rng = np.random.default_rng(4242)
    base = pd.Series(rng.normal(size=len(index)), index=index)
    effects = {
        name: pd.Series(rng.uniform(0.01, 0.2, size=len(index)), index=index)
        for name in FACTORS
    }

    def run(active):
        total = base.copy()
        for name in active:
            total = total + effects[name]
        if {"metro", "airport"} <= set(active):
            total = total + effects["metro"] * effects["airport"]
        return total

    return run, base, effects


@pytest.mark.acceptance
def test_acc_decomposition_reconciles(params):
    """"Decomposition: `Sigma lambda + interaction == total` to 1e-9.\"

    The tolerance is read from `price.decomposition.normalise_epsilon` (1e-9), which is
    the only 1e-9 in the price YAML.  The identity holds for the *raw* leave-one-out
    lambdas: the normalised lambdas sum to `total` on their own by construction, so
    adding `interaction` to those would double-count it.
    """
    index = pd.RangeIndex(N_DECOMP_CELLS)
    run, _, _ = _decomposition_run(index)
    result = L6.decompose(run, FACTORS, params)

    tolerance = params.value(L6.P_NORMALISE_EPSILON)
    np.testing.assert_allclose(
        (result.raw.sum(axis=1) + result.interaction).to_numpy(),
        result.total.to_numpy(),
        atol=tolerance,
        rtol=0.0,
    )
    assert result.reconciliation_error().abs().max() <= tolerance
    # The normalised lambdas reconcile on their own.
    np.testing.assert_allclose(
        result.normalised.sum(axis=1).to_numpy(), result.total.to_numpy(), atol=tolerance
    )
    # "report `interaction_i` explicitly in the output"
    assert "interaction" in result.to_frame().columns
    assert result.interaction.abs().max() > 0.0


@pytest.mark.acceptance
def test_acc_loo_reproduces_full_without_factor(params):
    """"Removing a factor and re-running FULL reproduces that factor's LOO run exactly.\""""
    index = pd.RangeIndex(N_DECOMP_CELLS)
    run, _, _ = _decomposition_run(index)
    result = L6.decompose(run, FACTORS, params)
    for name in FACTORS:
        reduced = L6.decompose(run, [f for f in FACTORS if f != name], params)
        np.testing.assert_array_equal(
            reduced.ln_p_full.to_numpy(), result.loo[name].to_numpy()
        )


def test_decomposition_flags_non_separable_factors(params):
    index = pd.RangeIndex(N_DECOMP_CELLS)

    def run(active):
        # Pure interaction: no factor does anything on its own.
        value = np.ones(len(index))
        if set(active) == set(FACTORS):
            value = value * 2.0
        return pd.Series(value, index=index)

    result = L6.decompose(run, FACTORS, params)
    assert not result.separable
    assert result.warning is not None
    assert "interaction" in result.warning


def test_decomposition_respects_the_max_factor_cap(params):
    index = pd.RangeIndex(2)
    cap = int(params.value(L6.P_MAX_FACTORS))
    names = [f"f{i}" for i in range(cap + 1)]
    with pytest.raises(ValueError) as excinfo:
        L6.decompose(lambda active: pd.Series(np.zeros(2), index=index), names, params)
    assert str(cap) in str(excinfo.value)


def test_decomposition_runs_are_the_spec_ordered_set(params):
    index = pd.RangeIndex(3)
    seen = []

    def run(active):
        seen.append(frozenset(active))
        return pd.Series(np.full(3, float(len(active))), index=index)

    L6.decompose(run, FACTORS, params)
    assert seen[0] == frozenset()  # Run 0: baseline, macro only
    assert seen[1] == frozenset(FACTORS)  # Run FULL
    assert set(seen[2:]) == {frozenset(FACTORS) - {f} for f in FACTORS}  # Run LOO_f


# --------------------------------------------------------------------------------------
# 13.5 — the residual and overheating detector
# --------------------------------------------------------------------------------------


N_OVERHEAT = 50


def _overheat_inputs(explosive_corridor: bool):
    index = pd.RangeIndex(N_OVERHEAT)
    rng = np.random.default_rng(7)
    residual = pd.Series(rng.normal(scale=0.01, size=N_OVERHEAT), index=index)
    price = pd.Series(np.full(N_OVERHEAT, 5000.0), index=index)
    rent = pd.Series(np.full(N_OVERHEAT, 5000.0 * 0.027 / 12.0), index=index)
    price_cagr = pd.Series(np.full(N_OVERHEAT, 0.03), index=index)
    builtup_cagr = pd.Series(np.full(N_OVERHEAT, 0.03), index=index)
    nightlight_cagr = pd.Series(np.full(N_OVERHEAT, 0.02), index=index)

    corridor = slice(0, 3)
    if explosive_corridor:
        residual.iloc[corridor] = 5.0  # model badly under-predicts: prices ran away
        rent.iloc[corridor] = 5000.0 * 0.010 / 12.0  # yield collapsed to 1.0%
        price_cagr.iloc[corridor] = 0.35  # 35%/yr with no physical change
    return index, residual, price, rent, price_cagr, builtup_cagr, nightlight_cagr


def _overheat(params, explosive: bool):
    index, residual, price, rent, pc, bc, nc = _overheat_inputs(explosive)
    return L6.overheating(
        params,
        residual=residual,
        price_inr_sqft=price,
        rent_inr_sqft_mo=rent,
        price_cagr_5y=pc,
        builtup_cagr_5y=bc,
        nightlight_cagr_5y=nc,
    )


@pytest.mark.acceptance
def test_overheating_detector_catches_an_explosive_corridor(params):
    """Section 13.5 x Section 21 "Agglomeration divergence -- prices explode in one
    corridor". Prices must not be able to run away undetected."""
    out = _overheat(params, explosive=True)
    corridor = out.iloc[:3]
    calm = out.iloc[3:]

    assert corridor["flag_residual"].all()
    assert corridor["flag_yield"].all()
    assert corridor["flag_physical"].all()
    assert (corridor["overheat_score"] == 3).all()
    assert (corridor["overheat_score_max"] == 3).all()
    assert not calm["flag_physical"].any()
    assert not calm["flag_yield"].any()
    assert calm["overheat_score"].max() < 3


def test_overheating_flags_are_reported_separately_never_one_opaque_number(params):
    out = _overheat(params, explosive=True)
    for column in (
        "residual",
        "gross_yield",
        "flag_residual",
        "flag_yield",
        "flag_physical",
        "overheat_score",
        "overheat_score_max",
    ):
        assert column in out.columns


def test_overheating_threshold_is_the_yaml_percentile(params):
    out = _overheat(params, explosive=False)
    percentile = params.value(L6.P_OVERHEAT_PERCENTILE)
    threshold = np.nanpercentile(out["residual"].to_numpy(), percentile)
    np.testing.assert_array_equal(
        out["flag_residual"].to_numpy(), out["residual"].to_numpy() > threshold
    )


def test_overheating_missing_rent_scores_out_of_two(params):
    index, residual, price, rent, pc, bc, nc = _overheat_inputs(True)
    rent.iloc[0] = np.nan
    out = L6.overheating(
        params,
        residual=residual,
        price_inr_sqft=price,
        rent_inr_sqft_mo=rent,
        price_cagr_5y=pc,
        builtup_cagr_5y=bc,
        nightlight_cagr_5y=nc,
    )
    # "Cells missing rent data get `flag_yield = null`, and the score is reported out of 2"
    assert pd.isna(out["flag_yield"].iloc[0])
    assert out["overheat_score_max"].iloc[0] == 2
    assert out["overheat_score"].iloc[0] == 2


def test_model_residual_matches_section_13_5(params):
    index = pd.RangeIndex(5)
    base = pd.Series(np.linspace(8.0, 9.0, 5), index=index)
    lambdas = pd.DataFrame(
        {"metro": np.full(5, 0.1), "airport": np.full(5, 0.05)}, index=index
    )
    observed = base + 0.2
    residual = L6.model_residual(observed, base, lambdas)
    np.testing.assert_allclose(residual.to_numpy(), np.full(5, 0.05), atol=1e-12)


# --------------------------------------------------------------------------------------
# 13.6 — uncertainty bands
# --------------------------------------------------------------------------------------


MC_DRAWS = 60
MC_YEARS = (2025, 2027, 2030, 2035)
BAND_LOW, BAND_HIGH = 10.0, 90.0


def _bands(params, cells, *, seed=20240101, years=MC_YEARS, draws=MC_DRAWS):
    paths = L6.monte_carlo_price_paths(
        cells,
        params,
        years=years,
        n_draws=draws,
        rng=np.random.default_rng(seed),
    )
    return L6.uncertainty_bands(
        paths,
        index=cells.index,
        years=years,
        percentile_low=BAND_LOW,
        percentile_high=BAND_HIGH,
    )


@pytest.mark.acceptance
def test_bands_widen_with_horizon(params, cells):
    bands = _bands(params, cells)
    width_by_year = bands.groupby("year")["width"].mean()
    assert list(width_by_year.index) == list(MC_YEARS)
    assert width_by_year.is_monotonic_increasing
    assert width_by_year.iloc[-1] > width_by_year.iloc[0]


@pytest.mark.acceptance
def test_bands_widen_with_input_uncertainty(params, cells):
    narrow = _widened(params, low=0.050, high=0.060)
    wide = _widened(params, low=0.010, high=0.100)
    narrow_bands = _bands(narrow, cells)
    wide_bands = _bands(wide, cells)
    assert wide_bands["width"].mean() > narrow_bands["width"].mean()
    for year in MC_YEARS:
        assert (
            wide_bands.loc[wide_bands["year"] == year, "width"].mean()
            > narrow_bands.loc[narrow_bands["year"] == year, "width"].mean()
        )


def test_bands_bracket_the_median(params, cells):
    bands = _bands(params, cells, draws=20, years=(2025, 2026))
    assert (bands["p_low"] <= bands["median"]).all()
    assert (bands["median"] <= bands["p_high"]).all()
    assert (bands["width"] >= 0).all()


def test_band_percentiles_missing_from_yaml_are_reported(params, cells):
    paths = L6.monte_carlo_price_paths(
        cells, params, years=(2025,), n_draws=2, rng=np.random.default_rng(1)
    )
    with pytest.raises(MissingParameter) as excinfo:
        L6.uncertainty_bands(paths, index=cells.index, years=(2025,), params=params)
    assert L6.P_BAND_PERCENTILE_LOW in str(excinfo.value)
    assert L6.P_BAND_PERCENTILE_HIGH in str(excinfo.value)


def test_monte_carlo_is_deterministic_for_a_given_seed(params, cells):
    kw = dict(years=(2025, 2026), n_draws=8)
    a = L6.monte_carlo_price_paths(
        cells, params, rng=np.random.default_rng(11), **kw
    )
    b = L6.monte_carlo_price_paths(
        cells, params, rng=np.random.default_rng(11), **kw
    )
    np.testing.assert_array_equal(a, b)
    c = L6.monte_carlo_price_paths(
        cells, params, rng=np.random.default_rng(12), **kw
    )
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------------------
# form_prices — purity, shape, determinism
# --------------------------------------------------------------------------------------


def test_form_prices_is_pure_and_preserves_index_and_row_count(params, cells):
    before = cells.copy(deep=True)
    out = L6.form_prices(cells, params, year=BASE_YEAR)
    assert len(out) == len(cells)
    pd.testing.assert_index_equal(out.index, cells.index)
    pd.testing.assert_frame_equal(cells, before)
    assert out is not cells
    # every input column survives untouched
    for column in cells.columns:
        pd.testing.assert_series_equal(out[column], cells[column])


def test_form_prices_is_deterministic(params, cells):
    a = L6.form_prices(cells, params, year=BASE_YEAR)
    b = L6.form_prices(cells, params, year=BASE_YEAR)
    pd.testing.assert_frame_equal(a, b)


def test_form_prices_records_diagnostics_for_provenance(params, cells):
    out = L6.form_prices(cells, params, year=BASE_YEAR)
    diagnostics = out.attrs[L6.ATTR_KEY]
    for key in ("year", "scenario", "phi_t", "eta", "gamma", "iterations", "converged"):
        assert key in diagnostics
    assert diagnostics["year"] == BASE_YEAR
    assert diagnostics["converged"] is True


def test_form_prices_price_level_follows_the_log_change(params, cells):
    out = L6.form_prices(cells, params, year=BASE_YEAR, phi_t=MACRO_PHI)
    observed = cells["price_res_inr_sqft"].to_numpy(dtype=float)
    expected = observed * np.exp(out["d_ln_P_fundamental"].to_numpy())
    np.testing.assert_allclose(
        out["price_res_inr_sqft_fundamental"].to_numpy(), expected, equal_nan=True
    )
    # A null price stays null — it is not silently imputed.
    assert out["price_res_inr_sqft_fundamental"].isna().sum() == int(
        cells["price_res_inr_sqft"].isna().sum()
    )


def test_supply_effects_shift_the_supply_intercept(params, cells):
    class _Effect:
        def __init__(self, cell, delta, year):
            self.cell = cell
            self.delta_floorspace_sqm = delta
            self.delta_capacity_sqm = 0.0
            self.start_year = year

    # A cell with slack, so the supply shift can actually show up in the quantity response
    # rather than being swallowed by the Section 13.2 capacity constraint.
    assert cells["headroom_sqm"].iloc[0] > 0
    target = str(cells["h3"].iloc[0])
    stock = float(cells["floorspace_res_sqm"].iloc[0])
    effect = _Effect(target, stock, BASE_YEAR)
    out = L6.form_prices(
        cells, params, year=BASE_YEAR, phi_t=0.0, supply_effects=[effect]
    )
    assert out["d_ln_S0"].iloc[0] == pytest.approx(np.log(2.0))
    assert out["d_ln_S0"].iloc[1] == 0.0
    # A positive supply shift depresses price.
    assert out["d_ln_P_fundamental"].iloc[0] < 0.0
    # An effect that has not started yet does nothing.
    later = L6.form_prices(
        cells,
        params,
        year=BASE_YEAR,
        phi_t=0.0,
        supply_effects=[_Effect(target, stock, BASE_YEAR + 1)],
    )
    np.testing.assert_array_equal(later["d_ln_S0"].to_numpy(), np.zeros(len(cells)))
