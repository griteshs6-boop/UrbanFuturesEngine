"""Layer 6 — price formation (spec Section 13).

This is where demand, supply and regulation meet: per cell, per year, the market clears in
log space, the capacity constraint decides how much of a demand shock lands on price rather
than on quantity, speculative overshoot is layered on top of (and reported separately from)
the fundamental, the movement is decomposed into named factors, and an overheating detector
looks for prices that have run away from the model.

The spatial hedonic that supplies `gamma` lives in :mod:`ufe.layers.hedonic` (Section 13.0),
which is a separate module because Section 13.0 is an estimation procedure with its own
mandatory diagnostics, not a per-year transform.

Public entry point
------------------
:func:`form_prices` — pure, returns a NEW frame with the same index and row count as its
input plus the columns listed in :data:`OUTPUT_COLUMNS`. The building blocks
(:func:`demand_shift`, :func:`supply_shift`, :func:`clear_market`, :func:`overshoot`,
:func:`decompose`, :func:`overheating`, :func:`monte_carlo_price_paths`,
:func:`uncertainty_bands`) are public too, so a runner can assemble the Section 13.4
leave-one-out ablation and the Section 13.6 Monte Carlo without re-deriving anything.

Units (spec Section 0.3) — read this before touching any area
--------------------------------------------------------------
    "Price | INR per square foot | the market convention; convert internally only for
    area math"

Prices (`price_res_inr_sqft`, `price_land_inr_sqft`) and rents (`rent_res_inr_sqft_mo`) are
per **square foot**. Every stock and capacity quantity (`floorspace_res_sqm`,
`headroom_sqm`, `capacity_sqm`, `absorption_cap_sqm`) is in **square metres**. A unit error
between them is invisible and catastrophic, so this module makes every conversion explicit
and routes them all through four named functions — :func:`sqft_from_sqm`,
:func:`sqm_from_sqft`, :func:`price_inr_per_sqm`, :func:`price_inr_per_sqft` — plus
:func:`value_inr`, the *only* place a price is ever multiplied by an area.

Three facts save the model from most of the risk:

1. Market clearing is done entirely in **log differences**, which are dimensionless. `d ln
   P` and `d ln Q` need no conversion at all.
2. The capacity constraint (Section 13.2) compares square metres with square metres.
3. The Section 13.5 yield flag divides an INR/sqft rent by an INR/sqft price — also
   dimensionless.

So the *only* genuine conversion in Layer 6 is :func:`value_inr`. The conversion factor is
not a Python literal: it is the EPSG definition of the international foot (EPSG unit 9002,
0.3048 m exactly) read out of pyproj's unit registry, the same registry :mod:`ufe.geo`
already depends on. Likewise ``MONTHS_PER_YEAR`` is read from :mod:`calendar` rather than
typed as ``12``.

Numeric policy (CONTRACT.md rule 1)
-----------------------------------
Every coefficient, elasticity, cap, percentile, tolerance and iteration limit is read from
YAML through :class:`ufe.params.Params`; the parameter paths are the ``P_*`` module
constants. The only literals in this file are ``0``, ``1`` and array indices. Two
definitional constants — a half is one over one plus one, a foot is an EPSG unit — are
derived rather than typed.

Parameters Section 13 needs that ``config/params/price.yaml`` does not define
---------------------------------------------------------------------------
* :data:`P_EPS_LAND_MULTIPLE` — Section 13.1 says the land pass uses "`eps_land` higher"
  but never says how much higher. :func:`form_prices` therefore defaults
  ``include_land=False``; asking for the land pass without supplying the multiple raises
  :class:`ufe.errors.MissingParameter` naming the path.
* :data:`P_BAND_PERCENTILE_LOW` / :data:`P_BAND_PERCENTILE_HIGH` / :data:`P_MC_DRAWS` —
  Section 13.6 defers the bands to Module 10 and names no percentiles.

Convergence controls (:data:`P_MAX_ITERATIONS`, :data:`P_CONVERGENCE_TOL`) are read from
``behaviour.agglomeration.*``: Section 13 names no convergence controls of its own, and
Section 21 lists the price-explosion guard under "Agglomeration divergence", whose
iteration limit and tolerance live there. A dedicated ``price.clearing.*`` block would be
better and is recommended in the build report.

Where this module knowingly departs from the Section 13 prose
--------------------------------------------------------------
Section 13.1 puts the macro trend `phi_t` *inside* `d ln D_i`, which is then divided by
`(eta + eps_i)`. The Section 13 ACCEPTANCE block requires that "running with no projects
and `phi_t = 0.055` produces exactly 5.5 log points of appreciation in every cell and zero
excess-over-trend anywhere". Those two statements cannot both hold: `phi / (eta + eps_i)`
varies with `eps_i` and equals `phi` only when `eta + eps_i == 1`. The acceptance block is
the test, so `phi_t` is applied as an additive trend on `ln P` and excluded from the
elasticity-divided local demand shift, and quantity responds to the local (relative) price
movement only. Both components are reported separately (`phi_t`, `d_ln_D_local`,
`d_ln_D`), so a caller who prefers the prose reading can reassemble it.
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ufe.errors import ConvergenceError, MissingParameter

logger = logging.getLogger(__name__)

__all__ = [
    "ATTR_KEY",
    "OUTPUT_COLUMNS",
    "MONTHS_PER_YEAR",
    "metres_per_foot",
    "sqm_per_sqft",
    "sqft_from_sqm",
    "sqm_from_sqft",
    "price_inr_per_sqm",
    "price_inr_per_sqft",
    "value_inr",
    "gross_yield",
    "field_effect",
    "demand_shift",
    "supply_shift",
    "macro_shift",
    "ClearingResult",
    "clear_market",
    "overshoot",
    "FactorDecomposition",
    "decompose",
    "model_residual",
    "overheating",
    "monte_carlo_price_paths",
    "uncertainty_bands",
    "form_prices",
]

# --------------------------------------------------------------------------------------
# parameter paths (spec Section 13, config/params/price.yaml)
# --------------------------------------------------------------------------------------

# 13.0 / 13.1 — hedonic and demand
P_GAMMA_BUILT = "price.hedonic.gamma_access_built"
P_GAMMA_LAND_MULTIPLE = "price.hedonic.gamma_land_multiple"
P_ETA = "price.hedonic.eta_demand_price"

# 13.1 — macro. The scenario name is interpolated into this template.
P_MACRO_SCENARIO_TEMPLATE = "price.macro.scenarios.{scenario}"
P_MACRO_PROBABILITY_TEMPLATE = "price.macro.scenario_probabilities.{scenario}"
DEFAULT_SCENARIO = "base"

# 9.4 / 13.1 — field caps
P_FIELD_CAP_LOW = "price.fields.cap_low"
P_FIELD_CAP_HIGH = "price.fields.cap_high"
P_FIELD_CAP_WARN_SHARE = "price.fields.cap_warn_share"

# 13.3 — overshoot
P_OVERSHOOT_PEAK = "price.overshoot.peak_factor"
P_OVERSHOOT_HALF_LIFE = "price.overshoot.half_life_years"
P_OVERSHOOT_TRIGGER = "price.overshoot.trigger_min_shock"

# 13.4 — decomposition
P_MAX_FACTORS = "price.decomposition.max_factors"
P_NORMALISE_EPSILON = "price.decomposition.normalise_epsilon"
P_INTERACTION_WARN_SHARE = "price.decomposition.interaction_warn_share_of_total"
P_INTERACTION_WARN_CELLS = "price.decomposition.interaction_warn_cell_share"

# 13.5 — residual and overheating detector
P_OVERHEAT_PERCENTILE = "price.residual.overheat_percentile"
P_PHYSICAL_DIVERGENCE_RATIO = "price.residual.physical_divergence_ratio"
P_PHYSICAL_MIN_PRICE_CAGR = "price.residual.physical_min_price_cagr"
P_OVERHEAT_YIELD_THRESHOLD = "price.yields.overheat_yield_threshold"
P_NORMAL_GROSS_YIELD = "price.yields.normal_gross_yield"

# 13.1 / 13.2 — convergence controls. See the module docstring: Section 13 names none, so
# these come from Section 21's agglomeration-divergence guard.
P_MAX_ITERATIONS = "behaviour.agglomeration.max_iterations"
P_CONVERGENCE_TOL = "behaviour.agglomeration.convergence_tol"

# --- paths Section 13 needs that price.yaml does not define -----------------------------
#: Section 13.1: the land pass uses "`eps_land` higher" — higher by an unstated amount.
P_EPS_LAND_MULTIPLE = "price.hedonic.eps_land_multiple"
#: Section 13.6 defers band construction to Module 10 and names no percentiles.
P_BAND_PERCENTILE_LOW = "price.uncertainty.band_percentile_low"
P_BAND_PERCENTILE_HIGH = "price.uncertainty.band_percentile_high"
P_MC_DRAWS = "price.uncertainty.monte_carlo_draws"

#: The frame-metadata slot the run diagnostics are attached to (cf. `l4_supply.ATTR_KEY`).
ATTR_KEY = "price_diagnostics"

# --------------------------------------------------------------------------------------
# column names
# --------------------------------------------------------------------------------------

COL_H3 = "h3"
COL_HOUSEHOLDS = "households"
COL_LNA = "lnA"
COL_EPS_SUPPLY = "eps_supply"
COL_HEADROOM = "headroom_sqm"
COL_FLOORSPACE_RES = "floorspace_res_sqm"
COL_PRICE_RES = "price_res_inr_sqft"
COL_PRICE_LAND = "price_land_inr_sqft"
COL_RENT_RES = "rent_res_inr_sqft_mo"

#: Columns :func:`form_prices` adds. None of these are declared in
#: ``ufe/store/schemas.py``'s strict `cells` schema — see the build report.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "phi_t",
    "d_ln_D_local",
    "d_ln_D",
    "d_ln_S0",
    "d_ln_P_local",
    "d_ln_P_fundamental",
    "d_ln_Q",
    "quantity_constrained",
    "excess_over_trend",
    "overshoot_log",
    "d_ln_P_reported",
    "price_res_inr_sqft_fundamental",
    "price_res_inr_sqft_reported",
    "floorspace_res_sqm_new",
)

LAND_OUTPUT_COLUMNS: tuple[str, ...] = (
    "d_ln_P_land",
    "price_land_inr_sqft_fundamental",
)

# --------------------------------------------------------------------------------------
# 0.3 — units. No conversion factor is a Python literal (CONTRACT.md rule 1).
# --------------------------------------------------------------------------------------

#: Twelve, taken from the Gregorian calendar rather than typed as a number. Used only by
#: the Section 13.5 gross-yield flag, which annualises a monthly rent.
MONTHS_PER_YEAR: int = len(calendar.month_abbr[1:])

#: A half, by the definition of a half-life. Section 13.3's ``exp(-ln2 * t / half_life)``
#: is identically ``(1/2) ** (t / half_life)``, which avoids typing ``2``.
_ONE_HALF: float = 1 / (1 + 1)

_EPSG_FOOT_UNIT = "foot"


@lru_cache(maxsize=None)
def metres_per_foot() -> float:
    """Metres in one international foot, from the EPSG unit registry (EPSG:9002).

    0.3048 exactly, by the 1959 international yard and pound agreement. Read from pyproj —
    which :mod:`ufe.geo` already depends on — rather than typed into Python, so the number
    has a citable authority and CONTRACT.md rule 1 is not bent.
    """
    import pyproj

    units = pyproj.get_units_map()
    try:
        return float(units[_EPSG_FOOT_UNIT].conv_factor)
    except KeyError as exc:  # pragma: no cover - pyproj always ships EPSG:9002
        raise MissingParameter(
            "pyproj's unit registry has no EPSG:9002 'foot'; cannot convert INR/sqft "
            "prices to areas (spec Section 0.3)"
        ) from exc


@lru_cache(maxsize=None)
def sqm_per_sqft() -> float:
    """Square metres in one square foot — 0.09290304 exactly."""
    foot = metres_per_foot()
    return foot * foot


def sqft_from_sqm(area_sqm: Any) -> Any:
    """Convert an area from square metres (the engine's area unit) to square feet."""
    return np.divide(area_sqm, sqm_per_sqft())


def sqm_from_sqft(area_sqft: Any) -> Any:
    """Convert an area from square feet back to square metres."""
    return np.multiply(area_sqft, sqm_per_sqft())


def price_inr_per_sqm(price_inr_sqft: Any) -> Any:
    """Convert a Section 0.3 INR/sqft price to INR/sqm.

    A price *per* square foot rises when restated per square metre, so this divides by
    ``sqm_per_sqft`` exactly as an area-in-sqft would — the reciprocal relationship is the
    classic place to invert the factor by mistake, which is why it is written once here
    and nowhere else.
    """
    return np.divide(price_inr_sqft, sqm_per_sqft())


def price_inr_per_sqft(price_inr_sqm: Any) -> Any:
    """Convert an INR/sqm price back to the Section 0.3 INR/sqft convention."""
    return np.multiply(price_inr_sqm, sqm_per_sqft())


def value_inr(price_inr_sqft: Any, area_sqm: Any) -> Any:
    """Money: an INR/**sqft** price times an area in **sqm**, in INR.

    The single place in Layer 6 where a price meets an area. Written as
    ``price_per_sqft * sqft_from_sqm(area)`` so the conversion is impossible to miss on
    review.
    """
    return np.multiply(price_inr_sqft, sqft_from_sqm(area_sqm))


def gross_yield(rent_inr_sqft_mo: Any, price_inr_sqft: Any) -> Any:
    """Section 13.5's ``(rent_i * 12) / price_i`` — dimensionless, no area conversion.

    Both arguments are INR per square foot (one monthly, one a capital value), so the
    square feet cancel. This is deliberately a named function so nobody is tempted to mix
    an INR/sqm rent into it.
    """
    rent = np.asarray(rent_inr_sqft_mo, dtype=float)
    price = np.asarray(price_inr_sqft, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rent * MONTHS_PER_YEAR / price


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _as_array(
    values: Any, index: pd.Index, *, fill: float = 0.0, name: str = "series"
) -> np.ndarray:
    """Coerce `values` to a float array aligned on `index`, defaulting to `fill`."""
    if values is None:
        return np.full(len(index), float(fill))
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            values = values.reindex(index)
            if values.isna().any():
                raise KeyError(
                    f"{name} does not cover every cell; {int(values.isna().sum())} "
                    "cells would be filled with NaN"
                )
        return values.to_numpy(dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(len(index), float(array))
    if len(array) != len(index):
        raise ValueError(f"{name} has length {len(array)}, expected {len(index)}")
    return array


def _assert_finite(name: str, values: np.ndarray, index: pd.Index | None = None) -> None:
    """Raise :class:`ConvergenceError` rather than propagate an inf/NaN price."""
    bad = ~np.isfinite(values)
    if not bad.any():
        return
    where = ""
    if index is not None:
        labels = list(np.asarray(index)[bad][: 1 + 1 + 1])
        where = f" (first offenders: {labels})"
    raise ConvergenceError(
        f"{name} is not finite in {int(bad.sum())} of {len(values)} cells{where}. "
        "Market clearing refuses to return a silently absurd number "
        "(spec Section 21, 'Agglomeration divergence')."
    )


def _scenario_path(scenario: str) -> str:
    return P_MACRO_SCENARIO_TEMPLATE.format(scenario=scenario)


def _require_path(params: Any, path: str, what: str) -> float:
    if params is None:
        raise MissingParameter(
            f"no Params supplied, so {path!r} ({what}) cannot be resolved"
        )
    try:
        return float(params.value(path))
    except MissingParameter as exc:
        raise MissingParameter(
            f"{path!r} is not defined in config/params/price.yaml but Section 13 needs "
            f"it: {what}. CONTRACT.md rule 1 forbids a default in Python — add it to the "
            "YAML or pass the value explicitly."
        ) from exc


# --------------------------------------------------------------------------------------
# 13.1 — demand shift
# --------------------------------------------------------------------------------------


def field_effect(
    field: Any, params: Any, *, index: pd.Index | None = None, cap: bool = True
) -> pd.Series:
    """The Section 9.4 premium/disamenity field, capped into ``[cap_low, cap_high]``.

    Section 9.4: "Log the number of cells that hit the cap; if more than 2% do, the
    parameters are wrong." The 2% is ``price.fields.cap_warn_share``.
    """
    if index is None:
        if isinstance(field, pd.Series):
            index = field.index
        else:
            raise ValueError("field_effect needs an index when `field` is not a Series")
    values = _as_array(field, index, name="field")
    if not cap:
        return pd.Series(values, index=index, name="field")

    low = _require_path(params, P_FIELD_CAP_LOW, "Section 9.4 field cap (low)")
    high = _require_path(params, P_FIELD_CAP_HIGH, "Section 9.4 field cap (high)")
    warn_share = _require_path(
        params, P_FIELD_CAP_WARN_SHARE, "Section 9.4 cap-hit warning share"
    )
    capped = np.clip(values, low, high)
    hit = int(np.count_nonzero(capped != values))
    if len(values) and hit / len(values) > warn_share:
        logger.warning(
            "%d of %d cells (%s) hit the Section 9.4 field cap, more than the %s "
            "allowed by %s: the field parameters are wrong",
            hit,
            len(values),
            format(hit / len(values), ".1%"),
            format(warn_share, ".1%"),
            P_FIELD_CAP_WARN_SHARE,
        )
    return pd.Series(capped, index=index, name="field")


def demand_shift(
    cells: pd.DataFrame,
    params: Any,
    *,
    d_lnA: Any = None,
    new_hh: Any = None,
    field: Any = None,
    gamma: float | None = None,
    land: bool = False,
    cap_field: bool = True,
) -> pd.Series:
    """The **local** part of Section 13.1's ``d ln D_i``.

    ::

        d ln D_i = gamma * d lnA_i                 # accessibility
                 + ln( (hh_i + new_hh_i) / hh_i )  # household demand
                 + field_i(residential)            # premium/disamenity net
                 + phi_t                           # city macro

    `phi_t` is deliberately **not** included here — see the module docstring for why the
    macro term is applied as an additive trend on `ln P` instead. Add
    :func:`macro_shift` to the result to recover the Section 13.1 expression verbatim.

    `gamma` defaults to ``price.hedonic.gamma_access_built``; with ``land=True`` it becomes
    ``gamma * price.hedonic.gamma_land_multiple`` (Section 13.1: "For raw land use
    `gamma * gamma_land_multiple`"). Pass `gamma` explicitly to use the *total* effect
    estimated by :func:`ufe.layers.hedonic.fit_hedonic`, which Section 13.0 requires when
    the chosen specification is a spatial lag.
    """
    index = cells.index
    if gamma is None:
        gamma = _require_path(
            params, P_GAMMA_BUILT, "accessibility elasticity of price (Section 13.1)"
        )
    if land:
        gamma = gamma * _require_path(
            params, P_GAMMA_LAND_MULTIPLE, "raw-land accessibility multiple (13.1)"
        )

    access_term = gamma * _as_array(d_lnA, index, name="d_lnA")

    if COL_HOUSEHOLDS not in cells.columns:
        raise KeyError(f"`cells` needs {COL_HOUSEHOLDS!r} for the Section 13.1 demand shift")
    households = np.asarray(cells[COL_HOUSEHOLDS], dtype=float)
    added = _as_array(new_hh, index, name="new_hh")
    with np.errstate(divide="ignore", invalid="ignore"):
        household_term = np.log((households + added) / households)
    # A cell with no households today has no proportional household demand shift; the
    # absolute arrival is Layer 5's business, not a log ratio's.
    household_term = np.where(households > 0, household_term, 0.0)

    field_term = field_effect(field, params, index=index, cap=cap_field).to_numpy()

    return pd.Series(
        access_term + household_term + field_term, index=index, name="d_ln_D_local"
    )


def macro_shift(
    params: Any, *, scenario: str = DEFAULT_SCENARIO, phi_t: float | None = None
) -> float:
    """`phi_t`, the city-wide macro trend in log points per year (Section 13.1)."""
    if phi_t is not None:
        return float(phi_t)
    return float(params.value(_scenario_path(scenario)))


def supply_shift(
    cells: pd.DataFrame,
    effects: Iterable[Any] = (),
    *,
    year: int,
    d_ln_S0: Any = None,
    quantity_col: str = COL_FLOORSPACE_RES,
) -> pd.Series:
    """Section 13.1's ``d ln S0_i`` — the *exogenous* supply shift only.

    "Supply shift `d ln S0_i` comes from `SupplyEffect` (townships, FAR changes) only.
    Ordinary construction is the endogenous `d ln Q`."

    `effects` is any iterable of objects carrying ``cell``, ``delta_floorspace_sqm`` and
    ``start_year`` — the shape :class:`ufe.layers.l4_supply.SupplyEffect` (and Layer 2's
    equivalent) exposes. It is duck-typed on purpose so this module never imports another
    layer. Only effects whose ``start_year`` equals `year` are applied; the shift is
    ``ln((Q_i + delta_i) / Q_i)``.
    """
    index = cells.index
    if d_ln_S0 is not None:
        return pd.Series(
            _as_array(d_ln_S0, index, name="d_ln_S0"), index=index, name="d_ln_S0"
        )

    delta = np.zeros(len(index))
    effects = list(effects)
    if effects:
        if COL_H3 not in cells.columns:
            raise KeyError(f"`cells` needs {COL_H3!r} to apply supply effects")
        position = {str(h): i for i, h in enumerate(cells[COL_H3])}
        for effect in effects:
            if int(getattr(effect, "start_year")) != int(year):
                continue
            cell = str(getattr(effect, "cell"))
            if cell not in position:
                continue
            delta[position[cell]] += float(getattr(effect, "delta_floorspace_sqm", 0.0))

    quantity = np.asarray(cells[quantity_col], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.log((quantity + delta) / quantity)
    shift = np.where((quantity > 0) & (quantity + delta > 0), shift, 0.0)
    return pd.Series(shift, index=index, name="d_ln_S0")


# --------------------------------------------------------------------------------------
# 13.1 / 13.2 — market clearing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClearingResult:
    """The outcome of one year's clearing (Sections 13.1 and 13.2)."""

    #: Total log price change, ``phi_t + d_ln_P_local``.
    d_ln_P: np.ndarray
    #: The elasticity-mediated part, excluding the macro trend.
    d_ln_P_local: np.ndarray
    #: Log quantity change, after the Section 13.2 capacity constraint.
    d_ln_Q: np.ndarray
    #: True where headroom or the absorption cap bound the quantity response.
    constrained: np.ndarray
    iterations: int
    converged: bool
    #: Largest change in `d_ln_P_local` at the final iteration.
    residual_change: float


def clear_market(
    params: Any,
    *,
    d_ln_D_local: Any,
    d_ln_S0: Any,
    eta: float,
    eps: Any,
    quantity_sqm: Any,
    headroom_sqm: Any,
    absorption_cap_sqm: Any = None,
    phi_t: float = 0.0,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    index: pd.Index | None = None,
) -> ClearingResult:
    """Clear the market, per cell, subject to the Section 13.2 capacity constraint.

    Section 13.1, in differences::

        d ln P_i = ( d ln D_i - d ln S0_i ) / ( eta + eps_i )
        d ln Q_i =   eps_i * d ln P_i + d ln S0_i

    Section 13.2::

        Q_new_i = Q_i * exp(d ln Q_i)
        allowed = min(Q_new_i, Q_i + headroom_sqm_i, Q_i + absorption_cap_sqm_i)
        if allowed < Q_new_i:
            d ln Q_i = ln(allowed / Q_i)
            d ln P_i = (d ln D_i - d ln Q_i) / eta

    "This is the mechanism that produces price spikes in constrained cells and flat prices
    in elastic ones. **It is the single most important behaviour in the model.**"

    The two equations are solved as a fixed point rather than in one pass, because the
    capacity test depends on the price and the constrained price depends on the capped
    quantity. Ordinary cells converge on the first iteration and constrained cells on the
    second; the loop exists so that the *guarantee* is explicit.

    Convergence guarantee (brief requirement 3). The function either converges or raises
    :class:`ufe.errors.ConvergenceError`. It never returns a silently absurd number:

    * a non-finite input (`d_ln_D_local`, `d_ln_S0`) raises immediately;
    * a non-finite iterate — for example ``eta + eps == 0``, a zero-denominator
      market — raises;
    * an iterate whose implied price ratio ``exp(d ln P)`` overflows to infinity raises,
      which caps the answer at a representable multiple without inventing a threshold;
    * failing to settle within ``behaviour.agglomeration.max_iterations`` raises.

    Parameters
    ----------
    eta:
        Demand price elasticity, ``price.hedonic.eta_demand_price``. Positive.
    eps:
        Per-cell supply elasticity, the `eps_supply` column.
    quantity_sqm:
        Current stock, square metres. Cells with no stock get ``d ln Q = 0`` (a log ratio
        against zero is undefined) and their unconstrained price.
    absorption_cap_sqm:
        Section 11.2's cap, from ``l4_supply``'s diagnostics. ``None`` means uncapped.
    """
    if index is None:
        index = pd.RangeIndex(len(np.atleast_1d(np.asarray(quantity_sqm))))

    dD = _as_array(d_ln_D_local, index, name="d_ln_D_local")
    dS0 = _as_array(d_ln_S0, index, name="d_ln_S0")
    eps_arr = _as_array(eps, index, name="eps")
    quantity = _as_array(quantity_sqm, index, name="quantity_sqm")
    headroom = _as_array(headroom_sqm, index, fill=np.inf, name="headroom_sqm")
    cap = _as_array(
        np.inf if absorption_cap_sqm is None else absorption_cap_sqm,
        index,
        fill=np.inf,
        name="absorption_cap_sqm",
    )

    _assert_finite("d_ln_D_local", dD, index)
    _assert_finite("d_ln_S0", dS0, index)
    if not np.isfinite(eta):
        raise ConvergenceError(f"eta must be finite, got {eta}")
    _assert_finite("eps", eps_arr, index)

    if max_iterations is None:
        max_iterations = int(
            _require_path(params, P_MAX_ITERATIONS, "market-clearing iteration cap")
        )
    if tolerance is None:
        tolerance = _require_path(params, P_CONVERGENCE_TOL, "market-clearing tolerance")

    denominator = eta + eps_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        unconstrained = (dD - dS0) / denominator
    _assert_finite(
        "the unconstrained d ln P (eta + eps is zero somewhere)", unconstrained, index
    )

    has_stock = quantity > 0
    ceiling = np.minimum(quantity + headroom, quantity + cap)

    current = unconstrained
    constrained = np.zeros(len(index), dtype=bool)
    d_ln_Q = np.zeros(len(index))
    converged = False
    change = np.inf
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            supply_response = eps_arr * current + dS0
            requested = quantity * np.exp(supply_response)
        allowed = np.minimum(requested, ceiling)
        constrained = has_stock & (allowed < requested)
        with np.errstate(divide="ignore", invalid="ignore"):
            capped_growth = np.log(allowed / quantity)
        d_ln_Q = np.where(
            has_stock, np.where(constrained, capped_growth, supply_response), 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            constrained_price = (dD - d_ln_Q) / eta
        nxt = np.where(constrained, constrained_price, unconstrained)
        _assert_finite("d ln P during market clearing", nxt, index)
        _assert_finite("d ln Q during market clearing", d_ln_Q, index)

        change = float(np.max(np.abs(nxt - current))) if len(nxt) else 0.0
        current = nxt
        if change <= tolerance:
            converged = True
            break

    if not converged:
        raise ConvergenceError(
            f"market clearing did not converge in {max_iterations} iterations "
            f"({P_MAX_ITERATIONS}); largest remaining change {change:.6g} exceeds the "
            f"tolerance {tolerance:.6g} ({P_CONVERGENCE_TOL}). Spec Section 21: prices "
            "exploding in one corridor must raise, not be reported."
        )

    total = float(phi_t) + current
    with np.errstate(over="ignore"):
        growth_factor = np.exp(total)
    _assert_finite(
        "the implied price ratio exp(d ln P) — the price change is not representable",
        growth_factor,
        index,
    )

    return ClearingResult(
        d_ln_P=total,
        d_ln_P_local=current,
        d_ln_Q=d_ln_Q,
        constrained=constrained,
        iterations=iterations,
        converged=True,
        residual_change=change,
    )


# --------------------------------------------------------------------------------------
# 13.3 — overshoot
# --------------------------------------------------------------------------------------


def overshoot(
    shock: Any,
    params: Any,
    *,
    year: float,
    announce_year: Any,
) -> np.ndarray:
    """Section 13.3's speculative overshoot, in log points.

    ::

        if shock_i > trigger_min_shock:
            overshoot_i(t) = peak_factor * shock_i * exp( -ln2 * (t - announce_year) / half_life )

    ``exp(-ln2 * x)`` is written as ``(1/2) ** x`` — identical, and it keeps a ``2`` out of
    the source. "Overshoot decays; the fundamental does not. Report the two separately —
    the overshoot component *is* the 'priced in on hype' number."
    """
    shock_arr = np.asarray(shock, dtype=float)
    announce = np.asarray(announce_year, dtype=float)
    peak_factor = _require_path(params, P_OVERSHOOT_PEAK, "Section 13.3 peak factor")
    half_life = _require_path(params, P_OVERSHOOT_HALF_LIFE, "Section 13.3 half-life")
    trigger = _require_path(params, P_OVERSHOOT_TRIGGER, "Section 13.3 trigger")
    if half_life <= 0:
        raise ValueError(f"{P_OVERSHOOT_HALF_LIFE} must be positive, got {half_life}")

    elapsed = float(year) - announce
    decay = np.where(elapsed >= 0, _ONE_HALF ** (elapsed / half_life), 0.0)
    return np.where(shock_arr > trigger, peak_factor * shock_arr * decay, 0.0)


# --------------------------------------------------------------------------------------
# 13.4 — factor decomposition
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorDecomposition:
    """Leave-one-out factor decomposition of a log price movement (Section 13.4)."""

    factors: tuple[str, ...]
    ln_p_base: pd.Series
    ln_p_full: pd.Series
    loo: dict[str, pd.Series]
    total: pd.Series
    #: ``raw_lambda_if = lnP_full_i - lnP_loo_f_i``.
    raw: pd.DataFrame
    #: ``lambda_if = raw_lambda_if * total_i / max(sum_f raw_lambda_if, epsilon)``.
    normalised: pd.DataFrame
    #: ``interaction_i = total_i - sum_f raw_lambda_if``. Reported, never hidden.
    interaction: pd.Series
    #: False when the interaction term is too large for the factors to be separable.
    separable: bool
    warning: str | None
    #: Cells where the normalisation denominator fell below `normalise_epsilon`, so the
    #: normalised lambdas are numerically meaningless there.
    degenerate: pd.Series

    def reconciliation_error(self) -> pd.Series:
        """``sum_f raw_lambda_if + interaction_i - total_i`` — zero by construction.

        The Section 13 ACCEPTANCE identity, exposed so the runner can assert it.
        """
        return self.raw.sum(axis=1) + self.interaction - self.total

    def to_frame(self) -> pd.DataFrame:
        """One row per cell: every normalised lambda, the interaction, and the total."""
        out = self.normalised.copy()
        out["interaction"] = self.interaction
        out["total"] = self.total
        return out


def decompose(
    run: Callable[[tuple[str, ...]], pd.Series],
    factors: Sequence[str],
    params: Any,
) -> FactorDecomposition:
    """Section 13.4's leave-one-out ablation.

    ::

        Run 0:      baseline — no projects, macro only          -> lnP_base_i
        Run FULL:   all factors active                          -> lnP_full_i
        Run LOO_f:  all factors except f                        -> lnP_loo_f_i

    `run` is called with the active factor set and must return a `pd.Series` of ``ln P``
    indexed by cell. It must be *pure*: the "Removing a factor and re-running FULL
    reproduces that factor's LOO run exactly" acceptance test is a purity test.

    Determinism (spec Section 15.2, "sort before iterating over sets"; Section 23 item 4,
    byte-identical output). The active set is handed to `run` as a **sorted tuple**, never
    as a `set` or `frozenset`. A `run` implementation that sums a float contribution per
    active factor would otherwise accumulate in string-hash order, which varies with
    ``PYTHONHASHSEED``, and floating-point addition is not associative — so the FULL and
    LOO runs would disagree in the last bits from one interpreter start to the next.
    `run` may still test membership (``name in active``) exactly as before.

    "Leave-one-out is not Shapley. It undercounts complementary factors and overcounts
    substitutes. **Report `interaction_i` explicitly in the output**; if it exceeds 20% of
    `total_i` for more than 5% of cells, the factors are not separable and the
    decomposition should be presented with a warning." Both thresholds are read from
    ``price.decomposition.*``; the warning is on :attr:`FactorDecomposition.warning`.

    Known defect in the spec's normalisation (reported in the build summary): the
    denominator is written ``max(sum_f raw_lambda_if, 1e-9)``, an unsigned maximum. When
    ``sum_f raw`` is negative (a factor group that *depresses* price) the denominator
    collapses to the epsilon and the normalised lambdas explode. The formula is
    implemented as written, and the affected cells are flagged on
    :attr:`FactorDecomposition.degenerate` and logged, rather than silently altered.
    """
    factors = tuple(dict.fromkeys(factors))
    max_factors = int(_require_path(params, P_MAX_FACTORS, "Section 13.4 factor cap"))
    if len(factors) > max_factors:
        raise ValueError(
            f"Section 13.4 caps the decomposition at {max_factors} factors "
            f"({P_MAX_FACTORS}); got {len(factors)}. 'runs scale linearly with factor "
            "count'."
        )
    epsilon = _require_path(
        params, P_NORMALISE_EPSILON, "Section 13.4 normalisation epsilon"
    )
    warn_share = _require_path(
        params, P_INTERACTION_WARN_SHARE, "Section 13.4 interaction warning share"
    )
    warn_cells = _require_path(
        params, P_INTERACTION_WARN_CELLS, "Section 13.4 interaction cell share"
    )

    # Sorted tuples, not frozensets: see the "Determinism" note in the docstring.
    all_factors: tuple[str, ...] = tuple(sorted(factors))
    ln_p_base = run(())
    ln_p_full = run(all_factors)
    index = ln_p_full.index
    loo = {
        name: run(tuple(f for f in all_factors if f != name)) for name in factors
    }

    total = (ln_p_full - ln_p_base).rename("total")
    raw = pd.DataFrame(
        {name: ln_p_full - series for name, series in loo.items()}, index=index
    )
    raw_sum = raw.sum(axis=1)
    interaction = (total - raw_sum).rename("interaction")

    denominator = np.maximum(raw_sum.to_numpy(), epsilon)
    degenerate = pd.Series(
        raw_sum.to_numpy() < epsilon, index=index, name="degenerate"
    )
    normalised = raw.mul(total.to_numpy() / denominator, axis=0)

    if bool(degenerate.any()):
        logger.warning(
            "%d of %d cells have sum(raw lambda) below %s = %g, so Section 13.4's "
            "max(sum, epsilon) normalisation is numerically meaningless there; the cells "
            "are flagged on FactorDecomposition.degenerate",
            int(degenerate.sum()),
            len(index),
            P_NORMALISE_EPSILON,
            epsilon,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.abs(interaction.to_numpy()) / np.abs(total.to_numpy())
    share = np.where(np.isfinite(share), share, 0.0)
    offending = float(np.mean(share > warn_share)) if len(index) else 0.0
    separable = offending <= warn_cells
    warning: str | None = None
    if not separable:
        warning = (
            f"the interaction term exceeds {warn_share:.0%} of the total price movement "
            f"in {offending:.1%} of cells (more than the {warn_cells:.0%} allowed by "
            f"{P_INTERACTION_WARN_CELLS}): the factors are not separable and this "
            "decomposition must be presented with a warning (spec Section 13.4)."
        )
        logger.warning("%s", warning)

    return FactorDecomposition(
        factors=factors,
        ln_p_base=ln_p_base,
        ln_p_full=ln_p_full,
        loo=loo,
        total=total,
        raw=raw,
        normalised=normalised,
        interaction=interaction,
        separable=separable,
        warning=warning,
        degenerate=degenerate,
    )


# --------------------------------------------------------------------------------------
# 13.5 — the residual and overheating detector
# --------------------------------------------------------------------------------------


def model_residual(
    ln_p_observed: pd.Series, ln_p_base_today: pd.Series, lambdas: pd.DataFrame
) -> pd.Series:
    """Section 13.5's residual.

    ::

        lnP_model_today_i = lnP_base_today_i + sum_f lambda_if(today)
        residual_i        = lnP_observed_i - lnP_model_today_i
    """
    model = ln_p_base_today + lambdas.sum(axis=1)
    return (ln_p_observed - model).rename("residual")


#: The three independent Section 13.5 flags, in the order they are summed.
FLAG_COLUMNS: tuple[str, ...] = ("flag_residual", "flag_yield", "flag_physical")


def overheating(
    params: Any,
    *,
    residual: pd.Series,
    price_inr_sqft: pd.Series,
    rent_inr_sqft_mo: pd.Series | None = None,
    price_cagr_5y: pd.Series | None = None,
    builtup_cagr_5y: pd.Series | None = None,
    nightlight_cagr_5y: pd.Series | None = None,
) -> pd.DataFrame:
    """Section 13.5's three independent flags and the 0-3 overheat score.

    ::

        flag_residual_i  = residual_i > percentile(residual, residual.overheat_percentile)
        flag_yield_i     = (rent_i * 12) / price_i < overheat_yield_threshold
        flag_physical_i  = price_cagr_5y_i > 1.8 * max(builtup_cagr_5y_i, nightlight_cagr_5y_i)
                           and price_cagr_5y_i > 0.08

    "Report the score and the constituent flags, never a single opaque number. Cells
    missing rent data get `flag_yield = null`, and the score is reported out of 2 with a
    note."

    This is the guard the brief pairs with Section 21's "Agglomeration divergence — prices
    explode in one corridor": a corridor whose price has left the model behind, whose
    rental yield has collapsed and whose physical development has not kept up scores 3.
    """
    index = residual.index
    percentile = _require_path(
        params, P_OVERHEAT_PERCENTILE, "Section 13.5 residual percentile"
    )
    yield_threshold = _require_path(
        params, P_OVERHEAT_YIELD_THRESHOLD, "Section 13.5 overheated gross yield"
    )
    ratio = _require_path(
        params, P_PHYSICAL_DIVERGENCE_RATIO, "Section 13.5 physical divergence ratio"
    )
    min_cagr = _require_path(
        params, P_PHYSICAL_MIN_PRICE_CAGR, "Section 13.5 minimum price CAGR"
    )

    residual_values = residual.to_numpy(dtype=float)
    if np.isfinite(residual_values).any():
        threshold = float(np.nanpercentile(residual_values, percentile))
    else:  # pragma: no cover - degenerate input
        threshold = np.inf
    flag_residual = residual_values > threshold

    yields = gross_yield(
        np.full(len(index), np.nan)
        if rent_inr_sqft_mo is None
        else rent_inr_sqft_mo.to_numpy(dtype=float),
        price_inr_sqft.to_numpy(dtype=float),
    )
    yield_known = np.isfinite(yields)
    flag_yield = pd.array(
        np.where(yield_known, yields < yield_threshold, None), dtype="boolean"
    )

    price_cagr = _as_array(price_cagr_5y, index, name="price_cagr_5y")
    builtup_cagr = _as_array(builtup_cagr_5y, index, name="builtup_cagr_5y")
    nightlight_cagr = _as_array(nightlight_cagr_5y, index, name="nightlight_cagr_5y")
    physical = np.maximum(builtup_cagr, nightlight_cagr)
    flag_physical = (price_cagr > ratio * physical) & (price_cagr > min_cagr)

    score = (
        flag_residual.astype(np.int64)
        + np.where(yield_known, flag_yield.fillna(False).to_numpy(), False).astype(
            np.int64
        )
        + flag_physical.astype(np.int64)
    )
    score_max = np.where(yield_known, len(FLAG_COLUMNS), len(FLAG_COLUMNS) - 1).astype(
        np.int64
    )

    return pd.DataFrame(
        {
            "residual": residual_values,
            "residual_threshold": np.full(len(index), threshold),
            "gross_yield": yields,
            "flag_residual": flag_residual,
            "flag_yield": flag_yield,
            "flag_physical": flag_physical,
            "overheat_score": score,
            "overheat_score_max": score_max,
        },
        index=index,
    )


# --------------------------------------------------------------------------------------
# 13.6 — uncertainty bands
# --------------------------------------------------------------------------------------


def monte_carlo_price_paths(
    cells: pd.DataFrame,
    params: Any,
    *,
    years: Sequence[int],
    n_draws: int | None = None,
    rng: np.random.Generator,
    scenario: str = DEFAULT_SCENARIO,
    base_year: int | None = None,
    d_lnA: Any = None,
    new_hh: Any = None,
    field: Any = None,
    absorption_cap_sqm: Any = None,
) -> np.ndarray:
    """Monte Carlo cumulative log price change, shape ``(n_draws, n_years, n_cells)``.

    Section 13.6: "Bands come from Monte Carlo (Module 10), not from an analytic formula.
    The deterministic run produces the median path only." This is the Layer 6 half of that
    — the draw machinery. Module 10 owns the sampling design; this function exists so the
    bands can be built (and their monotonicity tested) without it.

    Each draw walks one year at a time from ``base_year + 1`` to ``max(years)``, redrawing
    `phi_t`, `gamma` and `eta` from their YAML ranges with :meth:`ufe.params.Params.sample`
    at every step, and accumulates `d ln P`. Uncertainty therefore compounds with the
    horizon by construction, and widening the YAML range widens the band. Paths are in
    cumulative **log points relative to the base year**, not price levels, so cells with a
    null observed price still contribute.

    `rng` is required and explicit — no unseeded randomness anywhere (CONTRACT.md rule 5).
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("monte_carlo_price_paths needs an explicit numpy Generator")
    years = tuple(int(y) for y in years)
    if not years or list(years) != sorted(years):
        raise ValueError("`years` must be a non-empty increasing sequence")
    if base_year is None:
        base_year = int(params.city_config["base_year"])
    if years[0] <= base_year:
        raise ValueError(f"`years` must all follow the base year {base_year}")
    if n_draws is None:
        n_draws = int(
            _require_path(params, P_MC_DRAWS, "Section 13.6 Monte Carlo draw count")
        )

    index = cells.index
    n_cells = len(index)
    scenario_path = _scenario_path(scenario)
    params.value(scenario_path)  # fail fast on an unknown scenario

    eps = np.asarray(cells[COL_EPS_SUPPLY], dtype=float)
    quantity = np.asarray(cells[COL_FLOORSPACE_RES], dtype=float)
    headroom = np.asarray(cells[COL_HEADROOM], dtype=float)
    cap = _as_array(
        np.inf if absorption_cap_sqm is None else absorption_cap_sqm,
        index,
        fill=np.inf,
        name="absorption_cap_sqm",
    )
    access = _as_array(d_lnA, index, name="d_lnA")
    added_hh = _as_array(new_hh, index, name="new_hh")
    field_term = field_effect(field, params, index=index).to_numpy()
    households = np.asarray(cells[COL_HOUSEHOLDS], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        household_term = np.log((households + added_hh) / households)
    household_term = np.where(households > 0, household_term, 0.0)

    step_years = tuple(range(base_year + 1, years[-1] + 1))
    wanted = {year: position for position, year in enumerate(years)}
    paths = np.empty((int(n_draws), len(years), n_cells), dtype=float)

    for draw in range(int(n_draws)):
        cumulative = np.zeros(n_cells)
        for step_year in step_years:
            phi = params.sample(scenario_path, rng)
            gamma = params.sample(P_GAMMA_BUILT, rng)
            eta = params.sample(P_ETA, rng)
            local = gamma * access + household_term + field_term
            result = clear_market(
                params,
                d_ln_D_local=local,
                d_ln_S0=np.zeros(n_cells),
                eta=eta,
                eps=eps,
                quantity_sqm=quantity,
                headroom_sqm=headroom,
                absorption_cap_sqm=cap,
                phi_t=phi,
                index=index,
            )
            cumulative = cumulative + result.d_ln_P
            if step_year in wanted:
                paths[draw, wanted[step_year], :] = cumulative
    return paths


def uncertainty_bands(
    paths: np.ndarray,
    *,
    index: pd.Index,
    years: Sequence[int],
    params: Any = None,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
) -> pd.DataFrame:
    """Percentile bands over Monte Carlo `paths` (Section 13.6).

    Returns a long frame with one row per (cell, year) and columns ``cell``, ``year``,
    ``p_low``, ``median``, ``p_high``, ``width``. The band is in cumulative log points.

    Section 13.6 names no percentiles, so they are read from
    :data:`P_BAND_PERCENTILE_LOW` / :data:`P_BAND_PERCENTILE_HIGH`, which do not exist in
    ``price.yaml`` yet; pass them explicitly or add them to the YAML.
    """
    paths = np.asarray(paths, dtype=float)
    years = tuple(int(y) for y in years)
    if paths.ndim != 1 + 1 + 1:
        raise ValueError("`paths` must have shape (n_draws, n_years, n_cells)")
    if paths.shape[1] != len(years) or paths.shape[2] != len(index):
        raise ValueError(
            f"`paths` shape {paths.shape} does not match {len(years)} years and "
            f"{len(index)} cells"
        )

    missing: list[str] = []
    if percentile_low is None:
        try:
            percentile_low = _require_path(
                params, P_BAND_PERCENTILE_LOW, "Section 13.6 lower band percentile"
            )
        except MissingParameter:
            missing.append(P_BAND_PERCENTILE_LOW)
    if percentile_high is None:
        try:
            percentile_high = _require_path(
                params, P_BAND_PERCENTILE_HIGH, "Section 13.6 upper band percentile"
            )
        except MissingParameter:
            missing.append(P_BAND_PERCENTILE_HIGH)
    if missing:
        raise MissingParameter(
            "Section 13.6 uncertainty bands need percentiles that "
            "config/params/price.yaml does not define: "
            + ", ".join(missing)
            + ". Add them to the YAML (CONTRACT.md rule 1 forbids a default here) or pass "
            "percentile_low= / percentile_high= explicitly."
        )
    if not percentile_low < percentile_high:
        raise ValueError(
            f"{P_BAND_PERCENTILE_LOW} ({percentile_low}) must be below "
            f"{P_BAND_PERCENTILE_HIGH} ({percentile_high})"
        )

    low = np.percentile(paths, percentile_low, axis=0)
    high = np.percentile(paths, percentile_high, axis=0)
    median = np.median(paths, axis=0)

    cell_labels = np.asarray(index)
    return pd.DataFrame(
        {
            "cell": np.tile(cell_labels, len(years)),
            "year": np.repeat(np.asarray(years), len(index)),
            "p_low": low.ravel(),
            "median": median.ravel(),
            "p_high": high.ravel(),
            "width": (high - low).ravel(),
        }
    )


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def form_prices(
    cells: pd.DataFrame,
    params: Any,
    *,
    year: int,
    d_lnA: Any = None,
    new_hh: Any = None,
    field: Any = None,
    cap_field: bool = True,
    gamma: float | None = None,
    eta: float | None = None,
    scenario: str = DEFAULT_SCENARIO,
    phi_t: float | None = None,
    supply_effects: Iterable[Any] = (),
    d_ln_S0: Any = None,
    absorption_cap_sqm: Any = None,
    announcement_shock: Any = None,
    announce_year: Any = None,
    include_land: bool = False,
    eps_land_multiple: float | None = None,
) -> pd.DataFrame:
    """Form this year's prices (spec Section 13.1-13.3). Pure.

    Returns a NEW frame with the same index and row count as `cells`, every input column
    untouched, plus :data:`OUTPUT_COLUMNS` (and :data:`LAND_OUTPUT_COLUMNS` when
    `include_land`). Run diagnostics are attached at ``out.attrs[ATTR_KEY]`` for the
    manifest.

    Parameters
    ----------
    d_lnA:
        This year's change in log accessibility, from Layer 1.
    new_hh:
        Households arriving in the cell this year, from Layer 5.
    field:
        The Section 9.4 net premium/disamenity field for residential use, in log points.
        Capped into ``price.fields.[cap_low, cap_high]`` unless `cap_field` is False.
    gamma:
        Overrides ``price.hedonic.gamma_access_built``. Pass
        ``hedonic_fit.gamma("lnA")`` — the **total** effect — when a spatial lag was
        selected (Section 13.0).
    supply_effects, d_ln_S0:
        Section 13.1's exogenous supply shift. `d_ln_S0` takes precedence when both given.
    announcement_shock, announce_year:
        Section 13.3. `announcement_shock` is "the fundamental `d ln P` attributable to
        newly-announced projects this year"; the caller computes it (typically as the
        difference between a run with and without the new announcements) because Layer 6
        cannot know which part of `field` came from which project.
    include_land:
        Run Section 13.1's parallel raw-land pass. Off by default because
        ``price.yaml`` does not define the land supply elasticity — see
        :data:`P_EPS_LAND_MULTIPLE`.
    """
    if COL_EPS_SUPPLY not in cells.columns:
        raise KeyError(
            f"`cells` needs {COL_EPS_SUPPLY!r} from Layer 0 before prices can be formed"
        )
    index = cells.index
    out = cells.copy()

    phi = macro_shift(params, scenario=scenario, phi_t=phi_t)
    if eta is None:
        eta = _require_path(params, P_ETA, "demand price elasticity (Section 13.1)")
    if eta <= 0:
        raise ValueError(f"{P_ETA} must be positive, got {eta}")

    local_demand = demand_shift(
        cells,
        params,
        d_lnA=d_lnA,
        new_hh=new_hh,
        field=field,
        gamma=gamma,
        cap_field=cap_field,
    )
    supply = supply_shift(cells, supply_effects, year=year, d_ln_S0=d_ln_S0)

    eps = np.asarray(cells[COL_EPS_SUPPLY], dtype=float)
    quantity = np.asarray(cells[COL_FLOORSPACE_RES], dtype=float)
    headroom = _as_array(
        cells[COL_HEADROOM] if COL_HEADROOM in cells.columns else None,
        index,
        fill=np.inf,
        name=COL_HEADROOM,
    )

    result = clear_market(
        params,
        d_ln_D_local=local_demand,
        d_ln_S0=supply,
        eta=eta,
        eps=eps,
        quantity_sqm=quantity,
        headroom_sqm=headroom,
        absorption_cap_sqm=absorption_cap_sqm,
        phi_t=phi,
        index=index,
    )

    shock = _as_array(announcement_shock, index, name="announcement_shock")
    if announce_year is None:
        announce_year = year
    overshoot_log = overshoot(shock, params, year=year, announce_year=announce_year)

    observed_price = (
        np.asarray(cells[COL_PRICE_RES], dtype=float)
        if COL_PRICE_RES in cells.columns
        else np.full(len(index), np.nan)
    )
    d_ln_p_reported = result.d_ln_P + overshoot_log

    out["phi_t"] = phi
    out["d_ln_D_local"] = local_demand.to_numpy()
    out["d_ln_D"] = local_demand.to_numpy() + phi
    out["d_ln_S0"] = supply.to_numpy()
    out["d_ln_P_local"] = result.d_ln_P_local
    out["d_ln_P_fundamental"] = result.d_ln_P
    out["d_ln_Q"] = result.d_ln_Q
    out["quantity_constrained"] = result.constrained
    out["excess_over_trend"] = result.d_ln_P - phi
    out["overshoot_log"] = overshoot_log
    out["d_ln_P_reported"] = d_ln_p_reported
    out["price_res_inr_sqft_fundamental"] = observed_price * np.exp(result.d_ln_P)
    out["price_res_inr_sqft_reported"] = observed_price * np.exp(d_ln_p_reported)
    out["floorspace_res_sqm_new"] = quantity * np.exp(result.d_ln_Q)

    diagnostics: dict[str, Any] = {
        "year": int(year),
        "scenario": scenario,
        "phi_t": phi,
        "eta": float(eta),
        "gamma": float(
            gamma
            if gamma is not None
            else _require_path(params, P_GAMMA_BUILT, "accessibility elasticity")
        ),
        "iterations": result.iterations,
        "converged": result.converged,
        "residual_change": result.residual_change,
        "n_constrained": int(result.constrained.sum()),
        "params_hash": getattr(params, "hash", None),
        "land_pass": bool(include_land),
    }

    if include_land:
        if eps_land_multiple is None:
            eps_land_multiple = _require_path(
                params,
                P_EPS_LAND_MULTIPLE,
                "Section 13.1's raw-land supply elasticity, which the section describes "
                "only as '`eps_land` higher'",
            )
        if eps_land_multiple < 1:
            raise ValueError(
                f"{P_EPS_LAND_MULTIPLE} must be >= 1; Section 13.1 says eps_land is "
                f"*higher* than eps, got {eps_land_multiple}"
            )
        land_demand = demand_shift(
            cells,
            params,
            d_lnA=d_lnA,
            new_hh=new_hh,
            field=field,
            gamma=gamma,
            land=True,
            cap_field=cap_field,
        )
        land_result = clear_market(
            params,
            d_ln_D_local=land_demand,
            d_ln_S0=supply,
            eta=eta,
            eps=eps * eps_land_multiple,
            quantity_sqm=quantity,
            headroom_sqm=headroom,
            absorption_cap_sqm=absorption_cap_sqm,
            phi_t=phi,
            index=index,
        )
        observed_land = (
            np.asarray(cells[COL_PRICE_LAND], dtype=float)
            if COL_PRICE_LAND in cells.columns
            else np.full(len(index), np.nan)
        )
        out["d_ln_P_land"] = land_result.d_ln_P
        out["price_land_inr_sqft_fundamental"] = observed_land * np.exp(
            land_result.d_ln_P
        )
        diagnostics["eps_land_multiple"] = float(eps_land_multiple)

    out.attrs = dict(cells.attrs)
    out.attrs[ATTR_KEY] = diagnostics
    return out
