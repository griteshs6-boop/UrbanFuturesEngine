"""Layer 4 — supply (spec Section 11).

Tracks, per cell, the supply-side state that the rest of the engine treats as exogenous
to a single year's clearing: how much floorspace could exist (`capacity_sqm`), how much
already does (`built_sqm`), the resulting slack (`headroom_sqm`), and how fast that slack
can plausibly be absorbed in a given year (the Section 11.2 absorption cap).

Section 11.1 — "state carried forward"
---------------------------------------
Section 11 describes per-cell state that persists across simulation years. This module
stays a **pure function** (CONTRACT.md rule 2): it never mutates its inputs and holds no
module-level state. Instead, the previous year's state is passed in explicitly as a
:class:`SupplyState` and the new state comes back out. Because the public entry point's
required shape is ``apply_supply(cells, params, ...) -> pd.DataFrame`` (matching every
other layer in this codebase), the new :class:`SupplyState` is attached to the returned
frame's ``.attrs["supply_state"]`` — a plain dict slot pandas reserves for exactly this
kind of frame metadata, and one that survives ``.copy()``. The returned frame's
schema-visible columns (``capacity_sqm``, ``headroom_sqm``, ``inventory_months``,
``hist_absorption_sqm``) are *also* updated in place on the copy, so a caller who only
wants to inspect this year's numbers never needs to touch ``.attrs``.

The runner threads years like this::

    state = None
    for year in years:
        out = apply_supply(cells, params, state=state, year=year, demand_sqm=demand)
        state = out.attrs["supply_state"]
        cells = out          # next year's input

Section 11.2 — absorption cap
------------------------------
``absorption_cap_sqm`` bounds how much floorspace a cell can plausibly absorb in one
year: a historical absorption rate, grown at ``supply.absorption.base_growth``, scaled by
how attractive the cell is relative to the rest of the city this year (a softmax over an
exogenous allocation utility, renormalised to a city mean of 1.0) and damped when the
cell is already carrying a lot of unsold inventory. Every rate, cap and exponent in this
formula is read from ``config/params/supply.yaml``; the only literals in this module are
``0`` and ``1`` (CONTRACT.md rule 1).

Section 11.3 — applying supply effects
----------------------------------------
``SupplyEffect`` (spec Section 9.1) is produced and owned by Layer 2
(``ufe/layers/l2_shocks.py``). This module imports the canonical dataclass from there and
re-exports it, so ``l4_supply.SupplyEffect`` and ``l2_shocks.SupplyEffect`` are literally
the same class. (An earlier revision carried a local, field-for-field-identical copy
because Layer 2 did not exist yet; that copy is gone.)

A `SupplyEffect` with `start_year == year`:

* ``delta_capacity_sqm`` (land sterilisation, land banking) is applied to `capacity_sqm`
  immediately, at `start_year` — one-off, not spread over time.
* ``delta_floorspace_sqm`` (a launched township) is *not* applied to `built_sqm`
  immediately. It is added to a per-cell backlog (`committed_backlog_sqm`, part of the
  carried-forward state) that is delivered gradually, subject to the same Section 11.2
  absorption cap as ordinary demand — this is what makes the township "deliver over its
  absorption period" rather than appearing in one year. University land banking is simply
  two effects at different years (a negative one at acquisition, a positive one at
  `start_year + release_lag`); this module applies each on its own `start_year` and does
  not need to know about the lag itself.

Ambiguities resolved here (see the build report for the full list):

* Section 11 never says which of ``demand_sqm`` or the townships' backlog gets priority
  when both exist in the same cell in the same year, nor how they share one cap. This
  module delivers backlog first (already-committed supply is not discretionary) and lets
  ordinary demand claim whatever headroom the cap leaves.
* ``inventory_months`` is produced by the demand/price side of the engine (Section 13),
  not by this layer. Section 11.1 lists it as carried-forward state, so absent an explicit
  override for the year this module simply carries the previous value forward unchanged.
* ``hist_absorption_sqm`` (RERA-derived, Section 11.2) is a slowly-changing input, not a
  quantity this layer derives; it is carried forward unchanged unless the caller supplies
  a new value for the year.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import SchemaValidationError
from ufe.layers.l2_shocks import SupplyEffect as _SupplyEffect
from ufe.params import Params

logger = __import__("logging").getLogger(__name__)

__all__ = [
    "SupplyEffect",
    "SupplyState",
    "apply_supply",
    "relative_attractiveness",
    "inventory_damping",
    "absorption_cap_sqm",
    "ATTR_KEY",
    "CARRIED_FORWARD_COLUMNS",
]

# --------------------------------------------------------------------------------------
# parameter paths (spec Section 11, config/params/supply.yaml)
# --------------------------------------------------------------------------------------

P_BASE_GROWTH = "supply.absorption.base_growth"
P_INVENTORY_SOFT_CAP = "supply.absorption.inventory_months_soft_cap"
P_INVENTORY_DAMPING_EXPONENT = "supply.absorption.inventory_damping_exponent"

#: The frame-metadata slot the new :class:`SupplyState` is attached to (see module note).
ATTR_KEY = "supply_state"

#: Schema-visible columns this layer updates on the returned frame every year.
CARRIED_FORWARD_COLUMNS: tuple[str, ...] = (
    "capacity_sqm",
    "headroom_sqm",
    "inventory_months",
    "hist_absorption_sqm",
)

COL_H3 = "h3"
COL_CAPACITY = "capacity_sqm"
COL_HEADROOM = "headroom_sqm"
COL_INVENTORY_MONTHS = "inventory_months"
COL_HIST_ABSORPTION = "hist_absorption_sqm"
COL_FLOORSPACE_RES = "floorspace_res_sqm"
COL_FLOORSPACE_COM = "floorspace_com_sqm"

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    COL_H3,
    COL_CAPACITY,
    COL_HEADROOM,
    COL_INVENTORY_MONTHS,
    COL_HIST_ABSORPTION,
    COL_FLOORSPACE_RES,
    COL_FLOORSPACE_COM,
)


# --------------------------------------------------------------------------------------
# 9.1 (consumed here) — the SupplyEffect shape Layer 2 emits
# --------------------------------------------------------------------------------------


#: Re-exported from Layer 2, which owns the canonical definition (spec Section 9.1).
#: ``ufe.layers.l4_supply.SupplyEffect`` stays a valid reference for existing callers.
#:
#: ``delta_floorspace_sqm`` positive = new supply committed (delivered gradually,
#: Section 11.3). ``delta_capacity_sqm`` negative = sterilised (applied immediately at
#: ``start_year``).
SupplyEffect = _SupplyEffect


# --------------------------------------------------------------------------------------
# 11.1 — state carried forward
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SupplyState:
    """Per-cell state carried forward across simulation years (spec Section 11.1).

    Every series is indexed by ``h3``. Immutable: :func:`apply_supply` never mutates a
    `SupplyState` in place; it always returns a new one.

    Attributes
    ----------
    built_sqm:
        Residential + commercial floorspace actually standing in the cell.
    capacity_sqm, headroom_sqm, inventory_months, hist_absorption_sqm:
        Mirror the schema-visible ``cells`` columns of the same name.
    committed_backlog_sqm:
        Section 11.3 bookkeeping: `SupplyEffect.delta_floorspace_sqm` not yet delivered
        to `built_sqm`, because the absorption cap has not yet let it through.
    base_year:
        The `t=0` reference for the Section 11.2 growth term
        ``(1 + base_growth) ** (t - base_year)``.
    """

    built_sqm: pd.Series
    capacity_sqm: pd.Series
    headroom_sqm: pd.Series
    inventory_months: pd.Series
    hist_absorption_sqm: pd.Series
    committed_backlog_sqm: pd.Series
    base_year: int

    @staticmethod
    def initial(cells: pd.DataFrame, base_year: int) -> "SupplyState":
        """Build the first year's state directly from an ingested/Layer-0 ``cells`` frame."""
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in cells.columns]
        if missing:
            raise SchemaValidationError(
                f"Layer 4 needs column(s) {', '.join(missing)} on the cells frame "
                "(spec Section 11.1, Section 3.1)"
            )
        h3_index = pd.Index(cells[COL_H3].to_numpy(), name=COL_H3)
        built = (
            cells[COL_FLOORSPACE_RES].to_numpy(dtype=float)
            + cells[COL_FLOORSPACE_COM].to_numpy(dtype=float)
        )
        zeros = pd.Series(np.zeros(len(cells)), index=h3_index)
        return SupplyState(
            built_sqm=pd.Series(built, index=h3_index),
            capacity_sqm=pd.Series(
                cells[COL_CAPACITY].to_numpy(dtype=float), index=h3_index
            ),
            headroom_sqm=pd.Series(
                cells[COL_HEADROOM].to_numpy(dtype=float), index=h3_index
            ),
            inventory_months=pd.Series(
                cells[COL_INVENTORY_MONTHS].to_numpy(dtype=float), index=h3_index
            ),
            hist_absorption_sqm=pd.Series(
                cells[COL_HIST_ABSORPTION].to_numpy(dtype=float), index=h3_index
            ),
            committed_backlog_sqm=zeros,
            base_year=base_year,
        )


# --------------------------------------------------------------------------------------
# 11.2 — absorption cap
# --------------------------------------------------------------------------------------


def relative_attractiveness(utility: np.ndarray) -> np.ndarray:
    """Softmax over cells of the allocation utility, renormalised to a city mean of 1.0.

    Spec Section 11.2: ``relative_attractiveness_i(t) = softmax over cells of the
    allocation utility, normalised so the city mean is 1.0``. A softmax already sums to
    1 across the city, i.e. its mean is ``1 / n``; multiplying by ``n`` sets the mean to
    1 without changing the relative shape.
    """
    utility = np.asarray(utility, dtype=float)
    n = utility.shape[0]
    shifted = utility - np.max(utility)
    weights = np.exp(shifted)
    softmax = weights / weights.sum()
    return softmax * n


def inventory_damping(inventory_months: np.ndarray, params: Params) -> np.ndarray:
    """``clip((soft_cap / max(inventory_months, 1)) ** exponent, 0, 1)`` (Section 11.2)."""
    soft_cap = params.value(P_INVENTORY_SOFT_CAP)
    exponent = params.value(P_INVENTORY_DAMPING_EXPONENT)
    inventory_months = np.asarray(inventory_months, dtype=float)
    denom = np.maximum(inventory_months, 1)
    return np.clip((soft_cap / denom) ** exponent, 0, 1)


def absorption_cap_sqm(
    hist_absorption_sqm: np.ndarray,
    year: int,
    base_year: int,
    rel_attractiveness: np.ndarray,
    damping: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Section 11.2's ``absorption_cap_sqm_i(t)``."""
    base_growth = params.value(P_BASE_GROWTH)
    growth_factor = (1 + base_growth) ** (year - base_year)
    cap = (
        np.asarray(hist_absorption_sqm, dtype=float)
        * growth_factor
        * np.asarray(rel_attractiveness, dtype=float)
        * np.asarray(damping, dtype=float)
    )
    return np.maximum(0, cap)


# --------------------------------------------------------------------------------------
# 11.3 — applying supply effects
# --------------------------------------------------------------------------------------


def _apply_effects(
    state: SupplyState, effects: Sequence[SupplyEffect], year: int
) -> SupplyState:
    """Apply every effect whose `start_year == year` (spec Section 11.3)."""
    capacity = state.capacity_sqm.copy()
    backlog = state.committed_backlog_sqm.copy()
    for effect in effects:
        if effect.start_year != year:
            continue
        if effect.cell not in capacity.index:
            raise SchemaValidationError(
                f"SupplyEffect names cell {effect.cell!r}, which is not on the grid"
            )
        capacity.loc[effect.cell] += effect.delta_capacity_sqm
        backlog.loc[effect.cell] += effect.delta_floorspace_sqm
    capacity = capacity.clip(lower=0)
    backlog = backlog.clip(lower=0)
    return replace(state, capacity_sqm=capacity, committed_backlog_sqm=backlog)


def _as_series(
    value: pd.Series | Mapping[str, float] | None, index: pd.Index, default: float
) -> pd.Series:
    if value is None:
        return pd.Series(np.full(len(index), default), index=index)
    if isinstance(value, pd.Series):
        return value.reindex(index, fill_value=default).astype(float)
    return pd.Series(
        [float(value.get(h3_id, default)) for h3_id in index], index=index
    )


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def apply_supply(
    cells: pd.DataFrame,
    params: Params,
    *,
    year: int,
    state: SupplyState | None = None,
    demand_sqm: pd.Series | Mapping[str, float] | None = None,
    utility: pd.Series | Mapping[str, float] | None = None,
    effects: Sequence[SupplyEffect] = (),
    inventory_months: pd.Series | Mapping[str, float] | None = None,
    hist_absorption_sqm: pd.Series | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Apply one simulation year of Layer 4 supply (spec Section 11).

    Parameters
    ----------
    cells:
        The current `cells` frame. Never mutated. Used for its `h3` column and, when
        `state` is `None`, to seed the first year's :class:`SupplyState`.
    params:
        Resolved parameter tree; every coefficient is read from `config/params/supply.yaml`.
    year:
        The calendar year being simulated.
    state:
        The previous year's :class:`SupplyState`. `None` on the first call, in which case
        it is seeded from `cells` with `base_year = year`.
    demand_sqm:
        This year's desired floorspace growth per cell (from the demand/allocation side of
        the engine), keyed by `h3`. Defaults to zero everywhere — a cell with no demand
        input never grows.
    utility:
        This year's allocation utility per cell, keyed by `h3`, feeding
        :func:`relative_attractiveness`. Defaults to a uniform utility (every cell equally
        attractive, `relative_attractiveness == 1`).
    effects:
        `SupplyEffect` instances (Section 9.1 / 11.3) to resolve this year. Only those with
        `start_year == year` do anything; effects for other years are ignored (the caller
        re-passes the same full list every year, or filters it — either works).
    inventory_months, hist_absorption_sqm:
        Explicit overrides for this year's carried-forward values (Section 11.1 lists both
        as state this layer does not itself derive). `None` carries the previous value
        forward unchanged.

    Returns
    -------
    A new frame with the same index and row count as `cells`, with `capacity_sqm`,
    `headroom_sqm`, `inventory_months` and `hist_absorption_sqm` updated for the year, and
    `.attrs["supply_state"]` holding the new :class:`SupplyState` for the runner to thread
    into next year's call.
    """
    if state is None:
        state = SupplyState.initial(cells, base_year=year)

    h3_index = state.capacity_sqm.index

    # --- 11.3: land sterilisation / land banking / township commitments -------------
    state = _apply_effects(state, effects, year)

    # --- 11.1: carry forward inventory_months / hist_absorption_sqm -----------------
    inventory = (
        state.inventory_months
        if inventory_months is None
        else _as_series(inventory_months, h3_index, default=0).combine_first(
            state.inventory_months
        )
    )
    hist_absorption = (
        state.hist_absorption_sqm
        if hist_absorption_sqm is None
        else _as_series(hist_absorption_sqm, h3_index, default=0).combine_first(
            state.hist_absorption_sqm
        )
    )

    # --- headroom ahead of this year's absorption (capacity just updated above) -----
    headroom_before = (state.capacity_sqm - state.built_sqm).clip(lower=0)

    # --- 11.2: the absorption cap -----------------------------------------------------
    utility_arr = _as_series(utility, h3_index, default=0).to_numpy()
    damping = inventory_damping(inventory.to_numpy(), params)
    rel_attract = relative_attractiveness(utility_arr)
    cap = absorption_cap_sqm(
        hist_absorption.to_numpy(),
        year,
        state.base_year,
        rel_attract,
        damping,
        params,
    )
    cap_series = pd.Series(cap, index=h3_index)

    # --- demand: committed backlog delivers first, ordinary demand fills the rest ----
    demand = _as_series(demand_sqm, h3_index, default=0)
    total_demand = demand + state.committed_backlog_sqm
    delivered = np.minimum(np.minimum(total_demand, cap_series), headroom_before)
    delivered = delivered.clip(lower=0)

    backlog_delivered = np.minimum(delivered, state.committed_backlog_sqm)
    organic_delivered = delivered - backlog_delivered
    new_backlog = (state.committed_backlog_sqm - backlog_delivered).clip(lower=0)

    new_built = state.built_sqm + delivered
    new_headroom = (state.capacity_sqm - new_built).clip(lower=0)

    new_state = SupplyState(
        built_sqm=new_built,
        capacity_sqm=state.capacity_sqm,
        headroom_sqm=new_headroom,
        inventory_months=inventory,
        hist_absorption_sqm=hist_absorption,
        committed_backlog_sqm=new_backlog,
        base_year=state.base_year,
    )

    out = cells.copy(deep=True)
    by_h3 = out[COL_H3].to_numpy()
    out[COL_CAPACITY] = new_state.capacity_sqm.reindex(by_h3).to_numpy()
    out[COL_HEADROOM] = new_state.headroom_sqm.reindex(by_h3).to_numpy()
    out[COL_INVENTORY_MONTHS] = new_state.inventory_months.reindex(by_h3).to_numpy()
    out[COL_HIST_ABSORPTION] = new_state.hist_absorption_sqm.reindex(by_h3).to_numpy()
    out.attrs[ATTR_KEY] = new_state
    # Exposed for tests/diagnostics; not part of the schema-declared cells columns.
    out.attrs["absorption_cap_sqm"] = cap_series.reindex(by_h3)
    out.attrs["delivered_sqm"] = delivered.reindex(by_h3)
    out.attrs["organic_delivered_sqm"] = organic_delivered.reindex(by_h3)
    out.attrs["backlog_delivered_sqm"] = backlog_delivered.reindex(by_h3)
    return out
