"""Module 8 — Layer 5, allocation (spec Section 12).

Public entry points: :func:`allocate` (households, Sections 12.1-12.6) and
:func:`allocate_firms` (Section 12.7).

**Simulation-time module.** No I/O, no network, no LLM, no unseeded randomness — in fact no
randomness at all: every step here is a deterministic function of `(cells, params, year)`
and the effects handed in. Monte Carlo enters this layer through `Params.sample` at
parameter-resolution time (Section 16), not through a draw inside the loop, so there is no
`rng` argument to thread.

What it does
------------
12.1  the annual loop, with the agglomeration inner loop
12.2  household demand — exogenous growth plus job-driven in-migration, by income band
12.3  the residential utility function, including the cell fixed effect ``alpha_i``
12.4  constrained allocation by iterative proportional fitting against ``headroom_sqm``
12.5  induced service employment — **emitted here and nowhere else** (Section 21)
12.6  agglomeration damping with congestion feedback, and :class:`ConvergenceError`
12.7  firm allocation for non-service, non-pinned employment

State threading (the runner MUST do this)
-----------------------------------------
Like Layer 4, this layer is pure and returns the state it wants back next year on the
frame's ``.attrs``::

    state = None
    for year in years:
        out = allocate(cells, params, year=year, state=state, ...)
        state = out.attrs[l5_allocation.ATTR_STATE]
        cells = out

:class:`AllocationState` carries the calibrated :class:`AlphaFit`. Section 12.3 is explicit
that ``alpha_i`` is estimated **once**, in the base year, and held fixed: re-estimating it
every year would silently re-fit the model to its own output and make the null test
meaningless. If ``state`` is ``None`` and no ``alpha`` is supplied, this module calibrates
``alpha_i`` from the frame it is given and records the fit in the returned state.

The cell fixed effect (Section 12.3, Section 21)
------------------------------------------------
"Without this the model will relocate the entire city on day one." ``alpha_i`` is estimated
by inverting the observed base-year distribution:

    alpha_ik = ln(observed_share_ik) - ln(predicted_share_ik),  centred to mean 0 per band

Section 12.3 prints ``ln(observed_households_ik / predicted_share_ik)``, mixing a count with
a share; the two differ by ``ln(N_k)``, a per-band constant that the centring removes and
that the softmax would cancel anyway. Shares are used here so the expression is
dimensionally coherent.

The estimate is per cell **and band** — the residual it absorbs is band-specific (a cell can
be under-predicted for low-income households and over-predicted for high-income ones), and
only the band-specific form makes the allocation an exact fixed point of the observed
distribution, which is what the Section 12 null test asserts. The landed ``cells`` schema
has a single ``alpha_res`` column, so the household-weighted mean across bands is written
there and the full band matrix lives on :class:`AlphaFit` / :class:`AllocationState`.

Numeric policy
--------------
Every coefficient, elasticity, damping factor, tolerance and iteration cap is read from
``config/params/`` through :class:`~ufe.params.Params`. The only literals in this module are
``0`` and ``1`` and array indices (CONTRACT.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax

from ufe.errors import ConvergenceError, MissingParameter, UFEError
from ufe.layers.routing import MatrixSet, congestion_adjust
from ufe.store.schemas import INCOME_BANDS, SECTORS, IncomeBand, Sector
from ufe.store.schemas import income_band_boundaries as _shared_income_band_boundaries

logger = logging.getLogger(__name__)

__all__ = [
    "ATTR_STATE",
    "ATTR_DIAGNOSTICS",
    "COEFFICIENTS",
    "PURPOSES",
    "AlphaFit",
    "AllocationState",
    "HouseholdDemand",
    "allocate",
    "allocate_constrained",
    "allocate_firms",
    "band_accessibility",
    "choice_shares",
    "estimate_alpha_res",
    "household_demand",
    "income_band_boundaries",
    "induced_service_jobs",
    "utility",
    "wage_band",
]

ZERO, ONE = 0, 1

# --------------------------------------------------------------------------------------
# parameter paths — every number in this module comes from one of these
# --------------------------------------------------------------------------------------

BEHAVIOUR = "behaviour"
ACCESSIBILITY = "accessibility"
CASCADE = "cascade"
ARCHETYPES = "archetypes"

#: Section 12.3 utility coefficients, per income band.
P_LOGIT = f"{BEHAVIOUR}.logit"
#: Section 12.3 band-specific mode weights for `lnA_ik`.
P_ACCESS_MODE_WEIGHTS = "access_mode_weights"

P_WORKERS_PER_HOUSEHOLD = f"{BEHAVIOUR}.workers_per_household"
P_WAGE_PREMIUM = f"{BEHAVIOUR}.wage_to_band.household_wage_premium"
P_BAND_BOUNDARIES = f"{BEHAVIOUR}.income_bands.boundaries_inr_mo"
P_BAND_BASE_YEAR = f"{BEHAVIOUR}.income_bands.base_year"
P_BAND_INDEXING = f"{BEHAVIOUR}.income_bands.index_to_nominal_income"
P_INMIGRANT_SHARE = f"{BEHAVIOUR}.migration.inmigrant_share_by_sector"
P_NATURAL_GROWTH_RATE = f"{BEHAVIOUR}.natural_growth_rate"

P_SQM_PER_HH = f"{BEHAVIOUR}.sqm_per_hh_by_band"
P_PERSONS_PER_HH = f"{BEHAVIOUR}.persons_per_household_by_band"
P_SERVICE_JOBS_PER_RESIDENT = f"{BEHAVIOUR}.service_jobs_per_resident"
P_DORM_SERVICE_FACTOR = f"{BEHAVIOUR}.dorm_service_factor"

P_SPILLOVER_PHI = f"{BEHAVIOUR}.agglomeration.spillover_phi"
P_MAX_ITERATIONS = f"{BEHAVIOUR}.agglomeration.max_iterations"
P_CONVERGENCE_TOL = f"{BEHAVIOUR}.agglomeration.convergence_tol"
P_ALLOCATION_MAX_ITER = f"{BEHAVIOUR}.allocation.max_iter"
P_NULL_TEST_TOLERANCE = f"{BEHAVIOUR}.allocation.null_test_tolerance"

P_DEFAULT_RAMP_YEARS = f"{ARCHETYPES}._defaults.operational_ramp_years"

P_PURPOSE_WEIGHT = f"{ACCESSIBILITY}.purposes"
P_DECAY_BETA = f"{ACCESSIBILITY}.decay_beta"

P_FIRM_LOGIT = f"{CASCADE}.firm_logit"
P_FIRM_COEFFICIENTS = f"{P_FIRM_LOGIT}.coefficients"

#: Section 12.3 coefficient names, in the order the utility function prints them.
COEFFICIENTS: tuple[str, ...] = (
    "b_access",
    "b_price",
    "b_amenity",
    "b_disamenity",
    "b_agglom",
    "b_same_band",
)

#: Trip purposes, mirroring `ufe.layers.l1_accessibility.PURPOSES`.
PURPOSES: tuple[str, ...] = ("work", "retail", "education", "health")

#: `decay_beta.work` is tabulated for car/two_wheeler/transit/walk; metro is treated as
#: transit, exactly as Layer 1 does.
_BETA_MODE_ALIAS: Mapping[str, str] = {"metro": "transit"}

#: Section 12.7 firm-logit coefficient names.
FIRM_COEFFICIENTS: tuple[str, ...] = (
    "c_market",
    "c_labour",
    "c_land",
    "c_agglom",
    "c_freight",
)

#: Sectors for which the Section 12.7 `c_freight` term applies ("industrial only").
INDUSTRIAL_SECTORS: tuple[str, ...] = ("manuf_heavy", "manuf_light", "logistics")

#: Frame-metadata slots (see the module note on state threading).
ATTR_STATE = "allocation_state"
ATTR_DIAGNOSTICS = "allocation"

#: The sector induced service employment is booked to (Section 12.5).
SERVICE_SECTOR = Sector.retail_svc

COL_H3 = "h3"
COL_HOUSEHOLDS = "households"
COL_HH_BY_BAND = "hh_by_band"
COL_JOBS = "jobs_by_sector"
COL_POPULATION = "population"
COL_PRICE_RES = "price_res_inr_sqft"
COL_PRICE_LAND = "price_land_inr_sqft"
COL_AMENITY = "amenity"
COL_DISAMENITY = "disamenity"
COL_HEADROOM = "headroom_sqm"
COL_CAPACITY = "capacity_sqm"
COL_BUILTUP = "builtup_frac"
COL_LNA = "lnA"
COL_ALPHA = "alpha_res"

#: Columns :func:`utility` reads.
UTILITY_COLUMNS: tuple[str, ...] = (
    COL_PRICE_RES,
    COL_AMENITY,
    COL_DISAMENITY,
    COL_HOUSEHOLDS,
    COL_HH_BY_BAND,
)

#: Columns :func:`allocate` reads on top of those.
REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_H3,
    COL_JOBS,
    COL_POPULATION,
    COL_HEADROOM,
) + UTILITY_COLUMNS


# --------------------------------------------------------------------------------------
# a minimal structural view of the Section 9.1 EmploymentEffect
# --------------------------------------------------------------------------------------


class EmploymentEffectLike(Protocol):
    """The attributes this layer reads off a Section 9.1 ``EmploymentEffect``.

    ``ufe/layers/l2_shocks.py`` owns the real dataclass and is being written concurrently, so
    effects are consumed structurally (by attribute) and never by ``isinstance``. Nothing
    here imports Layer 2.
    """

    cell: str
    sector: int
    jobs: float
    median_wage_inr_mo: float
    start_year: int
    ramp_years: int
    dormitory_share: float


# --------------------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaFit:
    """The calibrated cell fixed effect (Section 12.3).

    Attributes
    ----------
    by_band:
        ``h3``-indexed frame, one column per income band. This is the object the model
        actually uses.
    per_cell:
        ``h3``-indexed household-weighted mean across bands, centred to mean 0. Written to
        the schema's single ``alpha_res`` column.
    base_year:
        The calibration year. Recorded so a run can prove the fit was not re-estimated.
    """

    by_band: pd.DataFrame
    per_cell: pd.Series
    base_year: int


@dataclass(frozen=True)
class AllocationState:
    """Per-run state carried forward across simulation years."""

    alpha: AlphaFit
    base_year: int
    cumulative_spill_by_band: np.ndarray = field(
        default_factory=lambda: np.zeros(len(INCOME_BANDS))
    )


@dataclass(frozen=True)
class HouseholdDemand:
    """Section 12.2 output."""

    by_band: np.ndarray
    exogenous: float
    job_driven: float
    dormitory_workers: pd.Series

    @property
    def total(self) -> float:
        return float(self.by_band.sum())


# --------------------------------------------------------------------------------------
# small parameter helpers
# --------------------------------------------------------------------------------------


def _coefficient(params: Any, band: str, name: str) -> float:
    return float(params.value(f"{P_LOGIT}.{band}.{name}"))


def _mode_weight(params: Any, band: str, mode: str) -> float:
    """Band-specific access mode weight (Section 12.3).

    A mode the band's table does not list contributes zero, matching Layer 1's documented
    treatment of a mode with no matrix: accessibility stays monotone in the network.
    """
    try:
        return float(params.value(f"{P_LOGIT}.{band}.{P_ACCESS_MODE_WEIGHTS}.{mode}"))
    except MissingParameter:
        logger.debug("band %s has no access_mode_weight for mode %s; contributing 0", band, mode)
        return float(ZERO)


def _purpose_weight(params: Any, purpose: str) -> float:
    return float(params.value(f"{P_PURPOSE_WEIGHT}.{purpose}.weight"))


def _beta(params: Any, purpose: str, mode: str) -> float:
    for candidate in (mode, _BETA_MODE_ALIAS.get(mode), "default"):
        if candidate is None:
            continue
        try:
            return float(params.value(f"{P_DECAY_BETA}.{purpose}.{candidate}"))
        except MissingParameter:
            continue
    raise MissingParameter(
        f"no decay beta for purpose {purpose!r} and mode {mode!r}: tried "
        f"{P_DECAY_BETA}.{purpose}.{{{mode},default}}"
    )


def income_band_boundaries(params: Any) -> list[float]:
    """The Section 3.7 monthly-income band boundaries.

    Reading the boundaries is :func:`ufe.store.schemas.income_band_boundaries`' job — it
    handles both the bare-scalar form the spec prints and the Section 4.1 leaf form the
    landed ``behaviour.yaml`` uses. This wrapper adds only the Section 3.7 indexing guard.

    ``income_bands.index_to_nominal_income`` is ``true`` and Section 3.7 requires the
    boundaries to be inflation-indexed to the base year. No inflation index exists anywhere
    in ``config/params/``, and the boundaries' own ``income_bands.base_year`` equals the
    city's ``base_year``, so the indexing factor is 1 for Vizag today. That is asserted, not
    assumed: a mismatch raises rather than quietly deflating every household into the wrong
    band.
    """
    boundaries = _shared_income_band_boundaries(params)

    if bool(params.get(P_BAND_INDEXING)):
        boundary_base = int(params.value(P_BAND_BASE_YEAR))
        city_base = int(params.city_config.get("base_year", boundary_base))
        if boundary_base != city_base:
            raise MissingParameter(
                f"{P_BAND_INDEXING} is true and {P_BAND_BASE_YEAR}={boundary_base} differs "
                f"from the city base_year={city_base}, but no inflation index exists in "
                "config/params/ to index the boundaries with (Section 3.7)"
            )
    return boundaries


def sqm_per_household(params: Any) -> np.ndarray:
    """``sqm_per_hh_by_band`` as an array in :data:`INCOME_BANDS` order (Section 12.4)."""
    return np.array(
        [float(params.value(f"{P_SQM_PER_HH}.{band}")) for band in INCOME_BANDS], dtype=float
    )


def persons_per_household(
    params: Any, override: Mapping[str, float] | Sequence[float] | None = None
) -> np.ndarray:
    """Persons per household by band (Section 12.5).

    ``behaviour.persons_per_household_by_band`` is **null on disk** — the supplied spec never
    gives values, and ``behaviour.yaml`` says so explicitly ("Module 8 / Module 5 must raise
    rather than substitute a default"). This raises :class:`MissingParameter` naming the
    exact paths unless the caller supplies the values.
    """
    if override is not None:
        if isinstance(override, Mapping):
            missing = [b for b in INCOME_BANDS if b not in override]
            if missing:
                raise MissingParameter(
                    f"persons_per_household_by_band override is missing band(s) {missing}"
                )
            return np.array([float(override[b]) for b in INCOME_BANDS], dtype=float)
        values = np.asarray(override, dtype=float)
        if values.shape != (len(INCOME_BANDS),):
            raise MissingParameter(
                "persons_per_household_by_band override must have one value per income band"
            )
        return values

    values = []
    for band in INCOME_BANDS:
        path = f"{P_PERSONS_PER_HH}.band_{band}"
        value = params.get(path)
        if value is None:
            raise MissingParameter(
                f"{path} is null in config/params/behaviour.yaml. Section 12.5 needs persons "
                "per household by band and the spec never supplies it; pass "
                "persons_per_household_by_band= explicitly or fit the parameter. This module "
                "will not substitute a default."
            )
        values.append(float(value["value"] if isinstance(value, Mapping) else value))
    return np.array(values, dtype=float)


# --------------------------------------------------------------------------------------
# frame helpers
# --------------------------------------------------------------------------------------


def _require(cells: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in cells.columns]
    if missing:
        raise UFEError(
            f"Layer 5 needs column(s) {missing} on the cells frame (spec Sections 3.1, 12.3)"
        )


def _matrix(cells: pd.DataFrame, column: str, width: int) -> np.ndarray:
    values = np.vstack(
        cells[column].map(lambda v: np.asarray(v, dtype=float)).to_numpy()
    ) if len(cells) else np.zeros((ZERO, width))
    if values.shape[ONE] != width:
        raise UFEError(f"{column} rows must have length {width}")
    return values


def _as_array(
    value: pd.Series | Mapping[str, float] | Sequence[float] | float | None,
    cells: pd.DataFrame,
    default: float,
) -> np.ndarray:
    n = len(cells)
    if value is None:
        return np.full(n, float(default))
    if isinstance(value, pd.Series):
        if value.index.equals(cells.index):
            return value.to_numpy(dtype=float)
        return value.reindex(cells[COL_H3].to_numpy()).fillna(default).to_numpy(dtype=float)
    if isinstance(value, Mapping):
        return np.array(
            [float(value.get(h, default)) for h in cells[COL_H3].to_numpy()], dtype=float
        )
    arr = np.asarray(value, dtype=float)
    if arr.ndim == ZERO:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise UFEError(f"expected {n} values, got {arr.shape}")
    return arr


def _positive_or_median(values: np.ndarray, what: str) -> np.ndarray:
    """Replace missing / non-positive values with the city median of the observed ones.

    Prices are nullable in the landed schema, and ``ln(price)`` is undefined for a missing or
    zero price. Imputing the city median is the least-informative choice that keeps the cell
    in the choice set; dropping the cell would silently shrink the city.
    """
    out = np.asarray(values, dtype=float).copy()
    good = np.isfinite(out) & (out > ZERO)
    if not good.any():
        raise UFEError(f"no usable {what} anywhere in the city; cannot form ln({what})")
    if not good.all():
        median = float(np.median(out[good]))
        logger.info("imputing city-median %s for %d cell(s)", what, int((~good).sum()))
        out[~good] = median
    return out


def _softmax(values: np.ndarray) -> np.ndarray:
    """Column-wise softmax over cells. ``-inf`` utilities get exactly zero share.

    ``scipy.special.softmax`` is used rather than a hand-rolled ``exp``/normalise so the
    shift stabilisation is the library's, not ours. (``xlogit`` is an *estimator* for mixed
    logit from observed choice data; Section 12.3's coefficients are given in YAML and there
    is no choice dataset to estimate from, so it has no role here.) A column with no finite
    utility at all — every cell ineligible — would make the shift ``-inf - -inf``; those
    columns are returned as zeros instead.
    """
    values = np.asarray(values, dtype=float)
    if values.size == ZERO:
        return values
    values = np.where(np.isfinite(values), values, -np.inf)
    out = np.zeros_like(values)
    usable = np.isfinite(values).any(axis=ZERO)
    if usable.any():
        out[:, usable] = scipy_softmax(values[:, usable], axis=ZERO)
    return out


# --------------------------------------------------------------------------------------
# 12.3 — band-specific accessibility
# --------------------------------------------------------------------------------------


def _opportunities(cells: pd.DataFrame, params: Any, destinations: Sequence[str]) -> dict:
    from ufe.layers.l1_accessibility import opportunities  # local import: no import cycle

    return opportunities(cells, params, destinations)


def _congestion_ratio(
    cells: pd.DataFrame,
    params: Any,
    matrices: MatrixSet,
    base_builtup: np.ndarray,
    new_builtup: np.ndarray,
) -> np.ndarray:
    """`t_new / t_base` from the Section 8.3 congestion law, as `builtup_frac` rises.

    Section 12.6: "as ``builtup_frac`` rises, arterial speeds fall (Section 8.3), which
    reduces ``lnA``. This is the physical mechanism that makes the loop converge; it must be
    active inside the inner loop."

    The matrices handed to this layer already carry the *base* congestion adjustment, so what
    is applied here is the incremental factor
    ``max(floor, 1 - k*corridor_base) / max(floor, 1 - k*corridor_new) >= 1``. It is obtained
    by calling the landed :func:`ufe.layers.routing.congestion_adjust` on a matrix of ones
    twice, so the functional form and its parameters (``accessibility.congestion.k`` and
    ``.floor``) are read from exactly one place.

    Approximation, reported: Layer 1 takes ``corridor_builtup`` as the mean of origin,
    destination and the cell nearest the straight-line midpoint. The midpoint lookup is
    private to ``routing``, so the corridor here is the mean of origin, destination and their
    arithmetic mean — algebraically ``(b_o + b_d) / 2``. Because the *same* approximation is
    used for numerator and denominator, the incremental factor is unaffected by the choice
    except through second-order curvature at the ``floor``.
    """
    o_index = pd.Index(matrices.origins)
    by_h3 = pd.Series(np.asarray(base_builtup, dtype=float), index=cells[COL_H3].to_numpy())
    by_h3_new = pd.Series(np.asarray(new_builtup, dtype=float), index=cells[COL_H3].to_numpy())

    o_base = by_h3.reindex(o_index).fillna(ZERO).to_numpy(dtype=float)
    o_new = by_h3_new.reindex(o_index).fillna(ZERO).to_numpy(dtype=float)

    parents = cells["h3_res8"].to_numpy()
    d_base = (
        pd.Series(np.asarray(base_builtup, dtype=float), index=parents)
        .groupby(level=ZERO)
        .mean()
        .reindex(matrices.destinations)
        .fillna(ZERO)
        .to_numpy(dtype=float)
    )
    d_new = (
        pd.Series(np.asarray(new_builtup, dtype=float), index=parents)
        .groupby(level=ZERO)
        .mean()
        .reindex(matrices.destinations)
        .fillna(ZERO)
        .to_numpy(dtype=float)
    )

    halves = len(("origin", "destination"))
    ones = np.ones((len(o_base), len(d_base)), dtype=np.float32)
    mid_base = np.add.outer(o_base, d_base) / halves
    mid_new = np.add.outer(o_new, d_new) / halves

    inv_base = congestion_adjust(ones, o_base, d_base, mid_base, params)
    inv_new = congestion_adjust(ones, o_new, d_new, mid_new, params)
    return np.asarray(inv_new, dtype=float) / np.asarray(inv_base, dtype=float)


def band_accessibility(
    cells: pd.DataFrame,
    params: Any,
    matrices: MatrixSet,
    *,
    base_builtup: pd.Series | np.ndarray | None = None,
    congestion_builtup: pd.Series | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """`lnA_ik` using the **band-specific** ``access_mode_weights`` (Section 12.3).

    "A low-income household's accessibility is walk-and-bus accessibility." Layer 1's
    ``lnA`` column combines modes with the *city* mode split, which is the wrong surface for
    a band-specific logit, so the gravity sum is re-combined here with each band's weights.
    Everything else — the opportunity definitions, the decay betas, the purpose weights, the
    ``ln(A + 1)`` combination — is Layer 1's, read from the same parameter paths and, for the
    opportunities, from Layer 1's own public function.

    Returns one array per band, aligned to ``cells`` rows, with ``NaN`` for cells that are
    not in ``matrices.origins`` — a cell with no computed accessibility is *missing*, not
    inaccessible, and :func:`allocate` treats it as ineligible rather than as unattractive.
    """
    opportunity = _opportunities(cells, params, matrices.destinations)
    modes = tuple(sorted(matrices.minutes))
    if not modes:
        raise UFEError("MatrixSet carries no travel-time matrices")

    ratio = None
    if congestion_builtup is not None:
        base = (
            cells[COL_BUILTUP].to_numpy(dtype=float)
            if base_builtup is None
            else _as_array(base_builtup, cells, ZERO)
        )
        ratio = _congestion_ratio(
            cells, params, matrices, base, _as_array(congestion_builtup, cells, ZERO)
        )

    # A^p_m for every (purpose, mode) — the expensive part, done once per call.
    by_purpose_mode: dict[tuple[str, str], np.ndarray] = {}
    for mode in modes:
        minutes = np.asarray(matrices.minutes[mode], dtype=float)
        if ratio is not None:
            minutes = minutes * ratio
        finite = np.isfinite(minutes)
        for purpose in PURPOSES:
            beta = _beta(params, purpose, mode)
            with np.errstate(over="ignore"):
                decay = np.where(finite, np.exp(-beta * np.where(finite, minutes, ZERO)), ZERO)
            by_purpose_mode[(purpose, mode)] = decay @ opportunity[purpose]

    index = pd.Index(cells[COL_H3].to_numpy(), name=COL_H3)
    origins = pd.Index(matrices.origins, name=COL_H3)

    out: dict[str, np.ndarray] = {}
    for band in INCOME_BANDS:
        lnA = np.zeros(len(origins), dtype=float)
        for purpose in PURPOSES:
            total = np.zeros(len(origins), dtype=float)
            for mode in modes:
                total = total + _mode_weight(params, band, mode) * by_purpose_mode[(purpose, mode)]
            lnA = lnA + _purpose_weight(params, purpose) * np.log(total + ONE)
        out[band] = pd.Series(lnA, index=origins).reindex(index).to_numpy(dtype=float)
    return out


# --------------------------------------------------------------------------------------
# 12.3 — the utility function
# --------------------------------------------------------------------------------------


def _alpha_frame(
    alpha: AlphaFit | pd.DataFrame | pd.Series | Mapping[str, float] | float | None,
    cells: pd.DataFrame,
) -> np.ndarray:
    """Resolve any accepted `alpha` shape to an `(n_cells, n_bands)` array."""
    n, k = len(cells), len(INCOME_BANDS)
    if alpha is None:
        return np.zeros((n, k))
    if isinstance(alpha, AlphaFit):
        alpha = alpha.by_band
    if isinstance(alpha, pd.DataFrame):
        missing = [b for b in INCOME_BANDS if b not in alpha.columns]
        if missing:
            raise UFEError(f"alpha frame is missing band column(s) {missing}")
        if alpha.index.equals(cells.index):
            return alpha[list(INCOME_BANDS)].to_numpy(dtype=float)
        return (
            alpha[list(INCOME_BANDS)]
            .reindex(cells[COL_H3].to_numpy())
            .fillna(ZERO)
            .to_numpy(dtype=float)
        )
    return np.repeat(_as_array(alpha, cells, ZERO)[:, None], k, axis=ONE)


def utility(
    cells: pd.DataFrame,
    params: Any,
    *,
    lnA_by_band: Mapping[str, Sequence[float]] | None = None,
    alpha: AlphaFit | pd.DataFrame | pd.Series | Mapping[str, float] | float | None = None,
    field_res: pd.Series | Mapping[str, float] | Sequence[float] | float | None = None,
) -> pd.DataFrame:
    """Section 12.3 residential utility, one column per income band.

    ``U_ik = b_access_k lnA_ik + b_price_k ln(price_i) + b_amenity_k amenity_i
    - b_disamenity_k disamenity_i + b_agglom_k ln(hh_i + 1)
    + b_same_band_k share_of_band_k_in_cell_i + field_i(residential) + alpha_ik``

    ``field_res`` is the Section 9.4 combined, capped residential field, produced by Layer 2
    and handed in; this layer never resolves projects itself.

    ``lnA_by_band`` defaults to the frame's Layer 1 ``lnA`` column for every band. That is
    the *city* mode split, not the band-specific surface Section 12.3 asks for; use
    :func:`band_accessibility` (or pass ``matrices=`` to :func:`allocate`) to get the real
    thing. Cells whose ``lnA`` is ``NaN`` come back with ``NaN`` utility and are excluded
    downstream.
    """
    _require(cells, UTILITY_COLUMNS)
    n = len(cells)

    if lnA_by_band is None:
        _require(cells, (COL_LNA,))
        base = cells[COL_LNA].to_numpy(dtype=float)
        lnA_by_band = {band: base for band in INCOME_BANDS}

    price = _positive_or_median(cells[COL_PRICE_RES].to_numpy(dtype=float), COL_PRICE_RES)
    ln_price = np.log(price)
    amenity = cells[COL_AMENITY].fillna(ZERO).to_numpy(dtype=float)
    disamenity = cells[COL_DISAMENITY].fillna(ZERO).to_numpy(dtype=float)
    households = cells[COL_HOUSEHOLDS].to_numpy(dtype=float)
    ln_households = np.log(households + ONE)

    bands = _matrix(cells, COL_HH_BY_BAND, len(INCOME_BANDS))
    with np.errstate(invalid="ignore", divide="ignore"):
        band_share = np.divide(
            bands, households[:, None], out=np.zeros_like(bands), where=households[:, None] > ZERO
        )

    field = _as_array(field_res, cells, ZERO)
    alpha_arr = _alpha_frame(alpha, cells)

    columns: dict[str, np.ndarray] = {}
    for k, band in enumerate(INCOME_BANDS):
        lnA = np.asarray(lnA_by_band[band], dtype=float)
        if lnA.shape != (n,):
            raise UFEError(f"lnA_by_band[{band!r}] has shape {lnA.shape}, expected {(n,)}")
        b = {name: _coefficient(params, band, name) for name in COEFFICIENTS}
        columns[band] = (
            b["b_access"] * lnA
            + b["b_price"] * ln_price
            + b["b_amenity"] * amenity
            - b["b_disamenity"] * disamenity
            + b["b_agglom"] * ln_households
            + b["b_same_band"] * band_share[:, k]
            + field
            + alpha_arr[:, k]
        )
    return pd.DataFrame(columns, index=cells.index)


def choice_shares(
    utility_frame: pd.DataFrame, eligible: np.ndarray | None = None
) -> pd.DataFrame:
    """Softmax of the utility over cells, per band. Columns sum to 1."""
    values = utility_frame.to_numpy(dtype=float).copy()
    mask = np.isfinite(values).all(axis=ONE)
    if eligible is not None:
        mask = mask & np.asarray(eligible, dtype=bool)
    values[~mask, :] = -np.inf
    return pd.DataFrame(
        _softmax(values), index=utility_frame.index, columns=utility_frame.columns
    )


# --------------------------------------------------------------------------------------
# 12.3 — the cell fixed effect
# --------------------------------------------------------------------------------------


def estimate_alpha_res(
    cells: pd.DataFrame,
    params: Any,
    *,
    lnA_by_band: Mapping[str, Sequence[float]] | None = None,
    field_res: pd.Series | Mapping[str, float] | Sequence[float] | float | None = None,
    base_year: int | None = None,
) -> AlphaFit:
    """Invert the observed base-year distribution to get ``alpha_i`` (Section 12.3).

    "It absorbs everything the model does not observe (schools, views, social networks,
    reputation) ... Without this the model will relocate the entire city on day one."

    A cell with zero observed households of band ``k`` would give ``ln(0)``. Rather than
    picking a floor out of the air, such a cell is given the *minimum finite* ``alpha`` of
    that band, i.e. it is made as unattractive as the least attractive cell that actually
    houses someone. That is data-derived and keeps the fit finite.
    """
    _require(cells, UTILITY_COLUMNS)
    base = utility(cells, params, lnA_by_band=lnA_by_band, alpha=None, field_res=field_res)
    eligible = np.isfinite(base.to_numpy(dtype=float)).all(axis=ONE)
    predicted = choice_shares(base, eligible).to_numpy(dtype=float)

    observed = _matrix(cells, COL_HH_BY_BAND, len(INCOME_BANDS))
    totals = observed.sum(axis=ZERO)

    alpha = np.zeros_like(observed)
    for k, band in enumerate(INCOME_BANDS):
        share = np.divide(
            observed[:, k], totals[k], out=np.zeros(len(cells)), where=totals[k] > ZERO
        )
        usable = eligible & (share > ZERO) & (predicted[:, k] > ZERO)
        column = np.full(len(cells), np.nan)
        column[usable] = np.log(share[usable]) - np.log(predicted[usable, k])
        if not usable.any():
            raise UFEError(
                f"cannot calibrate alpha_res for band {band!r}: no cell has both observed "
                "households and a positive predicted share"
            )
        column[~usable] = np.nanmin(column)
        alpha[:, k] = column - column[eligible].mean() if eligible.any() else column

    by_band = pd.DataFrame(alpha, columns=list(INCOME_BANDS), index=cells[COL_H3].to_numpy())
    by_band.index.name = COL_H3

    weights = totals / totals.sum() if totals.sum() > ZERO else np.full(len(INCOME_BANDS), ONE)
    per_cell = pd.Series(alpha @ weights, index=by_band.index, name=COL_ALPHA)
    per_cell = per_cell - per_cell.mean()

    return AlphaFit(
        by_band=by_band,
        per_cell=per_cell,
        base_year=int(base_year) if base_year is not None else ZERO,
    )


# --------------------------------------------------------------------------------------
# 12.2 — household demand
# --------------------------------------------------------------------------------------


def wage_band(wage_inr_mo: float, params: Any) -> int:
    """Section 9.5 wage -> income band.

    ``household_income = wage * workers_per_household * household_wage_premium``, then
    ``digitize`` against the indexed band boundaries.
    """
    income = (
        float(wage_inr_mo)
        * float(params.value(P_WORKERS_PER_HOUSEHOLD))
        * float(params.value(P_WAGE_PREMIUM))
    )
    return int(np.digitize(income, income_band_boundaries(params)))


def _ramp(year: int, start_year: int, ramp_years: float) -> float:
    """``ramp(k) = clip(k / operational_ramp_years, 0, 1)`` (Section 10.1, used by 12.2)."""
    k = year - int(start_year)
    if k <= ZERO:
        return float(ZERO)
    if ramp_years <= ZERO:
        return float(ONE)
    return float(min(ONE, k / ramp_years))


def _sector_name(sector: int | str) -> str:
    if isinstance(sector, str):
        if sector not in SECTORS:
            raise UFEError(f"unknown sector {sector!r}")
        return sector
    return SECTORS[int(sector)]


def household_demand(
    cells: pd.DataFrame,
    params: Any,
    *,
    year: int,
    employment_effects: Sequence[EmploymentEffectLike] = (),
    activation_weights: Sequence[float] | None = None,
    natural_growth_rate: float | None = None,
) -> HouseholdDemand:
    """Section 12.2 — this year's new households, by income band.

    ``exogenous_new_hh(t) = city_households(t-1) * natural_growth_rate``, split across bands
    in the current city distribution.

    ``job_driven_new_hh(t) = sum over active EmploymentEffects of
    jobs_active(t) * inmigrant_share[sector] * (1 - dormitory_share) / workers_per_household``
    with ``jobs_active(t) = jobs * w_project(t) * ramp(t - start_year)``.

    ``w_project(t)`` is Layer 3's activation weight; it is an *input* here
    (``activation_weights``, one per effect, defaulting to 1) because Layer 5 must not
    re-derive credibility.

    **Double counting.** ``behaviour.natural_growth_rate`` is documented in the YAML as
    "natural increase plus baseline migration" — Section 12.2's second option — so it is net
    of the job-driven component computed here and the two are simply added.

    Dormitory workers are counted separately and never become households (Section 9.5,
    Section 21: "Dormitory workers as apartment buyers").
    """
    _require(cells, (COL_H3, COL_HOUSEHOLDS, COL_HH_BY_BAND))
    n_bands = len(INCOME_BANDS)

    rate = (
        float(params.value(P_NATURAL_GROWTH_RATE))
        if natural_growth_rate is None
        else float(natural_growth_rate)
    )
    city_households = float(cells[COL_HOUSEHOLDS].sum())
    exogenous = city_households * rate

    observed = _matrix(cells, COL_HH_BY_BAND, n_bands).sum(axis=ZERO)
    distribution = (
        observed / observed.sum()
        if observed.sum() > ZERO
        else np.full(n_bands, ONE / n_bands)
    )
    by_band = exogenous * distribution

    if activation_weights is None:
        weights = np.ones(len(employment_effects))
    else:
        weights = np.asarray(activation_weights, dtype=float)
        if weights.shape != (len(employment_effects),):
            raise UFEError(
                "activation_weights must have one weight per employment effect "
                f"({len(employment_effects)}), got {weights.shape}"
            )

    wph = float(params.value(P_WORKERS_PER_HOUSEHOLD))
    if wph <= ZERO:
        raise UFEError(f"{P_WORKERS_PER_HOUSEHOLD} must be positive")
    default_ramp = float(params.value(P_DEFAULT_RAMP_YEARS))

    dormitory = pd.Series(np.zeros(len(cells)), index=cells.index)
    row_of = {h: i for i, h in enumerate(cells[COL_H3].to_numpy())}
    job_driven = float(ZERO)

    for effect, weight in zip(employment_effects, weights):
        ramp_years = float(getattr(effect, "ramp_years", ZERO) or default_ramp)
        active = float(effect.jobs) * float(weight) * _ramp(year, effect.start_year, ramp_years)
        if active <= ZERO:
            continue
        dorm_share = float(getattr(effect, "dormitory_share", ZERO) or ZERO)
        sector = _sector_name(effect.sector)
        share = float(params.value(f"{P_INMIGRANT_SHARE}.{sector}"))

        households = active * share * (ONE - dorm_share) / wph
        job_driven += households
        band = wage_band(float(effect.median_wage_inr_mo), params)
        by_band[band] += households

        row = row_of.get(str(effect.cell))
        if row is None:
            logger.warning("employment effect on unknown cell %s ignored", effect.cell)
            continue
        dormitory.iloc[row] += active * dorm_share

    return HouseholdDemand(
        by_band=by_band,
        exogenous=exogenous,
        job_driven=job_driven,
        dormitory_workers=dormitory,
    )


# --------------------------------------------------------------------------------------
# 12.4 — constrained allocation
# --------------------------------------------------------------------------------------


def allocate_constrained(
    utility_frame: pd.DataFrame,
    demand_by_band: Sequence[float],
    headroom_sqm: Sequence[float],
    sqm_per_hh_by_band: Sequence[float],
    params: Any,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Section 12.4 iterative proportional fitting against the Layer 4 capacity constraint.

    Returns ``(allocated, spill_by_band, capped)``:

    * ``allocated`` — ``(n_cells, n_bands)`` households placed;
    * ``spill_by_band`` — households that could not be housed. **Spill is a finding, not an
      error** (Section 12.4): the city is supply-constrained and prices will spike. It is
      returned for the price-clearing layer, and logged.
    * ``capped`` — the cells that hit ``headroom_sqm``.

    Departures from the printed pseudocode, all bug fixes rather than design changes: the
    printed loop never decrements ``remaining`` when nothing overflows (so it would allocate
    the demand once per iteration), and ``scale`` is applied before ``excess`` is derived
    from it. Both are corrected here; the algorithm — softmax over the still-active cells,
    cap the overflowing ones, return their excess to the pool, deactivate them — is exactly
    as written.
    """
    values = utility_frame.to_numpy(dtype=float)
    n, k = values.shape
    sqm = np.asarray(sqm_per_hh_by_band, dtype=float)
    headroom = np.asarray(headroom_sqm, dtype=float)
    remaining = np.asarray(demand_by_band, dtype=float).copy()

    allocated = np.zeros((n, k))
    capped = np.zeros(n, dtype=bool)

    active = np.isfinite(values).all(axis=ONE)
    if eligible is not None:
        active = active & np.asarray(eligible, dtype=bool)
    active = active & (headroom > ZERO)

    max_iter = int(params.value(P_ALLOCATION_MAX_ITER))
    for _ in range(max_iter):
        if remaining.sum() <= ZERO or not active.any():
            break
        shares = _softmax(np.where(active[:, None], values, -np.inf))
        allocated = allocated + shares * remaining[None, :]

        used = (allocated * sqm[None, :]).sum(axis=ONE)
        over = active & (used > headroom)
        if not over.any():
            remaining = np.zeros(k)
            break

        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(used > ZERO, headroom / used, ZERO)
        excess = allocated[over] * (ONE - scale[over][:, None])
        allocated[over] = allocated[over] * scale[over][:, None]
        remaining = excess.sum(axis=ZERO)
        capped = capped | over
        active = active & ~over

    if remaining.sum() > ZERO:
        logger.warning(
            "allocation spill: %.1f households could not be housed (Section 12.4 — the city "
            "is supply-constrained; this feeds the price clearing)",
            float(remaining.sum()),
        )
    return allocated, remaining, capped


# --------------------------------------------------------------------------------------
# 12.5 — induced service employment
# --------------------------------------------------------------------------------------


def induced_service_jobs(
    allocated: np.ndarray,
    params: Any,
    *,
    persons_per_household_by_band: Sequence[float],
    dormitory_workers: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 12.5 — ``(new_residents_i, new_service_jobs_i)``.

    ``new_service_jobs_i = service_jobs_per_resident * new_residents_i``, plus dormitory
    workers at ``dorm_service_factor`` of the rate ("low disposable income and remittance
    outflow").

    **Service jobs are emitted here and only here.** Section 9.2 forbids Layer 2 from
    emitting them and Section 21 lists the double count as a named failure mode; the symptom
    is employment growing faster than population, which there is a test for.
    """
    pph = np.asarray(persons_per_household_by_band, dtype=float)
    residents = np.asarray(allocated, dtype=float) @ pph
    rate = float(params.value(P_SERVICE_JOBS_PER_RESIDENT))
    jobs = rate * residents

    if dormitory_workers is not None:
        factor = float(params.value(P_DORM_SERVICE_FACTOR))
        jobs = jobs + rate * np.asarray(dormitory_workers, dtype=float) * factor
    return residents, jobs


# --------------------------------------------------------------------------------------
# 12.6 — agglomeration damping
# --------------------------------------------------------------------------------------


def _spillover_phi(params: Any) -> float:
    """``phi = spillover_phi``, with Section 12.6's ``phi < 1`` enforced.

    Section 21 names "Agglomeration divergence — prices explode in one corridor" and gives
    the guard as "``phi < 1``, congestion feedback, ``ConvergenceError``". ``phi >= 1`` makes
    the inner loop a non-contraction: each round of induced service jobs feeds back at least
    as strongly as it arrived. That is a divergent parameter set by construction, so it
    raises the divergence error immediately rather than burning ``max_iterations`` first.
    """
    phi = float(params.value(P_SPILLOVER_PHI))
    if not phi < ONE:
        raise ConvergenceError(
            f"{P_SPILLOVER_PHI} = {phi} but Section 12.6 requires phi < 1: the agglomeration "
            "inner loop is not a contraction and will diverge (Section 21, 'Agglomeration "
            "divergence')"
        )
    return phi


def _induced_builtup(
    cells: pd.DataFrame, allocated_sqm: np.ndarray, base_builtup: np.ndarray
) -> np.ndarray:
    """`builtup_frac` after this year's allocation lands (input to the congestion feedback).

    Modelling choice, reported: the spec says only "as ``builtup_frac`` rises". The rise is
    taken proportional to how much of the cell's remaining development capacity this year's
    allocation consumes, ``delta_i = (allocated_sqm_i / capacity_sqm_i) * (1 - builtup_i)``,
    which is monotone, bounded by 1, and zero for a cell with no capacity.
    """
    if COL_CAPACITY not in cells.columns:
        return base_builtup
    capacity = cells[COL_CAPACITY].fillna(ZERO).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.divide(
            allocated_sqm, capacity, out=np.zeros_like(allocated_sqm), where=capacity > ZERO
        )
    return np.clip(base_builtup + np.clip(fraction, ZERO, ONE) * (ONE - base_builtup), ZERO, ONE)


# --------------------------------------------------------------------------------------
# 12.1 — the annual loop
# --------------------------------------------------------------------------------------


def allocate(
    cells: pd.DataFrame,
    params: Any,
    *,
    year: int,
    state: AllocationState | None = None,
    employment_effects: Sequence[EmploymentEffectLike] = (),
    activation_weights: Sequence[float] | None = None,
    field_res: pd.Series | Mapping[str, float] | Sequence[float] | float | None = None,
    matrices: MatrixSet | None = None,
    lnA_by_band: Mapping[str, Sequence[float]] | None = None,
    alpha: AlphaFit | pd.DataFrame | pd.Series | Mapping[str, float] | float | None = None,
    headroom_sqm: pd.Series | Mapping[str, float] | Sequence[float] | float | None = None,
    persons_per_household_by_band: Mapping[str, float] | Sequence[float] | None = None,
    natural_growth_rate: float | None = None,
    reallocate_stock: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """One simulation year of Layer 5 (spec Section 12.1, steps 3-5).

    Pure: returns a NEW frame with the same index, row count and column set as ``cells``,
    with ``households``, ``hh_by_band``, ``population``, ``jobs_by_sector`` and ``alpha_res``
    updated. Steps 1-2 (resolving effects, fetching the network state's cached accessibility)
    and step 6 (market clearing) belong to the runner and to Module 9; this function consumes
    their outputs and produces the demand side.

    Parameters
    ----------
    year:
        The calendar year being simulated.
    state:
        Previous year's :class:`AllocationState`. ``None`` on the first call, in which case
        ``alpha_i`` is calibrated from this frame and recorded (Section 12.3: estimate once,
        hold fixed).
    employment_effects, activation_weights:
        Section 9.1 ``EmploymentEffect``s active in the horizon and Layer 3's ``w(t)`` for
        each. Consumed structurally; Layer 2 is never imported.
    field_res:
        The Section 9.4 combined, capped residential field per cell.
    matrices:
        A precomputed :class:`~ufe.layers.routing.MatrixSet`. When supplied, ``lnA_ik`` is
        computed with the band-specific mode weights and the Section 12.6 inner loop is live
        (service-job spillover, damped by ``phi``, against the congestion feedback). When
        omitted, the frame's city-mode-split ``lnA`` column is used for every band and there
        is no accessibility feedback: the loop terminates after one pass. That fallback is
        recorded in the diagnostics as ``band_accessibility='city_mode_split_fallback'``.
    lnA_by_band:
        Explicit band accessibility, overriding both of the above.
    alpha:
        An :class:`AlphaFit`, a frame/series, or a scalar. ``0`` disables the fixed effect —
        useful only for demonstrating that the null test has teeth.
    headroom_sqm:
        Overrides the frame's Layer 4 ``headroom_sqm``. ``np.inf`` removes the capacity
        constraint (the null test runs unconstrained, since it is a test of the behavioural
        model, not of supply).
    persons_per_household_by_band:
        Section 12.5 input; required as soon as anything is allocated, because
        ``behaviour.persons_per_household_by_band`` is null on disk.
    natural_growth_rate:
        Overrides ``behaviour.natural_growth_rate`` (the runner passes a scenario value).
    reallocate_stock:
        Calibration / null-test mode. Instead of allocating this year's *increment*, the
        entire household stock of the eligible cells is cleared and re-allocated. With a
        correctly estimated ``alpha_i`` this must reproduce the observed distribution — the
        Section 12 acceptance null test. No service jobs are emitted and ``population`` is
        untouched in this mode: it is a diagnostic, not a simulation year.
    """
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    _require(cells, REQUIRED_COLUMNS)

    n_bands = len(INCOME_BANDS)
    sqm = sqm_per_household(params)
    phi = _spillover_phi(params)
    tol = float(params.value(P_CONVERGENCE_TOL))
    max_iterations = int(params.value(P_MAX_ITERATIONS))

    # --- band accessibility (Section 12.3) ------------------------------------------
    source = "supplied"
    base_builtup = (
        cells[COL_BUILTUP].fillna(ZERO).to_numpy(dtype=float)
        if COL_BUILTUP in cells.columns
        else np.zeros(len(cells))
    )
    if lnA_by_band is None:
        if matrices is not None:
            lnA_by_band = band_accessibility(cells, params, matrices)
            source = "band_mode_weights"
        else:
            _require(cells, (COL_LNA,))
            base = cells[COL_LNA].to_numpy(dtype=float)
            lnA_by_band = {band: base for band in INCOME_BANDS}
            source = "city_mode_split_fallback"
            logger.info(
                "no MatrixSet supplied: Section 12.3's band-specific access_mode_weights "
                "cannot be applied and the agglomeration feedback of Section 12.6 is inert"
            )
    lnA_current = {band: np.asarray(lnA_by_band[band], dtype=float) for band in INCOME_BANDS}
    lnA_base = {band: values.copy() for band, values in lnA_current.items()}

    # --- the cell fixed effect (Section 12.3) ---------------------------------------
    if state is not None and alpha is None:
        alpha_fit: AlphaFit | Any = state.alpha
    elif alpha is None:
        alpha_fit = estimate_alpha_res(
            cells, params, lnA_by_band=lnA_base, field_res=field_res, base_year=year
        )
    else:
        alpha_fit = alpha

    eligible = np.isfinite(
        utility(cells, params, lnA_by_band=lnA_base, alpha=alpha_fit, field_res=field_res)
        .to_numpy(dtype=float)
    ).all(axis=ONE)

    # --- this year's demand (Section 12.2) ------------------------------------------
    observed = _matrix(cells, COL_HH_BY_BAND, n_bands)
    if reallocate_stock:
        demand = HouseholdDemand(
            by_band=observed[eligible].sum(axis=ZERO),
            exogenous=float(ZERO),
            job_driven=float(ZERO),
            dormitory_workers=pd.Series(np.zeros(len(cells)), index=cells.index),
        )
    else:
        demand = household_demand(
            cells,
            params,
            year=year,
            employment_effects=employment_effects,
            activation_weights=activation_weights,
            natural_growth_rate=natural_growth_rate,
        )

    headroom = (
        cells[COL_HEADROOM].fillna(ZERO).to_numpy(dtype=float)
        if headroom_sqm is None
        else _as_array(headroom_sqm, cells, ZERO)
    )
    dormitory = demand.dormitory_workers.to_numpy(dtype=float)
    needs_pph = demand.total > ZERO or dormitory.sum() > ZERO
    pph = (
        persons_per_household(params, persons_per_household_by_band)
        if (needs_pph and not reallocate_stock)
        else np.zeros(n_bands)
    )

    # --- 12.1 step 5: the agglomeration inner loop ----------------------------------
    allocated = np.zeros((len(cells), n_bands))
    spill = demand.by_band.copy()
    capped = np.zeros(len(cells), dtype=bool)
    residents = np.zeros(len(cells))
    service_jobs = np.zeros(len(cells))
    max_delta = float(ZERO)
    first_delta: float | None = None
    converged = matrices is None
    iterations = ZERO

    for iterations in range(ONE, max_iterations + ONE):
        u = utility(cells, params, lnA_by_band=lnA_current, alpha=alpha_fit, field_res=field_res)
        allocated, spill, capped = allocate_constrained(
            u, demand.by_band, headroom, sqm, params, eligible=eligible
        )
        if reallocate_stock:
            residents = np.zeros(len(cells))
            service_jobs = np.zeros(len(cells))
        else:
            residents, service_jobs = induced_service_jobs(
                allocated,
                params,
                persons_per_household_by_band=pph,
                dormitory_workers=dormitory,
            )

        if matrices is None:
            converged = True
            break

        # 5d: add service jobs -> update the opportunity surface -> recompute lnA, with the
        # congestion feedback of Section 12.6 active *inside* the loop.
        probe = cells.copy(deep=False)
        jobs = _matrix(cells, COL_JOBS, len(SECTORS)).copy()
        jobs[:, int(SERVICE_SECTOR)] += service_jobs
        probe[COL_JOBS] = list(jobs)
        allocated_sqm = (allocated * sqm[None, :]).sum(axis=ONE)
        probe_builtup = _induced_builtup(cells, allocated_sqm, base_builtup)

        raw = band_accessibility(
            probe,
            params,
            matrices,
            base_builtup=base_builtup,
            congestion_builtup=probe_builtup,
        )

        step_max = float(ZERO)
        for band in INCOME_BANDS:
            delta = np.asarray(raw[band], dtype=float) - lnA_current[band]
            delta = np.where(np.isfinite(delta), delta, ZERO)
            step = phi * delta
            lnA_current[band] = lnA_current[band] + step
            step_max = max(step_max, float(np.abs(step).max(initial=ZERO)))

        max_delta = step_max
        if first_delta is None:
            first_delta = step_max
        elif step_max > first_delta:
            raise ConvergenceError(
                f"agglomeration inner loop diverging at iteration {iterations}: "
                f"max|delta lnA| rose to {step_max:g} from an initial {first_delta:g}. "
                f"Check {P_SPILLOVER_PHI} (phi must be < 1) and the congestion feedback "
                "(Section 12.6, Section 21)"
            )
        if step_max < tol:
            converged = True
            break

    if not converged:
        raise ConvergenceError(
            f"agglomeration inner loop did not converge in {max_iterations} iterations "
            f"({P_MAX_ITERATIONS}): max|delta lnA| = {max_delta:g} against "
            f"{P_CONVERGENCE_TOL} = {tol:g}. A non-converging city means a parameter is "
            "wrong (Section 12.6)"
        )

    # --- write the year's state back onto a NEW frame --------------------------------
    out = cells.copy(deep=True)
    previous_total = cells[COL_HOUSEHOLDS].to_numpy(dtype=float)
    if reallocate_stock:
        new_bands = np.where(eligible[:, None], allocated, observed)
        new_total = np.where(eligible, allocated.sum(axis=ONE), previous_total)
    else:
        # Adding the increment to the stored total (rather than re-summing the bands) keeps a
        # zero-allocation year bit-identical to its input: `households` and `hh_by_band` are
        # independently stored in the schema and need not agree to the last ulp on ingest.
        new_bands = observed + allocated
        new_total = previous_total + allocated.sum(axis=ONE)
    out[COL_HH_BY_BAND] = [list(map(float, row)) for row in new_bands]
    out[COL_HOUSEHOLDS] = new_total

    if not reallocate_stock:
        out[COL_POPULATION] = cells[COL_POPULATION].to_numpy(dtype=float) + residents
        jobs = _matrix(cells, COL_JOBS, len(SECTORS)).copy()
        jobs[:, int(SERVICE_SECTOR)] += service_jobs
        out[COL_JOBS] = [list(map(float, row)) for row in jobs]

    if isinstance(alpha_fit, AlphaFit):
        out[COL_ALPHA] = (
            alpha_fit.per_cell.reindex(cells[COL_H3].to_numpy()).to_numpy(dtype=float)
        )
        new_state = AllocationState(
            alpha=alpha_fit,
            base_year=state.base_year if state is not None else int(year),
            cumulative_spill_by_band=(
                (state.cumulative_spill_by_band if state is not None else np.zeros(n_bands))
                + spill
            ),
        )
        out.attrs[ATTR_STATE] = new_state

    index = cells.index
    out.attrs[ATTR_DIAGNOSTICS] = {
        "year": int(year),
        "band_accessibility": source,
        "converged": bool(converged),
        "iterations": int(iterations),
        "max_delta_lnA": float(max_delta),
        "spillover_phi": phi,
        "demand_by_band": demand.by_band,
        "new_households": float(demand.total),
        "exogenous_households": float(demand.exogenous),
        "job_driven_households": float(demand.job_driven),
        "allocated_by_band": pd.DataFrame(allocated, index=index, columns=list(INCOME_BANDS)),
        "allocated_sqm": pd.Series((allocated * sqm[None, :]).sum(axis=ONE), index=index),
        "spill_by_band": spill,
        "spill_households": float(spill.sum()),
        "capped_cells": pd.Series(capped, index=index),
        "eligible": pd.Series(eligible, index=index),
        "new_residents": pd.Series(residents, index=index),
        "new_service_jobs": pd.Series(service_jobs, index=index),
        "dormitory_workers": demand.dormitory_workers,
        "lnA_by_band": {band: lnA_current[band].copy() for band in INCOME_BANDS},
    }
    return out


# --------------------------------------------------------------------------------------
# 12.7 — firm allocation
# --------------------------------------------------------------------------------------


def _firm_coefficient(params: Any, name: str) -> float:
    value = params.get(f"{P_FIRM_COEFFICIENTS}.{name}")
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        raise MissingParameter(
            f"{P_FIRM_COEFFICIENTS}.{name} is null in config/params/cascade.yaml. The "
            "Section 12.7 firm_logit coefficients are not specified anywhere in the supplied "
            "spec and were left null deliberately; Module 8 raises rather than inventing them."
        )
    return float(value)


def allocate_firms(
    cells: pd.DataFrame,
    params: Any,
    *,
    jobs_by_sector: Mapping[int | str, float],
    freight_access: pd.Series | Sequence[float] | None = None,
    zoning_gate: Mapping[int | str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Section 12.7 — a simpler logit for non-service, non-pinned employment.

    ``V_i^s = c_market lnA_i^market + c_labour lnA_i^labour + c_land ln(price_land_i)
    + c_agglom ln(jobs_i^s + 1) + c_freight freight_access_i (industrial only)
    + zoning_gate_i^s``

    "Only cascade-generated and organically-growing employment uses this. Project-located
    employment is pinned to its cell" — so the caller passes only the unpinned jobs.

    Ambiguities resolved, reported:

    * Section 12.7 writes ``lnA^market`` and ``lnA^labour``; Layer 1 emits no columns of
      those names. ``lnA_retail`` (access to consumers) is used for market access and
      ``lnA_work`` (access to jobs, hence to the labour pool) for labour access.
    * ``freight_access_i`` is not a landed column and has no definition in the spec, so it is
      an explicit argument. Omitting it drops the term.
    * ``cascade.candidate_filter.zone_classes`` is written as ``[industrial, mixed]`` but the
      landed ``ZONE_CLASSES`` vocabulary uses ``ind``. Rather than guess at the mapping, the
      gate is an explicit argument here and the vocabulary mismatch is reported.
    """
    _require(cells, (COL_H3, COL_JOBS, COL_PRICE_LAND, "lnA_work", "lnA_retail", "zone_class"))

    if not bool(params.get(f"{P_FIRM_LOGIT}.enabled")):
        logger.info("%s.enabled is false; coefficients are checked all the same", P_FIRM_LOGIT)
    coefficients = {name: _firm_coefficient(params, name) for name in FIRM_COEFFICIENTS}

    market = cells["lnA_retail"].to_numpy(dtype=float)
    labour = cells["lnA_work"].to_numpy(dtype=float)
    ln_land = np.log(
        _positive_or_median(cells[COL_PRICE_LAND].to_numpy(dtype=float), COL_PRICE_LAND)
    )
    freight = _as_array(freight_access, cells, ZERO)
    zones = cells["zone_class"].to_numpy()
    jobs = _matrix(cells, COL_JOBS, len(SECTORS)).copy()

    for sector, count in jobs_by_sector.items():
        name = _sector_name(sector)
        index = int(Sector[name])
        v = (
            coefficients["c_market"] * market
            + coefficients["c_labour"] * labour
            + coefficients["c_land"] * ln_land
            + coefficients["c_agglom"] * np.log(jobs[:, index] + ONE)
        )
        if name in INDUSTRIAL_SECTORS:
            v = v + coefficients["c_freight"] * freight
        if zoning_gate is not None and (sector in zoning_gate or name in zoning_gate):
            allowed = tuple(zoning_gate.get(sector, zoning_gate.get(name, ())))
            v = np.where(np.isin(zones, allowed), v, -np.inf)
        shares = _softmax(np.where(np.isfinite(v), v, -np.inf)[:, None])[:, ZERO]
        jobs[:, index] = jobs[:, index] + shares * float(count)

    out = cells.copy(deep=True)
    out[COL_JOBS] = [list(map(float, row)) for row in jobs]
    return out
