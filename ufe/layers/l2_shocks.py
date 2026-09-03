"""Layer 2 — shock resolution (spec Section 9).

This layer turns a pipeline of announced projects into *typed effects* on the cell grid.
It is **generic**: every magnitude, radius, decay shape, wage and share is read from
``config/params/archetypes.yaml`` (plus the caps in ``price.yaml`` and the household
routing constants in ``behaviour.yaml``).  There is no per-archetype branching anywhere in
this module, and no archetype name appears in it — adding an archetype is a YAML edit
(Section 9 ACCEPTANCE, first item).

Public entry point
------------------
``resolve_shocks(cells, projects, params, *, year, ...) -> pd.DataFrame``
    Pure.  Returns a new frame with the same index and row count as `cells`, plus the
    ``shock_*`` columns listed in :data:`ADDED_COLUMNS`.  The full typed-effect resolution
    is attached at ``out.attrs[ATTR_KEY]`` as a :class:`ShockResolution`.

Also public, because they are the interesting maths and are tested directly:
``field_decay`` (Section 9.3), ``wedge_factor`` (Section 9.3 directional wedge),
``compose_fields`` (Section 9.4), ``wage_band`` and ``route_households`` (Section 9.5).

Weighting by Layer 3
--------------------
Nothing in here recomputes credibility.  ``resolve_shocks`` calls
:func:`ufe.layers.l3_credibility.activation_weight` and multiplies **every** emitted
magnitude — jobs, field log-points, floorspace, capacity deltas — by the project's
``activation_weight`` for `year`.  `projects` must therefore already carry
``p_completion`` (from ``completion_probability``) and ``open_year`` (from
``delay_distribution``).  A project with weight 0 (dead, not yet announced, or forced to
``fails`` by the Section 10.5 counterfactual) contributes exactly nothing.

Because the effects are resolved *for a year*, the magnitudes on the returned effect
objects are the weighted ones.  That is what Layer 4 (Section 11.3) and Layer 5
(Section 12) consume.

Service employment
------------------
Section 9.2 step 2 and the Section 21 failure table are explicit: induced service
employment is computed in Layer 5 from resident population and is **never** emitted here.
This module does not read ``behaviour.service_jobs_per_resident`` and never writes to the
``retail_svc`` slot of ``shock_jobs_by_sector``.

CRS discipline
--------------
Every distance and every bearing is computed in the city's ``crs_metric`` via
:mod:`ufe.geo`.  This module never touches ``pyproj`` and never subtracts two degrees.

Ambiguities resolved here (see the build report)
-----------------------------------------------
* Section 9.5 prints ``effective_households = jobs * inmigrant_share[sector] /
  workers_per_household * (1 - dormitory_share)`` but introduces the block with "for
  archetypes with ``housing_typology.ownership_demand_share``".  Taken literally the
  formula does not use that share and the Section 9 ACCEPTANCE number ("under 2,000"
  effective households for a 30,000-job dormitory-typology plant) is unreachable — the
  literal formula gives ~4,634.  ``ownership_demand_share`` is therefore applied as a
  further multiplicative factor, defaulting to 1 for archetypes that declare none.
* Section 9.3 gives three decay shapes but ``archetypes.yaml`` carries no ``decay`` key on
  any field entry.  The default is :data:`DECAY_STEP`, which is also what makes the
  Section 9.3 exclusive-band rule and its ACCEPTANCE number (0.09, not 0.145) come out.
  A field entry may override it with ``decay: linear`` / ``decay: exponential``.
* Section 9.3's directional wedge is described "for airports", but a generic resolver
  cannot key behaviour off an archetype name.  The wedge is switched on by a
  ``directional_wedge: true`` flag on the archetype block (or on an individual field
  entry).  No shipped archetype carries it — reported, not invented.
* ``FieldEffect.end_year`` is treated as **exclusive**: a construction penalty running
  from ``construction_start_year`` to ``open_year`` is inactive in ``open_year`` itself.
* Section 3.7 says the income-band boundaries are "inflation-indexed by base year" but no
  index or inflation rate exists in the YAML.  The multiplier is the explicit
  ``income_index`` argument, defaulting to 1 (base-year nominal).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from ufe import geo
from ufe.errors import MissingParameter
from ufe.layers.l3_credibility import activation_weight
from ufe.params import Params
from ufe.store.schemas import INCOME_BANDS, SECTORS

logger = logging.getLogger(__name__)

__all__ = [
    "EmploymentEffect",
    "NetworkEffect",
    "FieldEffect",
    "SupplyEffect",
    "FloorspaceDemandEffect",
    "HouseholdDemand",
    "ShockResolution",
    "resolve_shocks",
    "field_decay",
    "wedge_factor",
    "compose_fields",
    "wage_band",
    "route_households",
    "ATTR_KEY",
    "ADDED_COLUMNS",
]


# --------------------------------------------------------------------------------------
# parameter paths — every number in this module comes from one of these
# --------------------------------------------------------------------------------------

ARCH = "archetypes"
DEFAULTS = f"{ARCH}._defaults"
P_DEFAULT_CONSTRUCTION_YEARS = f"{DEFAULTS}.construction_years"
P_DEFAULT_RAMP_YEARS = f"{DEFAULTS}.operational_ramp_years"

P_CAP_LOW = "price.fields.cap_low"
P_CAP_HIGH = "price.fields.cap_high"
P_CAP_WARN_SHARE = "price.fields.cap_warn_share"
P_EXPONENTIAL_DECAY_K = "price.fields.exponential_decay_k"
P_WEDGE_SCALE = "price.fields.airport_wedge_scale"

P_WORKERS_PER_HOUSEHOLD = "behaviour.workers_per_household"
P_HOUSEHOLD_WAGE_PREMIUM = "behaviour.wage_to_band.household_wage_premium"
P_INMIGRANT_SHARE = "behaviour.migration.inmigrant_share_by_sector"
P_INCOME_BAND_BOUNDARIES = "behaviour.income_bands.boundaries_inr_mo"
P_OFFICE_SQM_PER_SEAT = "behaviour.office_sqm_per_seat"


# --------------------------------------------------------------------------------------
# vocabularies (strings, not parameters)
# --------------------------------------------------------------------------------------

#: Section 9.3 decay shapes.
DECAY_STEP = "step"
DECAY_LINEAR = "linear"
DECAY_EXPONENTIAL = "exponential"
DECAYS: tuple[str, ...] = (DECAY_STEP, DECAY_LINEAR, DECAY_EXPONENTIAL)

#: Field entries carry no `decay` key in the shipped YAML; this is the default.
DEFAULT_DECAY = DECAY_STEP

#: Section 9.1 `FieldEffect.target`.
TARGET_RESIDENTIAL = "residential"
TARGET_COMMERCIAL = "commercial"
TARGET_OFFICE = "office"
TARGET_ALL = "all"
FIELD_TARGETS: tuple[str, ...] = (TARGET_RESIDENTIAL, TARGET_COMMERCIAL, TARGET_OFFICE)

#: The three field categories in `archetypes.yaml` (Section 9.2 step 4).
CATEGORY_PREMIUM = "premium"
CATEGORY_DISAMENITY = "disamenity"
CATEGORY_CONSTRUCTION_PENALTY = "construction_penalty"
FIELD_CATEGORIES: tuple[str, ...] = (
    CATEGORY_PREMIUM,
    CATEGORY_DISAMENITY,
    CATEGORY_CONSTRUCTION_PENALTY,
)

#: Section 9.1 `FloorspaceDemandEffect.use`.
USE_OFFICE = "office"
USE_DORMITORY = "dormitory"
USE_RETAIL = "retail"
USE_HOTEL = "hotel"
USE_WAREHOUSE = "warehouse"

#: Section 9.1 `NetworkEffect.kind`; `none` means "emit nothing".
NETWORK_NONE = "none"

#: Archetype keys read by this module.
KEY_SCALE_UNIT = "scale_unit"
KEY_NETWORK_EFFECT = "network_effect"
KEY_EMPLOYMENT = "employment"
KEY_HOUSING_TYPOLOGY = "housing_typology"
KEY_PREMIUM_MULTIPLIERS = "premium_multipliers"
KEY_STERILISES_LAND = "sterilises_land"
KEY_LAND_TAKE = "land_take_sqm_per_unit"
KEY_DIRECTIONAL_WEDGE = "directional_wedge"
KEY_APPLIES_WHEN = "applies_when"
KEY_MAX_M = "max_m"
KEY_TARGET = "target"
KEY_DECAY = "decay"
KEY_VALUE = "value"
KEY_TYPE = "type"
KEY_SPEED_KMH = "speed_kmh"

#: Error-handling modes, spelled as in `l3_credibility`.
RAISE = "raise"
IGNORE = "ignore"
_MODES = frozenset({RAISE, IGNORE})

#: The sector whose employment effects also imply office floorspace demand
#: (Section 9.2 step 6).  Read from the shared Section 3.6 taxonomy, not hardcoded.
OFFICE_SECTOR = SECTORS.index("it_office")
CONSTRUCTION_SECTOR = SECTORS.index("construction")


# --------------------------------------------------------------------------------------
# output columns
# --------------------------------------------------------------------------------------

COL_PREFIX = "shock_"

COL_FIELD_RESIDENTIAL = f"{COL_PREFIX}field_residential"
COL_FIELD_COMMERCIAL = f"{COL_PREFIX}field_commercial"
COL_FIELD_OFFICE = f"{COL_PREFIX}field_office"
COL_FIELD_CAP_HIT = f"{COL_PREFIX}field_cap_hit"
COL_JOBS_PERMANENT = f"{COL_PREFIX}jobs_permanent"
COL_JOBS_CONSTRUCTION = f"{COL_PREFIX}jobs_construction"
COL_JOBS_BY_SECTOR = f"{COL_PREFIX}jobs_by_sector"
COL_EFFECTIVE_HOUSEHOLDS = f"{COL_PREFIX}effective_households"
COL_HOUSEHOLDS_BY_BAND = f"{COL_PREFIX}households_by_band"
COL_DORMITORY_WORKERS = f"{COL_PREFIX}dormitory_workers"
COL_FLOORSPACE_DEMAND = f"{COL_PREFIX}floorspace_demand_sqm"
COL_DELTA_CAPACITY = f"{COL_PREFIX}delta_capacity_sqm"
COL_DELTA_FLOORSPACE = f"{COL_PREFIX}delta_floorspace_sqm"

#: Columns this layer adds to `cells`.  None of them is declared in
#: `ufe/store/schemas.py` yet (see the build report) — the schema owner must add them
#: before a frame carrying them is handed to `write_table`.
ADDED_COLUMNS: tuple[str, ...] = (
    COL_FIELD_RESIDENTIAL,
    COL_FIELD_COMMERCIAL,
    COL_FIELD_OFFICE,
    COL_FIELD_CAP_HIT,
    COL_JOBS_PERMANENT,
    COL_JOBS_CONSTRUCTION,
    COL_JOBS_BY_SECTOR,
    COL_EFFECTIVE_HOUSEHOLDS,
    COL_HOUSEHOLDS_BY_BAND,
    COL_DORMITORY_WORKERS,
    COL_FLOORSPACE_DEMAND,
    COL_DELTA_CAPACITY,
    COL_DELTA_FLOORSPACE,
)

#: `out.attrs` slot carrying the :class:`ShockResolution`.
ATTR_KEY = "shock_resolution"

_FIELD_TARGET_COLUMNS: dict[str, str] = {
    TARGET_RESIDENTIAL: COL_FIELD_RESIDENTIAL,
    TARGET_COMMERCIAL: COL_FIELD_COMMERCIAL,
    TARGET_OFFICE: COL_FIELD_OFFICE,
}

_REQUIRED_CELL_COLUMNS = ("h3", "lat", "lon")
_REQUIRED_PROJECT_COLUMNS = ("project_id", "archetype", "geom", "scale_value", "scale_unit")


# --------------------------------------------------------------------------------------
# 9.1 effect types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EmploymentEffect:
    """Permanent or construction employment sited in one cell (spec Section 9.1)."""

    cell: str
    sector: int
    jobs: float
    median_wage_inr_mo: float
    start_year: int
    ramp_years: int
    capture_radius_m: float
    dormitory_share: float = 0
    is_construction: bool = False
    duration_years: int | None = None
    project_id: str = ""


@dataclass(frozen=True)
class NetworkEffect:
    """A new or upgraded link in the network (spec Section 9.1).

    Consumed by Layer 1: it changes the travel-time matrices, so it is emitted as a
    geometry + speed + opening year and never as a price field.
    """

    kind: str
    geometry: LineString | BaseGeometry
    stations: list[Point] | None
    speed_kmh: float
    open_year: int
    project_id: str = ""


@dataclass(frozen=True)
class FieldEffect:
    """A signed log-point field around a geometry (spec Section 9.1).

    ``band_group`` is the Section 9.3 exclusive-band key: every effect sharing a
    ``band_group`` is a band of one multi-band field, and a cell takes the *narrowest*
    band that contains it, never the sum.
    """

    origin: BaseGeometry
    target: str
    max_m: float
    magnitude: float
    decay: str
    start_year: int
    end_year: int | None
    project_id: str = ""
    band_group: str = ""
    directional_wedge: bool = False


@dataclass(frozen=True)
class SupplyEffect:
    """Townships, land sterilisation, land banking (spec Section 9.1).

    THE CANONICAL DEFINITION.  ``ufe/layers/l4_supply.py`` currently declares a local copy
    of this dataclass because Layer 2 did not exist when it was written; the first four
    fields here are identical in name, order and meaning, so the two are drop-in
    compatible and `l4_supply` should import this one instead (see the build report).

    ``delta_floorspace_sqm`` positive = new supply committed (Section 11.3 delivers it
    gradually).  ``delta_capacity_sqm`` negative = sterilised (applied at ``start_year``).
    """

    cell: str
    delta_floorspace_sqm: float
    delta_capacity_sqm: float
    start_year: int
    project_id: str = ""


@dataclass(frozen=True)
class FloorspaceDemandEffect:
    """Non-residential floorspace demanded by a project (spec Section 9.1)."""

    cell: str
    use: str
    sqm: float
    start_year: int
    ramp_years: int
    project_id: str = ""


@dataclass(frozen=True)
class HouseholdDemand:
    """The Section 9.5 wage-band routing result for one :class:`EmploymentEffect`.

    Section 9.5 is a routing rule, not an effect type, but its output has to travel to
    Layer 5 somehow.  ``dormitory_workers`` deliberately do **not** appear in
    ``effective_households``: that is the mechanism that stops a dormitory-typology plant
    producing phantom apartment buyers.
    """

    cell: str
    band: int
    effective_households: float
    dormitory_workers: float
    household_income_inr_mo: float
    capture_radius_m: float
    start_year: int
    ramp_years: int
    project_id: str = ""


@dataclass(frozen=True)
class ShockResolution:
    """Everything Section 9.2 emitted for one year, plus diagnostics."""

    year: int
    employment: tuple[EmploymentEffect, ...] = ()
    network: tuple[NetworkEffect, ...] = ()
    fields: tuple[FieldEffect, ...] = ()
    supply: tuple[SupplyEffect, ...] = ()
    floorspace_demand: tuple[FloorspaceDemandEffect, ...] = ()
    households: tuple[HouseholdDemand, ...] = ()
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s): {', '.join(missing)}")


def _check_mode(name: str, mode: str) -> str:
    if mode not in _MODES:
        raise ValueError(f"{name} must be one of {sorted(_MODES)}, got {mode!r}")
    return mode


def _leaf(params: Params, path: str, *, monte_carlo: bool, rng: Any) -> float:
    """Deterministic value, or a Monte Carlo draw when asked (Section 14.1)."""
    if monte_carlo:
        return float(params.sample(path, rng))
    return float(params.value(path))


def _optional_node(params: Params, path: str) -> Any:
    try:
        return params.get(path)
    except MissingParameter:
        return None


def _as_entries(node: Any) -> list[Any]:
    """A field category is a list of bands or a single band; normalise to a list."""
    if node is None:
        return []
    if isinstance(node, Mapping):
        return [node]
    return list(node)


def _geometry(value: Any) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    return shapely_wkt.loads(str(value))


# --------------------------------------------------------------------------------------
# 9.3 field decay functions
# --------------------------------------------------------------------------------------


def field_decay(
    decay: str,
    magnitude: float,
    distance_m: np.ndarray,
    max_m: float,
    params: Params,
) -> np.ndarray:
    """Section 9.3, verbatim.

    ``step``        -> ``magnitude if d <= max_m else 0``
    ``linear``      -> ``magnitude * max(0, 1 - d / max_m)``
    ``exponential`` -> ``magnitude * exp(-k * d / max_m)`` inside ``max_m``, where ``k``
    is ``price.fields.exponential_decay_k`` (the spec prints 3, "~5% at max_m").
    """
    d = np.asarray(distance_m, dtype=float)
    reach = float(max_m)
    inside = d <= reach
    if decay == DECAY_STEP:
        return np.where(inside, float(magnitude), 0)
    if decay == DECAY_LINEAR:
        return float(magnitude) * np.maximum(0, 1 - d / reach)
    if decay == DECAY_EXPONENTIAL:
        k = float(params.value(P_EXPONENTIAL_DECAY_K))
        return np.where(inside, float(magnitude) * np.exp(-k * d / reach), 0)
    raise ValueError(f"unknown field decay {decay!r}; Section 9.3 defines {list(DECAYS)}")


def wedge_factor(
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
    cbd_xy: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Section 9.3 directional wedge, in metric coordinates.

    ``theta_i`` is the angle between ``origin -> cell_i`` and ``origin -> cbd``;
    ``wedge_i = scale * (1 + cos theta_i)`` with ``scale`` from
    ``price.fields.airport_wedge_scale``.  On the CBD bearing that is 1; directly opposite
    it is 0.  This is what makes the model directional rather than a radial buffer — the
    Section 21 "Radial airport model" guard.

    A cell sitting exactly on the origin has no bearing; it receives the full factor.
    """
    scale = float(params.value(P_WEDGE_SCALE))
    origin = np.asarray(origin_xy, dtype=float).reshape(-1)
    targets = np.asarray(target_xy, dtype=float).reshape(-1, len(origin))
    to_cbd = np.asarray(cbd_xy, dtype=float).reshape(-1) - origin
    to_cell = targets - origin

    cbd_norm = float(np.linalg.norm(to_cbd))
    if cbd_norm == 0:
        raise ValueError("the CBD coincides with the field origin; no bearing is defined")
    cell_norm = np.linalg.norm(to_cell, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = (to_cell @ to_cbd) / (cell_norm * cbd_norm)
    cos_theta = np.where(cell_norm == 0, 1, cos_theta)
    return scale * (1 + np.clip(cos_theta, -1, 1))


# --------------------------------------------------------------------------------------
# 9.4 overlapping fields
# --------------------------------------------------------------------------------------


def compose_fields(
    accumulated: Mapping[str, np.ndarray], params: Params
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Section 9.4: sum in log space (already done by the caller), then clip.

    ``total_field_i = clip(sum_effects magnitude_i, cap_low, cap_high)``.  Returns the
    clipped fields per target plus a boolean mask of the cells that hit either cap in any
    target — the quantity Section 9.4 says to log.
    """
    cap_low = float(params.value(P_CAP_LOW))
    cap_high = float(params.value(P_CAP_HIGH))
    if cap_low > cap_high:
        raise ValueError(f"{P_CAP_LOW} ({cap_low}) exceeds {P_CAP_HIGH} ({cap_high})")

    clipped: dict[str, np.ndarray] = {}
    hit = None
    for target, raw in accumulated.items():
        values = np.asarray(raw, dtype=float)
        clipped[target] = np.clip(values, cap_low, cap_high)
        target_hit = (values < cap_low) | (values > cap_high)
        hit = target_hit if hit is None else (hit | target_hit)
    if hit is None:
        hit = np.zeros(0, dtype=bool)
    return clipped, hit


# --------------------------------------------------------------------------------------
# 9.5 wage band routing
# --------------------------------------------------------------------------------------


def _band_boundaries(params: Params, income_index: float) -> np.ndarray:
    node = params.get(P_INCOME_BAND_BOUNDARIES)
    boundaries = np.array(
        [float(entry[KEY_VALUE]) for entry in node], dtype=float
    ) * float(income_index)
    expected = len(INCOME_BANDS) - 1
    if boundaries.size != expected:
        raise ValueError(
            f"{P_INCOME_BAND_BOUNDARIES} must hold {expected} boundaries for "
            f"{len(INCOME_BANDS)} bands, got {boundaries.size}"
        )
    return boundaries


def wage_band(
    median_wage_inr_mo: float, params: Params, *, income_index: float = 1
) -> tuple[int, float]:
    """Section 9.5: map a job's median wage to the household's income band.

    ``household_income = median_wage_inr_mo * workers_per_household *
    household_wage_premium``, then ``band = digitize(household_income, boundaries)``.
    Returns ``(band, household_income)``.
    """
    workers = float(params.value(P_WORKERS_PER_HOUSEHOLD))
    premium = float(params.value(P_HOUSEHOLD_WAGE_PREMIUM))
    income = float(median_wage_inr_mo) * workers * premium
    boundaries = _band_boundaries(params, income_index)
    return int(np.digitize(income, boundaries)), income


def route_households(
    jobs: float,
    sector: int,
    params: Params,
    *,
    dormitory_share: float = 0,
    ownership_demand_share: float = 1,
) -> tuple[float, float]:
    """Section 9.5: ``(effective_households, dormitory_workers)``.

    ``effective_households = jobs * inmigrant_share[sector] / workers_per_household *
    (1 - dormitory_share) * ownership_demand_share``.  The last factor is the ambiguity
    documented in the module docstring: without it the Section 9 ACCEPTANCE threshold for
    a dormitory-typology plant cannot be met.  It defaults to 1, so archetypes that
    declare no ``housing_typology.ownership_demand_share`` follow the printed formula
    exactly.

    ``dormitory_workers = jobs * dormitory_share`` and are excluded from the household
    allocation entirely.
    """
    sector_name = SECTORS[int(sector)]
    inmigrant = float(params.value(f"{P_INMIGRANT_SHARE}.{sector_name}"))
    workers = float(params.value(P_WORKERS_PER_HOUSEHOLD))
    effective = (
        float(jobs)
        * inmigrant
        / workers
        * (1 - float(dormitory_share))
        * float(ownership_demand_share)
    )
    return effective, float(jobs) * float(dormitory_share)


# --------------------------------------------------------------------------------------
# archetype access
# --------------------------------------------------------------------------------------


class _Archetype:
    """A thin, name-agnostic view of one ``archetypes.<key>`` block."""

    def __init__(self, params: Params, key: str) -> None:
        node = _optional_node(params, f"{ARCH}.{key}")
        if not isinstance(node, Mapping):
            raise MissingParameter(
                f"{ARCH}.{key} — no archetype block for {key!r} in archetypes.yaml"
            )
        self.key = key
        self.node = node
        self.path = f"{ARCH}.{key}"

    def get(self, name: str) -> Any:
        return self.node.get(name)

    @property
    def scale_unit(self) -> str | None:
        unit = self.node.get(KEY_SCALE_UNIT)
        return None if unit is None else str(unit)

    @property
    def employment(self) -> Mapping[str, Any] | None:
        node = self.node.get(KEY_EMPLOYMENT)
        return node if isinstance(node, Mapping) else None

    @property
    def housing_typology(self) -> Mapping[str, Any]:
        node = self.node.get(KEY_HOUSING_TYPOLOGY)
        return node if isinstance(node, Mapping) else {}

    @property
    def directional_wedge(self) -> bool:
        return bool(self.node.get(KEY_DIRECTIONAL_WEDGE, False))

    def employment_path(self, name: str) -> str:
        return f"{self.path}.{KEY_EMPLOYMENT}.{name}"


def _ramp_years(params: Params, archetype: _Archetype) -> float:
    node = archetype.get("operational_ramp_years")
    if isinstance(node, Mapping):
        return float(node[KEY_VALUE])
    return float(params.value(P_DEFAULT_RAMP_YEARS))


def _construction_years(params: Params, archetype: _Archetype) -> float:
    employment = archetype.employment or {}
    node = employment.get("construction_years") or archetype.get("construction_years")
    if isinstance(node, Mapping):
        return float(node[KEY_VALUE])
    return float(params.value(P_DEFAULT_CONSTRUCTION_YEARS))


# --------------------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------------------


def resolve_shocks(
    cells: pd.DataFrame,
    projects: pd.DataFrame,
    params: Params,
    *,
    year: int,
    cbd: Point | tuple[float, float] | None = None,
    crs_metric: str | None = None,
    project_flags: Mapping[str, Iterable[str]] | None = None,
    income_index: float = 1,
    monte_carlo: bool = False,
    rng: np.random.Generator | None = None,
    force_project_state: Mapping[str, str] | None = None,
    unknown_archetypes: str = RAISE,
    missing_office_sqm_per_seat: str = RAISE,
    check_scale_unit: bool = True,
) -> pd.DataFrame:
    """Resolve the project pipeline into cell-level shocks for `year` (spec Section 9).

    Parameters
    ----------
    cells:
        The current `cells` frame.  Never mutated.  Needs ``h3``, ``lat`` and ``lon``.
    projects:
        The pipeline, already carrying ``p_completion`` and ``open_year`` from Layer 3.
        Never mutated.
    params:
        Resolved parameter tree.  Every magnitude comes from here.
    year:
        The calendar year being resolved.
    cbd:
        The city centre used by the Section 9.3 directional wedge.  Defaults to the city
        config's ``cbd_point``.
    crs_metric:
        Override the city's metric CRS (tests only); defaults to
        :func:`ufe.geo.city_metric_crs`.
    project_flags:
        Per-project qualitative flags, keyed by ``project_id``.  A field entry carrying
        ``applies_when: <flag>`` is emitted only for projects holding that flag, and an
        ``premium_multipliers.<flag>`` entry multiplies that project's premiums.  This is
        how ``elevated`` alignments and interchange stations are expressed without an
        archetype name in the code.
    income_index:
        Multiplier applied to the Section 3.7 income-band boundaries (see module note).
    monte_carlo, rng:
        Draw magnitudes from their ``low..high`` ranges instead of taking ``.value``.
        ``monte_carlo=True`` requires an explicit :class:`numpy.random.Generator`.
    force_project_state:
        Section 10.5 counterfactual, passed straight through to
        :func:`ufe.layers.l3_credibility.activation_weight`.
    unknown_archetypes:
        ``'raise'`` (default) or ``'ignore'`` for a project whose archetype has no YAML
        block.  Only 3 of the 22 archetypes the spec names are transcribed, so pipelines
        assembled against the real config will hit this.
    missing_office_sqm_per_seat:
        ``'raise'`` (default) or ``'ignore'``.  ``behaviour.office_sqm_per_seat`` is
        deliberately null in the shipped YAML, whose own comment says "Module 8 / Module 5
        must raise rather than substitute a default".
    check_scale_unit:
        Raise when a project's ``scale_unit`` disagrees with its archetype's.

    Returns
    -------
    A new frame with the same index and row count as `cells`, plus
    :data:`ADDED_COLUMNS`.  ``out.attrs[ATTR_KEY]`` holds the :class:`ShockResolution`.
    """
    _require_columns(cells, _REQUIRED_CELL_COLUMNS, "cells")
    _check_mode("unknown_archetypes", unknown_archetypes)
    _check_mode("missing_office_sqm_per_seat", missing_office_sqm_per_seat)
    if monte_carlo and rng is None:
        raise ValueError("monte_carlo=True requires an explicit numpy Generator via rng=")

    year = int(year)
    n = len(cells)
    out = cells.copy(deep=True)
    flags = {str(k): set(v) for k, v in (project_flags or {}).items()}

    crs = crs_metric or geo.city_metric_crs(params)
    cell_points = gpd.GeoSeries(
        gpd.points_from_xy(cells["lon"].to_numpy(), cells["lat"].to_numpy()),
        crs=geo.GEOGRAPHIC_CRS,
        index=cells.index,
    )
    cell_points_m = geo.to_metric(cell_points, crs)
    cell_xy = np.column_stack(
        [cell_points_m.x.to_numpy(dtype=float), cell_points_m.y.to_numpy(dtype=float)]
    )
    cbd_m = geo.to_metric(_cbd_point(params, cbd), crs)
    cbd_xy = np.array([cbd_m.x, cbd_m.y], dtype=float)

    h3_values = cells["h3"].astype(str).to_numpy()
    row_of_h3 = {value: index for index, value in enumerate(h3_values)}

    accumulators = _Accumulators(n)
    employment_effects: list[EmploymentEffect] = []
    network_effects: list[NetworkEffect] = []
    field_effects: list[FieldEffect] = []
    supply_effects: list[SupplyEffect] = []
    demand_effects: list[FloorspaceDemandEffect] = []
    household_effects: list[HouseholdDemand] = []
    weights: dict[str, float] = {}

    if len(projects):
        _require_columns(projects, _REQUIRED_PROJECT_COLUMNS, "projects")
        resolved = activation_weight(
            projects, params, year, force_project_state=force_project_state
        )
        for _, project in resolved.iterrows():
            project_id = str(project["project_id"])
            key = str(project["archetype"])
            try:
                archetype = _Archetype(params, key)
            except MissingParameter:
                if unknown_archetypes == RAISE:
                    raise
                logger.info("ignoring project %s: unknown archetype %r", project_id, key)
                weights[project_id] = 0
                continue

            if check_scale_unit and archetype.scale_unit is not None:
                unit = str(project["scale_unit"])
                if unit != archetype.scale_unit:
                    raise ValueError(
                        f"project {project_id}: scale_unit {unit!r} disagrees with "
                        f"{archetype.path}.{KEY_SCALE_UNIT} = {archetype.scale_unit!r}"
                    )

            weight = float(project["activation_weight"])
            weights[project_id] = weight
            if weight == 0:
                continue

            _resolve_one(
                project=project,
                project_id=project_id,
                archetype=archetype,
                params=params,
                year=year,
                weight=weight,
                unit=float(project["scale_value"]),
                flags=flags.get(project_id, frozenset()),
                cell_points_m=cell_points_m,
                cell_xy=cell_xy,
                cbd_xy=cbd_xy,
                row_of_h3=row_of_h3,
                h3_values=h3_values,
                crs=crs,
                income_index=income_index,
                monte_carlo=monte_carlo,
                rng=rng,
                missing_office_sqm_per_seat=missing_office_sqm_per_seat,
                accumulators=accumulators,
                employment_effects=employment_effects,
                network_effects=network_effects,
                field_effects=field_effects,
                supply_effects=supply_effects,
                demand_effects=demand_effects,
                household_effects=household_effects,
            )

    fields, cap_hit = compose_fields(accumulators.fields, params)
    diagnostics = _cap_diagnostics(cap_hit, params)

    for target, column in _FIELD_TARGET_COLUMNS.items():
        out[column] = fields[target]
    out[COL_FIELD_CAP_HIT] = cap_hit
    out[COL_JOBS_PERMANENT] = accumulators.jobs_permanent
    out[COL_JOBS_CONSTRUCTION] = accumulators.jobs_construction
    out[COL_JOBS_BY_SECTOR] = list(accumulators.jobs_by_sector)
    out[COL_EFFECTIVE_HOUSEHOLDS] = accumulators.effective_households
    out[COL_HOUSEHOLDS_BY_BAND] = list(accumulators.households_by_band)
    out[COL_DORMITORY_WORKERS] = accumulators.dormitory_workers
    out[COL_FLOORSPACE_DEMAND] = accumulators.floorspace_demand
    out[COL_DELTA_CAPACITY] = accumulators.delta_capacity
    out[COL_DELTA_FLOORSPACE] = accumulators.delta_floorspace

    out.attrs[ATTR_KEY] = ShockResolution(
        year=year,
        employment=tuple(employment_effects),
        network=tuple(network_effects),
        fields=tuple(field_effects),
        supply=tuple(supply_effects),
        floorspace_demand=tuple(demand_effects),
        households=tuple(household_effects),
        weights=weights,
        diagnostics=diagnostics,
    )
    return out


def _cbd_point(params: Params, cbd: Point | tuple[float, float] | None) -> Point:
    if isinstance(cbd, Point):
        return cbd
    if cbd is not None:
        lat, lon = cbd
        return Point(float(lon), float(lat))
    node = params.city_config.get("cbd_point")
    if not isinstance(node, Mapping) or "lat" not in node or "lon" not in node:
        raise MissingParameter(
            "city config has no 'cbd_point'; the Section 9.3 directional wedge needs one "
            "(or pass cbd= explicitly)"
        )
    return Point(float(node["lon"]), float(node["lat"]))


def _cap_diagnostics(cap_hit: np.ndarray, params: Params) -> dict[str, Any]:
    """Section 9.4: "Log the number of cells that hit the cap; if more than 2% do ..."."""
    warn_share = float(params.value(P_CAP_WARN_SHARE))
    total = int(cap_hit.size)
    count = int(np.count_nonzero(cap_hit))
    share = (count / total) if total else 0
    warning = bool(share > warn_share)
    if count:
        logger.info("Section 9.4 field cap engaged in %d of %d cells", count, total)
    if warning:
        logger.warning(
            "a share %.4f of cells hit the Section 9.4 field cap, above the %s "
            "threshold of %.4f — the field parameters are wrong",
            share,
            P_CAP_WARN_SHARE,
            warn_share,
        )
    return {
        "cap_low": float(params.value(P_CAP_LOW)),
        "cap_high": float(params.value(P_CAP_HIGH)),
        "cap_hit_cells": count,
        "cap_hit_share": share,
        "cap_warn_share": warn_share,
        "cap_warning": warning,
    }


class _Accumulators:
    """Per-cell running totals; plain arrays so the resolver stays allocation-light."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.fields = {target: np.zeros(n, dtype=float) for target in FIELD_TARGETS}
        self.jobs_permanent = np.zeros(n, dtype=float)
        self.jobs_construction = np.zeros(n, dtype=float)
        self.jobs_by_sector = np.zeros((n, len(SECTORS)), dtype=float)
        self.effective_households = np.zeros(n, dtype=float)
        self.households_by_band = np.zeros((n, len(INCOME_BANDS)), dtype=float)
        self.dormitory_workers = np.zeros(n, dtype=float)
        self.floorspace_demand = np.zeros(n, dtype=float)
        self.delta_capacity = np.zeros(n, dtype=float)
        self.delta_floorspace = np.zeros(n, dtype=float)


def _resolve_one(
    *,
    project: pd.Series,
    project_id: str,
    archetype: _Archetype,
    params: Params,
    year: int,
    weight: float,
    unit: float,
    flags: Iterable[str],
    cell_points_m: gpd.GeoSeries,
    cell_xy: np.ndarray,
    cbd_xy: np.ndarray,
    row_of_h3: Mapping[str, int],
    h3_values: np.ndarray,
    crs: str,
    income_index: float,
    monte_carlo: bool,
    rng: Any,
    missing_office_sqm_per_seat: str,
    accumulators: _Accumulators,
    employment_effects: list[EmploymentEffect],
    network_effects: list[NetworkEffect],
    field_effects: list[FieldEffect],
    supply_effects: list[SupplyEffect],
    demand_effects: list[FloorspaceDemandEffect],
    household_effects: list[HouseholdDemand],
) -> None:
    """Section 9.2, steps 1-6, for one project.  Step 7 (cascade) is Module 9's."""
    flags = set(flags)
    geometry = _geometry(project["geom"])
    geometry_m = geo.to_metric(geometry, crs)
    distance_m = cell_points_m.distance(geometry_m).to_numpy(dtype=float)

    row = _site_row(geometry_m, cell_points_m, row_of_h3, project)
    cell = str(h3_values[row])

    open_year = int(np.floor(float(project["open_year"])))
    construction_start = int(np.floor(float(project["construction_start_year"])))
    ramp_years = _ramp_years(params, archetype)

    # ---------------------------------------------------------------- 1. employment
    employment = archetype.employment
    typology = archetype.housing_typology
    jobs = 0.0
    sector = None
    if employment is not None:
        per_unit = _leaf(
            params, archetype.employment_path("permanent_per_unit"), monte_carlo=monte_carlo, rng=rng
        )
        jobs = unit * per_unit * weight
        sector = SECTORS.index(str(employment["sector"]))
        wage = project.get("median_wage_inr_mo")
        if wage is None or (isinstance(wage, float) and np.isnan(wage)):
            wage = _leaf(
                params,
                archetype.employment_path("median_wage_inr_mo"),
                monte_carlo=monte_carlo,
                rng=rng,
            )
        wage = float(wage)
        radius = _leaf(
            params,
            archetype.employment_path("residential_capture_radius_m"),
            monte_carlo=monte_carlo,
            rng=rng,
        )
        dormitory_share = 0.0
        if "dormitory_share" in employment:
            dormitory_share = _leaf(
                params,
                archetype.employment_path("dormitory_share"),
                monte_carlo=monte_carlo,
                rng=rng,
            )

        employment_effects.append(
            EmploymentEffect(
                cell=cell,
                sector=sector,
                jobs=jobs,
                median_wage_inr_mo=wage,
                start_year=open_year,
                ramp_years=ramp_years,
                capture_radius_m=radius,
                dormitory_share=dormitory_share,
                project_id=project_id,
            )
        )
        accumulators.jobs_permanent[row] += jobs
        accumulators.jobs_by_sector[row, sector] += jobs

        # ------------------------------------------------------------ 5 (9.5) routing
        ownership = 1.0
        if "ownership_demand_share" in typology:
            ownership = _leaf(
                params,
                f"{archetype.path}.{KEY_HOUSING_TYPOLOGY}.ownership_demand_share",
                monte_carlo=monte_carlo,
                rng=rng,
            )
        households, dormitory_workers = route_households(
            jobs,
            sector,
            params,
            dormitory_share=dormitory_share,
            ownership_demand_share=ownership,
        )
        band, income = wage_band(wage, params, income_index=income_index)
        household_effects.append(
            HouseholdDemand(
                cell=cell,
                band=band,
                effective_households=households,
                dormitory_workers=dormitory_workers,
                household_income_inr_mo=income,
                capture_radius_m=radius,
                start_year=open_year,
                ramp_years=ramp_years,
                project_id=project_id,
            )
        )
        accumulators.effective_households[row] += households
        accumulators.households_by_band[row, band] += households
        accumulators.dormitory_workers[row] += dormitory_workers

        # ------------------------------------------------ construction employment
        if "construction_peak_per_unit" in employment:
            peak = _leaf(
                params,
                archetype.employment_path("construction_peak_per_unit"),
                monte_carlo=monte_carlo,
                rng=rng,
            )
            retention = _leaf(
                params,
                archetype.employment_path("construction_local_retention"),
                monte_carlo=monte_carlo,
                rng=rng,
            )
            construction_jobs = unit * peak * retention * weight
            construction_years = _construction_years(params, archetype)
            employment_effects.append(
                EmploymentEffect(
                    cell=cell,
                    sector=CONSTRUCTION_SECTOR,
                    jobs=construction_jobs,
                    median_wage_inr_mo=wage,
                    start_year=construction_start,
                    ramp_years=ramp_years,
                    capture_radius_m=radius,
                    is_construction=True,
                    duration_years=construction_years,
                    project_id=project_id,
                )
            )
            accumulators.jobs_construction[row] += construction_jobs
            accumulators.jobs_by_sector[row, CONSTRUCTION_SECTOR] += construction_jobs

    # 2. Induced service employment is NOT emitted here (Section 9.2 step 2).

    # ------------------------------------------------------------------- 3. network
    network = archetype.get(KEY_NETWORK_EFFECT)
    if isinstance(network, Mapping):
        kind = str(network.get(KEY_TYPE, NETWORK_NONE))
        if kind != NETWORK_NONE:
            speed = network.get(KEY_SPEED_KMH)
            network_effects.append(
                NetworkEffect(
                    kind=kind,
                    geometry=geometry,
                    stations=None,
                    speed_kmh=float(speed[KEY_VALUE]) if isinstance(speed, Mapping) else float("nan"),
                    open_year=open_year,
                    project_id=project_id,
                )
            )

    # -------------------------------------------------------------------- 4. fields
    _resolve_fields(
        archetype=archetype,
        params=params,
        year=year,
        weight=weight,
        project_id=project_id,
        flags=flags,
        geometry=geometry,
        geometry_m=geometry_m,
        distance_m=distance_m,
        cell_xy=cell_xy,
        cbd_xy=cbd_xy,
        open_year=open_year,
        construction_start=construction_start,
        announced_year=int(np.floor(float(project["announced_year"])))
        if "announced_year" in project.index
        else open_year,
        monte_carlo=monte_carlo,
        rng=rng,
        accumulators=accumulators,
        field_effects=field_effects,
    )

    # ---------------------------------------------------------------------- 5. land
    if bool(archetype.get(KEY_STERILISES_LAND)):
        land_take = _leaf(
            params, f"{archetype.path}.{KEY_LAND_TAKE}", monte_carlo=monte_carlo, rng=rng
        )
        delta_capacity = -unit * land_take * weight
        supply_effects.append(
            SupplyEffect(
                cell=cell,
                delta_floorspace_sqm=0,
                delta_capacity_sqm=delta_capacity,
                start_year=open_year,
                project_id=project_id,
            )
        )
        accumulators.delta_capacity[row] += delta_capacity

    # ---------------------------------------------------------- 6. floorspace demand
    if employment is not None:
        if "dormitory_sqm_per_worker" in typology:
            per_worker = _leaf(
                params,
                f"{archetype.path}.{KEY_HOUSING_TYPOLOGY}.dormitory_sqm_per_worker",
                monte_carlo=monte_carlo,
                rng=rng,
            )
            share = 0.0
            if "dormitory_share" in employment:
                share = _leaf(
                    params,
                    archetype.employment_path("dormitory_share"),
                    monte_carlo=monte_carlo,
                    rng=rng,
                )
            sqm = jobs * share * per_worker
            demand_effects.append(
                FloorspaceDemandEffect(
                    cell=cell,
                    use=USE_DORMITORY,
                    sqm=sqm,
                    start_year=open_year,
                    ramp_years=ramp_years,
                    project_id=project_id,
                )
            )
            accumulators.floorspace_demand[row] += sqm

        if sector == OFFICE_SECTOR:
            per_seat = _optional_node(params, P_OFFICE_SQM_PER_SEAT)
            if per_seat is None:
                if missing_office_sqm_per_seat == RAISE:
                    raise MissingParameter(
                        f"{P_OFFICE_SQM_PER_SEAT} is null in behaviour.yaml, so the "
                        "Section 9.2 step 6 office floorspace demand for project "
                        f"{project_id} cannot be resolved. The YAML's own comment says "
                        "Module 5 must raise rather than substitute a default; pass "
                        "missing_office_sqm_per_seat='ignore' to skip it instead."
                    )
                logger.info(
                    "skipping office floorspace demand for %s: %s is null",
                    project_id,
                    P_OFFICE_SQM_PER_SEAT,
                )
            else:
                sqm = jobs * float(params.value(P_OFFICE_SQM_PER_SEAT))
                demand_effects.append(
                    FloorspaceDemandEffect(
                        cell=cell,
                        use=USE_OFFICE,
                        sqm=sqm,
                        start_year=open_year,
                        ramp_years=ramp_years,
                        project_id=project_id,
                    )
                )
                accumulators.floorspace_demand[row] += sqm

    # 7. Cascade — Module 9 (`ufe/layers/cascade.py`), not this layer.


def _site_row(
    geometry_m: BaseGeometry,
    cell_points_m: gpd.GeoSeries,
    row_of_h3: Mapping[str, int],
    project: pd.Series,
) -> int:
    """The row of the cell a project sits in.

    Uses the project's declared ``cell`` / ``h3`` column when it names a cell present in
    the frame, and otherwise the nearest cell centroid **in the metric CRS** (never in
    degrees).  Deterministic: ties break on the lowest row index, which `argmin` gives.
    """
    for column in ("cell", "h3"):
        if column in project.index:
            key = project[column]
            if isinstance(key, str) and key in row_of_h3:
                return int(row_of_h3[key])
    anchor = geometry_m.representative_point()
    return int(np.argmin(cell_points_m.distance(anchor).to_numpy(dtype=float)))


def _resolve_fields(
    *,
    archetype: _Archetype,
    params: Params,
    year: int,
    weight: float,
    project_id: str,
    flags: set[str],
    geometry: BaseGeometry,
    geometry_m: BaseGeometry,
    distance_m: np.ndarray,
    cell_xy: np.ndarray,
    cbd_xy: np.ndarray,
    open_year: int,
    construction_start: int,
    announced_year: int,
    monte_carlo: bool,
    rng: Any,
    accumulators: _Accumulators,
    field_effects: list[FieldEffect],
) -> None:
    """Section 9.2 step 4 and Sections 9.3 / 9.4 for one project."""
    multiplier = _premium_multiplier(archetype, params, flags, monte_carlo=monte_carlo, rng=rng)
    wedge = None  # computed lazily; only wedge-flagged fields need it

    for category in FIELD_CATEGORIES:
        entries = _as_entries(archetype.get(category))
        if not entries:
            continue
        is_construction = category == CATEGORY_CONSTRUCTION_PENALTY
        if is_construction and not (construction_start <= year < open_year):
            continue
        start_year = construction_start if is_construction else announced_year
        end_year = open_year if is_construction else None
        scale = multiplier if category == CATEGORY_PREMIUM else 1

        # group by target, then apply the Section 9.3 exclusive-band rule
        grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                continue
            required = entry.get(KEY_APPLIES_WHEN)
            if required is not None and str(required) not in flags:
                continue
            grouped.setdefault(str(entry.get(KEY_TARGET, TARGET_ALL)), []).append(
                (index, entry)
            )

        for target, bands in grouped.items():
            ordered = sorted(bands, key=lambda item: float(item[1][KEY_MAX_M]))
            band_group = f"{project_id}:{category}:{target}"
            chosen = np.full(distance_m.shape, -1, dtype=int)
            for position, (_, entry) in enumerate(ordered):
                inside = distance_m <= float(entry[KEY_MAX_M])
                chosen = np.where((chosen < 0) & inside, position, chosen)

            total = np.zeros(distance_m.shape, dtype=float)
            for position, (index, entry) in enumerate(ordered):
                path = (
                    f"{archetype.path}.{category}.{KEY_VALUE}"
                    if len(entries) == 1 and isinstance(archetype.get(category), Mapping)
                    else f"{archetype.path}.{category}.{index}.{KEY_VALUE}"
                )
                magnitude = _leaf(params, path, monte_carlo=monte_carlo, rng=rng) * scale
                max_m = float(entry[KEY_MAX_M])
                decay = str(entry.get(KEY_DECAY, DEFAULT_DECAY))
                use_wedge = bool(
                    entry.get(KEY_DIRECTIONAL_WEDGE, archetype.directional_wedge)
                )

                values = field_decay(decay, magnitude, distance_m, max_m, params)
                if use_wedge:
                    if wedge is None:
                        anchor = geometry_m.centroid
                        wedge = wedge_factor(
                            np.array([anchor.x, anchor.y], dtype=float),
                            cell_xy,
                            cbd_xy,
                            params,
                        )
                    values = values * wedge
                total = total + np.where(chosen == position, values, 0)

                field_effects.append(
                    FieldEffect(
                        origin=geometry,
                        target=target,
                        max_m=max_m,
                        magnitude=magnitude * weight,
                        decay=decay,
                        start_year=start_year,
                        end_year=end_year,
                        project_id=project_id,
                        band_group=band_group,
                        directional_wedge=use_wedge,
                    )
                )

            total = total * weight
            for applied in FIELD_TARGETS if target == TARGET_ALL else (target,):
                if applied not in accumulators.fields:
                    raise ValueError(
                        f"{archetype.path}.{category}: unknown field target {applied!r}; "
                        f"Section 9.1 defines {list(FIELD_TARGETS) + [TARGET_ALL]}"
                    )
                accumulators.fields[applied] += total


def _premium_multiplier(
    archetype: _Archetype,
    params: Params,
    flags: set[str],
    *,
    monte_carlo: bool,
    rng: Any,
) -> float:
    """Product of the ``premium_multipliers`` entries whose flag the project carries."""
    node = archetype.get(KEY_PREMIUM_MULTIPLIERS)
    if not isinstance(node, Mapping):
        return 1
    multiplier = 1.0
    for flag in sorted(flags & set(node)):
        multiplier *= _leaf(
            params,
            f"{archetype.path}.{KEY_PREMIUM_MULTIPLIERS}.{flag}.{KEY_VALUE}",
            monte_carlo=monte_carlo,
            rng=rng,
        )
    return multiplier
