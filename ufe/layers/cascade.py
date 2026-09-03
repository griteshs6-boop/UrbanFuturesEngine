"""Module 10 — cascade (spec Section 14).

An *anchor* project of an archetype that declares a ``cascade:`` block spawns ancillary
employment nearby: a data centre pulls logistics, an electronics plant pulls light
manufacturing, an automotive OEM pulls two tiers of component suppliers. This module turns
that into :class:`ufe.layers.l2_shocks.EmploymentEffect` objects — the same effect type
Layer 2 emits — so the rest of the engine consumes cascade output through the existing
Section 9.1 vocabulary and nothing downstream needs to know a job was cascaded.

Structure
---------
``14.1 injection``  :func:`inject_cascade` — one generation of injection.
``14.2 generation cap``  :func:`resolve_cascades` — the fixed-point loop, capped.
``14.3 tiering``  handled by :func:`cascade_entries`: an archetype's ``cascade:`` may be a
    single mapping or a *list* of mappings. Every entry in the list hangs off the same
    anchor at the same generation. The tiers are never chained to each other.

Everything here is pure: no frame is mutated, no module-level state, and every draw takes
an explicit ``numpy.random.Generator``. No numeric literal appears below beyond ``0``/``1``
and array indices — the mechanism parameters live in ``config/params/cascade.yaml`` and the
per-archetype ratios, radii, lags and targets in ``config/params/archetypes.yaml``.

THE BLOCKER (read this before running cascade on the real config)
-----------------------------------------------------------------
``config/params/archetypes.yaml`` ships **3 of the 22** archetypes the spec references,
because Section 4.3 sources their values from a section of
``Urban_Futures_Engine_Specification.docx`` that was never supplied. Cascade must resolve
``cascade.target_archetype`` to read the injected jobs' ``sector`` and
``median_wage_inr_mo``, and **both shipped cascade targets are missing**:

    data_centre         -> logistics_park        (absent)
    electronics_assembly -> manufacturing_light  (absent)

So on the real config cascade cannot run for *any* anchor. This module does not paper over
that: :func:`inject_cascade` raises :class:`MissingArchetypeError`, naming the missing key,
the anchor archetype that referenced it and the parameter path. Tests supply the missing
archetypes through a ``tmp_path`` params overlay (the ``test_l2_shocks`` pattern); nothing
under ``config/`` is edited and no magnitude is invented.

A second, independent gap: ``cascade.firm_logit.coefficients.*`` are all ``null`` (Section
12.7 is not specified in the supplied spec), so :func:`ufe.layers.l5_allocation.allocate_firms`
raises ``MissingParameter``. Cascade uses that landed function as its allocator rather than
inventing a spread rule; the tests overlay coefficients the same way.

Ambiguities resolved here, reported rather than invented
--------------------------------------------------------
* Section 14.1's candidate filter needs ``freight_access_i`` — not a landed ``cells``
  column and undefined anywhere in the spec, exactly as Section 12.7 leaves it. It is an
  explicit ``freight_access=`` argument. Supplying it requires
  ``cascade.candidate_filter.freight_access_threshold`` to be non-null (it ships ``null``),
  and omitting it drops the term from the filter with a logged note. The same series is
  forwarded to the firm logit.
* Section 14.2 caps *generation*, but the spec never says where a generation-2 anchor
  sits or how big it is. The reading taken: each injected candidate cell becomes its own
  anchor for the next generation, sited at that cell's centroid, carrying its allocated
  share of the ancillary jobs, its own ``p`` and its own ``open_year``. That is the only
  reading under which "cascades may themselves have cascades" means anything.
* Section 14.1 says ``lag = uniform_int(C.lag_years[0], C.lag_years[1])`` but
  ``archetypes.yaml`` encodes ``lag_years`` in the Section 4.1 range form. ``monte_carlo``
  draws via ``Params.sample``; the deterministic path takes ``.value`` (the midpoint).
  Either way the lag is rounded to a whole year, because ``start_year`` is an integer
  calendar year (Section 0.3).
* ``zone_class in {industrial, mixed}`` uses a vocabulary the landed ``cells`` schema does
  not have (it spells it ``ind``). The mapping is data, in
  ``cascade.candidate_filter.zone_class_map``, and an unmapped name raises.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ufe import geo
from ufe.errors import MissingArchetypeError, MissingParameter
from ufe.layers.l2_shocks import EmploymentEffect
from ufe.params import Params
from ufe.store.schemas import SECTORS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# parameter paths and YAML keys — no magnitudes, only names
# --------------------------------------------------------------------------------------

CASCADE = "cascade"
ARCH = "archetypes"
DEFAULTS = f"{ARCH}._defaults"

P_MULTIPLIER = f"{CASCADE}.p_multiplier"
P_MAX_GENERATION = f"{CASCADE}.max_generation"
P_FILTER = f"{CASCADE}.candidate_filter"
P_FILTER_ZONE_CLASSES = f"{P_FILTER}.zone_classes"
P_FILTER_ZONE_CLASS_MAP = f"{P_FILTER}.zone_class_map"
P_FILTER_REQUIRE_POWER = f"{P_FILTER}.require_util_power"
P_FILTER_FREIGHT_THRESHOLD = f"{P_FILTER}.freight_access_threshold"
P_DEFAULT_RAMP_YEARS = f"{DEFAULTS}.operational_ramp_years"

KEY_CASCADE = "cascade"
KEY_RATIO = "ratio"
KEY_RADIUS_M = "radius_m"
KEY_LAG_YEARS = "lag_years"
KEY_TARGET_ARCHETYPE = "target_archetype"
KEY_EMPLOYMENT = "employment"
KEY_SECTOR = "sector"
KEY_WAGE = "median_wage_inr_mo"
KEY_CAPTURE_RADIUS = "residential_capture_radius_m"
KEY_RAMP_YEARS = "operational_ramp_years"
KEY_VALUE = "value"

COL_H3 = "h3"
COL_ZONE_CLASS = "zone_class"
COL_UTIL_POWER = "util_power"
COL_LAT = "lat"
COL_LON = "lon"

#: Anchor-frame columns :func:`inject_cascade` needs.  ``generation`` is optional and
#: defaults to ``0`` (a real, ingested project).
REQUIRED_ANCHOR_COLUMNS: tuple[str, ...] = (
    "project_id",
    "archetype",
    "geom",
    "p_completion",
    "open_year",
    "anchor_jobs",
)

COL_GENERATION = "generation"

#: Reasons a cascade entry produced nothing.  Section 14.1: zero candidates is a logged
#: warning and no injected jobs — never a crash and never a fallback onto the anchor cell.
SKIP_NO_CANDIDATES = "no_candidate_cells"
SKIP_GENERATION_CAP = "generation_cap"
SKIP_NO_CASCADE_BLOCK = "no_cascade_block"
SKIP_ZERO_JOBS = "zero_ancillary_jobs"

ZERO = 0
ONE = 1


# `MissingArchetypeError` now lives in the shared hierarchy in `ufe/errors.py`
# (CONTRACT.md: "Custom exceptions live in ufe/errors.py"). It is imported above and
# re-exported here, so `from ufe.layers.cascade import MissingArchetypeError` keeps
# working. It remains THE Module 10 blocker and is
# deliberately fatal: Section 14.1 reads the injected jobs' `sector` and
# `median_wage_inr_mo` off the target archetype, and there is no honest default for either.


# --------------------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeSkip:
    """One cascade entry that produced no jobs, and why (Section 14.1 / 14.2)."""

    anchor_project_id: str
    anchor_archetype: str
    tier: int
    generation: int
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CascadeAnchor:
    """One node of the cascade tree: a real project, or an injected child of one."""

    project_id: str
    archetype: str
    geom: BaseGeometry
    p_completion: float
    open_year: int
    anchor_jobs: float
    generation: int = ZERO
    parent_project_id: str | None = None
    cell: str | None = None


@dataclass(frozen=True)
class CascadeInjection:
    """Everything one ``(anchor, cascade entry)`` pair emitted (Section 14.1)."""

    anchor_project_id: str
    anchor_archetype: str
    target_archetype: str
    tier: int
    generation: int
    ancillary_jobs: float
    lag_years: int
    start_year: int
    p: float
    radius_m: float
    ratio: float
    candidate_cells: tuple[str, ...]
    effects: tuple[EmploymentEffect, ...]
    children: tuple[CascadeAnchor, ...]


@dataclass(frozen=True)
class CascadeResult:
    """The full cascade for a pipeline, all generations (Sections 14.1–14.3)."""

    injections: tuple[CascadeInjection, ...] = ()
    effects: tuple[EmploymentEffect, ...] = ()
    skipped: tuple[CascadeSkip, ...] = ()
    max_generation_reached: int = ZERO
    generations_run: int = ZERO
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s): {', '.join(missing)}")


def _archetype_node(params: Params, key: str) -> Mapping[str, Any]:
    try:
        node = params.get(f"{ARCH}.{key}")
    except MissingParameter:
        node = None
    if not isinstance(node, Mapping):
        raise MissingArchetypeError(
            f"archetypes.{key} is not defined in config/params/archetypes.yaml. "
            f"{key!r} is one of the 22 archetypes Section 4.3 says the file must contain; "
            "only 3 are present because Section 4.3 sources their values from a section of "
            "Urban_Futures_Engine_Specification.docx that was never supplied. Module 10 "
            "refuses to invent a sector or a wage for it (spec Section 0.1 rule 9)."
        )
    return node


def _leaf_value(
    params: Params, path: str, *, monte_carlo: bool, rng: np.random.Generator | None
) -> float:
    """Deterministic ``.value``, or a Monte Carlo draw from ``low..high``."""
    if monte_carlo:
        if rng is None:
            raise ValueError("monte_carlo=True requires an explicit rng (CONTRACT rule 5)")
        return float(params.sample(path, rng))
    return float(params.value(path))


def _geometry(value: Any) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, str):
        return shapely_wkt.loads(value)
    raise TypeError(f"cannot interpret {value!r} as a geometry")


def _sector_index(name: str, *, target: str) -> int:
    if name not in SECTORS:
        raise MissingArchetypeError(
            f"archetypes.{target}.employment.sector is {name!r}, which is not one of the "
            f"landed sector vocabulary {list(SECTORS)}."
        )
    return SECTORS.index(name)


# --------------------------------------------------------------------------------------
# 14.3 tiering — one archetype, one or many cascade entries
# --------------------------------------------------------------------------------------


def cascade_entries(params: Params, archetype: str) -> tuple[Mapping[str, Any], ...]:
    """Every cascade entry declared by `archetype` (spec Section 14.3).

    ``cascade:`` may be ``null`` (no cascade), a single mapping (one tier), or a list of
    mappings (the automotive-OEM two-tier case). Every entry returned hangs off the SAME
    anchor at the SAME generation — Section 14.3: "Do not chain them."
    """
    node = _archetype_node(params, archetype).get(KEY_CASCADE)
    if node is None:
        return ()
    if isinstance(node, Mapping):
        return (node,)
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        entries = tuple(e for e in node if isinstance(e, Mapping))
        if len(entries) != len(node):
            raise ValueError(f"archetypes.{archetype}.cascade has a non-mapping tier entry")
        return entries
    raise ValueError(
        f"archetypes.{archetype}.cascade must be null, a mapping, or a list of mappings"
    )


# --------------------------------------------------------------------------------------
# 14.1 the candidate-cell filter
# --------------------------------------------------------------------------------------


def _allowed_zone_classes(params: Params) -> tuple[str, ...]:
    wanted = list(params.get(P_FILTER_ZONE_CLASSES) or ())
    mapping = params.get(P_FILTER_ZONE_CLASS_MAP) or {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{P_FILTER_ZONE_CLASS_MAP} must be a mapping")
    unmapped = [name for name in wanted if name not in mapping]
    if unmapped:
        raise MissingParameter(
            f"{P_FILTER_ZONE_CLASS_MAP} has no entry for {unmapped}; Section 14.1 spells the "
            "industrial zone class differently from the landed cells vocabulary and Module 10 "
            "will not guess the mapping."
        )
    return tuple(str(mapping[name]) for name in wanted)


def candidate_cells(
    cells: pd.DataFrame,
    params: Params,
    *,
    anchor_geom: BaseGeometry,
    radius_m: float,
    crs_metric: str,
    freight_access: pd.Series | Sequence[float] | None = None,
) -> pd.DataFrame:
    """The Section 14.1 candidate set for one cascade entry.

    ``cells within C.radius_m of anchor AND zone_class in {industrial, mixed} AND
    freight_access_i > threshold AND util_power == 1``

    Returns a NEW frame (a filtered copy). Distances are computed in `crs_metric`, never in
    degrees (Section 0.3). May be empty — the caller handles that per Section 14.1.
    """
    _require_columns(cells, (COL_H3, COL_LAT, COL_LON, COL_ZONE_CLASS), "cells")

    frame = cells.copy(deep=True)
    if freight_access is not None:
        frame = frame.assign(_freight_access=np.asarray(freight_access, dtype=float))

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(frame[COL_LON], frame[COL_LAT])],
        index=frame.index,
        crs=geo.GEOGRAPHIC_CRS,
    )
    origin = geo.to_metric(anchor_geom, crs_metric)
    distance = geo.to_metric(points, crs_metric).distance(origin).to_numpy(dtype=float)
    mask = distance <= float(radius_m)

    allowed = _allowed_zone_classes(params)
    mask &= frame[COL_ZONE_CLASS].isin(allowed).to_numpy()

    if bool(params.get(P_FILTER_REQUIRE_POWER)):
        if COL_UTIL_POWER not in frame.columns:
            raise ValueError(
                f"{P_FILTER_REQUIRE_POWER} is true but cells has no {COL_UTIL_POWER!r} column"
            )
        mask &= frame[COL_UTIL_POWER].to_numpy().astype(float) == ONE

    if freight_access is not None:
        threshold = params.get(P_FILTER_FREIGHT_THRESHOLD)
        if isinstance(threshold, Mapping):
            threshold = threshold.get(KEY_VALUE)
        if threshold is None:
            raise MissingParameter(
                f"{P_FILTER_FREIGHT_THRESHOLD} is null in config/params/cascade.yaml. "
                "Section 14.1 writes the gate as `freight_access_i > threshold` but never "
                "states the threshold, so it ships null. Supply it before passing "
                "freight_access=, or omit freight_access= to drop the term."
            )
        mask &= frame["_freight_access"].to_numpy(dtype=float) > float(threshold)
    else:
        logger.info(
            "cascade candidate filter: no freight_access supplied, dropping the Section 14.1 "
            "`freight_access_i > threshold` term (freight_access is not a landed cells column)"
        )

    out = frame.loc[mask].drop(columns=["_freight_access"], errors="ignore")
    return out.copy(deep=True)


# --------------------------------------------------------------------------------------
# allocation — Section 12.7's firm logit, used, not reinvented
# --------------------------------------------------------------------------------------

Allocator = Callable[..., np.ndarray]


def firm_logit_shares(
    candidates: pd.DataFrame,
    params: Params,
    *,
    sector: int,
    jobs: float,
    freight_access: pd.Series | Sequence[float] | None = None,
) -> np.ndarray:
    """Spread `jobs` across `candidates` with the Section 12.7 firm logit.

    Delegates to :func:`ufe.layers.l5_allocation.allocate_firms` — Section 14.1 says
    "allocate ancillary_jobs across candidates via the FIRM logit (Section 12.7)", so this
    calls that landed function and recovers the shares by differencing the sector column.
    It raises ``MissingParameter`` on the shipped config because
    ``cascade.firm_logit.coefficients`` are null; that is intentional (see module docstring).
    """
    from ufe.layers.l5_allocation import allocate_firms  # lazy: keeps the import graph flat

    before = np.asarray(
        [row[sector] for row in candidates["jobs_by_sector"]], dtype=float
    )
    allocated = allocate_firms(
        candidates,
        params,
        jobs_by_sector={SECTORS[sector]: float(jobs)},
        freight_access=freight_access,
    )
    after = np.asarray([row[sector] for row in allocated["jobs_by_sector"]], dtype=float)
    return after - before


# --------------------------------------------------------------------------------------
# 14.1 injection
# --------------------------------------------------------------------------------------


def _target_employment(params: Params, target: str) -> Mapping[str, Any]:
    node = _archetype_node(params, target).get(KEY_EMPLOYMENT)
    if not isinstance(node, Mapping):
        raise MissingArchetypeError(
            f"archetypes.{target}.employment is absent or null, so Section 14.1 cannot read "
            "the injected jobs' sector or median wage."
        )
    return node


def _ramp_years(params: Params, target: str) -> float:
    node = _archetype_node(params, target).get(KEY_RAMP_YEARS)
    if isinstance(node, Mapping):
        return float(node[KEY_VALUE])
    return float(params.value(P_DEFAULT_RAMP_YEARS))


def _capture_radius_m(params: Params, target: str, employment: Mapping[str, Any]) -> float:
    node = employment.get(KEY_CAPTURE_RADIUS)
    if isinstance(node, Mapping):
        return float(node[KEY_VALUE])
    raise MissingArchetypeError(
        f"archetypes.{target}.employment.{KEY_CAPTURE_RADIUS} is absent; "
        "EmploymentEffect requires a capture radius and Module 10 will not invent one."
    )


def _anchors_from_frame(anchors: pd.DataFrame) -> list[CascadeAnchor]:
    _require_columns(anchors, REQUIRED_ANCHOR_COLUMNS, "anchors")
    generations = (
        anchors[COL_GENERATION]
        if COL_GENERATION in anchors.columns
        else pd.Series(ZERO, index=anchors.index)
    )
    return [
        CascadeAnchor(
            project_id=str(row.project_id),
            archetype=str(row.archetype),
            geom=_geometry(row.geom),
            p_completion=float(row.p_completion),
            open_year=int(row.open_year),
            anchor_jobs=float(row.anchor_jobs),
            generation=int(generation),
        )
        for row, generation in zip(anchors.itertuples(index=False), generations)
    ]


def inject_cascade(
    cells: pd.DataFrame,
    anchors: pd.DataFrame | Iterable[CascadeAnchor],
    params: Params,
    *,
    crs_metric: str | None = None,
    freight_access: pd.Series | Sequence[float] | None = None,
    monte_carlo: bool = False,
    rng: np.random.Generator | None = None,
    allocator: Allocator | None = None,
) -> tuple[tuple[CascadeInjection, ...], tuple[CascadeSkip, ...]]:
    """ONE generation of Section 14.1 injection. Pure; `cells` is never mutated.

    Returns ``(injections, skips)``. A cascade entry is skipped — logged, never fatal —
    when the anchor's archetype declares no cascade, when the generation cap refuses it
    (Section 14.2), when the candidate set is empty, or when the ancillary job count
    rounds to nothing. It is FATAL when ``target_archetype`` is missing
    (:class:`MissingArchetypeError`), which is the shipped-config case.
    """
    crs = crs_metric or geo.city_metric_crs(params)
    allocate = allocator or firm_logit_shares
    p_multiplier = float(params.value(P_MULTIPLIER))
    max_generation = int(params.value(P_MAX_GENERATION))

    nodes = (
        _anchors_from_frame(anchors)
        if isinstance(anchors, pd.DataFrame)
        else list(anchors)
    )

    injections: list[CascadeInjection] = []
    skips: list[CascadeSkip] = []

    for anchor in nodes:
        # ---- 14.2 the cap, checked before any work is done for this anchor -----------
        if anchor.generation >= max_generation:
            skips.append(
                CascadeSkip(
                    anchor_project_id=anchor.project_id,
                    anchor_archetype=anchor.archetype,
                    tier=ZERO,
                    generation=anchor.generation,
                    reason=SKIP_GENERATION_CAP,
                    detail=(
                        f"generation {anchor.generation} >= {P_MAX_GENERATION}"
                        f"={max_generation}; refusing to cascade further (Section 14.2)"
                    ),
                )
            )
            logger.info(
                "cascade: refusing to cascade %s at generation %d (cap %d)",
                anchor.project_id,
                anchor.generation,
                max_generation,
            )
            continue

        entries = cascade_entries(params, anchor.archetype)
        if not entries:
            skips.append(
                CascadeSkip(
                    anchor_project_id=anchor.project_id,
                    anchor_archetype=anchor.archetype,
                    tier=ZERO,
                    generation=anchor.generation,
                    reason=SKIP_NO_CASCADE_BLOCK,
                    detail=f"archetypes.{anchor.archetype}.cascade is null",
                )
            )
            continue

        for tier, entry in enumerate(entries):
            injection, skip = _inject_one(
                cells,
                params,
                anchor=anchor,
                entry=entry,
                tier=tier,
                crs_metric=crs,
                freight_access=freight_access,
                monte_carlo=monte_carlo,
                rng=rng,
                p_multiplier=p_multiplier,
                allocate=allocate,
                entry_path=f"{ARCH}.{anchor.archetype}.{KEY_CASCADE}",
                tiered=len(entries) > ONE,
            )
            if injection is not None:
                injections.append(injection)
            if skip is not None:
                skips.append(skip)

    return tuple(injections), tuple(skips)


def _entry_path(base: str, tier: int, tiered: bool) -> str:
    return f"{base}.{tier}" if tiered else base


def _inject_one(
    cells: pd.DataFrame,
    params: Params,
    *,
    anchor: CascadeAnchor,
    entry: Mapping[str, Any],
    tier: int,
    crs_metric: str,
    freight_access: pd.Series | Sequence[float] | None,
    monte_carlo: bool,
    rng: np.random.Generator | None,
    p_multiplier: float,
    allocate: Allocator,
    entry_path: str,
    tiered: bool,
) -> tuple[CascadeInjection | None, CascadeSkip | None]:
    """Section 14.1 for one ``(anchor, cascade entry)`` pair."""
    path = _entry_path(entry_path, tier, tiered)

    target = entry.get(KEY_TARGET_ARCHETYPE)
    if not target:
        raise MissingParameter(f"{path}.{KEY_TARGET_ARCHETYPE} is absent")
    target = str(target)

    # Resolving the target is what fails on the shipped config. Do it FIRST, before any
    # geometry work, so the error message is the first thing a caller sees.
    employment = _target_employment(params, target)
    sector_name = employment.get(KEY_SECTOR)
    if sector_name is None:
        raise MissingArchetypeError(
            f"archetypes.{target}.employment.{KEY_SECTOR} is absent (Section 14.1 needs it)"
        )
    sector = _sector_index(str(sector_name), target=target)
    wage = _leaf_value(
        params, f"{ARCH}.{target}.{KEY_EMPLOYMENT}.{KEY_WAGE}", monte_carlo=monte_carlo, rng=rng
    )
    ramp_years = _ramp_years(params, target)
    capture_radius_m = _capture_radius_m(params, target, employment)

    ratio = _leaf_value(params, f"{path}.{KEY_RATIO}", monte_carlo=monte_carlo, rng=rng)
    radius_m = _leaf_value(params, f"{path}.{KEY_RADIUS_M}", monte_carlo=monte_carlo, rng=rng)
    lag = _leaf_value(params, f"{path}.{KEY_LAG_YEARS}", monte_carlo=monte_carlo, rng=rng)
    lag_years = int(round(lag))

    ancillary_jobs = float(anchor.anchor_jobs) * ratio
    p_cascade = float(anchor.p_completion) * p_multiplier
    start_year = int(anchor.open_year) + lag_years

    if not ancillary_jobs > ZERO:
        return None, CascadeSkip(
            anchor_project_id=anchor.project_id,
            anchor_archetype=anchor.archetype,
            tier=tier,
            generation=anchor.generation,
            reason=SKIP_ZERO_JOBS,
            detail=f"anchor_jobs={anchor.anchor_jobs} x ratio={ratio} = {ancillary_jobs}",
        )

    candidates = candidate_cells(
        cells,
        params,
        anchor_geom=anchor.geom,
        radius_m=radius_m,
        crs_metric=crs_metric,
        freight_access=freight_access,
    )

    if candidates.empty:
        # Section 14.1: "log warning, skip cascade -- do NOT silently place it on the
        # anchor cell." No effects, no fallback, no crash.
        logger.warning(
            "cascade: zero candidate cells within %.0f m of anchor %s (archetype %s, tier %d); "
            "skipping the cascade entirely rather than placing %.1f ancillary jobs on the "
            "anchor cell",
            radius_m,
            anchor.project_id,
            anchor.archetype,
            tier,
            ancillary_jobs,
        )
        return None, CascadeSkip(
            anchor_project_id=anchor.project_id,
            anchor_archetype=anchor.archetype,
            tier=tier,
            generation=anchor.generation,
            reason=SKIP_NO_CANDIDATES,
            detail=f"radius_m={radius_m}, ancillary_jobs={ancillary_jobs}",
        )

    candidate_freight = None
    if freight_access is not None:
        candidate_freight = pd.Series(
            np.asarray(freight_access, dtype=float), index=cells.index
        ).loc[candidates.index]

    allocated = np.asarray(
        allocate(
            candidates,
            params,
            sector=sector,
            jobs=ancillary_jobs,
            freight_access=candidate_freight,
        ),
        dtype=float,
    )
    if allocated.shape[ZERO] != len(candidates):
        raise ValueError(
            f"allocator returned {allocated.shape[ZERO]} shares for {len(candidates)} candidates"
        )

    h3s = list(candidates[COL_H3])
    lats = list(candidates[COL_LAT])
    lons = list(candidates[COL_LON])

    effects: list[EmploymentEffect] = []
    children: list[CascadeAnchor] = []
    for index, cell in enumerate(h3s):
        jobs = float(allocated[index])
        if not jobs > ZERO:
            continue
        effects.append(
            EmploymentEffect(
                cell=str(cell),
                sector=sector,
                jobs=jobs,
                median_wage_inr_mo=wage,
                start_year=start_year,
                ramp_years=int(round(ramp_years)),
                capture_radius_m=capture_radius_m,
                project_id=f"{anchor.project_id}::cascade{tier}::g{anchor.generation + ONE}",
            )
        )
        children.append(
            CascadeAnchor(
                project_id=f"{anchor.project_id}::cascade{tier}::{cell}",
                archetype=target,
                geom=Point(float(lons[index]), float(lats[index])),
                p_completion=p_cascade,
                open_year=start_year,
                anchor_jobs=jobs,
                generation=anchor.generation + ONE,
                parent_project_id=anchor.project_id,
                cell=str(cell),
            )
        )

    return (
        CascadeInjection(
            anchor_project_id=anchor.project_id,
            anchor_archetype=anchor.archetype,
            target_archetype=target,
            tier=tier,
            generation=anchor.generation + ONE,
            ancillary_jobs=ancillary_jobs,
            lag_years=lag_years,
            start_year=start_year,
            p=p_cascade,
            radius_m=radius_m,
            ratio=ratio,
            candidate_cells=tuple(str(c) for c in h3s),
            effects=tuple(effects),
            children=tuple(children),
        ),
        None,
    )


# --------------------------------------------------------------------------------------
# 14.2 the generation cap — the fixed-point loop that cannot run away
# --------------------------------------------------------------------------------------


def resolve_cascades(
    cells: pd.DataFrame,
    anchors: pd.DataFrame | Iterable[CascadeAnchor],
    params: Params,
    *,
    crs_metric: str | None = None,
    freight_access: pd.Series | Sequence[float] | None = None,
    monte_carlo: bool = False,
    rng: np.random.Generator | None = None,
    allocator: Allocator | None = None,
) -> CascadeResult:
    """Run the cascade to its fixed point, capped at ``cascade.max_generation``.

    THE TERMINATION ARGUMENT (Section 14.2). Every child produced by
    :func:`inject_cascade` carries ``generation = parent.generation + 1``, and
    :func:`inject_cascade` refuses any anchor with ``generation >= max_generation``. The
    frontier therefore strictly increases in generation each pass and is empty after at
    most ``max_generation`` passes, regardless of how many children each pass produces or
    how large the cascade ratios are. A ratio above 1 (each generation bigger than the
    last) or a self-referential ``target_archetype`` (an archetype whose cascade target is
    itself) still terminates; it just terminates with more jobs. The loop below also
    asserts that the frontier's generation advances, so a future bug that failed to
    increment ``generation`` raises instead of hanging.
    """
    max_generation = int(params.value(P_MAX_GENERATION))
    frontier = (
        _anchors_from_frame(anchors) if isinstance(anchors, pd.DataFrame) else list(anchors)
    )

    all_injections: list[CascadeInjection] = []
    all_skips: list[CascadeSkip] = []
    generations_run = ZERO
    max_reached = max((a.generation for a in frontier), default=ZERO)

    while frontier:
        pass_generation = min(a.generation for a in frontier)
        if pass_generation >= max_generation:
            # Every remaining anchor is at or past the cap; run one final pass purely to
            # record the refusals, then stop.
            _, skips = inject_cascade(
                cells,
                frontier,
                params,
                crs_metric=crs_metric,
                freight_access=freight_access,
                monte_carlo=monte_carlo,
                rng=rng,
                allocator=allocator,
            )
            all_skips.extend(skips)
            break

        injections, skips = inject_cascade(
            cells,
            frontier,
            params,
            crs_metric=crs_metric,
            freight_access=freight_access,
            monte_carlo=monte_carlo,
            rng=rng,
            allocator=allocator,
        )
        all_injections.extend(injections)
        all_skips.extend(skips)
        generations_run += ONE

        children = [child for inj in injections for child in inj.children]
        if children:
            child_generation = min(c.generation for c in children)
            if child_generation <= pass_generation:
                raise ValueError(
                    "cascade generation did not advance: a child was emitted at generation "
                    f"{child_generation} from a frontier at generation {pass_generation}. "
                    "Refusing to loop (Section 14.2)."
                )
            max_reached = max(max_reached, max(c.generation for c in children))
        frontier = children

    effects = tuple(e for inj in all_injections for e in inj.effects)
    diagnostics = {
        "max_generation": max_generation,
        "n_injections": len(all_injections),
        "n_effects": len(effects),
        "n_skipped": len(all_skips),
        "jobs_injected": float(sum(e.jobs for e in effects)),
        "skip_reasons": sorted({s.reason for s in all_skips}),
    }
    return CascadeResult(
        injections=tuple(all_injections),
        effects=effects,
        skipped=tuple(all_skips),
        max_generation_reached=max_reached,
        generations_run=generations_run,
        diagnostics=diagnostics,
    )
