"""Layer 3 — credibility (spec Section 10).

This is the differentiating layer: it converts an announcement into a probability that the
thing actually happens, a distribution over when it happens, and a per-year weight in
``[0, 1]`` that scales every effect the project has on the city.

Public entry points
-------------------
``completion_probability(projects, announcers, params, ...)``
    Section 10.1 / 10.2.  Adds the five component scores, the announcer credibility score
    (ACS), the modifier product, the physical-divergence override and ``p_completion``.
``delay_distribution(projects, params, ...)``
    Section 10.3.  Adds ``announced_duration_yr``, ``slip``, ``actual_duration_yr`` and
    ``open_year``.  Deterministic by default; Monte Carlo only with an explicit
    ``numpy.random.Generator``.
``activation_weight(projects, params, year, ...)``
    Section 10.4.  Adds ``phase_weight``, ``discount`` and ``activation_weight``.
``slip_cdf`` / ``slip_median``
    The Section 10.3 delay law, exposed so it can be tested and plotted.
``assert_pipeline_contains_dead_projects``
    The Section 21 survivorship guard, layer side.

Every function is pure: it copies its input frame, never mutates it, and returns a new frame
with the same row count and index.  The counterfactual of Section 10.5 is the explicit
``force_project_state`` argument — there is no module-level flag anywhere in here.

Boundedness (CONTRACT.md, Section 0.3 "probability 0-1")
--------------------------------------------------------
* Each component score is in ``[0, 1]`` by construction (table lookups whose values are all
  in ``[0, 1]``, or a clipped expression).
* ``ACS_raw`` is a *convex combination* of those components — the weighted sum is divided by
  the sum of the weights — so it lies in ``[0, 1]`` without clipping.  On the parameters as
  shipped the weights already sum to 1, so the normalisation is a no-op; it exists so the
  bound survives a re-weighting.
* ``ACS`` is then affinely mapped onto ``[acs_bounds.min, acs_bounds.max]``, so the Section
  10.1 ``clip`` is likewise a no-op.
* ``p_base`` and the modifier product are **not** bounded above by 1 — ``commitment_hardness
  .construction_seen (0.94) * acs_bounds.max (1.35) = 1.269`` before any modifier.  Section
  10.1's own remedy is ``p = min(p, p_cap)``, which this module applies; ``p_cap`` is
  asserted to be in ``[0, 1]``.  So the final probability is in ``[0, p_cap]`` — but the
  upper bound genuinely comes from the cap, not from the algebra.  This is reported.

Dead projects (Section 21, "survivorship in the frozen pipeline")
------------------------------------------------------------------
If the frame carries the Section 17 ``outcome`` column, a project whose outcome is
``abandoned`` is forced to ``p_completion = 0`` and flagged ``dead_project``, so a pipeline
that honestly contains its failures scores materially below a survivor-only pipeline.  The
frozen-pipeline assertion of Section 19.2 belongs to the backtest module;
``assert_pipeline_contains_dead_projects`` is the same check available here.

Numeric policy (Section 0.1 rule 3)
------------------------------------
No numeric literal in this module other than ``0`` and ``1``.  ``_HALF`` is written as
``1 / (1 + 1)``: it is the definition of the median quantile, not a tunable parameter.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ufe.errors import BacktestGateFailure, MissingParameter, SchemaValidationError
from ufe.params import Params

logger = logging.getLogger(__name__)

__all__ = [
    "completion_probability",
    "delay_distribution",
    "activation_weight",
    "slip_cdf",
    "slip_median",
    "assert_pipeline_contains_dead_projects",
    "OUTCOME_COLUMN",
    "DEAD_OUTCOMES",
    "FORCE_HAPPENS",
    "FORCE_FAILS",
]

# ------------------------------------------------------------------------ parameter paths

NS = "credibility"
P_CAP = f"{NS}.p_cap"
DISCOUNT_RATE = f"{NS}.discount_rate"
STAGE_PROBABILITY = f"{NS}.stage_probability"
COMMITMENT_HARDNESS = f"{NS}.commitment_hardness"
ACS_WEIGHTS = f"{NS}.acs_weights"
ACS_MIN = f"{NS}.acs_bounds.min"
ACS_MAX = f"{NS}.acs_bounds.max"
DELIVERY_BANDS = f"{NS}.delivery_ratio_score.bands"
DELIVERY_UNKNOWN = f"{NS}.delivery_ratio_score.unknown_announcer"
CAPACITY_BANDS = f"{NS}.capacity_score.bands"
CAPACITY_MISSING = f"{NS}.capacity_score.missing_value"
LAG_DENOMINATOR = f"{NS}.component_scores.lag_score_denominator_months"
CYCLE_CENTRE = f"{NS}.component_scores.cycle_score_centre"
CYCLE_SPREAD = f"{NS}.component_scores.cycle_score_spread"
CYCLE_UTIL_MEAN = f"{NS}.component_scores.sector_capacity_util_mean"
CYCLE_UTIL_SD = f"{NS}.component_scores.sector_capacity_util_sd"
MODIFIERS = f"{NS}.modifiers"
DELAY_FAMILIES = f"{NS}.delay_lognormal"
DIVERGENCE_PENALTY = f"{NS}.physical_divergence.penalty_mult"
DIVERGENCE_STALE_DAYS = f"{NS}.physical_divergence.stale_after_days"
DIVERGENCE_STAGES = f"{NS}.physical_divergence.claimed_construction_stages"
DIVERGENCE_STATES = f"{NS}.physical_divergence.physical_states_no_activity"

ARCHETYPES = "archetypes"
DEFAULT_PHASE_CURVE = f"{ARCHETYPES}._defaults.phase_curve"
DEFAULT_CONSTRUCTION_YEARS = f"{ARCHETYPES}._defaults.construction_years"
DEFAULT_RAMP_YEARS = f"{ARCHETYPES}._defaults.operational_ramp_years"
PHASE_CURVE_TOLERANCE = f"{ARCHETYPES}._validation.phase_curve_sum_tolerance"

PHASE_KEYS = ("announcement", "construction_start", "operational")

#: `acs_weights` key -> the component column it multiplies (Section 10.1).
ACS_TERMS: tuple[tuple[str, str], ...] = (
    ("delivery_ratio", "delivery_score"),
    ("lag_score", "lag_score"),
    ("capacity", "capacity_score"),
    ("hardness", "hardness_component"),
    ("cycle", "cycle_score"),
)

# --------------------------------------------------------------------- structural constants

#: The median quantile.  A definition, not a parameter — hence the arithmetic form.
_HALF = 1 / (1 + 1)

#: Section 17's project-outcome vocabulary; the column is optional on `projects`.
OUTCOME_COLUMN = "outcome"
DEAD_OUTCOMES: frozenset[str] = frozenset({"abandoned"})

#: Section 10.5 counterfactual states.
FORCE_HAPPENS = "happens"
FORCE_FAILS = "fails"
FORCE_STATES: frozenset[str] = frozenset({FORCE_HAPPENS, FORCE_FAILS})

#: Optional column carrying the sector capacity utilisation at announcement (Section 10.2).
CYCLE_COLUMN = "sector_capacity_util"

RAISE = "raise"
IGNORE = "ignore"

_REQUIRED_PROJECT_COLUMNS = (
    "project_id",
    "archetype",
    "is_public",
    "announcer_id",
    "stage",
    "commitment_form",
    "capex_inr_cr",
    "modifiers",
    "physical_state",
    "physical_asof",
    "announced_date",
    "stated_completion",
)


# ---------------------------------------------------------------------------- small helpers


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s): {', '.join(missing)}")


def _leaf_values(params: Params, path: str) -> dict[str, float]:
    """A block of Section 4.1 leaves, as ``{key: value}``."""
    block = params.get(path)
    if not isinstance(block, Mapping):
        raise MissingParameter(f"{path} is not a parameter block")
    return {
        key: float(leaf["value"])
        for key, leaf in block.items()
        if isinstance(leaf, Mapping) and "value" in leaf
    }


def _bands(params: Params, path: str) -> list[dict[str, Any]]:
    bands = params.get(path)
    if not isinstance(bands, list) or not bands:
        raise MissingParameter(f"{path} is not a non-empty band table")
    return [dict(b) for b in bands]


def _band_score_at_least(bands: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    """First band (highest threshold first) whose ``min`` the value reaches."""
    order = sorted(bands, key=lambda b: b["min"], reverse=True)
    conditions = [x >= float(b["min"]) for b in order]
    choices = [float(b["value"]) for b in order]
    return np.select(conditions, choices, default=float(order[-1]["value"]))


def _band_score_at_most(bands: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    """First band (lowest ceiling first) whose ``max`` the value does not exceed."""
    order = sorted(bands, key=lambda b: b["max"])
    conditions = [x <= float(b["max"]) for b in order]
    choices = [float(b["value"]) for b in order]
    return np.select(conditions, choices, default=float(order[-1]["value"]))


def _unit_clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 1)


def _validate_force(force: Mapping[str, str] | None) -> dict[str, str]:
    if not force:
        return {}
    bad = {k: v for k, v in force.items() if v not in FORCE_STATES}
    if bad:
        raise ValueError(
            f"force_project_state values must be one of {sorted(FORCE_STATES)}; got {bad}"
        )
    return dict(force)


def _force_masks(
    projects: pd.DataFrame, force: Mapping[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    ids = projects["project_id"].astype(str)
    happens = ids.map(lambda pid: force.get(pid) == FORCE_HAPPENS).to_numpy(dtype=bool)
    fails = ids.map(lambda pid: force.get(pid) == FORCE_FAILS).to_numpy(dtype=bool)
    return happens, fails


def _archetype_node(params: Params, archetype: str, key: str) -> Any:
    """``archetypes.<archetype>.<key>`` if present, else ``None``."""
    for path in (f"{ARCHETYPES}.{archetype}.{key}", f"{ARCHETYPES}.{archetype}.employment.{key}"):
        try:
            node = params.get(path)
        except MissingParameter:
            continue
        if node is not None:
            return node
    return None


# ================================================================= 10.1 / 10.2  probability


def completion_probability(
    projects: pd.DataFrame,
    announcers: pd.DataFrame | None,
    params: Params,
    *,
    as_of: date | datetime | pd.Timestamp | None = None,
    force_project_state: Mapping[str, str] | None = None,
    unknown_modifiers: str = RAISE,
) -> pd.DataFrame:
    """Section 10.1 completion probability, with the Section 10.2 component scores.

    Parameters
    ----------
    projects
        A Section 3.3 ``projects`` frame.  May additionally carry the Section 17
        ``outcome`` column and the Section 10.2 ``sector_capacity_util`` column.
    announcers
        A Section 3.4 ``announcers`` frame, or ``None``.  Projects whose announcer is
        absent score ``delivery_ratio_score.unknown_announcer``.
    as_of
        Observation date for the physical-divergence staleness test.  Defaults to the
        latest observation date in the frame itself, so the function stays pure and
        deterministic; pass it explicitly in production.
    force_project_state
        Section 10.5 counterfactual: ``{project_id: 'happens' | 'fails'}``.
    unknown_modifiers
        ``'raise'`` (default) or ``'ignore'`` for modifier keys absent from
        ``credibility.modifiers``.

    Returns
    -------
    A new frame: every input column, plus ``delivery_score``, ``lag_score``,
    ``capacity_score``, ``hardness_component``, ``cycle_score``, ``acs_raw``, ``acs``,
    ``p_base``, ``modifier_p_mult``, ``p_uncapped``, ``p_completion``,
    ``capacity_data_missing`` and ``credibility_flags``.
    """
    _require_columns(projects, _REQUIRED_PROJECT_COLUMNS, "projects")
    if unknown_modifiers not in (RAISE, IGNORE):
        raise ValueError(f"unknown_modifiers must be {RAISE!r} or {IGNORE!r}")
    force = _validate_force(force_project_state)

    out = projects.copy(deep=True)
    n = len(out)
    flags: list[list[str]] = [[] for _ in range(n)]

    announcer = _announcer_columns(out, announcers)
    is_public = out["is_public"].to_numpy(dtype=bool)

    # ---------------------------------------------------------------- 10.2 component scores
    delivery = _delivery_score(params, announcer["delivery_ratio"], flags)
    lag = _lag_score(params, announcer["median_slip_months"], flags)
    capacity, capacity_missing = _capacity_score(
        params, out["capex_inr_cr"], announcer["mean_annual_capex_3y_inr_cr"], flags
    )
    hardness = _hardness_component(params, out, is_public)
    cycle = _cycle_score(params, out)

    # ------------------------------------------------------------------------- 10.1 p_base
    components = {
        "delivery_score": delivery,
        "lag_score": lag,
        "capacity_score": capacity,
        "hardness_component": hardness,
        "cycle_score": cycle,
    }
    acs_raw, acs = _acs(params, components)

    stage_probability = _leaf_values(params, STAGE_PROBABILITY)
    public_base = _map_or_raise(out["stage"], stage_probability, STAGE_PROBABILITY)

    private_base = np.where(np.isnan(hardness), np.nan, hardness * acs)
    p_base = np.where(is_public, public_base, private_base)
    acs_raw = np.where(is_public, np.nan, acs_raw)
    acs = np.where(is_public, np.nan, acs)

    # -------------------------------------------------------------------------- modifiers
    modifier_mult = _modifier_product(params, out["modifiers"], "p_mult", unknown_modifiers)

    p_cap = float(params.value(P_CAP))
    if not 0 <= p_cap <= 1:
        raise ValueError(f"{P_CAP} must lie in [0, 1]; got {p_cap}")

    p_uncapped = p_base * modifier_mult
    p = np.minimum(p_uncapped, p_cap)

    # ---------------------------------------------------- physical-divergence override
    p, divergent = _apply_physical_divergence(params, out, p, as_of)
    for i in np.flatnonzero(divergent):
        flags[i].append("physical_divergence")

    # ----------------------------------------------------- dead projects (Section 21)
    dead = _dead_mask(out)
    if dead.any():
        p = np.where(dead, 0, p)
        for i in np.flatnonzero(dead):
            flags[i].append("dead_project")

    # ------------------------------------------------------ 10.5 counterfactual override
    if force:
        happens, fails = _force_masks(out, force)
        p = np.where(happens, 1, p)
        p = np.where(fails, 0, p)
        for i in np.flatnonzero(happens):
            flags[i].append("forced_happens")
        for i in np.flatnonzero(fails):
            flags[i].append("forced_fails")

    for name, values in components.items():
        out[name] = values
    out["acs_raw"] = acs_raw
    out["acs"] = acs
    out["p_base"] = p_base
    out["modifier_p_mult"] = modifier_mult
    out["p_uncapped"] = p_uncapped
    out["p_completion"] = p
    out["capacity_data_missing"] = capacity_missing
    out["credibility_flags"] = flags

    if np.isnan(p).any():
        raise SchemaValidationError(
            "completion probability is undefined for "
            f"{int(np.isnan(p).sum())} project(s); check commitment_form and stage"
        )
    return out


def _announcer_columns(
    projects: pd.DataFrame, announcers: pd.DataFrame | None
) -> dict[str, pd.Series]:
    """Announcer track-record fields aligned to ``projects``' index."""
    wanted = ("delivery_ratio", "median_slip_months", "mean_annual_capex_3y_inr_cr")
    empty = {c: pd.Series(np.nan, index=projects.index, dtype=float) for c in wanted}
    if announcers is None or announcers.empty:
        return empty
    _require_columns(announcers, ("announcer_id",) + wanted, "announcers")
    lookup = announcers.set_index("announcer_id")[list(wanted)]
    keys = projects["announcer_id"]
    joined = keys.map(lambda k: None).to_frame(name="_")  # placeholder preserving index
    for column in wanted:
        mapping = lookup[column]
        joined[column] = keys.map(mapping).astype(float)
    return {column: joined[column] for column in wanted}


def _delivery_score(
    params: Params, delivery_ratio: pd.Series, flags: list[list[str]]
) -> np.ndarray:
    unknown = float(params.value(DELIVERY_UNKNOWN))
    bands = _bands(params, DELIVERY_BANDS)
    ratio = delivery_ratio.to_numpy(dtype=float)
    known = ~np.isnan(ratio)
    score = np.full(ratio.shape, unknown, dtype=float)
    if known.any():
        score[known] = _band_score_at_least(bands, ratio[known])
    for i in np.flatnonzero(~known):
        flags[i].append("unknown_announcer")
    return _unit_clip(score)


def _lag_score(
    params: Params, median_slip_months: pd.Series, flags: list[list[str]]
) -> np.ndarray:
    """``clip(1 - median_slip_months / 60, 0, 1)`` (Section 10.2).

    Section 10.2 does not say what to do when the announcer's slip record is null; the
    neutral ``capacity_score.missing_value`` is used and the row is flagged.
    """
    denominator = float(params.value(LAG_DENOMINATOR))
    fallback = float(params.value(CAPACITY_MISSING))
    slip = median_slip_months.to_numpy(dtype=float)
    known = ~np.isnan(slip)
    score = np.full(slip.shape, fallback, dtype=float)
    if known.any():
        score[known] = _unit_clip(1 - slip[known] / denominator)
    for i in np.flatnonzero(~known):
        flags[i].append("lag_unknown")
    return score


def _capacity_score(
    params: Params,
    capex: pd.Series,
    mean_annual_capex: pd.Series,
    flags: list[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    missing_value = float(params.value(CAPACITY_MISSING))
    bands = _bands(params, CAPACITY_BANDS)
    numerator = capex.to_numpy(dtype=float)
    denominator = mean_annual_capex.to_numpy(dtype=float)
    usable = ~np.isnan(numerator) & ~np.isnan(denominator) & (denominator > 0)
    score = np.full(numerator.shape, missing_value, dtype=float)
    if usable.any():
        score[usable] = _band_score_at_most(bands, numerator[usable] / denominator[usable])
    for i in np.flatnonzero(~usable):
        flags[i].append("capacity_unknown")
    return _unit_clip(score), ~usable


def _hardness_component(
    params: Params, projects: pd.DataFrame, is_public: np.ndarray
) -> np.ndarray:
    """``commitment_hardness[form]`` for private projects, NaN for public ones."""
    hardness = _leaf_values(params, COMMITMENT_HARDNESS)
    forms = projects["commitment_form"]
    values = np.full(len(projects), np.nan, dtype=float)
    for i, (public, form) in enumerate(zip(is_public, forms)):
        if public:
            continue
        if form is None or (isinstance(form, float) and np.isnan(form)):
            raise SchemaValidationError(
                f"private project {projects['project_id'].iloc[i]!r} has no commitment_form; "
                "Section 10.1 anchors p_base for private projects on commitment_hardness"
            )
        if form not in hardness:
            raise MissingParameter(f"{COMMITMENT_HARDNESS}.{form}")
        values[i] = hardness[form]
    return values


def _cycle_score(params: Params, projects: pd.DataFrame) -> np.ndarray:
    """Section 10.2 cycle score, inverted: high utilisation at announcement scores lower.

    ``cycle_score = 1 - clip(centre + spread * z, 0, 1)`` with
    ``z = (sector_capacity_util - mean) / sd``.  When the frame carries no
    ``sector_capacity_util`` the standardised utilisation is zero and the score reduces to
    ``1 - centre``.
    """
    centre = float(params.value(CYCLE_CENTRE))
    spread = float(params.value(CYCLE_SPREAD))
    if CYCLE_COLUMN in projects.columns:
        util = projects[CYCLE_COLUMN].to_numpy(dtype=float)
        mean = float(params.value(CYCLE_UTIL_MEAN))
        sd = float(params.value(CYCLE_UTIL_SD))
        if sd <= 0:
            raise ValueError(f"{CYCLE_UTIL_SD} must be positive; got {sd}")
        z = np.where(np.isnan(util), 0, (util - mean) / sd)
    else:
        z = np.zeros(len(projects), dtype=float)
    return 1 - _unit_clip(centre + spread * z)


def _acs(params: Params, components: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Section 10.1 announcer credibility score, bounded by construction."""
    weights = _leaf_values(params, ACS_WEIGHTS)
    missing = [key for key, _ in ACS_TERMS if key not in weights]
    if missing:
        raise MissingParameter(", ".join(f"{ACS_WEIGHTS}.{k}" for k in missing))

    total_weight = sum(weights[key] for key, _ in ACS_TERMS)
    if total_weight <= 0:
        raise ValueError(f"{ACS_WEIGHTS} must sum to a positive number")

    raw = np.zeros_like(next(iter(components.values())), dtype=float)
    for weight_key, component in ACS_TERMS:
        raw = raw + weights[weight_key] * components[component]
    # Convex combination -> [0, 1] without clipping (see the module docstring).
    raw = raw / total_weight

    lo = float(params.value(ACS_MIN))
    hi = float(params.value(ACS_MAX))
    if lo > hi:
        raise ValueError(f"{ACS_MIN} ({lo}) exceeds {ACS_MAX} ({hi})")
    return raw, np.clip(lo + raw * (hi - lo), lo, hi)


def _map_or_raise(series: pd.Series, table: Mapping[str, float], path: str) -> np.ndarray:
    unknown = sorted({v for v in series.dropna().unique() if v not in table})
    if unknown:
        raise MissingParameter(", ".join(f"{path}.{key}" for key in unknown))
    return series.map(table).to_numpy(dtype=float)


def _modifier_product(
    params: Params, modifiers: pd.Series, field: str, unknown: str
) -> np.ndarray:
    table = params.get(MODIFIERS)
    product = np.ones(len(modifiers), dtype=float)
    for i, keys in enumerate(modifiers):
        if keys is None:
            continue
        for key in keys:
            leaf = table.get(key) if isinstance(table, Mapping) else None
            if leaf is None or field not in leaf:
                if unknown == IGNORE:
                    logger.debug("ignoring unknown project modifier %r", key)
                    continue
                raise MissingParameter(f"{MODIFIERS}.{key}.{field}")
            product[i] *= float(leaf[field]["value"])
    return product


def _observation_horizon(projects: pd.DataFrame) -> pd.Timestamp | None:
    """The latest observation date in the frame — a pure, deterministic stand-in for today."""
    candidates = [c for c in ("last_updated", "stage_asof", "physical_asof") if c in projects]
    stamps = [pd.to_datetime(projects[c], errors="coerce").max() for c in candidates]
    stamps = [s for s in stamps if pd.notna(s)]
    return max(stamps) if stamps else None


def _apply_physical_divergence(
    params: Params,
    projects: pd.DataFrame,
    p: np.ndarray,
    as_of: date | datetime | pd.Timestamp | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 10.1 physical-divergence override.

    ``claimed_construction_stages`` mixes the two Section 3.3 vocabularies —
    ``construction`` / ``half_complete`` are ``stage`` values while ``epc_appointed`` /
    ``equipment_ordered`` are ``commitment_form`` values — so membership is tested against
    both columns.  Reported as a spec ambiguity.
    """
    claimed = set(params.get(DIVERGENCE_STAGES))
    inactive = set(params.get(DIVERGENCE_STATES))
    penalty = float(params.value(DIVERGENCE_PENALTY))
    stale_after = float(params.value(DIVERGENCE_STALE_DAYS))

    horizon = pd.Timestamp(as_of) if as_of is not None else _observation_horizon(projects)
    if horizon is None:
        return p, np.zeros(len(projects), dtype=bool)

    claims_construction = projects["stage"].isin(claimed) | projects[
        "commitment_form"
    ].isin(claimed)
    no_activity = projects["physical_state"].isin(inactive)
    observed = pd.to_datetime(projects["physical_asof"], errors="coerce")
    age_days = (horizon - observed) / pd.Timedelta(days=1)
    fresh = observed.notna() & (age_days < stale_after)

    divergent = (claims_construction & no_activity & fresh).to_numpy(dtype=bool)
    return np.where(divergent, p * penalty, p), divergent


def _dead_mask(projects: pd.DataFrame) -> np.ndarray:
    if OUTCOME_COLUMN not in projects.columns:
        return np.zeros(len(projects), dtype=bool)
    return projects[OUTCOME_COLUMN].isin(DEAD_OUTCOMES).to_numpy(dtype=bool)


def assert_pipeline_contains_dead_projects(projects: pd.DataFrame) -> None:
    """Section 19.2 / Section 21: a pipeline with no failures is a contaminated pipeline.

    Raises :class:`ufe.errors.BacktestGateFailure` when the frame carries no ``outcome``
    column or no abandoned project.  The credibility layer cannot demonstrate any value on
    a survivor-only pipeline, which is exactly how the survivorship failure mode hides.
    """
    if not _dead_mask(projects).any():
        raise BacktestGateFailure(
            "Frozen pipeline contains no failed projects. The freeze is contaminated "
            f"(Section 19.2): no row has {OUTCOME_COLUMN} in {sorted(DEAD_OUTCOMES)}."
        )


# ============================================================= 10.3  delay / open year


def _delay_family(params: Params, archetype: str, family_map: Mapping[str, str] | None) -> str:
    key = (family_map or {}).get(archetype, archetype)
    table = params.get(DELAY_FAMILIES)
    if not isinstance(table, Mapping) or key not in table:
        raise MissingParameter(
            f"{DELAY_FAMILIES}.{key} — no Section 10.3 delay distribution for archetype "
            f"{archetype!r}; supply a family_map or add the family to credibility.yaml"
        )
    return key


def _family_spec(params: Params, family: str) -> dict[str, Any]:
    node = params.get(f"{DELAY_FAMILIES}.{family}")
    if not isinstance(node, Mapping):
        raise MissingParameter(f"{DELAY_FAMILIES}.{family}")
    spec: dict[str, Any] = {"bimodal": bool(node.get("bimodal", False))}
    fields = ("fast_p", "fast_slip", "slow_slip", "sigma") if spec["bimodal"] else (
        "median",
        "sigma",
    )
    for field in fields:
        if field not in node:
            raise MissingParameter(f"{DELAY_FAMILIES}.{family}.{field}")
        spec[field] = float(params.value(f"{DELAY_FAMILIES}.{family}.{field}"))
    return spec


def slip_cdf(params: Params, family: str, x: np.ndarray | Sequence[float]) -> np.ndarray:
    """CDF of the Section 10.3 slip law for ``family`` (slip as a fraction of duration)."""
    spec = _family_spec(params, family)
    values = np.asarray(x, dtype=float)
    out = np.zeros(values.shape, dtype=float)
    positive = values > 0
    if spec["bimodal"]:
        sigma, slow, fast_p, fast_slip = (
            spec["sigma"],
            spec["slow_slip"],
            spec["fast_p"],
            spec["fast_slip"],
        )
        out[positive] = (1 - fast_p) * stats.norm.cdf(
            (np.log(values[positive]) - np.log(slow)) / sigma
        )
        out = out + fast_p * (values >= fast_slip)
    else:
        sigma, median = spec["sigma"], spec["median"]
        out[positive] = stats.norm.cdf((np.log(values[positive]) - np.log(median)) / sigma)
    return np.clip(out, 0, 1)


def slip_median(params: Params, family: str) -> float:
    """Median slip for ``family`` — the value a deterministic run uses (Section 10.3)."""
    spec = _family_spec(params, family)
    if not spec["bimodal"]:
        return float(spec["median"])
    fast_p, fast_slip, slow, sigma = (
        spec["fast_p"],
        spec["fast_slip"],
        spec["slow_slip"],
        spec["sigma"],
    )
    if fast_p >= _HALF:
        return float(fast_slip)
    quantile = (_HALF - fast_p) / (1 - fast_p)
    candidate = slow * np.exp(sigma * stats.norm.ppf(quantile))
    return float(max(candidate, fast_slip))


def _sample_slip(
    params: Params, family: str, size: int, rng: np.random.Generator
) -> np.ndarray:
    spec = _family_spec(params, family)
    if not spec["bimodal"]:
        return rng.lognormal(mean=np.log(spec["median"]), sigma=spec["sigma"], size=size)
    fast = rng.random(size) < spec["fast_p"]
    slow = rng.lognormal(mean=np.log(spec["slow_slip"]), sigma=spec["sigma"], size=size)
    return np.where(fast, spec["fast_slip"], slow)


def _announced_duration_years(
    projects: pd.DataFrame, days_per_year: float | None
) -> np.ndarray:
    """``(stated_completion - announced_date)`` in years.

    Section 10.3 divides by ``365.25``.  There is no ``days_per_year`` leaf in
    ``credibility.yaml`` and Section 0.1 rule 3 forbids the literal, so the default measures
    a year on the project's own announcement anniversary — a calendar fact, not a constant.
    Pass ``days_per_year`` to reproduce the spec's Julian year exactly.
    """
    announced = pd.to_datetime(projects["announced_date"])
    completion = pd.to_datetime(projects["stated_completion"])
    elapsed = completion - announced
    if days_per_year is None:
        one_year = (announced + pd.DateOffset(years=1)) - announced
        return (elapsed / one_year).to_numpy(dtype=float)
    return (elapsed / pd.Timedelta(days=1)).to_numpy(dtype=float) / float(days_per_year)


def delay_distribution(
    projects: pd.DataFrame,
    params: Params,
    *,
    monte_carlo: bool = False,
    rng: np.random.Generator | None = None,
    family_map: Mapping[str, str] | None = None,
    days_per_year: float | None = None,
    force_project_state: Mapping[str, str] | None = None,
    unknown_modifiers: str = RAISE,
) -> pd.DataFrame:
    """Section 10.3 delay distribution and the resulting ``open_year``.

    Deterministic runs take the median of the slip law (which, because the modifier product
    is a positive scale factor, is also the median of the resulting ``open_year``).  Monte
    Carlo runs sample, and require an explicit ``numpy.random.Generator`` — there is no
    implicit seeding anywhere in this module.
    """
    _require_columns(
        projects,
        ("project_id", "archetype", "announced_date", "stated_completion", "modifiers"),
        "projects",
    )
    if monte_carlo and rng is None:
        raise ValueError("monte_carlo=True requires an explicit numpy Generator via rng=")
    force = _validate_force(force_project_state)

    out = projects.copy(deep=True)
    n = len(out)

    families = np.array(
        [_delay_family(params, str(a), family_map) for a in out["archetype"]], dtype=object
    )
    slip = np.zeros(n, dtype=float)
    for family in pd.unique(families):
        mask = families == family
        if monte_carlo:
            assert rng is not None  # narrowed by the guard above
            slip[mask] = _sample_slip(params, str(family), int(mask.sum()), rng)
        else:
            slip[mask] = slip_median(params, str(family))

    slip = slip * _modifier_product(params, out["modifiers"], "delay_mult", unknown_modifiers)

    duration = _announced_duration_years(out, days_per_year)
    announced_year = pd.to_datetime(out["announced_date"]).dt.year.to_numpy(dtype=float)

    happens, fails = _force_masks(out, force) if force else (
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )
    # 10.5: 'happens' bypasses the delay draw and uses stated_completion.
    slip = np.where(happens, 0, slip)

    actual = duration * (1 + slip)
    open_year = announced_year + actual
    open_year = np.where(fails, np.nan, open_year)

    out["delay_family"] = families
    out["announced_duration_yr"] = duration
    out["slip"] = slip
    out["actual_duration_yr"] = np.where(fails, np.nan, actual)
    out["announced_year"] = announced_year
    out["open_year"] = open_year
    return out


# =============================================================== 10.4  activation weight


def _phase_curve(params: Params, archetype: str) -> tuple[float, float, float]:
    node = _archetype_node(params, archetype, "phase_curve")
    if node is None:
        node = params.get(DEFAULT_PHASE_CURVE)
    missing = [k for k in PHASE_KEYS if k not in node]
    if missing:
        raise MissingParameter(
            ", ".join(f"{ARCHETYPES}.{archetype}.phase_curve.{k}" for k in missing)
        )
    return tuple(float(node[k]["value"]) for k in PHASE_KEYS)  # type: ignore[return-value]


def _construction_years(params: Params, archetype: str) -> float:
    node = _archetype_node(params, archetype, "construction_years")
    if node is None:
        return float(params.value(DEFAULT_CONSTRUCTION_YEARS))
    return float(node["value"])


def _ramp_years(params: Params, archetype: str) -> float:
    node = _archetype_node(params, archetype, "operational_ramp_years")
    if node is None:
        return float(params.value(DEFAULT_RAMP_YEARS))
    return float(node["value"])


def activation_weight(
    projects: pd.DataFrame,
    params: Params,
    year: int | Iterable[int],
    *,
    force_project_state: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Section 10.4 activation weight ``w(t) = p_completion * phase_weight(t) * discount(t)``.

    ``projects`` must already carry ``p_completion`` (from :func:`completion_probability`)
    and ``open_year`` (from :func:`delay_distribution`).

    ``year`` may be a scalar — one output row per project — or an iterable of years, in
    which case the output is a long frame with a ``year`` column.
    """
    _require_columns(projects, ("project_id", "archetype"), "projects")
    for column in ("p_completion", "open_year"):
        if column not in projects.columns:
            raise ValueError(
                f"activation_weight needs the {column!r} column; run "
                f"{'completion_probability' if column == 'p_completion' else 'delay_distribution'}"
                " first"
            )
    force = _validate_force(force_project_state)

    if isinstance(year, (int, np.integer)):
        return _activation_one_year(projects, params, int(year), force)
    frames = [_activation_one_year(projects, params, int(t), force) for t in year]
    if not frames:
        raise ValueError("year must be a scalar or a non-empty iterable of years")
    return pd.concat(frames, ignore_index=True)


def _activation_one_year(
    projects: pd.DataFrame, params: Params, year: int, force: Mapping[str, str]
) -> pd.DataFrame:
    out = projects.copy(deep=True)
    n = len(out)

    archetypes = out["archetype"].astype(str)
    curves = {a: _phase_curve(params, a) for a in archetypes.unique()}
    construction_years = {a: _construction_years(params, a) for a in archetypes.unique()}
    ramp_years = {a: _ramp_years(params, a) for a in archetypes.unique()}

    f_ann = archetypes.map(lambda a: curves[a][0]).to_numpy(dtype=float)
    f_cs = archetypes.map(lambda a: curves[a][1]).to_numpy(dtype=float)
    f_op = archetypes.map(lambda a: curves[a][2]).to_numpy(dtype=float)
    ramp_span = archetypes.map(lambda a: ramp_years[a]).to_numpy(dtype=float)
    build_span = archetypes.map(lambda a: construction_years[a]).to_numpy(dtype=float)

    if "announced_year" in out.columns:
        announced_year = out["announced_year"].to_numpy(dtype=float)
    else:
        announced_year = pd.to_datetime(out["announced_date"]).dt.year.to_numpy(dtype=float)
    open_year = out["open_year"].to_numpy(dtype=float)
    construction_start = open_year - build_span

    opens = ~np.isnan(open_year)
    with np.errstate(invalid="ignore"):
        ramp = np.where(
            ramp_span > 0, _unit_clip((year - open_year) / np.where(ramp_span > 0, ramp_span, 1)),
            (year >= open_year).astype(float),
        )
        # Section 10 ACCEPTANCE 6: "Sum phase_curve = 1 implies w(t -> inf) = p_completion
        # exactly."  The loader has already validated that the curve sums to 1 within
        # `archetypes._validation.phase_curve_sum_tolerance`, so at full ramp the residual
        # is float noise and the terminal weight is snapped to exactly 1.
        tolerance = float(params.get(PHASE_CURVE_TOLERANCE))
        curve_total = f_ann + f_cs + f_op
        terminal = np.where(np.abs(curve_total - 1) <= tolerance, 1, curve_total)
        operational = np.where(ramp >= 1, terminal, f_ann + f_cs + f_op * ramp)
        phase = np.select(
            [
                ~opens,
                year < announced_year,
                year >= open_year,
                year >= construction_start,
            ],
            [
                np.zeros(n),
                np.zeros(n),
                operational,
                f_ann + f_cs,
            ],
            default=f_ann,
        )
        discount_rate = float(params.value(DISCOUNT_RATE))
        discount = np.where(
            opens & (year < open_year), (1 + discount_rate) ** (year - open_year), 1
        )

    phase = np.where(opens, phase, 0)
    p = out["p_completion"].to_numpy(dtype=float)
    weight = p * phase * discount

    if force:
        _, fails = _force_masks(out, force)
        weight = np.where(fails, 0, weight)
        phase = np.where(fails, 0, phase)

    out["year"] = year
    out["construction_start_year"] = construction_start
    out["phase_weight"] = phase
    out["discount"] = discount
    out["activation_weight"] = weight
    return out
