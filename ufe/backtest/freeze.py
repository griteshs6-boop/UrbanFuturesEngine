"""Historical freeze at origin year ``t0`` (spec Section 19.1) and the look-ahead guards.

This module is the reason the backtest is worth anything. Section 21 names look-ahead as a
failure mode whose symptom is "suspiciously high Spearman, >0.8" and whose guard is the
"parameter provenance check in ``freeze.py``". Section 19.1 is explicit that the check
"must be enforced in code, not by discipline". So every path that builds a frozen snapshot
runs :func:`assert_no_lookahead` on the *result*, and every individual guard is also a
public function so it can be attacked directly by a test.

What contamination looks like, and which guard catches it
---------------------------------------------------------
1. **A parameter fitted on post-t0 data.** Section 4.1 requires
   ``fitted_on: {cities: [...], data_through: YYYY}`` on every estimated ``scope: global``
   leaf. :func:`assert_parameter_provenance` walks the resolved tree, resolves inherited
   ``_provenance`` blocks, and raises :class:`LookAheadError` naming every leaf whose
   ``data_through`` exceeds ``t0``. A leaf carrying only a ``citation`` is admissible: it
   came from the literature and was not fitted here. A leaf with neither, in a namespace
   listed under ``backtest.freeze.provenance.strict_namespaces``, is refused as
   unprovenanced — ``ufe.params`` already refuses it at load time, and the freeze re-checks
   because it is the last line of defence.
2. **A post-t0 data observation inside the cell state.** A ``cells`` frame carries no year
   column, so its vintage cannot be inferred — it has to be *declared*. Section 19.1 step 1
   names the three that matter ("the building vintage, nightlight year, and population
   estimate at t0"); this module additionally requires the price surface's vintage.
   :func:`assert_data_vintage` refuses an undeclared vintage and refuses any vintage later
   than ``t0``. The ``cells_history`` panel does carry a year, so
   :func:`assert_history_asof` refuses any row in the *frozen* panel dated after ``t0``.
3. **A post-t0 project stage change.** Section 19.1 step 2: "ONLY projects announced on or
   before t0, with their stage AS OF t0. This requires ``project_history``."
   :func:`roll_back_stages` reconstructs each project's stage from the latest pre-t0
   ``stage`` transition. A project whose recorded ``stage_asof`` postdates ``t0`` and for
   which no pre-t0 transition exists cannot have its t0 stage reconstructed, and the freeze
   raises rather than guessing — guessing here is indistinguishable from knowing the future.

Outcomes are labels, never state
--------------------------------
Terminal project outcomes are read from the **full** ``project_history``, including rows
dated after ``t0``, because that is the answer key. They are attached to
:class:`FrozenSnapshot.outcomes` and are deliberately *not* merged into
``FrozenSnapshot.projects``: the frozen pipeline is what the model sees, and the model must
not see how the story ends. :func:`assert_dead_projects` then enforces Section 19.2.

Nothing in this module does I/O and nothing here imports ``ufe.ai`` (CONTRACT.md rule 4).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ufe.errors import LookAheadError, SurvivorshipContamination, UFEError

logger = logging.getLogger(__name__)

__all__ = [
    "LookAheadError",
    "SurvivorshipContamination",
    "ProvenanceRecord",
    "FrozenSnapshot",
    "parameter_provenance",
    "assert_parameter_provenance",
    "assert_data_vintage",
    "assert_history_asof",
    "assert_projects_asof",
    "roll_back_stages",
    "derive_outcomes",
    "assert_dead_projects",
    "assert_no_lookahead",
    "freeze",
]


# --------------------------------------------------------------------------------------
# Exceptions
#
# CONTRACT.md says custom exceptions live in `ufe/errors.py`, and `LookAheadError` and
# `SurvivorshipContamination` now do. They are imported above and re-exported here — and
# named in `__all__` — so `from ufe.backtest.freeze import LookAheadError` keeps working.
#
# `LookAheadError` (Section 19.1 / Section 21): a frozen run would have used information
# that postdates `t0`. `SurvivorshipContamination` (Section 19.2): the frozen pipeline has
# been filtered to projects that ultimately succeeded. Both are raised, never warned.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Parameter paths (no thresholds in the Python — Section 0.1 rule 3)
# --------------------------------------------------------------------------------------

P_STRICT_NAMESPACES = "backtest.freeze.provenance.strict_namespaces"
P_REQUIRE_PROVENANCE = "backtest.freeze.provenance.require_provenance"
P_ALLOW_EQUAL_T0 = "backtest.freeze.provenance.allow_data_through_equal_t0"
P_VINTAGE_KEYS = "backtest.freeze.vintage.required_keys"
P_VINTAGE_MAX_LAG = "backtest.freeze.vintage.max_lag_years"
P_MIN_PRICE_ZONE_FRAC = "backtest.freeze.price_reconstruction.min_observed_zone_frac"
P_OUTCOME_FIELD = "backtest.freeze.dead_projects.outcome_field"
P_ABANDONED_VALUE = "backtest.freeze.dead_projects.abandoned_value"
P_KNOWN_OUTCOMES = "backtest.freeze.dead_projects.known_values"
P_MIN_ABANDONED_COUNT = "backtest.freeze.dead_projects.min_abandoned_count"
P_MIN_ABANDONED_SHARE = "backtest.freeze.dead_projects.min_abandoned_share"

#: Column names this module reads. Named so the strings appear once.
COL_H3 = "h3"
COL_ZONE = "h3_res8"
COL_YEAR = "year"
COL_PRICE = "price_res_inr_sqft"
COL_PROJECT_ID = "project_id"
COL_STAGE = "stage"
COL_STAGE_ASOF = "stage_asof"
COL_ANNOUNCED = "announced_date"
COL_FIELD = "field"
COL_NEW_VALUE = "new_value"
COL_CHANGED_AT = "changed_at"

#: `cells_history` columns that overwrite the corresponding `cells` column at t0.
HISTORY_STATE_COLUMNS = ("builtup_frac", "nightlight", "population", COL_PRICE)

#: `projects` date columns that must not postdate t0 in a frozen pipeline.
PROJECT_DATE_COLUMNS = (COL_ANNOUNCED, COL_STAGE_ASOF, "first_seen", "last_updated",
                        "physical_asof")

_ZERO = 0
_ONE = 1


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """The effective provenance of one parameter leaf (Section 4.1)."""

    path: str
    scope: str
    data_through: int | None
    cities: tuple[str, ...]
    citation: str | None

    @property
    def is_fitted(self) -> bool:
        return self.data_through is not None

    def postdates(self, t0: int, *, allow_equal: bool) -> bool:
        if self.data_through is None:
            return False
        return self.data_through > t0 if allow_equal else self.data_through >= t0


def _leaf_paths_with_provenance(params: Any) -> list[tuple[str, dict, dict]]:
    """(path, leaf, inherited-provenance) for every leaf, using the loader's own walker."""
    from ufe.params import _walk_with_provenance  # noqa: PLC0415 - internal by design

    return list(_walk_with_provenance(params.resolved))


def parameter_provenance(
    params: Any, *, namespaces: Sequence[str] | None = None
) -> list[ProvenanceRecord]:
    """Effective provenance for every ``scope: global`` leaf, sorted by path.

    ``namespaces`` restricts the walk to those top-level parameter namespaces; ``None``
    walks the whole tree. ``scope: local`` leaves are excluded: they are calibrated from
    the city itself and Section 4.9 already forbids them from carrying the cross-city
    evidence base, so there is no cross-city fit date to check.
    """
    from ufe.params import _effective_provenance, _tokens  # noqa: PLC0415

    wanted = None if namespaces is None else set(namespaces)
    records: list[ProvenanceRecord] = []
    for path, leaf, inherited in _leaf_paths_with_provenance(params):
        if leaf.get("scope") != "global":
            continue
        tokens = _tokens(path)
        if not tokens:
            continue
        if wanted is not None and tokens[_ZERO] not in wanted:
            continue
        record = _effective_provenance(leaf, inherited)
        fitted = record.get("fitted_on") or {}
        data_through = fitted.get("data_through") if isinstance(fitted, Mapping) else None
        cities = fitted.get("cities") if isinstance(fitted, Mapping) else None
        records.append(
            ProvenanceRecord(
                path=path,
                scope="global",
                data_through=None if data_through is None else int(data_through),
                cities=tuple(cities or ()),
                citation=record.get("citation"),
            )
        )
    return sorted(records, key=lambda r: r.path)


def assert_parameter_provenance(params: Any, t0: int) -> list[ProvenanceRecord]:
    """Refuse to freeze at ``t0`` if any parameter was fitted on data after ``t0``.

    This is the Section 21 guard. It returns the provenance records it checked so the
    caller can record them in the run manifest — a backtest whose provenance is not written
    down is a backtest nobody can re-audit.
    """
    namespaces = list(params.value(P_STRICT_NAMESPACES))
    allow_equal = bool(params.value(P_ALLOW_EQUAL_T0))
    require = bool(params.value(P_REQUIRE_PROVENANCE))

    records = parameter_provenance(params, namespaces=namespaces)

    contaminated = [r for r in records if r.postdates(t0, allow_equal=allow_equal)]
    if contaminated:
        detail = "\n  ".join(
            f"{r.path}: fitted_on.data_through={r.data_through} > t0={t0} "
            f"(cities={', '.join(r.cities) or 'unstated'})"
            for r in contaminated
        )
        raise LookAheadError(
            f"{len(contaminated)} parameter(s) were fitted on data that postdates the "
            f"freeze origin t0={t0}. Freezing on them would leak the future into the "
            f"backtest (spec Sections 19.1 and 21). Re-estimate with "
            f"`ufe estimate global --data-through {t0}` or exclude these parameters:\n  "
            + detail
        )

    if require:
        unprovenanced = [r for r in records if not r.is_fitted and not r.citation]
        if unprovenanced:
            raise LookAheadError(
                f"{len(unprovenanced)} scope:global parameter(s) declare neither "
                f"`fitted_on` nor `citation`, so it cannot be shown that they predate "
                f"t0={t0} (spec Section 4.1):\n  "
                + "\n  ".join(r.path for r in unprovenanced)
            )

    logger.info(
        "parameter provenance clean for t0=%s: %d scope:global leaves checked, "
        "%d fitted, latest data_through=%s",
        t0,
        len(records),
        sum(_ONE for r in records if r.is_fitted),
        max((r.data_through for r in records if r.is_fitted), default=None),
    )
    return records


# --------------------------------------------------------------------------------------
# Data vintage and history slicing
# --------------------------------------------------------------------------------------


def assert_data_vintage(vintage: Mapping[str, int], t0: int, params: Any) -> dict[str, int]:
    """Refuse a cell snapshot whose declared source vintages do not sit at ``t0``.

    A ``cells`` frame has no year column, so a 2023 nightlight composite dropped into a
    2012 freeze is undetectable from the data alone. The caller therefore has to state, per
    attribute, the year the value describes. Undeclared is refused (not defaulted), later
    than ``t0`` is refused as look-ahead, and older than
    ``backtest.freeze.vintage.max_lag_years`` is refused as not describing ``t0`` at all.
    """
    required = list(params.value(P_VINTAGE_KEYS))
    max_lag = int(params.value(P_VINTAGE_MAX_LAG))

    missing = [key for key in required if key not in vintage]
    if missing:
        raise LookAheadError(
            "the cell snapshot's source vintage is undeclared for "
            f"{', '.join(sorted(missing))}. Section 19.1 step 1 requires the building "
            "vintage, nightlight year and population estimate at t0 to be known; an "
            "undeclared vintage is refused rather than assumed."
        )

    future = {k: int(v) for k, v in vintage.items() if int(v) > t0}
    if future:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(future.items()))
        raise LookAheadError(
            f"cell-state vintage postdates the freeze origin t0={t0}: {detail}. "
            "A frozen snapshot may not contain an observation the model could not have "
            "had (spec Sections 19.1 and 21)."
        )

    stale = {k: int(v) for k, v in vintage.items() if int(v) < t0 - max_lag}
    if stale:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(stale.items()))
        raise LookAheadError(
            f"cell-state vintage lags t0={t0} by more than {max_lag} year(s): {detail}. "
            "This is not look-ahead, but the snapshot does not describe the origin year "
            "either, and scoring it would be dishonest in the other direction."
        )
    return {str(k): int(v) for k, v in vintage.items()}


def _year_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Calendar year of a date column, as a nullable integer Series (Section 0.3)."""
    return pd.to_datetime(frame[column], errors="coerce").dt.year


def assert_history_asof(history: pd.DataFrame, t0: int, *, label: str) -> None:
    """Refuse a frozen panel containing an observation dated after ``t0``."""
    if history.empty:
        return
    if COL_YEAR in history.columns:
        years = pd.to_numeric(history[COL_YEAR], errors="coerce")
    elif COL_CHANGED_AT in history.columns:
        years = _year_series(history, COL_CHANGED_AT)
    else:
        raise LookAheadError(
            f"{label} carries neither {COL_YEAR!r} nor {COL_CHANGED_AT!r}, so it cannot "
            "be shown to predate t0"
        )
    future = years[years > t0]
    if len(future) > _ZERO:
        raise LookAheadError(
            f"{label} in the frozen snapshot contains {len(future)} observation(s) dated "
            f"after t0={t0} (latest {int(future.max())}). The freeze must be sliced to "
            f"year <= {t0} before it is scored (spec Section 19.1)."
        )


def assert_projects_asof(projects: pd.DataFrame, t0: int) -> None:
    """Refuse a frozen pipeline whose project records carry post-``t0`` dates."""
    if projects.empty:
        return
    offences: list[str] = []
    for column in PROJECT_DATE_COLUMNS:
        if column not in projects.columns:
            continue
        years = _year_series(projects, column)
        late = projects.loc[years > t0, COL_PROJECT_ID]
        if len(late) > _ZERO:
            shown = ", ".join(map(str, sorted(late)))
            offences.append(f"{column}: {shown}")
    if offences:
        raise LookAheadError(
            f"the frozen project pipeline carries record dates after t0={t0}. Section "
            "19.1 step 2 admits only projects announced on or before t0, with their stage "
            "AS OF t0:\n  " + "\n  ".join(offences)
        )


def roll_back_stages(
    projects: pd.DataFrame, project_history: pd.DataFrame, t0: int
) -> pd.DataFrame:
    """Restate every project's ``stage``/``stage_asof`` as it stood at ``t0``.

    Uses the latest ``field == 'stage'`` transition dated on or before ``t0``. A project
    whose current record postdates ``t0`` and has no such transition cannot be rolled back:
    its t0 stage is unknown, and Section 19.1 says the list must then be reconstructed by
    hand from contemporaneous sources. That reconstruction is a human job, so this raises.
    """
    out = projects.copy()
    if out.empty:
        return out

    stage_rows = pd.DataFrame(columns=[COL_PROJECT_ID, COL_NEW_VALUE, COL_CHANGED_AT])
    if not project_history.empty and COL_FIELD in project_history.columns:
        mask = project_history[COL_FIELD] == COL_STAGE
        candidates = project_history.loc[mask].copy()
        if not candidates.empty:
            candidates[COL_CHANGED_AT] = pd.to_datetime(candidates[COL_CHANGED_AT])
            candidates = candidates.loc[candidates[COL_CHANGED_AT].dt.year <= t0]
            if not candidates.empty:
                candidates = candidates.sort_values([COL_PROJECT_ID, COL_CHANGED_AT])
                stage_rows = candidates.groupby(COL_PROJECT_ID, as_index=False).last()

    latest = stage_rows.set_index(COL_PROJECT_ID) if not stage_rows.empty else None
    stale_year = _year_series(out, COL_STAGE_ASOF) > t0

    unrecoverable: list[str] = []
    stages = out[COL_STAGE].tolist()
    asof = list(pd.to_datetime(out[COL_STAGE_ASOF]))
    for position, project_id in enumerate(out[COL_PROJECT_ID].tolist()):
        known = latest is not None and project_id in latest.index
        if known:
            stages[position] = str(latest.loc[project_id, COL_NEW_VALUE])
            asof[position] = pd.Timestamp(latest.loc[project_id, COL_CHANGED_AT])
        elif bool(stale_year.iloc[position]):
            unrecoverable.append(str(project_id))

    if unrecoverable:
        raise LookAheadError(
            f"{len(unrecoverable)} project(s) record a stage change after t0={t0} and "
            "carry no pre-t0 stage transition in `project_history`, so their stage AS OF "
            "t0 is unknown. Using the recorded stage would import a post-freeze stage "
            "change into the snapshot (spec Section 19.1 step 2). Reconstruct these from "
            "sources dated t0, BEFORE consulting any outcome data: "
            + ", ".join(sorted(unrecoverable))
        )

    out[COL_STAGE] = stages
    out[COL_STAGE_ASOF] = asof
    return out


# --------------------------------------------------------------------------------------
# Outcomes and the Section 19.2 dead-project requirement
# --------------------------------------------------------------------------------------


def derive_outcomes(project_history: pd.DataFrame, params: Any) -> pd.Series:
    """Terminal outcome per project, read from the FULL history including post-t0 rows.

    Outcomes are the answer key, not state. They are read here from every ``field ==
    'outcome'`` row regardless of date, and the caller must keep them out of the frozen
    snapshot the model sees.
    """
    outcome_field = str(params.value(P_OUTCOME_FIELD))
    known = set(params.value(P_KNOWN_OUTCOMES))
    empty = pd.Series(dtype=object, name=outcome_field)
    if project_history.empty or COL_FIELD not in project_history.columns:
        return empty

    rows = project_history.loc[project_history[COL_FIELD] == outcome_field].copy()
    if rows.empty:
        return empty
    rows[COL_CHANGED_AT] = pd.to_datetime(rows[COL_CHANGED_AT])
    rows = rows.sort_values([COL_PROJECT_ID, COL_CHANGED_AT])
    latest = rows.groupby(COL_PROJECT_ID, as_index=False).last()

    unknown = sorted(set(latest[COL_NEW_VALUE]) - known)
    if unknown:
        raise UFEError(
            f"project_history records unknown {outcome_field} value(s) "
            f"{', '.join(map(str, unknown))}; the vocabulary is "
            f"{sorted(known)} (config/params/backtest.yaml)"
        )
    result = latest.set_index(COL_PROJECT_ID)[COL_NEW_VALUE]
    result.name = outcome_field
    return result


def assert_dead_projects(
    project_ids: Iterable[str], outcomes: pd.Series, params: Any
) -> dict[str, Any]:
    """Section 19.2, enforced.

    ``assert any(p.outcome == 'abandoned' for p in frozen_projects)`` — plus a share floor,
    because one dead project among two hundred is survivorship in all but name and leaves
    the credibility layer just as untestable.
    """
    abandoned_value = str(params.value(P_ABANDONED_VALUE))
    min_count = int(params.value(P_MIN_ABANDONED_COUNT))
    min_share = float(params.value(P_MIN_ABANDONED_SHARE))

    ids = [str(pid) for pid in project_ids]
    total = len(ids)
    if total == _ZERO:
        raise SurvivorshipContamination(
            "the frozen pipeline is empty, so it contains no failed projects and the "
            "credibility layer cannot be tested (spec Section 19.2)."
        )

    labelled = outcomes.reindex(ids)
    unlabelled = int(labelled.isna().sum())
    abandoned = [pid for pid in ids if str(labelled.get(pid)) == abandoned_value]
    share = len(abandoned) / total

    summary = {
        "frozen_projects": total,
        "labelled": total - unlabelled,
        "unlabelled": unlabelled,
        "abandoned": len(abandoned),
        "abandoned_share": share,
        "abandoned_ids": sorted(abandoned),
    }

    if len(abandoned) < min_count:
        raise SurvivorshipContamination(
            f"Frozen pipeline contains no failed projects. The freeze is contaminated. "
            f"({total} project(s) at t0, {unlabelled} without a recorded outcome, "
            f"{len(abandoned)} abandoned, minimum {min_count}.) Section 19.2: a t0 "
            "pipeline containing only projects that ultimately completed makes the "
            "credibility layer untestable, and this assertion is not optional. The usual "
            "cause is a project list rebuilt from today's records, which quietly drops "
            "everything that was cancelled."
        )
    if share < min_share:
        raise SurvivorshipContamination(
            f"Frozen pipeline is survivorship-contaminated: {len(abandoned)}/{total} "
            f"projects abandoned (share {share:.3f}), below the {min_share:.3f} floor in "
            "config/params/backtest.yaml. Section 19.2."
        )

    logger.info(
        "dead-project requirement satisfied: %d of %d frozen projects abandoned "
        "(share %.3f)",
        len(abandoned),
        total,
        share,
    )
    return summary


# --------------------------------------------------------------------------------------
# The frozen snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSnapshot:
    """A historical snapshot at ``t0``, provably free of post-``t0`` information.

    ``outcomes`` is the exception and is documented as such: it is the answer key, read
    from post-t0 history, held here so the scorer and the Section 19.2 assertion can reach
    it. It is never merged into ``projects``.
    """

    city_id: str
    t0: int
    cells: pd.DataFrame
    cells_history: pd.DataFrame
    projects: pd.DataFrame
    project_history: pd.DataFrame
    outcomes: pd.Series
    params: Any
    vintage: Mapping[str, int]
    price_test_enabled: bool
    provenance: tuple[ProvenanceRecord, ...] = ()
    dead_projects: Mapping[str, Any] = field(default_factory=dict)
    seed: int = _ZERO

    @property
    def freeze_hash(self) -> str:
        """Stable identity of the freeze: params hash + t0 + frame content hashes."""
        payload = {
            "city_id": self.city_id,
            "t0": self.t0,
            "params_hash": getattr(self.params, "hash", None),
            "vintage": {str(k): int(v) for k, v in sorted(self.vintage.items())},
            "seed": self.seed,
            "frames": {
                name: _frame_hash(frame)
                for name, frame in (
                    ("cells", self.cells),
                    ("cells_history", self.cells_history),
                    ("projects", self.projects),
                    ("project_history", self.project_history),
                )
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Everything a reader needs to re-audit this freeze."""
        return {
            "city_id": self.city_id,
            "t0": self.t0,
            "freeze_hash": self.freeze_hash,
            "params_hash": getattr(self.params, "hash", None),
            "seed": self.seed,
            "vintage": dict(self.vintage),
            "price_test_enabled": self.price_test_enabled,
            "n_cells": int(len(self.cells)),
            "n_projects": int(len(self.projects)),
            "dead_projects": dict(self.dead_projects),
            "provenance_checked": len(self.provenance),
            "latest_data_through": max(
                (r.data_through for r in self.provenance if r.is_fitted), default=None
            ),
        }


def _frame_hash(frame: pd.DataFrame) -> str:
    try:
        from ufe.store.db import content_hash  # noqa: PLC0415 - optional dependency

        return content_hash(frame)
    except Exception:  # pragma: no cover - store is present in this repo
        blob = frame.to_json(orient="split", date_format="iso", default_handler=str)
        return hashlib.sha256((blob or "").encode("utf-8")).hexdigest()


def _price_test_enabled(cells_t0: pd.DataFrame, params: Any) -> bool:
    """Section 19.1 step 1: skip the price test where the t0 surface is too sparse."""
    minimum = float(params.value(P_MIN_PRICE_ZONE_FRAC))
    if COL_ZONE not in cells_t0.columns or COL_PRICE not in cells_t0.columns:
        return False
    observed = cells_t0.groupby(COL_ZONE)[COL_PRICE].apply(lambda s: bool(s.notna().any()))
    if len(observed) == _ZERO:
        return False
    fraction = float(observed.mean())
    enabled = fraction >= minimum
    if not enabled:
        logger.warning(
            "price test disabled: observed-price zone share %.3f is below the required "
            "%.3f; the settlement test runs alone (spec Section 19.1)",
            fraction,
            minimum,
        )
    return enabled


def _cells_at_t0(cells: pd.DataFrame, history: pd.DataFrame, t0: int) -> pd.DataFrame:
    """Overwrite the history-tracked cell columns with their ``t0`` observation."""
    out = cells.copy()
    if history.empty:
        return out
    at_t0 = history.loc[pd.to_numeric(history[COL_YEAR], errors="coerce") == t0]
    if at_t0.empty:
        logger.warning(
            "cells_history has no rows at t0=%d; the cell snapshot is used as supplied "
            "and its declared vintage is the only guarantee it describes t0",
            t0,
        )
        return out
    indexed = at_t0.set_index(COL_H3)
    for column in HISTORY_STATE_COLUMNS:
        if column not in indexed.columns or column not in out.columns:
            continue
        replacement = out[COL_H3].map(indexed[column])
        out[column] = replacement.where(replacement.notna(), out[column])
    return out


def assert_no_lookahead(snapshot: FrozenSnapshot) -> None:
    """Run every look-ahead guard against an already-built snapshot.

    :func:`freeze` calls this on its own output, but it is public and takes a snapshot
    rather than raw inputs precisely so a test can hand it a hand-planted contaminated
    snapshot and watch it refuse.
    """
    assert_parameter_provenance(snapshot.params, snapshot.t0)
    assert_data_vintage(snapshot.vintage, snapshot.t0, snapshot.params)
    assert_history_asof(snapshot.cells_history, snapshot.t0, label="cells_history")
    assert_history_asof(snapshot.project_history, snapshot.t0, label="project_history")
    assert_projects_asof(snapshot.projects, snapshot.t0)

    outcome_field = str(snapshot.params.value(P_OUTCOME_FIELD))
    if outcome_field in snapshot.projects.columns:
        raise LookAheadError(
            f"the frozen `projects` frame carries a {outcome_field!r} column. Outcomes are "
            "the answer key and are resolved after t0; the model must not see them. They "
            "belong on FrozenSnapshot.outcomes."
        )


def freeze(
    *,
    city_id: str,
    t0: int,
    cells: pd.DataFrame,
    cells_history: pd.DataFrame,
    projects: pd.DataFrame,
    project_history: pd.DataFrame,
    params: Any,
    vintage: Mapping[str, int],
    seed: int | None = None,
    require_dead_projects: bool = True,
) -> FrozenSnapshot:
    """Construct the Section 19.1 historical snapshot at ``t0``.

    The order matters. Provenance is checked *first*, before a single row is sliced,
    because a contaminated parameter file invalidates the whole exercise and there is no
    point building a snapshot around it. Then the panels are sliced to ``year <= t0``, the
    project pipeline is restricted to what had been announced and rolled back to its t0
    stage, and finally :func:`assert_no_lookahead` re-checks the *constructed* artifact —
    belt and braces, because the slicing code is exactly the code most likely to acquire an
    off-by-one.

    ``vintage`` declares the source year of each cell attribute (Section 19.1 step 1);
    see :func:`assert_data_vintage`. ``require_dead_projects`` exists only so that
    Section 19.2's assertion can be exercised in isolation; production callers leave it on.
    """
    provenance = assert_parameter_provenance(params, t0)
    checked_vintage = assert_data_vintage(vintage, t0, params)

    seed = int(params.value("backtest.baselines.seed")) if seed is None else int(seed)

    frozen_history = cells_history.loc[
        pd.to_numeric(cells_history[COL_YEAR], errors="coerce") <= t0
    ].reset_index(drop=True)

    announced_year = _year_series(projects, COL_ANNOUNCED)
    frozen_projects = projects.loc[announced_year <= t0].reset_index(drop=True)

    frozen_ids = set(frozen_projects[COL_PROJECT_ID]) if not frozen_projects.empty else set()
    if project_history.empty:
        frozen_project_history = project_history.copy()
    else:
        changed_year = _year_series(project_history, COL_CHANGED_AT)
        frozen_project_history = project_history.loc[
            (changed_year <= t0) & project_history[COL_PROJECT_ID].isin(frozen_ids)
        ].reset_index(drop=True)

    frozen_projects = roll_back_stages(frozen_projects, frozen_project_history, t0)
    frozen_projects = _clip_project_dates(frozen_projects, t0)

    cells_t0 = _cells_at_t0(cells, frozen_history, t0)

    # Outcomes come from the FULL history, deliberately: they are the answer key.
    outcomes = derive_outcomes(project_history, params)

    snapshot = FrozenSnapshot(
        city_id=city_id,
        t0=t0,
        cells=cells_t0,
        cells_history=frozen_history,
        projects=frozen_projects,
        project_history=frozen_project_history,
        outcomes=outcomes,
        params=params,
        vintage=checked_vintage,
        price_test_enabled=_price_test_enabled(cells_t0, params),
        provenance=tuple(provenance),
        seed=seed,
    )
    assert_no_lookahead(snapshot)

    dead = (
        assert_dead_projects(frozen_projects[COL_PROJECT_ID], outcomes, params)
        if require_dead_projects
        else {}
    )
    snapshot = FrozenSnapshot(
        city_id=snapshot.city_id,
        t0=snapshot.t0,
        cells=snapshot.cells,
        cells_history=snapshot.cells_history,
        projects=snapshot.projects,
        project_history=snapshot.project_history,
        outcomes=snapshot.outcomes,
        params=snapshot.params,
        vintage=snapshot.vintage,
        price_test_enabled=snapshot.price_test_enabled,
        provenance=snapshot.provenance,
        dead_projects=dead,
        seed=snapshot.seed,
    )
    logger.info("freeze %s@%d -> %s", city_id, t0, snapshot.freeze_hash)
    return snapshot


def _clip_project_dates(projects: pd.DataFrame, t0: int) -> pd.DataFrame:
    """Drop record-keeping timestamps that postdate ``t0``.

    ``first_seen``, ``last_updated`` and ``physical_asof`` are ingestion bookkeeping, not
    model inputs, but a post-t0 value in them still means the row was touched after the
    freeze — so they are nulled where nullable and, where the schema forbids null, pinned
    back to the project's ``stage_asof``. Nothing is invented: the clipped value is always
    a date the record demonstrably already had.
    """
    if projects.empty:
        return projects
    out = projects.copy()
    stage_asof = pd.to_datetime(out[COL_STAGE_ASOF])
    for column in ("first_seen", "last_updated"):
        if column not in out.columns:
            continue
        values = pd.to_datetime(out[column])
        out[column] = values.where(values.dt.year <= t0, stage_asof)
    if "physical_asof" in out.columns:
        values = pd.to_datetime(out["physical_asof"])
        late = values.dt.year > t0
        out.loc[late, "physical_asof"] = pd.NaT
        if "physical_state" in out.columns:
            out.loc[late, "physical_state"] = None
    return out


def spearman_alarm_threshold(params: Any) -> float:
    """The Section 21 "suspiciously high Spearman" threshold, for the scorer to flag on."""
    return float(params.value("backtest.scoring.lookahead.spearman_alarm"))
