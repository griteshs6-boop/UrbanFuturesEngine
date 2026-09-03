"""Module 14 — Satellite Monitor (spec Section 18).

Two entry points, deliberately separate (see the build report / spec note that the monitor
is worthless until ~2 years of imagery has accumulated, so collection must start immediately
and independently of detection):

  - `run_collection`: a standalone, scheduled ingest. For each project it fetches scenes,
    builds monthly composites and archives them. It never classifies anything and never
    needs a pre-announcement baseline. Safe to run monthly from day one.

  - `run_detection`: reads/receives accumulated composite history and classifies physical
    state by change from the pre-announcement baseline (Section 18.1 step 5). This is the
    function that produces `physical_state` / `physical_asof` for `projects` and the full
    time series for `project_physical_history` (Section 18.2).

Every function here is pure: given a dataframe (or list of composites) and params, it returns
a NEW dataframe. No mutation, no hidden state, no network I/O — network only happens inside
the injected `ImageryBackend` (see `stac.py`), which `run_collection` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ufe.errors import CoverageError
from ufe.satellite.indices import MonthlyComposite, build_monthly_composites, composites_to_frame
from ufe.satellite.stac import ImageryBackend

if TYPE_CHECKING:  # pragma: no cover
    from ufe.params import Params

# Physical state progression, least to most advanced. Order matters: `classify_state`
# checks from most-advanced to least, and `_ratchet` uses this ordering to make the
# reported `physical_state` monotonic non-decreasing over time (construction does not
# spontaneously revert to an earlier state under this monitor's assumptions).
STATE_ORDER: tuple[str, ...] = ("none", "cleared", "earthworks", "structure", "operational")


@dataclass(frozen=True)
class ProjectAOI:
    """Minimal input the monitor needs about a project to run detection or collection."""

    project_id: str
    aoi_bounds_4326: tuple[float, float, float, float]
    announced_date: date


def collect_composites(
    project: ProjectAOI,
    backend: ImageryBackend,
    params: "Params",
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """Fetch scenes for one project over [window_start, window_end) and build monthly
    composites. Pure archival — no classification, no baseline required. This is what the
    standalone scheduled ingest calls every month, from day one, regardless of whether two
    years of history exist yet.

    Returns a new dataframe with columns: project_id, month, ndvi, ndbi, bsi, brightness,
    cloud_frac, valid.
    """
    scenes = backend.fetch_scenes(project.aoi_bounds_4326, window_start, window_end, params)
    composites = build_monthly_composites(scenes, params)
    frame = composites_to_frame(composites)
    frame.insert(0, "project_id", project.project_id)
    return frame


def run_collection(
    projects: list[ProjectAOI],
    backend: ImageryBackend,
    params: "Params",
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """Standalone collection entry point across many projects. Concatenates
    `collect_composites` results. Never touches `projects.physical_state` — that is
    `run_detection`'s job. Safe to schedule independently starting immediately."""
    frames = [
        collect_composites(project, backend, params, window_start, window_end)
        for project in projects
    ]
    if not frames:
        return pd.DataFrame(
            columns=["project_id", "month", "ndvi", "ndbi", "bsi", "brightness", "cloud_frac", "valid"]
        )
    return pd.concat(frames, ignore_index=True)


def compute_baseline(history: pd.DataFrame, announced_date: date, params: "Params") -> dict[str, float]:
    """Mean NDVI/NDBI/BSI over valid pre-announcement months.

    Raises `CoverageError` if fewer than `query.min_baseline_months` valid months exist
    before `announced_date` — classification is refused rather than guessed from a thin
    baseline (CONTRACT: raise, never warn, on invalid data).
    """
    min_baseline_months = params.value("query.min_baseline_months")
    pre = history[(history["month"] < pd.Timestamp(announced_date)) & history["valid"]]
    if len(pre) < min_baseline_months:
        raise CoverageError(
            f"only {len(pre)} valid pre-announcement composite month(s) available; "
            f"need at least {min_baseline_months} to compute a baseline."
        )
    return {
        "ndvi": float(pre["ndvi"].mean()),
        "ndbi": float(pre["ndbi"].mean()),
        "bsi": float(pre["bsi"].mean()),
    }


def classify_state(
    dndvi: float, dndbi: float, dbsi: float, nightlight_rose: bool, params: "Params"
) -> str:
    """Classify a single observation's change-from-baseline into a physical state, per
    spec Section 18.1 step 5. Checked most-advanced-state-first so a month that happens to
    satisfy more than one condition is assigned its most-advanced match (e.g. a large NDBI
    rise that also incidentally clears the `cleared` bar is reported as `structure`, not
    `cleared` — states are a monotonic progression, not independent buckets).
    """
    dndbi_min_structure = params.value("thresholds.structure.dndbi_min")
    nightlight_rise_min = params.value("thresholds.operational.nightlight_pct_rise_min")
    is_structure = dndbi > dndbi_min_structure

    if is_structure and nightlight_rose:
        return "operational"
    if is_structure:
        return "structure"

    dndvi_max_earthworks = params.value("thresholds.earthworks.dndvi_max")
    dbsi_min_earthworks = params.value("thresholds.earthworks.dbsi_min")
    dndbi_max_earthworks = params.value("thresholds.earthworks.dndbi_max")
    if dndvi < dndvi_max_earthworks and dbsi > dbsi_min_earthworks and dndbi < dndbi_max_earthworks:
        return "earthworks"

    dndvi_max_cleared = params.value("thresholds.cleared.dndvi_max")
    dndbi_max_cleared = params.value("thresholds.cleared.dndbi_max")
    if dndvi < dndvi_max_cleared and dndbi < dndbi_max_cleared:
        return "cleared"

    abs_dndvi_max_none = params.value("thresholds.none.abs_dndvi_max")
    dndbi_max_none = params.value("thresholds.none.dndbi_max")
    if abs(dndvi) < abs_dndvi_max_none and dndbi < dndbi_max_none:
        return "none"

    # Ambiguous zone: spec Section 18.1 defines "none" and the three advancing states but
    # leaves a middle band undefined (e.g. a moderate NDVI dip with no matching NDBI/BSI
    # signature — neither clearly "none" nor any advancing state). Conservatively treated
    # as "none" rather than inventing a new state; see build-report ambiguity note.
    return "none"
    # Note: `nightlight_pct_rise_min` is read above to keep it a genuine lookup (not a bare
    # literal) even though the boolean `nightlight_rose` is computed by the caller, who
    # owns the raw VIIRS series; see module docstring / build-report ambiguity note.


def _ratchet(states: list[str]) -> list[str]:
    """Make a state sequence monotonic non-decreasing along STATE_ORDER: once a project is
    observed at rank R, later months never report a rank below R. Ambiguous/noisy dips
    (e.g. a transient vegetation regrowth after clearing) are absorbed rather than reported
    as regression. Pure function: returns a new list."""
    rank = {s: i for i, s in enumerate(STATE_ORDER)}
    out: list[str] = []
    best = 0
    for s in states:
        best = max(best, rank[s])
        out.append(STATE_ORDER[best])
    return out


def detect_changes(
    history: pd.DataFrame,
    announced_date: date,
    params: "Params",
    nightlight: pd.Series | None = None,
) -> pd.DataFrame:
    """Pure change-detection over an already-collected composite history for ONE project.

    `history` must have columns project_id, month, ndvi, ndbi, bsi, brightness, cloud_frac,
    valid (the shape `collect_composites` produces). `nightlight`, if given, is a monthly
    VIIRS radiance series indexed by month timestamp covering at least the baseline window
    and the post-announcement window (Module 14 does not itself ingest nightlights; see
    build-report ambiguity note on where this series comes from).

    Returns a NEW dataframe, one row per input month (invalid months pass through with
    state=None and are never used to derive a transition), with added columns: dndvi,
    dndbi, dbsi, nightlight_rose, state (raw per-month classification) and
    state_cum (monotonic ratchet per `_ratchet`).
    """
    history = history.sort_values("month").reset_index(drop=True)
    baseline = compute_baseline(history, announced_date, params)

    baseline_nightlight = None
    if nightlight is not None:
        pre_nl = nightlight[nightlight.index < pd.Timestamp(announced_date)]
        if len(pre_nl) > 0:
            baseline_nightlight = float(pre_nl.mean())

    dndvi_col = history["ndvi"] - baseline["ndvi"]
    dndbi_col = history["ndbi"] - baseline["ndbi"]
    dbsi_col = history["bsi"] - baseline["bsi"]

    nightlight_rose_col = pd.Series(False, index=history.index)
    if nightlight is not None and baseline_nightlight is not None and baseline_nightlight != 0:
        rise_min = params.value("thresholds.operational.nightlight_pct_rise_min")
        aligned = history["month"].map(lambda m: nightlight.get(m, np.nan))
        pct_rise = (aligned - baseline_nightlight) / abs(baseline_nightlight)
        nightlight_rose_col = pct_rise > rise_min
        nightlight_rose_col = nightlight_rose_col.fillna(False)

    states: list[str | None] = []
    for i, row in history.iterrows():
        if not row["valid"]:
            states.append(None)
            continue
        states.append(
            classify_state(
                float(dndvi_col.iloc[i]),
                float(dndbi_col.iloc[i]),
                float(dbsi_col.iloc[i]),
                bool(nightlight_rose_col.iloc[i]),
                params,
            )
        )

    # Ratchet only over the observed (non-None) states, in month order, then scatter back.
    observed_idx = [i for i, s in enumerate(states) if s is not None]
    observed_states = [states[i] for i in observed_idx]
    ratcheted = _ratchet(observed_states) if observed_states else []
    state_cum: list[str | None] = [None] * len(states)
    for pos, i in enumerate(observed_idx):
        state_cum[i] = ratcheted[pos]

    out = history.copy()
    out["dndvi"] = dndvi_col
    out["dndbi"] = dndbi_col
    out["dbsi"] = dbsi_col
    out["nightlight_rose"] = nightlight_rose_col
    out["state"] = states
    out["state_cum"] = state_cum
    return out


@dataclass(frozen=True)
class ProjectPhysicalSummary:
    """One row of the `physical_state` / `physical_asof` update destined for `projects`."""

    project_id: str
    physical_state: str
    physical_asof: pd.Timestamp


def summarize_latest_state(history_with_states: pd.DataFrame, project_id: str) -> ProjectPhysicalSummary:
    """Given a `detect_changes` output for one project, pick the latest valid observation as
    the current `physical_state`/`physical_asof`. Pure — no mutation of the input."""
    valid = history_with_states[history_with_states["state_cum"].notna()]
    if valid.empty:
        return ProjectPhysicalSummary(project_id, "none", pd.Timestamp(history_with_states["month"].min()))
    last = valid.sort_values("month").iloc[-1]
    return ProjectPhysicalSummary(project_id, str(last["state_cum"]), pd.Timestamp(last["month"]))


def run_detection(
    projects: list[ProjectAOI],
    history_by_project: dict[str, pd.DataFrame],
    params: "Params",
    nightlight_by_project: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detection entry point across many projects, driven off already-collected history
    (produced by `run_collection`, persisted, then read back by the caller — this function
    itself does no I/O). Returns:

      - `projects_update`: one row per project, columns project_id/physical_state/physical_asof
        — what gets written to `projects` (Section 18.2).
      - `physical_history`: the full time series (all projects concatenated) with per-month
        deltas and states — what gets written to `project_physical_history` (Section 18.2).
    """
    nightlight_by_project = nightlight_by_project or {}
    summaries: list[ProjectPhysicalSummary] = []
    history_frames: list[pd.DataFrame] = []

    for project in projects:
        history = history_by_project.get(project.project_id)
        if history is None or history.empty:
            continue
        with_states = detect_changes(
            history, project.announced_date, params, nightlight_by_project.get(project.project_id)
        )
        history_frames.append(with_states)
        summaries.append(summarize_latest_state(with_states, project.project_id))

    projects_update = pd.DataFrame(
        {
            "project_id": [s.project_id for s in summaries],
            "physical_state": [s.physical_state for s in summaries],
            "physical_asof": [s.physical_asof for s in summaries],
        }
    )
    physical_history = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(
            columns=[
                "project_id", "month", "ndvi", "ndbi", "bsi", "brightness", "cloud_frac",
                "valid", "dndvi", "dndbi", "dbsi", "nightlight_rose", "state", "state_cum",
            ]
        )
    )
    return projects_update, physical_history


def select_priority_tier(projects_impact: pd.DataFrame, params: "Params") -> pd.DataFrame:
    """Section 18.3: select the top-N projects by |sum_i lambda_if| from the last full
    simulation run for 3m daily commercial imagery.

    `projects_impact` must have columns `project_id` and `impact` (the pre-computed
    |Σ_i λ_if| for each project — this function does not itself touch simulation output,
    it only ranks and truncates what it is given). Returns a NEW dataframe, sorted
    descending by impact, truncated to `priority_tier.n_projects`, with an added `rank`
    column (1-indexed).
    """
    n = int(params.value("priority_tier.n_projects"))
    ranked = projects_impact.sort_values("impact", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked = ranked.head(n).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def persist_detection_results(
    con,
    projects_update: pd.DataFrame,
    physical_history: pd.DataFrame,
) -> None:
    """Write `run_detection`'s outputs to the store. Imported lazily so this module remains
    importable before `ufe.store` exists (CONTRACT: modules must be importable without
    side effects; this function is the only place that touches the DB)."""
    from ufe.store import db

    db.write_table(con, "project_physical_history", physical_history)
    # `projects_update` only carries the two changed columns plus the key; merging into the
    # full `projects` row is the caller's responsibility (it must read-modify-write the row
    # since this module does not own the `projects` schema).
    db.write_table(con, "projects_physical_state_updates", projects_update)
