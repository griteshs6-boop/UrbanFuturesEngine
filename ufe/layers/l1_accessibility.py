"""Module 4 — Layer 1, accessibility (spec Section 8).

Public entry point: :func:`apply_accessibility`.

**Simulation-time module.** It reads an already-materialised
:class:`~ufe.layers.routing.MatrixSet` and performs no I/O of any kind — no network, no
filesystem, no database (CONTRACT.md rule 3). Building the matrices is the job of
`ufe.layers.routing.precompute_matrices`, which runs at ingestion time; nothing here can
reach a routing backend, and there are tests that assert it.

What it computes
----------------
Section 8.1  gravity accessibility per purpose, combined across modes by share and across
             purposes in logs
Section 8.2  opportunity definitions, with the POI fallbacks
Section 8.3  consumes the mode-specific travel-time matrices
Section 8.4  station proximity weight, with **exclusive** distance bands (Section 21)
Section 8.5  the output column set

Every coefficient is read from ``config/params/accessibility.yaml``.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import MissingParameter, UFEError
from ufe.layers.routing import MatrixSet, StationBand, station_decay_bands
from ufe.params import Params

logger = logging.getLogger(__name__)

__all__ = [
    "OUTPUT_COLUMNS",
    "PURPOSES",
    "apply_accessibility",
    "opportunities",
    "station_decay_weight",
    "cumulative_thresholds_min",
]

ZERO, ONE = 0, 1
NAMESPACE = "accessibility"

PURPOSES: tuple[str, ...] = ("work", "retail", "education", "health")

# Section 8.5. The `jobs_Xmin` names also carry the cumulative-opportunity thresholds; see
# `cumulative_thresholds_min`.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "lnA",
    "lnA_work",
    "lnA_retail",
    "lnA_education",
    "lnA_health",
    "jobs_30min",
    "jobs_45min",
    "jobs_60min",
    "station_weight",
)

_JOBS_BAND = re.compile(r"^jobs_(\d+)min$")

# `decay_beta.work` is tabulated for car/two_wheeler/transit/walk. Metro is not listed, so
# for beta purposes it is treated as transit. Reported as a spec ambiguity.
_BETA_MODE_ALIAS: Mapping[str, str] = {"metro": "transit"}


# --------------------------------------------------------------------------------------
# parameter access
# --------------------------------------------------------------------------------------


def cumulative_thresholds_min(params: Params) -> tuple[int, ...]:
    """The `jobs_Xmin` cut-offs, in minutes (Section 8.5).

    Preference order:

    1. ``accessibility.cumulative_opportunity.thresholds_min`` — **this leaf does not exist
       in ``config/params/accessibility.yaml`` on disk.** It is read first so that adding it
       takes effect immediately, and its absence is reported rather than papered over with
       a Python literal.
    2. failing that, the thresholds are parsed out of the *column names* the landed
       ``cells`` schema mandates (``jobs_30min`` / ``jobs_45min`` / ``jobs_60min``). The
       numbers therefore still come from a contract artefact rather than from a literal
       typed into this module, but they are pinned by the schema, not tunable.
    """
    try:
        raw = params.value(f"{NAMESPACE}.cumulative_opportunity.thresholds_min")
        return tuple(int(x) for x in raw)
    except MissingParameter:
        thresholds = tuple(
            int(m.group(ONE)) for m in (_JOBS_BAND.match(c) for c in OUTPUT_COLUMNS) if m
        )
        if not thresholds:
            raise
        return thresholds


def _purpose_weight(params: Params, purpose: str) -> float:
    return float(params.value(f"{NAMESPACE}.purposes.{purpose}.weight"))


def _mode_share(params: Params, mode: str) -> float:
    return float(params.value(f"{NAMESPACE}.modes.{mode}.share"))


def _beta(params: Params, purpose: str, mode: str) -> float:
    for candidate in (mode, _BETA_MODE_ALIAS.get(mode), "default"):
        if candidate is None:
            continue
        try:
            return float(params.value(f"{NAMESPACE}.decay_beta.{purpose}.{candidate}"))
        except MissingParameter:
            continue
    raise MissingParameter(
        f"no decay beta for purpose {purpose!r} and mode {mode!r}: tried "
        f"{NAMESPACE}.decay_beta.{purpose}.{{{mode},default}}"
    )


# --------------------------------------------------------------------------------------
# Section 8.2 — opportunities
# --------------------------------------------------------------------------------------


def opportunities(
    cells: pd.DataFrame, params: Params, destinations: Sequence[str]
) -> dict[str, np.ndarray]:
    """`O_j` per purpose, aggregated to the destination (res-8) cells — Section 8.2.

    Ambiguity, resolved and reported: Section 8.2 defines retail opportunity as
    "`floorspace_com_sqm_j` restricted to retail POIs". No column carries commercial
    floorspace split by POI category, so this uses `floorspace_com_sqm` where the cell has
    at least one retail POI, and the `retail_poi_count * retail_sqm_per_poi` fallback
    otherwise. Education and health follow the printed rule exactly: the measured count
    when present, the per-POI imputation when it is zero.
    """
    imputation = f"{NAMESPACE}.opportunity_imputation"
    jobs = cells["jobs_by_sector"].map(lambda v: float(np.sum(np.asarray(v, dtype=float))))

    def col(name: str) -> np.ndarray:
        if name not in cells.columns:
            return np.zeros(len(cells), dtype=float)
        return cells[name].fillna(ZERO).to_numpy(dtype=float)

    retail_poi = col("retail_poi_count")
    com = col("floorspace_com_sqm")
    retail = np.where(
        (retail_poi > ZERO) & (com > ZERO),
        com,
        retail_poi * float(params.value(f"{imputation}.retail_sqm_per_poi")),
    )

    seats = col("school_seats")
    education = np.where(
        seats > ZERO,
        seats,
        col("education_poi_count")
        * float(params.value(f"{imputation}.education_seats_per_poi")),
    )

    beds = col("hospital_beds")
    health = np.where(
        beds > ZERO,
        beds,
        col("health_poi_count") * float(params.value(f"{imputation}.health_beds_per_poi")),
    )

    frame = pd.DataFrame(
        {
            "h3_res8": cells["h3_res8"].to_numpy(),
            "work": jobs.to_numpy(dtype=float),
            "retail": retail,
            "education": education,
            "health": health,
        }
    )
    grouped = frame.groupby("h3_res8", sort=True).sum()
    index = pd.Index(destinations, name="h3_res8")
    grouped = grouped.reindex(index).fillna(ZERO)
    return {purpose: grouped[purpose].to_numpy(dtype=float) for purpose in PURPOSES}


# --------------------------------------------------------------------------------------
# Section 8.4 — station proximity, EXCLUSIVE bands
# --------------------------------------------------------------------------------------


def station_decay_weight(
    distance_m: float, bands: Sequence[StationBand], *, feeder: bool = False
) -> float:
    """Weight for ONE station at `distance_m` walk-network metres.

    Section 21's failure mode is a cumulative band ladder — "a 300 m cell gets both the
    0-500 and 500-1000 premium". The bands here are **exclusive**: the first band whose
    `max_m` contains the distance wins and the function returns immediately. Nothing is ever
    summed. Bands carrying `requires: feeder_or_park_ride` apply only to feeder / park-and-
    ride stations; beyond the last applicable band the weight is 0.
    """
    for band in bands:
        if band.requires and not feeder:
            continue
        if distance_m <= band.max_m:
            return float(band.w)
    return float(ZERO)


def _station_weights(matrices: MatrixSet, params: Params) -> np.ndarray:
    n_o = len(matrices.origins)
    if matrices.station_walk_dist_m is None:
        return np.zeros(n_o, dtype=float)

    bands = station_decay_bands(params)
    dist = np.asarray(matrices.station_walk_dist_m, dtype=float)
    feeder = np.asarray(
        matrices.station_feeder
        if matrices.station_feeder
        else [False] * dist.shape[ONE],
        dtype=bool,
    )

    # Vectorised exclusive-band lookup: pick each band's weight only where the distance
    # falls in that band and in no earlier one.
    weights = np.zeros_like(dist, dtype=float)
    assigned = np.zeros_like(dist, dtype=bool)
    for band in bands:
        eligible = np.ones_like(dist, dtype=bool) if not band.requires else np.broadcast_to(
            feeder[None, :], dist.shape
        )
        hit = (~assigned) & eligible & (dist <= band.max_m)
        weights = np.where(hit, float(band.w), weights)
        assigned = assigned | hit
    # Section 8.4: max over stations.
    return weights.max(axis=ONE) if dist.shape[ONE] else np.zeros(n_o, dtype=float)


# --------------------------------------------------------------------------------------
# Section 8.1 / 8.5 — the layer
# --------------------------------------------------------------------------------------


def _modes(matrices: MatrixSet) -> tuple[str, ...]:
    return tuple(sorted(matrices.minutes))


def _accessibility_by_purpose(
    matrices: MatrixSet, params: Params, opp: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """`A_i^p = sum_m share_m * sum_j O_j^p exp(-beta_{m,p} t^m_ij)` — Section 8.1.

    A mode with no matrix contributes **zero**, rather than having its share redistributed.
    That is the documented choice: it keeps `A` monotone in the network, so opening a metro
    line can only raise accessibility, which is what the Section 8 ACCEPTANCE block asserts.
    Renormalising instead would let a new mode *lower* `lnA` for cells it does not serve.
    """
    n_o = len(matrices.origins)
    out: dict[str, np.ndarray] = {}
    for purpose in PURPOSES:
        total = np.zeros(n_o, dtype=float)
        for mode in _modes(matrices):
            share = _mode_share(params, mode)
            beta = _beta(params, purpose, mode)
            t = np.asarray(matrices.minutes[mode], dtype=float)
            with np.errstate(over="ignore"):
                decay = np.exp(-beta * t)
            decay = np.where(np.isfinite(t), decay, ZERO)
            total = total + share * (decay @ opp[purpose])
        out[purpose] = total
    return out


def _cumulative_jobs(
    matrices: MatrixSet, params: Params, jobs: np.ndarray
) -> dict[int, np.ndarray]:
    """Mode-share-weighted cumulative opportunity within each threshold — Section 8.5."""
    n_o = len(matrices.origins)
    out: dict[int, np.ndarray] = {}
    for threshold in cumulative_thresholds_min(params):
        total = np.zeros(n_o, dtype=float)
        for mode in _modes(matrices):
            share = _mode_share(params, mode)
            t = np.asarray(matrices.minutes[mode], dtype=float)
            within = (t <= threshold).astype(float)
            total = total + share * (within @ jobs)
        out[threshold] = total
    return out


def apply_accessibility(
    cells: pd.DataFrame,
    params: Params,
    matrices: MatrixSet,
    **kwargs: object,
) -> pd.DataFrame:
    """Add the Section 8.5 accessibility columns to `cells`.

    Pure: returns a NEW frame with the same index and row count. Cells that are not in
    `matrices.origins` (Section 5.2 restricts the matrix to in-city, occupied or developable
    cells) get NaN, not zero — a cell with no computed accessibility is missing, not
    inaccessible.

    Parameters
    ----------
    cells
        A `cells` frame. Only reads columns; never mutates.
    params
        Loaded parameter tree.
    matrices
        Precomputed travel times for one network state. Never a backend: this function
        performs no I/O.
    """
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if not isinstance(matrices, MatrixSet):
        raise UFEError(
            "apply_accessibility takes a precomputed MatrixSet, not a routing backend "
            "(CONTRACT.md rule 3: no network calls at simulation time)"
        )
    if not matrices.minutes:
        raise UFEError("MatrixSet carries no travel-time matrices")

    missing = [c for c in ("h3", "h3_res8", "jobs_by_sector") if c not in cells.columns]
    if missing:
        raise UFEError(f"cells is missing required column(s): {missing}")

    opp = opportunities(cells, params, matrices.destinations)
    A = _accessibility_by_purpose(matrices, params, opp)

    ln_by_purpose = {p: np.log(A[p] + ONE) for p in PURPOSES}
    lnA = np.zeros(len(matrices.origins), dtype=float)
    for purpose in PURPOSES:
        lnA = lnA + _purpose_weight(params, purpose) * ln_by_purpose[purpose]

    cumulative = _cumulative_jobs(matrices, params, opp["work"])
    weights = _station_weights(matrices, params)

    computed = {"lnA": lnA, "station_weight": weights}
    for purpose in PURPOSES:
        computed[f"lnA_{purpose}"] = ln_by_purpose[purpose]
    for threshold, values in cumulative.items():
        computed[f"jobs_{threshold}min"] = values

    unknown = set(computed) - set(OUTPUT_COLUMNS)
    if unknown:
        raise UFEError(
            f"computed column(s) {sorted(unknown)} are not in the Section 8.5 output set; "
            "the cells schema is strict=True, so report the addition rather than emitting it"
        )

    origin_index = pd.Index(matrices.origins, name="h3")
    if origin_index.has_duplicates:
        raise UFEError("MatrixSet.origins contains duplicate h3 ids")
    aligned = pd.DataFrame(computed, index=origin_index).reindex(
        pd.Index(cells["h3"].to_numpy(), name="h3")
    )

    out = cells.copy()
    for column in OUTPUT_COLUMNS:
        out[column] = aligned[column].to_numpy(dtype=float)
    return out
