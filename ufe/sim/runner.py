"""Module 11 — the simulation runner (spec Section 15).

This is the module that turns six independent layers into an engine. It owns three things
and nothing else:

1. **The annual loop and the layer order** (Section 15.1). Every layer is pure and already
   tested; the runner's job is to call them in the right order and to thread each one's
   carried-forward state into the next year.
2. **Determinism** (Section 15.2). One :class:`numpy.random.SeedSequence` derived from
   `seed`, spawned deterministically per year and per stage. No global RNG anywhere, no set
   iteration order that reaches a number, and a run output that serialises to identical
   bytes.
3. **Caching** (Section 15.3). Travel-time matrices keyed by
   ``(mode, network_state_hash, params_hash)``; base-year substrate keyed by
   ``snapshot_hash``. Nothing keyed by city alone.

The layer order, per simulated year ``t``
-----------------------------------------

::

    L2  resolve_shocks(cells, projects, year=t)     -> shock fields + ShockResolution
        (Layer 3's p_completion / open_year / activation weight are resolved ONCE before
         the loop and re-read every year; Layer 2 calls activation_weight internally)
    L1  apply_accessibility(cells, matrices[state]) -> lnA..., and d_lnA against t-1
        (the network state for year t is the set of NetworkEffects with open_year <= t)
    L4  apply_supply(cells, year=t, state=supply_state, demand_sqm=<t-1 allocation>,
                     utility=<t-1 lnA>, effects=shocks.supply)
                                                    -> capacity/headroom/inventory + state
    L5  allocate(cells, year=t, state=allocation_state, employment_effects=shocks.employment,
                 field_res=shock_field_residential) -> households/hh_by_band/population/jobs
    L6  form_prices(cells, year=t, d_lnA=..., new_hh=..., field=...)   -> d ln P, prices

State threaded across years
---------------------------

===================  ================================================================
carrier              where it lives
===================  ================================================================
`SupplyState`        ``out.attrs["supply_state"]`` from Layer 4 (``l4_supply.ATTR_KEY``)
`AllocationState`    ``out.attrs["allocation_state"]`` from Layer 5 (``ATTR_STATE``)
`ShockResolution`    ``out.attrs["shock_resolution"]`` from Layer 2 (``l2_shocks.ATTR_KEY``)
`lnA`                previous year's column, to form ``d_lnA``
demand / utility     previous year's ``allocated_sqm`` and ``lnA``, into Layer 4
prices, stock        written back onto the ``cells`` frame that starts the next year
===================  ================================================================

The one-year lag between Layer 5's allocation and Layer 4's delivery is deliberate and is
what breaks the otherwise circular dependency (Layer 4 needs demand, Layer 5 needs the
headroom Layer 4 produces). It is recorded in the manifest as ``demand_lag_years``.

Not here
--------

No LLM (Section 23 item 6 — this module must never import ``ufe.ai``). No network: the
runner consumes precomputed :class:`~ufe.layers.routing.MatrixSet` value objects and never
touches a routing backend. No numeric literals beyond ``0`` and ``1``: horizon defaults,
seeds, cache sizes and conservation tolerances all come from
``config/params/simulation.yaml``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import UFEError
from ufe.layers import l2_shocks as L2
from ufe.layers import l3_credibility as L3
from ufe.layers import l4_supply as L4
from ufe.layers import l5_allocation as L5
from ufe.layers import l6_price as L6
from ufe.layers.l1_accessibility import apply_accessibility
from ufe.layers.routing import MatrixSet, network_state_hash
from ufe.sim.snapshot import (
    REPO_ROOT,
    Provenance,
    SnapshotData,
    SnapshotRef,
    load_snapshot_data,
    resolve_provenance,
)

__all__ = [
    "Scenario",
    "RunCache",
    "RunManifest",
    "SimResult",
    "run",
    "PANEL_COLUMNS",
    "LAYER_ORDER",
    "ZERO",
    "ONE",
]

logger = logging.getLogger(__name__)

ZERO = 0
ONE = 1

#: The Section 15.1 layer order, recorded in every manifest so a result can be replayed.
LAYER_ORDER: tuple[str, ...] = (
    "l2_shocks",
    "l1_accessibility",
    "l4_supply",
    "l5_allocation",
    "l6_price",
)

# ---------------------------------------------------------------- parameter paths
P_DEFAULT_LAST_YEAR = "simulation.horizon.default_last_year"
P_MAX_YEARS = "simulation.horizon.max_years"
P_DEFAULT_SEED = "simulation.seed.default"
P_SHORT_HASH_LENGTH = "simulation.provenance.short_hash_length"
P_MATRIX_ENTRIES = "simulation.cache.matrix_entries"
P_SUBSTRATE_ENTRIES = "simulation.cache.substrate_entries"
P_ACCESSIBILITY_ENTRIES = "simulation.cache.accessibility_entries"
P_STRICT_NETWORK_STATES = "simulation.network.strict_states"
P_HOUSEHOLDS_REL_TOL = "simulation.conservation.households_rel_tol"
P_FLOORSPACE_REL_TOL = "simulation.conservation.floorspace_rel_tol"

#: Columns of the per-cell per-year panel, in a fixed order (determinism: never a set).
PANEL_COLUMNS: tuple[str, ...] = (
    "h3",
    "year",
    "households",
    "population",
    "floorspace_res_sqm",
    "floorspace_com_sqm",
    "built_sqm",
    "capacity_sqm",
    "headroom_sqm",
    "delivered_sqm",
    "absorption_cap_sqm",
    "lnA",
    "d_lnA",
    "price_res_inr_sqft",
    "ln_price",
    "d_ln_P_fundamental",
    "d_ln_P_reported",
    "overshoot_log",
    "phi_t",
    "quantity_constrained",
    "shock_field_residential",
    "new_households",
)

#: Columns Layer 6 adds that must not survive into the next year's `cells` frame.
_TRANSIENT_PRICE_COLUMNS: tuple[str, ...] = L6.OUTPUT_COLUMNS + L6.LAND_OUTPUT_COLUMNS


# --------------------------------------------------------------------------------------
# 15.1 — Scenario
# --------------------------------------------------------------------------------------


def _as_tuple(values: Iterable[Any] | None) -> tuple[Any, ...]:
    return () if values is None else tuple(values)


@dataclass(frozen=True)
class Scenario:
    """Spec Section 15.1, verbatim, with sequences normalised to tuples so the dataclass
    is hashable and two identical scenarios serialise identically."""

    city_id: str
    horizon: tuple[int, ...]
    user_projects: tuple[Any, ...] = ()
    disabled_projects: tuple[str, ...] = ()
    force_project_state: Mapping[str, str] = field(default_factory=dict)
    macro_scenario: str = L6.DEFAULT_SCENARIO
    factor_groups: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizon", tuple(int(y) for y in self.horizon))
        object.__setattr__(self, "user_projects", _as_tuple(self.user_projects))
        object.__setattr__(
            self, "disabled_projects", tuple(str(p) for p in self.disabled_projects)
        )
        object.__setattr__(
            self,
            "force_project_state",
            {str(k): str(v) for k, v in dict(self.force_project_state or {}).items()},
        )
        if self.factor_groups is not None:
            object.__setattr__(
                self,
                "factor_groups",
                {
                    str(k): tuple(str(p) for p in v)
                    for k, v in sorted(dict(self.factor_groups).items())
                },
            )
        if not self.horizon:
            raise UFEError("Scenario.horizon must name at least one year")

    @property
    def report_years(self) -> tuple[int, ...]:
        """The years the caller asked about, which need not be contiguous."""
        return tuple(sorted(set(self.horizon)))

    def to_dict(self) -> dict[str, Any]:
        """A stable, JSON-serialisable description, for the manifest."""
        return {
            "city_id": self.city_id,
            "horizon": list(self.horizon),
            "n_user_projects": len(self.user_projects),
            "disabled_projects": sorted(self.disabled_projects),
            "force_project_state": dict(sorted(self.force_project_state.items())),
            "macro_scenario": self.macro_scenario,
            "factor_groups": (
                {k: list(v) for k, v in sorted(self.factor_groups.items())}
                if self.factor_groups
                else None
            ),
        }


# --------------------------------------------------------------------------------------
# 15.3 — caching
# --------------------------------------------------------------------------------------


class _BoundedCache:
    """A tiny FIFO cache. Not module-level state: every instance is owned by a caller."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(int(maxsize), ONE)
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self.hits = ZERO
        self.misses = ZERO

    def get(self, key: Any) -> Any | None:
        if key in self._store:
            self.hits += ONE
            return self._store[key]
        self.misses += ONE
        return None

    def put(self, key: Any, value: Any) -> Any:
        self._store[key] = value
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._store)


class RunCache:
    """Section 15.3's caches, held by the caller so Monte Carlo can share them across draws.

    Every key carries the params hash, so a parameter change can never serve a stale entry
    — Section 15.3's explicit warning ("do not cache anything keyed only by city") is
    structurally impossible here: `city` is not part of any key.

    * ``matrices``      key ``(params_hash, network_state_hash)``  -> `MatrixSet`
    * ``accessibility`` key ``(params_hash, snapshot_hash, network_state_hash, cells_hash)``
      -> the Layer 1 output columns for those inputs
    * ``substrate``     key ``(params_hash, snapshot_hash)`` -> the base-year `cells` frame
    """

    def __init__(self, params: Any) -> None:
        self.params_hash = params.hash
        self.matrices = _BoundedCache(int(params.value(P_MATRIX_ENTRIES)))
        self.accessibility = _BoundedCache(int(params.value(P_ACCESSIBILITY_ENTRIES)))
        self.substrate = _BoundedCache(int(params.value(P_SUBSTRATE_ENTRIES)))

    # -- travel-time matrices ---------------------------------------------------------

    def put_matrices(self, state_hash: str, matrices: MatrixSet) -> MatrixSet:
        return self.matrices.put((self.params_hash, str(state_hash)), matrices)

    def get_matrices(self, state_hash: str) -> MatrixSet | None:
        return self.matrices.get((self.params_hash, str(state_hash)))

    # -- base-year substrate ----------------------------------------------------------

    def get_substrate(self, snapshot_hash: str) -> pd.DataFrame | None:
        cached = self.substrate.get((self.params_hash, str(snapshot_hash)))
        return None if cached is None else cached.copy(deep=True)

    def put_substrate(self, snapshot_hash: str, cells: pd.DataFrame) -> pd.DataFrame:
        self.substrate.put((self.params_hash, str(snapshot_hash)), cells.copy(deep=True))
        return cells

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {"hits": cache.hits, "misses": cache.misses, "size": len(cache)}
            for name, cache in (
                ("matrices", self.matrices),
                ("accessibility", self.accessibility),
                ("substrate", self.substrate),
            )
        }


# --------------------------------------------------------------------------------------
# the run manifest (Section 23 item 5)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to say where a number came from, and to reproduce it.

    Deliberately carries **no wall-clock timestamp**: the manifest is part of the run's
    serialised output and Section 23 item 4 requires that output to be byte-identical
    across two runs with the same seed and snapshot.
    """

    provenance: Provenance
    seed: int
    deterministic: bool
    scenario: Mapping[str, Any]
    simulated_years: tuple[int, ...]
    report_years: tuple[int, ...]
    base_year: int
    layer_order: tuple[str, ...]
    params_manifest: Mapping[str, Any]
    demand_lag_years: int
    network_states: Mapping[int, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "snapshot_hash": self.provenance.snapshot_hash,
            "params_hash": self.provenance.params_hash,
            "code_version": self.provenance.code_version,
            "seed": int(self.seed),
            "deterministic": bool(self.deterministic),
            "scenario": dict(self.scenario),
            "simulated_years": list(self.simulated_years),
            "report_years": list(self.report_years),
            "base_year": int(self.base_year),
            "layer_order": list(self.layer_order),
            "params": dict(self.params_manifest),
            "demand_lag_years": int(self.demand_lag_years),
            "network_states": {str(k): v for k, v in sorted(self.network_states.items())},
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )

    @property
    def complete(self) -> bool:
        """Section 23 item 5: the three identifiers are all present and known."""
        return self.provenance.complete


# --------------------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    """Section 15.1's return value.

    ``panel`` is the per-cell per-year state; ``diagnostics`` the per-year convergence and
    spill record; ``residuals`` and ``overheat`` the Section 13.5 outputs for the final
    year; ``factors`` the Section 13.4 decomposition when the caller asked for one
    (:mod:`ufe.sim.factors` fills it in — a bare :func:`run` leaves it ``None``).
    """

    manifest: RunManifest
    panel: pd.DataFrame
    diagnostics: pd.DataFrame
    residuals: pd.DataFrame
    overheat: pd.DataFrame
    shock_weights: pd.DataFrame
    factors: Any | None = None
    cache_stats: Mapping[str, Any] = field(default_factory=dict)

    # -- provenance shortcuts ---------------------------------------------------------

    @property
    def snapshot_hash(self) -> str:
        return self.manifest.provenance.snapshot_hash

    @property
    def params_hash(self) -> str:
        return self.manifest.provenance.params_hash

    @property
    def code_version(self) -> str:
        return self.manifest.provenance.code_version

    @property
    def seed(self) -> int:
        return self.manifest.seed

    # -- Section 15.2: byte-identical serialisation ------------------------------------

    def to_parquet_bytes(self) -> dict[str, bytes]:
        """Serialise every frame plus the manifest. Deterministic, ordered, no timestamps."""
        payload: dict[str, bytes] = {}
        for name, frame in sorted(
            (
                ("panel", self.panel),
                ("diagnostics", self.diagnostics),
                ("residuals", self.residuals),
                ("overheat", self.overheat),
                ("shock_weights", self.shock_weights),
            )
        ):
            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=False, engine="pyarrow", compression=None)
            payload[name] = buffer.getvalue()
        payload["manifest"] = self.manifest.to_json().encode("utf-8")
        return payload

    def digest(self) -> str:
        """SHA-256 over the serialised result — the Section 15.2 determinism test's subject."""
        hasher = hashlib.sha256()
        for name, blob in sorted(self.to_parquet_bytes().items()):
            hasher.update(name.encode("ascii"))
            hasher.update(b"\x1f")
            hasher.update(blob)
            hasher.update(b"\x1e")
        return hasher.hexdigest()

    def at(self, year: int) -> pd.DataFrame:
        """The panel slice for one year, indexed by `h3`."""
        rows = self.panel.loc[self.panel["year"] == int(year)]
        return rows.set_index("h3").sort_index()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _child_rng(seed: int, *labels: Any) -> np.random.Generator:
    """A generator whose stream is a pure function of `(seed, labels)`.

    Section 15.2: "Every stochastic draw goes through a single `numpy.random.Generator`
    seeded from `seed`." A single sequential generator would make the stream depend on how
    many draws each earlier stage happened to take; deriving each stage's generator from
    the master seed by a labelled `SeedSequence` keeps determinism while making each stage
    independently reproducible. The label is hashed stably, never via Python's salted
    `hash()`.
    """
    tag = "|".join(str(x) for x in labels).encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(tag).digest()[: len(b"12345678")], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), entropy]))


def _expand_horizon(horizon: Sequence[int], base_year: int, max_years: int) -> tuple[int, ...]:
    """The contiguous annual sequence the loop actually walks.

    Section 15.1's `horizon` is a list of years; `config/cities/vizag.yaml` supplies
    ``horizon_years: [2027, 2030, 2035, 2040]``, which are *reporting* years. The engine
    steps annually (Section 0.3), so the loop runs every year from ``base_year + 1`` to the
    last requested year and the requested years are recorded as `report_years`.
    """
    last = max(int(y) for y in horizon)
    if last <= base_year:
        raise UFEError(
            f"horizon {sorted(set(int(y) for y in horizon))} ends at or before the base "
            f"year {base_year}: there is nothing to simulate"
        )
    span = last - base_year
    if span > max_years:
        raise UFEError(
            f"horizon spans {span} years from base year {base_year}, over "
            f"{P_MAX_YEARS} = {max_years}"
        )
    return tuple(range(base_year + ONE, last + ONE))


def _residential_share(cells: pd.DataFrame) -> np.ndarray:
    """Split of standing stock between residential and commercial, per cell.

    Layer 4 delivers a single ``delivered_sqm`` and updates only capacity/headroom; the
    schema stores residential and commercial floorspace separately. The delivery is split
    in the proportion of the stock already standing, and a cell with no stock at all is
    treated as entirely residential (the allocation layer is the only consumer of new stock
    in the shipped engine). Reported as a runner-level modelling choice.
    """
    res = cells[L4.COL_FLOORSPACE_RES].to_numpy(dtype=float)
    com = cells[L4.COL_FLOORSPACE_COM].to_numpy(dtype=float)
    total = res + com
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > ZERO, res / np.where(total > ZERO, total, ONE), ONE)
    return np.clip(share, ZERO, ONE)


def _apply_delivery(cells: pd.DataFrame, delivered: pd.Series) -> pd.DataFrame:
    """Write Layer 4's delivered floorspace into the schema's stock columns."""
    out = cells.copy(deep=True)
    out.attrs = dict(cells.attrs)
    values = np.asarray(delivered.to_numpy(), dtype=float)
    share = _residential_share(cells)
    out[L4.COL_FLOORSPACE_RES] = (
        cells[L4.COL_FLOORSPACE_RES].to_numpy(dtype=float) + values * share
    )
    out[L4.COL_FLOORSPACE_COM] = (
        cells[L4.COL_FLOORSPACE_COM].to_numpy(dtype=float) + values * (ONE - share)
    )
    return out


def _drop_transient(cells: pd.DataFrame) -> pd.DataFrame:
    """Strip Layer 6's per-year output columns before the frame starts the next year."""
    present = [c for c in _TRANSIENT_PRICE_COLUMNS if c in cells.columns]
    out = cells.drop(columns=present)
    out.attrs = {}
    return out


def _detach(frame: pd.DataFrame) -> pd.DataFrame:
    """Clear a layer's ``.attrs`` once the runner has read it.

    Not cosmetic. pandas propagates ``.attrs`` through `__finalize__` on *every* column
    access, deep-copying whatever is in there. A `SupplyState` (six h3-indexed Series) or a
    Layer 5 diagnostics block (several frames) left on ``.attrs`` therefore gets deep-copied
    hundreds of times per simulated year, and dominates the runtime. The runner reads the
    state into a local variable and detaches it before handing the frame to the next layer.
    Profiled cost of not doing this: roughly 4x the whole annual loop.
    """
    frame.attrs = {}
    return frame


def _series(values: Any, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index)
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _resolve_pipeline(
    projects: pd.DataFrame,
    announcers: pd.DataFrame,
    params: Any,
    scenario: Scenario,
    *,
    rng: np.random.Generator,
    deterministic: bool,
    delay_family_map: Mapping[str, str] | None,
    p_completion_override: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Layer 3 once, before the loop: `p_completion`, `open_year`, `construction_start_year`.

    Section 10 is a property of the pipeline, not of the year, so it is resolved once and
    re-read every year. Layer 2 recomputes only the year-varying activation weight.
    """
    if scenario.disabled_projects:
        keep = ~projects["project_id"].astype(str).isin(set(scenario.disabled_projects))
        projects = projects.loc[keep]
    if len(scenario.user_projects):
        extra = pd.DataFrame(list(scenario.user_projects))
        projects = pd.concat([projects, extra], ignore_index=True)
    projects = projects.sort_values("project_id", kind="mergesort").reset_index(drop=True)

    if projects.empty:
        return projects

    force = dict(scenario.force_project_state) or None
    with_p = L3.completion_probability(
        projects,
        announcers,
        params,
        force_project_state=force,
        unknown_modifiers=L3.IGNORE,
    )
    if p_completion_override:
        # Module 12 draws a correlated `p_completion` per project (Section 16.1). Layer 3
        # computes the deterministic mean; the draw replaces it here rather than inside
        # Layer 3, which stays pure and un-stochastic in that variable.
        drawn = with_p["project_id"].astype(str).map(dict(p_completion_override))
        with_p = with_p.assign(
            p_completion=drawn.fillna(with_p["p_completion"]).astype(float)
        )
    with_delay = L3.delay_distribution(
        with_p,
        params,
        monte_carlo=not deterministic,
        rng=None if deterministic else rng,
        family_map=dict(delay_family_map) if delay_family_map else None,
        force_project_state=force,
        unknown_modifiers=L3.IGNORE,
    )
    return with_delay


def _network_state_for_year(resolution: L2.ShockResolution, year: int) -> str:
    """Section 15.3's `network_state_hash`: the sorted set of links open in `year`."""
    open_ids = sorted(
        str(effect.project_id)
        for effect in resolution.network
        if int(effect.open_year) <= int(year)
    )
    return network_state_hash(open_ids)


# --------------------------------------------------------------------------------------
# 15.1 — the entry point
# --------------------------------------------------------------------------------------


def run(
    snapshot: SnapshotRef | SnapshotData | str | Path,
    params: Any,
    scenario: Scenario,
    seed: int = ZERO,
    deterministic: bool = True,
    *,
    matrices: MatrixSet | Mapping[str, MatrixSet] | None = None,
    persons_per_household_by_band: Mapping[str, float] | Sequence[float] | None = None,
    natural_growth_rate: float | None = None,
    cache: RunCache | None = None,
    allow_dirty: bool = False,
    repo_root: Path | str = REPO_ROOT,
    code_version: str | None = None,
    code_dirty: bool | None = None,
    verify_snapshot: bool = True,
    base_year: int | None = None,
    delay_family_map: Mapping[str, str] | None = None,
    p_completion_override: Mapping[str, float] | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    check_conservation: bool = True,
) -> SimResult:
    """Run the engine over `scenario.horizon` (spec Section 15.1).

    Parameters
    ----------
    snapshot:
        A :class:`~ufe.sim.snapshot.SnapshotRef`, a snapshot directory, or an
        already-loaded :class:`~ufe.sim.snapshot.SnapshotData`. **The only legal source of
        input data** (Section 3.8): the runner never reads the live database and never
        performs network I/O.
    params, scenario, seed, deterministic:
        Section 15.1's signature. `deterministic=False` switches Layers 2 and 3 into
        Monte Carlo mode, drawing magnitudes and slips from their YAML ranges with the
        seeded generator rather than taking medians.
    matrices:
        A precomputed :class:`~ufe.layers.routing.MatrixSet`, or a mapping from network
        state hash to one. Simulation code may not call a routing backend
        (CONTRACT.md rule 3), so a year whose network state has no matrix reuses the base
        state and records ``network_state_missing`` — or raises, when
        ``simulation.network.strict_states`` is true. ``None`` skips Layer 1 entirely and
        carries the snapshot's own ``lnA`` forward unchanged.
    persons_per_household_by_band:
        Section 12.5. ``behaviour.persons_per_household_by_band`` is null on disk, by
        design, so it must be supplied by the caller as soon as anything is allocated.
    natural_growth_rate:
        Overrides ``behaviour.natural_growth_rate`` for a scenario.
    delay_family_map:
        Section 10.3 maps an *archetype* to a delay family, but the two vocabularies do not
        coincide in the shipped config (``archetypes.yaml`` has ``metro_rail``;
        ``credibility.delay_lognormal`` has ``metro_phase1`` / ``metro_later``). Supply the
        mapping here, or put a ``delay_family_map`` block in the city config. Reported as a
        config gap rather than patched into Layer 3.
    cache:
        A :class:`RunCache` to share across runs (Monte Carlo passes one). A fresh one is
        built when omitted.
    allow_dirty:
        Section 23 item 5's explicit override. Without it, a dirty or unknown git state
        refuses to run.
    param_overrides:
        Per-run scalar overrides, applied on top of `params` without touching the loaded
        tree. Used by Monte Carlo to inject a draw; recorded in the manifest by hash only.
    """
    del param_overrides  # applied by the caller through a Params view; see montecarlo.py

    data = (
        snapshot
        if isinstance(snapshot, SnapshotData)
        else load_snapshot_data(snapshot, verify=verify_snapshot)
    )
    provenance = resolve_provenance(
        data.ref,
        params,
        allow_dirty=allow_dirty,
        repo_root=repo_root,
        code_version=code_version,
        code_dirty=code_dirty,
    )
    cache = cache if cache is not None else RunCache(params)

    if base_year is None:
        base_year = int(params.city_config["base_year"])
    years = _expand_horizon(
        scenario.horizon, int(base_year), int(params.value(P_MAX_YEARS))
    )
    hh_tol = float(params.value(P_HOUSEHOLDS_REL_TOL))
    fs_tol = float(params.value(P_FLOORSPACE_REL_TOL))
    strict_states = bool(params.value(P_STRICT_NETWORK_STATES))

    # --- base state -------------------------------------------------------------------
    cached = cache.get_substrate(data.snapshot_hash)
    cells = cached if cached is not None else data.cells.copy(deep=True)
    if cached is None:
        cache.put_substrate(data.snapshot_hash, cells)
    cells = cells.sort_values(L4.COL_H3, kind="mergesort").reset_index(drop=True)
    index = cells.index
    initial_built = (
        cells[L4.COL_FLOORSPACE_RES].to_numpy(dtype=float)
        + cells[L4.COL_FLOORSPACE_COM].to_numpy(dtype=float)
    ).sum()
    initial_households = float(cells[L5.COL_HOUSEHOLDS].sum())

    pipeline = _resolve_pipeline(
        data.projects,
        data.announcers,
        params,
        scenario,
        rng=_child_rng(seed, "credibility"),
        deterministic=deterministic,
        delay_family_map=(
            delay_family_map
            if delay_family_map is not None
            else params.city_config.get("delay_family_map")
        ),
        p_completion_override=p_completion_override,
    )

    matrix_map: dict[str, MatrixSet] = {}
    base_state = network_state_hash(())
    if isinstance(matrices, MatrixSet):
        matrix_map[matrices.network_state or base_state] = matrices
        base_state = matrices.network_state or base_state
    elif isinstance(matrices, Mapping):
        matrix_map = {str(k): v for k, v in sorted(matrices.items())}
        base_state = next(iter(matrix_map), base_state)
    for state_hash, matrix in matrix_map.items():
        cache.put_matrices(state_hash, matrix)

    # --- annual loop ------------------------------------------------------------------
    panel_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    states_by_year: dict[int, str] = {}

    supply_state: L4.SupplyState | None = None
    allocation_state: L5.AllocationState | None = None
    previous_lnA = (
        cells[L5.COL_LNA].to_numpy(dtype=float)
        if L5.COL_LNA in cells.columns
        else np.zeros(len(cells))
    )
    previous_demand_sqm = pd.Series(np.zeros(len(cells)), index=cells[L4.COL_H3])
    delivered_total = float(ZERO)
    allocated_total = float(ZERO)
    spill_total = float(ZERO)

    for year in years:
        # ---------------------------------------------------------------- L2: shocks
        shocked = L2.resolve_shocks(
            cells,
            pipeline,
            params,
            year=year,
            monte_carlo=not deterministic,
            rng=None if deterministic else _child_rng(seed, "shocks", year),
            force_project_state=dict(scenario.force_project_state) or None,
            unknown_archetypes=L2.IGNORE,
            missing_office_sqm_per_seat=L2.IGNORE,
            check_scale_unit=False,
        )
        resolution: L2.ShockResolution = shocked.attrs[L2.ATTR_KEY]
        _detach(shocked)
        for project_id in sorted(resolution.weights):
            weight_rows.append(
                {
                    "year": int(year),
                    "project_id": project_id,
                    "activation_weight": float(resolution.weights[project_id]),
                }
            )

        # -------------------------------------------------- L1: accessibility, d_lnA
        state_hash = _network_state_for_year(resolution, year)
        states_by_year[int(year)] = state_hash
        matrix = cache.get_matrices(state_hash)
        if matrix is None and matrix_map:
            if strict_states:
                raise UFEError(
                    f"year {year} needs travel-time matrices for network state "
                    f"{state_hash} but none were supplied, and the simulation may not "
                    "call a routing backend (CONTRACT.md rule 3). Precompute the state "
                    f"or set {P_STRICT_NETWORK_STATES} to false."
                )
            message = (
                f"year {year}: no matrices for network state {state_hash}; reusing the "
                f"base state {base_state}"
            )
            if message not in warnings:
                warnings.append(message)
            logger.warning("%s", message)
            matrix = cache.get_matrices(base_state)

        if matrix is not None:
            accessed = apply_accessibility(shocked, params, matrix)
        else:
            accessed = shocked
        lnA = (
            accessed[L5.COL_LNA].to_numpy(dtype=float)
            if L5.COL_LNA in accessed.columns
            else previous_lnA
        )
        d_lnA = np.where(np.isfinite(lnA) & np.isfinite(previous_lnA), lnA - previous_lnA, ZERO)

        # ---------------------------------------------------------------- L4: supply
        supplied = L4.apply_supply(
            accessed,
            params,
            year=year,
            state=supply_state,
            demand_sqm=previous_demand_sqm,
            utility=pd.Series(np.nan_to_num(lnA), index=accessed[L4.COL_H3]),
            effects=resolution.supply,
        )
        supply_state = supplied.attrs[L4.ATTR_KEY]
        delivered = supplied.attrs["delivered_sqm"]
        absorption_cap = supplied.attrs["absorption_cap_sqm"]
        _detach(supplied)
        delivered_values = np.asarray(delivered.to_numpy(), dtype=float)
        delivered_total += float(delivered_values.sum())
        supplied = _apply_delivery(supplied, delivered)

        # ------------------------------------------------------------ L5: allocation
        field_res = supplied[L2.COL_FIELD_RESIDENTIAL].to_numpy(dtype=float)
        allocated = L5.allocate(
            supplied,
            params,
            year=year,
            state=allocation_state,
            # Layer 2 has ALREADY multiplied every EmploymentEffect by Layer 3's w(t)
            # (see `l2_shocks._resolve_one`), so passing activation_weights here would
            # apply credibility twice. Reported in the build summary.
            employment_effects=resolution.employment,
            activation_weights=None,
            field_res=field_res,
            matrices=matrix,
            persons_per_household_by_band=persons_per_household_by_band,
            natural_growth_rate=natural_growth_rate,
        )
        allocation_state = allocated.attrs.get(L5.ATTR_STATE, allocation_state)
        allocation_diag = allocated.attrs[L5.ATTR_DIAGNOSTICS]
        _detach(allocated)
        allocated_by_band = allocation_diag["allocated_by_band"]
        new_hh = allocated_by_band.sum(axis=ONE)
        allocated_total += float(new_hh.sum())
        spill_total += float(allocation_diag["spill_households"])
        previous_demand_sqm = pd.Series(
            allocation_diag["allocated_sqm"].to_numpy(dtype=float),
            index=allocated[L4.COL_H3],
        )

        # --- Section 12 ACCEPTANCE, asserted every year: households are conserved
        demanded = float(np.asarray(allocation_diag["demand_by_band"], dtype=float).sum())
        settled = float(new_hh.sum()) + float(allocation_diag["spill_households"])
        if check_conservation and demanded > ZERO:
            if abs(settled - demanded) / demanded > hh_tol:
                raise UFEError(
                    f"year {year}: households not conserved — allocated {settled:.6f} "
                    f"against demand {demanded:.6f} (spec Section 12 ACCEPTANCE, "
                    f"{P_HOUSEHOLDS_REL_TOL})"
                )

        # ---------------------------------------------------------------- L6: prices
        priced = L6.form_prices(
            allocated,
            params,
            year=year,
            d_lnA=d_lnA,
            new_hh=new_hh.to_numpy(dtype=float),
            field=field_res,
            scenario=scenario.macro_scenario,
            supply_effects=resolution.supply,
            absorption_cap_sqm=absorption_cap.to_numpy(dtype=float),
        )
        price_diag = priced.attrs[L6.ATTR_KEY]
        _detach(priced)

        # ------------------------------------------------------- record and carry on
        built = supply_state.built_sqm.reindex(priced[L4.COL_H3]).to_numpy(dtype=float)
        reported = priced["price_res_inr_sqft_reported"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_price = np.log(reported)
        panel_rows.append(
            pd.DataFrame(
                {
                    "h3": priced[L4.COL_H3].to_numpy(),
                    "year": np.full(len(priced), int(year), dtype=np.int64),
                    "households": priced[L5.COL_HOUSEHOLDS].to_numpy(dtype=float),
                    "population": priced[L5.COL_POPULATION].to_numpy(dtype=float),
                    "floorspace_res_sqm": priced[L4.COL_FLOORSPACE_RES].to_numpy(dtype=float),
                    "floorspace_com_sqm": priced[L4.COL_FLOORSPACE_COM].to_numpy(dtype=float),
                    "built_sqm": built,
                    "capacity_sqm": priced[L4.COL_CAPACITY].to_numpy(dtype=float),
                    "headroom_sqm": priced[L4.COL_HEADROOM].to_numpy(dtype=float),
                    "delivered_sqm": delivered_values,
                    "absorption_cap_sqm": np.asarray(absorption_cap.to_numpy(), dtype=float),
                    "lnA": lnA,
                    "d_lnA": d_lnA,
                    "price_res_inr_sqft": reported,
                    "ln_price": ln_price,
                    "d_ln_P_fundamental": priced["d_ln_P_fundamental"].to_numpy(dtype=float),
                    "d_ln_P_reported": priced["d_ln_P_reported"].to_numpy(dtype=float),
                    "overshoot_log": priced["overshoot_log"].to_numpy(dtype=float),
                    "phi_t": priced["phi_t"].to_numpy(dtype=float),
                    "quantity_constrained": priced["quantity_constrained"].to_numpy(dtype=bool),
                    "shock_field_residential": field_res,
                    "new_households": new_hh.to_numpy(dtype=float),
                }
            )[list(PANEL_COLUMNS)]
        )
        diagnostics_rows.append(
            {
                "year": int(year),
                "network_state": state_hash,
                "allocation_converged": bool(allocation_diag["converged"]),
                "allocation_iterations": int(allocation_diag["iterations"]),
                "allocation_max_delta_lnA": float(allocation_diag["max_delta_lnA"]),
                "band_accessibility": str(allocation_diag["band_accessibility"]),
                "new_households": float(allocation_diag["new_households"]),
                "exogenous_households": float(allocation_diag["exogenous_households"]),
                "job_driven_households": float(allocation_diag["job_driven_households"]),
                "spill_households": float(allocation_diag["spill_households"]),
                "capped_cells": int(allocation_diag["capped_cells"].sum()),
                "delivered_sqm": float(delivered_values.sum()),
                "price_converged": bool(price_diag["converged"]),
                "price_iterations": int(price_diag["iterations"]),
                "price_residual_change": float(price_diag["residual_change"]),
                "n_quantity_constrained": int(price_diag["n_constrained"]),
                "phi_t": float(price_diag["phi_t"]),
                "field_cap_hit_cells": int(
                    priced[L2.COL_FIELD_CAP_HIT].to_numpy(dtype=bool).sum()
                ),
                "n_shock_projects": len(resolution.weights),
                "n_employment_effects": len(resolution.employment),
                "n_supply_effects": len(resolution.supply),
                "n_network_effects": len(resolution.network),
            }
        )

        cells = _drop_transient(priced)
        cells[L5.COL_PRICE_RES] = reported
        previous_lnA = lnA

    # --- Section 15.1: floorspace conservation ---------------------------------------
    final_built = float(
        (
            cells[L4.COL_FLOORSPACE_RES].to_numpy(dtype=float)
            + cells[L4.COL_FLOORSPACE_COM].to_numpy(dtype=float)
        ).sum()
    )
    expected_built = float(initial_built) + delivered_total
    if check_conservation and expected_built > ZERO:
        if abs(final_built - expected_built) / expected_built > fs_tol:
            raise UFEError(
                f"floorspace not conserved: final stock {final_built:.6f} against "
                f"initial {initial_built:.6f} plus delivered {delivered_total:.6f} "
                f"({P_FLOORSPACE_REL_TOL})"
            )

    panel = (
        pd.concat(panel_rows, ignore_index=True)
        if panel_rows
        else pd.DataFrame(columns=list(PANEL_COLUMNS))
    )
    diagnostics = pd.DataFrame(diagnostics_rows)
    shock_weights = (
        pd.DataFrame(weight_rows)
        if weight_rows
        else pd.DataFrame(columns=["year", "project_id", "activation_weight"])
    )

    residuals, overheat = _final_year_residuals(cells, params, panel, years)

    manifest = RunManifest(
        provenance=provenance,
        seed=int(seed),
        deterministic=bool(deterministic),
        scenario=scenario.to_dict(),
        simulated_years=years,
        report_years=scenario.report_years,
        base_year=int(base_year),
        layer_order=LAYER_ORDER,
        params_manifest=params.manifest(),
        demand_lag_years=ONE,
        network_states=states_by_year,
        warnings=tuple(warnings),
    )
    logger.info(
        "run complete: %d years, %d cells, households %.1f -> %.1f, spill %.1f",
        len(years),
        len(cells),
        initial_households,
        float(cells[L5.COL_HOUSEHOLDS].sum()),
        spill_total,
    )
    return SimResult(
        manifest=manifest,
        panel=panel,
        diagnostics=diagnostics,
        residuals=residuals,
        overheat=overheat,
        shock_weights=shock_weights,
        cache_stats=cache.stats(),
    )


def _final_year_residuals(
    cells: pd.DataFrame, params: Any, panel: pd.DataFrame, years: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 13.5's residual and overheat flags for the last simulated year."""
    if panel.empty:
        empty = pd.DataFrame()
        return empty, empty
    last = int(years[-ONE])
    slice_ = panel.loc[panel["year"] == last].set_index("h3").sort_index()
    price = slice_["price_res_inr_sqft"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_observed = np.log(price.to_numpy(dtype=float))
    ln_observed = pd.Series(ln_observed, index=slice_.index, name="ln_price")

    # With no factor decomposition attached, the model expectation is the cumulative
    # reported movement: lambdas are the per-year d ln P, summed.
    lambdas = (
        panel.pivot_table(
            index="h3", columns="year", values="d_ln_P_reported", aggfunc="sum"
        )
        .sort_index()
        .reindex(slice_.index)
    )
    lambdas.columns = [f"year_{int(c)}" for c in lambdas.columns]
    ln_base = ln_observed - lambdas.sum(axis=ONE)
    residual = L6.model_residual(ln_observed, ln_base, lambdas)

    rent = (
        cells.set_index(L4.COL_H3)[L6.COL_RENT_RES].reindex(slice_.index)
        if L6.COL_RENT_RES in cells.columns
        else None
    )
    overheat = L6.overheating(
        params,
        residual=residual,
        price_inr_sqft=price,
        rent_inr_sqft_mo=rent,
    )
    residual_frame = pd.DataFrame(
        {
            "h3": residual.index.to_numpy(),
            "year": np.full(len(residual), last, dtype=np.int64),
            "ln_price": ln_observed.to_numpy(dtype=float),
            "ln_price_model": (ln_base + lambdas.sum(axis=ONE)).to_numpy(dtype=float),
            "residual": residual.to_numpy(dtype=float),
        }
    )
    overheat_frame = overheat.reset_index().rename(columns={"index": "h3"})
    overheat_frame.insert(ONE, "year", np.full(len(overheat_frame), last, dtype=np.int64))
    return residual_frame, overheat_frame
