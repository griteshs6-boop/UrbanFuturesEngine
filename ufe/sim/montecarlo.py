"""Module 12 — Monte Carlo (spec Section 16).

Three jobs, in the order Section 16 states them.

**16.1 Sampling.** Every draw samples the parameter tree through
:meth:`ufe.params.Params.sample`, with an explicitly seeded generator per draw, and the
draws are *not* independent: a Gaussian copula imposes the correlation structure Section
16.1 names —

===================================================  ======
`p_completion` across projects sharing an announcer  ρ = 0.5
`p_completion` across public projects, same state    ρ = 0.3
`phi_t` and `eta` (demand price elasticity)          ρ = −0.3
===================================================  ======

The copula is applied on the latent normal scale and the resulting uniforms are pushed
through each parameter's own marginal by handing ``Params.sample`` a generator whose
``uniform`` returns the pre-drawn value (:class:`_CopulaRng`). Nothing about the marginals
is re-implemented here: a triangular-tagged leaf, a lognormal leaf and a bare scalar all
keep whatever ``Params.sample`` does with them.

**16.2 Efficiency.** The expensive step is accessibility, so the ensemble shares one
:class:`~ufe.sim.runner.RunCache` across every draw: travel-time matrices are built (or
loaded) once per network state, keyed by ``(params_hash, network_state_hash)``, never once
per draw. Section 16.2 then proposes interpolating ``lnA`` over a 3-point beta grid; this
implementation does **not** approximate. Once the matrix is cached, ``apply_accessibility``
is a decay-weighted mat-vec over that matrix, which is cheap next to the routing it
replaces, so each draw recomputes ``lnA`` *exactly* at its own drawn beta. The
``montecarlo.efficiency.beta_grid_points`` knob is kept in YAML for the day the exact path
becomes the bottleneck; the approximation it describes is documented but unused, and the
deviation is reported.

**16.3 Outputs.** Per cell per year: p10/p25/p50/p75/p90 of ln price and of built
floorspace. Per cell: P(outperform the city median) over the full horizon, and P(top
decile) at each horizon year. Per factor: the distribution of lambda (opt-in, because it
multiplies the run count by ``2 + len(factors)``).

Determinism. The whole ensemble is reproducible from one master seed: draw ``k`` uses
``base_seed + k`` exactly as Section 16.2 specifies, so a draw's result never depends on how
many workers ran or in what order they finished. `workers > 1` therefore changes the wall
clock and nothing else.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats

from ufe.errors import UFEError
from ufe.sim import factors as _factors
from ufe.sim import runner as _runner
from ufe.sim.runner import RunCache, Scenario, SimResult
from ufe.sim.snapshot import SnapshotData, SnapshotRef, load_snapshot_data

__all__ = [
    "MonteCarloConfig",
    "Draw",
    "EnsembleResult",
    "ParamsDraw",
    "load_config",
    "correlation_matrix",
    "sample_draw",
    "run_ensemble",
]

logger = logging.getLogger(__name__)

ZERO = _runner.ZERO
ONE = _runner.ONE

# ---------------------------------------------------------------- parameter paths
NS = "montecarlo"
P_DEFAULT_N = f"{NS}.draws.default_n"
P_BASE_SEED = f"{NS}.draws.base_seed"
P_CONVERGENCE_TOL = f"{NS}.draws.convergence_p50_rel_tol"
P_WORKERS = f"{NS}.parallel.workers"
P_CHUNK_SIZE = f"{NS}.parallel.chunk_size"
P_MAX_NETWORK_STATES = f"{NS}.efficiency.max_network_states"
P_BETA_GRID_POINTS = f"{NS}.efficiency.beta_grid_points"
P_QUANTILES = f"{NS}.quantiles"
P_NOMINAL_COVERAGE = f"{NS}.bands.nominal_coverage"
P_COVERAGE_TOLERANCE = f"{NS}.bands.coverage_tolerance"
P_RHO_ANNOUNCER = f"{NS}.correlation.same_announcer_p_completion"
P_RHO_PUBLIC = f"{NS}.correlation.same_state_public_p_completion"
P_RHO_PHI_ETA = f"{NS}.correlation.phi_eta"
P_MIN_EIGENVALUE = f"{NS}.correlation.min_eigenvalue"
P_ACCEPTANCE_TOL = f"{NS}.correlation.acceptance_tolerance"
P_BETA_CONCENTRATION = f"{NS}.p_completion.beta_concentration"
P_MEAN_FLOOR = f"{NS}.p_completion.mean_floor"
P_SAMPLED = f"{NS}.sampled_paths"
P_TOP_DECILE_SHARE = "simulation.output.top_decile_share"

#: The two Section 16.1 paths the phi/eta copula pair refers to.
P_ETA = "price.hedonic.eta_demand_price"

#: Latent copula dimension names that are not projects.
DIM_PHI = "__phi_t__"
DIM_ETA = "__eta__"


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloConfig:
    """`config/params/montecarlo.yaml`, resolved. Nothing here is a Python literal."""

    n_draws: int
    base_seed: int
    workers: int
    chunk_size: int
    max_network_states: int
    beta_grid_points: int
    quantiles: tuple[float, ...]
    quantile_names: tuple[str, ...]
    nominal_coverage: float
    coverage_tolerance: float
    rho_announcer: float
    rho_public: float
    rho_phi_eta: float
    min_eigenvalue: float
    acceptance_tolerance: float
    beta_concentration: float
    mean_floor: float
    top_decile_share: float
    convergence_p50_rel_tol: float
    sampled_paths: tuple[str, ...]
    mode_share_parent: str
    mode_share_leaf: str
    macro_scenarios_prefix: str
    macro_probabilities_prefix: str


def load_config(params: Any) -> MonteCarloConfig:
    """Resolve :class:`MonteCarloConfig` from the loaded parameter tree."""
    quantile_block = params.get(P_QUANTILES)
    names = tuple(sorted(quantile_block))
    quantiles = tuple(float(params.value(f"{P_QUANTILES}.{n}")) for n in names)

    sampled_block = params.get(P_SAMPLED)
    paths: list[str] = []
    for group in sorted(sampled_block):
        node = sampled_block[group]
        value = node.get("value") if isinstance(node, Mapping) else None
        if isinstance(value, list):
            paths.extend(str(p) for p in value)

    return MonteCarloConfig(
        n_draws=int(params.value(P_DEFAULT_N)),
        base_seed=int(params.value(P_BASE_SEED)),
        workers=int(params.value(P_WORKERS)),
        chunk_size=int(params.value(P_CHUNK_SIZE)),
        max_network_states=int(params.value(P_MAX_NETWORK_STATES)),
        beta_grid_points=int(params.value(P_BETA_GRID_POINTS)),
        quantiles=quantiles,
        quantile_names=names,
        nominal_coverage=float(params.value(P_NOMINAL_COVERAGE)),
        coverage_tolerance=float(params.value(P_COVERAGE_TOLERANCE)),
        rho_announcer=float(params.value(P_RHO_ANNOUNCER)),
        rho_public=float(params.value(P_RHO_PUBLIC)),
        rho_phi_eta=float(params.value(P_RHO_PHI_ETA)),
        min_eigenvalue=float(params.value(P_MIN_EIGENVALUE)),
        acceptance_tolerance=float(params.value(P_ACCEPTANCE_TOL)),
        beta_concentration=float(params.value(P_BETA_CONCENTRATION)),
        mean_floor=float(params.value(P_MEAN_FLOOR)),
        top_decile_share=float(params.value(P_TOP_DECILE_SHARE)),
        convergence_p50_rel_tol=float(params.value(P_CONVERGENCE_TOL)),
        sampled_paths=tuple(paths),
        mode_share_parent=str(params.value(f"{P_SAMPLED}.mode_share_parent")),
        mode_share_leaf=str(params.value(f"{P_SAMPLED}.mode_share_leaf")),
        macro_scenarios_prefix=str(params.value(f"{P_SAMPLED}.macro_scenarios_prefix")),
        macro_probabilities_prefix=str(
            params.value(f"{P_SAMPLED}.macro_probabilities_prefix")
        ),
    )


# --------------------------------------------------------------------------------------
# 16.1 — the Gaussian copula
# --------------------------------------------------------------------------------------


class _CopulaRng:
    """A ``Generator``-shaped object whose ``uniform`` replays a pre-drawn copula value.

    :meth:`ufe.params.Params.sample` is the single sampling entry point (task requirement),
    and it draws with ``rng.uniform(low, high)``. Feeding it this object is what makes a
    correlated draw go through the same code path as an independent one — the correlation
    lives in *which* uniform is supplied, never in a second copy of the marginal.
    """

    def __init__(self, u: float) -> None:
        self._u = float(u)

    def uniform(self, low: float = ZERO, high: float = ONE, size: Any = None) -> float:
        del size
        return float(low) + self._u * (float(high) - float(low))


def correlation_matrix(
    projects: pd.DataFrame, config: MonteCarloConfig
) -> tuple[list[str], np.ndarray]:
    """Section 16.1's latent correlation matrix over `(projects..., phi_t, eta)`.

    Returns the ordered dimension names and a symmetric positive-definite matrix. Project
    ids are sorted, never taken from set iteration order (Section 15.2).
    """
    ids = sorted(str(p) for p in projects["project_id"]) if len(projects) else []
    dims = [*ids, DIM_PHI, DIM_ETA]
    n = len(dims)
    matrix = np.eye(n)

    if ids:
        by_id = projects.set_index(projects["project_id"].astype(str))
        announcer = (
            by_id["announcer_id"].astype(object).to_dict()
            if "announcer_id" in by_id.columns
            else {}
        )
        public = (
            by_id["is_public"].astype(bool).to_dict()
            if "is_public" in by_id.columns
            else {}
        )
        for i, a in enumerate(ids):
            for j in range(i + ONE, len(ids)):
                b = ids[j]
                rho = ZERO
                ann_a, ann_b = announcer.get(a), announcer.get(b)
                if ann_a is not None and ann_b is not None and ann_a == ann_b:
                    rho = max(rho, config.rho_announcer)
                if public.get(a, False) and public.get(b, False):
                    # A single run is a single city, hence a single state (Section 16.1's
                    # "public projects in the same state").
                    rho = max(rho, config.rho_public)
                matrix[i, j] = matrix[j, i] = rho

    matrix[n - 2, n - ONE] = matrix[n - ONE, n - 2] = config.rho_phi_eta
    return dims, _nearest_positive_definite(matrix, config.min_eigenvalue)


def _nearest_positive_definite(matrix: np.ndarray, floor: float) -> np.ndarray:
    """Clip eigenvalues at `floor` and rescale back to a correlation matrix.

    The three pairwise rules of Section 16.1 are stated independently and need not compose
    into a positive-definite matrix (a block of mutually-0.5 projects plus a mutually-0.3
    public block generally does not). Rather than silently reduce a correlation, the matrix
    is projected onto the nearest PD matrix and the projection is logged.
    """
    symmetric = (matrix + matrix.T) / (ONE + ONE)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() >= floor:
        return symmetric
    logger.info(
        "copula correlation matrix has minimum eigenvalue %g below %g; projecting onto "
        "the nearest positive-definite correlation matrix (spec Section 16.1)",
        float(eigenvalues.min()),
        floor,
    )
    clipped = eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
    scale = np.sqrt(np.diag(clipped))
    return clipped / np.outer(scale, scale)


#: One clip-and-rescale pass guarantees a positive-definite result but not that the
#: smallest eigenvalue lands exactly on `floor` — rescaling the diagonal back to 1 moves
#: the spectrum slightly. The guarantee the sampler needs is that `np.linalg.cholesky`
#: succeeds, which positive definiteness gives.


def _copula_uniforms(
    dims: Sequence[str], matrix: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """One correlated uniform per latent dimension."""
    cholesky = np.linalg.cholesky(matrix)
    z = cholesky @ rng.standard_normal(len(dims))
    u = special.ndtr(z)
    return {name: float(value) for name, value in zip(dims, u)}


# --------------------------------------------------------------------------------------
# 16.1 — one draw
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    """One Monte Carlo draw: sampled parameter values plus the scenario it implies."""

    index: int
    seed: int
    overrides: Mapping[str, float]
    p_completion: Mapping[str, float]
    macro_scenario: str
    uniforms: Mapping[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"draw": int(self.index), "seed": int(self.seed)}
        row["macro_scenario"] = self.macro_scenario
        row.update({path: float(v) for path, v in sorted(self.overrides.items())})
        return row


class ParamsDraw:
    """A read-only view of `Params` with one draw's values substituted.

    Duck-typed rather than a subclass: ``ufe/params.py`` belongs to another agent and is
    not edited. Everything the layers use — ``value``, ``get``, ``leaf``, ``sample``,
    ``conf``, ``scope``, ``city_config``, ``manifest``, ``hash`` — is either overridden or
    delegated.
    """

    def __init__(self, base: Any, overrides: Mapping[str, float], *, draw: int) -> None:
        self._base = base
        self._overrides = {str(k): v for k, v in overrides.items()}
        self._draw = int(draw)

    # -- overridden accessors ---------------------------------------------------------

    def value(self, path: str) -> Any:
        if path in self._overrides:
            return self._overrides[path]
        return self._base.value(path)

    def leaf(self, path: str) -> dict[str, Any]:
        leaf = dict(self._base.leaf(path))
        if path in self._overrides:
            leaf["value"] = self._overrides[path]
        return leaf

    def get(self, path: str) -> Any:
        if path in self._overrides:
            return self.leaf(path)
        node = self._base.get(path)
        if isinstance(node, Mapping) and "value" in node:
            return self.leaf(path)
        return node

    def sample(self, path: str, rng: Any) -> Any:
        if path in self._overrides:
            return self._overrides[path]
        return self._base.sample(path, rng)

    # -- identity ---------------------------------------------------------------------

    @property
    def hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {"base": self._base.hash, "overrides": self._overrides},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        base = dict(self._base.manifest())
        base["params_hash"] = self.hash
        base["monte_carlo_draw"] = self._draw
        base["monte_carlo_overrides"] = dict(sorted(self._overrides.items()))
        return base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _macro_scenario(
    params: Any, config: MonteCarloConfig, rng: np.random.Generator
) -> str:
    """Section 16.1: "Categorical by `scenario_probabilities`, then triangular within"."""
    block = params.get(config.macro_probabilities_prefix)
    names = sorted(block)
    weights = np.array(
        [float(params.value(f"{config.macro_probabilities_prefix}.{n}")) for n in names]
    )
    total = weights.sum()
    if total <= ZERO:
        raise UFEError(f"{config.macro_probabilities_prefix} sums to {total}")
    return str(rng.choice(names, p=weights / total))


def _mode_share_overrides(
    params: Any, config: MonteCarloConfig, rng: np.random.Generator
) -> dict[str, float]:
    """Section 16.1: draw the mode shares, then renormalise them to sum to 1."""
    parent = params.get(config.mode_share_parent)
    modes = sorted(m for m in parent if isinstance(parent[m], Mapping))
    paths = [
        f"{config.mode_share_parent}.{mode}.{config.mode_share_leaf}"
        for mode in modes
        if isinstance(parent[mode], Mapping) and config.mode_share_leaf in parent[mode]
    ]
    if not paths:
        return {}
    drawn = np.array([float(params.sample(path, rng)) for path in paths])
    total = drawn.sum()
    if total <= ZERO:
        return {}
    return {path: float(v) for path, v in zip(paths, drawn / total)}


def sample_draw(
    params: Any,
    projects: pd.DataFrame,
    config: MonteCarloConfig,
    *,
    index: int,
    base_seed: int,
    p_completion_base: Mapping[str, float] | None = None,
    dims: Sequence[str] | None = None,
    matrix: np.ndarray | None = None,
) -> Draw:
    """Sample one draw (spec Section 16.1). Deterministic in `(base_seed, index)`.

    `p_completion_base` is Layer 3's deterministic completion probability per project; the
    copula perturbs it through a Beta with that mean (see
    ``montecarlo.p_completion.beta_concentration`` and the gap it documents).
    """
    seed = int(base_seed) + int(index)
    rng = np.random.default_rng(seed)
    if dims is None or matrix is None:
        dims, matrix = correlation_matrix(projects, config)
    uniforms = _copula_uniforms(dims, matrix, rng)

    overrides: dict[str, float] = {}

    # Correlated pair: phi_t (through the chosen macro scenario) and eta.
    scenario_name = _macro_scenario(params, config, rng)
    phi_path = f"{config.macro_scenarios_prefix}.{scenario_name}"
    overrides[phi_path] = float(
        params.sample(phi_path, _CopulaRng(uniforms[DIM_PHI]))
    )
    overrides[P_ETA] = float(params.sample(P_ETA, _CopulaRng(uniforms[DIM_ETA])))

    # Everything else in the Section 16.1 table: independent marginals, one shared stream.
    for path in config.sampled_paths:
        if path in overrides:
            continue
        overrides[path] = float(params.sample(path, rng))
    overrides.update(_mode_share_overrides(params, config, rng))

    # Correlated p_completion per project.
    concentration = max(config.beta_concentration, ONE + ONE)
    floor = config.mean_floor
    p_completion: dict[str, float] = {}
    for project_id in sorted(dict(p_completion_base or {})):
        mean = float(np.clip(p_completion_base[project_id], floor, ONE - floor))
        alpha = mean * concentration
        beta = (ONE - mean) * concentration
        u = uniforms.get(project_id)
        if u is None:
            continue
        p_completion[project_id] = float(stats.beta.ppf(u, alpha, beta))

    return Draw(
        index=int(index),
        seed=seed,
        overrides=overrides,
        p_completion=p_completion,
        macro_scenario=scenario_name,
        uniforms=uniforms,
    )


# --------------------------------------------------------------------------------------
# 16.3 — ensemble outputs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleResult:
    """Section 16.3's outputs, plus the draw ledger that produced them."""

    quantiles: pd.DataFrame
    outperform: pd.DataFrame
    top_decile: pd.DataFrame
    draws: pd.DataFrame
    lambdas: pd.DataFrame | None
    n_draws: int
    base_seed: int
    years: tuple[int, ...]
    cells: tuple[str, ...]
    manifest: Mapping[str, Any]
    #: `(n_draws, n_years, n_cells)` of cumulative ln price, kept for convergence checks.
    ln_price: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0, 0)))
    built_sqm: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0, 0)))

    def band(self, variable: str, low: str, high: str) -> pd.DataFrame:
        """The `[low, high]` band for `variable` as a wide frame, for coverage tests."""
        frame = self.quantiles
        rows = frame.loc[frame["variable"] == variable]
        return rows.pivot_table(index=["h3", "year"], columns="quantile", values="value")[
            [low, high]
        ]


def _quantile_frame(
    values: np.ndarray,
    variable: str,
    cells: Sequence[str],
    years: Sequence[int],
    config: MonteCarloConfig,
) -> pd.DataFrame:
    """Long frame of `(h3, year, variable, quantile, value)`."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        quantiles = np.nanquantile(values, config.quantiles, axis=ZERO)
    rows = []
    n_years, n_cells = values.shape[ONE], values.shape[2]
    for q_index, name in enumerate(config.quantile_names):
        rows.append(
            pd.DataFrame(
                {
                    "h3": np.tile(np.asarray(cells, dtype=object), n_years),
                    "year": np.repeat(np.asarray(years, dtype=np.int64), n_cells),
                    "variable": variable,
                    "quantile": name,
                    "value": quantiles[q_index].reshape(-ONE),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _outperform_and_decile(
    ln_price: np.ndarray,
    cells: Sequence[str],
    years: Sequence[int],
    config: MonteCarloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 16.3's two probability statistics.

    Section 16.3 names them without defining the comparison, so both readings are pinned
    here and reported:

    * **P(outperform city median) over the full horizon** — "outperform" is read as
      *growth*: each draw's total ``ln P`` movement from the first to the last simulated
      year, compared with that draw's own city-wide median movement.
    * **P(top decile) at each horizon year** — read as *level*: the cell sits in the top
      ``simulation.output.top_decile_share`` of the city's ``ln P`` distribution in that
      year, in that draw. Level rather than growth, because the first simulated year's
      growth is identically zero for every cell and the statistic would be vacuous.
    """
    n_draws = ln_price.shape[ZERO]
    if n_draws == ZERO:
        empty = pd.DataFrame()
        return empty, empty

    with warnings.catch_warnings():
        # An all-NaN cell (no observed base-year price) legitimately yields NaN.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        total = ln_price[:, -ONE, :] - ln_price[:, ZERO, :]
        median = np.nanmedian(total, axis=ONE, keepdims=True)
        outperform = pd.DataFrame(
            {
                "h3": np.asarray(cells, dtype=object),
                "p_outperform_city_median": np.nanmean(total > median, axis=ZERO),
            }
        )

        share = config.top_decile_share
        rows = []
        for y_index, year in enumerate(years):
            level = ln_price[:, y_index, :]
            threshold = np.nanquantile(level, ONE - share, axis=ONE, keepdims=True)
            inside = np.where(np.isnan(level), np.nan, (level >= threshold).astype(float))
            rows.append(
                pd.DataFrame(
                    {
                        "h3": np.asarray(cells, dtype=object),
                        "year": np.full(len(cells), int(year), dtype=np.int64),
                        "p_top_decile": np.nanmean(inside, axis=ZERO),
                    }
                )
            )
    return outperform, pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# 16.2 — the ensemble
# --------------------------------------------------------------------------------------


def _draw_payload(
    data: SnapshotData,
    params: Any,
    scenario: Scenario,
    config: MonteCarloConfig,
    draw: Draw,
    run_kwargs: Mapping[str, Any],
    cache: RunCache | None,
) -> SimResult:
    """Execute one draw. Pure in `(snapshot, params, scenario, draw)`."""
    del config
    view = ParamsDraw(params, draw.overrides, draw=draw.index)
    variant = dataclasses.replace(scenario, macro_scenario=draw.macro_scenario)
    return _runner.run(
        data,
        view,
        variant,
        seed=draw.seed,
        deterministic=False,
        cache=cache,
        p_completion_override=dict(draw.p_completion),
        **dict(run_kwargs),
    )


_WORKER_STATE: dict[str, Any] = {}


def _worker_init(payload: Mapping[str, Any]) -> None:  # pragma: no cover - subprocess
    from ufe.params import load_params

    params = load_params(payload["city"])
    data = load_snapshot_data(payload["snapshot_path"], verify=False)
    _WORKER_STATE.update(
        {
            "params": params,
            "data": data,
            "scenario": payload["scenario"],
            "config": payload["config"],
            "run_kwargs": payload["run_kwargs"],
            "cache": RunCache(params),
        }
    )


def _worker_run(draw: Draw) -> tuple[int, pd.DataFrame]:  # pragma: no cover - subprocess
    state = _WORKER_STATE
    result = _draw_payload(
        state["data"],
        state["params"],
        state["scenario"],
        state["config"],
        draw,
        state["run_kwargs"],
        state["cache"],
    )
    return draw.index, result.panel[["h3", "year", "ln_price", "built_sqm"]]


def run_ensemble(
    snapshot: SnapshotRef | SnapshotData | str | Path,
    params: Any,
    scenario: Scenario,
    *,
    n_draws: int | None = None,
    base_seed: int | None = None,
    workers: int | None = None,
    config: MonteCarloConfig | None = None,
    cache: RunCache | None = None,
    decompose_factors: bool = False,
    progress_every: int | None = None,
    **run_kwargs: Any,
) -> EnsembleResult:
    """Run the Monte Carlo ensemble (spec Section 16).

    Every draw is a full engine run at that draw's parameter values. Draw ``k`` is seeded
    ``base_seed + k`` (Section 16.2), so the ensemble is reproducible from one master seed
    regardless of worker count or completion order.

    `decompose_factors` additionally runs the Section 13.4 ablation inside every draw,
    yielding the "per factor: distribution of lambda" output. It multiplies the cost by
    ``2 + len(factors)`` and is therefore off by default.
    """
    config = config if config is not None else load_config(params)
    n_draws = int(config.n_draws if n_draws is None else n_draws)
    base_seed = int(config.base_seed if base_seed is None else base_seed)
    workers = int(config.workers if workers is None else workers)
    if n_draws <= ZERO:
        raise UFEError(f"n_draws must be positive, got {n_draws}")

    data = (
        snapshot
        if isinstance(snapshot, SnapshotData)
        else load_snapshot_data(
            snapshot, verify=bool(run_kwargs.get("verify_snapshot", True))
        )
    )
    cache = cache if cache is not None else RunCache(params)

    # Layer 3's deterministic p_completion, the mean of every draw's Beta.
    p_base = _deterministic_p_completion(data, params, scenario)
    dims, matrix = correlation_matrix(data.projects, config)
    draws = [
        sample_draw(
            params,
            data.projects,
            config,
            index=k,
            base_seed=base_seed,
            p_completion_base=p_base,
            dims=dims,
            matrix=matrix,
        )
        for k in range(n_draws)
    ]

    panels: dict[int, pd.DataFrame] = {}
    lambda_rows: list[pd.DataFrame] = []
    manifest: Mapping[str, Any] = {}

    if workers > ONE:  # pragma: no cover - exercised outside the test suite
        payload = {
            "city": params.city_id,
            "snapshot_path": str(data.ref.path),
            "scenario": scenario,
            "config": config,
            "run_kwargs": dict(run_kwargs),
        }
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(payload,)
        ) as pool:
            for index, panel in pool.map(
                _worker_run, draws, chunksize=max(ONE, config.chunk_size)
            ):
                panels[index] = panel
    else:
        for draw in draws:
            result = _draw_payload(
                data, params, scenario, config, draw, run_kwargs, cache
            )
            panels[draw.index] = result.panel[["h3", "year", "ln_price", "built_sqm"]]
            if not manifest:
                manifest = result.manifest.to_dict()
            if decompose_factors:
                view = ParamsDraw(params, draw.overrides, draw=draw.index)
                decomposition = _factors.decompose_run(
                    data,
                    view,
                    scenario,
                    seed=draw.seed,
                    cache=cache,
                    deterministic=False,
                    **run_kwargs,
                )
                frame = _factors.lambda_frame(decomposition.decomposition)
                frame["draw"] = draw.index
                lambda_rows.append(frame)
            if progress_every and (draw.index + ONE) % int(progress_every) == ZERO:
                logger.info("monte carlo: %d/%d draws", draw.index + ONE, n_draws)

    first = panels[min(panels)]
    cells = tuple(
        first.loc[first["year"] == first["year"].min(), "h3"].astype(str).tolist()
    )
    years = tuple(int(y) for y in sorted(first["year"].unique()))
    shape = (n_draws, len(years), len(cells))
    ln_price = np.full(shape, np.nan, dtype=np.float32)
    built = np.full(shape, np.nan, dtype=np.float32)
    for index in sorted(panels):
        wide_price = panels[index].pivot_table(
            index="year", columns="h3", values="ln_price"
        )
        wide_built = panels[index].pivot_table(
            index="year", columns="h3", values="built_sqm"
        )
        ln_price[index] = wide_price.reindex(index=list(years), columns=list(cells)).to_numpy()
        built[index] = wide_built.reindex(index=list(years), columns=list(cells)).to_numpy()

    quantiles = pd.concat(
        [
            _quantile_frame(ln_price, "ln_price", cells, years, config),
            _quantile_frame(built, "built_sqm", cells, years, config),
        ],
        ignore_index=True,
    )
    outperform, top_decile = _outperform_and_decile(ln_price, cells, years, config)

    return EnsembleResult(
        quantiles=quantiles,
        outperform=outperform,
        top_decile=top_decile,
        draws=pd.DataFrame([d.to_row() for d in draws]),
        lambdas=pd.concat(lambda_rows, ignore_index=True) if lambda_rows else None,
        n_draws=n_draws,
        base_seed=base_seed,
        years=years,
        cells=cells,
        manifest=manifest,
        ln_price=ln_price,
        built_sqm=built,
    )


def _deterministic_p_completion(
    data: SnapshotData, params: Any, scenario: Scenario
) -> dict[str, float]:
    """Layer 3's `p_completion` per project, the mean of each draw's Beta."""
    from ufe.layers import l3_credibility as L3

    projects = data.projects
    if projects.empty:
        return {}
    if scenario.disabled_projects:
        projects = projects.loc[
            ~projects["project_id"].astype(str).isin(set(scenario.disabled_projects))
        ]
    scored = L3.completion_probability(
        projects,
        data.announcers,
        params,
        force_project_state=dict(scenario.force_project_state) or None,
        unknown_modifiers=L3.IGNORE,
    )
    return {
        str(pid): float(p)
        for pid, p in zip(scored["project_id"], scored["p_completion"])
    }


def p50_converged(
    coarse: EnsembleResult, fine: EnsembleResult, config: MonteCarloConfig
) -> tuple[bool, float]:
    """Section 16 ACCEPTANCE: "doubling draw count changes p50 by less than 1%".

    Returns `(converged, worst_relative_change)` on the median ln-price surface.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        a = np.nanmedian(coarse.ln_price, axis=ZERO)
        b = np.nanmedian(fine.ln_price, axis=ZERO)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(b - a) / np.abs(np.where(a == ZERO, np.nan, a))
    worst = float(np.nanmax(relative)) if np.isfinite(relative).any() else float(ZERO)
    return worst < config.convergence_p50_rel_tol, worst


