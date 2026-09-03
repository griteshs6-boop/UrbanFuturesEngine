"""Run-level factor decomposition (spec Section 13.4, driven from Module 11).

Section 13.4's arithmetic — the leave-one-out ablation, the normalisation, the interaction
term and the separability warning — already lives in
:func:`ufe.layers.l6_price.decompose`, which is tested against the Section 13 acceptance
block. Nothing here re-derives it. What this module adds is the *run* half: turning a named
factor group into a set of disabled projects, executing the ``2 + len(factors)`` runs the
ablation needs, and handing Layer 6 a pure ``tuple of active factor names -> ln P``
callable.  A sorted tuple, never a set: spec Section 15.2 requires a deterministic
iteration order over set-like things.

::

    Run 0     baseline: every project disabled, macro only        -> lnP_base
    Run FULL  every factor group enabled                          -> lnP_full
    Run LOO_f every group except f                                -> lnP_loo_f

Factor groups come from the city config's ``factors:`` block
(``config/cities/vizag.yaml``) unless the :class:`~ufe.sim.runner.Scenario` overrides them.
Runs scale linearly with the factor count, which is why Section 13.4 caps it at
``price.decomposition.max_factors``; the cap is enforced inside Layer 6.

Purity matters here and is load-bearing: the Section 13 acceptance test "removing a factor
and re-running FULL reproduces that factor's LOO run exactly" is a purity test of the whole
engine, not of the decomposition arithmetic. Every sub-run therefore uses the *same* seed,
the same snapshot and the same shared :class:`~ufe.sim.runner.RunCache`.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import UFEError
from ufe.layers import l6_price as L6
from ufe.sim import runner as _runner
from ufe.sim.runner import RunCache, Scenario, SimResult
from ufe.sim.snapshot import SnapshotData, SnapshotRef, load_snapshot_data

__all__ = [
    "RunDecomposition",
    "factor_groups_from_params",
    "decompose_run",
    "lambda_frame",
]

logger = logging.getLogger(__name__)

ZERO = _runner.ZERO
ONE = _runner.ONE

#: The city-config block Section 13.4's named factor groups live in.
CITY_FACTORS_KEY = "factors"


@dataclass(frozen=True)
class RunDecomposition:
    """A :class:`~ufe.layers.l6_price.FactorDecomposition` plus its run provenance."""

    decomposition: L6.FactorDecomposition
    year: int
    factor_groups: Mapping[str, tuple[str, ...]]
    #: The FULL run, kept so the caller has the panel and the manifest without re-running.
    full: SimResult
    #: Number of engine runs executed (``2 + len(factors)``).
    n_runs: int

    @property
    def factors(self) -> tuple[str, ...]:
        return self.decomposition.factors

    @property
    def separable(self) -> bool:
        return self.decomposition.separable

    @property
    def warning(self) -> str | None:
        return self.decomposition.warning

    def to_frame(self) -> pd.DataFrame:
        """One row per cell: every normalised lambda, the interaction and the total."""
        frame = self.decomposition.to_frame().reset_index()
        frame = frame.rename(columns={frame.columns[ZERO]: "h3"})
        frame.insert(ONE, "year", np.full(len(frame), int(self.year), dtype=np.int64))
        return frame

    def reconciliation_error(self) -> pd.Series:
        """Section 13 ACCEPTANCE identity, zero by construction. Exposed for the test."""
        return self.decomposition.reconciliation_error()


def factor_groups_from_params(
    params: Any, scenario: Scenario | None = None
) -> dict[str, tuple[str, ...]]:
    """Named factor groups: the scenario's, else the city config's ``factors:`` block."""
    if scenario is not None and scenario.factor_groups:
        return {k: tuple(v) for k, v in sorted(scenario.factor_groups.items())}
    block = params.city_config.get(CITY_FACTORS_KEY) or {}
    return {str(k): tuple(str(p) for p in v) for k, v in sorted(block.items())}


def lambda_frame(decomposition: L6.FactorDecomposition) -> pd.DataFrame:
    """Long frame of `(h3, factor, lambda)` — the shape Section 16.3's lambda
    distribution is aggregated over."""
    # `melt` rather than `stack`: a cell with no observed base-year price has a NaN lambda,
    # and the long frame must still carry one row per (cell, factor) so downstream joins
    # line up. `stack` would silently drop those rows.
    wide = decomposition.normalised.copy()
    wide.index = wide.index.rename("h3")
    return wide.reset_index().melt(
        id_vars="h3", var_name="factor", value_name="lambda"
    )


def decompose_run(
    snapshot: SnapshotRef | SnapshotData | str | Path,
    params: Any,
    scenario: Scenario,
    *,
    seed: int = ZERO,
    year: int | None = None,
    factor_groups: Mapping[str, Sequence[str]] | None = None,
    cache: RunCache | None = None,
    **run_kwargs: Any,
) -> RunDecomposition:
    """Section 13.4's ablation at the run level.

    Parameters
    ----------
    year:
        The horizon year the decomposition is taken at. Defaults to the last simulated year.
    factor_groups:
        Overrides both the scenario's and the city config's groups.
    run_kwargs:
        Passed unchanged to every :func:`ufe.sim.runner.run` call (matrices, household
        sizes, ``allow_dirty``, ...). Identical across sub-runs, by construction.
    """
    data = (
        snapshot
        if isinstance(snapshot, SnapshotData)
        else load_snapshot_data(
            snapshot, verify=bool(run_kwargs.get("verify_snapshot", True))
        )
    )
    groups = (
        {str(k): tuple(str(p) for p in v) for k, v in sorted(dict(factor_groups).items())}
        if factor_groups is not None
        else factor_groups_from_params(params, scenario)
    )
    if not groups:
        raise UFEError(
            "no factor groups: give the scenario a `factor_groups`, or add a `factors:` "
            "block to the city config (spec Section 13.4)"
        )

    all_projects = set(data.projects["project_id"].astype(str))
    grouped = {pid for members in groups.values() for pid in members}
    unknown = sorted(grouped - all_projects)
    if unknown:
        logger.warning(
            "factor groups name %d project id(s) that are not in the snapshot pipeline: "
            "%s. They contribute nothing; the decomposition is still well defined.",
            len(unknown),
            ", ".join(unknown),
        )
    # Projects in no group are held FIXED across every sub-run: the ablation asks what each
    # named factor contributed, not what the ungrouped remainder contributed.
    ungrouped = sorted(all_projects - grouped)

    cache = cache if cache is not None else RunCache(params)
    executed: dict[tuple[str, ...], SimResult] = {}

    def _run(active: tuple[str, ...]) -> pd.Series:
        disabled = sorted(
            pid
            for name, members in groups.items()
            if name not in active
            for pid in members
        )
        variant = dataclasses.replace(
            scenario,
            disabled_projects=tuple(disabled),
            factor_groups={k: tuple(v) for k, v in groups.items()},
        )
        result = _runner.run(data, params, variant, seed=seed, cache=cache, **run_kwargs)
        executed[active] = result
        target = int(year) if year is not None else int(result.manifest.simulated_years[-ONE])
        rows = result.panel.loc[result.panel["year"] == target]
        if rows.empty:
            raise UFEError(
                f"year {target} is not in the simulated horizon "
                f"{result.manifest.simulated_years}"
            )
        return (
            rows.set_index("h3")["ln_price"].sort_index().rename(f"lnP_{target}")
        )

    factors = tuple(groups)
    decomposition = L6.decompose(_run, factors, params)
    full = executed[tuple(sorted(factors))]
    resolved_year = int(year) if year is not None else int(full.manifest.simulated_years[-ONE])
    logger.info(
        "factor decomposition over %d group(s) at %d: %d runs, %d ungrouped projects held "
        "fixed, separable=%s",
        len(factors),
        resolved_year,
        len(executed),
        len(ungrouped),
        decomposition.separable,
    )
    return RunDecomposition(
        decomposition=decomposition,
        year=resolved_year,
        factor_groups=groups,
        full=full,
        n_runs=len(executed),
    )
