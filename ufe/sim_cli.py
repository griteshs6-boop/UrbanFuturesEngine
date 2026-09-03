"""`ufe sim` — the simulation sub-app (Modules 11 and 12).

Mounted by ``ufe/cli.py`` as ``sim``, so this module's :func:`run` is reachable as
``ufe sim run``. ``ufe/cli.py`` ALSO registers the very same function object at the top
level, so the Section 23 item 2 invocation ``ufe run --city vizag --horizon 2035`` works
verbatim. Both spellings are the same callable and therefore take the same options.

Nothing in this module does work at import time. `print` is confined to the CLI layer
(CONTRACT.md), and even here it goes through rich's console.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sim",
    help="Run simulations, Monte Carlo ensembles and factor decompositions.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)

ZERO = 0
ONE = 1


def _load(city: str) -> Any:
    from ufe.params import load_params

    return load_params(city)


def _scenario(params: Any, city: str, horizon: list[int] | None, macro: str) -> Any:
    from ufe.sim.runner import P_DEFAULT_LAST_YEAR, Scenario

    years = tuple(horizon) if horizon else (int(params.value(P_DEFAULT_LAST_YEAR)),)
    return Scenario(city_id=city, horizon=years, macro_scenario=macro)


def _delay_family(pairs: list[str] | None) -> dict[str, str] | None:
    """``--delay-family metro_rail=metro_phase1`` (repeatable).

    Section 10.3 keys its delay families by a vocabulary that does not coincide with the
    archetype names in ``archetypes.yaml``, and no mapping ships in either file. Rather than
    invent one, the CLI takes it explicitly; a ``delay_family_map:`` block in the city config
    is picked up automatically when present.
    """
    if not pairs:
        return None
    mapping: dict[str, str] = {}
    for pair in pairs:
        archetype, _, family = pair.partition("=")
        if not family:
            raise typer.BadParameter(
                f"--delay-family expects archetype=family, got {pair!r}"
            )
        mapping[archetype.strip()] = family.strip()
    return mapping


def _pph(pairs: list[str] | None) -> dict[str, float] | None:
    """``--persons-per-household low=4.6`` (repeatable).

    ``behaviour.persons_per_household_by_band`` is deliberately null in the shipped YAML
    (Section 12.5), so Layer 5 requires it from the caller.
    """
    if not pairs:
        return None
    out: dict[str, float] = {}
    for pair in pairs:
        band, _, value = pair.partition("=")
        if not value:
            raise typer.BadParameter(
                f"--persons-per-household expects band=persons, got {pair!r}"
            )
        out[band.strip()] = float(value)
    return out


def _matrices(cache_dir: Path | None) -> Any:
    if cache_dir is None:
        return None
    from ufe.layers.routing import load_matrices

    return load_matrices(cache_dir)


def _print_provenance(manifest: Any, params: Any) -> None:
    from ufe.sim.runner import P_SHORT_HASH_LENGTH

    short = manifest.provenance.short(int(params.value(P_SHORT_HASH_LENGTH)))
    table = Table(title="run provenance (spec Section 23 item 5)")
    table.add_column("field")
    table.add_column("value")
    table.add_row("snapshot", f"{manifest.provenance.snapshot_id} ({short['snapshot_hash']})")
    table.add_row("params hash", short["params_hash"])
    table.add_row("code version", short["code_version"])
    table.add_row("working tree", "dirty" if manifest.provenance.code_dirty else "clean")
    table.add_row("seed", str(manifest.seed))
    table.add_row("years", f"{manifest.simulated_years[ZERO]}..{manifest.simulated_years[-ONE]}")
    table.add_row("complete", "yes" if manifest.complete else "NO — override in effect")
    console.print(table)


@app.command()
def run(
    city: str = typer.Option(..., help="City id, e.g. vizag."),
    snapshot: Path = typer.Option(..., help="Snapshot directory to read (Section 3.8)."),
    horizon: list[int] = typer.Option(
        None, help="Horizon year(s). Defaults to simulation.horizon.default_last_year."
    ),
    seed: int = typer.Option(ZERO, help="Master seed (Section 15.1)."),
    macro: str = typer.Option("base", help="Macro scenario name from price.macro.scenarios."),
    ttm_cache: Path = typer.Option(
        None, help="Travel-time matrix cache directory. Omitted: accessibility is not recomputed."
    ),
    delay_family: list[str] = typer.Option(
        None, help="archetype=family mapping for Section 10.3 delay families (repeatable)."
    ),
    persons_per_household: list[str] = typer.Option(
        None, help="band=persons; behaviour.persons_per_household_by_band is null on disk."
    ),
    out: Path = typer.Option(None, help="Directory to write the panel and manifest to."),
    allow_dirty: bool = typer.Option(
        False, help="Proceed despite an unknown or dirty git state (Section 23 item 5)."
    ),
    digest: bool = typer.Option(True, help="Print the SHA-256 of the serialised result."),
) -> None:
    """One deterministic run (spec Section 15)."""
    from ufe.sim import runner as R

    params = _load(city)
    scenario = _scenario(params, city, horizon, macro)
    result = R.run(
        snapshot,
        params,
        scenario,
        seed=seed,
        matrices=_matrices(ttm_cache),
        delay_family_map=_delay_family(delay_family),
        persons_per_household_by_band=_pph(persons_per_household),
        allow_dirty=allow_dirty,
    )
    _print_provenance(result.manifest, params)
    console.print(result.diagnostics.to_string(index=False))
    if digest:
        console.print(f"[bold]result digest[/bold] {result.digest()}")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        result.panel.to_parquet(out / "panel.parquet", index=False)
        result.diagnostics.to_parquet(out / "diagnostics.parquet", index=False)
        (out / "MANIFEST.json").write_text(result.manifest.to_json())
        console.print(f"written to {out}")


@app.command()
def montecarlo(
    city: str = typer.Option(..., help="City id."),
    snapshot: Path = typer.Option(..., help="Snapshot directory."),
    horizon: list[int] = typer.Option(None, help="Horizon year(s)."),
    draws: int = typer.Option(None, help="Draw count. Defaults to montecarlo.draws.default_n."),
    base_seed: int = typer.Option(None, help="Master seed; draw k uses base_seed + k."),
    workers: int = typer.Option(None, help="Processes. Defaults to montecarlo.parallel.workers."),
    macro: str = typer.Option("base", help="Macro scenario for the deterministic reference."),
    ttm_cache: Path = typer.Option(None, help="Travel-time matrix cache directory."),
    delay_family: list[str] = typer.Option(None, help="archetype=family (repeatable)."),
    persons_per_household: list[str] = typer.Option(None, help="band=persons (repeatable)."),
    out: Path = typer.Option(None, help="Directory to write the Section 16.3 outputs to."),
    allow_dirty: bool = typer.Option(False, help="Proceed despite a dirty git state."),
) -> None:
    """A Monte Carlo ensemble (spec Section 16)."""
    from ufe.sim import montecarlo as MC

    params = _load(city)
    scenario = _scenario(params, city, horizon, macro)
    ensemble = MC.run_ensemble(
        snapshot,
        params,
        scenario,
        n_draws=draws,
        base_seed=base_seed,
        workers=workers,
        matrices=_matrices(ttm_cache),
        delay_family_map=_delay_family(delay_family),
        persons_per_household_by_band=_pph(persons_per_household),
        allow_dirty=allow_dirty,
    )
    console.print(
        f"{ensemble.n_draws} draws, base seed {ensemble.base_seed}, "
        f"years {ensemble.years[ZERO]}..{ensemble.years[-ONE]}, {len(ensemble.cells)} cells"
    )
    console.print(ensemble.quantiles.head().to_string(index=False))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        ensemble.quantiles.to_parquet(out / "quantiles.parquet", index=False)
        ensemble.outperform.to_parquet(out / "outperform.parquet", index=False)
        ensemble.top_decile.to_parquet(out / "top_decile.parquet", index=False)
        ensemble.draws.to_parquet(out / "draws.parquet", index=False)
        console.print(f"written to {out}")


@app.command()
def factors(
    city: str = typer.Option(..., help="City id."),
    snapshot: Path = typer.Option(..., help="Snapshot directory."),
    horizon: list[int] = typer.Option(None, help="Horizon year(s)."),
    seed: int = typer.Option(ZERO, help="Master seed."),
    macro: str = typer.Option("base", help="Macro scenario."),
    ttm_cache: Path = typer.Option(None, help="Travel-time matrix cache directory."),
    delay_family: list[str] = typer.Option(None, help="archetype=family (repeatable)."),
    persons_per_household: list[str] = typer.Option(None, help="band=persons (repeatable)."),
    allow_dirty: bool = typer.Option(False, help="Proceed despite a dirty git state."),
) -> None:
    """The Section 13.4 leave-one-out factor decomposition, at the run level."""
    from ufe.sim import factors as F

    params = _load(city)
    scenario = _scenario(params, city, horizon, macro)
    decomposition = F.decompose_run(
        snapshot,
        params,
        scenario,
        seed=seed,
        matrices=_matrices(ttm_cache),
        delay_family_map=_delay_family(delay_family),
        persons_per_household_by_band=_pph(persons_per_household),
        allow_dirty=allow_dirty,
    )
    table = Table(title=f"factor decomposition at {decomposition.year}")
    table.add_column("factor")
    table.add_column("mean lambda")
    frame = decomposition.decomposition.normalised
    for name in decomposition.factors:
        table.add_row(name, f"{float(frame[name].mean()):+.5f}")
    table.add_row("interaction", f"{float(decomposition.decomposition.interaction.mean()):+.5f}")
    console.print(table)
    console.print(f"runs executed: {decomposition.n_runs}")
    if decomposition.warning:
        console.print(f"[yellow]{decomposition.warning}[/yellow]")


@app.command()
def manifest(
    snapshot: Path = typer.Option(..., help="Snapshot directory."),
    city: str = typer.Option(..., help="City id, to resolve the params hash."),
    allow_dirty: bool = typer.Option(False, help="Do not refuse on a dirty git state."),
) -> None:
    """Print the provenance triple for a snapshot without running anything."""
    from ufe.sim.snapshot import open_snapshot, resolve_provenance

    params = _load(city)
    ref = open_snapshot(snapshot)
    provenance = resolve_provenance(ref, params, allow_dirty=allow_dirty)
    console.print(json.dumps(provenance.to_dict(), indent=ONE + ONE, sort_keys=True))
