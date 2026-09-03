"""Typer sub-app for Module 2 — data ingestion.

Commands map onto Section 20.2's onboarding steps 6-9:

    ufe ingest national --city vizag     # step 6: DEM, land cover, buildings, OSM, VIIRS
    ufe ingest state    --city vizag     # step 7: guidance values, RERA, cadastral
    ufe ingest city     --city vizag     # step 8: master plan, CZMP, utilities, prices
    ufe ingest coverage --city vizag     # step 9: the report and the refusal gate
    ufe ingest adapters                  # which states have an adapter (step 2)

Progress is reported with ``rich.progress`` (Section 2.1b forbids ``tqdm``). Every import of
``ufe.params`` / ``ufe.store`` / the ingesters is lazy, inside a command, so this module
stays importable with no geospatial stack loaded and ``--help`` costs nothing.

Each command runs the ingesters for its tier, writes ``ingest_runs`` and ``cell_imputation``
rows, and upserts the ``cells`` columns it owns. Nothing is written without a matching
``ingest_runs`` row — Section 6: "A cell attribute with no corresponding ingest run is
invalid."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Module 2 data ingestion: national, state and city tiers.")

_CITY_OPTION = typer.Option(..., help="City id, e.g. vizag")
_DB_OPTION = typer.Option("data/ufe.duckdb", help="Store path")
_RAW_OPTION = typer.Option(None, help="Root of the raw source tree (default: config/ingest.yaml)")
_FORCE_OPTION = typer.Option(False, help="Re-fetch instead of using the cache")
_MODE_OPTION = typer.Option("development", help="development | production (Section 6.7 gate)")


def _setup(city: str, db: str, raw_root: str | None, mode: str) -> tuple[Any, Any, Any, Any]:
    """Load params, open the store, build a reader and resolve the state adapter."""
    from ufe.ingest.adapters.base import get_adapter
    from ufe.ingest.core import CityConfig, LocalFileReader, cfg
    from ufe.params import load_params
    from ufe.store import db as store

    params = load_params(city)
    city_config = CityConfig.from_params(params, mode=mode)
    reader = LocalFileReader(Path(raw_root or cfg("reader.raw_root")) / city_config.city_id)
    connection = store.connect(db)
    store.migrate(connection)
    adapter = get_adapter(city_config.state_code, reader=reader)
    return params, city_config, reader, (connection, adapter)


def _progress(description: str, total: int) -> Any:
    """A ``rich.progress`` bar. ``tqdm`` is forbidden by Section 2.1b."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    bar.start()
    bar.add_task(description, total=total)
    return bar


def _run_tier(
    tier: str, city: str, db: str, raw_root: str | None, mode: str, force: bool
) -> None:
    from ufe.ingest import runner

    params, city_config, reader, (connection, adapter) = _setup(city, db, raw_root, mode)
    ingesters = runner.ingesters_for_tier(
        tier, reader=reader, city=city_config, params=params, adapter=adapter
    )
    bar = _progress(f"ingest {tier}", len(ingesters))
    try:
        result = runner.run_ingesters(
            ingesters,
            connection=connection,
            city=city_config,
            params=params,
            adapter=adapter,
            force=force,
            on_step=lambda name: bar.advance(bar.task_ids[0]),
        )
    finally:
        bar.stop()
    typer.echo(
        f"{tier}: {len(result.runs)} ingest run(s), "
        f"{len(result.columns)} cells column(s): {', '.join(sorted(result.columns))}"
    )
    if result.failures:
        typer.echo("failed: " + "; ".join(result.failures))
        raise typer.Exit(code=1)


@app.command()
def national(
    city: str = _CITY_OPTION,
    db: str = _DB_OPTION,
    raw_root: str = _RAW_OPTION,
    mode: str = _MODE_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Section 20.2 step 6: DEM, land cover, buildings, population, OSM, nightlights."""
    _run_tier("national", city, db, raw_root, mode, force)


@app.command()
def state(
    city: str = _CITY_OPTION,
    db: str = _DB_OPTION,
    raw_root: str = _RAW_OPTION,
    mode: str = _MODE_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Section 20.2 step 7: guidance values, RERA and cadastral, via the state adapter."""
    _run_tier("state", city, db, raw_root, mode, force)


@app.command(name="city")
def city_tier(
    city: str = _CITY_OPTION,
    db: str = _DB_OPTION,
    raw_root: str = _RAW_OPTION,
    mode: str = _MODE_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Section 20.2 step 8: master plan zoning, CZMP/CRZ, utilities and prices.

    Raises ``MissingCriticalLayer`` for a coastal city with no CZMP layer, by design.
    """
    _run_tier("city", city, db, raw_root, mode, force)


@app.command()
def coverage(
    city: str = _CITY_OPTION,
    db: str = _DB_OPTION,
    strict: bool = typer.Option(True, help="Apply the Section 20.2 step 9 refusal gate"),
) -> None:
    """Section 20.2 step 9: real vs imputed per column, and the price-coverage gate."""
    from ufe.errors import CoverageError
    from ufe.ingest import coverage as cov
    from ufe.ingest.core import CELL_IMPUTATION_TABLE
    from ufe.params import load_params
    from ufe.store import db as store

    params = load_params(city)
    connection = store.connect(db, read_only=True)
    cells = store.read_table(connection, "cells")
    try:
        imputation = connection.execute(f'SELECT * FROM "{CELL_IMPUTATION_TABLE}"').df()
    except Exception:
        imputation = None
        typer.echo("warning: no cell_imputation table; every value will look real")
    report = cov.coverage_report(cells, imputation)
    typer.echo(cov.format_report(report))
    _ = params
    if strict:
        try:
            cov.assert_coverage(report)
        except CoverageError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc


@app.command()
def adapters() -> None:
    """Section 20.2 step 2: which states have an adapter, and what each publishes."""
    from ufe.ingest.adapters.base import CAPABILITIES, registry

    table = registry()
    if not table:
        typer.echo("no state adapters registered")
        return
    for code in sorted(table):
        adapter = table[code]()
        caps = adapter.capabilities()
        missing = sorted(set(CAPABILITIES) - set(caps))
        typer.echo(
            f"{code}: {getattr(adapter, 'state_name', '')} "
            f"provides={sorted(caps)} unavailable={missing} "
            f"bulk_access={adapter.access_terms().get('bulk_access_allowed')}"
        )


if __name__ == "__main__":
    app()
