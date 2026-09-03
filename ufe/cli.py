"""Unified `ufe` command-line interface.

Each module owns its own typer sub-app and this module mounts them. Mounting is defensive:
a sub-app that is not yet on disk is skipped and reported by `ufe doctor`, so a partial
checkout still yields a usable CLI rather than an import error at startup.

Nothing here does work at import time.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ufe",
    help="Urban Futures Engine — multi-city urban development simulation.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubApp:
    """A mountable sub-command: where it lives and what to call it."""

    module: str
    attr: str
    name: str
    help: str


# The mount table. Order is the order commands appear in `ufe --help`.
SUB_APPS: tuple[SubApp, ...] = (
    SubApp("ufe.licences_cli", "app", "licences", "Licence and data-rights auditing."),
    SubApp("ufe.params_cli", "app", "params", "Inspect and validate parameter files."),
    SubApp("ufe.store.cli", "app", "store", "Database migrations and snapshots."),
    SubApp("ufe.grid.cli", "app", "grid", "Build the hex grid for a city."),
    SubApp("ufe.ingest_cli", "app", "ingest", "Ingest source data into the store."),
    SubApp("ufe.sim_cli", "app", "sim", "Run simulations and Monte Carlo."),
    SubApp("ufe.backtest_cli", "app", "backtest", "Freeze, score and gate the backtest."),
    SubApp("ufe.ai_cli", "app", "ai", "The AI extraction pipeline (never runs at sim time)."),
    SubApp("ufe.satellite_cli", "app", "satellite", "Satellite collection and change detection."),
    SubApp("ufe.api_cli", "app", "api", "Serve the API."),
)


def _mount_all() -> tuple[list[str], list[tuple[str, str]]]:
    """Mount every available sub-app. Returns (mounted names, [(name, reason)] skipped)."""
    mounted: list[str] = []
    skipped: list[tuple[str, str]] = []
    for sub in SUB_APPS:
        try:
            module = importlib.import_module(sub.module)
        except ImportError as exc:
            skipped.append((sub.name, f"module {sub.module} not importable: {exc}"))
            continue
        sub_app = getattr(module, sub.attr, None)
        if sub_app is None:
            skipped.append((sub.name, f"{sub.module} has no attribute '{sub.attr}'"))
            continue
        app.add_typer(sub_app, name=sub.name, help=sub.help)
        mounted.append(sub.name)
    return mounted, skipped


MOUNTED, SKIPPED = _mount_all()


@dataclass(frozen=True)
class TopLevelAlias:
    """A sub-app command promoted to the top level, e.g. `ufe sim run` -> `ufe run`."""

    module: str
    attr: str
    name: str
    help: str
    #: What the alias delegates to, for `ufe doctor`.
    delegates_to: str


# Spec Section 23 item 2 documents the invocation as `ufe run --city vizag --horizon 2035`.
# The runner lives in the `sim` sub-app, so the same function object is ALSO registered at
# the top level here. It is the identical callable, so the two spellings cannot drift: they
# take the same options and do the same work.
TOP_LEVEL_ALIASES: tuple[TopLevelAlias, ...] = (
    TopLevelAlias(
        "ufe.sim_cli",
        "run",
        "run",
        "One deterministic run (spec Section 23 item 2). Identical to `ufe sim run`.",
        "sim run",
    ),
)


def _mount_aliases() -> tuple[list[str], list[tuple[str, str]]]:
    """Promote the aliased sub-app commands. Returns (aliased, [(name, reason)] skipped).

    Defensive in the same way as `_mount_all`: a sub-app that is not on disk yields a
    skipped alias reported by `ufe doctor`, never an import error at startup.
    """
    aliased: list[str] = []
    skipped: list[tuple[str, str]] = []
    for alias in TOP_LEVEL_ALIASES:
        try:
            module = importlib.import_module(alias.module)
        except ImportError as exc:
            skipped.append((alias.name, f"module {alias.module} not importable: {exc}"))
            continue
        command = getattr(module, alias.attr, None)
        if command is None or not callable(command):
            skipped.append(
                (alias.name, f"{alias.module} has no callable '{alias.attr}'")
            )
            continue
        app.command(name=alias.name, help=alias.help)(command)
        aliased.append(alias.name)
    return aliased, skipped


ALIASED, ALIAS_SKIPPED = _mount_aliases()


@app.command()
def doctor() -> None:
    """Report which sub-commands are available and which are missing, and why."""
    table = Table(title="ufe sub-commands")
    table.add_column("command")
    table.add_column("status")
    table.add_column("detail")
    for name in MOUNTED:
        table.add_row(name, "[green]mounted[/green]", "")
    for name, reason in SKIPPED:
        table.add_row(name, "[yellow]unavailable[/yellow]", reason)
    delegate = {a.name: a.delegates_to for a in TOP_LEVEL_ALIASES}
    for name in ALIASED:
        table.add_row(name, "[green]mounted[/green]", f"alias for `ufe {delegate[name]}`")
    for name, reason in ALIAS_SKIPPED:
        table.add_row(name, "[yellow]unavailable[/yellow]", reason)
    console.print(table)


@app.command()
def version() -> None:
    """Print the installed engine version."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        console.print(pkg_version("urban-futures"))
    except PackageNotFoundError:
        console.print("urban-futures (not installed as a distribution)")


if __name__ == "__main__":
    app()
