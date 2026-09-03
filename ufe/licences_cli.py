"""`ufe licences ...` sub-app (Section 2.4 / Section 22.2).

Mounted onto the main CLI by `ufe/cli.py` (owned by another agent) as:

    from ufe.licences_cli import app as licences_app
    app.add_typer(licences_app, name="licences")
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from ufe import licences

app = typer.Typer(help="Licence compliance audits (Section 2.4 software, Section 22 data).")
console = Console()


@app.callback()
def _callback() -> None:
    """Licence compliance audits (Section 2.4 software, Section 22 data).

    A no-op callback: it exists only so Typer keeps `audit` addressable as a named
    subcommand (`ufe licences audit`) instead of collapsing this single-command sub-app
    down to `ufe licences` with no subcommand name, which is Typer's default behaviour for
    a Typer() instance with exactly one registered command.
    """


def _print_findings(title: str, result: licences.AuditResult) -> None:
    if result.skipped:
        console.print(f"[yellow]SKIPPED[/yellow] — {result.skip_reason}")
        return

    table = Table(title=title)
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Licence")
    table.add_column("Class")
    table.add_column("Direct")
    table.add_column("Documented")

    for f in result.findings:
        cls_colour = {"green": "green", "amber": "yellow", "red": "red", "unknown": "red"}.get(
            f.licence_class, "white"
        )
        table.add_row(
            f.package,
            f.version,
            f.licence_raw,
            f"[{cls_colour}]{f.licence_class}[/{cls_colour}]",
            "yes" if f.is_direct_dependency else "",
            "yes" if f.documented_in_dependencies_md else "[red]no[/red]",
        )
    console.print(table)

    if result.errors:
        console.print("[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  - {err}")
    else:
        console.print("[bold green]OK — no licence policy violations.[/bold green]")


@app.command("audit")
def audit(
    data: bool = typer.Option(
        False,
        "--data",
        help="Also audit data-source licences (config/sources.yaml vs "
        "config/data_sources_licences.yaml, Section 22.2) in addition to the "
        "software dependency audit.",
    ),
) -> None:
    """Enumerate installed distributions, classify their licences, and fail on any
    Red-class package or any undocumented direct dependency (Section 2.4). With `--data`,
    also cross-checks declared data sources against the known-licence table (Section 22.2).
    """
    dep_result = licences.audit_dependencies()
    _print_findings("Software dependency licence audit", dep_result)

    overall_ok = dep_result.ok

    if data:
        data_result = licences.audit_data_sources()
        console.print()
        _print_findings("Data source licence audit", data_result)
        overall_ok = overall_ok and (data_result.ok or data_result.skipped)

    if not overall_ok:
        raise typer.Exit(code=1)
