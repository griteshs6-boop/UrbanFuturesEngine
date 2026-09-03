"""Module 13 — the review queue CLI (spec Section 17.10).

`ufe review list|approve|reject|edit` — a minimal CLI, deliberately not a web UI
("Do not build a web UI for this yet"). This module exposes a module-level `app`
(a `typer.Typer()` sub-app) so the top-level `ufe` CLI (owned by another agent, per
CONTRACT file ownership) can mount it with `app.add_typer(ai_cli.app, name="review")`.

Every action here writes `verified_by` and a timestamp onto the candidate (Section 17.10).
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from ufe.ai.queue import CandidateStatus, ReviewQueue

app = typer.Typer(help="Module 13 review queue: approve, reject, or edit AI-extracted candidates.")

console = Console()

# Process-lifetime in-memory queue. A real deployment would back this with the
# `project_candidates` store table (Section 17.1); ufe.store is not this module's file to
# create, so the CLI is wired against an injectable `ReviewQueue` for testability and can
# be pointed at a store-backed queue by whoever owns that wiring.
_queue = ReviewQueue()


def get_queue() -> ReviewQueue:
    """Indirection point so tests (and a future store-backed wiring) can swap the queue."""
    return _queue


def set_queue(queue: ReviewQueue) -> None:
    global _queue
    _queue = queue


@app.command("list")
def list_pending() -> None:
    """Show every pending (or parse_failed) candidate alongside its source."""
    queue = get_queue()
    table = Table(title="Pending review candidates")
    table.add_column("candidate_id")
    table.add_column("status")
    table.add_column("record_type")
    table.add_column("confidence")
    table.add_column("source_url")
    table.add_column("extracted_by")
    for candidate in queue.list_pending():
        table.add_row(
            candidate.candidate_id,
            candidate.status.value,
            candidate.record_type.value,
            f"{candidate.confidence:.2f}" if candidate.confidence is not None else "-",
            candidate.source_url or "-",
            candidate.extracted_by,
        )
    console.print(table)


@app.command("show")
def show(candidate_id: str) -> None:
    """Show a single candidate's full payload and source document alongside it."""
    queue = get_queue()
    candidate = queue.get(candidate_id)
    console.print(f"[bold]{candidate.candidate_id}[/bold]  status={candidate.status.value}")
    console.print(f"source_url: {candidate.source_url}")
    console.print(f"extracted_by: {candidate.extracted_by}")
    console.print(json.dumps(candidate.payload, indent=2, default=str))
    if candidate.parse_error:
        console.print(f"[red]parse_error: {candidate.parse_error}[/red]")


@app.command("approve")
def approve(candidate_id: str, verified_by: str = typer.Option(..., "--verified-by")) -> None:
    queue = get_queue()
    candidate = queue.approve(candidate_id, verified_by)
    console.print(f"Approved {candidate.candidate_id} by {verified_by} at {candidate.decided_at}")


@app.command("reject")
def reject(
    candidate_id: str,
    verified_by: str = typer.Option(..., "--verified-by"),
    reason: str = typer.Option(None, "--reason"),
) -> None:
    queue = get_queue()
    candidate = queue.reject(candidate_id, verified_by, reason)
    console.print(f"Rejected {candidate.candidate_id} by {verified_by} at {candidate.decided_at}")


@app.command("edit")
def edit(
    candidate_id: str,
    verified_by: str = typer.Option(..., "--verified-by"),
    field: list[str] = typer.Option([], "--set", help="field=value, repeatable"),
) -> None:
    """Edit one or more fields on a candidate's payload, then approve it."""
    edits: dict[str, str] = {}
    for item in field:
        key, _, value = item.partition("=")
        edits[key] = value
    queue = get_queue()
    candidate = queue.edit_and_approve(candidate_id, verified_by, edits)
    console.print(f"Edited and approved {candidate.candidate_id} by {verified_by}")


if __name__ == "__main__":
    app()
