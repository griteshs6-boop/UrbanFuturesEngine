"""Typer sub-app for Module 14 — the satellite monitor.

Exposes `collect` (standalone ingest, safe to schedule from day one) and `detect` (the
change-detection entry point, which needs a real baseline and should be run separately —
see `ufe/satellite/monitor.py` module docstring) as two distinct commands, plus
`priority-tier` for Section 18.3 selection. Imports of `ufe.params` / `ufe.store` are lazy
(inside each command) so this module stays importable even before those modules exist.
"""

from __future__ import annotations

from datetime import date, datetime

import typer

app = typer.Typer(help="Module 14 satellite monitor: imagery collection and physical-state detection.")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.command()
def collect(
    city: str = typer.Option(..., help="City id, e.g. vizag"),
    window_start: str = typer.Option(..., help="YYYY-MM-DD"),
    window_end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Standalone scheduled ingest: fetch scenes and archive monthly composites for every
    project with an AOI. Does not classify or touch `projects.physical_state`."""
    from ufe.params import load_params
    from ufe.satellite.monitor import ProjectAOI, run_collection
    from ufe.satellite.stac import StacImageryBackend
    from ufe.store import db

    params = load_params(city=city)
    con = db.connect()
    projects_df = db.read_table(con, "projects")

    projects = [
        ProjectAOI(
            project_id=row["project_id"],
            aoi_bounds_4326=row["aoi_bounds_4326"],
            announced_date=row["announced_date"],
        )
        for _, row in projects_df.iterrows()
    ]
    backend = StacImageryBackend()
    result = run_collection(projects, backend, params, _parse_date(window_start), _parse_date(window_end))
    db.write_table(con, "project_physical_composites", result)
    typer.echo(f"collected {len(result)} monthly composite rows for {len(projects)} project(s)")


@app.command()
def detect(city: str = typer.Option(..., help="City id, e.g. vizag")) -> None:
    """Change-detection entry point: read archived composite history, classify physical
    state relative to each project's pre-announcement baseline, write `physical_state` /
    `physical_asof` updates and the full `project_physical_history` time series."""
    from ufe.params import load_params
    from ufe.satellite.monitor import ProjectAOI, persist_detection_results, run_detection
    from ufe.store import db

    params = load_params(city=city)
    con = db.connect()
    projects_df = db.read_table(con, "projects")
    composites_df = db.read_table(con, "project_physical_composites")

    projects = [
        ProjectAOI(
            project_id=row["project_id"],
            aoi_bounds_4326=row["aoi_bounds_4326"],
            announced_date=row["announced_date"],
        )
        for _, row in projects_df.iterrows()
    ]
    history_by_project = {
        pid: frame.reset_index(drop=True) for pid, frame in composites_df.groupby("project_id")
    }
    projects_update, physical_history = run_detection(projects, history_by_project, params)
    persist_detection_results(con, projects_update, physical_history)
    typer.echo(f"detected physical state for {len(projects_update)} project(s)")


@app.command(name="priority-tier")
def priority_tier(city: str = typer.Option(..., help="City id, e.g. vizag")) -> None:
    """Select the top-impact projects (Section 18.3) for 3m daily commercial imagery."""
    from ufe.params import load_params
    from ufe.satellite.monitor import select_priority_tier
    from ufe.store import db

    params = load_params(city=city)
    con = db.connect()
    projects_impact = db.read_table(con, "project_impact_scores")
    ranked = select_priority_tier(projects_impact, params)
    typer.echo(ranked.to_string(index=False))


if __name__ == "__main__":
    app()
