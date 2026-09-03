"""Typer sub-app for store operations.

Mounted by the owner of ``ufe/cli.py`` with::

    from ufe.store.cli import app as store_app
    app.add_typer(store_app, name="store")

Nothing here is imported at simulation time.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ufe.store import db
from ufe.store import schemas as S

app = typer.Typer(help="DuckDB store: migrate, inspect, snapshot.")


@app.command("migrate")
def migrate_cmd(
    path: Path = typer.Option(Path(db.DEFAULT_DB_PATH), help="DuckDB file."),
) -> None:
    """Create or upgrade the store schema.  Idempotent."""
    con = db.connect(path)
    db.migrate(con)
    typer.echo(f"store at {path} is up to date")


@app.command("tables")
def tables_cmd(
    path: Path = typer.Option(Path(db.DEFAULT_DB_PATH), help="DuckDB file."),
) -> None:
    """Show row counts for every Section 3 table."""
    con = db.connect(path, read_only=True)
    for name in S.SCHEMAS:
        count = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        typer.echo(f"{name:16s} {count}")


@app.command("snapshot")
def snapshot_cmd(
    city: str = typer.Option(..., help="City id recorded in the manifest."),
    created_by: str = typer.Option(..., help="Creating user."),
    path: Path = typer.Option(Path(db.DEFAULT_DB_PATH), help="DuckDB file."),
    params_dir: Path = typer.Option(db.DEFAULT_PARAMS_DIR, help="config/params to copy."),
    out_root: Path = typer.Option(db.DEFAULT_SNAPSHOT_ROOT, help="Snapshot root."),
) -> None:
    """Write an immutable, hashed snapshot (Section 3.8)."""
    con = db.connect(path)
    ref = db.write_snapshot(
        con,
        city_id=city,
        created_by=created_by,
        params_dir=params_dir,
        out_root=out_root,
    )
    typer.echo(f"{ref.snapshot_id}\n{ref.snapshot_hash}\n{ref.path}")
