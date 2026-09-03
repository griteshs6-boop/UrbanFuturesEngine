"""`ufe grid` -- Module 1 CLI (spec Section 20.2 step 5: `ufe grid build --city {id}`)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ufe.grid.build import build_grid, load_boundary
from ufe.params import load_params

logger = logging.getLogger(__name__)

app = typer.Typer(help="Module 1 -- grid construction (spec Section 5).")


@app.command("build")
def build(
    city: str = typer.Option(..., "--city", help="City id, matching config/cities/{city}.yaml."),
    boundary: Path = typer.Option(
        None,
        "--boundary",
        help="Override boundary file path (defaults to the city config's boundary_source).",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Parquet output path (defaults to data/cache/grid/{city}_cells_geometry.parquet).",
    ),
) -> None:
    """Build the simulation grid for `city` and write its geometry columns to Parquet.

    Only the columns Module 1 owns (`h3, h3_res8, in_city, geometry, lat, lon, area_sqm`,
    Section 5.1 step 6) are written here. The remaining `SCHEMAS['cells']` columns are
    filled in by Module 2's ingest layers before the combined frame is written to the
    `cells` store table -- the `cells` pandera schema is `strict=True` with most base
    columns non-nullable, so a geometry-only frame cannot pass `ufe.store.db.write_table`
    on its own.
    """
    params = load_params(city)
    boundary_path = boundary if boundary is not None else params.city_config.get("boundary_source")
    if boundary_path is None:
        raise typer.BadParameter(
            f"city {city!r} config has no 'boundary_source' and --boundary was not given"
        )

    geometry = load_boundary(boundary_path)
    frame = build_grid(geometry, params)

    out_path = out if out is not None else Path("data/cache/grid") / f"{city}_cells_geometry.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path)

    typer.echo(
        f"wrote {len(frame)} cells ({int(frame['in_city'].sum())} in_city) for {city!r} to {out_path}"
    )


if __name__ == "__main__":
    app()
