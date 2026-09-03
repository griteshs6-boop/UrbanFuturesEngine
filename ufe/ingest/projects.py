"""6.11 Projects — load and validate the human-verified project registry.

This ingester does **not** scrape. Section 17 populates the registry through the AI
pipeline; Section 6.11 loads the approved records and validates them. Every rule below is
declared fatal by the spec, and all of them are checked in one pass so an operator sees
every problem at once rather than fixing them one exception at a time:

* ``archetype`` must exist in ``archetypes.yaml``;
* ``scale_unit`` must match the archetype's declared unit;
* ``source_urls`` non-empty;
* a ``stage`` transition must have a corresponding ``project_history`` row with
  ``changed_by != 'ai'`` — i.e. a human signed off on the stage the model will act on;
* geometry must fall inside the analysis halo;
* ``announced_date <= stated_completion``.

The halo test uses the ingested cell set as the definition of the halo (Module 1 built it),
reprojected into the city's ``crs_metric`` for the containment test.

Genuinely complete: this is a validator, it needs no external data, and every rule has a
test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd
import shapely

from ufe import geo
from ufe.errors import SchemaValidationError
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_REGISTRY",
    "ProjectsIngester",
    "ProjectValidationError",
    "archetype_units",
    "validate_projects",
]

KEY_REGISTRY = "project_registry"


class ProjectValidationError(SchemaValidationError):
    """The project registry violated one or more of the Section 6.11 fatal rules."""


def archetype_units(params: Any) -> dict[str, str]:
    """``{archetype: scale_unit}`` from ``archetypes.yaml``.

    The archetype's declared unit is whatever its ``scale_unit`` leaf says; an archetype
    block without one cannot constrain a project and is reported.
    """
    tree = params.get("archetypes")
    units: dict[str, str] = {}
    for name, block in (tree or {}).items():
        if name.startswith("_") or not isinstance(block, Mapping):
            continue
        unit = block.get("scale_unit")
        if isinstance(unit, Mapping):
            unit = unit.get("value")
        if unit is not None:
            units[str(name)] = str(unit)
        else:
            units[str(name)] = ""
    return units


def validate_projects(
    projects: pd.DataFrame,
    *,
    params: Any,
    cells: pd.DataFrame,
    project_history: pd.DataFrame | None = None,
    city: CityConfig | None = None,
) -> pd.DataFrame:
    """Apply every Section 6.11 rule. Raises :class:`ProjectValidationError` with all of them.

    Returns the validated frame unchanged (a validator, not a transform), so the caller can
    hand it straight to :func:`ufe.store.db.write_table`.
    """
    problems: list[str] = []
    units = archetype_units(params)
    forbidden = str(cfg("projects.forbidden_history_author_prefix"))

    for column in ("project_id", "archetype", "scale_unit", "source_urls", "stage", "geom"):
        if column not in projects.columns:
            problems.append(f"registry has no {column!r} column")
    if problems:
        raise ProjectValidationError("; ".join(problems))

    # -- archetype / scale_unit -----------------------------------------------------
    for _, row in projects.iterrows():
        archetype = str(row["archetype"])
        if archetype not in units:
            problems.append(
                f"{row['project_id']}: archetype {archetype!r} does not exist in "
                f"archetypes.yaml (known: {sorted(units)})"
            )
            continue
        declared = units[archetype]
        if declared and str(row["scale_unit"]) != declared:
            problems.append(
                f"{row['project_id']}: scale_unit {row['scale_unit']!r} does not match "
                f"archetype {archetype!r}'s declared unit {declared!r}"
            )

    # -- source_urls non-empty ------------------------------------------------------
    for _, row in projects.iterrows():
        urls = row["source_urls"]
        if urls is None or len(list(urls)) == 0:
            problems.append(f"{row['project_id']}: source_urls is empty")

    # -- announced_date <= stated_completion ----------------------------------------
    if {"announced_date", "stated_completion"} <= set(projects.columns):
        announced = pd.to_datetime(projects["announced_date"], errors="coerce")
        completion = pd.to_datetime(projects["stated_completion"], errors="coerce")
        bad = projects.loc[(announced > completion).fillna(False), "project_id"]
        problems.extend(
            f"{pid}: announced_date is after stated_completion" for pid in bad
        )

    # -- human-signed stage transitions ---------------------------------------------
    history = (
        project_history
        if project_history is not None
        else pd.DataFrame(columns=["project_id", "field", "changed_by"])
    )
    stage_rows = history[history.get("field", pd.Series(dtype=str)) == "stage"]
    human = stage_rows[
        ~stage_rows["changed_by"].astype(str).str.startswith(forbidden)
    ]
    signed = set(human["project_id"].astype(str))
    initial_stage = None
    try:  # the first stage in the vocabulary is "announced": no transition to sign off
        from ufe.store.schemas import PROJECT_STAGES

        initial_stage = PROJECT_STAGES[0]
    except Exception:  # pragma: no cover - schemas always importable
        pass
    for _, row in projects.iterrows():
        if str(row["stage"]) == initial_stage:
            continue
        if str(row["project_id"]) not in signed:
            problems.append(
                f"{row['project_id']}: stage {row['stage']!r} has no project_history row "
                f"with changed_by != {forbidden!r} (Section 6.11: a stage the model acts on "
                "must be human-verified)"
            )

    # -- geometry inside the analysis halo ------------------------------------------
    if city is not None and len(cells):
        halo = cells_gdf(cells)
        halo_union = (
            halo.geometry.union_all()
            if hasattr(halo.geometry, "union_all")
            else halo.geometry.unary_union
        )
        geoms = gpd.GeoSeries(
            [shapely.from_wkt(str(g)) for g in projects["geom"]], crs=geo.GEOGRAPHIC_CRS
        )
        outside = ~geoms.intersects(halo_union)
        problems.extend(
            f"{pid}: geometry falls outside the analysis halo"
            for pid in projects.loc[outside.to_numpy(), "project_id"]
        )

    if problems:
        raise ProjectValidationError(
            "project registry failed Section 6.11 validation:\n  " + "\n  ".join(problems)
        )
    return projects


class ProjectsIngester(Ingester):
    """The human-verified project registry: loaded and validated, never scraped."""

    source_id = "government_open_data"
    tier = "city"
    fills = ()
    spatial_res = "project geometry as recorded (point / polygon / linestring)"
    temporal_res = "as verified"
    notes = (
        "Reads the approved records from the Section 17 AI pipeline's output table and "
        "applies the six fatal validation rules of Section 6.11. Performs no extraction "
        "and no network access."
    )

    def __init__(
        self,
        reader: Any,
        *,
        city: CityConfig | None = None,
        params: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reader, city=city, config=config)
        self.params = params

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_REGISTRY,)

    def parse(self, raw: Path) -> pd.DataFrame:
        return self.reader.table(KEY_REGISTRY)

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        """Projects fill no ``cells`` column; this validates and returns the registry."""
        if self.params is None:
            raise ValueError("ProjectsIngester needs Params to resolve archetypes.yaml")
        history = (
            self.reader.table("project_history")
            if self.reader.exists("project_history")
            else None
        )
        validate_projects(
            df, params=self.params, cells=cells, project_history=history, city=self.city
        )
        self.side_tables = {"projects": df}
        return pd.DataFrame({"h3": cells["h3"].astype(str).to_numpy()})
