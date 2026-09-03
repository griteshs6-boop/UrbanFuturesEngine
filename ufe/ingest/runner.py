"""Tier orchestration for Module 2 — what ``ufe ingest <tier>`` actually does.

Section 6.0 splits data three ways by portability, and Section 20.2 runs the three tiers as
separate gated steps (6, 7, 8). This module holds that mapping and the one loop that is
common to every ingester:

    fetch (injectable reader)  ->  parse (pure)  ->  to_cells (pure)
        ->  ingest_runs row  ->  cell_imputation rows  ->  cells columns

Nothing here is ingester-specific and nothing branches on a state code: the state-tier
ingesters are handed a :class:`~ufe.ingest.adapters.base.StateAdapter` and that is the whole
of the multi-city mechanism (Section 23 item 11).

A failure in one ingester does not abort the tier — it is collected into
:attr:`TierResult.failures` and reported at the end — with one deliberate exception:
:class:`ufe.errors.MissingCriticalLayer` propagates immediately, because Section 20.2 step 4
makes a coastal city with no CRZ layer a hard stop rather than a partial result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ufe.errors import MissingCriticalLayer, SchemaValidationError
from ufe.ingest import core
from ufe.ingest.adapters.base import missing_capabilities

logger = logging.getLogger(__name__)

__all__ = ["TIERS", "TierResult", "ingesters_for_tier", "run_ingesters"]

#: Tier -> the ingester classes it runs, in dependency order. Buildings before population
#: (dasymetric refinement needs ``floorspace_res_sqm``); zoning before buildings' storey
#: fallback in a full rebuild, which is why the city tier is run last in Section 20.2.
TIERS: Mapping[str, tuple[str, ...]] = {
    "national": (
        "ufe.ingest.terrain:TerrainIngester",
        "ufe.ingest.landcover:LandcoverIngester",
        "ufe.ingest.buildings:BuildingsIngester",
        "ufe.ingest.population:PopulationIngester",
        "ufe.ingest.osm:OsmIngester",
        "ufe.ingest.nightlights:NightlightsIngester",
    ),
    "state": (
        "ufe.ingest.rera:ReraIngester",
        "ufe.ingest.cadastral:CadastralIngester",
    ),
    "city": (
        "ufe.ingest.zoning:ZoningIngester",
        "ufe.ingest.prices:PricesIngester",
        "ufe.ingest.projects:ProjectsIngester",
    ),
}

#: Ingesters that need the state adapter injected.
_NEEDS_ADAPTER = {"ReraIngester", "CadastralIngester", "PricesIngester"}
#: Ingesters that need the resolved parameter tree.
_NEEDS_PARAMS = {"PricesIngester", "ProjectsIngester"}


@dataclass
class TierResult:
    """What one tier produced."""

    cells: pd.DataFrame | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)
    imputation: pd.DataFrame = field(default_factory=pd.DataFrame)
    side_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    columns: set[str] = field(default_factory=set)
    failures: list[str] = field(default_factory=list)


def _load(spec: str) -> type:
    module_name, class_name = spec.split(":")
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def ingesters_for_tier(
    tier: str,
    *,
    reader: Any,
    city: Any,
    params: Any = None,
    adapter: Any = None,
) -> list[Any]:
    """Instantiate every ingester for ``tier``, injecting what each one declares it needs."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {sorted(TIERS)}")
    instances: list[Any] = []
    for spec in TIERS[tier]:
        cls = _load(spec)
        kwargs: dict[str, Any] = {"city": city}
        if cls.__name__ in _NEEDS_ADAPTER:
            kwargs["adapter"] = adapter
        if cls.__name__ in _NEEDS_PARAMS:
            kwargs["params"] = params
        instances.append(cls(reader, **kwargs))
    return instances


def run_ingesters(
    ingesters: Sequence[Any],
    *,
    connection: Any = None,
    cells: pd.DataFrame | None = None,
    city: Any,
    params: Any = None,
    adapter: Any = None,
    force: bool = False,
    on_step: Callable[[str], None] | None = None,
) -> TierResult:
    """Run the fetch/parse/to_cells loop for every ingester and persist the ledgers.

    ``cells`` defaults to the store's current ``cells`` table. The returned frame carries
    every column the tier produced plus a recomputed ``data_conf``; it is upserted into the
    store when ``connection`` is given and the frame validates.
    """
    from ufe.store import db as store

    result = TierResult()
    if cells is None:
        if connection is None:
            raise ValueError("run_ingesters needs either a cells frame or a connection")
        cells = store.read_table(connection, "cells")
    working = cells.copy()
    frames: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []

    for ingester in ingesters:
        name = type(ingester).__name__
        try:
            raw = ingester.fetch(city, force=force)
            parsed = ingester.parse(raw)
            produced = ingester.to_cells(parsed, working)
        except MissingCriticalLayer:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad source must not lose the rest
            logger.exception("%s failed", name)
            result.failures.append(f"{name}: {exc}")
            if on_step:
                on_step(name)
            continue

        provenance = ingester.provenance()
        columns = [c for c in core.value_columns(produced) if c != "h3"]
        run = core.ingest_run(
            provenance,
            city_id=city.city_id,
            tier=getattr(ingester, "tier", ""),
            columns=columns,
            rows=len(produced),
            params_hash=getattr(params, "hash", "") or "",
            content_hash=store.content_hash(produced.drop(columns=core.flag_companions(produced))),
        )
        result.runs.append(run)
        result.columns.update(columns)
        ledgers.append(
            core.imputation_long(
                produced, source_id=provenance["source_id"], run_id=run["run_id"]
            )
        )
        if columns:
            frames.append(produced)
            working = core.merge_ingested(working, produced)
        result.side_tables.update(getattr(ingester, "side_tables", {}) or {})
        if on_step:
            on_step(name)

    result.imputation = (
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(
            columns=list(core.CELL_IMPUTATION_COLUMNS)
        )
    )
    absent = missing_capabilities(adapter) if adapter is not None else set()
    # data_conf is always recomputed: it is a function of the imputation ledger and of the
    # adapter's missing capabilities (Section 6.0), both of which this run may have changed.
    working["data_conf"] = core.data_conf(
        result.imputation, working, missing_capabilities=absent
    )
    result.columns.add("data_conf")
    # Section 6: a cell attribute with no ingest_runs row is invalid, and that includes a
    # derived one. Its provenance is the set of runs it was computed from.
    result.runs.append(
        core.derived_run(
            ["data_conf"],
            city_id=city.city_id,
            from_run_ids=[run["run_id"] for run in result.runs],
            rows=len(working),
            params_hash=getattr(params, "hash", "") or "",
        )
    )
    result.cells = working

    if connection is not None:
        core.write_ingest_runs(connection, result.runs)
        core.write_cell_imputation(connection, result.imputation)
        try:
            store.write_table(connection, "cells", working, mode="upsert")
        except SchemaValidationError as exc:
            from ufe.store import schemas as schema_module

            absent = [
                name
                for name, column in schema_module.CELLS.columns.items()
                if column.required and name not in working.columns
            ]
            hint = (
                "run the remaining tiers (Section 20.2 steps 6-8) first"
                if absent
                else "see the validation error"
            )
            if {"households", "hh_by_band"} & set(absent):
                hint += (
                    "; note that households/hh_by_band cannot be derived until "
                    "behaviour.persons_per_household_by_band is populated — see "
                    "ufe.ingest.population.households_from_population"
                )
            result.failures.append(
                "cells not written: the frame does not yet satisfy schemas.CELLS. "
                f"Missing required column(s): {absent or 'none'} — {hint}. ({exc})"
            )
    return result
