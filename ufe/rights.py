"""Data-rights enforcement (Section 22): the OSM Produced Work rule and attribution rendering.

## The rule (Section 22.1)

OSM is ODbL 1.0. `cells` mixes OSM-derived columns with columns from other sources and is, on
any cautious reading, a Derivative Database. Share-alike is triggered only by publicly
distributing that database — so the product must never expose raw OSM-derived columns as bulk
per-cell data. It may freely expose **Produced Work**: computed outputs (prices, rankings,
factor loadings, residuals, scenario results, rendered maps/reports) synthesised from the
substrate.

## Column classification

Every column of the `cells` table (Section 3.1) is classified into exactly one of:

  * `"osm_derived_raw"` — a per-cell attribute whose value comes directly from OSM extraction
    (Section 6.5: road network, POI density by category, power infrastructure tags). Exposing
    one of these as bulk per-cell data IS distributing a Derivative Database.
  * `"clean"` — a `cells` column sourced from something other than OSM (DEM, land cover,
    census/GHSL/WorldPop, master plan, pricing sources, VIIRS, geometry). Safe with respect to
    the OSM rule specifically, though Section 22.1 still discourages exposing the raw grid as
    bulk data generally — that broader "no bulk cells table" rule is an API/product design
    constraint, not something this module can detect from a column name alone.
  * `"produced_work"` — anything NOT in the known `cells` schema. This is the default for a
    column name this module doesn't recognise, on the assumption that a name outside the
    substrate schema is a computed output column (e.g. `price_index`, `rank`, `scenario_delta`,
    a residual or factor loading) produced by a later layer. This is a permissive default by
    design: `assert_exposable` only ever blocks names it can positively identify as raw OSM
    columns, per the Section 23 item 9 requirement ("no API route exposes a raw OSM-derived
    column") — it does not attempt to police arbitrary future output-schema column names.

## The guard

    from ufe.rights import assert_exposable

    def build_response_schema(columns: list[str]) -> None:
        assert_exposable(columns)  # raises ufe.errors.DataRightsViolation on a raw OSM column

Call this on the column list of every API response schema before it goes out. It is a pure
function with no I/O and no side effects, safe to call at import time or per-request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import yaml

from ufe.errors import DataRightsViolation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_LICENCES_PATH = REPO_ROOT / "config" / "data_sources_licences.yaml"

# --- OSM Produced Work column classification (Section 22.1, Section 3.1) -------------------

#: `cells` columns whose value is a direct per-cell expression of OSM-extracted data
#: (Section 6.5: road network -> dist_arterial_m; power=substation/power=line -> util_power;
#: POI density by category, used as-is as a jobs_by_sector proxy -> jobs_by_sector).
CELLS_OSM_DERIVED_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "dist_arterial_m",
        "util_power",
        "jobs_by_sector",
    }
)

#: The remaining columns of the `cells` schema (Section 3.1), sourced from DEM, land cover,
#: census/GHSL/WorldPop, the master plan, pricing sources, VIIRS, or plain geometry — never OSM.
CELLS_CLEAN_COLUMNS: frozenset[str] = frozenset(
    {
        "h3",
        "lat",
        "lon",
        "area_sqm",
        "elev_m",
        "slope_pct",
        "landcover",
        "builtup_frac",
        "undevelopable_frac",
        "zone_class",
        "permitted_far",
        "crz_class",
        "population",
        "households",
        "hh_by_band",
        "floorspace_res_sqm",
        "floorspace_com_sqm",
        "price_res_inr_sqft",
        "price_land_inr_sqft",
        "rent_res_inr_sqft_mo",
        "mean_parcel_sqm",
        "parcel_count",
        "util_water",
        "util_sewer",
        "dist_cbd_m",
        "dist_coast_m",
        "nightlight",
        "data_conf",
    }
)

ColumnClass = Literal["osm_derived_raw", "clean", "produced_work"]


def classify_column(name: str) -> ColumnClass:
    """Classify a single column name per the OSM Produced Work rule.

    Exact, case-sensitive match against the `cells` schema (Section 3.1). Any name outside
    that schema defaults to `"produced_work"` — see the module docstring for why that default
    is safe for this guard's purpose.
    """
    if name in CELLS_OSM_DERIVED_RAW_COLUMNS:
        return "osm_derived_raw"
    if name in CELLS_CLEAN_COLUMNS:
        return "clean"
    return "produced_work"


def classify_columns(columns: Iterable[str]) -> dict[str, ColumnClass]:
    """Classify every column in `columns`, preserving input order."""
    return {c: classify_column(c) for c in columns}


def assert_exposable(columns: Iterable[str]) -> None:
    """Guard for API response schemas (Section 22.1, Section 23 item 9).

    Raises `ufe.errors.DataRightsViolation` if any name in `columns` classifies as
    `"osm_derived_raw"`. Call this on every API route's response column list before it can
    reach a client. Safe to call with an empty iterable or with column names that are not part
    of the `cells` schema (those are treated as produced-work outputs and pass).

    This function does NOT open a database connection, read a schema file, or otherwise
    perform I/O — it is a pure name lookup, safe to call per-request without a performance
    penalty and safe to unit test without fixtures.
    """
    offending = [c for c in columns if classify_column(c) == "osm_derived_raw"]
    if offending:
        raise DataRightsViolation(
            "Refusing to expose raw OSM-derived column(s) "
            f"{offending} as bulk per-cell data. Per Section 22.1, OpenStreetMap is ODbL-1.0: "
            "a per-cell attribute extracted directly from OSM (road network, POI density, "
            "power infrastructure) is part of our Derivative Database, and returning it as "
            "bulk data would distribute that database and trigger share-alike. Expose a "
            "computed output derived FROM this column instead (a price, ranking, factor "
            "loading, residual, or scenario result) — never the column itself."
        )


# --- Attribution rendering (Section 22.4) ---------------------------------------------------


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _load_data_licences(path: Path = DEFAULT_DATA_LICENCES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_source_key(key: str, sources: dict) -> str | None:
    norm = _normalise(key)
    if norm in sources:
        return norm
    for canonical, entry in sources.items():
        if norm == _normalise(canonical):
            return canonical
        for alias in entry.get("aliases", []) or []:
            if norm == _normalise(alias):
                return canonical
    return None


def get_attribution_text(
    source_keys: Iterable[str] | None = None,
    *,
    data_licences_path: Path = DEFAULT_DATA_LICENCES_PATH,
) -> str:
    """Render the attribution text required for a report footer, the about page, or the
    `/attributions` API endpoint (Section 22.4).

    `source_keys`: the data sources actually used in the artefact being built (e.g. the
    ingesters a given report drew on). Each key may be a canonical key or an alias from
    `config/data_sources_licences.yaml`. If omitted, renders the full attribution block for
    every known source (suitable for the product's about page).

    Raises `ufe.errors.DataRightsViolation` if a requested source key cannot be resolved —
    per Section 22.4, "a report build that cannot resolve an attribution for a source it used
    must fail," rather than silently omitting the attribution.
    """
    data = _load_data_licences(data_licences_path)
    sources = data.get("sources", {})

    if source_keys is None:
        keys = list(sources.keys())
    else:
        keys = []
        for requested in source_keys:
            resolved = _resolve_source_key(requested, sources)
            if resolved is None:
                raise DataRightsViolation(
                    f"No known attribution for data source '{requested}' — add it to "
                    "config/data_sources_licences.yaml before shipping this report/page."
                )
            keys.append(resolved)

    lines = [sources[k]["attribution"] for k in keys if sources[k].get("attribution")]
    return "\n".join(lines)
