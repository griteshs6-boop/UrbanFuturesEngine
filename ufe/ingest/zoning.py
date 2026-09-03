"""6.10 Zoning and CZMP/CRZ — the city-tier contracted GIS deliverables.

Section 6.10 is explicit that this is **not an automated ingest**: a GIS operator
georeferences the master plan and digitises the zoning polygons, and this module reads the
GeoPackage they deliver. Required attributes per polygon: ``zone_class``, ``permitted_far``,
``plan_name``, ``plan_year``, ``source_sheet`` — :func:`assert_required_attributes` refuses a
deliverable that is missing any of them, because a zoning layer with no provenance sheet
cannot be audited later.

Assignment rule, verbatim: "Assign to cells by majority area. Where a cell spans multiple
zones, store the area-weighted ``permitted_far`` and the majority ``zone_class``." Both are
computed in the city's ``crs_metric``.

**The hard requirement.** Section 6.10: "Fail loudly if a coastal city has no CRZ layer. Do
not silently substitute a distance buffer." Section 20.2 step 4 says the same thing and adds
"by design". :func:`crz_to_cells` therefore raises :class:`ufe.errors.MissingCriticalLayer`
for a city whose config says ``coastal: true`` when the CZMP layer is absent — there is no
flag, no fallback and no ``--force`` to get past it, because CRZ-I is a hard development
gate in Layer 0 and a distance buffer would produce a plausible, wrong answer.

This module also carries municipal utility coverage (``util_water``, ``util_sewer``), which
Section 6.0 lists as city-tier data alongside the master plan and which no other ingester
owns. Absent a utility layer both are 0 and flagged — an unmapped network is not an absent
one, and Layer 0's utility gate must know the difference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.errors import MissingCriticalLayer
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed
from ufe.store import schemas as S

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_ZONING",
    "KEY_CZMP",
    "KEY_UTILITIES",
    "ZONE_CLASS_COLUMN",
    "PERMITTED_FAR_COLUMN",
    "CRZ_CLASS_COLUMN",
    "UTIL_WATER_COLUMN",
    "UTIL_SEWER_COLUMN",
    "ZoningIngester",
    "assert_required_attributes",
    "zoning_to_cells",
    "crz_to_cells",
    "utilities_to_cells",
]

KEY_ZONING = "master_plan_zoning"
KEY_CZMP = "czmp_crz"
KEY_UTILITIES = "municipal_utilities"

ZONE_CLASS_COLUMN = "zone_class"
PERMITTED_FAR_COLUMN = "permitted_far"
CRZ_CLASS_COLUMN = "crz_class"
UTIL_WATER_COLUMN = "util_water"
UTIL_SEWER_COLUMN = "util_sewer"

_AREA = "_overlap_sqm"


def assert_required_attributes(
    layer: gpd.GeoDataFrame, required: Iterable[str], *, what: str
) -> None:
    """Refuse a contracted deliverable that is missing a required attribute (Section 6.10)."""
    missing = [column for column in required if column not in layer.columns]
    if missing:
        raise MissingCriticalLayer(
            f"{what} is missing required attribute(s) {missing}. Section 6.10 requires "
            f"{list(required)} on every polygon; a deliverable without them cannot be "
            "audited back to a plan sheet and is rejected."
        )


def _overlaps(
    cells: pd.DataFrame, layer: gpd.GeoDataFrame, *, crs_metric: str, columns: list[str]
) -> gpd.GeoDataFrame:
    """Cell x polygon intersections with their metric areas, for majority/weighted stats."""
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    polygons = layer if layer.crs is not None else layer.set_crs(geo.GEOGRAPHIC_CRS)
    polygons = geo.to_metric(
        polygons[[c for c in columns if c in polygons.columns] + [polygons.geometry.name]],
        crs_metric,
    )
    pieces = gpd.overlay(hexes, polygons.reset_index(drop=True), how="intersection", keep_geom_type=False)
    if len(pieces):
        pieces[_AREA] = pieces.geometry.area
    return pieces


def zoning_to_cells(
    zoning: gpd.GeoDataFrame, cells: pd.DataFrame, *, crs_metric: str, config: Any = None
) -> pd.DataFrame:
    """Majority ``zone_class`` and area-weighted ``permitted_far`` per cell (Section 6.10)."""
    required = [str(c) for c in cfg("zoning.required_attributes", config)]
    assert_required_attributes(zoning, required, what="master plan zoning deliverable")
    min_share = float(cfg("zoning.min_zoned_area_share", config))
    unzoned = str(cfg("zoning.unzoned_zone_class", config))
    if unzoned not in S.ZONE_CLASSES:
        raise MissingCriticalLayer(
            f"zoning.unzoned_zone_class {unzoned!r} is not in schemas.ZONE_CLASSES"
        )

    index = pd.Index(cells["h3"].astype(str), name="h3")
    pieces = _overlaps(cells, zoning, crs_metric=crs_metric, columns=required)
    out = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            ZONE_CLASS_COLUMN: unzoned,
            PERMITTED_FAR_COLUMN: 0.0,
        }
    )
    covered = np.zeros(len(index), dtype=bool)
    if len(pieces):
        hex_area = (
            geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
            .set_index("h3")
            .geometry.area.reindex(index)
        )
        zoned_area = pieces.groupby("h3")[_AREA].sum().reindex(index).fillna(0.0)
        share = np.divide(
            zoned_area.to_numpy(dtype=float),
            hex_area.to_numpy(dtype=float),
            out=np.zeros(len(index)),
            where=hex_area.to_numpy(dtype=float) > 0,
        )
        covered = share >= min_share

        majority = (
            pieces.sort_values(_AREA, ascending=False)
            .drop_duplicates("h3")
            .set_index("h3")[ZONE_CLASS_COLUMN]
            .reindex(index)
        )
        weighted = (
            pieces.assign(
                _w=pd.to_numeric(pieces[PERMITTED_FAR_COLUMN], errors="coerce")
                * pieces[_AREA]
            )
            .groupby("h3")[["_w", _AREA]]
            .sum()
        )
        far = (weighted["_w"] / weighted[_AREA]).reindex(index)
        out[ZONE_CLASS_COLUMN] = np.where(covered, majority.fillna(unzoned), unzoned)
        out[PERMITTED_FAR_COLUMN] = np.where(covered, far.fillna(0.0), 0.0)

    unknown = sorted(set(out[ZONE_CLASS_COLUMN]) - set(S.ZONE_CLASSES))
    if unknown:
        raise MissingCriticalLayer(
            f"master plan zoning uses zone_class values not in schemas.ZONE_CLASSES: "
            f"{unknown}. The GIS deliverable must be mapped onto the engine's vocabulary."
        )
    for column in (ZONE_CLASS_COLUMN, PERMITTED_FAR_COLUMN):
        out = mark_imputed(out, column, ~covered, "outside_master_plan_coverage")
    return out


def crz_to_cells(
    czmp: gpd.GeoDataFrame | None,
    cells: pd.DataFrame,
    *,
    city: CityConfig,
    config: Any = None,
) -> pd.DataFrame:
    """Majority ``crz_class`` per cell. **Raises for a coastal city with no CZMP layer.**

    Section 6.10 / Section 20.2 step 4: this is a hard requirement by design. CRZ-I is a
    hard gate in Layer 0, and substituting a coastline distance buffer would yield a
    plausible answer that is wrong in exactly the places that matter.
    """
    no_crz = str(cfg("zoning.no_crz_class", config))
    required = [str(c) for c in cfg("zoning.crz_required_attributes", config)]
    index = pd.Index(cells["h3"].astype(str), name="h3")

    if czmp is None or not len(czmp):
        if city.coastal:
            raise MissingCriticalLayer(
                f"{city.city_id} is declared coastal (config/cities/{city.city_id}.yaml: "
                "coastal: true) but no CZMP/CRZ layer was supplied. Section 20.2 step 4 "
                "makes CZMP digitisation a HARD REQUIREMENT and Section 6.10 forbids "
                "substituting a distance buffer. Commission the digitisation."
            )
        out = pd.DataFrame({"h3": index.to_numpy(), CRZ_CLASS_COLUMN: no_crz})
        return mark_imputed(
            out, CRZ_CLASS_COLUMN, np.zeros(len(out), dtype=bool), ""
        )

    assert_required_attributes(czmp, required, what="CZMP/CRZ deliverable")
    pieces = _overlaps(cells, czmp, crs_metric=city.crs_metric, columns=required)
    majority = (
        pieces.sort_values(_AREA, ascending=False)
        .drop_duplicates("h3")
        .set_index("h3")[CRZ_CLASS_COLUMN]
        .reindex(index)
        if len(pieces)
        else pd.Series(index=index, dtype=object)
    )
    values = majority.fillna(no_crz).astype(str)
    unknown = sorted(set(values) - set(S.CRZ_CLASSES))
    if unknown:
        raise MissingCriticalLayer(
            f"CZMP layer uses crz_class values not in schemas.CRZ_CLASSES: {unknown}"
        )
    out = pd.DataFrame({"h3": index.to_numpy(), CRZ_CLASS_COLUMN: values.to_numpy()})
    # Outside the CZMP polygons "none" is the correct observed answer, not an imputation.
    return mark_imputed(out, CRZ_CLASS_COLUMN, np.zeros(len(out), dtype=bool), "")


def utilities_to_cells(
    utilities: gpd.GeoDataFrame | None,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    water_column: str = "water_served",
    sewer_column: str = "sewer_served",
) -> pd.DataFrame:
    """Municipal water and sewer coverage per cell (city tier, Section 6.0).

    A cell is served when its centroid falls inside a served polygon. With no layer at all
    both flags are 0 and every cell is flagged, because Layer 0's utility gate must not
    read "unmapped" as "unserved" without knowing it did so.
    """
    index = pd.Index(cells["h3"].astype(str), name="h3")
    out = pd.DataFrame(
        {
            "h3": index.to_numpy(),
            UTIL_WATER_COLUMN: np.zeros(len(index), dtype=np.int64),
            UTIL_SEWER_COLUMN: np.zeros(len(index), dtype=np.int64),
        }
    )
    if utilities is None or not len(utilities):
        for column in (UTIL_WATER_COLUMN, UTIL_SEWER_COLUMN):
            out = mark_imputed(
                out, column, np.ones(len(out), dtype=bool), "no_utility_layer_zero"
            )
        return out

    layer = utilities if utilities.crs is not None else utilities.set_crs(geo.GEOGRAPHIC_CRS)
    keep = [c for c in (water_column, sewer_column) if c in layer.columns]
    layer = geo.to_metric(layer[keep + [layer.geometry.name]], crs_metric)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    centroids = gpd.GeoDataFrame(
        {"h3": hexes["h3"].to_numpy()}, geometry=hexes.geometry.centroid, crs=hexes.crs
    )
    joined = (
        gpd.sjoin(centroids, layer, how="left", predicate="within")
        .drop_duplicates("h3")
        .set_index("h3")
    )
    for column, source in (
        (UTIL_WATER_COLUMN, water_column),
        (UTIL_SEWER_COLUMN, sewer_column),
    ):
        if source in keep:
            served = (
                joined[source].reindex(index).fillna(0).astype(float) > 0
            ).astype(np.int64)
            out[column] = served.to_numpy()
            out = mark_imputed(out, column, np.zeros(len(out), dtype=bool), "")
        else:
            out = mark_imputed(
                out, column, np.ones(len(out), dtype=bool), "utility_layer_lacks_column"
            )
    # Section 7.3 precedence: sewer without water is not a state the engine models.
    out[UTIL_SEWER_COLUMN] = out[UTIL_SEWER_COLUMN] * out[UTIL_WATER_COLUMN]
    return out


class ZoningIngester(Ingester):
    """Master plan + CZMP + municipal utilities -> the regulation and utility columns."""

    source_id = "government_open_data"
    tier = "city"
    fills = (
        ZONE_CLASS_COLUMN,
        PERMITTED_FAR_COLUMN,
        CRZ_CLASS_COLUMN,
        UTIL_WATER_COLUMN,
        UTIL_SEWER_COLUMN,
    )
    spatial_res = "master plan / CZMP polygons as digitised"
    temporal_res = "per plan vintage (plan_year)"
    notes = (
        "Manual, contracted GIS deliverables read from a GeoPackage (Section 6.10), not an "
        "automated ingest. The underlying documents are government publications; the "
        "georeferencing is contracted work. NOTE: there is no 'contracted GIS' entry in "
        "config/data_sources_licences.yaml, so this run is recorded against the government "
        "open-data licence and the missing registry entry is reported."
    )

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_ZONING, KEY_CZMP, KEY_UTILITIES)

    def parse(self, raw: Path) -> pd.DataFrame:
        """The zoning polygons; CZMP and utilities are read by key in ``to_cells``."""
        return self.reader.vector(KEY_ZONING)

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("ZoningIngester needs a CityConfig")
        crs = self.city.crs_metric
        zones = zoning_to_cells(df, cells, crs_metric=crs, config=self.config)
        czmp = self.reader.vector(KEY_CZMP) if self.reader.exists(KEY_CZMP) else None
        crz = crz_to_cells(czmp, cells, city=self.city, config=self.config)
        utilities = (
            self.reader.vector(KEY_UTILITIES) if self.reader.exists(KEY_UTILITIES) else None
        )
        util = utilities_to_cells(utilities, cells, crs_metric=crs)
        out = zones.merge(crz, on="h3", validate="one_to_one").merge(
            util, on="h3", validate="one_to_one"
        )
        return out
