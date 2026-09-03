"""6.5 OSM — the state ``.osm.pbf``, clipped to the halo.

Extracts, per Section 6.5: the road network (routing itself is handled outside this module,
by the OSRM build), POIs by category, power infrastructure (``power=substation``,
``power=line``), railway and landuse polygons. The cell columns produced here are:

===================== ==========================================================
``util_power``        1 if a substation is within 2 000 m **or** an 11/33 kV line
                      passes within 500 m (Section 6.5, verbatim)
``dist_arterial_m``   metric distance to the nearest arterial highway
``dist_cbd_m``        metric distance to the city's ``cbd_point``
``dist_coast_m``      metric distance to the OSM coastline (nullable)
``retail_poi_count``  POI counts by category, which Section 8.2 consumes and
``education_poi_count`` Section 6.5 also uses as the ``jobs_by_sector`` proxy
``health_poi_count``
``jobs_by_sector``    **proxy only** — POI density times a jobs-per-POI figure.
                      Flagged imputed on every cell it touches.
===================== ==========================================================

Every distance is computed in the city's ``crs_metric`` through :mod:`ufe.geo` — the
Section 6 ACCEPTANCE item "two points 10 km apart in the metric CRS return 10 000 +- 50 m"
is a test against :func:`distance_to_points_m`.

Data rights (Section 22.1)
--------------------------
``dist_arterial_m``, ``util_power`` and ``jobs_by_sector`` are exactly the columns
:data:`ufe.rights.CELLS_OSM_DERIVED_RAW_COLUMNS` marks as raw OSM-derived. They may be
computed into ``cells`` and consumed by the layers, and must never be served as bulk
per-cell data. ODbL 1.0 attribution is recorded in ``provenance()``.

Genuinely complete for the transform half. The ``.osm.pbf`` clip and tag extraction is a
structural stub: it reads pre-extracted layers from the injected reader rather than parsing
a PBF, because there is no PBF here and osmium/pyrosm parsing is not exercisable offline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ufe import geo
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed
from ufe.store import schemas as S

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_SUBSTATIONS",
    "KEY_POWER_LINES",
    "KEY_ROADS",
    "KEY_COASTLINE",
    "KEY_POIS",
    "UTIL_POWER_COLUMN",
    "DIST_CBD_COLUMN",
    "DIST_COAST_COLUMN",
    "DIST_ARTERIAL_COLUMN",
    "POI_COLUMNS",
    "OsmIngester",
    "distance_to_points_m",
    "distance_to_layer_m",
    "util_power_flag",
    "poi_counts",
    "jobs_by_sector_proxy",
    "distances_to_cells",
    "osm_to_cells",
]

KEY_SUBSTATIONS = "osm/substations"
KEY_POWER_LINES = "osm/power_lines"
KEY_ROADS = "osm/roads"
KEY_COASTLINE = "osm/coastline"
KEY_POIS = "osm/pois"

UTIL_POWER_COLUMN = "util_power"
DIST_CBD_COLUMN = "dist_cbd_m"
DIST_COAST_COLUMN = "dist_coast_m"
DIST_ARTERIAL_COLUMN = "dist_arterial_m"
POI_COLUMNS = {
    "retail": "retail_poi_count",
    "education": "education_poi_count",
    "health": "health_poi_count",
}

_JOBS_COLUMN = "jobs_by_sector"


# --------------------------------------------------------------------------------------
# Distances — always metric (Section 0.3)
# --------------------------------------------------------------------------------------


def distance_to_points_m(
    origins: Sequence[tuple[float, float]],
    destination: tuple[float, float],
    *,
    crs_metric: str,
) -> np.ndarray:
    """Metric distance from each ``(lat, lon)`` origin to one ``(lat, lon)`` destination.

    Goes through :func:`ufe.geo.metric_distance_m`, so a geographic ``crs_metric`` raises
    :class:`ufe.geo.NonMetricCRSError` instead of quietly returning degrees. This is the
    function the Section 6 ACCEPTANCE known-answer distance test drives.
    """
    a = gpd.GeoSeries(
        [Point(float(lon), float(lat)) for lat, lon in origins], crs=geo.GEOGRAPHIC_CRS
    )
    b = gpd.GeoSeries(
        [Point(float(destination[1]), float(destination[0]))] * len(a), crs=geo.GEOGRAPHIC_CRS
    )
    return np.asarray(geo.metric_distance_m(a, b, crs_metric), dtype=float)


def distance_to_layer_m(
    cells: pd.DataFrame,
    layer: gpd.GeoDataFrame | None,
    *,
    crs_metric: str,
    ceiling: float | None = None,
) -> np.ndarray:
    """Distance from each cell's centroid to the nearest feature in ``layer``, in metres.

    Returns ``ceiling`` (or NaN when no ceiling is given) for every cell when the layer is
    absent or empty — never a fabricated distance.
    """
    n = len(cells)
    if layer is None or not len(layer):
        return np.full(n, np.nan if ceiling is None else float(ceiling), dtype=float)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    centroids = gpd.GeoDataFrame(
        {"h3": hexes["h3"].to_numpy()}, geometry=hexes.geometry.centroid, crs=hexes.crs
    )
    target = layer if layer.crs is not None else layer.set_crs(geo.GEOGRAPHIC_CRS)
    target = geo.to_metric(target[[target.geometry.name]], crs_metric)
    union = target.union_all() if hasattr(target, "union_all") else target.unary_union
    distances = centroids.geometry.distance(union).to_numpy(dtype=float)
    if ceiling is not None:
        distances = np.minimum(distances, float(ceiling))
    return distances


def distances_to_cells(
    cells: pd.DataFrame,
    *,
    city: CityConfig,
    roads: gpd.GeoDataFrame | None = None,
    coastline: gpd.GeoDataFrame | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """``dist_cbd_m``, ``dist_arterial_m`` and ``dist_coast_m`` per cell."""
    arterial_values = [str(v) for v in cfg("osm.arterial_highway_values", config)]
    ceiling = float(cfg("osm.dist_arterial_ceiling_m", config))

    arterials = None
    if roads is not None and len(roads):
        if "highway" in roads.columns:
            arterials = roads[roads["highway"].astype(str).isin(arterial_values)]
        else:
            arterials = roads

    out = pd.DataFrame({"h3": cells["h3"].astype(str).to_numpy()})
    out[DIST_CBD_COLUMN] = distance_to_points_m(
        list(zip(cells["lat"].to_numpy(dtype=float), cells["lon"].to_numpy(dtype=float))),
        (city.cbd_lat, city.cbd_lon),
        crs_metric=city.crs_metric,
    )
    out = mark_imputed(out, DIST_CBD_COLUMN, np.zeros(len(out), dtype=bool), "")

    arterial = distance_to_layer_m(
        cells, arterials, crs_metric=city.crs_metric, ceiling=ceiling
    )
    out[DIST_ARTERIAL_COLUMN] = arterial
    missing_roads = arterials is None or not len(arterials)
    out = mark_imputed(
        out,
        DIST_ARTERIAL_COLUMN,
        np.full(len(out), bool(missing_roads)),
        "no_arterial_layer_ceiling",
    )

    coast = distance_to_layer_m(cells, coastline, crs_metric=city.crs_metric)
    out[DIST_COAST_COLUMN] = coast
    missing_coast = coastline is None or not len(coastline)
    if missing_coast and city.coastal:
        logger.warning(
            "%s is declared coastal but no OSM coastline layer was supplied; "
            "dist_coast_m is left null (the schema allows it) and flagged",
            city.city_id,
        )
    out = mark_imputed(
        out, DIST_COAST_COLUMN, np.full(len(out), bool(missing_coast)), "no_coastline_layer"
    )
    return out


# --------------------------------------------------------------------------------------
# Power infrastructure
# --------------------------------------------------------------------------------------


def util_power_flag(
    cells: pd.DataFrame,
    *,
    substations: gpd.GeoDataFrame | None,
    power_lines: gpd.GeoDataFrame | None,
    crs_metric: str,
    config: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 6.5, verbatim: substation within 2 000 m OR an 11/33 kV line within 500 m.

    Returns ``(util_power, imputed_mask)``. With neither layer present the flag is 0 and
    every cell is flagged — an unmapped network is not the same as an absent one.
    """
    substation_radius = float(cfg("osm.substation_radius_m", config))
    line_radius = float(cfg("osm.power_line_radius_m", config))
    voltages = [float(v) for v in cfg("osm.power_line_voltages_v", config)]

    lines = power_lines
    if lines is not None and len(lines) and "voltage" in lines.columns:
        volts = pd.to_numeric(lines["voltage"], errors="coerce")
        lines = lines[volts.isin(voltages)]

    d_sub = distance_to_layer_m(cells, substations, crs_metric=crs_metric)
    d_line = distance_to_layer_m(cells, lines, crs_metric=crs_metric)
    near_sub = np.nan_to_num(d_sub, nan=np.inf) <= substation_radius
    near_line = np.nan_to_num(d_line, nan=np.inf) <= line_radius
    flag = (near_sub | near_line).astype(np.int64)

    have_nothing = (substations is None or not len(substations)) and (
        lines is None or not len(lines)
    )
    return flag, np.full(len(cells), bool(have_nothing))


# --------------------------------------------------------------------------------------
# POIs and the jobs proxy
# --------------------------------------------------------------------------------------


def poi_counts(
    cells: pd.DataFrame,
    pois: gpd.GeoDataFrame | None,
    *,
    crs_metric: str,
    config: Any = None,
) -> pd.DataFrame:
    """POI counts per cell per category (Section 6.5).

    A POI is attributed to the cell whose polygon contains it. Category membership is the
    ``osm.poi_categories`` tag-value lookup in ``config/ingest.yaml``, matched against the
    ``amenity`` / ``shop`` / ``office`` value the extract carries in a ``tag_value`` column.
    """
    categories: Mapping[str, Sequence[str]] = cfg("osm.poi_categories", config)
    index = pd.Index(cells["h3"].astype(str), name="h3")
    out = pd.DataFrame({column: np.zeros(len(index)) for column in POI_COLUMNS.values()})
    out.insert(0, "h3", index.to_numpy())
    if pois is None or not len(pois):
        return out

    layer = pois if pois.crs is not None else pois.set_crs(geo.GEOGRAPHIC_CRS)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    layer = geo.to_metric(
        layer[[c for c in ("tag_value",) if c in layer.columns] + [layer.geometry.name]],
        crs_metric,
    )
    joined = gpd.sjoin(layer, hexes, how="inner", predicate="within")
    if not len(joined):
        return out
    values = (
        joined["tag_value"].astype(str) if "tag_value" in joined.columns else pd.Series("", index=joined.index)
    )
    for category, column in POI_COLUMNS.items():
        wanted = {str(v) for v in categories.get(category, ())}
        subset = joined[values.isin(wanted)]
        counts = subset.groupby("h3").size().reindex(index).fillna(0.0)
        out[column] = counts.to_numpy(dtype=float)
    return out


def jobs_by_sector_proxy(
    counts: pd.DataFrame, *, config: Any = None
) -> tuple[list[list[float]], np.ndarray]:
    """POI density -> ``jobs_by_sector`` (Section 6.5: "used as a proxy for jobs_by_sector").

    A structural guess by construction, so the mask returned is all-True: every cell's
    ``jobs_by_sector`` is flagged imputed and lowers ``data_conf``. The per-POI job figures
    live in ``config/ingest.yaml``.
    """
    per_poi: Mapping[str, float] = cfg("osm.jobs_per_poi", config)
    sector_of_category = {"retail": "retail_svc", "education": "public_edu"}
    vectors: list[list[float]] = []
    for _, row in counts.iterrows():
        vector = [0.0] * len(S.SECTORS)
        for category, column in POI_COLUMNS.items():
            sector = sector_of_category.get(category)
            if sector is None or sector not in per_poi:
                continue
            vector[S.Sector[sector].value] = float(row[column]) * float(per_poi[sector])
        vectors.append(vector)
    return vectors, np.ones(len(counts), dtype=bool)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def osm_to_cells(
    cells: pd.DataFrame,
    *,
    city: CityConfig,
    substations: gpd.GeoDataFrame | None = None,
    power_lines: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
    coastline: gpd.GeoDataFrame | None = None,
    pois: gpd.GeoDataFrame | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """Every OSM-derived cell column in one frame."""
    out = distances_to_cells(
        cells, city=city, roads=roads, coastline=coastline, config=config
    )
    flag, power_imputed = util_power_flag(
        cells,
        substations=substations,
        power_lines=power_lines,
        crs_metric=city.crs_metric,
        config=config,
    )
    out[UTIL_POWER_COLUMN] = flag
    out = mark_imputed(out, UTIL_POWER_COLUMN, power_imputed, "no_power_layer_zero")

    counts = poi_counts(cells, pois, crs_metric=city.crs_metric, config=config)
    for column in POI_COLUMNS.values():
        out[column] = counts[column].to_numpy()
        out = mark_imputed(
            out,
            column,
            np.full(len(out), pois is None or not len(pois)),
            "no_poi_layer_zero",
        )
    vectors, jobs_imputed = jobs_by_sector_proxy(counts, config=config)
    out[_JOBS_COLUMN] = vectors
    out = mark_imputed(out, _JOBS_COLUMN, jobs_imputed, "poi_density_proxy")
    return out


class OsmIngester(Ingester):
    """OpenStreetMap -> power, distances, POI counts and the ``jobs_by_sector`` proxy."""

    source_id = "openstreetmap"
    tier = "national"
    fills = (
        UTIL_POWER_COLUMN,
        DIST_CBD_COLUMN,
        DIST_COAST_COLUMN,
        DIST_ARTERIAL_COLUMN,
        _JOBS_COLUMN,
    ) + tuple(POI_COLUMNS.values())
    spatial_res = "vector, OSM node/way precision"
    temporal_res = "continuous; pinned to the .osm.pbf snapshot date"
    notes = (
        "ODbL 1.0. dist_arterial_m, util_power and jobs_by_sector are raw OSM-derived "
        "columns (Section 22.1) and must never be exposed as bulk per-cell data; "
        "ufe.rights.assert_exposable enforces that at the API boundary."
    )

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_ROADS, KEY_SUBSTATIONS, KEY_POWER_LINES, KEY_COASTLINE, KEY_POIS)

    def parse(self, raw: Path) -> pd.DataFrame:
        """A manifest of the layers the reader holds; the layers are read in ``to_cells``."""
        if self.city is None:
            raise ValueError("OsmIngester needs a CityConfig")
        return pd.DataFrame(
            [{"layer": key, "available": self.reader.exists(key)} for key in self.keys(self.city)]
        )

    def _layer(self, key: str) -> gpd.GeoDataFrame | None:
        return self.reader.vector(key) if self.reader.exists(key) else None

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        """The road layer is the primary artefact; the rest are read lazily by key."""
        for key in self.keys(city):
            if self.reader.exists(key):
                if not force and key in self._fetched:
                    return self._fetched[key]
                path = self.reader.path(key, force=force)
                self._fetched[key] = path
                return path
        from ufe.ingest.core import MissingSource

        raise MissingSource(f"no OSM layer available for {city.city_id}")

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("OsmIngester needs a CityConfig")
        return osm_to_cells(
            cells,
            city=self.city,
            substations=self._layer(KEY_SUBSTATIONS),
            power_lines=self._layer(KEY_POWER_LINES),
            roads=self._layer(KEY_ROADS),
            coastline=self._layer(KEY_COASTLINE),
            pois=self._layer(KEY_POIS),
            config=self.config,
        )
