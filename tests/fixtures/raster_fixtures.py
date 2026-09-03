"""Deterministic synthetic rasters and vectors for offline Module 2 tests.

There is no real source data in this environment and no network at test time, so every
ingest test drives the *transform* half of an ingester against data built here. Everything
is seeded or analytic: same inputs, byte-identical output (spec Section 0.1 rule 7).

The cell set always comes from ``tests/fixtures/synthetic.py`` — the shared synthetic city —
so these fixtures are aligned with the hexagons every other module's tests use. Rasters are
written in the *metric* CRS of that city box (EPSG:32644, UTM 44N, matching
``config/cities/vizag.yaml``) covering the hexes' bounding box with a margin.

Two rasters are deliberately *analytic* rather than random, so a test can assert a
hand-computed answer rather than a golden value:

``synthetic_dem``          a planar tilt of a known gradient, so the true ``slope_pct`` is
                           known in closed form (``100 * hypot(gx, gy)``).
``synthetic_landcover``    a raster of horizontal bands with known pixel counts, so a
                           cell's class fractions are computable by hand.

Every constant lives in :data:`FIXTURE_CONFIG` (``raster_fixtures.yaml``), so this module
holds no bare numeric literals either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from ufe import geo
from ufe.ingest.core import InMemoryReader
from tests.fixtures.synthetic import synthetic_cells

__all__ = [
    "FIXTURE_CONFIG",
    "METRIC_CRS",
    "RasterGrid",
    "cells_frame",
    "hex_gdf",
    "metric_bounds",
    "raster_grid",
    "write_raster",
    "synthetic_dem",
    "synthetic_landcover",
    "synthetic_height_raster",
    "synthetic_population_raster",
    "synthetic_nightlight_months",
    "synthetic_footprints",
    "synthetic_footprints_by_vintage",
    "synthetic_wards",
    "synthetic_forest",
    "synthetic_substations",
    "synthetic_power_lines",
    "synthetic_roads",
    "synthetic_coastline",
    "synthetic_pois",
    "synthetic_parcels",
    "synthetic_zoning",
    "synthetic_czmp",
    "synthetic_utilities",
    "synthetic_guidance_values",
    "synthetic_listing_localities",
    "synthetic_listing_points",
    "synthetic_broker_panel",
    "synthetic_rera_projects",
    "build_reader",
]

_CONFIG_PATH = Path(__file__).with_name("raster_fixtures.yaml")
FIXTURE_CONFIG: Mapping[str, Any] = yaml.safe_load(_CONFIG_PATH.read_text())

#: The metric CRS of the fixture city, matching ``config/cities/vizag.yaml``.
METRIC_CRS = str(FIXTURE_CONFIG["crs_metric"])

_ZERO, _ONE = 0, 1


def cells_frame(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """The shared synthetic ``cells`` frame, subset to ``n`` rows."""
    count = int(FIXTURE_CONFIG["n_cells"]) if n is None else int(n)
    return synthetic_cells(n=count, seed=seed).reset_index(drop=True)


def hex_gdf(cells: pd.DataFrame, *, crs: str | None = None) -> gpd.GeoDataFrame:
    """Cell polygons as a GeoDataFrame, optionally reprojected into ``crs``."""
    from ufe.ingest.core import cells_gdf

    gdf = cells_gdf(cells)[["h3", "geometry"]]
    return gdf if crs is None else geo.to_metric(gdf, crs)


def metric_bounds(cells: pd.DataFrame, *, crs: str = METRIC_CRS) -> tuple[float, ...]:
    """The cells' bounding box in ``crs``, expanded by the configured margin."""
    margin = float(FIXTURE_CONFIG["raster"]["margin_m"])
    minx, miny, maxx, maxy = hex_gdf(cells, crs=crs).total_bounds
    return (minx - margin, miny - margin, maxx + margin, maxy + margin)


@dataclass(frozen=True)
class RasterGrid:
    """A raster's geometry: the affine transform, shape and CRS."""

    transform: Any
    width: int
    height: int
    crs: str

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates as ``(x, y)`` meshgrids."""
        cols = np.arange(self.width) + float(FIXTURE_CONFIG["raster"]["pixel_centre_offset"])
        rows = np.arange(self.height) + float(FIXTURE_CONFIG["raster"]["pixel_centre_offset"])
        xs = self.transform.c + cols * self.transform.a
        ys = self.transform.f + rows * self.transform.e
        return np.meshgrid(xs, ys)


def raster_grid(
    cells: pd.DataFrame, *, pixel_m: float | None = None, crs: str = METRIC_CRS
) -> RasterGrid:
    """A raster grid covering ``cells`` at ``pixel_m`` resolution in ``crs``."""
    size = float(FIXTURE_CONFIG["raster"]["pixel_m"]) if pixel_m is None else float(pixel_m)
    minx, miny, maxx, maxy = metric_bounds(cells, crs=crs)
    width = int(np.ceil((maxx - minx) / size))
    height = int(np.ceil((maxy - miny) / size))
    return RasterGrid(from_origin(minx, maxy, size, size), width, height, crs)


def write_raster(
    path: str | Path, data: np.ndarray, grid: RasterGrid, *, nodata: float | None = None
) -> Path:
    """Write a single-band GeoTIFF. The only place these fixtures touch the filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": _ONE,
        "dtype": str(data.dtype),
        "crs": grid.crs,
        "transform": grid.transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, _ONE)
    return path


# --------------------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------------------


def synthetic_dem(
    path: str | Path, cells: pd.DataFrame, *, gradient: tuple[float, float] | None = None
) -> tuple[Path, float]:
    """A DEM that is an exact plane, so the true slope is known in closed form.

    ``elev = base + gx * (x - x0) + gy * (y - y0)`` in metres, hence
    ``slope_pct = 100 * hypot(gx, gy)`` **everywhere**. Returns ``(path, expected_slope_pct)``
    so a test can assert against the analytic answer rather than a golden file.
    """
    spec = FIXTURE_CONFIG["dem"]
    gx, gy = gradient if gradient is not None else (
        float(spec["gradient_x"]),
        float(spec["gradient_y"]),
    )
    grid = raster_grid(cells, pixel_m=float(spec["pixel_m"]))
    xs, ys = grid.xy()
    elevation = float(spec["base_elev_m"]) + gx * (xs - xs.min()) + gy * (ys - ys.min())
    write_raster(path, elevation.astype("float32"), grid, nodata=float(spec["nodata"]))
    return Path(path), float(FIXTURE_CONFIG["slope_percent_scale"]) * float(np.hypot(gx, gy))


def synthetic_landcover(
    path: str | Path, cells: pd.DataFrame, *, codes: Sequence[int] | None = None
) -> tuple[Path, dict[int, float]]:
    """A categorical raster of equal horizontal bands with known area shares.

    Returns ``(path, {code: share_of_raster})``. Because the bands are exact row blocks, a
    cell's class fractions are computable by hand from its row extent.
    """
    spec = FIXTURE_CONFIG["landcover"]
    class_codes = [int(c) for c in (codes if codes is not None else spec["codes"])]
    grid = raster_grid(cells, pixel_m=float(spec["pixel_m"]))
    band_height = max(_ONE, grid.height // len(class_codes))
    data = np.empty((grid.height, grid.width), dtype="int32")
    for i in range(grid.height):
        data[i, :] = class_codes[min(i // band_height, len(class_codes) - _ONE)]
    write_raster(path, data, grid, nodata=int(spec["nodata"]))
    shares = {
        code: float((data == code).sum()) / float(data.size) for code in class_codes
    }
    return Path(path), shares


def synthetic_height_raster(path: str | Path, cells: pd.DataFrame) -> Path:
    """A constant building-height raster, so ``storeys`` is exactly computable."""
    spec = FIXTURE_CONFIG["height"]
    grid = raster_grid(cells, pixel_m=float(spec["pixel_m"]))
    data = np.full((grid.height, grid.width), float(spec["height_m"]), dtype="float32")
    return write_raster(path, data, grid, nodata=float(spec["nodata"]))


def synthetic_population_raster(path: str | Path, cells: pd.DataFrame) -> tuple[Path, float]:
    """A constant population-count raster. Returns ``(path, persons_per_pixel)``."""
    spec = FIXTURE_CONFIG["population_raster"]
    grid = raster_grid(cells, pixel_m=float(spec["pixel_m"]))
    per_pixel = float(spec["persons_per_pixel"])
    data = np.full((grid.height, grid.width), per_pixel, dtype="float32")
    write_raster(path, data, grid, nodata=float(spec["nodata"]))
    return Path(path), per_pixel


def synthetic_nightlight_months(
    directory: str | Path, cells: pd.DataFrame
) -> dict[tuple[int, int], Path]:
    """Monthly VIIRS-like composites with a deliberate negative value and a deliberate spike.

    Each month is constant except for one hot pixel and one negative pixel, so the
    outlier mask and the annual median have hand-checkable effects.
    """
    spec = FIXTURE_CONFIG["nightlights"]
    directory = Path(directory)
    grid = raster_grid(cells, pixel_m=float(spec["pixel_m"]))
    out: dict[tuple[int, int], Path] = {}
    for offset, radiance in enumerate(spec["monthly_radiance"]):
        year = int(spec["year"])
        month = offset + _ONE
        data = np.full((grid.height, grid.width), float(radiance), dtype="float32")
        data[_ZERO, _ZERO] = float(spec["outlier_radiance"])
        data[_ZERO, _ONE] = float(spec["negative_radiance"])
        out[(year, month)] = write_raster(
            directory / f"viirs_{year}_{month:02d}.tif",
            data,
            grid,
            nodata=float(spec["nodata"]),
        )
    return out


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------


def _hex_centroids(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    hexes = hex_gdf(cells, crs=METRIC_CRS)
    return gpd.GeoDataFrame(
        {"h3": hexes["h3"].to_numpy()}, geometry=hexes.geometry.centroid, crs=METRIC_CRS
    )


def synthetic_footprints(
    cells: pd.DataFrame, *, side_m: float | None = None, every: int | None = None
) -> gpd.GeoDataFrame:
    """One square footprint at the centroid of every ``every``-th cell.

    A square of side ``side_m`` has area ``side_m ** 2`` exactly, so the expected
    ``builtup_frac`` of a covered cell is ``side_m ** 2 / area_sqm`` — a hand-computed
    answer for the Section 6 ACCEPTANCE range check.
    """
    spec = FIXTURE_CONFIG["footprints"]
    side = float(spec["side_m"]) if side_m is None else float(side_m)
    step = int(spec["every_nth_cell"]) if every is None else int(every)
    centroids = _hex_centroids(cells).iloc[::step]
    half = side / (_ONE + _ONE)
    squares = [
        box(point.x - half, point.y - half, point.x + half, point.y + half)
        for point in centroids.geometry
    ]
    return gpd.GeoDataFrame(
        {"h3": centroids["h3"].to_numpy(), "footprint_sqm": side * side},
        geometry=squares,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_footprints_by_vintage(
    cells: pd.DataFrame, years: Iterable[int] | None = None
) -> dict[int, gpd.GeoDataFrame]:
    """A growing footprint panel: each vintage covers strictly more cells than the last."""
    spec = FIXTURE_CONFIG["footprints"]
    vintages = list(years if years is not None else spec["vintages"])
    steps = sorted(int(spec["every_nth_cell"]) * (len(vintages) - i) for i in range(len(vintages)))
    return {
        int(year): synthetic_footprints(cells, every=max(_ONE, step))
        for year, step in zip(vintages, sorted(steps, reverse=True))
    }


def synthetic_wards(cells: pd.DataFrame, *, n: int | None = None) -> gpd.GeoDataFrame:
    """Census-style ward polygons tiling the cell box, each with a known population total."""
    spec = FIXTURE_CONFIG["wards"]
    count = int(spec["count"]) if n is None else int(n)
    minx, miny, maxx, maxy = metric_bounds(cells)
    step = (maxx - minx) / count
    polygons = [
        box(minx + i * step, miny, minx + (i + _ONE) * step, maxy) for i in range(count)
    ]
    return gpd.GeoDataFrame(
        {
            "ward_id": [f"ward-{i}" for i in range(count)],
            "population": [float(p) for p in spec["populations"][:count]],
        },
        geometry=polygons,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_forest(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """A declared forest boundary covering the configured fraction of the cell box."""
    share = float(FIXTURE_CONFIG["forest"]["x_share"])
    minx, miny, maxx, maxy = metric_bounds(cells)
    return gpd.GeoDataFrame(
        {"name": ["Reserved Forest"]},
        geometry=[box(minx, miny, minx + (maxx - minx) * share, maxy)],
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_substations(cells: pd.DataFrame, *, n: int | None = None) -> gpd.GeoDataFrame:
    """``power=substation`` points at the centroids of the first ``n`` cells."""
    count = int(FIXTURE_CONFIG["power"]["n_substations"]) if n is None else int(n)
    centroids = _hex_centroids(cells).iloc[:count]
    return gpd.GeoDataFrame(
        {"power": ["substation"] * len(centroids)},
        geometry=centroids.geometry.to_numpy(),
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_power_lines(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """One 11 kV line and one out-of-scope line, so the voltage filter is exercised."""
    spec = FIXTURE_CONFIG["power"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    mid = (miny + maxy) / (_ONE + _ONE)
    return gpd.GeoDataFrame(
        {"power": ["line", "line"], "voltage": [spec["in_scope_voltage"], spec["out_of_scope_voltage"]]},
        geometry=[
            LineString([(minx, mid), (maxx, mid)]),
            LineString([(minx, miny), (maxx, miny)]),
        ],
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_roads(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """One arterial (``highway=primary``) and one residential street."""
    minx, miny, maxx, maxy = metric_bounds(cells)
    mid = (minx + maxx) / (_ONE + _ONE)
    return gpd.GeoDataFrame(
        {"highway": [str(FIXTURE_CONFIG["roads"]["arterial_value"]), "residential"]},
        geometry=[
            LineString([(mid, miny), (mid, maxy)]),
            LineString([(minx, maxy), (maxx, maxy)]),
        ],
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_coastline(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """An ``natural=coastline`` way along the eastern edge of the cell box."""
    minx, miny, maxx, maxy = metric_bounds(cells)
    return gpd.GeoDataFrame(
        {"natural": ["coastline"]},
        geometry=[LineString([(maxx, miny), (maxx, maxy)])],
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_pois(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """POI points inside known cells, one per configured tag value."""
    spec = FIXTURE_CONFIG["pois"]
    centroids = _hex_centroids(cells)
    rows: list[dict[str, Any]] = []
    geometries = []
    for offset, tag in enumerate(spec["tag_values"]):
        row = centroids.iloc[offset % len(centroids)]
        for _ in range(int(spec["per_tag"])):
            rows.append({"tag_value": str(tag), "h3": row["h3"]})
            geometries.append(row.geometry)
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=METRIC_CRS).to_crs(
        geo.GEOGRAPHIC_CRS
    )


def synthetic_parcels(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """Survey-number parcels of two known sizes inside the first cells.

    Two sizes rather than one so the Gini coefficient is non-zero and testable.
    """
    spec = FIXTURE_CONFIG["parcels"]
    centroids = _hex_centroids(cells).iloc[: int(spec["n_cells"])]
    geometries, records = [], []
    for index, row in enumerate(centroids.itertuples()):
        for order, side in enumerate(spec["side_m"]):
            half = float(side) / (_ONE + _ONE)
            offset = float(spec["offset_m"]) * order
            geometries.append(
                box(
                    row.geometry.x + offset - half,
                    row.geometry.y - half,
                    row.geometry.x + offset + half,
                    row.geometry.y + half,
                )
            )
            records.append(
                {"survey_no": f"{index}/{order}", "expected_sqm": float(side) ** (_ONE + _ONE)}
            )
    return gpd.GeoDataFrame(records, geometry=geometries, crs=METRIC_CRS).to_crs(
        geo.GEOGRAPHIC_CRS
    )


def synthetic_zoning(cells: pd.DataFrame, *, cover_all: bool = True) -> gpd.GeoDataFrame:
    """Master-plan zoning polygons with every Section 6.10 required attribute."""
    spec = FIXTURE_CONFIG["zoning"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    zones = list(spec["zones"])
    fars = [float(f) for f in spec["permitted_far"]]
    limit = maxx if cover_all else minx + (maxx - minx) * float(spec["partial_x_share"])
    step = (limit - minx) / len(zones)
    polygons = [
        box(minx + i * step, miny, minx + (i + _ONE) * step, maxy) for i in range(len(zones))
    ]
    return gpd.GeoDataFrame(
        {
            "zone_class": zones,
            "permitted_far": fars,
            "plan_name": [str(spec["plan_name"])] * len(zones),
            "plan_year": [int(spec["plan_year"])] * len(zones),
            "source_sheet": [f"sheet-{i}" for i in range(len(zones))],
        },
        geometry=polygons,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_czmp(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """CZMP polygons carrying ``crz_class``, along the eastern (seaward) side."""
    spec = FIXTURE_CONFIG["czmp"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    classes = list(spec["classes"])
    step = (maxx - minx) * float(spec["x_share"]) / len(classes)
    start = maxx - (maxx - minx) * float(spec["x_share"])
    polygons = [
        box(start + i * step, miny, start + (i + _ONE) * step, maxy)
        for i in range(len(classes))
    ]
    return gpd.GeoDataFrame(
        {"crz_class": classes, "plan_name": [str(spec["plan_name"])] * len(classes)},
        geometry=polygons,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_utilities(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """Municipal water/sewer service polygons over the western part of the box."""
    spec = FIXTURE_CONFIG["utilities"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    water_x = minx + (maxx - minx) * float(spec["water_x_share"])
    sewer_x = minx + (maxx - minx) * float(spec["sewer_x_share"])
    return gpd.GeoDataFrame(
        {"water_served": [_ONE, _ONE], "sewer_served": [_ONE, _ZERO]},
        geometry=[box(minx, miny, sewer_x, maxy), box(sewer_x, miny, water_x, maxy)],
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


# --------------------------------------------------------------------------------------
# Tabular / state-tier fixtures
# --------------------------------------------------------------------------------------


def synthetic_guidance_values(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """SRO locality polygons with guidance values in INR per **square yard**.

    Per square yard deliberately: the AP adapter is responsible for normalising to the
    engine's INR/sqft, and a fixture in the target unit would not test that.
    """
    spec = FIXTURE_CONFIG["guidance"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    values = [float(v) for v in spec["inr_per_sqyd"]]
    step = (maxx - minx) / len(values)
    polygons = [
        box(minx + i * step, miny, minx + (i + _ONE) * step, maxy) for i in range(len(values))
    ]
    return gpd.GeoDataFrame(
        {
            "locality_id": [f"loc-{i}" for i in range(len(values))],
            "sro_code": [f"SRO-{i}" for i in range(len(values))],
            "locality_name": [f"Locality {i}" for i in range(len(values))],
            "guidance_inr_sqyd": values,
            "effective_year": [int(spec["effective_year"])] * len(values),
        },
        geometry=polygons,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_listing_localities(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """Listing-portal locality polygons carrying an asking price and a rent."""
    spec = FIXTURE_CONFIG["listings"]
    minx, miny, maxx, maxy = metric_bounds(cells)
    asks = [float(v) for v in spec["ask_inr_sqft"]]
    rents = [float(v) for v in spec["rent_inr_sqft_mo"]]
    share = float(spec["x_share"])
    limit = minx + (maxx - minx) * share
    step = (limit - minx) / len(asks)
    polygons = [
        box(minx + i * step, miny, minx + (i + _ONE) * step, maxy) for i in range(len(asks))
    ]
    return gpd.GeoDataFrame(
        {"ask_inr_sqft": asks, "rent_inr_sqft_mo": rents},
        geometry=polygons,
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_listing_points(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """Point listings, for the 500 m Gaussian smear path."""
    spec = FIXTURE_CONFIG["listings"]
    centroids = _hex_centroids(cells).iloc[:: int(spec["point_every_nth_cell"])]
    return gpd.GeoDataFrame(
        {
            "ask_inr_sqft": [float(spec["point_ask_inr_sqft"])] * len(centroids),
            "rent_inr_sqft_mo": [float(spec["point_rent_inr_sqft_mo"])] * len(centroids),
        },
        geometry=centroids.geometry.to_numpy(),
        crs=METRIC_CRS,
    ).to_crs(geo.GEOGRAPHIC_CRS)


def synthetic_broker_panel(cells: pd.DataFrame, *, n: int | None = None) -> pd.DataFrame:
    """A broker panel in the Section 6.7c schema, sited on known cells."""
    spec = FIXTURE_CONFIG["broker"]
    count = int(spec["n"]) if n is None else int(n)
    sample = cells.iloc[:count]
    return pd.DataFrame(
        {
            "date": [str(spec["date"])] * len(sample),
            "lat": sample["lat"].to_numpy(dtype=float),
            "lon": sample["lon"].to_numpy(dtype=float),
            "area_sqft": np.full(len(sample), float(spec["area_sqft"])),
            "total_price_inr": np.full(len(sample), float(spec["area_sqft"]))
            * float(spec["inr_sqft"]),
            "property_type": [str(spec["property_type"])] * len(sample),
            "transaction_type": [str(spec["transaction_type"])] * len(sample),
        }
    )


def synthetic_rera_projects(cells: pd.DataFrame, *, n: int | None = None) -> pd.DataFrame:
    """A RERA portal extract in the shape the AP adapter normalises to."""
    spec = FIXTURE_CONFIG["rera"]
    count = int(spec["n"]) if n is None else int(n)
    sample = cells.iloc[:count]
    promoters = list(spec["promoters"])
    return pd.DataFrame(
        {
            "rera_id": [f"AP/RERA/{i:04d}" for i in range(len(sample))],
            "project_name": [f"Synthetic Tower {i}" for i in range(len(sample))],
            "promoter": [promoters[i % len(promoters)] for i in range(len(sample))],
            "lat": sample["lat"].to_numpy(dtype=float),
            "lon": sample["lon"].to_numpy(dtype=float),
            "total_units": np.full(len(sample), float(spec["total_units"])),
            "declared_start": [str(spec["declared_start"])] * len(sample),
            "declared_completion": [str(spec["declared_completion"])] * len(sample),
            "progress_pct": np.full(len(sample), float(spec["progress_pct"])),
            "quarter": [str(spec["quarter"])] * len(sample),
            "units_booked": np.full(len(sample), float(spec["units_booked"])),
        }
    )


def synthetic_project_registry(
    cells: pd.DataFrame, params: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A Section 6.11-valid project registry and its human-signed history.

    Built from the shared synthetic ``projects`` frame, then corrected so it *passes* every
    fatal rule: a real archetype from ``archetypes.yaml`` with its declared ``scale_unit``,
    geometry sited on a real cell, and the initial stage (which needs no sign-off). The
    negative cases are constructed per-test by breaking one rule at a time.
    """
    from tests.fixtures.synthetic import synthetic_projects
    from ufe.ingest.projects import archetype_units
    from ufe.params import load_params

    spec = FIXTURE_CONFIG["projects"]
    n = int(spec["n"])
    params = load_params(str(spec["city"])) if params is None else params
    units = archetype_units(params)
    archetype = sorted(units)[_ZERO]

    frame = synthetic_projects(n=n, cells=cells).copy()
    frame["archetype"] = archetype
    frame["scale_unit"] = units[archetype]
    frame["stage"] = str(spec["stage"])
    frame["geom"] = [
        Point(float(cells["lon"].iat[i]), float(cells["lat"].iat[i])).wkt for i in range(n)
    ]
    history = pd.DataFrame(
        {
            "project_id": frame["project_id"],
            "field": "stage",
            "old_value": None,
            "new_value": frame["stage"],
            "changed_at": pd.Timestamp(str(spec["changed_at"])),
            "source_url": [urls[_ZERO] for urls in frame["source_urls"]],
            "changed_by": str(spec["changed_by"]),
        }
    )
    return frame, history


# --------------------------------------------------------------------------------------
# One reader holding everything
# --------------------------------------------------------------------------------------


def build_reader(
    tmp_path: str | Path, cells: pd.DataFrame | None = None, *, coastal: bool = True
) -> tuple[InMemoryReader, pd.DataFrame, dict[str, Any]]:
    """An :class:`InMemoryReader` wired with every synthetic layer.

    Returns ``(reader, cells, expectations)`` where ``expectations`` carries the analytic
    answers a test asserts against (the plane's slope, the landcover shares, the
    persons-per-pixel of the population raster, the footprint side).

    ``coastal=False`` omits the CZMP layer, which is how the
    :class:`ufe.errors.MissingCriticalLayer` test is set up.
    """
    from ufe.ingest.adapters import ap
    from ufe.ingest.buildings import KEY_FOOTPRINTS, KEY_HEIGHT
    from ufe.ingest.landcover import KEY_FOREST, KEY_LANDCOVER
    from ufe.ingest.nightlights import monthly_key
    from ufe.ingest.osm import (
        KEY_COASTLINE,
        KEY_POIS,
        KEY_POWER_LINES,
        KEY_ROADS,
        KEY_SUBSTATIONS,
    )
    from ufe.ingest.population import KEY_CENSUS_WARDS, KEY_WORLDPOP
    from ufe.ingest.prices import KEY_BROKER_PANEL, KEY_LISTING_POINTS, KEY_LISTINGS
    from ufe.ingest.projects import KEY_REGISTRY
    from ufe.ingest.terrain import KEY_DEM
    from ufe.ingest.zoning import KEY_CZMP, KEY_UTILITIES, KEY_ZONING

    root = Path(tmp_path)
    cells = cells_frame() if cells is None else cells

    dem_path, expected_slope = synthetic_dem(root / "dem.tif", cells)
    lc_path, lc_shares = synthetic_landcover(root / "worldcover.tif", cells)
    height_path = synthetic_height_raster(root / "height.tif", cells)
    pop_path, per_pixel = synthetic_population_raster(root / "worldpop.tif", cells)
    months = synthetic_nightlight_months(root / "viirs", cells)
    registry, registry_history = synthetic_project_registry(cells)

    reader = InMemoryReader(
        paths={
            KEY_DEM: dem_path,
            KEY_LANDCOVER: lc_path,
            KEY_HEIGHT: height_path,
            KEY_WORLDPOP: pop_path,
            **{monthly_key(y, m): p for (y, m), p in months.items()},
        },
        vectors={
            KEY_FOREST: synthetic_forest(cells),
            KEY_FOOTPRINTS: synthetic_footprints(cells),
            KEY_CENSUS_WARDS: synthetic_wards(cells),
            KEY_SUBSTATIONS: synthetic_substations(cells),
            KEY_POWER_LINES: synthetic_power_lines(cells),
            KEY_ROADS: synthetic_roads(cells),
            KEY_COASTLINE: synthetic_coastline(cells),
            KEY_POIS: synthetic_pois(cells),
            KEY_ZONING: synthetic_zoning(cells),
            KEY_UTILITIES: synthetic_utilities(cells),
            KEY_LISTINGS: synthetic_listing_localities(cells),
            KEY_LISTING_POINTS: synthetic_listing_points(cells),
            ap.KEY_GUIDANCE: synthetic_guidance_values(cells),
            ap.KEY_PARCELS: synthetic_parcels(cells),
        },
        tables={
            KEY_BROKER_PANEL: synthetic_broker_panel(cells),
            ap.KEY_RERA: synthetic_rera_projects(cells),
            KEY_REGISTRY: registry,
            "project_history": registry_history,
        },
    )
    if coastal:
        reader.add_vector(KEY_CZMP, synthetic_czmp(cells))

    expectations = {
        "slope_pct": expected_slope,
        "landcover_shares": lc_shares,
        "persons_per_pixel": per_pixel,
        "footprint_side_m": float(FIXTURE_CONFIG["footprints"]["side_m"]),
        "footprint_every_nth": int(FIXTURE_CONFIG["footprints"]["every_nth_cell"]),
        "nightlight_months": months,
        "crs_metric": METRIC_CRS,
    }
    return reader, cells, expectations
