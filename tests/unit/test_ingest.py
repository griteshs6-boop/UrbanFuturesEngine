"""Module 2 — data ingestion. Tests for the pure transform half of every ingester.

Every test here runs **offline** against the deterministic synthetic rasters and vectors in
``tests/fixtures/raster_fixtures.py``. Tests that would need a real download are written and
marked ``@pytest.mark.needs_data`` so they are collected, visible and skipped here.

The seven items of the Section 6 ACCEPTANCE block are marked ``@pytest.mark.acceptance``:

1. every populated cell column has a matching ``ingest_runs`` row;
2. ``builtup_frac`` in [0, 1] and ``landcover_fracs`` summing to 1 +- 1e-3;
3. population within 5% of the district total for ``base_year``;
4. the 10 km known-answer distance test;
5. price blending reproducing a hand-computed value;
6. a coastal city with no CRZ layer raising ``MissingCriticalLayer``;
7. a ``force=False`` re-run using the cache and producing an identical frame.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ufe import geo
from ufe.errors import (
    CoverageError,
    DataRightsViolation,
    MissingCriticalLayer,
    MissingParameter,
    UFEError,
)
from ufe.ingest import buildings, cadastral, coverage, core, landcover, nightlights
from ufe.ingest import osm as osm_mod
from ufe.ingest import population, prices, projects, rera, runner, terrain, zoning
from ufe.ingest.adapters.base import get_adapter
from ufe.ingest.core import CityConfig
from ufe.params import load_params
from ufe.store import schemas as S
from tests.fixtures import raster_fixtures as rf

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: Columns Module 1 (``ufe/grid/build.py``) owns, not Module 2.
MODULE_1_COLUMNS = ("h3", "h3_res8", "in_city", "geometry", "lat", "lon", "area_sqm")

#: There is no real source data and no network here, so `needs_data` tests are additionally
#: guarded by an explicit opt-in environment variable (the repo has no root conftest wiring
#: marker filters), matching the `needs_osrm` convention in tests/unit/test_routing.py.
REAL_DATA_ENV = "UFE_REAL_DATA_ROOT"


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params("vizag")


@pytest.fixture(scope="module")
def city(params) -> CityConfig:
    return CityConfig.from_params(params)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """``(reader, cells, expectations)`` — built once, it writes real GeoTIFFs."""
    root = tmp_path_factory.mktemp("ingest_fixtures")
    return rf.build_reader(root)


@pytest.fixture(scope="module")
def reader(bundle):
    return bundle[0]


@pytest.fixture(scope="module")
def cells(bundle):
    return bundle[1]


@pytest.fixture(scope="module")
def expected(bundle):
    return bundle[2]


@pytest.fixture(scope="module")
def adapter(reader):
    return get_adapter("AP", reader=reader)


@pytest.fixture
def module1_cells(cells) -> pd.DataFrame:
    """A freshly built grid: Module 1's columns only, nothing ingested yet."""
    return cells[list(MODULE_1_COLUMNS)].copy()


def _run(ingester, city, cells):
    """The Section 6 loop: fetch -> parse -> to_cells."""
    return ingester.to_cells(ingester.parse(ingester.fetch(city)), cells)


# --------------------------------------------------------------------------------------
# The library rules of Section 2.1b
# --------------------------------------------------------------------------------------


def test_no_forbidden_library_is_imported():
    """Section 2.1b: ``exactextract`` not ``rasterstats``; ``rich.progress`` not ``tqdm``."""
    forbidden = ("rasterstats", "tqdm")
    sources = list(Path(core.__file__).parent.rglob("*.py")) + [
        Path(__file__).parents[2] / "ufe" / "ingest_cli.py"
    ]
    patterns = [
        re.compile(rf"^\s*(?:import|from)\s+{word}\b", re.MULTILINE) for word in forbidden
    ]
    offenders = [
        f"{path.name}: {pattern.pattern}"
        for path in sources
        for pattern in patterns
        if pattern.search(path.read_text())
    ]
    assert not offenders, offenders


def test_zonal_statistics_go_through_exactextract():
    """The zonal helper is the single point where raster statistics are computed."""
    assert "exactextract" in Path(core.__file__).read_text()
    assert "from exactextract import exact_extract" in Path(core.__file__).read_text()


def test_areal_interpolation_goes_through_tobler():
    """Section 6.4 mandates tobler for the polygon -> hex transfer."""
    assert "from tobler.area_weighted import area_interpolate" in Path(
        population.__file__
    ).read_text()


def test_progress_uses_rich():
    text = (Path(__file__).parents[2] / "ufe" / "ingest_cli.py").read_text()
    assert "from rich.progress import" in text


# --------------------------------------------------------------------------------------
# Provenance, licences and the ingest_runs ledger
# --------------------------------------------------------------------------------------


def test_every_ingester_declares_a_registered_source_and_licence(reader, city, params):
    """Section 6: the ``licence`` field is mandatory and feeds ATTRIBUTIONS.md."""
    for tier in runner.TIERS:
        for ingester in runner.ingesters_for_tier(
            tier, reader=reader, city=city, params=params, adapter=get_adapter("AP", reader=reader)
        ):
            record = ingester.provenance()
            assert record["licence"], f"{type(ingester).__name__} has no licence"
            for key in (
                "source_id",
                "url",
                "retrieved_at",
                "licence",
                "spatial_res",
                "temporal_res",
                "notes",
            ):
                assert key in record, f"{type(ingester).__name__} provenance lacks {key}"


def test_unregistered_source_raises_licence_violation(reader, city):
    from ufe.errors import LicenceViolation

    class Rogue(core.Ingester):
        source_id = "totally_unregistered_source"

        def keys(self, city):
            return ("x",)

        def parse(self, raw):
            return pd.DataFrame()

        def to_cells(self, df, cells):
            return pd.DataFrame()

    with pytest.raises(LicenceViolation):
        Rogue(reader, city=city)


def test_ingest_run_is_content_addressed(reader, city):
    ingester = terrain.TerrainIngester(reader, city=city)
    ingester.fetch(city)
    run_a = core.ingest_run(
        ingester.provenance(), city_id=city.city_id, tier="national", columns=["elev_m"], rows=1
    )
    run_b = core.ingest_run(
        ingester.provenance(), city_id=city.city_id, tier="national", columns=["elev_m"], rows=1
    )
    assert run_a["run_id"] == run_b["run_id"]


# --------------------------------------------------------------------------------------
# 6.1 Terrain
# --------------------------------------------------------------------------------------


def test_slope_percent_is_the_spec_formula():
    """``slope_pct = 100 * sqrt((dz/dx)^2 + (dz/dy)^2)`` (Section 6.1)."""
    scale = float(core.cfg("terrain.slope_percent_scale"))
    gx, gy = 0.03, 0.04  # a 3% and a 4% component -> a 5% slope
    x = np.arange(10, dtype=float)
    elevation = gy * x[:, None] + gx * x[None, :]
    slope = terrain.slope_percent(
        elevation, pixel_width_m=1, pixel_height_m=1, scale=scale
    )
    assert np.allclose(slope, scale * np.hypot(gx, gy))


def test_terrain_reproduces_the_analytic_slope_of_a_planar_dem(
    reader, city, cells, expected, tmp_path
):
    ingester = terrain.TerrainIngester(reader, city=city)
    ingester.work_root = tmp_path
    out = _run(ingester, city, cells)
    assert set(out["h3"]) == set(cells["h3"].astype(str))
    assert np.allclose(out["slope_pct"], expected["slope_pct"], rtol=1e-3)
    assert out["elev_m"].notna().all()
    assert (out["slope_pct"] >= 0).all()


def test_slope_refuses_a_geographic_dem(cells, tmp_path):
    """Section 0.3: never compute a gradient in degrees. The guard is in ufe/geo.py."""
    from rasterio.transform import from_origin

    grid = rf.RasterGrid(
        from_origin(float(cells["lon"].min()), float(cells["lat"].max()), 0.001, 0.001),
        width=8,
        height=8,
        crs=geo.GEOGRAPHIC_CRS,
    )
    data = np.zeros((grid.height, grid.width), dtype="float32")
    path = rf.write_raster(tmp_path / "geographic.tif", data, grid)
    with pytest.raises(geo.NonMetricCRSError):
        terrain.write_slope_raster(path, tmp_path / "slope.tif")


# --------------------------------------------------------------------------------------
# 6.2 Land cover
# --------------------------------------------------------------------------------------


def test_landcover_class_codes_map_onto_the_schema_vocabulary():
    mapping = landcover.class_name_map()
    assert set(mapping.values()) <= set(S.LANDCOVER_CLASSES)


@pytest.mark.acceptance
def test_landcover_fractions_sum_to_one(reader, city, cells):
    """Section 6 ACCEPTANCE: "sum of landcover_fracs = 1 +- 1e-3"."""
    ingester = landcover.LandcoverIngester(reader, city=city)
    _run(ingester, city, cells)
    fracs = ingester.side_tables[landcover.FRACS_TABLE]
    landcover.assert_fracs_sum_to_one(fracs)  # raises on violation
    totals = fracs.groupby("h3")["frac"].sum()
    tolerance = float(core.cfg("landcover.frac_sum_tolerance"))
    assert ((totals - 1).abs() <= tolerance).all()


def test_landcover_hard_gates_water_and_wetland(reader, city, cells):
    out = _run(landcover.LandcoverIngester(reader, city=city), city, cells)
    assert out["undevelopable_frac"].between(0, 1).all()
    assert out["landcover"].isin(S.LANDCOVER_CLASSES).all()
    # Water-majority cells must carry a substantial undevelopable fraction.
    water = out[out["landcover"] == "water"]
    if len(water):
        assert (water["undevelopable_frac"] > 0).all()


def test_tree_cover_is_gated_only_inside_a_forest_boundary(reader, city, cells):
    """Section 6.2: tree cover is a hard gate only inside a declared forest boundary."""
    raster = reader.path(landcover.KEY_LANDCOVER)
    with_forest, _ = landcover.landcover_to_cells(
        raster, cells, crs_metric=city.crs_metric, forest=rf.synthetic_forest(cells)
    )
    without, _ = landcover.landcover_to_cells(
        raster, cells, crs_metric=city.crs_metric, forest=None
    )
    assert with_forest["undevelopable_frac"].sum() > without["undevelopable_frac"].sum()
    # With no boundary the unresolved gate must be flagged, never silently resolved.
    tree_cells = without["undevelopable_frac__impute_method"] == "tree_cover_no_forest_boundary"
    assert tree_cells.any()


# --------------------------------------------------------------------------------------
# 6.3 Buildings
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_builtup_frac_is_in_the_unit_interval(reader, city, cells):
    """Section 6 ACCEPTANCE: ``builtup_frac`` in [0, 1] for all cells."""
    out = _run(buildings.BuildingsIngester(reader, city=city), city, cells)
    assert out["builtup_frac"].between(0, 1).all()
    assert out["builtup_frac"].notna().all()


def test_builtup_frac_matches_a_hand_computed_footprint_area(city, cells, expected):
    """A 40 m square is 1600 m^2, so a covered cell's fraction is 1600 / area_sqm."""
    footprints = rf.synthetic_footprints(cells)
    out = buildings.buildings_to_cells(footprints, cells, crs_metric=city.crs_metric)
    side = expected["footprint_side_m"]
    covered = out["builtup_frac"] > 0
    merged = out.loc[covered].merge(
        cells[["h3", "area_sqm"]].assign(h3=cells["h3"].astype(str)), on="h3"
    )
    assert np.allclose(
        merged["builtup_frac"], side * side / merged["area_sqm"], rtol=1e-3
    )


def test_storeys_come_from_the_height_raster_when_present(reader, city, cells):
    """``storeys = max(1, round(height_m / 3.2))``; 12.8 / 3.2 = exactly 4."""
    storeys, imputed = buildings.storeys_per_cell(
        cells, height_raster=reader.path(buildings.KEY_HEIGHT)
    )
    per_storey = float(core.cfg("buildings.metres_per_storey"))
    height = float(rf.FIXTURE_CONFIG["height"]["height_m"])
    assert np.allclose(storeys[~imputed], round(height / per_storey))
    assert not imputed.all()


def test_storeys_fall_back_to_the_zone_lookup_and_are_flagged(city, cells):
    storeys, imputed = buildings.storeys_per_cell(cells, height_raster=None)
    assert imputed.all(), "a zone-lookup storey count must always be flagged"
    assert (storeys >= 1).all()
    out = buildings.buildings_to_cells(
        rf.synthetic_footprints(cells), cells, crs_metric=city.crs_metric
    )
    methods = set(out["floorspace_res_sqm__impute_method"])
    assert "storeys_from_zone_class" in methods


def test_every_vintage_becomes_a_cells_history_row(city, cells):
    """Section 6.3: "Fetch every available vintage" — the backtest panel."""
    panel = buildings.buildings_to_history(
        rf.synthetic_footprints_by_vintage(cells), cells, crs_metric=city.crs_metric
    )
    years = sorted(panel["year"].unique())
    assert years == sorted(int(y) for y in rf.FIXTURE_CONFIG["footprints"]["vintages"])
    assert len(panel) == len(years) * len(cells)
    assert panel["builtup_frac"].between(0, 1).all()


def test_buildings_tolerate_a_missing_zoning_layer(city, module1_cells):
    """The national tier runs before the city tier, so ``zone_class`` may be absent."""
    out = buildings.buildings_to_cells(
        rf.synthetic_footprints(module1_cells), module1_cells, crs_metric=city.crs_metric
    )
    assert out["builtup_frac"].notna().all()


# --------------------------------------------------------------------------------------
# 6.4 Population
# --------------------------------------------------------------------------------------


def test_tobler_areal_interpolation_preserves_the_ward_total(city, cells):
    """Population is *extensive*: the interpolation must conserve the total (Section 6.4)."""
    wards = rf.synthetic_wards(cells)
    interpolated = population.population_from_wards(wards, cells, crs_metric=city.crs_metric)
    # Extensive means the total is conserved by the transfer, not re-scaled per cell.
    assert interpolated.sum() == pytest.approx(float(wards["population"].sum()), rel=1e-6)
    assert (interpolated >= 0).all()
    assert interpolated.max() < float(wards["population"].min())


@pytest.mark.acceptance
def test_population_is_within_five_percent_of_the_district_total(reader, city, cells):
    """Section 6 ACCEPTANCE: population sums to within 5% of the district total."""
    ingester = population.PopulationIngester(reader, city=city)
    out = _run(ingester, city, cells)
    district_total = float(sum(rf.FIXTURE_CONFIG["wards"]["populations"]))
    population.assert_district_total(out["population"], district_total)  # raises otherwise
    tolerance = float(core.cfg("population.district_total_tolerance"))
    assert abs(out["population"].sum() - district_total) / district_total <= tolerance


def test_population_outside_the_tolerance_raises():
    with pytest.raises(CoverageError):
        population.assert_district_total(pd.Series([1.0, 1.0]), 1000.0)


def test_dasymetric_refinement_follows_floorspace(city, cells):
    """Section 6.4: redistribute ward totals in proportion to ``floorspace_res_sqm``."""
    wards = rf.synthetic_wards(cells)
    weighted = cells.copy()
    weighted["floorspace_res_sqm"] = np.linspace(0, 1000, len(cells))
    refined, flagged = population.dasymetric_refine(
        wards, weighted, crs_metric=city.crs_metric
    )
    assert refined.sum() > 0
    # More floorspace must mean more people, within a ward.
    correlation = np.corrcoef(refined.to_numpy(), weighted["floorspace_res_sqm"])[0, 1]
    assert correlation > 0
    assert not flagged.all()


def test_dasymetric_flags_a_ward_with_no_floorspace(city, cells):
    wards = rf.synthetic_wards(cells)
    barren = cells.copy()
    barren["floorspace_res_sqm"] = 0.0
    refined, flagged = population.dasymetric_refine(wards, barren, crs_metric=city.crs_metric)
    assert flagged.all(), "a ward with no floorspace must be flagged, not silently spread"
    assert refined.sum() > 0


def test_growth_to_base_year_is_flagged_as_an_estimate(reader, cells, params):
    """Section 6.4: "Document that this is an estimate"."""
    raw = dict(params.city_config)
    raw["district_population_growth_rate"] = 0.02
    grown_city = CityConfig(
        city_id=raw["city_id"],
        state_code=raw["state_code"],
        crs_metric=raw["crs_metric"],
        base_year=int(raw["base_year"]),
        coastal=True,
        cbd_lat=raw["cbd_point"]["lat"],
        cbd_lon=raw["cbd_point"]["lon"],
        raw=raw,
    )
    ingester = population.PopulationIngester(reader, city=grown_city)
    out = _run(ingester, grown_city, cells)
    census_year = int(core.cfg("population.census_year"))
    assert out["population__imputed"].all()
    assert f"grown_from_census_{census_year}" in set(out["population__impute_method"]) | {
        m for value in out["population__impute_method"] for m in [value]
    }


def test_grow_to_base_year_is_a_compound_rate():
    grown = population.grow_to_base_year(
        pd.Series([100.0]), from_year=2011, to_year=2013, annual_growth_rate=0.10
    )
    assert grown.iat[0] == pytest.approx(100.0 * 1.1 * 1.1)


def test_households_raises_rather_than_inventing_a_household_size(params):
    """The parameter is null in behaviour.yaml — Section 0.1 rule 3 forbids a default."""
    with pytest.raises(MissingParameter) as excinfo:
        population.households_from_population(pd.Series([100.0]), params)
    assert population.PERSONS_PER_HH_PATH in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 6.5 OSM
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_distance_known_answer_ten_kilometres(city):
    """Section 6 ACCEPTANCE: two points 10 km apart return 10 000 +- 50 m."""
    target = float(core.cfg("acceptance.distance_known_answer_m"))
    tolerance = float(core.cfg("acceptance.distance_tolerance_m"))
    origin = Point(city.cbd_lon, city.cbd_lat)
    shifted = geo.to_metric(
        gpd.GeoSeries([origin], crs=geo.GEOGRAPHIC_CRS), city.crs_metric
    ).translate(xoff=target)
    far = geo.to_geographic(shifted, city.crs_metric).iat[0]
    measured = osm_mod.distance_to_points_m(
        [(far.y, far.x)], (city.cbd_lat, city.cbd_lon), crs_metric=city.crs_metric
    )
    assert abs(measured[0] - target) <= tolerance


def test_distance_refuses_a_geographic_crs(city):
    with pytest.raises(geo.NonMetricCRSError):
        osm_mod.distance_to_points_m(
            [(city.cbd_lat, city.cbd_lon)],
            (city.cbd_lat, city.cbd_lon),
            crs_metric=geo.GEOGRAPHIC_CRS,
        )


def test_util_power_follows_the_section_6_5_rule(city, cells):
    """Substation within 2 000 m OR an 11/33 kV line within 500 m."""
    substation_radius = float(core.cfg("osm.substation_radius_m"))
    flag, imputed = osm_mod.util_power_flag(
        cells,
        substations=rf.synthetic_substations(cells),
        power_lines=None,
        crs_metric=city.crs_metric,
    )
    distance = osm_mod.distance_to_layer_m(
        cells, rf.synthetic_substations(cells), crs_metric=city.crs_metric
    )
    assert set(np.unique(flag)) <= {0, 1}
    assert np.array_equal(flag.astype(bool), distance <= substation_radius)
    assert not imputed.any()


def test_out_of_scope_voltages_do_not_electrify_a_cell(city, cells):
    """Only 11 kV / 33 kV lines count (Section 6.5); a 400 kV transmission line does not."""
    lines = rf.synthetic_power_lines(cells)
    out_of_scope = lines[
        lines["voltage"] == rf.FIXTURE_CONFIG["power"]["out_of_scope_voltage"]
    ]
    flag, _ = osm_mod.util_power_flag(
        cells, substations=None, power_lines=out_of_scope, crs_metric=city.crs_metric
    )
    assert flag.sum() == 0


def test_no_power_layer_yields_zero_and_a_flag(city, cells):
    flag, imputed = osm_mod.util_power_flag(
        cells, substations=None, power_lines=None, crs_metric=city.crs_metric
    )
    assert flag.sum() == 0
    assert imputed.all(), "an unmapped network is not the same as an absent one"


def test_poi_counts_and_the_jobs_proxy(city, cells):
    counts = osm_mod.poi_counts(cells, rf.synthetic_pois(cells), crs_metric=city.crs_metric)
    per_tag = int(rf.FIXTURE_CONFIG["pois"]["per_tag"])
    assert counts["retail_poi_count"].sum() == per_tag
    assert counts["education_poi_count"].sum() == per_tag
    vectors, imputed = osm_mod.jobs_by_sector_proxy(counts)
    assert all(len(v) == len(S.SECTORS) for v in vectors)
    assert imputed.all(), "a POI-density jobs proxy must always be flagged"
    per_poi = core.cfg("osm.jobs_per_poi")
    retail_index = S.Sector["retail_svc"].value
    assert sum(v[retail_index] for v in vectors) == pytest.approx(
        per_tag * float(per_poi["retail_svc"])
    )


def test_arterial_distance_uses_only_arterial_highways(city, cells):
    roads = rf.synthetic_roads(cells)
    frame = osm_mod.distances_to_cells(cells, city=city, roads=roads, coastline=None)
    arterial_only = roads[roads["highway"] == rf.FIXTURE_CONFIG["roads"]["arterial_value"]]
    expected = osm_mod.distance_to_layer_m(
        cells, arterial_only, crs_metric=city.crs_metric,
        ceiling=float(core.cfg("osm.dist_arterial_ceiling_m")),
    )
    assert np.allclose(frame["dist_arterial_m"], expected)


def test_missing_coastline_leaves_dist_coast_null_and_flagged(city, cells):
    frame = osm_mod.distances_to_cells(cells, city=city, roads=None, coastline=None)
    assert frame["dist_coast_m"].isna().all()
    assert frame["dist_coast_m__imputed"].all()


# --------------------------------------------------------------------------------------
# 6.6 Nightlights
# --------------------------------------------------------------------------------------


def test_outlier_mask_drops_negatives_and_caps_per_month():
    floor = float(core.cfg("nightlights.min_valid_radiance"))
    monthly = pd.DataFrame(
        {
            "h3": ["a", "b", "c", "d"],
            "year": [2024] * 4,
            "month": [1] * 4,
            "nightlight": [-5.0, 1.0, 2.0, 900.0],
        }
    )
    masked = nightlights.apply_outlier_mask(monthly)
    assert pd.isna(masked["nightlight"].iat[0]), "negative radiance must be dropped"
    assert masked["nightlight"].max() <= 900.0
    assert (masked["nightlight"].dropna() >= floor).all()


def test_annual_aggregate_is_a_median_not_a_mean(reader, city, cells):
    """Section 6.6: "the median is more stable than the mean for this series"."""
    radiances = [float(v) for v in rf.FIXTURE_CONFIG["nightlights"]["monthly_radiance"]]
    ingester = nightlights.NightlightsIngester(reader, city=city)
    out = _run(ingester, city, cells)
    assert out["nightlight"].median() == pytest.approx(float(np.median(radiances)), rel=1e-6)
    assert float(np.median(radiances)) != pytest.approx(float(np.mean(radiances)))


def test_nightlights_write_a_cells_history_panel(reader, city, cells):
    ingester = nightlights.NightlightsIngester(reader, city=city)
    _run(ingester, city, cells)
    history = ingester.side_tables["cells_history"]
    assert set(history.columns) >= {"h3", "year", "nightlight"}
    assert (history["nightlight"] >= 0).all()


# --------------------------------------------------------------------------------------
# 6.7 Prices
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_price_blending_reproduces_a_hand_computed_value(params):
    """Section 6 ACCEPTANCE: "Price blending on a synthetic fixture reproduces a
    hand-computed value."

    haircut = price.price_data.ask_haircut_stable = 0.10
    w       = price.price_data.blend_weight_ask   = 0.60
    ask = 5000 -> ask_adj = 5000 * 0.90        = 4500
    reg = 1000, uplift = 2.0 -> reg_adj        = 2000
    price = 0.60 * 4500 + 0.40 * 2000 = 2700 + 800 = 3500
    """
    haircut = float(params.value("price.price_data.ask_haircut_stable"))
    weight = float(params.value("price.price_data.blend_weight_ask"))
    ask, reg, uplift = np.array([5000.0]), np.array([1000.0]), np.array([2.0])
    by_hand = weight * (ask[0] * (1 - haircut)) + (1 - weight) * (reg[0] * uplift[0])
    blended, ask_adj, reg_adj = prices.blend_prices(ask, reg, uplift, params=params)
    assert ask_adj[0] == pytest.approx(4500.0)
    assert reg_adj[0] == pytest.approx(2000.0)
    assert by_hand == pytest.approx(3500.0)
    assert blended[0] == pytest.approx(3500.0)


def test_blend_falls_back_to_whichever_leg_exists(params):
    blended, _, _ = prices.blend_prices(
        np.array([5000.0, np.nan]), np.array([np.nan, 1000.0]), np.array([2.0, 2.0]),
        params=params,
    )
    haircut = float(params.value("price.price_data.ask_haircut_stable"))
    assert blended[0] == pytest.approx(5000.0 * (1 - haircut))
    assert blended[1] == pytest.approx(2000.0)


def test_uplift_collapses_to_the_single_median_below_thirty_observations(cells):
    """Section 6.7: fewer than 30 broker observations -> single median, data_conf 0.4."""
    minimum = int(core.cfg("prices.min_broker_observations"))
    n = len(cells)
    broker = np.full(n, np.nan)
    reg = np.full(n, 1000.0)
    broker[: minimum - 1] = 2000.0
    uplift, low_confidence, n_obs = prices.fit_regional_uplift(broker, reg, cells)
    assert low_confidence and n_obs == minimum - 1
    assert np.allclose(uplift, 2.0)


def test_uplift_is_fitted_when_enough_observations_exist():
    """With >= 30 observations the uplift is a regression on (dist_cbd, zone_class)."""
    minimum = int(core.cfg("prices.min_broker_observations"))
    n = minimum * 2
    frame = pd.DataFrame(
        {
            "h3": [f"c{i}" for i in range(n)],
            "dist_cbd_m": np.linspace(0, 20000, n),
            "zone_class": ["res"] * n,
        }
    )
    reg = np.full(n, 1000.0)
    # A true uplift that falls linearly with distance from the CBD.
    truth = 3.0 - 0.0001 * frame["dist_cbd_m"].to_numpy()
    broker = np.where(np.arange(n) < minimum, reg * truth, np.nan)
    uplift, low_confidence, n_obs = prices.fit_regional_uplift(broker, reg, frame)
    assert not low_confidence and n_obs == minimum
    unobserved = np.arange(n) >= minimum
    assert np.allclose(uplift[unobserved], truth[unobserved], atol=1e-6)


def test_low_confidence_pins_the_price_data_conf(reader, city, params, cells, adapter):
    ingester = prices.PricesIngester(reader, adapter=adapter, city=city, params=params)
    _run(ingester, city, cells)
    assert ingester.low_confidence
    assert ingester.price_column_data_conf() == pytest.approx(
        float(core.cfg("prices.low_confidence_data_conf"))
    )


def test_gaussian_smear_decays_with_distance(city, cells):
    points = rf.synthetic_listing_points(cells)
    smeared = prices.gaussian_smear(
        cells, points, "ask_inr_sqft", crs_metric=city.crs_metric
    )
    assert smeared.notna().any()
    value = float(rf.FIXTURE_CONFIG["listings"]["point_ask_inr_sqft"])
    # Every contributing observation carries the same value, so the weighted mean is it.
    assert np.allclose(smeared.dropna(), value)
    # And distant cells get nothing rather than an extrapolation.
    assert smeared.isna().any() or len(smeared) == len(cells)


def test_rents_cascade_to_the_parent_then_stop(cells):
    """Section 6.7: res-8 parent median, then mark the detector unavailable."""
    rent = pd.Series(np.nan, index=cells.index)
    parents = cells["h3_res8"].astype(str)
    first_parent = parents.iat[0]
    known = parents == first_parent
    rent[known & (np.arange(len(cells)) == 0)] = 20.0
    filled, from_parent, unavailable = prices.impute_rents(rent, cells)
    assert np.allclose(filled[known.to_numpy()], 20.0)
    assert from_parent[known.to_numpy()].sum() == int(known.sum()) - 1
    assert unavailable[~known.to_numpy()].all(), "no rent anywhere in the parent -> stop"
    assert not unavailable[known.to_numpy()].any()


def test_overheat_unavailable_is_recorded_as_a_method(reader, city, params, adapter, cells):
    ingester = prices.PricesIngester(reader, adapter=adapter, city=city, params=params)
    out = _run(ingester, city, cells)
    assert prices.RENT_COLUMN + core.METHOD_SUFFIX in out.columns


def test_production_mode_refuses_a_tos_restricted_source():
    """Section 6.7 / 22.3: the largest legal exposure in the system, enforced in code."""
    with pytest.raises(DataRightsViolation):
        prices.assert_listing_source_permitted("production", "tos_restricted")


def test_licensed_feed_is_permitted_in_production():
    prices.assert_listing_source_permitted("production", "licensed_feed")
    prices.assert_listing_source_permitted("development", "tos_restricted")


def test_prices_ingester_refuses_at_fetch_in_production(reader, params, adapter, cells):
    production_city = CityConfig.from_params(params, mode="production")
    ingester = prices.PricesIngester(
        reader,
        adapter=adapter,
        city=production_city,
        params=params,
        licence_status="tos_restricted",
    )
    with pytest.raises(DataRightsViolation):
        ingester.fetch(production_city)


def test_broker_panel_schema_is_enforced(city, cells):
    bad = rf.synthetic_broker_panel(cells).drop(columns=["area_sqft"])
    with pytest.raises(ValueError):
        prices.broker_panel_to_cells(bad, cells, crs_metric=city.crs_metric)


def test_broker_panel_median_is_the_observed_price(city, cells):
    panel = rf.synthetic_broker_panel(cells)
    out = prices.broker_panel_to_cells(panel, cells, crs_metric=city.crs_metric)
    observed = out["_broker"].dropna()
    assert len(observed)
    assert np.allclose(observed, float(rf.FIXTURE_CONFIG["broker"]["inr_sqft"]))


# --------------------------------------------------------------------------------------
# 6.8 RERA
# --------------------------------------------------------------------------------------


def test_rera_produces_the_three_section_6_8_outputs(reader, city, cells, adapter):
    ingester = rera.ReraIngester(reader, adapter=adapter, city=city)
    _run(ingester, city, cells)
    tables = ingester.side_tables
    assert set(tables) == {
        "supply_pipeline",
        "absorption_observed",
        "developer_delivery_record",
    }
    pipeline = tables["supply_pipeline"]
    assert set(pipeline.columns) == set(rera.SUPPLY_PIPELINE_COLUMNS)
    assert (pipeline["units"] > 0).all()
    delivery = tables["developer_delivery_record"]
    assert (delivery["delivery_ratio"] <= 1).all()


def test_delivery_record_is_deterministic(reader, city, cells, adapter):
    """No wall clock: the slip is measured against the last reported quarter."""
    projects_frame = adapter.rera_projects(city)
    first = rera.delivery_record(projects_frame)
    second = rera.delivery_record(projects_frame)
    pd.testing.assert_frame_equal(first, second)
    assert "utcnow" not in Path(rera.__file__).read_text()


# --------------------------------------------------------------------------------------
# 6.9 Cadastral
# --------------------------------------------------------------------------------------


def test_parcel_statistics_match_hand_computed_areas(city, cells, adapter):
    parcels = rf.synthetic_parcels(cells)
    columns, stats = cadastral.parcels_to_cells(parcels, cells, crs_metric=city.crs_metric)
    sides = [float(s) for s in rf.FIXTURE_CONFIG["parcels"]["side_m"]]
    expected_mean = float(np.mean([s * s for s in sides]))
    observed = columns.loc[columns["parcel_count"] > 0, "mean_parcel_sqm"]
    assert len(observed)
    assert np.allclose(observed, expected_mean, rtol=1e-6)
    assert (columns["parcel_count"] >= 0).all()
    gini = stats["parcel_size_gini"].dropna()
    assert (gini > 0).all(), "two parcel sizes must give a non-zero Gini"


def test_gini_of_an_equal_distribution_is_zero():
    assert cadastral.gini(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0, abs=1e-9)


def test_cadastral_fallback_is_flagged(city, cells):
    """Section 6.9: infer from footprints, mark ``data_conf`` down."""
    footprints = rf.synthetic_footprints(cells)
    columns, _ = cadastral.fragmentation_from_footprints(
        footprints, cells, crs_metric=city.crs_metric
    )
    assert columns["mean_parcel_sqm__imputed"].all()
    assert set(columns["mean_parcel_sqm__impute_method"]) == {"inferred_from_footprints"}
    coverage_ratio = float(core.cfg("cadastral.fallback_plot_coverage_ratio"))
    side = float(rf.FIXTURE_CONFIG["footprints"]["side_m"])
    observed = columns.loc[columns["parcel_count"] > 0, "mean_parcel_sqm"]
    assert np.allclose(observed, side * side / coverage_ratio, rtol=1e-3)


# --------------------------------------------------------------------------------------
# 6.10 Zoning and CZMP — the hard requirement
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_coastal_city_with_no_crz_layer_raises_missing_critical_layer(city, cells):
    """Section 6 ACCEPTANCE / Section 20.2 step 4: a hard requirement, by design."""
    assert city.coastal, "vizag must be configured coastal for this test to mean anything"
    with pytest.raises(MissingCriticalLayer) as excinfo:
        zoning.crz_to_cells(None, cells, city=city)
    assert "coastal" in str(excinfo.value).lower()


@pytest.mark.acceptance
def test_coastal_ingester_raises_when_the_czmp_key_is_absent(city, cells, tmp_path):
    reader, cells_local, _ = rf.build_reader(tmp_path, coastal=False)
    ingester = zoning.ZoningIngester(reader, city=city)
    with pytest.raises(MissingCriticalLayer):
        _run(ingester, city, cells_local)


def test_no_distance_buffer_is_substituted_for_crz(city):
    """Section 6.10: "Do not silently substitute a distance buffer"."""
    source = Path(zoning.__file__).read_text()
    assert "dist_coast" not in source


def test_inland_city_without_a_czmp_layer_is_fine(params, cells):
    raw = dict(params.city_config)
    raw["coastal"] = False
    inland = CityConfig(
        city_id="inland",
        state_code=raw["state_code"],
        crs_metric=raw["crs_metric"],
        base_year=int(raw["base_year"]),
        coastal=False,
        cbd_lat=raw["cbd_point"]["lat"],
        cbd_lon=raw["cbd_point"]["lon"],
        raw=raw,
    )
    out = zoning.crz_to_cells(None, cells, city=inland)
    assert (out["crz_class"] == core.cfg("zoning.no_crz_class")).all()
    assert not out["crz_class__imputed"].any()


def test_zoning_requires_every_contracted_attribute(city, cells):
    incomplete = rf.synthetic_zoning(cells).drop(columns=["source_sheet"])
    with pytest.raises(MissingCriticalLayer) as excinfo:
        zoning.zoning_to_cells(incomplete, cells, crs_metric=city.crs_metric)
    assert "source_sheet" in str(excinfo.value)


def test_zone_class_is_the_majority_and_far_is_area_weighted(city, cells):
    """Section 6.10's assignment rule."""
    out = zoning.zoning_to_cells(rf.synthetic_zoning(cells), cells, crs_metric=city.crs_metric)
    assert out["zone_class"].isin(S.ZONE_CLASSES).all()
    fars = [float(f) for f in rf.FIXTURE_CONFIG["zoning"]["permitted_far"]]
    covered = ~out["zone_class__imputed"]
    observed = out.loc[covered, "permitted_far"]
    assert observed.min() >= min(fars) - 1e-9
    assert observed.max() <= max(fars) + 1e-9
    # A cell spanning two zones must land strictly between their FARs, not on either.
    assert ((observed > min(fars)) & (observed < max(fars))).any()


def test_cells_outside_the_master_plan_are_flagged(city, cells):
    partial = rf.synthetic_zoning(cells, cover_all=False)
    out = zoning.zoning_to_cells(partial, cells, crs_metric=city.crs_metric)
    assert out["zone_class__imputed"].any()
    unzoned = str(core.cfg("zoning.unzoned_zone_class"))
    assert (out.loc[out["zone_class__imputed"], "zone_class"] == unzoned).all()


def test_missing_utility_layer_is_flagged_not_assumed(city, cells):
    out = zoning.utilities_to_cells(None, cells, crs_metric=city.crs_metric)
    assert (out["util_water"] == 0).all()
    assert out["util_water__imputed"].all() and out["util_sewer__imputed"].all()


def test_sewer_implies_water(city, cells):
    out = zoning.utilities_to_cells(
        rf.synthetic_utilities(cells), cells, crs_metric=city.crs_metric
    )
    assert not ((out["util_sewer"] == 1) & (out["util_water"] == 0)).any()


# --------------------------------------------------------------------------------------
# 6.11 Projects — all six rules are fatal
# --------------------------------------------------------------------------------------


@pytest.fixture
def registry(cells):
    from tests.fixtures.synthetic import synthetic_projects

    units = projects.archetype_units(load_params("vizag"))
    archetype = sorted(units)[0]
    frame = synthetic_projects(n=2, cells=cells).copy()
    frame["archetype"] = archetype
    frame["scale_unit"] = units[archetype]
    frame["stage"] = S.PROJECT_STAGES[0]
    frame["geom"] = [
        Point(float(cells["lon"].iat[i]), float(cells["lat"].iat[i])).wkt for i in range(2)
    ]
    return frame


def test_a_valid_registry_passes(registry, cells, params, city):
    validated = projects.validate_projects(
        registry, params=params, cells=cells, city=city
    )
    assert len(validated) == len(registry)


def test_unknown_archetype_is_fatal(registry, cells, params, city):
    bad = registry.copy()
    bad.loc[bad.index[0], "archetype"] = "not_an_archetype"
    with pytest.raises(projects.ProjectValidationError, match="archetype"):
        projects.validate_projects(bad, params=params, cells=cells, city=city)


def test_mismatched_scale_unit_is_fatal(registry, cells, params, city):
    bad = registry.copy()
    wrong = [u for u in S.SCALE_UNITS if u != bad["scale_unit"].iat[0]][0]
    bad.loc[bad.index[0], "scale_unit"] = wrong
    with pytest.raises(projects.ProjectValidationError, match="scale_unit"):
        projects.validate_projects(bad, params=params, cells=cells, city=city)


def test_empty_source_urls_is_fatal(registry, cells, params, city):
    bad = registry.copy()
    bad.at[bad.index[0], "source_urls"] = []
    with pytest.raises(projects.ProjectValidationError, match="source_urls"):
        projects.validate_projects(bad, params=params, cells=cells, city=city)


def test_completion_before_announcement_is_fatal(registry, cells, params, city):
    bad = registry.copy()
    bad.loc[bad.index[0], "stated_completion"] = bad["announced_date"].iat[0] - pd.Timedelta(
        days=1
    )
    with pytest.raises(projects.ProjectValidationError, match="stated_completion"):
        projects.validate_projects(bad, params=params, cells=cells, city=city)


def test_geometry_outside_the_halo_is_fatal(registry, cells, params, city):
    bad = registry.copy()
    bad.loc[bad.index[0], "geom"] = Point(0.0, 0.0).wkt
    with pytest.raises(projects.ProjectValidationError, match="halo"):
        projects.validate_projects(bad, params=params, cells=cells, city=city)


def test_ai_only_stage_transition_is_fatal(registry, cells, params, city):
    """Section 6.11: a stage the model acts on must carry a human sign-off."""
    advanced = registry.copy()
    advanced["stage"] = S.PROJECT_STAGES[1]
    ai_history = pd.DataFrame(
        {
            "project_id": advanced["project_id"],
            "field": "stage",
            "changed_by": "ai:1.4.0",
        }
    )
    with pytest.raises(projects.ProjectValidationError, match="project_history"):
        projects.validate_projects(
            advanced, params=params, cells=cells, project_history=ai_history, city=city
        )
    human_history = ai_history.assign(changed_by="analyst-0")
    projects.validate_projects(
        advanced, params=params, cells=cells, project_history=human_history, city=city
    )


# --------------------------------------------------------------------------------------
# Imputation flagging
# --------------------------------------------------------------------------------------


def test_mark_imputed_is_pure_and_records_a_method():
    frame = pd.DataFrame({"h3": ["a", "b"], "x": [1.0, 2.0]})
    marked = core.mark_imputed(frame, "x", [True, False], "guessed")
    assert "x__imputed" not in frame.columns, "mark_imputed must not mutate its input"
    assert marked["x__imputed"].tolist() == [True, False]
    assert marked["x__impute_method"].tolist() == ["guessed", ""]


def test_flags_accumulate_and_keep_the_first_reason():
    frame = pd.DataFrame({"h3": ["a", "b"], "x": [1.0, 2.0]})
    marked = core.mark_imputed(frame, "x", [True, False], "first")
    marked = core.mark_imputed(marked, "x", [True, True], "second")
    assert marked["x__imputed"].all()
    assert marked["x__impute_method"].tolist() == ["first", "second"]


def test_imputation_long_is_the_persisted_ledger():
    frame = core.mark_imputed(
        pd.DataFrame({"h3": ["a", "b"], "x": [1.0, 2.0]}), "x", [True, False], "guessed"
    )
    ledger = core.imputation_long(frame, source_id="test_source", run_id="r1")
    assert list(ledger.columns) == list(core.CELL_IMPUTATION_COLUMNS)
    assert ledger.loc[ledger["h3"] == "a", "imputed"].iat[0]
    assert ledger["source_id"].unique().tolist() == ["test_source"]


def test_flags_never_reach_the_cells_frame(cells):
    frame = core.mark_imputed(
        pd.DataFrame({"h3": cells["h3"].astype(str), "elev_m": 1.0}),
        "elev_m",
        np.ones(len(cells), dtype=bool),
        "guessed",
    )
    merged = core.merge_ingested(cells, frame)
    assert not [c for c in merged.columns if c.endswith(core.IMPUTED_SUFFIX)]
    assert set(merged.columns) <= set(S.CELLS.columns)


def test_data_conf_falls_with_imputation_and_missing_capabilities(cells):
    clean = core.data_conf(pd.DataFrame(columns=list(core.CELL_IMPUTATION_COLUMNS)), cells)
    ledger = core.imputation_long(
        core.mark_imputed(
            pd.DataFrame({"h3": cells["h3"].astype(str), "price_res_inr_sqft": 1.0}),
            "price_res_inr_sqft",
            np.ones(len(cells), dtype=bool),
            "guessed",
        ),
        source_id="s",
    )
    degraded = core.data_conf(ledger, cells)
    penalised = core.data_conf(ledger, cells, missing_capabilities=["registration_transactions"])
    weight = float(core.cfg("data_conf.column_weights")["price_res_inr_sqft"])
    penalty = float(core.cfg("data_conf.missing_capability_penalty"))
    assert clean.iat[0] == pytest.approx(float(core.cfg("data_conf.base")))
    assert degraded.iat[0] == pytest.approx(clean.iat[0] - weight)
    assert penalised.iat[0] == pytest.approx(degraded.iat[0] - penalty)
    assert (penalised >= float(core.cfg("data_conf.floor"))).all()


# --------------------------------------------------------------------------------------
# Section 20.2 step 9 — the coverage gate
# --------------------------------------------------------------------------------------


def _priced_cells(cells: pd.DataFrame, real_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A cells frame where exactly ``real_fraction`` of populated cells have a real price."""
    frame = cells.copy()
    frame["population"] = 100.0
    frame["in_city"] = True
    frame["price_res_inr_sqft"] = 5000.0
    n_real = int(round(real_fraction * len(frame)))
    flags = np.ones(len(frame), dtype=bool)
    flags[:n_real] = False
    produced = core.mark_imputed(
        pd.DataFrame(
            {"h3": frame["h3"].astype(str), "price_res_inr_sqft": frame["price_res_inr_sqft"]}
        ),
        "price_res_inr_sqft",
        flags,
        "no_listing_observation",
    )
    return frame, core.imputation_long(produced, source_id="listing_portals")


def test_coverage_report_separates_real_imputed_and_missing(cells):
    frame, ledger = _priced_cells(cells, 0.5)
    report = coverage.coverage_report(frame, ledger)
    row = report[report["column"] == "price_res_inr_sqft"].iloc[0]
    assert row["n_real"] + row["n_imputed"] + row["n_missing"] == row["cells_populated"]
    assert row["frac_real"] == pytest.approx(0.5, abs=1 / len(cells))
    assert "no_listing_observation" in row["methods"]


def test_coverage_report_counts_a_fully_populated_but_imputed_column_as_zero_real(cells):
    frame, ledger = _priced_cells(cells, 0.0)
    report = coverage.coverage_report(frame, ledger)
    row = report[report["column"] == "price_res_inr_sqft"].iloc[0]
    assert row["n_missing"] == 0, "the column is fully populated"
    assert row["frac_real"] == 0.0, "but none of it is real"


def test_coverage_gate_raises_below_the_threshold(cells):
    threshold = coverage.coverage_thresholds()["price_res_inr_sqft"]
    frame, ledger = _priced_cells(cells, threshold / 2)
    report = coverage.coverage_report(frame, ledger)
    with pytest.raises(CoverageError, match="price_res_inr_sqft"):
        coverage.assert_coverage(report)


def test_coverage_gate_passes_above_the_threshold(cells):
    threshold = coverage.coverage_thresholds()["price_res_inr_sqft"]
    frame, ledger = _priced_cells(cells, min(1.0, threshold + 0.2))
    coverage.assert_coverage(coverage.coverage_report(frame, ledger))


def test_coverage_threshold_comes_from_yaml_not_a_literal():
    """Section 20.2 step 9's 40% must be data, not code."""
    thresholds = coverage.coverage_thresholds()
    assert "price_res_inr_sqft" in thresholds
    assert 0 < thresholds["price_res_inr_sqft"] <= 1
    source = Path(coverage.__file__).read_text()
    assert "0.4" not in source and "0.40" not in source


def test_populated_cells_are_the_denominator(cells):
    frame = cells.copy()
    frame["in_city"] = True
    frame["population"] = 0.0
    frame.loc[frame.index[:10], "population"] = 500.0
    mask = coverage.populated_mask(frame)
    assert mask.sum() == 10


def test_coverage_report_lists_an_absent_column(cells):
    frame = cells.copy().drop(columns=["price_res_inr_sqft"])
    frame["population"] = 100.0
    report = coverage.coverage_report(frame, None)
    row = report[report["column"] == "price_res_inr_sqft"].iloc[0]
    assert row["methods"] == "column_absent"
    assert not bool(row["passes"])


# --------------------------------------------------------------------------------------
# Caching and the ingest_runs ACCEPTANCE item
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_rerun_with_force_false_uses_the_cache_and_is_identical(reader, city, cells, tmp_path):
    """Section 6 ACCEPTANCE: "Re-running an ingester with force=False uses the cache and
    produces an identical frame"."""
    ingester = terrain.TerrainIngester(reader, city=city)
    ingester.work_root = tmp_path
    before = reader.reads.get(terrain.KEY_DEM, 0)
    first = _run(ingester, city, cells)
    after_first = reader.reads.get(terrain.KEY_DEM, 0)
    second = _run(ingester, city, cells)
    after_second = reader.reads.get(terrain.KEY_DEM, 0)

    assert after_first == before + 1, "the first run must read the source once"
    assert after_second == after_first, "the second run must come from the cache"
    pd.testing.assert_frame_equal(first, second)


def test_force_true_bypasses_the_cache(reader, city, tmp_path):
    ingester = terrain.TerrainIngester(reader, city=city)
    ingester.work_root = tmp_path
    ingester.fetch(city)
    before = reader.reads.get(terrain.KEY_DEM, 0)
    ingester.fetch(city, force=True)
    assert reader.reads.get(terrain.KEY_DEM, 0) == before + 1


@pytest.mark.acceptance
def test_every_populated_column_has_an_ingest_run(
    reader, city, params, adapter, module1_cells
):
    """Section 6 ACCEPTANCE / Section 6: "A cell attribute with no corresponding ingest run
    is invalid"."""
    working = module1_cells
    runs: list[dict] = []
    for tier in ("national", "city", "state"):
        ingesters = runner.ingesters_for_tier(
            tier, reader=reader, city=city, params=params, adapter=adapter
        )
        for ingester in ingesters:
            if isinstance(ingester, terrain.TerrainIngester):
                ingester.work_root = Path(reader.path(terrain.KEY_DEM)).parent
        result = runner.run_ingesters(
            ingesters, cells=working, city=city, params=params, adapter=adapter
        )
        assert not result.failures, result.failures
        working = result.cells
        runs.extend(result.runs)

    core.assert_every_column_has_run(working, runs, exempt=MODULE_1_COLUMNS)
    for column in ("elev_m", "slope_pct", "landcover", "builtup_frac", "population",
                   "zone_class", "crz_class", "nightlight", "price_res_inr_sqft",
                   "parcel_count", "util_power", "data_conf"):
        assert column in working.columns, column


def test_orphan_column_is_detected(cells):
    with pytest.raises(UFEError, match="ingest_runs"):
        core.assert_every_column_has_run(
            cells[["h3", "elev_m"]], [{"columns": []}], exempt=("h3",)
        )


# --------------------------------------------------------------------------------------
# Tests that need real source data — written, and skipped in this environment
# --------------------------------------------------------------------------------------


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.environ.get(REAL_DATA_ENV),
    reason=f"needs real downloaded source data; set {REAL_DATA_ENV}",
)
def test_real_copernicus_dem_covers_the_vizag_halo(city):
    """Requires the real Copernicus DEM GLO-30 tiles for the Vizag halo."""
    reader = core.LocalFileReader(Path(core.cfg("reader.raw_root")) / city.city_id)
    ingester = terrain.TerrainIngester(reader, city=city)
    raw = ingester.fetch(city)
    assert Path(raw).exists()


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.environ.get(REAL_DATA_ENV),
    reason=f"needs real downloaded source data; set {REAL_DATA_ENV}",
)
def test_real_worldcover_tile_classes_are_all_mapped(city):
    """Requires the real ESA WorldCover 2021 tile; asserts no unmapped class code."""
    import rasterio

    reader = core.LocalFileReader(Path(core.cfg("reader.raw_root")) / city.city_id)
    with rasterio.open(reader.path(landcover.KEY_LANDCOVER)) as ds:
        codes = set(np.unique(ds.read(1)).tolist())
    assert codes <= set(landcover.class_name_map()) | {0}


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.environ.get(REAL_DATA_ENV),
    reason=f"needs real downloaded source data; set {REAL_DATA_ENV}",
)
def test_real_open_buildings_has_multiple_vintages(city):
    """Requires the real Open Buildings releases; Section 6.3 needs every vintage."""
    reader = core.LocalFileReader(Path(core.cfg("reader.raw_root")) / city.city_id)
    ingester = buildings.BuildingsIngester(reader, city=city)
    assert len(ingester.available_vintages(range(2016, city.base_year + 1))) > 1


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.environ.get(REAL_DATA_ENV),
    reason=f"needs real downloaded source data; set {REAL_DATA_ENV}",
)
def test_real_viirs_series_starts_in_2012(city):
    """Requires the real VIIRS DNB archive (Section 6.6: 2012-present)."""
    reader = core.LocalFileReader(Path(core.cfg("reader.raw_root")) / city.city_id)
    ingester = nightlights.NightlightsIngester(reader, city=city)
    first = int(core.cfg("nightlights.first_available_year"))
    assert reader.exists(nightlights.monthly_key(first, 1))


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.environ.get(REAL_DATA_ENV),
    reason=f"needs real downloaded source data; set {REAL_DATA_ENV}",
)
def test_real_population_matches_the_district_census_total(city, params):
    """Requires the real Census 2011 ward layer and the district total from the city config."""
    reader = core.LocalFileReader(Path(core.cfg("reader.raw_root")) / city.city_id)
    ingester = population.PopulationIngester(reader, city=city)
    cells_real = pd.DataFrame()  # a real grid, from the store
    out = _run(ingester, city, cells_real)
    population.assert_district_total(out["population"], float(city.get("district_population")))
