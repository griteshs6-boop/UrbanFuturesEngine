"""Tests for the OSM Produced Work rule and attribution rendering (Section 22)."""

from __future__ import annotations

import pytest

from ufe import rights
from ufe.errors import DataRightsViolation

# All 30 `cells` columns per Section 3.1, for exhaustiveness checks.
ALL_CELLS_COLUMNS = [
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
    "jobs_by_sector",
    "floorspace_res_sqm",
    "floorspace_com_sqm",
    "price_res_inr_sqft",
    "price_land_inr_sqft",
    "rent_res_inr_sqft_mo",
    "mean_parcel_sqm",
    "parcel_count",
    "util_water",
    "util_sewer",
    "util_power",
    "dist_cbd_m",
    "dist_coast_m",
    "dist_arterial_m",
    "nightlight",
    "data_conf",
]


# --- classify_column -------------------------------------------------------------------------


@pytest.mark.parametrize("column", sorted(rights.CELLS_OSM_DERIVED_RAW_COLUMNS))
def test_classify_column_osm_derived_raw(column):
    assert rights.classify_column(column) == "osm_derived_raw"


@pytest.mark.parametrize("column", sorted(rights.CELLS_CLEAN_COLUMNS))
def test_classify_column_clean(column):
    assert rights.classify_column(column) == "clean"


@pytest.mark.parametrize(
    "column",
    ["price_index", "rank", "scenario_delta_pct", "residual", "factor_loading_metro", "not_a_real_column"],
)
def test_classify_column_unknown_defaults_to_produced_work(column):
    assert rights.classify_column(column) == "produced_work"


def test_every_cells_column_is_classified_exactly_once():
    """The full Section 3.1 `cells` schema must partition into osm_derived_raw/clean with no
    column left unclassified (falling through to the produced_work default would silently
    make it exposable).
    """
    classified = rights.classify_columns(ALL_CELLS_COLUMNS)
    assert set(classified.values()) <= {"osm_derived_raw", "clean"}
    assert set(ALL_CELLS_COLUMNS) == rights.CELLS_OSM_DERIVED_RAW_COLUMNS | rights.CELLS_CLEAN_COLUMNS


def test_osm_derived_and_clean_sets_are_disjoint():
    assert rights.CELLS_OSM_DERIVED_RAW_COLUMNS.isdisjoint(rights.CELLS_CLEAN_COLUMNS)


# --- assert_exposable — ACCEPTANCE, Section 22.1 / Section 23 item 9 -------------------------


@pytest.mark.acceptance
@pytest.mark.parametrize("column", sorted(rights.CELLS_OSM_DERIVED_RAW_COLUMNS))
def test_assert_exposable_raises_on_each_raw_osm_column(column):
    with pytest.raises(DataRightsViolation):
        rights.assert_exposable([column])


@pytest.mark.acceptance
def test_assert_exposable_raises_when_raw_osm_column_mixed_with_safe_ones():
    with pytest.raises(DataRightsViolation):
        rights.assert_exposable(["price_res_inr_sqft", "dist_arterial_m", "rank"])


@pytest.mark.acceptance
def test_assert_exposable_allows_computed_outputs_only():
    """A Produced Work response — prices, rankings, scenario results — must be exposable."""
    rights.assert_exposable(
        ["price_res_inr_sqft", "rank", "scenario_delta_pct", "factor_loading_metro"]
    )  # must not raise


@pytest.mark.acceptance
def test_assert_exposable_allows_empty_column_list():
    rights.assert_exposable([])  # must not raise


def test_assert_exposable_error_message_names_the_offending_columns():
    try:
        rights.assert_exposable(["util_power"])
    except DataRightsViolation as exc:
        assert "util_power" in str(exc)
    else:
        pytest.fail("expected DataRightsViolation")


# --- attribution rendering (Section 22.4) -----------------------------------------------------


def test_get_attribution_text_full_block_contains_osm():
    text = rights.get_attribution_text()
    assert "OpenStreetMap" in text
    assert "ODbL" in text


def test_get_attribution_text_filtered_by_source_keys():
    text = rights.get_attribution_text(["openstreetmap"])
    assert "OpenStreetMap" in text
    assert "WorldPop" not in text


def test_get_attribution_text_resolves_aliases():
    text_by_alias = rights.get_attribution_text(["osm"])
    text_by_key = rights.get_attribution_text(["openstreetmap"])
    assert text_by_alias == text_by_key


@pytest.mark.acceptance
def test_get_attribution_text_raises_on_unresolvable_source():
    """Section 22.4: a report build that cannot resolve an attribution for a source it used
    must fail, not silently omit the attribution.
    """
    with pytest.raises(DataRightsViolation):
        rights.get_attribution_text(["some_source_nobody_registered"])


def test_get_attribution_text_renders_into_a_footer_string():
    """Simulates the Module 21 report-footer use case: the function must return a plain,
    embeddable multi-line string, not a data structure.
    """
    footer = rights.get_attribution_text(["openstreetmap", "esa_worldcover"])
    assert isinstance(footer, str)
    lines = [l for l in footer.splitlines() if l.strip()]
    assert len(lines) == 2
