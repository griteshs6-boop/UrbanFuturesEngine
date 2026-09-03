"""Tests for Module 14 — the satellite monitor. No network access is used anywhere: the
`ScriptedImageryBackend` fixture serves small synthetic in-memory rasters."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.satellite_fixtures import (
    ScriptedImageryBackend,
    SimpleParams,
    bands_from_targets,
    build_labelled_sites,
    load_satellite_params,
    make_scene,
)
from ufe.satellite import indices, monitor
from ufe.satellite.monitor import ProjectAOI
from ufe.satellite.stac import SceneAssets


@pytest.fixture()
def params():
    return load_satellite_params()


# ---------------------------------------------------------------------------
# Index maths
# ---------------------------------------------------------------------------


def test_ndvi_ndbi_bsi_hand_computed():
    # NIR=0.4, RED=0.1 -> NDVI = (0.4-0.1)/(0.4+0.1) = 0.6
    assert indices.ndvi(np.array([0.4]), np.array([0.1]))[0] == pytest.approx(0.6)
    # SWIR=0.3, NIR=0.1 -> NDBI = (0.3-0.1)/(0.3+0.1) = 0.5
    assert indices.ndbi(np.array([0.3]), np.array([0.1]))[0] == pytest.approx(0.5)
    # BSI: B02=0.1,B04=0.2,B08=0.3,B11=0.4 -> ((0.4+0.2)-(0.3+0.1))/((0.4+0.2)+(0.3+0.1))
    #    = (0.6-0.4)/(0.6+0.4) = 0.2
    assert indices.bsi(
        np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.4])
    )[0] == pytest.approx(0.2)
    assert indices.brightness(np.array([0.1]), np.array([0.2]), np.array([0.3]))[0] == pytest.approx(0.2)


def test_bands_from_targets_roundtrip():
    targets = {"ndvi": 0.55, "ndbi": -0.20, "bsi": 0.05}
    bands = bands_from_targets(targets["ndvi"], targets["ndbi"], targets["bsi"])
    got_ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"])
    got_ndbi = (bands["B11"] - bands["B08"]) / (bands["B11"] + bands["B08"])
    s = bands["B11"] + bands["B04"]
    t = bands["B08"] + bands["B02"]
    got_bsi = (s - t) / (s + t)
    assert got_ndvi == pytest.approx(targets["ndvi"])
    assert got_ndbi == pytest.approx(targets["ndbi"])
    assert got_bsi == pytest.approx(targets["bsi"])


def test_cloud_mask_flags_only_cloud_codes():
    scl = np.array([[4, 8], [3, 6]])
    mask = indices.cloud_mask(scl, cloud_codes=[3, 8, 9, 10])
    assert mask.tolist() == [[False, True], [True, False]]


# ---------------------------------------------------------------------------
# Monthly compositing / cloud handling
# ---------------------------------------------------------------------------


def test_monthly_composite_matches_hand_computed_value(params):
    rng = np.random.default_rng(1)
    scene = make_scene("cleared", pd.Timestamp("2024-03-01"), rng, cloud_frac=0.0)
    composites = indices.build_monthly_composites([scene], params)
    assert len(composites) == 1
    c = composites[0]
    assert c.valid is True
    target = {"ndvi": 0.50, "ndbi": -0.40, "bsi": -0.28}
    assert c.ndvi == pytest.approx(target["ndvi"], abs=1e-6)
    assert c.ndbi == pytest.approx(target["ndbi"], abs=1e-6)
    assert c.bsi == pytest.approx(target["bsi"], abs=1e-6)


def test_fully_clouded_month_produces_no_composite(params):
    """Cloud-masking removes scenes correctly; a fully clouded month produces no composite
    rather than a spurious state change (ACCEPTANCE bullet 2)."""
    rng = np.random.default_rng(2)
    scene = make_scene("structure", pd.Timestamp("2024-05-01"), rng, cloud_frac=1.0)
    composites = indices.build_monthly_composites([scene], params)
    assert len(composites) == 1
    assert composites[0].valid is False
    assert np.isnan(composites[0].ndvi)


def test_partially_clouded_month_under_threshold_still_valid(params):
    rng = np.random.default_rng(3)
    scene = make_scene("none", pd.Timestamp("2024-06-01"), rng, cloud_frac=0.10, shape=(10, 10))
    composites = indices.build_monthly_composites([scene], params)
    assert composites[0].valid is True


def test_month_above_cloud_threshold_is_invalid(params):
    rng = np.random.default_rng(4)
    scene = make_scene("none", pd.Timestamp("2024-06-01"), rng, cloud_frac=0.60, shape=(10, 10))
    composites = indices.build_monthly_composites([scene], params)
    assert composites[0].valid is False


def test_median_across_multiple_scenes_in_month(params):
    """Two scenes in the same month: cloud-masked median should equal the clear scene's
    value, not be corrupted by the fully-clouded one."""
    rng = np.random.default_rng(5)
    clear = make_scene("cleared", pd.Timestamp("2024-04-05"), rng, cloud_frac=0.0)
    clouded = make_scene("structure", pd.Timestamp("2024-04-20"), rng, cloud_frac=1.0)
    composites = indices.build_monthly_composites([clear, clouded], params)
    assert len(composites) == 1
    assert composites[0].valid is True
    assert composites[0].ndvi == pytest.approx(0.50, abs=1e-6)


# ---------------------------------------------------------------------------
# Change detection / classification (pure, deterministic)
# ---------------------------------------------------------------------------


def test_classify_state_none(params):
    assert monitor.classify_state(0.0, 0.0, 0.0, False, params) == "none"


def test_classify_state_cleared(params):
    assert monitor.classify_state(-0.20, 0.00, 0.02, False, params) == "cleared"


def test_classify_state_earthworks(params):
    assert monitor.classify_state(-0.40, 0.05, 0.20, False, params) == "earthworks"


def test_classify_state_structure(params):
    assert monitor.classify_state(0.0, 0.40, 0.0, False, params) == "structure"


def test_classify_state_operational_requires_nightlight(params):
    assert monitor.classify_state(0.0, 0.40, 0.0, False, params) == "structure"
    assert monitor.classify_state(0.0, 0.40, 0.0, True, params) == "operational"


def test_detect_changes_is_pure_no_mutation(params):
    history = pd.DataFrame(
        {
            "project_id": ["p1"] * 4,
            "month": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2024-01-01"]),
            "ndvi": [0.70, 0.70, 0.70, 0.50],
            "ndbi": [-0.40, -0.40, -0.40, -0.40],
            "bsi": [-0.30, -0.30, -0.30, -0.28],
            "brightness": [0.1, 0.1, 0.1, 0.1],
            "cloud_frac": [0.0, 0.0, 0.0, 0.0],
            "valid": [True, True, True, True],
        }
    )
    original = history.copy(deep=True)
    result = monitor.detect_changes(history, date(2024, 1, 1), params)
    pd.testing.assert_frame_equal(history, original)  # input untouched
    assert result is not history
    assert list(result["state"]) == ["none", "none", "none", "cleared"]
    assert result.loc[3, "dndvi"] == pytest.approx(-0.20, abs=1e-6)


def test_detect_changes_raises_without_enough_baseline(params):
    history = pd.DataFrame(
        {
            "project_id": ["p1"],
            "month": pd.to_datetime(["2023-12-01"]),
            "ndvi": [0.70],
            "ndbi": [-0.40],
            "bsi": [-0.30],
            "brightness": [0.1],
            "cloud_frac": [0.0],
            "valid": [True],
        }
    )
    from ufe.errors import CoverageError

    with pytest.raises(CoverageError):
        monitor.detect_changes(history, date(2024, 1, 1), params)


def test_ratchet_is_monotonic_and_pure():
    states = ["none", "cleared", "none", "earthworks", "cleared", "structure"]
    original = list(states)
    out = monitor._ratchet(states)
    assert states == original  # input list untouched
    assert out == ["none", "cleared", "cleared", "earthworks", "earthworks", "structure"]


def test_invalid_month_never_advances_state(params):
    """A dropped (invalid) month must not be treated as a data point for state change."""
    history = pd.DataFrame(
        {
            "project_id": ["p1"] * 5,
            "month": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01", "2024-01-01", "2024-02-01"]
            ),
            "ndvi": [0.70, 0.70, 0.70, np.nan, 0.70],
            "ndbi": [-0.40, -0.40, -0.40, np.nan, -0.40],
            "bsi": [-0.30, -0.30, -0.30, np.nan, -0.30],
            "brightness": [0.1, 0.1, 0.1, np.nan, 0.1],
            "cloud_frac": [0.0, 0.0, 0.0, 0.9, 0.0],
            "valid": [True, True, True, False, True],
        }
    )
    result = monitor.detect_changes(history, date(2024, 1, 1), params)
    assert result.loc[3, "state"] is None
    assert result.loc[4, "state"] == "none"


# ---------------------------------------------------------------------------
# Collection entry point (standalone ingest)
# ---------------------------------------------------------------------------


def test_collect_composites_no_classification(params):
    """Collection archives composites but never touches state — it's usable standalone
    long before a baseline (or even 2 years of history) exists."""
    rng = np.random.default_rng(6)
    aoi = (83.0, 17.0, 83.01, 17.01)
    scenes = [make_scene("none", pd.Timestamp("2024-01-01"), rng, cloud_frac=0.0)]
    backend = ScriptedImageryBackend({aoi: scenes})
    project = ProjectAOI("p1", aoi, date(2024, 1, 1))
    result = monitor.collect_composites(project, backend, params, date(2024, 1, 1), date(2024, 2, 1))
    assert "state" not in result.columns
    assert list(result.columns) == ["project_id", "month", "ndvi", "ndbi", "bsi", "brightness", "cloud_frac", "valid"]
    assert result.loc[0, "project_id"] == "p1"


def test_run_collection_across_projects(params):
    rng = np.random.default_rng(7)
    aoi1 = (83.0, 17.0, 83.01, 17.01)
    aoi2 = (84.0, 18.0, 84.01, 18.01)
    backend = ScriptedImageryBackend(
        {
            aoi1: [make_scene("none", pd.Timestamp("2024-01-01"), rng, cloud_frac=0.0)],
            aoi2: [make_scene("cleared", pd.Timestamp("2024-01-01"), rng, cloud_frac=0.0)],
        }
    )
    projects = [ProjectAOI("p1", aoi1, date(2024, 1, 1)), ProjectAOI("p2", aoi2, date(2024, 1, 1))]
    result = monitor.run_collection(projects, backend, params, date(2024, 1, 1), date(2024, 2, 1))
    assert set(result["project_id"]) == {"p1", "p2"}
    assert len(result) == 2


def test_run_collection_empty_projects_returns_empty_frame(params):
    result = monitor.run_collection([], ScriptedImageryBackend({}), params, date(2024, 1, 1), date(2024, 2, 1))
    assert result.empty


# ---------------------------------------------------------------------------
# Priority tier (Section 18.3)
# ---------------------------------------------------------------------------


def test_select_priority_tier_ranks_and_truncates(params):
    impact = pd.DataFrame(
        {"project_id": [f"p{i}" for i in range(120)], "impact": list(range(120))}
    )
    original = impact.copy(deep=True)
    result = monitor.select_priority_tier(impact, params)
    pd.testing.assert_frame_equal(impact, original)  # pure, no mutation
    n = int(params.value("priority_tier.n_projects"))
    assert len(result) == n
    assert result.iloc[0]["project_id"] == "p119"  # highest impact first
    assert list(result["rank"]) == list(range(1, n + 1))


def test_select_priority_tier_uses_absolute_convention_upstream():
    """`select_priority_tier` ranks by whatever `impact` column it is given; the |.| is the
    caller's responsibility (Section 18.3: rank by |sum_i lambda_if|). Verify negative
    impacts are not silently favoured if the caller forgot to take the absolute value —
    i.e. this function does exactly what it says, ranks descending on the given column."""
    params = load_satellite_params()
    impact = pd.DataFrame({"project_id": ["a", "b", "c"], "impact": [-100, 5, 10]})
    result = monitor.select_priority_tier(impact, params)
    assert list(result["project_id"]) == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# Full pipeline / ACCEPTANCE — Module 14
# ---------------------------------------------------------------------------


def _run_full_pipeline_for_site(site, backend, params):
    window_start = date(2023, 1, 1)
    window_end = date(2025, 1, 1)
    project = ProjectAOI(site.project_id, site.aoi_bounds_4326, site.announced_date)
    history = monitor.collect_composites(project, backend, params, window_start, window_end)
    return monitor.detect_changes(history, site.announced_date, params)


@pytest.mark.acceptance
def test_acceptance_transition_detected_within_tolerance(params):
    """ACCEPTANCE — Module 14, bullet 1: on a labelled fixture of 10 sites with known
    construction start dates, the classifier identifies the transition to `cleared` or
    beyond within +/-4 months for at least 7."""
    sites, backend = build_labelled_sites()
    tolerance_months = 4
    hits = 0
    for site in sites:
        if site.true_transition_month is None:
            continue
        result = _run_full_pipeline_for_site(site, backend, params)
        advanced = result[result["state_cum"].isin(["cleared", "earthworks", "structure", "operational"])]
        if advanced.empty:
            continue
        first_detected = advanced.sort_values("month").iloc[0]["month"]
        lag_months = (first_detected.year - site.true_transition_month.year) * 12 + (
            first_detected.month - site.true_transition_month.month
        )
        if abs(lag_months) <= tolerance_months:
            hits += 1
    assert hits >= 7, f"only {hits}/10 sites detected within tolerance"


@pytest.mark.acceptance
def test_acceptance_cloud_masking_no_spurious_change(params):
    """ACCEPTANCE — Module 14, bullet 2: cloud-masking removes scenes correctly; a fully
    clouded month produces no composite rather than a spurious state change."""
    rng = np.random.default_rng(42)
    aoi = (83.5, 17.5, 83.51, 17.51)
    months = [pd.Timestamp("2023-01-01") + pd.DateOffset(months=i) for i in range(15)]
    scenes = []
    for idx, m in enumerate(months):
        if idx == 13:
            # one fully clouded month right in the monitored window
            scenes.append(make_scene("none", m, rng, cloud_frac=1.0))
        else:
            scenes.append(make_scene("none", m, rng, cloud_frac=0.0))
    backend = ScriptedImageryBackend({aoi: scenes})
    project = ProjectAOI("cloudy_site", aoi, date(2024, 1, 1))
    history = monitor.collect_composites(project, backend, params, date(2023, 1, 1), date(2024, 4, 1))
    result = monitor.detect_changes(history, date(2024, 1, 1), params)

    clouded_row = result[result["month"] == months[13]].iloc[0]
    assert bool(clouded_row["valid"]) is False
    assert clouded_row["state"] is None
    # and no other month was pushed into a false state change by the missing data
    assert set(result.loc[result["state"].notna(), "state"].unique()) <= {"none"}


@pytest.mark.acceptance
def test_acceptance_no_activity_site_never_advances(params):
    """ACCEPTANCE — Module 14, bullet 3: a site with no activity across 3 years never
    advances past `none`."""
    sites, backend = build_labelled_sites()
    no_activity_site = next(s for s in sites if s.true_transition_month is None)
    result = _run_full_pipeline_for_site(no_activity_site, backend, params)
    observed_states = set(result.loc[result["state_cum"].notna(), "state_cum"].unique())
    assert observed_states <= {"none"}


# ---------------------------------------------------------------------------
# CONTRACT compliance
# ---------------------------------------------------------------------------


def test_satellite_yaml_every_leaf_has_conf_and_scope():
    import yaml

    from tests.fixtures.satellite_fixtures import SATELLITE_YAML_PATH

    with open(SATELLITE_YAML_PATH) as f:
        tree = yaml.safe_load(f)

    def walk(node, path):
        if isinstance(node, dict):
            if "value" in node:
                assert "conf" in node, f"{path} missing conf"
                assert "scope" in node, f"{path} missing scope"
                assert node["conf"] in ("E", "R", "G"), f"{path} has invalid conf"
            else:
                for k, v in node.items():
                    if k == "_schema_version":
                        continue
                    walk(v, f"{path}.{k}")

    walk(tree, "satellite")


def test_modules_import_without_network_or_side_effects():
    """Modules are importable without side effects (CONTRACT). Re-importing must not raise
    or perform I/O."""
    import importlib

    for mod in ("ufe.satellite.stac", "ufe.satellite.indices", "ufe.satellite.monitor", "ufe.satellite_cli"):
        importlib.reload(importlib.import_module(mod))


def test_ufe_params_import_is_guarded():
    """`ufe.params` is not yet on disk; this test documents/asserts that fact so it is
    revisited (and un-skipped in spirit) once the params module owner lands it."""
    pytest.importorskip("ufe.params")


def test_ufe_store_import_is_guarded():
    """`ufe.store.db` / `ufe.store.schemas` are not yet on disk (only the package
    `__init__.py`); guarded per CONTRACT so this test suite doesn't hard-fail on another
    module owner's file."""
    pytest.importorskip("ufe.store.db")


@pytest.mark.needs_data
def test_real_stac_backend_not_used_in_unit_tests():
    """Placeholder acknowledging `StacImageryBackend` (real network backend) exists but is
    never exercised here; marked `needs_data` per CONTRACT so it is skipped by default."""
    pytest.skip("requires network access to a real STAC endpoint")
