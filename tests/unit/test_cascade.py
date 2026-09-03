"""Tests for Module 10, cascade (spec Section 14).

The Section 14 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "An OEM with 4,000 jobs and a 1.15 tier-1 ratio injects ~4,600 tier-1 jobs, all in
  industrial-zoned cells, none on the anchor cell unless it is itself industrial-zoned and
  passes the filters" -> ``test_acc_oem_tier1_injection``
  and ``test_acc_non_industrial_anchor_cell_is_not_used``
* "Cascade `p` equals anchor `p x 0.75`" -> ``test_acc_cascade_p_is_anchor_p_times_multiplier``
* "Generation cap: no project has `generation > 2`" -> ``test_acc_generation_cap_binds``
  and ``test_acc_explosive_cascade_terminates``
* "Zero candidate cells produces a logged warning and no injected jobs, not a crash and not
  a fallback placement" -> ``test_acc_zero_candidates_warns_and_skips``

THE BLOCKER, tested rather than worked around
---------------------------------------------
``config/params/archetypes.yaml`` ships 3 of the 22 archetypes, so BOTH shipped cascade
targets (``logistics_park``, ``manufacturing_light``) are undefined and cascade cannot run
against the real config for any anchor. ``test_blocker_*`` pin that behaviour: a clear,
named ``MissingArchetypeError`` identifying the missing key. Every other test loads the real
parameter tree through a ``tmp_path`` overlay (the ``test_l2_shocks`` pattern) that appends
the missing archetypes and the null Section 12.7 firm-logit coefficients. Nothing under
``config/`` is touched and no shipped magnitude is invented — the overlay values are TEST
FIXTURES and are asserted against by recomputation, never as magic constants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import Point

from ufe.errors import MissingParameter
from ufe.layers import cascade as C
from ufe.layers.l2_shocks import EmploymentEffect
from ufe.params import (
    DEFAULT_CITIES_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_PARAMS_DIR,
    load_params,
)
from ufe.store.schemas import SECTORS

from tests.fixtures.synthetic import (  # noqa: F401  (registers the session fixture)
    synthetic_city,
)

CITY = "vizag"

ANCHOR_OPEN_YEAR = 2030

#: TEST FIXTURES, not parameters. They exist only inside a tmp_path overlay because
#: Section 4.3's archetype table was never supplied (see the module docstring).
OVERLAY_ARCHETYPES: dict[str, dict] = {
    # The two targets the SHIPPED cascade blocks point at.
    "logistics_park": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "logistics",
            "median_wage_inr_mo": {"value": 22000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 9000, "conf": "G", "scope": "global"},
        },
        "cascade": None,
    },
    "manufacturing_light": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "manuf_light",
            "median_wage_inr_mo": {"value": 18000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 7000, "conf": "G", "scope": "global"},
        },
        "cascade": None,
    },
    # Section 14.3: the automotive OEM's TWO-TIER cascade, as two entries in a list, both
    # hanging off the same anchor at generation 1. The tier-1 ratio is the ACCEPTANCE
    # block's 1.15.
    "automotive_oem": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "manuf_heavy",
            "median_wage_inr_mo": {"value": 28000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 15000, "conf": "G", "scope": "global"},
        },
        "cascade": [
            {
                "ratio": {"value": 1.15, "low": 0.9, "high": 1.4, "conf": "G", "scope": "global"},
                "radius_m": {"value": 30000, "conf": "G", "scope": "global"},
                "lag_years": {
                    "value": 2, "low": 1, "high": 3, "conf": "G", "dist": "uniform",
                    "scope": "global",
                },
                "target_archetype": "auto_tier1",
            },
            {
                "ratio": {"value": 0.6, "low": 0.4, "high": 0.8, "conf": "G", "scope": "global"},
                "radius_m": {"value": 60000, "conf": "G", "scope": "global"},
                "lag_years": {
                    "value": 4, "low": 3, "high": 5, "conf": "G", "dist": "uniform",
                    "scope": "global",
                },
                "target_archetype": "auto_tier2",
            },
        ],
    },
    "auto_tier1": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "manuf_light",
            "median_wage_inr_mo": {"value": 20000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 10000, "conf": "G", "scope": "global"},
        },
        "cascade": None,
    },
    "auto_tier2": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "manuf_light",
            "median_wage_inr_mo": {"value": 15000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 8000, "conf": "G", "scope": "global"},
        },
        "cascade": None,
    },
    # A deliberately EXPLOSIVE archetype: its cascade target is itself, with a ratio above
    # 1, so every generation is strictly larger than the last. Without the Section 14.2 cap
    # this runs forever. It exists only to prove the cap binds.
    "runaway": {
        "_provenance": {"citation": "structural_assumption"},
        "scale_unit": "jobs",
        "network_effect": {"type": "none"},
        "employment": {
            "permanent_per_unit": {"value": 1.0, "conf": "G", "scope": "global"},
            "sector": "manuf_light",
            "median_wage_inr_mo": {"value": 12000, "conf": "G", "scope": "global"},
            "residential_capture_radius_m": {"value": 5000, "conf": "G", "scope": "global"},
        },
        "cascade": {
            "ratio": {"value": 3.0, "low": 2.0, "high": 4.0, "conf": "G", "scope": "global"},
            "radius_m": {"value": 90000, "conf": "G", "scope": "global"},
            "lag_years": {
                "value": 1, "low": 1, "high": 1, "conf": "G", "dist": "uniform",
                "scope": "global",
            },
            "target_archetype": "runaway",
        },
    },
}

#: Section 12.7's firm-logit coefficients ship null (nothing in the supplied spec states
#: them). Overlaid here so the allocator can run at all — again a fixture, not a parameter.
OVERLAY_FIRM_LOGIT = {
    "enabled": True,
    "coefficients": {
        "c_market": {"value": 0.4, "conf": "G", "scope": "global"},
        "c_labour": {"value": 0.3, "conf": "G", "scope": "global"},
        "c_land": {"value": -0.2, "conf": "G", "scope": "global"},
        "c_agglom": {"value": 0.25, "conf": "G", "scope": "global"},
        "c_freight": {"value": 0.5, "conf": "G", "scope": "global"},
    },
}


# --------------------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------------------


def _overlay(tmp_path_factory, *, archetypes: dict | None = None, cascade_patch: dict | None = None):
    target = tmp_path_factory.mktemp("cascade_params")
    for src in sorted(Path(DEFAULT_PARAMS_DIR).glob("*.yaml")):
        (target / src.name).write_text(src.read_text())

    if archetypes:
        path = target / "archetypes.yaml"
        tree = yaml.safe_load(path.read_text())
        tree.update(archetypes)
        path.write_text(yaml.safe_dump(tree, sort_keys=False))

    if cascade_patch:
        path = target / "cascade.yaml"
        tree = yaml.safe_load(path.read_text())
        tree.update(cascade_patch)
        path.write_text(yaml.safe_dump(tree, sort_keys=False))

    return load_params(
        CITY,
        params_dir=target,
        cities_dir=DEFAULT_CITIES_DIR,
        classes_file=DEFAULT_CLASSES_FILE,
    )


@pytest.fixture(scope="module")
def shipped():
    """The REAL parameter tree, untouched. Cascade cannot resolve a target on it."""
    return load_params(CITY)


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    return _overlay(
        tmp_path_factory,
        archetypes=OVERLAY_ARCHETYPES,
        cascade_patch={"firm_logit": OVERLAY_FIRM_LOGIT},
    )


# --------------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cells(synthetic_city):  # noqa: F811
    """The synthetic city, with the columns the Section 12.7 firm logit needs.

    ``lnA_work`` / ``lnA_retail`` are Layer 1 outputs and the synthetic fixture predates
    Layer 1, so they are derived deterministically here from a landed column rather than
    drawn — this keeps the test seedless and byte-reproducible.
    """
    frame = synthetic_city.cells.copy(deep=True)
    dist = frame["dist_cbd_m"].to_numpy(dtype=float)
    frame["lnA_work"] = np.log1p(np.max(dist) - dist)
    frame["lnA_retail"] = frame["lnA_work"]
    frame["price_land_inr_sqft"] = frame["price_land_inr_sqft"].fillna(
        frame["price_land_inr_sqft"].median()
    )
    return frame


def _industrial_cells(cells: pd.DataFrame, params) -> pd.DataFrame:
    allowed = C._allowed_zone_classes(params)
    return cells.loc[
        cells["zone_class"].isin(allowed) & (cells["util_power"].astype(float) == 1)
    ]


def _anchor(
    cells: pd.DataFrame,
    params,
    *,
    archetype: str,
    jobs: float,
    p_completion: float = 1.0,
    at: Point | None = None,
    project_id: str = "anchor-1",
    generation: int = 0,
) -> pd.DataFrame:
    """An anchor sited on the centroid of the industrial cluster (or at `at`)."""
    if at is None:
        ind = _industrial_cells(cells, params)
        at = Point(float(ind["lon"].mean()), float(ind["lat"].mean()))
    return pd.DataFrame(
        [
            {
                "project_id": project_id,
                "archetype": archetype,
                "geom": at,
                "p_completion": p_completion,
                "open_year": ANCHOR_OPEN_YEAR,
                "anchor_jobs": jobs,
                "generation": generation,
            }
        ]
    )


def _even_allocator(candidates, params, *, sector, jobs, freight_access=None):
    """A deterministic stand-in for the Section 12.7 logit.

    Used where the test is about the CASCADE MECHANISM rather than about the allocation
    rule, so a test does not silently depend on the overlaid firm-logit coefficients.
    """
    n = len(candidates)
    return np.full(n, float(jobs) / n, dtype=float)


# --------------------------------------------------------------------------------------
# THE BLOCKER
# --------------------------------------------------------------------------------------


def test_blocker_shipped_config_defines_only_three_archetypes(shipped):
    """Documents the gap: 3 of 22 archetypes, and both cascade targets are missing."""
    defined = {
        k for k, v in shipped.get("archetypes").items() if not k.startswith("_")
    }
    assert defined == {"metro_rail", "data_centre", "electronics_assembly"}

    targets = {
        entry["target_archetype"]
        for key in defined
        for entry in C.cascade_entries(shipped, key)
    }
    assert targets == {"logistics_park", "manufacturing_light"}
    assert not (targets & defined), "cascade targets are unreachable on the shipped config"


@pytest.mark.parametrize(
    ("anchor_archetype", "missing_target"),
    [("data_centre", "logistics_park"), ("electronics_assembly", "manufacturing_light")],
)
def test_blocker_missing_target_archetype_raises_named_error(
    cells, shipped, anchor_archetype, missing_target
):
    """Both shipped cascade anchors raise a clear, NAMED error identifying the archetype."""
    anchors = _anchor(cells, shipped, archetype=anchor_archetype, jobs=1000.0)
    with pytest.raises(C.MissingArchetypeError) as excinfo:
        C.inject_cascade(cells, anchors, shipped, allocator=_even_allocator)
    message = str(excinfo.value)
    assert f"archetypes.{missing_target}" in message
    assert "archetypes.yaml" in message
    # And it is not a bare Exception: it is a UFEError subclass with its own name.
    assert type(excinfo.value).__name__ == "MissingArchetypeError"


def test_blocker_firm_logit_coefficients_are_null_on_shipped_config(cells, shipped):
    """The independent second gap: Section 12.7's coefficients ship null, so the real
    allocator raises rather than inventing a spread."""
    ind = _industrial_cells(cells, shipped)
    with pytest.raises(MissingParameter) as excinfo:
        C.firm_logit_shares(ind, shipped, sector=SECTORS.index("logistics"), jobs=100.0)
    assert "firm_logit" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 14.1 injection — ACCEPTANCE
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_oem_tier1_injection(cells, overlay):
    """4,000 anchor jobs x a 1.15 tier-1 ratio -> ~4,600 tier-1 jobs, all industrial-zoned."""
    anchor_jobs = 4000.0
    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=anchor_jobs)

    injections, skips = C.inject_cascade(
        cells, anchors, overlay, allocator=_even_allocator
    )

    # Section 14.3: two tiers, both at generation 1, off the same anchor. Not chained.
    assert len(injections) == 2
    assert {i.generation for i in injections} == {1}
    assert [i.tier for i in injections] == [0, 1]
    assert [i.target_archetype for i in injections] == ["auto_tier1", "auto_tier2"]
    assert not skips

    tier1 = injections[0]
    ratio = overlay.value("archetypes.automotive_oem.cascade.0.ratio")
    assert ratio == pytest.approx(1.15)
    assert tier1.ancillary_jobs == pytest.approx(anchor_jobs * ratio)
    assert sum(e.jobs for e in tier1.effects) == pytest.approx(anchor_jobs * ratio)

    # "all in industrial-zoned cells"
    allowed = C._allowed_zone_classes(overlay)
    placed = cells.set_index("h3").loc[[e.cell for e in tier1.effects]]
    assert set(placed["zone_class"]) <= set(allowed)
    assert (placed["util_power"].astype(float) == 1).all()

    # Sector and wage come from the TARGET archetype, not the anchor (Section 14.1).
    assert {e.sector for e in tier1.effects} == {SECTORS.index("manuf_light")}
    wage = overlay.value("archetypes.auto_tier1.employment.median_wage_inr_mo")
    assert {e.median_wage_inr_mo for e in tier1.effects} == {wage}

    # start_year = anchor.open_year + lag
    lag = int(round(overlay.value("archetypes.automotive_oem.cascade.0.lag_years")))
    assert tier1.lag_years == lag
    assert {e.start_year for e in tier1.effects} == {ANCHOR_OPEN_YEAR + lag}


@pytest.mark.acceptance
def test_acc_cascade_p_is_anchor_p_times_multiplier(cells, overlay):
    """Section 14.1: `p_cascade = anchor.p_completion * 0.75`."""
    p_anchor = 0.8
    anchors = _anchor(
        cells, overlay, archetype="automotive_oem", jobs=4000.0, p_completion=p_anchor
    )
    injections, _ = C.inject_cascade(cells, anchors, overlay, allocator=_even_allocator)

    multiplier = overlay.value("cascade.p_multiplier")
    assert multiplier == pytest.approx(0.75)
    for injection in injections:
        assert injection.p == pytest.approx(p_anchor * multiplier)
        for child in injection.children:
            assert child.p_completion == pytest.approx(p_anchor * multiplier)


@pytest.mark.acceptance
def test_acc_non_industrial_anchor_cell_is_not_used(cells, overlay):
    """"none on the anchor cell unless it is itself industrial-zoned and passes the filters"."""
    allowed = C._allowed_zone_classes(overlay)
    excluded = cells.loc[~cells["zone_class"].isin(allowed)]
    anchor_cell = excluded.iloc[0]
    anchors = _anchor(
        cells,
        overlay,
        archetype="automotive_oem",
        jobs=4000.0,
        at=Point(float(anchor_cell["lon"]), float(anchor_cell["lat"])),
    )
    injections, _ = C.inject_cascade(cells, anchors, overlay, allocator=_even_allocator)
    placed = {e.cell for i in injections for e in i.effects}
    assert placed, "the surrounding cluster should still receive jobs"
    assert anchor_cell["h3"] not in placed


@pytest.mark.acceptance
def test_acc_zero_candidates_warns_and_skips(cells, overlay, tmp_path_factory, caplog):
    """Zero candidates -> a logged WARNING, no jobs, no crash, no fallback placement."""
    tiny_radius = dict(OVERLAY_ARCHETYPES)
    tiny = _overlay(
        tmp_path_factory,
        archetypes={
            **tiny_radius,
            "isolated_anchor": {
                "_provenance": {"citation": "structural_assumption"},
                "scale_unit": "jobs",
                "network_effect": {"type": "none"},
                "employment": None,
                "cascade": {
                    "ratio": {"value": 1.0, "conf": "G", "scope": "global"},
                    # One metre: no cell centroid can be inside it.
                    "radius_m": {"value": 1, "conf": "G", "scope": "global"},
                    "lag_years": {"value": 1, "conf": "G", "scope": "global"},
                    "target_archetype": "auto_tier1",
                },
            },
        },
        cascade_patch={"firm_logit": OVERLAY_FIRM_LOGIT},
    )
    # Site the anchor far offshore so nothing is within a metre of it.
    anchors = _anchor(
        cells,
        tiny,
        archetype="isolated_anchor",
        jobs=4000.0,
        at=Point(float(cells["lon"].max()) + 1.0, float(cells["lat"].max()) + 1.0),
    )

    with caplog.at_level(logging.WARNING, logger="ufe.layers.cascade"):
        injections, skips = C.inject_cascade(
            cells, anchors, tiny, allocator=_even_allocator
        )

    assert injections == ()
    assert [s.reason for s in skips] == [C.SKIP_NO_CANDIDATES]
    assert any(
        record.levelno == logging.WARNING and "zero candidate cells" in record.message
        for record in caplog.records
    )


# --------------------------------------------------------------------------------------
# 14.2 the generation cap — ACCEPTANCE
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_generation_cap_binds(cells, overlay):
    """No project exceeds `cascade.max_generation`, and the cap is what stops it."""
    cap = int(overlay.value("cascade.max_generation"))
    assert cap == 2

    anchors = _anchor(cells, overlay, archetype="runaway", jobs=100.0)
    result = C.resolve_cascades(cells, anchors, overlay, allocator=_even_allocator)

    assert result.max_generation_reached == cap
    assert result.generations_run == cap
    generations = {i.generation for i in result.injections}
    assert generations == {1, 2}
    assert max(generations) <= cap
    for injection in result.injections:
        for child in injection.children:
            assert child.generation <= cap

    # The cap is what stopped it: the final pass recorded explicit refusals.
    assert C.SKIP_GENERATION_CAP in {s.reason for s in result.skipped}
    refusals = [s for s in result.skipped if s.reason == C.SKIP_GENERATION_CAP]
    assert refusals and all(s.generation == cap for s in refusals)


@pytest.mark.acceptance
def test_acc_explosive_cascade_terminates(cells, overlay):
    """A deliberately explosive cascade (ratio 3.0, target = itself) TERMINATES.

    Every generation is three times the last and every child is itself a valid anchor, so
    without Section 14.2 this never halts. With the cap it returns, and the jobs injected
    are exactly the two capped generations' worth.
    """
    anchor_jobs = 100.0
    anchors = _anchor(cells, overlay, archetype="runaway", jobs=anchor_jobs)
    result = C.resolve_cascades(cells, anchors, overlay, allocator=_even_allocator)

    ratio = overlay.value("archetypes.runaway.cascade.ratio")
    assert ratio > 1, "the fixture must actually be explosive"

    gen1 = anchor_jobs * ratio
    gen2 = gen1 * ratio
    assert result.diagnostics["jobs_injected"] == pytest.approx(gen1 + gen2)
    assert result.generations_run == int(overlay.value("cascade.max_generation"))


def test_generation_cap_is_checked_before_any_work(cells, overlay):
    """An anchor already AT the cap is refused outright — Section 14.2's exact wording,
    "refuses to cascade a project with generation >= 2"."""
    cap = int(overlay.value("cascade.max_generation"))
    anchors = _anchor(
        cells, overlay, archetype="automotive_oem", jobs=4000.0, generation=cap
    )
    injections, skips = C.inject_cascade(cells, anchors, overlay, allocator=_even_allocator)
    assert injections == ()
    assert [s.reason for s in skips] == [C.SKIP_GENERATION_CAP]


def test_resolve_cascades_raises_if_generation_fails_to_advance(cells, overlay, monkeypatch):
    """The loop's own guard: a child emitted at the parent's generation raises instead of
    spinning. This is the failure mode the cap exists to prevent."""

    real = C.inject_cascade

    def _stuck(cells_, anchors_, params_, **kw):
        injections, skips = real(cells_, anchors_, params_, **kw)
        patched = tuple(
            type(i)(
                **{
                    **i.__dict__,
                    "children": tuple(
                        type(c)(**{**c.__dict__, "generation": 0}) for c in i.children
                    ),
                }
            )
            for i in injections
        )
        return patched, skips

    monkeypatch.setattr(C, "inject_cascade", _stuck)
    anchors = _anchor(cells, overlay, archetype="runaway", jobs=100.0)
    with pytest.raises(ValueError, match="generation did not advance"):
        C.resolve_cascades(cells, anchors, overlay, allocator=_even_allocator)


# --------------------------------------------------------------------------------------
# 14.3 tiering
# --------------------------------------------------------------------------------------


def test_tiering_is_two_entries_not_a_chain(cells, overlay):
    """Section 14.3: "two cascade entries ... both hanging off the same anchor at
    generation 1. Do not chain them.\""""
    entries = C.cascade_entries(overlay, "automotive_oem")
    assert len(entries) == 2

    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=4000.0)
    injections, _ = C.inject_cascade(cells, anchors, overlay, allocator=_even_allocator)

    # Both tiers name the anchor as their parent — tier 2 does not hang off tier 1.
    parents = {c.parent_project_id for i in injections for c in i.children}
    assert parents == {"anchor-1"}
    # Different ratios, radii and lags, per Section 14.3.
    assert injections[0].ratio != injections[1].ratio
    assert injections[0].radius_m != injections[1].radius_m
    assert injections[0].lag_years != injections[1].lag_years


def test_single_mapping_cascade_is_one_tier(overlay):
    assert len(C.cascade_entries(overlay, "runaway")) == 1
    assert C.cascade_entries(overlay, "metro_rail") == ()


# --------------------------------------------------------------------------------------
# the candidate filter and purity
# --------------------------------------------------------------------------------------


def test_candidate_filter_applies_every_section_14_1_clause(cells, overlay):
    radius_m = 30000.0
    ind_centre = _industrial_cells(cells, overlay)
    anchor_geom = Point(float(ind_centre["lon"].mean()), float(ind_centre["lat"].mean()))
    out = C.candidate_cells(
        cells, overlay, anchor_geom=anchor_geom, radius_m=radius_m, crs_metric="EPSG:32644"
    )
    allowed = C._allowed_zone_classes(overlay)
    assert not out.empty
    assert set(out["zone_class"]) <= set(allowed)
    assert (out["util_power"].astype(float) == 1).all()
    assert len(out) < len(cells)


def test_candidate_filter_raises_when_freight_threshold_is_null(cells, overlay):
    """Passing `freight_access` demands the threshold Section 14.1 never states."""
    ind = _industrial_cells(cells, overlay)
    anchor_geom = Point(float(ind["lon"].mean()), float(ind["lat"].mean()))
    with pytest.raises(MissingParameter, match="freight_access_threshold"):
        C.candidate_cells(
            cells,
            overlay,
            anchor_geom=anchor_geom,
            radius_m=30000.0,
            crs_metric="EPSG:32644",
            freight_access=np.ones(len(cells)),
        )


def test_zone_class_map_is_data_not_code(overlay):
    """The Section 14.1 `{industrial, mixed}` vocabulary maps onto the landed `cells`
    vocabulary through YAML, and an unmapped name raises rather than being guessed."""
    assert C._allowed_zone_classes(overlay) == ("ind", "mixed")


def test_injection_is_pure(cells, overlay):
    before = cells.copy(deep=True)
    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=4000.0)
    anchors_before = anchors.copy(deep=True)
    C.inject_cascade(cells, anchors, overlay, allocator=_even_allocator)
    pd.testing.assert_frame_equal(cells, before)
    pd.testing.assert_frame_equal(anchors, anchors_before)


def test_monte_carlo_draw_is_seeded_and_reproducible(cells, overlay):
    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=4000.0)
    kw = dict(monte_carlo=True, allocator=_even_allocator)
    a, _ = C.inject_cascade(
        cells, anchors, overlay, rng=np.random.default_rng(20240101), **kw
    )
    b, _ = C.inject_cascade(
        cells, anchors, overlay, rng=np.random.default_rng(20240101), **kw
    )
    c, _ = C.inject_cascade(
        cells, anchors, overlay, rng=np.random.default_rng(19990101), **kw
    )
    assert [i.ancillary_jobs for i in a] == [i.ancillary_jobs for i in b]
    assert [i.ancillary_jobs for i in a] != [i.ancillary_jobs for i in c]


def test_monte_carlo_requires_an_explicit_rng(cells, overlay):
    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=4000.0)
    with pytest.raises(ValueError, match="explicit rng"):
        C.inject_cascade(
            cells, anchors, overlay, monte_carlo=True, rng=None, allocator=_even_allocator
        )


def test_real_firm_logit_allocates_all_the_jobs(cells, overlay):
    """The Section 12.7 logit path, exercised end to end with overlaid coefficients."""
    anchors = _anchor(cells, overlay, archetype="automotive_oem", jobs=4000.0)
    injections, _ = C.inject_cascade(cells, anchors, overlay)  # default allocator
    tier1 = injections[0]
    assert sum(e.jobs for e in tier1.effects) == pytest.approx(tier1.ancillary_jobs)
    assert all(isinstance(e, EmploymentEffect) for e in tier1.effects)


def test_cascade_does_not_import_ufe_ai():
    """CONTRACT rule 4 / Section 23 item 6: no simulation module imports `ufe.ai`."""
    source = Path(C.__file__).read_text()
    assert "ufe.ai" not in source
