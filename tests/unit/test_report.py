"""Tests for the report renderer and its figure verification (spec Sections 0.2, 20.2,
22.4, 23 items 5 and 7).

* Section 23 item 7 — "the report renderer's figure verification passes on a full generated
  report" -> ``test_acc_figure_verification_passes_on_a_full_report``
  (``@pytest.mark.acceptance``), plus one failure case per clause of the gate.
* Section 23 item 5 — "Every number in the output traces to a snapshot hash, a params hash,
  and a git commit" -> ``test_acc_report_provenance_is_complete``.
* Section 0.2 — confidence tags "are surfaced in report output"
  -> ``test_confidence_tags_are_surfaced``.
* Section 20.2 — a class-default city's every report carries the flag
  -> ``test_acc_class_default_city_report_carries_the_flag``.
* Section 22.4 — ATTRIBUTIONS.md renders into every report footer
  -> ``test_acc_every_report_footer_renders_attributions``.

Offline: no store, no network, no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ufe.api import report as R
from ufe.errors import DataRightsViolation
from ufe.params import (
    DEFAULT_CITIES_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_PARAMS_DIR,
    load_params,
)

CITY = "vizag"
SNAPSHOT_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
REPORT_ID = "vizag-base-2035"

HEADLINE_PATHS = (
    "price.macro.scenarios.base",
    "archetypes.data_centre.employment.permanent_per_unit",
    "archetypes.metro_rail.premium.0.value",
)

RUN_DATA = {
    "zones": {
        "KOM": {"price_change_pct": 14.0, "drivers": {"metro": 62.0}},
        "MDL": {"price_change_pct": 6.5, "drivers": {"metro": 20.0}},
    },
    "city": {"mean_price_change_pct": 10.25, "units_delivered": 41200},
}

FIGURES = (
    R.Figure("zone_price_change", "Zone price change", "map", "figures/zpc.png"),
    R.Figure("driver_decomposition", "Driver decomposition", "chart", "figures/dd.svg"),
    R.Figure("delivery_path", "Delivery path", "chart", "figures/dp.svg"),
)

SECTIONS = (
    R.ReportSection(
        "Headline",
        "Zone KOM rises 14.0 [zones.KOM.price_change_pct] percent against a city mean of "
        "10.25 [city.mean_price_change_pct] percent. [[fig:zone_price_change]]",
    ),
    R.ReportSection(
        "Drivers",
        "62.0 [zones.KOM.drivers.metro] percent of KOM's rise is metro exposure, versus "
        "20.0 [zones.MDL.drivers.metro] percent in MDL. [[fig:driver_decomposition]]",
    ),
    R.ReportSection(
        "Supply",
        "41200 [city.units_delivered] units are delivered over the horizon. "
        "[[fig:delivery_path]]",
    ),
)


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def class_default_params(tmp_path_factory):
    """A city on class defaults. Built in a tmp dir; `config/` is untouched."""
    cities = tmp_path_factory.mktemp("cities")
    config = yaml.safe_load((Path(DEFAULT_CITIES_DIR) / f"{CITY}.yaml").read_text())
    config["city_id"] = "demoville"
    config["calibration_level"] = "class_default"
    (cities / "demoville.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return load_params(
        "demoville",
        params_dir=DEFAULT_PARAMS_DIR,
        cities_dir=cities,
        classes_file=DEFAULT_CLASSES_FILE,
    )


def _render(params, **overrides):
    kwargs = dict(
        report_id=REPORT_ID,
        title="Vizag base case, 2035",
        sections=SECTIONS,
        figures=FIGURES,
        run_data=RUN_DATA,
        snapshot_hash=SNAPSHOT_HASH,
        headline_parameter_paths=HEADLINE_PATHS,
    )
    kwargs.update(overrides)
    return R.render_report(params, **kwargs)


# --------------------------------------------------------------------------------------
# Section 23 item 7 — figure verification
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_figure_verification_passes_on_a_full_report(params):
    """Section 23 item 7: "the report renderer's figure verification passes on a full
    generated report"."""
    verification = R.verify_report(SECTIONS, FIGURES, RUN_DATA, params)
    assert verification.figures_checked == len(FIGURES)
    assert verification.references_checked == 5
    assert verification.numbers_checked == 5


def test_verification_fails_on_a_figure_referenced_but_not_generated(params):
    """Clause 1: every figure referenced by the narrative must exist."""
    sections = SECTIONS + (R.ReportSection("Extra", "See [[fig:ghost]]."),)
    with pytest.raises(R.FigureVerificationError, match="ghost"):
        R.verify_report(sections, FIGURES, RUN_DATA, params)


def test_verification_fails_on_a_figure_generated_but_not_referenced(params):
    """Clause 2: every figure generated must be referenced."""
    figures = FIGURES + (R.Figure("orphan", "Orphan", "chart", "figures/o.svg"),)
    with pytest.raises(R.FigureVerificationError, match="orphan"):
        R.verify_report(SECTIONS, figures, RUN_DATA, params)


def test_verification_fails_on_a_quoted_number_that_does_not_match(params):
    """Clause 3: every number quoted must match the run data."""
    sections = (
        R.ReportSection(
            "Headline",
            "Zone KOM rises 22.0 [zones.KOM.price_change_pct] percent. "
            "[[fig:zone_price_change]]",
        ),
        SECTIONS[1],
        SECTIONS[2],
    )
    with pytest.raises(R.FigureVerificationError) as excinfo:
        R.verify_report(sections, FIGURES, RUN_DATA, params)
    assert "22.0" in str(excinfo.value) and "14.0" in str(excinfo.value)


def test_verification_fails_on_a_field_path_that_does_not_exist(params):
    sections = (
        R.ReportSection(
            "Headline",
            "Zone ZZZ rises 14.0 [zones.ZZZ.price_change_pct] percent. "
            "[[fig:zone_price_change]]",
        ),
        SECTIONS[1],
        SECTIONS[2],
    )
    with pytest.raises(R.FigureVerificationError, match="does not exist"):
        R.verify_report(sections, FIGURES, RUN_DATA, params)


def test_verification_fails_on_a_reference_with_no_number_in_front_of_it(params):
    """An untraceable claim is a build failure (Section 23 item 5)."""
    sections = (
        R.ReportSection(
            "Headline",
            "KOM rises sharply [zones.KOM.price_change_pct]. [[fig:zone_price_change]]",
        ),
        SECTIONS[1],
        SECTIONS[2],
    )
    with pytest.raises(R.FigureVerificationError, match="no numeric figure"):
        R.verify_report(sections, FIGURES, RUN_DATA, params)


def test_verification_fails_on_a_non_numeric_run_data_value(params):
    run_data = {**RUN_DATA, "city": {**RUN_DATA["city"], "units_delivered": "many"}}
    with pytest.raises(R.FigureVerificationError, match="not numeric"):
        R.verify_report(SECTIONS, FIGURES, run_data, params)


def test_verification_fails_on_duplicate_figure_ids(params):
    figures = FIGURES + (FIGURES[0],)
    with pytest.raises(R.FigureVerificationError, match="duplicate figure id"):
        R.verify_report(SECTIONS, figures, RUN_DATA, params)


def test_verification_accepts_rounding_within_tolerance(params):
    """The tolerances come from YAML, not from a literal in the renderer."""
    rel_tol = params.value("api.report.figure_verification.rel_tol")
    abs_tol = params.value("api.report.figure_verification.abs_tol")
    actual = RUN_DATA["zones"]["KOM"]["price_change_pct"]
    inside = actual + max(abs_tol, rel_tol * abs(actual)) / 2
    sections = (
        R.ReportSection(
            "Headline",
            f"KOM rises {inside:.4f} [zones.KOM.price_change_pct] percent. "
            "[[fig:zone_price_change]]",
        ),
        SECTIONS[1],
        SECTIONS[2],
    )
    R.verify_report(sections, FIGURES, RUN_DATA, params)


def test_render_refuses_to_emit_an_unverified_report(params):
    """Verification runs BEFORE rendering: a bad report never reaches a renderer."""
    bad = (R.ReportSection("Headline", "KOM rises 99.9 [zones.KOM.price_change_pct]."),)
    with pytest.raises(R.FigureVerificationError):
        _render(params, sections=bad, figures=())


def test_figure_reference_syntax_does_not_collide_with_field_reference_syntax(params):
    """`[[fig:x]]` and `[field.path]` are different grammars and must not cross-match."""
    assert R.referenced_figure_ids(SECTIONS) == [
        "zone_price_change",
        "driver_decomposition",
        "delivery_path",
    ]
    assert R.ANY_REF_RE.findall("[[fig:zone_price_change]]") == []


# --------------------------------------------------------------------------------------
# Section 23 item 5 / Section 0.2 — provenance
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_report_provenance_is_complete(params):
    """"Every number in the output traces to a snapshot hash, a params hash, and a git
    commit"."""
    report = _render(params)
    provenance = report.provenance
    assert provenance.snapshot_hash == SNAPSHOT_HASH
    assert provenance.params_hash == params.hash
    assert provenance.git_commit and provenance.git_commit != ""
    assert provenance.city == CITY
    assert provenance.calibration_level == "full"
    # All three appear in the rendered document, not only on the object.
    for value in (SNAPSHOT_HASH, params.hash, provenance.git_commit):
        assert value in report.markdown


def test_git_commit_is_read_from_the_checkout():
    """This working tree currently has no commit on its branch ref, so the reader honestly
    returns "unknown" rather than inventing one. Either outcome is valid; a fabricated
    hash is not."""
    commit = R.git_commit()
    assert commit == "unknown" or (
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    )


SHA = "0" * 39 + "1"


def test_git_commit_reads_a_loose_ref(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(SHA + "\n")
    assert R.git_commit(tmp_path) == SHA


def test_git_commit_reads_packed_refs(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{SHA} refs/heads/main\n"
    )
    assert R.git_commit(tmp_path) == SHA


def test_git_commit_reads_a_detached_head(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text(SHA + "\n")
    assert R.git_commit(tmp_path) == SHA


def test_git_commit_is_unknown_outside_a_checkout(tmp_path):
    """It never invents a plausible-looking hash."""
    assert R.git_commit(tmp_path) == "unknown"


def test_confidence_tags_are_surfaced(params):
    """Section 0.2: the E/R/G tags "are surfaced in report output"."""
    report = _render(params)
    tags = report.provenance.confidence_tags
    assert set(tags) == set(HEADLINE_PATHS)
    for path, tag in tags.items():
        assert tag == params.conf(path)
        assert tag in ("E", "R", "G")
        assert f"`{path}`: **{tag}**" in report.markdown


def test_confidence_tags_raise_on_an_unknown_tag():
    """A stripped or mistyped `conf:` surfaces at report build, not in front of a client.

    `ufe.params` refuses to load a tree with an invalid `conf`, so the only way to reach
    this branch is with a stand-in — which is the point: the renderer does not trust its
    input to have been validated upstream.
    """

    class _FakeParams:
        def get(self, _):
            return ["E", "R", "G"]

        def conf(self, _):
            return "X"

    with pytest.raises(R.FigureVerificationError, match="conf="):
        R.confidence_tags(_FakeParams(), ["some.path"])


# --------------------------------------------------------------------------------------
# Section 20.2 — the calibration flag
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_class_default_city_report_carries_the_flag(class_default_params):
    """Section 20.2: a class-default city "is a demonstration, not a product" and that flag
    "must appear in every report it produces"."""
    report = _render(class_default_params)
    assert report.provenance.calibration_level == "class_default"
    assert report.provenance.is_class_default
    assert report.calibration_flag is not None
    assert "demonstration, not a product" in report.calibration_flag
    # In the body AND in the footer, so it survives a first-page-only reading.
    assert report.calibration_flag in report.markdown
    assert report.calibration_flag in report.footer
    assert report.markdown.index(report.calibration_flag) < report.markdown.index("## Provenance")


def test_a_fully_calibrated_city_carries_no_flag(params):
    report = _render(params)
    assert report.provenance.calibration_level == "full"
    assert report.calibration_flag is None
    assert "demonstration, not a product" not in report.markdown


# --------------------------------------------------------------------------------------
# Section 22.4 — attribution
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_every_report_footer_renders_attributions(params, class_default_params):
    """Section 22.4: ATTRIBUTIONS.md renders into every report footer."""
    on_disk = Path(R.ATTRIBUTIONS_PATH).read_text(encoding="utf-8").strip()
    for tree in (params, class_default_params):
        report = _render(tree)
        assert report.attributions == on_disk
        assert report.attributions in report.footer
        assert report.attributions in report.markdown
        assert "OpenStreetMap" in report.footer


def test_about_page_renders_attributions(params):
    """Section 22.4: "It renders into: the product about page ..."."""
    page = R.about_page(params)
    assert page["product"] == "Urban Futures Engine"
    assert page["exposure_policy"] == "produced_work_only"
    assert page["attributions"] == Path(R.ATTRIBUTIONS_PATH).read_text(encoding="utf-8").strip()


def test_report_build_fails_when_a_source_cannot_be_attributed(params):
    """Section 22.4: "A report build that cannot resolve an attribution for a source it
    used must fail"."""
    with pytest.raises(DataRightsViolation, match="no_such_source"):
        _render(params, attribution_sources=["no_such_source"])


def test_named_attribution_sources_render_only_those_sources(params):
    report = _render(params, attribution_sources=["openstreetmap"])
    assert "OpenStreetMap" in report.attributions
    assert len(report.attributions.splitlines()) == 1


def test_attribution_block_falls_back_when_attributions_md_is_absent(tmp_path):
    text = R.attribution_block(attributions_path=tmp_path / "nope.md")
    assert "OpenStreetMap" in text


# --------------------------------------------------------------------------------------
# rendering mechanics
# --------------------------------------------------------------------------------------


def test_rendered_markdown_strips_field_references_and_inlines_figures(params):
    report = _render(params)
    assert "[zones.KOM.price_change_pct]" not in report.markdown
    assert "14.0" in report.markdown
    assert "[[fig:zone_price_change]]" not in report.markdown
    assert "![Zone price change](figures/zpc.png)" in report.markdown


def test_report_carries_no_cell_level_data(params):
    """A report is a Produced Work (Section 22.1): no grid, no OSM column, ever."""
    from ufe.rights import CELLS_OSM_DERIVED_RAW_COLUMNS, assert_exposable

    report = _render(params)
    for column in CELLS_OSM_DERIVED_RAW_COLUMNS:
        assert column not in report.markdown
    assert_exposable(report.markdown.split())


def test_report_is_deterministic(params):
    """CONTRACT rule 5: same inputs -> byte-identical output."""
    assert _render(params).markdown == _render(params).markdown


def test_strip_references_keeps_the_numbers():
    assert R.strip_references("up 14.0 [a.b] percent") == "up 14.0 percent"
