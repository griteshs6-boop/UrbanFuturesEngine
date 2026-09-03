"""Tests for the licence audit engine (Section 2.4) and its CLI sub-app."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ufe import licences
from ufe.errors import LicenceViolation
from ufe.licences_cli import app as licences_app

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


# --- classify_licence -----------------------------------------------------------------------


@pytest.fixture()
def policy() -> dict:
    return licences.load_policy()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MIT", "green"),
        ("BSD-3-Clause", "green"),
        ("BSD 3-Clause License", "green"),
        ("Apache-2.0", "green"),
        ("Apache Software License", "green"),
        ("ISC", "green"),
        ("CC0-1.0", "green"),
        ("MPL-2.0", "amber"),
        ("Mozilla Public License 2.0 (MPL 2.0)", "amber"),
        ("LGPL-3.0", "amber"),
        ("GNU Lesser General Public License v3", "amber"),
        ("AGPL-3.0", "red"),
        ("GNU Affero General Public License v3", "red"),
        ("GPL-2.0", "red"),
        ("GPL-3.0", "red"),
        ("SSPL", "red"),
        ("Server Side Public License", "red"),
        ("BUSL-1.1", "red"),
        ("Some Totally Unrecognised Licence String", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_licence(policy, raw, expected):
    assert licences.classify_licence(raw, policy) == expected


def test_classify_licence_lgpl_not_confused_with_gpl(policy):
    """LGPL contains the substring GPL — must not be misclassified as Red."""
    assert licences.classify_licence("LGPL-2.1", policy) == "amber"


def test_classify_licence_agpl_not_confused_with_gpl(policy):
    assert licences.classify_licence("AGPL-3.0-or-later", policy) == "red"


# --- parse_dependencies_md -------------------------------------------------------------------


def test_parse_dependencies_md_covers_every_pyproject_direct_dependency():
    documented = licences.parse_dependencies_md()
    direct = licences.direct_dependency_names_from_pyproject()
    missing = sorted(direct - set(documented.keys()))
    assert missing == [], f"direct dependencies missing a DEPENDENCIES.md row: {missing}"


def test_parse_dependencies_md_classes_are_valid():
    documented = licences.parse_dependencies_md()
    assert documented, "expected at least one parsed row"
    assert set(documented.values()) <= {"green", "amber", "red"}


def test_parse_dependencies_md_hypothesis_is_amber():
    documented = licences.parse_dependencies_md()
    assert documented["hypothesis"] == "amber"


# --- direct_dependency_names_from_pyproject ---------------------------------------------------


def test_direct_dependency_names_includes_dev_extra():
    direct = licences.direct_dependency_names_from_pyproject()
    assert "pandas" in direct
    assert "pytest" in direct
    assert "hypothesis" in direct


# --- audit_dependencies: the real environment should currently pass ------------------------


@pytest.mark.acceptance
def test_real_environment_audit_passes():
    """The environment as actually built for this repo must have zero Red-class packages
    and every direct dependency documented — this is what CI's `ufe licences audit` checks.
    """
    result = licences.audit_dependencies()
    assert result.ok, result.errors


@pytest.mark.acceptance
def test_real_environment_has_no_red_class_findings():
    result = licences.audit_dependencies()
    red = [f for f in result.findings if f.licence_class == "red"]
    assert red == []


# --- ACCEPTANCE — Section 2.4 ----------------------------------------------------------------


@pytest.mark.acceptance
def test_audit_fails_on_injected_agpl_package():
    """`ufe licences audit` exits non-zero when an AGPL package is present in the environment.

    Per the task brief, this is proven by INJECTING a fake distribution record into the
    audit's input rather than actually installing an AGPL package.
    """
    fake = licences.Distribution(name="pandana", version="0.7.0", licence_raw="AGPL-3.0")
    result = licences.audit_dependencies(distributions=[fake])
    assert not result.ok
    assert any("Red-class" in e for e in result.errors)
    assert result.findings[0].licence_class == "red"


@pytest.mark.acceptance
def test_audit_fails_on_injected_gpl_package():
    fake = licences.Distribution(name="r-futures", version="1.0", licence_raw="GPL-2.0")
    result = licences.audit_dependencies(distributions=[fake])
    assert not result.ok
    assert any("Red-class" in e for e in result.errors)


@pytest.mark.acceptance
def test_audit_fails_on_undocumented_direct_dependency():
    """Adding a package without a DEPENDENCIES.md row fails the audit."""
    fake = licences.Distribution(name="totally-new-package", version="1.0", licence_raw="MIT")
    result = licences.audit_dependencies(
        distributions=[fake],
        direct_dependency_names=["totally-new-package"],
    )
    assert not result.ok
    assert any("no DEPENDENCIES.md row" in e for e in result.errors)


def test_audit_does_not_fail_on_undocumented_transitive_dependency():
    """A transitive dependency (not in pyproject.toml's direct list) with no
    DEPENDENCIES.md row must NOT fail the audit — only direct deps require documentation.
    """
    fake = licences.Distribution(name="some-transitive-thing", version="1.0", licence_raw="MIT")
    result = licences.audit_dependencies(distributions=[fake])
    assert result.ok


def test_raise_if_not_ok_raises_licence_violation():
    fake = licences.Distribution(name="pandana", version="0.7.0", licence_raw="AGPL-3.0")
    result = licences.audit_dependencies(distributions=[fake])
    with pytest.raises(LicenceViolation):
        licences.raise_if_not_ok(result)


def test_raise_if_not_ok_noop_when_ok():
    licences.raise_if_not_ok(licences.AuditResult(findings=[], errors=[]))


# --- audit_data_sources ------------------------------------------------------------------------


def test_audit_data_sources_skips_gracefully_when_sources_yaml_absent(tmp_path):
    """config/sources.yaml is owned by another agent and may not exist yet — the data audit
    must skip gracefully rather than fail or error.
    """
    missing_path = tmp_path / "sources.yaml"
    assert not missing_path.exists()
    result = licences.audit_data_sources(sources_path=missing_path)
    assert result.skipped is True
    assert result.ok  # skipped counts as passing, not failing


def test_audit_data_sources_passes_for_a_well_formed_sources_file(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        """
sources:
  openstreetmap:
    licence: "ODbL-1.0"
  esa_worldcover:
    licence: "CC-BY-4.0"
"""
    )
    result = licences.audit_data_sources(sources_path=sources_yaml)
    assert result.ok, result.errors


def test_audit_data_sources_fails_on_unmapped_source(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        """
sources:
  some_mystery_feed:
    licence: "Who Knows"
"""
    )
    result = licences.audit_data_sources(sources_path=sources_yaml)
    assert not result.ok
    assert any("some_mystery_feed" in e for e in result.errors)


def test_audit_data_sources_fails_on_licence_string_disagreement(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        """
sources:
  openstreetmap:
    licence: "MIT"
"""
    )
    result = licences.audit_data_sources(sources_path=sources_yaml)
    assert not result.ok


# --- CLI sub-app ---------------------------------------------------------------------------


def test_cli_audit_exits_zero_on_real_environment():
    result = runner.invoke(licences_app, ["audit"])
    assert result.exit_code == 0, result.output


def test_cli_audit_data_flag_runs_without_crashing_when_sources_yaml_absent():
    result = runner.invoke(licences_app, ["audit", "--data"])
    # config/sources.yaml may not exist yet (owned by another agent) -> skip, not crash/fail.
    assert result.exit_code == 0, result.output
