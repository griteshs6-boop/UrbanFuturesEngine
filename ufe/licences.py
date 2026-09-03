"""Software licence audit engine (Section 2.4).

`ufe licences audit` (see `ufe/licences_cli.py`) enumerates installed distributions, classifies
each one's licence against `config/licence_policy.yaml`, and fails when:

  * any distribution — direct or transitive — is Red-class (AGPL/GPL/SSPL/BUSL/...), or
  * a *direct* dependency (from `pyproject.toml`) has no row in `DEPENDENCIES.md`, or
  * a *direct* dependency's licence string does not match anything in the policy table
    ("unknown" is treated as a failed audit, not a pass — Section 2.4's rule is "no dependency
    is added without a recorded licence check").

Everything here is a pure function over injectable inputs (a list of `Distribution` records,
a policy dict, parsed `DEPENDENCIES.md` rows) so tests can exercise the Red-class failure path
by constructing a fake `Distribution` rather than installing a GPL package into the environment.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from ufe.errors import LicenceViolation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "licence_policy.yaml"
DEFAULT_DEPENDENCIES_PATH = REPO_ROOT / "DEPENDENCIES.md"
DEFAULT_DATA_LICENCES_PATH = REPO_ROOT / "config" / "data_sources_licences.yaml"
DEFAULT_SOURCES_PATH = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

LicenceClass = str  # "green" | "amber" | "red" | "unknown"


@dataclass(frozen=True)
class Distribution:
    """A minimal, injectable stand-in for an `importlib.metadata.Distribution`.

    Real distributions are converted to this shape by `get_installed_distributions()`. Tests
    build these directly to simulate a package (e.g. a GPL one) without installing it.
    """

    name: str
    version: str
    licence_raw: str


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    licence_raw: str
    licence_class: LicenceClass
    is_direct_dependency: bool
    documented_in_dependencies_md: bool


@dataclass(frozen=True)
class AuditResult:
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalise_name(name: str) -> str:
    """PEP 503-style normalisation so `PyYAML`, `pyyaml`, `py-yaml` all compare equal."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _extract_licence_string(dist: importlib_metadata.Distribution) -> str:
    """Prefer `License ::` trove classifiers (short, canonical); fall back to the raw
    `License` / `License-Expression` metadata field, which is sometimes the full licence text.
    """
    md = dist.metadata
    classifiers = [c for c in (md.get_all("Classifier") or []) if c.startswith("License")]
    license_field = md.get("License-Expression") or md.get("License") or ""
    parts: list[str] = []
    if classifiers:
        parts.extend(c.split("::")[-1].strip() for c in classifiers)
    if license_field and (not classifiers or len(license_field) <= 60):
        parts.append(license_field.strip())
    if not parts and license_field:
        parts.append(license_field.strip())
    return " | ".join(p for p in parts if p)


def get_installed_distributions() -> list[Distribution]:
    """Enumerate every distribution installed in the current environment."""
    out: list[Distribution] = []
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or "unknown"
        version = dist.version or "unknown"
        out.append(Distribution(name=name, version=version, licence_raw=_extract_licence_string(dist)))
    return out


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def classify_licence(raw: str, policy: dict) -> LicenceClass:
    """Classify a raw licence string into green/amber/red/unknown per `licence_policy.yaml`.

    Step 1: exact, case-insensitive match against each class's `exact` list.
    Step 2: ordered regex `rules` (first match wins — this is why AGPL/LGPL/MPL/EPL are checked
    before the bare "GPL" pattern in the policy file).
    Step 3: "unknown" if nothing matched.
    """
    if not raw:
        return "unknown"
    raw_lower = raw.strip().lower()

    classes = policy.get("classes", {})
    for cls_name, cls_def in classes.items():
        for exact in cls_def.get("exact", []) or []:
            if exact.strip().lower() == raw_lower:
                return cls_name

    for rule in policy.get("rules", []) or []:
        pattern = rule["pattern"]
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return rule["class"]

    return "unknown"


_MD_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


def parse_dependencies_md(path: Path = DEFAULT_DEPENDENCIES_PATH) -> dict[str, str]:
    """Parse the markdown tables in `DEPENDENCIES.md` into `{normalised_name: class}`.

    Recognises any row of the form `| package | version | licence | Class | why |`
    (class cell may be wrapped in `**bold**`) and skips header/separator rows.
    """
    if not path.exists():
        return {}

    documented: dict[str, str] = {}
    valid_classes = {"green", "amber", "red"}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _MD_ROW_RE.match(line.strip())
            if not m:
                continue
            cells = [c.strip() for c in m.group(1).split("|")]
            if len(cells) < 4:
                continue
            name_cell, _version_cell, _licence_cell, class_cell = cells[0], cells[1], cells[2], cells[3]
            if not name_cell or set(name_cell) <= {"-", ":"}:
                continue  # separator row
            if name_cell.lower() in {"package"}:
                continue  # header row
            class_clean = class_cell.strip("* ").lower()
            if class_clean not in valid_classes:
                continue
            documented[_normalise_name(name_cell)] = class_clean
    return documented


def direct_dependency_names_from_pyproject(path: Path = DEFAULT_PYPROJECT_PATH) -> set[str]:
    """Extract direct dependency names (main + `dev` extra) from `pyproject.toml`."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py3.11+ always has tomllib
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    names: set[str] = set()
    specs: list[str] = list(data.get("project", {}).get("dependencies", []))
    for extra_deps in data.get("project", {}).get("optional-dependencies", {}).values():
        specs.extend(extra_deps)

    spec_re = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
    for spec in specs:
        m = spec_re.match(spec.strip())
        if m:
            names.add(_normalise_name(m.group(1)))
    return names


def audit_dependencies(
    *,
    distributions: Sequence[Distribution] | None = None,
    policy_path: Path = DEFAULT_POLICY_PATH,
    dependencies_md_path: Path = DEFAULT_DEPENDENCIES_PATH,
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH,
    direct_dependency_names: Iterable[str] | None = None,
) -> AuditResult:
    """Run the Section 2.4 software licence audit.

    `distributions`, when provided, REPLACES the live environment enumeration — this is the
    injection point tests use to simulate a Red-class package being present without installing
    one. `direct_dependency_names`, when provided, overrides the set read from `pyproject.toml`
    (used by tests that simulate a brand-new undocumented dependency).
    """
    policy = load_policy(policy_path)
    dists = list(distributions) if distributions is not None else get_installed_distributions()
    documented = parse_dependencies_md(dependencies_md_path)
    direct = (
        {_normalise_name(n) for n in direct_dependency_names}
        if direct_dependency_names is not None
        else direct_dependency_names_from_pyproject(pyproject_path)
    )

    findings: list[Finding] = []
    errors: list[str] = []

    for dist in dists:
        norm = _normalise_name(dist.name)
        cls = classify_licence(dist.licence_raw, policy)
        is_direct = norm in direct
        is_documented = norm in documented
        findings.append(
            Finding(
                package=dist.name,
                version=dist.version,
                licence_raw=dist.licence_raw,
                licence_class=cls,
                is_direct_dependency=is_direct,
                documented_in_dependencies_md=is_documented,
            )
        )

        if cls == "red":
            errors.append(
                f"{dist.name} {dist.version} is Red-class ({dist.licence_raw!r}) — "
                "not permitted as a dependency (Section 2.4)."
            )
        elif cls == "unknown" and is_direct:
            errors.append(
                f"{dist.name} {dist.version} has an unrecognised licence string "
                f"({dist.licence_raw!r}); add it to config/licence_policy.yaml with a class."
            )

        if is_direct and not is_documented:
            errors.append(
                f"{dist.name} is a direct dependency but has no DEPENDENCIES.md row."
            )

    return AuditResult(findings=findings, errors=errors)


def audit_data_sources(
    *,
    sources_path: Path = DEFAULT_SOURCES_PATH,
    data_licences_path: Path = DEFAULT_DATA_LICENCES_PATH,
) -> AuditResult:
    """Run the Section 22.2 data-source licence audit.

    Cross-checks every source declared in `config/sources.yaml` (owned by the ingestion agent)
    against the known-licence table in `config/data_sources_licences.yaml`.

    `config/sources.yaml` does not exist yet at the time this module was written. Expected
    shape, documented here so both sides can code against it independently:

        sources:
          <source_key>:
            licence: "<licence string, matched case-insensitively against a key or alias
                       in config/data_sources_licences.yaml>"
            ... (other ingester-owned fields, ignored by this audit)

    When the file is absent, the audit is SKIPPED (not failed) — `AuditResult.skipped` is
    True — so that CI does not fail merely because the ingestion agent hasn't landed yet.
    Once the file exists, a source with an unrecognised licence string fails the audit.
    """
    if not sources_path.exists():
        return AuditResult(
            skipped=True,
            skip_reason=f"{sources_path} does not exist yet — data-source audit skipped.",
        )

    with open(data_licences_path, "r", encoding="utf-8") as fh:
        known = yaml.safe_load(fh) or {}
    known_sources = known.get("sources", {})

    # Build a lookup from every canonical key and alias -> canonical key.
    lookup: dict[str, str] = {}
    for key, entry in known_sources.items():
        lookup[_normalise_name(key)] = key
        for alias in entry.get("aliases", []) or []:
            lookup[_normalise_name(alias)] = key

    with open(sources_path, "r", encoding="utf-8") as fh:
        declared = yaml.safe_load(fh) or {}
    declared_sources = declared.get("sources", {})

    errors: list[str] = []
    findings: list[Finding] = []
    for source_key, source_def in declared_sources.items():
        licence_str = (source_def or {}).get("licence", "")
        norm = _normalise_name(source_key)
        canonical = lookup.get(norm) or lookup.get(_normalise_name(licence_str))
        ok = canonical is not None and (
            not licence_str
            or _normalise_name(licence_str) == _normalise_name(known_sources[canonical]["licence"])
            or norm in lookup
        )
        # Recognise the source primarily by key; if the declared licence string disagrees with
        # the table's recorded licence for that key, that is also a failure.
        if canonical is None:
            errors.append(
                f"data source '{source_key}' (licence {licence_str!r}) is not mapped in "
                "config/data_sources_licences.yaml."
            )
            continue
        expected_licence = known_sources[canonical]["licence"]
        if licence_str and _normalise_name(licence_str) != _normalise_name(expected_licence):
            errors.append(
                f"data source '{source_key}' declares licence {licence_str!r} but "
                f"config/data_sources_licences.yaml records {expected_licence!r} for it."
            )
        findings.append(
            Finding(
                package=source_key,
                version="",
                licence_raw=licence_str or expected_licence,
                licence_class="data-source",
                is_direct_dependency=True,
                documented_in_dependencies_md=True,
            )
        )

    return AuditResult(findings=findings, errors=errors)


def raise_if_not_ok(result: AuditResult) -> None:
    if not result.ok:
        raise LicenceViolation("; ".join(result.errors))
