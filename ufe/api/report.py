"""Report rendering and figure verification (spec Sections 0.2, 20.2, 22.4, 23 items 5 & 7).

What a report is here
---------------------
A :class:`Report` is a Produced Work in the ODbL sense (Section 22.1): prose, figures and
computed numbers, never the underlying grid. It is built from

* ``sections`` — narrative prose. Each section may reference figures as ``[[fig:ID]]`` and
  must attach a ``[field.path]`` reference to every number it quotes, exactly as Prompt G
  is instructed to (Section 17.9). The prose may come from :mod:`ufe.ai.narrate` or from a
  human; this module does not care and does not import ``ufe.ai``.
* ``figures`` — the figures actually generated for the run.
* ``run_data`` — the run's output object, the ground truth every quoted number is checked
  against.
* ``provenance`` — Section 23 item 5: "Every number in the output traces to a snapshot
  hash, a params hash, and a git commit," plus the Section 0.2 confidence tags of the
  parameters behind the headline numbers, plus the Section 20.2 ``calibration_level``.

Figure verification (Section 23 item 7)
---------------------------------------
:func:`verify_report` is a hard gate with three clauses, and it raises
:class:`FigureVerificationError` rather than warning:

1. **Every figure referenced by the narrative exists.** A ``[[fig:ID]]`` with no matching
   figure is a broken report.
2. **Every figure generated is referenced.** An orphan figure means either the narrative
   silently dropped a finding or a figure was built for nothing; both are bugs.
3. **Every number quoted in the narrative matches the run data.** Each ``N [field.path]``
   pair is resolved against ``run_data`` and compared within the tolerances in
   ``config/params/api.yaml``.

Clause 3 is the same check :func:`ufe.ai.narrate.verify_narrative` performs on Prompt G
output. This module re-implements the resolution and comparison locally rather than
importing it, for two reasons: a report's prose need not have come from an LLM, and
``ufe.api.report`` must stay importable in an environment with no ``ufe.ai`` prompt assets.
The regexes and tolerance semantics are deliberately identical.

Attribution (Section 22.4)
--------------------------
``ATTRIBUTIONS.md`` renders into the product about page AND every report footer. The text
comes from :func:`ufe.rights.get_attribution_text`; :func:`attribution_block` reads the
generated ``ATTRIBUTIONS.md`` when it is present and falls back to rendering from
``config/data_sources_licences.yaml``. A report whose sources cannot all be attributed
fails to build — ``get_attribution_text`` raises ``DataRightsViolation``.

No numeric literal appears in this module beyond ``0`` and ``1``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ufe.errors import UFEError
from ufe.params import Params
from ufe.rights import get_attribution_text

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTIONS_PATH = REPO_ROOT / "ATTRIBUTIONS.md"

# --- parameter paths ----------------------------------------------------------------------

P_REL_TOL = "api.report.figure_verification.rel_tol"
P_ABS_TOL = "api.report.figure_verification.abs_tol"
P_REQUIRE_EVERY_FIGURE_REFERENCED = (
    "api.report.figure_verification.require_every_figure_referenced"
)
P_REQUIRE_EVERY_REFERENCE_RESOLVED = (
    "api.report.figure_verification.require_every_reference_resolved"
)
P_CALIBRATION_FLAG_LEVEL = "api.report.calibration.flag_level"
P_CALIBRATION_FLAG_TEXT = "api.report.calibration.flag_text"
P_CONFIDENCE_TAGS = "api.report.confidence_tags"
P_FOOTER_ATTRIBUTIONS = "api.report.footer.include_attributions"
P_FOOTER_PROVENANCE = "api.report.footer.include_provenance"

ZERO = 0
ONE = 1

# --- the two reference grammars -------------------------------------------------------------

#: ``[[fig:hex_price_change]]`` — a narrative reference to a figure.
FIGURE_REF_RE = re.compile(r"\[\[fig:(?P<figure_id>[A-Za-z0-9_.-]+)\]\]")

#: ``+14% [zones.KOM.base_case_pct]`` — a quoted number and the field it came from.
#: Identical to `ufe.ai.narrate._FIGURE_REF_RE` (Section 17.9).
NUMBER_REF_RE = re.compile(r"(?P<figure>[+-]?\d+(?:\.\d+)?)\s*%?\s*\[(?P<path>[A-Za-z0-9_.]+)\]")

#: Any bracketed field reference, so a bracket with no number in front of it is caught too.
ANY_REF_RE = re.compile(r"(?<!\[)\[(?P<path>[A-Za-z0-9_.]+)\](?!\])")


class FigureVerificationError(UFEError):
    """The Section 23 item 7 figure-verification gate failed. The report build must fail."""


# --------------------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """One generated figure. `uri` is whatever the renderer emitted — never raw grid data."""

    figure_id: str
    title: str
    kind: str
    uri: str | None = None
    caption: str = ""


@dataclass(frozen=True)
class ReportSection:
    """One narrative section. `body` carries `[[fig:ID]]` and `N [field.path]` references."""

    name: str
    body: str


@dataclass(frozen=True)
class Provenance:
    """Section 23 item 5 plus Section 0.2 and Section 20.2."""

    city: str
    snapshot_hash: str
    params_hash: str
    git_commit: str
    calibration_level: str
    #: Section 0.2: the E/R/G tag of every parameter behind a headline number.
    confidence_tags: Mapping[str, str] = field(default_factory=dict)
    class_defaults_applied: tuple[str, ...] = ()
    params_source_files: tuple[str, ...] = ()

    @property
    def is_class_default(self) -> bool:
        return self.calibration_level == "class_default"


@dataclass(frozen=True)
class FigureVerification:
    """The outcome of the Section 23 item 7 gate. Only ever constructed on success."""

    figures_checked: int
    references_checked: int
    numbers_checked: int


@dataclass(frozen=True)
class Report:
    """A rendered, verified report."""

    report_id: str
    title: str
    city: str
    sections: tuple[ReportSection, ...]
    figures: tuple[Figure, ...]
    provenance: Provenance
    attributions: str
    footer: str
    markdown: str
    verification: FigureVerification
    calibration_flag: str | None = None


# --------------------------------------------------------------------------------------
# provenance (Section 23 item 5, Section 0.2, Section 20.2)
# --------------------------------------------------------------------------------------


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    """The current git commit, read straight from `.git` — no subprocess, no network.

    Returns the literal string ``"unknown"`` when the tree is not a git checkout, so that a
    report always carries the field. It never invents a plausible-looking hash.
    """
    git_dir = Path(repo_root) / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if not text.startswith("ref:"):
        return text
    ref = text.split(":", ONE)[ONE].strip()
    loose = git_dir / ref
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == ONE + ONE and parts[ONE] == ref:
                return parts[ZERO]
    return "unknown"


def confidence_tags(params: Params, paths: Iterable[str]) -> dict[str, str]:
    """Section 0.2: the E/R/G tag of each parameter path behind a headline number.

    Raises if a path resolves to a tag outside the known vocabulary, so a stripped or
    mistyped `conf:` field surfaces at report build rather than in front of a client.
    """
    known = set(params.get(P_CONFIDENCE_TAGS))
    tags: dict[str, str] = {}
    for path in paths:
        tag = params.conf(path)
        if tag not in known:
            raise FigureVerificationError(
                f"parameter {path!r} carries conf={tag!r}, which is not one of {sorted(known)}; "
                "Section 0.2 confidence tags are data and must not be stripped."
            )
        tags[path] = tag
    return tags


def build_provenance(
    params: Params,
    *,
    snapshot_hash: str,
    headline_parameter_paths: Iterable[str] = (),
    repo_root: Path = REPO_ROOT,
) -> Provenance:
    """Assemble the provenance block every report carries (Section 23 item 5)."""
    manifest = params.manifest()
    return Provenance(
        city=str(manifest["city"]),
        snapshot_hash=snapshot_hash,
        params_hash=str(manifest["params_hash"]),
        git_commit=git_commit(repo_root),
        calibration_level=str(manifest.get("calibration_level")),
        confidence_tags=confidence_tags(params, headline_parameter_paths),
        class_defaults_applied=tuple(manifest.get("class_defaults_applied") or ()),
        params_source_files=tuple(manifest.get("source_files") or ()),
    )


def calibration_flag(params: Params, provenance: Provenance) -> str | None:
    """Section 20.2: the flag a class-default city's every report must carry."""
    if provenance.calibration_level != params.get(P_CALIBRATION_FLAG_LEVEL):
        return None
    return str(params.get(P_CALIBRATION_FLAG_TEXT)).strip()


# --------------------------------------------------------------------------------------
# attribution (Section 22.4)
# --------------------------------------------------------------------------------------


def attribution_block(
    source_keys: Iterable[str] | None = None,
    *,
    attributions_path: Path = ATTRIBUTIONS_PATH,
) -> str:
    """The attribution text for the about page and every report footer (Section 22.4).

    With no `source_keys`, returns the generated ``ATTRIBUTIONS.md`` verbatim if it is on
    disk (Section 22.4: "`ATTRIBUTIONS.md` is generated, not hand-maintained"), else the
    full rendered block from ``config/data_sources_licences.yaml``. With `source_keys`,
    always renders just those sources — and raises ``DataRightsViolation`` if one cannot be
    resolved, because "a report build that cannot resolve an attribution for a source it
    used must fail."
    """
    if source_keys is not None:
        return get_attribution_text(source_keys)
    path = Path(attributions_path)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return get_attribution_text()


# --------------------------------------------------------------------------------------
# figure verification (Section 23 item 7)
# --------------------------------------------------------------------------------------


def referenced_figure_ids(sections: Sequence[ReportSection]) -> list[str]:
    """Every ``[[fig:ID]]`` in narrative order, duplicates preserved."""
    return [
        match.group("figure_id")
        for section in sections
        for match in FIGURE_REF_RE.finditer(section.body)
    ]


def _resolve_path(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as exc:
                raise FigureVerificationError(
                    f"cannot resolve index {part!r} in field path {path!r}: {exc}"
                ) from exc
        elif isinstance(cur, Mapping):
            if part not in cur:
                raise FigureVerificationError(
                    f"field path {path!r} does not exist in the run data (missing {part!r})"
                )
            cur = cur[part]
        else:
            raise FigureVerificationError(
                f"field path {path!r} does not resolve: hit a non-container at {part!r}"
            )
    return cur


def _matches(stated: float, actual: Any, *, rel_tol: float, abs_tol: float) -> bool:
    try:
        actual_f = float(actual)
    except (TypeError, ValueError) as exc:
        raise FigureVerificationError(
            f"referenced run-data value {actual!r} is not numeric, so the quoted figure "
            f"{stated} cannot be verified"
        ) from exc
    return abs(stated - actual_f) <= max(abs_tol, rel_tol * abs(actual_f))


def verify_report(
    sections: Sequence[ReportSection],
    figures: Sequence[Figure],
    run_data: Mapping[str, Any],
    params: Params,
) -> FigureVerification:
    """The Section 23 item 7 gate. Raises :class:`FigureVerificationError` on any failure.

    Three clauses, in order: every referenced figure exists; every generated figure is
    referenced; every quoted number matches the run data within the
    ``api.report.figure_verification`` tolerances.
    """
    available = [f.figure_id for f in figures]
    duplicates = sorted({fid for fid in available if available.count(fid) > ONE})
    if duplicates:
        raise FigureVerificationError(
            f"duplicate figure id(s) {duplicates}: a figure reference would be ambiguous"
        )
    available_set = set(available)
    referenced = referenced_figure_ids(sections)
    referenced_set = set(referenced)

    # 1. every figure referenced by the narrative must exist
    dangling = sorted(referenced_set - available_set)
    if dangling:
        raise FigureVerificationError(
            f"the narrative references figure(s) {dangling} that were never generated. "
            f"Generated figures: {sorted(available_set)}."
        )

    # 2. every figure generated must be referenced
    if bool(params.get(P_REQUIRE_EVERY_FIGURE_REFERENCED)):
        orphans = sorted(available_set - referenced_set)
        if orphans:
            raise FigureVerificationError(
                f"figure(s) {orphans} were generated but are never referenced by the "
                "narrative. Either cite them or stop generating them — an unreferenced "
                "figure is a finding the report silently dropped."
            )

    # 3. every number quoted in the narrative must match the run data
    rel_tol = float(params.value(P_REL_TOL))
    abs_tol = float(params.value(P_ABS_TOL))
    require_resolved = bool(params.get(P_REQUIRE_EVERY_REFERENCE_RESOLVED))
    numbers_checked = ZERO
    references_checked = ZERO

    for section in sections:
        body = section.body
        for match in ANY_REF_RE.finditer(body):
            references_checked += ONE
            path = match.group("path")
            preceding = body[: match.end()]
            numeric = list(NUMBER_REF_RE.finditer(preceding))
            if not numeric or numeric[-ONE].end() != match.end():
                if not require_resolved:
                    continue
                raise FigureVerificationError(
                    f"section {section.name!r}: reference [{path}] has no numeric figure "
                    "immediately preceding it, so nothing can be verified against the run "
                    "data. Every claim in a report must be traceable (Section 23 item 5)."
                )
            stated = float(numeric[-ONE].group("figure"))
            actual = _resolve_path(run_data, path)
            if not _matches(stated, actual, rel_tol=rel_tol, abs_tol=abs_tol):
                raise FigureVerificationError(
                    f"section {section.name!r}: the narrative quotes {stated} for [{path}] "
                    f"but the run data says {actual} (rel_tol={rel_tol}, abs_tol={abs_tol})."
                )
            numbers_checked += ONE

    return FigureVerification(
        figures_checked=len(available),
        references_checked=references_checked,
        numbers_checked=numbers_checked,
    )


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def strip_references(text: str) -> str:
    """Drop the `[field.path]` brackets after verification, keeping the numbers."""
    stripped = ANY_REF_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", stripped)


def _provenance_markdown(provenance: Provenance) -> str:
    lines = [
        "## Provenance",
        "",
        f"- City: `{provenance.city}`",
        f"- Snapshot hash: `{provenance.snapshot_hash}`",
        f"- Params hash: `{provenance.params_hash}`",
        f"- Git commit: `{provenance.git_commit}`",
        f"- Calibration level: `{provenance.calibration_level}`",
    ]
    if provenance.confidence_tags:
        lines.append("- Confidence of the parameters behind the headline numbers (Section 0.2):")
        for path, tag in provenance.confidence_tags.items():
            lines.append(f"    - `{path}`: **{tag}**")
    if provenance.class_defaults_applied:
        lines.append(
            "- City-class defaults applied: "
            + ", ".join(f"`{p}`" for p in provenance.class_defaults_applied)
        )
    return "\n".join(lines)


def render_report(
    params: Params,
    *,
    report_id: str,
    title: str,
    sections: Sequence[ReportSection],
    figures: Sequence[Figure],
    run_data: Mapping[str, Any],
    snapshot_hash: str,
    headline_parameter_paths: Iterable[str] = (),
    attribution_sources: Iterable[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> Report:
    """Verify and render a report. Raises rather than emitting an unverified report.

    Order matters: verification runs BEFORE any prose is rendered, so a report that would
    quote a wrong number never reaches a renderer, let alone a client.
    """
    verification = verify_report(sections, figures, run_data, params)

    provenance = build_provenance(
        params,
        snapshot_hash=snapshot_hash,
        headline_parameter_paths=headline_parameter_paths,
        repo_root=repo_root,
    )
    flag = calibration_flag(params, provenance)
    attributions = attribution_block(attribution_sources)

    figures_by_id = {f.figure_id: f for f in figures}

    body_parts: list[str] = [f"# {title}", ""]
    if flag:
        body_parts += [f"> **{flag}**", ""]
    for section in sections:
        rendered = strip_references(section.body)

        def _inline(match: re.Match[str]) -> str:
            figure = figures_by_id[match.group("figure_id")]
            target = figure.uri or figure.figure_id
            return f"![{figure.title}]({target})"

        rendered = FIGURE_REF_RE.sub(_inline, rendered)
        body_parts += [f"## {section.name}", "", rendered.strip(), ""]

    footer_parts: list[str] = ["---", ""]
    if bool(params.get(P_FOOTER_PROVENANCE)):
        footer_parts += [_provenance_markdown(provenance), ""]
    if flag:
        # Section 20.2: "that flag must appear in EVERY report it produces" — header and
        # footer both, so it survives a page-one-only reading of the document.
        footer_parts += [f"**{flag}**", ""]
    if bool(params.get(P_FOOTER_ATTRIBUTIONS)):
        footer_parts += ["## Attribution", "", attributions, ""]
    footer = "\n".join(footer_parts).strip()

    markdown = "\n".join(body_parts).strip() + "\n\n" + footer + "\n"

    return Report(
        report_id=report_id,
        title=title,
        city=provenance.city,
        sections=tuple(sections),
        figures=tuple(figures),
        provenance=provenance,
        attributions=attributions,
        footer=footer,
        markdown=markdown,
        verification=verification,
        calibration_flag=flag,
    )


def about_page(params: Params, *, attributions_path: Path = ATTRIBUTIONS_PATH) -> dict[str, Any]:
    """The product about page (Section 22.4: ATTRIBUTIONS.md renders into it)."""
    return {
        "product": str(params.get("api.about.product")),
        "summary": str(params.get("api.about.summary")).strip(),
        "exposure_policy": str(params.get("api.exposure_policy")),
        "attributions": attribution_block(attributions_path=attributions_path),
    }
