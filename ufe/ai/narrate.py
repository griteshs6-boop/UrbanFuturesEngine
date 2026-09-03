"""Module 13 — Prompt G, report narrative generation and its mandatory verification gate
(spec Section 17.9).

Prompt G runs strictly after a simulation completes, never during (CONTRACT rule 4 / spec
Section 17: no `ufe.layers`/`ufe.sim`/`ufe.backtest` module may import `ufe.ai` at all, so
this module cannot be reached from generation time even by accident).

The verification step is a hard gate: every `[field.path]` reference the model appends
after a figure is resolved against the simulation output object and must match to within
rounding. Any unmatched or unresolvable reference fails the report build — this module
raises `NarrativeVerificationError` rather than silently passing a bad report through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ufe.ai.client import AIClient, PromptTemplate, load_prompt

PROMPT_G_NAME = "G_report_narrative"

# Matches a numeric figure immediately followed by a `[field.path]` reference, e.g.
# "+14% [zones.KOM.factors.metro.lambda]" or "-3.5 [foo.bar]".
_FIGURE_REF_RE = re.compile(r"(?P<figure>[+-]?\d+(?:\.\d+)?)\s*%?\s*\[(?P<path>[A-Za-z0-9_.]+)\]")

# Matches any bracketed reference at all, so we can also catch a bracket with no
# parseable figure in front of it (also a build failure — an untraceable claim).
_ANY_REF_RE = re.compile(r"\[(?P<path>[A-Za-z0-9_.]+)\]")


class NarrativeVerificationError(Exception):
    """The mandatory Section 17.9 verification gate failed. The report build must fail,
    not warn — CONTRACT: 'Raise, never warn, on invalid data.'"""


@dataclass(frozen=True)
class VerifiedNarrative:
    text: str
    stripped_text: str
    references_checked: int


def generate_narrative(
    client: AIClient,
    *,
    sim_result_json: str,
    section_name: str,
    audience: str,
    n: int,
    prompt_version: str = "v1",
) -> str:
    """Run Prompt G. Returns the raw prose (with `[field.path]` references still attached)
    — callers must run `verify_narrative` before treating the text as safe to publish."""
    prompt: PromptTemplate = load_prompt(PROMPT_G_NAME, prompt_version)
    return client.complete_raw(
        prompt,
        {
            "sim_result_json": sim_result_json,
            "section_name": section_name,
            "audience": audience,
            "n": n,
        },
    )


def _resolve_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as exc:
                raise NarrativeVerificationError(f"Cannot resolve list index '{part}' in path '{path}': {exc}") from exc
        elif isinstance(cur, dict):
            if part not in cur:
                raise NarrativeVerificationError(f"Field path '{path}' does not exist in the output object (missing '{part}')")
            cur = cur[part]
        else:
            raise NarrativeVerificationError(f"Field path '{path}' does not resolve: hit non-container at '{part}'")
    return cur


def _figure_matches(stated: float, actual: Any, *, rel_tol: float, abs_tol: float) -> bool:
    try:
        actual_f = float(actual)
    except (TypeError, ValueError) as exc:
        raise NarrativeVerificationError(f"Referenced field value {actual!r} is not numeric") from exc
    diff = abs(stated - actual_f)
    return diff <= max(abs_tol, rel_tol * abs(actual_f))


def verify_narrative(
    text: str,
    output_obj: dict[str, Any],
    *,
    rel_tol: float,
    abs_tol: float,
) -> VerifiedNarrative:
    """The Section 17.9 hard gate. Raises `NarrativeVerificationError` on the first
    reference that does not resolve, or whose stated figure does not match the output
    object's value within rounding. On success, returns the text with brackets stripped
    ('the rendering layer strips these brackets after verification')."""
    references_checked = 0
    for match in _ANY_REF_RE.finditer(text):
        path = match.group("path")
        # Check whether a numeric figure immediately precedes this exact bracket.
        preceding = text[: match.end()]
        figure_matches = list(_FIGURE_REF_RE.finditer(preceding))
        if not figure_matches or figure_matches[-1].end() != match.end():
            raise NarrativeVerificationError(
                f"Reference [{path}] has no parseable numeric figure immediately preceding it"
            )
        stated = float(figure_matches[-1].group("figure"))
        actual = _resolve_path(output_obj, path)
        if not _figure_matches(stated, actual, rel_tol=rel_tol, abs_tol=abs_tol):
            raise NarrativeVerificationError(
                f"Stated figure {stated} for [{path}] does not match computed value {actual} "
                f"(rel_tol={rel_tol}, abs_tol={abs_tol})"
            )
        references_checked += 1

    stripped = _ANY_REF_RE.sub("", text)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return VerifiedNarrative(text=text, stripped_text=stripped, references_checked=references_checked)
