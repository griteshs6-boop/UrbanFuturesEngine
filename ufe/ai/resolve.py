"""Module 13 — Prompt C, entity resolution (spec Section 17.5).

Matches a mentioned company name against a set of candidate canonical `announcers`
records. Never guesses: a match below confidence, or with `matched_id=null`, always
routes to the review queue rather than silently linking a project to the wrong company.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ufe.ai.client import AIClient, ExtractionOutcome, load_prompt

PROMPT_C_NAME = "C_entity_resolution"


class SuggestedNewRecord(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)


class EntityResolutionResult(BaseModel):
    matched_id: str | None = None
    match_type: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_new_record: SuggestedNewRecord | None = None


def run_prompt_c(
    client: AIClient,
    *,
    mentioned: str,
    surrounding_sentence: str,
    candidates: list[dict[str, Any]],
    prompt_version: str = "v1",
) -> ExtractionOutcome[EntityResolutionResult]:
    prompt = load_prompt(PROMPT_C_NAME, prompt_version)
    return client.extract(
        prompt,
        {
            "mentioned": mentioned,
            "surrounding_sentence": surrounding_sentence,
            "candidates_json": json.dumps(candidates, sort_keys=True),
        },
        EntityResolutionResult,
    )


def is_confident_match(result: EntityResolutionResult, *, threshold: float) -> bool:
    """A resolution is only usable without human review if it names a specific match
    (`matched_id` set, `match_type != "none"`) and clears the confidence bar. Section
    17.5: 'You never guess. If the match is not clear, return null and explain.'"""
    return result.matched_id is not None and result.match_type != "none" and result.confidence > threshold
