"""Module 13 — extraction prompts A, B, D, E, F.

Each `run_prompt_*` function drives `ufe.ai.client.AIClient.extract` with the matching
prompt, validates the JSON response into a pydantic v2 model, and returns an
`ExtractionOutcome`. Routing the parsed (or failed) result into `ufe.ai.queue.ReviewQueue`
is the caller's job (kept separate so callers can decide `project_id`/`is_stage_change`
context that only they have) — see `queue_extraction_result` below for the shared plumbing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ufe.ai.client import AIClient, ExtractionOutcome, PromptTemplate, load_prompt
from ufe.ai.queue import CandidateStatus, RecordType, ReviewCandidate, ReviewQueue

PROMPT_A_NAME = "A_project_extraction"
PROMPT_B_NAME = "B_commitment_hardness"
PROMPT_D_NAME = "D_delivery_record"
PROMPT_E_NAME = "E_ec_extraction"
PROMPT_F_NAME = "F_change_monitoring"


# --------------------------------------------------------------------------------------
# Prompt A — project extraction (Section 17.3)
# --------------------------------------------------------------------------------------


class ProjectExtractionItem(BaseModel):
    name: str
    archetype_guess: str
    location_text: str | None = None
    announcer_name: str | None = None
    is_public: bool | None = None
    scale_value: float | None = None
    scale_unit: str | None = None
    capex_inr_cr: float | None = None
    stated_jobs: float | None = None
    stated_completion_text: str | None = None
    commitment_form_evidence: str | None = None
    quoted_by: str | None = None
    is_reannouncement: bool
    confidence: float = Field(ge=0.0, le=1.0)
    # Not in the literal JSON schema in Section 17.3, but the prose rule immediately
    # above the schema requires it ("record it as one record with location_text=null and
    # flag needs_location=true") — see the ambiguity note in the final report. Modeled as
    # an additional optional field, defaulting False, rather than dropped.
    needs_location: bool = False


class ProjectExtractionResult(BaseModel):
    projects: list[ProjectExtractionItem]
    document_type: str
    notes: str | None = None


def run_prompt_a(
    client: AIClient,
    *,
    raw_text: str,
    url: str,
    pub_date: str,
    prompt_version: str = "v1",
) -> ExtractionOutcome[ProjectExtractionResult]:
    prompt = load_prompt(PROMPT_A_NAME, prompt_version)
    archetype_list = ", ".join(client.settings.allowed_archetypes)
    return client.extract(
        prompt,
        {"raw_text": raw_text, "url": url, "pub_date": pub_date, "archetype_list": archetype_list},
        ProjectExtractionResult,
    )


# --------------------------------------------------------------------------------------
# Prompt B — commitment hardness classification (Section 17.4)
# --------------------------------------------------------------------------------------

COMMITMENT_FORMS = (
    "verbal",
    "summit_mou",
    "govt_mou_signed",
    "land_allotted",
    "board_approved",
    "land_possessed",
    "ec_granted",
    "epc_appointed",
    "equipment_ordered",
    "construction_seen",
)


class CommitmentHardnessResult(BaseModel):
    commitment_form: str
    evidence_quote: str
    modifiers: list[str] = Field(default_factory=list)
    modifier_evidence: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_note: str | None = None


def run_prompt_b(
    client: AIClient,
    *,
    project_name: str,
    announcer_name: str | None,
    raw_text: str,
    modifier_list_with_descriptions: str,
    prompt_version: str = "v1",
) -> ExtractionOutcome[CommitmentHardnessResult]:
    prompt = load_prompt(PROMPT_B_NAME, prompt_version)
    return client.extract(
        prompt,
        {
            "project_name": project_name,
            "announcer_name": announcer_name,
            "raw_text": raw_text,
            "modifier_list_with_descriptions": modifier_list_with_descriptions,
        },
        CommitmentHardnessResult,
    )


# --------------------------------------------------------------------------------------
# Prompt D — announcer delivery record reconstruction (Section 17.6)
# --------------------------------------------------------------------------------------


class DeliveryProjectOutcome(BaseModel):
    name: str
    announced_date: str
    announced_capex_inr_cr: float | None = None
    stated_completion: str | None = None
    outcome: str
    outcome_evidence: str
    actual_completion: str | None = None
    slip_months: float | None = None


class DeliveryRecordResult(BaseModel):
    projects: list[DeliveryProjectOutcome]
    total_announced_inr_cr: float
    total_deployed_inr_cr: float | None = None
    deployed_source: str
    coverage_note: str
    confidence: float = Field(ge=0.0, le=1.0)


def run_prompt_d(
    client: AIClient,
    *,
    name: str,
    announcement_docs: str,
    annual_report_extracts: str,
    prompt_version: str = "v1",
) -> ExtractionOutcome[DeliveryRecordResult]:
    prompt = load_prompt(PROMPT_D_NAME, prompt_version)
    return client.extract(
        prompt,
        {
            "name": name,
            "announcement_docs": announcement_docs,
            "annual_report_extracts": annual_report_extracts,
        },
        DeliveryRecordResult,
    )


# --------------------------------------------------------------------------------------
# Prompt E — environmental clearance filing extraction (Section 17.7)
# --------------------------------------------------------------------------------------


class ECExtractionResult(BaseModel):
    proposal_number: str | None = None
    project_name: str
    proponent_name: str
    category: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    land_area_ha: float | None = None
    capacity_value: float | None = None
    capacity_unit: str | None = None
    capex_inr_cr: float | None = None
    clearance_status: str
    status_date: str | None = None
    employment_direct: float | None = None
    employment_indirect: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Called out in prose but absent from the literal schema, same pattern as Prompt A's
    # needs_location — see the ambiguity note in the final report.
    coord_source: str | None = None
    coord_all: list[list[float]] | None = None


def run_prompt_e(
    client: AIClient,
    *,
    raw_text: str,
    prompt_version: str = "v1",
) -> ExtractionOutcome[ECExtractionResult]:
    prompt = load_prompt(PROMPT_E_NAME, prompt_version)
    return client.extract(prompt, {"raw_text": raw_text}, ECExtractionResult)


# --------------------------------------------------------------------------------------
# Prompt F — change monitoring triage (Section 17.8)
# --------------------------------------------------------------------------------------


class ChangeMonitoringResult(BaseModel):
    is_material_change: bool
    change_type: str | None = None
    proposed_updates: dict[str, Any] = Field(default_factory=dict)
    evidence_quote: str
    requires_human_review: bool
    confidence: float = Field(ge=0.0, le=1.0)


def run_prompt_f(
    client: AIClient,
    *,
    project_json: str,
    doc_date: str,
    raw_text: str,
    prompt_version: str = "v1",
) -> ExtractionOutcome[ChangeMonitoringResult]:
    prompt = load_prompt(PROMPT_F_NAME, prompt_version)
    return client.extract(
        prompt,
        {"project_json": project_json, "doc_date": doc_date, "raw_text": raw_text},
        ChangeMonitoringResult,
    )


# --------------------------------------------------------------------------------------
# Shared plumbing: route an ExtractionOutcome into the review queue.
# --------------------------------------------------------------------------------------


def queue_extraction_result(
    queue: ReviewQueue,
    outcome: ExtractionOutcome[BaseModel],
    *,
    record_type: RecordType,
    auto_apply_threshold: float,
    source_url: str | None = None,
    project_id: str | None = None,
    is_new_project: bool = False,
    is_stage_change: bool = False,
    confidence_override: float | None = None,
) -> ReviewCandidate:
    """Turn one `AIClient.extract` outcome into a queue candidate.

    A malformed response (outcome.ok is False) always becomes `parse_failed` and never
    reaches `projects`/`announcers`, regardless of any confidence field content — Section
    17.2: 'parse failures ... go to the review queue as parse_failed'.
    """
    extracted_by = outcome.extracted_by
    if not outcome.ok:
        candidate = ReviewCandidate(
            record_type=record_type,
            payload={},
            extracted_by=extracted_by,
            prompt_name=outcome.prompt_name,
            prompt_version=outcome.prompt_version,
            model_id=outcome.model_id,
            settings_hash=outcome.settings_hash,
            confidence=None,
            source_url=source_url,
            project_id=project_id,
            is_new_project=is_new_project,
            is_stage_change=is_stage_change,
            status=CandidateStatus.PARSE_FAILED,
            parse_error=outcome.error,
        )
        return queue.enqueue(candidate, auto_apply_threshold=auto_apply_threshold)

    assert outcome.parsed is not None
    payload = outcome.parsed.model_dump()
    confidence = confidence_override if confidence_override is not None else payload.get("confidence")
    candidate = ReviewCandidate(
        record_type=record_type,
        payload=payload,
        extracted_by=extracted_by,
        prompt_name=outcome.prompt_name,
        prompt_version=outcome.prompt_version,
        model_id=outcome.model_id,
        settings_hash=outcome.settings_hash,
        confidence=confidence,
        source_url=source_url,
        project_id=project_id,
        is_new_project=is_new_project,
        is_stage_change=is_stage_change,
    )
    return queue.enqueue(candidate, auto_apply_threshold=auto_apply_threshold)
