"""Module 13 — the human review queue (spec Section 17.10).

Holds candidate records produced by the AI extraction pipeline (`ufe.ai.extract`,
`ufe.ai.resolve`) before they are allowed to touch `projects` / `announcers`.

Decision rule (Section 17.1):
    - New projects and stage transitions ALWAYS require human approval, regardless of
      confidence. They never auto-apply.
    - Attribute updates that do not change stage MAY auto-apply if extraction confidence
      exceeds `confidence_auto_apply_threshold` (from config/params/ai.yaml). Auto-applied
      updates are still logged to `project_history`.
    - Parse failures (malformed JSON after the retry in `ufe.ai.client`) always land in
      the queue with status `parse_failed` and never auto-apply.

This module has no hard dependency on `ufe.store` — it works as a pure in-memory queue
so the AI pipeline is fully testable in isolation, per the task brief ("ufe.ai may import
from ufe.store and ufe.errors" — optional, not required). `to_project_history_row` produces
a dict shaped like the `PROJECT_HISTORY` table (Section 3.5) for callers that do have a
store connection to persist into.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CandidateStatus(str, Enum):
    PENDING = "pending"
    PARSE_FAILED = "parse_failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPLIED = "auto_applied"


class RecordType(str, Enum):
    NEW_PROJECT = "new_project"
    NEW_ANNOUNCER = "new_announcer"
    STAGE_CHANGE = "stage_change"
    ATTRIBUTE_UPDATE = "attribute_update"
    ENTITY_RESOLUTION = "entity_resolution"
    OTHER = "other"


@dataclass
class ReviewCandidate:
    """One row of the `project_candidates` table (Section 17.1)."""

    record_type: RecordType
    payload: dict[str, Any]
    extracted_by: str
    prompt_name: str
    prompt_version: str
    model_id: str
    settings_hash: str
    confidence: float | None
    source_url: str | None = None
    project_id: str | None = None
    is_stage_change: bool = False
    is_new_project: bool = False
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: CandidateStatus = CandidateStatus.PENDING
    verified_by: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    decided_at: datetime | None = None
    rejection_reason: str | None = None
    parse_error: str | None = None


def decide_auto_apply(candidate: ReviewCandidate, auto_apply_threshold: float) -> bool:
    """Section 17.1's auto-apply rule.

    Stage transitions and brand-new projects always require a human, no matter the
    confidence. Everything else may auto-apply once confidence clears the threshold.
    """
    if candidate.status == CandidateStatus.PARSE_FAILED:
        return False
    if candidate.is_new_project or candidate.is_stage_change:
        return False
    if candidate.record_type in (RecordType.NEW_PROJECT, RecordType.STAGE_CHANGE):
        return False
    if candidate.confidence is None:
        return False
    return candidate.confidence > auto_apply_threshold


def to_project_history_row(candidate: ReviewCandidate, *, field_name: str, old_value: Any, new_value: Any) -> dict[str, Any]:
    """Shape a candidate's decision as a `project_history` row (Section 3.5), append-only.

    `(project_id, field, old_value, new_value, changed_at, source_url, changed_by)`.
    """
    return {
        "project_id": candidate.project_id,
        "field": field_name,
        "old_value": old_value,
        "new_value": new_value,
        "changed_at": candidate.decided_at or _utcnow(),
        "source_url": candidate.source_url,
        "changed_by": candidate.verified_by or candidate.extracted_by,
    }


class ReviewQueue:
    """In-memory review queue. Section 17.10: 'a minimal CLI that shows pending candidates
    with the source document alongside, and accepts approve / reject / edit.'"""

    def __init__(self) -> None:
        self._items: dict[str, ReviewCandidate] = {}
        self._order = itertools.count()
        self._sequence: dict[str, int] = {}

    def enqueue(self, candidate: ReviewCandidate, *, auto_apply_threshold: float) -> ReviewCandidate:
        """Add a candidate. If it qualifies for auto-apply, mark it applied immediately
        (still recorded here, and still expected to be logged to project_history by the
        caller via `to_project_history_row`)."""
        if candidate.status != CandidateStatus.PARSE_FAILED and decide_auto_apply(candidate, auto_apply_threshold):
            candidate.status = CandidateStatus.AUTO_APPLIED
            candidate.verified_by = candidate.verified_by or "system:auto_apply"
            candidate.decided_at = _utcnow()
        self._items[candidate.candidate_id] = candidate
        self._sequence[candidate.candidate_id] = next(self._order)
        return candidate

    def get(self, candidate_id: str) -> ReviewCandidate:
        return self._items[candidate_id]

    def all(self) -> list[ReviewCandidate]:
        return [self._items[cid] for cid in sorted(self._items, key=lambda c: self._sequence[c])]

    def list_pending(self) -> list[ReviewCandidate]:
        return [c for c in self.all() if c.status in (CandidateStatus.PENDING, CandidateStatus.PARSE_FAILED)]

    def approve(self, candidate_id: str, verified_by: str) -> ReviewCandidate:
        candidate = self._items[candidate_id]
        candidate.status = CandidateStatus.APPROVED
        candidate.verified_by = verified_by
        candidate.decided_at = _utcnow()
        return candidate

    def reject(self, candidate_id: str, verified_by: str, reason: str | None = None) -> ReviewCandidate:
        candidate = self._items[candidate_id]
        candidate.status = CandidateStatus.REJECTED
        candidate.verified_by = verified_by
        candidate.rejection_reason = reason
        candidate.decided_at = _utcnow()
        return candidate

    def edit_and_approve(self, candidate_id: str, verified_by: str, edits: dict[str, Any]) -> ReviewCandidate:
        candidate = self._items[candidate_id]
        candidate.payload = {**candidate.payload, **edits}
        candidate.status = CandidateStatus.APPROVED
        candidate.verified_by = verified_by
        candidate.decided_at = _utcnow()
        return candidate
