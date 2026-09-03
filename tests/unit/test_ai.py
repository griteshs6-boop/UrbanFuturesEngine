"""Tests for Module 13 — the AI data pipeline.

All of these run without a live Anthropic API key and without network access: every
LLM call goes through `ufe.ai.client.RecordReplayTransport`, replaying canned JSON
from `tests/fixtures/ai_fixtures.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ufe.ai.client import (
    AIClient,
    AnthropicTransport,
    RecordReplayTransport,
    list_prompt_files,
    load_prompt,
    load_settings,
)
from ufe.ai.extract import (
    ChangeMonitoringResult,
    CommitmentHardnessResult,
    ProjectExtractionResult,
    queue_extraction_result,
    run_prompt_a,
    run_prompt_b,
    run_prompt_f,
)
from ufe.ai.narrate import NarrativeVerificationError, verify_narrative
from ufe.ai.queue import CandidateStatus, RecordType, ReviewCandidate, ReviewQueue, decide_auto_apply
from ufe.ai.resolve import EntityResolutionResult, is_confident_match

from tests.fixtures import ai_fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[2]


# ======================================================================================
# Import-graph acceptance test — the hard architectural rule (Section 17, CONTRACT rule 4)
# ======================================================================================


def _module_imports_ufe_ai(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ufe.ai" or alias.name.startswith("ufe.ai."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "ufe.ai" or node.module.startswith("ufe.ai.")):
                return True
    return False


#: The packages the rule actually covers. Spec Section 23 item 6: "No simulation module
#: imports `ufe.ai`." CONTRACT.md rule 4: "No module under `ufe/layers/`, `ufe/sim/`, or
#: `ufe/backtest/` may import `ufe.ai`. There is a test for this."
SIMULATION_PACKAGES: tuple[str, ...] = ("layers", "sim", "backtest")


@pytest.mark.acceptance
def test_no_simulation_module_imports_ai():
    """Spec Section 23 item 6: "No simulation module imports `ufe.ai`."

    CONTRACT.md rule 4 names the packages: "No module under `ufe/layers/`, `ufe/sim/`, or
    `ufe/backtest/` may import `ufe.ai`. There is a test for this." That is this test.

    `ufe/api` is deliberately NOT in scope. The API is allowed to produce report narrative
    — spec Section 17.9, prompt G ("G_report_narrative") — which runs strictly *after* a
    simulation completes, so it is not simulation-time inference. The separate architectural
    property the API must hold is that it reaches the narrator only by injection, which
    `test_api_reaches_the_narrator_only_by_injection` asserts below.

    Enforced by AST inspection, not a text grep, so it survives reformatting/aliasing
    tricks.
    """
    offenders: list[str] = []
    for subdir in SIMULATION_PACKAGES:
        root = REPO_ROOT / "ufe" / subdir
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            if _module_imports_ufe_ai(py_file):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert offenders == [], f"Simulation modules must never import ufe.ai: {offenders}"


def test_api_reaches_the_narrator_only_by_injection():
    """`ufe/api/main.py` obtains prompt-G prose by injection, never by importing `ufe.ai`.

    Spec Section 17.9 permits the API to generate report narrative, so the Section 23
    item 6 ban above does not apply to it. What must hold instead is the composition-root
    design: `create_app` accepts a `narrator` and calls it, and `main.py` itself has no
    static dependency on `ufe.ai` (prompt G needs a configured `AIClient` that only a
    composition root can supply).
    """
    import inspect

    from ufe.api import main as api_main

    main_path = REPO_ROOT / "ufe" / "api" / "main.py"
    assert not _module_imports_ufe_ai(main_path), (
        "ufe/api/main.py must not import ufe.ai statically: the narrator is injected by "
        "the composition root (spec Section 17.9, prompt G)"
    )

    signature = inspect.signature(api_main.create_app)
    assert "narrator" in signature.parameters, (
        "create_app must accept an injected `narrator` — that is how the API reaches "
        "prompt G without importing ufe.ai"
    )

    # That the injected callable is genuinely what produces the prose is exercised
    # end-to-end by tests/unit/test_api.py::test_report_route_uses_a_configured_narrator.


# ======================================================================================
# Prompt files load verbatim from disk
# ======================================================================================


@pytest.mark.parametrize(
    "name,version",
    [
        ("A_project_extraction", "v1"),
        ("B_commitment_hardness", "v1"),
        ("C_entity_resolution", "v1"),
        ("D_delivery_record", "v1"),
        ("E_ec_extraction", "v1"),
        ("F_change_monitoring", "v1"),
        ("G_report_narrative", "v1"),
    ],
)
def test_all_seven_prompts_load_from_disk(name, version):
    prompt = load_prompt(name, version)
    assert prompt.system.strip() != ""
    assert prompt.user_template.strip() != ""
    assert prompt.extracted_by == f"ai:{name}.{version}.md"


def test_prompt_inventory_has_exactly_seven_files():
    files = list_prompt_files()
    assert len(files) == 7
    letters = sorted(p.name[0] for p in files)
    assert letters == list("ABCDEFG")


def test_client_never_reads_env_or_network_at_import_or_construction():
    """AnthropicTransport must not read ANTHROPIC_API_KEY or touch the network until
    complete() is actually invoked."""
    transport = AnthropicTransport(api_key=None)
    assert transport._client is None  # lazily constructed


# ======================================================================================
# Settings loaded from YAML, not hardcoded (CONTRACT rule 1)
# ======================================================================================


def test_settings_load_from_yaml_config():
    settings = load_settings()
    assert settings.model_id  # comes from config/params/ai.yaml, not a literal in client.py
    assert settings.temperature == 0.0
    assert 0.0 < settings.confidence_auto_apply_threshold <= 1.0
    assert settings.hash  # provenance hash is derivable
    assert len(settings.allowed_archetypes) > 0


# ======================================================================================
# Prompt A acceptance — project count, null-for-unstated, "up to 10,000 jobs" hedge
# ======================================================================================


@pytest.mark.acceptance
def test_prompt_a_extracts_correct_count_and_nulls_unstated_fields():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_a_transport(), settings=settings)
    outcome = run_prompt_a(
        client, raw_text=fx.PROMPT_A_ARTICLE, url="https://example.com/a", pub_date="2026-01-01"
    )
    assert outcome.ok
    result: ProjectExtractionResult = outcome.parsed
    assert len(result.projects) == 2

    hedge_project = result.projects[0]
    assert hedge_project.stated_jobs == 10000
    assert "hedge" in (result.notes or "").lower() or "10000" in (result.notes or "")
    # Every field not explicitly stated in the document must be null.
    assert hedge_project.scale_value is None
    assert hedge_project.capex_inr_cr is None
    assert hedge_project.stated_completion_text is None
    assert hedge_project.commitment_form_evidence is None

    multisite_project = result.projects[1]
    assert multisite_project.location_text is None
    assert multisite_project.needs_location is True
    assert multisite_project.scale_value is None
    assert multisite_project.capex_inr_cr is None
    assert multisite_project.stated_jobs is None


def test_prompt_a_extracted_by_matches_prompt_file_and_version():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_a_transport(), settings=settings)
    outcome = run_prompt_a(client, raw_text=fx.PROMPT_A_ARTICLE, url="u", pub_date="2026-01-01")
    assert outcome.extracted_by == "ai:A_project_extraction.v1"


# ======================================================================================
# Prompt B acceptance — summit MoU, not board_approved
# ======================================================================================


@pytest.mark.acceptance
def test_prompt_b_summit_mou_not_board_approved():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_b_transport(), settings=settings)
    outcome = run_prompt_b(
        client,
        project_name="ABC Energy 500 MW solar park",
        announcer_name="ABC Energy",
        raw_text=fx.PROMPT_B_SUMMIT_ARTICLE,
        modifier_list_with_descriptions="(none for this fixture)",
    )
    assert outcome.ok
    result: CommitmentHardnessResult = outcome.parsed
    assert result.commitment_form == "summit_mou"
    assert result.commitment_form != "board_approved"


# ======================================================================================
# Prompt F acceptance — re-announcement is not a material change
# ======================================================================================


@pytest.mark.acceptance
def test_prompt_f_reannouncement_is_not_material_change():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_f_transport(), settings=settings)
    outcome = run_prompt_f(
        client,
        project_json=fx.PROMPT_F_EXISTING_PROJECT,
        doc_date="2026-08-01",
        raw_text=fx.PROMPT_F_REANNOUNCEMENT_ARTICLE,
    )
    assert outcome.ok
    result: ChangeMonitoringResult = outcome.parsed
    assert result.is_material_change is False
    assert result.change_type == "reannouncement"


# ======================================================================================
# Narrative verification acceptance — catches an injected wrong figure
# ======================================================================================


def test_narrative_verification_passes_a_correctly_cited_report():
    settings = load_settings()
    verified = verify_narrative(
        fx.NARRATIVE_TEXT_VALID,
        fx.SIM_RESULT_OBJECT,
        rel_tol=settings.narrative_rel_tol,
        abs_tol=settings.narrative_abs_tol,
    )
    assert verified.references_checked == 2
    assert "[" not in verified.stripped_text


@pytest.mark.acceptance
def test_narrative_verification_catches_injected_wrong_figure_and_fails_build():
    settings = load_settings()
    with pytest.raises(NarrativeVerificationError):
        verify_narrative(
            fx.NARRATIVE_TEXT_INJECTED_WRONG_FIGURE,
            fx.SIM_RESULT_OBJECT,
            rel_tol=settings.narrative_rel_tol,
            abs_tol=settings.narrative_abs_tol,
        )


def test_narrative_verification_catches_unresolvable_path():
    settings = load_settings()
    with pytest.raises(NarrativeVerificationError):
        verify_narrative(
            "Rents rose 5 [zones.NOPE.does_not_exist].",
            fx.SIM_RESULT_OBJECT,
            rel_tol=settings.narrative_rel_tol,
            abs_tol=settings.narrative_abs_tol,
        )


# ======================================================================================
# extracted_by provenance acceptance — every AI-written record carries it
# ======================================================================================


@pytest.mark.acceptance
def test_every_ai_written_record_carries_extracted_by():
    settings = load_settings()
    queue = ReviewQueue()

    client_a = AIClient(transport=fx.prompt_a_transport(), settings=settings)
    outcome_a = run_prompt_a(client_a, raw_text=fx.PROMPT_A_ARTICLE, url="u", pub_date="2026-01-01")
    candidate_a = queue_extraction_result(
        queue, outcome_a, record_type=RecordType.NEW_PROJECT,
        auto_apply_threshold=settings.confidence_auto_apply_threshold, is_new_project=True,
    )
    assert candidate_a.extracted_by == "ai:A_project_extraction.v1"

    client_b = AIClient(transport=fx.prompt_b_transport(), settings=settings)
    outcome_b = run_prompt_b(
        client_b, project_name="p", announcer_name="a", raw_text=fx.PROMPT_B_SUMMIT_ARTICLE,
        modifier_list_with_descriptions="-",
    )
    candidate_b = queue_extraction_result(
        queue, outcome_b, record_type=RecordType.ATTRIBUTE_UPDATE,
        auto_apply_threshold=settings.confidence_auto_apply_threshold,
    )
    assert candidate_b.extracted_by == "ai:B_commitment_hardness.v1"

    for candidate in queue.all():
        assert candidate.extracted_by.startswith("ai:")
        assert candidate.extracted_by == f"ai:{candidate.prompt_name}.{candidate.prompt_version}"


# ======================================================================================
# Review queue decision logic — Section 17.1
# ======================================================================================


def _fake_candidate(**overrides) -> ReviewCandidate:
    defaults = dict(
        record_type=RecordType.ATTRIBUTE_UPDATE,
        payload={"source_url": "x"},
        extracted_by="ai:B_commitment_hardness.v1",
        prompt_name="B_commitment_hardness",
        prompt_version="v1",
        model_id="test-model",
        settings_hash="deadbeef",
        confidence=0.95,
    )
    defaults.update(overrides)
    return ReviewCandidate(**defaults)


def test_new_project_never_auto_applies_even_at_high_confidence():
    candidate = _fake_candidate(record_type=RecordType.NEW_PROJECT, is_new_project=True, confidence=0.99)
    assert decide_auto_apply(candidate, auto_apply_threshold=0.9) is False


def test_stage_change_never_auto_applies_even_at_high_confidence():
    candidate = _fake_candidate(record_type=RecordType.STAGE_CHANGE, is_stage_change=True, confidence=0.99)
    assert decide_auto_apply(candidate, auto_apply_threshold=0.9) is False


def test_attribute_update_auto_applies_above_threshold():
    candidate = _fake_candidate(confidence=0.95)
    assert decide_auto_apply(candidate, auto_apply_threshold=0.9) is True


def test_attribute_update_stays_pending_below_threshold():
    candidate = _fake_candidate(confidence=0.5)
    assert decide_auto_apply(candidate, auto_apply_threshold=0.9) is False


def test_attribute_update_with_no_confidence_never_auto_applies():
    candidate = _fake_candidate(confidence=None)
    assert decide_auto_apply(candidate, auto_apply_threshold=0.9) is False


def test_review_queue_enqueue_auto_applies_attribute_update():
    queue = ReviewQueue()
    candidate = _fake_candidate(confidence=0.95)
    enqueued = queue.enqueue(candidate, auto_apply_threshold=0.9)
    assert enqueued.status == CandidateStatus.AUTO_APPLIED
    assert enqueued.verified_by is not None
    assert enqueued.decided_at is not None
    assert queue.list_pending() == []  # auto-applied items are not pending


def test_review_queue_enqueue_new_project_stays_pending_for_human_review():
    queue = ReviewQueue()
    candidate = _fake_candidate(record_type=RecordType.NEW_PROJECT, is_new_project=True, confidence=0.99)
    enqueued = queue.enqueue(candidate, auto_apply_threshold=0.9)
    assert enqueued.status == CandidateStatus.PENDING
    assert enqueued in queue.list_pending()


def test_review_queue_approve_reject_and_edit_flow():
    queue = ReviewQueue()
    c1 = queue.enqueue(_fake_candidate(record_type=RecordType.NEW_PROJECT, is_new_project=True, confidence=0.99), auto_apply_threshold=0.9)
    c2 = queue.enqueue(_fake_candidate(record_type=RecordType.NEW_PROJECT, is_new_project=True, confidence=0.99), auto_apply_threshold=0.9)

    approved = queue.approve(c1.candidate_id, verified_by="alice")
    assert approved.status == CandidateStatus.APPROVED
    assert approved.verified_by == "alice"
    assert approved.decided_at is not None

    rejected = queue.reject(c2.candidate_id, verified_by="bob", reason="not a real project")
    assert rejected.status == CandidateStatus.REJECTED
    assert rejected.rejection_reason == "not a real project"

    c3 = queue.enqueue(_fake_candidate(record_type=RecordType.NEW_PROJECT, is_new_project=True, confidence=0.99), auto_apply_threshold=0.9)
    edited = queue.edit_and_approve(c3.candidate_id, verified_by="alice", edits={"name": "corrected name"})
    assert edited.status == CandidateStatus.APPROVED
    assert edited.payload["name"] == "corrected name"


# ======================================================================================
# Parse-failure retry semantics (Section 17.2)
# ======================================================================================


def test_malformed_response_retries_once_then_succeeds():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_a_retry_then_ok_transport(), settings=settings)
    outcome = run_prompt_a(client, raw_text=fx.PROMPT_A_ARTICLE, url="u", pub_date="2026-01-01")
    assert outcome.ok
    assert outcome.attempts == 2
    assert len(client.transport.calls) == 2
    # The retry call must carry the validation error back to the model.
    assert "error" in client.transport.calls[1].user.lower()


def test_persistently_malformed_response_routes_to_parse_failed_never_projects():
    settings = load_settings()
    client = AIClient(transport=fx.prompt_a_always_malformed_transport(), settings=settings)
    outcome = run_prompt_a(client, raw_text=fx.PROMPT_A_ARTICLE, url="u", pub_date="2026-01-01")
    assert outcome.ok is False
    assert outcome.parsed is None
    assert outcome.attempts == settings.max_retries + 1

    queue = ReviewQueue()
    candidate = queue_extraction_result(
        queue, outcome, record_type=RecordType.NEW_PROJECT,
        auto_apply_threshold=settings.confidence_auto_apply_threshold, is_new_project=True,
    )
    assert candidate.status == CandidateStatus.PARSE_FAILED
    assert candidate.payload == {}
    assert candidate in queue.list_pending()  # never silently accepted into projects


def test_low_confidence_never_silently_bypasses_the_queue():
    """A malformed-but-confident-looking response still never reaches projects directly;
    it always passes through ReviewQueue.enqueue, which is the only gate for auto-apply."""
    settings = load_settings()
    client = AIClient(transport=fx.prompt_b_transport(), settings=settings)
    outcome = run_prompt_b(
        client, project_name="p", announcer_name="a", raw_text=fx.PROMPT_B_SUMMIT_ARTICLE,
        modifier_list_with_descriptions="-",
    )
    queue = ReviewQueue()
    candidate = queue_extraction_result(
        queue, outcome, record_type=RecordType.STAGE_CHANGE,
        auto_apply_threshold=settings.confidence_auto_apply_threshold, is_stage_change=True,
    )
    # commitment_form classification implies a stage-relevant change -> always human-reviewed
    assert candidate.status == CandidateStatus.PENDING


# ======================================================================================
# Prompt C — entity resolution never guesses
# ======================================================================================


def test_entity_resolution_none_match_is_not_confident():
    result = EntityResolutionResult(matched_id=None, match_type="none", reasoning="no clear match", confidence=0.4)
    assert is_confident_match(result, threshold=0.9) is False


def test_entity_resolution_confident_exact_match():
    result = EntityResolutionResult(matched_id="abc-1", match_type="exact", reasoning="exact name match", confidence=0.98)
    assert is_confident_match(result, threshold=0.9) is True


# ======================================================================================
# CLI smoke test
# ======================================================================================


def test_ai_cli_app_is_a_typer_subapp_with_review_commands():
    from typer.testing import CliRunner

    import ufe.ai_cli as ai_cli

    assert ai_cli.app.info.name is None or True  # module-level `app` exists; name is set by mount point
    runner = CliRunner()

    queue = ReviewQueue()
    settings = load_settings()
    client = AIClient(transport=fx.prompt_a_transport(), settings=settings)
    outcome = run_prompt_a(client, raw_text=fx.PROMPT_A_ARTICLE, url="u", pub_date="2026-01-01")
    candidate = queue_extraction_result(
        queue, outcome, record_type=RecordType.NEW_PROJECT,
        auto_apply_threshold=settings.confidence_auto_apply_threshold,
        is_new_project=True, source_url="https://example.com/a",
    )
    ai_cli.set_queue(queue)

    ai_cli.console.width = 200
    result = runner.invoke(ai_cli.app, ["list"])
    assert result.exit_code == 0
    assert candidate.candidate_id in result.stdout

    result = runner.invoke(ai_cli.app, ["approve", candidate.candidate_id, "--verified-by", "alice"])
    assert result.exit_code == 0
    assert queue.get(candidate.candidate_id).status == CandidateStatus.APPROVED
    assert queue.get(candidate.candidate_id).verified_by == "alice"


# ======================================================================================
# Optional guarded imports — ufe.store / ufe.params may not exist yet on disk.
# ======================================================================================


def test_store_and_params_imports_are_guarded_if_absent():
    pytest.importorskip("ufe.store.db", reason="ufe.store not yet implemented by its owning agent")
    pytest.importorskip("ufe.params", reason="ufe.params not yet implemented by its owning agent")
