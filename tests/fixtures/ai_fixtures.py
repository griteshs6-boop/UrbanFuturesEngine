"""Canned LLM responses + sample source documents for Module 13 tests.

Every response here is a hand-authored fixture standing in for what the real Anthropic
API would return. Nothing in this file (or in tests/unit/test_ai.py) makes a network
call — `ufe.ai.client.RecordReplayTransport` just replays these strings.
"""

from __future__ import annotations

import json

from ufe.ai.client import RecordReplayTransport

# --------------------------------------------------------------------------------------
# Prompt A fixture — two projects: one with a "up to 10,000 jobs" hedge, one an
# unspecified multi-site plan (location_text=null, needs_location=true).
# --------------------------------------------------------------------------------------

PROMPT_A_ARTICLE = """\
XYZ Industries announced today that it will build a new electronics assembly plant in
Devanahalli, Bengaluru. A company spokesperson said the facility could create up to
10,000 jobs once fully operational. No investment figure or completion date was
disclosed in the statement.

Separately, XYZ Industries said it plans to invest across multiple as-yet-unidentified
sites in Karnataka as part of a broader five-year expansion strategy.
"""

PROMPT_A_RESPONSE = json.dumps(
    {
        "projects": [
            {
                "name": "XYZ Industries electronics assembly plant, Devanahalli",
                "archetype_guess": "manufacturing_light",
                "location_text": "Devanahalli, Bengaluru",
                "announcer_name": "XYZ Industries",
                "is_public": False,
                "scale_value": None,
                "scale_unit": None,
                "capex_inr_cr": None,
                "stated_jobs": 10000,
                "stated_completion_text": None,
                "commitment_form_evidence": None,
                "quoted_by": "a company spokesperson",
                "is_reannouncement": False,
                "confidence": 0.82,
                "needs_location": False,
            },
            {
                "name": "XYZ Industries Karnataka multi-site expansion",
                "archetype_guess": "unknown",
                "location_text": None,
                "announcer_name": "XYZ Industries",
                "is_public": None,
                "scale_value": None,
                "scale_unit": None,
                "capex_inr_cr": None,
                "stated_jobs": None,
                "stated_completion_text": None,
                "commitment_form_evidence": None,
                "quoted_by": None,
                "is_reannouncement": False,
                "confidence": 0.6,
                "needs_location": True,
            },
        ],
        "document_type": "news",
        "notes": (
            "First record: 'up to 10,000 jobs' is a hedge, recorded literally as "
            "stated_jobs=10000. Second record covers an unspecified multi-site "
            "investment plan with no named location; needs_location flagged."
        ),
    }
)


def prompt_a_transport() -> RecordReplayTransport:
    return RecordReplayTransport([PROMPT_A_RESPONSE])


# --------------------------------------------------------------------------------------
# Prompt A fixture — malformed response, then a valid one on retry.
# --------------------------------------------------------------------------------------

PROMPT_A_MALFORMED_RESPONSE = "{not valid json"


def prompt_a_retry_then_ok_transport() -> RecordReplayTransport:
    return RecordReplayTransport([PROMPT_A_MALFORMED_RESPONSE, PROMPT_A_RESPONSE])


def prompt_a_always_malformed_transport() -> RecordReplayTransport:
    return RecordReplayTransport([PROMPT_A_MALFORMED_RESPONSE, PROMPT_A_MALFORMED_RESPONSE])


# --------------------------------------------------------------------------------------
# Prompt B fixture — a summit MoU, worded to tempt a naive classifier toward
# board_approved (it mentions a stock-exchange filing about the MoU itself, not about
# board approval of the project).
# --------------------------------------------------------------------------------------

PROMPT_B_SUMMIT_ARTICLE = """\
At the Global Investors Summit held in the state capital, ABC Energy signed a
memorandum of understanding with the state government to set up a 500 MW solar park.
The company's stock exchange filing disclosing the MoU noted that "no board resolution
approving capital expenditure has been passed at this stage" and that the project
remains subject to land allotment and further approvals.
"""

PROMPT_B_RESPONSE = json.dumps(
    {
        "commitment_form": "summit_mou",
        "evidence_quote": (
            "At the Global Investors Summit held in the state capital, ABC Energy "
            "signed a memorandum of understanding with the state government to set up "
            "a 500 MW solar park."
        ),
        "modifiers": [],
        "modifier_evidence": {},
        "confidence": 0.88,
        "ambiguity_note": (
            "The stock exchange filing disclosing the MoU could be mistaken for board "
            "approval, but the filing explicitly states no board resolution has been "
            "passed; commitment_form is summit_mou, not board_approved."
        ),
    }
)


def prompt_b_transport() -> RecordReplayTransport:
    return RecordReplayTransport([PROMPT_B_RESPONSE])


# --------------------------------------------------------------------------------------
# Prompt F fixture — a re-announcement article.
# --------------------------------------------------------------------------------------

PROMPT_F_EXISTING_PROJECT = json.dumps(
    {
        "project_id": "abc-energy-solar-park-01",
        "name": "ABC Energy 500 MW solar park",
        "stage": "feasibility",
        "commitment_form": "summit_mou",
    }
)

PROMPT_F_REANNOUNCEMENT_ARTICLE = """\
The chief minister today reiterated the state's commitment to the ABC Energy 500 MW
solar park, first announced at last year's investors summit, calling it "a flagship
project for the state's renewable energy push."
"""

PROMPT_F_RESPONSE = json.dumps(
    {
        "is_material_change": False,
        "change_type": "reannouncement",
        "proposed_updates": {},
        "evidence_quote": (
            "The chief minister today reiterated the state's commitment to the ABC "
            "Energy 500 MW solar park, first announced at last year's investors summit."
        ),
        "requires_human_review": False,
        "confidence": 0.91,
    }
)


def prompt_f_transport() -> RecordReplayTransport:
    return RecordReplayTransport([PROMPT_F_RESPONSE])


# --------------------------------------------------------------------------------------
# Prompt G fixture — a simulation output object, a correctly-cited narrative, and a
# tampered narrative with an injected wrong figure.
# --------------------------------------------------------------------------------------

SIM_RESULT_OBJECT = {
    "zones": {
        "KOM": {
            "factors": {
                "metro": {"lambda": 14},
            },
            "price_change_pct": 14,
        }
    },
    "confidence_tag": "R",
}

NARRATIVE_TEXT_VALID = (
    "Prices in the KOM zone rose 14% [zones.KOM.price_change_pct], driven largely by "
    "the new metro connection, whose effect on price is measured at 14 "
    "[zones.KOM.factors.metro.lambda]."
)

# Same claim, but the number before the second bracket has been tampered with (99
# instead of 14) — this must fail verification.
NARRATIVE_TEXT_INJECTED_WRONG_FIGURE = (
    "Prices in the KOM zone rose 14% [zones.KOM.price_change_pct], driven largely by "
    "the new metro connection, whose effect on price is measured at 99 "
    "[zones.KOM.factors.metro.lambda]."
)
