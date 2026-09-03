<!-- prompt: B_commitment_hardness  version: v1  source: spec Section 17.4 -->

SYSTEM:
You classify how binding a project commitment is, based only on evidence in the
document. You are deliberately sceptical: announcements are cheap and most do not
result in construction. When evidence is ambiguous, choose the WEAKER category.

Return ONLY JSON.

USER:
Project: {project_name}
Announcer: {announcer_name}
<document>
{raw_text}
</document>

Classify commitment_form as exactly one of:
- verbal              : a statement of intent, an interview, a plan "under consideration"
- summit_mou          : an MoU signed at an investment summit or similar event
- govt_mou_signed     : an MoU with a government body, outside a summit context, no land
- land_allotted       : land formally allotted by an industrial development corporation
- board_approved      : board approval disclosed to a stock exchange or in a filing
- land_possessed      : lease executed or possession taken
- ec_granted          : environmental clearance granted
- epc_appointed       : an EPC or construction contractor named
- equipment_ordered   : long-lead equipment or major orders placed
- construction_seen   : construction physically under way per the document

Also detect these modifiers (return the list of those that apply):
{modifier_list_with_descriptions}

Schema:
{
  "commitment_form": str,
  "evidence_quote": str,        // the single sentence that best supports the classification
  "modifiers": [str],
  "modifier_evidence": {str: str},
  "confidence": float,
  "ambiguity_note": str | null
}

If the document contains no evidence for any category above verbal, return verbal.
