<!-- prompt: D_delivery_record  version: v1  source: spec Section 17.6 -->

Run over a corpus of that announcer's historical announcements plus its annual reports.

SYSTEM:
You reconstruct a company's record of delivering on announced capital projects.
You work only from the documents provided. Where a project's outcome is not
determinable from the documents, mark it unknown rather than guessing.

USER:
Company: {name}
<announcements>
{announcement_docs}     // dated announcements over the review window
</announcements>
<financials>
{annual_report_extracts} // capex lines by year
</financials>

For each announced project, determine its outcome.

Schema:
{
  "projects": [{
    "name": str,
    "announced_date": str,          // ISO
    "announced_capex_inr_cr": float | null,
    "stated_completion": str | null,
    "outcome": str,                 // completed|under_construction|abandoned|stalled|unknown
    "outcome_evidence": str,
    "actual_completion": str | null,
    "slip_months": float | null
  }],
  "total_announced_inr_cr": float,
  "total_deployed_inr_cr": float | null,
  "deployed_source": str,           // how you determined deployed capex
  "coverage_note": str,             // what fraction of the window the documents cover
  "confidence": float
}

Rules:
- total_deployed must come from reported capex in the financials, not from summing
  project estimates.
- If the documents cover less than 60% of the review window, set confidence below 0.5
  and say so in coverage_note.
- "unknown" is an acceptable and often correct outcome. Do not force a classification.
