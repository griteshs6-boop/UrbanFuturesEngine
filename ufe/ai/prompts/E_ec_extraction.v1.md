<!-- prompt: E_ec_extraction  version: v1  source: spec Section 17.7 -->

EC filings are structured and high-value: they carry coordinates, land area, capacity and stage.

SYSTEM:
You extract project data from an environmental clearance application or grant.
These documents are structured; extract literally.

USER:
<document>{raw_text}</document>

Schema:
{
  "proposal_number": str | null,
  "project_name": str,
  "proponent_name": str,
  "category": str | null,
  "latitude": float | null,
  "longitude": float | null,
  "land_area_ha": float | null,
  "capacity_value": float | null,
  "capacity_unit": str | null,
  "capex_inr_cr": float | null,
  "clearance_status": str,   // applied|tor_granted|ec_granted|rejected|withdrawn
  "status_date": str | null,
  "employment_direct": float | null,
  "employment_indirect": float | null,
  "confidence": float
}

Coordinates are usually given in DMS. Convert to decimal degrees and state the
original string in a "coord_source" field. If multiple coordinate pairs are given
(polygon vertices), return the centroid and put all pairs in "coord_all".
