# Attributions

This file lists every attribution the product owes, either to a software licence (Section 2.4)
or to a data source (Section 22.2/22.4). The data-source portion is machine-readable at
`config/data_sources_licences.yaml`, and `ufe.rights.get_attribution_text()` renders it (in
full, or filtered to the sources a specific artefact used) for embedding in report footers, the
product about page, and the API `/attributions` endpoint. This file and that function must stay
in sync — the function is the source of truth for the data-source block; this file mirrors it
for human readers and adds the software attributions the function does not cover.

## Data sources (render into every report footer / about page / `/attributions`)

- © OpenStreetMap contributors, under the Open Database Licence (ODbL) 1.0.
- Building footprints from the Google-Microsoft Open Buildings dataset, CC BY 4.0.
- Land cover from ESA WorldCover, CC BY 4.0.
- Contains modified Copernicus data, under the Copernicus licence.
- Contains modified Copernicus Sentinel data, under the Copernicus open licence.
- VIIRS nightlights courtesy of NOAA/NASA and the Earth Observation Group (EOG).
- Population estimates from WorldPop, CC BY 4.0.
- Settlement data from the Global Human Settlement Layer (GHSL), CC BY 4.0.
- Government of India / State Government open data portals, per-portal terms apply.
- Registration/RERA/EC portal data used under each portal's published terms.

Listing-portal and broker-panel data carry no public attribution string by design — see
Section 22.2/22.3: they are proprietary/contractual sources, never redistributed, and never
themselves exposed in a paid product. If a specific licensing arrangement with a listing
portal or broker requires a named attribution, add it to `config/data_sources_licences.yaml`
under that source's `attribution` field before it is used in anything customer-facing.

## Software (Section 2.4)

All 37 Green-class runtime and dev dependencies (MIT, BSD-2/3, Apache-2.0, ISC) require, at
minimum, retention of their copyright notice and licence text when redistributed — satisfied by
shipping the packages themselves (each carries its own `LICENSE`/`COPYING` file inside its
distribution) rather than by reproducing every notice here. See `DEPENDENCIES.md` for the full
per-package table (name, version, licence, class, why it is needed).

One Amber-class dependency:

- **hypothesis** (MPL-2.0) — dev-only, not shipped in the built wheel (`[tool.hatch.build...
  ] packages = ["ufe"]` excludes `tests/`). No file-level obligation is triggered because we do
  not modify or redistribute `hypothesis`'s own source.

Zero Red-class dependencies.

## How this file is kept honest

- `DEPENDENCIES.md` is the source of truth for the software table and is checked by
  `ufe licences audit` (fails the build on an undocumented direct dependency or a Red-class
  package).
- `config/data_sources_licences.yaml` is the source of truth for the data-source table and is
  checked by `ufe licences audit --data` against `config/sources.yaml`.
- `ufe.rights.get_attribution_text()` raises `ufe.errors.DataRightsViolation` if a report or
  page asks for an attribution for a source that isn't in `config/data_sources_licences.yaml`,
  which is how Section 22.4's "a report build that cannot resolve an attribution ... must fail"
  requirement is enforced in code rather than only in this document.
