# Dependencies and licence audit

Generated against the environment installed under `.venv/`, one row per direct dependency
declared in `pyproject.toml` (main + `dev` extra). Licence and version were verified with
`importlib.metadata` (equivalently `pip show`) against the *installed* distribution, not copied
from spec comments. See "Discrepancies vs `s02_2_environment.md`" below.

Class follows Section 2.4:

- **Green** — MIT, BSD-2/3, Apache-2.0, ISC, PSF, CC0. Use freely.
- **Amber** — LGPL, MPL-2.0, EPL. Dev-only, or a separate unmodified process, with written
  justification.
- **Red** — AGPL-3.0, GPL-2/3, SSPL, BUSL, any source-available licence. Not permitted.

## Runtime dependencies

| Package | Version (installed) | Licence (verified) | Class | Why it is needed |
|---|---|---|---|---|
| pandas | 2.3.3 | BSD-3-Clause | Green | Core dataframe type used across the whole engine. |
| numpy | 2.4.6 | BSD-3-Clause (bundles 0BSD/MIT/Zlib/CC0-1.0 for vendored components, all Green) | Green | Array math underlying pandas, geopandas, statsmodels, scikit-learn. |
| pyarrow | 25.0.1 | Apache-2.0 | Green | Parquet I/O for snapshots (`cells.parquet`) and DuckDB interop. |
| duckdb | 1.5.5 | MIT | Green | Embedded store for all tabular data (Section 2.2). |
| geopandas | 1.1.4 | BSD-3-Clause | Green | Geometry-aware dataframes for the grid and layer modules. |
| shapely | 2.1.2 | BSD-3-Clause | Green | Geometry primitives underlying geopandas/osmnx. |
| pyproj | 3.7.2 | MIT | Green | CRS transforms (EPSG:4326 <-> `crs_metric`, Section 0.1 rule 7). |
| rasterio | 1.4.4 | BSD-3-Clause | Green | Reads DEM / land-cover / satellite rasters. |
| rtree | 1.4.1 | MIT | Green | Spatial index backing geopandas nearest/overlay ops. |
| h3 | 4.5.0 | Apache-2.0 | Green | H3 hexagon indexing — the grid's primary key. |
| osmnx | 2.1.1 | MIT | Green | OSM network/POI download and graph construction for ingestion. |
| networkx | 3.6.1 | BSD-3-Clause | Green | Graph structures used by osmnx and the transit graph (Section 8.3). |
| exactextract | 0.3.0 | Apache-2.0 | Green | Exact cell-area-weighted zonal stats (nightlights, land cover) — replaces `rasterstats` per Section 2.1b. |
| tobler | 0.13.0 | BSD-3-Clause | Green | PySAL areal interpolation (dasymetric population refinement, Section 6.4). |
| libpysal | 4.14.1 | BSD-3-Clause | Green | Spatial weights/dependency of `tobler`/`spreg`. |
| spreg | 1.9.0 | BSD-3-Clause | Green | Spatial regression for calibration work. |
| momepy | 0.11.0 | BSD-3-Clause | Green | Urban morphology metrics (optional per spec, kept as a direct dependency). |
| statsmodels | 0.15.0 | BSD-3-Clause | Green | Discrete-choice / regression models in the behaviour and credibility layers. |
| scikit-learn | 1.9.0 | BSD-3-Clause | Green | General ML utilities (imputation, scaling, clustering). |
| scipy | 1.17.1 | BSD-3-Clause | Green | Numerical routines underlying statsmodels/scikit-learn/spreg. |
| xlogit | 0.2.7 | MIT | Green | Mixed logit estimation for discrete location-choice models. |
| SALib | 1.5.2 | MIT | Green | Sensitivity analysis for the Monte Carlo module. |
| stackstac | 0.5.1 | MIT | Green | Lazy xarray access to STAC-catalogued satellite imagery. |
| pystac-client | 0.9.0 | Apache-2.0 | Green | STAC API client for satellite scene discovery. |
| xarray | 2026.7.0 | Apache-2.0 | Green | Labeled arrays for satellite raster stacks. |
| dask | 2026.8.0 | BSD-3-Clause | Green | Lazy/chunked computation backing `stackstac`/`xarray`. |
| pandera | 0.33.1 | MIT | Green | DataFrame schema validation on every store write (Section 3, Contract). |
| pydantic | 2.7+ (installed 2.13.5) | MIT | Green | Parameter-tree and config validation. |
| PyYAML | 6.0.3 | MIT | Green | All `config/*.yaml` and `config/params/*.yaml` parsing. |
| typer | 0.27.2 | MIT | Green | `ufe` CLI framework. |
| rich | 15.0.0 | MIT | Green | CLI/report progress and table rendering (replaces `tqdm`, Section 2.1). |
| httpx | 0.28.1 | BSD-3-Clause | Green | HTTP client for OSRM and STAC calls. |
| anthropic | 1.3.0 | MIT | Green | AI pipeline (Module 13) only — never imported by `ufe.layers`/`ufe.sim`/`ufe.backtest` (Contract rule 4). |
| fastapi | 0.141.1 | MIT | Green | API layer serving computed outputs (Section 22.1). |
| uvicorn | 0.52.4 | BSD-3-Clause | Green | ASGI server for the FastAPI app. |

## Dev-only dependencies (`[project.optional-dependencies].dev`, never shipped)

| Package | Version (installed) | Licence (verified) | Class | Why it is needed |
|---|---|---|---|---|
| pytest | 9.1.1 | MIT | Green | Test runner. |
| pytest-cov | 7.1.0 | MIT | Green | Coverage reporting in CI. |
| hypothesis | 6.167.1 | MPL-2.0 | **Amber** | Property-based testing. **Justification:** dev-only, never imported by shipped code or the built wheel (`[tool.hatch.build.targets.wheel] packages = ["ufe"]` excludes `tests/`); MPL-2.0's file-level copyleft is triggered only by modifying and distributing `hypothesis`'s own source files, which we do not do. Matches the explicit allowance in `s02_2_environment.md` Section 2.1 ("MPL-2.0 — dev-only, acceptable"). |
| pip-licenses | 5.5.5 | MIT | Green | Reference tool for cross-checking installed-package licence metadata; not imported by `ufe.licences` (which reads `importlib.metadata` directly to avoid a runtime dependency on it), but kept available for manual audits. |

## Tally

| Class | Count |
|---|---|
| Green | 37 |
| Amber | 1 (`hypothesis`, dev-only, justified above) |
| Red | 0 |

## Discrepancies vs `s02_2_environment.md`

None found. Every licence string reported by the installed distribution's metadata (via
`importlib.metadata.distribution(name).metadata`, cross-checked with `Classifier:` trove
entries where the `License` field held full licence text rather than an SPDX-style short name —
namely `pandas`, `h3`, `exactextract`, `scipy`, `shapely`, `rasterio`, `libpysal`, `tobler`,
`momepy`, `pandera`, `SALib`, `pystac-client`, `pip-licenses`) matches the class the spec asserts
for that package. `hypothesis` is MPL-2.0 exactly as the spec's Amber note anticipates.

This DEPENDENCIES.md is read directly by `ufe/licences.py`: `ufe licences audit` fails the
build if a direct dependency present in the environment has no row here, or if any row's
class is Red.
