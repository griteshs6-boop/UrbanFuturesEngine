# Landed contracts — read this before writing a layer

Wave 1 is complete. These interfaces exist on disk and are tested. Use them; do not reinvent them.

## `ufe/store/schemas.py`

`SCHEMAS: dict[str, pandera.DataFrameSchema]` with keys
`cells`, `cells_history`, `projects`, `announcers`, `project_history`, `snapshots`.

Schemas are `strict=True` — an unknown column raises. If your layer adds a column to `cells`,
it must already be declared in the optional block below, or you must report the addition so the
schema owner can add it. Do not edit `ufe/store/schemas.py` yourself.

**`cells` required columns (34):**
`h3, h3_res8, in_city, geometry, lat, lon, area_sqm, elev_m, slope_pct, landcover, builtup_frac,
undevelopable_frac, zone_class, permitted_far, crz_class, population, households, hh_by_band,
jobs_by_sector, floorspace_res_sqm, floorspace_com_sqm, price_res_inr_sqft, price_land_inr_sqft,
rent_res_inr_sqft_mo, mean_parcel_sqm, parcel_count, util_water, util_sewer, util_power,
dist_cbd_m, dist_coast_m, dist_arterial_m, nightlight, data_conf`

Nullable among these: `price_res_inr_sqft`, `price_land_inr_sqft`, `rent_res_inr_sqft_mo`,
`mean_parcel_sqm`, `dist_coast_m`.

**`cells` optional columns (`required=False`, 27)** — these are the layer outputs:
`utility_state, slope_cost_mult, capacity_sqm, headroom_sqm, elasticity_class, eps_supply,
regulatory_index, lnA, lnA_work, lnA_retail, lnA_education, lnA_health, jobs_30min, jobs_45min,
jobs_60min, station_weight, retail_poi_count, education_poi_count, health_poi_count, school_seats,
hospital_beds, amenity, disamenity, alpha_res, inventory_months, hist_absorption_sqm,
dist_existing_builtup_m`

`hh_by_band` and `jobs_by_sector` are `DOUBLE[]` (python lists), length-checked against the
`IncomeBand` / `Sector` IntEnums exported as `INCOME_BANDS` and `SECTORS`.

## `ufe/store/db.py`

`connect(path, read_only=False)`, `migrate(con)`, `write_table(con, name, df)` (validates, raises
`ufe.errors.SchemaValidationError`), `read_table(con, name, **filters)`, `geometry_metadata(con)`,
`content_hash(df)`, `file_hash(p)`, `snapshot_hash(hashes)`, `write_snapshot(...)`.

Geometry: `cells.geometry` is WKB in a `BLOB`; `projects.geom` is WKT in `VARCHAR`. Both EPSG:4326.

## `tests/fixtures/synthetic.py`

```python
synthetic_cells(n=300, seed=20240101)
synthetic_cells_history(cells, seed)
synthetic_announcers(n=5, seed)
synthetic_projects(n=12, seed, announcers, cells)
synthetic_project_history(projects, seed)
build_city() -> SyntheticCity     # dataclass bundling all of the above
```
plus a session-scoped pytest fixture `synthetic_city`. Everything is seeded and reproducible.
Constants live in `tests/fixtures/synthetic.yaml`. **Use this fixture — do not build your own
synthetic city.** If it lacks a column you need, extend it additively and say so in your report.

## `ufe/params.py`

Per `CONTRACT.md`: `load_params(city)` -> `Params` with `.get/.value/.sample/.conf/.scope/.hash`.
All parameter YAML lives in `config/params/` and `config/cities/vizag.yaml`. Read the actual
files on disk to learn the real parameter paths — do not guess path names.

## `ufe/errors.py`

`UFEError, MissingCriticalLayer, ParameterScopeViolation, MissingParameter,
ParameterValidationError, SchemaValidationError, ConvergenceError, DeterminismError,
LicenceViolation, DataRightsViolation, BacktestGateFailure, CoverageError`

## `ufe/rights.py`

`assert_exposable(columns: Iterable[str]) -> None` raises `DataRightsViolation` for raw
OSM-derived columns. The API layer must call it.

## Environment facts

- Docker is NOT available. OSRM cannot run here. Route everything through the
  `TravelTimeBackend` protocol in `ufe/layers/routing.py` and test with the non-OSRM backend.
- No network at test time. Every test must pass offline.
- Real Vizag boundary is at `data/raw/boundaries/vizag_osm.geojson` (GVMC, 625 km², EPSG:4326).
- Run tests as `.venv/bin/python -m pytest`.

---

# Wave 2 addendum — more landed interfaces (read this too)

## `ufe/geo.py` (CRS discipline — use it, do not reimplement)
`GEOGRAPHIC_CRS`, `NonMetricCRSError`, `is_geographic(crs)`, `assert_metric_crs(crs)`,
`city_metric_crs(params)`, `to_metric`, `to_geographic`, `metric_area_sqm`,
`metric_distance_m`, `metric_buffer`. All accept shapely geometry, GeoSeries or GeoDataFrame.

## `ufe/grid/build.py`
Real Vizag grid: 136,566 res-9 cells (5,458 `in_city`) over the GVMC boundary plus a 50 km halo.
Mean cell area 114,562 m². Builds in ~9 s.

## `ufe/layers/l0_substrate.py`
`assemble_substrate(cells, params, *, gates, ward_jobs_2011, sector_growth, ward_col,
poi_columns, use_regression, regression_coefficients) -> pd.DataFrame`
Adds: `undevelopable_frac, slope_cost_mult, utility_state, capacity_sqm, headroom_sqm,
elasticity_class, eps_supply`, and `jobs_by_sector` when census inputs are supplied.
Also `fit_elasticity_regression(zones, params) -> RegressionFit`.

## `ufe/layers/routing.py`
`TravelTimeBackend` Protocol: `matrix(origins, destinations, profile) -> np.ndarray` of float32
MINUTES, `inf` = unreachable; origins/destinations are `(n,2)` lat/lon arrays.
`DistanceCapableBackend` adds `distance_matrix` in metres. `OSRMBackend` (real HTTP, needs Docker,
unavailable here). `HaversineBackend` (offline fallback: no circuity, no barriers, never returns
`inf` — optimistic). Only `precompute_matrices(...)` touches a backend; it caches float32 `.npy`
under `{cache}/ttm/{mode}/{key}.npy`. Simulation code consumes a `MatrixSet` value object and
must do NO I/O.

## `ufe/layers/l1_accessibility.py`
`apply_accessibility(cells, params, matrices: MatrixSet, ...) -> pd.DataFrame`
Adds: `lnA, lnA_work, lnA_retail, lnA_education, lnA_health, jobs_30min, jobs_45min, jobs_60min,
station_weight`.

## `ufe/layers/l3_credibility.py`
`completion_probability(...)`, `activation_weight(...)`, both pure, both taking an explicit
`force_project_state` argument for Section 10.5 counterfactual mode ('happens' / 'does_not_happen'
/ None) and an explicit `rng` where a draw is required. Also `slip_cdf`, `slip_median`, and an
`unknown_modifiers='raise'|'ignore'` argument.

## `ufe/layers/l4_supply.py`
`apply_supply(cells, params, *, year, state=None, demand_sqm=None, utility=None, effects=(),
inventory_months=None, hist_absorption_sqm=None) -> pd.DataFrame`
**State threading (the runner MUST do this):** the new state is attached at
`out.attrs[l4_supply.ATTR_KEY]` (`"supply_state"`). Read it and pass it as `state=` next year.
`SupplyState` is a frozen dataclass of h3-indexed Series: `built_sqm, capacity_sqm, headroom_sqm,
inventory_months, hist_absorption_sqm, committed_backlog_sqm`, plus `base_year: int`.
Diagnostics also on `.attrs`: `absorption_cap_sqm, delivered_sqm, organic_delivered_sqm,
backlog_delivered_sqm`.
`SupplyEffect(cell, delta_floorspace_sqm, delta_capacity_sqm, start_year)` is produced and
owned by **Layer 2** (`ufe/layers/l2_shocks.py`, spec Section 9.1). `l4_supply` imports and
re-exports it (`SupplyEffect = _SupplyEffect`), so `l4_supply.SupplyEffect` and
`l2_shocks.SupplyEffect` are literally the same class. There is no duplicate definition; do not
create one.

## Known open issues (do not "fix" someone else's file — report instead)
- `config/params/archetypes.yaml` defines only 3 of the 22 archetypes (`metro_rail`,
  `data_centre`, `electronics_assembly`) — the spec references a `.docx` section that was not
  supplied. **Cascade cannot resolve `target_archetype` for the other 19 until they exist**, and
  raises `ufe.errors.MissingArchetypeError` rather than inventing values. Design around this and
  report it rather than inventing archetype values.

## Resolved (these were open issues; they are fixed — do not re-report them)
- `accessibility.grid.halo_buffer_m` **exists** (`config/params/accessibility.yaml`, 50 000 m,
  `conf: G`, `scope: global`).
- `accessibility.modes.two_wheeler.speed_factor` **exists**
  (`config/params/accessibility.yaml`, 0.85, `conf: R`, `scope: local`).
- `config/cities/vizag.yaml` `boundary_source` now points at the real file on disk,
  `data/raw/boundaries/vizag_osm.geojson` (GVMC).
- The L4/L2 `SupplyEffect` duplication is gone — see the single-definition note above.
