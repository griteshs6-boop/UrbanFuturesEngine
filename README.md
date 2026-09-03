# Urban Futures Engine

A parameterised, deterministic simulation of how a city's land and housing market responds to
infrastructure and industrial announcements. Given a frozen snapshot of a city — a hex grid, a
substrate of terrain / land cover / zoning / utilities, a travel-time network, a population and
jobs baseline, and a registry of announced projects — the engine walks year by year to a horizon
and reports, per cell:

- how much accessibility changed and why,
- how credible each announced project is and how much of it to believe this year,
- how much floorspace could physically and legally be delivered,
- where households and firms would locate given that,
- what price clears the market, split into a fundamental and a speculative overshoot,
- and a leave-one-out attribution of that price movement to named factors ("the metro", "the
  data centres"), reported alongside the interaction term it cannot separate.

The pipeline is a stack of layers, each a pure function of `(cells, params, ...)` returning a new
frame:

| Layer | Module | What it does |
|---|---|---|
| L0 | `ufe/layers/l0_substrate.py` | Developability, capacity, headroom, supply elasticity |
| L1 | `ufe/layers/l1_accessibility.py` | `lnA` and cumulative-opportunity access, per purpose |
| L2 | `ufe/layers/l2_shocks.py` | Announced projects → employment / network / field / supply effects |
| L3 | `ufe/layers/l3_credibility.py` | Completion probability and activation weight per project |
| L4 | `ufe/layers/l4_supply.py` | Absorption-capped delivery, inventory, backlog |
| L5 | `ufe/layers/l5_allocation.py` | Household and firm location choice |
| L6 | `ufe/layers/l6_price.py` | Market clearing, overshoot, factor decomposition, overheating |

Around them: `ufe/grid` (H3 grid construction), `ufe/ingest` (raster and vector ingesters plus
state adapters), `ufe/store` (DuckDB + pandera schemas + content-hashed snapshots), `ufe/sim`
(runner, Monte Carlo, factor decomposition), `ufe/backtest` (freeze, baselines, scoring, ship
gate), `ufe/ai` (an offline LLM extraction pipeline that never runs at simulation time),
`ufe/satellite` (change detection), and `ufe/api` (a read-only FastAPI surface with a
data-rights guard).

Two rules run through everything and are enforced by tests:

- **No numeric parameter lives in Python.** Every coefficient, threshold, elasticity and
  tolerance is in `config/params/*.yaml` with a confidence class (`E`/`R`/`G`) and a scope
  (`global`/`local`). The only literals in `.py` are `0`, `1` and array indices.
- **Determinism.** Same snapshot + same seed + same params hash ⇒ byte-identical output. RNGs
  are always explicit `numpy.random.Generator`s; sets are sorted before they are iterated.

---

## Environment constraints — read this before you try to run anything

These are properties of the environment this repository was built in, not bugs.

- **Docker is not available, so OSRM cannot run here.** Real travel times require a running
  `osrm-routed`. All routing therefore goes through the `TravelTimeBackend` protocol in
  `ufe/layers/routing.py`. `OSRMBackend` is written and is the production path; the offline
  fallback used by every test is `HaversineBackend`, which has **no circuity, no barriers, and
  never returns `inf`** — it is systematically optimistic and is a test double, not a substitute.
  Tests that genuinely need OSRM are written and marked `@pytest.mark.needs_osrm`, and are
  skipped here.
- **No real source data is present.** The only real file on disk is the GVMC boundary at
  `data/raw/boundaries/vizag_osm.geojson`. There are no DEMs, no land-cover rasters, no
  building footprints, no price panel, no RERA extract, no cadastral or zoning digitisation.
  Everything the test suite exercises runs against the seeded synthetic city in
  `tests/fixtures/synthetic.py`. Tests needing real rasters are marked
  `@pytest.mark.needs_data` and are skipped.
- **Tests run fully offline.** No test makes a network call, reads an API key, or contacts an
  LLM. `ufe/ai` is exercised against recorded fixtures.
- **`config/params/archetypes.yaml` defines 3 of the 22 project archetypes** the spec
  references (`metro_rail`, `data_centre`, `electronics_assembly`); the source document for the
  rest was not supplied. Cascade raises `MissingArchetypeError` for the others rather than
  inventing a sector and a median wage.

---

## Install and run the tests

Python 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install the project + pytest
uv run pytest                # the whole suite, offline
```

Equivalently, against the existing virtualenv:

```bash
.venv/bin/python -m pytest -q
```

Useful selections:

```bash
uv run pytest -m acceptance          # only the spec's ACCEPTANCE blocks
uv run pytest -m "not slow"
uv run pytest tests/unit/test_l6_price.py
```

Markers: `acceptance` (maps 1:1 onto the spec's ACCEPTANCE blocks), `slow`, `needs_osrm`,
`needs_data`. The last two are skipped in this environment for the reasons above.

CI (`.github/workflows`) runs the licence audit first and the test suite second; a Red-class
package licence or an unmapped data source fails the build before any test runs.

---

## The `ufe` CLI

`ufe doctor` reports which sub-commands mounted and, for anything missing, why. The surface:

```
ufe doctor                 which sub-commands are available, and why any are not
ufe version                installed engine version
ufe run                    one deterministic run — spec Section 23 item 2;
                           the same command as `ufe sim run` (same options)

ufe licences  audit
ufe params    estimate
ufe store     migrate | tables | snapshot
ufe grid      build
ufe ingest    national | state | city | coverage | adapters
ufe sim       run | montecarlo | factors | manifest
ufe backtest  gate | freeze | rolling | loco | ablate | sobol
ufe ai        list | show | approve | reject | edit
ufe satellite collect | detect | priority-tier
ufe api       serve | routes | exposure | attributions
```

The spec's documented invocation is `ufe run --city vizag --horizon 2035`. It also needs
`--snapshot`, because the engine reads a content-hashed snapshot rather than a live database:
every number it prints traces back to a snapshot hash, a params hash and a git commit.

Note that no snapshot exists in this repository (see "no real source data" above), so `ufe run`
has nothing to run against here.

---

## Build-sequence status (spec Section 20)

Spec Section 20 lays out 22 steps. "Built" below means the module exists, is importable, and its
ACCEPTANCE block is a passing test against synthetic fixtures. It does **not** mean the module
has ever seen real data.

| # | Step | Status |
|---|---|---|
| 0 | `DEPENDENCIES.md`, `ATTRIBUTIONS.md`, `ufe licences audit`, CI gate | Built |
| 1 | `ufe/params.py` + YAML, including `scope` enforcement | Built |
| 2 | `ufe/store/` — DuckDB, schemas, migrations | Built |
| 3 | `ufe/grid/build.py` (Module 1) | Built |
| 4 | `ufe/ingest/` core + `adapters/base.py`, `adapters/ap.py` | Built; never run on real rasters |
| 5 | `ufe/layers/l0_substrate.py` (Module 3) | Built |
| 6 | OSRM setup + `ufe/layers/l1_accessibility.py` (Module 4) | Layer built; **OSRM not runnable here** — tested on `HaversineBackend` only |
| 7 | `ufe/layers/l3_credibility.py` (Module 6) | Built |
| 8 | `ufe/layers/l2_shocks.py` (Module 5) | Built; only 3 of 22 archetypes are parameterised |
| 9 | `ufe/layers/l4_supply.py` (Module 7) | Built |
| 10 | `ufe/layers/l5_allocation.py` (Module 8) | Built; calibration exercised on synthetic data only — see below |
| 11 | `ufe/layers/l6_price.py` (Module 9) | Built |
| 12 | `ufe/sim/runner.py` (Module 11) | Built |
| 13 | `ufe/backtest/` (Module 15) | Built |
| 14 | **GATE: run the backtest. If it fails B2, stop and reassess.** | **NOT RUN.** No historical panel exists. See below. |
| 15 | `ufe/layers/cascade.py` (Module 10) | Built; blocked on the missing archetypes for 19 of 22 targets |
| 16 | `ufe/sim/montecarlo.py` (Module 12) | Built |
| 17 | `ufe/sim/factors.py` — decomposition | Built |
| 18 | Remaining ingesters — prices, RERA, cadastral, zoning, nightlights | Built; never run on real sources |
| 19 | `ufe/ai/` (Module 13) | Built; exercised against recorded fixtures, no live LLM calls |
| 20 | `ufe/satellite/` (Module 14) | Built; no imagery collected |
| 21 | `ufe/api/` + report rendering + attributions footer | Built |
| 22 | Second-city validation (Section 20.3) | **Not done.** `config/cities/` contains only `vizag.yaml` |

Steps 15–21 were built despite step 14 not having been run. Section 20's note on step 14 says
"This is a real stop... If the model does not beat momentum out of sample, do not build the
interface, the Monte Carlo, or the AI pipeline." Those things exist here as code that has not
earned its place. Treat them accordingly.

---

## What is NOT validated

**This engine has never been validated against real data. There is no evidence that it
predicts anything.** Specifically:

1. **The Section 19 backtest gate is UNRUN.** `ufe backtest gate` is implemented and tested —
   against synthetic and deliberately-weak fixtures, which is a test that the gate *logic*
   works, not a test that the *model* passes it. The gate itself
   (`median spearman > 0.55`, `median beat_b2 > 0`, bootstrap CI lower bound `> 0`, band coverage
   in `[0.70, 0.90]`, on at least three hold-out cities) has never been evaluated against a
   historical price panel, because no historical price panel is present. Spec Section 20 step 14
   is a hard stop that has not been cleared.

2. **There is no claim that the model beats momentum.** Baseline B2 is "last five years'
   appreciation, extrapolated". Whether this engine outperforms that naive extrapolation
   out of sample is **unknown**. It has not been measured. Nothing in this repository should be
   read as asserting otherwise, and any output it produces should be treated as an illustration
   of the mechanism, not a forecast.

3. **The Section 20.2 step 10 cell fixed-effect calibration has only been exercised on
   synthetic data.** `alpha_i` estimation and the Module 8 null test run, pass, and are covered
   by tests — on the seeded synthetic city, whose fixed effects were generated by the same
   assumptions the estimator recovers. The spec's own warning applies in full: *"Step 10 is the
   one that gets skipped. It is also the one that silently destroys the model."* Passing the
   null test on synthetic data does not tell you it will pass on Vizag.

Further, and less centrally: accessibility has never been computed on real road-network travel
times (Haversine only); no city carries a real `calibration_level`; supply elasticities have not
been hand-checked against known constrained and unconstrained localities (Section 20.2 step 11);
logit coefficients have not been estimated from an observed household distribution (step 12);
and mode split and macro trend are city-class defaults, not local survey and index data
(steps 13–14). Per Section 20.2, **a city on class defaults is a demonstration, not a product.**

What the test suite *does* establish is narrower and worth stating precisely: that the code
implements the specified equations, that every ACCEPTANCE block in the spec passes, that the
parameter discipline holds, that runs are reproducible byte for byte, and that the
architectural boundaries (no LLM at simulation time, no network at simulation time, no raw
OSM-derived column exposed by the API, no `scope: global` override from a city config) are
enforced rather than merely intended.

---

## Repository map

```
config/          parameter YAML (params/, cities/, city_classes.yaml) — every number lives here
data/            raw/, cache/, snapshots/ — only the GVMC boundary is real
osrm/            OSRM profiles; unusable without Docker
ufe/             the engine (see the layer table above)
tests/           unit/, integration/, fixtures/ — the synthetic city lives in fixtures/
CONTRACT.md      build conventions: the non-negotiable rules
SCHEMA_CONTRACT.md  landed interfaces and the remaining open issues
DEPENDENCIES.md  dependency licence classification (Section 2.4)
ATTRIBUTIONS.md  data-source attribution required in every report
```
