# Build conventions — read before writing any code

This file is the shared contract between parallel build agents. The authoritative source is
the spec (`/agent/workspace/spec/s*.md`). Where this file and the spec disagree, the spec wins,
but flag the disagreement.

## Non-negotiable rules (spec Section 0.1)

1. **No numeric parameter in Python.** Every coefficient, threshold, elasticity, multiplier
   lives in YAML under `config/params/`. The only literals allowed in `.py` are `0`, `1`, and
   array indices. Not `2`, not `0.5`, not `1e-6`. If you need a tolerance, it goes in YAML.
2. **Pure functions.** Every function that transforms cell state takes `(df, params, ...)` and
   returns a NEW dataframe. No in-place mutation, no module-level mutable state.
3. **No network calls at simulation time.** Anything under `ufe/layers/` and `ufe/sim/` must not
   perform I/O beyond reading the local store.
4. **No LLM calls at simulation time.** No module under `ufe/layers/`, `ufe/sim/`, or
   `ufe/backtest/` may import `ufe.ai`. There is a test for this.
5. **Determinism.** Same inputs + same seed + same params version = byte-identical output.
   Never use unseeded RNG. Always thread an explicit `numpy.random.Generator` or seed.
6. **Tests before implementation.** Every module's ACCEPTANCE block in the spec becomes a test
   marked `@pytest.mark.acceptance`.
7. **CRS.** Geometry is stored in EPSG:4326. All metric computation (distance, area, buffer)
   happens in the city's `crs_metric`. Never compute distance in degrees.

## Units (spec Section 0.3)

metres · minutes (float) · square metres · INR · INR/sqft for price · integer calendar year ·
probability 0–1 · natural log (`ln`) throughout.

## Python conventions

- Python 3.11. Type hints on every public function. `from __future__ import annotations`.
- Modules are importable without side effects. No work at import time.
- Raise, never warn, on invalid data. Custom exceptions live in `ufe/errors.py`.
- Logging via `logging.getLogger(__name__)`. Never `print` outside `ufe/cli.py`.

## Shared interfaces you must use, not reinvent

```python
# ufe/errors.py
class UFEError(Exception): ...
class MissingCriticalLayer(UFEError): ...
class ParameterScopeViolation(UFEError): ...
class ConvergenceError(UFEError): ...
class SchemaValidationError(UFEError): ...
class MissingParameter(UFEError): ...

# ufe/params.py
class Params:
    """Loaded, validated parameter tree. Access by dotted path."""
    def get(self, path: str) -> Any: ...          # scalar/resolved value, raises MissingParameter
    def value(self, path: str) -> float: ...      # deterministic value of a scalar-or-range
    def sample(self, path: str, rng) -> float: ...# Monte Carlo draw from a range
    def conf(self, path: str) -> str: ...         # 'E' | 'R' | 'G'
    def scope(self, path: str) -> str: ...        # 'global' | 'local'
    @property
    def hash(self) -> str: ...                    # sha256 of the resolved tree, for provenance

def load_params(city: str, params_dir: Path = ..., cities_dir: Path = ...) -> Params: ...

# ufe/store/db.py
def connect(path: str | Path = "data/ufe.duckdb", read_only: bool = False) -> duckdb.DuckDBPyConnection: ...
def migrate(con) -> None: ...
def write_table(con, name: str, df: pd.DataFrame) -> None:   # validates against pandera schema, raises on invalid
def read_table(con, name: str, **filters) -> pd.DataFrame: ...

# ufe/store/schemas.py
CELLS: pandera.DataFrameSchema
CELLS_HISTORY: pandera.DataFrameSchema
PROJECTS: pandera.DataFrameSchema
ANNOUNCERS: pandera.DataFrameSchema
PROJECT_HISTORY: pandera.DataFrameSchema
SNAPSHOTS: pandera.DataFrameSchema
SCHEMAS: dict[str, pandera.DataFrameSchema]
```

Every layer module exposes ONE public entry point named for its layer, e.g.:

```python
def apply_accessibility(cells: pd.DataFrame, params: Params, **kw) -> pd.DataFrame: ...
```

It must return a new frame with the same row count and index as the input, plus new columns.

## Travel-time backend (Section 8.3)

Docker is NOT available in this environment, so OSRM cannot be run here. Accessibility must be
written against an abstract backend so tests pass without OSRM:

```python
# ufe/layers/routing.py
class TravelTimeBackend(Protocol):
    def matrix(self, origins: np.ndarray, destinations: np.ndarray, profile: str) -> np.ndarray:
        """Return an (n_origins, n_destinations) matrix of travel times in MINUTES."""

class OSRMBackend:    # real; HTTP to a running osrm-routed, ingestion/precompute time only
class HaversineBackend:  # deterministic fallback for tests, speeds read from YAML
```

The precomputed matrix is persisted to the store; simulation reads it and never calls a backend.

## Testing

- `pytest -m acceptance` must map 1:1 onto the spec's ACCEPTANCE blocks.
- Tests that need OSRM: `@pytest.mark.needs_osrm`. Tests that need real rasters:
  `@pytest.mark.needs_data`. Both are skipped by default in this environment but must be written.
- Synthetic fixtures live in `tests/fixtures/`. Build a small synthetic city (a few hundred
  hexes) once, in `tests/fixtures/synthetic.py`, and reuse it.

## File ownership

Do not edit files outside the set assigned to you in your task brief. If you need a change in
someone else's file, report it in your final summary instead of making it.
