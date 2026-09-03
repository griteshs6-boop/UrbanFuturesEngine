"""Shared machinery for Module 2 — data ingestion (spec Section 6).

There is no real source data in this environment and no network at test time, so every
ingester is split in two halves and only the second half is ever exercised offline:

**(a) the fetch/read step** goes through :class:`SourceReader`, an injectable interface.
    Three implementations ship here: :class:`LocalFileReader` (files already on disk),
    :class:`InMemoryReader` (the test double, fed by ``tests/fixtures/raster_fixtures.py``)
    and :class:`HttpArchiveReader` — the *only* class in ``ufe/ingest`` that touches the
    network, and the only one a test must never construct.

**(b) the transform step** is pure: ``parse(raw) -> DataFrame`` and
    ``to_cells(df, cells) -> DataFrame``. It takes data and a cell frame and returns a new
    frame of ``cells`` columns. No I/O, no module state, no mutation of the input
    (CONTRACT.md rule 2). Every test in ``tests/unit/test_ingest.py`` drives this half.

Provenance and the ``ingest_runs`` ledger
-----------------------------------------
Section 6: "Every ingest run writes a row to an ``ingest_runs`` table. A cell attribute
with no corresponding ingest run is invalid." :func:`ingest_run` builds that row,
:func:`write_ingest_runs` persists it, and :func:`assert_every_column_has_run` is the
Section 6 ACCEPTANCE check. ``provenance()`` is assembled from ``config/sources.yaml`` and
cross-checked against ``config/data_sources_licences.yaml``, so an ingester wired to an
unregistered source or an unrecognised licence string raises
:class:`ufe.errors.LicenceViolation` at construction rather than shipping unattributed
data (Section 22.2).

How imputation is flagged
-------------------------
Every transform that fills a value it did not observe marks it, in the frame, with two
companion columns per value column::

    <column>__imputed        bool    True where the value was not observed
    <column>__impute_method  str     short machine-readable reason, "" where observed

``cells`` is a ``strict=True`` pandera schema, so those companions cannot be stored in it.
:func:`imputation_long` melts them into the tidy ``cell_imputation(h3, column, imputed,
method, source_id)`` ledger that is persisted alongside ``ingest_runs``, and
:func:`data_conf` folds them into the ``cells.data_conf`` column the schema does declare.
Nothing is ever imputed silently: an unflagged fill is a bug, and the coverage report in
:mod:`ufe.ingest.coverage` is computed from these flags, not from null-counting.

Numeric policy
--------------
Zero numeric literals beyond ``0`` and ``1`` and array indices. Every constant comes from
``config/ingest.yaml`` via :func:`load_ingest_config`, or from ``Params`` when the spec says
it is a model parameter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

from ufe import geo
from ufe.errors import LicenceViolation, UFEError

logger = logging.getLogger(__name__)

__all__ = [
    "IMPUTED_SUFFIX",
    "METHOD_SUFFIX",
    "INGEST_RUNS_COLUMNS",
    "CELL_IMPUTATION_COLUMNS",
    "INGEST_RUNS_TABLE",
    "CELL_IMPUTATION_TABLE",
    "MissingSource",
    "IngestConfigError",
    "CityConfig",
    "SourceReader",
    "LocalFileReader",
    "InMemoryReader",
    "HttpArchiveReader",
    "Ingester",
    "load_ingest_config",
    "cfg",
    "cells_gdf",
    "zonal",
    "raster_crs",
    "mark_imputed",
    "value_columns",
    "flag_columns",
    "imputation_long",
    "data_conf",
    "merge_ingested",
    "provenance_for",
    "ingest_run",
    "write_ingest_runs",
    "read_ingest_runs",
    "write_cell_imputation",
    "assert_every_column_has_run",
]

# --------------------------------------------------------------------------------------
# Column naming conventions
# --------------------------------------------------------------------------------------

#: Companion boolean column marking an imputed value.
IMPUTED_SUFFIX = "__imputed"
#: Companion string column naming the imputation method.
METHOD_SUFFIX = "__impute_method"

#: The ``ingest_runs`` ledger (Section 6). There is no pandera schema for it in
#: ``ufe/store/schemas.py`` — that file belongs to another module, so the addition is
#: reported rather than made, and this module writes the table with plain SQL.
INGEST_RUNS_TABLE = "ingest_runs"
INGEST_RUNS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "source_id",
    "city_id",
    "tier",
    "columns",
    "rows",
    "url",
    "retrieved_at",
    "licence",
    "spatial_res",
    "temporal_res",
    "notes",
    "content_hash",
    "params_hash",
    "created_at",
)

#: The tidy imputation ledger described in the module docstring.
CELL_IMPUTATION_TABLE = "cell_imputation"
CELL_IMPUTATION_COLUMNS: tuple[str, ...] = (
    "h3",
    "column",
    "imputed",
    "method",
    "source_id",
    "run_id",
)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ingest.yaml"
_SOURCES_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
_DATA_LICENCES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "data_sources_licences.yaml"
)

_H3 = "h3"


class MissingSource(UFEError):
    """A :class:`SourceReader` was asked for a key it does not hold."""


class IngestConfigError(UFEError):
    """``config/ingest.yaml`` is missing a key an ingester needs."""


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_yaml(path: str) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise IngestConfigError(f"{path} does not contain a YAML mapping")
    return data


def load_ingest_config(path: str | Path = _CONFIG_PATH) -> Mapping[str, Any]:
    """The parsed ``config/ingest.yaml`` (cached per path)."""
    return _load_yaml(str(path))


def cfg(dotted: str, config: Mapping[str, Any] | None = None) -> Any:
    """Look up a dotted path in ``config/ingest.yaml``, raising if it is absent.

    This is the "clearly-named lookup that fails loudly" that replaces a Python literal.
    """
    node: Any = load_ingest_config() if config is None else config
    for token in dotted.split("."):
        if not isinstance(node, Mapping) or token not in node:
            raise IngestConfigError(
                f"config/ingest.yaml has no key {dotted!r} (stopped at {token!r})"
            )
        node = node[token]
    return node


# --------------------------------------------------------------------------------------
# City configuration (the `CityConfig` of the Section 6 protocol signatures)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CityConfig:
    """The city-level facts every ingester and state adapter needs (Section 6.0).

    Built from a loaded :class:`ufe.params.Params` so that nothing here is a literal:
    ``crs_metric``, ``state_code``, ``base_year`` and the CBD point all come from
    ``config/cities/<city>.yaml``.
    """

    city_id: str
    state_code: str
    crs_metric: str
    base_year: int
    coastal: bool
    cbd_lat: float
    cbd_lon: float
    city_class: str | None = None
    calibration_level: str | None = None
    boundary_source: str | None = None
    mode: str = "development"
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: Any, *, mode: str = "development") -> "CityConfig":
        city = params.city_config
        cbd = city.get("cbd_point") or {}
        missing = [k for k in ("state_code", "base_year") if not city.get(k)]
        if missing:
            raise IngestConfigError(
                f"city config {city.get('city_id')!r} is missing {', '.join(missing)} "
                "(spec Section 4.8)"
            )
        return cls(
            city_id=str(city.get("city_id", params.city_id)),
            state_code=str(city["state_code"]),
            crs_metric=geo.city_metric_crs(params),
            base_year=int(city["base_year"]),
            coastal=bool(city.get("coastal", False)),
            cbd_lat=float(cbd["lat"]),
            cbd_lon=float(cbd["lon"]),
            city_class=city.get("city_class"),
            calibration_level=city.get("calibration_level"),
            boundary_source=city.get("boundary_source"),
            mode=mode,
            raw=city,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """A city-config key the spec asks for but which may be absent."""
        return self.raw.get(key, default)

    def work_dir(self, root: str | Path | None = None) -> Path:
        """Scratch directory for derived rasters, keyed by city so runs never collide."""
        base = Path(root) if root is not None else Path(cfg("reader.cache_root"))
        out = base / "derived" / self.city_id
        out.mkdir(parents=True, exist_ok=True)
        return out


# --------------------------------------------------------------------------------------
# (a) the injectable reader interface
# --------------------------------------------------------------------------------------


@runtime_checkable
class SourceReader(Protocol):
    """The fetch/read half of every ingester (spec Section 6, ``fetch``).

    A *key* is a logical source name — ``"dem"``, ``"worldcover"``, ``"buildings/2023"`` —
    not a URL and not a path. The reader maps it to bytes on the local filesystem. Every
    ingester holds one of these and never opens a URL itself, which is what makes the
    transform half testable offline.
    """

    def exists(self, key: str) -> bool:
        """True when this reader can supply ``key``."""

    def path(self, key: str, *, force: bool = False) -> Path:
        """A local path for ``key``, materialising it if necessary.

        ``force=False`` must reuse anything already materialised (Section 6 ACCEPTANCE:
        "Re-running an ingester with ``force=False`` uses the cache and produces an
        identical frame").
        """

    def vector(self, key: str, *, force: bool = False) -> gpd.GeoDataFrame:
        """``key`` as a GeoDataFrame."""

    def table(self, key: str, *, force: bool = False) -> pd.DataFrame:
        """``key`` as a plain DataFrame."""

    def retrieved_at(self, key: str) -> datetime:
        """When ``key`` was obtained — goes straight into ``provenance()``."""


class _ReaderBase:
    """Shared bookkeeping: read counts (so cache behaviour is testable) and timestamps."""

    def __init__(self) -> None:
        self.reads: dict[str, int] = {}
        self._retrieved: dict[str, datetime] = {}

    def _record(self, key: str) -> None:
        self.reads[key] = self.reads.get(key, 0) + 1
        self._retrieved.setdefault(key, datetime.now(timezone.utc))

    def retrieved_at(self, key: str) -> datetime:
        return self._retrieved.get(key, datetime.now(timezone.utc))

    @staticmethod
    def _read_vector(path: Path) -> gpd.GeoDataFrame:
        return gpd.read_file(path)

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix in (".json", ".geojson"):
            return pd.json_normalize(json.loads(path.read_text()))
        raise MissingSource(f"cannot read {path} as a table")


class LocalFileReader(_ReaderBase):
    """Reads sources already on disk under a root directory (no network at all).

    Keys are paths relative to ``root``; an optional ``manifest`` maps a logical key onto a
    relative path so an ingester never has to know a filename.
    """

    def __init__(self, root: str | Path, manifest: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.manifest = dict(manifest or {})

    def _resolve(self, key: str) -> Path:
        return self.root / self.manifest.get(key, key)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def path(self, key: str, *, force: bool = False) -> Path:
        target = self._resolve(key)
        if not target.exists():
            raise MissingSource(f"{key!r} not found at {target}")
        self._record(key)
        return target

    def vector(self, key: str, *, force: bool = False) -> gpd.GeoDataFrame:
        return self._read_vector(self.path(key, force=force))

    def table(self, key: str, *, force: bool = False) -> pd.DataFrame:
        return self._read_table(self.path(key, force=force))


class InMemoryReader(_ReaderBase):
    """The test double. Holds raster paths, in-memory vectors and in-memory tables.

    ``tests/fixtures/raster_fixtures.py`` builds one of these from synthetic GeoTIFFs and
    GeoDataFrames, so the whole transform half of every ingester runs offline.
    """

    def __init__(
        self,
        *,
        paths: Mapping[str, str | Path] | None = None,
        vectors: Mapping[str, gpd.GeoDataFrame] | None = None,
        tables: Mapping[str, pd.DataFrame] | None = None,
        retrieved_at: datetime | None = None,
    ) -> None:
        super().__init__()
        self._paths = {k: Path(v) for k, v in (paths or {}).items()}
        self._vectors = dict(vectors or {})
        self._tables = dict(tables or {})
        self._fixed_time = retrieved_at

    # -- mutation helpers used by fixtures ------------------------------------------
    def add_path(self, key: str, value: str | Path) -> "InMemoryReader":
        self._paths[key] = Path(value)
        return self

    def add_vector(self, key: str, value: gpd.GeoDataFrame) -> "InMemoryReader":
        self._vectors[key] = value
        return self

    def add_table(self, key: str, value: pd.DataFrame) -> "InMemoryReader":
        self._tables[key] = value
        return self

    def drop(self, *keys: str) -> "InMemoryReader":
        """Remove keys — how a test simulates an absent layer (e.g. no CRZ)."""
        for key in keys:
            self._paths.pop(key, None)
            self._vectors.pop(key, None)
            self._tables.pop(key, None)
        return self

    # -- SourceReader ----------------------------------------------------------------
    def exists(self, key: str) -> bool:
        return key in self._paths or key in self._vectors or key in self._tables

    def path(self, key: str, *, force: bool = False) -> Path:
        if key in self._paths:
            self._record(key)
            return self._paths[key]
        if key in self._vectors or key in self._tables:
            # An in-memory layer has no file. `fetch` still has to return a Path to satisfy
            # the Section 6 protocol, so it gets a sentinel that identifies the key and
            # that nothing will try to open.
            self._record(key)
            return Path(f"memory://{key}")
        raise MissingSource(f"InMemoryReader holds no path for {key!r}")

    def vector(self, key: str, *, force: bool = False) -> gpd.GeoDataFrame:
        if key in self._vectors:
            self._record(key)
            return self._vectors[key].copy()
        if key in self._paths:
            return self._read_vector(self.path(key, force=force))
        raise MissingSource(f"InMemoryReader holds no vector for {key!r}")

    def table(self, key: str, *, force: bool = False) -> pd.DataFrame:
        if key in self._tables:
            self._record(key)
            return self._tables[key].copy()
        if key in self._paths:
            return self._read_table(self.path(key, force=force))
        raise MissingSource(f"InMemoryReader holds no table for {key!r}")

    def retrieved_at(self, key: str) -> datetime:
        return self._fixed_time or super().retrieved_at(key)


class HttpArchiveReader(_ReaderBase):
    """The real fetch step. **The only networked class in ``ufe/ingest``.**

    Downloads ``urls[key]`` into ``cache_root`` once and reuses it thereafter, so
    ``force=False`` re-runs never re-download (Section 6 ACCEPTANCE). ``min_interval_s``
    honours the rate limit a portal states in its terms of service — the state adapter
    declares it in ``access_terms()`` and the caller passes it here (Sections 6.0, 22.2).

    No test in this repository may construct this class: everything that needs it is
    marked ``@pytest.mark.needs_data``.
    """

    def __init__(
        self,
        urls: Mapping[str, str],
        *,
        cache_root: str | Path | None = None,
        client: Any = None,
        min_interval_s: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        super().__init__()
        self.urls = dict(urls)
        self.cache_root = Path(cache_root or cfg("reader.cache_root"))
        self._client = client
        self.min_interval_s = (
            float(cfg("reader.default_min_seconds_between_requests"))
            if min_interval_s is None
            else float(min_interval_s)
        )
        self.timeout_s = (
            float(cfg("reader.http_timeout_seconds")) if timeout_s is None else float(timeout_s)
        )
        self._last_request: float | None = None

    def exists(self, key: str) -> bool:
        return key in self.urls or self._cache_path(key).exists()

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(self.urls.get(key, key).encode("utf-8")).hexdigest()
        suffix = Path(self.urls.get(key, key)).suffix
        return self.cache_root / f"{digest}{suffix}"

    def _throttle(self) -> None:
        if self._last_request is None:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def path(self, key: str, *, force: bool = False) -> Path:
        target = self._cache_path(key)
        if target.exists() and not force:
            logger.debug("cache hit for %s", key)
            self._retrieved.setdefault(
                key, datetime.fromtimestamp(target.stat().st_mtime, timezone.utc)
            )
            return target
        if key not in self.urls:
            raise MissingSource(f"no URL registered for {key!r}")
        import httpx  # local import: this module must import without network libs loaded

        client = self._client or httpx.Client(timeout=self.timeout_s)
        self._throttle()
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.get(self.urls[key])
        response.raise_for_status()
        target.write_bytes(response.content)
        self._last_request = time.monotonic()
        self._record(key)
        return target

    def vector(self, key: str, *, force: bool = False) -> gpd.GeoDataFrame:
        return self._read_vector(self.path(key, force=force))

    def table(self, key: str, *, force: bool = False) -> pd.DataFrame:
        return self._read_table(self.path(key, force=force))


# --------------------------------------------------------------------------------------
# The Section 6 Ingester protocol, as an ABC with the shared bits filled in
# --------------------------------------------------------------------------------------


class Ingester(ABC):
    """``source_id`` / ``fetch`` / ``parse`` / ``to_cells`` / ``provenance`` (Section 6).

    Subclasses implement ``keys``, ``parse`` and ``to_cells``. ``fetch`` and
    ``provenance`` are shared: ``fetch`` memoises per key so a ``force=False`` re-run reads
    the reader once, and ``provenance`` is assembled from ``config/sources.yaml``.
    """

    #: Key in ``config/sources.yaml``.
    source_id: ClassVar[str]
    #: ``national`` | ``state`` | ``city`` (Section 6.0 data tiers).
    tier: ClassVar[str] = "national"
    #: ``cells`` columns this ingester is responsible for.
    fills: ClassVar[tuple[str, ...]] = ()
    #: Spatial / temporal resolution strings for ``provenance()``.
    spatial_res: ClassVar[str] = ""
    temporal_res: ClassVar[str] = ""
    notes: ClassVar[str] = ""

    def __init__(
        self,
        reader: SourceReader,
        *,
        city: CityConfig | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.reader = reader
        #: The city this instance was constructed for. ``parse`` needs it whenever the
        #: transform is CRS-dependent (terrain slope must be computed in ``crs_metric``),
        #: which the Section 6 ``parse(raw)`` signature cannot express.
        self.city = city
        self.config = config or load_ingest_config()
        # Validates the source registration and the licence string up front (Section 22.2).
        self._provenance_base = provenance_for(self.source_id)
        self._fetched: dict[str, Path] = {}

    # -- (a) fetch -------------------------------------------------------------------

    @abstractmethod
    def keys(self, city: CityConfig) -> tuple[str, ...]:
        """The reader keys this ingester needs for ``city``, primary key first."""

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        """Materialise the primary source for ``city`` and return its local path.

        Memoised: a second call with ``force=False`` returns the same path without
        touching the reader again.
        """
        key = self.keys(city)[0]
        if not force and key in self._fetched:
            return self._fetched[key]
        path = self.reader.path(key, force=force)
        self._fetched[key] = path
        return path

    # -- (b) pure transform ----------------------------------------------------------

    @abstractmethod
    def parse(self, raw: Path) -> pd.DataFrame:
        """Turn the raw artefact into a tidy frame. Pure apart from reading ``raw``."""

    @abstractmethod
    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        """Turn parsed source data into ``cells`` columns. Pure.

        Returns a NEW frame keyed by ``h3`` carrying this ingester's value columns plus
        their ``__imputed`` / ``__impute_method`` companions.
        """

    # -- provenance ------------------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        """``{source_id, url, retrieved_at, licence, spatial_res, temporal_res, notes}``."""
        keys = tuple(self._fetched) or ()
        retrieved = (
            self.reader.retrieved_at(keys[0]) if keys else datetime.now(timezone.utc)
        )
        record = dict(self._provenance_base)
        record.update(
            {
                "source_id": self.source_id,
                "retrieved_at": retrieved.isoformat(),
                "spatial_res": self.spatial_res,
                "temporal_res": self.temporal_res,
                "notes": self.notes or record.get("notes", ""),
            }
        )
        return record


# --------------------------------------------------------------------------------------
# Provenance / licence resolution (Section 22.2)
# --------------------------------------------------------------------------------------


def provenance_for(source_id: str) -> dict[str, Any]:
    """The licence block for ``source_id``, validated against both registries.

    Raises :class:`ufe.errors.LicenceViolation` when the source is not declared in
    ``config/sources.yaml`` or when its licence string is not recognised by
    ``config/data_sources_licences.yaml`` — the same authority ``ufe licences audit --data``
    uses. The ``licence`` field is mandatory (Section 6) and feeds ``ATTRIBUTIONS.md``.
    """
    sources = _load_yaml(str(_SOURCES_PATH)).get("sources") or {}
    if source_id not in sources:
        raise LicenceViolation(
            f"ingester source_id {source_id!r} is not declared in config/sources.yaml; "
            "every source must carry a licence (spec Section 6 / Section 22.2)"
        )
    entry = dict(sources[source_id])
    licence = entry.get("licence")
    known = _load_yaml(str(_DATA_LICENCES_PATH)).get("sources") or {}
    recognised = {
        str(spec.get("licence", "")).lower() for spec in known.values() if spec
    }
    if not licence or str(licence).lower() not in recognised:
        raise LicenceViolation(
            f"source {source_id!r} declares licence {licence!r}, which is not in "
            "config/data_sources_licences.yaml (spec Section 22.2)"
        )
    return {
        "source_id": source_id,
        "url": entry.get("url", ""),
        "licence": licence,
        "obligation": entry.get("obligation", ""),
        "commercial_use": entry.get("commercial_use"),
        "attribution": entry.get("attribution"),
        "tier": entry.get("tier", ""),
        "notes": entry.get("notes", ""),
    }


# --------------------------------------------------------------------------------------
# Geometry helpers — CRS discipline lives in ufe/geo.py, this just applies it
# --------------------------------------------------------------------------------------


def cells_gdf(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """A GeoDataFrame in EPSG:4326 from a ``cells`` frame's WKB ``geometry`` column.

    Accepts a frame that is already a GeoDataFrame (returned unchanged apart from a CRS
    assertion) so callers can pass either.
    """
    if isinstance(cells, gpd.GeoDataFrame) and cells.geometry.name in cells:
        out = cells.copy()
        if out.crs is None:
            out = out.set_crs(geo.GEOGRAPHIC_CRS)
        return out
    if "geometry" not in cells.columns:
        raise MissingSource("cells frame has no 'geometry' column")
    geometry = shapely.from_wkb(cells["geometry"].to_numpy())
    attrs = cells.drop(columns=["geometry"])
    return gpd.GeoDataFrame(attrs, geometry=geometry, crs=geo.GEOGRAPHIC_CRS)


def raster_crs(path: str | Path) -> str:
    """The CRS of a raster, as a string."""
    import rasterio

    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise MissingSource(f"raster {path} declares no CRS")
        return str(ds.crs)


def zonal(
    raster_path: str | Path,
    cells: pd.DataFrame,
    ops: Sequence[str],
    *,
    weights: str | Path | None = None,
) -> pd.DataFrame:
    """Exact-area-weighted zonal statistics per cell, via **exactextract**.

    Spec Section 2.1b mandates ``exactextract`` and forbids ``rasterstats``; Section 6.1
    explains why (a res-9 hexagon is roughly 3x a 30 m DEM pixel, so centroid or
    approximate-coverage sampling introduces real error at cell edges).

    Cell polygons are reprojected from EPSG:4326 into the raster's own CRS before
    extraction, so no area weighting is ever done in degrees.
    """
    from exactextract import exact_extract

    gdf = cells_gdf(cells)[[_H3, "geometry"]]
    target = raster_crs(raster_path)
    if geo.is_geographic(target):
        projected = gdf
    else:
        projected = geo.to_metric(gdf, target)
    result = exact_extract(
        str(raster_path),
        projected,
        list(ops),
        output="pandas",
        include_cols=[_H3],
    )
    out = pd.DataFrame(result)
    out[_H3] = out[_H3].astype(str)
    return out


# --------------------------------------------------------------------------------------
# Imputation flagging
# --------------------------------------------------------------------------------------


def mark_imputed(
    df: pd.DataFrame, column: str, mask: Any, method: str
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``column`` flagged imputed where ``mask`` is True.

    Creates the two companion columns if they do not exist yet. Flags accumulate: marking
    twice with different methods keeps the first non-empty method for the rows already
    flagged, so the earliest (most specific) reason survives.
    """
    out = df.copy()
    flag, method_col = column + IMPUTED_SUFFIX, column + METHOD_SUFFIX
    if flag not in out.columns:
        out[flag] = False
    if method_col not in out.columns:
        out[method_col] = ""
    mask_arr = np.asarray(mask, dtype=bool)
    fresh = mask_arr & ~out[flag].to_numpy(dtype=bool)
    out.loc[fresh, method_col] = method
    out[flag] = out[flag].to_numpy(dtype=bool) | mask_arr
    return out


def value_columns(df: pd.DataFrame) -> list[str]:
    """The value columns of an ingest frame — companions and ``h3`` excluded."""
    return [
        c
        for c in df.columns
        if c != _H3 and not c.endswith(IMPUTED_SUFFIX) and not c.endswith(METHOD_SUFFIX)
    ]


def flag_columns(df: pd.DataFrame) -> list[str]:
    """The value columns that carry an imputation flag."""
    return [c for c in value_columns(df) if c + IMPUTED_SUFFIX in df.columns]


def imputation_long(
    df: pd.DataFrame, *, source_id: str, run_id: str = ""
) -> pd.DataFrame:
    """Melt the ``__imputed`` companions into the tidy ``cell_imputation`` ledger."""
    rows: list[pd.DataFrame] = []
    for column in flag_columns(df):
        rows.append(
            pd.DataFrame(
                {
                    "h3": df[_H3].to_numpy(),
                    "column": column,
                    "imputed": df[column + IMPUTED_SUFFIX].to_numpy(dtype=bool),
                    "method": df.get(
                        column + METHOD_SUFFIX, pd.Series([""] * len(df))
                    ).to_numpy(),
                    "source_id": source_id,
                    "run_id": run_id,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=list(CELL_IMPUTATION_COLUMNS))
    return pd.concat(rows, ignore_index=True)[list(CELL_IMPUTATION_COLUMNS)]


def data_conf(
    imputation: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    missing_capabilities: Iterable[str] = (),
    config: Mapping[str, Any] | None = None,
) -> pd.Series:
    """``cells.data_conf`` from the imputation ledger (Sections 3.1, 6.0).

    ``data_conf = clip(base - sum(weight_c over imputed columns c)
    - penalty * n_missing_capabilities, floor, 1)``. Section 6.0: "A missing capability
    lowers ``data_conf``; it never silently imputes" — hence the second term, driven by the
    state adapter's ``capabilities()``.
    """
    conf_cfg = cfg("data_conf", config)
    base = float(conf_cfg["base"])
    floor = float(conf_cfg["floor"])
    default_weight = float(conf_cfg["default_column_weight"])
    weights = {k: float(v) for k, v in (conf_cfg.get("column_weights") or {}).items()}
    penalty = float(conf_cfg["missing_capability_penalty"]) * len(list(missing_capabilities))

    index = pd.Index(cells[_H3].astype(str), name=_H3)
    deduction = pd.Series(0.0, index=index)
    if len(imputation):
        flagged = imputation[imputation["imputed"].astype(bool)]
        if len(flagged):
            per_cell = (
                flagged.assign(
                    weight=flagged["column"].map(lambda c: weights.get(c, default_weight))
                )
                .groupby("h3")["weight"]
                .sum()
            )
            deduction = deduction.add(per_cell.reindex(index).fillna(0.0), fill_value=0.0)
    values = (base - penalty - deduction).clip(lower=floor, upper=1)
    return pd.Series(values.to_numpy(dtype=float), index=cells.index, name="data_conf")


def merge_ingested(
    cells: pd.DataFrame, *frames: pd.DataFrame, drop_flags: bool = True
) -> pd.DataFrame:
    """Left-join ingest output onto a ``cells`` frame, keeping row count and order.

    With ``drop_flags=True`` the companion columns are stripped, which is what a caller
    writing to the ``strict=True`` ``cells`` schema needs; the flags are persisted
    separately by :func:`write_cell_imputation`.
    """
    out = cells.copy()
    for frame in frames:
        if frame is None or not len(frame):
            continue
        payload = frame.drop(columns=flag_companions(frame)) if drop_flags else frame.copy()
        cols = [c for c in payload.columns if c != _H3]
        out = out.drop(columns=[c for c in cols if c in out.columns]).merge(
            payload, on=_H3, how="left", validate="one_to_one"
        )
    return out


def flag_companions(df: pd.DataFrame) -> list[str]:
    """The companion columns in ``df`` (both suffixes)."""
    return [
        c for c in df.columns if c.endswith(IMPUTED_SUFFIX) or c.endswith(METHOD_SUFFIX)
    ]


# --------------------------------------------------------------------------------------
# ingest_runs ledger
# --------------------------------------------------------------------------------------


def ingest_run(
    provenance: Mapping[str, Any],
    *,
    city_id: str,
    tier: str,
    columns: Sequence[str],
    rows: int,
    params_hash: str = "",
    content_hash: str = "",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """One ``ingest_runs`` row (Section 6).

    ``run_id`` is a content-addressed digest of the run's identity, so re-ingesting the
    same source for the same city with the same provenance produces the same id and the
    ledger stays idempotent.
    """
    created = created_at or datetime.now(timezone.utc)
    identity = json.dumps(
        {
            "source_id": provenance["source_id"],
            "city_id": city_id,
            "retrieved_at": provenance.get("retrieved_at"),
            "columns": sorted(columns),
            "content_hash": content_hash,
        },
        sort_keys=True,
    )
    return {
        "run_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "source_id": provenance["source_id"],
        "city_id": city_id,
        "tier": tier,
        "columns": list(columns),
        "rows": int(rows),
        "url": provenance.get("url", ""),
        "retrieved_at": provenance.get("retrieved_at", ""),
        "licence": provenance["licence"],
        "spatial_res": provenance.get("spatial_res", ""),
        "temporal_res": provenance.get("temporal_res", ""),
        "notes": provenance.get("notes", ""),
        "content_hash": content_hash,
        "params_hash": params_hash,
        "created_at": created.isoformat(),
    }


#: ``source_id`` / ``licence`` for a column that is *computed* from other ingest runs rather
#: than read from a source. ``data_conf`` is the only one today. Section 6 requires every
#: populated cell attribute to have an ``ingest_runs`` row, and a derived column's honest
#: provenance is the set of runs it was computed from, which ``notes`` records.
#:
#: NOTE for the licence owner: ``ufe licences audit --data`` cross-checks
#: ``config/sources.yaml`` against ``config/data_sources_licences.yaml`` and does not read
#: the ``ingest_runs`` table, so this string is not audited today. If the audit is extended
#: to the ledger, ``Derived`` needs an entry (or an exemption) there.
DERIVED_SOURCE_ID = "derived"
DERIVED_LICENCE = "Derived"


def derived_run(
    columns: Sequence[str],
    *,
    city_id: str,
    from_run_ids: Sequence[str],
    rows: int,
    params_hash: str = "",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """An ``ingest_runs`` row for a column computed from other runs (e.g. ``data_conf``)."""
    provenance = {
        "source_id": DERIVED_SOURCE_ID,
        "url": "",
        "licence": DERIVED_LICENCE,
        "retrieved_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "spatial_res": "per cell",
        "temporal_res": "per ingest run",
        "notes": "computed from ingest runs: " + ", ".join(sorted(from_run_ids)),
    }
    return ingest_run(
        provenance,
        city_id=city_id,
        tier="derived",
        columns=columns,
        rows=rows,
        params_hash=params_hash,
        created_at=created_at,
    )


def _ensure_table(con: Any, name: str, ddl: str) -> None:
    con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" ({ddl})')


def write_ingest_runs(con: Any, runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Append ``runs`` to the ``ingest_runs`` table, creating it if necessary.

    Written with plain SQL rather than :func:`ufe.store.db.write_table` because
    ``ufe/store/schemas.py`` declares no ``ingest_runs`` schema and that file belongs to
    another module — the addition is reported, not made here.
    """
    frame = pd.DataFrame(list(runs), columns=list(INGEST_RUNS_COLUMNS))
    frame["columns"] = frame["columns"].map(lambda v: list(v) if v is not None else [])
    _ensure_table(
        con,
        INGEST_RUNS_TABLE,
        "run_id VARCHAR, source_id VARCHAR, city_id VARCHAR, tier VARCHAR, "
        '"columns" VARCHAR[], rows BIGINT, url VARCHAR, retrieved_at VARCHAR, '
        "licence VARCHAR, spatial_res VARCHAR, temporal_res VARCHAR, notes VARCHAR, "
        "content_hash VARCHAR, params_hash VARCHAR, created_at VARCHAR",
    )
    if not len(frame):
        return frame
    con.register("_ufe_ingest_runs", frame)
    quoted = ", ".join(f'"{c}"' for c in INGEST_RUNS_COLUMNS)
    con.execute(
        f'DELETE FROM "{INGEST_RUNS_TABLE}" WHERE run_id IN '
        "(SELECT run_id FROM _ufe_ingest_runs)"
    )
    con.execute(
        f'INSERT INTO "{INGEST_RUNS_TABLE}" ({quoted}) SELECT {quoted} FROM _ufe_ingest_runs'
    )
    con.unregister("_ufe_ingest_runs")
    return frame


def read_ingest_runs(con: Any) -> pd.DataFrame:
    _ensure_table(
        con,
        INGEST_RUNS_TABLE,
        "run_id VARCHAR, source_id VARCHAR, city_id VARCHAR, tier VARCHAR, "
        '"columns" VARCHAR[], rows BIGINT, url VARCHAR, retrieved_at VARCHAR, '
        "licence VARCHAR, spatial_res VARCHAR, temporal_res VARCHAR, notes VARCHAR, "
        "content_hash VARCHAR, params_hash VARCHAR, created_at VARCHAR",
    )
    frame = con.execute(f'SELECT * FROM "{INGEST_RUNS_TABLE}"').df()
    if "columns" in frame.columns:
        frame["columns"] = frame["columns"].map(lambda v: list(v) if v is not None else [])
    return frame


def write_cell_imputation(con: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Persist the tidy imputation ledger so downstream layers can tell real from imputed."""
    payload = frame.reindex(columns=list(CELL_IMPUTATION_COLUMNS))
    _ensure_table(
        con,
        CELL_IMPUTATION_TABLE,
        'h3 VARCHAR, "column" VARCHAR, imputed BOOLEAN, method VARCHAR, '
        "source_id VARCHAR, run_id VARCHAR",
    )
    if not len(payload):
        return payload
    con.register("_ufe_cell_imputation", payload)
    quoted = ", ".join(f'"{c}"' for c in CELL_IMPUTATION_COLUMNS)
    con.execute(
        f'INSERT INTO "{CELL_IMPUTATION_TABLE}" ({quoted}) '
        f"SELECT {quoted} FROM _ufe_cell_imputation"
    )
    con.unregister("_ufe_cell_imputation")
    return payload


def assert_every_column_has_run(
    cells: pd.DataFrame,
    runs: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    exempt: Iterable[str] = (),
) -> None:
    """Section 6 ACCEPTANCE: "Every populated cell column has a matching ``ingest_runs`` row."

    A column counts as populated when it holds at least one non-null value. ``exempt``
    covers the columns Module 1 owns (``h3``, geometry, ``area_sqm``, ...) and the derived
    columns later layers write.
    """
    if isinstance(runs, pd.DataFrame):
        covered = {c for row in runs.get("columns", []) for c in (row or [])}
    else:
        covered = {c for row in runs for c in row.get("columns", [])}
    exempt = set(exempt)
    populated = {
        c
        for c in value_columns(cells)
        if c not in exempt and cells[c].notna().any()
    }
    orphans = sorted(populated - covered)
    if orphans:
        raise UFEError(
            "cell attributes with no corresponding ingest_runs row (spec Section 6): "
            + ", ".join(orphans)
        )
