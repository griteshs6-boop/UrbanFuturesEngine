"""DuckDB store: connect / migrate / write_table / read_table, plus snapshots (Section 3.8).

Every write is validated against the table's pandera schema in :mod:`ufe.store.schemas`.
An invalid frame **raises** :class:`ufe.errors.SchemaValidationError`; nothing is warned and
nothing partial is written (Section 3, preamble).

Geometry and CRS
----------------
All geometry is EPSG:4326 on disk (Section 0.3).  Two encodings are used:

* ``cells.geometry`` — **WKB** in a DuckDB ``BLOB``.  Compact: a city is 150k–300k hexagons.
* ``projects.geom``  — **WKT** in a DuckDB ``VARCHAR``, because Section 3.3 specifies WKT.

DuckDB's base build has no geometry type, so the CRS cannot ride along in the column type.
It is recorded instead in the ``_geometry_columns`` metadata table
(``table_name, column_name, encoding, crs``) created by migration 1, and is readable with
:func:`geometry_metadata`.  Anything that projects geometry must read the CRS from there and
reproject into the city's ``crs_metric`` before doing metric maths — never compute distance
in degrees.

Numeric policy (Section 0.1 rule 3): this module contains no model parameters.  The only
named numbers are :data:`SHORT_HASH_LEN`, a string-formatting width for the snapshot
directory name mandated by Section 3.8's ``{YYYY-MM-DD}_{shorthash}`` pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import pandera.errors as pa_errors

from ufe.errors import SchemaValidationError
from ufe.store import schemas as S
from ufe.store.migrations import LEDGER_TABLE, MIGRATIONS, GEOMETRY_METADATA_TABLE

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_SNAPSHOT_ROOT",
    "DEFAULT_PARAMS_DIR",
    "SHORT_HASH_LEN",
    "SnapshotRef",
    "connect",
    "migrate",
    "write_table",
    "read_table",
    "table_exists",
    "geometry_metadata",
    "content_hash",
    "file_hash",
    "snapshot_hash",
    "write_snapshot",
    "read_snapshot_table",
    "read_manifest",
]

DEFAULT_DB_PATH = "data/ufe.duckdb"
DEFAULT_SNAPSHOT_ROOT = Path("data/snapshots")
DEFAULT_PARAMS_DIR = Path("config/params")

#: Width of the ``{shorthash}`` component of a snapshot directory name (Section 3.8).
#: A presentation width, not a model parameter.
SHORT_HASH_LEN: int = 8

#: File-read buffer for hashing.  An I/O size, not a model parameter.
_HASH_CHUNK_BYTES: int = 1 << 20

#: ``json.dumps`` indentation for MANIFEST.json.  Formatting, not a model parameter.
_JSON_INDENT: int = 2

_MANIFEST_NAME = "MANIFEST.json"
_PARAMS_SUBDIR = "params"
_SNAPSHOT_TABLES: tuple[str, ...] = ("cells", "projects", "announcers")
_DATE_DIR_FORMAT = "%Y-%m-%d"

_WRITE_MODES = ("append", "replace", "upsert")
_DATETIME_DTYPE = "datetime64[ns]"


# --------------------------------------------------------------------------------------
# Connection & migration
# --------------------------------------------------------------------------------------


def connect(
    path: str | Path = DEFAULT_DB_PATH, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open (and, for on-disk paths, create the parent directory of) the store.

    ``path=":memory:"`` gives an ephemeral database, which is what the tests use.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(path), read_only=read_only)


def _applied_versions(con: duckdb.DuckDBPyConnection) -> set[int]:
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{LEDGER_TABLE}" ('
        " version BIGINT PRIMARY KEY, name VARCHAR, applied_at TIMESTAMP)"
    )
    rows = con.execute(f'SELECT version FROM "{LEDGER_TABLE}"').fetchall()
    return {int(r[0]) for r in rows}


def migrate(con: duckdb.DuckDBPyConnection) -> None:
    """Bring the store up to the latest schema version.

    Idempotent: migrations already recorded in ``_ufe_migrations`` are skipped, and every
    migration body is itself written with ``IF NOT EXISTS`` / ``INSERT OR REPLACE`` so a
    half-applied ledger cannot wedge the database.
    """
    applied = _applied_versions(con)
    for migration in sorted(MIGRATIONS, key=lambda m: m.version):
        if migration.version in applied:
            continue
        logger.info("applying migration %s (%s)", migration.version, migration.name)
        con.execute("BEGIN TRANSACTION")
        try:
            migration.apply(con)
            con.execute(
                f'INSERT INTO "{LEDGER_TABLE}" VALUES (?, ?, ?)',
                [migration.version, migration.name, datetime.now(timezone.utc)],
            )
        except Exception:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchall()
    return bool(rows)


def geometry_metadata(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The ``_geometry_columns`` table: encoding and CRS per geometry column."""
    return con.execute(
        f'SELECT * FROM "{GEOMETRY_METADATA_TABLE}" ORDER BY table_name, column_name'
    ).df()


# --------------------------------------------------------------------------------------
# Validation & writing
# --------------------------------------------------------------------------------------


def _normalise_lists(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Coerce numpy arrays / tuples in list columns to plain python lists."""
    out = df
    for col in S.LIST_COLUMNS.get(table, ()):
        if col not in out.columns:
            continue
        if out is df:
            out = df.copy()
        out[col] = out[col].map(
            lambda v: None if v is None else [x.item() if hasattr(x, "item") else x for x in v]
        )
    return out


def validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against the pandera schema for ``table``.

    Returns the coerced frame.  Raises :class:`ufe.errors.SchemaValidationError` — never
    warns — on any failure, including unknown columns (the schemas are ``strict=True``).
    """
    if table not in S.SCHEMAS:
        raise SchemaValidationError(f"unknown table {table!r}; known: {sorted(S.SCHEMAS)}")
    schema = S.SCHEMAS[table]
    candidate = _normalise_lists(df, table)
    try:
        return schema.validate(candidate, lazy=True)
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as exc:
        raise SchemaValidationError(f"{table}: {exc}") from exc


def _prepare_for_insert(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the physical column order, materialising absent optional columns."""
    out = df.copy()
    for column in S.column_order(table):
        if column not in out.columns:
            out[column] = None
    return out[S.column_order(table)]


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    df: pd.DataFrame,
    mode: str = "append",
) -> None:
    """Validate ``df`` against its pandera schema and insert it into ``name``.

    ``mode``:
      * ``append``  (default) — plain insert.
      * ``replace`` — truncate the table first.
      * ``upsert``  — delete rows whose primary key appears in ``df``, then insert.

    Raises :class:`ufe.errors.SchemaValidationError` if the frame does not validate.  The
    validation happens before any SQL is issued, so an invalid frame leaves the store
    untouched.
    """
    if mode not in _WRITE_MODES:
        raise ValueError(f"mode must be one of {_WRITE_MODES}, got {mode!r}")
    validated = validate(name, df)
    if not table_exists(con, name):
        raise SchemaValidationError(
            f"table {name!r} does not exist; call ufe.store.db.migrate(con) first"
        )

    payload = _prepare_for_insert(name, validated)
    quoted = ", ".join(f'"{c}"' for c in payload.columns)

    con.execute("BEGIN TRANSACTION")
    try:
        if mode == "replace":
            con.execute(f'DELETE FROM "{name}"')
        elif mode == "upsert":
            keys = S.PRIMARY_KEYS.get(name, ())
            if not keys:
                raise SchemaValidationError(f"table {name!r} has no primary key; cannot upsert")
            con.register("_ufe_upsert_keys", payload[list(keys)])
            predicate = " AND ".join(f't."{k}" = s."{k}"' for k in keys)
            con.execute(
                f'DELETE FROM "{name}" t WHERE EXISTS '
                f"(SELECT 1 FROM _ufe_upsert_keys s WHERE {predicate})"
            )
            con.unregister("_ufe_upsert_keys")
        con.register("_ufe_write_src", payload)
        con.execute(f'INSERT INTO "{name}" ({quoted}) SELECT {quoted} FROM _ufe_write_src')
        con.unregister("_ufe_write_src")
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    logger.debug("wrote %d rows to %s (mode=%s)", len(payload), name, mode)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def _is_null_scalar(v: Any) -> bool:
    """True for a missing value, whatever null sentinel DuckDB/pandas hands back.

    A never-written *optional* list column (e.g. the ``shock_*`` arrays of Section 9)
    reads back as ``pandas.NA``, not ``None``, so a bare ``v is None`` guard is not
    enough.  Sequences are never null, and ``pd.isna`` on a sequence returns an array
    rather than a bool, so they are excluded before it is called.
    """
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return False
    return bool(pd.isna(v))


def _postprocess(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Restore python-native types DuckDB widens on the way out.

    * ``LIST`` columns come back as ``numpy.ndarray`` -> plain ``list``.
    * ``BLOB`` columns come back as ``bytearray`` -> immutable ``bytes``, so a WKB value
      written equals the WKB value read.
    """
    out = df
    for col in S.LIST_COLUMNS.get(table, ()):
        if col in out.columns:
            if out is df:
                out = df.copy()
            out[col] = out[col].map(
                lambda v: None
                if _is_null_scalar(v)
                else [x.item() if hasattr(x, "item") else x for x in v]
            )
    for col, (encoding, _crs) in S.GEOMETRY_COLUMNS.get(table, {}).items():
        if encoding == "WKB" and col in out.columns:
            if out is df:
                out = df.copy()
            out[col] = out[col].map(lambda v: None if v is None else bytes(v))
    # DuckDB returns TIMESTAMP/DATE as microsecond-resolution numpy datetimes; the schemas
    # declare nanosecond resolution, so normalise to keep round-trips dtype-stable.
    for col, spec in S.SCHEMAS[table].columns.items():
        if str(spec.dtype) == _DATETIME_DTYPE and col in out.columns:
            if str(out[col].dtype) != _DATETIME_DTYPE:
                if out is df:
                    out = df.copy()
                out[col] = out[col].astype(_DATETIME_DTYPE)
    return out


def read_table(con: duckdb.DuckDBPyConnection, name: str, **filters: Any) -> pd.DataFrame:
    """Read ``name``, optionally filtered by equality on any column.

    Scalar filter values become ``col = ?``; list/tuple/set values become ``col IN (...)``.
    ``None`` becomes ``col IS NULL``.  The returned frame has the schema's column order and
    validates against the schema.
    """
    if name not in S.SCHEMAS:
        raise SchemaValidationError(f"unknown table {name!r}; known: {sorted(S.SCHEMAS)}")
    known = set(S.column_order(name))
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in filters.items():
        if column not in known:
            raise SchemaValidationError(f"{name}: unknown filter column {column!r}")
        if value is None:
            clauses.append(f'"{column}" IS NULL')
        elif isinstance(value, (list, tuple, set, frozenset)):
            values = sorted(value, key=repr)
            if not values:
                clauses.append("FALSE")
                continue
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f'"{column}" IN ({placeholders})')
            params.extend(values)
        else:
            clauses.append(f'"{column}" = ?')
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    frame = con.execute(f'SELECT * FROM "{name}"{where}', params).df()
    return _postprocess(name, frame)


# --------------------------------------------------------------------------------------
# 3.8 Snapshots — content hashing so every number traces to a hash
# --------------------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    """A stable byte encoding for a single dataframe cell."""
    if value is None:
        return b"\x00null"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"\x01" + bytes(value)
    if isinstance(value, str):
        return b"\x02" + value.encode("utf-8")
    if isinstance(value, (list, tuple, np.ndarray)):
        return b"\x03" + b"\x1f".join(_canonical_bytes(v) for v in value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return b"\x04" + pd.Timestamp(value).isoformat().encode("ascii")
    if value is pd.NaT:
        return b"\x00null"
    if isinstance(value, (np.bool_, bool)):
        return b"\x05" + (b"true" if value else b"false")
    if isinstance(value, (int, np.integer)):
        return b"\x06" + repr(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if fv != fv:  # NaN
            return b"\x00null"
        return b"\x07" + float.hex(fv).encode("ascii")
    return b"\x08" + repr(value).encode("utf-8")


def content_hash(df: pd.DataFrame, *, sort_by: Sequence[str] | None = None) -> str:
    """SHA-256 of a dataframe's content, independent of row and column order.

    Columns are hashed in sorted name order; rows in ``sort_by`` order when given (defaults
    to the table's primary key when the frame is a known table).  This is the primitive that
    lets any number in an output be traced back to the snapshot it came from.
    """
    frame = df
    if sort_by:
        frame = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    for column in sorted(frame.columns):
        digest.update(b"\x1e" + str(column).encode("utf-8"))
        for value in frame[column].tolist():
            digest.update(b"\x1d")
            digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_hash(file_hashes: Iterable[str]) -> str:
    """``sha256`` over the sorted concatenation of individual file hashes (Section 3.8)."""
    joined = "".join(sorted(str(h) for h in file_hashes))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SnapshotRef:
    """Handle to an immutable snapshot directory.

    A simulation may only read from a snapshot, never from the live DB (Section 3.8).
    """

    snapshot_id: str
    snapshot_hash: str
    path: Path
    city_id: str
    params_hash: str

    def table_path(self, name: str) -> Path:
        return self.path / f"{name}.parquet"

    @property
    def params_path(self) -> Path:
        return self.path / _PARAMS_SUBDIR

    @property
    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_NAME


def _iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != _MANIFEST_NAME)


def write_snapshot(
    con: duckdb.DuckDBPyConnection,
    *,
    city_id: str,
    created_by: str,
    params_dir: str | Path = DEFAULT_PARAMS_DIR,
    out_root: str | Path = DEFAULT_SNAPSHOT_ROOT,
    created_at: datetime | None = None,
    ingest_run_ids: Sequence[str] = (),
    params_hash: str | None = None,
) -> SnapshotRef:
    """Materialise ``data/snapshots/{YYYY-MM-DD}_{shorthash}/`` and register it.

    Writes ``cells.parquet``, ``projects.parquet``, ``announcers.parquet``, a full copy of the
    ``config/params/`` tree, and ``MANIFEST.json`` carrying source row counts, the hash of
    each file, the creating user and the ingest run ids.  Then inserts a row into the
    ``snapshots`` table.  The directory is renamed to its final ``{shorthash}`` name only
    after every file is written, so a partially written snapshot is never addressable.
    """
    created_at = created_at or datetime.now(timezone.utc)
    out_root = Path(out_root)
    stamp = created_at.strftime(_DATE_DIR_FORMAT)
    staging = out_root / f"{stamp}_staging_{created_at.strftime('%H%M%S%f')}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    row_counts: dict[str, int] = {}
    for table in _SNAPSHOT_TABLES:
        frame = read_table(con, table)
        row_counts[table] = len(frame)
        frame.to_parquet(staging / f"{table}.parquet", index=False)

    params_dir = Path(params_dir)
    params_target = staging / _PARAMS_SUBDIR
    if params_dir.is_dir():
        shutil.copytree(params_dir, params_target)
    else:
        params_target.mkdir()
        logger.warning("params dir %s does not exist; snapshot params/ is empty", params_dir)

    files = _iter_files(staging)
    hashes = {str(p.relative_to(staging)): file_hash(p) for p in files}
    digest = snapshot_hash(hashes.values())
    short = digest[:SHORT_HASH_LEN]
    snapshot_id = f"{stamp}_{short}"

    manifest = {
        "snapshot_id": snapshot_id,
        "snapshot_hash": digest,
        "city_id": city_id,
        "created_at": created_at.isoformat(),
        "created_by": created_by,
        "created_from": "live_db",
        "source_row_counts": row_counts,
        "file_hashes": hashes,
        "ingest_run_ids": list(ingest_run_ids),
        "params_hash": params_hash or "",
        "geometry_crs": S.GEOMETRY_CRS,
        "geometry_encoding": {
            f"{t}.{c}": enc for t, cols in S.GEOMETRY_COLUMNS.items() for c, (enc, _) in cols.items()
        },
        "content_hashes": {
            table: content_hash(
                read_table(con, table), sort_by=S.PRIMARY_KEYS.get(table) or None
            )
            for table in _SNAPSHOT_TABLES
        },
    }
    (staging / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=_JSON_INDENT, sort_keys=True))

    final = out_root / snapshot_id
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)

    row = pd.DataFrame(
        [
            {
                "snapshot_id": snapshot_id,
                "snapshot_hash": digest,
                "city_id": city_id,
                "created_at": pd.Timestamp(created_at).tz_localize(None),
                "created_by": created_by,
                "path": str(final),
                "params_hash": params_hash or "",
                "cells_rows": row_counts["cells"],
                "projects_rows": row_counts["projects"],
                "announcers_rows": row_counts["announcers"],
                "file_hashes": sorted(hashes.values()),
                "ingest_run_ids": list(ingest_run_ids),
            }
        ]
    )
    write_table(con, "snapshots", row, mode="upsert")

    return SnapshotRef(
        snapshot_id=snapshot_id,
        snapshot_hash=digest,
        path=final,
        city_id=city_id,
        params_hash=params_hash or "",
    )


def read_snapshot_table(ref: SnapshotRef | str | Path, name: str) -> pd.DataFrame:
    """Read a table out of a snapshot directory (the only legal read path for a simulation)."""
    root = ref.path if isinstance(ref, SnapshotRef) else Path(ref)
    frame = pd.read_parquet(root / f"{name}.parquet")
    return _postprocess(name, frame)


def read_manifest(ref: SnapshotRef | str | Path) -> Mapping[str, Any]:
    root = ref.path if isinstance(ref, SnapshotRef) else Path(ref)
    return json.loads((root / _MANIFEST_NAME).read_text())
