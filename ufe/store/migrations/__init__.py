"""Versioned, idempotent schema migrations for the DuckDB store.

A migration is a small dataclass with an integer ``version``, a human ``name`` and an
``apply(con)`` callable.  :func:`ufe.store.db.migrate` runs every migration whose version is
absent from the ``_ufe_migrations`` ledger, in ascending order, inside one transaction each.
Re-running ``migrate`` is a no-op — that property is under test.

Migrations are Python rather than raw ``.sql`` files for one reason: migration 1 generates its
DDL directly from the pandera schemas in :mod:`ufe.store.schemas`, so the physical tables can
never drift from the validation contract.  Later, additive migrations may of course be plain
SQL strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import duckdb

from ufe.store import schemas as S

__all__ = ["Migration", "MIGRATIONS", "LEDGER_TABLE", "GEOMETRY_METADATA_TABLE", "ddl_for"]

LEDGER_TABLE = "_ufe_migrations"
GEOMETRY_METADATA_TABLE = "_geometry_columns"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[duckdb.DuckDBPyConnection], None]


def ddl_for(table: str) -> str:
    """``CREATE TABLE IF NOT EXISTS`` statement derived from the table's pandera schema."""
    cols = ",\n    ".join(
        f'"{name}" {S.sql_type(table, name)}' for name in S.column_order(table)
    )
    return f'CREATE TABLE IF NOT EXISTS "{table}" (\n    {cols}\n)'


def _create_core_tables(con: duckdb.DuckDBPyConnection) -> None:
    for table in S.SCHEMAS:
        con.execute(ddl_for(table))


def _record_geometry_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Record the encoding and CRS of every geometry column (Section 0.3).

    DuckDB has no native geometry type in the base build, so the CRS cannot live in the
    column type.  It lives here instead, and readers are expected to consult it.
    """
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{GEOMETRY_METADATA_TABLE}" ('
        ' table_name VARCHAR, column_name VARCHAR, encoding VARCHAR, crs VARCHAR,'
        ' PRIMARY KEY (table_name, column_name))'
    )
    for table, cols in S.GEOMETRY_COLUMNS.items():
        for column, (encoding, crs) in cols.items():
            con.execute(
                f'INSERT OR REPLACE INTO "{GEOMETRY_METADATA_TABLE}" VALUES (?, ?, ?, ?)',
                [table, column, encoding, crs],
            )


def _initial(con: duckdb.DuckDBPyConnection) -> None:
    _create_core_tables(con)
    _record_geometry_columns(con)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=1, name="initial_core_tables", apply=_initial),
)
