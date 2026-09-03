"""Tests for the storage layer (spec Section 3).

Section 3 has no explicit ``ACCEPTANCE`` block; its normative statements are in the section
preamble and in 3.6/3.7/3.8.  Each of those becomes an ``@pytest.mark.acceptance`` test here:

* "Every table has a pandera schema in ``ufe/store/schemas.py``"        -> ``test_acc_every_section_3_table_has_a_schema``
* "... and is validated on write"                                        -> ``test_acc_every_table_is_validated_on_write``
* "Writing an invalid frame must raise, not warn"                        -> ``test_acc_invalid_frame_raises_and_does_not_warn``
* 3.6 eight sectors / 3.7 four bands, as enums not free strings          -> ``test_acc_sector_taxonomy``, ``test_acc_income_bands``
* 3.7 "Band boundaries live in behaviour.yaml ... do not hardcode"       -> ``test_acc_income_band_boundaries_come_from_yaml``
* 3.8 snapshot layout and ``sha256`` over sorted file hashes             -> ``test_acc_snapshot_layout``, ``test_acc_snapshot_hash_definition``
* 0.3 geometry is EPSG:4326 on disk, CRS recorded                        -> ``test_acc_geometry_crs_recorded``
"""

from __future__ import annotations

import ast
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely import wkb, wkt

from ufe.errors import SchemaValidationError
from ufe.store import db
from ufe.store import schemas as S
from ufe.store.migrations import MIGRATIONS, ddl_for

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    build_city,
    synthetic_announcers,
    synthetic_cells,
    synthetic_cells_history,
    synthetic_city,
    synthetic_project_history,
    synthetic_projects,
)

SECTION_3_TABLES = (
    "cells",
    "cells_history",
    "projects",
    "announcers",
    "project_history",
    "snapshots",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def con():
    connection = db.connect(":memory:")
    db.migrate(connection)
    yield connection
    connection.close()


@pytest.fixture()
def loaded(con, synthetic_city):  # noqa: F811
    db.write_table(con, "cells", synthetic_city.cells)
    db.write_table(con, "cells_history", synthetic_city.cells_history)
    db.write_table(con, "announcers", synthetic_city.announcers)
    db.write_table(con, "projects", synthetic_city.projects)
    db.write_table(con, "project_history", synthetic_city.project_history)
    return con


def _frames(city):
    return {
        "cells": city.cells,
        "cells_history": city.cells_history,
        "announcers": city.announcers,
        "projects": city.projects,
        "project_history": city.project_history,
    }


# ======================================================================================
# ACCEPTANCE — Section 3
# ======================================================================================


@pytest.mark.acceptance
def test_acc_every_section_3_table_has_a_schema():
    """3 preamble: every table lives in DuckDB and has a pandera schema."""
    assert set(S.SCHEMAS) == set(SECTION_3_TABLES)
    for name in SECTION_3_TABLES:
        assert isinstance(S.SCHEMAS[name], type(S.CELLS))
        assert S.SCHEMAS[name].columns, f"{name} has no columns"
    # The contract also names the individual module-level constants.
    assert (S.CELLS, S.CELLS_HISTORY, S.PROJECTS) == (
        S.SCHEMAS["cells"],
        S.SCHEMAS["cells_history"],
        S.SCHEMAS["projects"],
    )
    assert (S.ANNOUNCERS, S.PROJECT_HISTORY, S.SNAPSHOTS) == (
        S.SCHEMAS["announcers"],
        S.SCHEMAS["project_history"],
        S.SCHEMAS["snapshots"],
    )


@pytest.mark.acceptance
@pytest.mark.parametrize("table", ["cells", "cells_history", "projects", "announcers",
                                   "project_history"])
def test_acc_every_table_is_validated_on_write(con, synthetic_city, table):  # noqa: F811
    """3 preamble: '... and is validated on write'."""
    frame = _frames(synthetic_city)[table]
    db.write_table(con, table, frame)
    assert len(db.read_table(con, table)) == len(frame)

    # Same frame with one required column removed must be rejected by write_table.
    required = [c for c, col in S.SCHEMAS[table].columns.items() if col.required]
    broken = frame.drop(columns=[required[0]])
    with pytest.raises(SchemaValidationError):
        db.write_table(con, table, broken)


@pytest.mark.acceptance
def test_acc_invalid_frame_raises_and_does_not_warn(con, synthetic_city):  # noqa: F811
    """3 preamble: 'Writing an invalid frame must raise, not warn.'"""
    bad = synthetic_city.cells.copy()
    bad.loc[bad.index[0], "builtup_frac"] = len(S.INCOME_BANDS)  # far outside [0, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        with pytest.raises(SchemaValidationError):
            db.write_table(con, "cells", bad)

    # Nothing was written: validation happens before any SQL is issued.
    assert len(db.read_table(con, "cells")) == 0


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.assign(builtup_frac=-d["builtup_frac"] - 1), id="out_of_range"),
        pytest.param(lambda d: d.assign(zone_class="not_a_zone"), id="bad_enum"),
        pytest.param(lambda d: d.assign(crz_class="V"), id="bad_crz"),
        pytest.param(lambda d: pd.concat([d, d.iloc[[0]]]), id="duplicate_primary_key"),
        pytest.param(lambda d: d.assign(surprise_column=0), id="unknown_column"),
        pytest.param(lambda d: d.drop(columns=["h3"]), id="missing_key"),
        pytest.param(
            lambda d: d.assign(hh_by_band=[list(v)[:-1] for v in d["hh_by_band"]]),
            id="wrong_band_count",
        ),
        pytest.param(
            lambda d: d.assign(jobs_by_sector=[list(v)[:-1] for v in d["jobs_by_sector"]]),
            id="wrong_sector_count",
        ),
        pytest.param(lambda d: d.assign(util_water=d["util_water"] + 2), id="non_boolean_flag"),
        pytest.param(lambda d: d.assign(data_conf=d["data_conf"] + 1), id="probability_gt_one"),
        pytest.param(lambda d: d.assign(geometry="not wkb"), id="geometry_not_bytes"),
    ],
)
def test_acc_invalid_cells_variants_raise(con, synthetic_city, mutate):  # noqa: F811
    with pytest.raises(SchemaValidationError):
        db.write_table(con, "cells", mutate(synthetic_city.cells))


@pytest.mark.acceptance
def test_acc_invalid_projects_variants_raise(con, synthetic_city):  # noqa: F811
    projects = synthetic_city.projects
    for mutate in (
        lambda d: d.assign(stage="wishful"),
        lambda d: d.assign(geom_type="multipolygon"),
        lambda d: d.assign(scale_unit="furlongs"),
        lambda d: d.assign(source_urls=[[] for _ in range(len(d))]),  # 3.3: non-empty
        lambda d: d.assign(extracted_by="magic"),  # must be human / ai:{version}
        lambda d: d.assign(funding_source="vibes"),
    ):
        with pytest.raises(SchemaValidationError):
            db.write_table(con, "projects", mutate(projects))


@pytest.mark.acceptance
def test_acc_sector_taxonomy():
    """3.6: eight sectors, in order, exposed as a validated enum rather than free strings."""
    assert [s.name for s in S.Sector] == [
        "agri",
        "manuf_heavy",
        "manuf_light",
        "logistics",
        "it_office",
        "retail_svc",
        "public_edu",
        "construction",
    ]
    assert [s.value for s in S.Sector] == list(range(len(S.SECTORS)))
    assert S.SECTORS == tuple(s.name for s in S.Sector)
    # jobs_by_sector length is checked against the taxonomy, not a literal.
    assert S.CELLS.columns["jobs_by_sector"].metadata["list_len"] == len(S.SECTORS)


@pytest.mark.acceptance
def test_acc_income_bands():
    """3.7: four bands, in order, exposed as a validated enum."""
    assert [b.name for b in S.IncomeBand] == ["low", "mid", "upper_mid", "high"]
    assert [b.value for b in S.IncomeBand] == list(range(len(S.INCOME_BANDS)))
    assert S.CELLS.columns["hh_by_band"].metadata["list_len"] == len(S.INCOME_BANDS)


@pytest.mark.acceptance
def test_acc_income_band_boundaries_come_from_yaml():
    """3.7: 'Band boundaries live in behaviour.yaml ... Do not hardcode.'"""
    assert S.INCOME_BAND_BOUNDARY_PATH == "behaviour.income_bands.boundaries_inr_mo"
    expected = len(S.INCOME_BANDS) - 1

    class _FakeParams:
        def __init__(self, value):
            self.value = value

        def get(self, path):
            assert path == S.INCOME_BAND_BOUNDARY_PATH
            return self.value

    good = [float(i) for i in range(expected)]
    assert S.income_band_boundaries(_FakeParams(good)) == good
    with pytest.raises(ValueError):
        S.income_band_boundaries(_FakeParams(good[:-1]))

    # Section 4.1 leaf form -- what the landed behaviour.yaml actually encodes.
    leaves = [{"value": b, "conf": "E", "scope": "local"} for b in good]
    assert S.income_band_boundaries(_FakeParams(leaves)) == good
    with pytest.raises(ValueError):
        S.income_band_boundaries(_FakeParams(leaves[:-1]))
    # A mapping with no `value` key is a malformed leaf, not a silently-skipped one.
    with pytest.raises(ValueError):
        S.income_band_boundaries(_FakeParams([{"conf": "E"}] * expected))


def test_income_band_boundaries_read_the_real_behaviour_yaml():
    """The shared helper must read the on-disk config, not just the spec's scalar form."""
    from ufe.layers import l5_allocation as L5
    from ufe.params import load_params

    params = load_params("vizag")
    boundaries = S.income_band_boundaries(params)

    assert len(boundaries) == len(S.INCOME_BANDS) - 1
    assert boundaries == sorted(boundaries)
    assert all(isinstance(b, float) for b in boundaries)
    # L5 no longer keeps a local copy; its wrapper only adds the indexing guard.
    assert L5.income_band_boundaries(params) == boundaries


@pytest.mark.acceptance
def test_acc_geometry_crs_recorded(con):
    """0.3: geometry is EPSG:4326 on disk, and the CRS is recorded alongside it."""
    meta = db.geometry_metadata(con)
    recorded = {(r.table_name, r.column_name): (r.encoding, r.crs) for r in meta.itertuples()}
    assert recorded == {
        ("cells", "geometry"): ("WKB", "EPSG:4326"),
        ("projects", "geom"): ("WKT", "EPSG:4326"),
    }
    assert set(recorded) == set(S.GEOMETRY_ENCODING)
    assert all(crs == S.GEOMETRY_CRS for _, crs in recorded.values())
    assert S.sql_type("cells", "geometry") == "BLOB"
    assert S.sql_type("projects", "geom") == "VARCHAR"


@pytest.mark.acceptance
def test_acc_snapshot_hash_definition(tmp_path):
    """3.8: 'snapshot_hash = sha256 over the sorted concatenation of individual file hashes'."""
    a, b, c = (tmp_path / n for n in ("a.bin", "b.bin", "c.bin"))
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    c.write_bytes(b"gamma")
    hashes = [db.file_hash(p) for p in (a, b, c)]
    assert hashes[0] == hashlib.sha256(b"alpha").hexdigest()

    expected = hashlib.sha256("".join(sorted(hashes)).encode("ascii")).hexdigest()
    assert db.snapshot_hash(hashes) == expected
    # Order-independent, by construction.
    assert db.snapshot_hash(reversed(hashes)) == expected
    # Any change to any file changes the snapshot hash.
    c.write_bytes(b"gamma!")
    assert db.snapshot_hash([db.file_hash(p) for p in (a, b, c)]) != expected


@pytest.mark.acceptance
def test_acc_snapshot_layout(loaded, tmp_path, synthetic_city):  # noqa: F811
    """3.8: the on-disk snapshot layout, MANIFEST contents, and the snapshots table row."""
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "behaviour.yaml").write_text("income_bands:\n  boundaries_inr_mo: [1, 2, 3]\n")
    root = tmp_path / "snapshots"

    ref = db.write_snapshot(
        loaded,
        city_id="vizag",
        created_by="tester",
        params_dir=params_dir,
        out_root=root,
        created_at=datetime(2026, 9, 3),
        ingest_run_ids=["ingest-a", "ingest-b"],
        params_hash="deadbeef",
    )

    assert ref.path.name == f"2026-09-03_{ref.snapshot_hash[: db.SHORT_HASH_LEN]}"
    assert ref.snapshot_id == ref.path.name
    for table in ("cells", "projects", "announcers"):
        assert ref.table_path(table).is_file()
    assert (ref.params_path / "behaviour.yaml").is_file()
    assert ref.manifest_path.is_file()
    # No staging directory left behind.
    assert [p.name for p in root.iterdir()] == [ref.snapshot_id]

    manifest = db.read_manifest(ref)
    assert manifest["source_row_counts"]["cells"] == len(synthetic_city.cells)
    assert manifest["created_by"] == "tester"
    assert manifest["ingest_run_ids"] == ["ingest-a", "ingest-b"]
    assert manifest["geometry_crs"] == "EPSG:4326"
    assert set(manifest["file_hashes"]) >= {
        "cells.parquet",
        "projects.parquet",
        "announcers.parquet",
        "params/behaviour.yaml",
    }
    assert manifest["snapshot_hash"] == db.snapshot_hash(manifest["file_hashes"].values())

    row = db.read_table(loaded, "snapshots").iloc[0]
    assert row["snapshot_hash"] == ref.snapshot_hash
    assert row["cells_rows"] == len(synthetic_city.cells)
    assert row["params_hash"] == "deadbeef"
    assert sorted(row["file_hashes"]) == sorted(manifest["file_hashes"].values())


@pytest.mark.acceptance
def test_acc_snapshot_is_readable_without_the_live_db(loaded, tmp_path, synthetic_city):  # noqa: F811
    """3.8: 'A simulation may only read from a snapshot, never from the live DB.'"""
    ref = db.write_snapshot(
        loaded,
        city_id="vizag",
        created_by="tester",
        params_dir=tmp_path / "absent",
        out_root=tmp_path / "snapshots",
    )
    loaded.close()  # the live DB is gone; the snapshot must still be readable
    cells = db.read_snapshot_table(ref, "cells")
    assert len(cells) == len(synthetic_city.cells)
    assert list(cells.columns) == S.column_order("cells")
    assert isinstance(cells["geometry"].iloc[0], bytes)


# ======================================================================================
# Round-trip, migrations, and the synthetic fixture
# ======================================================================================


def test_round_trip_preserves_dtypes_and_values(con, synthetic_city):  # noqa: F811
    for table, frame in _frames(synthetic_city).items():
        db.write_table(con, table, frame)
        back = db.read_table(con, table)
        assert list(back.columns) == S.column_order(table)
        key = S.PRIMARY_KEYS.get(table) or list(frame.columns[: len(S.INCOME_BANDS)])
        left = frame.sort_values(list(key)).reset_index(drop=True)
        right = back.sort_values(list(key)).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=True, check_like=False)


def test_round_trip_preserves_geometry(con, synthetic_city):  # noqa: F811
    db.write_table(con, "cells", synthetic_city.cells)
    db.write_table(con, "announcers", synthetic_city.announcers)
    db.write_table(con, "projects", synthetic_city.projects)

    cells = db.read_table(con, "cells").set_index("h3")
    source = synthetic_city.cells.set_index("h3")
    for h3_index in list(source.index)[: len(S.SECTORS)]:
        stored = cells.loc[h3_index, "geometry"]
        assert isinstance(stored, bytes)
        geom = wkb.loads(stored)
        assert geom.equals(wkb.loads(source.loc[h3_index, "geometry"]))
        # Centroid must agree with the stored lat/lon, i.e. the WKB is (lon, lat) EPSG:4326.
        assert geom.centroid.x == pytest.approx(source.loc[h3_index, "lon"], abs=1e-6)
        assert geom.centroid.y == pytest.approx(source.loc[h3_index, "lat"], abs=1e-6)

    projects = db.read_table(con, "projects")
    for geom_text, geom_type in zip(projects["geom"], projects["geom_type"]):
        parsed = wkt.loads(geom_text)
        assert parsed.geom_type.lower() == geom_type
        assert parsed.is_valid


def test_round_trip_preserves_list_columns(con, synthetic_city):  # noqa: F811
    db.write_table(con, "cells", synthetic_city.cells)
    back = db.read_table(con, "cells").set_index("h3")
    source = synthetic_city.cells.set_index("h3")
    for h3_index in list(source.index)[: len(S.SECTORS)]:
        for column, length in (
            ("hh_by_band", len(S.INCOME_BANDS)),
            ("jobs_by_sector", len(S.SECTORS)),
        ):
            stored = back.loc[h3_index, column]
            assert isinstance(stored, list) and len(stored) == length
            assert stored == pytest.approx(source.loc[h3_index, column])


def test_round_trip_preserves_nulls(con, synthetic_city):  # noqa: F811
    db.write_table(con, "cells", synthetic_city.cells)
    back = db.read_table(con, "cells").set_index("h3")
    source = synthetic_city.cells.set_index("h3")
    for column in ("price_res_inr_sqft", "rent_res_inr_sqft_mo"):
        assert source[column].isna().any(), f"fixture should exercise nulls in {column}"
        assert back[column].isna().sum() == source[column].isna().sum()


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "ufe.duckdb"
    con = db.connect(path)
    db.migrate(con)
    first = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns"
        " ORDER BY table_name, column_name"
    ).fetchall()
    ledger_first = con.execute('SELECT version, name FROM "_ufe_migrations"').fetchall()

    for _ in range(len(S.INCOME_BANDS)):
        db.migrate(con)

    second = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns"
        " ORDER BY table_name, column_name"
    ).fetchall()
    assert first == second
    assert con.execute('SELECT version, name FROM "_ufe_migrations"').fetchall() == ledger_first
    assert len(ledger_first) == len(MIGRATIONS)

    # Geometry metadata is not duplicated by a second run either.
    assert len(db.geometry_metadata(con)) == sum(len(v) for v in S.GEOMETRY_COLUMNS.values())
    con.close()

    # And on a re-opened database.
    con = db.connect(path)
    db.migrate(con)
    assert con.execute('SELECT count(*) FROM "_ufe_migrations"').fetchone()[0] == len(MIGRATIONS)
    con.close()


def test_migration_versions_are_unique_and_ordered():
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(set(versions))


def test_ddl_matches_schema_columns():
    for table in S.SCHEMAS:
        ddl = ddl_for(table)
        for column in S.column_order(table):
            assert f'"{column}"' in ddl


def test_write_table_rejects_unknown_table(con, synthetic_city):  # noqa: F811
    with pytest.raises(SchemaValidationError):
        db.write_table(con, "not_a_table", synthetic_city.cells)


def test_write_table_requires_migrated_store(synthetic_city):  # noqa: F811
    raw = db.connect(":memory:")
    with pytest.raises(SchemaValidationError):
        db.write_table(raw, "cells", synthetic_city.cells)
    raw.close()


def test_write_modes(con, synthetic_city):  # noqa: F811
    cells = synthetic_city.cells
    db.write_table(con, "cells", cells)
    db.write_table(con, "cells", cells, mode="replace")
    assert len(db.read_table(con, "cells")) == len(cells)

    bumped = cells.copy()
    bumped["nightlight"] = bumped["nightlight"] + 1
    db.write_table(con, "cells", bumped, mode="upsert")
    back = db.read_table(con, "cells")
    assert len(back) == len(cells)
    assert back["nightlight"].sum() == pytest.approx(cells["nightlight"].sum() + len(cells))


def test_read_table_filters(loaded, synthetic_city):  # noqa: F811
    cells = synthetic_city.cells
    one = cells["h3"].iloc[0]
    assert list(db.read_table(loaded, "cells", h3=one)["h3"]) == [one]

    pair = list(cells["h3"].iloc[: len(S.INCOME_BANDS)])
    assert sorted(db.read_table(loaded, "cells", h3=pair)["h3"]) == sorted(pair)

    in_city = db.read_table(loaded, "cells", in_city=True)
    assert len(in_city) == int(cells["in_city"].sum())

    nulls = db.read_table(loaded, "cells", price_res_inr_sqft=None)
    assert len(nulls) == int(cells["price_res_inr_sqft"].isna().sum())

    assert len(db.read_table(loaded, "cells", h3=[])) == 0

    with pytest.raises(SchemaValidationError):
        db.read_table(loaded, "cells", nope="x")


def test_optional_columns_may_be_absent(con):
    """A Layer 0 frame that has not computed the derived fields still writes."""
    cells = synthetic_cells()
    optional = [c for c, col in S.CELLS.columns.items() if not col.required]
    assert optional, "the cells schema should have optional, layer-derived columns"
    lean = cells.drop(columns=optional)
    db.write_table(con, "cells", lean)
    back = db.read_table(con, "cells")
    assert len(back) == len(lean)
    assert back[optional].isna().all().all()


def test_layer2_shock_columns_are_declared_optional_on_cells():
    """Every `l2_shocks.ADDED_COLUMNS` entry is a declared, optional `cells` column.

    The `cells` schema is `strict=True`, so an undeclared `shock_*` column would make
    `write_table` reject any post-Layer-2 frame (spec Section 9).
    """
    from ufe.layers import l2_shocks

    for column in l2_shocks.ADDED_COLUMNS:
        assert column in S.CELLS.columns, column
        assert not S.CELLS.columns[column].required, column
    assert "shock_jobs_by_sector" in S.LIST_COLUMNS["cells"]
    assert "shock_households_by_band" in S.LIST_COLUMNS["cells"]


def test_a_post_layer2_frame_round_trips_through_the_store(con):
    """A frame carrying the `shock_*` columns writes, reads back and keeps its lists."""
    from ufe.layers import l2_shocks

    cells = synthetic_cells()
    assert set(l2_shocks.ADDED_COLUMNS) <= set(cells.columns)

    db.write_table(con, "cells", cells)
    back = db.read_table(con, "cells").set_index("h3").loc[cells["h3"]].reset_index()

    for column in l2_shocks.ADDED_COLUMNS:
        assert column in back.columns
    assert back["shock_jobs_by_sector"].map(len).eq(len(S.SECTORS)).all()
    assert back["shock_households_by_band"].map(len).eq(len(S.INCOME_BANDS)).all()
    assert not back["shock_field_cap_hit"].any()


# --- synthetic fixture -----------------------------------------------------------------


@pytest.mark.parametrize("table", ["cells", "cells_history", "projects", "announcers",
                                   "project_history"])
def test_synthetic_fixture_validates(synthetic_city, table):  # noqa: F811
    frame = _frames(synthetic_city)[table]
    validated = db.validate(table, frame)
    assert len(validated) == len(frame)


def test_synthetic_cells_column_set_matches_schema_exactly(synthetic_city):  # noqa: F811
    assert list(synthetic_city.cells.columns) == S.column_order("cells")


def test_synthetic_is_reproducible():
    a = synthetic_cells()
    b = synthetic_cells()
    pd.testing.assert_frame_equal(a, b)
    assert db.content_hash(a, sort_by=["h3"]) == db.content_hash(b, sort_by=["h3"])

    seed = len(S.SECTORS)
    pd.testing.assert_frame_equal(synthetic_cells(seed=seed), synthetic_cells(seed=seed))
    assert db.content_hash(synthetic_cells(seed=seed), sort_by=["h3"]) != db.content_hash(
        a, sort_by=["h3"]
    )

    n = len(S.SECTORS) * len(S.INCOME_BANDS)
    small = synthetic_cells(n=n)
    assert len(small) == n
    pd.testing.assert_frame_equal(small, synthetic_cells(n=n))

    for builder in (synthetic_announcers, synthetic_projects):
        pd.testing.assert_frame_equal(builder(), builder())
    pd.testing.assert_frame_equal(synthetic_cells_history(), synthetic_cells_history())
    pd.testing.assert_frame_equal(synthetic_project_history(), synthetic_project_history())


def test_synthetic_city_is_internally_consistent(synthetic_city):  # noqa: F811
    city = synthetic_city
    known = set(city.announcers["announcer_id"])
    linked = city.projects["announcer_id"].dropna()
    assert set(linked) <= known
    # 3.3: announcer_id is null for public projects.
    assert city.projects.loc[city.projects["is_public"], "announcer_id"].isna().all()
    assert set(city.project_history["project_id"]) <= set(city.projects["project_id"])
    assert set(city.cells_history["h3"]) == set(city.cells["h3"])
    assert (city.cells["headroom_sqm"] >= 0).all()
    assert (city.projects["stated_completion"] >= city.projects["announced_date"]).all()


def test_synthetic_city_has_a_few_hundred_hexes(synthetic_city):  # noqa: F811
    """CONTRACT: 'a small synthetic city (a few hundred hexes)'."""
    hundred = len(S.SECTORS) * len(S.SECTORS) + len(S.SECTORS)
    assert hundred < len(synthetic_city.cells) < hundred * len(S.INCOME_BANDS) * len(S.SECTORS)
    assert synthetic_city.cells["h3"].is_unique
    assert synthetic_city.cells["in_city"].any()
    assert not synthetic_city.cells["in_city"].all()


# --- content hashing --------------------------------------------------------------------


def test_content_hash_is_stable_and_order_independent(synthetic_city):  # noqa: F811
    cells = synthetic_city.cells
    base = db.content_hash(cells, sort_by=["h3"])
    shuffled = cells.sample(frac=1, random_state=len(S.SECTORS))
    assert db.content_hash(shuffled, sort_by=["h3"]) == base
    reordered = cells[list(reversed(S.column_order("cells")))]
    assert db.content_hash(reordered, sort_by=["h3"]) == base

    nudged = cells.copy()
    nudged.loc[nudged.index[0], "nightlight"] += 1
    assert db.content_hash(nudged, sort_by=["h3"]) != base


def test_content_hash_survives_a_store_round_trip(con, synthetic_city):  # noqa: F811
    db.write_table(con, "cells", synthetic_city.cells)
    assert db.content_hash(db.read_table(con, "cells"), sort_by=["h3"]) == db.content_hash(
        synthetic_city.cells, sort_by=["h3"]
    )


def test_snapshot_hash_is_reproducible(loaded, tmp_path):  # noqa: F811
    kwargs = dict(city_id="vizag", created_by="tester", params_dir=tmp_path / "absent")
    first = db.write_snapshot(loaded, out_root=tmp_path / "s1", **kwargs)
    second = db.write_snapshot(loaded, out_root=tmp_path / "s2", **kwargs)
    manifest_a = json.loads(first.manifest_path.read_text())
    manifest_b = json.loads(second.manifest_path.read_text())
    assert manifest_a["content_hashes"] == manifest_b["content_hashes"]


# --- Section 0.1 rule 3: no numeric parameters in Python ---------------------------------

_ALLOWED_NUMBERS = {0, 1}


def _offending_numbers(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    const_assignments: set[int] = set()
    subscript_nodes: set[int] = set()
    check_call_nodes: set[int] = set()

    for node in ast.walk(tree):
        # Named module-level constants (SHORT_HASH_LEN = 8) are declarations, not literals
        # buried in an expression.
        if isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.lstrip("_").isupper() for t in node.targets
        ):
            const_assignments.update(id(n) for n in ast.walk(node.value))
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.lstrip("_").isupper() and node.value is not None:
                const_assignments.update(id(n) for n in ast.walk(node.value))
        if isinstance(node, ast.Subscript):
            subscript_nodes.update(id(n) for n in ast.walk(node.slice))
        # Schema bounds declared through pandera Checks are allowed by the brief.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "Check":
                    check_call_nodes.update(id(n) for n in ast.walk(node))

    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if not isinstance(node.value, (int, float)):
            continue
        if node.value in _ALLOWED_NUMBERS:
            continue
        if id(node) in const_assignments or id(node) in subscript_nodes:
            continue
        if id(node) in check_call_nodes:
            continue
        bad.append(f"{path.name}:{node.lineno}: {node.value!r}")
    return bad


@pytest.mark.parametrize(
    "relative",
    [
        "ufe/store/db.py",
        "ufe/store/schemas.py",
        "ufe/store/migrations/__init__.py",
        "tests/fixtures/synthetic.py",
    ],
)
def test_no_numeric_parameters_in_python(relative):
    """0.1 rule 3 / CONTRACT: only 0, 1, array indices and named declarations."""
    assert _offending_numbers(REPO_ROOT / relative) == []


def test_synthetic_generation_constants_live_in_yaml():
    from tests.fixtures import synthetic

    assert synthetic.CONFIG_PATH.is_file()
    assert {"grid", "cells", "projects", "announcers", "history"} <= set(synthetic.CONFIG)
