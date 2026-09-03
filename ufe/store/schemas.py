"""Pandera schemas for every table in the Urban Futures Engine store (spec Section 3).

Every table in DuckDB has exactly one :class:`pandera.DataFrameSchema` here and is validated
on write by :func:`ufe.store.db.write_table`.  Writing an invalid frame **raises**
:class:`ufe.errors.SchemaValidationError` — it never warns (Section 3, preamble).

Conventions
-----------
* Units follow Section 0.3: metres, minutes, square metres, INR, INR/sqft, integer calendar
  year, probability 0–1.
* Geometry is stored in ``EPSG:4326`` (Section 0.3).  ``cells.geometry`` is **WKB** (DuckDB
  ``BLOB``) because there is one polygon per cell and a city has 150k–300k of them;
  ``projects.geom`` is **WKT** (DuckDB ``VARCHAR``) because Section 3.3 says so explicitly.
  The encoding and CRS of every geometry column are recorded in the ``_geometry_columns``
  metadata table written by the migrations.
* ``hh_by_band`` and ``jobs_by_sector`` are fixed-length ``list[float]`` columns whose length
  is checked against :data:`INCOME_BANDS` and :data:`SECTORS` respectively — never against a
  literal.
* Columns marked ``required=False`` are the derived fields produced by later layers
  (Sections 7, 8, 11, 12, 19).  They live in the same physical table so that downstream
  modules have one place to read and write cell state; a Layer 0 frame that has not yet
  computed them still validates.

Numeric policy (Section 0.1 rule 3): the only numbers in this module are the structural
bounds ``0`` and ``1`` used in schema declarations.  Behavioural thresholds — including the
income-band boundaries of Section 3.7 — are read from YAML via :func:`income_band_boundaries`.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ufe.params import Params

__all__ = [
    "Sector",
    "IncomeBand",
    "SECTORS",
    "INCOME_BANDS",
    "ZONE_CLASSES",
    "CRZ_CLASSES",
    "LANDCOVER_CLASSES",
    "UTILITY_STATES",
    "ELASTICITY_CLASSES",
    "PROJECT_STAGES",
    "COMMITMENT_FORMS",
    "FUNDING_SOURCES",
    "GEOM_TYPES",
    "SCALE_UNITS",
    "PHYSICAL_STATES",
    "GEOMETRY_CRS",
    "GEOMETRY_ENCODING",
    "INCOME_BAND_BOUNDARY_PATH",
    "PARAMETER_VALUE_KEY",
    "income_band_boundaries",
    "CELLS",
    "CELLS_HISTORY",
    "PROJECTS",
    "ANNOUNCERS",
    "PROJECT_HISTORY",
    "SNAPSHOTS",
    "SCHEMAS",
    "LIST_COLUMNS",
    "GEOMETRY_COLUMNS",
    "PRIMARY_KEYS",
    "sql_type",
    "column_order",
]


# --------------------------------------------------------------------------------------
# 3.6 Sector taxonomy / 3.7 Income bands — validated enums, not free strings
# --------------------------------------------------------------------------------------


class Sector(enum.IntEnum):
    """The eight sectors used everywhere jobs are counted (Section 3.6).

    The integer value is the position in ``jobs_by_sector``.
    """

    agri = 0
    manuf_heavy = enum.auto()
    manuf_light = enum.auto()
    logistics = enum.auto()
    it_office = enum.auto()
    retail_svc = enum.auto()
    public_edu = enum.auto()
    construction = enum.auto()


class IncomeBand(enum.IntEnum):
    """The four monthly-household-income bands (Section 3.7).

    The integer value is the position in ``hh_by_band``.  The *boundaries* are not defined
    here — they live in ``behaviour.yaml`` and are inflation-indexed by base year.
    """

    low = 0
    mid = enum.auto()
    upper_mid = enum.auto()
    high = enum.auto()


SECTORS: tuple[str, ...] = tuple(s.name for s in Sector)
INCOME_BANDS: tuple[str, ...] = tuple(b.name for b in IncomeBand)

#: Dotted parameter path holding the income-band boundaries (Section 3.7, ``behaviour.yaml``).
INCOME_BAND_BOUNDARY_PATH = "behaviour.income_bands.boundaries_inr_mo"


#: The key holding the deterministic value of a Section 4.1 parameter leaf.
PARAMETER_VALUE_KEY = "value"


def _leaf_value(entry: Any, path: str) -> float:
    """Coerce one boundary entry to a float.

    ``behaviour.yaml`` encodes the Section 3.7 boundaries as a sequence of Section 4.1
    parameter leaves (``{value, conf, scope}``) so each boundary carries its own confidence
    tag and scope, but the spec prints them as bare scalars.  Both forms are accepted here.
    """
    if isinstance(entry, Mapping):
        if PARAMETER_VALUE_KEY not in entry:
            raise ValueError(
                f"{path} entry {entry!r} is a mapping with no {PARAMETER_VALUE_KEY!r} key: "
                "a Section 4.1 parameter leaf must carry one"
            )
        entry = entry[PARAMETER_VALUE_KEY]
    return float(entry)


def income_band_boundaries(params: "Params") -> list[float]:
    """Return the monthly-income band boundaries from ``behaviour.yaml``.

    Section 3.7 is explicit: "Band boundaries live in ``behaviour.yaml`` and must be
    inflation-indexed by base year.  Do not hardcode."  There are ``len(INCOME_BANDS) - 1``
    of them.  Each may be either a bare scalar (as the spec prints it) or a Section 4.1
    parameter leaf (as the landed ``behaviour.yaml`` encodes it) — see :func:`_leaf_value`.
    """
    boundaries = list(params.get(INCOME_BAND_BOUNDARY_PATH))
    expected = len(INCOME_BANDS) - 1
    if len(boundaries) != expected:
        raise ValueError(
            f"{INCOME_BAND_BOUNDARY_PATH} must have {expected} boundaries for "
            f"{len(INCOME_BANDS)} bands, got {len(boundaries)}"
        )
    return [_leaf_value(b, INCOME_BAND_BOUNDARY_PATH) for b in boundaries]


# --------------------------------------------------------------------------------------
# Categorical vocabularies
# --------------------------------------------------------------------------------------

#: Section 3.1 ``zone_class``.
ZONE_CLASSES: tuple[str, ...] = ("res", "com", "ind", "agri", "public", "eco", "mixed")

#: Section 3.1 ``crz_class``.
CRZ_CLASSES: tuple[str, ...] = ("none", "I", "II", "III", "IV")

#: Section 6.2 — ESA WorldCover dominant class, mapped to readable names.
LANDCOVER_CLASSES: tuple[str, ...] = (
    "tree_cover",
    "shrubland",
    "grassland",
    "cropland",
    "builtup",
    "bare",
    "snow_ice",
    "water",
    "herbaceous_wetland",
    "mangroves",
    "moss_lichen",
)

#: Section 7.3 — derived from the three utility booleans, in order of precedence.
UTILITY_STATES: tuple[str, ...] = ("none", "water", "water_sewer", "water_sewer_power")

#: Section 7.4 — supply-elasticity classes.
ELASTICITY_CLASSES: tuple[str, ...] = (
    "dense_core",
    "constrained_periph",
    "open_fringe",
    "typical_periph",
)

#: Project stage enum.  Section 3.3 cites "§11.1" but the stage vocabulary is actually
#: defined by ``credibility.yaml: stage_probability`` (Sections 4.4 / 10.1) — see the
#: module note in the build report.
PROJECT_STAGES: tuple[str, ...] = (
    "announced",
    "feasibility",
    "dpr_prepared",
    "dpr_approved",
    "funded",
    "land_majority",
    "construction",
    "half_complete",
)

#: Private-project commitment form.  Section 3.3 cites "§11.5"; the vocabulary is
#: ``credibility.yaml: commitment_hardness`` (Sections 4.4 / 10.1).
COMMITMENT_FORMS: tuple[str, ...] = (
    "verbal",
    "summit_mou",
    "govt_mou_signed",
    "land_allotted",
    "board_approved",
    "land_possessed",
    "ec_granted",
    "epc_appointed",
    "equipment_ordered",
    "construction_seen",
)

#: Section 3.3 ``funding_source``.  The spec says "enum" but never enumerates it; this set
#: is the build's proposal and is flagged as a gap.
FUNDING_SOURCES: tuple[str, ...] = (
    "state_budget",
    "central_scheme",
    "multilateral",
    "ppp",
    "private_equity",
    "internal_accrual",
    "debt",
    "unknown",
)

#: Section 3.3 ``geom_type``.
GEOM_TYPES: tuple[str, ...] = ("point", "polygon", "linestring")

#: Section 3.3 ``scale_unit``.
SCALE_UNITS: tuple[str, ...] = (
    "mw",
    "seats",
    "beds",
    "acres",
    "units_per_year",
    "mppa",
    "km",
    "lakh_sqft",
)

#: Section 18.1 ``physical_state``.
PHYSICAL_STATES: tuple[str, ...] = (
    "none",
    "cleared",
    "earthworks",
    "structure",
    "operational",
    "unobserved",
)

#: Section 0.3 — everything on disk is EPSG:4326.
GEOMETRY_CRS = "EPSG:4326"

#: Per-column geometry encoding.  See the module docstring for the rationale.
GEOMETRY_ENCODING: dict[tuple[str, str], str] = {
    ("cells", "geometry"): "WKB",
    ("projects", "geom"): "WKT",
}


# --------------------------------------------------------------------------------------
# Reusable checks
# --------------------------------------------------------------------------------------

_UNIT_INTERVAL = Check.in_range(0, 1)
_NON_NEGATIVE = Check.greater_than_or_equal_to(0)
_BOOL_INT = Check.isin((0, 1))


def _isin(values: Sequence[str]) -> Check:
    return Check.isin(tuple(values))


def _fixed_length_floats(n: int, what: str) -> Check:
    """Every element must be a length-``n`` sequence of finite numbers."""

    def _ok(series: pd.Series) -> pd.Series:
        def check_one(v: Any) -> bool:
            if v is None:
                return False
            try:
                seq = list(v)
            except TypeError:
                return False
            if len(seq) != n:
                return False
            return all(isinstance(x, (int, float)) or hasattr(x, "__float__") for x in seq)

        return series.map(check_one)

    return Check(_ok, element_wise=False, error=f"must be a list of {n} floats ({what})")


def _list_of_str(*, non_empty: bool = False) -> Check:
    def _ok(series: pd.Series) -> pd.Series:
        def check_one(v: Any) -> bool:
            if v is None:
                return not non_empty
            try:
                seq = list(v)
            except TypeError:
                return False
            if non_empty and not seq:
                return False
            return all(isinstance(x, str) for x in seq)

        return series.map(check_one)

    suffix = ", non-empty" if non_empty else ""
    return Check(_ok, element_wise=False, error=f"must be a list of strings{suffix}")


def _bytes_column() -> Check:
    def _ok(series: pd.Series) -> pd.Series:
        return series.map(lambda v: v is None or isinstance(v, (bytes, bytearray, memoryview)))

    return Check(_ok, element_wise=False, error="must be WKB bytes")


def _num(sql: str, *, nullable: bool = False, required: bool = True, checks=None) -> Column:
    return Column(
        float,
        checks=checks,
        nullable=nullable,
        required=required,
        coerce=True,
        metadata={"sql": sql},
    )


_F = "DOUBLE"


# --------------------------------------------------------------------------------------
# 3.1 cells
# --------------------------------------------------------------------------------------

CELLS = DataFrameSchema(
    name="cells",
    strict=True,
    coerce=True,
    columns={
        # --- identity & geometry (Sections 3.1, 5.1) -----------------------------------
        "h3": Column(str, unique=True, nullable=False, metadata={"sql": "VARCHAR"}),
        "h3_res8": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "in_city": Column(bool, nullable=False, coerce=True, metadata={"sql": "BOOLEAN"}),
        "geometry": Column(
            object,
            checks=_bytes_column(),
            nullable=False,
            metadata={"sql": "BLOB", "geometry": "WKB", "crs": GEOMETRY_CRS},
        ),
        "lat": _num(_F, checks=Check.in_range(-90, 90)),
        "lon": _num(_F, checks=Check.in_range(-180, 180)),
        "area_sqm": _num(_F, checks=Check.greater_than(0)),
        # --- terrain & land cover (Sections 3.1, 6.1, 6.2) -----------------------------
        "elev_m": _num(_F),
        "slope_pct": _num(_F, checks=_NON_NEGATIVE),
        "landcover": Column(
            str, checks=_isin(LANDCOVER_CLASSES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "builtup_frac": _num(_F, checks=_UNIT_INTERVAL),
        "undevelopable_frac": _num(_F, checks=_UNIT_INTERVAL),
        # --- regulation (Sections 3.1, 6.9) --------------------------------------------
        "zone_class": Column(
            str, checks=_isin(ZONE_CLASSES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "permitted_far": _num(_F, checks=_NON_NEGATIVE),
        "crz_class": Column(
            str, checks=_isin(CRZ_CLASSES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        # --- people & jobs --------------------------------------------------------------
        "population": _num(_F, checks=_NON_NEGATIVE),
        "households": _num(_F, checks=_NON_NEGATIVE),
        "hh_by_band": Column(
            object,
            checks=_fixed_length_floats(len(INCOME_BANDS), "income bands, Section 3.7"),
            nullable=False,
            metadata={"sql": "DOUBLE[]", "list_len": len(INCOME_BANDS)},
        ),
        "jobs_by_sector": Column(
            object,
            checks=_fixed_length_floats(len(SECTORS), "sectors, Section 3.6"),
            nullable=False,
            metadata={"sql": "DOUBLE[]", "list_len": len(SECTORS)},
        ),
        # --- stock & price ----------------------------------------------------------------
        "floorspace_res_sqm": _num(_F, checks=_NON_NEGATIVE),
        "floorspace_com_sqm": _num(_F, checks=_NON_NEGATIVE),
        "price_res_inr_sqft": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "price_land_inr_sqft": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "rent_res_inr_sqft_mo": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "mean_parcel_sqm": _num(_F, nullable=True, checks=Check.greater_than(0)),
        "parcel_count": Column(
            "int64", checks=_NON_NEGATIVE, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        # --- utilities (0/1 flags, Section 3.1) --------------------------------------------
        "util_water": Column(
            "int64", checks=_BOOL_INT, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        "util_sewer": Column(
            "int64", checks=_BOOL_INT, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        "util_power": Column(
            "int64", checks=_BOOL_INT, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        # --- distances & remote sensing -----------------------------------------------------
        "dist_cbd_m": _num(_F, checks=_NON_NEGATIVE),
        "dist_coast_m": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "dist_arterial_m": _num(_F, checks=_NON_NEGATIVE),
        "nightlight": _num(_F, checks=_NON_NEGATIVE),
        "data_conf": _num(_F, checks=_UNIT_INTERVAL),
        # --- derived by Layer 0 (Section 7); absent on a freshly ingested frame -------------
        "utility_state": Column(
            str,
            checks=_isin(UTILITY_STATES),
            nullable=True,
            required=False,
            metadata={"sql": "VARCHAR"},
        ),
        "slope_cost_mult": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "capacity_sqm": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "headroom_sqm": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "elasticity_class": Column(
            str,
            checks=_isin(ELASTICITY_CLASSES),
            nullable=True,
            required=False,
            metadata={"sql": "VARCHAR"},
        ),
        "eps_supply": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "regulatory_index": _num(_F, nullable=True, required=False),
        # --- derived by Layer 1 accessibility (Section 8.5) ---------------------------------
        "lnA": _num(_F, nullable=True, required=False),
        "lnA_work": _num(_F, nullable=True, required=False),
        "lnA_retail": _num(_F, nullable=True, required=False),
        "lnA_education": _num(_F, nullable=True, required=False),
        "lnA_health": _num(_F, nullable=True, required=False),
        "jobs_30min": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "jobs_45min": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "jobs_60min": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "station_weight": _num(_F, nullable=True, required=False, checks=_UNIT_INTERVAL),
        # --- opportunity inputs for accessibility (Section 8.2) -----------------------------
        "retail_poi_count": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "education_poi_count": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "health_poi_count": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "school_seats": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "hospital_beds": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        # --- allocation utility terms (Sections 9.x, 12.3) ----------------------------------
        "amenity": _num(_F, nullable=True, required=False),
        "disamenity": _num(_F, nullable=True, required=False),
        "alpha_res": _num(_F, nullable=True, required=False),
        # --- supply state (Section 11) -------------------------------------------------------
        "inventory_months": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "hist_absorption_sqm": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        # --- backtest baseline B4 covariate (Section 19.3) -----------------------------------
        "dist_existing_builtup_m": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        # --- derived by Layer 2 shock resolution (Section 9) ---------------------------------
        # These are exactly `ufe.layers.l2_shocks.ADDED_COLUMNS`; the resolution is per-year,
        # so a frame carrying them is a snapshot of one simulated year, not a stock.
        #
        # Section 9.4 composed premium/disamenity fields, in log points.  They are clipped by
        # `price.fields.cap_low` / `cap_high`, which are YAML and may be negative (a
        # disamenity field is a negative log-point shift), so no sign check is declared here.
        "shock_field_residential": _num(_F, nullable=True, required=False),
        "shock_field_commercial": _num(_F, nullable=True, required=False),
        "shock_field_office": _num(_F, nullable=True, required=False),
        # True where composition hit either field cap in any target (Section 9.4 logs it).
        "shock_field_cap_hit": Column(
            bool,
            nullable=False,
            required=False,
            coerce=True,
            metadata={"sql": "BOOLEAN"},
        ),
        # Section 9.2 employment, activation-weighted, sited in the cell.
        "shock_jobs_permanent": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "shock_jobs_construction": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        "shock_jobs_by_sector": Column(
            object,
            checks=_fixed_length_floats(len(SECTORS), "sectors, Section 3.6"),
            nullable=False,
            required=False,
            metadata={"sql": "DOUBLE[]", "list_len": len(SECTORS)},
        ),
        # Section 9.5 household routing.
        "shock_effective_households": _num(
            _F, nullable=True, required=False, checks=_NON_NEGATIVE
        ),
        "shock_households_by_band": Column(
            object,
            checks=_fixed_length_floats(len(INCOME_BANDS), "income bands, Section 3.7"),
            nullable=False,
            required=False,
            metadata={"sql": "DOUBLE[]", "list_len": len(INCOME_BANDS)},
        ),
        "shock_dormitory_workers": _num(_F, nullable=True, required=False, checks=_NON_NEGATIVE),
        # Section 9.2 step 6 non-residential floorspace demand.
        "shock_floorspace_demand_sqm": _num(
            _F, nullable=True, required=False, checks=_NON_NEGATIVE
        ),
        # Section 9.1 `SupplyEffect` deltas, summed per cell.  `delta_capacity_sqm` is
        # negative when land is sterilised, so neither delta carries a sign check.
        "shock_delta_capacity_sqm": _num(_F, nullable=True, required=False),
        "shock_delta_floorspace_sqm": _num(_F, nullable=True, required=False),
    },
)


# --------------------------------------------------------------------------------------
# 3.2 cells_history
# --------------------------------------------------------------------------------------

CELLS_HISTORY = DataFrameSchema(
    name="cells_history",
    strict=True,
    coerce=True,
    columns={
        "h3": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "year": Column("int64", nullable=False, coerce=True, metadata={"sql": "BIGINT"}),
        "builtup_frac": _num(_F, nullable=True, checks=_UNIT_INTERVAL),
        "nightlight": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "population": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "price_res_inr_sqft": _num(_F, nullable=True, checks=_NON_NEGATIVE),
    },
    unique=["h3", "year"],
)


# --------------------------------------------------------------------------------------
# 3.3 projects
# --------------------------------------------------------------------------------------

_DATE = "DATE"
_TS = "TIMESTAMP"


def _date_col(*, nullable: bool = False, sql: str = _DATE) -> Column:
    return Column(
        "datetime64[ns]", nullable=nullable, coerce=True, metadata={"sql": sql}
    )


PROJECTS = DataFrameSchema(
    name="projects",
    strict=True,
    coerce=True,
    columns={
        "project_id": Column(str, unique=True, nullable=False, metadata={"sql": "VARCHAR"}),
        "name": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "archetype": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "geom_type": Column(
            str, checks=_isin(GEOM_TYPES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "geom": Column(
            str,
            nullable=False,
            metadata={"sql": "VARCHAR", "geometry": "WKT", "crs": GEOMETRY_CRS},
        ),
        "announcer_id": Column(str, nullable=True, metadata={"sql": "VARCHAR"}),
        "is_public": Column(bool, nullable=False, coerce=True, metadata={"sql": "BOOLEAN"}),
        "scale_value": _num(_F, checks=_NON_NEGATIVE),
        "scale_unit": Column(
            str, checks=_isin(SCALE_UNITS), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "capex_inr_cr": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "stated_jobs": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "median_wage_inr_mo": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "announced_date": _date_col(),
        "stated_completion": _date_col(),
        "stage": Column(
            str, checks=_isin(PROJECT_STAGES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "stage_asof": _date_col(),
        "commitment_form": Column(
            str,
            checks=_isin(COMMITMENT_FORMS),
            nullable=True,
            metadata={"sql": "VARCHAR"},
        ),
        "land_possession_pct": _num(_F, nullable=True, checks=_UNIT_INTERVAL),
        "funding_source": Column(
            str, checks=_isin(FUNDING_SOURCES), nullable=False, metadata={"sql": "VARCHAR"}
        ),
        "modifiers": Column(
            object,
            checks=_list_of_str(),
            nullable=False,
            metadata={"sql": "VARCHAR[]", "list_of": "str"},
        ),
        "physical_state": Column(
            str,
            checks=_isin(PHYSICAL_STATES),
            nullable=True,
            metadata={"sql": "VARCHAR"},
        ),
        "physical_asof": _date_col(nullable=True),
        "source_urls": Column(
            object,
            checks=_list_of_str(non_empty=True),
            nullable=False,
            metadata={"sql": "VARCHAR[]", "list_of": "str"},
        ),
        "extracted_by": Column(
            str,
            checks=Check.str_matches(r"^(human|ai:[\w.\-]+)$"),
            nullable=False,
            metadata={"sql": "VARCHAR"},
        ),
        "verified_by": Column(str, nullable=True, metadata={"sql": "VARCHAR"}),
        "first_seen": _date_col(sql=_TS),
        "last_updated": _date_col(sql=_TS),
    },
)


# --------------------------------------------------------------------------------------
# 3.4 announcers
# --------------------------------------------------------------------------------------

ANNOUNCERS = DataFrameSchema(
    name="announcers",
    strict=True,
    coerce=True,
    columns={
        "announcer_id": Column(str, unique=True, nullable=False, metadata={"sql": "VARCHAR"}),
        "name": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "aliases": Column(
            object, checks=_list_of_str(), nullable=False, metadata={"sql": "VARCHAR[]"}
        ),
        "parent_id": Column(str, nullable=True, metadata={"sql": "VARCHAR"}),
        "is_listed": Column(bool, nullable=False, coerce=True, metadata={"sql": "BOOLEAN"}),
        "announced_capex_10y_inr_cr": _num(_F, checks=_NON_NEGATIVE),
        "deployed_capex_10y_inr_cr": _num(_F, checks=_NON_NEGATIVE),
        "delivery_ratio": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "median_slip_months": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "mean_annual_capex_3y_inr_cr": _num(_F, nullable=True, checks=_NON_NEGATIVE),
        "net_debt_ebitda": _num(_F, nullable=True),
        "record_sources": Column(
            object,
            checks=_list_of_str(non_empty=True),
            nullable=False,
            metadata={"sql": "VARCHAR[]"},
        ),
        "record_asof": _date_col(),
    },
)


# --------------------------------------------------------------------------------------
# 3.5 project_history — append-only audit log
# --------------------------------------------------------------------------------------

PROJECT_HISTORY = DataFrameSchema(
    name="project_history",
    strict=True,
    coerce=True,
    columns={
        "project_id": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "field": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        # Values are serialised to text so one log can hold every field type.
        "old_value": Column(str, nullable=True, metadata={"sql": "VARCHAR"}),
        "new_value": Column(str, nullable=True, metadata={"sql": "VARCHAR"}),
        "changed_at": _date_col(sql=_TS),
        "source_url": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "changed_by": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
    },
)


# --------------------------------------------------------------------------------------
# 3.8 snapshots
# --------------------------------------------------------------------------------------

SNAPSHOTS = DataFrameSchema(
    name="snapshots",
    strict=True,
    coerce=True,
    columns={
        # ``{YYYY-MM-DD}_{shorthash}`` — the on-disk directory name.
        "snapshot_id": Column(str, unique=True, nullable=False, metadata={"sql": "VARCHAR"}),
        "snapshot_hash": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "city_id": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "created_at": _date_col(sql=_TS),
        "created_by": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "path": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "params_hash": Column(str, nullable=False, metadata={"sql": "VARCHAR"}),
        "cells_rows": Column(
            "int64", checks=_NON_NEGATIVE, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        "projects_rows": Column(
            "int64", checks=_NON_NEGATIVE, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        "announcers_rows": Column(
            "int64", checks=_NON_NEGATIVE, nullable=False, coerce=True, metadata={"sql": "BIGINT"}
        ),
        "file_hashes": Column(
            object, checks=_list_of_str(non_empty=True), nullable=False,
            metadata={"sql": "VARCHAR[]"},
        ),
        "ingest_run_ids": Column(
            object, checks=_list_of_str(), nullable=False, metadata={"sql": "VARCHAR[]"}
        ),
    },
)


SCHEMAS: dict[str, DataFrameSchema] = {
    "cells": CELLS,
    "cells_history": CELLS_HISTORY,
    "projects": PROJECTS,
    "announcers": ANNOUNCERS,
    "project_history": PROJECT_HISTORY,
    "snapshots": SNAPSHOTS,
}

#: Table -> list-typed columns.  DuckDB hands these back as ``numpy.ndarray``; the store
#: converts them to plain lists on read so round-trips are value-stable.
LIST_COLUMNS: dict[str, tuple[str, ...]] = {
    table: tuple(
        name
        for name, col in schema.columns.items()
        if str(col.metadata.get("sql", "")).endswith("[]")
    )
    for table, schema in SCHEMAS.items()
}

#: Table -> {column: (encoding, crs)} for every geometry column.
GEOMETRY_COLUMNS: dict[str, dict[str, tuple[str, str]]] = {
    table: {
        name: (col.metadata["geometry"], col.metadata["crs"])
        for name, col in schema.columns.items()
        if col.metadata and "geometry" in col.metadata
    }
    for table, schema in SCHEMAS.items()
}

#: Logical primary keys.  Enforced by pandera (``unique=``), not by DuckDB constraints —
#: bulk loads are faster without them and the schema is the single source of truth.
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "cells": ("h3",),
    "cells_history": ("h3", "year"),
    "projects": ("project_id",),
    "announcers": ("announcer_id",),
    "project_history": (),
    "snapshots": ("snapshot_id",),
}


def sql_type(table: str, column: str) -> str:
    """DuckDB column type for ``table.column``, taken from the pandera metadata."""
    col = SCHEMAS[table].columns[column]
    return str(col.metadata["sql"])


def column_order(table: str) -> list[str]:
    """Physical column order for ``table``, i.e. the declaration order in this module."""
    return list(SCHEMAS[table].columns)
