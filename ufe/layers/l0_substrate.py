"""Layer 0 — substrate assembly (spec Section 7).

Assembles the ingested cell attributes into the derived Layer 0 fields:

======================  =========================================================
Section                 Column(s) produced
======================  =========================================================
7.1                     ``undevelopable_frac``
7.2                     ``slope_cost_mult``
7.3                     ``utility_state``, ``capacity_sqm``, ``headroom_sqm``
7.4                     ``elasticity_class``, ``eps_supply``
7.5                     ``jobs_by_sector`` (only when census inputs are supplied)
======================  =========================================================

The single public entry point is :func:`assemble_substrate`.  It is pure: it never mutates
its argument, holds no module state, uses no RNG, and returns a new frame with the same
index and row count as the input plus the derived columns (CONTRACT.md).

Every threshold, coefficient, slope band, FAR gate and elasticity cutoff is read from
``config/params/supply.yaml`` through :class:`ufe.params.Params`; the only literals in this
module are ``0`` and ``1`` (spec Section 0.1 rule 3).

Section 7.1 requires the hard-gate layers to be *unioned geometrically* before their area
share is taken, "not summed — summing double-counts".  Pass those layers as ``gates``:
a mapping of gate name to a sequence of shapely geometries in EPSG:4326.  They are unioned,
reprojected into the city's ``crs_metric`` and intersected with each cell.  With no
``gates`` the layer falls back to the ``undevelopable_frac`` produced at ingest (Section
6.2), which is already a union share, and adds only the Section 7.2 slope gate to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import MissingParameter, SchemaValidationError
from ufe.params import Params
from ufe.store import schemas as S

logger = logging.getLogger(__name__)

# A metric-CRS helper is owned by ``ufe.geo``; it may or may not exist yet.
try:  # pragma: no cover - exercised by whichever of the two branches is live
    from ufe import geo as _geo
except ImportError:  # pragma: no cover
    _geo = None

__all__ = [
    "assemble_substrate",
    "assembly_feasibility_tiers",
    "assembly_multiplier",
    "fit_elasticity_regression",
    "AssemblyTier",
    "RegressionFit",
    "DERIVED_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "DEFAULT_WARD_COL",
]


# --------------------------------------------------------------------------------------
# parameter paths (spec Section 7, config/params/supply.yaml)
# --------------------------------------------------------------------------------------

P_SLOPE_CUTOFF_PCT = "supply.slope.cutoff_pct"
P_SLOPE_PENALTY_START_PCT = "supply.slope.penalty_start_pct"
P_SLOPE_PENALTY_PER_PCT = "supply.slope.penalty_per_pct"

P_UTILITY_GATE = "supply.utility_gate"
P_ASSEMBLY_FEASIBILITY = "supply.assembly_feasibility"
P_ASSEMBLY_MIN_PARCEL_KEY = "min_parcel_sqm"
P_LEAF_VALUE_KEY = "value"

P_ELASTICITY_CLASS = "supply.elasticity_class"
P_ELASTICITY_CLASS_DEFAULT = "supply.elasticity_class_default"
P_DENSE_CORE_BUILTUP_MIN = "supply.elasticity_classifier.dense_core_builtup_frac_min"
P_DENSE_CORE_HEADROOM_RATIO_MAX = "supply.elasticity_classifier.dense_core_headroom_ratio_max"
P_CONSTRAINED_UNDEVELOPABLE_MIN = "supply.elasticity_classifier.constrained_undevelopable_min"
P_OPEN_FRINGE_BUILTUP_MAX = "supply.elasticity_classifier.open_fringe_builtup_frac_max"

P_REGRESSION_ENABLED = "supply.elasticity_regression.enabled"
P_REGRESSION_COEFFICIENTS = "supply.elasticity_regression.coefficients"
P_REGRESSION_MIN_ABS_DLN_PRICE = "supply.elasticity_regression.fit.min_abs_dln_price"
P_REGRESSION_LOG_OFFSET = "supply.elasticity_regression.fit.log_offset"

#: Section 7.5 step 2 maps each sector onto a POI category.  The mapping is data, not code;
#: it is read from here when present and otherwise the dasymetric surface falls back to
#: commercial floorspace alone.  See the module note in the build report.
P_SECTOR_POI_COLUMN = "supply.jobs.sector_poi_column"

CITY_CRS_METRIC_KEY = "crs_metric"

REGRESSION_COEFFICIENT_NAMES = ("a0", "a1", "a2", "a3", "a4")


# --------------------------------------------------------------------------------------
# vocabularies (taken from the schema, never re-spelled)
# --------------------------------------------------------------------------------------

STATE_NONE, STATE_WATER, STATE_WATER_SEWER, STATE_WATER_SEWER_POWER = S.UTILITY_STATES
CLASS_DENSE_CORE, CLASS_CONSTRAINED_PERIPH, CLASS_OPEN_FRINGE, CLASS_TYPICAL_PERIPH = (
    S.ELASTICITY_CLASSES
)

#: Columns :func:`assemble_substrate` writes.  Every one is declared in ``schemas.CELLS``.
DERIVED_COLUMNS: tuple[str, ...] = (
    "undevelopable_frac",
    "slope_cost_mult",
    "utility_state",
    "capacity_sqm",
    "headroom_sqm",
    "elasticity_class",
    "eps_supply",
    "jobs_by_sector",
)

#: Columns the layer reads and therefore requires on the input frame.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "area_sqm",
    "slope_pct",
    "builtup_frac",
    "undevelopable_frac",
    "permitted_far",
    "floorspace_res_sqm",
    "floorspace_com_sqm",
    "mean_parcel_sqm",
    "util_water",
    "util_sewer",
    "util_power",
    "jobs_by_sector",
)

#: Section 7.5 works ward by ward.  The synthetic grid and the real grid both carry the
#: res-8 parent as the coarsest stable grouping; a real ward id column can be named instead.
DEFAULT_WARD_COL = "h3_res8"

COL_FLOORSPACE_COM = "floorspace_com_sqm"
COL_REGULATORY_INDEX = "regulatory_index"
COL_GEOMETRY = "geometry"


@dataclass(frozen=True)
class AssemblyTier:
    """One rung of the Section 7.3 ``assembly_feasibility(mean_parcel_sqm)`` step function."""

    min_parcel_sqm: float
    value: float


@dataclass(frozen=True)
class RegressionFit:
    """Result of the Section 7.4 elasticity OLS, for the run manifest."""

    coefficients: dict[str, float] = field(default_factory=dict)
    r2: float = 0.0
    n: int = 0


# --------------------------------------------------------------------------------------
# 7.1 undevelopable fraction
# --------------------------------------------------------------------------------------


def _metric_geometries(geometries: Sequence[Any], crs_metric: str) -> list[Any]:
    """Reproject EPSG:4326 geometries into the city's metric CRS (Section 0.3)."""
    helper = getattr(_geo, "to_metric", None) if _geo is not None else None
    if callable(helper):
        import geopandas as gpd

        series = gpd.GeoSeries(list(geometries), crs=S.GEOMETRY_CRS)
        return list(helper(series, crs_metric))

    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(S.GEOMETRY_CRS, crs_metric, always_xy=True)
    return [shapely_transform(transformer.transform, geom) for geom in geometries]


def _iter_gate_geometries(gates: Mapping[str, Any]) -> list[Any]:
    geoms: list[Any] = []
    for layer in gates.values():
        if layer is None:
            continue
        if hasattr(layer, "geometry") and not isinstance(layer, (list, tuple)):
            layer = list(layer.geometry)  # a GeoDataFrame / GeoSeries
        geoms.extend(geom for geom in layer if geom is not None and not geom.is_empty)
    return geoms


def _gate_union_fraction(
    cells: pd.DataFrame, gates: Mapping[str, Any], crs_metric: str
) -> np.ndarray:
    """Per-cell area share of the geometric UNION of every hard-gate layer (Section 7.1)."""
    from shapely import wkb as shapely_wkb
    from shapely.ops import unary_union

    gate_geoms = _iter_gate_geometries(gates)
    if not gate_geoms:
        return np.zeros(len(cells))
    if COL_GEOMETRY not in cells.columns:
        raise SchemaValidationError(
            "geometric hard gates were supplied but the cells frame has no "
            f"{COL_GEOMETRY!r} column"
        )

    cell_geoms = [shapely_wkb.loads(bytes(blob)) for blob in cells[COL_GEOMETRY]]
    projected = _metric_geometries(list(gate_geoms) + cell_geoms, crs_metric)
    union = unary_union(projected[: len(gate_geoms)])

    shares = np.zeros(len(cells))
    for position, cell in enumerate(projected[len(gate_geoms) :]):
        area = cell.area
        if area <= 0:
            continue
        shares[position] = cell.intersection(union).area / area
    return shares


def _slope_gate_fraction(slope_pct: np.ndarray, cutoff_pct: float) -> np.ndarray:
    """``slope_gt_cutoff_frac``: a cell above the Section 7.2 cutoff is fully gated."""
    return np.where(slope_pct > cutoff_pct, 1.0, 0.0)


def _undevelopable_fraction(
    cells: pd.DataFrame,
    params: Params,
    gates: Mapping[str, Any] | None,
    crs_metric: str,
) -> np.ndarray:
    cutoff = params.value(P_SLOPE_CUTOFF_PCT)
    slope_frac = _slope_gate_fraction(cells["slope_pct"].to_numpy(dtype=float), cutoff)

    if not gates:
        # No gate geometry supplied: the ingested fraction is already a union share.
        base = cells["undevelopable_frac"].to_numpy(dtype=float)
    else:
        base = _gate_union_fraction(cells, gates, crs_metric)

    return np.clip(np.nan_to_num(base) + slope_frac, 0, 1)


# --------------------------------------------------------------------------------------
# 7.2 slope cost multiplier
# --------------------------------------------------------------------------------------


def _slope_cost_multiplier(slope_pct: np.ndarray, params: Params) -> np.ndarray:
    cutoff = params.value(P_SLOPE_CUTOFF_PCT)
    start = params.value(P_SLOPE_PENALTY_START_PCT)
    per_pct = params.value(P_SLOPE_PENALTY_PER_PCT)

    penalised = 1 + per_pct * np.maximum(0, slope_pct - start)
    return np.where(slope_pct > cutoff, np.inf, penalised)


# --------------------------------------------------------------------------------------
# 7.3 utility state, assembly feasibility, capacity, headroom
# --------------------------------------------------------------------------------------


def _utility_state(cells: pd.DataFrame) -> np.ndarray:
    """``none`` / ``water`` / ``water_sewer`` / ``water_sewer_power``, in that precedence.

    "power alone without water counts as ``none``" (Section 7.3).
    """
    water = cells["util_water"].to_numpy(dtype=float) > 0
    sewer = cells["util_sewer"].to_numpy(dtype=float) > 0
    power = cells["util_power"].to_numpy(dtype=float) > 0

    state = np.full(len(cells), STATE_NONE, dtype=object)
    state[water] = STATE_WATER
    state[water & sewer] = STATE_WATER_SEWER
    state[water & sewer & power] = STATE_WATER_SEWER_POWER
    return state


def _utility_multiplier(state: np.ndarray, params: Params) -> np.ndarray:
    gates = {name: params.value(f"{P_UTILITY_GATE}.{name}") for name in S.UTILITY_STATES}
    return np.array([gates[value] for value in state], dtype=float)


def assembly_feasibility_tiers(params: Params) -> tuple[AssemblyTier, ...]:
    """The Section 7.3 ``assembly_feasibility`` table, in the order declared in YAML."""
    rows = params.get(P_ASSEMBLY_FEASIBILITY)
    return tuple(
        AssemblyTier(
            min_parcel_sqm=float(row[P_ASSEMBLY_MIN_PARCEL_KEY]),
            value=float(row[P_LEAF_VALUE_KEY]),
        )
        for row in rows
    )


def assembly_multiplier(mean_parcel_sqm: np.ndarray, params: Params) -> np.ndarray:
    """``assembly_feasibility(mean_parcel_sqm)`` — a step function read from YAML.

    A cell with no parcel observation (``mean_parcel_sqm`` is nullable, Section 3.1) takes
    the lowest tier: unknown fragmentation is treated as the hardest to assemble.
    """
    tiers = sorted(assembly_feasibility_tiers(params), key=lambda t: t.min_parcel_sqm)
    parcel = np.asarray(mean_parcel_sqm, dtype=float)
    out = np.full(parcel.shape, tiers[0].value, dtype=float)
    for tier in tiers:
        out = np.where(parcel >= tier.min_parcel_sqm, tier.value, out)
    return out


# --------------------------------------------------------------------------------------
# 7.4 supply elasticity
# --------------------------------------------------------------------------------------


def _classify_elasticity(
    cells: pd.DataFrame,
    params: Params,
    undevelopable_frac: np.ndarray,
    headroom_sqm: np.ndarray,
) -> np.ndarray:
    builtup = cells["builtup_frac"].to_numpy(dtype=float)
    area = cells["area_sqm"].to_numpy(dtype=float)
    water = cells["util_water"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(area > 0, headroom_sqm / area, np.nan)

    dense_min = params.value(P_DENSE_CORE_BUILTUP_MIN)
    ratio_max = params.value(P_DENSE_CORE_HEADROOM_RATIO_MAX)
    undev_min = params.value(P_CONSTRAINED_UNDEVELOPABLE_MIN)
    open_max = params.value(P_OPEN_FRINGE_BUILTUP_MAX)

    dense = (builtup > dense_min) & (ratio < ratio_max)
    constrained = ~dense & (undevelopable_frac > undev_min)
    open_fringe = ~dense & ~constrained & (builtup < open_max) & (water == 0)

    out = np.full(len(cells), CLASS_TYPICAL_PERIPH, dtype=object)
    out[open_fringe] = CLASS_OPEN_FRINGE
    out[constrained] = CLASS_CONSTRAINED_PERIPH
    out[dense] = CLASS_DENSE_CORE

    # "overridden by city.elasticity_class_default where the classifier is ambiguous":
    # a cell missing any classifier input cannot be placed, so it takes the city default.
    ambiguous = (
        np.isnan(builtup)
        | np.isnan(ratio)
        | np.isnan(undevelopable_frac)
        | np.isnan(water)
    )
    if ambiguous.any():
        default = params.value(P_ELASTICITY_CLASS_DEFAULT)
        if default not in S.ELASTICITY_CLASSES:
            raise MissingParameter(
                f"{P_ELASTICITY_CLASS_DEFAULT} must be one of {S.ELASTICITY_CLASSES}, "
                f"got {default!r}"
            )
        out[ambiguous] = default
    return out


def _eps_from_class(elasticity_class: np.ndarray, params: Params) -> np.ndarray:
    values = {
        name: params.value(f"{P_ELASTICITY_CLASS}.{name}") for name in S.ELASTICITY_CLASSES
    }
    return np.array([values[name] for name in elasticity_class], dtype=float)


def _regression_design(
    undevelopable_frac: np.ndarray,
    regulatory_index: np.ndarray,
    builtup_frac: np.ndarray,
    mean_parcel_sqm: np.ndarray,
    offset: float,
) -> np.ndarray:
    """The Section 7.4 design matrix ``[1, ln(1-u+o), reg, ln(1-b+o), ln(parcel)]``."""
    return np.column_stack(
        [
            np.ones(len(undevelopable_frac)),
            np.log(1 - undevelopable_frac + offset),
            regulatory_index,
            np.log(1 - builtup_frac + offset),
            np.log(mean_parcel_sqm),
        ]
    )


def _resolve_coefficients(
    params: Params, override: Mapping[str, float] | None
) -> dict[str, float]:
    source: Mapping[str, Any]
    if override is not None:
        source = override
    else:
        node = params.get(P_REGRESSION_COEFFICIENTS)
        source = node if isinstance(node, Mapping) else {}
    missing = [
        name for name in REGRESSION_COEFFICIENT_NAMES if source.get(name) is None
    ]
    if missing:
        raise MissingParameter(
            "the Section 7.4 elasticity regression is enabled but "
            + ", ".join(f"{P_REGRESSION_COEFFICIENTS}.{name}" for name in missing)
            + " has no value; fit it with fit_elasticity_regression() and write the "
            "coefficients into config/params/supply.yaml, or leave "
            f"{P_REGRESSION_ENABLED} false to use the class assignment"
        )
    return {name: float(source[name]) for name in REGRESSION_COEFFICIENT_NAMES}


def _eps_from_regression(
    cells: pd.DataFrame,
    params: Params,
    undevelopable_frac: np.ndarray,
    coefficients: Mapping[str, float],
) -> np.ndarray:
    offset = params.value(P_REGRESSION_LOG_OFFSET)
    if COL_REGULATORY_INDEX not in cells.columns:
        raise SchemaValidationError(
            "the Section 7.4 elasticity regression needs the "
            f"{COL_REGULATORY_INDEX!r} column, which is not on the frame"
        )
    design = _regression_design(
        undevelopable_frac,
        cells[COL_REGULATORY_INDEX].to_numpy(dtype=float),
        cells["builtup_frac"].to_numpy(dtype=float),
        cells["mean_parcel_sqm"].to_numpy(dtype=float),
        offset,
    )
    beta = np.array(
        [coefficients[name] for name in REGRESSION_COEFFICIENT_NAMES], dtype=float
    )
    return np.exp(design @ beta)


def fit_elasticity_regression(zones: pd.DataFrame, params: Params) -> RegressionFit:
    """Fit ``ln eps = a0 + a1 ln(1-u+o) + a2 reg + a3 ln(1-b+o) + a4 ln(parcel)`` by OLS.

    ``zones`` needs ``dln_price``, ``dln_floorspace``, ``undevelopable_frac``,
    ``regulatory_index``, ``builtup_frac`` and ``mean_parcel_sqm`` — one row per zone with
    a ten-year history of both series (Section 7.4).  Zones whose ``|dln_price|`` falls
    below ``supply.elasticity_regression.fit.min_abs_dln_price`` are dropped for division
    instability, as are zones whose implied elasticity is not strictly positive (its log
    is undefined).  ``r2`` and ``n`` go into the run manifest.
    """
    required = (
        "dln_price",
        "dln_floorspace",
        "undevelopable_frac",
        "regulatory_index",
        "builtup_frac",
        "mean_parcel_sqm",
    )
    missing = [column for column in required if column not in zones.columns]
    if missing:
        raise SchemaValidationError(
            f"fit_elasticity_regression needs column(s) {', '.join(missing)}"
        )

    min_abs = params.value(P_REGRESSION_MIN_ABS_DLN_PRICE)
    offset = params.value(P_REGRESSION_LOG_OFFSET)

    dln_price = zones["dln_price"].to_numpy(dtype=float)
    dln_q = zones["dln_floorspace"].to_numpy(dtype=float)
    keep = np.abs(dln_price) >= min_abs
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_observed = np.where(keep, dln_q / np.where(keep, dln_price, 1), np.nan)
    keep &= np.isfinite(eps_observed) & (eps_observed > 0)

    if not keep.any():
        raise SchemaValidationError(
            "no zone survives the Section 7.4 fitting filters "
            f"(|dln_price| >= {min_abs} and a positive observed elasticity)"
        )

    design = _regression_design(
        zones["undevelopable_frac"].to_numpy(dtype=float)[keep],
        zones["regulatory_index"].to_numpy(dtype=float)[keep],
        zones["builtup_frac"].to_numpy(dtype=float)[keep],
        zones["mean_parcel_sqm"].to_numpy(dtype=float)[keep],
        offset,
    )
    target = np.log(eps_observed[keep])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)

    residual = target - design @ beta
    ss_res = float(residual @ residual)
    centred = target - target.mean()
    ss_tot = float(centred @ centred)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return RegressionFit(
        coefficients=dict(zip(REGRESSION_COEFFICIENT_NAMES, (float(b) for b in beta))),
        r2=float(r2),
        n=int(keep.sum()),
    )


# --------------------------------------------------------------------------------------
# 7.5 jobs by sector
# --------------------------------------------------------------------------------------


def _sector_poi_columns(
    params: Params, override: Mapping[str, str] | None
) -> dict[str, str]:
    if override is not None:
        return dict(override)
    try:
        node = params.get(P_SECTOR_POI_COLUMN)
    except MissingParameter:
        logger.info(
            "%s is not in the parameter tree; the Section 7.5 dasymetric surface falls "
            "back to commercial floorspace alone",
            P_SECTOR_POI_COLUMN,
        )
        return {}
    return {str(k): str(v) for k, v in dict(node).items()}


def _ward_jobs_frame(ward_jobs_2011: Any) -> pd.DataFrame:
    frame = (
        ward_jobs_2011
        if isinstance(ward_jobs_2011, pd.DataFrame)
        else pd.DataFrame.from_dict(
            {k: list(v) for k, v in dict(ward_jobs_2011).items()},
            orient="index",
            columns=list(S.SECTORS),
        )
    )
    missing = [sector for sector in S.SECTORS if sector not in frame.columns]
    if missing:
        raise SchemaValidationError(
            f"ward_jobs_2011 is missing sector column(s) {', '.join(missing)} "
            f"(Section 3.6 defines exactly {len(S.SECTORS)})"
        )
    return frame


def _estimate_jobs_by_sector(
    cells: pd.DataFrame,
    params: Params,
    ward_jobs_2011: Any,
    sector_growth: Mapping[str, float] | None,
    ward_col: str,
    poi_columns: Mapping[str, str] | None,
) -> list[list[float]]:
    """Section 7.5 steps 1-4: grow the 2011 ward totals, redistribute dasymetrically."""
    if ward_col not in cells.columns:
        raise SchemaValidationError(
            f"jobs estimation needs the ward column {ward_col!r} on the cells frame"
        )
    wards = _ward_jobs_frame(ward_jobs_2011)
    cell_wards = cells[ward_col].to_numpy()
    unknown = sorted(set(wards.index) - set(cell_wards))
    if unknown:
        raise SchemaValidationError(
            "ward_jobs_2011 names ward(s) that are not on the grid, so their employment "
            f"cannot be placed: {', '.join(map(str, unknown))}"
        )

    growth = dict(sector_growth or {})
    mapping = _sector_poi_columns(params, poi_columns)
    floorspace = cells[COL_FLOORSPACE_COM].to_numpy(dtype=float)
    positions = {
        ward: np.asarray(rows, dtype=int)
        for ward, rows in pd.Series(np.arange(len(cells)), index=cell_wards).groupby(
            level=0
        )
    }

    jobs = np.zeros((len(cells), len(S.SECTORS)), dtype=float)
    for sector_index, sector in enumerate(S.SECTORS):
        poi_column = mapping.get(sector)
        if poi_column is not None and poi_column in cells.columns:
            weight = cells[poi_column].to_numpy(dtype=float) * floorspace
        else:
            weight = floorspace.copy()
        weight = np.nan_to_num(np.maximum(weight, 0))

        for ward, grown_2011 in wards[sector].items():
            rows = positions[ward]
            total = float(grown_2011) * float(growth.get(sector, 1))
            block = weight[rows]
            block_sum = block.sum()
            share = (
                block / block_sum
                if block_sum > 0
                else np.full(len(rows), 1 / len(rows))
            )
            jobs[rows, sector_index] = total * share

    return [[float(x) for x in row] for row in jobs]


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def assemble_substrate(
    cells: pd.DataFrame,
    params: Params,
    *,
    gates: Mapping[str, Any] | None = None,
    ward_jobs_2011: Any | None = None,
    sector_growth: Mapping[str, float] | None = None,
    ward_col: str = DEFAULT_WARD_COL,
    poi_columns: Mapping[str, str] | None = None,
    use_regression: bool | None = None,
    regression_coefficients: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble Layer 0 (spec Section 7).

    Parameters
    ----------
    cells:
        An ingested ``cells`` frame.  Never mutated.
    params:
        The resolved parameter tree for the city.
    gates:
        Optional mapping of hard-gate layer name to shapely geometries in EPSG:4326
        (Section 7.1).  Unioned geometrically before their area share is taken.  When
        omitted, the ingested ``undevelopable_frac`` is used as the gate union share.
    ward_jobs_2011, sector_growth, ward_col, poi_columns:
        Section 7.5 inputs.  With no ``ward_jobs_2011`` the existing ``jobs_by_sector``
        column is passed through untouched — this layer never invents employment.
    use_regression, regression_coefficients:
        Section 7.4.  ``use_regression`` defaults to ``supply.elasticity_regression.enabled``;
        supplying ``regression_coefficients`` turns the regression on.

    Returns
    -------
    A NEW frame with the same index and row count as ``cells`` plus
    :data:`DERIVED_COLUMNS`.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in cells.columns]
    if missing:
        raise SchemaValidationError(
            f"Layer 0 needs column(s) {', '.join(missing)} on the cells frame "
            "(spec Section 3.1)"
        )

    out = cells.copy(deep=True)
    crs_metric = str(params.city_config.get(CITY_CRS_METRIC_KEY, S.GEOMETRY_CRS))

    # --- 7.1 / 7.2 -------------------------------------------------------------------
    undevelopable_frac = _undevelopable_fraction(cells, params, gates, crs_metric)
    slope_cost_mult = _slope_cost_multiplier(
        cells["slope_pct"].to_numpy(dtype=float), params
    )

    # --- 7.3 -------------------------------------------------------------------------
    utility_state = _utility_state(cells)
    gross_far_capacity = (
        cells["area_sqm"].to_numpy(dtype=float)
        * (1 - undevelopable_frac)
        * cells["permitted_far"].to_numpy(dtype=float)
    )
    capacity_sqm = np.maximum(
        0,
        gross_far_capacity
        * _utility_multiplier(utility_state, params)
        * assembly_multiplier(cells["mean_parcel_sqm"].to_numpy(dtype=float), params),
    )
    built = cells["floorspace_res_sqm"].to_numpy(dtype=float) + cells[
        "floorspace_com_sqm"
    ].to_numpy(dtype=float)
    headroom_sqm = np.maximum(0, capacity_sqm - built)

    # --- 7.4 -------------------------------------------------------------------------
    elasticity_class = _classify_elasticity(
        cells, params, undevelopable_frac, headroom_sqm
    )
    if use_regression is None:
        use_regression = bool(params.get(P_REGRESSION_ENABLED)) or (
            regression_coefficients is not None
        )
    if use_regression:
        coefficients = _resolve_coefficients(params, regression_coefficients)
        eps_supply = _eps_from_regression(
            cells, params, undevelopable_frac, coefficients
        )
    else:
        eps_supply = _eps_from_class(elasticity_class, params)

    out["undevelopable_frac"] = undevelopable_frac
    out["slope_cost_mult"] = slope_cost_mult
    out["utility_state"] = utility_state
    out["capacity_sqm"] = capacity_sqm
    out["headroom_sqm"] = headroom_sqm
    out["elasticity_class"] = elasticity_class
    out["eps_supply"] = eps_supply

    # --- 7.5 -------------------------------------------------------------------------
    if ward_jobs_2011 is not None:
        out["jobs_by_sector"] = pd.Series(
            _estimate_jobs_by_sector(
                cells, params, ward_jobs_2011, sector_growth, ward_col, poi_columns
            ),
            index=out.index,
            dtype=object,
        )
    else:
        out["jobs_by_sector"] = pd.Series(
            [[float(x) for x in row] for row in cells["jobs_by_sector"]],
            index=out.index,
            dtype=object,
        )

    return out
