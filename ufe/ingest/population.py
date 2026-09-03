"""6.4 Population — Census 2011 wards, then GHSL POP 100 m, then WorldPop 100 m.

Section 6.4 in order:

1. sources in priority order: census ward polygons where available, then GHSL, then
   WorldPop — the first one the reader can supply wins, and which one was used is recorded
   in ``provenance()`` and in the ``ingest_runs`` row;
2. rasterise / zonal-sum to cells;
3. **dasymetric refinement**: redistribute ward totals within the ward in proportion to
   ``floorspace_res_sqm``;
4. grow 2011 forward to ``base_year`` with district growth applied uniformly, then
   re-refine against the latest building vintage. "Document that this is an estimate" — so
   every grown value is flagged ``grown_from_census_<year>``.

Why **tobler** and not hand-rolled areal interpolation, quoting Section 6.4: "Hand-rolled
areal interpolation gets the extensive/intensive distinction wrong roughly every time:
population is extensive (sums), density is intensive (averages)." So the polygon->hex
transfer goes through :func:`tobler.area_weighted.area_interpolate` with ``population`` as
an *extensive* variable, in the city's ``crs_metric`` (tobler's area weights are areas, and
an area in degrees is not an area).

A note on ``households`` and ``hh_by_band``
-------------------------------------------
Both are non-nullable in ``schemas.CELLS``, and both need a household size that the
parameter tree does not supply: ``behaviour.persons_per_household_by_band`` exists but every
band is ``null``. :func:`households_from_population` therefore raises
:class:`ufe.errors.MissingParameter` rather than inventing a household size — the gap is
reported, not defaulted (see ``config/ingest.yaml: reported_parameter_gaps``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.errors import CoverageError, MissingParameter
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed, zonal

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_CENSUS_WARDS",
    "KEY_GHSL",
    "KEY_WORLDPOP",
    "POPULATION_COLUMN",
    "HOUSEHOLDS_COLUMN",
    "PERSONS_PER_HH_PATH",
    "PopulationIngester",
    "population_from_wards",
    "population_from_raster",
    "dasymetric_refine",
    "grow_to_base_year",
    "households_from_population",
    "assert_district_total",
    "source_priority",
]

KEY_CENSUS_WARDS = "census_wards"
KEY_GHSL = "ghsl_pop"
KEY_WORLDPOP = "worldpop"

POPULATION_COLUMN = "population"
HOUSEHOLDS_COLUMN = "households"

#: The parameter path Section 6.4 needs and which resolves to ``null`` today.
PERSONS_PER_HH_PATH = "behaviour.persons_per_household_by_band"

_WARD_ID = "ward_id"
_WARD_POP = "population"
_FLOORSPACE = "floorspace_res_sqm"


def source_priority(config: Any = None) -> tuple[str, ...]:
    """``census_wards`` -> ``ghsl_pop`` -> ``worldpop`` (Section 6.4), from YAML."""
    return tuple(str(s) for s in cfg("population.source_priority", config))


# --------------------------------------------------------------------------------------
# Step 2 — transfer to cells
# --------------------------------------------------------------------------------------


def population_from_wards(
    wards: gpd.GeoDataFrame,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    population_column: str = _WARD_POP,
) -> pd.Series:
    """Areal interpolation of ward population onto cells, via **tobler**.

    ``population`` is passed as an *extensive* variable, so ward totals are preserved by
    construction and a half-covered hexagon receives half the people rather than the whole
    ward's density.
    """
    from tobler.area_weighted import area_interpolate

    source = wards if wards.crs is not None else wards.set_crs(geo.GEOGRAPHIC_CRS)
    source = geo.to_metric(source[[population_column, source.geometry.name]], crs_metric)
    target = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    interpolated = area_interpolate(
        source_df=source.reset_index(drop=True),
        target_df=target.reset_index(drop=True),
        extensive_variables=[population_column],
    )
    values = np.asarray(interpolated[population_column], dtype=float)
    return pd.Series(values, index=pd.Index(cells["h3"].astype(str), name="h3"))


def population_from_raster(raster: str | Path, cells: pd.DataFrame) -> pd.Series:
    """Zonal *sum* of a population-count raster per cell, exact-area weighted."""
    stats = zonal(raster, cells, ["sum"]).set_index("h3").reindex(cells["h3"].astype(str))
    values = pd.to_numeric(stats["sum"], errors="coerce").fillna(0.0)
    return pd.Series(
        values.to_numpy(dtype=float), index=pd.Index(cells["h3"].astype(str), name="h3")
    )


# --------------------------------------------------------------------------------------
# Step 3 — dasymetric refinement
# --------------------------------------------------------------------------------------


def dasymetric_refine(
    wards: gpd.GeoDataFrame,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    population_column: str = _WARD_POP,
    weight_column: str = _FLOORSPACE,
    config: Any = None,
) -> tuple[pd.Series, np.ndarray]:
    """Redistribute each ward's total within the ward, in proportion to ``weight_column``.

    Returns ``(population, weight_missing_mask)``. A ward with no residential floorspace at
    all cannot be refined; its total is then spread by intersected area instead and every
    cell in it is flagged, which is the honest answer to "we do not know where these people
    are inside this ward".

    Each cell is assigned to the ward it overlaps most, in ``crs_metric``.
    """
    min_weight = float(cfg("population.min_ward_floorspace_sqm", config))
    source = wards if wards.crs is not None else wards.set_crs(geo.GEOGRAPHIC_CRS)
    ward_cols = [c for c in (_WARD_ID, population_column) if c in source.columns]
    source = source[ward_cols + [source.geometry.name]].reset_index(drop=True)
    if _WARD_ID not in source.columns:
        source[_WARD_ID] = [f"ward-{i}" for i in range(len(source))]
    source_m = geo.to_metric(source, crs_metric)

    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    # The building layer may not be ingested yet (Section 20.2 runs buildings and population
    # in the same tier). With no weight, every ward falls back to the area spread below and
    # every cell is flagged, rather than the run failing.
    weights = (
        np.asarray(cells[weight_column], dtype=float)
        if weight_column in cells.columns
        else np.zeros(len(cells), dtype=float)
    )
    hexes[weight_column] = weights

    pieces = gpd.overlay(hexes, source_m, how="intersection", keep_geom_type=False)
    if not len(pieces):
        zeros = pd.Series(0.0, index=pd.Index(cells["h3"].astype(str), name="h3"))
        return zeros, np.ones(len(cells), dtype=bool)
    pieces["_share_area"] = pieces.geometry.area
    # One ward per cell: the one it overlaps most.
    assignment = (
        pieces.sort_values("_share_area", ascending=False)
        .drop_duplicates("h3")
        .set_index("h3")[[_WARD_ID, population_column]]
    )

    index = pd.Index(cells["h3"].astype(str), name="h3")
    frame = pd.DataFrame(
        {
            "ward": assignment[_WARD_ID].reindex(index),
            "ward_pop": pd.to_numeric(assignment[population_column].reindex(index)),
            "weight": np.asarray(cells[weight_column], dtype=float),
            "area": pd.Series(hexes.geometry.area.to_numpy(), index=index),
        },
        index=index,
    )
    grouped = frame.groupby("ward", dropna=True)
    weight_totals = grouped["weight"].transform("sum")
    area_totals = grouped["area"].transform("sum")
    use_area = ~(weight_totals > min_weight)
    share = np.where(
        use_area,
        np.divide(
            frame["area"], area_totals, out=np.zeros(len(frame)), where=area_totals > 0
        ),
        np.divide(
            frame["weight"], weight_totals, out=np.zeros(len(frame)), where=weight_totals > 0
        ),
    )
    population = pd.Series(
        np.nan_to_num(frame["ward_pop"].to_numpy(dtype=float)) * share, index=index
    )
    unassigned = frame["ward"].isna().to_numpy()
    return population, (use_area.to_numpy() | unassigned)


# --------------------------------------------------------------------------------------
# Step 4 — grow forward
# --------------------------------------------------------------------------------------


def grow_to_base_year(
    population: pd.Series,
    *,
    from_year: int,
    to_year: int,
    annual_growth_rate: float,
) -> pd.Series:
    """Uniform district growth from ``from_year`` to ``to_year`` (Section 6.4).

    Deliberately uniform: the spec asks for district growth "applied uniformly", followed by
    a dasymetric re-refinement against the latest building vintage. This is an estimate and
    every value it produces is flagged.
    """
    years = int(to_year) - int(from_year)
    return population * (1 + float(annual_growth_rate)) ** years


def households_from_population(population: pd.Series, params: Any) -> pd.Series:
    """``households = population / persons_per_household``.

    Raises :class:`ufe.errors.MissingParameter` today: every band of
    ``behaviour.persons_per_household_by_band`` is ``null`` in ``config/params/``. Section
    0.1 rule 3 and CONTRACT.md rule 1 forbid substituting a literal, and Section 6.0's
    "never silently imputes" forbids inventing one, so this raises and the gap is reported.
    """
    try:
        node = params.get(PERSONS_PER_HH_PATH)
    except MissingParameter:
        node = None
    values = [v for v in (node or {}).values() if v is not None] if isinstance(node, Mapping) else []
    if not values:
        raise MissingParameter(
            f"{PERSONS_PER_HH_PATH} resolves to null for every band, so households cannot "
            "be derived from population (spec Section 6.4). Populate it in "
            "config/params/behaviour.yaml — this module will not substitute a default "
            "(Section 0.1 rule 3)."
        )
    persons_per_hh = float(np.mean([float(v) for v in values]))
    return population / persons_per_hh


def assert_district_total(
    population: pd.Series, district_total: float, *, config: Any = None
) -> None:
    """Section 6 ACCEPTANCE: population sums to within 5% of the district total."""
    tolerance = float(cfg("population.district_total_tolerance", config))
    total = float(np.nansum(np.asarray(population, dtype=float)))
    district_total = float(district_total)
    if district_total <= 0:
        raise CoverageError("district_total must be positive to check the Section 6 tolerance")
    error = abs(total - district_total) / district_total
    if error > tolerance:
        raise CoverageError(
            f"ingested population {total:,.0f} differs from the district total "
            f"{district_total:,.0f} by {error:.1%}, outside the "
            f"{tolerance:.0%} tolerance of the Module 2 ACCEPTANCE block"
        )


# --------------------------------------------------------------------------------------
# The ingester
# --------------------------------------------------------------------------------------


class PopulationIngester(Ingester):
    """Census wards / GHSL / WorldPop -> ``population`` (dasymetrically refined)."""

    source_id = "government_open_data"
    tier = "national"
    fills = (POPULATION_COLUMN,)
    spatial_res = "census ward polygons; 100 m where raster"
    temporal_res = "2011 census, grown to base_year"
    notes = (
        "Areal interpolation by tobler with population as an extensive variable, then "
        "dasymetric refinement against floorspace_res_sqm, then uniform district growth to "
        "base_year. Growth is an estimate and is flagged as such (Section 6.4)."
    )

    #: Set by :meth:`parse` to whichever source was actually used.
    resolved_source: str | None = None

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_CENSUS_WARDS, KEY_GHSL, KEY_WORLDPOP)

    def resolve_source(self, city: CityConfig) -> str:
        """The highest-priority source the reader can actually supply (Section 6.4)."""
        for key in source_priority(self.config):
            if self.reader.exists(key):
                return key
        raise MissingParameter(
            "no population source available; tried "
            f"{list(source_priority(self.config))} (Section 6.4)"
        )

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        key = self.resolve_source(city)
        self.resolved_source = key
        if not force and key in self._fetched:
            return self._fetched[key]
        path = self.reader.path(key, force=force)
        self._fetched[key] = path
        return path

    def parse(self, raw: Path) -> pd.DataFrame:
        """Ward polygons as a GeoDataFrame, or a one-row manifest for a raster source.

        The layer is read back through the reader rather than with ``gpd.read_file(raw)``:
        the reader is the read half of the split (it may hold the layer in memory, or
        behind a cache), and ``raw`` is the handle it issued.
        """
        if self.resolved_source == KEY_CENSUS_WARDS:
            return self.reader.vector(KEY_CENSUS_WARDS)
        return pd.DataFrame([{"path": str(raw)}])

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("PopulationIngester needs a CityConfig")
        index = cells["h3"].astype(str)
        if isinstance(df, gpd.GeoDataFrame):
            population, weight_missing = dasymetric_refine(
                df, cells, crs_metric=self.city.crs_metric, config=self.config
            )
            method = "ward_total_spread_by_area"
        else:
            population = population_from_raster(df["path"].iat[0], cells)
            weight_missing = np.ones(len(cells), dtype=bool)
            method = "gridded_population_product"

        census_year = int(cfg("population.census_year", self.config))
        growth = self.city.get("district_population_growth_rate")
        grown = np.zeros(len(cells), dtype=bool)
        if growth is not None and self.city.base_year != census_year:
            population = grow_to_base_year(
                population,
                from_year=census_year,
                to_year=self.city.base_year,
                annual_growth_rate=float(growth),
            )
            grown = np.ones(len(cells), dtype=bool)

        out = pd.DataFrame(
            {"h3": index.to_numpy(), POPULATION_COLUMN: population.reindex(index).to_numpy()}
        )
        out = mark_imputed(out, POPULATION_COLUMN, weight_missing, method)
        out = mark_imputed(
            out, POPULATION_COLUMN, grown, f"grown_from_census_{census_year}"
        )
        return out
