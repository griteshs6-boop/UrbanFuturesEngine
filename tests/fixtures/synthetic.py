"""Deterministic synthetic city, shared by every module's tests.

Import this rather than rolling your own fixture.  Everything is seeded: same ``n`` + same
``seed`` gives byte-identical frames (spec Section 0.1 rule 7).  Every generation constant
lives in ``tests/fixtures/synthetic.yaml``, so this module contains no numeric literals.

Public API
----------
``synthetic_cells(n=..., seed=...)``      -> ``cells`` frame, validates against ``schemas.CELLS``
``synthetic_cells_history(cells, seed=)`` -> ``cells_history`` frame
``synthetic_announcers(n=..., seed=...)`` -> ``announcers`` frame
``synthetic_projects(n=..., seed=..., announcers=None, cells=None)`` -> ``projects`` frame
``synthetic_project_history(projects, seed=...)`` -> ``project_history`` frame
``synthetic_city(...)``                   -> pytest fixture yielding :class:`SyntheticCity`
``SyntheticCity``                         -> dataclass with ``.cells .cells_history .projects
                                             .announcers .project_history``

Using the pytest fixture from another test module::

    from tests.fixtures.synthetic import synthetic_city  # noqa: F401  (registers the fixture)

    def test_something(synthetic_city):
        cells = synthetic_city.cells

Columns provided in ``cells`` (exactly the column set of ``ufe.store.schemas.CELLS``, in
schema order)
-----------------------------------------------------------------------------------------
Identity / geometry
    ``h3``, ``h3_res8``, ``in_city``, ``geometry`` (WKB bytes, EPSG:4326), ``lat``, ``lon``,
    ``area_sqm``
Terrain and cover
    ``elev_m``, ``slope_pct``, ``landcover``, ``builtup_frac``, ``undevelopable_frac``
Regulation
    ``zone_class``, ``permitted_far``, ``crz_class``
People and jobs
    ``population``, ``households``, ``hh_by_band`` (4 floats, Section 3.7),
    ``jobs_by_sector`` (8 floats, Section 3.6)
Stock and price
    ``floorspace_res_sqm``, ``floorspace_com_sqm``, ``price_res_inr_sqft`` (nullable),
    ``price_land_inr_sqft`` (nullable), ``rent_res_inr_sqft_mo`` (nullable),
    ``mean_parcel_sqm`` (nullable), ``parcel_count``
Utilities and distances
    ``util_water``, ``util_sewer``, ``util_power``, ``dist_cbd_m``, ``dist_coast_m``,
    ``dist_arterial_m``, ``nightlight``, ``data_conf``
Layer 0 derived (Section 7)
    ``utility_state``, ``slope_cost_mult``, ``capacity_sqm``, ``headroom_sqm``,
    ``elasticity_class``, ``eps_supply``, ``regulatory_index``
Layer 1 accessibility (Section 8.5)
    ``lnA``, ``lnA_work``, ``lnA_retail``, ``lnA_education``, ``lnA_health``, ``jobs_30min``,
    ``jobs_45min``, ``jobs_60min``, ``station_weight``
Opportunity inputs (Section 8.2)
    ``retail_poi_count``, ``education_poi_count``, ``health_poi_count``, ``school_seats``,
    ``hospital_beds``
Allocation utility terms (Sections 9, 12.3)
    ``amenity``, ``disamenity``, ``alpha_res``
Supply state (Section 11)
    ``inventory_months``, ``hist_absorption_sqm``
Backtest baseline B4 covariate (Section 19.3)
    ``dist_existing_builtup_m``

The derived columns are filled with *plausible, internally consistent* values so downstream
tests have something to chew on; they are not the output of the real layers and must be
recomputed by the layer under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import h3
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, Point, Polygon

from ufe.store import schemas as S

__all__ = [
    "CONFIG_PATH",
    "CONFIG",
    "SyntheticCity",
    "synthetic_cells",
    "synthetic_cells_history",
    "synthetic_projects",
    "synthetic_announcers",
    "synthetic_project_history",
    "build_city",
    "synthetic_city",
]

CONFIG_PATH = Path(__file__).with_suffix(".yaml")
CONFIG: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())

_ZERO = 0
_ONE = 1


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _uniform(rng: np.random.Generator, spec: dict[str, float], n: int) -> np.ndarray:
    return rng.uniform(float(spec["min"]), float(spec["max"]), size=n)


def _randint(rng: np.random.Generator, spec: dict[str, int], n: int) -> np.ndarray:
    return rng.integers(int(spec["min"]), int(spec["max"]) + _ONE, size=n)


def _bernoulli(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    return (rng.random(n) < float(p)).astype(np.int64)


def _choice(rng: np.random.Generator, values: Sequence[str], weights, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return rng.choice(np.asarray(values, dtype=object), size=n, p=w / w.sum())


def _mask_to_nan(values: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    out = values.astype(float).copy()
    out[rng.random(len(out)) < float(p)] = np.nan
    return out


def _grid_cells(n: int) -> list[str]:
    """The first ``n`` h3 cells of the fixture box, in sorted (deterministic) order."""
    g = CONFIG["grid"]
    lat, lon = float(g["center_lat"]), float(g["center_lon"])
    dlat, dlon = float(g["half_span_lat_deg"]), float(g["half_span_lon_deg"])
    poly = h3.LatLngPoly(
        [
            (lat - dlat, lon - dlon),
            (lat - dlat, lon + dlon),
            (lat + dlat, lon + dlon),
            (lat + dlat, lon - dlon),
        ]
    )
    cells = sorted(h3.h3shape_to_cells(poly, int(g["resolution"])))
    if n > len(cells):
        raise ValueError(
            f"synthetic grid holds {len(cells)} cells at res {g['resolution']}; asked for {n}. "
            "Widen grid.half_span_*_deg in synthetic.yaml."
        )
    return cells[:n]


def _cell_wkb(cell: str) -> bytes:
    """Hexagon boundary as WKB in EPSG:4326 (x=lon, y=lat)."""
    ring = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
    return Polygon(ring).wkb


# --------------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------------


def synthetic_cells(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """A deterministic ``cells`` frame with every column of ``schemas.CELLS`` populated."""
    g = CONFIG["grid"]
    c = CONFIG["cells"]
    n = int(g["default_n"]) if n is None else int(n)
    seed = int(g["default_seed"]) if seed is None else int(seed)
    rng = np.random.default_rng(seed)

    cells = _grid_cells(n)
    latlng = np.array([h3.cell_to_latlng(x) for x in cells], dtype=float)
    lat, lon = latlng[:, _ZERO], latlng[:, _ONE]
    parent_res = int(g["parent_resolution"])

    radius = float(g["in_city_radius_deg"])
    d_centre = np.hypot(lat - float(g["center_lat"]), lon - float(g["center_lon"]))
    in_city = d_centre <= radius

    area_sqm = _uniform(rng, c["area_sqm"], n)
    builtup_frac = _uniform(rng, c["builtup_frac"], n)
    undevelopable_frac = _uniform(rng, c["undevelopable_frac"], n)
    permitted_far = _uniform(rng, c["permitted_far"], n)

    population = builtup_frac * _uniform(rng, c["population_per_builtup"], n)
    pph = float(c["persons_per_household"])
    households = population / pph

    band_shares = rng.dirichlet(np.asarray(c["band_alpha"], dtype=float), size=n)
    hh_by_band = [list(map(float, row)) for row in band_shares * households[:, None]]

    jobs_total = _uniform(rng, c["jobs_total"], n) * builtup_frac
    sector_shares = rng.dirichlet(np.asarray(c["sector_alpha"], dtype=float), size=n)
    jobs_by_sector = [list(map(float, row)) for row in sector_shares * jobs_total[:, None]]

    floorspace_res_sqm = population * _uniform(rng, c["floorspace_res_per_person_sqm"], n)
    com_share = _uniform(rng, c["floorspace_com_share"], n)
    floorspace_com_sqm = floorspace_res_sqm * com_share

    price_res = _mask_to_nan(
        _uniform(rng, c["price_res_inr_sqft"], n), rng, c["price_missing_p"]
    )
    price_land = price_res * _uniform(rng, c["price_land_share"], n)
    rent = _mask_to_nan(
        price_res * _uniform(rng, c["rent_yield_annual"], n) / float(c["months_per_year"]),
        rng,
        c["rent_missing_p"],
    )

    util_water = _bernoulli(rng, c["util_water_p"], n)
    util_sewer = _bernoulli(rng, c["util_sewer_p"], n) * util_water
    util_power = _bernoulli(rng, c["util_power_p"], n)

    # Section 7.3: none / water / water_sewer / water_sewer_power, in that precedence.
    utility_state = np.where(
        util_water == _ZERO,
        S.UTILITY_STATES[_ZERO],
        np.where(
            util_sewer == _ZERO,
            S.UTILITY_STATES[_ONE],
            np.where(util_power == _ZERO, S.UTILITY_STATES[2], S.UTILITY_STATES[3]),
        ),
    )

    capacity_sqm = (
        area_sqm
        * (_ONE - undevelopable_frac)
        * permitted_far
        * _uniform(rng, c["capacity_utilisation"], n)
    )
    built = floorspace_res_sqm + floorspace_com_sqm
    capacity_sqm = np.maximum(capacity_sqm, built)
    headroom_sqm = np.maximum(_ZERO, capacity_sqm - built)

    elasticity_class = _choice(
        rng,
        S.ELASTICITY_CLASSES,
        np.ones(len(S.ELASTICITY_CLASSES)),
        n,
    )

    lnA = _uniform(rng, c["lnA"], n)
    spread = float(c["lnA_spread"])
    jobs_30 = _uniform(rng, c["jobs_30min"], n)

    retail_poi = _randint(rng, c["retail_poi_count"], n).astype(float)
    edu_poi = _randint(rng, c["education_poi_count"], n).astype(float)
    health_poi = _randint(rng, c["health_poi_count"], n).astype(float)

    frame = pd.DataFrame(
        {
            "h3": cells,
            "h3_res8": [h3.cell_to_parent(x, parent_res) for x in cells],
            "in_city": in_city,
            "geometry": [_cell_wkb(x) for x in cells],
            "lat": lat,
            "lon": lon,
            "area_sqm": area_sqm,
            "elev_m": _uniform(rng, c["elev_m"], n),
            "slope_pct": _uniform(rng, c["slope_pct"], n),
            "landcover": _choice(rng, c["landcover_classes"], c["landcover_weights"], n),
            "builtup_frac": builtup_frac,
            "undevelopable_frac": undevelopable_frac,
            "zone_class": _choice(rng, c["zone_classes"], c["zone_weights"], n),
            "permitted_far": permitted_far,
            "crz_class": _choice(rng, c["crz_classes"], c["crz_weights"], n),
            "population": population,
            "households": households,
            "hh_by_band": hh_by_band,
            "jobs_by_sector": jobs_by_sector,
            "floorspace_res_sqm": floorspace_res_sqm,
            "floorspace_com_sqm": floorspace_com_sqm,
            "price_res_inr_sqft": price_res,
            "price_land_inr_sqft": price_land,
            "rent_res_inr_sqft_mo": rent,
            "mean_parcel_sqm": _uniform(rng, c["mean_parcel_sqm"], n),
            "parcel_count": _randint(rng, c["parcel_count"], n),
            "util_water": util_water,
            "util_sewer": util_sewer,
            "util_power": util_power,
            "dist_cbd_m": _uniform(rng, c["dist_cbd_m"], n),
            "dist_coast_m": _uniform(rng, c["dist_coast_m"], n),
            "dist_arterial_m": _uniform(rng, c["dist_arterial_m"], n),
            "nightlight": _uniform(rng, c["nightlight"], n),
            "data_conf": _uniform(rng, c["data_conf"], n),
            "utility_state": utility_state,
            "slope_cost_mult": _uniform(rng, c["slope_cost_mult"], n),
            "capacity_sqm": capacity_sqm,
            "headroom_sqm": headroom_sqm,
            "elasticity_class": elasticity_class,
            "eps_supply": _uniform(rng, c["eps_supply"], n),
            "regulatory_index": _uniform(rng, c["regulatory_index"], n),
            "lnA": lnA,
            "lnA_work": lnA + rng.normal(_ZERO, spread, n),
            "lnA_retail": lnA + rng.normal(_ZERO, spread, n),
            "lnA_education": lnA + rng.normal(_ZERO, spread, n),
            "lnA_health": lnA + rng.normal(_ZERO, spread, n),
            "jobs_30min": jobs_30,
            "jobs_45min": jobs_30 * float(c["jobs_45min_mult"]),
            "jobs_60min": jobs_30 * float(c["jobs_60min_mult"]),
            "station_weight": _uniform(rng, c["station_weight"], n),
            "retail_poi_count": retail_poi,
            "education_poi_count": edu_poi,
            "health_poi_count": health_poi,
            "school_seats": edu_poi * float(c["seats_per_education_poi"]),
            "hospital_beds": health_poi * float(c["beds_per_health_poi"]),
            "amenity": _uniform(rng, c["amenity"], n),
            "disamenity": _uniform(rng, c["disamenity"], n),
            "alpha_res": _uniform(rng, c["alpha_res"], n),
            "inventory_months": _uniform(rng, c["inventory_months"], n),
            "hist_absorption_sqm": _uniform(rng, c["hist_absorption_sqm"], n),
            "dist_existing_builtup_m": _uniform(rng, c["dist_existing_builtup_m"], n),
            # Layer 2 shock resolution (Section 9). The synthetic city is a base state
            # with no resolved project shocks, so every one of these is the identity
            # (zero / not-capped) rather than a random draw -- deliberately, so a test
            # that expects "no shocks yet" reads the fixture correctly.
            "shock_field_residential": np.full(n, _ZERO, dtype=float),
            "shock_field_commercial": np.full(n, _ZERO, dtype=float),
            "shock_field_office": np.full(n, _ZERO, dtype=float),
            "shock_field_cap_hit": np.zeros(n, dtype=bool),
            "shock_jobs_permanent": np.full(n, _ZERO, dtype=float),
            "shock_jobs_construction": np.full(n, _ZERO, dtype=float),
            "shock_jobs_by_sector": [
                [float(_ZERO)] * len(S.SECTORS) for _ in range(n)
            ],
            "shock_effective_households": np.full(n, _ZERO, dtype=float),
            "shock_households_by_band": [
                [float(_ZERO)] * len(S.INCOME_BANDS) for _ in range(n)
            ],
            "shock_dormitory_workers": np.full(n, _ZERO, dtype=float),
            "shock_floorspace_demand_sqm": np.full(n, _ZERO, dtype=float),
            "shock_delta_capacity_sqm": np.full(n, _ZERO, dtype=float),
            "shock_delta_floorspace_sqm": np.full(n, _ZERO, dtype=float),
        }
    )
    return frame[S.column_order("cells")]


# --------------------------------------------------------------------------------------
# cells_history
# --------------------------------------------------------------------------------------


def synthetic_cells_history(
    cells: pd.DataFrame | None = None, seed: int | None = None
) -> pd.DataFrame:
    """A ``cells_history`` panel back-cast from ``cells`` by per-cell CAGRs (Section 3.2)."""
    h = CONFIG["history"]
    cells = synthetic_cells() if cells is None else cells
    seed = int(CONFIG["grid"]["default_seed"]) if seed is None else int(seed)
    rng = np.random.default_rng(seed)

    base_year = int(h["base_year"])
    years = list(range(base_year - int(h["span_years"]), base_year + _ONE))
    n = len(cells)

    g_built = _uniform(rng, h["builtup_cagr"], n)
    g_price = _uniform(rng, h["price_cagr"], n)
    g_pop = _uniform(rng, h["population_cagr"], n)
    g_night = _uniform(rng, h["nightlight_cagr"], n)

    rows = []
    for year in years:
        back = base_year - year
        price = cells["price_res_inr_sqft"].to_numpy(dtype=float) / (_ONE + g_price) ** back
        price = _mask_to_nan(price, rng, h["price_missing_p"])
        rows.append(
            pd.DataFrame(
                {
                    "h3": cells["h3"].to_numpy(),
                    "year": np.full(n, year, dtype=np.int64),
                    "builtup_frac": np.clip(
                        cells["builtup_frac"].to_numpy(dtype=float) / (_ONE + g_built) ** back,
                        _ZERO,
                        _ONE,
                    ),
                    "nightlight": cells["nightlight"].to_numpy(dtype=float)
                    / (_ONE + g_night) ** back,
                    "population": cells["population"].to_numpy(dtype=float)
                    / (_ONE + g_pop) ** back,
                    "price_res_inr_sqft": price,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    return out[S.column_order("cells_history")]


# --------------------------------------------------------------------------------------
# announcers
# --------------------------------------------------------------------------------------


def synthetic_announcers(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """A deterministic ``announcers`` frame (Section 3.4)."""
    a = CONFIG["announcers"]
    n = int(a["default_n"]) if n is None else int(n)
    seed = int(a["default_seed"]) if seed is None else int(seed)
    rng = np.random.default_rng(seed)

    ids = [f"ann-{i:03d}" for i in range(n)]
    delivery_ratio = _uniform(rng, a["delivery_ratio"], n)
    announced = _uniform(rng, a["announced_capex_10y_inr_cr"], n)
    n_alias = _randint(rng, a["n_aliases"], n)
    n_src = _randint(rng, a["n_record_sources"], n)
    has_parent = rng.random(n) < float(a["parent_p"])

    asof = datetime(
        int(a["record_asof_year"]), int(a["record_asof_month"]), int(a["record_asof_day"])
    )

    frame = pd.DataFrame(
        {
            "announcer_id": ids,
            "name": [f"Synthetic Announcer {i}" for i in range(n)],
            "aliases": [[f"{ids[i]}-alias-{k}" for k in range(int(n_alias[i]))] for i in range(n)],
            "parent_id": [
                ids[_ZERO] if (has_parent[i] and i > _ZERO) else None for i in range(n)
            ],
            "is_listed": rng.random(n) < float(a["listed_p"]),
            "announced_capex_10y_inr_cr": announced,
            "deployed_capex_10y_inr_cr": announced * delivery_ratio,
            "delivery_ratio": delivery_ratio,
            "median_slip_months": _uniform(rng, a["median_slip_months"], n),
            "mean_annual_capex_3y_inr_cr": _uniform(rng, a["mean_annual_capex_3y_inr_cr"], n),
            "net_debt_ebitda": _mask_to_nan(
                _uniform(rng, a["net_debt_ebitda"], n), rng, a["net_debt_missing_p"]
            ),
            "record_sources": [
                [f"https://example.invalid/{ids[i]}/source/{k}" for k in range(int(n_src[i]))]
                or [f"https://example.invalid/{ids[i]}"]
                for i in range(n)
            ],
            "record_asof": [asof] * n,
        }
    )
    return frame[S.column_order("announcers")]


# --------------------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------------------


def _project_geometry(
    rng: np.random.Generator, geom_type: str, lat: float, lon: float
) -> str:
    p = CONFIG["projects"]
    if geom_type == "point":
        return Point(lon, lat).wkt
    if geom_type == "polygon":
        d = float(p["polygon_half_side_deg"])
        return Polygon(
            [
                (lon - d, lat - d),
                (lon + d, lat - d),
                (lon + d, lat + d),
                (lon - d, lat + d),
            ]
        ).wkt
    span = float(p["linestring_span_deg"])
    return LineString([(lon, lat), (lon + span, lat + span)]).wkt


def synthetic_projects(
    n: int | None = None,
    seed: int | None = None,
    announcers: pd.DataFrame | None = None,
    cells: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """A deterministic ``projects`` frame (Section 3.3), sited on the synthetic grid."""
    p = CONFIG["projects"]
    n = int(p["default_n"]) if n is None else int(n)
    seed = int(p["default_seed"]) if seed is None else int(seed)
    rng = np.random.default_rng(seed)

    cells = synthetic_cells() if cells is None else cells
    announcers = synthetic_announcers() if announcers is None else announcers
    announcer_ids = list(announcers["announcer_id"])

    site_idx = rng.integers(_ZERO, len(cells), size=n)
    lats = cells["lat"].to_numpy(dtype=float)[site_idx]
    lons = cells["lon"].to_numpy(dtype=float)[site_idx]

    is_public = rng.random(n) < float(p["public_p"])
    geom_types = _choice(rng, p["geom_types"], p["geom_type_weights"], n)

    announced_year = _randint(rng, p["announced_year"], n)
    announced_month = _randint(rng, p["announced_month"], n)
    announced_day = _randint(rng, p["announced_day"], n)
    duration = _randint(rng, p["duration_years"], n)
    announced = [
        datetime(int(announced_year[i]), int(announced_month[i]), int(announced_day[i]))
        for i in range(n)
    ]
    completion = [
        datetime(
            int(announced_year[i]) + int(duration[i]),
            int(announced_month[i]),
            int(announced_day[i]),
        )
        for i in range(n)
    ]
    stage_asof = [
        announced[i] + timedelta(days=int(x))
        for i, x in enumerate(_randint(rng, p["stage_asof_lag_days"], n))
    ]
    physical_asof_raw = [
        announced[i] + timedelta(days=int(x))
        for i, x in enumerate(_randint(rng, p["physical_asof_lag_days"], n))
    ]
    physical_missing = rng.random(n) < float(p["physical_missing_p"])

    n_mod = _randint(rng, p["n_modifiers"], n)
    n_url = _randint(rng, p["n_source_urls"], n)
    human = rng.random(n) < float(p["human_extracted_p"])
    verified = rng.random(n) < float(p["verified_p"])
    modifier_keys = list(p["modifier_keys"])

    equal = lambda seq: np.ones(len(seq))  # noqa: E731 - uniform categorical weights

    frame = pd.DataFrame(
        {
            "project_id": [f"proj-{i:03d}" for i in range(n)],
            "name": [f"Synthetic Project {i}" for i in range(n)],
            "archetype": _choice(rng, p["archetypes"], equal(p["archetypes"]), n),
            "geom_type": geom_types,
            "geom": [
                _project_geometry(rng, str(geom_types[i]), float(lats[i]), float(lons[i]))
                for i in range(n)
            ],
            "announcer_id": [
                None if is_public[i] else announcer_ids[int(site_idx[i]) % len(announcer_ids)]
                for i in range(n)
            ],
            "is_public": is_public,
            "scale_value": _uniform(rng, p["scale_value"], n),
            "scale_unit": _choice(rng, p["scale_units"], equal(p["scale_units"]), n),
            "capex_inr_cr": _mask_to_nan(
                _uniform(rng, p["capex_inr_cr"], n), rng, p["capex_missing_p"]
            ),
            "stated_jobs": _mask_to_nan(
                _uniform(rng, p["stated_jobs"], n), rng, p["jobs_missing_p"]
            ),
            "median_wage_inr_mo": _mask_to_nan(
                _uniform(rng, p["median_wage_inr_mo"], n), rng, p["wage_missing_p"]
            ),
            "announced_date": announced,
            "stated_completion": completion,
            "stage": _choice(rng, S.PROJECT_STAGES, equal(S.PROJECT_STAGES), n),
            "stage_asof": stage_asof,
            "commitment_form": [
                None
                if is_public[i]
                else str(_choice(rng, S.COMMITMENT_FORMS, equal(S.COMMITMENT_FORMS), _ONE)[_ZERO])
                for i in range(n)
            ],
            "land_possession_pct": _mask_to_nan(
                _uniform(rng, p["land_possession_pct"], n), rng, p["land_possession_missing_p"]
            ),
            "funding_source": _choice(rng, S.FUNDING_SOURCES, equal(S.FUNDING_SOURCES), n),
            "modifiers": [
                list(rng.choice(modifier_keys, size=int(n_mod[i]), replace=False))
                for i in range(n)
            ],
            "physical_state": [
                None
                if physical_missing[i]
                else str(_choice(rng, S.PHYSICAL_STATES, equal(S.PHYSICAL_STATES), _ONE)[_ZERO])
                for i in range(n)
            ],
            "physical_asof": [
                pd.NaT if physical_missing[i] else physical_asof_raw[i] for i in range(n)
            ],
            "source_urls": [
                [f"https://example.invalid/proj-{i:03d}/{k}" for k in range(max(_ONE, int(n_url[i])))]
                for i in range(n)
            ],
            "extracted_by": [
                "human" if human[i] else f"ai:{p['ai_prompt_version']}" for i in range(n)
            ],
            "verified_by": [f"analyst-{i % len(announcer_ids)}" if verified[i] else None
                            for i in range(n)],
            "first_seen": [
                announced[i] + timedelta(days=int(x))
                for i, x in enumerate(_randint(rng, p["first_seen_lag_days"], n))
            ],
            "last_updated": [
                stage_asof[i] + timedelta(days=int(x))
                for i, x in enumerate(_randint(rng, p["last_updated_lag_days"], n))
            ],
        }
    )
    return frame[S.column_order("projects")]


# --------------------------------------------------------------------------------------
# project_history
# --------------------------------------------------------------------------------------


def synthetic_project_history(
    projects: pd.DataFrame | None = None, seed: int | None = None
) -> pd.DataFrame:
    """One append-only ``stage`` transition per project (Section 3.5)."""
    projects = synthetic_projects() if projects is None else projects
    seed = int(CONFIG["projects"]["default_seed"]) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    stages = list(S.PROJECT_STAGES)

    rows = []
    for _, row in projects.iterrows():
        idx = stages.index(row["stage"])
        previous = stages[max(_ZERO, idx - _ONE)]
        rows.append(
            {
                "project_id": row["project_id"],
                "field": "stage",
                "old_value": previous,
                "new_value": row["stage"],
                "changed_at": row["stage_asof"],
                "source_url": row["source_urls"][_ZERO],
                "changed_by": row["verified_by"] or "ingest",
            }
        )
    _ = rng  # deterministic; kept for signature symmetry with the other builders
    return pd.DataFrame(rows)[S.column_order("project_history")]


# --------------------------------------------------------------------------------------
# the whole city
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticCity:
    """All five core frames, mutually consistent (FKs resolve, sites lie on the grid)."""

    cells: pd.DataFrame
    cells_history: pd.DataFrame
    announcers: pd.DataFrame
    projects: pd.DataFrame
    project_history: pd.DataFrame


def build_city(n_cells: int | None = None, seed: int | None = None) -> SyntheticCity:
    """Build the whole synthetic city deterministically."""
    cells = synthetic_cells(n=n_cells, seed=seed)
    announcers = synthetic_announcers(seed=seed)
    projects = synthetic_projects(seed=seed, announcers=announcers, cells=cells)
    return SyntheticCity(
        cells=cells,
        cells_history=synthetic_cells_history(cells, seed=seed),
        announcers=announcers,
        projects=projects,
        project_history=synthetic_project_history(projects, seed=seed),
    )


@pytest.fixture(scope="session")
def synthetic_city() -> SyntheticCity:
    """Session-scoped synthetic city.  Import this name to use the fixture elsewhere."""
    return build_city()
