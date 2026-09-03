"""Tests for `ufe.layers.l1_accessibility` — Module 4, Layer 1 (spec Section 8).

Written before the implementation (spec Section 0.1 rule 2).  Every ACCEPTANCE bullet of
Section 8 appears here marked ``acceptance``, plus the Section 21 exclusive-band guard.

The maths is checked against hand calculations on a five-cell grid with a single job
cluster, not with smoke tests.  Every coefficient in the expected value is read back out of
``config/params/accessibility.yaml`` through ``Params``.
"""

from __future__ import annotations

import builtins
import math
import socket

import numpy as np
import pandas as pd
import pytest

from ufe.layers import routing as R
from ufe.layers.l1_accessibility import (
    OUTPUT_COLUMNS,
    apply_accessibility,
    opportunities,
    station_decay_weight,
)
from ufe.params import load_params
from ufe.store import schemas as S
from tests.fixtures.synthetic import synthetic_cells

CITY = "vizag"
ZERO, ONE = 0, 1


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def five_cells() -> pd.DataFrame:
    """Five cells with five *distinct* res-8 parents, one job cluster, nothing else."""
    cells = synthetic_cells(n=300)
    cells = cells[cells["in_city"]]  # Section 5.2 restricts origins to in-city cells
    picked = cells.drop_duplicates(subset="h3_res8").head(5).reset_index(drop=True)
    n = len(picked)
    assert n == 5 and picked["h3_res8"].nunique() == n

    out = picked.copy()
    zeros = [[float(ZERO)] * len(S.SECTORS) for _ in range(n)]
    out["jobs_by_sector"] = zeros
    # one job cluster, entirely in the first cell
    cluster = [float(ZERO)] * len(S.SECTORS)
    cluster[ZERO] = 1000.0
    out.at[ZERO, "jobs_by_sector"] = cluster
    # silence every non-work purpose so lnA reduces to the work term alone
    for col in (
        "floorspace_com_sqm",
        "retail_poi_count",
        "education_poi_count",
        "health_poi_count",
        "school_seats",
        "hospital_beds",
    ):
        out[col] = float(ZERO)
    return out


def _matrixset(cells: pd.DataFrame, minutes: dict[str, np.ndarray], **kw) -> R.MatrixSet:
    return R.MatrixSet(
        origins=tuple(cells["h3"]),
        destinations=tuple(cells["h3_res8"]),
        minutes={m: np.asarray(v, dtype=np.float32) for m, v in minutes.items()},
        network_state=kw.pop("network_state", ""),
        station_walk_dist_m=kw.pop("station_walk_dist_m", None),
        station_feeder=kw.pop("station_feeder", ()),
    )


def _ramp(n: int, step: float) -> np.ndarray:
    """t[i, j] = |i - j| * step minutes."""
    idx = np.arange(n, dtype=float)
    return np.abs(idx[:, None] - idx[None, :]) * step


def _hand_A(params, minutes: dict[str, np.ndarray], opp: np.ndarray, purpose: str) -> np.ndarray:
    """Section 8.1, spelled out longhand for the test to compare against."""
    total = np.zeros(next(iter(minutes.values())).shape[ZERO], dtype=float)
    for mode, t in minutes.items():
        share = float(params.value(f"accessibility.modes.{mode}.share"))
        beta = float(params.value(f"accessibility.decay_beta.{purpose}.{mode}"))
        total = total + share * (np.exp(-beta * np.asarray(t, dtype=float)) @ opp)
    return total


# --------------------------------------------------------------------------------------
# Section 8.2 — opportunity definitions
# --------------------------------------------------------------------------------------


def test_work_opportunity_is_the_sum_of_jobs_by_sector(params, five_cells):
    opp = opportunities(five_cells, params, tuple(five_cells["h3_res8"]))
    assert opp["work"][ZERO] == pytest.approx(sum(five_cells["jobs_by_sector"][ZERO]))
    assert opp["work"][ONE:].sum() == pytest.approx(ZERO)


def test_opportunity_fallbacks_use_the_yaml_per_poi_constants(params):
    cells = synthetic_cells(n=300).drop_duplicates(subset="h3_res8").head(3).reset_index(drop=True)
    cells = cells.copy()
    cells["floorspace_com_sqm"] = float(ZERO)
    cells["school_seats"] = float(ZERO)
    cells["hospital_beds"] = float(ZERO)
    cells["retail_poi_count"] = float(ONE)
    cells["education_poi_count"] = float(ONE)
    cells["health_poi_count"] = float(ONE)

    opp = opportunities(cells, params, tuple(cells["h3_res8"]))
    assert opp["retail"][ZERO] == pytest.approx(
        params.value("accessibility.opportunity_imputation.retail_sqm_per_poi")
    )
    assert opp["education"][ZERO] == pytest.approx(
        params.value("accessibility.opportunity_imputation.education_seats_per_poi")
    )
    assert opp["health"][ZERO] == pytest.approx(
        params.value("accessibility.opportunity_imputation.health_beds_per_poi")
    )


def test_measured_opportunity_beats_the_poi_fallback(params):
    cells = synthetic_cells(n=300).drop_duplicates(subset="h3_res8").head(2).reset_index(drop=True)
    cells = cells.copy()
    cells["education_poi_count"] = float(ONE)
    cells["school_seats"] = 12345.0
    opp = opportunities(cells, params, tuple(cells["h3_res8"]))
    assert opp["education"][ZERO] == pytest.approx(12345.0)


def test_opportunities_aggregate_to_the_destination_resolution(params):
    """Destinations are res-8; several res-9 origins share one parent (Section 5.2)."""
    cells = synthetic_cells(n=60).copy()
    dests = tuple(sorted(set(cells["h3_res8"])))
    opp = opportunities(cells, params, dests)
    assert len(opp["work"]) == len(dests)
    total_jobs = sum(sum(v) for v in cells["jobs_by_sector"])
    assert opp["work"].sum() == pytest.approx(total_jobs)


# --------------------------------------------------------------------------------------
# Section 8.1 — the core formula
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_A_matches_a_hand_calculation_on_a_five_cell_grid(params, five_cells):
    """Section 8 ACCEPTANCE bullet 1: A_i matches a hand calculation to 1e-6."""
    step = 5.0
    minutes = {"car": _ramp(len(five_cells), step), "walk": _ramp(len(five_cells), step * 3)}
    ms = _matrixset(five_cells, minutes)

    out = apply_accessibility(five_cells, params, ms)

    opp = np.zeros(len(five_cells))
    opp[ZERO] = sum(five_cells["jobs_by_sector"][ZERO])
    expected_A = _hand_A(params, minutes, opp, "work")

    got_A = np.exp(out["lnA_work"].to_numpy()) - ONE
    assert got_A == pytest.approx(expected_A, abs=1e-6, rel=1e-6)


@pytest.mark.acceptance
def test_lnA_combines_purposes_in_logs_with_the_yaml_weights(params, five_cells):
    step = 5.0
    minutes = {"car": _ramp(len(five_cells), step)}
    out = apply_accessibility(five_cells, params, _matrixset(five_cells, minutes))

    w_work = float(params.value("accessibility.purposes.work.weight"))
    # the other three purposes have zero opportunity, so ln(0 + 1) = 0 and drop out
    assert out["lnA"].to_numpy() == pytest.approx(w_work * out["lnA_work"].to_numpy())
    for purpose in ("retail", "education", "health"):
        assert out[f"lnA_{purpose}"].to_numpy() == pytest.approx(ZERO)


def test_the_plus_one_guards_the_log_at_zero_opportunity(params, five_cells):
    cells = five_cells.copy()
    cells["jobs_by_sector"] = [[float(ZERO)] * len(S.SECTORS) for _ in range(len(cells))]
    minutes = {"car": _ramp(len(cells), 5.0)}
    out = apply_accessibility(cells, params, _matrixset(cells, minutes))
    assert np.isfinite(out["lnA"].to_numpy()).all()
    assert out["lnA"].to_numpy() == pytest.approx(ZERO)


@pytest.mark.acceptance
def test_lnA_decreases_monotonically_with_distance_from_the_job_centre(params, five_cells):
    """Section 8 ACCEPTANCE bullet 2."""
    step = 7.0
    minutes = {"car": _ramp(len(five_cells), step), "walk": _ramp(len(five_cells), step * 3)}
    out = apply_accessibility(five_cells, params, _matrixset(five_cells, minutes))
    lna = out["lnA"].to_numpy()
    assert (np.diff(lna) < ZERO).all()


def test_unreachable_destinations_contribute_nothing(params, five_cells):
    n = len(five_cells)
    finite = _ramp(n, 5.0)
    with_inf = finite.copy()
    with_inf[:, ZERO] = np.inf
    a = apply_accessibility(five_cells, params, _matrixset(five_cells, {"car": with_inf}))
    # the only job cluster is destination 0, so making it unreachable zeroes A
    assert a["lnA_work"].to_numpy() == pytest.approx(ZERO)


def test_a_mode_with_no_matrix_contributes_zero_rather_than_reweighting(params, five_cells):
    """Documented choice: absent modes contribute 0 so that adding one strictly helps."""
    n = len(five_cells)
    car_only = apply_accessibility(
        five_cells, params, _matrixset(five_cells, {"car": _ramp(n, 5.0)})
    )
    both = apply_accessibility(
        five_cells,
        params,
        _matrixset(five_cells, {"car": _ramp(n, 5.0), "walk": _ramp(n, 5.0)}),
    )
    assert (both["lnA_work"].to_numpy() >= car_only["lnA_work"].to_numpy()).all()
    assert both["lnA_work"][ZERO] > car_only["lnA_work"][ZERO]


# --------------------------------------------------------------------------------------
# Section 8 ACCEPTANCE — metro
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_adding_a_metro_line_raises_lnA_in_catchment_and_leaves_far_cells_unchanged(
    params, five_cells
):
    """Section 8 ACCEPTANCE bullet 3."""
    n = len(five_cells)
    base = {"car": _ramp(n, 5.0)}
    # metro reaches the job cluster (destination 0) from cells 0 and 1 only; the rest are
    # beyond the walk-access catchment and stay unreachable.
    metro = np.full((n, n), np.inf, dtype=float)
    metro[ZERO, ZERO] = 2.0
    metro[ONE, ZERO] = 8.0

    before = apply_accessibility(five_cells, params, _matrixset(five_cells, base))
    after = apply_accessibility(
        five_cells, params, _matrixset(five_cells, {**base, "metro": metro})
    )

    b, a = before["lnA"].to_numpy(), after["lnA"].to_numpy()
    assert (a[: ONE + ONE] > b[: ONE + ONE]).all()          # in catchment: strictly up
    assert a[ONE + ONE :] == pytest.approx(b[ONE + ONE :])  # out of catchment: unchanged


# --------------------------------------------------------------------------------------
# Section 8 ACCEPTANCE — congestion
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_doubling_corridor_builtup_raises_travel_time_and_lowers_lnA(params, five_cells):
    """Section 8 ACCEPTANCE bullet 4, through the real precompute path."""
    low = five_cells.copy()
    low["builtup_frac"] = ONE / 4
    high = five_cells.copy()
    high["builtup_frac"] = ONE / 2

    backend = R.HaversineBackend(params)
    ms_low = R.precompute_matrices(low, params, backend, network_state=frozenset())
    ms_high = R.precompute_matrices(high, params, backend, network_state=frozenset())

    for mode, t_low in ms_low.minutes.items():
        t_high = ms_high.minutes[mode]
        off = ~np.eye(t_low.shape[ZERO], t_low.shape[ONE], dtype=bool)
        assert (t_high[off] > t_low[off]).all()

    a_low = apply_accessibility(low, params, ms_low)["lnA"].to_numpy()
    a_high = apply_accessibility(high, params, ms_high)["lnA"].to_numpy()
    assert (a_high <= a_low).all()
    assert a_high.sum() < a_low.sum()


# --------------------------------------------------------------------------------------
# Section 8 ACCEPTANCE — float32
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_float32_matrix_storage_round_trips_lnA_within_tolerance(params, five_cells, tmp_path):
    """Section 8 ACCEPTANCE bullet 6."""
    n = len(five_cells)
    exact = {"car": _ramp(n, 5.0) + ONE / 3}
    ms64 = R.MatrixSet(
        origins=tuple(five_cells["h3"]),
        destinations=tuple(five_cells["h3_res8"]),
        minutes={"car": np.asarray(exact["car"], dtype=np.float64)},
    )
    ms32 = _matrixset(five_cells, exact)
    R.save_matrices(ms32, tmp_path, params)
    reloaded = R.load_matrices(tmp_path)

    tol = float(params.value("accessibility.matrix.float32_tolerance"))
    a64 = apply_accessibility(five_cells, params, ms64)["lnA"].to_numpy()
    a32 = apply_accessibility(five_cells, params, reloaded)["lnA"].to_numpy()
    assert np.abs(a64 - a32).max() < tol


# --------------------------------------------------------------------------------------
# Section 8.4 / Section 21 — station proximity, EXCLUSIVE bands
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_station_decay_bands_are_exclusive(params):
    """Section 21: a 300 m cell must NOT collect both the 0-400 and the 400-800 weight."""
    bands = R.station_decay_bands(params)
    first, second = bands[ZERO], bands[ONE]
    w = station_decay_weight(300.0, bands, feeder=False)
    assert w == pytest.approx(first.w)
    assert w != pytest.approx(first.w + second.w)
    assert w <= ONE


@pytest.mark.acceptance
def test_station_weight_column_is_the_exclusive_band_not_a_sum(params, five_cells):
    bands = R.station_decay_bands(params)
    n = len(five_cells)
    dist = np.full((n, ONE), 300.0, dtype=np.float32)
    ms = _matrixset(
        five_cells,
        {"car": _ramp(n, 5.0)},
        station_walk_dist_m=dist,
        station_feeder=(False,),
    )
    out = apply_accessibility(five_cells, params, ms)
    assert out["station_weight"].to_numpy() == pytest.approx(bands[ZERO].w)
    assert out["station_weight"].max() <= ONE


def test_each_band_returns_exactly_its_own_weight(params):
    bands = R.station_decay_bands(params)
    previous_max = ZERO
    for band in bands:
        if band.requires:
            continue
        midpoint = (previous_max + band.max_m) / 2
        assert station_decay_weight(midpoint, bands, feeder=False) == pytest.approx(band.w)
        previous_max = band.max_m


def test_beyond_the_last_open_band_the_weight_is_zero(params):
    bands = R.station_decay_bands(params)
    open_bands = [b for b in bands if not b.requires]
    assert station_decay_weight(open_bands[-ONE].max_m * 2, bands, feeder=False) == ZERO


def test_the_feeder_gated_band_applies_only_to_feeder_stations(params):
    bands = R.station_decay_bands(params)
    gated = [b for b in bands if b.requires][ZERO]
    d = (max(b.max_m for b in bands if not b.requires) + gated.max_m) / 2
    assert station_decay_weight(d, bands, feeder=False) == ZERO
    assert station_decay_weight(d, bands, feeder=True) == pytest.approx(gated.w)


def test_station_weight_is_the_max_over_stations(params, five_cells):
    bands = R.station_decay_bands(params)
    n = len(five_cells)
    far = bands[-ONE].max_m * 2
    dist = np.array([[far, 300.0]] * n, dtype=np.float32)
    ms = _matrixset(
        five_cells,
        {"car": _ramp(n, 5.0)},
        station_walk_dist_m=dist,
        station_feeder=(False, False),
    )
    out = apply_accessibility(five_cells, params, ms)
    assert out["station_weight"].to_numpy() == pytest.approx(bands[ZERO].w)


def test_station_weight_is_zero_when_there_are_no_stations(params, five_cells):
    out = apply_accessibility(
        five_cells, params, _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    )
    assert out["station_weight"].to_numpy() == pytest.approx(ZERO)


# --------------------------------------------------------------------------------------
# Section 8.5 — output
# --------------------------------------------------------------------------------------


def test_output_columns_are_exactly_the_section_8_5_set(params, five_cells):
    out = apply_accessibility(
        five_cells, params, _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    )
    added = [c for c in out.columns if c not in five_cells.columns]
    assert set(added) | set(OUTPUT_COLUMNS) == set(OUTPUT_COLUMNS)
    assert set(OUTPUT_COLUMNS) <= set(out.columns)


def test_output_validates_against_the_landed_cells_schema(params, five_cells):
    out = apply_accessibility(
        five_cells, params, _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    )
    S.SCHEMAS["cells"].validate(out)


def test_cumulative_job_bands_are_nested_and_mode_share_weighted(params, five_cells):
    n = len(five_cells)
    minutes = {"car": _ramp(n, 20.0)}
    out = apply_accessibility(five_cells, params, _matrixset(five_cells, minutes))
    assert (out["jobs_30min"] <= out["jobs_45min"]).all()
    assert (out["jobs_45min"] <= out["jobs_60min"]).all()

    share = float(params.value("accessibility.modes.car.share"))
    jobs = sum(five_cells["jobs_by_sector"][ZERO])
    # cell 0 is 0 minutes from the cluster, cell 1 is 20, cell 2 is 40 minutes away
    assert out["jobs_30min"][ZERO] == pytest.approx(share * jobs)
    assert out["jobs_30min"][2] == pytest.approx(ZERO)
    assert out["jobs_45min"][2] == pytest.approx(share * jobs)


def test_cells_outside_the_origin_set_get_null_not_zero(params):
    cells = synthetic_cells(n=300).drop_duplicates(subset="h3_res8").head(5).reset_index(drop=True)
    subset = cells.head(3)
    ms = R.MatrixSet(
        origins=tuple(subset["h3"]),
        destinations=tuple(cells["h3_res8"]),
        minutes={"car": np.zeros((len(subset), len(cells)), dtype=np.float32)},
    )
    out = apply_accessibility(cells, params, ms)
    assert out["lnA"].isna().sum() == len(cells) - len(subset)
    assert out["lnA"].notna().sum() == len(subset)


# --------------------------------------------------------------------------------------
# CONTRACT rules 2 and 3 — purity and no I/O at simulation time
# --------------------------------------------------------------------------------------


def test_apply_accessibility_is_pure(params, five_cells):
    before = five_cells.copy(deep=True)
    ms = _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    out = apply_accessibility(five_cells, params, ms)
    assert out is not five_cells
    assert len(out) == len(five_cells)
    assert out.index.equals(five_cells.index)
    pd.testing.assert_frame_equal(five_cells, before)


def test_apply_accessibility_is_deterministic(params, five_cells):
    ms = _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    a = apply_accessibility(five_cells, params, ms)
    b = apply_accessibility(five_cells, params, ms)
    pd.testing.assert_frame_equal(a, b)


def test_apply_accessibility_performs_no_io(params, five_cells, monkeypatch):
    """CONTRACT rule 3: no network calls, and no file access, at simulation time."""

    def forbidden(*a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError("apply_accessibility performed I/O")

    ms = _matrixset(five_cells, {"car": _ramp(len(five_cells), 5.0)})
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    apply_accessibility(five_cells, params, ms)


def test_l1_module_does_not_import_a_network_client():
    import ufe.layers.l1_accessibility as mod

    source = open(mod.__file__).read()
    for banned in ("import httpx", "import requests", "import duckdb", "from ufe.ai"):
        assert banned not in source


def test_l1_module_cannot_reach_a_backend():
    """The simulation-time module must not even name the backend classes."""
    import ufe.layers.l1_accessibility as mod

    source = open(mod.__file__).read()
    assert "OSRMBackend" not in source
    assert "HaversineBackend" not in source
    assert not hasattr(mod, "precompute_matrices")
