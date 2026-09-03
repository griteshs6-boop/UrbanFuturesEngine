"""The four backtest baselines, B1..B4, plus the B0 city-average anchor (spec Section 19.3).

All of them share one contract: given the frozen ``t0`` state and a horizon in years, return
a ``pd.Series`` indexed by ``h3`` holding **predicted total fractional price appreciation
over the horizon** — ``P(t0 + years) / P(t0) - 1`` — except :func:`b4_urban_growth`, which
predicts **Δ builtup_frac** and is scored on ``settlement_spearman`` only. Section 19.3 is
explicit about that last point and it is worth repeating: B4 does not predict prices, and
comparing it on price metrics would be dishonest.

Every baseline reads only the frozen snapshot. None of them touches the network, the store
or ``ufe.ai`` (CONTRACT.md rules 3 and 4), and every stochastic step takes an explicit
seeded generator (rule 5).

Why B4 is written out longhand here
-----------------------------------
Section 19.3: "The obvious candidates in the literature — FUTURES (a GRASS GIS addon) and
SLEUTH — are GPL or of unclear licence, and cannot be linked into a commercial product."
FUTURES ships as a GRASS addon under GPL-2.0; linking it into the served product would put
the whole engine under copyleft, which Section 0.1 rule 8 calls "an unrecoverable commercial
error". So the logistic cellular-automaton below is written from the standard formulation
using ``scikit-learn`` (BSD-3-Clause, already a declared dependency) and ``h3``
(Apache-2.0), and we own it outright. **Do not add GRASS, r.futures, or a SLEUTH port to
this repo or to CI.** If a peer-reviewed FUTURES comparison is wanted, run it offline on a
separate machine and paste the resulting *number* into a report — consuming a benchmark
score is not a derivative work; bundling the code is.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ufe.errors import MissingParameter, UFEError

logger = logging.getLogger(__name__)

__all__ = [
    "BaselineError",
    "city_price_cagr",
    "b0_city_average",
    "b1_distance_cbd",
    "b2_momentum",
    "b3_naive_announcement",
    "b4_urban_growth",
    "BASELINES",
    "PRICE_BASELINES",
    "SETTLEMENT_BASELINES",
]


class BaselineError(UFEError):
    """A baseline cannot be computed from the data it was given."""


# ------------------------------------------------------------------ parameter paths
P_SEED = "backtest.baselines.seed"
P_MOMENTUM_YEARS = "backtest.horizon.momentum_lookback_years"
P_CONVERSION_YEARS = "backtest.horizon.conversion_lookback_years"
P_MIN_HISTORY_YEARS = "backtest.horizon.min_history_years"
P_MIN_CAGR = "backtest.baselines.b0.min_cagr"
P_MAX_CAGR = "backtest.baselines.b0.max_cagr"
P_B1_OFFSET = "backtest.baselines.b1.distance_offset_m"
P_B1_EXP_MIN = "backtest.baselines.b1.exponent_min"
P_B1_EXP_MAX = "backtest.baselines.b1.exponent_max"
P_B2_FALLBACK = "backtest.baselines.b2.fallback_to_city_cagr"
P_B3_P = "backtest.baselines.b3.completion_probability"
P_B3_SLIP = "backtest.baselines.b3.slip_years"
P_B3_DEFAULT_PREMIUM = "backtest.baselines.b3.default_premium"
P_B3_DEFAULT_MAX_M = "backtest.baselines.b3.default_premium_max_m"
P_B3_TARGET = "backtest.baselines.b3.premium_target"
P_B4_KRING = "backtest.baselines.b4.neighbourhood_kring"
P_B4_BUILT_THRESHOLD = "backtest.baselines.b4.built_threshold"
P_B4_NEG_PER_POS = "backtest.baselines.b4.negative_per_positive"
P_B4_MIN_POSITIVES = "backtest.baselines.b4.min_positives"
P_B4_MAX_ITER = "backtest.baselines.b4.max_iter"
P_B4_C = "backtest.baselines.b4.inverse_regularisation_c"
P_B4_SOLVER = "backtest.baselines.b4.solver"
P_B4_STEPS_PER_YEAR = "backtest.baselines.b4.steps_per_year"
P_B4_MIN_STEP_AREA = "backtest.baselines.b4.min_step_area_sqm"
P_B4_DROP_FIRST = "backtest.baselines.b4.drop_first_zone_dummy"

# ------------------------------------------------------------------ column names
COL_H3 = "h3"
COL_YEAR = "year"
COL_PRICE = "price_res_inr_sqft"
COL_BUILTUP = "builtup_frac"
COL_AREA = "area_sqm"
COL_UNDEV = "undevelopable_frac"
COL_SLOPE = "slope_pct"
COL_DIST_CBD = "dist_cbd_m"
COL_DIST_ARTERIAL = "dist_arterial_m"
COL_DIST_BUILTUP = "dist_existing_builtup_m"
COL_ZONE_CLASS = "zone_class"
COL_LAT = "lat"
COL_LON = "lon"
NEIGHBOURHOOD_COL = "neighbourhood_builtup_frac"

_ZERO = 0
_ONE = 1


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def _t0_from_history(history: pd.DataFrame, t0: int | None) -> int:
    if t0 is not None:
        return int(t0)
    if history is None or history.empty:
        raise BaselineError("no history supplied and no explicit t0: cannot locate the origin")
    return int(pd.to_numeric(history[COL_YEAR]).max())


def _require_history_span(history: pd.DataFrame, t0: int, lookback: int, params: Any) -> None:
    minimum = int(params.value(P_MIN_HISTORY_YEARS))
    years = pd.to_numeric(history[COL_YEAR]).unique()
    inside = [y for y in years if t0 - lookback <= y <= t0]
    if len(inside) < minimum:
        raise BaselineError(
            f"history holds {len(inside)} year(s) in [{t0 - lookback}, {t0}]; at least "
            f"{minimum} are required (backtest.horizon.min_history_years). Extrapolating a "
            "trend from fewer points is guesswork with a decimal point on it."
        )


def _year_slice(history: pd.DataFrame, year: int, column: str) -> pd.Series:
    """``column`` at ``year``, indexed by h3. Missing years give an empty Series."""
    rows = history.loc[pd.to_numeric(history[COL_YEAR]) == year]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.set_index(COL_H3)[column].astype(float)


def _filled(values: pd.Series, fallback: np.ndarray) -> np.ndarray:
    """`values` as a float array, falling back element-wise where it is missing."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(numeric), fallback, numeric)


def _clip_cagr(value: float, params: Any) -> float:
    return float(
        np.clip(value, float(params.value(P_MIN_CAGR)), float(params.value(P_MAX_CAGR)))
    )


def city_price_cagr(
    history: pd.DataFrame, t0: int, lookback: int, params: Any
) -> float:
    """City-wide price CAGR over ``[t0 - lookback, t0]`` (the B0 anchor).

    The city index is the cross-sectional mean of observed cell prices. A median would be
    more robust but shifts with the *composition* of the observed set, and the observed set
    changes year to year in exactly the way that matters here.
    """
    _require_history_span(history, t0, lookback, params)
    start = _year_slice(history, t0 - lookback, COL_PRICE).dropna()
    end = _year_slice(history, t0, COL_PRICE).dropna()
    if start.empty or end.empty:
        raise BaselineError(
            f"no observed prices at t0={t0} or t0-{lookback}={t0 - lookback}; the city "
            "CAGR is undefined and the price test must be skipped (Section 19.1)"
        )
    ratio = float(end.mean()) / float(start.mean())
    if ratio <= _ZERO:
        raise BaselineError("city price index is non-positive; the panel is not usable")
    return _clip_cagr(ratio ** (_ONE / lookback) - _ONE, params)


def _appreciation(cagr: Any, years: int) -> Any:
    """Total fractional appreciation implied by a per-year CAGR over ``years``."""
    return (_ONE + np.asarray(cagr, dtype=float)) ** years - _ONE


def _as_series(values: Any, cells_t0: pd.DataFrame, name: str) -> pd.Series:
    out = pd.Series(np.asarray(values, dtype=float), index=pd.Index(cells_t0[COL_H3], name=COL_H3))
    out.name = name
    return out


# --------------------------------------------------------------------------------------
# B0 — everyone at the city CAGR
# --------------------------------------------------------------------------------------


def b0_city_average(
    cells_t0: pd.DataFrame,
    years: int,
    *,
    params: Any,
    history: pd.DataFrame,
    t0: int | None = None,
    lookback: int | None = None,
) -> pd.Series:
    """Every cell appreciates at the observed city CAGR (Section 19.3).

    Deliberately rank-degenerate: its Spearman is undefined, which is the point. B0 exists
    to calibrate *level* metrics (`mape_cagr`, `band_coverage`) and to prove that a model
    with no spatial content scores zero on the spatial metrics.
    """
    t0 = _t0_from_history(history, t0)
    lookback = int(params.value(P_MOMENTUM_YEARS)) if lookback is None else int(lookback)
    growth = city_price_cagr(history, t0, lookback, params)
    return _as_series(np.full(len(cells_t0), _appreciation(growth, years)), cells_t0, "b0")


# --------------------------------------------------------------------------------------
# B1 — monotonic decay in distance to the CBD, fitted exponent
# --------------------------------------------------------------------------------------


def b1_distance_cbd(
    cells_t0: pd.DataFrame,
    years: int,
    *,
    params: Any,
    history: pd.DataFrame,
    t0: int | None = None,
    lookback: int | None = None,
) -> pd.Series:
    """Appreciation decays monotonically with distance to the CBD; exponent fitted on the
    pre-``t0`` window only (Section 19.3).

    ``ln(1 + a_i) = a + b · ln(dist_i + offset)`` is fitted by least squares on realised
    appreciation over ``[t0 - lookback, t0]``. ``b`` is clipped to a non-positive range: the
    baseline's declared form is *monotonic decay*, so a training window in which the fringe
    happened to outrun the core does not get to silently invert the baseline into
    "the further out the better" — it gets a flat prediction, and the clip is logged.
    The intercept is then shifted so the predicted mean log growth matches the city CAGR,
    which fixes the level without touching the ranking.
    """
    t0 = _t0_from_history(history, t0)
    lookback = int(params.value(P_MOMENTUM_YEARS)) if lookback is None else int(lookback)
    offset = float(params.value(P_B1_OFFSET))
    growth = city_price_cagr(history, t0, lookback, params)

    start = _year_slice(history, t0 - lookback, COL_PRICE)
    end = _year_slice(history, t0, COL_PRICE)
    realised = (end / start).dropna()
    realised = realised[realised > _ZERO]

    distances = cells_t0.set_index(COL_H3)[COL_DIST_CBD].astype(float)
    ln_d_all = np.log(distances.to_numpy() + offset)

    exponent = _ZERO
    intercept = np.log(_ONE + growth) * lookback
    common = realised.index.intersection(distances.index)
    if len(common) > _ONE:
        y = np.log(realised.loc[common].to_numpy())
        x = np.log(distances.loc[common].to_numpy() + offset)
        if float(np.std(x)) > _ZERO:
            design = np.column_stack([np.ones(len(x)), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            intercept, exponent = float(coefficients[_ZERO]), float(coefficients[_ONE])
    else:
        logger.warning(
            "B1 has %d cell(s) with appreciation at both ends of the training window; "
            "falling back to a flat city-CAGR prediction",
            len(common),
        )

    clipped = float(
        np.clip(exponent, float(params.value(P_B1_EXP_MIN)), float(params.value(P_B1_EXP_MAX)))
    )
    if clipped != exponent:
        logger.warning(
            "B1 fitted exponent %.4f is outside the monotonic-decay range and was clipped "
            "to %.4f; over this window distance to the CBD did not predict appreciation",
            exponent,
            clipped,
        )

    log_growth_window = intercept + clipped * ln_d_all
    # Shift the level onto the observed city CAGR; ranking is unaffected.
    log_growth_window = (
        log_growth_window
        - float(np.mean(log_growth_window))
        + np.log(_ONE + growth) * lookback
    )
    per_year = log_growth_window / lookback
    return _as_series(np.exp(per_year * years) - _ONE, cells_t0, "b1")


# --------------------------------------------------------------------------------------
# B2 — momentum. The one that matters: the gate is defined against it.
# --------------------------------------------------------------------------------------


def b2_momentum(
    cells_t0: pd.DataFrame,
    history: pd.DataFrame,
    years: int,
    *,
    params: Any,
    t0: int | None = None,
    lookback: int | None = None,
) -> pd.Series:
    """Each cell's last-``N``-year CAGR persists over the horizon (Section 19.3).

    This is the baseline the gate is written against (``beat_b2``), and it is a genuinely
    hard one to beat over short horizons in a trending market. Cells with no price at both
    ends of the window fall back to the city CAGR rather than being dropped, so B2 covers
    the same zone set as the model and the Spearman comparison stays like-for-like.
    """
    t0 = _t0_from_history(history, t0)
    lookback = int(params.value(P_MOMENTUM_YEARS)) if lookback is None else int(lookback)
    fallback = bool(params.value(P_B2_FALLBACK))
    city = city_price_cagr(history, t0, lookback, params)

    start = _year_slice(history, t0 - lookback, COL_PRICE)
    end = _year_slice(history, t0, COL_PRICE)
    ratio = (end / start).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio > _ZERO)
    per_cell = ratio ** (_ONE / lookback) - _ONE

    aligned = per_cell.reindex(cells_t0[COL_H3].to_numpy())
    if fallback:
        aligned = aligned.fillna(city)
    else:
        aligned = aligned.dropna()
    clipped = aligned.clip(
        lower=float(params.value(P_MIN_CAGR)), upper=float(params.value(P_MAX_CAGR))
    )
    out = pd.Series(_appreciation(clipped.to_numpy(), years), index=pd.Index(clipped.index, name=COL_H3))
    out.name = "b2"
    return out


# --------------------------------------------------------------------------------------
# B3 — naive announcement: the engine with the credibility layer disabled
# --------------------------------------------------------------------------------------


def _cell_points_metric(cells_t0: pd.DataFrame, params: Any):
    import geopandas as gpd  # noqa: PLC0415 - heavy import, only needed for B3

    from ufe.geo import city_metric_crs, to_metric  # noqa: PLC0415

    points = gpd.GeoSeries(
        gpd.points_from_xy(cells_t0[COL_LON].astype(float), cells_t0[COL_LAT].astype(float)),
        crs="EPSG:4326",
    )
    return to_metric(points, city_metric_crs(params))


def _residential_premium_bands(
    params: Any, archetype: str
) -> list[tuple[float, float]]:
    """Exclusive (max_m, log-premium) bands for an archetype, nearest band first.

    Section 21 names "cumulative distance bands" as a failure mode — "a 300 m cell gets both
    the 0-500 and 500-1000 premium" — so the bands are returned sorted ascending and the
    caller applies exactly the first one that contains the cell.
    """
    target = str(params.value(P_B3_TARGET))
    try:
        rows = params.get(f"archetypes.{archetype}.premium")
    except MissingParameter:
        return []
    if not isinstance(rows, Sequence):
        return []
    bands: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("target") != target:
            continue
        if row.get("max_m") is None:
            continue
        bands.append((float(row["max_m"]), float(row["value"])))
    return sorted(bands)


def b3_naive_announcement(
    cells_t0: pd.DataFrame,
    projects: pd.DataFrame,
    years: int,
    *,
    params: Any,
    history: pd.DataFrame,
    t0: int | None = None,
    lookback: int | None = None,
    runner: Any = None,
    scenario: Any = None,
    snapshot: Any = None,
) -> pd.Series:
    """Every project completes on ``stated_completion``, at full effect, ``p = 1``, no delay.

    Section 19.3: "B3 is the full engine with the credibility layer disabled. Its purpose is
    to isolate the credibility layer's contribution." When a runner is available that is
    exactly what happens: ``runner.run`` is called with every project forced to ``happens``
    (the Section 15.1 ``Scenario.force_project_state`` hook), which is what "credibility
    disabled" means operationally.

    ``ufe/sim/runner.py`` is being built in parallel and may not be on disk. Rather than
    fail or, worse, quietly return something that looks like an engine run, this falls back
    to an explicit analytic stand-in — the archetype residual-premium table applied at
    ``p = 1`` with no slip — and stamps ``series.attrs['backend'] = 'analytic_fallback'`` so
    no scorecard can claim an engine comparison it did not make. If a ``runner`` is passed
    explicitly there is no fallback: a broken runner raises.
    """
    t0 = _t0_from_history(history, t0)
    lookback = int(params.value(P_MOMENTUM_YEARS)) if lookback is None else int(lookback)

    resolved_runner = runner
    if resolved_runner is None and snapshot is not None:
        # Only auto-discover the engine when there is a snapshot to run it against.
        # `ufe/sim/runner.py` may not exist yet (it is built in parallel), and even when it
        # does, calling it without a snapshot would fail deep inside the runner rather than
        # here, which is a worse error message and a worse failure mode.
        try:  # Section 15.1.
            from ufe.sim import runner as sim_runner  # noqa: PLC0415

            resolved_runner = sim_runner if hasattr(sim_runner, "run") else None
        except ImportError:
            resolved_runner = None

    if resolved_runner is not None:
        out = _b3_via_runner(
            resolved_runner, cells_t0, projects, years, params, t0, scenario, snapshot
        )
        out.attrs["backend"] = "runner"
        return out

    logger.warning(
        "B3 is running its analytic stand-in: no simulation runner was available. The "
        "resulting scorecard measures the credibility layer against a premium-table "
        "approximation, not against the engine. Do not quote it as an engine comparison."
    )
    out = _b3_analytic(cells_t0, projects, years, params, history, t0, lookback)
    out.attrs["backend"] = "analytic_fallback"
    return out


def _b3_via_runner(
    runner: Any,
    cells_t0: pd.DataFrame,
    projects: pd.DataFrame,
    years: int,
    params: Any,
    t0: int,
    scenario: Any,
    snapshot: Any,
) -> pd.Series:
    """Call the Section 15.1 runner with credibility disabled and read the price path."""
    if scenario is None:
        scenario = _build_forced_scenario(runner, params, projects, t0, years)
    result = runner.run(snapshot, params, scenario)
    prices = _price_frame_from_result(result)
    start = prices.loc[prices[COL_YEAR] == t0].set_index(COL_H3)[COL_PRICE].astype(float)
    end = prices.loc[prices[COL_YEAR] == t0 + years].set_index(COL_H3)[COL_PRICE].astype(float)
    ratio = (end / start).reindex(cells_t0[COL_H3].to_numpy())
    out = pd.Series(ratio.to_numpy() - _ONE, index=pd.Index(cells_t0[COL_H3], name=COL_H3))
    out.name = "b3"
    return out


def _build_forced_scenario(
    runner: Any, params: Any, projects: pd.DataFrame, t0: int, years: int
) -> Any:
    scenario_cls = getattr(runner, "Scenario", None)
    if scenario_cls is None:
        raise BaselineError(
            "the supplied runner exposes no `Scenario` (spec Section 15.1); B3 cannot "
            "force every project to 'happens' without it"
        )
    forced = {str(pid): "happens" for pid in sorted(projects["project_id"])}
    return scenario_cls(
        city_id=params.city_id,
        horizon=list(range(t0, t0 + years + _ONE)),
        force_project_state=forced,
    )


def _price_frame_from_result(result: Any) -> pd.DataFrame:
    for attribute in ("cells_by_year", "panel", "cell_years", "state"):
        frame = getattr(result, attribute, None)
        if isinstance(frame, pd.DataFrame) and {COL_H3, COL_YEAR, COL_PRICE} <= set(frame.columns):
            return frame
    raise BaselineError(
        "the runner's SimResult exposes no per-cell per-year frame carrying "
        f"{COL_H3}/{COL_YEAR}/{COL_PRICE}; B3 cannot read a price path out of it"
    )


def _b3_analytic(
    cells_t0: pd.DataFrame,
    projects: pd.DataFrame,
    years: int,
    params: Any,
    history: pd.DataFrame,
    t0: int,
    lookback: int,
) -> pd.Series:
    """Archetype residual premia at ``p = 1``, no slip, on top of the city CAGR."""
    probability = float(params.value(P_B3_P))
    slip = int(params.value(P_B3_SLIP))
    default_premium = float(params.value(P_B3_DEFAULT_PREMIUM))
    default_max_m = float(params.value(P_B3_DEFAULT_MAX_M))
    growth = city_price_cagr(history, t0, lookback, params)

    uplift = np.zeros(len(cells_t0), dtype=float)
    if not projects.empty:
        from shapely import wkt  # noqa: PLC0415

        import geopandas as gpd  # noqa: PLC0415

        from ufe.geo import city_metric_crs, to_metric  # noqa: PLC0415

        cell_points = _cell_points_metric(cells_t0, params)
        crs_metric = city_metric_crs(params)

        announced = pd.to_datetime(projects["announced_date"]).dt.year
        completion = pd.to_datetime(projects["stated_completion"]).dt.year + slip
        live = projects.loc[(announced <= t0) & (completion <= t0 + years)]

        for _, project in live.iterrows():
            bands = _residential_premium_bands(params, str(project["archetype"]))
            if not bands:
                bands = [(default_max_m, default_premium)]
            geometry = to_metric(
                gpd.GeoSeries([wkt.loads(str(project["geom"]))], crs="EPSG:4326"), crs_metric
            )
            distance = cell_points.distance(geometry.iloc[_ZERO]).to_numpy()
            premium = np.zeros(len(cells_t0), dtype=float)
            assigned = np.zeros(len(cells_t0), dtype=bool)
            for max_m, value in bands:  # ascending: exclusive bands, nearest wins
                inside = (distance <= max_m) & (~assigned)
                premium[inside] = value
                assigned |= inside
            uplift += np.log(_ONE + premium * probability)

    log_total = np.log(_ONE + growth) * years + uplift
    return _as_series(np.exp(log_total) - _ONE, cells_t0, "b3")


# --------------------------------------------------------------------------------------
# B4 — the in-house logistic cellular-automata urban growth model
# --------------------------------------------------------------------------------------


def _kring_neighbours(cells_t0: pd.DataFrame, k: int) -> list[np.ndarray]:
    """Integer positions of each cell's k-ring neighbours, self excluded."""
    import h3  # noqa: PLC0415

    position = {str(cell): index for index, cell in enumerate(cells_t0[COL_H3])}
    out: list[np.ndarray] = []
    for cell in cells_t0[COL_H3]:
        cell = str(cell)
        ring = h3.grid_disk(cell, k)
        indices = [position[n] for n in sorted(ring) if n in position and n != cell]
        out.append(np.asarray(indices, dtype=np.int64))
    return out


def _neighbourhood_mean(values: np.ndarray, neighbours: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for index, positions in enumerate(neighbours):
        out[index] = float(values[positions].mean()) if len(positions) else values[index]
    return out


def _zone_dummies(cells_t0: pd.DataFrame, params: Any) -> pd.DataFrame:
    from ufe.store.schemas import ZONE_CLASSES  # noqa: PLC0415

    drop_first = bool(params.value(P_B4_DROP_FIRST))
    classes = list(ZONE_CLASSES)[_ONE:] if drop_first else list(ZONE_CLASSES)
    series = cells_t0[COL_ZONE_CLASS].astype(str).to_numpy()
    return pd.DataFrame(
        {f"zone_{name}": (series == name).astype(float) for name in classes},
        index=cells_t0.index,
    )


def _b4_design(
    cells_t0: pd.DataFrame, neighbourhood: np.ndarray, params: Any
) -> pd.DataFrame:
    """Section 19.3 step 2's covariate block, in a fixed column order."""
    missing = [
        column
        for column in (COL_SLOPE, COL_DIST_ARTERIAL, COL_DIST_CBD, COL_DIST_BUILTUP, COL_UNDEV)
        if column not in cells_t0.columns
    ]
    if missing:
        raise BaselineError(
            f"B4 needs {', '.join(missing)} on the cell frame (spec Section 19.3 step 2)"
        )
    base = pd.DataFrame(
        {
            COL_SLOPE: cells_t0[COL_SLOPE].astype(float).to_numpy(),
            COL_DIST_ARTERIAL: cells_t0[COL_DIST_ARTERIAL].astype(float).to_numpy(),
            COL_DIST_CBD: cells_t0[COL_DIST_CBD].astype(float).to_numpy(),
            COL_DIST_BUILTUP: cells_t0[COL_DIST_BUILTUP].astype(float).fillna(_ZERO).to_numpy(),
            NEIGHBOURHOOD_COL: np.asarray(neighbourhood, dtype=float),
            COL_UNDEV: cells_t0[COL_UNDEV].astype(float).to_numpy(),
        },
        index=cells_t0.index,
    )
    return pd.concat([base, _zone_dummies(cells_t0, params)], axis=_ONE)


def _standardise(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = design.mean(axis=_ZERO)
    scale = design.std(axis=_ZERO)
    scale = np.where(scale > _ZERO, scale, _ONE)
    return (design - mean) / scale, mean, scale


def b4_urban_growth(
    cells_t0: pd.DataFrame,
    history: pd.DataFrame,
    years: int,
    *,
    params: Any,
    t0: int | None = None,
    lookback: int | None = None,
    exogenous_growth_sqm: float | None = None,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Logistic cellular-automata urban growth: predicted Δ ``builtup_frac`` per cell.

    Written in-house (see the module docstring on why FUTURES and SLEUTH are off-limits).
    Follows Section 19.3's five steps exactly:

    1. positives are cells that crossed ``built_threshold`` between ``t0 - lookback`` and
       ``t0``; negatives are a seeded random sample of cells that stayed below it;
    2. covariates are slope, distance to arterial, distance to the CBD, distance to
       existing built-up, the k-ring-2 neighbourhood built-up fraction, one-hot
       ``zone_class`` and ``undevelopable_frac``, all evaluated at ``t0 - lookback`` so the
       fit never sees the state it is being asked to explain;
    3. ``sklearn.linear_model.LogisticRegression`` on standardised covariates;
    4. total city growth over the horizon is exogenous, taken from the observed city trend,
       and is allocated in annual steps to cells in descending ``p_i``, capped by
       ``(1 - undevelopable_frac_i) · area_i``, **recomputing the neighbourhood term after
       every step** — that recomputation is the cellular automaton and it is what makes
       growth propagate outward from existing fabric instead of scattering;
    5. the return value is Δ ``builtup_frac``.

    Scored on ``settlement_spearman`` only.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    t0 = _t0_from_history(history, t0)
    lookback = int(params.value(P_CONVERSION_YEARS)) if lookback is None else int(lookback)
    threshold = float(params.value(P_B4_BUILT_THRESHOLD))
    kring = int(params.value(P_B4_KRING))
    rng = np.random.default_rng(int(params.value(P_SEED))) if rng is None else rng

    _require_history_span(history, t0, lookback, params)
    start_built = _year_slice(history, t0 - lookback, COL_BUILTUP).reindex(
        cells_t0[COL_H3].to_numpy()
    )
    end_built = _year_slice(history, t0, COL_BUILTUP).reindex(cells_t0[COL_H3].to_numpy())
    if start_built.isna().all() or end_built.isna().all():
        raise BaselineError(
            f"cells_history carries no {COL_BUILTUP} at {t0 - lookback} or {t0}; B4 has no "
            "conversion events to learn from"
        )
    fallback = cells_t0[COL_BUILTUP].to_numpy(dtype=float)
    start_values = _filled(start_built, fallback)
    end_values = _filled(end_built, fallback)

    neighbours = _kring_neighbours(cells_t0, kring)

    was_built = start_values >= threshold
    is_built = end_values >= threshold
    converted = (~was_built) & is_built
    stayed = (~was_built) & (~is_built)

    n_positive = int(converted.sum())
    minimum = int(params.value(P_B4_MIN_POSITIVES))
    if n_positive < minimum:
        raise BaselineError(
            f"B4 found {n_positive} conversion(s) between {t0 - lookback} and {t0}; at "
            f"least {minimum} are required (backtest.baselines.b4.min_positives). A "
            "logistic fit on fewer is noise with a coefficient table attached."
        )

    ratio = float(params.value(P_B4_NEG_PER_POS))
    candidate_negatives = np.flatnonzero(stayed)
    wanted = min(len(candidate_negatives), int(round(n_positive * ratio)))
    chosen = rng.choice(candidate_negatives, size=wanted, replace=False) if wanted else np.array([], dtype=np.int64)
    training_rows = np.sort(np.concatenate([np.flatnonzero(converted), chosen]))
    labels = converted[training_rows].astype(int)

    design_start = _b4_design(cells_t0, _neighbourhood_mean(start_values, neighbours), params)
    scaled, mean, scale = _standardise(design_start.to_numpy(dtype=float))

    model = LogisticRegression(
        C=float(params.value(P_B4_C)),
        max_iter=int(params.value(P_B4_MAX_ITER)),
        solver=str(params.value(P_B4_SOLVER)),
        random_state=int(params.value(P_SEED)),
    )
    model.fit(scaled[training_rows], labels)

    # ---- step 4: allocate exogenous city growth, cellular-automata style ---------------
    area = cells_t0[COL_AREA].astype(float).to_numpy()
    undevelopable = cells_t0[COL_UNDEV].astype(float).to_numpy()
    current = np.asarray(end_values, dtype=float).copy()
    ceiling = _ONE - undevelopable

    if exogenous_growth_sqm is None:
        built_start = float(np.nansum(start_values * area))
        built_end = float(np.nansum(end_values * area))
        annual = (built_end - built_start) / lookback
        exogenous_growth_sqm = max(_ZERO, annual * years)

    steps_per_year = int(params.value(P_B4_STEPS_PER_YEAR))
    n_steps = max(_ONE, years * steps_per_year)
    per_step = float(exogenous_growth_sqm) / n_steps
    min_step_area = float(params.value(P_B4_MIN_STEP_AREA))
    order_key = cells_t0[COL_H3].astype(str).to_numpy()

    for _ in range(n_steps):
        if per_step < min_step_area:
            break
        design = _b4_design(cells_t0, _neighbourhood_mean(current, neighbours), params)
        probability = model.predict_proba(
            (design.to_numpy(dtype=float) - mean) / scale
        )[:, _ONE]
        headroom = np.maximum(_ZERO, (ceiling - current)) * area
        # Descending probability, ties broken on h3 so the allocation is deterministic.
        order = np.lexsort((order_key, -probability))
        remaining = per_step
        for position in order:
            if remaining < min_step_area:
                break
            take = min(remaining, headroom[position])
            if take <= _ZERO:
                continue
            current[position] += take / area[position]
            remaining -= take

    delta = current - np.asarray(end_values, dtype=float)
    return _as_series(delta, cells_t0, "b4")


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------

#: Baselines scored against price appreciation.
PRICE_BASELINES: dict[str, Any] = {
    "b0": b0_city_average,
    "b1": b1_distance_cbd,
    "b2": b2_momentum,
    "b3": b3_naive_announcement,
}

#: Baselines scored against settlement only (Section 19.3: "never against price metrics").
SETTLEMENT_BASELINES: dict[str, Any] = {"b4": b4_urban_growth}

BASELINES: dict[str, Any] = {**PRICE_BASELINES, **SETTLEMENT_BASELINES}
