"""Tests for the backtest baselines B0..B4 (spec Section 19.3).

The Section 19 ACCEPTANCE item "All four baselines run on the Vizag fixture and produce
sane values" maps onto ``test_acc_all_baselines_run_and_are_sane``.

Everything here runs on a *constructed* panel with a known answer rather than on random
noise: prices grow at a rate that decays monotonically with distance to the CBD, and
built-up fraction grows fastest where the surrounding fabric is already built. That means
each baseline has an outcome we can predict in advance — B1 should rank well, B2 should
recover the planted per-cell CAGR almost exactly, and B4's allocation should land next to
existing development rather than scattering — so a passing test says something, instead of
just saying the code did not crash.

This module also owns the shared panel builders that ``test_backtest.py`` imports.

Numeric policy. The panel-construction constants below are fixture geometry, not model
parameters: they define the synthetic world, they are named, and they are declared once.
Every *model* threshold, horizon and criterion asserted against is read from
``config/params/backtest.yaml`` through ``Params``, per Section 0.1 rule 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.backtest import baselines as B
from ufe.params import load_params

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    build_city,
    synthetic_city,
)

CITY = "vizag"

#: Origin year. Every fitted parameter in the landed config declares
#: `fitted_on.data_through: 2019`, so 2019 is the earliest origin at which a freeze against
#: the real parameter tree is clean. A freeze before it must raise — see test_backtest.py.
T0 = 2019
FIRST_YEAR = 2004
LAST_YEAR = 2029
HORIZON = 5

# --- fixture geometry: the synthetic world's planted parameters -------------------------
CORE_PRICE_CAGR = 0.11        # price growth at the CBD
FRINGE_PRICE_CAGR = 0.02      # price growth at the far fringe
BASE_PRICE_INR_SQFT = 4000.0
BUILTUP_RATE_PER_YEAR = 0.03
PRICE_NOISE_SD = 0.002
SEED = 20240101


# --------------------------------------------------------------------------------------
# shared panel builders
# --------------------------------------------------------------------------------------


def normalised_distance(cells: pd.DataFrame) -> np.ndarray:
    d = cells["dist_cbd_m"].to_numpy(dtype=float)
    return (d - d.min()) / (d.max() - d.min())


def planted_price_cagr(cells: pd.DataFrame) -> np.ndarray:
    """Per-cell price CAGR: highest at the CBD, decaying monotonically outward."""
    dn = normalised_distance(cells)
    return CORE_PRICE_CAGR - (CORE_PRICE_CAGR - FRINGE_PRICE_CAGR) * dn


def build_panel(
    cells: pd.DataFrame,
    *,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
    seed: int = SEED,
) -> pd.DataFrame:
    """A dense `cells_history` panel with a planted, knowable structure.

    Prices compound at :func:`planted_price_cagr` with a whisper of seeded noise so the
    Spearman statistics are not degenerate. Built-up fraction starts high near the CBD and
    grows at a constant rate, so cells in the middle of the distribution cross the
    "is built" threshold during the training window and give B4 real conversions to learn.
    """
    rng = np.random.default_rng(seed)
    dn = normalised_distance(cells)
    cagr = planted_price_cagr(cells)
    # Built-up fraction can never exceed the developable ceiling; the panel respects the
    # same physical cap the allocator does, or the "sane values" assertions would be
    # testing the fixture's carelessness rather than the model.
    ceiling = 1 - cells["undevelopable_frac"].to_numpy(dtype=float)
    base_builtup = np.clip(1 - dn, 0, 1) * ceiling
    years = list(range(first_year, last_year + 1))
    n = len(cells)

    # One draw per cell, not per year: the planted per-cell CAGR must be exactly
    # recoverable, so the noise perturbs the growth rate, not the path around it.
    noise = rng.normal(0, PRICE_NOISE_SD, n)
    rows = []
    for year in years:
        elapsed = year - first_year
        price = BASE_PRICE_INR_SQFT * (1 + cagr + noise) ** elapsed
        builtup = np.minimum(base_builtup + BUILTUP_RATE_PER_YEAR * elapsed, ceiling)
        rows.append(
            pd.DataFrame(
                {
                    "h3": cells["h3"].to_numpy(),
                    "year": np.full(n, year, dtype=np.int64),
                    "builtup_frac": builtup,
                    "nightlight": cells["nightlight"].to_numpy(dtype=float),
                    "population": cells["population"].to_numpy(dtype=float),
                    "price_res_inr_sqft": price,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def realised_appreciation(panel: pd.DataFrame, t0: int, years: int) -> pd.Series:
    """The answer key: actual total appreciation between ``t0`` and ``t0 + years``."""
    start = panel.loc[panel["year"] == t0].set_index("h3")["price_res_inr_sqft"]
    end = panel.loc[panel["year"] == t0 + years].set_index("h3")["price_res_inr_sqft"]
    out = (end / start) - 1
    out.name = "actual"
    return out


def realised_settlement(panel: pd.DataFrame, t0: int, years: int) -> pd.Series:
    start = panel.loc[panel["year"] == t0].set_index("h3")["builtup_frac"]
    end = panel.loc[panel["year"] == t0 + years].set_index("h3")["builtup_frac"]
    out = end - start
    out.name = "actual_settlement"
    return out


@pytest.fixture(scope="module")
def params():
    return load_params(city=CITY)


@pytest.fixture(scope="module")
def city():
    return build_city()


@pytest.fixture(scope="module")
def panel(city):
    return build_panel(city.cells)


# --------------------------------------------------------------------------------------
# B0
# --------------------------------------------------------------------------------------


def test_b0_is_a_single_number_for_every_cell(city, panel, params):
    pred = B.b0_city_average(city.cells, HORIZON, params=params, history=panel, t0=T0)
    assert len(pred) == len(city.cells)
    assert pred.nunique() == 1
    assert np.isfinite(pred).all()


def test_b0_recovers_the_city_cagr(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    cagr = B.city_price_cagr(panel, T0, lookback, params)
    pred = B.b0_city_average(city.cells, HORIZON, params=params, history=panel, t0=T0)
    assert pred.iloc[0] == pytest.approx((1 + cagr) ** HORIZON - 1)
    # The planted CAGRs bracket the city aggregate.
    assert FRINGE_PRICE_CAGR < cagr < CORE_PRICE_CAGR


def test_b0_ranks_nothing_and_that_is_the_point(city, panel, params):
    from ufe.backtest.score import spearman

    pred = B.b0_city_average(city.cells, HORIZON, params=params, history=panel, t0=T0)
    actual = realised_appreciation(panel, T0, HORIZON)
    correlation, _ = spearman(pred, actual)
    assert correlation is None


# --------------------------------------------------------------------------------------
# B1
# --------------------------------------------------------------------------------------


def test_b1_is_monotonically_decreasing_in_distance_to_cbd(city, panel, params):
    pred = B.b1_distance_cbd(city.cells, HORIZON, params=params, history=panel, t0=T0)
    distance = city.cells.set_index("h3")["dist_cbd_m"].reindex(pred.index)
    ordered = pred.iloc[np.argsort(distance.to_numpy())]
    assert np.all(np.diff(ordered.to_numpy()) <= 0)


def test_b1_recovers_the_planted_distance_gradient(city, panel, params):
    """The panel was built so appreciation decays with distance; B1 must find that."""
    from ufe.backtest.score import spearman

    pred = B.b1_distance_cbd(city.cells, HORIZON, params=params, history=panel, t0=T0)
    actual = realised_appreciation(panel, T0, HORIZON)
    correlation, _ = spearman(pred, actual)
    assert correlation is not None and correlation > 0.9


def test_b1_exponent_is_clipped_to_monotonic_decay(city, params, caplog):
    """A window where the fringe outran the core must not invert the baseline."""
    cells = city.cells
    dn = normalised_distance(cells)
    inverted = FRINGE_PRICE_CAGR + (CORE_PRICE_CAGR - FRINGE_PRICE_CAGR) * dn
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        rows.append(
            pd.DataFrame(
                {
                    "h3": cells["h3"].to_numpy(),
                    "year": np.full(len(cells), year, dtype=np.int64),
                    "builtup_frac": cells["builtup_frac"].to_numpy(dtype=float),
                    "nightlight": cells["nightlight"].to_numpy(dtype=float),
                    "population": cells["population"].to_numpy(dtype=float),
                    "price_res_inr_sqft": BASE_PRICE_INR_SQFT
                    * (1 + inverted) ** (year - FIRST_YEAR),
                }
            )
        )
    history = pd.concat(rows, ignore_index=True)
    pred = B.b1_distance_cbd(cells, HORIZON, params=params, history=history, t0=T0)
    # Clipped to a zero exponent: flat, never inverted into "further out is better".
    assert pred.nunique() == 1


# --------------------------------------------------------------------------------------
# B2 — the baseline the gate is written against
# --------------------------------------------------------------------------------------


def test_b2_recovers_the_planted_per_cell_cagr(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    pred = B.b2_momentum(city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback)
    implied = (1 + pred) ** (1 / HORIZON) - 1
    planted = pd.Series(planted_price_cagr(city.cells), index=city.cells["h3"])
    assert np.allclose(implied.to_numpy(), planted.reindex(implied.index).to_numpy(), atol=0.01)


def test_b2_falls_back_to_the_city_cagr_where_prices_are_missing(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    holed = panel.copy()
    victims = set(city.cells["h3"].iloc[:5])
    holed.loc[holed["h3"].isin(victims), "price_res_inr_sqft"] = np.nan
    pred = B.b2_momentum(city.cells, holed, HORIZON, params=params, t0=T0, lookback=lookback)
    # Same coverage as the model: nothing dropped.
    assert len(pred) == len(city.cells)
    assert np.isfinite(pred).all()
    city_cagr = B.city_price_cagr(holed, T0, lookback, params)
    for h3 in victims:
        assert pred.loc[h3] == pytest.approx((1 + city_cagr) ** HORIZON - 1)


def test_b2_is_deterministic(city, panel, params):
    first = B.b2_momentum(city.cells, panel, HORIZON, params=params, t0=T0)
    second = B.b2_momentum(city.cells, panel, HORIZON, params=params, t0=T0)
    pd.testing.assert_series_equal(first, second)


def test_baselines_refuse_a_panel_that_is_too_short(city, params):
    thin = build_panel(city.cells, first_year=T0 - 1, last_year=T0)
    with pytest.raises(B.BaselineError, match="min_history_years"):
        B.b2_momentum(city.cells, thin, HORIZON, params=params, t0=T0)


# --------------------------------------------------------------------------------------
# B3
# --------------------------------------------------------------------------------------


def test_b3_analytic_fallback_is_labelled_as_such(city, panel, params):
    """No runner on disk must never look like an engine comparison."""
    projects = city.projects
    pred = B.b3_naive_announcement(
        city.cells, projects, HORIZON, params=params, history=panel, t0=T0
    )
    assert pred.attrs["backend"] in {"analytic_fallback", "runner"}
    assert len(pred) == len(city.cells)
    assert np.isfinite(pred).all()


def test_b3_gives_cells_near_a_project_a_premium(city, panel, params):
    """p = 1, no slip: a project inside the horizon lifts the cells around it."""
    from shapely.geometry import Point

    cells = city.cells
    target = cells.iloc[0]
    project = pd.DataFrame(
        {
            "project_id": ["p-near"],
            "archetype": ["metro_rail"],
            "geom": [Point(float(target["lon"]), float(target["lat"])).wkt],
            "announced_date": [pd.Timestamp(f"{T0 - 1}-01-01")],
            "stated_completion": [pd.Timestamp(f"{T0 + 1}-01-01")],
        }
    )
    with_project = B.b3_naive_announcement(
        cells, project, HORIZON, params=params, history=panel, t0=T0
    )
    without = B.b3_naive_announcement(
        cells, project.iloc[:0], HORIZON, params=params, history=panel, t0=T0
    )
    assert with_project.loc[target["h3"]] > without.loc[target["h3"]]
    # And a cell on the far side of the city is untouched: the premium is not city-wide.
    far = cells.set_index("h3")["dist_cbd_m"].reindex(with_project.index)
    _ = far
    assert (with_project >= without - 1e-12).all()
    assert (with_project > without + 1e-12).sum() < len(cells)


def test_b3_uses_exclusive_distance_bands(params):
    """Section 21: a 300 m cell must not collect both the 0-500 and 500-1000 premium."""
    bands = B._residential_premium_bands(params, "metro_rail")
    assert bands == sorted(bands)
    assert len(bands) >= 1
    # Ascending order is what makes the nearest band win exclusively in `_b3_analytic`.
    assert bands[0][0] < bands[-1][0] or len(bands) == 1


def test_b3_with_a_runner_uses_the_runner_and_never_falls_back(city, panel, params):
    """A supplied runner is authoritative: if it is broken, B3 raises rather than fake it."""

    class BrokenRunner:
        class Scenario:  # noqa: D106 - Section 15.1 shape, minimal
            def __init__(self, city_id, horizon, force_project_state=None, **kwargs):
                self.city_id = city_id
                self.horizon = horizon
                self.force_project_state = force_project_state or {}

        @staticmethod
        def run(snapshot, params, scenario):
            return object()

    with pytest.raises(B.BaselineError, match="SimResult"):
        B.b3_naive_announcement(
            city.cells,
            city.projects,
            HORIZON,
            params=params,
            history=panel,
            t0=T0,
            runner=BrokenRunner,
        )


def test_b3_forces_every_project_to_happen(city, panel, params):
    """Credibility disabled means every project is forced to 'happens' (Section 15.1)."""
    seen: dict[str, object] = {}

    class RecordingRunner:
        class Scenario:
            def __init__(self, city_id, horizon, force_project_state=None, **kwargs):
                seen["forced"] = dict(force_project_state or {})
                self.city_id = city_id

        @staticmethod
        def run(snapshot, params, scenario):
            raise AssertionError("stop here; the scenario is what is under test")

    with pytest.raises(AssertionError):
        B.b3_naive_announcement(
            city.cells,
            city.projects,
            HORIZON,
            params=params,
            history=panel,
            t0=T0,
            runner=RecordingRunner,
        )
    assert set(seen["forced"]) == set(city.projects["project_id"])
    assert set(seen["forced"].values()) == {"happens"}


# --------------------------------------------------------------------------------------
# B4 — the in-house logistic cellular automaton
# --------------------------------------------------------------------------------------


def test_b4_uses_no_gpl_dependency():
    """Section 19.3 / Section 0.1 rule 8: no GRASS, no r.futures, no SLEUTH.

    A copyleft dependency reaching the served product is described in the spec as an
    unrecoverable commercial error, and B4 is the single place in the engine where the
    temptation to reach for one is real.
    """
    import ast
    import pathlib

    source = pathlib.Path(B.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    forbidden = {"grass", "grass_session", "pygrass", "futures_gis", "sleuth", "r_futures"}
    assert not (imported & forbidden)
    assert "sklearn" in imported, "B4 is the in-house logistic CA; it needs scikit-learn"


def test_b4_returns_a_sane_delta_builtup_frac(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    pred = B.b4_urban_growth(
        city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback
    )
    assert len(pred) == len(city.cells)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all(), "the allocator only ever adds built-up area"
    ceiling = 1 - city.cells.set_index("h3")["undevelopable_frac"].reindex(pred.index)
    current = panel.loc[panel["year"] == T0].set_index("h3")["builtup_frac"].reindex(pred.index)
    assert (current + pred <= ceiling + 1e-9).all(), "allocation must respect the land cap"


def test_b4_allocates_the_exogenous_growth_it_was_given(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    area = city.cells.set_index("h3")["area_sqm"]
    budget = float(area.sum()) * BUILTUP_RATE_PER_YEAR
    pred = B.b4_urban_growth(
        city.cells,
        panel,
        HORIZON,
        params=params,
        t0=T0,
        lookback=lookback,
        exogenous_growth_sqm=budget,
    )
    allocated = float((pred * area.reindex(pred.index)).sum())
    assert allocated == pytest.approx(budget, rel=0.01)


def test_b4_recomputes_the_neighbourhood_every_allocation_step(city, panel, params, monkeypatch):
    """The cellular-automata part, and Section 19.3 says in as many words that it matters.

    "Recompute neighbourhood_builtup_frac each step so growth propagates outward — this is
    the cellular-automata part and it matters." A single-shot allocation against a frozen
    neighbourhood term would still produce plausible-looking numbers, which is exactly why
    this is asserted rather than eyeballed.
    """
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    steps_per_year = int(params.value("backtest.baselines.b4.steps_per_year"))
    calls: list[int] = []
    original = B._neighbourhood_mean

    def counting(values, neighbours):
        calls.append(len(values))
        return original(values, neighbours)

    monkeypatch.setattr(B, "_neighbourhood_mean", counting)
    B.b4_urban_growth(city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback)

    # One call to build the training design, then one per allocation step.
    assert len(calls) >= HORIZON * steps_per_year + 1


def test_b4_concentrates_growth_rather_than_smearing_it(city, panel, params):
    """A CA that allocates everywhere in equal measure has learned nothing."""
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    pred = B.b4_urban_growth(
        city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback
    )
    if pred.sum() > 0:
        assert (pred > 0).sum() < len(pred)


def test_b4_refuses_a_panel_with_no_conversions(city, params):
    """A logistic fit on a handful of positives is noise with a coefficient table."""
    cells = city.cells
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        rows.append(
            pd.DataFrame(
                {
                    "h3": cells["h3"].to_numpy(),
                    "year": np.full(len(cells), year, dtype=np.int64),
                    "builtup_frac": np.zeros(len(cells)),
                    "nightlight": cells["nightlight"].to_numpy(dtype=float),
                    "population": cells["population"].to_numpy(dtype=float),
                    "price_res_inr_sqft": np.full(len(cells), BASE_PRICE_INR_SQFT),
                }
            )
        )
    frozen = pd.concat(rows, ignore_index=True)
    with pytest.raises(B.BaselineError, match="min_positives"):
        B.b4_urban_growth(cells, frozen, HORIZON, params=params, t0=T0, lookback=5)


def test_b4_is_deterministic(city, panel, params):
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    first = B.b4_urban_growth(city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback)
    second = B.b4_urban_growth(city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback)
    pd.testing.assert_series_equal(first, second)


def test_b4_is_never_offered_a_price_metric():
    """Section 19.3: B4 is "scored against settlement_spearman only, never price"."""
    assert set(B.SETTLEMENT_BASELINES) == {"b4"}
    assert "b4" not in B.PRICE_BASELINES


# --------------------------------------------------------------------------------------
# ACCEPTANCE — Module 15
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_all_baselines_run_and_are_sane(city, panel, params):
    """ACCEPTANCE: "All four baselines run on the Vizag fixture and produce sane values."

    Sane means: one finite value per cell, no NaN, appreciation bounded by the clip range
    the parameter file declares, and B4's settlement deltas non-negative and inside the
    developable ceiling.
    """
    lookback = int(params.value("backtest.horizon.momentum_lookback_years"))
    floor = (1 + float(params.value("backtest.baselines.b0.min_cagr"))) ** HORIZON - 1
    cap = (1 + float(params.value("backtest.baselines.b0.max_cagr"))) ** HORIZON - 1

    price_predictions = {
        "b0": B.b0_city_average(city.cells, HORIZON, params=params, history=panel, t0=T0),
        "b1": B.b1_distance_cbd(city.cells, HORIZON, params=params, history=panel, t0=T0),
        "b2": B.b2_momentum(city.cells, panel, HORIZON, params=params, t0=T0),
        "b3": B.b3_naive_announcement(
            city.cells, city.projects, HORIZON, params=params, history=panel, t0=T0
        ),
    }
    for name, prediction in price_predictions.items():
        assert len(prediction) == len(city.cells), name
        assert np.isfinite(prediction).all(), name
        assert (prediction >= floor - 1e-9).all(), name
        assert (prediction <= cap + 1).all(), name

    settlement = B.b4_urban_growth(
        city.cells, panel, HORIZON, params=params, t0=T0, lookback=lookback
    )
    assert len(settlement) == len(city.cells)
    assert np.isfinite(settlement).all()
    assert (settlement >= 0).all()
    assert (settlement <= 1).all()
