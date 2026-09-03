"""Tests for the backtest harness — freeze, scoring and the ship gate (spec Section 19).

The Section 19 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "``freeze.py`` raises when a parameter's ``fitted_on.data_through`` exceeds ``t0``"
  -> ``test_acc_freeze_raises_on_post_t0_fitted_parameter``
* "The dead-project assertion fires on a contaminated fixture"
  -> ``test_acc_dead_project_assertion_fires_on_contaminated_fixture``
* "All four baselines run on the Vizag fixture and produce sane values"
  -> ``tests/unit/test_baselines.py::test_acc_all_baselines_run_and_are_sane``
* "``ship_gate`` returns False on a deliberately weak model fixture"
  -> ``test_acc_ship_gate_fails_a_weak_model``

Beyond the acceptance block, the adversarial section below plants each of the three
look-ahead vectors Section 21 warns about — a post-freeze parameter, a post-freeze price
observation and a post-freeze project stage change — and asserts each one is caught. Those
three tests are the ones to keep working if anything here is ever refactored: they are the
difference between a backtest and a number.

**The gate has never been run against real data in this environment.** There is no
historical panel here, so ``ufe backtest gate`` cannot produce a genuine verdict. Every
gate test below runs on synthetic scorecards with a known answer, and the real-data path is
marked ``@pytest.mark.needs_data`` and skipped. Nothing in this file should be read as
evidence that the model beats momentum.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from ufe.backtest import freeze as F
from ufe.backtest import gate as G
from ufe.backtest import score as S
from ufe.params import Params, load_params

from tests.fixtures.synthetic import build_city
from tests.unit.test_baselines import (
    HORIZON,
    T0,
    build_panel,
    realised_appreciation,
)

CITY = "vizag"

#: Every fitted leaf in the landed parameter tree declares `data_through: 2019`.
FITTED_THROUGH = 2019
#: An origin before the fit window: freezing here is look-ahead and must be refused.
CONTAMINATED_T0 = 2016

SEED = 20240101
N_HOLDOUT_CITIES = 3

#: There is no real historical panel here and the repo has no root conftest wiring marker
#: filters, so the `needs_data` test is additionally guarded by an explicit opt-in
#: environment variable, matching the convention in tests/unit/test_ingest.py.
REAL_PANEL_ENV = "UFE_REAL_PANEL_ROOT"


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params(city=CITY)


@pytest.fixture(scope="module")
def city():
    return build_city()


@pytest.fixture(scope="module")
def panel(city):
    return build_panel(city.cells)


def clean_vintage(t0: int = T0) -> dict[str, int]:
    """A declared cell-state vintage that sits exactly at ``t0``."""
    return {
        "builtup_frac": t0,
        "nightlight": t0,
        "population": t0,
        "price_res_inr_sqft": t0,
    }


def rebuilt_params(params: Params, tree: dict) -> Params:
    """A `Params` around a mutated tree, for planting contamination."""
    return Params(
        city_id=params.city_id,
        city_class=params.city_class,
        resolved=tree,
        deviations=[],
        class_defaults_applied=[],
        source_files=[],
        city_config=params.city_config,
    )


def with_data_through(params: Params, path: str, data_through: int) -> tuple[Params, str]:
    """Plant a post-freeze fit date on one leaf and return the new Params and its path."""
    tree = copy.deepcopy(params.resolved)
    node = tree
    for token in path.split("."):
        node = node[int(token)] if isinstance(node, list) else node[token]
    node["fitted_on"] = {"cities": ["planted"], "data_through": data_through}
    return rebuilt_params(params, tree), path


def strip_provenance(params: Params, path: str) -> Params:
    """Remove every provenance record governing ``path``, inherited ones included."""
    tree = copy.deepcopy(params.resolved)
    node = tree
    for token in path.split("."):
        if isinstance(node, dict):
            node.pop("_provenance", None)
        node = node[int(token)] if isinstance(node, list) else node[token]
    node.pop("fitted_on", None)
    node.pop("citation", None)
    return rebuilt_params(params, tree)


def build_pipeline(t0: int = T0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small project pipeline with pre-``t0`` stage history and post-``t0`` outcomes.

    Three projects announced before ``t0``, one announced after it (which the freeze must
    drop), and outcomes resolved after ``t0`` — including one abandonment, which is the
    whole point of Section 19.2.
    """
    projects = pd.DataFrame(
        {
            "project_id": ["p-alive", "p-dead", "p-slow", "p-future"],
            "archetype": ["metro_rail"] * 4,
            "geom": ["POINT (83.22 17.72)"] * 4,
            "announced_date": pd.to_datetime(
                [f"{t0 - 4}-03-01", f"{t0 - 3}-06-01", f"{t0 - 2}-09-01", f"{t0 + 2}-01-01"]
            ),
            "stated_completion": pd.to_datetime(
                [f"{t0 + 3}-01-01", f"{t0 + 4}-01-01", f"{t0 + 6}-01-01", f"{t0 + 8}-01-01"]
            ),
            "stage": ["construction", "funded", "feasibility", "announced"],
            "stage_asof": pd.to_datetime(
                [f"{t0 - 1}-01-01", f"{t0 - 1}-01-01", f"{t0 - 1}-01-01", f"{t0 + 2}-01-01"]
            ),
        }
    )
    history = pd.DataFrame(
        {
            "project_id": [
                "p-alive", "p-dead", "p-slow", "p-future",
                "p-alive", "p-dead", "p-slow",
            ],
            "field": ["stage"] * 4 + ["outcome"] * 3,
            "old_value": [
                "announced", "announced", "announced", None,
                None, None, None,
            ],
            "new_value": [
                "construction", "funded", "feasibility", "announced",
                "completed", "abandoned", "stalled",
            ],
            "changed_at": pd.to_datetime(
                [
                    f"{t0 - 1}-01-01", f"{t0 - 1}-01-01", f"{t0 - 1}-01-01", f"{t0 + 2}-01-01",
                    f"{t0 + 4}-01-01", f"{t0 + 3}-01-01", f"{t0 + 5}-01-01",
                ]
            ),
            "source_url": ["https://example.invalid/x"] * 7,
            "changed_by": ["ingest"] * 7,
        }
    )
    return projects, history


def do_freeze(city, panel, params, *, t0: int = T0, **kwargs):
    projects, history = build_pipeline(t0)
    defaults = dict(
        city_id=CITY,
        t0=t0,
        cells=city.cells,
        cells_history=panel,
        projects=projects,
        project_history=history,
        params=params,
        vintage=clean_vintage(t0),
    )
    defaults.update(kwargs)
    return F.freeze(**defaults)


# --------------------------------------------------------------------------------------
# ADVERSARIAL — the three look-ahead vectors of Section 21
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_freeze_raises_on_post_t0_fitted_parameter(city, panel, params):
    """ACCEPTANCE: freeze raises when `fitted_on.data_through` exceeds `t0`.

    No planting is needed to show this: the landed parameter tree really is fitted on data
    through 2019, so any freeze before 2019 really is look-ahead and really must be refused.
    """
    with pytest.raises(F.LookAheadError) as raised:
        do_freeze(city, panel, params, t0=CONTAMINATED_T0)
    message = str(raised.value)
    assert f"data_through={FITTED_THROUGH}" in message
    assert f"t0={CONTAMINATED_T0}" in message
    assert "archetypes." in message


def test_adversarial_planted_post_freeze_parameter_is_caught(params):
    """Vector 1: a single parameter re-fitted after the freeze date."""
    path = "credibility.stage_probability.announced"
    planted, _ = with_data_through(params, path, T0 + 3)
    # Clean at an origin after the planted fit date...
    F.assert_parameter_provenance(planted, T0 + 3)
    # ...and refused at the freeze origin, naming the offending leaf.
    with pytest.raises(F.LookAheadError, match=path):
        F.assert_parameter_provenance(planted, T0)


def test_adversarial_unprovenanced_global_parameter_is_refused(params):
    """A global leaf with neither `fitted_on` nor `citation` cannot be shown to predate t0."""
    path = "credibility.stage_probability.announced"
    stripped = strip_provenance(params, path)
    with pytest.raises(F.LookAheadError, match="neither"):
        F.assert_parameter_provenance(stripped, T0)


def test_adversarial_planted_post_freeze_price_observation_is_caught(city, panel, params):
    """Vector 2: a price surface stamped with a vintage after the freeze date.

    A `cells` frame carries no year, so the only way this is detectable is by requiring the
    vintage to be declared. Both halves are tested: a post-t0 declaration is refused, and an
    *undeclared* vintage is refused rather than defaulted — silently assuming t0 would let
    the same contamination straight through.
    """
    contaminated = clean_vintage()
    contaminated["price_res_inr_sqft"] = T0 + 2
    with pytest.raises(F.LookAheadError, match="price_res_inr_sqft"):
        do_freeze(city, panel, params, vintage=contaminated)

    undeclared = clean_vintage()
    undeclared.pop("nightlight")
    with pytest.raises(F.LookAheadError, match="nightlight"):
        do_freeze(city, panel, params, vintage=undeclared)


def test_adversarial_post_freeze_price_row_inside_the_snapshot_is_caught(city, panel, params):
    """The same vector at panel level: a `cells_history` row dated after t0."""
    snapshot = do_freeze(city, panel, params)
    smuggled = pd.concat(
        [
            snapshot.cells_history,
            panel.loc[panel["year"] == T0 + 2],
        ],
        ignore_index=True,
    )
    contaminated = F.FrozenSnapshot(
        city_id=snapshot.city_id,
        t0=snapshot.t0,
        cells=snapshot.cells,
        cells_history=smuggled,
        projects=snapshot.projects,
        project_history=snapshot.project_history,
        outcomes=snapshot.outcomes,
        params=snapshot.params,
        vintage=snapshot.vintage,
        price_test_enabled=snapshot.price_test_enabled,
    )
    with pytest.raises(F.LookAheadError, match="cells_history"):
        F.assert_no_lookahead(contaminated)


def test_adversarial_planted_post_freeze_stage_change_is_caught(city, panel, params):
    """Vector 3: a project whose stage advanced after the freeze date.

    The project's record says `construction`, as of two years after t0. There is no pre-t0
    stage transition to roll it back to, so its stage at t0 is unknowable and the freeze
    refuses rather than importing tomorrow's stage into today's pipeline.
    """
    projects, history = build_pipeline()
    projects.loc[projects["project_id"] == "p-slow", "stage"] = "construction"
    projects.loc[projects["project_id"] == "p-slow", "stage_asof"] = pd.Timestamp(
        f"{T0 + 2}-01-01"
    )
    history = history.loc[history["project_id"] != "p-slow"]

    with pytest.raises(F.LookAheadError, match="p-slow"):
        do_freeze(city, panel, params, projects=projects, project_history=history)


def test_stage_is_rolled_back_to_its_t0_value_when_history_allows(city, panel, params):
    """With a pre-t0 transition on record, the stage is restated, not refused."""
    projects, history = build_pipeline()
    projects.loc[projects["project_id"] == "p-slow", "stage"] = "construction"
    projects.loc[projects["project_id"] == "p-slow", "stage_asof"] = pd.Timestamp(
        f"{T0 + 2}-01-01"
    )
    snapshot = do_freeze(city, panel, params, projects=projects, project_history=history)
    frozen = snapshot.projects.set_index("project_id")
    assert frozen.loc["p-slow", "stage"] == "feasibility"
    assert pd.Timestamp(frozen.loc["p-slow", "stage_asof"]).year <= T0


# --------------------------------------------------------------------------------------
# freeze — the rest of Section 19.1
# --------------------------------------------------------------------------------------


def test_freeze_is_clean_at_an_origin_after_the_fit_window(city, panel, params):
    snapshot = do_freeze(city, panel, params)
    assert snapshot.t0 == T0
    assert snapshot.provenance
    assert max(r.data_through for r in snapshot.provenance if r.is_fitted) <= T0
    F.assert_no_lookahead(snapshot)


def test_freeze_admits_only_projects_announced_on_or_before_t0(city, panel, params):
    snapshot = do_freeze(city, panel, params)
    assert set(snapshot.projects["project_id"]) == {"p-alive", "p-dead", "p-slow"}
    assert "p-future" not in set(snapshot.project_history["project_id"])


def test_freeze_slices_the_panel_to_t0(city, panel, params):
    snapshot = do_freeze(city, panel, params)
    assert int(snapshot.cells_history["year"].max()) <= T0
    assert int(snapshot.project_history["changed_at"].dt.year.max()) <= T0


def test_freeze_keeps_outcomes_out_of_the_frozen_pipeline(city, panel, params):
    """Outcomes are the answer key. The model must not see how the story ends."""
    snapshot = do_freeze(city, panel, params)
    assert "outcome" not in snapshot.projects.columns
    assert snapshot.outcomes.loc["p-dead"] == "abandoned"
    # ...and a snapshot that does carry them is refused.
    leaked = snapshot.projects.copy()
    leaked["outcome"] = "completed"
    with pytest.raises(F.LookAheadError, match="outcome"):
        F.assert_no_lookahead(
            F.FrozenSnapshot(
                city_id=snapshot.city_id,
                t0=snapshot.t0,
                cells=snapshot.cells,
                cells_history=snapshot.cells_history,
                projects=leaked,
                project_history=snapshot.project_history,
                outcomes=snapshot.outcomes,
                params=snapshot.params,
                vintage=snapshot.vintage,
                price_test_enabled=snapshot.price_test_enabled,
            )
        )


def test_freeze_uses_the_t0_observation_for_history_tracked_columns(city, panel, params):
    snapshot = do_freeze(city, panel, params)
    expected = panel.loc[panel["year"] == T0].set_index("h3")["builtup_frac"]
    got = snapshot.cells.set_index("h3")["builtup_frac"]
    assert np.allclose(got.to_numpy(), expected.reindex(got.index).to_numpy())


def test_freeze_hash_is_stable_and_sensitive(city, panel, params):
    first = do_freeze(city, panel, params)
    second = do_freeze(city, panel, params)
    assert first.freeze_hash == second.freeze_hash
    other = do_freeze(city, panel, params, t0=T0 + 1)
    assert other.freeze_hash != first.freeze_hash
    assert first.manifest()["params_hash"] == params.hash


def test_freeze_refuses_a_vintage_that_lags_t0_too_far(city, panel, params):
    stale = clean_vintage()
    stale["population"] = T0 - int(params.value("backtest.freeze.vintage.max_lag_years")) - 1
    with pytest.raises(F.LookAheadError, match="lags"):
        do_freeze(city, panel, params, vintage=stale)


def test_price_test_is_skipped_when_the_t0_surface_is_too_sparse(city, panel, params):
    """Section 19.1: "where not, the price test is skipped ... only the settlement test runs"."""
    sparse = panel.copy()
    keep = set(city.cells["h3"].iloc[:2])
    sparse.loc[~sparse["h3"].isin(keep), "price_res_inr_sqft"] = np.nan
    # The reconstructed t0 surface is the history value where there is one and the cell
    # frame's own value otherwise, so both have to be blank for the surface to be missing.
    unpriced = city.cells.copy()
    unpriced.loc[~unpriced["h3"].isin(keep), "price_res_inr_sqft"] = np.nan
    snapshot = do_freeze(city, panel, params, cells=unpriced, cells_history=sparse)
    assert snapshot.price_test_enabled is False
    assert do_freeze(city, panel, params).price_test_enabled is True


# --------------------------------------------------------------------------------------
# Section 19.2 — the dead-project requirement
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acc_dead_project_assertion_fires_on_contaminated_fixture(city, panel, params):
    """ACCEPTANCE: the dead-project assertion fires on a contaminated fixture.

    "Contaminated" here means survivorship-filtered: the pipeline has been rebuilt from
    today's records, so everything in it completed and the cancellations are gone. That
    pipeline makes the credibility layer untestable, and Section 19.2 says the assertion is
    not optional.
    """
    projects, history = build_pipeline()
    survivors = history.copy()
    survivors.loc[survivors["field"] == "outcome", "new_value"] = "completed"

    with pytest.raises(F.SurvivorshipContamination) as raised:
        do_freeze(city, panel, params, projects=projects, project_history=survivors)
    assert "no failed projects" in str(raised.value)
    assert "contaminated" in str(raised.value).lower()


def test_dead_project_requirement_passes_when_a_project_died(city, panel, params):
    snapshot = do_freeze(city, panel, params)
    assert snapshot.dead_projects["abandoned"] >= int(
        params.value("backtest.freeze.dead_projects.min_abandoned_count")
    )
    assert snapshot.dead_projects["abandoned_ids"] == ["p-dead"]


def test_dead_project_requirement_rejects_a_token_single_death(params):
    """One dead project in a large pipeline is survivorship in all but name."""
    ids = [f"p-{index:03d}" for index in range(100)]
    outcomes = pd.Series(["completed"] * len(ids), index=ids)
    outcomes.iloc[0] = "abandoned"
    with pytest.raises(F.SurvivorshipContamination, match="survivorship"):
        F.assert_dead_projects(ids, outcomes, params)


def test_dead_project_requirement_rejects_an_empty_pipeline(params):
    with pytest.raises(F.SurvivorshipContamination, match="empty"):
        F.assert_dead_projects([], pd.Series(dtype=object), params)


def test_outcome_vocabulary_is_validated(params):
    history = pd.DataFrame(
        {
            "project_id": ["p"],
            "field": ["outcome"],
            "new_value": ["vanished"],
            "changed_at": pd.to_datetime([f"{T0 + 1}-01-01"]),
        }
    )
    with pytest.raises(Exception, match="unknown"):
        F.derive_outcomes(history, params)


# --------------------------------------------------------------------------------------
# Section 19.4 — scoring
# --------------------------------------------------------------------------------------


def perfect_and_useless(actual: pd.Series, seed: int = SEED):
    """A prediction that ranks the answer key exactly, and one that ranks noise."""
    rng = np.random.default_rng(seed)
    perfect = actual.copy()
    perfect.name = "perfect"
    useless = pd.Series(rng.permutation(actual.to_numpy()), index=actual.index, name="useless")
    return perfect, useless


@pytest.fixture(scope="module")
def zone_actual(city, panel):
    cell_actual = realised_appreciation(panel, T0, HORIZON)
    return S.to_zone(cell_actual, city.cells)


def test_spearman_is_one_for_a_perfect_ranking(zone_actual):
    perfect, _ = perfect_and_useless(zone_actual)
    correlation, pvalue = S.spearman(perfect, zone_actual)
    assert correlation == pytest.approx(1.0)
    assert pvalue is not None


def test_top3_precision_counts_predicted_winners_in_the_actual_top_decile(zone_actual, params):
    perfect, useless = perfect_and_useless(zone_actual)
    k = int(params.value("backtest.scoring.top_k_zones"))
    card = S.score(perfect, zone_actual, params=params, years=HORIZON)
    assert card.top3_precision == pytest.approx(1.0)
    assert len(zone_actual) >= k


def test_mape_cagr_is_zero_for_a_perfect_prediction(zone_actual, params):
    perfect, _ = perfect_and_useless(zone_actual)
    card = S.score(perfect, zone_actual, params=params, years=HORIZON)
    assert card.mape_cagr == pytest.approx(0.0, abs=1e-9)


def test_band_coverage_and_reliability_diagram(zone_actual, params):
    """Bands that bracket every zone give coverage 1; a nested pair gives a diagram."""
    wide = pd.DataFrame(
        {
            "p10": zone_actual - 1,
            "p90": zone_actual + 1,
            "p25": zone_actual - 1,
            "p75": zone_actual + 1,
        },
        index=zone_actual.index,
    )
    card = S.score(zone_actual, zone_actual, wide, params=params, years=HORIZON)
    assert card.band_coverage == pytest.approx(1.0)
    assert len(card.reliability) == 2
    nominals = sorted(point[0] for point in card.reliability)
    assert nominals == pytest.approx([0.5, 0.8])

    narrow = pd.DataFrame(
        {"p10": zone_actual + 1, "p90": zone_actual + 2}, index=zone_actual.index
    )
    assert S.score(
        zone_actual, zone_actual, narrow, params=params, years=HORIZON
    ).band_coverage == pytest.approx(0.0)


def test_overheat_precision(zone_actual, params):
    """Of the zones flagged overheated, the fraction that underperformed the city median."""
    ranked = zone_actual.sort_values()
    losers = ranked.index[: len(ranked) // 4]
    flags = pd.Series(False, index=zone_actual.index)
    flags.loc[losers] = True
    card = S.score(
        zone_actual, zone_actual, params=params, years=HORIZON, overheat_flags=flags
    )
    assert card.overheat_precision == pytest.approx(1.0)


def test_beat_b2_and_its_bootstrap_ci(zone_actual, params):
    perfect, useless = perfect_and_useless(zone_actual)
    card = S.score(perfect, zone_actual, params=params, years=HORIZON, b2_pred=useless)
    assert card.beat_b2 is not None and card.beat_b2 > 0
    assert card.beat_b2_ci_lower is not None
    assert card.beat_b2_ci_lower <= card.beat_b2 <= card.beat_b2_ci_upper

    # Symmetry: the useless model loses to momentum by the same margin.
    reverse = S.score(useless, zone_actual, params=params, years=HORIZON, b2_pred=perfect)
    assert reverse.beat_b2 == pytest.approx(-card.beat_b2)
    assert reverse.beat_b2_ci_upper < 0


def test_bootstrap_ci_is_deterministic(zone_actual, params):
    perfect, useless = perfect_and_useless(zone_actual)
    first = S.bootstrap_difference_ci(perfect, useless, zone_actual, params)
    second = S.bootstrap_difference_ci(perfect, useless, zone_actual, params)
    assert first == second


def test_lookahead_alarm_fires_on_a_suspiciously_high_spearman(zone_actual, params):
    """Section 21: "Look-ahead in backtest — symptom: suspiciously high Spearman, >0.8"."""
    alarm = float(params.value("backtest.scoring.lookahead.spearman_alarm"))
    perfect, useless = perfect_and_useless(zone_actual)
    hot = S.score(perfect, zone_actual, params=params, years=HORIZON)
    assert hot.spearman > alarm
    assert hot.suspicious_spearman is True
    assert any("LOOK-AHEAD ALARM" in note for note in hot.notes)
    assert "Look-ahead alarm" in hot.to_markdown()

    cool = S.score(useless, zone_actual, params=params, years=HORIZON)
    assert cool.suspicious_spearman is False


def test_settlement_spearman_and_beat_b4_are_reported(zone_actual, params):
    """Section 19.3: B4 is scored on settlement only — and reported, never gated."""
    settlement = zone_actual.copy()
    model = settlement.copy()
    b4 = pd.Series(
        np.random.default_rng(SEED).permutation(settlement.to_numpy()),
        index=settlement.index,
    )
    card = S.score(
        zone_actual,
        zone_actual,
        params=params,
        years=HORIZON,
        settlement_pred=model,
        settlement_actual=settlement,
        b4_settlement_pred=b4,
    )
    assert card.settlement_spearman == pytest.approx(1.0)
    assert card.beat_b4 is not None and card.beat_b4 > 0
    assert any("never gated" in note for note in card.notes)


def test_price_test_disabled_reports_settlement_only(zone_actual, params):
    card = S.score(
        zone_actual,
        zone_actual,
        params=params,
        years=HORIZON,
        settlement_pred=zone_actual,
        settlement_actual=zone_actual,
        price_test_enabled=False,
    )
    assert card.spearman is None
    assert card.mape_cagr is None
    assert card.settlement_spearman is not None
    assert "price test skipped" in " ".join(card.notes)


def test_scorecard_round_trips_through_json_and_renders_markdown(zone_actual, params):
    perfect, useless = perfect_and_useless(zone_actual)
    card = S.score(perfect, zone_actual, params=params, years=HORIZON, b2_pred=useless)
    restored = S.ScoreCard.from_dict(json.loads(card.to_json()))
    assert restored.spearman == pytest.approx(card.spearman)
    assert restored.beat_b2 == pytest.approx(card.beat_b2)
    markdown = card.to_markdown()
    assert "spearman" in markdown and "beat_b2" in markdown


# --------------------------------------------------------------------------------------
# Section 19.6 — the gate
# --------------------------------------------------------------------------------------


def make_card(city_id: str, **kwargs) -> S.ScoreCard:
    """A scorecard with every gated field set to a passing value unless overridden."""
    defaults = dict(
        city_id=city_id,
        t0=T0,
        horizon_years=HORIZON,
        n_zones=100,
        spearman=0.65,
        beat_b2=0.08,
        beat_b2_ci_lower=0.02,
        beat_b2_ci_upper=0.14,
        band_coverage=0.82,
        top3_precision=0.67,
        mape_cagr=0.2,
        settlement_spearman=0.5,
    )
    defaults.update(kwargs)
    return S.ScoreCard(**defaults)


def holdout(**kwargs) -> list[S.ScoreCard]:
    return [make_card(f"holdout-{index}", **kwargs) for index in range(N_HOLDOUT_CITIES)]


@pytest.mark.acceptance
def test_acc_ship_gate_fails_a_weak_model():
    """ACCEPTANCE: `ship_gate` returns False on a deliberately weak model fixture.

    Weak means what Section 19.6 says it means: it does not out-rank momentum. Every other
    criterion here is set to a passing value, so the FAIL is attributable.
    """
    params = load_params(city=CITY)
    weak = holdout(spearman=0.31, beat_b2=-0.05, beat_b2_ci_lower=-0.12, beat_b2_ci_upper=-0.01)
    result = G.ship_gate(weak, params)

    assert result.passed is False
    assert bool(result) is False
    assert result.verdict == "FAIL"
    failing = {criterion.name for criterion in result.failing}
    assert {"median_spearman", "median_beat_b2", "beat_b2_ci_lower"} <= failing
    assert "Nothing ships on FAIL" in result.reasoning()


def test_ship_gate_passes_a_strong_model_on_three_holdout_cities(params):
    result = G.ship_gate(holdout(), params)
    assert result.passed is True
    assert result.n_cities == N_HOLDOUT_CITIES
    assert result.reasoning().startswith("PASS")


def test_ship_gate_requires_three_holdout_cities(params):
    """Section 23 item 3: PASS on at least three hold-out cities."""
    minimum = int(params.value("backtest.gate.min_holdout_cities"))
    assert minimum == N_HOLDOUT_CITIES
    result = G.ship_gate(holdout()[: minimum - 1], params)
    assert result.passed is False
    assert "holdout_cities" in {criterion.name for criterion in result.failing}


def test_ship_gate_enforces_both_ends_of_the_band_coverage_window(params):
    low = float(params.value("backtest.gate.min_median_band_coverage"))
    high = float(params.value("backtest.gate.max_median_band_coverage"))
    for coverage in (low - 0.1, high + 0.05):
        result = G.ship_gate(holdout(band_coverage=coverage), params)
        assert result.passed is False
        assert "median_band_coverage" in {c.name for c in result.failing}
    assert G.ship_gate(holdout(band_coverage=(low + high) / 2), params).passed


def test_ship_gate_treats_an_unmeasured_metric_as_a_failure(params):
    """A metric that was never measured never passes its criterion."""
    result = G.ship_gate(holdout(beat_b2=None, beat_b2_ci_lower=None), params)
    assert result.passed is False
    names = {criterion.name for criterion in result.failing}
    assert {"median_beat_b2", "beat_b2_ci_lower"} <= names


def test_ship_gate_refuses_an_underpowered_scorecard(params):
    result = G.ship_gate(holdout(n_zones=1), params)
    assert result.passed is False
    assert "scorecards_powered" in {criterion.name for criterion in result.failing}


def test_ship_gate_reports_but_does_not_gate_beat_b4(params):
    """Section 19.6: `beat_b4` is reported, not gated — losing on settlement is not fatal."""
    assert bool(params.value("backtest.gate.gate_on_beat_b4")) is False
    result = G.ship_gate(holdout(beat_b4=-0.2), params)
    assert result.passed is True
    assert result.reported["median_beat_b4_settlement"] == pytest.approx(-0.2)
    assert any("loses to the in-house logistic-CA baseline" in w for w in result.warnings)


def test_ship_gate_surfaces_the_lookahead_alarm_in_its_reasoning(params):
    result = G.ship_gate(holdout(spearman=0.93, suspicious_spearman=True), params)
    assert result.passed is True
    assert any("LOOK-AHEAD ALARM" in warning for warning in result.warnings)
    assert "LOOK-AHEAD ALARM" in result.reasoning()


def test_ship_gate_lists_every_failing_criterion_not_just_the_first(params):
    result = G.ship_gate(holdout(spearman=0.1, band_coverage=0.2, beat_b2=-1.0), params)
    assert len(result.failing) >= 3
    for criterion in result.failing:
        assert criterion.line().startswith("[FAIL]")


# --------------------------------------------------------------------------------------
# End to end on synthetic panels with a known answer
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_pair(city, panel, params):
    """Two constructed worlds: one where the model must beat momentum, one where it must not.

    Both share the same answer key. In the first, the model's ranking is the answer and
    momentum is noise; in the second, momentum is the answer and the model is noise. Neither
    is a claim about the real engine — they are a calibration of the harness itself, which
    is the only thing that can be validated without a historical panel.
    """
    from ufe.backtest import baselines as B

    cell_actual = realised_appreciation(panel, T0, HORIZON)
    actual = S.to_zone(cell_actual, city.cells)
    momentum = S.to_zone(
        B.b2_momentum(city.cells, panel, HORIZON, params=params, t0=T0), city.cells
    )
    noise = pd.Series(
        np.random.default_rng(SEED).permutation(actual.to_numpy()),
        index=actual.index,
        name="noise",
    )
    # Bands calibrated to land inside the gate's 0.70-0.90 coverage window: exactly 80% of
    # zones are bracketed and the rest deliberately are not. Perfectly-covering bands would
    # FAIL the gate, and rightly so — a band that always contains the answer says nothing.
    covered = np.zeros(len(actual), dtype=bool)
    covered[: int(round(len(actual) * float(params.value("backtest.scoring.bands.nominal_coverage"))))] = True
    bands = pd.DataFrame(
        {
            "p10": np.where(covered, actual - 1, actual + 1),
            "p90": np.where(covered, actual + 1, actual + 2),
        },
        index=actual.index,
    )
    winner = S.score(
        actual, actual, bands, params=params, years=HORIZON, b2_pred=noise,
        city_id="win", t0=T0,
    )
    loser = S.score(
        noise, actual, bands, params=params, years=HORIZON, b2_pred=actual,
        city_id="lose", t0=T0,
    )
    return winner, loser


def test_synthetic_world_where_the_model_must_beat_momentum(scored_pair, params):
    winner, _ = scored_pair
    assert winner.spearman == pytest.approx(1.0)
    assert winner.beat_b2 > 0
    assert winner.beat_b2_ci_lower > 0
    cards = [
        S.ScoreCard.from_dict({**winner.to_dict(), "city_id": f"holdout-{index}"})
        for index in range(N_HOLDOUT_CITIES)
    ]
    assert G.ship_gate(cards, params).passed is True


def test_synthetic_world_where_the_model_must_not_beat_momentum(scored_pair, params):
    _, loser = scored_pair
    assert loser.beat_b2 < 0
    assert loser.beat_b2_ci_upper < 0
    cards = [
        S.ScoreCard.from_dict({**loser.to_dict(), "city_id": f"holdout-{index}"})
        for index in range(N_HOLDOUT_CITIES)
    ]
    result = G.ship_gate(cards, params)
    assert result.passed is False
    assert "median_beat_b2" in {criterion.name for criterion in result.failing}


# --------------------------------------------------------------------------------------
# CLI (Section 19.6: prints PASS or FAIL, exits non-zero on FAIL)
# --------------------------------------------------------------------------------------


def run_cli(argv: list[str]):
    from typer.testing import CliRunner

    from ufe.backtest_cli import app

    return CliRunner().invoke(app, argv)


def test_cli_gate_fails_loudly_without_a_historical_panel():
    """There is no real panel here, and the gate must say so rather than pass."""
    result = run_cli(["gate", "--city", CITY])
    assert result.exit_code != 0
    assert "FAIL" in result.stdout
    assert "No historical panel" in result.stdout


def test_cli_gate_prints_pass_and_exits_zero_on_a_passing_scorecard_set(tmp_path):
    path = tmp_path / "scorecards.json"
    path.write_text(json.dumps([card.to_dict() for card in holdout()]))
    result = run_cli(["gate", "--city", CITY, "--scorecards", str(path)])
    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert "median_beat_b2" in result.stdout


def test_cli_gate_exits_non_zero_on_fail(tmp_path):
    weak = holdout(spearman=0.1, beat_b2=-0.2, beat_b2_ci_lower=-0.3, beat_b2_ci_upper=-0.1)
    path = tmp_path / "weak.json"
    path.write_text(json.dumps([card.to_dict() for card in weak]))
    result = run_cli(["gate", "--city", CITY, "--scorecards", str(path)])
    assert result.exit_code != 0
    assert "FAIL" in result.stdout
    assert "Nothing ships on FAIL" in result.stdout


def test_cli_gate_writes_markdown_reports(tmp_path):
    cards = tmp_path / "cards.json"
    cards.write_text(json.dumps([card.to_dict() for card in holdout()]))
    report = tmp_path / "report.md"
    result = run_cli(
        ["gate", "--city", CITY, "--scorecards", str(cards), "--markdown", str(report)]
    )
    assert result.exit_code == 0
    assert report.exists()
    assert "Backtest scorecard" in report.read_text()


def test_cli_robustness_commands_refuse_rather_than_improvise():
    for argv in (
        ["rolling", "--city", CITY],
        ["loco", "--city", CITY],
        ["ablate", "--city", CITY],
        ["sobol", "--city", CITY],
        ["freeze", "--city", CITY, "--t0", str(T0)],
    ):
        result = run_cli(argv)
        assert result.exit_code != 0, argv
        assert "No historical panel" in result.stdout, argv


def test_cli_ablate_rejects_an_unknown_layer():
    result = run_cli(["ablate", "--city", CITY, "--layers", "l9"])
    assert result.exit_code != 0
    assert "unknown layer" in result.stdout


def test_cli_is_mounted_on_the_root_app():
    from ufe.cli import MOUNTED

    assert "backtest" in MOUNTED


# --------------------------------------------------------------------------------------
# Contract compliance
# --------------------------------------------------------------------------------------


def test_backtest_never_imports_ufe_ai():
    """CONTRACT.md rule 4 / Section 17: no LLM calls at simulation time."""
    import ast
    import pathlib

    import ufe.backtest_cli

    package = pathlib.Path(F.__file__).parent
    modules = sorted(package.glob("*.py")) + [pathlib.Path(ufe.backtest_cli.__file__)]
    for module in modules:
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("ufe.ai") for a in node.names), module
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("ufe.ai"), module


def test_no_numeric_literals_in_the_backtest_package():
    """Section 0.1 rule 3: only 0, 1 and array indices may appear in the Python.

    Enforced by reading the AST rather than by discipline, because "just don't type a
    number" is exactly the rule everybody breaks at 2am. Subscripts are exempt (array
    indices are allowed), as are dataclass/`enumerate` bookkeeping via the named `_ZERO`
    and `_ONE` constants.
    """
    import ast
    import pathlib

    import ufe.backtest_cli

    allowed = {0, 1}
    package = pathlib.Path(F.__file__).parent
    modules = sorted(package.glob("*.py")) + [pathlib.Path(ufe.backtest_cli.__file__)]
    offences: list[str] = []
    for module in modules:
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # Array indices are explicitly permitted.
                for child in ast.walk(node.slice):
                    child._ufe_index = True  # type: ignore[attr-defined]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if getattr(node, "_ufe_index", False):
                continue
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                continue
            if node.value in allowed:
                continue
            offences.append(f"{module.name}:{node.lineno}: {node.value}")
    assert not offences, "numeric literals outside {0, 1}:\n  " + "\n  ".join(offences)


# --------------------------------------------------------------------------------------
# The real-data path — never run here
# --------------------------------------------------------------------------------------


@pytest.mark.needs_data
@pytest.mark.skipif(
    os.environ.get(REAL_PANEL_ENV) is None,
    reason=(
        "no real historical panel in this environment; the ship gate is UNRUN against real "
        f"data. Set {REAL_PANEL_ENV} to point at one to run it."
    ),
)
def test_real_backtest_gate_on_three_holdout_cities():
    """The only test that can honestly answer "does the model beat momentum?".

    It needs a real historical panel — reconstructed price surfaces, project pipelines with
    stage history and recorded outcomes, for at least three hold-out cities. None of that
    exists in this environment, so this test is skipped by default and **the gate is unrun
    against real data**. Everything else in this file validates the harness, not the model.
    """
    from ufe.store import db

    params = load_params(city=CITY)
    con = db.connect(read_only=True)
    cells = db.read_table(con, "cells")
    history = db.read_table(con, "cells_history")
    projects = db.read_table(con, "projects")
    project_history = db.read_table(con, "project_history")

    origins = list(params.value("backtest.robustness.rolling_origins"))
    horizon = int(params.value("backtest.horizon.default_years"))
    cards: list[S.ScoreCard] = []
    for t0 in origins:
        snapshot = F.freeze(
            city_id=CITY,
            t0=t0,
            cells=cells,
            cells_history=history,
            projects=projects,
            project_history=project_history,
            params=params,
            vintage={key: t0 for key in params.value("backtest.freeze.vintage.required_keys")},
        )
        assert snapshot.dead_projects["abandoned"] > 0
    result = G.ship_gate(cards, params)
    assert result.passed, result.reasoning()
