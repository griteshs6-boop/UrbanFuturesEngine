"""Tests for Module 12 — Monte Carlo (spec Section 16).

The Section 16 ACCEPTANCE block maps onto the ``@pytest.mark.acceptance`` tests here:

* "Band calibration on synthetic data: generate truth from a known parameter draw, run MC,
  verify ~80% of truths fall in the p10-p90 band."
  -> ``test_acc_band_calibration_on_synthetic_data``
* "Correlation structure is present: `p_completion` for two projects sharing an announcer
  has sample correlation 0.5 +/- 0.1."
  -> ``test_acc_shared_announcer_p_completion_correlation``
  with ``test_acc_public_projects_share_the_same_state_correlation`` and
  ``test_acc_phi_and_eta_are_negatively_correlated`` for the other two Section 16.1 pairs.
* "Doubling draw count changes p50 by less than 1% (convergence check)."
  -> ``test_acc_doubling_draws_moves_p50_by_less_than_one_percent``
* "Runtime target met on the Vizag fixture." -> NOT ASSERTABLE HERE. Real Vizag data is not
  on disk and this environment is not the 16-core machine Section 16.2 specifies. What IS
  measured is ``test_per_draw_cost_is_recorded``, which prints the per-draw cost on the
  synthetic fixture; the honest extrapolation is in the build report, not in an assertion
  that would pass or fail for reasons unrelated to the engine.

Every expected number is either read back out of ``config/params/montecarlo.yaml`` through
``Params`` or is a TEST INPUT (a draw count, a horizon, household sizes).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from ufe.errors import UFEError
from ufe.layers.routing import HaversineBackend, precompute_matrices
from ufe.params import load_params
from ufe.sim import montecarlo as MC
from ufe.sim import runner as R
from ufe.sim.snapshot import load_snapshot_data
from ufe.store import db

from tests.fixtures.synthetic import build_city
from tests.unit.test_runner import (
    ARCHETYPE_UNITS,
    BASE_YEAR,
    CITY,
    DELAY_FAMILY_MAP,
    PPH,
)

#: A deliberately small city: the Monte Carlo tests run dozens of full engine runs, and the
#: statistics being tested are about the *ensemble*, not about the city's size.
N_CELLS = 120

#: Draw counts. Test inputs: small enough to run in seconds, large enough for the Section 16
#: acceptance statistics to be meaningful. Everything is seeded, so these numbers are
#: reproducible exactly, not approximately.
ENSEMBLE_DRAWS = 60
TRUTH_DRAWS = 30
CORRELATION_DRAWS = 2000

#: Truth draws come from a seed block disjoint from the ensemble's.
TRUTH_SEED_BLOCK = 1_000_000


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def config(params):
    return MC.load_config(params)


def _relabel(projects: pd.DataFrame) -> pd.DataFrame:
    names = sorted(ARCHETYPE_UNITS)
    assigned = [names[i % len(names)] for i in range(len(projects))]
    out = projects.copy(deep=True)
    out["archetype"] = assigned
    out["scale_unit"] = [ARCHETYPE_UNITS[a] for a in assigned]
    return out


@pytest.fixture(scope="module")
def city():
    return build_city(n_cells=N_CELLS)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, params, city):
    root = tmp_path_factory.mktemp("mc_snapshot")
    con = db.connect(root / "ufe.duckdb")
    db.migrate(con)
    db.write_table(con, "cells", city.cells)
    db.write_table(con, "announcers", city.announcers)
    db.write_table(con, "projects", _relabel(city.projects))
    ref = db.write_snapshot(
        con,
        city_id=CITY,
        created_by="test_montecarlo",
        out_root=root / "snapshots",
        params_hash=params.hash,
    )
    con.close()
    return ref


@pytest.fixture(scope="module")
def data(snapshot):
    return load_snapshot_data(snapshot)


@pytest.fixture(scope="module")
def matrices(params, city):
    return precompute_matrices(city.cells, params, HaversineBackend(params))


@pytest.fixture(scope="module")
def kwargs(matrices):
    return {
        "matrices": matrices,
        "persons_per_household_by_band": PPH,
        "delay_family_map": DELAY_FAMILY_MAP,
        "allow_dirty": True,
    }


@pytest.fixture(scope="module")
def scenario():
    return R.Scenario(city_id=CITY, horizon=(BASE_YEAR + 2,))


@pytest.fixture(scope="module")
def cache(params):
    """One cache shared across every draw — Section 16.2's whole point."""
    return R.RunCache(params)


@pytest.fixture(scope="module")
def ensemble(snapshot, params, scenario, kwargs, cache):
    return MC.run_ensemble(
        snapshot,
        params,
        scenario,
        n_draws=ENSEMBLE_DRAWS,
        base_seed=0,
        cache=cache,
        **kwargs,
    )


@pytest.fixture(scope="module")
def p_base(data, params, scenario):
    return MC._deterministic_p_completion(data, params, scenario)


# ======================================================================================
# 16.1 — configuration comes from YAML
# ======================================================================================


def test_config_is_read_from_montecarlo_yaml(params, config):
    assert config.n_draws == int(params.value(MC.P_DEFAULT_N))
    assert config.base_seed == int(params.value(MC.P_BASE_SEED))
    assert config.quantile_names == ("p10", "p25", "p50", "p75", "p90")
    assert config.quantiles == tuple(
        float(params.value(f"{MC.P_QUANTILES}.{n}")) for n in config.quantile_names
    )
    assert config.rho_announcer == float(params.value(MC.P_RHO_ANNOUNCER))
    assert config.rho_public == float(params.value(MC.P_RHO_PUBLIC))
    assert config.rho_phi_eta == float(params.value(MC.P_RHO_PHI_ETA))
    assert config.sampled_paths, "montecarlo.sampled_paths is empty"


def test_every_sampled_path_resolves(params, config):
    """A typo in `montecarlo.sampled_paths` must not fail silently at draw time."""
    for path in config.sampled_paths:
        leaf = params.leaf(path)
        assert "value" in leaf


def test_section_16_2_caps_are_present(params, config):
    assert config.max_network_states == int(
        params.value("accessibility.matrix.max_network_states")
    ), "montecarlo and accessibility disagree on the Section 8.3/16.2 network state cap"
    assert config.beta_grid_points >= 3


# ======================================================================================
# 16.1 — the Gaussian copula
# ======================================================================================


def test_copula_rng_maps_a_uniform_onto_the_leaf_range():
    rng = MC._CopulaRng(0.25)
    assert rng.uniform(0.0, 4.0) == pytest.approx(1.0)
    assert MC._CopulaRng(0.0).uniform(-1.0, 1.0) == pytest.approx(-1.0)
    assert MC._CopulaRng(1.0).uniform(-1.0, 1.0) == pytest.approx(1.0)


def test_sampling_goes_through_params_sample(params, data, config, p_base, monkeypatch):
    """Task requirement / Section 16.1: draws come from `Params.sample(path, rng)`."""
    seen: list[str] = []
    real = type(params).sample

    def _spy(self, path, rng):
        seen.append(path)
        return real(self, path, rng)

    monkeypatch.setattr(type(params), "sample", _spy)
    MC.sample_draw(
        params, data.projects, config, index=0, base_seed=0, p_completion_base=p_base
    )
    assert set(config.sampled_paths) <= set(seen)


def test_drawn_values_stay_inside_their_yaml_ranges(params, data, config, p_base):
    draw = MC.sample_draw(
        params, data.projects, config, index=3, base_seed=0, p_completion_base=p_base
    )
    for path, value in draw.overrides.items():
        leaf = params.leaf(path)
        low, high = leaf.get("low"), leaf.get("high")
        if low is None or high is None:
            continue
        if path.startswith(config.mode_share_parent):
            continue  # renormalised, so it may leave its own range
        assert low - abs(low) * 1e-9 <= value <= high + abs(high) * 1e-9, path


def test_mode_shares_renormalise_to_one(params, data, config, p_base):
    draw = MC.sample_draw(
        params, data.projects, config, index=1, base_seed=0, p_completion_base=p_base
    )
    shares = [
        v for k, v in draw.overrides.items() if k.startswith(config.mode_share_parent)
    ]
    assert shares
    assert sum(shares) == pytest.approx(1.0)


def test_correlation_matrix_is_symmetric_and_positive_definite(data, config):
    dims, matrix = MC.correlation_matrix(data.projects, config)
    assert len(dims) == len(data.projects) + 2
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0
    np.linalg.cholesky(matrix)  # must not raise
    assert matrix[-2, -1] == pytest.approx(config.rho_phi_eta)


def test_correlation_matrix_projection_is_logged_not_silent(config, caplog):
    """A block of mutually-0.5 correlations plus a mutually-0.3 block need not be PD."""
    broken = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with caplog.at_level("INFO", logger="ufe.sim.montecarlo"):
        fixed = MC._nearest_positive_definite(broken, config.min_eigenvalue)
    assert np.linalg.eigvalsh(fixed).min() > 0
    np.linalg.cholesky(fixed)  # the guarantee the sampler needs: it must factorise
    np.testing.assert_allclose(np.diag(fixed), np.ones(3))
    assert "positive-definite" in caplog.text


def _shared_announcer_projects(base: pd.DataFrame) -> pd.DataFrame:
    """Two private projects on the SAME announcer, plus two public ones.

    The shipped synthetic fixture happens to give every private project a distinct
    announcer, so the Section 16 acceptance pair has to be constructed. Test input.
    """
    out = base.copy(deep=True).reset_index(drop=True)
    out.loc[out.index[0], "is_public"] = False
    out.loc[out.index[1], "is_public"] = False
    out.loc[out.index[0], "announcer_id"] = "ann-000"
    out.loc[out.index[1], "announcer_id"] = "ann-000"
    out.loc[out.index[2], "is_public"] = True
    out.loc[out.index[2], "announcer_id"] = None
    out.loc[out.index[3], "is_public"] = True
    out.loc[out.index[3], "announcer_id"] = None
    return out


def _p_completion_samples(params, projects, config, p_base, n: int) -> pd.DataFrame:
    dims, matrix = MC.correlation_matrix(projects, config)
    draws = [
        MC.sample_draw(
            params,
            projects,
            config,
            index=k,
            base_seed=0,
            p_completion_base=p_base,
            dims=dims,
            matrix=matrix,
        )
        for k in range(n)
    ]
    return pd.DataFrame([d.p_completion for d in draws]), draws


@pytest.mark.acceptance
def test_acc_shared_announcer_p_completion_correlation(params, data, config, p_base):
    """Section 16 ACCEPTANCE: "sample correlation 0.5 +/- 0.1"."""
    projects = _shared_announcer_projects(data.projects)
    samples, _ = _p_completion_samples(
        params, projects, config, p_base, CORRELATION_DRAWS
    )
    a, b = projects["project_id"].iloc[0], projects["project_id"].iloc[1]
    observed = float(np.corrcoef(samples[a], samples[b])[0, 1])
    assert observed == pytest.approx(
        config.rho_announcer, abs=config.acceptance_tolerance
    ), f"shared-announcer correlation {observed:.3f} against {config.rho_announcer}"


@pytest.mark.acceptance
def test_acc_public_projects_share_the_same_state_correlation(
    params, data, config, p_base
):
    projects = _shared_announcer_projects(data.projects)
    samples, _ = _p_completion_samples(
        params, projects, config, p_base, CORRELATION_DRAWS
    )
    a, b = projects["project_id"].iloc[2], projects["project_id"].iloc[3]
    observed = float(np.corrcoef(samples[a], samples[b])[0, 1])
    assert observed == pytest.approx(config.rho_public, abs=config.acceptance_tolerance)


@pytest.mark.acceptance
def test_acc_phi_and_eta_are_negatively_correlated(params, data, config, p_base):
    """Section 16.1's third named pair: rho(phi_t, eta) = -0.3."""
    dims, matrix = MC.correlation_matrix(data.projects, config)
    draws = [
        MC.sample_draw(
            params,
            data.projects,
            config,
            index=k,
            base_seed=0,
            p_completion_base=p_base,
            dims=dims,
            matrix=matrix,
        )
        for k in range(CORRELATION_DRAWS)
    ]
    # phi_t is drawn inside the chosen macro scenario, so compare on the latent uniforms
    # (which is where the copula lives) as well as on the realised eta.
    phi_u = np.array([d.uniforms[MC.DIM_PHI] for d in draws])
    eta = np.array([d.overrides[MC.P_ETA] for d in draws])
    observed = float(np.corrcoef(phi_u, eta)[0, 1])
    assert observed == pytest.approx(
        config.rho_phi_eta, abs=config.acceptance_tolerance
    )


def test_uncorrelated_projects_stay_uncorrelated(params, data, config, p_base):
    """The copula must not smear correlation across unrelated projects."""
    projects = _shared_announcer_projects(data.projects)
    samples, _ = _p_completion_samples(
        params, projects, config, p_base, CORRELATION_DRAWS
    )
    a = projects["project_id"].iloc[0]  # private, ann-000
    other = projects.loc[
        (~projects["is_public"].astype(bool))
        & (projects["announcer_id"].astype(object) != "ann-000"),
        "project_id",
    ]
    assert len(other), "fixture has no unrelated private project"
    observed = float(np.corrcoef(samples[a], samples[other.iloc[0]])[0, 1])
    assert abs(observed) < config.acceptance_tolerance


def test_macro_scenario_is_categorical_by_scenario_probabilities(
    params, data, config, p_base
):
    """Section 16.1: "Categorical by `scenario_probabilities`, then triangular within"."""
    draws = [
        MC.sample_draw(
            params, data.projects, config, index=k, base_seed=0, p_completion_base=p_base
        )
        for k in range(CORRELATION_DRAWS)
    ]
    counts = pd.Series([d.macro_scenario for d in draws]).value_counts(normalize=True)
    for name in ("base", "bull", "bear"):
        expected = float(
            params.value(f"{config.macro_probabilities_prefix}.{name}")
        )
        assert counts[name] == pytest.approx(expected, abs=config.acceptance_tolerance)


# ======================================================================================
# 16.2 — seeding, efficiency, determinism
# ======================================================================================


def test_draw_seed_is_base_seed_plus_index(params, data, config, p_base):
    """Section 16.2: "each worker with `seed = base_seed + draw_index`"."""
    for k in (0, 1, 17):
        draw = MC.sample_draw(
            params,
            data.projects,
            config,
            index=k,
            base_seed=100,
            p_completion_base=p_base,
        )
        assert draw.seed == 100 + k


def test_a_draw_is_a_pure_function_of_base_seed_and_index(params, data, config, p_base):
    a = MC.sample_draw(
        params, data.projects, config, index=5, base_seed=0, p_completion_base=p_base
    )
    b = MC.sample_draw(
        params, data.projects, config, index=5, base_seed=0, p_completion_base=p_base
    )
    c = MC.sample_draw(
        params, data.projects, config, index=6, base_seed=0, p_completion_base=p_base
    )
    assert a.overrides == b.overrides and a.p_completion == b.p_completion
    assert a.overrides != c.overrides


def test_ensemble_is_reproducible_from_one_master_seed(
    snapshot, params, scenario, kwargs, cache
):
    a = MC.run_ensemble(
        snapshot, params, scenario, n_draws=4, base_seed=11, cache=cache, **kwargs
    )
    b = MC.run_ensemble(
        snapshot, params, scenario, n_draws=4, base_seed=11, cache=cache, **kwargs
    )
    np.testing.assert_array_equal(np.nan_to_num(a.ln_price), np.nan_to_num(b.ln_price))
    pd.testing.assert_frame_equal(a.quantiles, b.quantiles)


def test_a_different_master_seed_changes_the_ensemble(
    snapshot, params, scenario, kwargs, cache
):
    a = MC.run_ensemble(
        snapshot, params, scenario, n_draws=4, base_seed=11, cache=cache, **kwargs
    )
    b = MC.run_ensemble(
        snapshot, params, scenario, n_draws=4, base_seed=99, cache=cache, **kwargs
    )
    assert not np.array_equal(np.nan_to_num(a.ln_price), np.nan_to_num(b.ln_price))


@pytest.mark.acceptance
def test_acc_accessibility_is_computed_once_per_network_state_not_once_per_draw(
    ensemble, cache
):
    """Section 16.2: "Structure the run so it is computed once per network state, not once
    per draw." The travel-time cache is shared across the whole ensemble."""
    stats = cache.stats()["matrices"]
    assert stats["size"] <= 32, "more cached states than the Section 16.2 cap"
    assert stats["hits"] > ensemble.n_draws, (
        "matrices are not being reused across draws: "
        f"{stats['hits']} hits for {ensemble.n_draws} draws"
    )


def test_zero_or_negative_draw_count_raises(snapshot, params, scenario, kwargs):
    with pytest.raises(UFEError, match="n_draws"):
        MC.run_ensemble(snapshot, params, scenario, n_draws=0, **kwargs)


# ======================================================================================
# 16.1 — ParamsDraw, the view that injects a draw
# ======================================================================================


def test_params_draw_overrides_value_leaf_and_sample(params):
    view = MC.ParamsDraw(params, {MC.P_ETA: 0.75}, draw=3)
    assert view.value(MC.P_ETA) == 0.75
    assert view.leaf(MC.P_ETA)["value"] == 0.75
    assert view.sample(MC.P_ETA, np.random.default_rng(0)) == 0.75
    # untouched paths delegate
    assert view.value("price.hedonic.gamma_access_built") == params.value(
        "price.hedonic.gamma_access_built"
    )
    assert view.city_config == params.city_config


def test_params_draw_hash_differs_per_draw(params):
    a = MC.ParamsDraw(params, {MC.P_ETA: 0.75}, draw=1)
    b = MC.ParamsDraw(params, {MC.P_ETA: 0.76}, draw=1)
    assert a.hash != b.hash != params.hash
    assert a.manifest()["monte_carlo_draw"] == 1
    assert a.manifest()["monte_carlo_overrides"] == {MC.P_ETA: 0.75}


def test_drawn_p_completion_actually_reaches_the_run(
    data, params, scenario, kwargs, config, p_base
):
    """A draw that forces every project's p_completion to the floor must change the run."""
    draw = MC.sample_draw(
        params, data.projects, config, index=0, base_seed=0, p_completion_base=p_base
    )
    dead = MC.Draw(
        index=draw.index,
        seed=draw.seed,
        overrides=draw.overrides,
        p_completion={pid: 0.0 for pid in draw.p_completion},
        macro_scenario=draw.macro_scenario,
    )
    alive = MC.Draw(
        index=draw.index,
        seed=draw.seed,
        overrides=draw.overrides,
        p_completion={pid: 1.0 for pid in draw.p_completion},
        macro_scenario=draw.macro_scenario,
    )
    a = MC._draw_payload(data, params, scenario, config, dead, kwargs, None)
    b = MC._draw_payload(data, params, scenario, config, alive, kwargs, None)
    assert a.digest() != b.digest()
    assert a.shock_weights["activation_weight"].sum() < b.shock_weights[
        "activation_weight"
    ].sum()


# ======================================================================================
# 16.3 — outputs
# ======================================================================================


def test_quantile_output_has_the_section_16_3_shape(ensemble, config):
    frame = ensemble.quantiles
    assert set(frame["variable"]) == {"ln_price", "built_sqm"}
    assert set(frame["quantile"]) == set(config.quantile_names)
    expected = (
        len(config.quantile_names) * 2 * len(ensemble.cells) * len(ensemble.years)
    )
    assert len(frame) == expected


def test_quantiles_are_monotonic_in_the_probability(ensemble):
    wide = ensemble.quantiles.pivot_table(
        index=["variable", "h3", "year"], columns="quantile", values="value"
    ).dropna()
    assert (wide["p10"] <= wide["p25"]).all()
    assert (wide["p25"] <= wide["p50"]).all()
    assert (wide["p50"] <= wide["p75"]).all()
    assert (wide["p75"] <= wide["p90"]).all()


def test_outperform_and_top_decile_are_probabilities(ensemble, config):
    values = ensemble.outperform["p_outperform_city_median"].dropna()
    assert ((values >= 0) & (values <= 1)).all()
    decile = ensemble.top_decile["p_top_decile"].dropna()
    assert ((decile >= 0) & (decile <= 1)).all()
    assert set(ensemble.top_decile["year"]) == set(ensemble.years)
    # roughly a decile of the city is in the top decile, in every year
    per_year = ensemble.top_decile.groupby("year")["p_top_decile"].mean()
    assert (per_year <= config.top_decile_share * 2).all()


def test_draw_ledger_records_every_draw(ensemble):
    assert len(ensemble.draws) == ensemble.n_draws
    assert list(ensemble.draws["draw"]) == list(range(ensemble.n_draws))
    assert (ensemble.draws["seed"] == ensemble.draws["draw"] + ensemble.base_seed).all()
    assert MC.P_ETA in ensemble.draws.columns


def test_lambda_distribution_is_available_on_request(
    snapshot, params, kwargs, cache, data
):
    """Section 16.3's "per factor: distribution of lambda". Two factor groups over the
    fixture's own project ids, so the ablation is cheap."""
    ids = list(data.projects["project_id"].astype(str))
    scenario = R.Scenario(
        city_id=CITY,
        horizon=(BASE_YEAR + 1,),
        factor_groups={"first": tuple(ids[:2]), "second": tuple(ids[2:4])},
    )
    ensemble = MC.run_ensemble(
        snapshot,
        params,
        scenario,
        n_draws=2,
        base_seed=0,
        cache=cache,
        decompose_factors=True,
        **kwargs,
    )
    assert ensemble.lambdas is not None
    assert set(ensemble.lambdas["factor"]) == {"first", "second"}
    assert set(ensemble.lambdas["draw"]) == {0, 1}


# ======================================================================================
# Section 16 ACCEPTANCE — band calibration and convergence
# ======================================================================================


@pytest.mark.acceptance
def test_acc_band_calibration_on_synthetic_data(
    ensemble, data, params, scenario, config, kwargs, cache, p_base
):
    """"Generate truth from a known parameter draw, run MC, verify ~80% of truths fall in
    the p10-p90 band."

    One truth draw is not enough: a draw's parameters move every cell in the same direction
    (the macro scenario alone is city-wide), so cells within a draw are strongly correlated
    and a single truth is essentially one Bernoulli trial. Coverage is therefore pooled over
    `TRUTH_DRAWS` independent truths from a seed block disjoint from the ensemble's.
    """
    dims, matrix = MC.correlation_matrix(data.projects, config)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        low = np.nanquantile(ensemble.ln_price, config.quantiles[0], axis=0)
        high = np.nanquantile(ensemble.ln_price, config.quantiles[-1], axis=0)

    inside: list[float] = []
    for k in range(TRUTH_DRAWS):
        truth_draw = MC.sample_draw(
            params,
            data.projects,
            config,
            index=k,
            base_seed=TRUTH_SEED_BLOCK,
            p_completion_base=p_base,
            dims=dims,
            matrix=matrix,
        )
        truth = MC._draw_payload(
            data, params, scenario, config, truth_draw, kwargs, cache
        )
        surface = (
            truth.panel.pivot_table(index="year", columns="h3", values="ln_price")
            .reindex(index=list(ensemble.years), columns=list(ensemble.cells))
            .to_numpy()
        )
        valid = np.isfinite(surface) & np.isfinite(low) & np.isfinite(high)
        inside.append(float(((surface >= low) & (surface <= high))[valid].mean()))

    coverage = float(np.mean(inside))
    assert coverage == pytest.approx(
        config.nominal_coverage, abs=config.coverage_tolerance
    ), (
        f"p10-p90 band covers {coverage:.1%} of {TRUTH_DRAWS} independent truths, against "
        f"a nominal {config.nominal_coverage:.0%} +/- {config.coverage_tolerance:.0%}"
    )


@pytest.mark.acceptance
def test_acc_doubling_draws_moves_p50_by_less_than_one_percent(
    ensemble, snapshot, params, scenario, kwargs, cache, config
):
    """Section 16 ACCEPTANCE convergence check."""
    doubled = MC.run_ensemble(
        snapshot,
        params,
        scenario,
        n_draws=2 * ENSEMBLE_DRAWS,
        base_seed=0,
        cache=cache,
        **kwargs,
    )
    converged, worst = MC.p50_converged(ensemble, doubled, config)
    assert converged, (
        f"doubling {ENSEMBLE_DRAWS} draws to {2 * ENSEMBLE_DRAWS} moved the median ln "
        f"price by {worst:.2%}, over {config.convergence_p50_rel_tol:.0%}"
    )


def test_p50_convergence_check_has_teeth(config):
    """The convergence check must be able to fail."""
    coarse = MC.EnsembleResult(
        quantiles=pd.DataFrame(),
        outperform=pd.DataFrame(),
        top_decile=pd.DataFrame(),
        draws=pd.DataFrame(),
        lambdas=None,
        n_draws=1,
        base_seed=0,
        years=(2025,),
        cells=("a",),
        manifest={},
        ln_price=np.ones((1, 1, 1)),
    )
    fine = MC.EnsembleResult(
        quantiles=pd.DataFrame(),
        outperform=pd.DataFrame(),
        top_decile=pd.DataFrame(),
        draws=pd.DataFrame(),
        lambdas=None,
        n_draws=2,
        base_seed=0,
        years=(2025,),
        cells=("a",),
        manifest={},
        ln_price=np.full((1, 1, 1), 2.0),
    )
    converged, worst = MC.p50_converged(coarse, fine, config)
    assert converged is False
    assert worst == pytest.approx(1.0)


# ======================================================================================
# runtime — measured and reported, never asserted (see the module docstring)
# ======================================================================================


def test_per_draw_cost_is_recorded(snapshot, params, scenario, kwargs, cache, capsys):
    """Section 16.2's runtime target cannot be validated here: this is not a 16-core
    machine and real Vizag data is not on disk. The per-draw cost on the synthetic fixture
    is measured and printed so the extrapolation in the build report is grounded."""
    n = 10
    start = time.perf_counter()
    MC.run_ensemble(
        snapshot, params, scenario, n_draws=n, base_seed=500, cache=cache, **kwargs
    )
    elapsed = time.perf_counter() - start
    per_draw = elapsed / n
    years = max(scenario.horizon) - BASE_YEAR
    with capsys.disabled():
        print(
            f"\n[montecarlo] {per_draw * 1000:.0f} ms/draw over {years} year(s) and "
            f"{N_CELLS} cells (serial, one shared RunCache)"
        )
    assert per_draw > 0
