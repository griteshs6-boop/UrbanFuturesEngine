"""End-to-end integration: the whole engine, every layer, several years.

This is the centrepiece test for Modules 11 and 12. It builds the synthetic city, writes it
to a real snapshot through the real store, precomputes real (offline, Haversine) travel-time
matrices, and then runs the full annual loop across Layers 0-6 for eight years.

What it asserts, marked ``@pytest.mark.acceptance`` where the spec asks for it:

* the run completes and **every** layer contributed something to the output
  (Section 15.1's layer order is not decorative)
* **households are conserved**: ``sum(allocated) + spill == sum(new_hh)`` every year
  (Section 12 ACCEPTANCE) and the city total only grows by what was demanded
* **floorspace is conserved**: standing stock moves by exactly what Layer 4 delivered, and
  never exceeds capacity (Section 11)
* **prices are finite** everywhere the base year had an observed price, and every log price
  movement is finite everywhere
* the run is **deterministic**: two runs with the same seed and snapshot serialise to
  identical bytes (Section 15.2, Section 23 item 4)
* the run is **traceable**: snapshot hash + params hash + git commit (Section 23 item 5)
* the Section 13.4 factor decomposition reconciles, and its purity property holds
* a Monte Carlo ensemble runs on top of the same snapshot and produces Section 16.3 outputs

Everything is offline: no OSRM, no network, no LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.layers.routing import HaversineBackend, precompute_matrices
from ufe.params import load_params
from ufe.sim import factors as F
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

#: Eight simulated years — long enough that the state threading has to actually work
#: (construction starts, ramps, backlog delivery and overshoot decay all span years).
HORIZON_YEARS = 8
LAST_YEAR = BASE_YEAR + HORIZON_YEARS

#: Floating-point tolerance, derived rather than typed.
STRICT = float(np.sqrt(np.finfo(float).eps))


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture(scope="module")
def city():
    built = build_city()
    projects = built.projects.copy(deep=True)
    names = sorted(ARCHETYPE_UNITS)
    assigned = [names[i % len(names)] for i in range(len(projects))]
    projects["archetype"] = assigned
    projects["scale_unit"] = [ARCHETYPE_UNITS[a] for a in assigned]
    import dataclasses

    return dataclasses.replace(built, projects=projects)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, params, city):
    root = tmp_path_factory.mktemp("e2e")
    con = db.connect(root / "ufe.duckdb")
    db.migrate(con)
    db.write_table(con, "cells", city.cells)
    db.write_table(con, "announcers", city.announcers)
    db.write_table(con, "projects", city.projects)
    ref = db.write_snapshot(
        con,
        city_id=CITY,
        created_by="test_end_to_end",
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
    return R.Scenario(city_id=CITY, horizon=(LAST_YEAR,))


@pytest.fixture(scope="module")
def result(data, params, scenario, kwargs):
    return R.run(data, params, scenario, seed=20240101, **kwargs)


# ======================================================================================
# the run completes, and every layer contributed
# ======================================================================================


@pytest.mark.acceptance
def test_acc_full_annual_loop_completes_over_every_layer(result, data, matrices):
    """Section 15.1: the runner threads the annual loop across every layer, for real."""
    years = result.manifest.simulated_years
    assert years == tuple(range(BASE_YEAR + 1, LAST_YEAR + 1))
    assert len(result.panel) == len(data.cells) * HORIZON_YEARS
    assert len(result.diagnostics) == HORIZON_YEARS

    # Layer 2 resolved the pipeline into shocks
    assert (result.diagnostics["n_shock_projects"] == len(data.projects)).all()
    assert result.diagnostics["n_employment_effects"].max() > 0
    assert result.shock_weights["activation_weight"].max() > 0
    assert result.panel["shock_field_residential"].abs().max() > 0

    # Layer 1 produced a finite accessibility surface on the origin set. Cells outside
    # `MatrixSet.origins` get NaN by design (Section 5.2 restricts the matrix; Layer 1's
    # docstring: "a cell with no computed accessibility is missing, not inaccessible").
    origins = set(matrices.origins)
    on_grid = result.panel["h3"].isin(origins)
    assert on_grid.any()
    assert np.isfinite(result.panel.loc[on_grid, "lnA"]).all()
    assert result.panel.loc[~on_grid, "lnA"].isna().all()
    assert result.panel.loc[on_grid, "lnA"].nunique() > 1
    assert np.isfinite(result.panel["d_lnA"]).all()

    # Layer 4 delivered floorspace
    assert result.diagnostics["delivered_sqm"].sum() > 0

    # Layer 5 allocated households
    assert result.diagnostics["new_households"].sum() > 0
    assert result.diagnostics["allocation_converged"].all()

    # Layer 6 formed prices
    assert result.diagnostics["price_converged"].all()
    assert result.panel["d_ln_P_reported"].abs().sum() > 0


@pytest.mark.acceptance
def test_acc_run_is_traceable_to_snapshot_params_and_code(result, snapshot, params):
    """Section 23 item 5."""
    provenance = result.manifest.provenance
    assert provenance.snapshot_hash == snapshot.snapshot_hash
    assert provenance.params_hash == params.hash
    assert provenance.code_version
    assert provenance.city_id == CITY
    payload = result.manifest.to_dict()
    assert payload["provenance"]["snapshot_id"] == snapshot.snapshot_id
    assert payload["params"]["source_files"], "the manifest lost the params source list"


# ======================================================================================
# conservation
# ======================================================================================


@pytest.mark.acceptance
def test_acc_households_are_conserved_every_year(result):
    """Section 12 ACCEPTANCE: ``sum(allocated) + spill == sum(new_hh)``."""
    allocated = result.panel.groupby("year")["new_households"].sum()
    demanded = result.diagnostics.set_index("year")["new_households"]
    spill = result.diagnostics.set_index("year")["spill_households"]
    np.testing.assert_allclose(
        (allocated + spill).to_numpy(), demanded.to_numpy(), rtol=STRICT
    )
    assert demanded.sum() > 0, "nothing was demanded, so nothing was tested"


@pytest.mark.acceptance
def test_acc_city_household_stock_grows_only_by_what_was_allocated(result, data):
    initial = float(data.cells["households"].sum())
    per_year = result.panel.groupby("year")["households"].sum()
    allocated = result.panel.groupby("year")["new_households"].sum()
    expected = initial + allocated.cumsum()
    np.testing.assert_allclose(per_year.to_numpy(), expected.to_numpy(), rtol=STRICT)


@pytest.mark.acceptance
def test_acc_floorspace_is_conserved_and_never_exceeds_capacity(result, data):
    """Section 11: stock moves by exactly what Layer 4 delivered, and stays under capacity."""
    initial = float(
        (data.cells["floorspace_res_sqm"] + data.cells["floorspace_com_sqm"]).sum()
    )
    delivered = result.diagnostics.set_index("year")["delivered_sqm"]
    stock = result.panel.groupby("year").apply(
        lambda g: float((g["floorspace_res_sqm"] + g["floorspace_com_sqm"]).sum()),
        include_groups=False,
    )
    np.testing.assert_allclose(
        stock.to_numpy(), (initial + delivered.cumsum()).to_numpy(), rtol=STRICT
    )
    assert delivered.sum() > 0, "nothing was delivered, so nothing was tested"

    # the schema's two stock columns must always agree with Layer 4's own `built_sqm`
    np.testing.assert_allclose(
        (result.panel["floorspace_res_sqm"] + result.panel["floorspace_com_sqm"]).to_numpy(),
        result.panel["built_sqm"].to_numpy(),
        rtol=STRICT,
    )
    # No cell ever GROWS beyond its capacity. The weaker form is deliberate: Section 11.3
    # land sterilisation can drop `capacity_sqm` below stock that is already standing (a
    # data-centre land take under an occupied cell), so `built <= capacity` is not an
    # invariant of the model — "delivery never pushes a cell over capacity" is.
    grew = result.panel["delivered_sqm"] > 0
    assert grew.any(), "no cell grew, so nothing was tested"
    assert (
        result.panel.loc[grew, "built_sqm"]
        <= result.panel.loc[grew, "capacity_sqm"] + STRICT
    ).all()
    assert (result.panel["headroom_sqm"] >= 0).all()
    assert (result.panel["delivered_sqm"] >= 0).all()


@pytest.mark.acceptance
def test_acc_delivery_never_exceeds_the_absorption_cap(result):
    """Section 11.2's absorption cap is a cap, not a suggestion."""
    assert (
        result.panel["delivered_sqm"] <= result.panel["absorption_cap_sqm"] + STRICT
    ).all()


# ======================================================================================
# finiteness
# ======================================================================================


@pytest.mark.acceptance
def test_acc_prices_are_finite(result, data):
    """Every price movement is finite, and every cell with an observed base-year price ends
    the horizon with a finite, positive price."""
    assert np.isfinite(result.panel["d_ln_P_fundamental"]).all()
    assert np.isfinite(result.panel["d_ln_P_reported"]).all()
    assert np.isfinite(result.panel["phi_t"]).all()
    assert np.isfinite(result.panel["overshoot_log"]).all()

    observed = data.cells.set_index("h3")["price_res_inr_sqft"]
    known = observed.dropna().index
    assert len(known), "the fixture has no observed prices at all"
    final = result.at(LAST_YEAR)["price_res_inr_sqft"].reindex(known)
    assert np.isfinite(final).all()
    assert (final > 0).all()

    # a cell with no observed base-year price stays missing rather than being invented
    missing = observed[observed.isna()].index
    if len(missing):
        assert result.at(LAST_YEAR)["price_res_inr_sqft"].reindex(missing).isna().all()


@pytest.mark.acceptance
def test_acc_no_negative_or_nan_quantities(result):
    for column in (
        "households",
        "population",
        "floorspace_res_sqm",
        "floorspace_com_sqm",
        "built_sqm",
        "capacity_sqm",
        "headroom_sqm",
    ):
        values = result.panel[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{column} has non-finite values"
        assert (values >= -STRICT).all(), f"{column} went negative"


# ======================================================================================
# determinism (Section 15.2, Section 23 item 4)
# ======================================================================================


@pytest.mark.acceptance
def test_acc_the_whole_engine_is_deterministic(data, params, scenario, kwargs, result):
    """Two runs, same seed, same snapshot: byte-identical serialised output."""
    again = R.run(data, params, scenario, seed=20240101, **kwargs)
    first, second = result.to_parquet_bytes(), again.to_parquet_bytes()
    for name in sorted(first):
        assert first[name] == second[name], f"{name} is not byte-identical"
    assert again.digest() == result.digest()


@pytest.mark.acceptance
def test_acc_determinism_survives_a_cold_cache(snapshot, params, scenario, kwargs, result):
    """Reproducibility must not depend on cache warmth — it must survive re-reading the
    snapshot from disk into a brand-new cache."""
    cold = R.run(
        snapshot, params, scenario, seed=20240101, cache=R.RunCache(params), **kwargs
    )
    assert cold.digest() == result.digest()


# ======================================================================================
# Section 13.4 — factor decomposition over the run
# ======================================================================================


@pytest.fixture(scope="module")
def decomposition(data, params, kwargs):
    ids = list(data.projects["project_id"].astype(str))
    scenario = R.Scenario(
        city_id=CITY,
        horizon=(BASE_YEAR + 3,),
        factor_groups={"group_a": tuple(ids[:3]), "group_b": tuple(ids[3:6])},
    )
    return F.decompose_run(data, params, scenario, seed=20240101, **kwargs)


@pytest.mark.acceptance
def test_acc_factor_decomposition_reconciles(decomposition):
    """Section 13 ACCEPTANCE: ``sum_f raw_lambda + interaction == total``, exactly."""
    error = decomposition.reconciliation_error().to_numpy()
    # Cells with no observed base-year price carry a NaN ln price all the way through, so
    # the identity is asserted where it is defined.
    defined = np.isfinite(error)
    assert defined.sum() > 0
    assert np.abs(error[defined]).max() < STRICT
    assert decomposition.n_runs == len(decomposition.factors) + 2


@pytest.mark.acceptance
def test_acc_removing_a_factor_reproduces_its_leave_one_out_run(
    decomposition, data, params, kwargs
):
    """Section 13 ACCEPTANCE, and a purity test of the WHOLE engine: re-running with a
    factor's projects disabled must reproduce that factor's LOO series exactly."""
    groups = decomposition.factor_groups
    target = decomposition.factors[0]
    scenario = R.Scenario(
        city_id=CITY,
        horizon=(BASE_YEAR + 3,),
        disabled_projects=tuple(groups[target]),
        factor_groups={k: tuple(v) for k, v in groups.items()},
    )
    rerun = R.run(data, params, scenario, seed=20240101, **kwargs)
    rows = rerun.panel.loc[rerun.panel["year"] == decomposition.year]
    reproduced = rows.set_index("h3")["ln_price"].sort_index()
    expected = decomposition.decomposition.loo[target]
    np.testing.assert_array_equal(
        np.nan_to_num(reproduced.to_numpy()), np.nan_to_num(expected.to_numpy())
    )


def test_factor_frame_has_one_row_per_cell(decomposition, data):
    frame = decomposition.to_frame()
    assert len(frame) == len(data.cells)
    for name in decomposition.factors:
        assert name in frame.columns
    assert {"interaction", "total", "year"} <= set(frame.columns)


def test_lambda_frame_is_long_and_complete(decomposition, data):
    long = F.lambda_frame(decomposition.decomposition)
    assert len(long) == len(data.cells) * len(decomposition.factors)
    assert set(long["factor"]) == set(decomposition.factors)


def test_factor_groups_default_to_the_city_config(params):
    groups = F.factor_groups_from_params(params)
    assert set(groups) == {"metro", "airport", "data_centres", "port_indl"}


def test_decompose_without_any_factor_groups_raises(data, params, kwargs):
    from ufe.errors import UFEError

    scenario = R.Scenario(city_id=CITY, horizon=(BASE_YEAR + 1,), factor_groups={})
    with pytest.raises(UFEError, match="no factor groups"):
        F.decompose_run(data, params, scenario, factor_groups={}, **kwargs)


# ======================================================================================
# Monte Carlo on top of the same snapshot
# ======================================================================================


@pytest.mark.acceptance
def test_acc_monte_carlo_runs_over_the_same_snapshot(snapshot, params, kwargs):
    """Section 16.3's outputs, produced end to end from the same immutable snapshot."""
    scenario = R.Scenario(city_id=CITY, horizon=(BASE_YEAR + 2,))
    ensemble = MC.run_ensemble(
        snapshot, params, scenario, n_draws=8, base_seed=0, **kwargs
    )
    assert ensemble.n_draws == 8
    assert set(ensemble.quantiles["variable"]) == {"ln_price", "built_sqm"}
    assert not ensemble.outperform.empty
    assert not ensemble.top_decile.empty
    # the ensemble is reproducible from its master seed
    again = MC.run_ensemble(
        snapshot, params, scenario, n_draws=8, base_seed=0, **kwargs
    )
    pd.testing.assert_frame_equal(ensemble.quantiles, again.quantiles)


# ======================================================================================
# the CLI, end to end
# ======================================================================================


def test_cli_run_writes_a_panel_and_a_manifest(snapshot, tmp_path):
    import json

    from typer.testing import CliRunner

    import ufe.sim_cli as sim_cli

    out = tmp_path / "run"
    result = CliRunner().invoke(
        sim_cli.app,
        [
            "run",
            "--city",
            CITY,
            "--snapshot",
            str(snapshot.path),
            "--horizon",
            str(BASE_YEAR + 1),
            "--out",
            str(out),
            "--allow-dirty",
        ]
        + [f"--delay-family={k}={v}" for k, v in DELAY_FAMILY_MAP.items()]
        + [f"--persons-per-household={k}={v}" for k, v in PPH.items()],
    )
    assert result.exit_code == 0, result.output
    panel = pd.read_parquet(out / "panel.parquet")
    assert list(panel.columns) == list(R.PANEL_COLUMNS)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    for key in ("snapshot_hash", "params_hash", "code_version", "seed", "layer_order"):
        assert key in manifest


def test_cli_run_refuses_a_dirty_tree_without_the_override(snapshot):
    from typer.testing import CliRunner

    import ufe.sim_cli as sim_cli
    from ufe.sim.snapshot import git_commit, git_is_dirty

    if not (git_is_dirty() or git_commit() == "unknown"):
        pytest.skip("working tree is clean, so there is nothing to refuse")
    result = CliRunner().invoke(
        sim_cli.app,
        [
            "run",
            "--city",
            CITY,
            "--snapshot",
            str(snapshot.path),
            "--horizon",
            str(BASE_YEAR + 1),
        ]
        + [f"--delay-family={k}={v}" for k, v in DELAY_FAMILY_MAP.items()]
        + [f"--persons-per-household={k}={v}" for k, v in PPH.items()],
    )
    assert result.exit_code != 0
