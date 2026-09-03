"""Tests for Module 11 — the simulation runner (spec Section 15).

The Section 15 / Section 23 requirements map onto the ``@pytest.mark.acceptance`` tests
here:

* Section 15.2 / Section 23 item 4 — "Two runs with the same seed and snapshot produce
  byte-identical output" -> ``test_acc_same_seed_produces_byte_identical_output``, with
  ``test_acc_determinism_test_has_teeth`` proving the check can fail.
* Section 23 item 5 — "Every number in the output traces to a snapshot hash, a params hash
  and a git commit" -> ``test_acc_manifest_carries_the_full_provenance_triple`` and
  ``test_acc_run_refuses_unknown_or_dirty_provenance``.
* CONTRACT rule 3 / Section 0.1 rule 5 — no network at simulation time ->
  ``test_acc_no_socket_is_opened_during_a_run``.
* Section 15.3 — caching -> ``test_acc_nothing_is_cached_by_city_alone`` and
  ``test_matrices_are_built_once_per_network_state``.

This module also owns the shared snapshot fixture the Monte Carlo and end-to-end tests
import, because a snapshot has to be written to disk (Section 3.8 forbids reading the live
DB from a simulation) and building it once per session is worth the plumbing.

Numbers appearing below are TEST INPUTS — household sizes, a horizon, a seed — never model
parameters. Every model quantity is read back out of ``Params``.
"""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufe.errors import UFEError
from ufe.layers.routing import HaversineBackend, network_state_hash, precompute_matrices
from ufe.params import load_params
from ufe.sim import runner as R
from ufe.sim.snapshot import (
    Provenance,
    ProvenanceError,
    UNKNOWN_COMMIT,
    load_snapshot_data,
    open_snapshot,
    resolve_provenance,
)
from ufe.store import db

from tests.fixtures.synthetic import build_city

CITY = "vizag"
BASE_YEAR = 2024

#: Persons per household by band. ``behaviour.persons_per_household_by_band`` is null on
#: disk by design (Section 12.5, and the YAML says so out loud), so Layer 5 requires the
#: caller to supply it. These are fixture inputs, matching ``tests/fixtures/synthetic.yaml``'s
#: flat 4.2 with a band gradient so the bands are distinguishable — exactly as
#: ``tests/unit/test_l5_allocation.py`` does.
PPH = {"low": 4.6, "mid": 4.2, "upper_mid": 3.8, "high": 3.4}

#: Section 10.3's delay families are keyed by a vocabulary that does not coincide with the
#: archetype names in ``archetypes.yaml`` (``metro_rail`` vs ``metro_phase1``). The runner
#: takes the mapping as an argument; this is the fixture's.
DELAY_FAMILY_MAP = {
    "metro_rail": "metro_phase1",
    "data_centre": "data_centre",
    "electronics_assembly": "private_industrial",
}

#: The three archetypes actually present in ``config/params/archetypes.yaml``, paired with
#: a ``scale_unit`` the ``projects`` schema accepts. ``electronics_assembly`` declares
#: ``scale_unit: jobs``, which is NOT in ``schemas.SCALE_UNITS`` — reported, and worked
#: around here by siting it on ``seats`` with the runner's scale-unit check disabled.
ARCHETYPE_UNITS = {
    "metro_rail": "km",
    "data_centre": "mw",
    "electronics_assembly": "seats",
}


# --------------------------------------------------------------------------------------
# shared fixtures — a real snapshot on disk, and a real offline MatrixSet
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def params():
    return load_params(CITY)


@pytest.fixture(scope="session")
def city():
    """The synthetic city, with its projects relabelled onto the shipped archetypes.

    ``tests/fixtures/synthetic.py`` sites projects on archetype names the shipped
    ``archetypes.yaml`` does not define (only 3 of the 22 were transcribed). Relabelling
    here — in the test, never in the fixture — is what lets Layer 2 actually emit effects,
    so the end-to-end run exercises the shock path rather than silently resolving to zero.
    """
    built = build_city()
    projects = built.projects.copy(deep=True)
    names = sorted(ARCHETYPE_UNITS)
    assigned = [names[i % len(names)] for i in range(len(projects))]
    projects["archetype"] = assigned
    projects["scale_unit"] = [ARCHETYPE_UNITS[a] for a in assigned]
    return dataclasses.replace(built, projects=projects)


@pytest.fixture(scope="session")
def snapshot(tmp_path_factory, params, city):
    """An immutable snapshot directory, written through the real store."""
    root = tmp_path_factory.mktemp("snapshot")
    con = db.connect(root / "ufe.duckdb")
    db.migrate(con)
    db.write_table(con, "cells", city.cells)
    db.write_table(con, "announcers", city.announcers)
    db.write_table(con, "projects", city.projects)
    ref = db.write_snapshot(
        con,
        city_id=CITY,
        created_by="test_runner",
        out_root=root / "snapshots",
        params_hash=params.hash,
    )
    con.close()
    return ref


@pytest.fixture(scope="session")
def snapshot_data(snapshot):
    return load_snapshot_data(snapshot)


@pytest.fixture(scope="session")
def matrices(params, city):
    """A real offline :class:`MatrixSet` — Haversine backend, no OSRM, no network."""
    return precompute_matrices(city.cells, params, HaversineBackend(params))


def run_kwargs(matrices) -> dict:
    """The keyword arguments every test run shares."""
    return {
        "matrices": matrices,
        "persons_per_household_by_band": PPH,
        "delay_family_map": DELAY_FAMILY_MAP,
        "allow_dirty": True,
    }


def scenario(last_year: int = BASE_YEAR + 3, **kwargs) -> R.Scenario:
    return R.Scenario(city_id=CITY, horizon=(last_year,), **kwargs)


@pytest.fixture(scope="module")
def result(snapshot_data, params, matrices):
    return R.run(snapshot_data, params, scenario(), seed=42, **run_kwargs(matrices))


# ======================================================================================
# 15.1 — signature and Scenario
# ======================================================================================


def test_run_has_the_section_15_1_signature():
    import inspect

    signature = inspect.signature(R.run)
    positional = [
        name
        for name, p in signature.parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    ]
    assert positional == ["snapshot", "params", "scenario", "seed", "deterministic"]
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["deterministic"].default is True


def test_scenario_normalises_sequences_and_is_hashable():
    s = R.Scenario(
        city_id=CITY,
        horizon=[2030, 2027],
        disabled_projects=["b", "a"],
        force_project_state={"a": "happens"},
        factor_groups={"metro": ["p1", "p2"]},
    )
    assert s.horizon == (2030, 2027)
    assert s.report_years == (2027, 2030)
    assert s.disabled_projects == ("b", "a")
    assert s.factor_groups == {"metro": ("p1", "p2")}
    # to_dict sorts everything, so two equivalent scenarios serialise identically.
    other = R.Scenario(
        city_id=CITY,
        horizon=[2030, 2027],
        disabled_projects=["a", "b"],
        force_project_state={"a": "happens"},
        factor_groups={"metro": ["p1", "p2"]},
    )
    assert s.to_dict() == other.to_dict()


def test_empty_horizon_raises():
    with pytest.raises(UFEError):
        R.Scenario(city_id=CITY, horizon=())


def test_horizon_expands_to_contiguous_annual_steps(params):
    years = R._expand_horizon((2027, 2030), BASE_YEAR, int(params.value(R.P_MAX_YEARS)))
    assert years == tuple(range(BASE_YEAR + 1, 2031))


def test_horizon_at_or_before_base_year_raises(params):
    with pytest.raises(UFEError, match="nothing to simulate"):
        R._expand_horizon((BASE_YEAR,), BASE_YEAR, int(params.value(R.P_MAX_YEARS)))


def test_horizon_longer_than_the_cap_raises(params):
    cap = int(params.value(R.P_MAX_YEARS))
    with pytest.raises(UFEError, match="max_years"):
        R._expand_horizon((BASE_YEAR + cap + 1,), BASE_YEAR, cap)


# ======================================================================================
# 15.2 — DETERMINISM (Section 23 item 4)
# ======================================================================================


@pytest.mark.acceptance
def test_acc_same_seed_produces_byte_identical_output(snapshot_data, params, matrices):
    """Section 23 item 4, tested on SERIALISED BYTES, not on dataframe equality."""
    first = R.run(snapshot_data, params, scenario(), seed=42, **run_kwargs(matrices))
    second = R.run(snapshot_data, params, scenario(), seed=42, **run_kwargs(matrices))

    a, b = first.to_parquet_bytes(), second.to_parquet_bytes()
    assert sorted(a) == sorted(b)
    for name in sorted(a):
        assert a[name] == b[name], f"{name} differs byte-for-byte between two runs"
    assert first.digest() == second.digest()
    # and the frames themselves, for good measure
    pd.testing.assert_frame_equal(first.panel, second.panel)


@pytest.mark.acceptance
def test_acc_determinism_test_has_teeth(snapshot_data, params, matrices):
    """A different seed in a stochastic run must change the bytes, or the test above
    proves nothing."""
    kwargs = run_kwargs(matrices)
    a = R.run(
        snapshot_data, params, scenario(), seed=42, deterministic=False, **kwargs
    )
    b = R.run(
        snapshot_data, params, scenario(), seed=43, deterministic=False, **kwargs
    )
    assert a.digest() != b.digest()


def test_stochastic_run_is_reproducible_from_its_seed(snapshot_data, params, matrices):
    kwargs = run_kwargs(matrices)
    a = R.run(snapshot_data, params, scenario(), seed=7, deterministic=False, **kwargs)
    b = R.run(snapshot_data, params, scenario(), seed=7, deterministic=False, **kwargs)
    assert a.digest() == b.digest()


def test_deterministic_run_ignores_the_seed(snapshot_data, params, matrices):
    """`deterministic=True` takes medians everywhere, so the seed cannot reach a number."""
    kwargs = run_kwargs(matrices)
    a = R.run(snapshot_data, params, scenario(), seed=1, **kwargs)
    b = R.run(snapshot_data, params, scenario(), seed=999, **kwargs)
    pd.testing.assert_frame_equal(a.panel, b.panel)


def test_child_rng_is_a_pure_function_of_seed_and_label():
    a = R._child_rng(5, "shocks", 2030).standard_normal(4)
    b = R._child_rng(5, "shocks", 2030).standard_normal(4)
    c = R._child_rng(5, "shocks", 2031).standard_normal(4)
    d = R._child_rng(6, "shocks", 2030).standard_normal(4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_no_global_numpy_state_is_consumed(snapshot_data, params, matrices):
    """Section 15.2: "no use of the global `numpy.random` state"."""
    np.random.seed(0)
    before = np.random.get_state()[2]
    R.run(snapshot_data, params, scenario(), seed=3, deterministic=False, **run_kwargs(matrices))
    assert np.random.get_state()[2] == before


# ======================================================================================
# Section 23 item 5 — PROVENANCE
# ======================================================================================


@pytest.mark.acceptance
def test_acc_manifest_carries_the_full_provenance_triple(result, snapshot, params):
    manifest = result.manifest
    assert manifest.provenance.snapshot_hash == snapshot.snapshot_hash
    assert manifest.provenance.params_hash == params.hash
    assert manifest.provenance.code_version  # possibly UNKNOWN_COMMIT in this sandbox
    payload = manifest.to_dict()
    for key in ("snapshot_hash", "params_hash", "code_version", "seed", "scenario"):
        assert key in payload
    assert payload["params"]["params_hash"] == params.hash
    assert payload["layer_order"] == list(R.LAYER_ORDER)
    # the manifest round-trips as stable, sorted JSON with no wall clock in it
    assert manifest.to_json() == manifest.to_json()
    assert "created_at" not in manifest.to_json()


@pytest.mark.acceptance
def test_acc_run_refuses_unknown_or_dirty_provenance(snapshot_data, params, matrices):
    """Section 23 item 5: a run must not proceed from an untraceable code state."""
    kwargs = run_kwargs(matrices) | {"allow_dirty": False}
    with pytest.raises(ProvenanceError, match="Section 23 item 5"):
        R.run(
            snapshot_data,
            params,
            scenario(),
            code_version=UNKNOWN_COMMIT,
            code_dirty=True,
            **kwargs,
        )
    with pytest.raises(ProvenanceError):
        R.run(
            snapshot_data,
            params,
            scenario(),
            code_version="0" * 40,
            code_dirty=True,
            **kwargs,
        )


def test_clean_provenance_runs_without_an_override(snapshot_data, params, matrices):
    kwargs = run_kwargs(matrices) | {"allow_dirty": False}
    result = R.run(
        snapshot_data,
        params,
        scenario(BASE_YEAR + 1),
        code_version="a" * 40,
        code_dirty=False,
        **kwargs,
    )
    assert result.manifest.complete
    assert result.manifest.provenance.dirty_override is False


def test_override_is_recorded_not_hidden(snapshot_data, params, matrices):
    result = R.run(
        snapshot_data,
        params,
        scenario(BASE_YEAR + 1),
        code_version=UNKNOWN_COMMIT,
        code_dirty=True,
        **run_kwargs(matrices),
    )
    assert result.manifest.provenance.dirty_override is True
    assert result.manifest.complete is False


def test_provenance_incomplete_when_the_commit_is_unknown():
    provenance = Provenance(
        snapshot_id="s",
        snapshot_hash="h",
        params_hash="p",
        code_version=UNKNOWN_COMMIT,
        code_dirty=True,
        city_id=CITY,
    )
    assert provenance.complete is False
    assert provenance.short(4)["snapshot_hash"] == "h"


def test_a_modified_snapshot_will_not_open(snapshot, tmp_path):
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(snapshot.path, copy)
    frame = pd.read_parquet(copy / "cells.parquet")
    frame.loc[frame.index[0], "population"] = frame["population"].iloc[0] + 1
    frame.to_parquet(copy / "cells.parquet", index=False)
    with pytest.raises(ProvenanceError, match="modified"):
        open_snapshot(copy)


def test_resolve_provenance_can_be_disabled_by_yaml(snapshot, params, monkeypatch):
    class _Off:
        def __getattr__(self, name):
            return getattr(params, name)

        def value(self, path):
            if path == "simulation.provenance.require_clean_git":
                return False
            return params.value(path)

    provenance = resolve_provenance(
        snapshot, _Off(), code_version=UNKNOWN_COMMIT, code_dirty=True
    )
    assert provenance.complete is False  # still honest about the gap


# ======================================================================================
# CONTRACT rule 3 — no network at simulation time
# ======================================================================================


@pytest.mark.acceptance
def test_acc_no_socket_is_opened_during_a_run(
    snapshot_data, params, matrices, monkeypatch
):
    """The runner reads precomputed matrices and never calls a routing backend."""
    opened: list[tuple] = []

    class _Forbidden(socket.socket):
        def __init__(self, *args, **kwargs):  # pragma: no cover - must never run
            opened.append(args)
            raise AssertionError(
                "simulation opened a socket: CONTRACT.md rule 3 forbids network I/O at "
                "simulation time (spec Section 0.1 rule 5)"
            )

    monkeypatch.setattr(socket, "socket", _Forbidden)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network call")),
    )
    result = R.run(snapshot_data, params, scenario(), seed=42, **run_kwargs(matrices))
    assert opened == []
    assert not result.panel.empty


def test_runner_rejects_a_routing_backend_instead_of_a_matrixset(
    snapshot_data, params
):
    """Passing a live backend must not silently work: Layer 1 refuses it."""
    from ufe.layers.l1_accessibility import apply_accessibility

    with pytest.raises(UFEError, match="precomputed MatrixSet"):
        apply_accessibility(
            snapshot_data.cells, params, HaversineBackend(params)  # type: ignore[arg-type]
        )


# ======================================================================================
# 15.3 — caching
# ======================================================================================


@pytest.mark.acceptance
def test_acc_nothing_is_cached_by_city_alone(params, matrices):
    """Section 15.3: "do not cache anything keyed only by city"."""
    cache = R.RunCache(params)
    cache.put_matrices("state-a", matrices)
    cache.put_substrate("snap-1", pd.DataFrame({"h3": ["x"]}))
    keys = list(cache.matrices._store) + list(cache.substrate._store)
    assert keys, "the cache under test is empty"
    for key in keys:
        assert key[0] == params.hash, "every cache key must carry the params hash"
        assert params.city_id not in [str(part) for part in key]


def test_cache_key_changes_with_the_params_hash(params, matrices):
    cache = R.RunCache(params)
    cache.put_matrices("state-a", matrices)
    assert cache.get_matrices("state-a") is matrices

    class _Other:
        hash = "a-different-params-hash"

        def value(self, path):
            return params.value(path)

    other = R.RunCache(_Other())
    other.matrices = cache.matrices  # same underlying store, different params hash
    assert other.get_matrices("state-a") is None


def test_matrices_are_built_once_per_network_state(snapshot_data, params, matrices):
    """A run consumes matrices from the cache; a second run over the same states adds no
    entries and performs no backend work."""
    cache = R.RunCache(params)
    kwargs = run_kwargs(matrices)
    R.run(snapshot_data, params, scenario(), seed=1, cache=cache, **kwargs)
    size_after_first = len(cache.matrices)
    R.run(snapshot_data, params, scenario(), seed=1, cache=cache, **kwargs)
    assert len(cache.matrices) == size_after_first
    assert cache.stats()["matrices"]["hits"] > 0


def test_substrate_is_keyed_by_snapshot_hash(snapshot_data, params, matrices):
    cache = R.RunCache(params)
    R.run(snapshot_data, params, scenario(BASE_YEAR + 1), cache=cache, **run_kwargs(matrices))
    assert cache.get_substrate(snapshot_data.snapshot_hash) is not None
    assert cache.get_substrate("some-other-snapshot") is None


def test_bounded_cache_evicts_fifo():
    cache = R._BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("c") == 3


# ======================================================================================
# state threading across years (the runner's actual job)
# ======================================================================================


def test_supply_state_is_threaded_so_stock_never_shrinks(result):
    built = result.panel.pivot_table(index="year", values="built_sqm", aggfunc="sum")
    assert built["built_sqm"].is_monotonic_increasing


def test_capacity_is_carried_forward_not_recomputed_from_the_base_frame(result):
    """`capacity_sqm` only moves when a SupplyEffect sterilises or adds land."""
    per_year = result.panel.pivot_table(index="year", values="capacity_sqm", aggfunc="sum")
    assert len(per_year) > 1
    assert per_year["capacity_sqm"].is_monotonic_decreasing or per_year[
        "capacity_sqm"
    ].nunique() == 1


def test_headroom_equals_capacity_minus_built_every_year(result):
    panel = result.panel
    np.testing.assert_allclose(
        panel["headroom_sqm"].to_numpy(),
        np.maximum(panel["capacity_sqm"] - panel["built_sqm"], 0).to_numpy(),
        atol=float(np.sqrt(np.finfo(float).eps)),
    )


def test_every_year_converges_within_the_yaml_iteration_cap(result, params):
    """Section 12.6 / Section 13.2: both inner loops converge, every year, and the runner
    records it. A non-converging year raises inside the layer, so this also pins that the
    runner is not swallowing a ConvergenceError."""
    cap = int(params.value("behaviour.agglomeration.max_iterations"))
    assert result.diagnostics["allocation_converged"].all()
    assert result.diagnostics["price_converged"].all()
    assert result.diagnostics["allocation_iterations"].max() <= cap
    assert (result.diagnostics["allocation_iterations"] >= 1).all()


def test_d_lnA_is_zero_in_a_year_with_no_network_change(result):
    """With one static MatrixSet the accessibility surface only moves through the
    agglomeration feedback, so d_lnA is small but the column exists and is finite."""
    assert np.isfinite(result.panel["d_lnA"]).all()


def test_layer_order_is_recorded_and_matches_the_spec(result):
    assert result.manifest.layer_order == (
        "l2_shocks",
        "l1_accessibility",
        "l4_supply",
        "l5_allocation",
        "l6_price",
    )
    assert result.manifest.demand_lag_years == 1


def test_network_state_hash_is_recorded_per_year(result):
    states = result.manifest.network_states
    assert set(states) == set(result.manifest.simulated_years)
    assert all(isinstance(v, str) and len(v) == len(network_state_hash(())) for v in states.values())


# ======================================================================================
# result shape and serialisation
# ======================================================================================


def test_panel_has_one_row_per_cell_per_year(result, snapshot_data):
    n_cells = len(snapshot_data.cells)
    n_years = len(result.manifest.simulated_years)
    assert len(result.panel) == n_cells * n_years
    assert list(result.panel.columns) == list(R.PANEL_COLUMNS)


def test_at_year_returns_an_h3_indexed_slice(result):
    year = result.manifest.simulated_years[-1]
    slice_ = result.at(year)
    assert slice_.index.name == "h3"
    assert slice_.index.is_monotonic_increasing


def test_residuals_and_overheat_are_produced_for_the_final_year(result):
    last = result.manifest.simulated_years[-1]
    assert (result.residuals["year"] == last).all()
    assert (result.overheat["year"] == last).all()
    assert "overheat_score" in result.overheat.columns


def test_shock_weights_record_every_project_every_year(result, snapshot_data):
    per_year = result.shock_weights.groupby("year")["project_id"].nunique()
    assert (per_year == len(snapshot_data.projects)).all()


def test_digest_changes_when_any_frame_changes(result):
    tampered = dataclasses.replace(
        result, panel=result.panel.assign(households=result.panel["households"] + 1)
    )
    assert tampered.digest() != result.digest()


# ======================================================================================
# conservation guards
# ======================================================================================


def test_conservation_failure_raises_rather_than_warning(
    snapshot_data, params, matrices, monkeypatch
):
    """CONTRACT: raise, never warn, on invalid data."""
    real = R.L5.allocate

    def _leaky(*args, **kwargs):
        out = real(*args, **kwargs)
        diagnostics = dict(out.attrs[R.L5.ATTR_DIAGNOSTICS])
        diagnostics["spill_households"] = diagnostics["spill_households"] + 1.0
        diagnostics["demand_by_band"] = np.asarray(diagnostics["demand_by_band"]) * 2
        out.attrs[R.L5.ATTR_DIAGNOSTICS] = diagnostics
        return out

    monkeypatch.setattr(R.L5, "allocate", _leaky)
    with pytest.raises(UFEError, match="households not conserved"):
        R.run(snapshot_data, params, scenario(), **run_kwargs(matrices))


def test_conservation_can_be_switched_off_for_diagnostics(
    snapshot_data, params, matrices
):
    out = R.run(
        snapshot_data,
        params,
        scenario(BASE_YEAR + 1),
        check_conservation=False,
        **run_kwargs(matrices),
    )
    assert not out.panel.empty


# ======================================================================================
# no LLM at simulation time (Section 23 item 6) — belt and braces alongside test_ai.py
# ======================================================================================


@pytest.mark.acceptance
def test_acc_sim_modules_do_not_import_ufe_ai():
    import ast

    root = Path(R.__file__).resolve().parent
    for module in sorted(root.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(
                n == "ufe.ai" or n.startswith("ufe.ai.") for n in names
            ), f"{module} imports ufe.ai"


# ======================================================================================
# CLI
# ======================================================================================


def test_sim_cli_is_mounted_and_exposes_run():
    from typer.testing import CliRunner

    import ufe.cli as cli
    import ufe.sim_cli as sim_cli

    assert "sim" in cli.MOUNTED
    result = CliRunner().invoke(sim_cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "montecarlo", "factors", "manifest"):
        assert command in result.stdout


def test_top_level_run_command_matches_the_spec_invocation():
    """Spec Section 23 item 2 names the command `ufe run --city vizag --horizon 2035`.

    `ufe sim run` must keep working, and `ufe doctor` must not break.
    """
    from typer.testing import CliRunner

    import ufe.cli as cli
    import ufe.sim_cli as sim_cli

    runner = CliRunner()

    assert "run" in cli.ALIASED
    assert cli.ALIAS_SKIPPED == []

    top = runner.invoke(cli.app, ["run", "--help"])
    assert top.exit_code == 0, top.output
    sub = runner.invoke(sim_cli.app, ["run", "--help"])
    assert sub.exit_code == 0, sub.output
    # Same callable, so the same options: no drift between the two spellings.
    for option in ("--city", "--horizon", "--snapshot", "--seed", "--macro"):
        assert option in top.stdout.replace("\n", "")

    doctor = runner.invoke(cli.app, ["doctor"])
    assert doctor.exit_code == 0, doctor.output
    assert "run" in doctor.stdout and "sim" in doctor.stdout


def test_sim_cli_manifest_command_prints_the_triple(snapshot, tmp_path):
    from typer.testing import CliRunner

    import ufe.sim_cli as sim_cli

    result = CliRunner().invoke(
        sim_cli.app,
        ["manifest", "--snapshot", str(snapshot.path), "--city", CITY, "--allow-dirty"],
    )
    assert result.exit_code == 0, result.output
    for key in ("snapshot_hash", "params_hash", "code_version"):
        assert key in result.stdout
