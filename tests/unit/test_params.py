"""Tests for Module 1 — `ufe/params.py` and the YAML parameter tree (spec Section 4).

The ACCEPTANCE block of Section 4 is transcribed below as `@pytest.mark.acceptance`
tests, one test per bullet. Everything else is structural invariants over the YAML
files themselves.
"""

from __future__ import annotations

import ast
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ufe import params as params_mod
from ufe.errors import MissingParameter, ParameterScopeViolation, ParameterValidationError

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
PARAMS_DIR = CONFIG / "params"
CITIES_DIR = CONFIG / "cities"
CLASSES_FILE = CONFIG / "city_classes.yaml"

PARAM_FILES = [
    "accessibility.yaml",
    "archetypes.yaml",
    "credibility.yaml",
    "supply.yaml",
    "behaviour.yaml",
    "price.yaml",
    "cascade.yaml",
]

# Blocks whose leaves are probabilities / shares and must lie in 0..1 (Section 0.3).
PROBABILITY_PREFIXES = (
    "credibility.stage_probability",
    "credibility.commitment_hardness",
    "credibility.p_cap",
    "credibility.physical_divergence.penalty_mult",
    "supply.utility_gate",
    "supply.absorption.base_growth",
    "price.macro.scenarios",
    "price.macro.scenario_probabilities",
    "price.yields",
    "price.price_data.ask_haircut_stable",
    "price.price_data.ask_haircut_soft",
    "price.price_data.blend_weight_ask",
    "behaviour.migration.inmigrant_share_by_sector",
    "behaviour.agglomeration.spillover_phi",
    "accessibility.purposes",
    "accessibility.modes",
    "accessibility.congestion",
    "cascade.p_multiplier",
)


# --------------------------------------------------------------------------- helpers


def leaves(tree, prefix=""):
    """Yield (dotted_path, leaf_dict) for every leaf in a raw or resolved tree."""
    if isinstance(tree, dict):
        if params_mod.is_leaf(tree):
            yield prefix, tree
            return
        for key, sub in tree.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            yield from leaves(sub, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(tree, list):
        for idx, sub in enumerate(tree):
            yield from leaves(sub, f"{prefix}.{idx}")


@pytest.fixture(scope="module")
def vizag():
    return params_mod.load(city="vizag")


@pytest.fixture
def sandbox(tmp_path):
    """A writable copy of `config/` so tests can corrupt it safely."""
    dst = tmp_path / "config"
    shutil.copytree(CONFIG, dst)
    return dst


def load_sandbox(sandbox, city="vizag"):
    return params_mod.load_params(
        city,
        params_dir=sandbox / "params",
        cities_dir=sandbox / "cities",
        classes_file=sandbox / "city_classes.yaml",
    )


def rewrite(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


# --------------------------------------------------------------- YAML file invariants


@pytest.mark.parametrize("name", PARAM_FILES)
def test_param_file_loads(name):
    data = yaml.safe_load((PARAMS_DIR / name).read_text())
    assert isinstance(data, dict) and data


def test_city_and_registry_files_load():
    for path in (CITIES_DIR / "vizag.yaml", CLASSES_FILE, CONFIG / "sources.yaml"):
        assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_every_leaf_has_conf_and_scope():
    for name in PARAM_FILES:
        data = yaml.safe_load((PARAMS_DIR / name).read_text())
        found = list(leaves(data, Path(name).stem))
        assert found, name
        for path, leaf in found:
            assert leaf.get("conf") in {"E", "R", "G"}, f"{path} conf={leaf.get('conf')!r}"
            assert leaf.get("scope") in {"global", "local"}, f"{path} scope missing"


def test_every_range_has_low_le_high(vizag):
    for path, leaf in leaves(vizag.resolved):
        low, high = leaf.get("low"), leaf.get("high")
        if low is None or high is None:
            continue
        assert low <= high, path
        if leaf.get("value") is not None:
            assert low <= leaf["value"] <= high, path


def test_probabilities_within_unit_interval(vizag):
    checked = 0
    for path, leaf in leaves(vizag.resolved):
        if not path.startswith(PROBABILITY_PREFIXES):
            continue
        for key in ("value", "low", "high"):
            val = leaf.get(key)
            if isinstance(val, (int, float)):
                assert 0 <= val <= 1, f"{path}.{key} = {val}"
                checked += 1
    assert checked > 0


def test_no_leaf_left_requiring_a_local_value(vizag):
    for path, leaf in leaves(vizag.resolved):
        if leaf.get("requires_local"):
            assert leaf.get("value") is not None, path


def test_sources_registry_has_a_licence_for_every_source():
    data = yaml.safe_load((CONFIG / "sources.yaml").read_text())
    assert data["sources"]
    for source_id, block in data["sources"].items():
        assert block.get("licence"), source_id
        assert block.get("obligation"), source_id
        assert "commercial_use" in block, source_id


def test_no_numeric_literals_in_params_module():
    """Section 0.1 rule 3 / CONTRACT rule 1: only 0 and 1 may appear in Python."""
    tree = ast.parse((REPO / "ufe" / "params.py").read_text())
    bad = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value not in (0, 1)
    ]
    assert bad == []


# ------------------------------------------------------------------------- Params API


def test_get_returns_leaf_and_subtree(vizag):
    leaf = vizag.get("credibility.stage_probability.funded")
    assert leaf["value"] == pytest.approx(0.77)
    assert set(vizag.get("credibility.stage_probability")) >= {"announced", "funded"}


def test_get_raises_on_unknown_path(vizag):
    with pytest.raises(MissingParameter):
        vizag.get("credibility.stage_probability.imaginary")


def test_value_conf_scope(vizag):
    assert vizag.value("credibility.discount_rate") == pytest.approx(0.12)
    assert vizag.value("credibility.discount_rate.value") == pytest.approx(0.12)
    assert vizag.conf("credibility.discount_rate") == "R"
    assert vizag.scope("credibility.discount_rate") == "global"
    assert vizag.scope("supply.elasticity_class.dense_core") == "local"


def test_sample_is_within_range_and_deterministic(vizag):
    path = "credibility.stage_probability.announced"
    leaf = vizag.get(path)
    draws = [vizag.sample(path, np.random.default_rng(7)) for _ in range(3)]
    assert draws[0] == draws[1] == draws[2]
    assert leaf["low"] <= draws[0] <= leaf["high"]
    many = [vizag.sample(path, np.random.default_rng(s)) for s in range(50)]
    assert min(many) < max(many)


def test_sample_of_a_scalar_returns_its_value(vizag):
    path = "accessibility.transit_penalties_min.transfer"
    assert vizag.sample(path, np.random.default_rng(0)) == vizag.value(path)


def test_city_overrides_are_applied(vizag):
    assert vizag.value("supply.elasticity_class.dense_core") == pytest.approx(0.10)
    assert vizag.value("price.macro.scenarios.base") == pytest.approx(0.061)


def test_city_class_defaults_are_applied_and_recorded(vizag):
    assert vizag.value("behaviour.natural_growth_rate") == pytest.approx(0.021)
    assert vizag.value("accessibility.modes.two_wheeler.share") == pytest.approx(0.42)
    assert "behaviour.natural_growth_rate.value" in vizag.class_defaults_applied


def test_manifest_carries_provenance(vizag):
    manifest = vizag.manifest()
    assert manifest["params_hash"] == vizag.hash
    assert manifest["city"] == "vizag"
    assert manifest["deviations"] == []
    assert manifest["city_class"] == "tier2_coastal_south"


def test_archetype_transcription_is_faithful(vizag):
    assert vizag.value("archetypes.metro_rail.premium.0.value") == pytest.approx(0.09)
    assert vizag.get("archetypes.metro_rail.premium.0")["max_m"] == 500
    assert vizag.value("archetypes.data_centre.employment.permanent_per_unit") == pytest.approx(0.42)
    assert vizag.conf("archetypes.data_centre.cascade.ratio") == "G"
    assert vizag.value("archetypes.electronics_assembly.employment.dormitory_share") == pytest.approx(0.68)


def test_field_caps_live_in_price_yaml(vizag):
    """Section 9.4 says the overlapping-field cap is a price.yaml parameter."""
    assert vizag.value("price.fields.cap_low") == pytest.approx(-0.45)
    assert vizag.value("price.fields.cap_high") == pytest.approx(0.60)


# -------------------------------------------------------- ACCEPTANCE — spec Section 4


@pytest.mark.acceptance
def test_acceptance_load_returns_validated_params():
    p = params_mod.load(city="vizag")
    assert isinstance(p, params_mod.Params)
    assert p.city_id == "vizag"
    assert p.value("behaviour.workers_per_household") == pytest.approx(1.45)


@pytest.mark.acceptance
@pytest.mark.parametrize("missing", ["value", "conf", "scope"])
def test_acceptance_loader_raises_when_a_leaf_lacks_value_conf_or_scope(sandbox, missing):
    def mutate(data):
        del data["discount_rate"][missing]

    rewrite(sandbox / "params" / "credibility.yaml", mutate)
    with pytest.raises(ParameterValidationError):
        load_sandbox(sandbox)


@pytest.mark.acceptance
def test_acceptance_low_greater_than_high_raises(sandbox):
    def mutate(data):
        data["discount_rate"]["low"] = data["discount_rate"]["high"] + 1

    rewrite(sandbox / "params" / "credibility.yaml", mutate)
    with pytest.raises(ParameterValidationError):
        load_sandbox(sandbox)


@pytest.mark.acceptance
def test_acceptance_phase_curve_sums_to_one(vizag):
    curve = vizag.get("archetypes.metro_rail.phase_curve")
    total = sum(leaf["value"] for _, leaf in leaves(curve))
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.acceptance
def test_acceptance_phase_curve_not_summing_to_one_raises(sandbox):
    def mutate(data):
        data["metro_rail"]["phase_curve"]["operational"]["value"] = 0.9

    rewrite(sandbox / "params" / "archetypes.yaml", mutate)
    with pytest.raises(ParameterValidationError):
        load_sandbox(sandbox)


@pytest.mark.acceptance
def test_acceptance_global_override_from_city_config_raises(sandbox):
    def mutate(data):
        data["overrides"]["credibility.discount_rate.value"] = 0.2

    rewrite(sandbox / "cities" / "vizag.yaml", mutate)
    with pytest.raises(ParameterScopeViolation):
        load_sandbox(sandbox)
    # the spec (Section 4.9) names this exception GlobalParameterOverride
    assert params_mod.GlobalParameterOverride is ParameterScopeViolation


@pytest.mark.acceptance
def test_acceptance_justified_global_override_succeeds_and_is_logged(sandbox, caplog):
    def mutate(data):
        data["_global_overrides_justification"]["credibility.discount_rate.value"] = {
            "value": 0.2,
            "reason": "State infrastructure bonds price 800bp above the panel mean.",
            "approved_by": "a.rao",
            "date": "2026-01-15",
        }

    rewrite(sandbox / "cities" / "vizag.yaml", mutate)
    with caplog.at_level("WARNING"):
        p = load_sandbox(sandbox)
    assert p.value("credibility.discount_rate") == pytest.approx(0.2)
    deviations = p.manifest()["deviations"]
    assert len(deviations) == 1
    assert deviations[0]["path"] == "credibility.discount_rate.value"
    assert deviations[0]["approved_by"] == "a.rao"
    assert "credibility.discount_rate.value" in caplog.text


@pytest.mark.acceptance
@pytest.mark.parametrize("dropped", ["reason", "approved_by", "value", "date"])
def test_acceptance_incomplete_justification_raises(sandbox, dropped):
    block = {
        "value": 0.2,
        "reason": "Local mode survey shows materially different commute tolerance.",
        "approved_by": "a.rao",
        "date": "2026-01-15",
    }
    del block[dropped]

    def mutate(data):
        data["_global_overrides_justification"]["credibility.discount_rate.value"] = block

    rewrite(sandbox / "cities" / "vizag.yaml", mutate)
    with pytest.raises(ParameterScopeViolation):
        load_sandbox(sandbox)


@pytest.mark.acceptance
def test_acceptance_estimate_global_refuses_to_write_a_local_parameter(sandbox):
    from ufe import params_cli

    with pytest.raises(ParameterScopeViolation):
        params_cli.write_global_estimates(
            params_dir=sandbox / "params",
            updates={"supply.elasticity_class.dense_core.value": 0.2},
            cities=["hyderabad", "bengaluru"],
            data_through=2019,
        )
    # ... and accepts a genuinely global one
    params_cli.write_global_estimates(
        params_dir=sandbox / "params",
        updates={"credibility.discount_rate.value": 0.13},
        cities=["hyderabad", "bengaluru"],
        data_through=2019,
    )
    reloaded = load_sandbox(sandbox)
    assert reloaded.value("credibility.discount_rate") == pytest.approx(0.13)
    assert reloaded.get("credibility.discount_rate")["fitted_on"]["data_through"] == 2019


@pytest.mark.acceptance
def test_acceptance_estimate_global_cli_exits_nonzero_for_local_parameter(sandbox, tmp_path):
    from typer.testing import CliRunner

    from ufe import params_cli

    updates = tmp_path / "updates.json"
    updates.write_text(json.dumps({"supply.elasticity_class.dense_core.value": 0.2}))
    result = CliRunner().invoke(
        params_cli.app,
        [
            "estimate",
            "global",
            "--cities",
            "hyderabad,bengaluru",
            "--data-through",
            "2019",
            "--output",
            str(sandbox / "params"),
            "--updates",
            str(updates),
        ],
    )
    assert result.exit_code != 0


@pytest.mark.acceptance
def test_acceptance_global_param_without_fitted_on_or_citation_raises(sandbox):
    def mutate(data):
        data["a_new_global_knob"] = {"value": 0.5, "conf": "R", "scope": "global"}

    rewrite(sandbox / "params" / "price.yaml", mutate)
    with pytest.raises(ParameterValidationError):
        load_sandbox(sandbox)


@pytest.mark.acceptance
def test_acceptance_global_param_with_citation_is_accepted(sandbox):
    def mutate(data):
        data["a_new_global_knob"] = {
            "value": 0.5,
            "conf": "R",
            "scope": "global",
            "citation": "Glaeser & Gyourko 2018, Journal of Economic Perspectives",
        }

    rewrite(sandbox / "params" / "price.yaml", mutate)
    assert load_sandbox(sandbox).value("price.a_new_global_knob") == pytest.approx(0.5)


@pytest.mark.acceptance
def test_acceptance_hash_is_stable_across_loads_and_processes(vizag):
    again = params_mod.load(city="vizag")
    assert vizag.hash == again.hash
    assert len(vizag.hash) == len(vizag.hash.strip()) > 0

    code = "from ufe import params; print(params.load(city='vizag').hash)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == vizag.hash


def test_hash_changes_when_a_value_changes(vizag, sandbox):
    def mutate(data):
        data["overrides"]["price.macro.scenarios.base.value"] = 0.07

    rewrite(sandbox / "cities" / "vizag.yaml", mutate)
    assert load_sandbox(sandbox).hash != vizag.hash


def test_resolved_tree_is_not_shared_between_loads(vizag):
    other = params_mod.load(city="vizag")
    other.resolved["credibility"]["discount_rate"]["value"] = 0.99
    assert vizag.value("credibility.discount_rate") == pytest.approx(0.12)


def test_deepcopy_of_resolved_tree_round_trips(vizag):
    assert copy.deepcopy(vizag.resolved) == vizag.resolved
