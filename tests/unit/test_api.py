"""Tests for the API surface (spec Sections 22 and 23).

* Section 23 item 9 / Section 22.1 — "No API route exposes a raw OSM-derived column"
  -> ``test_acc_no_route_exposes_a_raw_osm_column`` (``@pytest.mark.acceptance``). The spec
  names this test ``tests/integration/test_no_derived_db_exposure.py``; it lives here
  instead because it needs no integration fixtures — it enumerates the live app's routes.
* Section 22.4 — attribution rendering into the about page and the `/attributions`
  endpoint -> ``test_about_page_renders_attributions`` and
  ``test_attributions_endpoint_renders_attributions_md``.
* Section 23 items 5 / 7 and Section 20.2 — provenance and figure verification over HTTP.

Everything runs offline: no store, no network, no API key. The app is constructed with an
in-memory :class:`ufe.api.main.RunStore` and, where a narrative is needed, the client
supplies it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ufe.api import report as report_lib
from ufe.api.main import (
    RightsGuardedRoute,
    RightsGuardedRouter,
    RunData,
    RunStore,
    ZoneResult,
    create_app,
    iter_api_routes,
)
from ufe.api.schemas import ProducedWork
from ufe.errors import DataRightsViolation
from ufe.params import (
    DEFAULT_CITIES_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_PARAMS_DIR,
    load_params,
)
from ufe.rights import CELLS_OSM_DERIVED_RAW_COLUMNS

CITY = "vizag"
SCENARIO = "base"
HORIZON = 2035
SNAPSHOT_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

#: Parameters behind the headline numbers, for the Section 0.2 confidence tags.
HEADLINE_PATHS = (
    "price.macro.scenarios.base",
    "archetypes.metro_rail.premium.0.value",
)

RUN_OUTPUT = {
    "zones": {
        "KOM": {"price_change_pct": 14.0, "drivers": {"metro": 62.0, "port": 38.0}},
        "MDL": {"price_change_pct": 6.5, "drivers": {"metro": 20.0, "port": 80.0}},
    },
    "city": {"mean_price_change_pct": 10.25},
}

FIGURES = (
    report_lib.Figure(
        figure_id="zone_price_change",
        title="Zone price change, base case",
        kind="map",
        uri="figures/zone_price_change.png",
    ),
    report_lib.Figure(
        figure_id="driver_decomposition",
        title="Driver decomposition",
        kind="chart",
        uri="figures/driver_decomposition.svg",
    ),
)

SECTIONS = (
    report_lib.ReportSection(
        name="Headline",
        body=(
            "Zone KOM rises 14.0 [zones.KOM.price_change_pct] percent in the base case, "
            "against a city mean of 10.25 [city.mean_price_change_pct] percent. "
            "[[fig:zone_price_change]]"
        ),
    ),
    report_lib.ReportSection(
        name="Drivers",
        body=(
            "KOM's rise is 62.0 [zones.KOM.drivers.metro] percent attributable to metro "
            "exposure. [[fig:driver_decomposition]]"
        ),
    ),
)


def _run(city: str = CITY) -> RunData:
    return RunData(
        city=city,
        scenario=SCENARIO,
        horizon_year=HORIZON,
        snapshot_hash=SNAPSHOT_HASH,
        zones=(
            ZoneResult("KOM", "Kommadi", 14.0, 1, {"metro": 62.0, "port": 38.0}),
            ZoneResult("MDL", "Madhurawada", 6.5, 2, {"metro": 20.0, "port": 80.0}),
        ),
        output=RUN_OUTPUT,
        figures=FIGURES,
        headline_parameter_paths=HEADLINE_PATHS,
    )


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture()
def client(params):
    application = create_app(params=params, run_store=RunStore([_run()]))
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def application(params):
    return create_app(params=params, run_store=RunStore([_run()]))


# --------------------------------------------------------------------------------------
# Section 22.1 / Section 23 item 9 — THE acceptance test
# --------------------------------------------------------------------------------------


def _model_field_names(model: type[BaseModel], seen: set[type] | None = None) -> set[str]:
    """Every field name a response model can put on the wire, recursively."""
    seen = seen if seen is not None else set()
    if model in seen or not (isinstance(model, type) and issubclass(model, BaseModel)):
        return set()
    seen.add(model)
    names: set[str] = set()
    for name, info in model.model_fields.items():
        names.add(name)
        if info.alias:
            names.add(info.alias)
        if info.serialization_alias:
            names.add(info.serialization_alias)
        for arg in (info.annotation, *getattr(info.annotation, "__args__", ())):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                names |= _model_field_names(arg, seen)
            for nested in getattr(arg, "__args__", ()):
                if isinstance(nested, type) and issubclass(nested, BaseModel):
                    names |= _model_field_names(nested, seen)
    return names


@pytest.mark.acceptance
def test_acc_no_route_exposes_a_raw_osm_column(application):
    """Section 22.1: enumerate every route and assert no response schema contains a raw
    OSM-derived column. Section 23 item 9."""
    routes = iter_api_routes(application)
    assert routes, "the app must actually have routes for this test to mean anything"

    for route in routes:
        model = route.response_model
        assert model is not None, f"route {route.path} has no response model to audit"
        assert issubclass(model, ProducedWork), (
            f"route {route.path} returns {model.__name__}, which does not subclass "
            "ProducedWork and so never ran the Section 22.1 check"
        )
        names = _model_field_names(model)
        offending = names & CELLS_OSM_DERIVED_RAW_COLUMNS
        assert not offending, f"route {route.path} exposes raw OSM column(s) {offending}"
        # And the guard itself agrees.
        from ufe.rights import assert_exposable

        assert_exposable(names)


@pytest.mark.acceptance
def test_acc_every_route_is_wrapped_by_the_response_time_guard(application):
    """The guard is structural: it is the route class, not a call each author remembers."""
    for route in iter_api_routes(application):
        assert isinstance(route, RightsGuardedRoute), route.path
    assert application.router.route_class is RightsGuardedRoute


def test_defining_a_leaking_response_model_fails_at_class_creation():
    """Definition-time half of the guard: the module would not even import."""
    with pytest.raises(DataRightsViolation) as excinfo:

        class Leaky(ProducedWork):  # noqa: D401 - deliberately illegal
            zone_id: str
            util_power: int

    assert "util_power" in str(excinfo.value)


def test_registering_a_non_produced_work_response_model_is_refused():
    """Registration-time half: a plain BaseModel cannot sneak past the base class."""

    class Sneaky(BaseModel):
        util_power: int

    router = RightsGuardedRouter(route_class=RightsGuardedRoute)
    with pytest.raises(DataRightsViolation, match="ProducedWork"):

        @router.get("/sneaky", response_model=Sneaky)
        def sneaky() -> Sneaky:  # pragma: no cover - never registered
            return Sneaky(util_power=1)


def test_response_time_guard_blocks_an_untyped_leak():
    """Response-time half: the one that cannot be bypassed.

    A route with NO response model, returning a bare dict — the exact shape that slips past
    both schema checks — is still blocked, because the guard is the route class.
    """
    leaky = FastAPI()
    leaky.router.route_class = RightsGuardedRoute
    router = APIRouter(route_class=RightsGuardedRoute)

    @router.get("/raw")
    def raw() -> dict:
        return {"h3": "8928308280fffff", "util_power": 1, "dist_arterial_m": 120.0}

    leaky.include_router(router)

    with TestClient(leaky) as test_client:
        with pytest.raises(DataRightsViolation) as excinfo:
            test_client.get("/raw")
    assert "util_power" in str(excinfo.value) or "dist_arterial_m" in str(excinfo.value)


def test_response_time_guard_catches_a_nested_leak():
    """Nested objects and arrays are walked, not just top-level keys."""
    leaky = FastAPI()
    router = APIRouter(route_class=RightsGuardedRoute)

    @router.get("/nested")
    def nested() -> dict:
        return {"zones": [{"zone_id": "KOM", "detail": {"jobs_by_sector": [1, 2, 3]}}]}

    leaky.include_router(router)
    with TestClient(leaky) as test_client:
        with pytest.raises(DataRightsViolation, match="jobs_by_sector"):
            test_client.get("/nested")


def test_data_rights_violation_is_served_as_an_error_not_the_payload(params):
    """When the guard fires on the real app the client gets an error body, not the column."""
    application = create_app(params=params, run_store=RunStore([_run()]))
    router = APIRouter(route_class=RightsGuardedRoute)

    @router.get("/oops")
    def oops() -> dict:
        return {"util_power": 1}

    application.include_router(router)
    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.get("/oops")
    assert response.status_code == 500
    assert response.json()["error"] == "data_rights_violation"
    assert "util_power" not in response.text


def test_exposure_policy_route_lists_the_blocked_columns(client):
    body = client.get("/policy/exposure").json()
    assert body["policy"] == "produced_work_only"
    assert set(body["blocked_columns"]) == CELLS_OSM_DERIVED_RAW_COLUMNS
    assert "the cells table" in body["never_exposes"]


def test_there_is_no_bulk_cells_route(application):
    """Section 22.1's design constraint: no endpoint returns the grid."""
    paths = {route.path for route in iter_api_routes(application)}
    assert not any("cell" in path for path in paths), paths
    assert not any("grid" in path for path in paths), paths


# --------------------------------------------------------------------------------------
# Section 22.4 — attribution rendering
# --------------------------------------------------------------------------------------


def test_attributions_endpoint_renders_attributions_md(client):
    body = client.get("/attributions").json()
    on_disk = Path(report_lib.ATTRIBUTIONS_PATH).read_text(encoding="utf-8").strip()
    assert body["text"] == on_disk
    assert body["sources"], "the per-source licence table must be served too"
    assert any("OpenStreetMap" in s["attribution"] for s in body["sources"])


def test_about_page_renders_attributions(client):
    """Section 22.4: "It renders into: the product about page ..."."""
    body = client.get("/about").json()
    assert body["exposure_policy"] == "produced_work_only"
    assert "OpenStreetMap" in body["attributions"]
    assert body["attributions"].strip()


# --------------------------------------------------------------------------------------
# provenance and outlooks
# --------------------------------------------------------------------------------------


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_provenance_route_carries_every_section_23_item_5_field(client):
    body = client.get(f"/cities/{CITY}/provenance").json()
    assert body["snapshot_hash"] == SNAPSHOT_HASH
    assert len(body["params_hash"]) == len(SNAPSHOT_HASH)
    assert body["git_commit"]
    assert body["calibration_level"] == "full"
    # Section 0.2 confidence tags on the parameters behind the headline numbers.
    assert set(body["confidence_tags"]) == set(HEADLINE_PATHS)
    assert set(body["confidence_tags"].values()) <= {"E", "R", "G"}


def test_outlook_returns_produced_work_only(client):
    body = client.get(f"/cities/{CITY}/outlook", params={"scenario": SCENARIO, "horizon_year": HORIZON}).json()
    assert [z["zone_id"] for z in body["zones"]] == ["KOM", "MDL"]
    assert body["zones"][0]["price_change_pct"] == 14.0
    assert {d["factor"] for d in body["zones"][0]["drivers"]} == {"metro", "port"}
    # No cell-level anything anywhere in the payload.
    assert "h3" not in json.dumps(body)


def test_unknown_run_is_404(client):
    assert client.get(f"/cities/{CITY}/outlook", params={"scenario": "nope", "horizon_year": HORIZON}).status_code == 404


def test_scenario_post(client):
    response = client.post(
        f"/cities/{CITY}/scenarios", json={"scenario": SCENARIO, "horizon_year": HORIZON}
    )
    assert response.status_code == 200
    assert response.json()["horizon_year"] == HORIZON


def test_scenario_request_forbids_unknown_fields(client):
    response = client.post(
        f"/cities/{CITY}/scenarios",
        json={"scenario": SCENARIO, "horizon_year": HORIZON, "return_cells": True},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------------------
# reports over HTTP (Section 23 item 7, Section 22.4, Section 20.2)
# --------------------------------------------------------------------------------------


def _report_body(sections=SECTIONS) -> dict:
    return {
        "title": "Vizag base case, 2035",
        "scenario": SCENARIO,
        "horizon_year": HORIZON,
        "sections": [{"name": s.name, "body": s.body} for s in sections],
    }


@pytest.mark.acceptance
def test_acc_report_route_passes_figure_verification(client):
    """Section 23 item 7, over HTTP: a full generated report verifies and renders."""
    response = client.post(f"/cities/{CITY}/reports", json=_report_body())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verification"] == {
        "figures_checked": len(FIGURES),
        "references_checked": 3,
        "numbers_checked": 3,
        "passed": True,
    }
    # Section 22.4: every report footer renders the attribution block.
    assert "OpenStreetMap" in body["footer"]
    assert "OpenStreetMap" in body["markdown"]
    # Section 23 item 5.
    assert body["provenance"]["snapshot_hash"] == SNAPSHOT_HASH
    assert body["provenance"]["git_commit"] in body["markdown"]
    assert body["calibration_flag"] is None  # vizag is calibration_level: full


def test_report_route_rejects_an_unverifiable_number(client):
    bad = (
        report_lib.ReportSection(
            name="Headline",
            body="Zone KOM rises 99.0 [zones.KOM.price_change_pct] percent. [[fig:zone_price_change]]",
        ),
        SECTIONS[1],
    )
    response = client.post(f"/cities/{CITY}/reports", json=_report_body(bad))
    assert response.status_code == 422
    assert "99.0" in response.json()["detail"]


def test_report_route_rejects_a_dangling_figure_reference(client):
    bad = (
        report_lib.ReportSection(
            name="Headline",
            body=(
                "Zone KOM rises 14.0 [zones.KOM.price_change_pct] percent. "
                "[[fig:zone_price_change]] [[fig:does_not_exist]]"
            ),
        ),
        SECTIONS[1],
    )
    response = client.post(f"/cities/{CITY}/reports", json=_report_body(bad))
    assert response.status_code == 422
    assert "does_not_exist" in response.json()["detail"]


def test_report_route_without_a_narrator_and_without_sections_is_400(client):
    response = client.post(
        f"/cities/{CITY}/reports",
        json={"title": "T", "scenario": SCENARIO, "horizon_year": HORIZON},
    )
    assert response.status_code == 400


def test_report_route_uses_a_configured_narrator(params):
    """The API MAY generate the narrative; nothing under ufe/layers or ufe/sim may."""
    calls: list[str] = []

    def narrator(run: RunData, title: str):
        calls.append(title)
        return SECTIONS

    application = create_app(
        params=params, run_store=RunStore([_run()]), narrator=narrator
    )
    with TestClient(application) as test_client:
        response = test_client.post(
            f"/cities/{CITY}/reports",
            json={"title": "Narrated", "scenario": SCENARIO, "horizon_year": HORIZON},
        )
    assert response.status_code == 200, response.text
    assert calls == ["Narrated"]


# --------------------------------------------------------------------------------------
# Section 20.2 — a class_default city's report carries the flag, over HTTP
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def class_default_params(tmp_path_factory):
    """A city identical to vizag but declaring `calibration_level: class_default`.

    Built in a tmp dir; nothing under `config/` is touched.
    """
    cities = tmp_path_factory.mktemp("cities")
    config = yaml.safe_load((Path(DEFAULT_CITIES_DIR) / f"{CITY}.yaml").read_text())
    config["city_id"] = "demoville"
    config["calibration_level"] = "class_default"
    (cities / "demoville.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return load_params(
        "demoville",
        params_dir=DEFAULT_PARAMS_DIR,
        cities_dir=cities,
        classes_file=DEFAULT_CLASSES_FILE,
    )


@pytest.mark.acceptance
def test_acc_class_default_city_report_carries_the_flag(class_default_params):
    """Section 20.2: the flag "must appear in every report it produces"."""
    application = create_app(
        params=class_default_params, run_store=RunStore([_run("demoville")])
    )
    with TestClient(application) as test_client:
        body = test_client.post("/cities/demoville/reports", json=_report_body()).json()
        provenance = test_client.get("/cities/demoville/provenance").json()

    assert body["calibration_flag"] is not None
    assert "demonstration, not a product" in body["calibration_flag"]
    assert body["provenance"]["calibration_level"] == "class_default"
    assert body["provenance"]["calibration_warning"] == body["calibration_flag"]
    assert body["calibration_flag"] in body["markdown"]
    assert body["calibration_flag"] in body["footer"]

    # And the standalone provenance route agrees.
    assert provenance["calibration_level"] == "class_default"
    assert provenance["calibration_warning"] == body["calibration_flag"]


# --------------------------------------------------------------------------------------
# the no-LLM-at-simulation-time boundary (Section 23 item 6, CONTRACT rule 4)
# --------------------------------------------------------------------------------------


def test_no_simulation_module_imports_ufe_ai():
    """`ufe/layers`, `ufe/sim` and `ufe/backtest` must not reach `ufe.ai`."""
    import ast

    root = Path(report_lib.REPO_ROOT) / "ufe"
    offenders: list[str] = []
    for package in ("layers", "sim", "backtest"):
        for path in sorted((root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(n == "ufe.ai" or n.startswith("ufe.ai.") for n in names):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders


def test_the_api_does_not_import_ufe_ai_either():
    """Prompt G reaches the API as an injected callable, never as an import.

    CONTRACT rule 4 names only `ufe/layers`, `ufe/sim` and `ufe/backtest`, but the landed
    `tests/unit/test_ai.py` check also covers `ufe/api`, and there is no reason for the API
    to import the narrator: it cannot construct an `AIClient` anyway. Injection satisfies
    both readings.
    """
    import ast

    for path in (Path(report_lib.REPO_ROOT) / "ufe" / "api").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(
                n == "ufe.ai" or n.startswith("ufe.ai.") for n in names
            ), f"{path}:{node.lineno}"


def test_report_module_does_not_import_ufe_ai():
    source = Path(report_lib.__file__).read_text()
    assert "import ufe.ai" not in source
    assert "from ufe.ai" not in source


# --------------------------------------------------------------------------------------
# the CLI sub-app
# --------------------------------------------------------------------------------------


def test_cli_sub_app_is_a_module_level_typer_app():
    import typer

    from ufe import api_cli

    assert isinstance(api_cli.app, typer.Typer)


def test_cli_routes_command_runs():
    from typer.testing import CliRunner

    from ufe import api_cli

    result = CliRunner().invoke(api_cli.app, ["routes"])
    assert result.exit_code == 0, result.output
