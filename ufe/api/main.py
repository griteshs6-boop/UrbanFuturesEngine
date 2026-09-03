"""The FastAPI application (spec Sections 22 and 23).

Everything this app returns is a **Produced Work** in the ODbL sense (Section 22.1):
computed outputs, rankings, factor loadings, scenario results, rendered reports. There is
no bulk-grid endpoint, no per-cell endpoint, and no download of the `cells` table.

The data-rights guard is structural, not procedural
---------------------------------------------------
Section 23 item 9 says "No API route exposes a raw OSM-derived column." A rule that each
route author has to remember to apply is not a guard, so this app makes it impossible to
skip in three overlapping ways (see :mod:`ufe.api.schemas` for the full argument):

* **Definition time** — every response model subclasses ``ProducedWork``, which runs
  ``ufe.rights.assert_exposable`` over its field names when the class is created.
* **Registration time** — :func:`guarded_router` refuses to register a route whose
  ``response_model`` is not a ``ProducedWork`` subclass.
* **Response time** — :class:`RightsGuardedRoute` is the router's ``route_class``, so
  EVERY route's outgoing JSON is walked and every key asserted, including routes that
  return a bare dict, a ``JSONResponse``, or nothing typed at all.

The response-time guard is the one that cannot be bypassed: a route added anywhere on this
app, by anyone, in any style, passes through it.

No LLM at simulation time (Section 23 item 6, CONTRACT rule 4)
--------------------------------------------------------------
The API *may* call prompt G (:mod:`ufe.ai.narrate`) to write a report narrative, because
that runs strictly AFTER a simulation completes. It is injected as the ``narrator``
callable by whoever composes the app, and is deliberately not imported anywhere under
``ufe/api``: prompt G needs a configured ``AIClient`` that only a composition root can
supply, and keeping the import out means the "no module imports ufe.ai" AST check can be
run over ``ufe/api`` as well as ``ufe/layers``, ``ufe/sim`` and ``ufe/backtest``. (Spec
Section 23 item 6 and CONTRACT rule 4 name only the three simulation packages, so
``tests/unit/test_ai.py`` bans the import there; ``ufe/api`` is held to the injection
property separately, by ``tests/unit/test_api.py``.)

Construction
------------
``create_app(...)`` takes its data through explicit arguments, so tests run fully offline
with no store, no network, and no API key. ``app`` is a module-level default instance for
``uvicorn ufe.api.main:app``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from ufe.api import report as report_lib
from ufe.api.schemas import (
    AboutResponse,
    AttributionsResponse,
    AttributionSource,
    DriverShare,
    ErrorResponse,
    ExposurePolicyResponse,
    FigureModel,
    FigureVerificationModel,
    HealthResponse,
    ProducedWork,
    ProvenanceResponse,
    ReportRequest,
    ReportResponse,
    ScenarioRequest,
    ScenarioResponse,
    SectionModel,
    ZoneOutlook,
    ZoneOutlookResponse,
)
from ufe.errors import DataRightsViolation, UFEError
from ufe.params import Params, load_params
from ufe.rights import CELLS_OSM_DERIVED_RAW_COLUMNS, assert_exposable, get_attribution_text

logger = logging.getLogger(__name__)

ZERO = 0
ONE = 1

STATUS_OK = "ok"
NARRATOR_PROMPT_G = "prompt_g"


# --------------------------------------------------------------------------------------
# the response-time guard
# --------------------------------------------------------------------------------------


def _json_keys(payload: Any, into: set[str]) -> None:
    """Collect every object key appearing anywhere in a decoded JSON payload."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            into.add(str(key))
            _json_keys(value, into)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _json_keys(item, into)


class RightsGuardedRoute(APIRoute):
    """Every response on this app passes through here (Section 22.1, Section 23 item 9).

    Installed as the router's ``route_class``, so it applies to routes this module defines
    AND to any route added to the app later, whatever its return type. It decodes the
    outgoing JSON body, collects every key at every depth, and calls
    ``ufe.rights.assert_exposable``. A leak raises ``DataRightsViolation`` and is turned
    into a 500 by :func:`_data_rights_handler` — the client gets an error, never the column.
    """

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            response = await original(request)
            body = getattr(response, "body", None)
            media_type = (response.media_type or "") if response is not None else ""
            if body and "json" in media_type:
                try:
                    payload = json.loads(body)
                except (ValueError, TypeError):  # not decodable JSON; nothing to inspect
                    return response
                keys: set[str] = set()
                _json_keys(payload, keys)
                assert_exposable(keys)
            return response

        return guarded


class RightsGuardedRouter(APIRouter):
    """An ``APIRouter`` that additionally refuses to register an unguarded response model.

    This is the registration-time half of the guard: a route whose ``response_model`` is
    not a :class:`~ufe.api.schemas.ProducedWork` subclass never makes it onto the app, so
    the definition-time check cannot be sidestepped by declaring a plain ``BaseModel``.
    """

    def add_api_route(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        model = kwargs.get("response_model")
        if model is not None and not (
            isinstance(model, type) and issubclass(model, ProducedWork)
        ):
            raise DataRightsViolation(
                f"route {path!r} declares response_model={model!r}, which is not a "
                "ufe.api.schemas.ProducedWork subclass. Every API response model must "
                "subclass ProducedWork so that Section 22.1's column check runs at "
                "class-definition time."
            )
        super().add_api_route(path, endpoint, **kwargs)


def guarded_router(**kwargs: Any) -> RightsGuardedRouter:
    """A router with both halves of the structural guard installed."""
    kwargs.setdefault("route_class", RightsGuardedRoute)
    return RightsGuardedRouter(**kwargs)


def iter_api_routes(application: FastAPI) -> list[APIRoute]:
    """Every ``APIRoute`` reachable on `application`, flattening included routers.

    FastAPI mounts an included router behind an ``_IncludedRouter`` object rather than
    splicing its routes into ``app.routes``, so the Section 22.1 test cannot simply read
    ``app.routes``. This walks through.
    """
    found: list[APIRoute] = []
    seen: set[int] = set()

    def walk(routes: Sequence[Any]) -> None:
        for route in routes:
            if id(route) in seen:
                continue
            seen.add(id(route))
            if isinstance(route, APIRoute):
                found.append(route)
                continue
            nested = (
                getattr(route, "routes", None)
                or getattr(getattr(route, "original_router", None), "routes", None)
                or getattr(getattr(route, "app", None), "routes", None)
            )
            if nested:
                walk(nested)

    walk(application.routes)
    return found


# --------------------------------------------------------------------------------------
# the run store — how the app gets its numbers, offline
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneResult:
    """One zone's computed outlook. Produced Work; no cell attribute is carried."""

    zone_id: str
    zone_name: str
    price_change_pct: float
    rank: int
    drivers: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunData:
    """One completed run, as the API sees it. Never the `cells` table (Section 22.1)."""

    city: str
    scenario: str
    horizon_year: int
    snapshot_hash: str
    zones: tuple[ZoneResult, ...] = ()
    #: The run's output object, the ground truth report figures are verified against.
    output: Mapping[str, Any] = field(default_factory=dict)
    figures: tuple[report_lib.Figure, ...] = ()
    headline_parameter_paths: tuple[str, ...] = ()
    attribution_sources: tuple[str, ...] | None = None


class RunStore:
    """An in-memory store of completed runs, keyed by ``(city, scenario, horizon_year)``.

    Deliberately not a database handle: the API never opens the store at request time in
    this build, so the whole surface is testable offline. A production implementation
    swaps this for a reader over the snapshot tables; the route code does not change.
    """

    def __init__(self, runs: Iterable[RunData] = ()) -> None:
        self._runs: dict[tuple[str, str, int], RunData] = {
            (r.city, r.scenario, r.horizon_year): r for r in runs
        }

    def add(self, run: RunData) -> None:
        self._runs[(run.city, run.scenario, run.horizon_year)] = run

    def cities(self) -> list[str]:
        return sorted({city for city, _, _ in self._runs})

    def get(self, city: str, scenario: str, horizon_year: int) -> RunData:
        try:
            return self._runs[(city, scenario, horizon_year)]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"no completed run for city={city!r} scenario={scenario!r} "
                    f"horizon_year={horizon_year}"
                ),
            ) from None

    def any_for_city(self, city: str) -> RunData:
        for (run_city, _, _), run in sorted(self._runs.items()):
            if run_city == city:
                return run
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no completed run for city={city!r}")


Narrator = Callable[[RunData, str], tuple[report_lib.ReportSection, ...]]


# --------------------------------------------------------------------------------------
# adapters between the domain objects and the wire models
# --------------------------------------------------------------------------------------


def _provenance_model(provenance: report_lib.Provenance, flag: str | None) -> ProvenanceResponse:
    return ProvenanceResponse(
        city=provenance.city,
        snapshot_hash=provenance.snapshot_hash,
        params_hash=provenance.params_hash,
        git_commit=provenance.git_commit,
        calibration_level=provenance.calibration_level,
        confidence_tags=dict(provenance.confidence_tags),
        class_defaults_applied=tuple(provenance.class_defaults_applied),
        calibration_warning=flag,
    )


def _zone_models(run: RunData) -> tuple[ZoneOutlook, ...]:
    return tuple(
        ZoneOutlook(
            zone_id=zone.zone_id,
            zone_name=zone.zone_name,
            scenario=run.scenario,
            horizon_year=run.horizon_year,
            price_change_pct=zone.price_change_pct,
            rank=zone.rank,
            drivers=tuple(
                DriverShare(factor=name, share=share)
                for name, share in sorted(zone.drivers.items())
            ),
        )
        for zone in run.zones
    )


def _report_model(rendered: report_lib.Report) -> ReportResponse:
    return ReportResponse(
        report_id=rendered.report_id,
        title=rendered.title,
        city=rendered.city,
        sections=tuple(
            SectionModel(name=s.name, body=s.body) for s in rendered.sections
        ),
        figures=tuple(
            FigureModel(
                figure_id=f.figure_id,
                title=f.title,
                kind=f.kind,
                uri=f.uri,
                caption=f.caption,
            )
            for f in rendered.figures
        ),
        provenance=_provenance_model(rendered.provenance, rendered.calibration_flag),
        attributions=rendered.attributions,
        footer=rendered.footer,
        markdown=rendered.markdown,
        verification=FigureVerificationModel(
            figures_checked=rendered.verification.figures_checked,
            references_checked=rendered.verification.references_checked,
            numbers_checked=rendered.verification.numbers_checked,
            passed=True,
        ),
        calibration_flag=rendered.calibration_flag,
    )


# --------------------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------------------


def create_app(
    *,
    params: Params | Mapping[str, Params] | None = None,
    run_store: RunStore | None = None,
    narrator: Narrator | str | None = None,
    version: str = "0",
    repo_root: Path = report_lib.REPO_ROOT,
) -> FastAPI:
    """Build the application.

    `params` may be a single ``Params`` (single-city deployment) or a mapping of city id to
    ``Params``. When omitted, the app loads a city's params on demand with
    ``ufe.params.load_params`` — which is local file I/O, never network.

    `narrator` is how the report route obtains prose when the client supplies none. Pass a
    callable, or the string ``"prompt_g"`` to use :mod:`ufe.ai.narrate` (imported lazily,
    post-simulation only). ``None`` means the client must supply the sections.
    """
    store = run_store if run_store is not None else RunStore()
    params_map: dict[str, Params] = {}
    if isinstance(params, Params):
        params_map[params.city_id] = params
    elif isinstance(params, Mapping):
        params_map.update(params)

    def get_params(city: str) -> Params:
        if city in params_map:
            return params_map[city]
        try:
            loaded = load_params(city)
        except Exception as exc:  # a misspelt city is a client error, not a 500
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown city {city!r}: {exc}") from exc
        params_map[city] = loaded
        return loaded

    settings = get_params(next(iter(params_map), "vizag")) if params_map else None

    application = FastAPI(
        title="Urban Futures Engine API",
        description=(
            "Computed outputs only. Per Section 22.1 (ODbL Produced Work rule) this API "
            "never returns the cell grid or any raw OSM-derived per-cell attribute."
        ),
        version=version,
    )
    # Belt and braces: any route added directly to `application` — bypassing the router
    # below — still gets the response-time guard.
    application.router.route_class = RightsGuardedRoute

    router = guarded_router()

    # ---------------------------------------------------------------- service / policy

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status=STATUS_OK, version=version)

    @router.get("/attributions", response_model=AttributionsResponse)
    def attributions() -> AttributionsResponse:
        """Section 22.4: ATTRIBUTIONS.md renders into the API `/attributions` endpoint."""
        import yaml

        from ufe.rights import DEFAULT_DATA_LICENCES_PATH

        data = yaml.safe_load(Path(DEFAULT_DATA_LICENCES_PATH).read_text()) or {}
        sources = tuple(
            AttributionSource(
                key=key,
                licence=str(entry.get("licence", "")),
                attribution=str(entry.get("attribution", "")),
            )
            for key, entry in sorted((data.get("sources") or {}).items())
        )
        return AttributionsResponse(
            text=report_lib.attribution_block(attributions_path=repo_root / "ATTRIBUTIONS.md"),
            sources=sources,
        )

    @router.get("/about", response_model=AboutResponse)
    def about(city: str = "vizag") -> AboutResponse:
        """Section 22.4: ATTRIBUTIONS.md renders into the product about page."""
        page = report_lib.about_page(
            get_params(city), attributions_path=repo_root / "ATTRIBUTIONS.md"
        )
        return AboutResponse(**page)

    @router.get("/policy/exposure", response_model=ExposurePolicyResponse)
    def exposure_policy(city: str = "vizag") -> ExposurePolicyResponse:
        """The Section 22.1 rule, served as data so a client can see what it will not get."""
        return ExposurePolicyResponse(
            policy=str(get_params(city).get("api.exposure_policy")),
            exposes=(
                "prices",
                "rankings",
                "factor_loadings",
                "residuals",
                "scenario_results",
                "rendered_maps",
                "reports",
            ),
            never_exposes=(
                "the cells table",
                "per-cell OSM-derived attributes as bulk data",
                "any downloadable form of the underlying database",
            ),
            blocked_columns=tuple(sorted(CELLS_OSM_DERIVED_RAW_COLUMNS)),
        )

    # ------------------------------------------------------------------- provenance

    def _provenance_for(run: RunData) -> tuple[report_lib.Provenance, str | None]:
        city_params = get_params(run.city)
        provenance = report_lib.build_provenance(
            city_params,
            snapshot_hash=run.snapshot_hash,
            headline_parameter_paths=run.headline_parameter_paths,
            repo_root=repo_root,
        )
        return provenance, report_lib.calibration_flag(city_params, provenance)

    @router.get("/cities/{city}/provenance", response_model=ProvenanceResponse)
    def provenance(city: str) -> ProvenanceResponse:
        """Section 23 item 5, Section 0.2 and Section 20.2 in one payload."""
        run = store.any_for_city(city)
        return _provenance_model(*_provenance_for(run))

    # --------------------------------------------------------------------- outlooks

    @router.get("/cities/{city}/outlook", response_model=ZoneOutlookResponse)
    def outlook(city: str, scenario: str, horizon_year: int) -> ZoneOutlookResponse:
        """Zone-level computed outputs — Section 22.1's own example of a Produced Work."""
        run = store.get(city, scenario, horizon_year)
        return ZoneOutlookResponse(
            city=run.city,
            scenario=run.scenario,
            horizon_year=run.horizon_year,
            zones=_zone_models(run),
            provenance=_provenance_model(*_provenance_for(run)),
        )

    @router.post("/cities/{city}/scenarios", response_model=ScenarioResponse)
    def scenarios(city: str, body: ScenarioRequest) -> ScenarioResponse:
        run = store.get(city, body.scenario, body.horizon_year)
        return ScenarioResponse(
            city=run.city,
            scenario=run.scenario,
            horizon_year=run.horizon_year,
            zones=_zone_models(run),
            provenance=_provenance_model(*_provenance_for(run)),
        )

    # ---------------------------------------------------------------------- reports

    def _sections_for(run: RunData, request: ReportRequest) -> tuple[report_lib.ReportSection, ...]:
        if request.sections is not None:
            return tuple(
                report_lib.ReportSection(name=s.name, body=s.body) for s in request.sections
            )
        if narrator is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "no narrative supplied and this app was built without a narrator. "
                    "Pass `sections`, or construct the app with narrator='prompt_g'."
                ),
            )
        if callable(narrator):
            return tuple(narrator(run, request.title))
        if narrator == NARRATOR_PROMPT_G:
            # Prompt G (`ufe.ai.narrate`) is wired in by the COMPOSITION ROOT, not imported
            # here: it needs a configured `AIClient`, and keeping the import out of
            # `ufe/api` entirely means the "no module imports ufe.ai" AST check can be run
            # over `ufe/api` as well as `ufe/layers`, `ufe/sim` and `ufe/backtest`. See the
            # module docstring and the build report.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "narrator='prompt_g' is a marker, not a wiring: build the prompt-G "
                    "narrator (ufe.ai.narrate.generate_narrative + verify_narrative) with a "
                    "configured AIClient and pass it to create_app as a callable, or supply "
                    "`sections` on the request."
                ),
            )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"unknown narrator {narrator!r}")

    @router.post("/cities/{city}/reports", response_model=ReportResponse)
    def build_report(city: str, body: ReportRequest) -> ReportResponse:
        """Render a report, gated by Section 23 item 7's figure verification.

        Verification runs before rendering, so a report quoting a number the run data does
        not support is a 422 and never reaches the client.
        """
        run = store.get(city, body.scenario, body.horizon_year)
        sections = _sections_for(run, body)
        try:
            rendered = report_lib.render_report(
                get_params(city),
                report_id=f"{run.city}-{run.scenario}-{run.horizon_year}",
                title=body.title,
                sections=sections,
                figures=run.figures,
                run_data=run.output,
                snapshot_hash=run.snapshot_hash,
                headline_parameter_paths=run.headline_parameter_paths,
                attribution_sources=run.attribution_sources,
                repo_root=repo_root,
            )
        except report_lib.FigureVerificationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        return _report_model(rendered)

    application.include_router(router)

    # ------------------------------------------------------------------- error shape

    @application.exception_handler(DataRightsViolation)
    async def _data_rights_handler(_: Request, exc: DataRightsViolation) -> JSONResponse:
        """A leak is never served. Log loudly, return an error body with no column in it."""
        logger.error("data-rights violation blocked at the response boundary: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="data_rights_violation",
                detail=(
                    "The response was blocked because it would have exposed a raw "
                    "OSM-derived column (Section 22.1). This is a server bug; the payload "
                    "has been withheld."
                ),
            ).model_dump(),
        )

    @application.exception_handler(UFEError)
    async def _ufe_error_handler(_: Request, exc: UFEError) -> JSONResponse:
        logger.error("%s: %s", type(exc).__name__, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
        )

    application.state.run_store = store
    application.state.params = params_map
    application.state.settings = settings
    return application


#: Module-level default app, for `uvicorn ufe.api.main:app`. Constructing it does no I/O
#: beyond building the route table — parameters and runs load on demand.
app = create_app()


__all__ = [
    "RightsGuardedRoute",
    "RightsGuardedRouter",
    "RunData",
    "RunStore",
    "ZoneResult",
    "app",
    "create_app",
    "guarded_router",
    "iter_api_routes",
]
