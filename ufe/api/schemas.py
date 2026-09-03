"""API request/response models, and the structural Section 22.1 data-rights guard.

Section 23 item 9 is a hard gate: **no API route may expose a raw OSM-derived column.**
`ufe.rights.assert_exposable` does the classification; the job of this module is to make
calling it *unavoidable* rather than a step each route author has to remember. Three
independent mechanisms, any one of which would catch a leak:

1. **Definition time.** Every response model subclasses :class:`ProducedWork`, whose
   ``__pydantic_init_subclass__`` hook runs :func:`~ufe.rights.assert_exposable` over the
   model's field names and serialisation aliases the moment the class body finishes
   executing. A module that declares a leaking model cannot even be imported, so the leak
   is a build failure, not a runtime one. Nested models are covered because they too must
   subclass ``ProducedWork``.
2. **Response time.** :class:`RightsGuardedRoute` (in :mod:`ufe.api.main`) is installed as
   the router's ``route_class``, so it wraps EVERY route on the app — including routes
   added later, routes with no ``response_model``, and routes that return a bare ``dict``
   or a ``JSONResponse``. It walks the outgoing JSON and asserts every key is exposable.
3. **Test time.** ``tests/unit/test_api.py::test_acc_no_route_exposes_a_raw_osm_column``
   enumerates the live app's routes and their response models (the Section 22.1 test).

The three overlap deliberately. (1) cannot see a route that returns an untyped dict; (2)
cannot run for a route nobody calls; (3) cannot see a key that only appears for certain
data. Together they close the gap.

Every model here is a **Produced Work** in the ODbL sense (Section 22.1): a price, a
ranking, a factor loading, a residual, a scenario result, a rendered map or a report.
Nothing here is a per-cell attribute and there is no bulk-grid endpoint to attach one to.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ufe.rights import assert_exposable


class ProducedWork(BaseModel):
    """Base class for every API model. Enforces Section 22.1 at class-definition time.

    Subclassing this is not a convention a route author may skip: :class:`ufe.api.main`'s
    router refuses to register a route whose ``response_model`` is not a subclass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        names: set[str] = set()
        for name, info in cls.model_fields.items():
            names.add(name)
            if info.alias:
                names.add(info.alias)
            if info.serialization_alias:
                names.add(info.serialization_alias)
        # Raises ufe.errors.DataRightsViolation, at import time, naming the column.
        assert_exposable(names)


# --------------------------------------------------------------------------------------
# service and policy
# --------------------------------------------------------------------------------------


class HealthResponse(ProducedWork):
    status: str
    version: str


class AttributionSource(ProducedWork):
    key: str
    licence: str
    attribution: str


class AttributionsResponse(ProducedWork):
    """Section 22.4: the `/attributions` endpoint renders ATTRIBUTIONS.md."""

    text: str
    sources: tuple[AttributionSource, ...] = ()


class AboutResponse(ProducedWork):
    """Section 22.4: ATTRIBUTIONS.md renders into the product about page."""

    product: str
    summary: str
    exposure_policy: str
    attributions: str


class ExposurePolicyResponse(ProducedWork):
    """What the product will and will not return (Section 22.1), served as data."""

    policy: str
    exposes: tuple[str, ...]
    never_exposes: tuple[str, ...]
    blocked_columns: tuple[str, ...]


# --------------------------------------------------------------------------------------
# provenance (Section 23 item 5, Section 0.2, Section 20.2)
# --------------------------------------------------------------------------------------


class ProvenanceResponse(ProducedWork):
    city: str
    snapshot_hash: str
    params_hash: str
    git_commit: str
    calibration_level: str
    #: Section 0.2 E/R/G tags, keyed by parameter path.
    confidence_tags: dict[str, str] = Field(default_factory=dict)
    class_defaults_applied: tuple[str, ...] = ()
    calibration_warning: str | None = None


# --------------------------------------------------------------------------------------
# scenario results (Produced Work: computed outputs, aggregated to zones)
# --------------------------------------------------------------------------------------


class DriverShare(ProducedWork):
    """A factor loading: "driven 62% by metro exposure" (Section 22.1's own example)."""

    factor: str
    share: float


class ZoneOutlook(ProducedWork):
    """Zone-level computed output. Zones, never cells; percentages, never attributes."""

    zone_id: str
    zone_name: str
    scenario: str
    horizon_year: int
    price_change_pct: float
    rank: int
    drivers: tuple[DriverShare, ...] = ()


class ZoneOutlookResponse(ProducedWork):
    city: str
    scenario: str
    horizon_year: int
    zones: tuple[ZoneOutlook, ...]
    provenance: ProvenanceResponse


class ScenarioRequest(BaseModel):
    """Request bodies are not Produced Work and carry no exposure risk, but they are
    ``extra="forbid"`` so a client cannot smuggle an unknown field through."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    horizon_year: int


class ScenarioResponse(ProducedWork):
    city: str
    scenario: str
    horizon_year: int
    zones: tuple[ZoneOutlook, ...]
    provenance: ProvenanceResponse


# --------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------


class FigureModel(ProducedWork):
    figure_id: str
    title: str
    kind: str
    uri: str | None = None
    caption: str = ""


class SectionModel(ProducedWork):
    name: str
    body: str


class FigureVerificationModel(ProducedWork):
    figures_checked: int
    references_checked: int
    numbers_checked: int
    passed: bool


class ReportResponse(ProducedWork):
    report_id: str
    title: str
    city: str
    sections: tuple[SectionModel, ...]
    figures: tuple[FigureModel, ...]
    provenance: ProvenanceResponse
    #: Section 22.4: every report footer renders ATTRIBUTIONS.md.
    attributions: str
    footer: str
    markdown: str
    verification: FigureVerificationModel
    #: Section 20.2: present on every report a class-default city produces.
    calibration_flag: str | None = None


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    scenario: str
    horizon_year: int
    #: Optional pre-written narrative. When omitted the app asks its configured narrator
    #: (``ufe.ai.narrate``, prompt G) — which runs strictly after simulation, never during.
    sections: tuple[SectionModel, ...] | None = None


class ErrorResponse(ProducedWork):
    error: str
    detail: str
