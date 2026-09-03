"""The state adapter pattern (spec Section 6.0).

Data splits three ways by portability — national, state, city. The *state* tier is what
makes the engine multi-city rather than single-city, and it is reached through an adapter
selected by ``city.state_code``. Section 23 item 11 is the design constraint this module
exists to satisfy:

    onboarding a second city needs "no code changes outside ``ufe/ingest/adapters/``".

Concretely that means:

* every state-tier ingester (:mod:`ufe.ingest.prices`, :mod:`ufe.ingest.rera`,
  :mod:`ufe.ingest.cadastral`) takes a :class:`StateAdapter` and calls only the seven
  methods of the protocol — it never branches on a state code;
* a new state is added by dropping one module into ``ufe/ingest/adapters/`` that defines a
  class decorated with :func:`register` (or subclassing :class:`StateAdapterBase`).
  :func:`discover` imports every module in this package, so nothing outside the package —
  no registry list, no import statement, no ``if state == ...`` — has to change.
  ``tests/unit/test_adapters.py`` asserts exactly that by writing a fictional adapter into
  this package at test time and checking the whole state-tier pipeline picks it up.

Honesty rules from Section 6.0, enforced here rather than left to each adapter:

* "Any method may return ``None`` where the state does not publish it." The base class
  returns ``None`` for everything and each adapter overrides what its state actually has.
* ``capabilities()`` is derived from :attr:`StateAdapterBase.provides` and is written into
  the manifest "so a report can state honestly which layers were unavailable".
* "A missing capability lowers ``data_conf``; it never silently imputes" —
  :func:`missing_capabilities` is what :func:`ufe.ingest.core.data_conf` is fed.
* ``access_terms()`` "is not decorative": :class:`AccessTerms` carries the portal terms,
  the bulk-access policy and a rate limit, and :meth:`AccessTerms.assert_bulk_access` is
  called by every ingester before it reads a portal in bulk (Section 22.2).

This module contains no numeric literals: the default rate limit comes from
``config/ingest.yaml``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Iterable, Mapping, Protocol, runtime_checkable

import geopandas as gpd
import pandas as pd

from ufe.errors import DataRightsViolation, UFEError

logger = logging.getLogger(__name__)

__all__ = [
    "CAPABILITIES",
    "CAP_GUIDANCE_VALUES",
    "CAP_REGISTRATION_TRANSACTIONS",
    "CAP_RERA_PROJECTS",
    "CAP_CADASTRAL_PARCELS",
    "CAP_INDUSTRIAL_ALLOTMENTS",
    "AdapterError",
    "UnknownStateAdapter",
    "AccessTerms",
    "StateAdapter",
    "StateAdapterBase",
    "register",
    "registry",
    "discover",
    "get_adapter",
    "available_state_codes",
    "satisfies_protocol",
    "assert_adapter",
    "missing_capabilities",
    "ADAPTER_METHODS",
]


# --------------------------------------------------------------------------------------
# Capability vocabulary — one per data-returning method of the protocol
# --------------------------------------------------------------------------------------

CAP_GUIDANCE_VALUES = "guidance_values"
CAP_REGISTRATION_TRANSACTIONS = "registration_transactions"
CAP_RERA_PROJECTS = "rera_projects"
CAP_CADASTRAL_PARCELS = "cadastral_parcels"
CAP_INDUSTRIAL_ALLOTMENTS = "industrial_allotments"

#: The full capability set. ``capabilities()`` returns a subset; the complement is what
#: lowers ``data_conf`` and is named in the report as unavailable.
CAPABILITIES: tuple[str, ...] = (
    CAP_GUIDANCE_VALUES,
    CAP_REGISTRATION_TRANSACTIONS,
    CAP_RERA_PROJECTS,
    CAP_CADASTRAL_PARCELS,
    CAP_INDUSTRIAL_ALLOTMENTS,
)

#: Every method a conforming adapter must expose (Section 6.0).
ADAPTER_METHODS: tuple[str, ...] = CAPABILITIES + ("capabilities", "access_terms")


class AdapterError(UFEError):
    """An adapter is malformed or does not satisfy the Section 6.0 protocol."""


class UnknownStateAdapter(UFEError):
    """No adapter is registered for a city's ``state_code`` (Section 20.2 step 2)."""


# --------------------------------------------------------------------------------------
# access_terms()
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AccessTerms:
    """The terms an adapter operates under (Section 6.0, Section 22.2).

    ``bulk_access_allowed=False`` means the portal's terms of service prohibit bulk
    automated collection. :meth:`assert_bulk_access` then raises
    :class:`ufe.errors.DataRightsViolation` rather than letting an ingester scrape it —
    Section 22.3 ranks unauthorised bulk access to government portals as the second
    largest legal exposure in the system.
    """

    state_code: str
    portals: Mapping[str, str] = field(default_factory=dict)
    tos_urls: Mapping[str, str] = field(default_factory=dict)
    bulk_access_allowed: bool = False
    min_seconds_between_requests: float | None = None
    licence: str = ""
    notes: str = ""

    def rate_limit_s(self) -> float:
        """The rate limit to honour, defaulting to ``config/ingest.yaml``."""
        if self.min_seconds_between_requests is not None:
            return float(self.min_seconds_between_requests)
        from ufe.ingest.core import cfg

        return float(cfg("reader.default_min_seconds_between_requests"))

    def assert_bulk_access(self, what: str) -> None:
        """Refuse a bulk automated read the portal's terms do not permit."""
        if not self.bulk_access_allowed:
            raise DataRightsViolation(
                f"{self.state_code}: bulk automated access to {what} is not permitted "
                f"under the terms recorded in access_terms() "
                f"({self.tos_urls or self.portals}). Section 22.2 requires the adapter to "
                "declare the terms it operates under and the ingester to respect them."
            )

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["portals"] = dict(self.portals)
        record["tos_urls"] = dict(self.tos_urls)
        record["rate_limit_s"] = self.rate_limit_s()
        return record


# --------------------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class StateAdapter(Protocol):
    """Section 6.0, verbatim. Any method may return ``None`` where the state does not
    publish that dataset."""

    state_code: str

    def guidance_values(self, city: Any) -> pd.DataFrame:
        """Registration-department guidance values by SRO and village/ward (Section 6.7b)."""

    def registration_transactions(self, city: Any) -> pd.DataFrame | None:
        """Recorded transactions, where the state publishes them."""

    def rera_projects(self, city: Any) -> pd.DataFrame:
        """Registered project listings from the state RERA portal (Section 6.8)."""

    def cadastral_parcels(self, city: Any) -> gpd.GeoDataFrame | None:
        """Parcel polygons with survey-number attributes (Section 6.9)."""

    def industrial_allotments(self, city: Any) -> pd.DataFrame | None:
        """State industrial-corporation land allotments."""

    def capabilities(self) -> set[str]:
        """Which of :data:`CAPABILITIES` this state actually publishes."""

    def access_terms(self) -> dict:
        """Portal ToS, rate limits, bulk-access policy (Section 22.2)."""


class StateAdapterBase:
    """Optional base class: implements the "returns ``None``" default for every method.

    A state adapter is free to satisfy :class:`StateAdapter` structurally without
    inheriting from this, but subclassing gives:

    * ``capabilities()`` derived from :attr:`provides`, so the two can never disagree;
    * ``None`` for every dataset the state does not publish, with a log line naming it;
    * automatic registration by ``state_code``.
    """

    #: Two-letter state code, e.g. ``"AP"``. Must be set by the subclass.
    state_code: ClassVar[str] = ""
    #: The subset of :data:`CAPABILITIES` this state publishes.
    provides: ClassVar[frozenset[str]] = frozenset()
    #: Human-readable state name, for reports.
    state_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        unknown = set(cls.provides) - set(CAPABILITIES)
        if unknown:
            raise AdapterError(
                f"{cls.__name__}.provides names unknown capabilities: {sorted(unknown)}; "
                f"known: {list(CAPABILITIES)}"
            )
        if cls.state_code:
            register(cls)

    # -- the seven protocol methods ---------------------------------------------------

    def _unavailable(self, capability: str) -> None:
        logger.info(
            "%s does not publish %s; the layer is reported unavailable and data_conf is "
            "lowered — it is never imputed (spec Section 6.0)",
            self.state_code or type(self).__name__,
            capability,
        )
        return None

    def guidance_values(self, city: Any) -> pd.DataFrame:
        raise AdapterError(
            f"{type(self).__name__} declares {CAP_GUIDANCE_VALUES} but does not implement it"
        )

    def registration_transactions(self, city: Any) -> pd.DataFrame | None:
        return self._unavailable(CAP_REGISTRATION_TRANSACTIONS)

    def rera_projects(self, city: Any) -> pd.DataFrame:
        raise AdapterError(
            f"{type(self).__name__} declares {CAP_RERA_PROJECTS} but does not implement it"
        )

    def cadastral_parcels(self, city: Any) -> gpd.GeoDataFrame | None:
        return self._unavailable(CAP_CADASTRAL_PARCELS)

    def industrial_allotments(self, city: Any) -> pd.DataFrame | None:
        return self._unavailable(CAP_INDUSTRIAL_ALLOTMENTS)

    def capabilities(self) -> set[str]:
        return set(self.provides)

    def access_terms(self) -> dict:
        raise AdapterError(
            f"{type(self).__name__} must declare access_terms() (spec Section 6.0/22.2)"
        )


# --------------------------------------------------------------------------------------
# Registry + discovery
# --------------------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}
_DISCOVERED = False


def register(cls: type) -> type:
    """Register an adapter class by its ``state_code``. Usable as a decorator."""
    code = getattr(cls, "state_code", "")
    if not code:
        raise AdapterError(f"{cls.__name__} has no state_code; cannot register it")
    key = str(code).upper()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls and existing.__name__ != cls.__name__:
        logger.warning(
            "state adapter for %s re-registered: %s replaces %s",
            key,
            cls.__name__,
            existing.__name__,
        )
    _REGISTRY[key] = cls
    return cls


def discover(force: bool = False) -> dict[str, type]:
    """Import every module in ``ufe/ingest/adapters/`` so each one registers itself.

    This is the whole of the "no code changes outside ``ufe/ingest/adapters/``" mechanism:
    a new state is a new file in this package. Modules that fail to import are logged and
    skipped rather than breaking every other state.
    """
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_REGISTRY)
    package = importlib.import_module(__package__)
    # A module dropped into the package after this process started is invisible to
    # `pkgutil.iter_modules` until the import system's directory cache is dropped. Without
    # this, "onboarding a state is one new file" would only work after a restart.
    importlib.invalidate_caches()
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        name = f"{__package__}.{info.name}"
        try:
            module = importlib.import_module(name)
            if force:
                module = importlib.reload(module)
        except Exception:  # pragma: no cover - a broken adapter must not break the rest
            logger.exception("failed to import state adapter module %s", name)
            continue
        # Belt and braces: register any conforming class the module forgot to decorate.
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and getattr(obj, "state_code", "")
                and obj is not StateAdapterBase
                and str(obj.state_code).upper() not in _REGISTRY
                and satisfies_protocol(obj)
            ):
                register(obj)
    _DISCOVERED = True
    return dict(_REGISTRY)


def registry() -> dict[str, type]:
    """The registered adapters, running discovery first."""
    discover()
    return dict(_REGISTRY)


def available_state_codes() -> list[str]:
    return sorted(registry())


def get_adapter(state_code: str, **kwargs: Any) -> StateAdapter:
    """Instantiate the adapter for ``state_code`` (Section 6.0: selected by the city config).

    Raises :class:`UnknownStateAdapter` when the state has no adapter — Section 20.2 step 2
    ("Confirm a state adapter exists for ``city.state_code``. If not, build one (~1 wk)")
    is a gate, not a fallback.
    """
    table = registry()
    key = str(state_code).upper()
    if key not in table:
        raise UnknownStateAdapter(
            f"no state adapter for {state_code!r}; available: {sorted(table)}. "
            "Add one module to ufe/ingest/adapters/ (spec Section 20.2 step 2) — no code "
            "outside that package needs to change."
        )
    adapter = table[key](**kwargs)
    assert_adapter(adapter)
    return adapter


def satisfies_protocol(candidate: Any) -> bool:
    """True when ``candidate`` (class or instance) exposes the whole Section 6.0 protocol."""
    if not getattr(candidate, "state_code", ""):
        return False
    return all(callable(getattr(candidate, name, None)) for name in ADAPTER_METHODS)


def assert_adapter(adapter: Any) -> None:
    """Raise :class:`AdapterError` unless ``adapter`` satisfies the protocol."""
    missing = [name for name in ADAPTER_METHODS if not callable(getattr(adapter, name, None))]
    if not getattr(adapter, "state_code", ""):
        missing.append("state_code")
    if missing:
        raise AdapterError(
            f"{type(adapter).__name__} does not satisfy the Section 6.0 StateAdapter "
            f"protocol; missing: {sorted(missing)}"
        )
    caps = adapter.capabilities()
    unknown = set(caps) - set(CAPABILITIES)
    if unknown:
        raise AdapterError(
            f"{type(adapter).__name__}.capabilities() names unknown capabilities "
            f"{sorted(unknown)}; known: {list(CAPABILITIES)}"
        )
    terms = adapter.access_terms()
    if not isinstance(terms, Mapping):
        raise AdapterError(
            f"{type(adapter).__name__}.access_terms() must return a mapping "
            "(spec Section 6.0)"
        )


def missing_capabilities(adapter: Any, wanted: Iterable[str] = CAPABILITIES) -> set[str]:
    """Capabilities ``adapter`` does not provide — the ``data_conf`` deduction (Section 6.0)."""
    return set(wanted) - set(adapter.capabilities())
