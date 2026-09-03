"""State adapters (spec Section 6.0).

Adding a state means adding **one module to this package** and nothing else: the registry
in :mod:`ufe.ingest.adapters.base` discovers every module here at run time, so no import
list, no dispatch table and no ingester has to change (Section 23 item 11).

The re-exports below are conveniences for callers; they are not the registration mechanism.
"""

from __future__ import annotations

from ufe.ingest.adapters.base import (
    CAPABILITIES,
    AccessTerms,
    AdapterError,
    StateAdapter,
    StateAdapterBase,
    UnknownStateAdapter,
    available_state_codes,
    discover,
    get_adapter,
    missing_capabilities,
    register,
    registry,
)

__all__ = [
    "CAPABILITIES",
    "AccessTerms",
    "AdapterError",
    "StateAdapter",
    "StateAdapterBase",
    "UnknownStateAdapter",
    "available_state_codes",
    "discover",
    "get_adapter",
    "missing_capabilities",
    "register",
    "registry",
]
