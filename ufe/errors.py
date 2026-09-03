"""Exception hierarchy for the Urban Futures Engine.

Every failure mode the spec names as "must raise, not warn" gets a named exception here.
"""

from __future__ import annotations


class UFEError(Exception):
    """Base class for every error raised by the engine."""


class MissingCriticalLayer(UFEError):
    """A data layer the spec marks as a hard requirement is absent.

    Raised by ingesters, e.g. CZMP/CRZ for a coastal city (Section 20.2 step 4).
    """


class ParameterScopeViolation(UFEError):
    """A `scope: global` parameter was overridden from a city config (Section 4.9)."""


class MissingParameter(UFEError):
    """A parameter path was requested that does not exist in the loaded tree."""


class ParameterValidationError(UFEError):
    """A parameter file failed schema validation."""


class SchemaValidationError(UFEError):
    """A dataframe failed its pandera schema on write (Section 3)."""


class ConvergenceError(UFEError):
    """An iterative solve failed to converge — agglomeration divergence, market clearing."""


class DeterminismError(UFEError):
    """Two runs with identical inputs and seed produced different output."""


class LicenceViolation(UFEError):
    """A Red-class dependency or an unmapped data source was detected (Section 2.4)."""


class DataRightsViolation(UFEError):
    """An API route attempted to expose a raw OSM-derived column (Section 22.1)."""


class BacktestGateFailure(UFEError):
    """The backtest gate did not pass (Section 19.6)."""


class CoverageError(UFEError):
    """Data coverage fell below the threshold required to proceed (Section 20.2 step 9)."""


class NonMetricCRSError(UFEError):
    """A metric operation (area / distance / buffer) was attempted on a geographic CRS.

    Spec Section 0.3: "Never compute distance in degrees. There is a test for this."
    Re-exported from :mod:`ufe.geo`, where it was originally defined.
    """


class MissingArchetypeError(UFEError):
    """A cascade entry names a ``target_archetype`` that ``archetypes.yaml`` does not define.

    Section 14.1 reads the injected jobs' ``sector`` and ``median_wage_inr_mo`` off the
    target archetype and there is no honest default for either, so this is deliberately
    fatal. Re-exported from :mod:`ufe.layers.cascade`, where it was originally defined.
    """


class LookAheadError(UFEError):
    """A frozen backtest run would have used information that postdates ``t0``.

    Section 19.1 / Section 21. Raised, never warned: a contaminated backtest that runs is
    worse than one that refuses to, because it produces a number somebody will quote.
    Re-exported from :mod:`ufe.backtest.freeze`, where it was originally defined.
    """


class SurvivorshipContamination(UFEError):
    """The frozen pipeline has been filtered to projects that ultimately succeeded.

    Section 19.2. "A t0 pipeline containing only projects that ultimately completed makes
    the credibility layer untestable. This assertion is not optional."
    Re-exported from :mod:`ufe.backtest.freeze`, where it was originally defined.
    """
