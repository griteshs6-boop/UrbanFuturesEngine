"""Parameter loading, validation, scope enforcement and provenance (spec Section 4).

The public surface is `load_params()` / `load()` and the `Params` object described in
CONTRACT.md: `get` / `value` / `sample` / `conf` / `scope` / `hash`.

Structure of the tree
---------------------
Every file in `config/params/` becomes a top-level namespace named after the file stem, so
`config/params/supply.yaml` -> `supply.*`. Paths are dotted; list elements are addressed by
integer index, e.g. `archetypes.metro_rail.premium.0.value`.

A *leaf* is any mapping that carries a `value` key (Section 4.1: scalar, range and
lookup-table forms are all leaves). Every leaf must carry `value`, `conf` in E/R/G and
`scope` in global/local; the loader raises `ParameterValidationError` on any missing one.
Ranges must satisfy `low <= value <= high`. Keys `_provenance`, `_validation` and
`_schema_version` are metadata, not parameters.

Provenance (Section 4.1, Section 19.1)
--------------------------------------
Every `scope: global` leaf must state either `fitted_on: {cities, data_through}` (estimated)
or `citation: <str>` (taken from the literature, or the sentinel `structural_assumption` for
a declared structural guess). Because most global leaves share a provenance record, a block
may declare `_provenance: {fitted_on: ..., citation: ...}` which is inherited by every leaf
beneath it unless the leaf states its own. `freeze.py` (Section 19.1) reads the effective
record via `Params.provenance(path)`.

Resolution order for a city
---------------------------
1. the global parameter files in `config/params/`;
2. the city-class defaults in `config/city_classes.yaml`, mapped onto parameter paths by
   that file's `_class_default_map` (Section 20.2), applied only where the city itself is
   silent, and recorded in `Params.class_defaults_applied`;
3. the city config's `overrides:` block — `scope: local` parameters ONLY;
4. the city config's `_global_overrides_justification:` block — the Section 4.9 escape
   hatch for `scope: global` parameters.

Scope enforcement and the approved-justification mechanism (Section 4.9, Section 23 item 10)
--------------------------------------------------------------------------------------------
Section 4.9 specifies the enforcement in pseudocode but does not fully specify what an
"approved justification" is, so the mechanism implemented here is stated explicitly:

* A path in `overrides:` that resolves to a `scope: global` leaf raises
  `ParameterScopeViolation` (Section 4.9 names this exception `GlobalParameterOverride`,
  which is exported here as an alias of the CONTRACT.md name).
* The only way to change a global parameter from a city config is to name it under
  `_global_overrides_justification:` with a block carrying ALL FOUR of
  `value`, `reason`, `approved_by`, `date`. Any missing field raises
  `ParameterScopeViolation`; `date` must be an ISO-8601 calendar date; `reason` and
  `approved_by` must be non-empty. There is no implicit approver and no wildcard.
* A `scope: local` path under `_global_overrides_justification` is also an error: it must
  go in `overrides:`, so that the deviation list only ever contains real deviations.
* Every accepted deviation is logged at WARNING with the path, the base value, the new
  value and the approver, and is recorded in `Params.deviations`, which
  `Params.manifest()` embeds so that it is "written into every simulation manifest" and
  can be flagged in report output.
* Deviations are part of the params hash, so a run with a justified override is not
  provenance-identical to a run without one.

Determinism
-----------
`value(path)` returns the deterministic `.value` used by deterministic runs.
`sample(path, rng)` draws uniformly from `low..high` with the supplied
`numpy.random.Generator` and returns `.value` when the leaf declares no range. No module
state, no global RNG.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterator, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ufe.errors import MissingParameter, ParameterScopeViolation, ParameterValidationError

logger = logging.getLogger(__name__)

# Section 4.9 calls the scope-violation exception GlobalParameterOverride; CONTRACT.md
# calls it ParameterScopeViolation. They are the same exception.
GlobalParameterOverride = ParameterScopeViolation

_PKG_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _PKG_DIR.parent / "config"
DEFAULT_PARAMS_DIR = _CONFIG_DIR / "params"
DEFAULT_CITIES_DIR = _CONFIG_DIR / "cities"
DEFAULT_CLASSES_FILE = _CONFIG_DIR / "city_classes.yaml"

VALUE_KEY = "value"
PROVENANCE_KEY = "_provenance"
VALIDATION_KEY = "_validation"
SCHEMA_VERSION_KEY = "_schema_version"
METADATA_KEYS = frozenset({PROVENANCE_KEY, VALIDATION_KEY, SCHEMA_VERSION_KEY})
# Fields that only ever appear on a leaf. A mapping carrying one of these but no `value`
# is a malformed leaf, not a block, and the loader raises on it. `low`/`high` are
# deliberately NOT in this set: Section 4.6 uses them as income-band names
# (`logit.low`, `sqm_per_hh_by_band.high`), so they are not reliable leaf markers.
LEAF_FIELDS = frozenset({"conf", "scope"})

OVERRIDES_KEY = "overrides"
JUSTIFICATION_KEY = "_global_overrides_justification"
JUSTIFICATION_FIELDS = ("value", "reason", "approved_by", "date")
CLASS_DEFAULT_MAP_KEY = "_class_default_map"
CLASSES_KEY = "classes"

SCOPE_GLOBAL = "global"
SCOPE_LOCAL = "local"
PHASE_CURVE_KEY = "phase_curve"
PHASE_CURVE_TOLERANCE_KEY = "phase_curve_sum_tolerance"
ARCHETYPES_NAMESPACE = "archetypes"
STRUCTURAL_CITATION = "structural_assumption"
PATH_SEP = "."

# The parameter namespaces Section 4 defines (one per file in `config/params/`). Provenance
# for `scope: global` leaves is enforced for these. Namespaces contributed by other modules
# (e.g. `satellite.yaml`, `ai.yaml`) are still validated structurally — every leaf needs
# value/conf/scope and a valid range — but their `fitted_on` / `citation` discipline is
# their own module's responsibility, not this loader's, so that adding a new operational
# settings file cannot break every city load.
SECTION_4_NAMESPACES = frozenset(
    {
        "accessibility",
        "archetypes",
        "behaviour",
        "cascade",
        "credibility",
        "price",
        "supply",
    }
)


def is_leaf(node: Any) -> bool:
    """True when `node` is a Section 4.1 parameter leaf (a mapping carrying `value`)."""
    return isinstance(node, dict) and VALUE_KEY in node


class FittedOn(BaseModel):
    """The `fitted_on` provenance block read by `freeze.py` (Section 19.1)."""

    model_config = ConfigDict(extra="forbid")

    cities: list[str]
    data_through: int


class LeafModel(BaseModel):
    """Schema for a Section 4.1 leaf. Extra keys (`max_m`, `min`, `target`, ...) allowed."""

    model_config = ConfigDict(extra="allow")

    # `value` is required (a mapping without it is not a leaf) but may hold any scalar or,
    # for enumerations such as a band-code list, a sequence.
    value: Any = Field(...)
    low: float | None = None
    high: float | None = None
    conf: Literal["E", "R", "G"]
    scope: Literal["global", "local"]
    dist: Literal["triangular", "uniform", "lognormal", "beta"] = "triangular"
    kind: Literal["number", "category"] = "number"
    requires_local: bool = False
    fitted_on: FittedOn | None = None
    citation: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> LeafModel:
        if self.value is None and not self.requires_local:
            raise ValueError("value is null but the leaf is not marked requires_local")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low ({self.low}) > high ({self.high})")
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            if self.low is not None and self.value < self.low:
                raise ValueError(f"value ({self.value}) < low ({self.low})")
            if self.high is not None and self.value > self.high:
                raise ValueError(f"value ({self.value}) > high ({self.high})")
        return self


# --------------------------------------------------------------------------- tree walking


def _walk(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (path, node) for every leaf, skipping metadata keys."""
    if isinstance(node, dict):
        if is_leaf(node):
            yield prefix, node
            return
        for key, sub in node.items():
            if key in METADATA_KEYS:
                continue
            yield from _walk(sub, _join(prefix, str(key)))
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from _walk(sub, _join(prefix, str(index)))


def _join(prefix: str, token: str) -> str:
    return f"{prefix}{PATH_SEP}{token}" if prefix else token


def _tokens(path: str) -> list[str]:
    return [token for token in path.split(PATH_SEP) if token]


def _child(node: Any, token: str) -> Any:
    if isinstance(node, dict):
        if token not in node:
            raise KeyError(token)
        return node[token]
    if isinstance(node, list):
        try:
            index = int(token)
        except ValueError as exc:
            raise KeyError(token) from exc
        if index < 0 or index >= len(node):
            raise KeyError(token)
        return node[index]
    raise KeyError(token)


def _lookup(tree: Any, path: str) -> Any:
    node = tree
    for token in _tokens(path):
        try:
            node = _child(node, token)
        except KeyError as exc:
            raise MissingParameter(f"no parameter at path {path!r}") from exc
    return node


def _lookup_leaf(tree: Any, path: str) -> dict[str, Any]:
    """Return the leaf governing `path`, accepting either the leaf or its `.value`."""
    node = tree
    leaf: dict[str, Any] | None = None
    for token in _tokens(path):
        try:
            node = _child(node, token)
        except KeyError as exc:
            raise MissingParameter(f"no parameter at path {path!r}") from exc
        if is_leaf(node):
            leaf = node
    if leaf is None:
        raise MissingParameter(f"path {path!r} does not resolve to a parameter leaf")
    return leaf


# ------------------------------------------------------------------------- validation


def _validate_leaves(tree: dict[str, Any]) -> None:
    errors: list[str] = []
    for path, leaf in _walk(tree):
        try:
            LeafModel.model_validate(leaf)
        except ValidationError as exc:
            for err in exc.errors():
                location = PATH_SEP.join(str(part) for part in err["loc"]) or VALUE_KEY
                errors.append(f"{path}: {location}: {err['msg']}")
    _require_conf_and_scope(tree, errors)
    if errors:
        raise ParameterValidationError(
            "invalid parameter leaves (spec Section 4.1):\n  " + "\n  ".join(sorted(errors))
        )


def _require_conf_and_scope(tree: dict[str, Any], errors: list[str]) -> None:
    """Section 4 ACCEPTANCE: every leaf has value, conf and scope."""
    for path, leaf in _walk(tree):
        for key in (VALUE_KEY, "conf", "scope"):
            if key not in leaf:
                errors.append(f"{path}: missing required field {key!r}")
    for path, node in _walk_mappings(tree):
        if is_leaf(node) or not (LEAF_FIELDS & set(node)):
            continue
        errors.append(f"{path}: missing required field {VALUE_KEY!r}")


def _walk_mappings(node: Any, prefix: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield every mapping in the tree, including leaves, skipping metadata keys."""
    if isinstance(node, dict):
        yield prefix, node
        if is_leaf(node):
            return
        for key, sub in node.items():
            if key in METADATA_KEYS:
                continue
            yield from _walk_mappings(sub, _join(prefix, str(key)))
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from _walk_mappings(sub, _join(prefix, str(index)))


def _validate_provenance(tree: dict[str, Any]) -> None:
    """Section 4 ACCEPTANCE: a global param with no fitted_on and no citation raises."""
    errors: list[str] = []
    for path, leaf, inherited in _walk_with_provenance(tree):
        if leaf.get("scope") != SCOPE_GLOBAL:
            continue
        if _tokens(path)[0] not in SECTION_4_NAMESPACES:
            continue
        record = _effective_provenance(leaf, inherited)
        if not record.get("fitted_on") and not record.get("citation"):
            errors.append(path)
    if errors:
        raise ParameterValidationError(
            "scope:global parameters with neither a fitted_on block nor a literature "
            "citation (spec Section 4.1):\n  " + "\n  ".join(sorted(errors))
        )


def _walk_with_provenance(
    node: Any, prefix: str = "", inherited: dict[str, Any] | None = None
) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    inherited = inherited or {}
    if isinstance(node, dict):
        local = node.get(PROVENANCE_KEY)
        if isinstance(local, dict):
            inherited = {**inherited, **local}
        if is_leaf(node):
            yield prefix, node, inherited
            return
        for key, sub in node.items():
            if key in METADATA_KEYS:
                continue
            yield from _walk_with_provenance(sub, _join(prefix, str(key)), inherited)
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from _walk_with_provenance(sub, _join(prefix, str(index)), inherited)


def _effective_provenance(
    leaf: dict[str, Any], inherited: dict[str, Any]
) -> dict[str, Any]:
    record = dict(inherited)
    for key in ("fitted_on", "citation"):
        if leaf.get(key) is not None:
            record[key] = leaf[key]
    return record


def _validate_phase_curves(tree: dict[str, Any]) -> None:
    """Section 4 ACCEPTANCE: phase_curve fractions sum to 1.0 within the YAML tolerance."""
    archetypes = tree.get(ARCHETYPES_NAMESPACE) or {}
    tolerance = (archetypes.get(VALIDATION_KEY) or {}).get(PHASE_CURVE_TOLERANCE_KEY)
    if tolerance is None:
        raise ParameterValidationError(
            f"archetypes.{VALIDATION_KEY}.{PHASE_CURVE_TOLERANCE_KEY} is required "
            "(the phase-curve sum tolerance may not be a literal in Python)"
        )
    errors: list[str] = []
    for path, curve in _find_key(archetypes, PHASE_CURVE_KEY, ARCHETYPES_NAMESPACE):
        if not isinstance(curve, dict):
            continue
        total = 0
        for _, leaf in _walk(curve):
            total += leaf[VALUE_KEY]
        if abs(total - 1) > tolerance:
            errors.append(f"{path}: fractions sum to {total}, not 1")
    if errors:
        raise ParameterValidationError(
            "phase_curve fractions must sum to 1 (spec Section 4.3):\n  "
            + "\n  ".join(sorted(errors))
        )


def _find_key(node: Any, wanted: str, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, sub in node.items():
            path = _join(prefix, str(key))
            if key == wanted:
                yield path, sub
            else:
                yield from _find_key(sub, wanted, path)
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from _find_key(sub, wanted, _join(prefix, str(index)))


def _validate(tree: dict[str, Any]) -> None:
    """Full validation of the authored files, before any override is applied."""
    _validate_leaves(tree)
    _validate_provenance(tree)
    _validate_phase_curves(tree)


def _validate_resolved(tree: dict[str, Any]) -> None:
    """Re-validate after overrides.

    The `low <= value <= high` rule is deliberately NOT re-applied here. `low`/`high` state
    the range over which the parameter was estimated, which is a property of the authored
    file; an approved Section 4.9 deviation, or a locally calibrated value, may legitimately
    fall outside it. Landing outside the range is logged as a warning by `_assign` instead.
    """
    errors: list[str] = []
    _require_conf_and_scope(tree, errors)
    for path, leaf in _walk(tree):
        low, high = leaf.get("low"), leaf.get("high")
        if low is not None and high is not None and low > high:
            errors.append(f"{path}: low ({low}) > high ({high})")
    if errors:
        raise ParameterValidationError(
            "invalid resolved parameter leaves (spec Section 4.1):\n  "
            + "\n  ".join(sorted(errors))
        )
    _validate_provenance(tree)
    _validate_phase_curves(tree)


# --------------------------------------------------------------------------- overriding


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ParameterValidationError(f"missing configuration file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ParameterValidationError(f"{path} does not contain a YAML mapping")
    return data


def _assign(tree: dict[str, Any], path: str, value: Any) -> Any:
    """Set the value at `path`, returning the previous value. Raises on a non-leaf path."""
    tokens = _tokens(path)
    if not tokens:
        raise MissingParameter("empty parameter path")
    container: Any = tree
    for token in tokens[:-1]:
        try:
            container = _child(container, token)
        except KeyError as exc:
            raise MissingParameter(f"no parameter at path {path!r}") from exc
    last = tokens[-1]
    try:
        target = _child(container, last)
    except KeyError as exc:
        raise MissingParameter(f"no parameter at path {path!r}") from exc

    if last == VALUE_KEY and is_leaf(container):
        previous = container[VALUE_KEY]
        container[VALUE_KEY] = value
        _warn_if_outside_range(path, container, value)
        return previous
    if is_leaf(target):
        previous = target[VALUE_KEY]
        target[VALUE_KEY] = value
        _warn_if_outside_range(path, target, value)
        return previous
    raise ParameterValidationError(
        f"override path {path!r} does not address a parameter leaf or its value"
    )


def _warn_if_outside_range(path: str, leaf: dict[str, Any], value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    low, high = leaf.get("low"), leaf.get("high")
    if (low is not None and value < low) or (high is not None and value > high):
        logger.warning(
            "%s resolved to %s, outside its estimated range [%s, %s]",
            path,
            value,
            low,
            high,
        )


def _apply_overrides(
    tree: dict[str, Any], city_cfg: dict[str, Any], city_id: str
) -> None:
    """Section 4.9 `apply_overrides`: refuse scope:global paths outright."""
    overrides = city_cfg.get(OVERRIDES_KEY) or {}
    for path in sorted(overrides):
        leaf = _lookup_leaf(tree, path)
        if leaf.get("scope") == SCOPE_GLOBAL:
            raise ParameterScopeViolation(
                f"{path} is scope:global and is estimated across cities. "
                f"To change it, re-estimate on the full city panel and update "
                f"config/params/ — not config/cities/. "
                f"(city={city_id}; the Section 4.9 escape hatch is "
                f"{JUSTIFICATION_KEY} with {', '.join(JUSTIFICATION_FIELDS)}.)"
            )
    for path in sorted(overrides):
        _assign(tree, path, overrides[path])


def _apply_justified_overrides(
    tree: dict[str, Any], city_cfg: dict[str, Any], city_id: str
) -> list[dict[str, Any]]:
    """Section 4.9 escape hatch. Returns the recorded deviations, one per override."""
    blocks = city_cfg.get(JUSTIFICATION_KEY) or {}
    deviations: list[dict[str, Any]] = []
    for path in sorted(blocks):
        block = blocks[path]
        if not isinstance(block, dict):
            raise ParameterScopeViolation(
                f"{JUSTIFICATION_KEY}[{path!r}] must be a mapping with "
                f"{', '.join(JUSTIFICATION_FIELDS)}"
            )
        missing = [field for field in JUSTIFICATION_FIELDS if field not in block]
        if missing:
            raise ParameterScopeViolation(
                f"{JUSTIFICATION_KEY}[{path!r}] is missing required field(s) "
                f"{', '.join(missing)}. A global-parameter deviation requires a value, a "
                f"reason, a named approver and a date (spec Section 4.9)."
            )
        for field in ("reason", "approved_by"):
            if not str(block[field]).strip():
                raise ParameterScopeViolation(
                    f"{JUSTIFICATION_KEY}[{path!r}].{field} must not be empty"
                )
        try:
            approved_on = _date.fromisoformat(str(block["date"]))
        except ValueError as exc:
            raise ParameterScopeViolation(
                f"{JUSTIFICATION_KEY}[{path!r}].date must be an ISO-8601 date"
            ) from exc

        leaf = _lookup_leaf(tree, path)
        if leaf.get("scope") != SCOPE_GLOBAL:
            raise ParameterScopeViolation(
                f"{path} is scope:{leaf.get('scope')} and does not need a justification. "
                f"Move it to the {OVERRIDES_KEY!r} block."
            )
        previous = _assign(tree, path, block["value"])
        deviation = {
            "path": path,
            "city": city_id,
            "base_value": previous,
            "override_value": block["value"],
            "reason": str(block["reason"]),
            "approved_by": str(block["approved_by"]),
            "date": approved_on.isoformat(),
            "conf": leaf.get("conf"),
            "scope": leaf.get("scope"),
        }
        deviations.append(deviation)
        logger.warning(
            "GLOBAL PARAMETER DEVIATION: %s overridden for city %s from %s to %s "
            "(approved_by=%s, date=%s, reason=%s)",
            path,
            city_id,
            previous,
            block["value"],
            deviation["approved_by"],
            deviation["date"],
            deviation["reason"],
        )
    return deviations


def _class_default_overrides(
    classes: dict[str, Any], city_class: str | None
) -> dict[str, Any]:
    """Map `city_classes.yaml` defaults onto parameter paths (Section 20.2)."""
    if city_class is None:
        return {}
    mapping = classes.get(CLASS_DEFAULT_MAP_KEY) or {}
    blocks = classes.get(CLASSES_KEY) or {}
    if city_class not in blocks:
        raise ParameterValidationError(
            f"unknown city_class {city_class!r}; known: {sorted(blocks)}"
        )
    block = blocks[city_class]
    resolved: dict[str, Any] = {}
    for source_key in sorted(mapping):
        node: Any = block
        for token in _tokens(source_key):
            if not isinstance(node, dict) or token not in node:
                node = None
                break
            node = node[token]
        if node is not None:
            resolved[mapping[source_key]] = node
    return resolved


def _apply_class_defaults(
    tree: dict[str, Any], defaults: dict[str, Any], claimed: set[str]
) -> list[str]:
    applied: list[str] = []
    for path in sorted(defaults):
        if path in claimed:
            continue
        leaf = _lookup_leaf(tree, path)
        if leaf.get("scope") == SCOPE_GLOBAL:
            raise ParameterScopeViolation(
                f"{path} is scope:global and cannot be supplied by a city-class default"
            )
        _assign(tree, path, defaults[path])
        applied.append(path)
    return applied


def _check_local_values_supplied(tree: dict[str, Any]) -> None:
    missing = [
        path
        for path, leaf in _walk(tree)
        if leaf.get("requires_local") and leaf.get(VALUE_KEY) is None
    ]
    if missing:
        raise ParameterValidationError(
            "parameters marked requires_local have no value after resolving the city "
            "config and city-class defaults:\n  " + "\n  ".join(sorted(missing))
        )


# ------------------------------------------------------------------------------- Params


class Params:
    """Loaded, validated parameter tree. Access by dotted path (CONTRACT.md)."""

    def __init__(
        self,
        city_id: str,
        city_class: str | None,
        resolved: dict[str, Any],
        deviations: list[dict[str, Any]],
        class_defaults_applied: list[str],
        source_files: list[str],
        city_config: dict[str, Any],
    ) -> None:
        self.city_id = city_id
        self.city_class = city_class
        self._resolved = resolved
        self._deviations = deviations
        self._class_defaults_applied = class_defaults_applied
        self._source_files = source_files
        self._city_config = city_config
        self._hash = self._compute_hash()

    # ------------------------------------------------------------------ introspection

    @property
    def resolved(self) -> dict[str, Any]:
        """The resolved tree. Each `load_params` call owns its own copy."""
        return self._resolved

    @property
    def deviations(self) -> list[dict[str, Any]]:
        """Approved `scope: global` deviations, for the manifest and report output."""
        return copy.deepcopy(self._deviations)

    @property
    def class_defaults_applied(self) -> list[str]:
        return list(self._class_defaults_applied)

    @property
    def city_config(self) -> dict[str, Any]:
        return copy.deepcopy(self._city_config)

    # ---------------------------------------------------------------------- accessors

    def get(self, path: str) -> Any:
        """Resolved node at `path` — a leaf mapping or a subtree."""
        return _lookup(self._resolved, path)

    def leaf(self, path: str) -> dict[str, Any]:
        """The leaf governing `path`, accepting either the leaf or its `.value`."""
        return _lookup_leaf(self._resolved, path)

    def value(self, path: str) -> float:
        """Deterministic value of a scalar-or-range (Section 4.1)."""
        return self.leaf(path)[VALUE_KEY]

    def sample(self, path: str, rng: Any) -> float:
        """Monte Carlo draw: uniform on low..high, or `.value` when there is no range."""
        leaf = self.leaf(path)
        low, high = leaf.get("low"), leaf.get("high")
        if low is None or high is None:
            return leaf[VALUE_KEY]
        return float(rng.uniform(low, high))

    def conf(self, path: str) -> str:
        """'E' | 'R' | 'G' (Section 0.2)."""
        return self.leaf(path)["conf"]

    def scope(self, path: str) -> str:
        """'global' | 'local' (Section 4.9)."""
        return self.leaf(path)["scope"]

    def provenance(self, path: str) -> dict[str, Any]:
        """Effective `fitted_on` / `citation` for `path`, including inherited records."""
        target = self.leaf(path)
        for _, leaf, inherited in _walk_with_provenance(self._resolved):
            if leaf is target:
                return _effective_provenance(leaf, inherited)
        return {}

    # ----------------------------------------------------------------------- identity

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "city": self.city_id,
                "city_class": self.city_class,
                "params": self._resolved,
                "deviations": self._deviations,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        """sha256 of the resolved tree, for provenance (CONTRACT.md)."""
        return self._hash

    def manifest(self) -> dict[str, Any]:
        """The provenance block embedded in every simulation manifest (Section 4.9)."""
        return {
            "params_hash": self._hash,
            "city": self.city_id,
            "city_class": self.city_class,
            "calibration_level": self._city_config.get("calibration_level"),
            "source_files": list(self._source_files),
            "class_defaults_applied": list(self._class_defaults_applied),
            "deviations": copy.deepcopy(self._deviations),
        }

    def __repr__(self) -> str:
        return f"Params(city={self.city_id!r}, hash={self._hash!r})"


# --------------------------------------------------------------------------- entry point


def load_params(
    city: str,
    params_dir: Path = DEFAULT_PARAMS_DIR,
    cities_dir: Path = DEFAULT_CITIES_DIR,
    classes_file: Path = DEFAULT_CLASSES_FILE,
) -> Params:
    """Load, validate and resolve the parameter tree for `city` (spec Section 4)."""
    params_dir, cities_dir = Path(params_dir), Path(cities_dir)
    files = sorted(params_dir.glob("*.yaml"))
    if not files:
        raise ParameterValidationError(f"no parameter files found in {params_dir}")

    tree: dict[str, Any] = {path.stem: _read_yaml(path) for path in files}
    _validate(tree)

    city_cfg = _read_yaml(cities_dir / f"{city}.yaml")
    city_id = city_cfg.get("city_id", city)
    city_class = city_cfg.get("city_class")

    classes = _read_yaml(Path(classes_file))
    claimed = set(city_cfg.get(OVERRIDES_KEY) or {}) | set(
        city_cfg.get(JUSTIFICATION_KEY) or {}
    )
    defaults = _class_default_overrides(classes, city_class)
    class_defaults_applied = _apply_class_defaults(tree, defaults, claimed)

    _apply_overrides(tree, city_cfg, city_id)
    deviations = _apply_justified_overrides(tree, city_cfg, city_id)

    _check_local_values_supplied(tree)
    _validate_resolved(tree)

    if class_defaults_applied and city_cfg.get("calibration_level") != "class_default":
        logger.info(
            "city %s declares calibration_level=%s but inherits %d city-class default(s): %s",
            city_id,
            city_cfg.get("calibration_level"),
            len(class_defaults_applied),
            ", ".join(class_defaults_applied),
        )

    return Params(
        city_id=city_id,
        city_class=city_class,
        resolved=tree,
        deviations=deviations,
        class_defaults_applied=class_defaults_applied,
        source_files=[str(path.name) for path in files],
        city_config=city_cfg,
    )


def load(city: str, **kwargs: Any) -> Params:
    """Alias for `load_params`, matching the Section 4 ACCEPTANCE block's spelling."""
    return load_params(city, **kwargs)
