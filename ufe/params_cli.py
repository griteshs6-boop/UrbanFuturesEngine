"""`ufe estimate global` — the offline global-parameter estimator (spec Section 4.9).

Section 4.9: "Estimating global parameters. A separate process, never the simulation:

    ufe estimate global --cities hyderabad,bengaluru,kochi,chennai,ahmedabad \\
                        --data-through 2019 --output config/params/

It writes the fitted value, the `fitted_on` block and a fit report. It refuses to write a
`scope: local` parameter."

Only the refusal and the write are implemented here; the estimation itself needs the
cross-city panel, which is Module 15's territory. The fitted values are supplied as a JSON
mapping of `{parameter_path: value}` via `--updates`, which is what an estimator run would
emit.

Note: rewriting a parameter file through PyYAML does not preserve comments. That is
acceptable for the estimator (it is the tool of record for global values) but it is the
reason nothing else in the engine writes `config/params/`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
import yaml

from ufe.errors import ParameterScopeViolation
from ufe.params import (
    DEFAULT_PARAMS_DIR,
    SCOPE_GLOBAL,
    _assign,
    _lookup_leaf,
    _read_yaml,
    _tokens,
)

logger = logging.getLogger(__name__)

app = typer.Typer(help="Parameter tooling (spec Section 4).")
estimate_app = typer.Typer(help="Offline parameter estimation (Section 4.9).")
app.add_typer(estimate_app, name="estimate")


def write_global_estimates(
    params_dir: Path,
    updates: dict[str, Any],
    cities: list[str],
    data_through: int,
) -> list[str]:
    """Write fitted `scope: global` values plus their `fitted_on` block.

    Raises `ParameterScopeViolation` if any target is `scope: local` — the estimator is
    forbidden from touching locally calibrated parameters (Section 4.9).
    """
    params_dir = Path(params_dir)
    files = sorted(params_dir.glob("*.yaml"))
    tree: dict[str, Any] = {path.stem: _read_yaml(path) for path in files}

    for path in sorted(updates):
        leaf = _lookup_leaf(tree, path)
        if leaf.get("scope") != SCOPE_GLOBAL:
            raise ParameterScopeViolation(
                f"{path} is scope:{leaf.get('scope')}. `ufe estimate global` refuses to "
                f"write a scope:local parameter — local values belong in "
                f"config/cities/ (spec Section 4.9)."
            )

    touched: set[str] = set()
    for path in sorted(updates):
        leaf = _lookup_leaf(tree, path)
        _assign(tree, path, updates[path])
        leaf["fitted_on"] = {"cities": list(cities), "data_through": int(data_through)}
        touched.add(_tokens(path)[0])
        logger.info("estimated %s = %s (fitted_on %s)", path, updates[path], data_through)

    for namespace in sorted(touched):
        target = params_dir / f"{namespace}.yaml"
        target.write_text(yaml.safe_dump(tree[namespace], sort_keys=False))
    return sorted(touched)


@estimate_app.command("global")
def estimate_global(
    cities: str = typer.Option(..., help="Comma-separated city ids in the fitting panel."),
    data_through: int = typer.Option(..., help="Last year of data used (look-ahead guard)."),
    output: Path = typer.Option(DEFAULT_PARAMS_DIR, help="Parameter directory to write."),
    updates: Path = typer.Option(..., help="JSON mapping of {parameter_path: value}."),
) -> None:
    """Write fitted global parameters. Refuses `scope: local` targets."""
    payload = json.loads(Path(updates).read_text())
    try:
        written = write_global_estimates(
            params_dir=output,
            updates=payload,
            cities=[part.strip() for part in cities.split(",") if part.strip()],
            data_through=data_through,
        )
    except ParameterScopeViolation as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {', '.join(written)}")
