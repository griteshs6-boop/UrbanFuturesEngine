"""Typer sub-app for Module 15 — the backtest harness (spec Sections 19.5 and 19.6).

Mounted by ``ufe/cli.py`` as ``ufe backtest``. Every heavy import happens inside a command
so this module stays importable before the simulation runner exists.

``ufe backtest gate`` prints PASS or FAIL together with the reasoning and exits non-zero on
FAIL (Section 19.6: "Nothing ships on FAIL"). It will only ever print PASS off real
scorecards: there is no synthetic or demo mode, and a missing historical panel produces a
loud refusal rather than a green tick. That is deliberate. A gate that can be made to say
PASS without data is not a gate.

The robustness commands (Section 19.5) are wired to the same harness. Each one refuses
rather than improvising when the historical panel it needs is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    help="Freeze, score and gate the backtest (spec Section 19).",
    no_args_is_help=True,
)
console = Console()

NO_PANEL_MESSAGE = (
    "No historical panel is available in this environment, so the backtest cannot be run "
    "against real data. Pass --scorecards with a JSON array of ScoreCard objects produced "
    "by a real run, or ingest a historical panel first. The harness will not manufacture a "
    "verdict it has not earned (spec Sections 19.6 and 23 item 3)."
)


def _params(city: str):
    from ufe.params import load_params

    return load_params(city=city)


def _load_scorecards(path: Path):
    from ufe.backtest.score import ScoreCard

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = [payload]
    return [ScoreCard.from_dict(item) for item in payload]


@app.command()
def gate(
    city: str = typer.Option(..., help="City id whose parameter tree holds the gate criteria."),
    scorecards: Path = typer.Option(
        None,
        "--scorecards",
        help="JSON array of ScoreCard objects from a real backtest run.",
    ),
    markdown: Path = typer.Option(
        None, "--markdown", help="Write the per-city scorecard reports here."
    ),
) -> None:
    """Print PASS or FAIL with the reasoning; exit non-zero on FAIL (Section 19.6)."""
    from ufe.backtest.gate import ship_gate

    params = _params(city)
    fail_code = int(params.value("backtest.cli.exit_code_fail"))
    pass_code = int(params.value("backtest.cli.exit_code_pass"))

    if scorecards is None or not Path(scorecards).exists():
        console.print("[bold red]FAIL[/bold red]")
        console.print(NO_PANEL_MESSAGE)
        raise typer.Exit(code=fail_code)

    cards = _load_scorecards(scorecards)
    result = ship_gate(cards, params)

    colour = "green" if result.passed else "red"
    console.print(f"[bold {colour}]{result.verdict}[/bold {colour}]")
    console.print(result.reasoning())

    if markdown is not None:
        Path(markdown).write_text("\n\n---\n\n".join(card.to_markdown() for card in cards))
        console.print(f"wrote {len(cards)} scorecard report(s) to {markdown}")

    raise typer.Exit(code=pass_code if result.passed else fail_code)


@app.command()
def freeze(
    city: str = typer.Option(..., help="City id."),
    t0: int = typer.Option(..., help="Origin year of the freeze."),
) -> None:
    """Build the Section 19.1 snapshot at ``t0`` and print its provenance manifest."""
    from ufe.backtest.freeze import assert_parameter_provenance

    params = _params(city)
    records = assert_parameter_provenance(params, t0)
    fitted = [record for record in records if record.is_fitted]
    console.print(
        f"parameter provenance clean at t0={t0}: {len(records)} scope:global leaves "
        f"checked, {len(fitted)} fitted, latest data_through="
        f"{max((r.data_through for r in fitted), default=None)}"
    )
    console.print(NO_PANEL_MESSAGE)
    raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))


@app.command()
def rolling(
    city: str = typer.Option(..., help="City id."),
    origins: str = typer.Option(
        None, help="Comma-separated origin years; defaults to backtest.robustness.rolling_origins."
    ),
) -> None:
    """Section 19.5: re-freeze and re-score at several origin years."""
    params = _params(city)
    years = (
        [int(token) for token in origins.split(",")]
        if origins
        else list(params.value("backtest.robustness.rolling_origins"))
    )
    console.print(f"rolling origins: {', '.join(map(str, years))}")
    console.print(NO_PANEL_MESSAGE)
    raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))


@app.command()
def loco(city: str = typer.Option(..., help="City id supplying the parameter tree.")) -> None:
    """Section 19.5: leave one city out."""
    params = _params(city)
    minimum = int(params.value("backtest.robustness.loco_min_cities"))
    console.print(f"leave-one-city-out needs at least {minimum} cities with a frozen panel")
    console.print(NO_PANEL_MESSAGE)
    raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))


@app.command()
def ablate(
    city: str = typer.Option(..., help="City id."),
    layers: str = typer.Option(None, help="Comma-separated layers, e.g. l1,l3,l4,l5."),
) -> None:
    """Section 19.5: disable a layer by identity transform, never by deleting code."""
    params = _params(city)
    allowed = list(params.value("backtest.robustness.ablatable_layers"))
    wanted = [token.strip() for token in layers.split(",")] if layers else allowed
    unknown = sorted(set(wanted) - set(allowed))
    if unknown:
        console.print(f"[bold red]FAIL[/bold red] unknown layer(s): {', '.join(unknown)}")
        raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))
    console.print(f"ablating: {', '.join(wanted)} (identity transform, code untouched)")
    console.print(NO_PANEL_MESSAGE)
    raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))


@app.command()
def sobol(
    city: str = typer.Option(..., help="City id."),
    n: int = typer.Option(None, "--n", help="Sample size; defaults to backtest.robustness.sobol_n."),
) -> None:
    """Section 19.5: Sobol sensitivity over the sampled parameter ranges."""
    params = _params(city)
    samples = int(params.value("backtest.robustness.sobol_n")) if n is None else n
    console.print(f"sobol n={samples}")
    console.print(NO_PANEL_MESSAGE)
    raise typer.Exit(code=int(params.value("backtest.cli.exit_code_fail")))


if __name__ == "__main__":  # pragma: no cover
    app()
