"""`ufe api` — the API sub-app, mounted by `ufe/cli.py`.

Nothing here does work at import time: the FastAPI app and the parameter tree are built
inside the command bodies. Host, port and every other setting come from
`config/params/api.yaml` (Section 0.1 rule 3).
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="api",
    help="Serve the API and inspect what it is allowed to expose (Section 22).",
    no_args_is_help=True,
)
console = Console()

DEFAULT_CITY = "vizag"

P_HOST = "api.serve.host"
P_PORT = "api.serve.port"


@app.command()
def serve(
    city: str = typer.Option(DEFAULT_CITY, help="City whose parameter tree the app loads."),
    host: str = typer.Option(None, help="Override api.serve.host."),
    port: int = typer.Option(None, help="Override api.serve.port."),
    reload: bool = typer.Option(False, help="Reload on source change (development only)."),
) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    from ufe.params import load_params

    params = load_params(city)
    uvicorn.run(
        "ufe.api.main:app",
        host=host or str(params.get(P_HOST)),
        port=int(port if port is not None else params.value(P_PORT)),
        reload=reload,
    )


@app.command()
def routes(city: str = typer.Option(DEFAULT_CITY)) -> None:
    """List every route and its response model — the Section 22.1 audit, by hand."""
    from ufe.api.main import create_app, iter_api_routes
    from ufe.api.schemas import ProducedWork
    from ufe.params import load_params

    application = create_app(params=load_params(city))
    table = Table(title="ufe API routes (Produced Work only, Section 22.1)")
    table.add_column("method")
    table.add_column("path")
    table.add_column("response model")
    table.add_column("guarded")
    for route in sorted(iter_api_routes(application), key=lambda r: r.path):
        model = route.response_model
        guarded = isinstance(model, type) and issubclass(model, ProducedWork)
        table.add_row(
            ",".join(sorted(route.methods or ())),
            route.path,
            getattr(model, "__name__", "-"),
            "[green]yes[/green]" if guarded else "[red]NO[/red]",
        )
    console.print(table)


@app.command()
def exposure() -> None:
    """Print the columns the data-rights guard will refuse to expose (Section 22.1)."""
    from ufe.rights import CELLS_OSM_DERIVED_RAW_COLUMNS

    console.print_json(json.dumps({"blocked_columns": sorted(CELLS_OSM_DERIVED_RAW_COLUMNS)}))


@app.command()
def attributions() -> None:
    """Print the attribution block rendered into the about page and every report footer."""
    from ufe.api.report import attribution_block

    console.print(attribution_block())
