"""pypic command-line interface."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from pypic import __version__
from pypic._codegen_cli import app as export_app
from pypic.cli import compare, convert, inspect, plot, reduce, serve
from pypic.cli._shared import parse_steps
from pypic.schema.cli import app as schema_app

__all__ = ["app", "main", "parse_steps"]


app = typer.Typer(
    name="pypic",
    help="Inspect and compare plasma simulation output.",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pypic {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level: debug, info, warning, error.",
        ),
    ] = "warning",
    quiet: Annotated[
        bool,
        typer.Option("-q", "--quiet", help="Shorthand for --log-level error."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show full tracebacks on error."),
    ] = False,
) -> None:
    """Inspect and compare plasma simulation output."""
    level_name = "ERROR" if quiet else log_level.upper()
    if level_name not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        msg = f"Invalid --log-level {log_level!r}. Use debug, info, warning, or error."
        raise typer.BadParameter(msg)
    logging.basicConfig(level=getattr(logging, level_name), force=True)
    logging.captureWarnings(True)
    app.pretty_exceptions_enable = debug


app.command()(inspect.info)
app.command()(inspect.fields)
app.command()(inspect.stats)
app.command()(compare.compare)
app.command()(inspect.validate)
app.command()(plot.plot)
app.command(name="plot-compare")(plot.plot_compare)
app.command()(serve.serve)
app.add_typer(convert.convert_app, name="convert")
app.add_typer(schema_app, name="schema")
app.add_typer(export_app, name="export")
app.add_typer(reduce.reduce_app, name="reduce")
