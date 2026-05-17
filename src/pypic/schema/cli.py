"""Typer CLI for the schema utilities.

Exposes ``pypic schema export`` (the JSON Schema generator). Imports typer
inside the module body so :mod:`pypic.schema` itself stays free of the
optional CLI dependency — code that imports from ``pypic.schema`` for
validation only never pulls typer into the working set.

The same commands are reachable as ``python -m pypic.schema.cli export``
without installing the ``cli`` extra, useful for build scripts that don't
want to entangle their dependency graph with rich/typer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from pypic.schema._export import build_schema, dump_schema
from pypic.schema._models import SCHEMA_VERSION

app = typer.Typer(
    name="schema",
    help="Schema utilities: JSON Schema export, future validation helpers.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Stub callback that keeps subcommands from being elided.

    Without this, typer promotes a single-command app to that command at
    the root, which would break ``pypic schema export`` (typer would
    expose only ``pypic schema``). The callback forces the subcommand
    layer to stay even when only ``export`` exists today.
    """


@app.command("export")
def export(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Destination path. Use '-' (default) to write to stdout.",
        ),
    ] = Path("-"),
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty/--compact",
            help="Indented JSON (default) vs minified single-line.",
        ),
    ] = True,
    include_x_extensions: Annotated[
        bool,
        typer.Option(
            "--include-x-extensions",
            help=(
                "Annotate non-strict object schemas with patternProperties "
                "for the v1.0 'x-*' extension namespace."
            ),
        ),
    ] = False,
    inline_single_use_defs: Annotated[
        bool,
        typer.Option(
            "--inline-single-use-defs",
            help=(
                "Inline $defs entries referenced exactly once. Produces "
                "flatter Zod from json-schema-to-zod at the cost of a "
                "larger root schema."
            ),
        ),
    ] = False,
    schema_version: Annotated[
        str,
        typer.Option(
            "--schema-version",
            help=(
                "Version stamped in $id and title. Defaults to the value "
                "of pypic.schema.SCHEMA_VERSION."
            ),
        ),
    ] = SCHEMA_VERSION,
) -> None:
    """Export the simulation.toml schema as JSON Schema 2020-12.

    The default ``--pretty`` form is what gets committed to
    ``src/pypic/schema/simulation.schema.v{SCHEMA_VERSION}.json``;
    ``--compact`` is for piping into transport (HTTP responses,
    fixture-comparison harnesses).
    """
    schema = build_schema(
        include_x_extensions=include_x_extensions,
        inline_single_use_defs=inline_single_use_defs,
        schema_version=schema_version,
    )
    payload = dump_schema(schema, pretty=pretty)
    if str(output) == "-":
        sys.stdout.write(payload)
        if not pretty:
            sys.stdout.write("\n")
    else:
        output.write_text(payload)


if __name__ == "__main__":
    app()
