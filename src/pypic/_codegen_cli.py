"""Typer CLI for codegen export — JSON bundle for cross-language tooling.

Mirrors :mod:`pypic.schema.cli`: thin commands delegating to the pure
helpers in :mod:`pypic.codegen`, with stable ``sort_keys`` JSON output.
``bundle`` is the one webpic's codegen consumes; the per-table commands
(``aliases``, ``recipes``, ``fields``) exist for inspection and debugging.

typer is imported here only so library users of :mod:`pypic.codegen`
don't pull the optional CLI dependency into their working set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from pypic.codegen import export_aliases, export_bundle, export_fields, export_recipes
from pypic.schema._models import SCHEMA_VERSION

app = typer.Typer(
    name="export",
    help="Export pypic schema/aliases/recipes/fields as JSON for codegen.",
    no_args_is_help=True,
)

_OutputOption = Annotated[
    Path,
    typer.Option(
        "--output", "-o", help="Destination path. Use '-' (default) for stdout."
    ),
]
_PrettyOption = Annotated[
    bool,
    typer.Option(
        "--pretty/--compact", help="Indented JSON (default) vs minified single-line."
    ),
]


@app.callback()
def _root() -> None:
    """Stub callback so subcommands aren't elided (see pypic.schema.cli._root)."""


def _write(payload: dict[str, Any], output: Path, *, pretty: bool) -> None:
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    if str(output) == "-":
        sys.stdout.write(text)
    else:
        output.write_text(text)


@app.command("bundle")
def bundle(
    output: _OutputOption = Path("-"),
    pretty: _PrettyOption = True,
    include_x_extensions: Annotated[
        bool,
        typer.Option(
            "--include-x-extensions",
            help="Annotate non-strict objects with the 'x-*' extension namespace.",
        ),
    ] = False,
    inline_single_use_defs: Annotated[
        bool,
        typer.Option(
            "--inline-single-use-defs",
            help="Inline single-use $defs in the schema (flatter Zod output).",
        ),
    ] = False,
    schema_version: Annotated[
        str,
        typer.Option(
            "--schema-version", help="Version stamped in the embedded schema."
        ),
    ] = SCHEMA_VERSION,
) -> None:
    """Export the unified codegen bundle (schema + aliases + recipes + fields)."""
    payload = export_bundle(
        schema_version=schema_version,
        inline_single_use_defs=inline_single_use_defs,
        include_x_extensions=include_x_extensions,
    )
    _write(payload, output, pretty=pretty)


@app.command("aliases")
def aliases(output: _OutputOption = Path("-"), pretty: _PrettyOption = True) -> None:
    """Export the alias tables (compute aliases, group aliases, species regex)."""
    _write(export_aliases(), output, pretty=pretty)


@app.command("recipes")
def recipes(output: _OutputOption = Path("-"), pretty: _PrettyOption = True) -> None:
    """Export the recipe registry and per-species templates (metadata only)."""
    _write(export_recipes(), output, pretty=pretty)


@app.command("fields")
def fields(output: _OutputOption = Path("-"), pretty: _PrettyOption = True) -> None:
    """Export static field metadata (units, LaTeX, long names)."""
    _write(export_fields(), output, pretty=pretty)


if __name__ == "__main__":
    app()
