"""Typer CLI for the schema utilities.

Exposes three subcommands:

- ``export`` — emits the JSON Schema 2020-12 document for
  ``simulation.toml`` v1.0 (the cross-tool contract).
- ``validate`` — checks one or more ``simulation.toml`` files against
  the schema; reports per-file pass/fail with Pydantic error details.
- ``diff`` — diffs two JSON Schema documents (the second defaults to
  the bundled current schema); text or structured JSON output.

Lives outside [`pypic.schema`][pypic.schema] because it imports typer:
the subpackage itself holds to stdlib and pydantic only, so it can be
lifted into a standalone distribution and so library users who only
need ``validate_simulation_toml`` don't pull the optional CLI
dependency into their working set. `pypic._codegen_cli` sits outside
`pypic.codegen` for the same reason.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pypic.schema._diff import schema_structured_diff, schema_text_diff
from pypic.schema._export import build_schema, dump_schema, get_schema_path
from pypic.schema._loader import ValidationError, validate_simulation_toml
from pypic.schema._models import SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(
    name="schema",
    help="Schema utilities: JSON Schema export, TOML validation, and schema diff.",
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

    The default --pretty form is what gets committed to
    src/pypic/schema/simulation.schema.v{SCHEMA_VERSION}.json;
    --compact is for piping into transport (HTTP responses,
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
    else:
        output.write_text(payload)


@app.command("validate")
def validate(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "simulation.toml file(s) to validate. Pass '-' to read TOML "
                "from stdin (single source only when stdin is used)."
            ),
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit machine-readable JSON ({files: [{path, ok, errors}]}) to "
                "stdout instead of the colored human summary."
            ),
        ),
    ] = False,
) -> None:
    """Validate simulation.toml file(s) against the v1.0 schema.

    Exit codes: 0 if every file passes, 1 if any file fails Pydantic
    validation, 2 if any file is missing, unreadable, or syntactically
    broken TOML (hard errors override validation failures in the exit
    code).
    """
    if len(paths) > 1 and any(str(p) == "-" for p in paths):
        typer.echo(
            "'-' (stdin) may only be used when it is the sole input.",
            err=True,
        )
        raise typer.Exit(code=2)

    results: list[dict[str, Any]] = []
    exit_code = 0

    for path in paths:
        display = "<stdin>" if str(path) == "-" else str(path)
        try:
            source = _read_source(path)
            validate_simulation_toml(source)
        except ValidationError as exc:
            results.append(
                {"path": display, "ok": False, "errors": _coerce_errors(exc)}
            )
            exit_code = max(exit_code, 1)
        except tomllib.TOMLDecodeError as exc:
            results.append(
                {
                    "path": display,
                    "ok": False,
                    "errors": [
                        {"type": "toml_decode_error", "loc": [], "msg": str(exc)}
                    ],
                }
            )
            exit_code = 2
        except (FileNotFoundError, OSError) as exc:
            results.append(
                {
                    "path": display,
                    "ok": False,
                    "errors": [{"type": "io_error", "loc": [], "msg": str(exc)}],
                }
            )
            exit_code = 2
        else:
            results.append({"path": display, "ok": True, "errors": []})

    if json_output:
        sys.stdout.write(json.dumps({"files": results}, indent=2) + "\n")
    else:
        _print_validate_human(results)

    raise typer.Exit(code=exit_code)


@app.command("diff")
def diff(
    a: Annotated[
        Path,
        typer.Argument(
            help="First JSON Schema document. Pass '-' to read from stdin.",
        ),
    ],
    b: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Second JSON Schema document. Defaults to the bundled current "
                "schema (resolved via --bundled-version)."
            ),
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'text' (unified diff, default) or 'json'.",
        ),
    ] = "text",
    schema_version: Annotated[
        str,
        typer.Option(
            "--bundled-version",
            help=(
                "Schema version to resolve when the second argument is "
                "omitted. Defaults to pypic.schema.SCHEMA_VERSION."
            ),
        ),
    ] = SCHEMA_VERSION,
) -> None:
    """Diff two JSON Schema documents; second defaults to bundled current.

    Direction is left → right: lines prefixed - and entries under
    removed are in the first document, lines prefixed + and
    entries under added are in the second.

    Exit codes follow git diff: 0 if the documents are identical, 1
    if they differ, 2 on hard errors (file not found, malformed JSON,
    unknown --format).
    """
    if fmt not in ("text", "json"):
        typer.echo(f"--format must be 'text' or 'json' (got {fmt!r})", err=True)
        raise typer.Exit(code=2)

    if b is None:
        b = get_schema_path(schema_version)
    elif str(b) == "-":
        typer.echo(
            "'-' (stdin) is only accepted for the first argument; "
            "the second must be a file path.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        a_doc = _load_json(a)
        b_doc = _load_json(b)
    except json.JSONDecodeError as exc:
        typer.echo(f"JSON decode error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if fmt == "text":
        out = schema_text_diff(a_doc, b_doc, fromfile=str(a), tofile=str(b))
        sys.stdout.write(out)
        has_diff = bool(out)
    else:
        structured = schema_structured_diff(a_doc, b_doc)
        sys.stdout.write(json.dumps(structured, indent=2) + "\n")
        has_diff = bool(
            structured["added"] or structured["removed"] or structured["changed"]
        )

    raise typer.Exit(code=1 if has_diff else 0)


# Helpers.


def _read_source(path: Path) -> bytes:
    """Return raw TOML bytes from a path or stdin (``-``)."""
    if str(path) == "-":
        return sys.stdin.buffer.read()
    return path.read_bytes()


def _load_json(path: Path) -> dict[str, Any]:
    """Parse a JSON file or stdin (``-``) into a dict."""
    if str(path) == "-":
        return json.loads(sys.stdin.read())  # type: ignore[no-any-return]
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _coerce_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Reduce ``ValidationError.errors()`` to the JSON-safe subset.

    Pydantic's native ``errors()`` already returns JSON-compatible
    dicts, but the optional ``ctx`` field can contain unserializable
    values (regex objects, exceptions); we project to the fields we
    actually care about.
    """
    out: list[dict[str, Any]] = []
    for err in exc.errors():
        out.append(
            {
                "type": err.get("type", "validation_error"),
                "loc": list(err.get("loc", ())),
                "msg": err.get("msg", ""),
            }
        )
    return out


def _print_validate_human(results: Iterable[dict[str, Any]]) -> None:
    """Render per-file pass/fail with indented Pydantic error lines."""
    from rich.console import Console

    console = Console(stderr=True)
    for entry in results:
        if entry["ok"]:
            console.print(f"[green]✓[/green] {entry['path']}")
            continue
        console.print(f"[red]✗[/red] {entry['path']}")
        for err in entry["errors"]:
            loc = ".".join(str(part) for part in err.get("loc", [])) or "<root>"
            console.print(f"    [dim]{loc}[/dim]: {err.get('msg', '')}")


if __name__ == "__main__":
    app()
