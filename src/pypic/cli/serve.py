"""``pypic serve``: the Arrow IPC server."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def serve(
    root: Annotated[
        Path,
        typer.Argument(help="Directory whose subdirectories contain simulation.toml."),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port number."),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Auto-reload on code change (dev)."),
    ] = False,
    cors_origin: Annotated[
        list[str] | None,
        typer.Option(
            "--cors-origin",
            help="CORS-allowed origin (repeatable). Default '*' for local dev.",
        ),
    ] = None,
) -> None:
    # The backslash before [server] is for typer's rich help renderer,
    # which otherwise parses the bracket as a style tag and silently
    # drops it — printing an install command that does not install the
    # extra. tests/test_cli.py pins the rendered output.
    r"""Run the Arrow IPC + JSON HTTP server.

    Exposes JSON discovery routes (/health, /sims, ...) and one WebSocket
    endpoint (/sims/{sim}/stream) that streams field slices as Arrow IPC
    bytes to webpic and other Arrow-aware clients.

    Requires the server extra: pip install "pypic-plasma\[server]"

    The wire protocol is documented at
    https://rusaitis.github.io/pypic/api/server/
    """
    try:
        from pypic.server.app import serve as _serve
    except ImportError as exc:
        typer.echo(
            "pypic serve requires the server extra. "
            'Install with: pip install "pypic-plasma[server]"',
            err=True,
        )
        raise typer.Exit(1) from exc

    origins = tuple(cors_origin) if cors_origin else ("*",)
    _serve(root, host=host, port=port, reload=reload, cors_origins=origins)
