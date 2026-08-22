"""FastAPI app factory for the pypic server.

:func:`create_app` builds a ready-to-mount ``FastAPI`` instance with
the discovery HTTP routes and the streaming WebSocket endpoint
wired up against a single simulation root.

The factory pattern (rather than a module-level app singleton) lets
tests construct independent apps against per-test fixtures, and lets
production deployments host multiple roots in one process if desired.

CORS is permissive (`allow_origins=["*"]`) by default to match the
local-dev story where webpic runs on a different port.  Production
deployments must tighten this via ``cors_origins=`` — wide-open
defaults must not survive the dev → prod transition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fastapi import FastAPI

__all__ = ["create_app", "serve"]


def create_app(
    root: Path,
    *,
    cors_origins: Sequence[str] = ("*",),
) -> FastAPI:
    """Build a FastAPI app serving simulations under *root*.

    Parameters
    ----------
    root : Path
        Directory whose subdirectories contain ``simulation.toml``
        files (one per simulation).  Discovery walks this tree on
        every ``GET /sims`` request; readers open lazily on first
        access to a named simulation.
    cors_origins : sequence of str
        Origins the CORS middleware accepts.  Default ``("*",)`` is
        permissive — suitable for local dev with webpic running on
        a different port.  Tighten to an explicit allowlist for
        production.

    Returns
    -------
    FastAPI
        The configured application.  Mount it under uvicorn /
        gunicorn / hypercorn; see :func:`serve` for the simple
        single-process launch path.
    """
    try:
        from fastapi import APIRouter, FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        msg = (
            "pypic.server requires FastAPI. "
            "Install with: pip install pypic-plasma[server]"
        )
        raise ImportError(msg) from exc

    from pypic.server._state import SimulationRegistry
    from pypic.server.exceptions import PypicError
    from pypic.server.routes import register_routes
    from pypic.server.stream import register_stream

    app = FastAPI(
        title="pypic",
        description=(
            "Read, analyze, and stream plasma simulation output. "
            "Discovery via JSON HTTP routes; binary field data via the "
            "Arrow IPC WebSocket at /sims/{sim}/stream."
        ),
        version=_pypic_version(),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.exception_handler(PypicError)
    async def _pypic_error_handler(
        request: Request,
        exc: PypicError,
    ) -> JSONResponse:
        """Route every typed pypic error to its declared HTTP status.

        Body shape is ``{"kind": <wire kind>, "detail": <message>}`` —
        ``detail`` stays back-compat with the previous
        ``HTTPException(detail=str(exc))`` shape; ``kind`` is additive
        and matches the WebSocket :class:`ErrorFrame.kind` literal so
        clients can dispatch identically across both transports.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"kind": exc.kind, "detail": exc.detail},
        )

    app.state.registry = SimulationRegistry(root)

    router = APIRouter()
    register_routes(router)
    register_stream(router)
    app.include_router(router)

    return app


def serve(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    cors_origins: Sequence[str] = ("*",),
) -> None:
    """Launch a uvicorn server hosting :func:`create_app`.

    Convenience wrapper for ``pypic serve``; production deployments
    typically construct the app via :func:`create_app` and invoke
    their own ASGI server.

    Raises
    ------
    ImportError
        uvicorn is not installed (``pip install pypic-plasma[server]``).
    """
    try:
        import uvicorn
    except ImportError as exc:
        msg = (
            "pypic serve requires uvicorn. "
            "Install with: pip install pypic-plasma[server]"
        )
        raise ImportError(msg) from exc

    app = create_app(root, cors_origins=cors_origins)
    uvicorn.run(app, host=host, port=port, reload=reload)


def _pypic_version() -> str:
    """Look up the installed pypic version (returns ``"unknown"`` if absent).

    Shared by the ``/health`` probe and the FastAPI ``version`` field
    so the two never drift.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version("pypic")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
