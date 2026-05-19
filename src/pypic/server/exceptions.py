"""Server-boundary exception hierarchy.

Re-exports the four library-raised typed exceptions from
:mod:`pypic.exceptions` so HTTP/WebSocket handlers have a single
import line, and defines :class:`ValidationFailedError` — the server
wrapper for :class:`pydantic.ValidationError` from request-frame
parsing and ``simulation.toml`` validation.

The library does not raise :class:`ValidationFailedError` directly;
the server constructs it at the boundary where pydantic errors are
caught, so :func:`pypic.server.app.create_app`'s single
``@exception_handler(PypicError)`` can route every server-visible
error type through the same path.
"""

from __future__ import annotations

from typing import ClassVar

from pypic.exceptions import (
    GeometryUnsupportedError,
    PypicError,
    UnknownFieldError,
    UnknownSimulationError,
    UnknownStepError,
)

__all__ = [
    "GeometryUnsupportedError",
    "PypicError",
    "UnknownFieldError",
    "UnknownSimulationError",
    "UnknownStepError",
    "ValidationFailedError",
]


class ValidationFailedError(PypicError, ValueError):
    """Pydantic validation failed at the server boundary.

    Wraps :class:`pydantic.ValidationError` from
    :meth:`SubscribeRequest.model_validate_json` (wire frame) and from
    :func:`validate_simulation_toml` (``simulation.toml`` parse during
    :func:`open_simulation`).  The original ``ValidationError`` is
    preserved on ``__cause__`` (via ``raise … from exc``) so callers
    that need the structured error tree can still reach it.
    """

    kind: ClassVar[str] = "validation"
    status_code: ClassVar[int] = 422
