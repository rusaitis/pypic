"""Server-boundary exception hierarchy.

Re-exports the library-raised typed exceptions from
[`pypic.exceptions`][pypic.exceptions] (base `PypicError` plus its four
subclasses) so HTTP/WebSocket handlers have a single import line, and
defines `ValidationFailedError` — the server wrapper for
`pydantic.ValidationError` from request-frame parsing and
``simulation.toml`` validation.

The library does not raise `ValidationFailedError` directly;
the server constructs it at the boundary where pydantic errors are
caught, so [`pypic.server.app.create_app`][pypic.server.app.create_app]'s single
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

    Wraps `pydantic.ValidationError` from
    `SubscribeRequest.model_validate_json` (wire frame) and from
    `validate_simulation_toml` (``simulation.toml`` parse during
    `open_simulation`).  The original ``ValidationError`` is
    preserved on ``__cause__`` (via ``raise … from exc``) so callers
    that need the structured error tree can still reach it.
    """

    kind: ClassVar[str] = "validation"
    status_code: ClassVar[int] = 422
