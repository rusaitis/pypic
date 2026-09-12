"""Server-boundary exception hierarchy and error routing.

Re-exports the library-raised typed exceptions from
[`pypic.exceptions`][pypic.exceptions] (base `PypicError` plus its four
subclasses) so HTTP/WebSocket handlers have a single import line,
defines `ValidationFailedError` — the server wrapper for
`pydantic.ValidationError` from request-frame parsing and
``simulation.toml`` validation — and owns `error_routing`, the one
table mapping exception type to wire kind and HTTP status. The library
raises the types; only this boundary knows what they mean over HTTP.

The library does not raise `ValidationFailedError` directly;
the server constructs it at the boundary where pydantic errors are
caught, so [`pypic.server.app.create_app`][pypic.server.app.create_app]'s single
``@exception_handler(PypicError)`` can route every server-visible
error type through the same path.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from pypic.exceptions import (
    GeometryUnsupportedError,
    PypicError,
    UnknownFieldError,
    UnknownSimulationError,
    UnknownStepError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypic.server.protocol import ErrorKind

__all__ = [
    "GeometryUnsupportedError",
    "PypicError",
    "UnknownFieldError",
    "UnknownSimulationError",
    "UnknownStepError",
    "ValidationFailedError",
    "error_routing",
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


_ROUTING: Mapping[type[PypicError], tuple[ErrorKind, int]] = MappingProxyType(
    {
        PypicError: ("internal", 500),
        UnknownSimulationError: ("unknown_sim", 404),
        UnknownFieldError: ("unknown_field", 404),
        UnknownStepError: ("unknown_step", 404),
        GeometryUnsupportedError: ("geometry_unsupported", 400),
        ValidationFailedError: ("validation", 422),
    }
)


def error_routing(exc: PypicError) -> tuple[ErrorKind, int]:
    """Return the ``(wire kind, HTTP status)`` for a raised pypic error.

    Walks the MRO rather than looking the exact type up: ``ErrorKind``
    (in [`pypic.server.protocol`][pypic.server.protocol]) is a closed
    vocabulary, so a subclass nobody mapped must degrade to its nearest
    mapped base instead of putting an unknown string on the wire.
    ``PypicError`` itself maps to ``internal`` / 500, which is why the
    walk always terminates.
    """
    for base in type(exc).__mro__:
        route = _ROUTING.get(base)
        if route is not None:
            return route
    return _ROUTING[PypicError]  # unreachable; mypy wants the exit
