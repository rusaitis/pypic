"""Typed exception hierarchy for pypic.

Library-side raise sites use these instead of bare ``KeyError`` /
``NotImplementedError`` so callers — the Starlette server in
particular — can dispatch on type rather than sniff exception
messages.  Subclasses inherit from the appropriate standard base
(``KeyError`` / ``NotImplementedError``) in addition to
`PypicError`, so existing ``except KeyError`` and
``except NotImplementedError`` callers keep working unchanged.

The ``kind`` and ``status_code`` classvars are server-routing
metadata: the Starlette/FastAPI layer reads them to build
[`pypic.server.protocol.ErrorFrame`][pypic.server.protocol.ErrorFrame] (WebSocket) and
`fastapi.HTTPException` (HTTP) responses without a
dispatch table.  They are inert for non-server callers — the
classvar values impose no behavior on the library itself.

The server-only
[`pypic.server.exceptions.ValidationFailedError`][pypic.server.exceptions.ValidationFailedError]
wraps `pydantic.ValidationError` from request-frame parsing
and ``simulation.toml`` validation; it lives next to the server
because the wire format is its only consumer.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "GeometryUnsupportedError",
    "PypicError",
    "UnknownFieldError",
    "UnknownSimulationError",
    "UnknownStepError",
]


class PypicError(Exception):
    """Base for every typed pypic exception.

    Subclasses set ``kind`` (matching `ErrorFrame.kind` literals
    on the WebSocket wire format) and ``status_code`` (the HTTP status
    a server boundary should emit).  Defaults route to the catch-all
    ``"internal"`` / ``500`` so a raw `PypicError` raised by
    accident is still routable.
    """

    kind: ClassVar[str] = "internal"
    status_code: ClassVar[int] = 500

    @property
    def detail(self) -> str:
        r"""Human-readable message, free of ``KeyError``-style requoting.

        The `KeyError`-inheriting subclasses (Unknown\*Error)
        otherwise ``str()`` to ``"'msg'"`` because ``KeyError.__str__``
        calls ``repr()`` on ``args[0]``.  Wire consumers (the HTTP body's
        ``detail`` field, `ErrorFrame.message`) want the bare message,
        which ``args[0]`` gives directly — stripping quotes off
        ``str(exc)`` would mangle messages that legitimately carry them.
        """
        if self.args:
            return str(self.args[0])
        return super().__str__()


class UnknownSimulationError(PypicError, KeyError):
    """No simulation with the requested name exists under the registry root.

    Subclass of `KeyError` so callers that catch the broader type
    still work, while letting the server route to the ``unknown_sim``
    error kind without inspecting message strings.
    """

    kind: ClassVar[str] = "unknown_sim"
    status_code: ClassVar[int] = 404


class UnknownFieldError(PypicError, KeyError):
    """A requested field name does not resolve in the dataset.

    Raised by `FieldDataset.resolve_key` and by
    `Simulation.read` when ``strict_fields=True`` (the default)
    and one of the requested names matched no loaded field.
    """

    kind: ClassVar[str] = "unknown_field"
    status_code: ClassVar[int] = 404


class UnknownStepError(PypicError, KeyError):
    """A requested timestep is not available for the simulation."""

    kind: ClassVar[str] = "unknown_step"
    status_code: ClassVar[int] = 404


class GeometryUnsupportedError(PypicError, NotImplementedError):
    """An operation is not implemented for the dataset's coordinate geometry.

    Examples: `regrid` on spherical geometry, derivative-based
    derived quantities on non-Cartesian grids, spatial-axis reductions
    on non-Cartesian grids.
    """

    kind: ClassVar[str] = "geometry_unsupported"
    status_code: ClassVar[int] = 400
