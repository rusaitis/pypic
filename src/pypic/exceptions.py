"""Typed exception hierarchy for pypic.

Library-side raise sites use these instead of bare ``KeyError`` /
``NotImplementedError`` so callers — the Starlette server in
particular — can dispatch on type rather than sniff exception
messages.  Subclasses inherit from the appropriate standard base
(``KeyError`` / ``NotImplementedError``) in addition to
:class:`PypicError`, so existing ``except KeyError`` and
``except NotImplementedError`` callers keep working unchanged.

The ``kind`` and ``status_code`` classvars are server-routing
metadata: the Starlette/FastAPI layer reads them to build
:class:`pypic.server.protocol.ErrorFrame` (WebSocket) and
:class:`fastapi.HTTPException` (HTTP) responses without a
dispatch table.  They are inert for non-server callers — the
classvar values impose no behavior on the library itself.

The server-only :class:`pypic.server.exceptions.ValidationFailedError`
wraps :class:`pydantic.ValidationError` from request-frame parsing
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

    Subclasses set ``kind`` (matching :class:`ErrorFrame.kind` literals
    on the WebSocket wire format) and ``status_code`` (the HTTP status
    a server boundary should emit).  Defaults route to the catch-all
    ``"internal"`` / ``500`` so a raw :class:`PypicError` raised by
    accident is still routable.
    """

    kind: ClassVar[str] = "internal"
    status_code: ClassVar[int] = 500


class UnknownSimulationError(PypicError, KeyError):
    """No simulation with the requested name exists under the registry root.

    Subclass of :class:`KeyError` so callers that catch the broader type
    still work, while letting the server route to the ``unknown_sim``
    error kind without inspecting message strings.
    """

    kind: ClassVar[str] = "unknown_sim"
    status_code: ClassVar[int] = 404


class UnknownFieldError(PypicError, KeyError):
    """A requested field name does not resolve in the dataset.

    Raised by :meth:`FieldDataset.resolve_key` and by
    :meth:`Simulation.read` when ``strict_fields=True`` (the default)
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

    Examples: :func:`regrid` on spherical geometry, derivative-based
    derived quantities on non-Cartesian grids, spatial-axis reductions
    on non-Cartesian grids.
    """

    kind: ClassVar[str] = "geometry_unsupported"
    status_code: ClassVar[int] = 400
