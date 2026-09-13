"""Typed exception hierarchy for pypic.

Library-side raise sites use these instead of bare ``KeyError`` /
``NotImplementedError`` so callers — the Starlette server in
particular — can dispatch on type rather than sniff exception
messages.  Subclasses inherit from the appropriate standard base
(``KeyError`` / ``NotImplementedError``) in addition to
`PypicError`, so existing ``except KeyError`` and
``except NotImplementedError`` callers keep working unchanged.

Wire kinds and HTTP status codes are *not* here: they are server
concerns, and
[`pypic.server.exceptions.error_routing`][pypic.server.exceptions.error_routing]
maps each type to its pair at the boundary that cares.

The server-only
[`pypic.server.exceptions.ValidationFailedError`][pypic.server.exceptions.ValidationFailedError]
wraps `pydantic.ValidationError` from request-frame parsing
and ``simulation.toml`` validation; it lives next to the server
because the wire format is its only consumer.
"""

from __future__ import annotations

__all__ = [
    "GeometryUnsupportedError",
    "PypicError",
    "UndeclaredNormalizationError",
    "UnknownFieldError",
    "UnknownSimulationError",
    "UnknownStepError",
    "UnsupportedGridError",
]


class PypicError(Exception):
    """Base for every typed pypic exception.

    Catching this catches every error pypic raises deliberately.
    """

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
    still work, while letting the server route on type rather than
    inspect message strings.
    """


class UnknownFieldError(PypicError, KeyError):
    """A requested field name does not resolve in the dataset.

    Raised by `FieldDataset.resolve_key` and by
    `Simulation.read` when ``strict_fields=True`` (the default)
    and one of the requested names matched no loaded field.
    """


class UnknownStepError(PypicError, KeyError):
    """A requested timestep is not available for the simulation."""


class UndeclaredNormalizationError(PypicError, ValueError):
    """SI conversion was asked for, but no unit system was ever declared.

    A reader that finds no ``simulation.toml`` cannot invent the one
    absolute anchor SI conversion needs — a PIC deck fixes only
    dimensionless ratios, so the reference density is a modelling
    choice, not data in the file.  Rather than return code units
    labelled tesla, `Normalization.si_factor` raises this for any
    dimensional quantity.  Dimensionless quantities (``beta``,
    ``M_A``, ``agyrotropy``) are exempt: they are correct under any
    anchor.

    Subclass of `ValueError`, matching the unknown-quantity raise
    from the same function.
    """


class GeometryUnsupportedError(PypicError, NotImplementedError):
    """An operation is not implemented for the dataset's coordinate geometry.

    Examples: `regrid` on spherical geometry, derivative-based
    derived quantities on non-Cartesian grids, spatial-axis reductions
    on non-Cartesian grids.
    """


class UnsupportedGridError(GeometryUnsupportedError):
    """The grid's *representation* is beyond what pypic's containers carry.

    Distinct from the parent, which is about the coordinate system: this
    one fires on a Cartesian grid whose cells are not uniformly spaced,
    where naming the geometry would be actively misleading.  Raised by
    `readers.config.load_config` on a ``[grid.stretched]`` deck, because
    `GridInfo` holds one scalar spacing per axis.

    A `NotImplementedError`, which is the distinction that matters at
    the reader boundary: the document is *valid* under the cross-tool
    schema and pypic simply cannot build a container from it.  A
    `ValueError` there would tell the user their deck is wrong when it
    is not.  Subclassing `GeometryUnsupportedError` keeps the server
    routing (``geometry_unsupported`` / 400) and every existing
    ``except GeometryUnsupportedError`` caller working unchanged.
    """
