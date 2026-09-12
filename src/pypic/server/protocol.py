"""Wire-format models for the pypic server.

Pydantic v2 models that describe every WebSocket request/response
shape — and only those.  HTTP responses are plain JSON dumps of
existing typed values (``SimulationConfig``, schema bodies), so they
do not need wrappers here.

The selection / reduction wire shapes are tagged-union dicts that
map onto the established ``BoxSelection`` / ``PlaneSelection`` /
``SphereSelection`` dataclasses + [`pypic.reduce`][pypic.reduce] kwargs.  Keeping
the JSON ↔ object conversion in this module preserves the architecture
rule that selections are pure region descriptions — they get
constructed from validated specs *outside* the dataclass definitions.

Recording selection provenance in a stored Zarr
(``attrs.selections``) will reuse `SelectionSpec` directly; designing
the wire shape here lets that storage side land later as a pure write
addition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

# Pydantic resolves the ``Reduction`` annotation at model build time
# via ``get_type_hints``, so it must be importable at runtime — not in
# a TYPE_CHECKING block.
from pypic.reductions import Reduction  # noqa: TC001
from pypic.selections import BoxSelection, PlaneSelection, SphereSelection

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "Ack",
    "BoxSpec",
    "ErrorFrame",
    "ErrorKind",
    "PlaneSpec",
    "ReductionSpec",
    "SelectionSpec",
    "SphereSpec",
    "SubscribeRequest",
    "to_reduction_kwargs",
    "to_selection",
]


# Closed vocabulary of error kinds on the wire. A client dispatches on
# these, so adding one is a wire-format change;
# `pypic.server.exceptions.error_routing` is where exception types map
# onto them.
type ErrorKind = Literal[
    "validation",
    "unknown_field",
    "unknown_sim",
    "unknown_step",
    "geometry_unsupported",
    "internal",
]


class _StrictModel(BaseModel):
    """Base for all wire models — reject unknown keys at the boundary."""

    model_config = ConfigDict(extra="forbid")


# -- Selection tagged union --------------------------------------------------


class BoxSpec(_StrictModel):
    """Wire form of [`BoxSelection`][pypic.selections.BoxSelection]."""

    kind: Literal["box"] = "box"
    ranges: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Axis name → (start, stop) integer index range. "
        "Empty dict is a no-op.",
    )


class PlaneSpec(_StrictModel):
    """Wire form of [`PlaneSelection`][pypic.selections.PlaneSelection]."""

    kind: Literal["plane"] = "plane"
    normal: str
    index: int | None = Field(
        default=None,
        description="None → midplane.",
    )


class SphereSpec(_StrictModel):
    """Wire form of [`SphereSelection`][pypic.selections.SphereSelection]."""

    kind: Literal["sphere"] = "sphere"
    center: tuple[float, float, float]
    radius: float = Field(gt=0.0)
    keep: Literal["inside", "outside"] = "inside"


SelectionSpec = Annotated[
    BoxSpec | PlaneSpec | SphereSpec,
    Field(discriminator="kind"),
]


def to_selection(
    spec: SelectionSpec,
) -> BoxSelection | PlaneSelection | SphereSelection:
    """Construct the corresponding selection dataclass from a validated spec."""
    match spec:
        case BoxSpec(ranges=r):
            return BoxSelection(ranges=dict(r))
        case PlaneSpec(normal=n, index=i):
            return PlaneSelection(normal=n, index=i)
        case SphereSpec(center=c, radius=r, keep=k):
            return SphereSelection(center=c, radius=r, keep=k)
        case _ as unreachable:
            assert_never(unreachable)


# -- Reduction spec ----------------------------------------------------------


class ReductionSpec(_StrictModel):
    """Wire form of a [`pypic.reduce`][pypic.reduce] call."""

    axis: str | list[str] = Field(
        description="Single axis name or list of axes (e.g. ['y', 'z']).",
    )
    op: Reduction = Field(
        default="integrate",
        description="Reduction operation. See pypic.Reduction.",
    )
    weight: str | None = Field(
        default=None,
        description="Optional weight field (only mean / integrate accept).",
    )
    nan_policy: Literal["omit", "propagate", "raise"] = "omit"


def to_reduction_kwargs(spec: ReductionSpec) -> dict[str, Any]:
    """Translate a ``ReductionSpec`` to kwargs for [`pypic.reduce`][pypic.reduce].

    The ``axis`` field accepts either a single string or a list; the
    list form gets passed through as a tuple (which pypic.reduce
    expects for multi-axis input).
    """
    axis: str | tuple[str, ...] = (
        tuple(spec.axis) if isinstance(spec.axis, list) else spec.axis
    )
    return {
        "axis": axis,
        "reduction": spec.op,
        "weight": spec.weight,
        "nan_policy": spec.nan_policy,
    }


# -- WebSocket frames --------------------------------------------------------


class SubscribeRequest(_StrictModel):
    """Inbound WebSocket frame requesting one FieldDataset slice."""

    type: Literal["subscribe"] = "subscribe"
    request_id: str = Field(
        description="Caller-supplied UUID echoed in every server response."
    )
    sim: str | None = Field(
        default=None,
        description="Simulation name. Optional when the WS path already "
        "carries /sims/{sim}/stream; required for multiplexed connections.",
    )
    step: int
    fields: list[str] = Field(
        default_factory=list,
        description="Canonical or alias field names. Empty → server "
        "returns every field available at the step.",
    )
    selection: SelectionSpec | None = None
    reduction: ReductionSpec | None = None
    units: Literal["code", "si"] = "code"


class Ack(_StrictModel):
    """Outbound JSON text frame announcing the binary payload to follow."""

    type: Literal["ack"] = "ack"
    request_id: str
    shape: list[int]
    dims: list[str]
    fields: list[str]
    units: Literal["code", "si"]


class ErrorFrame(_StrictModel):
    """Outbound JSON text frame on request failure.

    ``kind`` is a coarse category for client-side dispatch; ``message``
    is the human-readable detail straight from the exception.
    """

    type: Literal["error"] = "error"
    request_id: str
    kind: ErrorKind
    message: str
