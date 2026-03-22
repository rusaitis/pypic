"""FieldLine: frozen container for a traced magnetic (or other vector) field line."""

from __future__ import annotations

__all__ = ["FieldLine"]

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypic.types import FloatArray, Vector3
    from pypic.units import Normalization

_VALID_DIRECTIONS = frozenset({"both", "forward", "backward"})


@dataclass(frozen=True, slots=True)
class FieldLine:
    r"""Ordered sequence of 3D points tracing a vector field at a fixed time.

    A field line is parameterized by arc length, not time. It represents
    the spatial structure of a vector field (typically $\mathbf{B}$) at a
    single instant.

    Parameters
    ----------
    points : FloatArray
        Ordered positions along the line, shape ``(N, 3)`` with ``N >= 2``.
    field_name : str
        Name of the traced vector field (e.g. ``"B"``).
    seed_point : Vector3
        Starting point for the trace.
    normalization : Normalization
        Unit conversion for this data.
    time : float | None
        Simulation time at which the field was sampled.
    step : int | None
        Timestep index.
    direction : str
        Trace direction: ``"both"``, ``"forward"``, or ``"backward"``.
    scalars : dict[str, FloatArray]
        Named scalar quantities sampled along the line, each shape ``(N,)``.
    metadata : dict[str, Any]
        Arbitrary metadata (reader info, integration parameters, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.units import Normalization
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> fl = FieldLine(
    ...     points=pts, field_name="B", seed_point=(0.0, 0.0, 0.0),
    ...     normalization=Normalization.identity(),
    ... )
    >>> fl.n_points
    3
    >>> fl.start_point
    (0.0, 0.0, 0.0)
    >>> fl.end_point
    (2.0, 0.0, 0.0)
    >>> len(fl)
    3
    """

    points: FloatArray
    field_name: str
    seed_point: Vector3
    normalization: Normalization
    time: float | None = None
    step: int | None = None
    direction: str = "both"
    scalars: dict[str, FloatArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            msg = f"points must have shape (N, 3), got {self.points.shape}"
            raise ValueError(msg)
        if self.points.shape[0] < 2:
            msg = f"points must have at least 2 rows, got {self.points.shape[0]}"
            raise ValueError(msg)
        if self.direction not in _VALID_DIRECTIONS:
            msg = (
                f"direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )
            raise ValueError(msg)
        n = self.points.shape[0]
        for name, arr in self.scalars.items():
            if arr.shape != (n,):
                msg = (
                    f"scalar {name!r} has shape {arr.shape}, "
                    f"expected ({n},)"
                )
                raise ValueError(msg)
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_points(self) -> int:
        """Number of points along the field line."""
        return int(self.points.shape[0])

    @property
    def start_point(self) -> Vector3:
        """First point on the line as a 3-tuple."""
        p = self.points[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def end_point(self) -> Vector3:
        """Last point on the line as a 3-tuple."""
        p = self.points[-1]
        return (float(p[0]), float(p[1]), float(p[2]))

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __repr__(self) -> str:
        scalar_names = sorted(self.scalars) if self.scalars else []
        parts = [
            f"FieldLine({self.field_name!r}, n={self.n_points}",
            f"direction={self.direction!r}",
        ]
        if self.time is not None:
            parts.append(f"t={self.time}")
        if scalar_names:
            parts.append(f"scalars={scalar_names}")
        return ", ".join(parts) + ")"

    def with_scalars(self, **new_scalars: FloatArray) -> FieldLine:
        """Return a new FieldLine with additional or replaced scalars.

        Parameters
        ----------
        **new_scalars : FloatArray
            Scalar arrays to merge, each shape ``(N,)``.

        Returns
        -------
        FieldLine
        """
        merged = dict(self.scalars)
        merged.update(new_scalars)
        return copy.replace(self, scalars=merged)
