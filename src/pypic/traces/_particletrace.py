"""ParticleTrace: frozen container for a particle trajectory (worldline)."""

from __future__ import annotations

__all__ = ["ParticleTrace"]

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pypic.types import FloatArray, Vector3
    from pypic.units import Normalization, SpeciesInfo


@dataclass(frozen=True, slots=True)
class ParticleTrace:
    r"""Ordered sequence of spacetime points tracing a particle's worldline.

    Unlike `FieldLine`, a particle trace is parameterized by time (monotonic)
    and carries velocity at each point. Represents a physical particle's
    trajectory through the simulation domain.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)`` with ``N >= 2``.
    time : FloatArray
        Time at each point, shape ``(N,)``, monotonically increasing.
    velocity : FloatArray
        Velocity at each point, shape ``(N, 3)``.
    species_name : str
        Species name (e.g. ``"electrons"``).
    normalization : Normalization
        Unit conversion for this data.
    species : SpeciesInfo | None
        Species charge/mass info for derived calculations.
    particle_id : int | None
        Tracking ID, if available.
    scalars : dict[str, FloatArray]
        Named scalar quantities along the trajectory, each shape ``(N,)``.
    metadata : dict[str, Any]
        Arbitrary metadata.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.units import Normalization
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> t = np.array([0.0, 0.5, 1.0])
    >>> vel = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> tr = ParticleTrace(
    ...     points=pts, time=t, velocity=vel, species_name="electrons",
    ...     normalization=Normalization.identity(),
    ... )
    >>> tr.n_points
    3
    >>> tr.duration
    1.0
    >>> len(tr)
    3
    """

    points: FloatArray
    time: FloatArray
    velocity: FloatArray
    species_name: str
    normalization: Normalization
    species: SpeciesInfo | None = None
    particle_id: int | None = None
    scalars: dict[str, FloatArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            msg = f"points must have shape (N, 3), got {self.points.shape}"
            raise ValueError(msg)
        n = self.points.shape[0]
        if n < 2:
            msg = f"points must have at least 2 rows, got {n}"
            raise ValueError(msg)
        if self.time.shape != (n,):
            msg = f"time must have shape ({n},), got {self.time.shape}"
            raise ValueError(msg)
        if self.velocity.ndim != 2 or self.velocity.shape != (n, 3):
            msg = f"velocity must have shape ({n}, 3), got {self.velocity.shape}"
            raise ValueError(msg)
        if not np.all(np.diff(self.time) > 0):
            msg = "time must be strictly monotonically increasing"
            raise ValueError(msg)
        for name, arr in self.scalars.items():
            if arr.shape != (n,):
                msg = f"scalar {name!r} has shape {arr.shape}, expected ({n},)"
                raise ValueError(msg)
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_points(self) -> int:
        """Number of points along the trajectory."""
        return int(self.points.shape[0])

    @property
    def start_point(self) -> Vector3:
        """First position as a 3-tuple."""
        p = self.points[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def end_point(self) -> Vector3:
        """Last position as a 3-tuple."""
        p = self.points[-1]
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def start_time(self) -> float:
        """Time at first point."""
        return float(self.time[0])

    @property
    def end_time(self) -> float:
        """Time at last point."""
        return float(self.time[-1])

    @property
    def duration(self) -> float:
        """Total elapsed time."""
        return float(self.time[-1] - self.time[0])

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __repr__(self) -> str:
        scalar_names = sorted(self.scalars) if self.scalars else []
        parts = [
            f"ParticleTrace({self.species_name!r}, n={self.n_points}",
            f"t=[{self.start_time:.4g}, {self.end_time:.4g}]",
        ]
        if self.particle_id is not None:
            parts.append(f"id={self.particle_id}")
        if scalar_names:
            parts.append(f"scalars={scalar_names}")
        return ", ".join(parts) + ")"

    def with_scalars(self, **new_scalars: FloatArray) -> ParticleTrace:
        """Return a new ParticleTrace with additional or replaced scalars.

        Parameters
        ----------
        **new_scalars : FloatArray
            Scalar arrays to merge, each shape ``(N,)``.

        Returns
        -------
        ParticleTrace
        """
        n = self.n_points
        for name, arr in new_scalars.items():
            if arr.shape != (n,):
                msg = f"Scalar {name!r} has shape {arr.shape}, expected ({n},)"
                raise ValueError(msg)
        merged = dict(self.scalars)
        merged.update(new_scalars)
        return copy.replace(self, scalars=merged)
