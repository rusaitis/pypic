"""GridInfo: the structured-grid metadata every FieldDataset carries."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pypic.coordinates.geometry import (
    CARTESIAN,
    CYLINDRICAL,  # noqa: F401 — used in doctests
    SPHERICAL,  # noqa: F401 — used in doctests
    CoordinateGeometry,
)

if TYPE_CHECKING:
    from xarray import Dataset

    from pypic.types import FloatArray


@dataclass(frozen=True, slots=True)
class GridInfo:
    r"""Structured grid metadata for 1D/2D/3D simulation domains.

    Parameters
    ----------
    dimensions : tuple[int, ...]
        Number of cells along each axis.
    spacing : tuple[float, ...]
        Cell size along each axis in code units.
    origin : tuple[float, ...]
        Lower-left corner coordinate of the domain.
    geometry : CoordinateGeometry
        Coordinate system (Cartesian, spherical, cylindrical).
    dt : float | None
        Timestep size in code units, if known.
    boundary : tuple[str, ...] | None
        Boundary condition per axis (e.g. ``("periodic", "open", "periodic")``).

    Examples
    --------
    >>> grid = GridInfo(
    ...     dimensions=(4,), spacing=(0.5,), origin=(0.0,),
    ...     geometry=CARTESIAN,
    ... )
    >>> grid.coordinate_arrays()[0]
    array([0.25, 0.75, 1.25, 1.75])
    """

    dimensions: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...] = ()
    geometry: CoordinateGeometry = CARTESIAN
    dt: float | None = None
    boundary: tuple[str, ...] | None = None
    surviving_axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        ndim = len(self.dimensions)
        if not self.origin:
            object.__setattr__(self, "origin", (0.0,) * ndim)
        if len(self.spacing) != ndim or len(self.origin) != ndim:
            msg = (
                f"Length mismatch: dimensions({ndim}), "
                f"spacing({len(self.spacing)}), origin({len(self.origin)})"
            )
            raise ValueError(msg)
        for i, d in enumerate(self.dimensions):
            if d <= 0:
                raise ValueError(f"dimensions[{i}] must be > 0, got {d}")
        for i, s in enumerate(self.spacing):
            if s <= 0:
                raise ValueError(f"spacing[{i}] must be > 0, got {s}")
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.boundary is not None and len(self.boundary) != ndim:
            msg = (
                f"boundary length ({len(self.boundary)}) "
                f"must match dimensions length ({ndim})"
            )
            raise ValueError(msg)
        if self.surviving_axes is not None and len(self.surviving_axes) != ndim:
            msg = (
                f"surviving_axes length ({len(self.surviving_axes)}) "
                f"must match dimensions length ({ndim})"
            )
            raise ValueError(msg)

    @property
    def surviving_axis_names(self) -> tuple[str, ...]:
        """Axis names for the current dimensions.

        After slicing, returns only the names of axes that survived
        (e.g. ``("x", "z")`` after removing the y-axis). When no
        slicing has occurred, returns the first *ndim* names from
        the geometry.

        Examples
        --------
        >>> grid = GridInfo(
        ...     dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> grid.surviving_axis_names
        ('x', 'y', 'z')
        >>> import copy
        >>> sliced = copy.replace(
        ...     grid, dimensions=(4, 2), spacing=(1.0, 1.0),
        ...     origin=(0.0, 0.0), surviving_axes=(0, 2),
        ... )
        >>> sliced.surviving_axis_names
        ('x', 'z')
        """
        if self.surviving_axes is not None:
            return tuple(self.geometry.axis_names[i] for i in self.surviving_axes)
        return self.geometry.axis_names[: len(self.dimensions)]

    def coordinate_arrays(self) -> tuple[FloatArray, ...]:
        r"""Cell-centered coordinate arrays for each axis.

        Returns
        -------
        tuple[FloatArray, ...]
            One 1-D array per axis: ``origin[i] + (arange(n) + 0.5) * dx[i]``.

        Examples
        --------
        >>> grid = GridInfo(
        ...     dimensions=(3, 2), spacing=(1.0, 2.0), origin=(0.0, 0.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> x, y = grid.coordinate_arrays()
        >>> x
        array([0.5, 1.5, 2.5])
        >>> y
        array([1., 3.])
        """
        return tuple(
            self.origin[i] + (np.arange(self.dimensions[i]) + 0.5) * self.spacing[i]
            for i in range(len(self.dimensions))
        )


def _build_grid_from_dataset(old_grid: GridInfo, new_ds: Dataset) -> GridInfo:
    """Derive a reduced GridInfo from a sliced xr.Dataset."""
    current_names = old_grid.surviving_axis_names
    surviving: list[tuple[int, str]] = []  # (local_index, name)

    for local_idx, name in enumerate(current_names):
        if name in new_ds.dims:
            surviving.append((local_idx, name))

    # Map local indices back to original 3D geometry axis indices
    if old_grid.surviving_axes is not None:
        new_surviving = tuple(
            old_grid.surviving_axes[local_idx] for local_idx, _ in surviving
        )
    else:
        new_surviving = tuple(local_idx for local_idx, _ in surviving)

    new_boundary = None
    if old_grid.boundary is not None:
        new_boundary = tuple(old_grid.boundary[local_idx] for local_idx, _ in surviving)

    return copy.replace(
        old_grid,
        dimensions=tuple(int(new_ds.sizes[name]) for _, name in surviving),
        spacing=tuple(old_grid.spacing[local_idx] for local_idx, _ in surviving),
        origin=tuple(
            # invert cell-center formula: coord[0] = origin + 0.5*spacing
            float(new_ds.coords[name].values[0]) - 0.5 * old_grid.spacing[local_idx]
            for local_idx, name in surviving
        ),
        boundary=new_boundary,
        surviving_axes=new_surviving,
    )
