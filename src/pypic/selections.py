"""Region selections: describe sub-regions and produce new FieldDatasets."""

from __future__ import annotations

__all__ = ["BoxSelection", "PlaneSelection", "SphereSelection"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.dataset import FieldDataset


@dataclass(frozen=True, slots=True)
class PlaneSelection:
    r"""Select a 2D plane from a 3D dataset by slicing along one axis.

    Parameters
    ----------
    normal : str
        Axis name perpendicular to the plane (e.g. ``"z"``, ``"θ"``).
    index : int | None
        Integer index along the normal axis. ``None`` selects the
        midplane (``dimensions[axis] // 2``). For even dimension
        counts, this selects the lower midpoint.

    Examples
    --------
    >>> from pypic.selections import PlaneSelection
    >>> import copy
    >>> plane = PlaneSelection(normal="z", index=3)
    >>> plane.normal
    'z'
    >>> copy.replace(plane, index=0).index
    0
    """

    normal: str
    index: int | None = None

    def apply(self, data: FieldDataset) -> FieldDataset:
        """Slice the dataset along the normal axis.

        Parameters
        ----------
        data : FieldDataset
            Input dataset (must contain the ``normal`` dimension).

        Returns
        -------
        FieldDataset
            Reduced dataset with the normal dimension removed.

        Raises
        ------
        ValueError
            If ``normal`` is not a dimension in *data*.
        """
        axis_names = data.grid.surviving_axis_names
        if self.normal not in axis_names:
            msg = f"Axis {self.normal!r} not found in dataset dimensions {axis_names!r}"
            raise ValueError(msg)

        axis_idx = axis_names.index(self.normal)
        idx = data.grid.dimensions[axis_idx] // 2 if self.index is None else self.index

        dim_size = data.grid.dimensions[axis_idx]
        if idx < 0 or idx >= dim_size:
            msg = (
                f"Index {idx} out of bounds for axis {self.normal!r} "
                f"with dimension {dim_size}"
            )
            raise ValueError(msg)

        return data.isel({self.normal: idx})


@dataclass(frozen=True, slots=True)
class BoxSelection:
    r"""Select a rectangular sub-region via integer index ranges.

    Parameters
    ----------
    ranges : dict[str, tuple[int, int]]
        Axis name → ``(start, stop)`` half-open index range. Axes not
        listed are kept in full. An empty dict is a no-op.

    Examples
    --------
    >>> from pypic.selections import BoxSelection
    >>> import copy
    >>> box = BoxSelection(ranges={"x": (1, 5), "y": (0, 3)})
    >>> box.ranges["x"]
    (1, 5)
    >>> copy.replace(box, ranges={}).ranges
    {}
    """

    ranges: dict[str, tuple[int, int]]

    def apply(self, data: FieldDataset) -> FieldDataset:
        """Slice the dataset to the specified sub-region.

        Parameters
        ----------
        data : FieldDataset
            Input dataset.

        Returns
        -------
        FieldDataset
            Sub-region with updated grid metadata.

        Raises
        ------
        ValueError
            If any axis name in ``ranges`` is not a dimension in *data*.
        """
        if not self.ranges:
            return data

        axis_names = data.grid.surviving_axis_names
        for axis in self.ranges:
            if axis not in axis_names:
                msg = f"Axis {axis!r} not found in dataset dimensions {axis_names!r}"
                raise ValueError(msg)

        indexers = {
            axis: slice(start, stop) for axis, (start, stop) in self.ranges.items()
        }
        return data.isel(indexers)


@dataclass(frozen=True, slots=True)
class SphereSelection:
    r"""Mask fields inside or outside a sphere with NaN.

    Unlike `PlaneSelection` and `BoxSelection`, this
    preserves the grid shape — masked points become ``NaN`` rather than
    being removed.

    Parameters
    ----------
    center : tuple[float, ...]
        Center point in coordinate units. Length must match the dataset
        dimensionality (3 for 3D, 2 for a previously sliced 2D dataset).
    radius : float
        Radius in the same coordinate units as *center*.
    keep : str
        Which region to keep: ``"inside"`` (default) keeps points within
        the sphere and NaN-fills outside; ``"outside"`` keeps points
        beyond the sphere and NaN-fills inside.

    Examples
    --------
    >>> from pypic.selections import SphereSelection
    >>> import copy
    >>> s = SphereSelection(center=(0.0, 0.0, 0.0), radius=3.375, keep="outside")
    >>> s.radius
    3.375
    >>> copy.replace(s, keep="inside").keep
    'inside'
    """

    center: tuple[float, ...]
    radius: float
    keep: str = "inside"

    def apply(self, data: FieldDataset) -> FieldDataset:
        """Apply the spherical mask, returning a new FieldDataset.

        Parameters
        ----------
        data : FieldDataset
            Input dataset.

        Returns
        -------
        FieldDataset
            Same grid shape, with masked points set to ``NaN``.

        Raises
        ------
        ValueError
            If *center* length does not match the dataset dimensionality,
            *radius* is not positive, or *keep* is invalid.
        """
        coords = data.grid.coordinate_arrays()
        ndim = len(coords)

        if len(self.center) != ndim:
            msg = (
                f"center has {len(self.center)} components but dataset "
                f"has {ndim} dimensions"
            )
            raise ValueError(msg)
        if self.radius <= 0:
            msg = f"radius must be positive, got {self.radius}"
            raise ValueError(msg)
        if self.keep not in ("inside", "outside"):
            msg = f"keep must be 'inside' or 'outside', got {self.keep!r}"
            raise ValueError(msg)

        # Squared distance via broadcasting (avoids sqrt)
        r_sq: np.ndarray = np.zeros(data.grid.dimensions)
        for i in range(ndim):
            shape = [1] * ndim
            shape[i] = -1
            r_sq = r_sq + (coords[i].reshape(shape) - self.center[i]) ** 2

        radius_sq = self.radius**2
        mask = r_sq <= radius_sq if self.keep == "inside" else r_sq > radius_sq

        return data.where(mask)
