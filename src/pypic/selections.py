"""Region selections: describe sub-regions and produce new FieldDatasets."""

from __future__ import annotations

__all__ = ["BoxSelection", "PlaneSelection"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypic.readers.base import FieldDataset


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
        axis_names = data.grid.geometry.axis_names[: len(data.grid.dimensions)]
        if self.normal not in axis_names:
            msg = f"Axis {self.normal!r} not found in dataset dimensions {axis_names!r}"
            raise ValueError(msg)

        if self.index is None:
            axis_idx = axis_names.index(self.normal)
            idx = data.grid.dimensions[axis_idx] // 2
        else:
            idx = self.index

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

        axis_names = data.grid.geometry.axis_names[: len(data.grid.dimensions)]
        for axis in self.ranges:
            if axis not in axis_names:
                msg = f"Axis {axis!r} not found in dataset dimensions {axis_names!r}"
                raise ValueError(msg)

        indexers = {
            axis: slice(start, stop) for axis, (start, stop) in self.ranges.items()
        }
        return data.isel(indexers)
