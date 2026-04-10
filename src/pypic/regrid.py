"""Uniform-to-uniform grid interpolation.

Provides :func:`regrid` for interpolating a :class:`FieldDataset` from one
uniform Cartesian grid onto another, :func:`common_grid` for computing the
intersection grid at the finer resolution, and :func:`align_grids` as a
convenience that regrids two datasets onto their common grid.

Cartesian grids only — spherical and cylindrical geometries raise
:class:`NotImplementedError`, matching the convention in
:mod:`pypic.coordinates.operators`.

Examples
--------
>>> import numpy as np
>>> from pypic.dataset import FieldDataset
>>> from pypic.grid import GridInfo
>>> from pypic.units import Normalization
>>> grid_a = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
>>> grid_b = GridInfo(dimensions=(8,), spacing=(0.5,), origin=(0.0,))
>>> ds = FieldDataset.from_arrays(
...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid_a, Normalization.identity(),
... )
>>> result = regrid(ds, grid_b)
>>> result.grid.dimensions
(8,)
>>> result["B1"].shape
(8,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_never

import numpy as np

from pypic.coordinates.geometry import GeometryType
from pypic.grid import GridInfo

if TYPE_CHECKING:
    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray

__all__ = ["align_grids", "common_grid", "regrid"]


def _require_cartesian_grid(grid: GridInfo, label: str) -> None:
    """Raise if *grid* uses a non-Cartesian geometry."""
    match grid.geometry.type:
        case GeometryType.CARTESIAN:
            return
        case GeometryType.SPHERICAL | GeometryType.CYLINDRICAL:
            msg = (
                f"Regridding not implemented for "
                f"{grid.geometry.type.value} geometry ({label})"
            )
            raise NotImplementedError(msg)
        case _ as unreachable:
            assert_never(unreachable)


def _grid_extent(grid: GridInfo) -> tuple[float, ...]:
    """Upper-right corner of the domain: ``origin + dimensions * spacing``."""
    return tuple(
        grid.origin[i] + grid.dimensions[i] * grid.spacing[i]
        for i in range(len(grid.dimensions))
    )


def common_grid(a: GridInfo, b: GridInfo) -> GridInfo:
    r"""Compute the intersection grid at the finer resolution.

    The result covers the spatial overlap of *a* and *b* with per-axis
    spacing equal to ``min(a.spacing[i], b.spacing[i])``.

    Parameters
    ----------
    a, b : GridInfo
        Source grids.  Must be Cartesian with the same dimensionality.

    Returns
    -------
    GridInfo
        Intersection grid.

    Raises
    ------
    NotImplementedError
        If either grid is non-Cartesian.
    ValueError
        If dimensionalities differ or domains do not overlap.

    Examples
    --------
    >>> g1 = GridInfo(dimensions=(10,), spacing=(1.0,), origin=(0.0,))
    >>> g2 = GridInfo(dimensions=(10,), spacing=(0.5,), origin=(5.0,))
    >>> cg = common_grid(g1, g2)
    >>> cg.origin
    (5.0,)
    >>> cg.spacing
    (0.5,)
    """
    _require_cartesian_grid(a, "grid a")
    _require_cartesian_grid(b, "grid b")

    ndim_a = len(a.dimensions)
    ndim_b = len(b.dimensions)
    if ndim_a != ndim_b:
        msg = f"Grid dimensionality mismatch: {ndim_a}D vs {ndim_b}D"
        raise ValueError(msg)

    extent_a = _grid_extent(a)
    extent_b = _grid_extent(b)

    new_origin: list[float] = []
    new_spacing: list[float] = []
    new_dims: list[int] = []

    for i in range(ndim_a):
        lo = max(a.origin[i], b.origin[i])
        hi = min(extent_a[i], extent_b[i])
        if lo >= hi:
            msg = f"Grids do not overlap along axis {i}"
            raise ValueError(msg)

        dx = min(a.spacing[i], b.spacing[i])
        n = max(1, int((hi - lo) / dx))
        new_origin.append(lo)
        new_spacing.append(dx)
        new_dims.append(n)

    return GridInfo(
        dimensions=tuple(new_dims),
        spacing=tuple(new_spacing),
        origin=tuple(new_origin),
        geometry=a.geometry,
    )


def regrid(
    source: FieldDataset,
    target_grid: GridInfo,
    *,
    method: str = "linear",
    **kwargs: Any,  # noqa: ANN401 — scipy passthrough
) -> FieldDataset:
    r"""Interpolate all fields from *source* onto *target_grid*.

    Each field array is interpolated independently using
    :class:`~scipy.interpolate.RegularGridInterpolator`.  Points in
    *target_grid* that fall outside the source domain are filled with NaN
    (override via ``fill_value`` kwarg).

    Parameters
    ----------
    source : FieldDataset
        Dataset on the original grid.
    target_grid : GridInfo
        Target grid specification.
    method : str
        Interpolation method forwarded to ``RegularGridInterpolator``
        (e.g. ``"linear"``, ``"nearest"``, ``"cubic"``).
    **kwargs
        Extra keyword arguments forwarded to ``RegularGridInterpolator``
        (e.g. ``fill_value=0.0``).

    Returns
    -------
    FieldDataset
        New dataset on *target_grid* with all fields interpolated and
        all metadata (normalization, species, physics, frame, transforms)
        preserved from *source*.

    Raises
    ------
    NotImplementedError
        If either grid is non-Cartesian.
    ValueError
        If source and target dimensionalities differ.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> coarse = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
    >>> fine = GridInfo(dimensions=(8,), spacing=(0.5,), origin=(0.0,))
    >>> ds = FieldDataset.from_arrays(
    ...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     coarse, Normalization.identity(),
    ... )
    >>> result = regrid(ds, fine)
    >>> result.grid.spacing
    (0.5,)
    """
    _require_cartesian_grid(source.grid, "source")
    _require_cartesian_grid(target_grid, "target")

    src_ndim = len(source.grid.dimensions)
    tgt_ndim = len(target_grid.dimensions)
    if src_ndim != tgt_ndim:
        msg = f"Cannot regrid {src_ndim}D source onto {tgt_ndim}D target grid"
        raise ValueError(msg)

    # No-op shortcut when grids are identical.
    if source.grid == target_grid:
        return source

    from scipy.interpolate import RegularGridInterpolator

    src_coords = source.grid.coordinate_arrays()
    tgt_coords = target_grid.coordinate_arrays()
    tgt_mesh = np.meshgrid(*tgt_coords, indexing="ij")

    interp_kwargs: dict[str, Any] = {
        "method": method,
        "bounds_error": False,
        "fill_value": np.nan,
    }
    interp_kwargs.update(kwargs)

    new_fields: dict[str, FloatArray] = {}
    for name in source.field_names():
        interp = RegularGridInterpolator(
            src_coords,
            source[name],
            **interp_kwargs,
        )
        new_fields[name] = interp(tuple(tgt_mesh))

    from pypic.dataset import FieldDataset as _FieldDataset

    return _FieldDataset.from_arrays(
        new_fields,
        target_grid,
        source.normalization,
        species=list(source.species),
        physics=source.physics,
        metadata=dict(source.metadata),
        frame=source.frame,
        transforms=dict(source.transforms),
    )


def align_grids(
    a: FieldDataset,
    b: FieldDataset,
    *,
    method: str = "linear",
    **kwargs: Any,  # noqa: ANN401 — scipy passthrough
) -> tuple[FieldDataset, FieldDataset]:
    r"""Regrid both datasets onto their common intersection grid.

    Computes the intersection domain at the finer per-axis resolution
    via :func:`common_grid`, then regrids each dataset onto it via
    :func:`regrid`.

    Parameters
    ----------
    a, b : FieldDataset
        Input datasets on (possibly different) uniform Cartesian grids.
    method : str
        Interpolation method (default ``"linear"``).
    **kwargs
        Extra arguments forwarded to :func:`regrid`.

    Returns
    -------
    tuple[FieldDataset, FieldDataset]
        ``(a_regridded, b_regridded)`` on the common grid.

    Raises
    ------
    NotImplementedError
        If either grid is non-Cartesian.
    ValueError
        If dimensionalities differ or domains do not overlap.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> g1 = GridInfo(dimensions=(10,), spacing=(1.0,), origin=(0.0,))
    >>> g2 = GridInfo(dimensions=(20,), spacing=(0.5,), origin=(0.0,))
    >>> ds1 = FieldDataset.from_arrays(
    ...     {"B1": np.ones(10)}, g1, Normalization.identity(),
    ... )
    >>> ds2 = FieldDataset.from_arrays(
    ...     {"B1": np.ones(20)}, g2, Normalization.identity(),
    ... )
    >>> a_new, b_new = align_grids(ds1, ds2)
    >>> a_new.grid.spacing == b_new.grid.spacing
    True
    """
    target = common_grid(a.grid, b.grid)
    return (
        regrid(a, target, method=method, **kwargs),
        regrid(b, target, method=method, **kwargs),
    )
