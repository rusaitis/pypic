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
...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid_a, Normalization.identity(),
... )
>>> result = regrid(ds, grid_b)
>>> result.grid.dimensions
(8,)
>>> result["B_1"].shape
(8,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pypic.coordinates.geometry import GeometryType
from pypic.grid import GridInfo

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray

__all__ = ["align_grids", "common_grid", "regrid"]

# Mirror of scipy's ``RegularGridInterpolator`` method vocabulary so the
# no-op shortcut below rejects the same typos a live interpolation
# would. Kept as a hardcoded set rather than reaching into scipy
# internals; the list has been stable since scipy 1.10.
_VALID_INTERPOLATION_METHODS: frozenset[str] = frozenset(
    {"linear", "nearest", "slinear", "cubic", "quintic", "pchip"}
)


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


def _sample_bounds(
    grid: GridInfo,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-axis ``(sample_lo, sample_hi)`` for a cell-centered grid.

    `GridInfo.coordinate_arrays` is cell-centered, so the first valid
    sample sits at ``origin + 0.5*dx`` and the last at
    ``origin + (N - 0.5)*dx``. These are the inclusive bounds of the
    range over which `RegularGridInterpolator` accepts queries — anything
    outside becomes NaN. Use these (not the cell-volume edge
    ``origin + N*dx``) when computing the overlap of two grids.
    """
    los = tuple(
        grid.origin[i] + 0.5 * grid.spacing[i] for i in range(len(grid.dimensions))
    )
    his = tuple(
        grid.origin[i] + (grid.dimensions[i] - 0.5) * grid.spacing[i]
        for i in range(len(grid.dimensions))
    )
    return los, his


def common_grid(a: GridInfo, b: GridInfo) -> GridInfo:
    r"""Compute the intersection grid at the finer resolution.

    Returns a uniform Cartesian grid covering the **inclusive sample-range
    intersection** of *a* and *b* with per-axis spacing
    ``min(a.spacing[i], b.spacing[i])``. Because the bounds are computed
    from cell-centered sample positions (``origin + 0.5*dx`` to
    ``origin + (N - 0.5)*dx``) rather than cell-volume edges, every
    sample on the returned grid lies strictly inside both source sample
    ranges. This guarantees that interpolating either source onto the
    common grid never produces a synthetic boundary NaN.

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
        If dimensionalities differ or sample ranges do not overlap.

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

    los_a, his_a = _sample_bounds(a)
    los_b, his_b = _sample_bounds(b)

    new_origin: list[float] = []
    new_spacing: list[float] = []
    new_dims: list[int] = []

    for i in range(ndim_a):
        sample_lo = max(los_a[i], los_b[i])
        sample_hi = min(his_a[i], his_b[i])
        if sample_lo > sample_hi:
            msg = (
                f"Grids do not overlap along axis {i}: "
                f"a samples [{los_a[i]:g}, {his_a[i]:g}], "
                f"b samples [{los_b[i]:g}, {his_b[i]:g}]"
            )
            raise ValueError(msg)

        dx = min(a.spacing[i], b.spacing[i])
        # Number of samples on a closed interval [sample_lo, sample_hi]
        # at uniform spacing dx. The 1e-9 tolerance absorbs FP rounding
        # when ``(sample_hi - sample_lo) / dx`` is an exact integer;
        # safe for all spacings PIC/MHD readers emit (typically 1e-6
        # upward in code units). For pathologically small spacings
        # (< ~1e-9) the slack would shadow a legitimate sub-step, but
        # that regime does not arise in fluid/kinetic output.
        n = max(1, int((sample_hi - sample_lo) / dx + 1e-9) + 1)
        new_origin.append(sample_lo - 0.5 * dx)
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
    fields: Iterable[str] | None = None,
    method: str = "linear",
    **kwargs: Any,  # noqa: ANN401 — scipy passthrough
) -> FieldDataset:
    r"""Interpolate fields from *source* onto *target_grid*.

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
    fields : Iterable[str] | None
        Canonical field names (or aliases) to regrid. ``None`` (default)
        regrids every field in *source*. Passing a subset avoids wasted
        interpolation when a caller only needs a handful of fields from
        a large dataset — the stacked-field interpolator is still built
        once, but only over the requested subset. Raises ``KeyError`` on
        unknown names (including close-match suggestions from
        :meth:`FieldDataset.resolve_key`).
    method : str
        Interpolation method forwarded to ``RegularGridInterpolator``
        (e.g. ``"linear"``, ``"nearest"``, ``"cubic"``).
    **kwargs
        Extra keyword arguments forwarded to ``RegularGridInterpolator``
        (e.g. ``fill_value=0.0``).

    Returns
    -------
    FieldDataset
        New dataset on *target_grid* with the selected fields
        interpolated and all metadata (normalization, species, physics,
        frame, transforms) preserved from *source*. Metadata survives
        the regrid unchanged — see :func:`field_difference_dataset` for
        the comparison helper that deliberately replaces it.

    Raises
    ------
    NotImplementedError
        If either grid is non-Cartesian.
    ValueError
        If source and target dimensionalities differ.
    KeyError
        If *fields* names a field that does not exist in *source*.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> coarse = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
    >>> fine = GridInfo(dimensions=(8,), spacing=(0.5,), origin=(0.0,))
    >>> ds = FieldDataset.from_arrays(
    ...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
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

    # Validate method *before* the no-op shortcut so that typos raise
    # regardless of whether interpolation actually runs — otherwise
    # same-grid callers (including the classic ``compare_fields(ds, ds,
    # ...)`` smoke test) silently accept unknown methods.
    if method not in _VALID_INTERPOLATION_METHODS:
        msg = (
            f"Unknown interpolation method {method!r}. Must be one of "
            f"{sorted(_VALID_INTERPOLATION_METHODS)}."
        )
        raise ValueError(msg)

    # Resolve the field selection *before* the no-op shortcut so that
    # bad names always raise, even when no interpolation runs.
    if fields is None:
        names = list(source.field_names())
    else:
        seen: dict[str, None] = {}
        for raw in fields:
            seen[source.resolve_key(raw)] = None
        names = list(seen)

    # No-op shortcut: identical grids and a selection that covers every
    # field (or None). A strict subset still needs to build a narrower
    # dataset, so it falls through to the construction path below.
    if source.grid == target_grid and (
        fields is None or set(names) == set(source.field_names())
    ):
        return source

    src_coords = source.grid.coordinate_arrays()
    tgt_coords = target_grid.coordinate_arrays()
    tgt_mesh = np.meshgrid(*tgt_coords, indexing="ij")

    interp_kwargs: dict[str, Any] = {
        "method": method,
        "bounds_error": False,
        "fill_value": np.nan,
    }
    interp_kwargs.update(kwargs)

    # Stack the requested fields into one trailing value-dimension so
    # the RegularGridInterpolator is built and evaluated once. Every
    # field shares the source grid, target mesh, and interpolation
    # options, so per-field reconstruction is pure Python overhead (and
    # avoids per-field precomputation for ``method="cubic"``/``"quintic"``).
    new_fields: dict[str, FloatArray] = {}
    if names:
        stacked = np.stack([source[n] for n in names], axis=-1)
        interp = RegularGridInterpolator(src_coords, stacked, **interp_kwargs)
        sampled = interp(tuple(tgt_mesh))
        for i, name in enumerate(names):
            new_fields[name] = sampled[..., i]

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
        strict_fields=False,
    )


def align_grids(
    a: FieldDataset,
    b: FieldDataset,
    *,
    fields: Iterable[str] | None = None,
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
    fields : Iterable[str] | None
        Canonical field names (or aliases) to keep on the output. When
        *None* (default) every field in each dataset is regridded, which
        matches the historical behavior. Pass a subset to skip wasted
        interpolation — each name is resolved through **both** source
        alias tables independently, so ``"Bx"`` works even when one side
        only exposes the canonical ``"B_1"``.
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
    KeyError
        If *fields* names a field missing in either dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> g1 = GridInfo(dimensions=(10,), spacing=(1.0,), origin=(0.0,))
    >>> g2 = GridInfo(dimensions=(20,), spacing=(0.5,), origin=(0.0,))
    >>> ds1 = FieldDataset.from_arrays(
    ...     {"B_1": np.ones(10)}, g1, Normalization.identity(),
    ... )
    >>> ds2 = FieldDataset.from_arrays(
    ...     {"B_1": np.ones(20)}, g2, Normalization.identity(),
    ... )
    >>> a_new, b_new = align_grids(ds1, ds2)
    >>> a_new.grid.spacing == b_new.grid.spacing
    True
    """
    target = common_grid(a.grid, b.grid)
    # Materialize the selection now so both regrid calls see the same
    # field list — resolution happens against each dataset's own alias
    # table inside regrid(), which matters when one side uses canonical
    # names and the other exposes an extra alias.
    field_list = list(fields) if fields is not None else None
    return (
        regrid(a, target, fields=field_list, method=method, **kwargs),
        regrid(b, target, fields=field_list, method=method, **kwargs),
    )
