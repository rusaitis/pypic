"""Axis reductions: collapse a FieldDataset along one dimension.

The :func:`project` function reduces a :class:`~pypic.dataset.FieldDataset`
along a single axis using a chosen reduction (``"integrate"``, ``"sum"``,
``"mean"``, ``"max"``, ``"min"``, ``"std"``).  Column densities, line-of-
sight integrals, slab averages, and projected-max diagnostics all
compose from this single primitive paired with an optional
:class:`~pypic.selections.BoxSelection` or
:class:`~pypic.selections.SphereSelection`.

Why no ``SlabSelection``? Selections describe regions, not data
(CLAUDE.md §architecture).  A ``Slab`` would fuse "pick a thick slice"
(region) with "reduce along the thick axis" (data), which forces every
caller to think about both at once.  Splitting them keeps the existing
``BoxSelection`` reusable for non-projection workflows and gives
webpic-style viewers a clean wire format:
``{selection, axis, reduction}`` → one server-side ``project()`` call.
"""

from __future__ import annotations

__all__ = ["project"]

from typing import TYPE_CHECKING, Literal

import numpy as np

from pypic.coordinates.geometry import GeometryType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.dataset import FieldDataset
    from pypic.diagnostics import NanPolicy
    from pypic.selections import BoxSelection, SphereSelection


type Reduction = Literal["integrate", "sum", "mean", "max", "min", "std"]

_VALID_REDUCTIONS: tuple[str, ...] = (
    "integrate",
    "sum",
    "mean",
    "max",
    "min",
    "std",
)
_VALID_NAN_POLICIES: tuple[str, ...] = ("omit", "propagate", "raise")


def project(
    data: FieldDataset,
    axis: str,
    *,
    reduction: Reduction = "integrate",
    selection: BoxSelection | SphereSelection | None = None,
    fields: Iterable[str] | None = None,
    nan_policy: NanPolicy = "omit",
) -> FieldDataset:
    r"""Reduce a FieldDataset along one axis.

    Apply *selection* (if given) to crop or NaN-mask, then collapse the
    *axis* dimension using *reduction*.  Returns a FieldDataset with one
    fewer surviving axis — the natural input to ``plot_field_slice``,
    ``compute(...)``, ``in_si(...)``, and other dataset-consuming APIs.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (1-, 2-, or 3-D — must contain *axis*).
    axis : str
        Surviving-axis name to reduce away (e.g. ``"x"``, ``"z"``).
        Validated against ``data.grid.surviving_axis_names``.
    reduction : {"integrate", "sum", "mean", "max", "min", "std"}
        How to collapse the axis.  ``"integrate"`` uses xarray's
        trapezoidal-rule integration over the coordinate values — the
        physically correct line-of-sight / column integral for both
        uniform and (future) non-uniform 1-D coords.  ``"sum"`` is the
        unweighted Riemann sum (no ``× dx`` factor).  The remaining
        reductions are direct xarray equivalents.  Default
        ``"integrate"`` matches plasma-physics convention (column
        densities, integrated $\mathbf{J}\!\cdot\!\mathbf{E}$).
    selection : BoxSelection | SphereSelection | None
        Optional region selector applied before reduction.  ``None``
        projects the full domain.  Sugar for
        ``selection.apply(data)``-then-project; the two forms are
        observationally identical.
    fields : Iterable[str] | None
        Optional subset of field names (canonical or alias) to project.
        ``None`` projects every data variable.  Unknown names raise
        :class:`KeyError` with the full list (CLAUDE.md §architecture).
    nan_policy : {"omit", "propagate", "raise"}
        ``"omit"`` (default) → ``skipna=True``: NaN cells are dropped
        from each reduction.  Pairs naturally with
        :class:`~pypic.selections.SphereSelection`, which NaN-masks
        outside-region cells.  ``"propagate"`` lets NaN poison the
        result.  ``"raise"`` errors when any input cell is NaN.

    Returns
    -------
    FieldDataset
        Same metadata (normalization, species, frame, transforms) with
        ``axis`` removed from the grid and dimensions; each surviving
        DataArray gains an ``attrs["projection"]`` dict recording axis,
        reduction, and (for integrate/sum) the original axis length.

    Raises
    ------
    NotImplementedError
        Non-Cartesian geometry.  Spherical / cylindrical Jacobian-aware
        integration is deferred to TASKS Step 19b/20b.
    ValueError
        Unknown *axis*, invalid *reduction* or *nan_policy*, or — under
        ``nan_policy="raise"`` — any NaN in a projected field.
    KeyError
        Any name in *fields* fails to resolve via
        ``FieldDataset.resolve_key``.

    Notes
    -----
    **Unit caveat (MVP).** After ``reduction="integrate"`` the SI unit
    dimension shifts by one length factor along *axis* (e.g. number
    density m\ :sup:`-3` → column density m\ :sup:`-2`).  Per-field
    ``quantity_type`` and ``si_unit`` attrs are **preserved unchanged**
    by this MVP; ``in_si()`` therefore returns the value the *original*
    SI factor would yield — off by one length-unit.  Users needing
    correct SI units multiply by ``data.normalization.length_si``.
    Unit-aware projection is tracked alongside Step 20b
    (volume-weighted norms).

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.coordinates import CARTESIAN
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.reductions import project
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
    ...     origin=(0.0, 0.0, 0.0), geometry=CARTESIAN)
    >>> ds = FieldDataset.from_arrays(
    ...     {"B_1": np.ones((4, 3, 2))}, grid, Normalization.identity())
    >>> column = project(ds, "z", reduction="integrate")
    >>> column.grid.dimensions
    (4, 3)
    >>> column.grid.surviving_axis_names
    ('x', 'y')
    """
    if reduction not in _VALID_REDUCTIONS:
        msg = f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}"
        raise ValueError(msg)
    if nan_policy not in _VALID_NAN_POLICIES:
        msg = f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        raise ValueError(msg)

    if selection is not None:
        data = selection.apply(data)

    if data.grid.geometry.type is not GeometryType.CARTESIAN:
        msg = (
            f"project() supports Cartesian grids only "
            f"(got {data.grid.geometry.type.value}); "
            f"spherical/cylindrical Jacobian-aware integration is "
            f"deferred to TASKS Step 19b/20b"
        )
        raise NotImplementedError(msg)

    axis_names = data.grid.surviving_axis_names
    if axis not in axis_names:
        msg = f"Axis {axis!r} not found in dataset dimensions {axis_names!r}"
        raise ValueError(msg)
    local_idx = axis_names.index(axis)

    if fields is None:
        ds_to_reduce = data.xr
    else:
        canonicals: list[str] = []
        unresolved: list[str] = []
        for name in fields:
            try:
                canonicals.append(data.resolve_key(name))
            except KeyError:
                unresolved.append(name)
        if unresolved:
            msg = f"project(): unknown fields {unresolved!r}"
            raise KeyError(msg)
        ds_to_reduce = data.xr[canonicals]

    if nan_policy == "raise":
        for name in [str(n) for n in ds_to_reduce.data_vars]:
            arr = ds_to_reduce[name].values
            n_nan = int(np.isnan(arr).sum())
            if n_nan > 0:
                msg = f"project: input contains {n_nan} NaN cell(s) in {name!r}"
                raise ValueError(msg)

    skipna = nan_policy == "omit"

    if reduction == "integrate":
        reduced = ds_to_reduce.integrate(coord=axis)
        # xarray's Dataset.integrate doesn't expose keep_attrs; restore
        # per-DataArray attrs from the source so quantity_type / si_unit
        # / latex survive for downstream in_si()/field_info() lookups.
        for name in [str(n) for n in reduced.data_vars]:
            reduced[name].attrs = dict(data.xr[name].attrs)
    elif reduction == "sum":
        reduced = ds_to_reduce.sum(dim=axis, skipna=skipna, keep_attrs=True)
    elif reduction == "mean":
        reduced = ds_to_reduce.mean(dim=axis, skipna=skipna, keep_attrs=True)
    elif reduction == "max":
        reduced = ds_to_reduce.max(dim=axis, skipna=skipna, keep_attrs=True)
    elif reduction == "min":
        reduced = ds_to_reduce.min(dim=axis, skipna=skipna, keep_attrs=True)
    else:  # std
        reduced = ds_to_reduce.std(dim=axis, skipna=skipna, keep_attrs=True)

    projection_attr: dict[str, str | float] = {
        "axis": axis,
        "reduction": reduction,
    }
    if reduction in ("integrate", "sum"):
        projection_attr["length"] = float(
            data.grid.spacing[local_idx] * data.grid.dimensions[local_idx]
        )
    for name in [str(n) for n in reduced.data_vars]:
        reduced[name].attrs["projection"] = dict(projection_attr)

    return data._wrap_sliced(reduced)
