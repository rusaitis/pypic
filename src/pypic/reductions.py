"""Axis reductions: collapse a FieldDataset along one or more dimensions.

The `reduce` function collapses a [`FieldDataset`][pypic.dataset.FieldDataset]
along one axis (``axis="z"``) or several (``axis=("y", "z")``) using a
chosen reduction.  Supported reductions: ``"integrate"``, ``"sum"``,
``"mean"``, ``"median"``, ``"max"``, ``"min"``, ``"std"``, ``"var"``,
``"argmax"``, ``"argmin"``.  Column densities, line-of-sight integrals,
slab averages, projected-max diagnostics, and peak-position maps all
compose from this single primitive paired with an optional
[`BoxSelection`][pypic.selections.BoxSelection] or
[`SphereSelection`][pypic.selections.SphereSelection].

There is deliberately no ``SlabSelection``: selections describe regions,
not data, and a slab would fuse "pick a thick slice" with "reduce along
it".  Kept separate, ``BoxSelection`` stays reusable outside reductions
and a viewer request is just ``{selection, axis, reduction}``.

After an unweighted ``integrate`` the ``quantity_type`` and ``si_unit``
strings are preserved unchanged, and `FieldDataset.in_si` corrects the
value via ``length_ref ** length_axes`` from attrs.  The unit strings
themselves do not shift by the reduced length factors.
"""

from __future__ import annotations

__all__ = ["Reduction", "reduce"]

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from pypic.coordinates.geometry import GeometryType
from pypic.diagnostics import _VALID_NAN_POLICIES
from pypic.exceptions import GeometryUnsupportedError, UnknownFieldError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr

    from pypic.dataset import FieldDataset
    from pypic.diagnostics import NanPolicy
    from pypic.selections import BoxSelection, SphereSelection


type Reduction = Literal[
    "integrate",
    "sum",
    "mean",
    "median",
    "max",
    "min",
    "std",
    "var",
    "argmax",
    "argmin",
]

_VALID_REDUCTIONS: tuple[str, ...] = get_args(Reduction.__value__)

# Reductions accepting xarray's ``dim=[...]`` natively.
_MULTI_AXIS_DIM_REDUCERS: frozenset[str] = frozenset(
    {"sum", "mean", "median", "max", "min", "std", "var"}
)
# Reductions that return the *coordinate value* at the extremum (via
# xarray's ``idxmax`` / ``idxmin``).  Single-axis only — there is no
# meaningful coord-value to return over a multi-dim search.
_INDEX_REDUCERS: frozenset[str] = frozenset({"argmax", "argmin"})
# Reductions that accept a ``weight=`` kwarg.  ``sum`` is trivially
# achieved by pre-multiplying; ``max`` / ``min`` / ``median`` ignore
# weights semantically; ``std`` / ``var`` weighted versions are niche
# (revisit if requested); ``argmax`` / ``argmin`` return positions.
_WEIGHTABLE_REDUCERS: frozenset[str] = frozenset({"mean", "integrate"})


def reduce(
    data: FieldDataset,
    axis: str | tuple[str, ...],
    *,
    reduction: Reduction = "integrate",
    selection: BoxSelection | SphereSelection | None = None,
    fields: Iterable[str] | None = None,
    weight: str | None = None,
    nan_policy: NanPolicy = "omit",
) -> FieldDataset:
    r"""Reduce a FieldDataset along one or more axes.

    Apply *selection* (if given) to crop or NaN-mask, then collapse the
    *axis* dimension(s) using *reduction*.  Returns a FieldDataset with
    one fewer surviving axis per name passed — the natural input to
    ``plot_field_slice``, ``compute(...)``, ``in_si(...)``, and other
    dataset-consuming APIs.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (1-, 2-, or 3-D — must contain every name in *axis*).
    axis : str or tuple of str
        Dimension name(s) to reduce away (e.g. ``"x"``, ``("y", "z")``,
        ``"time"``).  Validated against the underlying xarray dataset's
        dims so non-grid dims like ``time`` (added by
        [`pypic.io.to_zarr_timeseries`][pypic.io.to_zarr_timeseries]) are accepted
        alongside the
        spatial ``grid.surviving_axis_names``.  Multi-axis reduction
        collapses several dimensions in one call — useful for going
        from 3D to 1D without chaining.  ``"argmax"`` and ``"argmin"``
        require a single axis.
    reduction : str
        How to collapse the axis.  Choices:

        * ``"integrate"`` — xarray trapezoidal-rule integration over the
          coordinate values (looped over axes for multi-axis input).
          Default — matches plasma-physics convention (column densities,
          integrated $\mathbf{J}\!\cdot\!\mathbf{E}$).
        * ``"sum"`` — unweighted Riemann sum (no ``× dx`` factor).
        * ``"mean"`` / ``"median"`` / ``"max"`` / ``"min"`` / ``"std"`` /
          ``"var"`` — direct xarray equivalents.
        * ``"argmax"`` / ``"argmin"`` — return the *coordinate value* of
          the extremum along *axis* (xarray's ``idxmax`` / ``idxmin``);
          single-axis only.  Result is a position along the reduced
          axis, so ``quantity_type`` is overridden to ``"length"``.
    selection : BoxSelection | SphereSelection | None
        Optional region selector applied before reduction.  ``None``
        reduces over the full domain.  Sugar for
        ``selection.apply(data)``-then-reduce; the two forms are
        observationally identical.
    fields : Iterable[str] | None
        Optional subset of field names (canonical or alias) to reduce.
        ``None`` reduces every data variable.  Unknown names raise
        `KeyError` with the full list rather than being skipped.
    weight : str | None
        Name of a field to weight the reduction by.  Only supported for
        ``reduction in {"mean", "integrate"}`` — other reductions raise
        ``ValueError`` (``sum`` is trivially achieved by pre-multiplying;
        ``max`` / ``min`` / ``median`` / ``std`` / ``var`` /
        ``argmax`` / ``argmin`` reject weights).  ``mean`` returns
        $\langle f \rangle_w = \sum f w / \sum w$ over the reduced axes;
        ``integrate`` returns the weighted line average
        $\int f w \, dx / \int w \, dx$ (yt's emission-weighted /
        density-weighted column average).  Resolves via
        ``data.resolve_key`` (aliases accepted); unknown names raise
        `KeyError`.  NaN handling under ``nan_policy="omit"``
        uses a joint mask: cells where the field *or* weight is NaN
        contribute zero to both the numerator and denominator and are
        skipped consistently.
    nan_policy : {"omit", "propagate", "raise"}
        ``"omit"`` (default) → ``skipna=True``: NaN cells are dropped
        from each reduction.  Pairs naturally with
        [`SphereSelection`][pypic.selections.SphereSelection], which NaN-masks
        outside-region cells.  ``"propagate"`` lets NaN poison the
        result.  ``"raise"`` errors when any input cell is NaN.
        Same vocabulary and semantics as
        [`pypic.diagnostics.l2_relative_error`][pypic.diagnostics.l2_relative_error] —
        see
        `conventions.md § Error Norms and Divergence` for the broader
        rationale.

    Returns
    -------
    FieldDataset
        Same metadata (normalization, species, frame, transforms) with
        the named axis (or axes) removed from the grid and dimensions.
        Each surviving DataArray gains an ``attrs["reduction"]`` dict
        recording ``{axis, op}`` — plus ``result_kind: "axis_position"``
        for ``argmax`` / ``argmin``, ``weight: <canonical name>`` when
        *weight* is set, and ``length_axes: <int>`` after unweighted
        ``integrate`` (the running count of length-dimension shifts
        across chained reductions).  The inner ``op`` key holds the
        reduction name (``"mean"``, ``"integrate"``, ...) to avoid
        shadowing the outer ``reduction`` key.

    Raises
    ------
    GeometryUnsupportedError
        Non-Cartesian geometry combined with a spatial-axis reduction.
        Spherical / cylindrical Jacobian-aware integration is not yet
        implemented.  Pure non-spatial reductions (e.g. along ``time``)
        bypass this check.
    ValueError
        Unknown *axis* name, invalid *reduction* or *nan_policy*,
        multi-axis input passed with ``reduction="argmax"`` /
        ``"argmin"``, or — under ``nan_policy="raise"`` — any NaN in a
        reduced field.
    KeyError
        Any name in *fields* fails to resolve via
        ``FieldDataset.resolve_key``.

    Notes
    -----
    **Unit shift after ``integrate``.** After unweighted
    ``reduction="integrate"`` the SI unit dimension shifts by one
    length factor along each reduced axis (e.g. number density
    m\ :sup:`-3` → column density m\ :sup:`-2`).  Per-field
    ``quantity_type`` and ``si_unit`` *strings* are preserved
    unchanged — but the numeric value returned by ``in_si()`` is
    correct: the ``length_axes`` provenance stamp here lets
    ``FieldDataset.in_si`` apply an extra ``length_ref **
    length_axes`` factor at the boundary.  Weighted ``integrate``
    does not stamp ``length_axes`` because the length factor
    cancels between numerator and denominator.  The displayed
    unit string and the openPMD 7-tuple do not yet shift with the
    reduction.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.coordinates import CARTESIAN
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.reductions import reduce
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
    ...     origin=(0.0, 0.0, 0.0), geometry=CARTESIAN)
    >>> ds = FieldDataset.from_arrays(
    ...     {"B_1": np.ones((4, 3, 2))}, grid, Normalization.identity())
    >>> column = reduce(ds, "z", reduction="integrate")
    >>> column.grid.dimensions
    (4, 3)
    >>> column.grid.surviving_axis_names
    ('x', 'y')
    >>> line = reduce(ds, ("y", "z"), reduction="mean")
    >>> line.grid.dimensions
    (4,)
    """
    axes: tuple[str, ...] = (axis,) if isinstance(axis, str) else tuple(axis)
    _validate_reduce(axes, reduction=reduction, weight=weight, nan_policy=nan_policy)
    if selection is not None:
        data = selection.apply(data)
    _validate_axes(data, axes, reduction)

    ds_to_reduce = data.xr if fields is None else data.xr[_resolve_fields(data, fields)]
    weight_canonical = None if weight is None else _resolve_weight(data, weight)
    weight_da = None if weight_canonical is None else data.xr[weight_canonical]
    if nan_policy == "raise":
        _reject_nan(ds_to_reduce, weight_da, weight_canonical)

    reduced = _collapse(
        ds_to_reduce,
        axes,
        reduction,
        weight_da=weight_da,
        nan_policy=nan_policy,
        source=data.xr,
    )
    _stamp_provenance(reduced, data.xr, axes, reduction, weight_canonical)
    # reductions sits above dataset in the layering, so it reuses the
    # dataset's own re-wrap rather than reconstructing grid and aliases.
    return data._wrap_sliced(reduced)  # noqa: SLF001


def _validate_reduce(
    axes: tuple[str, ...],
    *,
    reduction: str,
    weight: str | None,
    nan_policy: str,
) -> None:
    """Reject argument combinations `reduce` cannot honour, before any I/O."""
    if reduction not in _VALID_REDUCTIONS:
        msg = f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}"
        raise ValueError(msg)
    if nan_policy not in _VALID_NAN_POLICIES:
        msg = f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        raise ValueError(msg)
    if weight is not None and reduction not in _WEIGHTABLE_REDUCERS:
        msg = (
            f"weight= is only supported for reduction in "
            f"{sorted(_WEIGHTABLE_REDUCERS)!r}, got {reduction!r}"
        )
        raise ValueError(msg)
    if not axes:
        msg = "reduce(): axis must name at least one dimension"
        raise ValueError(msg)
    if len(set(axes)) != len(axes):
        msg = f"reduce(): duplicate axis names in {axes!r}"
        raise ValueError(msg)
    if reduction in _INDEX_REDUCERS and len(axes) > 1:
        msg = (
            f"reduction={reduction!r} requires a single axis "
            f"(got {axes!r}); idxmax/idxmin have no multi-axis form"
        )
        raise ValueError(msg)


def _validate_axes(data: FieldDataset, axes: tuple[str, ...], reduction: str) -> None:
    """Check *axes* against the (possibly selection-cropped) dataset."""
    # Accept any name present on the underlying xarray dataset, so
    # non-grid dims like ``time`` (added by ``to_zarr_timeseries``)
    # can be reduced too — not just the spatial ``surviving_axis_names``.
    xr_dims = tuple(str(d) for d in data.xr.dims)
    for ax in axes:
        if ax not in xr_dims:
            msg = f"Axis {ax!r} not found in dataset dimensions {xr_dims!r}"
            raise ValueError(msg)

    # The Cartesian gate only fires when at least one *spatial* axis is
    # being reduced — Jacobian-aware integration on non-Cartesian grids
    # is not implemented.  Pure non-spatial reductions (e.g.
    # ``reduce(ts, "time", "mean")``) are geometry-agnostic.
    reduces_spatial = any(ax in data.grid.surviving_axis_names for ax in axes)
    if reduces_spatial and data.grid.geometry.type is not GeometryType.CARTESIAN:
        msg = (
            f"reduce() supports Cartesian grids only for spatial-axis "
            f"reductions (got {data.grid.geometry.type.value}); "
            f"spherical/cylindrical Jacobian-aware integration is "
            f"not implemented"
        )
        raise GeometryUnsupportedError(msg)


def _resolve_fields(data: FieldDataset, fields: Iterable[str]) -> list[str]:
    """Canonical names for *fields*, raising once with every unknown name."""
    canonicals: list[str] = []
    unresolved: list[str] = []
    for name in fields:
        try:
            canonicals.append(data.resolve_key(name))
        except KeyError:
            unresolved.append(name)
    if unresolved:
        msg = f"reduce(): unknown fields {unresolved!r}"
        raise UnknownFieldError(msg)
    return canonicals


def _resolve_weight(data: FieldDataset, weight: str) -> str:
    try:
        return data.resolve_key(weight)
    except KeyError as exc:
        msg = f"reduce(): unknown weight field {weight!r}"
        raise UnknownFieldError(msg) from exc


def _reject_nan(
    ds: xr.Dataset, weight_da: xr.DataArray | None, weight_name: str | None
) -> None:
    """Raise on any NaN in the reduced fields or the weight (``nan_policy="raise"``)."""
    for name in [str(n) for n in ds.data_vars]:
        n_nan = int(np.isnan(ds[name].values).sum())
        if n_nan > 0:
            msg = f"reduce: input contains {n_nan} NaN cell(s) in {name!r}"
            raise ValueError(msg)
    if weight_da is not None:
        n_nan = int(np.isnan(weight_da.values).sum())
        if n_nan > 0:
            msg = f"reduce: weight field {weight_name!r} contains {n_nan} NaN cell(s)"
            raise ValueError(msg)


def _collapse(
    ds: xr.Dataset,
    axes: tuple[str, ...],
    reduction: str,
    *,
    weight_da: xr.DataArray | None,
    nan_policy: NanPolicy,
    source: xr.Dataset,
) -> xr.Dataset:
    """Apply *reduction* over *axes*: the per-operation dispatch of `reduce`."""
    skipna = nan_policy == "omit"
    reduced: xr.Dataset
    if reduction == "integrate":
        if weight_da is None:
            reduced = ds.integrate(coord=list(axes))
        else:
            reduced = _weighted_integrate(ds, weight_da, axes, nan_policy=nan_policy)
        # xarray's Dataset.integrate doesn't expose keep_attrs; restore
        # per-DataArray attrs from the source so quantity_type / si_unit
        # / latex survive for downstream in_si()/field_info() lookups.
        for name in [str(n) for n in reduced.data_vars]:
            reduced[name].attrs = dict(source[name].attrs)
        return reduced
    if reduction == "mean" and weight_da is not None:
        return _weighted_mean(ds, weight_da, axes, nan_policy=nan_policy)
    if reduction in _MULTI_AXIS_DIM_REDUCERS:
        reduced = getattr(ds, reduction)(dim=list(axes), skipna=skipna, keep_attrs=True)
        return reduced

    # argmax / argmin: single axis, enforced by _validate_reduce.
    method = ds.idxmax if reduction == "argmax" else ds.idxmin
    reduced = method(dim=axes[0], skipna=skipna, keep_attrs=True)
    # The result is a coordinate position along the reduced axis, not a
    # value of the original field, so the field-specific descriptors
    # would mislabel it downstream.
    for name in [str(n) for n in reduced.data_vars]:
        attrs = dict(reduced[name].attrs)
        attrs["quantity_type"] = "length"
        attrs["si_unit"] = "m"
        attrs["unit_dimension"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        attrs.pop("latex", None)
        attrs.pop("long_name", None)
        reduced[name].attrs = attrs
    return reduced


def _stamp_provenance(
    reduced: xr.Dataset,
    source: xr.Dataset,
    axes: tuple[str, ...],
    reduction: str,
    weight_canonical: str | None,
) -> None:
    """Record ``attrs["reduction"]`` on every reduced field.

    ``length_axes`` accumulates across chained reductions so ``in_si()``
    applies the right number of ``length_ref`` factors: unweighted
    ``integrate`` adds ``len(axes)``, weighted ``integrate`` cancels them
    between numerator and denominator, other reductions carry the count
    forward, and ``argmax``/``argmin`` return a length, which resets the
    unit dimension and makes the prior count moot.
    """
    base_attr: dict[str, str | int | tuple[str, ...]] = {
        "axis": axes[0] if len(axes) == 1 else axes,
        "op": reduction,
    }
    if reduction in _INDEX_REDUCERS:
        base_attr["result_kind"] = "axis_position"
    if weight_canonical is not None:
        base_attr["weight"] = weight_canonical

    for name in [str(n) for n in reduced.data_vars]:
        field_attr = dict(base_attr)
        if reduction not in _INDEX_REDUCERS:
            prior = source[name].attrs.get("reduction") or {}
            prior_length_axes = int(prior.get("length_axes", 0))
            added = (
                len(axes)
                if reduction == "integrate" and weight_canonical is None
                else 0
            )
            total = prior_length_axes + added
            if total:
                field_attr["length_axes"] = total
        reduced[name].attrs["reduction"] = field_attr


def _weighted_mean(
    ds: xr.Dataset,
    weight_da: xr.DataArray,
    axes: tuple[str, ...],
    *,
    nan_policy: NanPolicy,
) -> xr.Dataset:
    r"""Weighted mean :math:`\langle f \rangle_w = \sum f w / \sum w`.

    Joint NaN-masking under ``nan_policy="omit"``: cells where field or
    weight is NaN contribute zero to both numerator and denominator,
    skipping them consistently.  Under ``"propagate"``, NaN flows
    through naturally.  Under ``"raise"``, NaN was pre-checked.
    """

    def _one(field_da: xr.DataArray) -> xr.DataArray:
        fw, w = _joint_mask(field_da, weight_da, nan_policy=nan_policy)
        fw_sum = fw.sum(dim=list(axes), skipna=False, keep_attrs=True)
        w_sum = w.sum(dim=list(axes), skipna=False)
        result: xr.DataArray = fw_sum / w_sum
        result.attrs = dict(field_da.attrs)
        return result

    return ds.map(_one, keep_attrs=True)


def _weighted_integrate(
    ds: xr.Dataset,
    weight_da: xr.DataArray,
    axes: tuple[str, ...],
    *,
    nan_policy: NanPolicy,
) -> xr.Dataset:
    r"""Weighted line average :math:`\int f w \, dx / \int w \, dx`.

    yt's emission-weighted / density-weighted column average.  For
    multi-axis input the numerator and denominator are integrated over
    every named axis in turn (the integration is associative and
    commutative for trapezoidal-rule integration of a smooth product).
    Joint NaN-masking under ``nan_policy="omit"`` matches
    `_weighted_mean`.
    """
    coords = list(axes)

    def _one(field_da: xr.DataArray) -> xr.DataArray:
        fw, w = _joint_mask(field_da, weight_da, nan_policy=nan_policy)
        return fw.integrate(coord=coords) / w.integrate(coord=coords)

    return ds.map(_one, keep_attrs=True)


def _joint_mask(
    field_da: xr.DataArray,
    weight_da: xr.DataArray,
    *,
    nan_policy: NanPolicy,
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Return ``(field * weight, weight)`` with joint NaN masking.

    Under ``nan_policy="omit"``, cells where either the field or the
    weight is NaN are replaced with zero in *both* the numerator
    ``field * weight`` and the denominator ``weight``, so they
    contribute nothing to either integral and the resulting weighted
    average is taken over the non-NaN region.  Under ``"propagate"``
    (and ``"raise"``, where NaN was already screened out upstream) no
    masking is applied; NaN flows through naturally.
    """
    if nan_policy == "omit":
        nan_mask = field_da.isnull() | weight_da.isnull()
        fw = (field_da * weight_da).where(~nan_mask, 0.0)
        w = weight_da.where(~nan_mask, 0.0)
    else:
        fw = field_da * weight_da
        w = weight_da
    return fw, w
