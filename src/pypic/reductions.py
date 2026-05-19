"""Axis reductions: collapse a FieldDataset along one or more dimensions.

The :func:`reduce` function collapses a :class:`~pypic.dataset.FieldDataset`
along one axis (``axis="z"``) or several (``axis=("y", "z")``) using a
chosen reduction.  Supported reductions: ``"integrate"``, ``"sum"``,
``"mean"``, ``"median"``, ``"max"``, ``"min"``, ``"std"``, ``"var"``,
``"argmax"``, ``"argmin"``.  Column densities, line-of-sight integrals,
slab averages, projected-max diagnostics, and peak-position maps all
compose from this single primitive paired with an optional
:class:`~pypic.selections.BoxSelection` or
:class:`~pypic.selections.SphereSelection`.

Why no ``SlabSelection``? Selections describe regions, not data
(CLAUDE.md §architecture).  A ``Slab`` would fuse "pick a thick slice"
(region) with "reduce along the thick axis" (data), which forces every
caller to think about both at once.  Splitting them keeps the existing
``BoxSelection`` reusable for non-reduction workflows and gives
webpic-style viewers a clean wire format:
``{selection, axis, reduction}`` → one server-side ``reduce()`` call.

The verb is ``reduce`` (not ``project``) deliberately: ``project()`` is
already taken by Three.js (``Vector3.project(camera)``) for screen-space
camera projection, and webpic is a Three.js viewer.  Server-side
``reduce`` keeps the cross-stack vocabulary clean.

The ``length_axes`` attrs stamp is an interim mechanism shipped ahead of
TASKS Step 43c's openPMD ``unit_dimension`` 7-tuple generalization.
Today, after an unweighted ``integrate``, the displayed ``quantity_type``
and ``si_unit`` strings are preserved but the numeric value through
:meth:`FieldDataset.in_si` is corrected via ``length_ref ** length_axes``.
Step 43c will subsume this with proper post-reduction tuple arithmetic.
"""

from __future__ import annotations

__all__ = ["Reduction", "reduce"]

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from pypic.coordinates.geometry import GeometryType

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
_VALID_NAN_POLICIES: tuple[str, ...] = ("omit", "propagate", "raise")

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
        :func:`pypic.io.to_zarr_timeseries`) are accepted alongside the
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
        :class:`KeyError` with the full list (CLAUDE.md §architecture).
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
        :class:`KeyError`.  NaN handling under ``nan_policy="omit"``
        uses a joint mask: cells where the field *or* weight is NaN
        contribute zero to both the numerator and denominator and are
        skipped consistently.
    nan_policy : {"omit", "propagate", "raise"}
        ``"omit"`` (default) → ``skipna=True``: NaN cells are dropped
        from each reduction.  Pairs naturally with
        :class:`~pypic.selections.SphereSelection`, which NaN-masks
        outside-region cells.  ``"propagate"`` lets NaN poison the
        result.  ``"raise"`` errors when any input cell is NaN.
        Same vocabulary and semantics as
        :func:`pypic.diagnostics.l2_relative_error` — see
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
    NotImplementedError
        Non-Cartesian geometry combined with a spatial-axis reduction.
        Spherical / cylindrical Jacobian-aware integration is deferred
        to TASKS Step 43b.  Pure non-spatial reductions (e.g. along
        ``time``) bypass this check.
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
    unit string and the openPMD 7-tuple are still authoritatively
    fixed by TASKS Step 43c.

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

    if selection is not None:
        data = selection.apply(data)

    axes: tuple[str, ...] = (axis,) if isinstance(axis, str) else tuple(axis)
    if not axes:
        msg = "reduce(): axis must name at least one dimension"
        raise ValueError(msg)
    if len(set(axes)) != len(axes):
        msg = f"reduce(): duplicate axis names in {axes!r}"
        raise ValueError(msg)

    spatial_axes = data.grid.surviving_axis_names
    xr_dims = tuple(str(d) for d in data.xr.dims)
    for ax in axes:
        # Accept any name present on the underlying xarray dataset, so
        # non-grid dims like ``time`` (added by ``to_zarr_timeseries``)
        # can be reduced too — not just the spatial ``surviving_axis_names``.
        if ax not in xr_dims:
            msg = f"Axis {ax!r} not found in dataset dimensions {xr_dims!r}"
            raise ValueError(msg)

    # The Cartesian gate only fires when at least one *spatial* axis is
    # being reduced — Jacobian-aware integration on non-Cartesian grids
    # is deferred to Step 43b.  Pure non-spatial reductions (e.g.
    # ``reduce(ts, "time", "mean")``) are geometry-agnostic.
    reduces_spatial = any(ax in spatial_axes for ax in axes)
    if reduces_spatial and data.grid.geometry.type is not GeometryType.CARTESIAN:
        msg = (
            f"reduce() supports Cartesian grids only for spatial-axis "
            f"reductions (got {data.grid.geometry.type.value}); "
            f"spherical/cylindrical Jacobian-aware integration is "
            f"deferred to TASKS Step 43b"
        )
        raise NotImplementedError(msg)

    if reduction in _INDEX_REDUCERS and len(axes) > 1:
        msg = (
            f"reduction={reduction!r} requires a single axis "
            f"(got {axes!r}); idxmax/idxmin have no multi-axis form"
        )
        raise ValueError(msg)

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
            msg = f"reduce(): unknown fields {unresolved!r}"
            raise KeyError(msg)
        ds_to_reduce = data.xr[canonicals]

    weight_canonical: str | None = None
    weight_da = None
    if weight is not None:
        try:
            weight_canonical = data.resolve_key(weight)
        except KeyError as exc:
            msg = f"reduce(): unknown weight field {weight!r}"
            raise KeyError(msg) from exc
        weight_da = data.xr[weight_canonical]

    if nan_policy == "raise":
        for name in [str(n) for n in ds_to_reduce.data_vars]:
            arr = ds_to_reduce[name].values
            n_nan = int(np.isnan(arr).sum())
            if n_nan > 0:
                msg = f"reduce: input contains {n_nan} NaN cell(s) in {name!r}"
                raise ValueError(msg)
        if weight_da is not None:
            n_nan = int(np.isnan(weight_da.values).sum())
            if n_nan > 0:
                msg = (
                    f"reduce: weight field {weight_canonical!r} contains "
                    f"{n_nan} NaN cell(s)"
                )
                raise ValueError(msg)

    skipna = nan_policy == "omit"

    if reduction == "integrate":
        if weight_da is None:
            reduced = ds_to_reduce.integrate(coord=list(axes))
        else:
            reduced = _weighted_integrate(
                ds_to_reduce, weight_da, axes, nan_policy=nan_policy
            )
        # xarray's Dataset.integrate doesn't expose keep_attrs; restore
        # per-DataArray attrs from the source so quantity_type / si_unit
        # / latex survive for downstream in_si()/field_info() lookups.
        for name in [str(n) for n in reduced.data_vars]:
            reduced[name].attrs = dict(data.xr[name].attrs)
    elif reduction == "mean" and weight_da is not None:
        reduced = _weighted_mean(ds_to_reduce, weight_da, axes, nan_policy=nan_policy)
    elif reduction in _MULTI_AXIS_DIM_REDUCERS:
        method = getattr(ds_to_reduce, reduction)
        reduced = method(dim=list(axes), skipna=skipna, keep_attrs=True)
    else:  # argmax / argmin (single-axis enforced above)
        method_name = "idxmax" if reduction == "argmax" else "idxmin"
        method = getattr(ds_to_reduce, method_name)
        match axes:
            case (single_axis,):
                reduced = method(dim=single_axis, skipna=skipna, keep_attrs=True)
            case _:  # unreachable — guarded above
                raise AssertionError(f"argmax/argmin single-axis invariant: {axes!r}")
        # Override metadata: the result is a coordinate position along
        # the reduced axis, not a value of the original field.
        for name in [str(n) for n in reduced.data_vars]:
            attrs = dict(reduced[name].attrs)
            attrs["quantity_type"] = "length"
            attrs["si_unit"] = "m"
            attrs["unit_dimension"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            # The array now holds coordinate positions, not field values —
            # clear field-specific descriptors so downstream tooling doesn't
            # mislabel the result as the original quantity.
            attrs.pop("latex", None)
            attrs.pop("long_name", None)
            reduced[name].attrs = attrs

    reduction_axis: str | tuple[str, ...]
    match axes:
        case (single,):
            reduction_axis = single
        case _:
            reduction_axis = axes
    base_attr: dict[str, str | int | tuple[str, ...]] = {
        "axis": reduction_axis,
        "op": reduction,
    }
    if reduction in _INDEX_REDUCERS:
        base_attr["result_kind"] = "axis_position"
    if weight_canonical is not None:
        base_attr["weight"] = weight_canonical

    # Accumulate ``length_axes`` across chained reductions so ``in_si()``
    # can apply the right number of ``length_ref`` factors.  Unweighted
    # ``integrate`` adds ``len(axes)``; weighted ``integrate`` cancels
    # the length factors between numerator and denominator (units of
    # ``∫ f w dx / ∫ w dx`` equal units of ``f``); every other reduction
    # is unit-preserving so it carries the prior count forward
    # unchanged.  ``argmax``/``argmin`` overwrite ``quantity_type`` to
    # ``"length"`` and reset the unit dimension entirely — the prior
    # ``length_axes`` is moot and dropped.
    for name in [str(n) for n in reduced.data_vars]:
        field_attr = dict(base_attr)
        if reduction not in _INDEX_REDUCERS:
            prior = data.xr[name].attrs.get("reduction") or {}
            prior_length_axes = int(prior.get("length_axes", 0))
            added = len(axes) if (reduction == "integrate" and weight is None) else 0
            total = prior_length_axes + added
            if total:
                field_attr["length_axes"] = total
        reduced[name].attrs["reduction"] = field_attr

    return data._wrap_sliced(reduced)


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
    :func:`_weighted_mean`.
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
