r"""Grid-aware cross-model comparison utilities.

These three functions are the only place in pypic where diagnostic
math touches :class:`~pypic.dataset.FieldDataset`. Everything in
:mod:`pypic.diagnostics` stays pure (NumPy in, NumPy out); the
functions here add the glue layer — alignment via :mod:`pypic.regrid`,
alias resolution through both datasets, and SI conversion at the
comparison boundary — then delegate the actual norm evaluation back
to the pure helpers.

Cross-model comparisons (e.g. iPIC3D vs BATSRUS) default to SI because
different normalizations are incomparable in code units; same-model
runs can opt in to ``units="code"`` to skip the conversion.

Examples
--------
>>> import numpy as np
>>> from pypic.dataset import FieldDataset
>>> from pypic.grid import GridInfo
>>> from pypic.units import Normalization
>>> grid = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
>>> a = FieldDataset.from_arrays(
...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid, Normalization.identity(),
... )
>>> b = FieldDataset.from_arrays(
...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid, Normalization.identity(),
... )
>>> float(compare_fields(a, b, "B1"))
0.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pypic.diagnostics import (
    NanPolicy,
    field_difference,
    l2_relative_error,
    linf_error,
)
from pypic.regrid import align_grids

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.dataset import FieldDataset
    from pypic.grid import GridInfo
    from pypic.types import FloatArray

__all__ = [
    "compare_fields",
    "field_comparison_report",
    "field_difference_dataset",
]


_ALLOWED_UNITS = ("si", "code")
_ALLOWED_METRICS = ("l2", "linf")
_ALLOWED_NAN_POLICIES = ("omit", "propagate", "raise")
_RESOLUTION_WARNING_THRESHOLD = 10.0


def _validate_units(units: str) -> None:
    if units not in _ALLOWED_UNITS:
        msg = f"units must be one of {_ALLOWED_UNITS!r}, got {units!r}"
        raise ValueError(msg)


def _validate_metric(metric: str) -> None:
    if metric not in _ALLOWED_METRICS:
        msg = f"metric must be one of {_ALLOWED_METRICS!r}, got {metric!r}"
        raise ValueError(msg)


def _validate_nan_policy(nan_policy: str) -> None:
    if nan_policy not in _ALLOWED_NAN_POLICIES:
        msg = f"nan_policy must be one of {_ALLOWED_NAN_POLICIES!r}, got {nan_policy!r}"
        raise ValueError(msg)


def _resolve_common_field(a: FieldDataset, b: FieldDataset, name: str) -> str:
    """Resolve *name* through both datasets' aliases to a shared canonical."""
    canonical_a = a.resolve_key(name)
    canonical_b = b.resolve_key(name)
    if canonical_a != canonical_b:
        msg = (
            f"Field {name!r} resolves to different canonical names in the two "
            f"datasets: {canonical_a!r} in A, {canonical_b!r} in B"
        )
        raise ValueError(msg)
    return canonical_a


def _common_canonical_fields(a: FieldDataset, b: FieldDataset) -> list[str]:
    """Intersection of canonical field names, preserving A's order."""
    b_names = set(b.field_names())
    return [name for name in a.field_names() if name in b_names]


def _resolve_field_list(
    a: FieldDataset,
    b: FieldDataset,
    fields: Iterable[str] | None,
) -> list[str]:
    """Compute the ordered list of canonical names to compare."""
    if fields is None:
        names = _common_canonical_fields(a, b)
    else:
        seen: dict[str, None] = {}
        for raw in fields:
            seen[_resolve_common_field(a, b, raw)] = None
        names = list(seen)
    if not names:
        msg = "No common fields to compare between the two datasets"
        raise ValueError(msg)
    return names


def _extract_values(ds: FieldDataset, canonical_name: str, units: str) -> FloatArray:
    """Return the field array in the requested units.

    *units* is assumed pre-validated by the public function — see
    :func:`_validate_units`.
    """
    if units == "si":
        return ds.in_si(canonical_name)
    return ds[canonical_name]


def _resolution_ratio(a: GridInfo, b: GridInfo) -> tuple[float, ...]:
    """Per-axis ``max(dx) / min(dx)`` — always ≥ 1.0."""
    return tuple(
        max(da, db) / min(da, db) for da, db in zip(a.spacing, b.spacing, strict=True)
    )


def _warn_if_coarse_mismatch(a: GridInfo, b: GridInfo) -> None:
    """Warn once if any axis' spacing ratio exceeds the threshold."""
    ratios = _resolution_ratio(a, b)
    for axis, ratio in enumerate(ratios):
        if ratio > _RESOLUTION_WARNING_THRESHOLD:
            warnings.warn(
                f"Grid resolutions differ by {ratio:.1f}x along axis {axis} — "
                f"cross-scale comparison may be physically meaningless",
                UserWarning,
                stacklevel=3,
            )
            return


def compare_fields(
    a: FieldDataset,
    b: FieldDataset,
    field: str,
    *,
    metric: str = "l2",
    units: str = "si",
    method: str = "linear",
    nan_policy: NanPolicy = "omit",
) -> float:
    r"""Compute an error norm between one field of two datasets.

    Aligns *a* and *b* onto their common grid via :func:`pypic.regrid.align_grids`,
    resolves *field* through both datasets' aliases to a shared canonical
    name, converts to SI (by default) or leaves in code units, and
    delegates to the pure diagnostic in :mod:`pypic.diagnostics`.

    Parameters
    ----------
    a, b : FieldDataset
        Datasets to compare. Grids may differ in resolution, extent,
        or both — the overlap is computed automatically.
    field : str
        Field name; canonical or alias. Must resolve to the same
        canonical name in both datasets.
    metric : {"l2", "linf"}
        Error norm. ``"l2"`` uses :func:`~pypic.diagnostics.l2_relative_error`
        (relative to *b*); ``"linf"`` uses :func:`~pypic.diagnostics.linf_error`
        (absolute max).
    units : {"si", "code"}
        ``"si"`` converts both fields through :meth:`FieldDataset.in_si`
        before comparing — the default, safe for cross-model runs.
        ``"code"`` compares raw code-unit values; valid only when both
        datasets share a normalization.
    method : str
        Interpolation method passed through to
        :func:`pypic.regrid.align_grids` (e.g. ``"linear"``,
        ``"nearest"``, ``"cubic"``). Default ``"linear"``.
    nan_policy : {"omit", "propagate", "raise"}
        Forwarded to the pure diagnostic. Default ``"omit"`` masks NaN
        cells from the metric (with a warning naming the dropped count)
        — useful for sphere selections, masked regions, and other
        upstream sources of NaN. Use ``"propagate"`` for strict
        verification where any NaN should poison the result.

    Returns
    -------
    float
        Scalar error metric.

    Raises
    ------
    KeyError
        If *field* is missing in either dataset (message from
        :meth:`FieldDataset.resolve_key` includes close-match suggestions).
    ValueError
        If *metric* or *units* is unknown, if *field* resolves to
        different canonical names in the two datasets, or if the grids
        do not overlap.
    NotImplementedError
        If either grid is non-Cartesian (propagated from :func:`align_grids`).

    Warns
    -----
    UserWarning
        If any axis' spacing ratio exceeds 10×.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
    >>> ds = FieldDataset.from_arrays(
    ...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> float(compare_fields(ds, ds, "B1"))
    0.0
    """
    _validate_metric(metric)
    _validate_units(units)
    _validate_nan_policy(nan_policy)
    canonical = _resolve_common_field(a, b, field)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    a_aligned, b_aligned = align_grids(a, b, method=method)
    va = _extract_values(a_aligned, canonical, units)
    vb = _extract_values(b_aligned, canonical, units)
    if metric == "l2":
        return float(l2_relative_error(va, vb, nan_policy=nan_policy))
    return float(linf_error(va, vb, nan_policy=nan_policy))


def field_comparison_report(
    a: FieldDataset,
    b: FieldDataset,
    *,
    fields: Iterable[str] | None = None,
    units: str = "si",
    method: str = "linear",
    nan_policy: NanPolicy = "omit",
) -> dict[str, Any]:
    r"""Compute L2 and L∞ errors for every common field, plus grid context.

    Aligns the datasets once, then loops over the requested (or shared)
    canonical field names, computing both norms per field. The grid
    context captures domain extent and resolution ratio so a reader can
    judge whether the comparison is physically meaningful — e.g. a
    100× spacing mismatch between a kinetic-scale PIC run and an
    MHD-scale run is numerically computable but suspect.

    Parameters
    ----------
    a, b : FieldDataset
        Datasets to compare.
    fields : Iterable[str] | None
        Field names to report on. ``None`` reports on the intersection
        of canonical field names present in both datasets. Explicit names
        may be aliases; they resolve through both datasets.
    units : {"si", "code"}
        Unit convention; see :func:`compare_fields`.
    method : str
        Interpolation method passed through to
        :func:`pypic.regrid.align_grids`. Default ``"linear"``.
    nan_policy : {"omit", "propagate", "raise"}
        Forwarded to the pure diagnostics; see :func:`compare_fields`.

    Returns
    -------
    dict
        Report with keys

        - ``"fields"`` — ``{canonical_name: {"l2": float, "linf": float}}``
        - ``"grid"`` — common-grid and per-axis resolution-ratio info
        - ``"units"`` — echo of the *units* argument

    Raises
    ------
    ValueError
        If there are no common fields, or on invalid *units*.
    KeyError
        If an explicit *fields* entry is missing in either dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
    >>> ds = FieldDataset.from_arrays(
    ...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> report = field_comparison_report(ds, ds)
    >>> report["fields"]["B1"]["l2"]
    0.0
    >>> report["units"]
    'si'
    """
    _validate_units(units)
    _validate_nan_policy(nan_policy)
    # Resolve names against the originals so custom aliases from
    # ``from_arrays(aliases=...)`` survive (regrid only regenerates the
    # geometry-default aliases on its output) and so bad field names
    # raise *before* the expensive alignment step.
    names = _resolve_field_list(a, b, fields)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    a_aligned, b_aligned = align_grids(a, b, method=method)

    per_field: dict[str, dict[str, float]] = {}
    for name in names:
        va = _extract_values(a_aligned, name, units)
        vb = _extract_values(b_aligned, name, units)
        per_field[name] = {
            "l2": float(l2_relative_error(va, vb, nan_policy=nan_policy)),
            "linf": float(linf_error(va, vb, nan_policy=nan_policy)),
        }

    grid_context: dict[str, Any] = {
        "common_dimensions": a_aligned.grid.dimensions,
        "common_spacing": a_aligned.grid.spacing,
        "common_origin": a_aligned.grid.origin,
        "resolution_ratio": _resolution_ratio(a.grid, b.grid),
        "source_a_dimensions": a.grid.dimensions,
        "source_b_dimensions": b.grid.dimensions,
    }
    return {"fields": per_field, "grid": grid_context, "units": units}


def field_difference_dataset(
    a: FieldDataset,
    b: FieldDataset,
    *,
    fields: Iterable[str] | None = None,
    units: str = "si",
    method: str = "linear",
) -> FieldDataset:
    r"""Build a FieldDataset of pointwise differences on the common grid.

    Each requested field is computed as ``a[name] - b[name]`` after the
    two datasets are aligned. The returned dataset inherits *a*'s
    normalization, species, physics, and frame metadata, so the result
    plugs directly into :func:`pypic.plotting.plot_field_slice`. For the
    three-panel A | B | diff layout, run :func:`pypic.regrid.align_grids`
    yourself and pass the pair to :func:`pypic.plotting.plot_comparison`
    — that path does not need this helper.

    When ``units="si"``, the stored arrays carry SI values even though
    the dataset's normalization is from *a*; the SI choice is recorded
    in ``metadata["comparison"]["units"]`` for provenance.

    Parameters
    ----------
    a, b : FieldDataset
        Datasets to subtract. Grids may differ.
    fields : Iterable[str] | None
        Field names to include. ``None`` uses the full intersection of
        canonical names.
    units : {"si", "code"}
        Unit convention; see :func:`compare_fields`.
    method : str
        Interpolation method passed through to
        :func:`pypic.regrid.align_grids`. Default ``"linear"``.

        NaN cells in either input pass through to the difference array
        unchanged. Use the diagnostic functions or :func:`compare_fields`
        with ``nan_policy`` if you need to mask them when computing
        downstream metrics.

    Returns
    -------
    FieldDataset
        Dataset on ``common_grid(a.grid, b.grid)`` containing
        ``a - b`` for each selected field.

    Raises
    ------
    ValueError
        If there are no common fields, or on invalid *units*.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4,), spacing=(1.0,), origin=(0.0,))
    >>> a = FieldDataset.from_arrays(
    ...     {"B1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> b = FieldDataset.from_arrays(
    ...     {"B1": np.array([1.0, 1.5, 2.5, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> diff = field_difference_dataset(a, b)
    >>> diff["B1"]
    array([0. , 0.5, 0.5, 0. ])
    """
    _validate_units(units)
    # Resolve against originals — see field_comparison_report for the
    # rationale (custom aliases survive, bad names raise pre-alignment).
    names = _resolve_field_list(a, b, fields)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    a_aligned, b_aligned = align_grids(a, b, method=method)

    diff_fields: dict[str, FloatArray] = {}
    for name in names:
        va = _extract_values(a_aligned, name, units)
        vb = _extract_values(b_aligned, name, units)
        diff_fields[name] = field_difference(va, vb)

    from pypic.dataset import FieldDataset as _FieldDataset
    from pypic.units import Normalization

    # When units="si" the stored arrays already carry SI values, so the
    # result must use an identity normalization — otherwise calling
    # in_si() on the returned dataset would re-apply the SI factor and
    # silently double-convert. With identity, in_si() returns the same
    # SI values and __getitem__ also returns SI (which is now both
    # "code" and "SI" simultaneously, since the factors are 1.0).
    result_norm = Normalization.identity() if units == "si" else a_aligned.normalization

    return _FieldDataset.from_arrays(
        diff_fields,
        a_aligned.grid,
        result_norm,
        species=list(a_aligned.species),
        physics=a_aligned.physics,
        frame=a_aligned.frame,
        transforms=dict(a_aligned.transforms),
        metadata={
            "comparison": {
                "source_frames": (a.frame, b.frame),
                "units": units,
            }
        },
    )
