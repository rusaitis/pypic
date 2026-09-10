r"""Grid-aware cross-model comparison utilities.

These three functions are the only place in pypic where diagnostic
math touches [`FieldDataset`][pypic.dataset.FieldDataset]. Everything in
[`pypic.diagnostics`][pypic.diagnostics] stays pure (NumPy in, NumPy out); the
functions here add the glue layer — alignment via [`pypic.regrid`][pypic.regrid],
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
...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid, Normalization.identity(),
... )
>>> b = FieldDataset.from_arrays(
...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
...     grid, Normalization.identity(),
... )
>>> float(compare_fields(a, b, "B_1"))
0.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pypic.diagnostics import (
    _PYPIC_PREFIX,
    NanPolicy,
    field_difference,
    l2_relative_error,
    linf_error,
)
from pypic.dataset import FieldDataset
from pypic.regrid import align_grids
from pypic.units import Normalization

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def _validate_choice(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        msg = f"{name} must be one of {allowed!r}, got {value!r}"
        raise ValueError(msg)


def _validate_code_units_compatible(
    a: FieldDataset, b: FieldDataset, units: str
) -> None:
    """Refuse ``units='code'`` across mismatched normalizations.

    Code-unit comparison is only meaningful when both datasets share a
    normalization (same length/time/B/density references). Comparing
    PIC code values to MHD code values is physically meaningless even
    though the arithmetic succeeds — the *numeric* value of ``B_1`` in
    a PIC dump and a BATSRUS dump means very different things in SI.
    Use ``units='si'`` for cross-model.
    """
    if units == "code" and a.normalization != b.normalization:
        msg = (
            "units='code' requires both datasets to share a normalization "
            "(same length/time/B/density references), but A and B differ. "
            "Pass units='si' to compare in SI, or align normalizations "
            "upstream before calling."
        )
        raise ValueError(msg)


def _transform_or_raise(ds: FieldDataset, target: str, *, label: str) -> FieldDataset:
    """Transform *ds* to *target* frame, or raise a clear ValueError."""
    if ds.frame == target:
        return ds
    try:
        return ds.transform_to(target)
    except (KeyError, ValueError) as exc:
        msg = (
            f"Cannot align dataset {label} from frame {ds.frame!r} to "
            f"{target!r}: no transform registered on {label}. Register one "
            f"via [coordinates.transforms] in simulation.toml, or call "
            f"{label.lower()}.transform_to({target!r}) before comparing."
        )
        raise ValueError(msg) from exc


def _align_frames(
    a: FieldDataset,
    b: FieldDataset,
    *,
    frame: str | None = None,
) -> tuple[FieldDataset, FieldDataset]:
    """Bring both datasets into a common frame.

    When *frame* is ``None`` (the default), the common frame is *a*'s
    frame and only *b* may be transformed. When *frame* is given, both
    *a* and *b* are transformed to that frame — useful for comparing
    in a third reference frame neither input lives in natively.

    Frame alignment runs *before* grid alignment because a frame
    transform rotates and translates the grid itself; aligning grids
    first would compute a common domain across two incompatible
    coordinate systems.

    Frame transforms are static. When ``transform_to`` grows an
    ``epoch`` kwarg for dipole-tilt-style rotations and SPICE
    ephemerides, this helper forwards it; there is nothing to thread
    through until then.
    """
    if frame is not None and not frame:
        msg = "frame must be a non-empty string"
        raise ValueError(msg)
    target = a.frame if frame is None else frame
    if a.frame == target and b.frame == target:
        return a, b
    return (
        _transform_or_raise(a, target, label="A"),
        _transform_or_raise(b, target, label="B"),
    )


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
    `_validate_choice`.
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
    """Warn at most once per call if any axis spacing ratio exceeds the threshold.

    The warning fires for the first axis that crosses
    `_RESOLUTION_WARNING_THRESHOLD` and then returns, by design:
    a user running ``compare_fields(kinetic, mhd)`` wants one heads-up
    about the cross-scale comparison, not three redundant ones when all
    axes are 50× off. ``skip_file_prefixes`` routes the warning past
    the pypic frames to the user's call site regardless of layering.
    """
    ratios = _resolution_ratio(a, b)
    for axis, ratio in enumerate(ratios):
        if ratio > _RESOLUTION_WARNING_THRESHOLD:
            warnings.warn(
                f"Grid resolutions differ by {ratio:.1f}x along axis {axis} — "
                f"cross-scale comparison may be physically meaningless",
                UserWarning,
                skip_file_prefixes=_PYPIC_PREFIX,
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
    frame: str | None = None,
) -> float:
    r"""Compute an error norm between one field of two datasets.

    Aligns *a* and *b* onto their common grid via
    [`pypic.regrid.align_grids`][pypic.regrid.align_grids],
    resolves *field* through both datasets' aliases to a shared canonical
    name, converts to SI (by default) or leaves in code units, and
    delegates to the pure diagnostic in [`pypic.diagnostics`][pypic.diagnostics].

    When the two datasets are in different frames, *b* is transformed to
    *a*'s frame via `FieldDataset.transform_to` before alignment.
    Pass an explicit *frame* to compare in a third reference frame —
    both inputs are then transformed to that frame instead. A clear
    `ValueError` is raised if any required transform is missing.

    Parameters
    ----------
    a, b : FieldDataset
        Datasets to compare. Grids may differ in resolution, extent,
        or both — the overlap is computed automatically.
    field : str
        Field name; canonical or alias. Must resolve to the same
        canonical name in both datasets.
    metric : {"l2", "linf"}
        Error norm. ``"l2"`` uses
        [`l2_relative_error`][pypic.diagnostics.l2_relative_error]
        (relative to *b*); ``"linf"`` uses [`linf_error`][pypic.diagnostics.linf_error]
        (absolute max).
    units : {"si", "code"}
        ``"si"`` converts both fields through `FieldDataset.in_si`
        before comparing — the default, safe for cross-model runs.
        ``"code"`` compares raw code-unit values; valid only when both
        datasets share a normalization.
    method : str
        Interpolation method passed through to
        [`pypic.regrid.align_grids`][pypic.regrid.align_grids] (e.g. ``"linear"``,
        ``"nearest"``, ``"cubic"``). Default ``"linear"``.
    nan_policy : {"omit", "propagate", "raise"}
        Forwarded to the pure diagnostic. Default ``"omit"`` masks NaN
        cells from the metric (with a warning naming the dropped count)
        — useful for sphere selections, masked regions, and other
        upstream sources of NaN. Use ``"propagate"`` for strict
        verification where any NaN should poison the result.
    frame : str | None
        Reference frame to compare in. ``None`` (default) uses *a*'s
        frame, transforming *b* if needed. A non-``None`` value
        transforms both *a* and *b* to that frame first — useful when
        neither dataset lives natively in the frame you want to plot or
        report in. Each dataset must have a transform registered to the
        target (or already be in it).

    Returns
    -------
    float
        Scalar error metric.

    Raises
    ------
    KeyError
        If *field* is missing in either dataset (message from
        `FieldDataset.resolve_key` includes close-match suggestions).
    ValueError
        If *metric* or *units* is unknown, if *field* resolves to
        different canonical names in the two datasets, or if the grids
        do not overlap.
    NotImplementedError
        If either grid is non-Cartesian (propagated from `align_grids`).

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
    ...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> float(compare_fields(ds, ds, "B_1"))
    0.0
    """
    _validate_choice(metric, _ALLOWED_METRICS, "metric")
    _validate_choice(units, _ALLOWED_UNITS, "units")
    # Duplicates the check inside ``_apply_nan_policy`` intentionally:
    # a bad policy caught *here* fails before the expensive alignment
    # step, turning a wasted multi-field regrid into an instant error.
    _validate_choice(nan_policy, _ALLOWED_NAN_POLICIES, "nan_policy")
    _validate_code_units_compatible(a, b, units)
    # Resolve against the *original* datasets: alignment rebuilds them
    # without custom aliases (``transform_to`` drops them, ``regrid`` keeps
    # only geometry defaults), so a post-alignment lookup would lose anything
    # from ``from_arrays(aliases=...)``.  Canonical names survive both.
    canonical = _resolve_common_field(a, b, field)
    a, b = _align_frames(a, b, frame=frame)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    # Regrid only the requested field — 50× cheaper than full-dataset
    # alignment on a multi-moment PIC dump.
    a_aligned, b_aligned = align_grids(a, b, fields=[canonical], method=method)
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
    frame: str | None = None,
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
        Unit convention; see `compare_fields`.
    method : str
        Interpolation method passed through to
        [`pypic.regrid.align_grids`][pypic.regrid.align_grids]. Default ``"linear"``.
    nan_policy : {"omit", "propagate", "raise"}
        Forwarded to the pure diagnostics; see `compare_fields`.
    frame : str | None
        Reference frame to compare in; see `compare_fields`.

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
    ...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> report = field_comparison_report(ds, ds)
    >>> report["fields"]["B_1"]["l2"]
    0.0
    >>> report["units"]
    'si'
    """
    _validate_choice(units, _ALLOWED_UNITS, "units")
    _validate_choice(nan_policy, _ALLOWED_NAN_POLICIES, "nan_policy")
    _validate_code_units_compatible(a, b, units)
    # Resolve names against the *originals* so custom aliases from
    # ``from_arrays(aliases=...)`` survive — both transform_to and
    # regrid drop user aliases, so resolving post-alignment would lose
    # them. Bad names also raise *before* the expensive alignment step.
    names = _resolve_field_list(a, b, fields)
    a, b = _align_frames(a, b, frame=frame)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    a_aligned, b_aligned = align_grids(a, b, fields=names, method=method)

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
    frame: str | None = None,
) -> FieldDataset:
    r"""Build a FieldDataset of pointwise differences on the common grid.

    Each requested field is computed as ``a[name] - b[name]`` after the
    two datasets are aligned. The returned dataset inherits *a*'s
    species, physics, and frame metadata, so the result plugs directly
    into [`plot_field_slice`][pypic.plotting.plot_field_slice]. For the
    three-panel A | B | diff layout, run
    [`align_grids`][pypic.regrid.align_grids] yourself and pass the pair to
    [`plot_comparison`][pypic.plotting.plot_comparison] — that path does not
    need this helper.

    Unlike [`pypic.regrid.regrid`][pypic.regrid.regrid], which preserves *a*'s original
    metadata dict verbatim, this function **replaces** ``.metadata``
    with a fresh ``{"comparison": {"source_frames": ..., "units": ...}}``
    record — the diff is a new artifact, not a regrid of *a*, and any
    per-step provenance on the sources would be misleading if copied.

    NaN cells in either input pass through the difference array
    unchanged (NaN minus anything = NaN). There is no ``nan_policy``
    parameter because `field_difference` itself is pure
    subtraction; use `compare_fields` with ``nan_policy=...`` or
    [`l2_relative_error`][pypic.diagnostics.l2_relative_error] directly if you need
    masked reductions.

    When ``units="si"``, the stored arrays carry SI values but the
    dataset's normalization is set to `Normalization.identity`
    so that `FieldDataset.in_si` returns the same values instead
    of re-applying the SI factor. The actual unit choice is recorded
    in ``metadata["comparison"]["units"]`` for provenance.

    Parameters
    ----------
    a, b : FieldDataset
        Datasets to subtract. Grids may differ.
    fields : Iterable[str] | None
        Field names to include. ``None`` uses the full intersection of
        canonical names.
    units : {"si", "code"}
        Unit convention; see `compare_fields`.
    method : str
        Interpolation method passed through to
        [`pypic.regrid.align_grids`][pypic.regrid.align_grids]. Default ``"linear"``.
    frame : str | None
        Reference frame for the result; see `compare_fields`. The
        returned dataset's ``frame`` attribute reflects this choice
        (``a.frame`` when ``None``, otherwise the requested frame).

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
    ...     {"B_1": np.array([1.0, 2.0, 3.0, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> b = FieldDataset.from_arrays(
    ...     {"B_1": np.array([1.0, 1.5, 2.5, 4.0])},
    ...     grid, Normalization.identity(),
    ... )
    >>> diff = field_difference_dataset(a, b)
    >>> diff["B_1"]
    array([0. , 0.5, 0.5, 0. ])
    """
    _validate_choice(units, _ALLOWED_UNITS, "units")
    _validate_code_units_compatible(a, b, units)
    # Capture original frames *before* _align_frames for the provenance
    # record below; the metadata should reflect what the user passed in,
    # not the post-transform frame on B.
    source_frames = (a.frame, b.frame)
    # Resolve against originals — see field_comparison_report for the
    # rationale (custom aliases survive transform_to/regrid only when
    # resolved up front; bad names raise pre-alignment).
    names = _resolve_field_list(a, b, fields)
    a, b = _align_frames(a, b, frame=frame)
    _warn_if_coarse_mismatch(a.grid, b.grid)
    a_aligned, b_aligned = align_grids(a, b, fields=names, method=method)

    diff_fields: dict[str, FloatArray] = {}
    for name in names:
        va = _extract_values(a_aligned, name, units)
        vb = _extract_values(b_aligned, name, units)
        diff_fields[name] = field_difference(va, vb)

    # With units="si" the stored arrays are already SI, so the result needs
    # an identity normalization — otherwise ``in_si()`` would re-apply the
    # factor and silently double-convert.
    result_norm = Normalization.identity() if units == "si" else a_aligned.normalization

    return FieldDataset.from_arrays(
        diff_fields,
        a_aligned.grid,
        result_norm,
        species=list(a_aligned.species),
        physics=a_aligned.physics,
        frame=a_aligned.frame,
        transforms=dict(a_aligned.transforms),
        metadata={
            "comparison": {
                "source_frames": source_frames,
                "units": units,
            }
        },
        strict_fields=False,
    )
