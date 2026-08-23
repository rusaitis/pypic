r"""Destagger Yee-mesh arrays to cell-centered or node-centered grids.

Linear half-cell shifts driven by the openPMD ED-PIC ``position`` tuple
(per-component offset in ``[0.0, 1.0)``).  Covers Cartesian Yee meshes —
PIC codes that store $\mathbf{B}$ on cell faces and $\mathbf{E}$ on cell
edges (VPIC, WarpX, PIConGPU).

Divergence-preserving reconstruction for constrained-transport MHD meshes
(e.g. ARMS) is out of scope; that requires a vector-potential-based
scheme rather than the per-component linear average implemented here.

Output arrays are one element shorter along each shifted axis — the
natural outcome of a half-cell average.  Readers that need uniform-shape
fields handle boundary completion (cropping or extrapolation) themselves.
Readers should also construct a fresh `StaggerInfo` with
``interpolation_order=1`` and a ``notes`` string recording the
destagger provenance.

Currently unwired in production readers — this module is pre-built
infrastructure for the upcoming VPIC reader (``TASKS.md`` Step 35,
which destaggers ``cbx/cby/cbz`` on faces and ``ex/ey/ez`` on edges)
and the openPMD reader (``TASKS.md`` Step 42, which consumes the
ED-PIC per-component ``position`` tuple).  The Cartesian-Yee
algorithm is kept tested in ``tests/test_destagger.py`` so it is
ready to wire up when those readers land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypic.types import FloatArray

_CELL_CENTER: tuple[float, float, float] = (0.5, 0.5, 0.5)
_NODE: tuple[float, float, float] = (0.0, 0.0, 0.0)

_OFFSET_ATOL: float = 1e-9


def _average_along(arr: FloatArray, axis: int) -> FloatArray:
    """Linear half-cell average along *axis*.

    Output shape is one less along *axis*; otherwise unchanged.  This is
    the elementary operation behind every Yee destagger shift.
    """
    if not 0 <= axis < arr.ndim:
        msg = f"axis {axis} out of range for ndim={arr.ndim}"
        raise ValueError(msg)
    lo: list[slice] = [slice(None)] * arr.ndim
    hi: list[slice] = [slice(None)] * arr.ndim
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(lo)] + arr[tuple(hi)])


def _is_half_cell(value: float) -> bool:
    return abs(value) <= _OFFSET_ATOL or abs(value - 0.5) <= _OFFSET_ATOL


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _OFFSET_ATOL


def _destagger_one(
    arr: FloatArray,
    source: tuple[float, ...],
    target: tuple[float, ...],
    *,
    name: str,
) -> FloatArray:
    """Shift *arr* from source stagger position to target.

    Composes a half-cell average along every axis where ``source`` and
    ``target`` differ.  Each offset must be either 0.0 or 0.5;
    mixed-fractional positions (e.g. 0.25) require higher-order
    interpolation and are rejected with `ValueError`.
    """
    if len(source) != arr.ndim:
        msg = (
            f"position_map[{name!r}] has length {len(source)} but array "
            f"{name!r} has ndim {arr.ndim}"
        )
        raise ValueError(msg)
    if len(target) != arr.ndim:
        msg = (
            f"target position has length {len(target)} but array {name!r} "
            f"has ndim {arr.ndim}"
        )
        raise ValueError(msg)
    for label, vec in (("source", source), ("target", target)):
        for offset in vec:
            if not _is_half_cell(offset):
                msg = (
                    f"{label} position {offset} for {name!r} is not in "
                    "{0.0, 0.5} — linear destagger supports half-cell "
                    "shifts only"
                )
                raise ValueError(msg)
    result: FloatArray = arr
    for axis in range(arr.ndim):
        if not _close(source[axis], target[axis]):
            result = _average_along(result, axis)
    return result


def _destagger_to_target(
    arrays: dict[str, FloatArray],
    position_map: Mapping[str, tuple[float, ...]],
    *,
    target: tuple[float, float, float],
) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {}
    for name, arr in arrays.items():
        source = position_map.get(name)
        if source is None:
            out[name] = arr
            continue
        target_for_arr = target[: arr.ndim]
        if len(source) == arr.ndim and all(
            _close(s, t) for s, t in zip(source, target_for_arr, strict=True)
        ):
            out[name] = arr
            continue
        out[name] = _destagger_one(arr, source, target_for_arr, name=name)
    return out


def destagger_arrays_to_cell_centers(
    arrays: dict[str, FloatArray],
    position_map: Mapping[str, tuple[float, ...]],
) -> dict[str, FloatArray]:
    r"""Linearly interpolate Yee-staggered arrays to cell centers.

    Each array's source position is looked up in *position_map* and
    compared against the cell-center target ``(0.5, ..., 0.5)``.  When
    a key is missing or the position already matches the target, the
    array passes through unchanged (object identity preserved).  Output
    arrays are one element shorter along each axis where the source
    offset differed from 0.5.

    Parameters
    ----------
    arrays : dict[str, FloatArray]
        Map from canonical field name to NumPy array.  Arrays may be
        1-, 2-, or 3-dimensional.
    position_map : Mapping[str, tuple[float, ...]]
        Map from canonical field name to its ED-PIC ``position`` tuple,
        each offset in ``{0.0, 0.5}`` (only half-cell offsets are
        supported by linear interpolation).  Matches the shape of
        [`pypic.containers.StaggerInfo.position`][pypic.containers.StaggerInfo.position].

    Returns
    -------
    dict[str, FloatArray]
        Destaggered arrays.  Mismatched output shapes across fields are
        the caller's problem to reconcile (cropping or extrapolation).

    Raises
    ------
    ValueError
        If a position-tuple length does not match its array's ndim, or
        if any offset is neither 0.0 nor 0.5.

    Examples
    --------
    >>> import numpy as np
    >>> b_face = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = destagger_arrays_to_cell_centers(
    ...     arrays={"B_1": b_face},
    ...     position_map={"B_1": (0.5, 0.0)},
    ... )
    >>> out["B_1"]
    array([[1.5, 2.5],
           [4.5, 5.5]])
    """
    return _destagger_to_target(arrays, position_map, target=_CELL_CENTER)


def destagger_arrays_to_nodes(
    arrays: dict[str, FloatArray],
    position_map: Mapping[str, tuple[float, ...]],
) -> dict[str, FloatArray]:
    r"""Linearly interpolate Yee-staggered arrays to grid nodes.

    Mirror of `destagger_arrays_to_cell_centers` with target
    position ``(0.0, ..., 0.0)``.  See that function for parameter and
    error semantics.

    Examples
    --------
    >>> import numpy as np
    >>> b_node = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> out = destagger_arrays_to_nodes(
    ...     arrays={"B_1": b_node},
    ...     position_map={"B_1": (0.5, 0.0)},
    ... )
    >>> out["B_1"]
    array([[2., 3.]])
    """
    return _destagger_to_target(arrays, position_map, target=_NODE)
