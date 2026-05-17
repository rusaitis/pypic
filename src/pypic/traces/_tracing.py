"""RK4 field line tracer with fixed-step and adaptive variants."""

from __future__ import annotations

__all__ = [
    "TerminationReason",
    "VectorFieldInterpolator",
    "estimate_tracing_error",
    "trace_field_line",
    "trace_field_line_adaptive",
    "trace_field_lines_adaptive",
]

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pypic.numerics import (
    dormand_prince_step,
    dormand_prince_step_batched,
    embedded_error_norm,
    embedded_error_norm_batched,
    i_step_controller,
    i_step_controller_batched,
)
from pypic.traces._fieldline import _VALID_DIRECTIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.dataset import FieldDataset
    from pypic.traces._fieldline import FieldLine
    from pypic.types import BoolArray, FloatArray, IntArray, Vector3


class TerminationReason(StrEnum):
    """Why a field line trace stopped."""

    MAX_STEPS = "max_steps"  # reached step limit
    DOMAIN_EXIT = "domain_exit"  # left interpolation domain (NaN)
    NULL_POINT = "null_point"  # |B| below null_threshold
    CALLBACK = "callback"  # user terminate() returned True


@dataclass(frozen=True, slots=True)
class VectorFieldInterpolator:
    r"""Pre-built trilinear interpolator for a 3-component vector field.

    Wraps a single ``RegularGridInterpolator`` over a stacked
    ``(..., 3)`` value array so each ``__call__`` dispatches once
    instead of three times. Built once and reused across all RK4
    stages and seed points.
    """

    _interp: RegularGridInterpolator

    @classmethod
    def from_dataset(
        cls,
        data: FieldDataset,
        components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    ) -> Self:
        """Build from a FieldDataset.

        Parameters
        ----------
        data : FieldDataset
            Gridded field data.
        components : tuple[str, str, str]
            Names of the three vector components.

        Returns
        -------
        VectorFieldInterpolator
        """
        coord_arrays = data.grid.coordinate_arrays()
        stacked = np.stack(
            [np.asarray(data[c], dtype=np.float64) for c in components],
            axis=-1,
        )
        interp = RegularGridInterpolator(
            coord_arrays,
            stacked,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return cls(_interp=interp)

    def __call__(self, point: FloatArray) -> FloatArray:
        """Evaluate the vector field at a single point.

        Parameters
        ----------
        point : FloatArray
            Position, shape ``(3,)``.

        Returns
        -------
        FloatArray
            Field vector, shape ``(3,)``. NaN if outside domain.
        """
        return self._interp(point.reshape(1, 3))[0]  # type: ignore[no-any-return]

    def batch(self, points: FloatArray) -> FloatArray:
        """Evaluate the vector field at ``N`` points in one call.

        Used by the batched adaptive tracer so all seeds in a step share
        a single ``RegularGridInterpolator`` dispatch. Roughly N× faster
        than calling :meth:`__call__` N times because per-call Python /
        argument-marshaling overhead amortizes over the batch.

        Parameters
        ----------
        points : FloatArray
            Positions, shape ``(N, 3)``.

        Returns
        -------
        FloatArray
            Field vectors, shape ``(N, 3)``. Rows are NaN where the
            corresponding point is outside the interpolation domain.
        """
        return self._interp(points)


def _rhs(
    point: FloatArray,
    interp: VectorFieldInterpolator,
    sign: float,
    null_threshold: float,
) -> FloatArray | None:
    """Compute unit field direction at *point*, scaled by *sign*.

    Returns ``None`` if the point is outside the domain or at a null
    point (field magnitude below *null_threshold*).
    """
    b = interp(point)
    if np.any(np.isnan(b)):
        return None
    mag = np.linalg.norm(b)
    if mag < null_threshold:
        return None
    return sign * b / mag


def _classify_failure(
    point: FloatArray,
    interp: VectorFieldInterpolator,
) -> TerminationReason:
    """Classify a failed RHS evaluation as domain exit or null."""
    b = interp(point)
    if np.any(np.isnan(b)):
        return TerminationReason.DOMAIN_EXIT
    return TerminationReason.NULL_POINT


def _trace_single_direction(
    interp: VectorFieldInterpolator,
    seed: FloatArray,
    sign: float,
    step_size: float,
    max_steps: int,
    null_threshold: float,
    terminate: Callable[[FloatArray], bool] | None,
) -> tuple[FloatArray, TerminationReason]:
    """Fixed-step classical RK4 integration in one direction."""
    buf = np.empty((max_steps + 1, 3), dtype=np.float64)
    buf[0] = seed
    n = 0
    reason = TerminationReason.MAX_STEPS
    h = step_size

    for _ in range(max_steps):
        y = buf[n]

        k1 = _rhs(y, interp, sign, null_threshold)
        if k1 is None:
            reason = _classify_failure(y, interp)
            break

        p2 = y + 0.5 * h * k1
        k2 = _rhs(p2, interp, sign, null_threshold)
        if k2 is None:
            reason = _classify_failure(p2, interp)
            break

        p3 = y + 0.5 * h * k2
        k3 = _rhs(p3, interp, sign, null_threshold)
        if k3 is None:
            reason = _classify_failure(p3, interp)
            break

        p4 = y + h * k3
        k4 = _rhs(p4, interp, sign, null_threshold)
        if k4 is None:
            reason = _classify_failure(p4, interp)
            break

        n += 1
        buf[n] = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if terminate is not None and terminate(buf[n]):
            reason = TerminationReason.CALLBACK
            break

    return buf[: n + 1], reason


def _trace_single_direction_adaptive(
    interp: VectorFieldInterpolator,
    seed: FloatArray,
    sign: float,
    atol: float,
    rtol: float,
    step_size_init: float,
    min_step: float,
    max_step: float,
    max_steps: int,
    null_threshold: float,
    terminate: Callable[[FloatArray], bool] | None,
) -> tuple[FloatArray, TerminationReason, float]:
    """Dormand-Prince 5(4) adaptive integration in one direction.

    Bookkeeping (buffer, signed direction, null/out-of-domain
    classification, termination callback) lives here; the pure
    Dormand-Prince step + error estimator + step-size controller
    live in :mod:`pypic.numerics`.
    """
    buf = np.empty((max_steps + 1, 3), dtype=np.float64)
    buf[0] = seed
    n = 0
    reason = TerminationReason.MAX_STEPS
    h = step_size_init
    max_local_error = 0.0
    # FSAL: k_last from the previous accepted step seeds k[0] of the
    # next attempt. None on the very first attempt. On a rejection,
    # y is unchanged so k_carry stays valid for the retry — no reset.
    k_carry: FloatArray | None = None

    def rhs(yi: FloatArray) -> FloatArray | None:
        return _rhs(yi, interp, sign, null_threshold)

    while n < max_steps:
        result = dormand_prince_step(rhs, buf[n], h, k0=k_carry)
        if result.failed_stage is not None:
            assert result.failed_point is not None  # invariant of DPStepResult
            reason = _classify_failure(result.failed_point, interp)
            break

        assert result.y_new is not None  # success path
        assert result.err_vec is not None
        assert result.k_last is not None
        err_norm = embedded_error_norm(result.err_vec, result.y_new, atol, rtol)
        max_local_error = max(max_local_error, err_norm)
        h_new = i_step_controller(h, err_norm, min_step=min_step, max_step=max_step)

        if err_norm <= 1.0 or h <= min_step:
            n += 1
            buf[n] = result.y_new
            k_carry = result.k_last
            h = h_new
            if terminate is not None and terminate(result.y_new):
                reason = TerminationReason.CALLBACK
                break
        else:
            # Reject: y unchanged, retry with smaller h. k_carry stays
            # as-is (still f(buf[n]) from the prior accept), so the
            # retry also benefits from FSAL.
            h = h_new

    return buf[: n + 1], reason, max_local_error


# Reason codes for the batched tracer's per-seed reasons_int array.
# Keeping the encoding numeric lets the per-step bookkeeping stay in
# vectorized NumPy (rather than dropping into Python objects for
# enum values).
_R_MAX = 0
_R_DOMAIN = 1
_R_NULL = 2
_R_CALLBACK = 3

_REASON_FROM_INT: dict[int, TerminationReason] = {
    _R_MAX: TerminationReason.MAX_STEPS,
    _R_DOMAIN: TerminationReason.DOMAIN_EXIT,
    _R_NULL: TerminationReason.NULL_POINT,
    _R_CALLBACK: TerminationReason.CALLBACK,
}


def _trace_batch_single_direction_adaptive(
    interp: VectorFieldInterpolator,
    seeds: FloatArray,
    sign: float,
    atol: float,
    rtol: float,
    step_size_init: float,
    min_step: float,
    max_step: float,
    max_steps: int,
    null_threshold: float,
    terminate: Callable[[FloatArray], bool] | None,
) -> tuple[FloatArray, IntArray, IntArray, FloatArray]:
    """Batched Dormand-Prince 5(4) adaptive integration in one direction.

    Runs the kernel on the full ``(N, 3)`` batch each step; per-seed
    termination is tracked via a ``live`` mask in the closure RHS so
    dead seeds report ``valid=False`` and don't contaminate the
    surviving seeds (Butcher contraction is stage-axis only).

    Returns a tuple ``(buf, n_steps, reasons_int, max_local_error)``:
    ``buf`` has shape ``(N, max_steps + 1, 3)`` with the per-seed
    trajectory written in-place; ``n_steps[i]`` is the number of
    *accepted* steps for seed ``i`` (so ``buf[i, :n_steps[i] + 1]`` is
    the seed's trace, always at least one point — the seed itself);
    ``reasons_int[i]`` is the encoded :class:`TerminationReason`;
    ``max_local_error[i]`` is the worst per-step error norm for seed
    ``i``.
    """
    n_seeds = seeds.shape[0]
    buf = np.empty((n_seeds, max_steps + 1, 3), dtype=np.float64)
    buf[:, 0] = seeds
    n_steps = np.zeros(n_seeds, dtype=np.intp)
    reasons_int = np.full(n_seeds, -1, dtype=np.intp)
    h = np.full(n_seeds, step_size_init, dtype=np.float64)
    max_local_error = np.zeros(n_seeds, dtype=np.float64)
    k_carry: FloatArray | None = None
    live = np.ones(n_seeds, dtype=bool)

    def rhs_batched(y: FloatArray) -> tuple[FloatArray, BoolArray]:
        b = interp.batch(y)
        mag = np.linalg.norm(b, axis=-1)
        valid = live & np.isfinite(mag) & (mag >= null_threshold)
        out = np.zeros_like(b)
        if valid.any():
            out[valid] = sign * b[valid] / mag[valid, np.newaxis]
        return out, valid

    while live.any():
        # Terminate any seed that has filled its buffer (no more room
        # to advance — buf is shape (N, max_steps + 1, 3)). Check at
        # the top of the loop so we never overrun by accepting one
        # past the end.
        maxed = live & (n_steps >= max_steps)
        if maxed.any():
            reasons_int[maxed] = _R_MAX
            live &= ~maxed
            if not live.any():
                break

        y_cur = buf[np.arange(n_seeds), n_steps]  # (N, 3)
        result = dormand_prince_step_batched(rhs_batched, y_cur, h, k0=k_carry)

        err_norm = embedded_error_norm_batched(
            result.err_vec, result.y_new, atol, rtol
        )
        # Track per-seed worst error only for live seeds. Dead seeds
        # have garbage err_norm; keep their previous max_local_error.
        max_local_error = np.maximum(
            max_local_error, np.where(live, err_norm, 0.0)
        )
        h_new = i_step_controller_batched(
            h, err_norm, min_step=min_step, max_step=max_step
        )

        success = result.failed_stage == -1
        accept = live & success & ((err_norm <= 1.0) | (h <= min_step))
        newly_failed = live & ~success

        if accept.any():
            acc_idx = np.flatnonzero(accept)
            n_steps[acc_idx] += 1
            buf[acc_idx, n_steps[acc_idx]] = result.y_new[acc_idx]
            if k_carry is None:
                k_carry = np.zeros((n_seeds, 3), dtype=np.float64)
            k_carry[acc_idx] = result.k_last[acc_idx]

        # Update h for all live seeds. Rejected seeds (live & success &
        # ~accept) retry with the smaller h; dead seeds keep their last
        # h (irrelevant — they won't step again).
        h = np.where(live, h_new, h)

        if newly_failed.any():
            # Per-seed classify by re-evaluating the interpolator at the
            # failed point. This loop is N_failed long, not N_total, and
            # only runs on terminating steps.
            for i in np.flatnonzero(newly_failed):
                fp = result.failed_point[i]
                b_at = interp(fp)
                if np.any(np.isnan(b_at)):
                    reasons_int[i] = _R_DOMAIN
                else:
                    reasons_int[i] = _R_NULL
            live &= ~newly_failed

        if terminate is not None and accept.any():
            for i in np.flatnonzero(accept):
                if terminate(buf[i, n_steps[i]]):
                    reasons_int[i] = _R_CALLBACK
                    live[i] = False

    return buf, n_steps, reasons_int, max_local_error


_EMPTY_POINTS = np.empty((0, 3), dtype=np.float64)


def _assemble_field_line(
    fwd_points: FloatArray,
    fwd_reason: TerminationReason,
    bwd_points: FloatArray,
    bwd_reason: TerminationReason,
    seed: Vector3,
    direction: str,
    field_name: str,
    normalization: object,
    metadata: dict,  # type: ignore[type-arg]
) -> FieldLine:
    """Concatenate forward/backward traces into a FieldLine."""
    from pypic.traces._fieldline import FieldLine as _FieldLine

    match direction:
        case "forward":
            all_points = fwd_points
            reason = fwd_reason
        case "backward":
            all_points = bwd_points[::-1]
            reason = bwd_reason
        case _:
            # "both": reverse backward, drop duplicated seed, append forward.
            # Reason: MAX_STEPS if either direction was truncated,
            # otherwise forward reason (arbitrary but deterministic).
            bwd_rev = bwd_points[::-1]
            if len(bwd_rev) > 0 and len(fwd_points) > 0:
                all_points = np.concatenate([bwd_rev[:-1], fwd_points])
            elif len(bwd_rev) > 0:
                all_points = bwd_rev
            else:
                all_points = fwd_points
            is_max = (
                fwd_reason == TerminationReason.MAX_STEPS
                or bwd_reason == TerminationReason.MAX_STEPS
            )
            reason = TerminationReason.MAX_STEPS if is_max else fwd_reason

    metadata["reason"] = str(reason)
    metadata["n_steps"] = len(all_points) - 1

    return _FieldLine(
        points=all_points,
        field_name=field_name,
        seed_point=seed,
        normalization=normalization,  # type: ignore[arg-type]
        direction=direction,
        metadata=metadata,
    )


def trace_field_line(
    data: FieldDataset,
    seed: Vector3,
    *,
    step_size: float = 0.5,
    max_steps: int = 10_000,
    direction: str = "both",
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    null_threshold: float = 1e-12,
    terminate: Callable[[FloatArray], bool] | None = None,
    interpolator: VectorFieldInterpolator | None = None,
) -> FieldLine:
    r"""Trace a field line using classical (fixed-step) RK4.

    Integrates $d\mathbf{r}/ds = \hat{\mathbf{B}}(\mathbf{r})$ where
    $\hat{\mathbf{B}} = \mathbf{B}/|\mathbf{B}|$ and $s$ is arc length.

    Parameters
    ----------
    data : FieldDataset
        Gridded vector field data.
    seed : Vector3
        Starting point ``(x, y, z)`` for the trace.
    step_size : float
        Arc-length step size in code units.
    max_steps : int
        Maximum integration steps per direction.
    direction : str
        ``"forward"``, ``"backward"``, or ``"both"``.
    field_components : tuple[str, str, str]
        Names of the three vector field components.
    null_threshold : float
        Field magnitude below which the point is a null.
    terminate : Callable[[FloatArray], bool] | None
        Optional callback; stops if it returns ``True``.
    interpolator : VectorFieldInterpolator | None
        Pre-built interpolator. Built internally if ``None``.

    Returns
    -------
    FieldLine

    Raises
    ------
    ValueError
        If seed is outside domain or at a null point.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    >>> data = FieldDataset.from_arrays(
    ...     {
    ...         "B_1": np.ones((8, 8, 8)),
    ...         "B_2": np.zeros((8, 8, 8)),
    ...         "B_3": np.zeros((8, 8, 8)),
    ...     },
    ...     grid,
    ...     Normalization.identity(),
    ... )
    >>> fl = trace_field_line(
    ...     data, (4.0, 4.0, 4.0), step_size=0.5, max_steps=4, direction="forward"
    ... )
    >>> fl.n_points
    5
    >>> bool(fl.points[-1, 0] > fl.points[0, 0])  # advances along +x
    True
    """
    if direction not in _VALID_DIRECTIONS:
        msg = f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        raise ValueError(msg)

    if interpolator is None:
        interpolator = VectorFieldInterpolator.from_dataset(data, field_components)

    seed_arr = np.asarray(seed, dtype=np.float64)
    _validate_seed(seed_arr, interpolator, null_threshold)

    field_name = _field_name_from_components(field_components)
    meta: dict = {"step_size": step_size, "method": "rk4"}  # type: ignore[type-arg]
    args = (step_size, max_steps, null_threshold, terminate)

    match direction:
        case "forward":
            fwd, fwd_r = _trace_single_direction(interpolator, seed_arr, 1.0, *args)
            return _assemble_field_line(
                fwd,
                fwd_r,
                _EMPTY_POINTS,
                TerminationReason.MAX_STEPS,
                seed,
                direction,
                field_name,
                data.normalization,
                meta,
            )
        case "backward":
            bwd, bwd_r = _trace_single_direction(interpolator, seed_arr, -1.0, *args)
            return _assemble_field_line(
                _EMPTY_POINTS,
                TerminationReason.MAX_STEPS,
                bwd,
                bwd_r,
                seed,
                direction,
                field_name,
                data.normalization,
                meta,
            )
        case _:
            fwd, fwd_r = _trace_single_direction(interpolator, seed_arr, 1.0, *args)
            bwd, bwd_r = _trace_single_direction(interpolator, seed_arr, -1.0, *args)
            return _assemble_field_line(
                fwd,
                fwd_r,
                bwd,
                bwd_r,
                seed,
                direction,
                field_name,
                data.normalization,
                meta,
            )


def trace_field_line_adaptive(
    data: FieldDataset,
    seed: Vector3,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    step_size_init: float = 0.5,
    min_step: float = 1e-8,
    max_step: float = 2.0,
    max_steps: int = 10_000,
    direction: str = "both",
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    null_threshold: float = 1e-12,
    terminate: Callable[[FloatArray], bool] | None = None,
    interpolator: VectorFieldInterpolator | None = None,
) -> FieldLine:
    r"""Trace a field line using adaptive Dormand-Prince 5(4).

    Uses embedded error estimation to adapt step size, taking larger
    steps in smooth regions and smaller steps near strong curvature.

    Parameters
    ----------
    data : FieldDataset
        Gridded vector field data.
    seed : Vector3
        Starting point ``(x, y, z)`` for the trace.
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    step_size_init : float
        Initial arc-length step size.
    min_step : float
        Minimum allowed step size.
    max_step : float
        Maximum allowed step size.
    max_steps : int
        Maximum accepted steps per direction.
    direction : str
        ``"forward"``, ``"backward"``, or ``"both"``.
    field_components : tuple[str, str, str]
        Names of the three vector field components.
    null_threshold : float
        Field magnitude below which the point is a null.
    terminate : Callable[[FloatArray], bool] | None
        Optional callback; stops if it returns ``True``.
    interpolator : VectorFieldInterpolator | None
        Pre-built interpolator. Built internally if ``None``.

    Returns
    -------
    FieldLine

    Raises
    ------
    ValueError
        If seed is outside domain or at a null point.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    >>> data = FieldDataset.from_arrays(
    ...     {
    ...         "B_1": np.ones((8, 8, 8)),
    ...         "B_2": np.zeros((8, 8, 8)),
    ...         "B_3": np.zeros((8, 8, 8)),
    ...     },
    ...     grid,
    ...     Normalization.identity(),
    ... )
    >>> fl = trace_field_line_adaptive(
    ...     data, (4.0, 4.0, 4.0), max_steps=4, direction="forward"
    ... )
    >>> fl.metadata["method"]
    'rk45_dopri'
    >>> "max_local_error" in fl.metadata
    True
    """
    if direction not in _VALID_DIRECTIONS:
        msg = f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        raise ValueError(msg)

    if interpolator is None:
        interpolator = VectorFieldInterpolator.from_dataset(data, field_components)

    seed_arr = np.asarray(seed, dtype=np.float64)
    _validate_seed(seed_arr, interpolator, null_threshold)

    field_name = _field_name_from_components(field_components)

    def _adapt(sign: float) -> tuple[FloatArray, TerminationReason, float]:
        return _trace_single_direction_adaptive(
            interpolator,
            seed_arr,
            sign,
            atol,
            rtol,
            step_size_init,
            min_step,
            max_step,
            max_steps,
            null_threshold,
            terminate,
        )

    def _meta(err: float) -> dict:  # type: ignore[type-arg]
        return {
            "method": "rk45_dopri",
            "atol": atol,
            "rtol": rtol,
            "max_local_error": err,
        }

    match direction:
        case "forward":
            fwd, fwd_r, fwd_err = _adapt(1.0)
            return _assemble_field_line(
                fwd,
                fwd_r,
                _EMPTY_POINTS,
                TerminationReason.MAX_STEPS,
                seed,
                direction,
                field_name,
                data.normalization,
                _meta(fwd_err),
            )
        case "backward":
            bwd, bwd_r, bwd_err = _adapt(-1.0)
            return _assemble_field_line(
                _EMPTY_POINTS,
                TerminationReason.MAX_STEPS,
                bwd,
                bwd_r,
                seed,
                direction,
                field_name,
                data.normalization,
                _meta(bwd_err),
            )
        case _:
            fwd, fwd_r, fwd_err = _adapt(1.0)
            bwd, bwd_r, bwd_err = _adapt(-1.0)
            return _assemble_field_line(
                fwd,
                fwd_r,
                bwd,
                bwd_r,
                seed,
                direction,
                field_name,
                data.normalization,
                _meta(max(fwd_err, bwd_err)),
            )


def trace_field_lines_adaptive(
    data: FieldDataset,
    seeds: FloatArray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    step_size_init: float = 0.5,
    min_step: float = 1e-8,
    max_step: float = 2.0,
    max_steps: int = 10_000,
    direction: str = "both",
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    null_threshold: float = 1e-12,
    terminate: Callable[[FloatArray], bool] | None = None,
    interpolator: VectorFieldInterpolator | None = None,
) -> list[FieldLine]:
    r"""Trace ``N`` field lines adaptively in parallel via the batched DP kernel.

    Functionally equivalent to calling :func:`trace_field_line_adaptive`
    ``N`` times in a Python loop, but each Butcher-stage RHS evaluation
    is amortized across all seeds in one
    :class:`VectorFieldInterpolator` dispatch — ~10× faster for
    moderate ``N``, larger speedups for ``N`` in the thousands. Memory
    cost is ``N * (max_steps + 1) * 24`` bytes per direction; pick
    ``max_steps`` accordingly for large seed arrays.

    Per-seed termination state (live mask, step count, reason, max
    local error) is tracked in vectorized NumPy; dead seeds report
    ``valid=False`` to the kernel on subsequent steps and don't
    contaminate the surviving seeds (Butcher contraction is over the
    stage axis, not the seed axis).

    Parameters
    ----------
    data : FieldDataset
        Gridded vector field data.
    seeds : FloatArray
        Starting points, shape ``(N, 3)``.
    atol, rtol, step_size_init, min_step, max_step, max_steps : float
        Adaptive integration controls; semantics identical to
        :func:`trace_field_line_adaptive`. Applied per-seed.
    direction : str
        ``"forward"``, ``"backward"``, or ``"both"``.
    field_components : tuple[str, str, str]
        Names of the three vector field components.
    null_threshold : float
        Field magnitude below which a point is treated as a null.
    terminate : Callable[[FloatArray], bool] | None
        Optional per-seed callback; stops a seed when it returns
        ``True`` on the newly accepted point.
    interpolator : VectorFieldInterpolator | None
        Pre-built interpolator. Built internally if ``None``.

    Returns
    -------
    list[FieldLine]
        One :class:`FieldLine` per input seed, in seed order.

    Raises
    ------
    ValueError
        If ``seeds`` has the wrong shape, if any seed is outside the
        interpolation domain, or if any seed is at a field null. The
        upfront validation matches the single-seed contract — invalid
        seeds fail loudly rather than producing a 1-point trace.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    >>> data = FieldDataset.from_arrays(
    ...     {
    ...         "B_1": np.ones((8, 8, 8)),
    ...         "B_2": np.zeros((8, 8, 8)),
    ...         "B_3": np.zeros((8, 8, 8)),
    ...     },
    ...     grid,
    ...     Normalization.identity(),
    ... )
    >>> seeds = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    >>> lines = trace_field_lines_adaptive(
    ...     data, seeds, max_steps=4, direction="forward"
    ... )
    >>> len(lines)
    2
    >>> all(fl.metadata["method"] == "rk45_dopri" for fl in lines)
    True
    """
    if direction not in _VALID_DIRECTIONS:
        msg = f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        raise ValueError(msg)

    if interpolator is None:
        interpolator = VectorFieldInterpolator.from_dataset(data, field_components)

    seeds_arr = np.asarray(seeds, dtype=np.float64)
    if seeds_arr.ndim != 2 or seeds_arr.shape[1] != 3:
        msg = f"seeds must have shape (N, 3), got {seeds_arr.shape}"
        raise ValueError(msg)
    n_seeds = seeds_arr.shape[0]

    # Upfront per-seed validation: invalid seeds raise here rather than
    # producing 1-point traces that violate FieldLine's N >= 2 invariant.
    # O(N) Python overhead, negligible vs. the actual tracing.
    for i in range(n_seeds):
        _validate_seed(seeds_arr[i], interpolator, null_threshold)

    field_name = _field_name_from_components(field_components)
    args = (
        atol,
        rtol,
        step_size_init,
        min_step,
        max_step,
        max_steps,
        null_threshold,
        terminate,
    )

    def _build(
        fwd_pts: FloatArray,
        fwd_reason: TerminationReason,
        bwd_pts: FloatArray,
        bwd_reason: TerminationReason,
        seed_i: int,
        max_err: float,
    ) -> FieldLine:
        meta: dict = {  # type: ignore[type-arg]
            "method": "rk45_dopri",
            "atol": atol,
            "rtol": rtol,
            "max_local_error": max_err,
        }
        return _assemble_field_line(
            fwd_pts,
            fwd_reason,
            bwd_pts,
            bwd_reason,
            tuple(seeds_arr[seed_i].tolist()),
            direction,
            field_name,
            data.normalization,
            meta,
        )

    field_lines: list[FieldLine] = []

    match direction:
        case "forward":
            buf, n_steps, reasons, errs = _trace_batch_single_direction_adaptive(
                interpolator, seeds_arr, 1.0, *args
            )
            for i in range(n_seeds):
                field_lines.append(
                    _build(
                        buf[i, : int(n_steps[i]) + 1],
                        _REASON_FROM_INT[int(reasons[i])],
                        _EMPTY_POINTS,
                        TerminationReason.MAX_STEPS,
                        i,
                        float(errs[i]),
                    )
                )
        case "backward":
            buf, n_steps, reasons, errs = _trace_batch_single_direction_adaptive(
                interpolator, seeds_arr, -1.0, *args
            )
            for i in range(n_seeds):
                field_lines.append(
                    _build(
                        _EMPTY_POINTS,
                        TerminationReason.MAX_STEPS,
                        buf[i, : int(n_steps[i]) + 1],
                        _REASON_FROM_INT[int(reasons[i])],
                        i,
                        float(errs[i]),
                    )
                )
        case _:
            fwd_buf, fwd_n, fwd_reasons, fwd_errs = (
                _trace_batch_single_direction_adaptive(
                    interpolator, seeds_arr, 1.0, *args
                )
            )
            bwd_buf, bwd_n, bwd_reasons, bwd_errs = (
                _trace_batch_single_direction_adaptive(
                    interpolator, seeds_arr, -1.0, *args
                )
            )
            for i in range(n_seeds):
                field_lines.append(
                    _build(
                        fwd_buf[i, : int(fwd_n[i]) + 1],
                        _REASON_FROM_INT[int(fwd_reasons[i])],
                        bwd_buf[i, : int(bwd_n[i]) + 1],
                        _REASON_FROM_INT[int(bwd_reasons[i])],
                        i,
                        max(float(fwd_errs[i]), float(bwd_errs[i])),
                    )
                )

    return field_lines


def estimate_tracing_error(
    field_line: FieldLine,
    data: FieldDataset,
    *,
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    interpolator: VectorFieldInterpolator | None = None,
) -> float:
    r"""Estimate tracing error via Richardson extrapolation.

    Re-traces at half the original step size and compares endpoints.
    For an order-$p$ scheme, halving the step reduces truncation error
    by $2^p$; a ratio of ~16 confirms RK4's 4th-order convergence.

    Only works with fixed-step field lines (from ``trace_field_line``).
    Adaptive traces store ``max_local_error`` in metadata instead.

    Parameters
    ----------
    field_line : FieldLine
        Previously traced field line (needs ``step_size`` in metadata).
    data : FieldDataset
        Same data used for the original trace.
    field_components : tuple[str, str, str]
        Vector field component names.
    interpolator : VectorFieldInterpolator | None
        Pre-built interpolator. Built internally if ``None``.

    Returns
    -------
    float
        L2 distance between original and refined endpoints.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    >>> data = FieldDataset.from_arrays(
    ...     {
    ...         "B_1": np.ones((8, 8, 8)),
    ...         "B_2": np.zeros((8, 8, 8)),
    ...         "B_3": np.zeros((8, 8, 8)),
    ...     },
    ...     grid,
    ...     Normalization.identity(),
    ... )
    >>> fl = trace_field_line(
    ...     data, (4.0, 4.0, 4.0), step_size=1.0, max_steps=3, direction="forward"
    ... )
    >>> err = estimate_tracing_error(fl, data)
    >>> err < 1e-9  # uniform field, RK4 is exact
    True
    """
    for key in ("step_size", "n_steps"):
        if key not in field_line.metadata:
            msg = (
                f"FieldLine metadata missing required key {key!r}. "
                f"Only fixed-step traces (trace_field_line) support error estimation."
            )
            raise ValueError(msg)

    original_step = field_line.metadata["step_size"]
    n_steps = field_line.metadata["n_steps"]

    refined = trace_field_line(
        data,
        field_line.seed_point,
        step_size=original_step / 2.0,
        max_steps=n_steps * 2,
        direction=field_line.direction,
        field_components=field_components,
        interpolator=interpolator,
    )
    orig_end = np.array(field_line.end_point)
    ref_end = np.array(refined.end_point)
    return float(np.linalg.norm(orig_end - ref_end))


def _validate_seed(
    seed: FloatArray,
    interp: VectorFieldInterpolator,
    null_threshold: float,
) -> None:
    """Raise ValueError if seed is outside domain or at a null."""
    b = interp(seed)
    if np.any(np.isnan(b)):
        msg = f"Seed point {tuple(seed)} is outside the interpolation domain."
        raise ValueError(msg)
    if np.linalg.norm(b) < null_threshold:
        msg = f"Seed point {tuple(seed)} is at a field null (|B| < {null_threshold})."
        raise ValueError(msg)


def _field_name_from_components(
    components: tuple[str, str, str],
) -> str:
    """Infer a field name like ``'B'`` from ``('B_1', 'B_2', 'B_3')``.

    Validates that all components share the same prefix. Strips the
    Tier-3 trailing ``_<digit>`` suffix.
    """
    names = {c.rstrip("0123456789").rstrip("_") for c in components}
    if len(names) != 1:
        msg = f"Components must belong to the same field, got {components}"
        raise ValueError(msg)
    return names.pop()
