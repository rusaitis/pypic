"""RK4 field line tracer with fixed-step and adaptive variants."""

from __future__ import annotations

__all__ = [
    "TerminationReason",
    "VectorFieldInterpolator",
    "estimate_tracing_error",
    "trace_field_line",
    "trace_field_line_adaptive",
]

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pypic.numerics import (
    dormand_prince_step,
    embedded_error_norm,
    pi_step_controller,
)
from pypic.traces._fieldline import _VALID_DIRECTIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.dataset import FieldDataset
    from pypic.traces._fieldline import FieldLine
    from pypic.types import FloatArray, Vector3


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
    """Dormand-Prince RK4(5) adaptive integration in one direction.

    Bookkeeping (buffer, signed direction, null/out-of-domain
    classification, termination callback) lives here; the pure
    Dormand-Prince step + error estimator + PI controller live in
    :mod:`pypic.numerics`.
    """
    buf = np.empty((max_steps + 1, 3), dtype=np.float64)
    buf[0] = seed
    n = 0
    reason = TerminationReason.MAX_STEPS
    h = step_size_init
    max_local_error = 0.0

    def rhs(yi: FloatArray) -> FloatArray | None:
        return _rhs(yi, interp, sign, null_threshold)

    while n < max_steps:
        result = dormand_prince_step(rhs, buf[n], h)
        if result.failed_stage is not None:
            assert result.failed_point is not None  # invariant of DPStepResult
            reason = _classify_failure(result.failed_point, interp)
            break

        assert result.y_new is not None  # success path
        assert result.err_vec is not None
        err_norm = embedded_error_norm(result.err_vec, result.y_new, atol, rtol)
        max_local_error = max(max_local_error, err_norm)
        h_new = pi_step_controller(
            h, err_norm, min_step=min_step, max_step=max_step
        )

        if err_norm <= 1.0 or h <= min_step:
            n += 1
            buf[n] = result.y_new
            h = h_new
            if terminate is not None and terminate(result.y_new):
                reason = TerminationReason.CALLBACK
                break
        else:
            h = h_new

    return buf[: n + 1], reason, max_local_error


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
    r"""Trace a field line using adaptive Dormand-Prince RK4(5).

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
