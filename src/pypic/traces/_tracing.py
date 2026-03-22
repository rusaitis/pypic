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

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.readers.base import FieldDataset
    from pypic.traces._fieldline import FieldLine
    from pypic.types import FloatArray, Vector3


class TerminationReason(StrEnum):
    """Why a field line trace stopped."""

    MAX_STEPS = "max_steps"
    DOMAIN_EXIT = "domain_exit"
    NULL_POINT = "null_point"
    CALLBACK = "callback"


@dataclass(frozen=True, slots=True)
class VectorFieldInterpolator:
    r"""Pre-built trilinear interpolator for a 3-component vector field.

    Wraps three ``RegularGridInterpolator`` instances (one per
    component), built once and reused across all RK4 stages and
    seed points.
    """

    _interps: tuple[
        RegularGridInterpolator,
        RegularGridInterpolator,
        RegularGridInterpolator,
    ]

    @classmethod
    def from_dataset(
        cls,
        data: FieldDataset,
        components: tuple[str, str, str] = ("B1", "B2", "B3"),
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
        interps = tuple(
            RegularGridInterpolator(
                coord_arrays,
                np.asarray(data[c], dtype=np.float64),
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            for c in components
        )
        return cls(_interps=interps)  # type: ignore[arg-type]

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
        pt = point.reshape(1, 3)
        return np.array(
            [float(interp(pt)[0]) for interp in self._interps],
        )


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
) -> tuple[list[FloatArray], TerminationReason]:
    """Fixed-step classical RK4 integration in one direction."""
    points: list[FloatArray] = [seed.copy()]
    reason = TerminationReason.MAX_STEPS
    h = step_size

    for _ in range(max_steps):
        y = points[-1]

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

        y_new = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        points.append(y_new)

        if terminate is not None and terminate(y_new):
            reason = TerminationReason.CALLBACK
            break

    return points, reason


# Dormand-Prince RK4(5) Butcher tableau
_DP_A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ],
    dtype=np.float64,
)
_DP_B5 = np.array(
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
)
_DP_B4 = np.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
)
_DP_E = _DP_B5 - _DP_B4


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
) -> tuple[list[FloatArray], TerminationReason, float]:
    """Dormand-Prince RK4(5) adaptive integration in one direction."""
    points: list[FloatArray] = [seed.copy()]
    reason = TerminationReason.MAX_STEPS
    h = step_size_init
    max_local_error = 0.0
    steps_taken = 0

    while steps_taken < max_steps:
        y = points[-1]

        k = np.empty((7, 3))
        failed = False
        for i in range(7):
            yi = y + h * np.dot(_DP_A[i, :i], k[:i]) if i > 0 else y
            rhs_val = _rhs(yi, interp, sign, null_threshold)
            if rhs_val is None:
                reason = _classify_failure(yi, interp)
                failed = True
                break
            k[i] = rhs_val

        if failed:
            break

        y5 = y + h * np.dot(_DP_B5, k)
        err_vec = h * np.dot(_DP_E, k)
        scale = atol + rtol * np.abs(y5)
        err_norm = float(np.max(np.abs(err_vec) / scale))
        max_local_error = max(max_local_error, err_norm)

        if err_norm == 0.0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
        h_new = float(np.clip(h * factor, min_step, max_step))

        if err_norm <= 1.0 or h <= min_step:
            points.append(y5)
            steps_taken += 1
            h = h_new
            if terminate is not None and terminate(y5):
                reason = TerminationReason.CALLBACK
                break
        else:
            h = h_new

    return points, reason, max_local_error


def _assemble_field_line(
    fwd_points: list[FloatArray],
    fwd_reason: TerminationReason,
    bwd_points: list[FloatArray],
    bwd_reason: TerminationReason,
    seed: Vector3,
    direction: str,
    field_name: str,
    normalization: object,
    metadata: dict,  # type: ignore[type-arg]
) -> FieldLine:
    """Concatenate forward/backward traces into a FieldLine."""
    from pypic.traces._fieldline import FieldLine as _FieldLine

    if direction == "forward":
        all_points = np.array(fwd_points)
        reason = fwd_reason
    elif direction == "backward":
        all_points = np.array(bwd_points[::-1])
        reason = bwd_reason
    else:
        bwd_rev = list(reversed(bwd_points))
        if len(bwd_rev) > 0 and len(fwd_points) > 0:
            combined = bwd_rev[:-1] + fwd_points
        elif len(bwd_rev) > 0:
            combined = bwd_rev
        else:
            combined = fwd_points
        all_points = np.array(combined)
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


_VALID_DIRECTIONS = frozenset({"forward", "backward", "both"})


def trace_field_line(
    data: FieldDataset,
    seed: Vector3,
    *,
    step_size: float = 0.5,
    max_steps: int = 10_000,
    direction: str = "both",
    field_components: tuple[str, str, str] = ("B1", "B2", "B3"),
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
    """
    if direction not in _VALID_DIRECTIONS:
        msg = f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        raise ValueError(msg)

    if interpolator is None:
        interpolator = VectorFieldInterpolator.from_dataset(
            data,
            field_components,
        )

    seed_arr = np.asarray(seed, dtype=np.float64)
    _validate_seed(seed_arr, interpolator, null_threshold)

    field_name = _field_name_from_components(field_components)
    meta: dict = {"step_size": step_size, "method": "rk4"}  # type: ignore[type-arg]

    args = (step_size, max_steps, null_threshold, terminate)
    if direction == "forward":
        fwd, fwd_r = _trace_single_direction(
            interpolator,
            seed_arr,
            1.0,
            *args,
        )
        return _assemble_field_line(
            fwd,
            fwd_r,
            [],
            TerminationReason.MAX_STEPS,
            seed,
            direction,
            field_name,
            data.normalization,
            meta,
        )
    if direction == "backward":
        bwd, bwd_r = _trace_single_direction(
            interpolator,
            seed_arr,
            -1.0,
            *args,
        )
        return _assemble_field_line(
            [],
            TerminationReason.MAX_STEPS,
            bwd,
            bwd_r,
            seed,
            direction,
            field_name,
            data.normalization,
            meta,
        )
    # "both"
    fwd, fwd_r = _trace_single_direction(
        interpolator,
        seed_arr,
        1.0,
        *args,
    )
    bwd, bwd_r = _trace_single_direction(
        interpolator,
        seed_arr,
        -1.0,
        *args,
    )
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
    field_components: tuple[str, str, str] = ("B1", "B2", "B3"),
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
    """
    if direction not in _VALID_DIRECTIONS:
        msg = f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        raise ValueError(msg)

    if interpolator is None:
        interpolator = VectorFieldInterpolator.from_dataset(
            data,
            field_components,
        )

    seed_arr = np.asarray(seed, dtype=np.float64)
    _validate_seed(seed_arr, interpolator, null_threshold)

    field_name = _field_name_from_components(field_components)

    def _adapt(
        sign: float,
    ) -> tuple[list[FloatArray], TerminationReason, float]:
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

    if direction == "forward":
        fwd, fwd_r, fwd_err = _adapt(1.0)
        return _assemble_field_line(
            fwd,
            fwd_r,
            [],
            TerminationReason.MAX_STEPS,
            seed,
            direction,
            field_name,
            data.normalization,
            _meta(fwd_err),
        )
    if direction == "backward":
        bwd, bwd_r, bwd_err = _adapt(-1.0)
        return _assemble_field_line(
            [],
            TerminationReason.MAX_STEPS,
            bwd,
            bwd_r,
            seed,
            direction,
            field_name,
            data.normalization,
            _meta(bwd_err),
        )
    # "both"
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
    field_components: tuple[str, str, str] = ("B1", "B2", "B3"),
) -> float:
    r"""Estimate tracing error via Richardson extrapolation.

    Re-traces at half the original step size and compares endpoints.

    Parameters
    ----------
    field_line : FieldLine
        Previously traced field line (needs ``step_size`` in metadata).
    data : FieldDataset
        Same data used for the original trace.
    field_components : tuple[str, str, str]
        Vector field component names.

    Returns
    -------
    float
        L2 distance between original and refined endpoints.
    """
    original_step = field_line.metadata["step_size"]
    n_steps = field_line.metadata["n_steps"]

    refined = trace_field_line(
        data,
        field_line.seed_point,
        step_size=original_step / 2.0,
        max_steps=n_steps * 2,
        direction=field_line.direction,
        field_components=field_components,
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
    """Infer a field name like 'B' from ('B1', 'B2', 'B3')."""
    return components[0].rstrip("0123456789")
