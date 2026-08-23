"""Step-size controllers for embedded Runge-Kutta methods.

The current implementation is the **elementary (I) controller** from
[@HairerWanner1993] §II.4: one step's error norm sets the next step.
A true PI controller — which threads in the previous step's error —
is queued behind the ``err_prev`` kwarg for when a consumer needs it
(see the parameter docstring for citations).

References
----------
- Hairer, Nørsett & Wanner [@HairerWanner1993] §II.4 — elementary
  (I) step-size controller, safety factor, growth clamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.types import FloatArray

# Tuning constants shared across embedded-RK orders.
_SAFETY = 0.9  # Bias accepted steps slightly small.
_GROWTH_MIN = 0.2  # Minimum shrink ratio per accept/reject.
_GROWTH_MAX = 5.0  # Maximum growth ratio per accept.
_ERR_FLOOR = 1e-15  # Avoid pow(0, ·) when err is exactly machine zero.


def i_step_controller(
    h: float,
    err_norm: float,
    *,
    min_step: float,
    max_step: float,
    order: int = 5,
    err_prev: float | None = None,  # reserved for future PI upgrade
) -> float:
    r"""Next step size from the current step and its error norm.

    Applies the elementary order-$p$ step-size formula
    $h_{new} = h \cdot S \cdot (\mathrm{err})^{-1/p}$ with safety
    factor $S = 0.9$, then clamps the growth ratio to
    ``[_GROWTH_MIN, _GROWTH_MAX]`` and the absolute step to
    ``[min_step, max_step]``. ``order`` is the order of the embedded
    method's higher-order solution (5 for Dormand-Prince 5(4)).

    ``err_prev`` is reserved for a future PI upgrade and ignored
    today; passing it is harmless.

    Parameters
    ----------
    h : float
        Current step size.
    err_norm : float
        Scaled error norm from
        [`pypic.numerics.embedded_error_norm`][pypic.numerics.embedded_error_norm].
        Values
        ``<= 1`` indicate an acceptable step.
    min_step, max_step : float
        Lower / upper bounds on the returned step magnitude.
    order : int
        Order $p$ of the embedded higher-order solution. Default 5
        (Dormand-Prince 5(4)).
    err_prev : float or None
        Reserved for a future PI controller upgrade
        ([@Gustafsson1988]; [@HairerWanner1993] §IV.2) that would
        thread the previous step's error norm into the formula.
        Currently ignored; passing it is harmless.

    Returns
    -------
    float
        Next step size (always within ``[min_step, max_step]``).

    Examples
    --------
    >>> # err_norm = 1 → step factor ≈ safety = 0.9
    >>> float(round(i_step_controller(1.0, 1.0, min_step=1e-6, max_step=10.0), 6))
    0.9
    >>> # err_norm → 0 → growth clamped to _GROWTH_MAX
    >>> float(i_step_controller(1.0, 0.0, min_step=1e-6, max_step=10.0))
    5.0
    """
    del err_prev  # reserved; not yet used
    exponent = -1.0 / order
    factor = min(
        _GROWTH_MAX,
        max(
            _GROWTH_MIN,
            _SAFETY * max(err_norm, _ERR_FLOOR) ** exponent,
        ),
    )
    return float(np.clip(h * factor, min_step, max_step))


def i_step_controller_batched(
    h: FloatArray,
    err_norm: FloatArray,
    *,
    min_step: float,
    max_step: float,
    order: int = 5,
    err_prev: FloatArray | None = None,
) -> FloatArray:
    r"""Per-seed step-size update for a batched embedded RK integration.

    Vectorized form of `i_step_controller`. Applies the elementary
    order-$p$ formula $h_{new} = h \cdot S \cdot \mathrm{err}^{-1/p}$
    independently per seed, with the same safety factor, growth clamps,
    and absolute-step clamps as the scalar version.

    Parameters
    ----------
    h : NDArray
        Current per-seed step sizes, shape ``(N,)``.
    err_norm : NDArray
        Per-seed scaled error norms from
        [`pypic.numerics.embedded_error_norm_batched`][pypic.numerics.embedded_error_norm_batched],
        shape
        ``(N,)``. Values ``<= 1`` indicate acceptable steps.
    min_step, max_step : float
        Lower / upper bounds on each returned step (shared across seeds;
        per-seed bounds are not in scope for v1).
    order : int
        Order $p$ of the embedded higher-order solution. Default 5
        (Dormand-Prince 5(4)).
    err_prev : NDArray or None
        Reserved for a future per-seed PI controller upgrade
        ([@Gustafsson1988]; [@HairerWanner1993] §IV.2). Currently
        ignored; passing it is harmless.

    Returns
    -------
    FloatArray
        Per-seed next step sizes, shape ``(N,)``, each clamped to
        ``[min_step, max_step]``.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([1.0, 1.0])
    >>> err = np.array([1.0, 0.0])      # one at-tol, one zero-err
    >>> h_new = i_step_controller_batched(h, err, min_step=1e-6, max_step=10.0)
    >>> float(round(h_new[0], 6))
    0.9
    >>> float(h_new[1])
    5.0
    """
    del err_prev  # reserved; not yet used
    err_clamped = np.maximum(err_norm, _ERR_FLOOR)
    factor = _SAFETY * err_clamped ** (-1.0 / order)
    factor = np.clip(factor, _GROWTH_MIN, _GROWTH_MAX)
    return np.clip(h * factor, min_step, max_step)
