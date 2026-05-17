"""Step-size controllers for embedded Runge-Kutta methods.

The current implementation is the **elementary (I) controller** from
[@HairerWanner1993] §II.4: one step's error norm sets the next step. A
true PI controller ([@Gustafsson1988]; [@HairerWanner1993] §IV.2) —
which threads in the previous step's error — lands in this module when
a consumer needs it.

References
----------
- Hairer, Nørsett & Wanner [@HairerWanner1993] §II.4 — elementary
  (I) step-size controller, safety factor, growth clamps.
- Gustafsson [@Gustafsson1988] — PI controller (reserved for future
  upgrade via the ``err_prev`` kwarg).
"""

from __future__ import annotations

import numpy as np

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
        :func:`pypic.numerics.embedded_error_norm`. Values
        ``<= 1`` indicate an acceptable step.
    min_step, max_step : float
        Lower / upper bounds on the returned step magnitude.
    order : int
        Order $p$ of the embedded higher-order solution. Default 5
        (Dormand-Prince 5(4)).
    err_prev : float or None
        Reserved for the PI controller. Currently ignored.

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
