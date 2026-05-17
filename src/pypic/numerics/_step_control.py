"""PI step-size controller for embedded Runge-Kutta methods.

The standard order-controlled step adaptation from Hairer & Wanner
"Solving ODEs I" §II.4. The current implementation is scalar PI
suitable for the embedded RK4(5) pair; if future kernels need full
PI/PID with memory of prior errors, this module is where it lands.
"""

from __future__ import annotations

import numpy as np

# Standard tuning constants for the order-5 embedded method.
_DP_SAFETY = 0.9  # Bias accepted steps slightly small.
_DP_GROWTH_MIN = 0.2  # Minimum shrink ratio per accept/reject.
_DP_GROWTH_MAX = 5.0  # Maximum growth ratio per accept.
_DP_ERR_FLOOR = 1e-15  # Avoid pow(0, ·) when err is exactly machine zero.
_DP_EXPONENT = -0.2  # -1 / p with p = 5 (5th-order embedded method).


def pi_step_controller(
    h: float,
    err_norm: float,
    *,
    min_step: float,
    max_step: float,
) -> float:
    r"""Next step size from current step and last error norm.

    Applies the standard order-5 step-size formula
    $h_{new} = h \cdot S \cdot (\mathrm{err})^{-1/p}$ with safety
    factor $S = 0.9$ and order $p = 5$, then clamps the growth ratio
    to ``[_DP_GROWTH_MIN, _DP_GROWTH_MAX]`` and the absolute step to
    ``[min_step, max_step]``.

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

    Returns
    -------
    float
        Next step size (always within ``[min_step, max_step]``).

    Examples
    --------
    >>> # err_norm = 1 → step factor ≈ safety = 0.9
    >>> float(round(pi_step_controller(1.0, 1.0, min_step=1e-6, max_step=10.0), 6))
    0.9
    >>> # err_norm → 0 → growth clamped to _DP_GROWTH_MAX
    >>> float(pi_step_controller(1.0, 0.0, min_step=1e-6, max_step=10.0))
    5.0
    """
    factor = min(
        _DP_GROWTH_MAX,
        max(
            _DP_GROWTH_MIN,
            _DP_SAFETY * max(err_norm, _DP_ERR_FLOOR) ** _DP_EXPONENT,
        ),
    )
    return float(np.clip(h * factor, min_step, max_step))
