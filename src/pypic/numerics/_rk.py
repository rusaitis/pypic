"""Dormand-Prince 5(4) adaptive ODE step kernel.

Pure numerics: state-vector-agnostic, no dependency on field-line
tracing or any other pypic concept. The caller supplies the RHS as a
``Callable[[FloatArray], FloatArray | None]`` and decides what to do
with success / failure outcomes.

The notation ``p(q)`` follows Hairer & Wanner: propagate with order
*p*, embedded estimator order *q*. The 5th-order weights ``_DP_B5``
advance the solution; the 4th-order weights ``_DP_B4`` produce the
embedded error estimate. SciPy's ``RK45`` and MATLAB's ``ode45``
implement the same tableau.

The kernel exploits Dormand-Prince's First-Same-As-Last (FSAL)
property: row 6 of the Butcher matrix equals the 5th-order weights,
so the 7th stage of an accepted step is evaluated at the new
solution and equals the 1st stage of the next step. Callers can
pass ``k0`` from the previous accepted step's ``k_last`` to skip
one RHS evaluation per accepted step.

References
----------
- Dormand-Prince [@DormandPrince1980] — original tableau, embedded
  error estimator, and FSAL property.
- Hairer, Nørsett & Wanner [@HairerWanner1993] §II.4 — textbook
  treatment, step-size control, stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.types import FloatArray


# Dormand-Prince 5(4) Butcher tableau (7 stages, FSAL).
# Rows = stages, columns = weights on previous stages.
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


@dataclass(frozen=True, slots=True)
class DPStepResult:
    """Outcome of one Dormand-Prince 5(4) step.

    On success, ``failed_stage`` is ``None`` and ``y_new`` /
    ``err_vec`` / ``k_last`` are populated. On RHS-callable failure,
    ``failed_stage`` is the stage index (0-6) whose evaluation
    returned ``None`` and ``failed_point`` is the point passed to
    that stage so the caller can classify the failure mode (e.g.
    null hit, out-of-domain) in its own vocabulary.

    Attributes
    ----------
    y_new : FloatArray or None
        5th-order solution at ``t + h``. ``None`` on failure.
    err_vec : FloatArray or None
        Embedded 4(5) error estimate vector (same shape as ``y_new``).
        ``None`` on failure.
    k_last : FloatArray or None
        Last stage value ``f(y_new)`` from the FSAL row. ``None`` on
        failure. Re-pass to the next step via ``k0=`` to skip the
        stage-0 RHS call on accepted steps.
    failed_stage : int or None
        Index of the stage whose RHS evaluation returned ``None``,
        or ``None`` on success.
    failed_point : FloatArray or None
        The point passed to the failed stage (for caller-side
        classification). ``None`` on success.
    """

    y_new: FloatArray | None
    err_vec: FloatArray | None
    k_last: FloatArray | None
    failed_stage: int | None
    failed_point: FloatArray | None


def dormand_prince_step(
    f: Callable[[FloatArray], FloatArray | None],
    y: FloatArray,
    h: float,
    *,
    k0: FloatArray | None = None,
) -> DPStepResult:
    r"""One Dormand-Prince 5(4) step from ``y`` over a step of size ``h``.

    Evaluates the 7 stages of the Dormand-Prince tableau and returns
    both the 5th-order solution and the embedded 4th-order error
    estimate:

    $$y_{n+1} = y_n + h \sum_{i=1}^{7} b_i\, k_i, \qquad
    k_i = f\!\left(y_n + h \sum_{j<i} a_{ij}\, k_j\right)$$

    $$\mathrm{err} = h \sum_{i=1}^{7} (b_i - \hat{b}_i)\, k_i$$

    where $b_i$ are the 5th-order propagation weights (``_DP_B5``)
    and $\hat{b}_i$ are the embedded 4th-order weights (``_DP_B4``).

    The RHS callable ``f`` may return ``None`` to signal that the
    point is invalid (out-of-domain, at a magnetic null, ...); when
    that happens the step is aborted at the failing stage and the
    failure point is reported back for caller-side classification.

    Currently assumes ``y`` is 1-D ``(n,)``. Vectorize via
    ``np.tensordot`` when a batched consumer arrives.

    Parameters
    ----------
    f : callable
        RHS function ``f(y) -> dy/dt`` returning a same-shape array,
        or ``None`` to signal an invalid evaluation point.
    y : NDArray
        Current state, shape ``(n,)``.
    h : float
        Step size (sign-bearing — pass a negative ``h`` to integrate
        backward).
    k0 : NDArray or None
        Pre-computed first-stage value ``f(y)`` from a previous
        accepted step's ``k_last`` (FSAL re-use). When supplied,
        skips the stage-0 RHS evaluation. ``None`` (default)
        evaluates stage 0 normally.

    Returns
    -------
    DPStepResult
        Step outcome. See :class:`DPStepResult`.

    Examples
    --------
    >>> import numpy as np
    >>> result = dormand_prince_step(lambda y: -y, np.array([1.0]), 0.1)
    >>> bool(abs(result.y_new[0] - np.exp(-0.1)) < 1e-9)
    True
    >>> result.failed_stage is None
    True
    """
    n = y.shape[0]
    k = np.empty((7, n), dtype=np.float64)

    if k0 is None:
        rhs0 = f(y)
        if rhs0 is None:
            return DPStepResult(
                y_new=None,
                err_vec=None,
                k_last=None,
                failed_stage=0,
                failed_point=y,
            )
        k[0] = rhs0
    else:
        k[0] = k0

    for i in range(1, 7):
        yi = y + h * np.dot(_DP_A[i, :i], k[:i])
        rhs_val = f(yi)
        if rhs_val is None:
            return DPStepResult(
                y_new=None,
                err_vec=None,
                k_last=None,
                failed_stage=i,
                failed_point=yi,
            )
        k[i] = rhs_val

    y_new = y + h * np.dot(_DP_B5, k)
    err_vec = h * np.dot(_DP_E, k)
    return DPStepResult(
        y_new=y_new,
        err_vec=err_vec,
        k_last=k[6],
        failed_stage=None,
        failed_point=None,
    )


def embedded_error_norm(
    err_vec: FloatArray,
    y_new: FloatArray,
    atol: float,
    rtol: float,
) -> float:
    r"""RMS norm of ``err_vec`` scaled by ``atol + rtol * |y_new|``.

    $$\|\mathrm{err}\|_{\mathrm{RMS}} = \sqrt{\frac{1}{n}\sum_i
    \left(\frac{\mathrm{err}_i}
    {\mathrm{atol} + \mathrm{rtol}\,|y_{\mathrm{new},i}|}\right)^2}$$

    Standard mixed absolute/relative tolerance norm for embedded
    Runge-Kutta error estimators ([@HairerWanner1993] §II.4). A
    returned value $\le 1$ means the step is acceptable under the
    requested tolerances. Matches the convention used by
    ``scipy.integrate.RK45._estimate_error_norm`` so step-size
    sequences remain comparable between the two.

    Parameters
    ----------
    err_vec : NDArray
        Embedded error vector from
        :func:`dormand_prince_step`.
    y_new : NDArray
        Proposed solution at the new time (used for the relative
        tolerance scale).
    atol, rtol : float
        Absolute and relative tolerances.

    Returns
    -------
    float
        Scaled RMS norm of the error. For a single-component state
        the RMS collapses to $|\mathrm{err}| / \mathrm{scale}$.

    Examples
    --------
    >>> import numpy as np
    >>> err = np.array([1e-6, 2e-6])
    >>> y = np.array([1.0, 2.0])
    >>> float(round(embedded_error_norm(err, y, atol=1e-6, rtol=0.0), 6))
    1.581139
    """
    scale = atol + rtol * np.abs(y_new)
    scaled = err_vec / scale
    return float(np.sqrt(np.mean(scaled * scaled)))
