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

    from pypic.types import BoolArray, FloatArray, IntArray


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

    Operates on a 1-D state ``y`` of shape ``(n,)``. See
    :func:`dormand_prince_step_batched` for the vectorized form over
    a leading seed axis.

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
    requested tolerances. RMS (rather than the alternative max-norm)
    aligns with SciPy's ``RK45._estimate_error_norm`` and the
    Hairer-Nørsett-Wanner convention, so step-size sequences here
    are directly comparable to other Python and Fortran ODE
    integrators that follow the same recipe.

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


@dataclass(frozen=True, slots=True)
class DPStepResultBatched:
    """Outcome of one batched Dormand-Prince 5(4) step over N seeds.

    Same FSAL/embedded-error contract as :class:`DPStepResult`, lifted
    over an N-seed batch. ``y_new`` / ``err_vec`` / ``k_last`` are
    always populated as ``(N, n)`` arrays; per-seed validity is read
    from ``failed_stage`` (``-1`` = success, ``0..6`` = first stage at
    which the RHS reported invalid).

    Attributes
    ----------
    y_new : FloatArray
        5th-order solution at ``t + h``, shape ``(N, n)``. Slots where
        ``failed_stage != -1`` carry undefined values — the caller must
        filter by ``failed_stage`` before consuming.
    err_vec : FloatArray
        Embedded 4(5) error estimate, shape ``(N, n)``. Same caveat as
        ``y_new`` for failed seeds.
    k_last : FloatArray
        Last-stage value ``f(y_new)`` for FSAL re-use, shape ``(N, n)``.
        Re-pass via ``k0=`` on the next call to skip stage-0 evaluation
        for the surviving seeds.
    failed_stage : FloatArray
        Per-seed first-failed-stage index, shape ``(N,)`` int. ``-1``
        means the step succeeded for that seed; ``0..6`` indexes the
        Butcher row whose RHS evaluation returned ``valid=False``.
    failed_point : FloatArray
        Per-seed evaluation point at which the failure occurred, shape
        ``(N, n)``. NaN where ``failed_stage == -1``.
    """

    y_new: FloatArray
    err_vec: FloatArray
    k_last: FloatArray
    failed_stage: IntArray
    failed_point: FloatArray


def dormand_prince_step_batched(
    f: Callable[[FloatArray], tuple[FloatArray, BoolArray]],
    y: FloatArray,
    h: FloatArray | float,
    *,
    k0: FloatArray | None = None,
) -> DPStepResultBatched:
    r"""Batched Dormand-Prince 5(4) step over N independent seeds.

    Vectorized form of :func:`dormand_prince_step` — each of the seven
    Butcher stages becomes a single ``f`` call on the whole ``(N, n)``
    batch instead of N calls on individual ``(n,)`` vectors. The
    arithmetic for each seed is identical to the single-step kernel;
    the only contract difference is the RHS callable.

    The RHS ``f(y_batch)`` must return ``(rhs_values, valid_mask)``
    where ``rhs_values`` is the ``(N, n)`` RHS array and ``valid_mask``
    is an ``(N,)`` boolean array (``True`` = valid seed). Invalid
    seeds at any stage are recorded in the per-seed ``failed_stage``
    array of the returned result; the kernel still evaluates the
    remaining stages for the surviving seeds (their ``y_new`` and
    ``err_vec`` are unaffected by the failed seeds because the
    Butcher contraction is stage-axis only).

    Currently assumes ``y`` is 2-D ``(N, n)``. Field-line tracing
    uses ``n = 3``.

    Parameters
    ----------
    f : callable
        Batched RHS ``f(y) -> (rhs, valid_mask)``. ``rhs`` is
        ``(N, n)``; ``valid_mask`` is ``(N,)`` bool.
    y : NDArray
        Current state, shape ``(N, n)``.
    h : NDArray or float
        Step size. Scalar (shared across seeds) or shape ``(N,)``
        (per-seed). Sign-bearing — negative ``h`` integrates backward.
    k0 : NDArray or None
        Pre-computed first-stage value from a previous accepted step's
        ``k_last`` (FSAL re-use), shape ``(N, n)``. When supplied,
        skips the stage-0 RHS evaluation for the whole batch.

    Returns
    -------
    DPStepResultBatched

    Examples
    --------
    >>> import numpy as np
    >>> def rhs(y):
    ...     return -y, np.ones(y.shape[0], dtype=bool)
    >>> y0 = np.array([[1.0], [2.0]])
    >>> r = dormand_prince_step_batched(rhs, y0, 0.1)
    >>> bool(np.allclose(r.y_new[:, 0], y0[:, 0] * np.exp(-0.1), atol=1e-9))
    True
    >>> int(r.failed_stage.max())
    -1
    """
    n_seeds, n_dim = y.shape
    k = np.empty((7, n_seeds, n_dim), dtype=np.float64)
    failed_stage = np.full(n_seeds, -1, dtype=np.intp)
    failed_point = np.full((n_seeds, n_dim), np.nan, dtype=np.float64)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(n_seeds, float(h_arr), dtype=np.float64)
    elif h_arr.shape != (n_seeds,):
        msg = f"h must be scalar or shape ({n_seeds},), got {h_arr.shape}"
        raise ValueError(msg)
    h_col = h_arr[:, None]

    if k0 is None:
        rhs0, valid0 = f(y)
        k[0] = rhs0
        bad0 = ~valid0
        if bad0.any():
            failed_stage[bad0] = 0
            failed_point[bad0] = y[bad0]
    else:
        k[0] = k0

    for i in range(1, 7):
        yi = y + h_col * np.tensordot(_DP_A[i, :i], k[:i], axes=([0], [0]))
        rhs_i, valid_i = f(yi)
        k[i] = rhs_i
        newly_failed = (failed_stage < 0) & ~valid_i
        if newly_failed.any():
            failed_stage[newly_failed] = i
            failed_point[newly_failed] = yi[newly_failed]

    y_new = y + h_col * np.tensordot(_DP_B5, k, axes=([0], [0]))
    err_vec = h_col * np.tensordot(_DP_E, k, axes=([0], [0]))

    return DPStepResultBatched(
        y_new=y_new,
        err_vec=err_vec,
        k_last=k[6],
        failed_stage=failed_stage,
        failed_point=failed_point,
    )


def embedded_error_norm_batched(
    err_vec: FloatArray,
    y_new: FloatArray,
    atol: float,
    rtol: float,
) -> FloatArray:
    r"""Per-seed RMS error norm for a batched embedded RK step.

    Vectorized form of :func:`embedded_error_norm`. Same mixed
    absolute/relative scaling and RMS reduction as the scalar form
    ([@HairerWanner1993] §II.4), averaged over the *component* axis
    (``axis=-1``) so each seed gets its own scalar error norm.

    Parameters
    ----------
    err_vec : NDArray
        Embedded error vectors, shape ``(N, n)``.
    y_new : NDArray
        Proposed solutions at the new time, shape ``(N, n)``.
    atol, rtol : float
        Absolute and relative tolerances (shared across seeds; per-seed
        tolerances are not in scope for v1).

    Returns
    -------
    FloatArray
        Per-seed scaled RMS error norms, shape ``(N,)``. Values ``<= 1``
        mark acceptable steps under the requested tolerances.

    Examples
    --------
    >>> import numpy as np
    >>> err = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
    >>> y = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> norms = embedded_error_norm_batched(err, y, atol=1e-6, rtol=0.0)
    >>> norms.shape
    (2,)
    """
    scale = atol + rtol * np.abs(y_new)
    scaled = err_vec / scale
    return np.sqrt(np.mean(scaled * scaled, axis=-1))  # type: ignore[no-any-return]
