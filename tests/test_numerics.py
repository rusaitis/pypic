"""Unit tests for :mod:`pypic.numerics` — Dormand-Prince kernel + controller.

These tests pin the kernel's numerical behavior independently of any
particular consumer (field-line tracing, particle pushers, etc.) so a
future kernel-side change can't silently break the contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pypic.numerics import (
    DPStepResult,
    DPStepResultBatched,
    dormand_prince_step,
    dormand_prince_step_batched,
    embedded_error_norm,
    embedded_error_norm_batched,
    i_step_controller,
    i_step_controller_batched,
)
from pypic.numerics._step_control import _GROWTH_MAX, _GROWTH_MIN


def _integrate_fixed(f, y0: np.ndarray, t_end: float, h: float) -> np.ndarray:
    """Fixed-step DP integration over ``[0, t_end]`` with step ``h``.

    Direction follows ``sign(h) == sign(t_end)`` — caller passes a
    negative ``t_end`` and matching negative ``h`` to integrate backward.
    """
    y = y0.copy()
    t = 0.0
    sign = 1.0 if h > 0 else -1.0
    while sign * (t_end - t) > 1e-15:
        remaining = t_end - t
        step = h if abs(h) < abs(remaining) else remaining
        result = dormand_prince_step(f, y, step)
        assert result.failed_stage is None
        assert result.y_new is not None
        y = result.y_new
        t += step
    return y


class TestDormandPrinceStep:
    def test_exponential_decay_matches_analytic(self) -> None:
        """dy/dt = -y from y(0)=1 reaches exp(-t) within 5th-order error."""
        h = 0.01
        y_final = _integrate_fixed(lambda y: -y, np.array([1.0]), 1.0, h)
        np.testing.assert_allclose(y_final[0], math.exp(-1.0), atol=1e-10)

    def test_exponential_growth_matches_analytic(self) -> None:
        """dy/dt = +y from y(0)=1 reaches exp(t)."""
        h = 0.01
        y_final = _integrate_fixed(lambda y: y, np.array([1.0]), 1.0, h)
        np.testing.assert_allclose(y_final[0], math.exp(1.0), atol=1e-9)

    def test_harmonic_oscillator_returns_to_origin(self) -> None:
        """dy/dt = [y[1], -y[0]] is a rotation; one period returns to identity."""
        h = 0.001
        y0 = np.array([1.0, 0.0])
        y_final = _integrate_fixed(
            lambda y: np.array([y[1], -y[0]]), y0, 2.0 * math.pi, h
        )
        np.testing.assert_allclose(y_final, y0, atol=1e-10)

    def test_richardson_convergence_is_fifth_order(self) -> None:
        """Halving h reduces global error by ~2^5 = 32 (5th-order solution)."""
        # Stiff-ish oscillator: y'' + 4 y = 0, period π. Phase mismatch
        # at t = 1 makes a clean convergence test.
        rhs = lambda y: np.array([y[1], -4.0 * y[0]])  # noqa: E731
        y0 = np.array([1.0, 0.0])
        true_final = np.array([math.cos(2.0), -2.0 * math.sin(2.0)])

        err_coarse = np.linalg.norm(_integrate_fixed(rhs, y0, 1.0, 0.02) - true_final)
        err_fine = np.linalg.norm(_integrate_fixed(rhs, y0, 1.0, 0.01) - true_final)
        ratio = err_coarse / err_fine
        # 5th-order ⇒ ratio ≈ 32. Allow wide band so different floating-point
        # rounding doesn't flake; we mainly want to rule out 4th order (~16)
        # and 6th order (~64).
        assert 20.0 < ratio < 60.0, f"convergence ratio {ratio:.2f} not 5th-order"

    def test_failure_path_populates_diagnostic_fields(self) -> None:
        """When f returns None at a stage, failed_stage/failed_point are set."""
        calls = [0]

        def f(y: np.ndarray) -> np.ndarray | None:
            calls[0] += 1
            # Fail on the 3rd stage call (stage index 2)
            if calls[0] == 3:
                return None
            return -y

        result = dormand_prince_step(f, np.array([1.0]), 0.1)
        assert result.failed_stage == 2
        assert result.failed_point is not None
        assert result.y_new is None
        assert result.err_vec is None

    def test_success_path_populates_solution_fields(self) -> None:
        """Successful step leaves diagnostic fields None and solution fields set."""
        result = dormand_prince_step(lambda y: -y, np.array([1.0]), 0.1)
        assert result.failed_stage is None
        assert result.failed_point is None
        assert result.y_new is not None
        assert result.err_vec is not None
        assert result.k_last is not None
        assert result.k_last.shape == (1,)

    def test_k_last_equals_rhs_at_y_new(self) -> None:
        """FSAL property: k_last on success equals f(y_new)."""
        f = lambda y: -y  # noqa: E731
        result = dormand_prince_step(f, np.array([1.0]), 0.1)
        assert result.y_new is not None
        assert result.k_last is not None
        np.testing.assert_allclose(result.k_last, f(result.y_new))

    def test_fsal_k0_skips_stage_zero_eval(self) -> None:
        """Passing k0 skips the stage-0 RHS call: 6 evals per step instead of 7."""
        calls = [0]

        def f(y: np.ndarray) -> np.ndarray:
            calls[0] += 1
            return -y

        # Warm-up step to obtain k_last.
        warmup = dormand_prince_step(f, np.array([1.0]), 0.1)
        assert warmup.k_last is not None
        warmup_calls = calls[0]
        assert warmup_calls == 7  # cold start: all 7 stages evaluated

        # Reuse k_last as k0 — should only do 6 RHS calls this step.
        calls[0] = 0
        reused = dormand_prince_step(f, warmup.y_new, 0.1, k0=warmup.k_last)
        assert reused.failed_stage is None
        assert calls[0] == 6

    def test_fsal_k0_produces_same_solution(self) -> None:
        """FSAL re-use must not change the numerical result."""
        f = lambda y: -y  # noqa: E731
        warmup = dormand_prince_step(f, np.array([1.0]), 0.1)
        assert warmup.y_new is not None
        assert warmup.k_last is not None

        without_fsal = dormand_prince_step(f, warmup.y_new, 0.1)
        with_fsal = dormand_prince_step(f, warmup.y_new, 0.1, k0=warmup.k_last)
        assert without_fsal.y_new is not None
        assert with_fsal.y_new is not None
        np.testing.assert_allclose(with_fsal.y_new, without_fsal.y_new)

    def test_backward_integration_matches_analytic(self) -> None:
        """Backward integration of dy/dt=-y recovers y(0)=1 from y(1)=exp(-1).

        Pins the sign-bearing-``h`` claim in the docstring at 5th-order
        precision: a negative step size must integrate backward with the
        same precision as forward integration, not just "in the right
        direction".
        """
        y0 = np.array([math.exp(-1.0)])
        y_final = _integrate_fixed(lambda v: -v, y0, -1.0, -0.01)
        np.testing.assert_allclose(y_final[0], 1.0, atol=1e-9)


def _adaptive_solve(
    f,  # type: ignore[no-untyped-def]
    y0: np.ndarray,
    t_end: float,
    *,
    atol: float,
    rtol: float,
) -> np.ndarray:
    """Minimal adaptive driver wiring DP + error norm + I controller.

    Exists so the test below can exercise the three numerics primitives
    together at the ``pypic.numerics`` layer — the field-line tracer
    has its own end-to-end coverage but conflates kernel × interpolator
    × tracer-specific bookkeeping.
    """
    y = y0.copy()
    t = 0.0
    h = 0.1
    k0: np.ndarray | None = None
    while t < t_end - 1e-15:
        step_h = min(h, t_end - t)
        result = dormand_prince_step(f, y, step_h, k0=k0)
        assert result.failed_stage is None
        assert result.y_new is not None
        assert result.err_vec is not None
        err = embedded_error_norm(result.err_vec, result.y_new, atol, rtol)
        if err <= 1.0:
            y = result.y_new
            t += step_h
            k0 = result.k_last  # FSAL reuse after accept
        else:
            k0 = None  # FSAL invalid after reject; recompute stage 0
        h = i_step_controller(h, err, min_step=1e-8, max_step=1.0)
    return y


class TestAdaptiveIntegration:
    """End-to-end at the numerics layer: DP × error norm × I controller."""

    def test_tighter_tolerance_yields_smaller_error(self) -> None:
        """Halving atol/rtol drives a strictly smaller global error.

        Catches regressions in the controller × kernel interaction
        (safety factor, growth bound, FSAL re-use across rejects) that
        the isolated kernel and controller tests don't see.
        """
        rhs = lambda y: np.array([y[1], -4.0 * y[0]])  # noqa: E731
        y0 = np.array([1.0, 0.0])
        true_final = np.array([math.cos(2.0), -2.0 * math.sin(2.0)])

        err_loose = np.linalg.norm(
            _adaptive_solve(rhs, y0, 1.0, atol=1e-6, rtol=1e-6) - true_final
        )
        err_tight = np.linalg.norm(
            _adaptive_solve(rhs, y0, 1.0, atol=1e-9, rtol=1e-9) - true_final
        )
        # The tight run must achieve strictly smaller error than the loose run.
        # We don't assert a precise ratio — the controller's safety factor and
        # growth clamps make the per-tolerance error coupling looser than
        # err ∝ tol — only the ordering, which is the controller's contract.
        assert err_tight < err_loose
        # Both runs must clear their respective tolerances by a healthy margin.
        assert err_loose < 1e-4
        assert err_tight < 1e-7


class TestEmbeddedErrorNorm:
    def test_atol_dominant_branch(self) -> None:
        """When y is small, atol scales the norm."""
        err = np.array([1e-6])
        y = np.array([0.0])
        assert embedded_error_norm(err, y, atol=1e-6, rtol=1.0) == pytest.approx(1.0)

    def test_rtol_dominant_branch(self) -> None:
        """When |y| is large, rtol scales the norm."""
        err = np.array([1.0])
        y = np.array([100.0])
        # scale = 0 + 0.01 * 100 = 1, err / scale = 1
        assert embedded_error_norm(err, y, atol=0.0, rtol=0.01) == pytest.approx(1.0)

    def test_rms_norm_averages_over_components(self) -> None:
        """Multi-dim err uses RMS = sqrt(mean(scaled^2)), matching SciPy."""
        err = np.array([0.1, 1.0, 0.01])
        y = np.array([1.0, 1.0, 1.0])
        result = embedded_error_norm(err, y, atol=1.0, rtol=0.0)
        # sqrt((0.01 + 1.0 + 0.0001) / 3) ≈ 0.58026
        assert result == pytest.approx(np.sqrt(1.0101 / 3.0))


class TestIStepController:
    def test_unit_error_returns_safety_factor(self) -> None:
        """err_norm = 1 → factor = 0.9 (safety bias)."""
        h_new = i_step_controller(1.0, 1.0, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(0.9)

    def test_zero_error_clamps_to_max_growth(self) -> None:
        """err_norm → 0 → factor clamps to _GROWTH_MAX."""
        h_new = i_step_controller(1.0, 0.0, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(_GROWTH_MAX)

    def test_large_error_clamps_to_min_growth(self) -> None:
        """err_norm >> 1 → factor clamps to _GROWTH_MIN, not below."""
        h_new = i_step_controller(1.0, 1e6, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(_GROWTH_MIN)

    def test_absolute_clamp_min_step(self) -> None:
        """Returned h ≥ min_step regardless of growth ratio."""
        h_new = i_step_controller(1e-3, 1e6, min_step=0.5, max_step=10.0)
        assert h_new == pytest.approx(0.5)

    def test_absolute_clamp_max_step(self) -> None:
        """Returned h ≤ max_step regardless of growth ratio."""
        h_new = i_step_controller(2.0, 0.0, min_step=1e-6, max_step=3.0)
        assert h_new == pytest.approx(3.0)

    def test_order_parameter_changes_exponent(self) -> None:
        """order=p in the controller raises err to power -1/p."""
        # err = 0.5 (under tolerance): order 5 grows more than order 8
        # because 0.5**(-1/5) > 0.5**(-1/8). Same h, atol/rtol, just
        # different order claim.
        h_order5 = i_step_controller(1.0, 0.5, min_step=1e-6, max_step=10.0, order=5)
        h_order8 = i_step_controller(1.0, 0.5, min_step=1e-6, max_step=10.0, order=8)
        assert h_order5 > h_order8

    def test_err_prev_kwarg_accepted_and_ignored(self) -> None:
        """Reserved err_prev kwarg is accepted but does not change behavior."""
        h_without = i_step_controller(1.0, 0.5, min_step=1e-6, max_step=10.0)
        h_with = i_step_controller(1.0, 0.5, min_step=1e-6, max_step=10.0, err_prev=0.1)
        assert h_with == pytest.approx(h_without)

    def test_h_new_monotone_non_increasing_in_err(self) -> None:
        """``h_new`` never grows when ``err_norm`` grows.

        Three isolated points are already pinned (err=1 → safety, err→0
        and err→∞ → clamps). This sweep covers the unclamped middle and
        the transitions into both clamps so a future controller change
        that breaks the order is caught.
        """
        # Spans both clamps (err≈0 hits _GROWTH_MAX; err=1e6 hits _GROWTH_MIN)
        # with several unclamped samples around err=1.
        err_sweep = [0.0, 1e-3, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 100.0, 1e6]
        prev_h: float | None = None
        for err in err_sweep:
            h_new = i_step_controller(1.0, err, min_step=1e-8, max_step=100.0)
            if prev_h is not None:
                assert h_new <= prev_h + 1e-12, (
                    f"non-monotone at err={err}: {h_new} > prev {prev_h}"
                )
            prev_h = h_new


class TestDPStepResult:
    def test_is_frozen(self) -> None:
        """DPStepResult is immutable (frozen=True, slots=True)."""
        result = DPStepResult(
            y_new=None,
            err_vec=None,
            k_last=None,
            failed_stage=0,
            failed_point=np.array([0.0]),
        )
        with pytest.raises(AttributeError):
            result.failed_stage = 1  # type: ignore[misc]


def _all_valid_rhs(f_scalar):  # type: ignore[no-untyped-def]
    """Wrap a scalar-shape RHS so it answers the batched contract."""

    def rhs_batched(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return f_scalar(y), np.ones(y.shape[0], dtype=bool)

    return rhs_batched


class TestDormandPrinceStepBatched:
    def test_single_seed_matches_scalar_kernel(self) -> None:
        """For N=1, batched and scalar kernels agree bit-for-bit."""
        f_scalar = lambda y: -y  # noqa: E731
        y0 = np.array([[1.5]])
        scalar = dormand_prince_step(lambda y: -y, np.array([1.5]), 0.1)
        batched = dormand_prince_step_batched(_all_valid_rhs(f_scalar), y0, 0.1)
        assert scalar.y_new is not None
        assert scalar.err_vec is not None
        assert scalar.k_last is not None
        np.testing.assert_allclose(batched.y_new[0], scalar.y_new)
        np.testing.assert_allclose(batched.err_vec[0], scalar.err_vec)
        np.testing.assert_allclose(batched.k_last[0], scalar.k_last)
        assert int(batched.failed_stage[0]) == -1

    def test_n_seeds_match_per_seed_scalar_loop(self) -> None:
        """Batched advance of N seeds equals N independent scalar calls."""
        f_scalar = lambda y: -y  # noqa: E731
        y0 = np.array([[0.5], [1.0], [2.0], [-0.7]])
        h = 0.05

        scalar_results = [
            dormand_prince_step(f_scalar, y0[i], h) for i in range(y0.shape[0])
        ]
        batched = dormand_prince_step_batched(_all_valid_rhs(f_scalar), y0, h)

        for i, sc in enumerate(scalar_results):
            assert sc.y_new is not None
            np.testing.assert_allclose(batched.y_new[i], sc.y_new, atol=1e-15)

    def test_per_seed_h_advances_each_independently(self) -> None:
        """Per-seed h: each seed advances by its own step size."""
        f_scalar = lambda y: -y  # noqa: E731
        y0 = np.array([[1.0], [1.0]])
        h = np.array([0.05, 0.20])  # second seed steps 4× as far
        batched = dormand_prince_step_batched(_all_valid_rhs(f_scalar), y0, h)
        # Both seeds start at 1.0 and decay; second seed should be smaller.
        assert batched.y_new[1, 0] < batched.y_new[0, 0]
        np.testing.assert_allclose(batched.y_new[0, 0], np.exp(-0.05), atol=1e-10)
        np.testing.assert_allclose(batched.y_new[1, 0], np.exp(-0.20), atol=1e-10)

    def test_failed_seed_isolated_from_good_seeds(self) -> None:
        """One seed's stage failure does not contaminate good seeds' y_new."""

        def rhs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Seed 0 valid for all calls; seed 1 invalid (NaN-poisoned).
            values = -y
            valid = np.array([True, False])
            return values, valid

        y0 = np.array([[1.0], [1.0]])
        result = dormand_prince_step_batched(rhs, y0, 0.1)
        assert int(result.failed_stage[0]) == -1
        # Seed 1 fails at stage 0 (the very first RHS call).
        assert int(result.failed_stage[1]) == 0
        # Good seed should be untouched by the bad seed's failure.
        scalar = dormand_prince_step(lambda y: -y, np.array([1.0]), 0.1)
        assert scalar.y_new is not None
        np.testing.assert_allclose(result.y_new[0], scalar.y_new)

    def test_failed_point_captures_invalid_evaluation_point(self) -> None:
        """failed_point[bad] is the point passed to the failed stage."""

        def rhs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Seed 0 invalid at stage 0 → failed_point should equal y0.
            valid = np.array([False, True])
            return -y, valid

        y0 = np.array([[3.14, 2.71], [0.0, 0.0]])
        result = dormand_prince_step_batched(rhs, y0, 0.1)
        np.testing.assert_allclose(result.failed_point[0], y0[0])
        # Good seed's failed_point stays NaN.
        assert np.all(np.isnan(result.failed_point[1]))

    def test_fsal_k0_skips_stage_zero_eval(self) -> None:
        """Passing k0 skips the stage-0 RHS call across the whole batch."""
        calls = [0]
        f_scalar = lambda y: -y  # noqa: E731

        def rhs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            calls[0] += 1
            return -y, np.ones(y.shape[0], dtype=bool)

        y0 = np.array([[1.0], [2.0], [3.0]])
        # Cold start: 7 RHS calls.
        warmup = dormand_prince_step_batched(rhs, y0, 0.1)
        assert calls[0] == 7

        # Re-use k_last as k0 → 6 RHS calls on the next step.
        calls[0] = 0
        reused = dormand_prince_step_batched(rhs, warmup.y_new, 0.1, k0=warmup.k_last)
        assert calls[0] == 6
        # Result must match a fresh call (no FSAL).
        without_fsal = dormand_prince_step_batched(
            _all_valid_rhs(f_scalar), warmup.y_new, 0.1
        )
        np.testing.assert_allclose(reused.y_new, without_fsal.y_new)

    def test_h_shape_mismatch_raises(self) -> None:
        """h with a shape other than scalar / (N,) raises."""
        f_scalar = lambda y: -y  # noqa: E731
        y0 = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match=r"h must be scalar or shape"):
            dormand_prince_step_batched(_all_valid_rhs(f_scalar), y0, np.array([0.1]))

    def test_result_is_frozen(self) -> None:
        """DPStepResultBatched is immutable (frozen=True, slots=True)."""
        zero = np.zeros((1, 1))
        result = DPStepResultBatched(
            y_new=zero,
            err_vec=zero,
            k_last=zero,
            failed_stage=np.zeros(1, dtype=np.intp),
            failed_point=zero,
        )
        with pytest.raises(AttributeError):
            result.failed_stage = np.ones(1, dtype=np.intp)  # type: ignore[misc]


class TestEmbeddedErrorNormBatched:
    def test_collapses_to_scalar_kernel_for_n1(self) -> None:
        """For N=1, the batched RMS norm equals the scalar version."""
        err = np.array([[0.1, 0.2, 0.3]])
        y = np.array([[1.0, 1.0, 1.0]])
        batched = embedded_error_norm_batched(err, y, atol=1.0, rtol=0.0)
        scalar = embedded_error_norm(err[0], y[0], atol=1.0, rtol=0.0)
        assert batched.shape == (1,)
        assert batched[0] == pytest.approx(scalar)

    def test_per_seed_independence(self) -> None:
        """Each seed's RMS norm depends only on that seed's err and y."""
        err = np.array([[0.1, 0.2], [1.0, 2.0]])
        y = np.array([[1.0, 1.0], [1.0, 1.0]])
        norms = embedded_error_norm_batched(err, y, atol=1.0, rtol=0.0)
        # Seed 0: sqrt(mean([0.01, 0.04])) = sqrt(0.025) ≈ 0.158
        # Seed 1: sqrt(mean([1.0,  4.0])) = sqrt(2.5)   ≈ 1.581
        assert norms[0] == pytest.approx(np.sqrt(0.025))
        assert norms[1] == pytest.approx(np.sqrt(2.5))

    def test_single_component_collapses_to_absolute_scaled_error(self) -> None:
        """For n=1, RMS reduces to |err|/scale per seed (no averaging).

        Pins the docstring claim — covered for the scalar form already,
        but the batched path's mean-over-component reduction needs its
        own check.
        """
        err = np.array([[1e-6], [-3e-6]])
        y = np.array([[1.0], [2.0]])
        norms = embedded_error_norm_batched(err, y, atol=1e-6, rtol=0.0)
        # scale = atol = 1e-6 (rtol=0); per-seed result is |err|/atol.
        expected = np.array([1.0, 3.0])
        np.testing.assert_allclose(norms, expected)


class TestIStepControllerBatched:
    def test_per_seed_independence(self) -> None:
        """Each seed gets its own h_new based on its own err."""
        h = np.array([1.0, 1.0, 1.0])
        err = np.array([1.0, 0.0, 1e6])  # at-tol / zero / blown
        h_new = i_step_controller_batched(h, err, min_step=1e-6, max_step=10.0)
        assert h_new[0] == pytest.approx(0.9)  # safety factor
        assert h_new[1] == pytest.approx(_GROWTH_MAX)  # growth clamp
        assert h_new[2] == pytest.approx(_GROWTH_MIN)  # shrink clamp

    def test_absolute_clamps_apply_per_seed(self) -> None:
        """min_step / max_step clamps apply elementwise."""
        h = np.array([1e-3, 2.0])
        err = np.array([1e6, 0.0])  # first blown, second zero
        h_new = i_step_controller_batched(h, err, min_step=0.5, max_step=3.0)
        assert h_new[0] == pytest.approx(0.5)  # blown step pinned to min
        assert h_new[1] == pytest.approx(3.0)  # zero-err pinned to max

    def test_collapses_to_scalar_controller(self) -> None:
        """For N=1, batched controller agrees with scalar version."""
        h = np.array([1.0])
        err = np.array([0.5])
        batched = i_step_controller_batched(h, err, min_step=1e-6, max_step=10.0)
        scalar = i_step_controller(1.0, 0.5, min_step=1e-6, max_step=10.0)
        assert float(batched[0]) == pytest.approx(scalar)

    def test_all_zero_err_does_not_trigger_pow_zero(self) -> None:
        """An all-zero err vector clamps to _GROWTH_MAX, no NaN/Inf leakage.

        The scalar controller's zero-error path is covered by
        ``TestIStepController.test_zero_error_clamps_to_max_growth``;
        the batched path goes through ``np.maximum(err, _ERR_FLOOR)``
        as a single vectorized op and needs its own dedicated check
        against a regression that drops the floor.
        """
        h = np.array([1.0, 1.0, 1.0])
        err = np.zeros(3)
        h_new = i_step_controller_batched(h, err, min_step=1e-6, max_step=100.0)
        np.testing.assert_allclose(h_new, _GROWTH_MAX)
        assert np.all(np.isfinite(h_new))
