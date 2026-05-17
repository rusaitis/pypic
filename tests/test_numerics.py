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
    dormand_prince_step,
    embedded_error_norm,
    pi_step_controller,
)
from pypic.numerics._step_control import _DP_GROWTH_MAX, _DP_GROWTH_MIN


def _integrate_fixed(
    f, y0: np.ndarray, t_end: float, h: float
) -> np.ndarray:
    """Fixed-step DP integration; returns final state."""
    y = y0.copy()
    t = 0.0
    while t < t_end - 1e-15:
        step = min(h, t_end - t)
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
        assert result.k.shape == (7, 1)

    def test_negative_step_integrates_backward(self) -> None:
        """h < 0 integrates dy/dt = -y backward: from y(1)=exp(-1) → y(0)=1."""
        y = np.array([math.exp(-1.0)])
        result = dormand_prince_step(lambda v: -v, y, -1.0)
        # Coarse single-step backward integration won't be precise, but the
        # direction must be correct (y grows when integrating backward
        # through exponential decay).
        assert result.y_new is not None
        assert result.y_new[0] > y[0]


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

    def test_infinity_norm_picks_max(self) -> None:
        """Multi-dimensional err uses max, not sum."""
        err = np.array([0.1, 1.0, 0.01])
        y = np.array([1.0, 1.0, 1.0])
        result = embedded_error_norm(err, y, atol=1.0, rtol=0.0)
        assert result == pytest.approx(1.0)


class TestPIStepController:
    def test_unit_error_returns_safety_factor(self) -> None:
        """err_norm = 1 → factor = 0.9 (safety bias)."""
        h_new = pi_step_controller(1.0, 1.0, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(0.9)

    def test_zero_error_clamps_to_max_growth(self) -> None:
        """err_norm → 0 → factor clamps to _DP_GROWTH_MAX."""
        h_new = pi_step_controller(1.0, 0.0, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(_DP_GROWTH_MAX)

    def test_large_error_clamps_to_min_growth(self) -> None:
        """err_norm >> 1 → factor clamps to _DP_GROWTH_MIN, not below."""
        h_new = pi_step_controller(1.0, 1e6, min_step=1e-6, max_step=10.0)
        assert h_new == pytest.approx(_DP_GROWTH_MIN)

    def test_absolute_clamp_min_step(self) -> None:
        """Returned h ≥ min_step regardless of growth ratio."""
        h_new = pi_step_controller(1e-3, 1e6, min_step=0.5, max_step=10.0)
        assert h_new == pytest.approx(0.5)

    def test_absolute_clamp_max_step(self) -> None:
        """Returned h ≤ max_step regardless of growth ratio."""
        h_new = pi_step_controller(2.0, 0.0, min_step=1e-6, max_step=3.0)
        assert h_new == pytest.approx(3.0)


class TestDPStepResult:
    def test_is_frozen(self) -> None:
        """DPStepResult is immutable (frozen=True, slots=True)."""
        result = DPStepResult(
            y_new=None,
            err_vec=None,
            k=np.zeros((7, 1)),
            failed_stage=0,
            failed_point=np.array([0.0]),
        )
        with pytest.raises(AttributeError):
            result.failed_stage = 1  # type: ignore[misc]
