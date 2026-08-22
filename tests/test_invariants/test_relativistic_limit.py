# Source: docs/equations.md § 8.1 ("recovers (1/2) ρ_m v² for v ≪ c" for
#         e_k; "recovers B/√ρ_m for σ ≪ 1" for v_A; "practical cap at c"
#         for v_th; and the composition of the relativistic v_ms reducing
#         to v_A² + c_s² when v_A, c_s ≪ c) + CLAUDE.md "Relativistic via
#         c=None kwarg" ("When None, the non-relativistic formula is used.
#         When provided, the relativistic branch activates").
# Claim: for each derived function with ``c: float | None``, the
#        ``c=C`` branch agrees with ``c=None`` within O(1/C²) as C grows,
#        when the inputs stay in the non-relativistic regime.
# Excluded: ``enthalpy`` — its relativistic branch adds the rest-mass
#        ``c²`` term by design (equations.md § 2), so the two branches
#        diverge as C² rather than converge.
"""Non-relativistic limit of every derived function that accepts ``c``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypic.derived import (
    alfven_speed,
    kinetic_energy_density,
    magnetosonic_speed,
    sound_speed,
    thermal_speed,
)

SHAPE = (4, 3)
C_LARGE = 1e6


def _positive_array(
    min_value: float = 0.1, max_value: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    """Finite strictly-positive arrays in [0.1, 10] — B, ρ_m, P, T all
    live in R⁺. The narrow range keeps every derived speed ≲ 10 so that
    (v/C_LARGE)² ≲ 1e-10 stays comfortably above float64 roundoff but
    still safely inside the non-relativistic regime equations.md § 8.1
    promises the limit on.
    """
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


def _bounded_velocity(upper: float = 10.0) -> st.SearchStrategy[np.ndarray]:
    """Velocities well below ``C_LARGE`` (ratio ≤ 1e-5). Keeps the inputs
    in the regime where ``equations.md § 8.1`` promises the limit.
    """
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=0.1,
            max_value=upper,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


@given(b=_positive_array(), rho_m=_positive_array())
@settings(max_examples=40, deadline=None)
def test_alfven_speed_limit(b: np.ndarray, rho_m: np.ndarray) -> None:
    r"""$v_A^{rel}(c \to \infty) \to B/\sqrt{\rho_m}$ — equations.md § 8.1."""
    rel = alfven_speed(b, rho_m, c=C_LARGE)
    nonrel = alfven_speed(b, rho_m)
    # Leading-order deviation is O(v_A² / C²). With v_A ≤ 1e4 and C=1e6,
    # |rel - nonrel| / |nonrel| ≲ 1e-4 — but in practice the float64
    # truncation is tighter. atol comfortably bounds both.
    np.testing.assert_allclose(rel, nonrel, rtol=1e-6, atol=1e-10)


@given(p=_positive_array(), rho_m=_positive_array())
@settings(max_examples=40, deadline=None)
def test_sound_speed_limit(p: np.ndarray, rho_m: np.ndarray) -> None:
    r"""$c_s^{rel}(c \to \infty) \to \sqrt{\gamma P / \rho_m}$."""
    rel = sound_speed(p, rho_m, c=C_LARGE)
    nonrel = sound_speed(p, rho_m)
    np.testing.assert_allclose(rel, nonrel, rtol=1e-6, atol=1e-10)


@given(v_a=_positive_array(), c_s=_positive_array())
@settings(max_examples=40, deadline=None)
def test_magnetosonic_speed_limit(v_a: np.ndarray, c_s: np.ndarray) -> None:
    r"""$v_{ms}^{rel}(c \to \infty) \to \sqrt{v_A^2 + c_s^2}$.

    The relativistic composition subtracts $v_A^2 c_s^2 / c^2$, which
    vanishes to machine precision at this C.
    """
    rel = magnetosonic_speed(v_a, c_s, c=C_LARGE)
    nonrel = magnetosonic_speed(v_a, c_s)
    np.testing.assert_allclose(rel, nonrel, rtol=1e-6, atol=1e-10)


@given(
    temperature=_positive_array(),
    mass=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=40, deadline=None)
def test_thermal_speed_limit(temperature: np.ndarray, mass: float) -> None:
    r"""$v_{th}^{rel}(c \to \infty) \to \sqrt{T/m}$."""
    rel = thermal_speed(temperature, mass, c=C_LARGE)
    nonrel = thermal_speed(temperature, mass)
    np.testing.assert_allclose(rel, nonrel, rtol=1e-6, atol=1e-10)


@given(rho_m=_positive_array(), v=_bounded_velocity())
@settings(max_examples=40, deadline=None)
def test_kinetic_energy_density_limit(rho_m: np.ndarray, v: np.ndarray) -> None:
    r"""$(\gamma - 1)\rho_m c^2 \to \tfrac{1}{2}\rho_m v^2$ as $v/c \to 0$.

    equations.md § 8.1 column "Notes": "recovers (1/2) ρ_m v² for v ≪ c".

    The naive form ``(γ - 1) ρ_m c²`` catastrophically cancels once
    ``v/c < sqrt(eps) ≈ 1.5e-8`` — for c=1e6 and v=1 this is well within
    the problem regime. The algebraically equivalent form
    ``γ² v² ρ_m / (γ + 1)`` avoids the subtraction and recovers the
    non-relativistic limit cleanly at any v/c.
    """
    rel = kinetic_energy_density(rho_m, v, c=C_LARGE)
    nonrel = kinetic_energy_density(rho_m, v)
    np.testing.assert_allclose(rel, nonrel, rtol=1e-6, atol=1e-10)


def test_enthalpy_excluded_from_limit_is_ill_posed() -> None:
    """Documentation guard: ``enthalpy(..., c=C)`` adds the rest-mass
    ``c²`` term by physical convention (equations.md § 2 "h_rel = c² +
    γP/((γ-1)ρ_m)"), so it does NOT converge to the non-relativistic
    enthalpy as ``c → ∞``. The c→∞ limit claim applies only to the
    five functions exercised above. If this test ever fails (i.e.
    ``enthalpy`` starts converging), the property suite should be
    extended; today's behavior is by design.
    """
    from pypic.derived import enthalpy

    p = np.array([1.0])
    rho = np.array([1.0])
    rel = enthalpy(p, rho, c=C_LARGE)
    nonrel = enthalpy(p, rho)
    # h_rel - h_nonrel == c² bit-exact (Synge-type convention).
    assert rel[0] == pytest.approx(nonrel[0] + C_LARGE**2, rel=0.0, abs=1e-6)
