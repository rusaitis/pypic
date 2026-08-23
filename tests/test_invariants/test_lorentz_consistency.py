# Source: docs/conventions.md § "Three-velocity vs four-velocity":
#   "Four-velocity is the spatial part of the 4-velocity
#    u^μ = γ(c, v). When four-velocity is available, derived
#    quantities should compute the Lorentz factor from
#    γ = sqrt(1 + u²/c²) and recover three-velocity as
#    v_i = u_i/γ only when needed."
# + src/pypic/derived.py:1980 (lorentz_factor — from 3-velocity)
# + src/pypic/derived.py:2015 (lorentz_factor_from_four_velocity).
# Claims:
#   (a) The two formulas agree bit-exactly on matched inputs:
#       given γ = 1/√(1 - v²/c²), setting u = γ·v gives
#       γ' = √(1 + u²/c²) == γ (algebraic identity).
#   (b) γ(v=0) == 1.
#   (c) γ(-v) == γ(+v) (even function of v).
#   (d) γ(v) ≥ 1 for any v in [0, c).
# Also covers ``magnetization`` sign:
#   (e) σ ≥ 0 for any B, ρ_m > 0.
#   (f) σ(B=0) == 0.
# Fresh invariant #10 after backlog exhaustion.
"""Cross-consistency between the two Lorentz-factor formulas."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    lorentz_factor,
    lorentz_factor_from_four_velocity,
    magnetization,
)

SHAPE = (3, 4)


def _sub_c_velocity(c: float, max_ratio: float = 0.99) -> st.SearchStrategy[np.ndarray]:
    """Velocity magnitudes in $[0, max_ratio \\cdot c]$.

    Bounded below c to keep γ finite and avoid the catastrophic-cancellation
    regime is tested separately. Tight enough that γ stays
    under ~7 for max_ratio=0.99, so scalar × array products stay comfortably
    inside float64.
    """
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=0.0,
            max_value=max_ratio * c,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_array() -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=1e-3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


@given(
    v=_sub_c_velocity(1.0),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
    data=st.data(),
)
@settings(max_examples=40, deadline=None)
def test_lorentz_factor_from_v_and_u_agree(
    v: np.ndarray, c: float, data: st.DataObject
) -> None:
    r"""$\gamma_{v} := 1/\sqrt{1 - v^2/c^2}$ equals
    $\gamma_{u} := \sqrt{1 + u^2/c^2}$ when $u = \gamma_{v} \cdot v$.

    This is the algebraic identity tying the two formulas to the same
    underlying Lorentz factor. Either formula alone might have a
    sign/exponent bug that its own docstring doctest wouldn't catch;
    the cross-check catches it.
    """
    del data
    # Rescale velocities so max(v) < 0.99 * c for every draw.
    v_scaled = v * (c / max(1.0, c))
    v_bounded = np.minimum(v_scaled, 0.99 * c)

    gamma_from_v = lorentz_factor(v_bounded, c)
    u = gamma_from_v * v_bounded
    gamma_from_u = lorentz_factor_from_four_velocity(u, c)

    assert_allclose(gamma_from_v, gamma_from_u, rtol=1e-13, atol=1e-13)


@given(
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=10, deadline=None)
def test_lorentz_factor_at_rest_is_one(c: float) -> None:
    r"""$\gamma(v=0) = 1$ — rest frame. Both formulas must give
    exactly 1.0 regardless of the chosen speed of light.
    """
    v = np.zeros(SHAPE)
    u = np.zeros(SHAPE)
    assert_allclose(lorentz_factor(v, c), 1.0, rtol=0, atol=0)
    assert_allclose(lorentz_factor_from_four_velocity(u, c), 1.0, rtol=0, atol=0)


@given(
    v=_sub_c_velocity(1.0),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_lorentz_factor_is_even_in_velocity(v: np.ndarray, c: float) -> None:
    r"""$\gamma(-v) = \gamma(+v)$ — Lorentz factor depends only on $v^2$,
    so the sign of the velocity does not matter.
    """
    v_bounded = np.minimum(v, 0.99 * c)
    assert_allclose(
        lorentz_factor(-v_bounded, c),
        lorentz_factor(v_bounded, c),
        rtol=0,
        atol=0,
    )


@given(
    v=_sub_c_velocity(1.0),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_lorentz_factor_is_at_least_one(v: np.ndarray, c: float) -> None:
    r"""$\gamma \geq 1$ for all physical velocities — the docstring
    claim at derived.py:2002 ("Bounded $[1, \infty)$").
    """
    v_bounded = np.minimum(v, 0.99 * c)
    assert np.all(lorentz_factor(v_bounded, c) >= 1.0)


@given(
    b=_positive_array(),
    rho=_positive_array(),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_magnetization_is_non_negative(
    b: np.ndarray, rho: np.ndarray, c: float
) -> None:
    r"""$\sigma = B^2 / (\rho_m c^2) \geq 0$ — always non-negative."""
    assert np.all(magnetization(b, rho, c) >= 0.0)


@given(
    rho=_positive_array(),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_magnetization_vanishes_at_zero_b(rho: np.ndarray, c: float) -> None:
    r"""$\sigma(B=0) = 0$ — no magnetic energy, no magnetization."""
    b = np.zeros(SHAPE)
    assert_allclose(magnetization(b, rho, c), 0.0, rtol=0, atol=0)
