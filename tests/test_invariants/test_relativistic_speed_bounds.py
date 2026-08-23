# Source: docs/equations.md § 8.1 (relativistic v_A, v_ms, v_th bounded
#         by c) + docs/conventions.md § "Relativistic speed composition"
#         ("ensures v_ms < c always, unlike the non-relativistic
#         v_ms^2 = v_A^2 + c_s^2 which can exceed c when sigma is large")
#         + src/pypic/derived.py:1000-1002 docstring "guarantees
#         v_ms < c" + src/pypic/derived.py:1026 comment
#         "You cannae change the laws of physics - v_ms < c, always"
#         + src/pypic/derived.py:659-660 "caps the result at c"
#         + src/pypic/derived.py:166-169 "approaches c as sigma -> inf".
# Claims:
#   (a) alfven_speed(b, rho_m, c=c)    <= c for b >= 0, rho_m > 0.
#   (b) magnetosonic_speed(v_A, c_s, c=c) <= c for v_A, c_s in [0, c].
#   (c) magnetosonic_speed(v_A, c_s, c=c) >= max(v_A, c_s) for v_A, c_s
#       in [0, c] (fast-mode ordering - v_ms is the *fast* mode, so it
#       must exceed both constituent speeds).
#   (d) thermal_speed(T, m, c=c)       <= c for T >= 0, m > 0.
#   (e) Non-relativistic v_ms can exceed c (demonstrates why the
#       relativistic composition exists) - guards against a future
#       refactor silently capping the non-rel branch too.
# Fresh invariant #12 after backlog exhaustion.
"""Relativistic speed formulas respect the c-cap at any finite c."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypic.derived import alfven_speed, magnetosonic_speed, thermal_speed

SHAPE = (3, 4)


def _nonneg_array(
    min_value: float = 0.0, max_value: float = 1e6
) -> st.SearchStrategy[np.ndarray]:
    """Finite non-negative arrays. Spans the full magnetization range:
    with b up to 1e6 and rho_m down to 1e-3, sigma = b^2/(rho_m c^2)
    can reach 1e15 - well inside the regime where sigma/(1+sigma)
    rounds to 1 in float64, so the tests accept equality at c (not just
    strict <)."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_array(
    min_value: float = 1e-3, max_value: float = 1e6
) -> st.SearchStrategy[np.ndarray]:
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


def _speed_of_light() -> st.SearchStrategy[float]:
    """c in [0.5, 100] - normalized-units range used in the sibling
    relativistic invariant tests (test_lorentz_consistency,
    test_relativistic_limit)."""
    return st.floats(min_value=0.5, max_value=100.0, allow_nan=False)


def _subc_speed(c: float) -> st.SearchStrategy[np.ndarray]:
    """Speeds in [0, c] - valid inputs for the relativistic magnetosonic
    composition. v_A or c_s > c violates the formula's domain."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=0.0,
            max_value=c,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@given(b=_nonneg_array(), rho_m=_positive_array(), c=_speed_of_light())
@settings(max_examples=50, deadline=None)
def test_relativistic_alfven_speed_bounded_by_c(
    b: np.ndarray, rho_m: np.ndarray, c: float
) -> None:
    r"""$v_A^{rel}(b, \rho_m, c) \leq c$ — equations.md § 8.1.

    $v_A^{rel} = c \sqrt{\sigma / (1 + \sigma)}$ with
    $\sigma = b^2 / (\rho_m c^2) \geq 0$. Since $\sigma/(1+\sigma) \in
    [0, 1)$ for finite $\sigma$, $v_A^{rel} < c$ strictly; equality
    arises only when $\sigma$ is so large that float64 rounds
    $\sigma/(1+\sigma)$ to 1. Accept $\leq$ to cover both regimes.
    """
    v_a = alfven_speed(b, rho_m, c=c)
    assert np.all(v_a <= c)
    assert np.all(v_a >= 0.0)


@given(c=_speed_of_light(), data=st.data())
@settings(max_examples=40, deadline=None)
def test_relativistic_magnetosonic_speed_bounded_by_c(
    c: float, data: st.DataObject
) -> None:
    r"""$v_{ms}^{rel}(v_A, c_s, c) \leq c$ for $v_A, c_s \in [0, c]$.

    derived.py:1002 docstring: "guarantees $v_{ms} < c$".
    conventions.md § Relativistic speed composition: "ensures
    $v_{ms} < c$ always, unlike the non-relativistic form which can
    exceed $c$ when $\sigma$ is large".
    """
    v_a = data.draw(_subc_speed(c))
    c_s = data.draw(_subc_speed(c))
    v_ms = magnetosonic_speed(v_a, c_s, c=c)
    # At v_A = c or c_s = c, v_ms² algebraically equals c²; sqrt(c*c)
    # in float64 has ±1 ulp roundoff (IEEE 754), so v_ms - c can be
    # one ulp positive. Allow 4·eps·c to cover the worst case with
    # margin. The algebraic bound holds in infinite precision.
    atol = 4.0 * np.finfo(np.float64).eps * c
    assert np.all(v_ms <= c + atol)
    assert np.all(v_ms >= 0.0)


@given(c=_speed_of_light(), data=st.data())
@settings(max_examples=40, deadline=None)
def test_magnetosonic_speed_exceeds_both_components(
    c: float, data: st.DataObject
) -> None:
    r"""$v_{ms}^{rel} \geq \max(v_A, c_s)$ — the fast mode propagates
    at least as fast as either wave alone.

    From $v_{ms}^2 = v_A^2 + c_s^2 (1 - v_A^2/c^2)$:
    $v_{ms}^2 - v_A^2 = c_s^2 (1 - v_A^2/c^2) \geq 0$ for $v_A \leq c$,
    and symmetrically $v_{ms}^2 - c_s^2 = v_A^2 (1 - c_s^2/c^2) \geq 0$
    for $c_s \leq c$. Both inequalities are algebraic - any violation
    is a sign or exponent bug in the composition formula.
    """
    v_a = data.draw(_subc_speed(c))
    c_s = data.draw(_subc_speed(c))
    v_ms = magnetosonic_speed(v_a, c_s, c=c)
    # Tolerance at float64 roundoff of the compositional subtraction.
    atol = 1e-13 * c
    assert np.all(v_ms + atol >= v_a)
    assert np.all(v_ms + atol >= c_s)


@given(
    temperature=_nonneg_array(min_value=0.0, max_value=1e12),
    mass=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
    c=_speed_of_light(),
)
@settings(max_examples=40, deadline=None)
def test_relativistic_thermal_speed_bounded_by_c(
    temperature: np.ndarray, mass: float, c: float
) -> None:
    r"""$v_{th}^{rel} = v_{th} / \sqrt{1 + v_{th}^2/c^2} \leq c$ —
    derived.py:659-660 "caps the result at $c$".

    Exercised deliberately in the ultra-relativistic regime ($T/m$
    allowed up to $10^{12}$, giving $v_{th} \gg c$) where the cap is
    load-bearing: without it the thermal speed would exceed $c$ by
    many orders of magnitude.

    Tolerance: $v_{th}/\sqrt{1 + v_{th}^2/c^2}$ reduces to
    $c \cdot x/\sqrt{1+x^2}$ with $x = v_{th}/c$. At $x \gg 1$ the
    $+1$ underflows, giving $\sqrt{x^2} = x$ and the division
    $x/x$ in float64 rounds to the bit-exact $c$ — but with a
    possible $\pm$ 1-ulp overshoot from IEEE 754 sqrt (same
    property behind the $v_{ms}$ atol). Absorb 4·eps·c.
    """
    v_th = thermal_speed(temperature, mass, c=c)
    atol = 4.0 * np.finfo(np.float64).eps * c
    assert np.all(v_th <= c + atol)
    assert np.all(v_th >= 0.0)


def test_nonrelativistic_magnetosonic_can_exceed_c() -> None:
    r"""Guard: the *non-relativistic* branch $v_{ms}^2 = v_A^2 + c_s^2$
    is NOT capped — verified by constructing a case where it exceeds
    an external "c" reference.

    Documented in conventions.md: "the non-relativistic
    $v_{ms}^2 = v_A^2 + c_s^2$ which can exceed $c$ when $\sigma$ is
    large". If a future refactor silently caps the non-rel branch too,
    this test fails and forces a conscious decision: is the rewrite
    intended, or is it a regression?
    """
    v_a = np.array([0.9])
    c_s = np.array([0.9])
    c_ref = 1.0
    v_ms_nonrel = magnetosonic_speed(v_a, c_s)  # no c kwarg
    assert float(v_ms_nonrel[0]) > c_ref
    # And the relativistic branch correctly caps below c.
    v_ms_rel = magnetosonic_speed(v_a, c_s, c=c_ref)
    assert float(v_ms_rel[0]) < c_ref
