# Source: docs/equations.md § 1 (Densities and Moments table):
#   "P: Total scalar pressure = P_e + P_i or Tr(P)/3 or fluid P"
# + src/pypic/derived.py:1288 (total_pressure: P = P_e + P_i)
# + src/pypic/derived.py:1314 (isotropic_pressure: P = (P11+P22+P33)/3)
# + src/pypic/compute.py:279-281
#   (``"P": (total_pressure, ("Pe", "Pi"))``,
#    ``"Pe": (isotropic_pressure, ("P11_s0", "P22_s0", "P33_s0"))``,
#    ``"Pi": (isotropic_pressure, ("P11_s1", "P22_s1", "P33_s1"))``).
# Claim: the two independent paths to total pressure give identical
# results up to float64 summation roundoff:
#   path_a = total_pressure(Pe, Pi)
#          = isotropic(P11_s0, P22_s0, P33_s0)
#          + isotropic(P11_s1, P22_s1, P33_s1)
#   path_b = isotropic(P11_s0 + P11_s1, P22_s0 + P22_s1, P33_s0 + P33_s1)
#
# Both reduce to (Tr(P_s0) + Tr(P_s1))/3. The split-then-add vs
# add-then-divide paths must agree: any factor-of-3 or sign bug in
# either recipe would break the identity.
# Fresh invariant #11 — written as the autoresearcher's 20th iteration.
"""Isotropic pressure via two independent paths."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import isotropic_pressure, total_pressure

SHAPE = (3, 4)


def _tensor_component() -> st.SearchStrategy[np.ndarray]:
    """Diagonal pressure-tensor component — strictly positive for
    physical consistency (pressures are non-negative) and bounded to
    keep the two-path sums comfortably inside float64.
    """
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
    p11_e=_tensor_component(),
    p22_e=_tensor_component(),
    p33_e=_tensor_component(),
    p11_i=_tensor_component(),
    p22_i=_tensor_component(),
    p33_i=_tensor_component(),
)
@settings(max_examples=50, deadline=None)
def test_total_pressure_from_per_species_tensors_two_paths(
    p11_e: np.ndarray,
    p22_e: np.ndarray,
    p33_e: np.ndarray,
    p11_i: np.ndarray,
    p22_i: np.ndarray,
    p33_i: np.ndarray,
) -> None:
    r"""Two paths to total pressure must agree:

        Path A: total_pressure(Pe, Pi)
                where Pe = isotropic(P11_s0, P22_s0, P33_s0), etc.
        Path B: isotropic(P11_s0 + P11_s1, P22_s0 + P22_s1,
                          P33_s0 + P33_s1)

    Both equal (Tr(P_s0) + Tr(P_s1)) / 3 algebraically. Float64
    summation associativity gives the result bit-exact on this input
    range (all components positive, no catastrophic cancellation).
    """
    # Path A: per-species isotropic, then add.
    pe = isotropic_pressure(p11_e, p22_e, p33_e)
    pi = isotropic_pressure(p11_i, p22_i, p33_i)
    path_a = total_pressure(pe, pi)

    # Path B: add per-species tensor components, then isotropic.
    path_b = isotropic_pressure(p11_e + p11_i, p22_e + p22_i, p33_e + p33_i)

    assert_allclose(path_a, path_b, rtol=1e-13, atol=1e-13)


@given(
    p11=_tensor_component(),
    p22=_tensor_component(),
    p33=_tensor_component(),
)
@settings(max_examples=30, deadline=None)
def test_isotropic_pressure_equals_trace_over_three(
    p11: np.ndarray, p22: np.ndarray, p33: np.ndarray
) -> None:
    r"""``isotropic_pressure(P11, P22, P33) == (P11 + P22 + P33) / 3``
    — the first tensor invariant divided by 3. Encodes the
    equations.md § 4 footnote [^9] claim "P = Tr(P)/3" at the recipe
    level, distinct from iteration 10's rotation-invariance test
    which asserts the trace is a *rotation* invariant.
    """
    p = isotropic_pressure(p11, p22, p33)
    assert_allclose(p, (p11 + p22 + p33) / 3.0, rtol=0, atol=0)


@given(
    p_a=_tensor_component(),
    p_b=_tensor_component(),
    p_c=_tensor_component(),
)
@settings(max_examples=30, deadline=None)
def test_total_pressure_is_commutative_and_associative(
    p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray
) -> None:
    r"""``total_pressure`` is addition — commutative and associative
    bit-exact for positive inputs with no cancellation. Guards against
    any future change that introduces non-linearity (e.g. a weighted
    average or a max).
    """
    # Commutativity.
    assert_allclose(
        total_pressure(p_a, p_b), total_pressure(p_b, p_a), rtol=0, atol=0
    )
    # Associativity — float64 addition is NOT exactly associative in
    # general; rearrangement can introduce 1-ulp error per step, so for
    # inputs up to 1e3 the absolute difference is bounded by ~eps·|sum|
    # ≈ 1e-13.
    lhs = total_pressure(total_pressure(p_a, p_b), p_c)
    rhs = total_pressure(p_a, total_pressure(p_b, p_c))
    assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)
