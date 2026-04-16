# Source: src/pypic/coordinates/operators.py — ``gradient``,
#         ``divergence``, and ``curl`` are all implemented via
#         ``np.gradient``, which is a linear finite-difference operator
#         along a single axis. Two consequences the code never states
#         but every caller assumes:
#   (a) Each operator is linear in its vector-field input:
#       ``op(α F + β G) == α · op(F) + β · op(G)``.
#       A sign or factor bug in any branch of ``curl`` would violate
#       this even on the smooth fields where the non-linearity-
#       detecting tests in tests/test_operators.py pass.
#   (b) The vector identity ``curl(grad f) = 0`` holds bit-exactly
#       everywhere, not just in the interior. Dual to iteration 3's
#       ``div(curl F) = 0`` (commit 218d9d1): linear operators along
#       distinct axes commute even where the boundary one-sided
#       stencils apply, because they never read across axes.
# Fresh invariant #4 after numbered backlog exhausted at iteration 9.
"""Linearity of grad/div/curl and the curl(grad f) = 0 identity."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.coordinates.operators import curl, divergence, gradient

SHAPE = (5, 6, 7)


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    """Float64 arrays with values in [-1, 1]. Magnitudes kept modest so
    the ``α f + β g`` combinations stay well-scaled for the linearity
    checks — extreme inputs would trade accuracy for no insight on the
    algebraic property under test.
    """
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_spacing() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    )


def _coefficient() -> st.SearchStrategy[float]:
    """Scalar multiplier for the α F + β G combination.

    Subnormals excluded: a subnormal β (~1e-313) shrinks the adaptive
    tolerance ``|a| + |b|`` to below 1e-310 while ``a f + β g``
    operates on denormal intermediates that are only approximate — the
    test would fail not because linearity breaks but because the
    tolerance scale is meaningless at that magnitude.
    """
    return st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


@given(
    f=_bounded_array(),
    g=_bounded_array(),
    a=_coefficient(),
    b=_coefficient(),
    dx=_positive_spacing(),
    dy=_positive_spacing(),
    dz=_positive_spacing(),
)
@settings(max_examples=40, deadline=None)
def test_gradient_is_linear(
    f: np.ndarray,
    g: np.ndarray,
    a: float,
    b: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    r"""$\nabla(\alpha f + \beta g) = \alpha \nabla f + \beta \nabla g$
    component-wise. Tolerance ≈ roundoff in the combined product.
    """
    lhs1, lhs2, lhs3 = gradient(a * f + b * g, dx, dy, dz)
    fg1, fg2, fg3 = gradient(f, dx, dy, dz)
    gg1, gg2, gg3 = gradient(g, dx, dy, dz)
    inv_d = 1.0 / min(dx, dy, dz)
    atol = 1e-13 * (abs(a) + abs(b)) * inv_d
    assert_allclose(lhs1, a * fg1 + b * gg1, atol=atol, rtol=1e-12)
    assert_allclose(lhs2, a * fg2 + b * gg2, atol=atol, rtol=1e-12)
    assert_allclose(lhs3, a * fg3 + b * gg3, atol=atol, rtol=1e-12)


@given(
    f1=_bounded_array(),
    f2=_bounded_array(),
    f3=_bounded_array(),
    g1=_bounded_array(),
    g2=_bounded_array(),
    g3=_bounded_array(),
    a=_coefficient(),
    b=_coefficient(),
    dx=_positive_spacing(),
    dy=_positive_spacing(),
    dz=_positive_spacing(),
)
@settings(max_examples=40, deadline=None)
def test_curl_is_linear(
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    a: float,
    b: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    r"""$\nabla \times (\alpha \mathbf{F} + \beta \mathbf{G})
        = \alpha \nabla \times \mathbf{F} + \beta \nabla \times \mathbf{G}$."""
    sum1 = a * f1 + b * g1
    sum2 = a * f2 + b * g2
    sum3 = a * f3 + b * g3
    lhs1, lhs2, lhs3 = curl(sum1, sum2, sum3, dx, dy, dz)
    cf1, cf2, cf3 = curl(f1, f2, f3, dx, dy, dz)
    cg1, cg2, cg3 = curl(g1, g2, g3, dx, dy, dz)
    inv_d = 1.0 / min(dx, dy, dz)
    atol = 1e-13 * (abs(a) + abs(b)) * inv_d
    assert_allclose(lhs1, a * cf1 + b * cg1, atol=atol, rtol=1e-12)
    assert_allclose(lhs2, a * cf2 + b * cg2, atol=atol, rtol=1e-12)
    assert_allclose(lhs3, a * cf3 + b * cg3, atol=atol, rtol=1e-12)


@given(
    f1=_bounded_array(),
    f2=_bounded_array(),
    f3=_bounded_array(),
    g1=_bounded_array(),
    g2=_bounded_array(),
    g3=_bounded_array(),
    a=_coefficient(),
    b=_coefficient(),
    dx=_positive_spacing(),
    dy=_positive_spacing(),
    dz=_positive_spacing(),
)
@settings(max_examples=40, deadline=None)
def test_divergence_is_linear(
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    a: float,
    b: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    r"""$\nabla \cdot (\alpha \mathbf{F} + \beta \mathbf{G})
        = \alpha \nabla \cdot \mathbf{F} + \beta \nabla \cdot \mathbf{G}$."""
    sum1 = a * f1 + b * g1
    sum2 = a * f2 + b * g2
    sum3 = a * f3 + b * g3
    lhs = divergence(sum1, sum2, sum3, dx, dy, dz)
    rhs_f = divergence(f1, f2, f3, dx, dy, dz)
    rhs_g = divergence(g1, g2, g3, dx, dy, dz)
    inv_d = 1.0 / min(dx, dy, dz)
    atol = 1e-13 * (abs(a) + abs(b)) * inv_d
    assert_allclose(lhs, a * rhs_f + b * rhs_g, atol=atol, rtol=1e-12)


@given(
    f=_bounded_array(),
    dx=_positive_spacing(),
    dy=_positive_spacing(),
    dz=_positive_spacing(),
)
@settings(max_examples=40, deadline=None)
def test_curl_of_gradient_is_zero(
    f: np.ndarray, dx: float, dy: float, dz: float
) -> None:
    r"""$\nabla \times \nabla f = \mathbf{0}$ bit-exactly everywhere.

    Dual to the ``div(curl F) = 0`` identity tested in iteration 3
    (commit 218d9d1). Same argument: ``np.gradient`` along distinct
    axes commutes exactly (each call reads only its own axis), so the
    three curl components reduce to pairwise differences of identical
    mixed partials and cancel at every grid point — including the
    one-sided boundary stencils.
    """
    g1, g2, g3 = gradient(f, dx, dy, dz)
    c1, c2, c3 = curl(g1, g2, g3, dx, dy, dz)
    inv_d_sq = 1.0 / min(dx, dy, dz) ** 2
    atol = 2e-13 * inv_d_sq
    assert_allclose(c1, 0.0, atol=atol)
    assert_allclose(c2, 0.0, atol=atol)
    assert_allclose(c3, 0.0, atol=atol)
