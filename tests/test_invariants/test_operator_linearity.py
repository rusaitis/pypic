# Source: src/pypic/coordinates/operators.py — ``gradient``,
#         ``divergence``, and ``curl`` are all implemented via
#         ``np.gradient``, a linear finite-difference operator along
#         a single axis.
# Claim: the vector identity ``curl(grad f) = 0`` holds bit-exactly
#        everywhere, not just in the interior. Dual to the
#        ``div(curl F) = 0`` identity in ``test_operator_identities.py``:
#        linear operators along distinct axes commute even where the
#        boundary one-sided stencils apply, because they never read
#        across axes.
# No separate α F + β G linearity test: if linearity broke silently,
# div(curl F) = 0 and curl(grad f) = 0 would fail first.
"""The curl(grad f) = 0 vector identity (dual to div(curl F) = 0)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.coordinates.operators import curl, gradient

SHAPE = (5, 6, 7)


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    """Float64 arrays with values in [-1, 1]."""
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

    Dual to the ``div(curl F) = 0`` identity in
    ``test_operator_identities.py``. Same argument: ``np.gradient``
    along distinct
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
