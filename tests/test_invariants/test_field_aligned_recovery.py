# Source: src/pypic/derived.py:622 (parallel_component, A·b̂)
#         + :664 (perpendicular_vector, A - A_par·b̂)
#         + docs/equations.md § 4.2 (field-aligned decomposition).
# Claim: A_par·b̂ + A_perp = A componentwise, for arbitrary A and
#        non-zero B. The Pythagorean identity already covers
#        |A|² = A_par² + |A_perp|² (test_derived.py:614), but that
#        constrains magnitudes only — a sign flip in
#        ``perpendicular_vector`` (e.g. ``A + A_par b̂`` instead of
#        ``A - A_par b̂``) leaves magnitudes intact but breaks the
#        vector reconstruction. Both identities are needed to pin
#        down the decomposition up to a sign.
"""Vector recovery identity for the field-aligned decomposition."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import parallel_component, perpendicular_vector

SHAPE = (3, 4)


def _bounded_array(
    lo: float = -10.0, hi: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=lo,
            max_value=hi,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _nonzero_array(
    min_value: float = 0.5, max_value: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    """Arrays bounded away from zero — avoids 0/0 in |B| normalization."""
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


@given(
    a1=_bounded_array(),
    a2=_bounded_array(),
    a3=_bounded_array(),
    b1=_nonzero_array(),
    b2=_nonzero_array(),
    b3=_nonzero_array(),
)
@settings(max_examples=60, deadline=None)
def test_parallel_plus_perpendicular_recovers_vector(
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$A_\parallel \hat{b} + \mathbf{A}_\perp = \mathbf{A}$ componentwise.

    The Pythagorean identity $|\mathbf{A}|^2 = A_\parallel^2 +
    |\mathbf{A}_\perp|^2$ constrains magnitudes only; a sign error in
    ``perpendicular_vector`` (``A + A_\parallel \hat{b}`` instead of
    ``A - A_\parallel \hat{b}``) leaves the squared sum intact but
    breaks reconstruction. The vector recovery identity closes that
    gap.

    Tolerance: the reconstruction is ``A - A_par·b̂ + A_par·b̂``, so the
    error is a near-cancellation at scale $|A_\parallel|\cdot|\hat{b}|
    \le |A| \le 10\sqrt{3}$. Accumulated roundoff ~ ``100·eps·|A|`` ≈
    ``2e-14``; ``atol=1e-12`` absorbs that comfortably.
    """
    b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
    bhat1, bhat2, bhat3 = b1 / b_mag, b2 / b_mag, b3 / b_mag
    a_par = parallel_component(a1, a2, a3, b1, b2, b3)
    ap1, ap2, ap3 = perpendicular_vector(a1, a2, a3, b1, b2, b3)
    assert_allclose(a_par * bhat1 + ap1, a1, rtol=0, atol=1e-12)
    assert_allclose(a_par * bhat2 + ap2, a2, rtol=0, atol=1e-12)
    assert_allclose(a_par * bhat3 + ap3, a3, rtol=0, atol=1e-12)
