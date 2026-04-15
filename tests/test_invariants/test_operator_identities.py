# Source: docs/equations.md § 6 ("Magnitudes and Differential Operators")
#         + src/pypic/coordinates/operators.py docstrings (second-order central
#           differences, one-sided at boundaries, np.gradient convention).
# Claim: $\nabla \cdot (\nabla \times \mathbf{F}) = 0$ — a vector identity
# that the discrete operators must preserve. With np.gradient's stencil
# (linear along each axis independently), mixed partials commute bit-exactly
# at every grid point, so the identity holds to *machine* precision, not
# just truncation-error precision. Backlog #3 in autoresearcher-pypic.md.
"""Discrete vector identities for ``curl`` and ``divergence``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypic.coordinates.operators import curl, divergence


def _bounded_array(shape: tuple[int, int, int]) -> st.SearchStrategy[np.ndarray]:
    """Float64 arrays with values in [-1, 1]."""
    return arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_spacing() -> st.SearchStrategy[float]:
    """Grid spacings in [0.1, 10.0] — avoid denormals without stressing roundoff."""
    return st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    )


# Shapes are kept small: the identity is point-wise, not a convergence claim,
# so (5, 6, 7) already covers interior + boundary + every mixed-axis combination.
# A non-cube shape guards against axis-order bugs (e.g. confusing d1 with d2).
SHAPE = (5, 6, 7)


@given(
    f1=_bounded_array(SHAPE),
    f2=_bounded_array(SHAPE),
    f3=_bounded_array(SHAPE),
    d1=_positive_spacing(),
    d2=_positive_spacing(),
    d3=_positive_spacing(),
)
@settings(max_examples=50, deadline=None)
def test_div_curl_is_zero_to_machine_precision(
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    d1: float,
    d2: float,
    d3: float,
) -> None:
    r"""$\nabla \cdot (\nabla \times \mathbf{F}) = 0$ bit-exactly.

    The discrete identity holds for *any* array, not just smooth fields:
    ``np.gradient`` is a linear operator along a single axis, and linear
    operators along distinct axes commute exactly. Mixed partials cancel
    pairwise in ``div(curl F)`` at every grid point — including boundaries,
    where the one-sided stencils still commute with the orthogonal central
    stencils. Tolerance is set to accumulated float64 roundoff
    (~N_ops × eps × max(|f|/d²) ≈ 1e-11 for the worst combination of
    small spacing and many summations).
    """
    c1, c2, c3 = curl(f1, f2, f3, d1, d2, d3)
    result = divergence(c1, c2, c3, d1, d2, d3)
    # Scale tolerance by the worst inverse-spacing product that appears in
    # the second derivative (1 / min(d)**2). Keeps the bound realistic when
    # Hypothesis picks d = 0.1.
    inv_d_sq = 1.0 / min(d1, d2, d3) ** 2
    atol = 2e-13 * inv_d_sq
    np.testing.assert_allclose(result, 0.0, atol=atol)
