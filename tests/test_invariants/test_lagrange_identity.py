# Source: src/pypic/derived.py:373-424 (poynting_flux definition
#         S = E × B) + docs/equations.md § 3 (S = E × B).
#         The Lagrange identity |a × b|² + (a·b)² = |a|²|b|² is a
#         pure vector-algebra identity — it holds for ANY
#         three-vectors independent of physics context.
# Claim: |S|² + (E·B)² = |E|²|B|² to machine precision. Binds the
#        squared Poynting magnitude, the second Lorentz invariant
#        (E·B), and the two field magnitudes into a single algebraic
#        relation. Any sign error in the cross-product expansion that
#        leaves S·B = 0 intact (test_ohms_law_decomposition.py) could
#        still break this — the two constraints are independent and
#        both are needed to pin down the cross product up to the
#        right-hand-rule convention.
"""Lagrange identity for the Poynting flux."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import poynting_flux

SHAPE = (3, 4)


def _bounded_array(
    lo: float = -10.0, hi: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    """Finite signed arrays — E and B can be any sign, not just positive."""
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


@given(
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=60)
def test_lagrange_identity(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$|\mathbf{E} \times \mathbf{B}|^2 + (\mathbf{E} \cdot \mathbf{B})^2
    = |\mathbf{E}|^2 |\mathbf{B}|^2$ — Lagrange's identity.

    A pure vector-algebra identity. Any sign error in the cross-product
    expansion that leaves ``S·B = 0`` intact (covered in
    ``test_ohms_law_decomposition.py``) could still break this — the two
    constraints are independent. Both are needed to fully pin down the
    cross product up to the right-hand-rule convention.

    Tolerance: the LHS sums three squared products (six multiplications,
    two subtractions) plus $(\mathbf{E}\cdot\mathbf{B})^2$ — roughly
    $30 \cdot \varepsilon \cdot \max^4$ accumulated roundoff for values
    in $[-10, 10]$.
    """
    s1, s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
    lhs = s1**2 + s2**2 + s3**2 + (e1 * b1 + e2 * b2 + e3 * b3) ** 2
    e_sq = e1**2 + e2**2 + e3**2
    b_sq = b1**2 + b2**2 + b3**2
    rhs = e_sq * b_sq
    # 30·eps at |E|,|B|≤10 → |E|²|B|² ≤ 1e4 → roundoff ≈ 3e-12.
    assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
