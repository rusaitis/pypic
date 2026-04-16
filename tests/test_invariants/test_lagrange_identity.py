# Source: src/pypic/derived.py:373-424 (poynting_flux definition
#         S = E × B) + docs/equations.md § 3 (e_B = B²/2, e_E = E²/2,
#         S = E × B). The Lagrange identity |a × b|² + (a·b)² =
#         |a|²|b|² is a pure vector-algebra identity — it holds for
#         ANY three-vectors independent of physics context.
# Claims:
#   (a) Lagrange identity: |S|² + (E·B)² = |E|²|B|² to machine
#       precision. Binds the squared Poynting magnitude, the
#       second Lorentz invariant (E·B), and the two field magnitudes
#       into a single algebraic relation.
#   (b) Energy-density form: |S|² + (E·B)² = 4 · e_E · e_B, since
#       e_E = E²/2 and e_B = B²/2 imply |E|²|B|² = 4·e_E·e_B.
#   (c) S·E = 0 — complement to iter-15 S·B = 0. The cross product is
#       perpendicular to BOTH operands; iter-15 only tested one side.
# Fresh invariant #14 after backlog exhaustion.
"""Lagrange identity and S·E = 0 for the Poynting flux."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    electric_energy_density,
    magnetic_energy_density,
    poynting_flux,
)

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
@settings(max_examples=60, deadline=None)
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
    expansion (``derived.py:421-423``) that leaves ``S·B = 0`` intact
    (iter 15) could still break this — the two constraints are
    independent. Both are needed to fully pin down the cross product up
    to the right-hand-rule convention.

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


@given(
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=40, deadline=None)
def test_lagrange_identity_in_energy_form(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$|\mathbf{S}|^2 + (\mathbf{E}\cdot\mathbf{B})^2 = 4 e_E e_B$
    — Lagrange identity re-expressed through the energy-density
    recipes. Exercises ``magnetic_energy_density`` and
    ``electric_energy_density`` against the raw field magnitudes,
    catching any missing factor of $1/2$ in either.
    """
    s1, s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
    s_sq = s1**2 + s2**2 + s3**2
    e_dot_b = e1 * b1 + e2 * b2 + e3 * b3
    e_mag = np.sqrt(e1**2 + e2**2 + e3**2)
    b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
    e_E = electric_energy_density(e_mag)
    e_B = magnetic_energy_density(b_mag)
    assert_allclose(s_sq + e_dot_b**2, 4.0 * e_E * e_B, rtol=1e-12, atol=1e-12)


@given(
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50, deadline=None)
def test_poynting_flux_dot_e_is_zero(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{S} \cdot \mathbf{E} = (\mathbf{E} \times \mathbf{B})
    \cdot \mathbf{E} = 0$ — cross product perpendicular to its first
    operand. Iter 15 tested $\mathbf{S} \cdot \mathbf{B} = 0$;
    the first-operand complement is an independent constraint
    (e.g. a typo swapping $e_1 \leftrightarrow b_1$ in the
    expansion would preserve $\mathbf{S}\cdot\mathbf{B} = 0$ but
    break $\mathbf{S}\cdot\mathbf{E} = 0$).
    """
    s1, s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
    dot = s1 * e1 + s2 * e2 + s3 * e3
    assert_allclose(dot, 0.0, atol=1e-12)
