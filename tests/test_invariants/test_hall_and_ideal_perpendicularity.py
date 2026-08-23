# Source: src/pypic/derived.py:1623 (ideal_electric_field,
#         E_ideal = -V × B) + :1740 (hall_electric_field,
#         E_Hall = (J × B) / (n|q|)) + docs/equations.md footnote
#         [^12] (Generalized Ohm's law terms) +
#         test_ohms_law_decomposition.py, which establishes
#         E_ideal · B = 0; this file covers the three complementary
#         orthogonalities.
# Claims:
#   (a) E_ideal · V = 0 — the convective E = -V × B is perpendicular
#       to V as well as to B; test_ohms_law_decomposition.py covers
#       only the B side.
#   (b) E_Hall · J = 0 — J × B / (n|q|) is perpendicular to J.
#   (c) E_Hall · B = 0 — and to B.
#   (d) hall_electric_field(.., charge=+q) == hall_electric_field(.., charge=-q)
#       — docstring promises "The absolute value is used — sign does not
#       affect the result" (derived.py:1777). A missing abs() would
#       break this.
# Fresh invariant #15 after backlog exhaustion.
"""Hall and ideal E-field perpendicularities + Hall charge-sign invariance."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import hall_electric_field, ideal_electric_field

SHAPE = (3, 4)


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_array(
    min_value: float = 0.1, max_value: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    """Strictly-positive arrays for number density (appears in
    denominator of the Hall term)."""
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


def _perpendicularity_atol(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Float64 floor for ``a · b == 0`` when the dot is algebraically zero.

    The dot product accumulates ~``eps · |a| · |b|`` per term; a few
    dozen multiplies + cancellations push the constant up to ~100·eps.
    Returning the bound rather than a fixed atol keeps the assertion
    tight when ``|a|, |b|`` are small and forgiving when ``E_Hall``
    blows up because ``n|q|`` is small.
    """
    a_max = max(float(np.max(np.abs(c))) for c in a)
    b_max = max(float(np.max(np.abs(c))) for c in b)
    return 100.0 * np.finfo(np.float64).eps * a_max * b_max


@given(
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50, deadline=None)
def test_ideal_electric_field_perpendicular_to_v(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{E}_{ideal} \cdot \mathbf{V} = 0$ — $-\mathbf{V} \times
    \mathbf{B}$ is perpendicular to the first operand $\mathbf{V}$ as
    well as to $\mathbf{B}$. Iter 16 only tested the $\mathbf{B}$ side;
    both are needed to fully constrain the cross-product implementation
    (a swap of components that preserves $\cdot \mathbf{B} = 0$ could
    still break $\cdot \mathbf{V} = 0$).
    """
    ei1, ei2, ei3 = ideal_electric_field(v1, v2, v3, b1, b2, b3)
    dot = ei1 * v1 + ei2 * v2 + ei3 * v3
    assert_allclose(dot, 0.0, atol=1e-12)


@given(
    j1=_bounded_array(),
    j2=_bounded_array(),
    j3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
    n=_positive_array(),
    charge=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False).filter(
        lambda q: abs(q) > 0.1
    ),
)
@settings(max_examples=50, deadline=None)
def test_hall_electric_field_perpendicular_to_j(
    j1: np.ndarray,
    j2: np.ndarray,
    j3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    n: np.ndarray,
    charge: float,
) -> None:
    r"""$\mathbf{E}_{Hall} \cdot \mathbf{J} = 0$ — the Hall term is
    proportional to $\mathbf{J} \times \mathbf{B}$, perpendicular to
    $\mathbf{J}$. Division by the scalar $n|q|$ preserves direction.

    Tolerance: the dot product's float floor scales with
    ``|E_Hall| · |J|``; pin ``atol`` to that natural scale times
    ``100 · eps`` rather than a fixed constant, so the test stays
    robust when ``n|q|`` is small enough to amplify ``E_Hall``.
    """
    eh1, eh2, eh3 = hall_electric_field(j1, j2, j3, b1, b2, b3, n, charge)
    dot = eh1 * j1 + eh2 * j2 + eh3 * j3
    atol = _perpendicularity_atol((eh1, eh2, eh3), (j1, j2, j3))
    assert_allclose(dot, 0.0, atol=atol)


@given(
    j1=_bounded_array(),
    j2=_bounded_array(),
    j3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
    n=_positive_array(),
    charge=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False).filter(
        lambda q: abs(q) > 0.1
    ),
)
@settings(max_examples=50, deadline=None)
def test_hall_electric_field_perpendicular_to_b(
    j1: np.ndarray,
    j2: np.ndarray,
    j3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    n: np.ndarray,
    charge: float,
) -> None:
    r"""$\mathbf{E}_{Hall} \cdot \mathbf{B} = 0$ — completes the pair
    of perpendicularities that pin down the $\mathbf{J} \times \mathbf{B}$
    cross product. Tolerance scaling matches the J-side test above.
    """
    eh1, eh2, eh3 = hall_electric_field(j1, j2, j3, b1, b2, b3, n, charge)
    dot = eh1 * b1 + eh2 * b2 + eh3 * b3
    atol = _perpendicularity_atol((eh1, eh2, eh3), (b1, b2, b3))
    assert_allclose(dot, 0.0, atol=atol)


@given(
    j1=_bounded_array(),
    j2=_bounded_array(),
    j3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
    n=_positive_array(),
    charge_mag=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_hall_electric_field_uses_charge_magnitude(
    j1: np.ndarray,
    j2: np.ndarray,
    j3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    n: np.ndarray,
    charge_mag: float,
) -> None:
    r"""``hall_electric_field(..., charge=+q) == hall_electric_field(
    ..., charge=-q)`` — derived.py:1777 docstring: "The absolute value
    is used — sign does not affect the result". A missing ``abs()``
    would flip all three components under ``charge → -charge``, which
    is exactly the failure mode this test catches.
    """
    pos = hall_electric_field(j1, j2, j3, b1, b2, b3, n, +charge_mag)
    neg = hall_electric_field(j1, j2, j3, b1, b2, b3, n, -charge_mag)
    for p, q in zip(pos, neg, strict=True):
        assert_allclose(p, q, rtol=0, atol=0)
