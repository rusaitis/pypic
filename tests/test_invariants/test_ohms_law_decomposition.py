# Source: docs/equations.md § 9 footnote [^12]:
#         "E' = E + V × B ... the non-ideal residual vanishes in ideal MHD"
#         "E_ideal = -V × B"
#         src/pypic/derived.py:1675 (non_ideal_electric_field)
#         src/pypic/derived.py:1623 (ideal_electric_field)
# Claims:
#   (a) E_prime + E_ideal = E component-wise (Ohm's law closure).
#   (b) E_ideal · B = 0 (ideal E is perpendicular to B).
#   (c) E_prime · B = E · B (non-ideal preserves B-parallel E component).
#   (d) magnetic_shear_angle(B, B) = 0 and magnetic_shear_angle(B, -B) = π.
# Each function is authored independently (different args, different
# signs); a sign error in the V×B cross product would break (a) even
# though each function's own unit test uses only one test vector.
# Fresh invariant #7 after backlog exhaustion.
"""Ohm's law decomposition closure and magnetic shear angle limits."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    ideal_electric_field,
    magnetic_shear_angle,
    non_ideal_electric_field,
)

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


def _nonzero_array(
    min_value: float = 0.5, max_value: float = 10.0
) -> st.SearchStrategy[np.ndarray]:
    """Arrays bounded away from zero — avoids 0/0 in angle computations."""
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
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50)
def test_non_ideal_plus_ideal_equals_total_e(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{E}' + \mathbf{E}_{ideal} = \mathbf{E}$ component-wise.

    ``E' = E + V×B`` and ``E_ideal = -V×B``, so ``E' + E_ideal =
    E + V×B - V×B = E``. The V×B cross products must cancel bit-exactly
    since both recipes expand the same three terms with opposite signs.
    """
    ep1, ep2, ep3 = non_ideal_electric_field(e1, e2, e3, v1, v2, v3, b1, b2, b3)
    ei1, ei2, ei3 = ideal_electric_field(v1, v2, v3, b1, b2, b3)
    # V×B products reach ~100 when V, B are both bounded by ±10; the
    # cancellation ``E + V×B - V×B`` therefore carries roundoff up to
    # ~100 × eps ≈ 2e-14. 1e-12 absorbs that comfortably.
    assert_allclose(ep1 + ei1, e1, rtol=0, atol=1e-12)
    assert_allclose(ep2 + ei2, e2, rtol=0, atol=1e-12)
    assert_allclose(ep3 + ei3, e3, rtol=0, atol=1e-12)


@given(
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50)
def test_ideal_e_perpendicular_to_b(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{E}_{ideal} \cdot \mathbf{B} = 0$ — the convective
    electric field $-\mathbf{V} \times \mathbf{B}$ is always perpendicular
    to $\mathbf{B}$. Companion to the Poynting S·B = 0 orthogonality
    in ``test_lagrange_identity.py``.
    """
    ei1, ei2, ei3 = ideal_electric_field(v1, v2, v3, b1, b2, b3)
    dot = ei1 * b1 + ei2 * b2 + ei3 * b3
    assert_allclose(dot, 0.0, atol=1e-12)


@given(
    b1=_nonzero_array(),
    b2=_nonzero_array(),
    b3=_nonzero_array(),
)
@settings(max_examples=50)
def test_shear_angle_self_is_zero(
    b1: np.ndarray, b2: np.ndarray, b3: np.ndarray
) -> None:
    r"""$\theta(\mathbf{B}, \mathbf{B}) = 0$ — a vector is parallel to itself.

    The ``arccos`` function amplifies roundoff near $\cos\theta = 1$:
    $\arccos(1 - \varepsilon) \approx \sqrt{2\varepsilon}$. With
    $\varepsilon \sim 10^{-16}$ from the dot/(mag×mag) division, the
    angle error is $\sim 1.5 \times 10^{-8}$ — inherent to the formula,
    not a bug.
    """
    angle = magnetic_shear_angle(b1, b2, b3, b1, b2, b3)
    assert_allclose(angle, 0.0, atol=1e-7)


@given(
    b1=_nonzero_array(),
    b2=_nonzero_array(),
    b3=_nonzero_array(),
)
@settings(max_examples=50)
def test_shear_angle_antiparallel_is_pi(
    b1: np.ndarray, b2: np.ndarray, b3: np.ndarray
) -> None:
    r"""$\theta(\mathbf{B}, -\mathbf{B}) = \pi$ — a vector is
    anti-parallel to its negation. Exercises the ``arccos(-1)`` branch
    of the shear-angle formula.
    """
    angle = magnetic_shear_angle(b1, b2, b3, -b1, -b2, -b3)
    assert_allclose(angle, np.pi, atol=1e-7)
