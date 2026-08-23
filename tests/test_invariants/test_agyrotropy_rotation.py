# Source: src/pypic/derived.py:1471 (agyrotropy, Swisdak 2016) +
#         docs/schema.md § "Pressure tensor" ("agyrotropy — deviation
#         from gyrotropic symmetry") + docs/equations.md § 4
#         (agyrotropy is a tensor invariant under coordinate rotation).
# Claim: Q(R·P·R^T, R·B) == Q(P, B) for any rotation matrix R.
#        Q is built from Tr(P), P_par = b̂·P·b̂, and ‖Π‖²_F where
#        Π = (I - b̂b̂^T) P (I - b̂b̂^T) is the double-projected
#        perpendicular tensor. All three are scalars under orthogonal
#        conjugation — Tr is rotation-invariant, the quadratic form
#        b̂·P·b̂ is scalar when B rotates with P, and Frobenius norms
#        of orthogonally-similar tensors coincide. Any bug that
#        introduces an axis-aligned assumption in the Π construction
#        (e.g. hardcoding ẑ as the perpendicular-plane normal) would
#        break this test while still passing the single-axis unit tests
#        at derived.py:1525-1530 and in tests/test_derived.py::TestAgyrotropy.
# Complements the Tr/P_par/P_perp rotation invariance in
# test_pressure_rotation.py: this one exercises the agyrotropy compound
# formula, one level further.
"""Rotation invariance of the agyrotropy measure Q."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.coordinates.transforms import (
    rotate_pressure_tensor,
    rotate_vector_components,
)
from pypic.derived import agyrotropy
from tests.strategies import rotations

SHAPE = (3, 4)


def _tensor_component() -> st.SearchStrategy[np.ndarray]:
    """Bounded finite array for one off-diagonal pressure-tensor slot
    (signs allowed)."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_diagonal_component() -> st.SearchStrategy[np.ndarray]:
    """Strictly positive bounded array for a diagonal pressure slot.

    Degenerate-input narrowing: Hypothesis found that an all-zero
    pressure tensor produces Q = 0/0 = NaN (the formula Q = 1 - 4I₂/I₁²
    divides by I₁² = (Tr(P) - P_∥)², which vanishes when every
    component is zero). A zero pressure tensor is unphysical for any
    real plasma; restricting diagonals to [0.1, 5.0] keeps I₁ > 0
    almost everywhere while still spanning the interesting
    anisotropy regime. Off-diagonals retain their full [-5, 5]
    range — asymmetry in sign is what the rotation must mix.
    """
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=0.1,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


def _b_component() -> st.SearchStrategy[np.ndarray]:
    """Bounded B component — one axis is held away from zero separately
    (see ``b3`` below) to guarantee $|B| > 0$."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-3.0,
            max_value=3.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@given(
    p11=_positive_diagonal_component(),
    p22=_positive_diagonal_component(),
    p33=_positive_diagonal_component(),
    p12=_tensor_component(),
    p13=_tensor_component(),
    p23=_tensor_component(),
    b1=_b_component(),
    b2=_b_component(),
    b3=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    rotation=rotations(),
)
@settings(max_examples=40)
def test_agyrotropy_is_rotation_invariant(
    p11: np.ndarray,
    p22: np.ndarray,
    p33: np.ndarray,
    p12: np.ndarray,
    p13: np.ndarray,
    p23: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: float,
    rotation: tuple,
) -> None:
    r"""$Q(R \mathbf{P} R^T, R \mathbf{B}) = Q(\mathbf{P}, \mathbf{B})$
    for any rotation matrix $R$.

    The pressure tensor and magnetic field must rotate together.
    Tolerance: Q = 1 - 4 I₂/I₁² involves a ratio of quadratic tensor
    invariants; when $I_1 \to 0$ (near-isotropic $P$ or $\mathbf{B}$
    nearly aligned with a degenerate eigen-direction) the division
    amplifies float64 roundoff. 1e-9 bounds this comfortably without
    masking any O(1) coordinate bug.
    """
    b3_arr = np.full(SHAPE, b3, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)

    rp11, rp22, rp33, rp12, rp13, rp23 = rotate_pressure_tensor(
        p11, p22, p33, p12, p13, p23, r
    )
    rb1, rb2, rb3 = rotate_vector_components(b1, b2, b3_arr, r)

    q_before = agyrotropy(p11, p22, p33, p12, p13, p23, b1, b2, b3_arr)
    q_after = agyrotropy(rp11, rp22, rp33, rp12, rp13, rp23, rb1, rb2, rb3)

    # Drop any Q that came out NaN before — |B| could collapse to 0 in
    # some entries after rotation if b3 is small compared to the other
    # components in the stacked draw, though b3 ≥ 0.5 makes this rare.
    # Mask consistently on both sides.
    mask = np.isfinite(q_before) & np.isfinite(q_after)
    assert mask.any(), "Test vacuous: every element went NaN on both sides."
    assert_allclose(q_after[mask], q_before[mask], rtol=1e-9, atol=1e-10)


@given(
    p_scalar=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    b1=_b_component(),
    b2=_b_component(),
    b3=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_isotropic_pressure_is_gyrotropic_for_any_b(
    p_scalar: float, b1: np.ndarray, b2: np.ndarray, b3: float
) -> None:
    r"""$Q(P \mathbf{I}, \mathbf{B}) = 0$ for any $\mathbf{B}$.

    The isotropic tensor $P \mathbf{I}$ has a trivially-gyrotropic
    shape — $P_\parallel = P_\perp = P$ regardless of the $\mathbf{B}$
    direction. Existing ``test_derived.py::test_isotropic_is_zero``
    only exercises $\mathbf{B} = \hat{z}$; this extends the claim to
    arbitrary $\mathbf{B}$. An axis-specific coordinate leak in the
    Π projection would produce a spurious non-zero $Q$ for
    off-axis $\mathbf{B}$ and be caught here.
    """
    p_diag = np.full(SHAPE, p_scalar, dtype=np.float64)
    zeros = np.zeros(SHAPE, dtype=np.float64)
    b3_arr = np.full(SHAPE, b3, dtype=np.float64)

    q = agyrotropy(p_diag, p_diag, p_diag, zeros, zeros, zeros, b1, b2, b3_arr)
    # Q = 1 - 4 I₂/I₁² with I₁ = 2P, I₂ = (4P² - 2P²)/2 = P² → Q = 0.
    # Residual comes from the |B| normalization + quadratic projection.
    assert_allclose(q, 0.0, atol=1e-13)
