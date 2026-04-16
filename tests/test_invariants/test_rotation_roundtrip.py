# Source: src/pypic/coordinates/transforms.py:413 ("Rotate vector field
#         components by a 3×3 rotation matrix: v'_i = sum_j R_ij v_j")
#         + :444 ("Rotate a symmetric pressure tensor by a rotation
#         matrix: P'_ij = sum_kl R_ik R_jl P_kl").
# Orthogonality of the rotation matrix (required by
# ``FrameTransform.__post_init__`` at :110, ``|R^T R - I| < 1e-6``)
# implies two consequences the code never states directly but every
# caller relies on:
#   (a) Vector magnitude is preserved: ``|R·v|² == |v|²``.
#   (b) Composition with the transpose is the identity:
#       ``R^T · (R · v) == v``, and likewise for the tensor double
#       contraction.
# Existing tests in tests/test_transforms.py only cover specific axis
# swaps / single-angle rotations. Hypothesis over the full SO(3)
# (Rodrigues via tests.strategies.rotations) exercises arbitrary
# orientations. Fresh invariant #3 after backlog exhaustion.
"""Magnitude preservation and R → R^T round-trip for rotation routines."""

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
from tests.strategies import rotations

SHAPE = (4, 3)


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


@given(
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    rotation=rotations(),
)
@settings(max_examples=60, deadline=None)
def test_vector_magnitude_is_rotation_invariant(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, rotation: tuple
) -> None:
    r"""$|R \cdot \mathbf{v}|^2 = |\mathbf{v}|^2$ for any SO(3) R.

    The squared magnitude is a rank-0 tensor invariant; checking on
    squared magnitudes avoids a ``sqrt`` that would cut the attainable
    precision from 1e-14 to ~1e-7 for typical inputs.
    """
    r = np.asarray(rotation, dtype=np.float64)
    r1, r2, r3 = rotate_vector_components(v1, v2, v3, r)
    assert_allclose(
        r1**2 + r2**2 + r3**2,
        v1**2 + v2**2 + v3**2,
        rtol=1e-12,
        atol=1e-12,
    )


@given(
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    rotation=rotations(),
)
@settings(max_examples=60, deadline=None)
def test_vector_rotation_inverse_is_transpose(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, rotation: tuple
) -> None:
    r"""$R^T (R \mathbf{v}) = \mathbf{v}$ — for orthogonal R, the
    transpose is the inverse. Guards against axis-order swaps inside
    ``rotate_vector_components`` (e.g. summing over rows where columns
    were meant).
    """
    r = np.asarray(rotation, dtype=np.float64)
    r1, r2, r3 = rotate_vector_components(v1, v2, v3, r)
    u1, u2, u3 = rotate_vector_components(r1, r2, r3, r.T)
    assert_allclose(u1, v1, rtol=1e-12, atol=1e-12)
    assert_allclose(u2, v2, rtol=1e-12, atol=1e-12)
    assert_allclose(u3, v3, rtol=1e-12, atol=1e-12)


@given(
    p11=_bounded_array(),
    p22=_bounded_array(),
    p33=_bounded_array(),
    p12=_bounded_array(),
    p13=_bounded_array(),
    p23=_bounded_array(),
    rotation=rotations(),
)
@settings(max_examples=40, deadline=None)
def test_pressure_tensor_rotation_inverse_is_transpose(
    p11: np.ndarray,
    p22: np.ndarray,
    p33: np.ndarray,
    p12: np.ndarray,
    p13: np.ndarray,
    p23: np.ndarray,
    rotation: tuple,
) -> None:
    r"""$R^T (R P R^T) R = P$ — the tensor double contraction with the
    transpose recovers the original. This is the 6-component version
    of the 3-component vector round-trip above and catches any index
    confusion between ``R_ik R_jl`` and ``R_ki R_lj`` in
    ``rotate_pressure_tensor``.
    """
    r = np.asarray(rotation, dtype=np.float64)
    rp11, rp22, rp33, rp12, rp13, rp23 = rotate_pressure_tensor(
        p11, p22, p33, p12, p13, p23, r
    )
    out11, out22, out33, out12, out13, out23 = rotate_pressure_tensor(
        rp11, rp22, rp33, rp12, rp13, rp23, r.T
    )
    assert_allclose(out11, p11, rtol=1e-11, atol=1e-11)
    assert_allclose(out22, p22, rtol=1e-11, atol=1e-11)
    assert_allclose(out33, p33, rtol=1e-11, atol=1e-11)
    assert_allclose(out12, p12, rtol=1e-11, atol=1e-11)
    assert_allclose(out13, p13, rtol=1e-11, atol=1e-11)
    assert_allclose(out23, p23, rtol=1e-11, atol=1e-11)


@given(
    p11=_bounded_array(),
    p22=_bounded_array(),
    p33=_bounded_array(),
    p12=_bounded_array(),
    p13=_bounded_array(),
    p23=_bounded_array(),
    rotation=rotations(),
)
@settings(max_examples=40, deadline=None)
def test_pressure_tensor_frobenius_norm_is_rotation_invariant(
    p11: np.ndarray,
    p22: np.ndarray,
    p33: np.ndarray,
    p12: np.ndarray,
    p13: np.ndarray,
    p23: np.ndarray,
    rotation: tuple,
) -> None:
    r"""$\|P\|_F^2 = \sum_{ij} P_{ij}^2$ is the second symmetric-tensor
    invariant. For a 3×3 symmetric tensor:
    $\|P\|_F^2 = P_{11}^2 + P_{22}^2 + P_{33}^2 + 2(P_{12}^2 + P_{13}^2 + P_{23}^2)$.
    Complementing the trace invariance covered in iteration 10 at
    tests/test_invariants/test_pressure_rotation.py:79.
    """
    r = np.asarray(rotation, dtype=np.float64)
    rp11, rp22, rp33, rp12, rp13, rp23 = rotate_pressure_tensor(
        p11, p22, p33, p12, p13, p23, r
    )

    def _frob_sq(a11, a22, a33, a12, a13, a23):
        return a11**2 + a22**2 + a33**2 + 2.0 * (a12**2 + a13**2 + a23**2)

    assert_allclose(
        _frob_sq(rp11, rp22, rp33, rp12, rp13, rp23),
        _frob_sq(p11, p22, p33, p12, p13, p23),
        rtol=1e-11,
        atol=1e-11,
    )
