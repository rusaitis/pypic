# Source: src/pypic/derived.py:1362 ("P_par = b̂·P·b̂" — a double
#         contraction is a scalar, hence frame-invariant by construction)
#         + src/pypic/coordinates/transforms.py:457 ("The trace
#         P_11 + P_22 + P_33 is invariant [under rotation]")
#         + docs/equations.md § 4 footnote [^9]
#         ("$P = (P_\parallel + 2 P_\perp)/3$").
# Claims tested here:
#   (a) Trace of the rotated pressure tensor equals the original trace.
#   (b) ``parallel_pressure(R·P·R^T, R·B)`` == ``parallel_pressure(P, B)``
#       — R acts on both arguments, scalar falls out unchanged.
#   (c) Same for ``perpendicular_pressure``.
#   (d) Algebraic: ``(P_par + 2 P_perp)/3`` equals the trace/3, so the
#       isotropic scalar pressure agrees whether derived from the tensor
#       trace directly or via the parallel/perpendicular decomposition.
"""Rotation invariance and CGL identity for pressure decomposition."""

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
from pypic.derived import (
    isotropic_pressure,
    parallel_pressure,
    perpendicular_pressure,
)
from tests.strategies import rotations

SHAPE = (3, 4)


def _symmetric_tensor_component() -> st.SearchStrategy[np.ndarray]:
    """Float64 array with finite bounded entries for one tensor slot."""
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


def _nonzero_b_component() -> st.SearchStrategy[np.ndarray]:
    """``parallel_pressure`` divides by ``|B|``; bound B away from zero
    on at least one component (via B_3 below) to avoid 0/0 NaNs that are
    physically meaningful but test the arithmetic pipeline rather than
    the rotation-invariance claim.
    """
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


@given(
    p11=_symmetric_tensor_component(),
    p22=_symmetric_tensor_component(),
    p33=_symmetric_tensor_component(),
    p12=_symmetric_tensor_component(),
    p13=_symmetric_tensor_component(),
    p23=_symmetric_tensor_component(),
    rotation=rotations(),
)
@settings(max_examples=40, deadline=None)
def test_pressure_trace_is_rotation_invariant(
    p11: np.ndarray,
    p22: np.ndarray,
    p33: np.ndarray,
    p12: np.ndarray,
    p13: np.ndarray,
    p23: np.ndarray,
    rotation: tuple,
) -> None:
    """``Tr(R·P·R^T) == Tr(P)`` — the first symmetric-tensor invariant.

    Generalizes the single-angle test in ``tests/test_transforms.py:191``
    to the full SO(3) via Rodrigues. Tolerance reflects the 9 multiply-
    adds per rotated component times the ~40-term trace sum.
    """
    r = np.asarray(rotation, dtype=np.float64)
    rp11, rp22, rp33, _, _, _ = rotate_pressure_tensor(p11, p22, p33, p12, p13, p23, r)
    trace_before = p11 + p22 + p33
    trace_after = rp11 + rp22 + rp33
    assert_allclose(trace_after, trace_before, rtol=1e-12, atol=1e-12)


@given(
    p11=_symmetric_tensor_component(),
    p22=_symmetric_tensor_component(),
    p33=_symmetric_tensor_component(),
    p12=_symmetric_tensor_component(),
    p13=_symmetric_tensor_component(),
    p23=_symmetric_tensor_component(),
    b1=_nonzero_b_component(),
    b2=_nonzero_b_component(),
    b3=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    rotation=rotations(),
)
@settings(max_examples=40, deadline=None)
def test_parallel_pressure_is_rotation_invariant(
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
    """``parallel_pressure(R·P·R^T, R·B)`` == ``parallel_pressure(P, B)``.

    The bilinear contraction $\\hat b^T P \\hat b$ is a scalar (rank-0
    tensor) under orthogonal coordinate changes; rotating P and B
    together must not perturb it beyond float64 roundoff. ``b3`` is kept
    away from zero to guarantee $|B| > 0$ so the unit-vector division
    in ``parallel_pressure`` stays finite.
    """
    b3_arr = np.full(SHAPE, b3, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    rp11, rp22, rp33, rp12, rp13, rp23 = rotate_pressure_tensor(
        p11, p22, p33, p12, p13, p23, r
    )
    rb1, rb2, rb3 = rotate_vector_components(b1, b2, b3_arr, r)

    before = parallel_pressure(p11, p22, p33, p12, p13, p23, b1, b2, b3_arr)
    after = parallel_pressure(rp11, rp22, rp33, rp12, rp13, rp23, rb1, rb2, rb3)
    assert_allclose(after, before, rtol=1e-11, atol=1e-11)


@given(
    p11=_symmetric_tensor_component(),
    p22=_symmetric_tensor_component(),
    p33=_symmetric_tensor_component(),
    p12=_symmetric_tensor_component(),
    p13=_symmetric_tensor_component(),
    p23=_symmetric_tensor_component(),
    b1=_nonzero_b_component(),
    b2=_nonzero_b_component(),
    b3=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    rotation=rotations(),
)
@settings(max_examples=40, deadline=None)
def test_perpendicular_pressure_is_rotation_invariant(
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
    """``perpendicular_pressure`` is likewise invariant — derived from
    the already-invariant trace and ``P_par``.
    """
    b3_arr = np.full(SHAPE, b3, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    rp11, rp22, rp33, rp12, rp13, rp23 = rotate_pressure_tensor(
        p11, p22, p33, p12, p13, p23, r
    )
    rb1, rb2, rb3 = rotate_vector_components(b1, b2, b3_arr, r)

    before = perpendicular_pressure(p11, p22, p33, p12, p13, p23, b1, b2, b3_arr)
    after = perpendicular_pressure(rp11, rp22, rp33, rp12, rp13, rp23, rb1, rb2, rb3)
    assert_allclose(after, before, rtol=1e-11, atol=1e-11)


@given(
    p11=_symmetric_tensor_component(),
    p22=_symmetric_tensor_component(),
    p33=_symmetric_tensor_component(),
    p12=_symmetric_tensor_component(),
    p13=_symmetric_tensor_component(),
    p23=_symmetric_tensor_component(),
    b1=_nonzero_b_component(),
    b2=_nonzero_b_component(),
    b3=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_isotropic_pressure_equals_par_plus_two_perp_over_three(
    p11: np.ndarray,
    p22: np.ndarray,
    p33: np.ndarray,
    p12: np.ndarray,
    p13: np.ndarray,
    p23: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: float,
) -> None:
    r"""$(P_\parallel + 2 P_\perp) / 3 = \operatorname{Tr}(\mathbf{P})/3 = P$.

    equations.md § 4, footnote [^9]. This ties together the three
    scalar quantities derived from the same pressure tensor — if any
    recipe drifts relative to the others (e.g. a factor-of-three or
    sign bug in one of the three functions), this identity breaks.
    """
    b3_arr = np.full(SHAPE, b3, dtype=np.float64)
    p_par = parallel_pressure(p11, p22, p33, p12, p13, p23, b1, b2, b3_arr)
    p_perp = perpendicular_pressure(p11, p22, p33, p12, p13, p23, b1, b2, b3_arr)
    p_iso = isotropic_pressure(p11, p22, p33)

    assert_allclose((p_par + 2.0 * p_perp) / 3.0, p_iso, rtol=1e-13, atol=1e-13)
