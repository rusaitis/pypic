# Source: CLAUDE.md "transform-chaining rule" + docs/schema.md § 2
#         ("Transforms can chain"). The chain operation is implemented by
#         ``compose_transforms`` in src/pypic/coordinates/transforms.py:182;
#         associativity is required for ``resolve_transform`` (same file,
#         line 249) to produce a unique answer regardless of which pair
#         of adjacent edges the graph traversal happens to fuse first.
# Claim: compose(compose(T1, T2), T3) == compose(T1, compose(T2, T3)).
# Inverse is already covered at tests/test_transforms.py:67; associativity
# is the orthogonal algebraic property.
"""Hypothesis property: ``compose_transforms`` is associative."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

from pypic.coordinates.transforms import FrameTransform, compose_transforms
from tests.strategies import frame_transforms


@given(
    t1=frame_transforms("A", "B"),
    t2=frame_transforms("B", "C"),
    t3=frame_transforms("C", "D"),
)
@settings(max_examples=100, deadline=None)
def test_compose_is_associative(
    t1: FrameTransform, t2: FrameTransform, t3: FrameTransform
) -> None:
    r"""$(T_1 \circ T_2) \circ T_3 = T_1 \circ (T_2 \circ T_3)$ — left- vs
    right-associated composition must agree on origin, rotation, and scale
    to float64 roundoff. Associativity is what lets
    ``resolve_transform`` walk a graph of registered edges and return a
    single canonical chained transform regardless of traversal order.
    """
    left = compose_transforms(compose_transforms(t1, t2), t3)
    right = compose_transforms(t1, compose_transforms(t2, t3))

    assert left.source_frame == right.source_frame == "A"
    assert left.target_frame == right.target_frame == "D"
    assert_allclose(left.origin, right.origin, rtol=0.0, atol=1e-10)
    assert_allclose(left.rotation_matrix, right.rotation_matrix, atol=1e-14)
    assert_allclose(left.scale, right.scale, rtol=1e-14)


@given(
    t1=frame_transforms("A", "B"),
    t2=frame_transforms("B", "C"),
    t3=frame_transforms("C", "D"),
)
@settings(max_examples=100, deadline=None)
def test_compose_matches_sequential_application(
    t1: FrameTransform, t2: FrameTransform, t3: FrameTransform
) -> None:
    r"""Applying the composed transform to a point equals applying each
    transform in sequence. Guards against sign/order bugs in
    ``compose_transforms`` that would survive algebraic simplification
    but corrupt real point coordinates.

    Point transform: $\mathbf{y} = s\, R\, (\mathbf{x} - \mathbf{o})$.
    """

    def apply(t: FrameTransform, x: np.ndarray) -> np.ndarray:
        return t.scale * t.rotation_matrix @ (x - np.asarray(t.origin))

    composed = compose_transforms(compose_transforms(t1, t2), t3)
    x = np.array([2.5, -1.3, 0.7])
    sequential = apply(t3, apply(t2, apply(t1, x)))
    direct = apply(composed, x)
    assert_allclose(direct, sequential, rtol=1e-10, atol=1e-10)
