# Source: src/pypic/diagnostics.py:88 (l2_relative_error,
#         ε_L2 = ‖a-b‖₂ / ‖b‖₂) + :143 (linf_error,
#         ε_L∞ = max|a-b|) + docs/equations.md § 7 Diagnostics table
#         + docs/conventions.md § "Error Norms and Divergence"
#         ("L2 is discrete, unweighted ... volume factors cancel;
#         L∞ is absolute, not relative — relative L∞ is misleading
#         near field nulls").
# Claims (pure metric properties of the two error norms):
#   (a) Scale invariance of the relative L2:
#       l2(α a, α b) == l2(a, b) for α ≠ 0 — the scale factor cancels
#       between numerator and denominator. Catches any accidental
#       absolute-norm regression.
#   (b) Homogeneity of the absolute L∞:
#       linf(α a, α b) == |α| · linf(a, b) — the abs-max-difference
#       scales linearly with a uniform rescaling, matching the
#       "absolute, not relative" docstring contract.
#   (c) Symmetry of L∞: linf(a, b) == linf(b, a) — max(|a-b|) =
#       max(|b-a|) trivially but catches any abs() typo.
#   (d) Self-distance is zero: l2(a, a) == 0 and linf(a, a) == 0
#       bit-exact (subtracting an array from itself is exact in FP).
# Note on l2 symmetry: NOT a claim — l2_relative_error normalizes by
# the reference, so swapping args swaps the denominator. Documented
# asymmetry (conventions.md) that we do not test as an invariant.
# Fresh invariant #18 after backlog exhaustion.
"""Scale/symmetry/self-distance metric properties of l2 / linf."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.diagnostics import l2_relative_error, linf_error

SHAPE = (3, 4)


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    """Signed finite array with bounded magnitude. Zero is allowed —
    the norms handle it, and self-distance tests explicitly need it
    to hit in the `a == b == 0` corner."""
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


def _nonzero_reference() -> st.SearchStrategy[np.ndarray]:
    """Reference arrays with ‖b‖₂ bounded away from zero — l2_relative_error
    returns inf if ‖b‖₂ = 0, which makes the scale-invariance test
    vacuous. Ranged in [0.1, 10] away from zero guarantees a finite
    denominator without restricting sign.
    """
    positive = arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=0.1,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )
    # Randomly flip signs per-element so the reference still exercises
    # mixed-sign fields while keeping every element nonzero.
    sign_flip = arrays(
        dtype=np.int8,
        shape=SHAPE,
        elements=st.sampled_from([-1, 1]),
    )
    return st.tuples(positive, sign_flip).map(lambda pair: pair[0] * pair[1])


def _nonzero_scalar() -> st.SearchStrategy[float]:
    """Scalar α ≠ 0 for scale tests. Sign random, magnitude in [1e-3, 1e3]."""
    return (
        st.floats(
            min_value=1e-3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        )
        .flatmap(lambda x: st.sampled_from([x, -x]))
    )


@given(
    a=_bounded_array(),
    b=_nonzero_reference(),
    alpha=_nonzero_scalar(),
)
@settings(max_examples=50, deadline=None)
def test_l2_relative_error_is_scale_invariant(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> None:
    r"""$\varepsilon_{L_2}(\alpha a, \alpha b) = \varepsilon_{L_2}(a, b)$.

    The factor $|\alpha|$ multiplies both the numerator $\|\alpha(a-b)\|_2$
    and the denominator $\|\alpha b\|_2$, cancelling exactly. Any
    accidental switch to an absolute-L2 norm (e.g. dropping the
    denominator in a refactor) would scale linearly with $|\alpha|$
    and be caught by varying $\alpha$ across $[10^{-3}, 10^3]$.
    """
    baseline = l2_relative_error(a, b)
    scaled = l2_relative_error(alpha * a, alpha * b)
    # Three sqrt(sum(x²))s (two norms, one ratio) on values up to 10·|α|
    # — float64 roundoff ≈ 10·eps. 1e-13 rtol absorbs it.
    assert_allclose(scaled, baseline, rtol=1e-13, atol=1e-14)


@given(
    a=_bounded_array(),
    b=_bounded_array(),
    alpha=_nonzero_scalar(),
)
@settings(max_examples=50, deadline=None)
def test_linf_error_scales_with_alpha_magnitude(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> None:
    r"""$\varepsilon_{L_\infty}(\alpha a, \alpha b) = |\alpha| \cdot
    \varepsilon_{L_\infty}(a, b)$.

    The absolute max-norm scales linearly with the overall rescaling
    factor. Sign of $\alpha$ does not matter (abs-inside). A sign
    leak (e.g. `max(c - r)` instead of `max(abs(c - r))`) would flip
    the comparison under $\alpha < 0$.
    """
    baseline = linf_error(a, b)
    scaled = linf_error(alpha * a, alpha * b)
    assert_allclose(scaled, abs(alpha) * baseline, rtol=1e-13, atol=1e-13)


@given(a=_bounded_array(), b=_bounded_array())
@settings(max_examples=50, deadline=None)
def test_linf_error_is_symmetric(a: np.ndarray, b: np.ndarray) -> None:
    r"""$\varepsilon_{L_\infty}(a, b) = \varepsilon_{L_\infty}(b, a)$.

    max-abs of $(a-b)$ equals max-abs of $(b-a)$ — a trivial
    consequence of absolute value, but a regression guard that
    linf_error does not accidentally become signed or reference-normalized.
    """
    ab = linf_error(a, b)
    ba = linf_error(b, a)
    assert_allclose(ab, ba, rtol=0, atol=0)


@given(a=_bounded_array())
@settings(max_examples=30, deadline=None)
def test_self_distance_is_zero(a: np.ndarray) -> None:
    r"""$\varepsilon_{L_2}(a, a) = 0$ and $\varepsilon_{L_\infty}(a, a) = 0$.

    Subtracting an array from itself is exact in IEEE 754, so both
    norms return 0 bit-exact when the denominator is well-conditioned.
    For $l_2$, the denominator $\|a\|_2$ must be strictly positive in
    float64 to avoid a 0/0 NaN — documented at diagnostics.py:119-121
    ("nan if both are all zeros or no valid cells remain"). Skip cases
    where $\sum a^2$ underflows to 0 (which includes the all-zero
    corner AND Hypothesis-generated subnormal inputs like
    $a \sim 10^{-308}$ where squaring underflows below
    $2.2 \times 10^{-308}$). $l_\infty$ is robust to the underflow
    regime — it uses $\max|0|$, not a squared sum.
    """
    # Match the diagnostics.py NaN-on-zero-reference contract exactly.
    if np.sum(a * a) == 0.0:
        assert linf_error(a, a) == 0.0
        return
    assert l2_relative_error(a, a) == 0.0
    assert linf_error(a, a) == 0.0
