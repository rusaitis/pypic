# Source: src/pypic/derived.py:373 ("$\mathbf{S} = \mathbf{E} \times \mathbf{B}$")
#         — the cross product is anti-commutative: $A \times B = -(B \times A)$.
#         src/pypic/derived.py:1803 firehose_parameter docstring:
#         "$\mathcal{F} = (P_\parallel - P_\perp)/(B^2/2) - 1$"
#         src/pypic/derived.py:1838 mirror_parameter docstring:
#         "$\mathcal{M} = P_\perp / P_\parallel - 1 - 1/\beta_\perp$"
#         docs/equations.md § 9 footnote [^14]: "firehose unstable when
#         P_par - P_perp > B²/2; mirror unstable when
#         P_perp/P_par > 1 + 1/beta_perp".
# Claim:
#  (a) poynting_flux(E, B) == -poynting_flux(B, E) — anti-commutativity
#      of the cross product. Guards against sign/order bugs in the
#      component expansion.
#  (b) At isotropy (P_par == P_perp): firehose < 0 and mirror < 0
#      for any finite B and P > 0. At isotropy the plasma is ALWAYS
#      stable to both instabilities — a sign or denominator bug would
#      break this. Tests the analytical limit, not the formula itself.
# Fresh invariant #6 after backlog exhaustion.
"""Poynting-flux anti-commutativity and stability at pressure isotropy."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import firehose_parameter, mirror_parameter, poynting_flux

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


@given(
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50)
def test_poynting_flux_anti_commutativity(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{E} \times \mathbf{B} = -(\mathbf{B} \times \mathbf{E})$.

    Swapping the two vector arguments must negate every component of
    the result. This is not a tautology: the implementation
    (``derived.py:421``) is an explicit expansion of the cross product
    into six terms, where a single sign flip would break anti-commutativity
    while still producing a plausible-looking vector.
    """
    s1, s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
    t1, t2, t3 = poynting_flux(b1, b2, b3, e1, e2, e3)
    assert_allclose(s1, -t1, rtol=0, atol=1e-14)
    assert_allclose(s2, -t2, rtol=0, atol=1e-14)
    assert_allclose(s3, -t3, rtol=0, atol=1e-14)


@given(
    e1=_bounded_array(),
    e2=_bounded_array(),
    e3=_bounded_array(),
    b1=_bounded_array(),
    b2=_bounded_array(),
    b3=_bounded_array(),
)
@settings(max_examples=50)
def test_poynting_flux_dot_b_is_zero(
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> None:
    r"""$\mathbf{S} \cdot \mathbf{B} = (\mathbf{E} \times \mathbf{B})
    \cdot \mathbf{B} = 0$ — the cross product is always perpendicular
    to both input vectors. Guards against any term in the expansion that
    would introduce a component parallel to B.
    """
    s1, s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
    dot = s1 * b1 + s2 * b2 + s3 * b3
    assert_allclose(dot, 0.0, atol=1e-12)


@given(p=_positive_array(), b=_positive_array())
@settings(max_examples=50)
def test_firehose_stable_at_isotropy(p: np.ndarray, b: np.ndarray) -> None:
    r"""At isotropy ($P_\parallel = P_\perp = P$):
    $\mathcal{F} = (P - P)/(B^2/2) - 1 = -1 < 0$ — always stable.

    Independent of pressure magnitude and field strength.
    """
    result = firehose_parameter(p, p, b)
    assert_allclose(result, -1.0, rtol=0, atol=1e-14)


@given(p=_positive_array(), b=_positive_array())
@settings(max_examples=50)
def test_mirror_stable_at_isotropy(p: np.ndarray, b: np.ndarray) -> None:
    r"""At isotropy ($P_\parallel = P_\perp = P$):
    $\mathcal{M} = P/P - 1 - 1/\beta_\perp = -1/\beta_\perp < 0$ —
    always stable for finite $\beta$.

    The exact value is $-B^2/(2P)$ which is always negative for positive
    $P$ and $B$.
    """
    result = mirror_parameter(p, p, b)
    expected = -(b**2) / (2.0 * p)
    assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    assert np.all(result < 0.0)
