# Source: docs/equations.md § 2 (s = ln(P/n^γ) and
#         s_gyro = ln(P_par·P_perp²/n⁵)) + equations.md footnote [^1]
#         ("the gyrotropic exponent of 5 is independent of γ — it arises
#         from the CGL double-adiabatic invariants, not from the equation
#         of state. ... The coincidence with the numerator of 5/3 is
#         just that — a coincidence.") + docs/conventions.md §
#         "Gyrotropic entropy and CGL invariants" + src/pypic/derived.py:535
#         (entropy docstring) + src/pypic/derived.py:571
#         (gyrotropic_entropy docstring, "exponent 5 arises from combining
#         the two CGL invariants").
# Claims:
#   (a) At isotropy (P_par = P_perp = P), s_gyro(P,P,n) = 3·s(P,n,γ=5/3).
#       This is the CGL / 3D-γ coincidence made explicit — changing
#       either formula in isolation breaks it.
#   (b) Logarithmic scaling: s(α·P,n,γ) = s(P,n,γ) + ln(α).
#   (c) Logarithmic scaling: s(P,α·n,γ) = s(P,n,γ) - γ·ln(α).
#   (d) s_gyro(α·P_par, P_perp, n) = s_gyro(P_par, P_perp, n) + ln(α).
#   (e) s_gyro(P_par, α·P_perp, n) = s_gyro(P_par, P_perp, n) + 2·ln(α).
#   (f) s_gyro(P_par, P_perp, α·n) = s_gyro(P_par, P_perp, n) - 5·ln(α).
# Fresh invariant #13 after backlog exhaustion.
"""Algebraic identities between isotropic and gyrotropic entropy."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import entropy, gyrotropic_entropy

SHAPE = (3, 4)
GAMMA_3D = 5.0 / 3.0


def _positive_array(
    min_value: float = 1e-3, max_value: float = 1e3
) -> st.SearchStrategy[np.ndarray]:
    """Strictly positive finite arrays — entropy takes log(ratio) so
    zeros return NaN by design (see derived.py:568). Keep inputs in
    a range where the logs stay well-conditioned."""
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


def _positive_scalar() -> st.SearchStrategy[float]:
    """Strictly positive scalar for scaling factor α."""
    return st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    )


@given(pressure=_positive_array(), density=_positive_array())
@settings(max_examples=50)
def test_isotropy_identity_s_gyro_equals_three_s(
    pressure: np.ndarray, density: np.ndarray
) -> None:
    r"""$s_{gyro}(P, P, n) = 3\,s(P, n, \gamma=5/3)$ — the CGL / 3D-γ
    coincidence from equations.md [^1].

    Derivation: at isotropy, $s_{gyro} = \ln(P \cdot P^2 / n^5)
    = 3\ln(P/n^{5/3}) = 3\,s$. The exponent 5 in the CGL invariant
    matches $3 \cdot 5/3$ exactly — "a coincidence" per the docstring,
    but an exploitable algebraic identity. Any divergence between
    the two formulas in isolation breaks this test.
    """
    s_iso = entropy(pressure, density, gamma=GAMMA_3D)
    s_gyro = gyrotropic_entropy(pressure, pressure, density)
    assert_allclose(s_gyro, 3.0 * s_iso, rtol=1e-13, atol=1e-13)


@given(
    pressure=_positive_array(),
    density=_positive_array(),
    alpha=_positive_scalar(),
    gamma=st.floats(min_value=1.01, max_value=3.0, allow_nan=False),
)
@settings(max_examples=50)
def test_entropy_pressure_scaling(
    pressure: np.ndarray, density: np.ndarray, alpha: float, gamma: float
) -> None:
    r"""$s(\alpha P, n, \gamma) - s(P, n, \gamma) = \ln\alpha$.

    Multiplying pressure by α shifts entropy by ln α, independent of
    density and γ. Follows from $\ln(\alpha P / n^\gamma) =
    \ln\alpha + \ln(P/n^\gamma)$.
    """
    shifted = entropy(alpha * pressure, density, gamma=gamma)
    baseline = entropy(pressure, density, gamma=gamma)
    assert_allclose(shifted - baseline, np.log(alpha), rtol=1e-13, atol=1e-12)


@given(
    pressure=_positive_array(),
    density=_positive_array(),
    alpha=_positive_scalar(),
    gamma=st.floats(min_value=1.01, max_value=3.0, allow_nan=False),
)
@settings(max_examples=50)
def test_entropy_density_scaling(
    pressure: np.ndarray, density: np.ndarray, alpha: float, gamma: float
) -> None:
    r"""$s(P, \alpha n, \gamma) - s(P, n, \gamma) = -\gamma \ln\alpha$.

    Entropy scales as $-\gamma$ under density rescaling — the exponent
    $\gamma$ in the denominator of $P/n^\gamma$ made explicit.
    """
    shifted = entropy(pressure, alpha * density, gamma=gamma)
    baseline = entropy(pressure, density, gamma=gamma)
    assert_allclose(shifted - baseline, -gamma * np.log(alpha), rtol=1e-12, atol=1e-12)


@given(
    p_par=_positive_array(),
    p_perp=_positive_array(),
    density=_positive_array(),
    alpha=_positive_scalar(),
)
@settings(max_examples=50)
def test_gyrotropic_entropy_scaling_exponents(
    p_par: np.ndarray,
    p_perp: np.ndarray,
    density: np.ndarray,
    alpha: float,
) -> None:
    r"""Check the three CGL exponents (1, 2, 5) in
    $s_{gyro} = \ln(P_\parallel^1 P_\perp^2 / n^5)$ simultaneously.

    - Scaling $P_\parallel$ by $\alpha$ shifts $s_{gyro}$ by $\ln\alpha$.
    - Scaling $P_\perp$ by $\alpha$ shifts $s_{gyro}$ by $2\ln\alpha$.
    - Scaling $n$ by $\alpha$ shifts $s_{gyro}$ by $-5\ln\alpha$.

    A typo in any of the exponents (e.g. $P_\perp^3$ or $n^6$) would
    collapse the corresponding identity. Tests all three in one
    function (one assert-block per exponent) to keep them grouped and
    comparable — these are facets of a single CGL claim, not three
    independent behaviors.
    """
    baseline = gyrotropic_entropy(p_par, p_perp, density)

    shifted_par = gyrotropic_entropy(alpha * p_par, p_perp, density)
    shifted_perp = gyrotropic_entropy(p_par, alpha * p_perp, density)
    shifted_n = gyrotropic_entropy(p_par, p_perp, alpha * density)

    assert_allclose(shifted_par - baseline, np.log(alpha), rtol=1e-13, atol=1e-12)
    assert_allclose(
        shifted_perp - baseline, 2.0 * np.log(alpha), rtol=1e-13, atol=1e-12
    )
    assert_allclose(shifted_n - baseline, -5.0 * np.log(alpha), rtol=1e-13, atol=1e-11)
