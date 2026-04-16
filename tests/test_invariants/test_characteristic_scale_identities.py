# Source: docs/equations.md § 5 (Characteristic Scales table):
#   d  = c / ω_p          (skin depth)
#   r  = v_th / ω_c       (gyroradius)
#   λ_D = sqrt(T / (nq²)) (Debye length)
# Direct algebraic consequences linking the independently-authored recipes:
#   (i)   skin_depth × plasma_frequency == c
#   (ii)  gyroradius × gyrofrequency == thermal_speed
#   (iii) debye_length × plasma_frequency == thermal_speed
#         (since λ_D = v_th / ω_p)
#   (iv)  debye_length == thermal_speed / plasma_frequency
# Each function is implemented in src/pypic/derived.py as a separate
# pure function; a sign, factor, or exponent bug in any one of them
# (e.g. missing sqrt, q vs |q|, m vs 1/m) would break the identity
# while passing the function's own unit test. Fresh invariant #5 after
# backlog exhaustion.
"""Cross-consistency of characteristic plasma scales."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    debye_length,
    gyrofrequency,
    gyroradius,
    plasma_frequency,
    skin_depth,
    thermal_speed,
)

SHAPE = (3, 4)


def _positive_array(
    min_value: float = 0.1, max_value: float = 1e2
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


def _species_scalar(
    min_value: float = 0.01, max_value: float = 1e3
) -> st.SearchStrategy[float]:
    """Positive scalar for |q| or m. Sign of charge doesn't matter since
    all recipes take |q| internally; we pass the raw positive value.
    """
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    )


@given(
    n=_positive_array(),
    q=_species_scalar(),
    m=_species_scalar(),
    c=st.floats(min_value=0.5, max_value=100.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_skin_depth_times_plasma_frequency_is_c(
    n: np.ndarray, q: float, m: float, c: float
) -> None:
    r"""$d \cdot \omega_p = c$ — from $d = c / \omega_p$."""
    d = skin_depth(n, q, m, c)
    omega_p = plasma_frequency(n, q, m)
    assert_allclose(d * omega_p, c, rtol=1e-13, atol=1e-13)


@given(
    temperature=_positive_array(),
    b=_positive_array(),
    q=_species_scalar(),
    m=_species_scalar(),
)
@settings(max_examples=40, deadline=None)
def test_gyroradius_times_gyrofrequency_is_thermal_speed(
    temperature: np.ndarray, b: np.ndarray, q: float, m: float
) -> None:
    r"""$r \cdot \omega_c = v_{th}$ — from $r = v_{th}/\omega_c$."""
    r = gyroradius(temperature, b, q, m)
    omega_c = gyrofrequency(b, q, m)
    v_th = thermal_speed(temperature, m)
    assert_allclose(r * omega_c, v_th, rtol=1e-13, atol=1e-13)


@given(
    temperature=_positive_array(),
    n=_positive_array(),
    q=_species_scalar(),
    m=_species_scalar(),
)
@settings(max_examples=40, deadline=None)
def test_debye_length_is_thermal_speed_over_plasma_frequency(
    temperature: np.ndarray, n: np.ndarray, q: float, m: float
) -> None:
    r"""$\lambda_D = v_{th} / \omega_p$ — binds Debye length, thermal
    speed, and plasma frequency through three separate recipes.

    Derivation: $\lambda_D = \sqrt{T/(nq^2)}$, $v_{th} = \sqrt{T/m}$,
    $\omega_p = \sqrt{nq^2/m}$, so $v_{th}/\omega_p = \sqrt{T/m}
    / \sqrt{nq^2/m} = \sqrt{T/(nq^2)} = \lambda_D$.
    """
    lam = debye_length(temperature, n, q)
    v_th = thermal_speed(temperature, m)
    omega_p = plasma_frequency(n, q, m)
    assert_allclose(lam, v_th / omega_p, rtol=1e-13, atol=1e-13)


