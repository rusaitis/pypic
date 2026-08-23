# Source: docs/equations.md § 5 ("Characteristic Scales") — β, v_A, Alfvén
# and magnetosonic Mach numbers have closed-form algebraic definitions in
# pypic's SI-rationalized normalization (μ₀ = 1).
# Covers backlog #2b: compute-dispatch parity for the plasma-parameter
# recipes (beta family, v_A, M_A).
"""Algebraic identities for plasma-parameter recipes in ``compute._REGISTRY``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tests._helpers import make_test_dataset

SHAPE = (3, 4, 2)


def _positive_array(
    min_value: float = 1e-6, max_value: float = 1e3
) -> st.SearchStrategy[np.ndarray]:
    """Strictly-positive finite arrays. Pressures, densities, |B| share this."""
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


@pytest.mark.parametrize(
    ("name", "pressure_field"),
    [("beta", "P"), ("beta_e", "Pe"), ("beta_i", "Pi")],
)
@given(p=_positive_array(), b_mag=_positive_array())
@settings(max_examples=40)
def test_beta_identity(
    name: str, pressure_field: str, p: np.ndarray, b_mag: np.ndarray
) -> None:
    """β = 2 P / B² — equations.md § 5, pypic normalization (μ₀ = 1)."""
    # Seed |B| by giving B_1 = b_mag and B_2 = B_3 = 0 so |B| = b_mag exactly.
    ds = make_test_dataset(
        {
            pressure_field: p,
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
        },
        shape=SHAPE,
    )
    beta = ds.compute(name)
    expected = 2.0 * p / (b_mag * b_mag)
    np.testing.assert_allclose(beta, expected, rtol=1e-13)


@given(b_mag=_positive_array(), rho_m=_positive_array())
@settings(max_examples=40)
def test_alfven_speed_identity(b_mag: np.ndarray, rho_m: np.ndarray) -> None:
    """v_A = |B| / sqrt(ρ_m) — equations.md § 5 (non-relativistic branch)."""
    ds = make_test_dataset(
        {
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
            "rho_m": rho_m,
        },
        shape=SHAPE,
    )
    v_a = ds.compute("v_A")
    expected = b_mag / np.sqrt(rho_m)
    np.testing.assert_allclose(v_a, expected, rtol=1e-13)


@given(v_mag=_positive_array(), b_mag=_positive_array(), rho_m=_positive_array())
@settings(max_examples=40)
def test_alfven_mach_identity(
    v_mag: np.ndarray, b_mag: np.ndarray, rho_m: np.ndarray
) -> None:
    """M_A = |V| / v_A — composition of two registered recipes."""
    ds = make_test_dataset(
        {
            "V_1": v_mag,
            "V_2": np.zeros_like(v_mag),
            "V_3": np.zeros_like(v_mag),
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
            "rho_m": rho_m,
        },
        shape=SHAPE,
    )
    m_a = ds.compute("M_A")
    expected = v_mag / (b_mag / np.sqrt(rho_m))
    np.testing.assert_allclose(m_a, expected, rtol=1e-13)
