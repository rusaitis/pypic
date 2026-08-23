# Source: docs/equations.md § 5 (β = 2P/B², β_e = 2Pe/B², β_i = 2Pi/B²)
#         + docs/schema.md § 3 ("Electron/ion convenience aliases" + two-
#         species convention) + src/pypic/compute.py:135-137 (registry
#         wires all three betas to ``plasma_beta(P, |B|)`` with
#         species-appropriate pressures) + src/pypic/compute.py:279
#         (``P`` dispatches to ``total_pressure(Pe, Pi)`` when a total P
#         is not stored directly).
# Claim: ``ds.compute("beta") == ds.compute("beta_e") + ds.compute("beta_i")``
#        for any dataset whose total pressure is the two-species sum
#        ``P = Pe + Pi``. Follows algebraically from linearity of
#        ``2 · P / B²`` in its numerator. Guards against:
#          (a) a drift in the denominator (e.g. β_e using B² while β
#              uses |B|·|B| with a different precision path),
#          (b) a stray factor (missing/extra 2),
#          (c) ``total_pressure(Pe, Pi)`` accidentally becoming a
#              non-linear combination (geometric mean, max, etc.).
# Fresh invariant #21 after backlog exhaustion.
"""Two-fluid plasma-beta additivity through the compute registry."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from tests._helpers import make_test_dataset

SHAPE = (3, 4, 2)


def _positive_array(
    min_value: float = 1e-3, max_value: float = 1e3
) -> st.SearchStrategy[np.ndarray]:
    """Strictly positive arrays — Pe, Pi, |B| never legitimately zero
    in this identity (β is singular at B=0)."""
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
    pe=_positive_array(),
    pi=_positive_array(),
    b_mag=_positive_array(),
)
@settings(max_examples=40)
def test_beta_equals_sum_of_species_betas(
    pe: np.ndarray, pi: np.ndarray, b_mag: np.ndarray
) -> None:
    r"""$\beta = \beta_e + \beta_i$ when $P = P_e + P_i$.

    The three ``plasma_beta`` recipes share the denominator $B^2$;
    linearity of the numerator makes the split exact. Datasets carry
    only per-species pressures here, so ``ds.compute("beta")`` forces
    the composition path through ``total_pressure(Pe, Pi)`` —
    exercising the registered ``P`` dispatch together with the three
    β recipes in one shot.
    """
    ds = make_test_dataset(
        {
            "Pe": pe,
            "Pi": pi,
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
        },
        shape=SHAPE,
    )
    beta = ds.compute("beta")
    beta_e = ds.compute("beta_e")
    beta_i = ds.compute("beta_i")
    # Each β is 2·P/B², so β = β_e + β_i is algebraic. The only
    # roundoff source is the shared |B|·|B| intermediate, which is
    # computed once per recipe; tolerance ≈ few · eps · β.
    assert_allclose(beta, beta_e + beta_i, rtol=1e-13, atol=1e-14)


@given(
    pe=_positive_array(),
    pi=_positive_array(),
    b_mag=_positive_array(),
    alpha=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=30)
def test_beta_is_linear_in_pressure_split(
    pe: np.ndarray, pi: np.ndarray, b_mag: np.ndarray, alpha: float
) -> None:
    r"""Scaling $P_e \to \alpha P_e$ shifts only $\beta_e$ (and the
    total $\beta$) by $\alpha \cdot \beta_e$ — $\beta_i$ is untouched.

    A subtle per-species bug where, say, ``beta_e`` silently includes
    the ion pressure would make $\beta_i$ also change under the
    electron rescaling and fail this test.
    """
    ds_base = make_test_dataset(
        {
            "Pe": pe,
            "Pi": pi,
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
        },
        shape=SHAPE,
    )
    ds_scaled = make_test_dataset(
        {
            "Pe": alpha * pe,
            "Pi": pi,
            "B_1": b_mag,
            "B_2": np.zeros_like(b_mag),
            "B_3": np.zeros_like(b_mag),
        },
        shape=SHAPE,
    )
    beta_i_base = ds_base.compute("beta_i")
    beta_i_scaled = ds_scaled.compute("beta_i")
    assert_allclose(beta_i_scaled, beta_i_base, rtol=0, atol=0)

    beta_e_base = ds_base.compute("beta_e")
    beta_e_scaled = ds_scaled.compute("beta_e")
    assert_allclose(beta_e_scaled, alpha * beta_e_base, rtol=1e-13, atol=1e-14)
