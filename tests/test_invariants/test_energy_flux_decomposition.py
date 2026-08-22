# Source: docs/equations.md § 3 "Energy flux decomposition"
#         ("EF = KEF + EHF + q"); src/pypic/derived.py:1103
#         (kinetic_energy_flux_component, KEF_i = (1/2) n m |V|² V_i);
#         :1181 (enthalpy_flux_component, EHF_i = (γ/(γ-1)) P V_i).
# Claims:
#   (a) KEF_i(αV) = α³ · KEF_i(V). Encodes the |V|² · V_i structure:
#       a typo reducing to |V| · V_i (α²) or V_i alone (α) would be
#       caught by varying α.
#   (b) EHF is jointly linear in (P, V) — a single α-scaling of either
#       argument scales EHF by α; combined (αP, βV) scales by αβ.
# Dropped as ill-posed:
#   - "HF + KEF = EF bit-exact" and "q + EHF = HF bit-exact": Hypothesis
#     found ef=0.05, kef=1.0 produces a 1-ulp residual in (ef-kef)+kef-ef.
#     equations.md [^ef] "HF = EF - KEF is exact" means *physically* exact
#     (no closure assumption), not *float64*-exact. Weakening the assert
#     to a tolerance would make the test tautological — just "addition
#     partially inverts subtraction". Rather than weaken the property
#     to make it pass, these sub-tests are dropped.
"""Scaling identities for the energy flux decomposition recipes."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    enthalpy_flux_component,
    kinetic_energy_flux_component,
)

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
    v_comp=_bounded_array(),
    v1=_bounded_array(),
    v2=_bounded_array(),
    v3=_bounded_array(),
    rho_c=_positive_array(),
    charge=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    mass=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    alpha=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_kinetic_energy_flux_scales_as_velocity_cubed(
    v_comp: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    rho_c: np.ndarray,
    charge: float,
    mass: float,
    alpha: float,
) -> None:
    r"""$\mathrm{KEF}_i(\alpha\mathbf{V}) = \alpha^3 \cdot \mathrm{KEF}_i(\mathbf{V})$.

    From $\mathrm{KEF}_i = \tfrac{1}{2} n m |\mathbf{V}|^2 V_i$: two
    powers of $V$ from $|V|^2$ plus one from $V_i$ → cubic scaling
    under $\mathbf{V} \to \alpha\mathbf{V}$. A typo reducing $|V|^2$
    to $|V|$ would give $\alpha^2$; dropping the $V_i$ factor would
    give $\alpha^2$; either would fail this test under varied $\alpha$.
    """
    baseline = kinetic_energy_flux_component(v_comp, v1, v2, v3, rho_c, charge, mass)
    scaled = kinetic_energy_flux_component(
        alpha * v_comp, alpha * v1, alpha * v2, alpha * v3, rho_c, charge, mass
    )
    # |V|² = Σ(α·Vⱼ)² = α²·Σ Vⱼ², scaled by α·V_i: α³ overall.
    # Products span ~10³ before scaling, so roundoff ≈ 10 · eps · α³ · baseline.
    assert_allclose(scaled, (alpha**3) * baseline, rtol=1e-13, atol=1e-13)


@given(
    pressure=_positive_array(),
    v_comp=_bounded_array(),
    gamma=st.floats(min_value=1.1, max_value=3.0, allow_nan=False),
    alpha=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    beta=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_enthalpy_flux_is_bilinear_in_pressure_and_velocity(
    pressure: np.ndarray,
    v_comp: np.ndarray,
    gamma: float,
    alpha: float,
    beta: float,
) -> None:
    r"""EHF bilinearity: scaling $(P, V_i) \to (\alpha P, \beta V_i)$
    scales $\mathrm{EHF}_i$ by $\alpha\beta$.

    $\mathrm{EHF}_i = \frac{\gamma}{\gamma-1} P V_i$: linear in $P$,
    linear in $V_i$. Testing both scalings simultaneously catches any
    bug that breaks bilinearity (e.g. a spurious $P^2$ or missing $V$).
    """
    baseline = enthalpy_flux_component(pressure, v_comp, gamma=gamma)
    scaled = enthalpy_flux_component(alpha * pressure, beta * v_comp, gamma=gamma)
    assert_allclose(scaled, alpha * beta * baseline, rtol=1e-13, atol=1e-13)
