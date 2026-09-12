# Source: docs/equations.md § 2 (Thermodynamic Quantities table):
#           h      = γ P / ((γ - 1) ρ_m)                  # specific enthalpy
#           e_int  = P / ((γ - 1) ρ_m)                    # specific internal energy
#           e_th   = P / (γ - 1)                          # volumetric thermal energy
#           c_s    = √(γ P / ρ_m)                         # adiabatic sound speed
# Claim: direct algebraic consequences of those four formulas:
#   (i)  h - e_int = P / ρ_m           (pressure-to-density ratio)
#   (ii) h = γ · e_int                  (enthalpy = γ × internal energy)
#   (iii) e_th = ρ_m · e_int            (volumetric thermal = mass × specific internal)
#   (iv) c_s² = γ · (γ - 1) · e_int     (sound speed from specific energy)
# None of the four recipes knows about the others — if any drifts
# (bad factor, sign, swapped arg), these identities break.
"""Cross-consistency of enthalpy / internal energy / thermal density / c_s."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.derived import (
    enthalpy,
    internal_energy,
    sound_speed,
    thermal_energy_density,
)

SHAPE = (3, 4)


def _positive_array() -> st.SearchStrategy[np.ndarray]:
    """Strictly-positive finite arrays — P, ρ_m both live in R⁺."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=1e-3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


def _adiabatic_index() -> st.SearchStrategy[float]:
    """γ ∈ (1, 3]: excludes γ = 1 (the (γ-1) denominator would blow up)
    and stays inside the physical range — monoatomic 5/3, diatomic 7/5,
    isothermal → 1⁺, ultra-relativistic 4/3 — the span equations.md
    and docs/conventions.md both treat as canonical.
    """
    return st.floats(
        min_value=1.01,
        max_value=3.0,
        allow_nan=False,
        allow_infinity=False,
    )


@given(p=_positive_array(), rho=_positive_array(), gamma=_adiabatic_index())
@settings(max_examples=40)
def test_enthalpy_minus_internal_equals_pressure_over_density(
    p: np.ndarray, rho: np.ndarray, gamma: float
) -> None:
    r"""$h - e_{int} = \frac{P}{\rho_m}$ — the classical thermodynamic
    identity that falls out of combining the two recipes in
    ``equations.md § 2``.
    """
    assert_allclose(
        enthalpy(p, rho, gamma) - internal_energy(p, rho, gamma),
        p / rho,
        rtol=1e-13,
        atol=1e-13,
    )


@given(p=_positive_array(), rho=_positive_array(), gamma=_adiabatic_index())
@settings(max_examples=40)
def test_enthalpy_equals_gamma_times_internal_energy(
    p: np.ndarray, rho: np.ndarray, gamma: float
) -> None:
    r"""$h = \gamma \cdot e_{int}$ — both recipes share the same
    denominator $(γ - 1) ρ_m$ and differ only by the numerator factor
    of $\gamma$.
    """
    assert_allclose(
        enthalpy(p, rho, gamma),
        gamma * internal_energy(p, rho, gamma),
        rtol=1e-13,
        atol=1e-13,
    )


@given(p=_positive_array(), rho=_positive_array(), gamma=_adiabatic_index())
@settings(max_examples=40)
def test_thermal_energy_density_is_internal_energy_times_mass_density(
    p: np.ndarray, rho: np.ndarray, gamma: float
) -> None:
    r"""$e_{th} = \rho_m \cdot e_{int}$ — the volumetric thermal energy
    is the specific internal energy times mass density. Ties the
    *volumetric* ``e_th`` recipe (depends only on $P, \gamma$) to the
    *specific* ``e_int`` recipe (also depends on $\rho_m$). A bug in
    either (lost or spurious $(γ-1)$ factor) would break the identity.
    """
    assert_allclose(
        thermal_energy_density(p, gamma),
        rho * internal_energy(p, rho, gamma),
        rtol=1e-13,
        atol=1e-13,
    )


@given(p=_positive_array(), rho=_positive_array(), gamma=_adiabatic_index())
@settings(max_examples=40)
def test_sound_speed_squared_equals_gamma_gamma_minus_one_e_int(
    p: np.ndarray, rho: np.ndarray, gamma: float
) -> None:
    r"""$c_s^2 = \gamma (\gamma - 1) e_{int}$ — binds the MHD sound
    speed recipe to the internal-energy recipe through the textbook
    polytropic relation $c_s^2 = \gamma P / \rho_m$. If ``sound_speed``
    ever regresses (e.g. loses the $\sqrt{\gamma}$ factor), this
    identity catches it without duplicating the formula.
    """
    c_s = sound_speed(p, rho, gamma)
    assert_allclose(
        c_s**2,
        gamma * (gamma - 1.0) * internal_energy(p, rho, gamma),
        rtol=1e-13,
        atol=1e-13,
    )
