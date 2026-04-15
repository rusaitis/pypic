"""Hypothesis strategies for pypic invariant tests.

Each strategy documents the source claim it helps test — schema.md, a
CLAUDE.md rule, or a docstring invariant. Narrowing a strategy (shrinking
its range to avoid a counterexample) requires citing the degenerate case
in a comment; see ``autoresearcher-pypic.md`` for the loop rules.
"""

from __future__ import annotations

from hypothesis import strategies as st

from pypic.units import Normalization

BASE_QUANTITIES: tuple[str, ...] = (
    "length",
    "time",
    "velocity",
    "b_field",
    "e_field",
    "density",
)
"""The six base quantities ``Normalization.normalize`` accepts (units.py:20)."""

COMPOUND_QUANTITIES: tuple[str, ...] = (
    "dimensionless",
    "pressure",
    "temperature",
    "energy_density",
    "current_density",
    "frequency",
    "mass_density",
    "charge_density",
    "poynting_flux",
    "energy_flux",
    "b_field_per_length",
    "e_field_per_length",
    "velocity_per_length",
    "specific_energy",
    "power_density",
    "four_velocity",
)
"""Compound quantities that ``si_factor`` resolves (units.py:_COMPOUND_FACTORS)."""


def base_quantities() -> st.SearchStrategy[str]:
    """One of the six base quantity names."""
    return st.sampled_from(BASE_QUANTITIES)


def compound_quantities() -> st.SearchStrategy[str]:
    """One of the compound quantity names recognized by ``si_factor``."""
    return st.sampled_from(COMPOUND_QUANTITIES)


def finite_physical_floats(
    min_value: float = 1e-20, max_value: float = 1e20
) -> st.SearchStrategy[float]:
    """Finite positive floats over a wide physical range.

    Excludes zero (round-trip is trivially exact) and non-finite values.
    The ~40-decade range stresses normalization without pushing any
    intermediate product (``x * ref`` or ``x / ref``) into denormals
    when combined with the normalization strategies below.
    """
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    )


def pic_standard_norms() -> st.SearchStrategy[Normalization]:
    """``Normalization.pic_standard`` over (n, m, q) triples.

    Ranges chosen to cover solar-wind through laboratory-plasma densities
    and electron-through-heavy-ion species. All reference values stay
    positive and finite after the internal ``omega_ref`` computation.
    """
    return st.builds(
        Normalization.pic_standard,
        st.floats(1e10, 1e22, allow_nan=False, allow_infinity=False),
        st.floats(1e-31, 1e-25, allow_nan=False, allow_infinity=False),
        st.floats(1e-20, 1e-18, allow_nan=False, allow_infinity=False),
    )


def pic_electron_norms() -> st.SearchStrategy[Normalization]:
    """``Normalization.pic_electron(n_e)`` over plasma densities."""
    return st.builds(
        Normalization.pic_electron,
        st.floats(1e10, 1e22, allow_nan=False, allow_infinity=False),
    )


def mhd_standard_norms() -> st.SearchStrategy[Normalization]:
    """``Normalization.mhd_standard(l_0, rho_0, b_0)`` over MHD regimes."""
    return st.builds(
        Normalization.mhd_standard,
        st.floats(1e3, 1e10, allow_nan=False, allow_infinity=False),
        st.floats(1e-22, 1e-6, allow_nan=False, allow_infinity=False),
        st.floats(1e-15, 1.0, allow_nan=False, allow_infinity=False),
    )


def identity_norms() -> st.SearchStrategy[Normalization]:
    """Singleton strategy emitting ``Normalization.identity()``."""
    return st.just(Normalization.identity())


def normalizations() -> st.SearchStrategy[Normalization]:
    """Uniformly sample across all four standard Normalization constructors."""
    return st.one_of(
        pic_standard_norms(),
        pic_electron_norms(),
        mhd_standard_norms(),
        identity_norms(),
    )
