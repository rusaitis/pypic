"""String-based dispatch for derived quantities on FieldDataset.

Maps short names (``"|B|"``, ``"beta"``, ``"v_A"``, ...) to pure functions
in ``derived.py``, ``diagnostics.py``, and ``operators.py``. Never imports
FieldDataset — avoids circular dependencies. FieldDataset methods call into
this module via deferred imports.
"""

from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from scipy import constants

from pypic import derived, diagnostics
from pypic._aliases import (
    _COMPUTE_ALIASES,
    _get_field_alias_fallback,
)
from pypic.coordinates import operators
from pypic.coordinates.geometry import GeometryType
from pypic.fields import _FIELD_INFO, _SPECIES_QUANTITY_PATTERNS, QuantityType

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray
    from pypic.units import Normalization


class _SpeciesArgs(StrEnum):
    """Describes which species parameters a dynamic recipe needs."""

    CHARGE_MASS = "charge_mass"
    MASS_ONLY = "mass_only"
    CHARGE_ONLY = "charge_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class _Recipe:
    func: Callable[..., Any]
    fields: tuple[str, ...]
    species_index: int | None = None
    needs_grid: bool = False
    needs_gamma: bool = False
    needs_c: bool = False
    component: int | None = None
    species_args: _SpeciesArgs | None = None
    # When True, the dataset's geometry is passed to ``func`` as a
    # ``geometry=`` kwarg. Used by operator-backed recipes
    # (``div_B``, ``div_E``, ``curl_B*``, ``vort*``) so that the
    # FieldDataset path is automatically correct when spherical /
    # cylindrical operators are eventually implemented. ``psi`` opts
    # out because ``magnetic_flux_function`` has its own (Cartesian-only)
    # generalization path documented in its docstring.
    passes_geometry: bool = False
    # When True and ``physics.relativistic`` is set on the dataset,
    # ``c`` is injected as a keyword argument, activating the
    # relativistic branch of functions with a ``c=None`` kwarg.
    supports_relativistic: bool = False


_PRESSURE_TENSOR_FIELDS = ("P11", "P22", "P33", "P12", "P13", "P23")
_PRESSURE_TENSOR_AND_B = (*_PRESSURE_TENSOR_FIELDS, "B1", "B2", "B3")

_REGISTRY: dict[str, _Recipe] = {
    # Magnitudes
    "|B|": _Recipe(derived.magnetic_field_magnitude, ("B1", "B2", "B3")),
    "|E|": _Recipe(derived.electric_field_magnitude, ("E1", "E2", "E3")),
    "|J|": _Recipe(derived.current_density_magnitude, ("J1", "J2", "J3")),
    "|V|": _Recipe(derived.velocity_magnitude, ("V1", "V2", "V3")),
    "|Ve|": _Recipe(derived.velocity_magnitude, ("Ve1", "Ve2", "Ve3")),
    # Plasma parameters
    "beta": _Recipe(derived.plasma_beta, ("P", "|B|")),
    "beta_e": _Recipe(derived.plasma_beta, ("Pe", "|B|")),
    "beta_i": _Recipe(derived.plasma_beta, ("Pi", "|B|")),
    "v_A": _Recipe(derived.alfven_speed, ("|B|", "rho_m"), supports_relativistic=True),
    "c_s": _Recipe(
        derived.sound_speed, ("P", "rho_m"), needs_gamma=True,
        supports_relativistic=True,
    ),
    "c_ia": _Recipe(
        derived.ion_acoustic_speed,
        ("Te", "Ti"),
        species_index=1,
        species_args=_SpeciesArgs.MASS_ONLY,
    ),
    "v_ms": _Recipe(
        derived.magnetosonic_speed, ("v_A", "c_s"), supports_relativistic=True,
    ),
    "M_A": _Recipe(derived.alfven_mach, ("|V|", "v_A")),
    "M_ms": _Recipe(derived.magnetosonic_mach, ("|V|", "v_ms")),
    # Energies
    "e_B": _Recipe(derived.magnetic_energy_density, ("|B|",)),
    "e_E": _Recipe(derived.electric_energy_density, ("|E|",)),
    "e_k": _Recipe(
        derived.kinetic_energy_density, ("rho_m", "|V|"),
        supports_relativistic=True,
    ),
    "e_th": _Recipe(derived.thermal_energy_density, ("P",), needs_gamma=True),
    "e_th_trace": _Recipe(
        derived.thermal_energy_density_trace,
        ("P11", "P22", "P33"),
    ),
    # Thermodynamic
    "h": _Recipe(
        derived.enthalpy, ("P", "rho_m"), needs_gamma=True,
        supports_relativistic=True,
    ),
    "h_rel": _Recipe(
        derived.relativistic_enthalpy,
        ("P", "rho_m"),
        needs_gamma=True,
        needs_c=True,
    ),
    "gamma_L": _Recipe(derived.lorentz_factor, ("|V|",), needs_c=True),
    "sigma": _Recipe(derived.magnetization, ("|B|", "rho_m"), needs_c=True),
    "e_int": _Recipe(derived.internal_energy, ("P", "rho_m"), needs_gamma=True),
    "s": _Recipe(derived.entropy, ("P", "rho_m"), needs_gamma=True),
    "s_e": _Recipe(derived.entropy, ("Pe", "n_s0"), needs_gamma=True),
    "s_i": _Recipe(derived.entropy, ("Pi", "n_s1"), needs_gamma=True),
    "s_gyro_e": _Recipe(derived.gyrotropic_entropy, ("P_par", "P_perp", "n_s0")),
    "s_gyro_i": _Recipe(derived.gyrotropic_entropy, ("P_par", "P_perp", "n_s1")),
    # Poynting flux (tuple return — component selects)
    "S1": _Recipe(
        derived.poynting_flux,
        ("E1", "E2", "E3", "B1", "B2", "B3"),
        component=0,
    ),
    "S2": _Recipe(
        derived.poynting_flux,
        ("E1", "E2", "E3", "B1", "B2", "B3"),
        component=1,
    ),
    "S3": _Recipe(
        derived.poynting_flux,
        ("E1", "E2", "E3", "B1", "B2", "B3"),
        component=2,
    ),
    # Enthalpy flux (total, MHD): EHF_i = (gamma/(gamma-1)) P V_i
    "EHF1": _Recipe(
        derived.enthalpy_flux_component,
        ("P", "V1"),
        needs_gamma=True,
    ),
    "EHF2": _Recipe(
        derived.enthalpy_flux_component,
        ("P", "V2"),
        needs_gamma=True,
    ),
    "EHF3": _Recipe(
        derived.enthalpy_flux_component,
        ("P", "V3"),
        needs_gamma=True,
    ),
    # Species-dependent: electrons (species 0)
    "omega_pe": _Recipe(
        derived.plasma_frequency,
        ("n_s0",),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "omega_ce": _Recipe(
        derived.gyrofrequency,
        ("|B|",),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "d_e": _Recipe(
        derived.skin_depth,
        ("n_s0",),
        species_index=0,
        needs_c=True,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "v_th_e": _Recipe(
        derived.thermal_speed,
        ("Te",),
        species_index=0,
        species_args=_SpeciesArgs.MASS_ONLY,
        supports_relativistic=True,
    ),
    "r_e": _Recipe(
        derived.gyroradius,
        ("Te", "|B|"),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "lambda_D": _Recipe(
        derived.debye_length,
        ("Te", "n_s0"),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_ONLY,
    ),
    # Species-dependent: ions (species 1)
    "omega_pi": _Recipe(
        derived.plasma_frequency,
        ("n_s1",),
        species_index=1,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "omega_ci": _Recipe(
        derived.gyrofrequency,
        ("|B|",),
        species_index=1,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "d_i": _Recipe(
        derived.skin_depth,
        ("n_s1",),
        species_index=1,
        needs_c=True,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    "v_th_i": _Recipe(
        derived.thermal_speed,
        ("Ti",),
        species_index=1,
        species_args=_SpeciesArgs.MASS_ONLY,
        supports_relativistic=True,
    ),
    "r_i": _Recipe(
        derived.gyroradius,
        ("Ti", "|B|"),
        species_index=1,
        species_args=_SpeciesArgs.CHARGE_MASS,
    ),
    # Total pressure from partial pressures (Pe, Pi each resolve from
    # per-species tensor trace when not directly available).
    "P": _Recipe(derived.total_pressure, ("Pe", "Pi")),
    "Pe": _Recipe(derived.isotropic_pressure, ("P11_s0", "P22_s0", "P33_s0")),
    "Pi": _Recipe(derived.isotropic_pressure, ("P11_s1", "P22_s1", "P33_s1")),
    # Pressure tensor decomposition
    "P_par": _Recipe(derived.parallel_pressure, _PRESSURE_TENSOR_AND_B),
    "P_perp": _Recipe(derived.perpendicular_pressure, _PRESSURE_TENSOR_AND_B),
    "agyrotropy": _Recipe(derived.agyrotropy, _PRESSURE_TENSOR_AND_B),
    # Grid-dependent diagnostics
    "div_B": _Recipe(
        diagnostics.div_b,
        ("B1", "B2", "B3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    "div_E": _Recipe(
        diagnostics.div_e,
        ("E1", "E2", "E3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    # Curl of B (tuple return — component selects)
    "curl_B1": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=0,
        passes_geometry=True,
    ),
    "curl_B2": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=1,
        passes_geometry=True,
    ),
    "curl_B3": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=2,
        passes_geometry=True,
    ),
    # Vorticity (tuple return — component selects)
    "vort1": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=0,
        passes_geometry=True,
    ),
    "vort2": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=1,
        passes_geometry=True,
    ),
    "vort3": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=2,
        passes_geometry=True,
    ),
    # Vorticity magnitude — depends on vort1/2/3
    "|vort|": _Recipe(derived.velocity_magnitude, ("vort1", "vort2", "vort3")),
    # Reconnection diagnostics
    "J_dot_E": _Recipe(derived.j_dot_e, ("J1", "J2", "J3", "E1", "E2", "E3")),
    # Non-ideal electric field E' = E + VxB (component selects)
    "E_prime_1": _Recipe(
        derived.non_ideal_electric_field,
        ("E1", "E2", "E3", "V1", "V2", "V3", "B1", "B2", "B3"),
        component=0,
    ),
    "E_prime_2": _Recipe(
        derived.non_ideal_electric_field,
        ("E1", "E2", "E3", "V1", "V2", "V3", "B1", "B2", "B3"),
        component=1,
    ),
    "E_prime_3": _Recipe(
        derived.non_ideal_electric_field,
        ("E1", "E2", "E3", "V1", "V2", "V3", "B1", "B2", "B3"),
        component=2,
    ),
    # Ideal electric field E_ideal = -VxB (component selects)
    "E_ideal_1": _Recipe(
        derived.ideal_electric_field,
        ("V1", "V2", "V3", "B1", "B2", "B3"),
        component=0,
    ),
    "E_ideal_2": _Recipe(
        derived.ideal_electric_field,
        ("V1", "V2", "V3", "B1", "B2", "B3"),
        component=1,
    ),
    "E_ideal_3": _Recipe(
        derived.ideal_electric_field,
        ("V1", "V2", "V3", "B1", "B2", "B3"),
        component=2,
    ),
    # Hall electric field E_Hall = JxB/(nq) (component selects)
    "E_Hall_1": _Recipe(
        derived.hall_electric_field,
        ("J1", "J2", "J3", "B1", "B2", "B3", "n_s0"),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_ONLY,
        component=0,
    ),
    "E_Hall_2": _Recipe(
        derived.hall_electric_field,
        ("J1", "J2", "J3", "B1", "B2", "B3", "n_s0"),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_ONLY,
        component=1,
    ),
    "E_Hall_3": _Recipe(
        derived.hall_electric_field,
        ("J1", "J2", "J3", "B1", "B2", "B3", "n_s0"),
        species_index=0,
        species_args=_SpeciesArgs.CHARGE_ONLY,
        component=2,
    ),
    # Anisotropy instability parameters
    "firehose": _Recipe(derived.firehose_parameter, ("P_par", "P_perp", "|B|")),
    "mirror": _Recipe(derived.mirror_parameter, ("P_par", "P_perp", "|B|")),
    # Magnetic flux function (2D only)
    "psi": _Recipe(derived.magnetic_flux_function, ("B2",), needs_grid=True),
}


@dataclass(frozen=True, slots=True)
class _SpeciesTemplate:
    """Template for species-dependent derived quantities.

    Used to dynamically synthesize recipes for species index >= 2,
    where static registry entries don't exist.
    """

    func: Callable[..., Any]
    field_pattern: tuple[str, ...]
    species_args: _SpeciesArgs
    needs_gamma: bool = False
    needs_c: bool = False


_SPECIES_TEMPLATES: dict[str, _SpeciesTemplate] = {
    "omega_p": _SpeciesTemplate(
        derived.plasma_frequency, ("n_s{N}",), _SpeciesArgs.CHARGE_MASS
    ),
    "omega_c": _SpeciesTemplate(
        derived.gyrofrequency, ("|B|",), _SpeciesArgs.CHARGE_MASS
    ),
    "d": _SpeciesTemplate(
        derived.skin_depth, ("n_s{N}",), _SpeciesArgs.CHARGE_MASS, needs_c=True
    ),
    "v_th": _SpeciesTemplate(
        derived.thermal_speed, ("T_s{N}",), _SpeciesArgs.MASS_ONLY
    ),
    "r": _SpeciesTemplate(
        derived.gyroradius, ("T_s{N}", "|B|"), _SpeciesArgs.CHARGE_MASS
    ),
    "lambda_D": _SpeciesTemplate(
        derived.debye_length, ("T_s{N}", "n_s{N}"), _SpeciesArgs.CHARGE_ONLY
    ),
    "beta": _SpeciesTemplate(derived.plasma_beta, ("P_s{N}", "|B|"), _SpeciesArgs.NONE),
    "s": _SpeciesTemplate(
        derived.entropy, ("P_s{N}", "n_s{N}"), _SpeciesArgs.NONE, needs_gamma=True
    ),
    "s_gyro": _SpeciesTemplate(
        derived.gyrotropic_entropy, ("P_par", "P_perp", "n_s{N}"), _SpeciesArgs.NONE
    ),
    "T": _SpeciesTemplate(derived.temperature, ("P_s{N}", "n_s{N}"), _SpeciesArgs.NONE),
    "P": _SpeciesTemplate(
        derived.isotropic_pressure,
        ("P11_s{N}", "P22_s{N}", "P33_s{N}"),
        _SpeciesArgs.NONE,
    ),
    "V1": _SpeciesTemplate(
        derived.bulk_velocity, ("J1_s{N}", "rho_c_s{N}"), _SpeciesArgs.NONE
    ),
    "V2": _SpeciesTemplate(
        derived.bulk_velocity, ("J2_s{N}", "rho_c_s{N}"), _SpeciesArgs.NONE
    ),
    "V3": _SpeciesTemplate(
        derived.bulk_velocity, ("J3_s{N}", "rho_c_s{N}"), _SpeciesArgs.NONE
    ),
    "|V|": _SpeciesTemplate(
        derived.velocity_magnitude,
        ("V1_s{N}", "V2_s{N}", "V3_s{N}"),
        _SpeciesArgs.NONE,
    ),
    # Per-species mass density: rho_m_s = |rho_c_s| * m / |q|
    "rho_m": _SpeciesTemplate(
        derived.species_mass_density, ("rho_c_s{N}",), _SpeciesArgs.CHARGE_MASS
    ),
    # Per-species energy densities and thermodynamic quantities
    "e_k": _SpeciesTemplate(
        derived.kinetic_energy_density, ("rho_m_s{N}", "|V|_s{N}"), _SpeciesArgs.NONE
    ),
    "e_th": _SpeciesTemplate(
        derived.thermal_energy_density,
        ("P_s{N}",),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "e_th_trace": _SpeciesTemplate(
        derived.thermal_energy_density_trace,
        ("P11_s{N}", "P22_s{N}", "P33_s{N}"),
        _SpeciesArgs.NONE,
    ),
    "e_int": _SpeciesTemplate(
        derived.internal_energy,
        ("P_s{N}", "rho_m_s{N}"),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "h": _SpeciesTemplate(
        derived.enthalpy,
        ("P_s{N}", "rho_m_s{N}"),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    # Kinetic energy flux: KEF_i = (1/2) n m |V|² V_i
    "KEF1": _SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V1_s{N}", "V1_s{N}", "V2_s{N}", "V3_s{N}", "rho_c_s{N}"),
        _SpeciesArgs.CHARGE_MASS,
    ),
    "KEF2": _SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V2_s{N}", "V1_s{N}", "V2_s{N}", "V3_s{N}", "rho_c_s{N}"),
        _SpeciesArgs.CHARGE_MASS,
    ),
    "KEF3": _SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V3_s{N}", "V1_s{N}", "V2_s{N}", "V3_s{N}", "rho_c_s{N}"),
        _SpeciesArgs.CHARGE_MASS,
    ),
    # Heat flux: HF_i = EF_i - KEF_i (thermal + heat flux residual)
    "HF1": _SpeciesTemplate(
        derived.heat_flux_component, ("EF1_s{N}", "KEF1_s{N}"), _SpeciesArgs.NONE
    ),
    "HF2": _SpeciesTemplate(
        derived.heat_flux_component, ("EF2_s{N}", "KEF2_s{N}"), _SpeciesArgs.NONE
    ),
    "HF3": _SpeciesTemplate(
        derived.heat_flux_component, ("EF3_s{N}", "KEF3_s{N}"), _SpeciesArgs.NONE
    ),
    # Enthalpy flux (per-species): EHF_i = (gamma/(gamma-1)) P_s V_i_s
    "EHF1": _SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V1_s{N}"),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "EHF2": _SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V2_s{N}"),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "EHF3": _SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V3_s{N}"),
        _SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    # Conductive heat flux: q_i = HF_i - EHF_i (non-adiabatic residual)
    "q1": _SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF1_s{N}", "EHF1_s{N}"),
        _SpeciesArgs.NONE,
    ),
    "q2": _SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF2_s{N}", "EHF2_s{N}"),
        _SpeciesArgs.NONE,
    ),
    "q3": _SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF3_s{N}", "EHF3_s{N}"),
        _SpeciesArgs.NONE,
    ),
}

_SPECIES_SUFFIX_RE = re.compile(r"^(.+)_s(\d+)$")


def _try_species_recipe(name: str) -> _Recipe | None:
    """Try to build a recipe from species templates for names like ``omega_p_s2``.

    Returns ``None`` if the name doesn't match any template.
    """
    m = _SPECIES_SUFFIX_RE.match(name)
    if m is None:
        return None
    prefix, idx_str = m.group(1), m.group(2)
    species_index = int(idx_str)
    template = _SPECIES_TEMPLATES.get(prefix)
    if template is None:
        return None
    fields = tuple(f.replace("{N}", idx_str) for f in template.field_pattern)
    return _Recipe(
        func=template.func,
        fields=fields,
        species_index=species_index,
        needs_gamma=template.needs_gamma,
        needs_c=template.needs_c,
        species_args=template.species_args,
    )


# Display unit conversion: unit string → SI value.
# Hand-curated instead of a prefix parser — life's too short to rewrite pint.
_DISPLAY_UNITS: dict[str, float] = {
    "T": 1.0,
    "nT": 1e-9,
    "mT": 1e-3,
    "V/m": 1.0,
    "mV/m": 1e-3,
    "m/s": 1.0,
    "km/s": 1e3,
    "m^-3": 1.0,
    "cm^-3": 1e6,
    "/cc": 1e6,  # informal alias for cm^-3
    "Mp/cc": 1e6 * constants.m_p,  # proton masses per cc (BATSRUS)
    "amu/cc": 1e6 * constants.m_u,  # atomic mass units per cc (BATSRUS)
    "kg/m^3": 1.0,
    "Pa": 1.0,
    "nPa": 1e-9,
    "J/m^3": 1.0,
    "W/m^2": 1.0,
    "mW/m^2": 1e-3,
    "W/m^3": 1.0,
    "Hz": 1.0,
    "rad/s": 1.0,
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "m": 1.0,
    "km": 1e3,
    "R_E": 6.371e6,  # NASA NSSDCA Planetary Fact Sheet
    "eV": constants.eV,
    "keV": 1e3 * constants.eV,
    "K": constants.k,
    "A/m^2": 1.0,
    "uA/m^2": 1e-6,
    "nA/m^2": 1e-9,
    "uA/m2": 1e-6,  # BATSRUS header alias (no caret)
    "C/m^3": 1.0,
    "J/kg": 1.0,
    "MeV": 1e6 * constants.eV,
    "G": 1e-4,
    "mG": 1e-7,
    "pPa": 1e-12,
    "cm/s": 1e-2,
    "Mm": 1e6,
    "RE": 6.371e6,  # alias for R_E
    "R_S": 6.957e8,  # IAU 2015 nominal solar radius
    "R_Moon": 1.7374e6,  # NASA NSSDCA Moon Fact Sheet
    "R_Mercury": 2.4397e6,  # NASA NSSDCA Planetary Fact Sheet
    "R_Mars": 3.3895e6,  # NASA NSSDCA Planetary Fact Sheet
    "R_J": 6.9911e7,  # NASA NSSDCA Planetary Fact Sheet (volumetric mean)
    "R_Saturn": 5.8232e7,  # NASA NSSDCA Planetary Fact Sheet (volumetric mean)
    "AU": constants.au,
    "normalized": 1.0,
}

_MAX_DEPTH = 10


def _resolve_name(name: str) -> str:
    """Resolve a compute alias to its canonical registry name."""
    return _COMPUTE_ALIASES.get(name, name)


def _get_recipe(name: str) -> _Recipe:
    """Look up a recipe by name, raising KeyError with suggestions on miss."""
    canonical = _resolve_name(name)
    try:
        return _REGISTRY[canonical]
    except KeyError:
        pass
    # Try dynamic species template synthesis (e.g. omega_p_s2, T_s3)
    dynamic = _try_species_recipe(canonical)
    if dynamic is not None:
        return dynamic
    all_names = available_quantities()
    suggestions = difflib.get_close_matches(name, all_names, n=3, cutoff=0.4)
    msg = f"Unknown derived quantity {name!r}."
    if suggestions:
        msg += f" Did you mean: {suggestions}?"
    msg += " Call available_quantities() for the full list."
    raise KeyError(msg) from None


def _get_species_args(
    dataset: FieldDataset,
    recipe: _Recipe,
) -> list[float]:
    """Extract charge and mass from species info for a recipe."""
    if recipe.species_index is None:
        return []
    idx = recipe.species_index
    if not dataset.species or idx >= len(dataset.species):
        available = len(dataset.species)
        detail = (
            "no species defined"
            if not dataset.species
            else f"only {available} species available"
        )
        msg = (
            f"Cannot compute {recipe.func.__name__!r}: "
            f"requires species[{idx}] but {detail}"
        )
        raise ValueError(msg)
    sp = dataset.species[idx]
    if sp.charge is None or sp.mass is None:
        missing = [p for p in ("charge", "mass") if getattr(sp, p) is None]
        msg = (
            f"Species {sp.name!r} (index {idx}) is missing {', '.join(missing)}, "
            f"required by {recipe.func.__name__!r}"
        )
        raise ValueError(msg)
    return [sp.charge, sp.mass]


def _get_gamma(dataset: FieldDataset) -> float:
    """Get the adiabatic index from physics params."""
    return dataset.physics.gamma


def _get_c(dataset: FieldDataset) -> float:
    """Get the speed of light from physics params."""
    return dataset.physics.c


def _append_species_params(
    args: list[Any],
    species_args: list[float],
    kind: _SpeciesArgs,
) -> None:
    """Append the right species parameters based on the descriptor."""
    match kind:
        case _SpeciesArgs.CHARGE_MASS:
            args.extend(species_args)
        case _SpeciesArgs.MASS_ONLY:
            args.append(species_args[1])
        case _SpeciesArgs.CHARGE_ONLY:
            args.append(species_args[0])
        case _SpeciesArgs.NONE:
            pass
        case _ as unreachable:  # pragma: no cover
            from typing import assert_never

            assert_never(unreachable)


def compute_field(name: str, dataset: FieldDataset, _depth: int = 0) -> FloatArray:
    """Compute a derived quantity by name from a FieldDataset.

    If *name* is already present in the dataset, returns it directly.
    Otherwise dispatches to the registered pure function, recursively
    resolving any intermediate dependencies.

    Parameters
    ----------
    name : str
        Field or derived quantity name (e.g. ``"|B|"``, ``"beta"``).
    dataset : FieldDataset
        Source data.

    Returns
    -------
    FloatArray
        Computed array in code units.

    Raises
    ------
    KeyError
        If *name* is unknown and not in the dataset.
    ValueError
        If required species or physics info is missing.
    RecursionError
        If dependency chain exceeds depth limit.
    """
    if _depth > _MAX_DEPTH:
        msg = f"Dependency chain too deep (>{_MAX_DEPTH}) while computing {name!r}"
        raise RecursionError(msg)

    # Check original name first — raw fields take priority over aliases
    if dataset.has_field(name):
        return dataset[name]

    canonical = _resolve_name(name)

    # Direct field lookup after alias resolution
    if dataset.has_field(canonical):
        return dataset[canonical]

    recipe = _get_recipe(canonical)

    # Resolve field dependencies (recursive)
    args: list[Any] = []
    for field_name in recipe.fields:
        try:
            args.append(compute_field(field_name, dataset, _depth + 1))
        except KeyError:
            available = sorted(dataset.field_names())
            msg = (
                f"Cannot compute {canonical!r}: "
                f"requires {field_name!r} which is not available. "
                f"Available fields: {available}"
            )
            raise KeyError(msg) from None

    # Append species charge/mass
    species_args = _get_species_args(dataset, recipe)
    if species_args:
        if recipe.species_args is None:
            msg = (
                f"Recipe for {canonical!r} has species_index={recipe.species_index} "
                f"but no species_args descriptor"
            )
            raise ValueError(msg)
        _append_species_params(args, species_args, recipe.species_args)

    # Append gamma
    if recipe.needs_gamma:
        args.append(_get_gamma(dataset))

    # Append c
    if recipe.needs_c:
        args.append(_get_c(dataset))

    # Auto-inject c for relativistic simulations: when the dataset
    # declares physics.relativistic=True, functions with a c=None kwarg
    # get the speed of light passed in, activating their relativistic branch.
    kwargs: dict[str, Any] = {}
    if recipe.supports_relativistic and dataset.physics.relativistic:
        kwargs["c"] = _get_c(dataset)

    # Append grid spacing
    if recipe.needs_grid:
        if dataset.grid.geometry.type != GeometryType.CARTESIAN:
            msg = (
                f"Derived quantity {canonical!r} requires spatial derivatives, "
                f"which are only implemented for Cartesian geometry. "
                f"Dataset has {dataset.grid.geometry.type.value} geometry."
            )
            raise NotImplementedError(msg)
        args.extend(dataset.grid.spacing)
        # Operator-backed recipes get the dataset's geometry threaded
        # through as a kwarg. Today this is a no-op for the only
        # reachable code path (geometry is always Cartesian after the
        # check above) but it pre-wires the FieldDataset → recipe →
        # operator path so that when spherical/cylindrical operator
        # implementations land, the recipe automatically passes the
        # right geometry. At that point, the early raise above can be
        # relaxed for ``passes_geometry`` recipes.
        if recipe.passes_geometry:
            kwargs["geometry"] = dataset.grid.geometry.type

    result = recipe.func(*args, **kwargs)  # Make it so.

    if recipe.component is not None:
        return result[recipe.component]  # type: ignore[no-any-return]
    return result  # type: ignore[no-any-return]


def field_si_factor(name: str, normalization: Normalization) -> float:
    """Return the SI conversion factor for a field or derived quantity.

    Parameters
    ----------
    name : str
        Field or derived quantity name.
    normalization : Normalization
        Active normalization.

    Returns
    -------
    float
        Multiplicative factor: ``si_value = code_value * factor``.

    Raises
    ------
    ValueError
        If the quantity type for *name* is unknown.
    """
    canonical = _resolve_name(name)
    info = _FIELD_INFO.get(canonical)
    if info is None:
        # Resolve field aliases (Bx→B1, B_x→B1, P_e→Pe, etc.)
        fallback = _get_field_alias_fallback()
        canonical = fallback.get(canonical, canonical)
        info = _FIELD_INFO.get(canonical)
    quantity_type: str | None = info.quantity_type if info is not None else None
    if quantity_type is None:
        # Try regex patterns for per-species fields (n_s2, J1_s3, etc.)
        for pattern, qtype in _SPECIES_QUANTITY_PATTERNS:
            if pattern.match(canonical):
                quantity_type = qtype
                break
    if quantity_type is None:
        msg = (
            f"No SI conversion known for {name!r}. Known fields: {sorted(_FIELD_INFO)}"
        )
        raise ValueError(msg)
    return normalization.si_factor(quantity_type)


def display_unit_factor(unit_str: str) -> float:
    """Return the SI value of a display unit string.

    Parameters
    ----------
    unit_str : str
        Unit string (e.g. ``"nT"``, ``"km/s"``).

    Returns
    -------
    float
        Value of one display unit in SI.

    Raises
    ------
    ValueError
        If *unit_str* is not recognized.
    """
    try:
        return _DISPLAY_UNITS[unit_str]
    except KeyError:
        valid = sorted(_DISPLAY_UNITS)
        msg = f"Unknown unit {unit_str!r}. Valid: {valid}"
        raise ValueError(msg) from None


def available_quantities() -> list[str]:
    """Return sorted list of registered quantity names and aliases.

    Does not include dynamically synthesized per-species quantities
    (e.g. ``"omega_p_s2"``, ``"T_s3"``), which are also computable
    via :func:`compute_field`.

    Returns
    -------
    list[str]
    """
    return sorted(set(_REGISTRY) | set(_COMPUTE_ALIASES))


def field_dependencies(name: str, _depth: int = 0) -> set[str]:
    """Return the raw field names needed to compute *name*.

    Recursively walks the compute recipe graph. If *name* has no recipe
    (i.e. it is a raw field), returns ``{name}``.

    Parameters
    ----------
    name : str
        Field or derived quantity name (e.g. ``"Pi"``, ``"beta"``).

    Returns
    -------
    set[str]
        Leaf field names that must be present in the dataset.
    """
    if _depth > _MAX_DEPTH:
        return {name}

    canonical = _resolve_name(name)
    try:
        recipe = _get_recipe(canonical)
    except KeyError:
        return {canonical}

    deps: set[str] = set()
    for field in recipe.fields:
        deps |= field_dependencies(field, _depth + 1)
    return deps


_recipe_lock = threading.Lock()


def _find_sibling_components(name: str) -> dict[str, int]:
    r"""Find all recipes sharing the same func/fields as *name*.

    Returns a ``{name: component_index}`` dict for component-based
    recipes (e.g. ``S1/S2/S3``, ``curl_B1/2/3``). Returns an empty
    dict if *name* has no ``component``.
    """
    canonical = _resolve_name(name)
    try:
        recipe = _get_recipe(canonical)
    except KeyError:
        return {}
    if recipe.component is None:
        return {}
    siblings: dict[str, int] = {}
    for reg_name, reg_recipe in _REGISTRY.items():
        if (
            reg_recipe.func is recipe.func
            and reg_recipe.fields == recipe.fields
            and reg_recipe.component is not None
        ):
            siblings[reg_name] = reg_recipe.component
    return siblings


def register_recipe(
    name: str,
    func: Callable[..., Any],
    fields: tuple[str, ...],
    quantity_type: QuantityType | str,
    *,
    needs_grid: bool = False,
    needs_gamma: bool = False,
    needs_c: bool = False,
    long_name: str = "",
    latex: str = "",
) -> None:
    r"""Register a custom derived quantity.

    Registers both the computation recipe and the field metadata,
    so ``compute()``, ``in_si()``, ``field_info()``, and
    ``with_derived()`` all work for the custom field.

    Parameters
    ----------
    name : str
        Quantity name (e.g. ``"R_reconnection"``).
    func : Callable
        Pure function: takes arrays (one per field in *fields*),
        plus grid spacing if *needs_grid*, plus gamma if
        *needs_gamma*, plus c if *needs_c*. Returns a single array.
    fields : tuple[str, ...]
        Input field names (canonical or derived). Resolved
        recursively at compute time.
    quantity_type : QuantityType | str
        Physical quantity type for SI conversion.
    needs_grid : bool
        If ``True``, grid spacing ``(dx, dy, dz)`` is appended to args.
    needs_gamma : bool
        If ``True``, adiabatic index $\gamma$ is appended to args.
    needs_c : bool
        If ``True``, speed of light $c$ is appended to args.
    long_name : str
        Human-readable label for plot titles.
    latex : str
        LaTeX symbol for plot labels.

    Raises
    ------
    ValueError
        If *name* already exists in the recipe registry.

    Examples
    --------
    >>> import numpy as np
    >>> register_recipe(
    ...     "e_mag_ratio",
    ...     func=lambda eb, ee: eb / (eb + ee),
    ...     fields=("e_B", "e_E"),
    ...     quantity_type="dimensionless",
    ...     long_name="Magnetic-to-total EM energy ratio",
    ... )
    >>> "e_mag_ratio" in available_quantities()
    True
    >>> unregister_recipe("e_mag_ratio")
    """
    from pypic.fields import register_field as _register_field

    recipe = _Recipe(
        func=func,
        fields=fields,
        needs_grid=needs_grid,
        needs_gamma=needs_gamma,
        needs_c=needs_c,
    )
    with _recipe_lock:
        if name in _REGISTRY:
            msg = f"Recipe {name!r} already registered"
            raise ValueError(msg)
        _REGISTRY[name] = recipe

    try:
        _register_field(name, quantity_type, long_name=long_name, latex=latex)
    except Exception:
        with _recipe_lock:
            _REGISTRY.pop(name, None)
        raise


def unregister_recipe(name: str) -> None:
    r"""Remove a custom derived quantity.

    Removes both the computation recipe and the field metadata.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    from pypic.fields import unregister_field as _unregister_field

    with _recipe_lock:
        try:
            recipe = _REGISTRY.pop(name)
        except KeyError:
            msg = f"No recipe registered for {name!r}"
            raise KeyError(msg) from None

    try:
        _unregister_field(name)
    except Exception:
        with _recipe_lock:
            _REGISTRY[name] = recipe
        raise


__all__ = [
    "available_quantities",
    "compute_field",
    "display_unit_factor",
    "field_dependencies",
    "field_si_factor",
    "register_recipe",
    "unregister_recipe",
]
