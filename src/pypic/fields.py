"""Field metadata registry: names, units, and display labels.

Maps every canonical field name to its physical quantity type, SI unit
label, human-readable long name, and LaTeX symbol.  ``_FIELD_INFO`` is
the single source of truth — ``compute.field_si_factor`` reads it
directly at lookup time.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from dataclasses import dataclass
from enum import StrEnum

from pypic._aliases import _COMPUTE_ALIASES, _get_field_alias_fallback

log = logging.getLogger(__name__)

_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class FieldInfo:
    """Metadata for a single field or derived quantity.

    Parameters
    ----------
    quantity_type : str
        Physical quantity type matching ``Normalization.si_factor()``
        (e.g. ``"b_field"``, ``"pressure"``, ``"dimensionless"``).
    long_name : str
        Human-readable name (e.g. ``"Magnetic field component 1"``).
    si_unit : str
        SI unit label (e.g. ``"T"``, ``"Pa"``, ``""`` for dimensionless).
    latex : str
        LaTeX symbol for plot labels (e.g. ``r"$B_1$"``).
    """

    quantity_type: str
    long_name: str
    si_unit: str
    latex: str = ""


class QuantityType(StrEnum):
    """Physical quantity types for field metadata and SI conversion.

    Each member corresponds to a key in ``Normalization.si_factor()``
    and maps to a default SI unit label.  Since ``QuantityType`` is a
    ``StrEnum``, members compare equal to plain strings:
    ``QuantityType.B_FIELD == "b_field"`` is ``True``.

    Examples
    --------
    >>> QuantityType.B_FIELD
    <QuantityType.B_FIELD: 'b_field'>
    >>> QuantityType.B_FIELD == "b_field"
    True
    """

    B_FIELD = "b_field"
    E_FIELD = "e_field"
    VELOCITY = "velocity"
    LENGTH = "length"
    TIME = "time"
    DENSITY = "density"
    MASS_DENSITY = "mass_density"
    CHARGE_DENSITY = "charge_density"
    CURRENT_DENSITY = "current_density"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    ENERGY_DENSITY = "energy_density"
    FREQUENCY = "frequency"
    POYNTING_FLUX = "poynting_flux"
    ENERGY_FLUX = "energy_flux"
    B_FIELD_PER_LENGTH = "b_field_per_length"
    E_FIELD_PER_LENGTH = "e_field_per_length"
    VELOCITY_PER_LENGTH = "velocity_per_length"
    SPECIFIC_ENERGY = "specific_energy"
    POWER_DENSITY = "power_density"
    DIMENSIONLESS = "dimensionless"


_QUANTITY_UNITS: dict[str, str] = {
    "b_field": "T",
    "e_field": "V/m",
    "velocity": "m/s",
    "length": "m",
    "time": "s",
    "density": "m^-3",
    "mass_density": "kg/m^3",
    "charge_density": "C/m^3",
    "current_density": "A/m^2",
    "pressure": "Pa",
    "temperature": "J",
    "energy_density": "J/m^3",
    "frequency": "rad/s",
    "poynting_flux": "W/m^2",
    "energy_flux": "W/m^2",
    "b_field_per_length": "T/m",
    "e_field_per_length": "V/m^2",
    "velocity_per_length": "1/s",
    "specific_energy": "J/kg",
    "power_density": "W/m^3",
    "dimensionless": "",
}

assert set(QuantityType) == set(_QUANTITY_UNITS), (
    "QuantityType and _QUANTITY_UNITS out of sync"
)

# Helper to keep long FieldInfo constructors within 88 columns
_FI = FieldInfo

_FIELD_INFO: dict[str, FieldInfo] = {
    # Electromagnetic fields
    "B1": _FI("b_field", "Magnetic field component 1", "T", r"$B_1$"),
    "B2": _FI("b_field", "Magnetic field component 2", "T", r"$B_2$"),
    "B3": _FI("b_field", "Magnetic field component 3", "T", r"$B_3$"),
    "|B|": _FI("b_field", "Magnetic field magnitude", "T", r"$|B|$"),
    "B0_1": _FI("b_field", "Background B component 1", "T", r"$B_{0,1}$"),
    "B0_2": _FI("b_field", "Background B component 2", "T", r"$B_{0,2}$"),
    "B0_3": _FI("b_field", "Background B component 3", "T", r"$B_{0,3}$"),
    "E1": _FI("e_field", "Electric field component 1", "V/m", r"$E_1$"),
    "E2": _FI("e_field", "Electric field component 2", "V/m", r"$E_2$"),
    "E3": _FI("e_field", "Electric field component 3", "V/m", r"$E_3$"),
    "|E|": _FI("e_field", "Electric field magnitude", "V/m", r"$|E|$"),
    # Current density
    "J1": _FI(
        "current_density",
        "Current density component 1",
        "A/m^2",
        r"$J_1$",
    ),
    "J2": _FI(
        "current_density",
        "Current density component 2",
        "A/m^2",
        r"$J_2$",
    ),
    "J3": _FI(
        "current_density",
        "Current density component 3",
        "A/m^2",
        r"$J_3$",
    ),
    "|J|": _FI(
        "current_density",
        "Current density magnitude",
        "A/m^2",
        r"$|J|$",
    ),
    # Velocities
    "V1": _FI("velocity", "Bulk velocity component 1", "m/s", r"$V_1$"),
    "V2": _FI("velocity", "Bulk velocity component 2", "m/s", r"$V_2$"),
    "V3": _FI("velocity", "Bulk velocity component 3", "m/s", r"$V_3$"),
    "|V|": _FI("velocity", "Bulk velocity magnitude", "m/s", r"$|V|$"),
    "Ve1": _FI(
        "velocity",
        "Electron velocity component 1",
        "m/s",
        r"$V_{e,1}$",
    ),
    "Ve2": _FI(
        "velocity",
        "Electron velocity component 2",
        "m/s",
        r"$V_{e,2}$",
    ),
    "Ve3": _FI(
        "velocity",
        "Electron velocity component 3",
        "m/s",
        r"$V_{e,3}$",
    ),
    "|Ve|": _FI("velocity", "Electron velocity magnitude", "m/s", r"$|V_e|$"),
    "v_A": _FI("velocity", "Alfvén speed", "m/s", r"$v_A$"),
    "c_s": _FI("velocity", "Sound speed", "m/s", r"$c_s$"),
    "c_ia": _FI("velocity", "Ion acoustic speed", "m/s", r"$c_{ia}$"),
    "v_ms": _FI("velocity", "Fast magnetosonic speed", "m/s", r"$v_{ms}$"),
    "v_th_e": _FI("velocity", "Electron thermal speed", "m/s", r"$v_{th,e}$"),
    "v_th_i": _FI("velocity", "Ion thermal speed", "m/s", r"$v_{th,i}$"),
    # Four-velocity
    "u1": _FI("velocity", "Four-velocity component 1", "m/s", r"$u_1$"),
    "u2": _FI("velocity", "Four-velocity component 2", "m/s", r"$u_2$"),
    "u3": _FI("velocity", "Four-velocity component 3", "m/s", r"$u_3$"),
    # Densities
    "rho_m": _FI("mass_density", "Mass density", "kg/m^3", r"$\rho_m$"),
    "rho_c": _FI("charge_density", "Charge density", "C/m^3", r"$\rho_c$"),
    "n_s0": _FI("density", "Number density (species 0)", "m^-3", r"$n_{s0}$"),
    "n_s1": _FI("density", "Number density (species 1)", "m^-3", r"$n_{s1}$"),
    "n_e": _FI("density", "Electron number density", "m^-3", r"$n_e$"),
    "n_i": _FI("density", "Ion number density", "m^-3", r"$n_i$"),
    # Pressure
    "P": _FI("pressure", "Total scalar pressure", "Pa", r"$P$"),
    "Pe": _FI("pressure", "Electron pressure", "Pa", r"$P_e$"),
    "Pi": _FI("pressure", "Ion pressure", "Pa", r"$P_i$"),
    "P_par": _FI("pressure", "Parallel pressure", "Pa", r"$P_\parallel$"),
    "P_perp": _FI("pressure", "Perpendicular pressure", "Pa", r"$P_\perp$"),
    "P11": _FI("pressure", "Pressure tensor P11", "Pa", r"$P_{11}$"),
    "P22": _FI("pressure", "Pressure tensor P22", "Pa", r"$P_{22}$"),
    "P33": _FI("pressure", "Pressure tensor P33", "Pa", r"$P_{33}$"),
    "P12": _FI("pressure", "Pressure tensor P12", "Pa", r"$P_{12}$"),
    "P13": _FI("pressure", "Pressure tensor P13", "Pa", r"$P_{13}$"),
    "P23": _FI("pressure", "Pressure tensor P23", "Pa", r"$P_{23}$"),
    # Temperature
    "Te": _FI("temperature", "Electron temperature", "J", r"$T_e$"),
    "Ti": _FI("temperature", "Ion temperature", "J", r"$T_i$"),
    # Energy densities
    "e_B": _FI(
        "energy_density",
        "Magnetic energy density",
        "J/m^3",
        r"$e_B$",
    ),
    "e_E": _FI(
        "energy_density",
        "Electric energy density",
        "J/m^3",
        r"$e_E$",
    ),
    "e_k": _FI(
        "energy_density",
        "Kinetic energy density",
        "J/m^3",
        r"$e_k$",
    ),
    "e_th": _FI(
        "energy_density",
        "Thermal energy density",
        "J/m^3",
        r"$e_{th}$",
    ),
    # Frequencies
    "omega_pe": _FI(
        "frequency",
        "Electron plasma frequency",
        "rad/s",
        r"$\omega_{pe}$",
    ),
    "omega_pi": _FI(
        "frequency",
        "Ion plasma frequency",
        "rad/s",
        r"$\omega_{pi}$",
    ),
    "omega_ce": _FI(
        "frequency",
        "Electron cyclotron frequency",
        "rad/s",
        r"$\omega_{ce}$",
    ),
    "omega_ci": _FI(
        "frequency",
        "Ion cyclotron frequency",
        "rad/s",
        r"$\omega_{ci}$",
    ),
    # Lengths
    "d_e": _FI("length", "Electron skin depth", "m", r"$d_e$"),
    "d_i": _FI("length", "Ion skin depth", "m", r"$d_i$"),
    "r_e": _FI("length", "Electron thermal gyroradius", "m", r"$r_e$"),
    "r_i": _FI("length", "Ion thermal gyroradius", "m", r"$r_i$"),
    "lambda_D": _FI("length", "Electron Debye length", "m", r"$\lambda_D$"),
    # Poynting flux / energy flux
    "S1": _FI(
        "poynting_flux",
        "Poynting flux component 1",
        "W/m^2",
        r"$S_1$",
    ),
    "S2": _FI(
        "poynting_flux",
        "Poynting flux component 2",
        "W/m^2",
        r"$S_2$",
    ),
    "S3": _FI(
        "poynting_flux",
        "Poynting flux component 3",
        "W/m^2",
        r"$S_3$",
    ),
    "EF1": _FI(
        "energy_flux",
        "Energy flux component 1",
        "W/m^2",
        r"$EF_1$",
    ),
    "EF2": _FI(
        "energy_flux",
        "Energy flux component 2",
        "W/m^2",
        r"$EF_2$",
    ),
    "EF3": _FI(
        "energy_flux",
        "Energy flux component 3",
        "W/m^2",
        r"$EF_3$",
    ),
    "EHF1": _FI("energy_flux", "Enthalpy flux component 1", "W/m^2", r"$EHF_1$"),
    "EHF2": _FI("energy_flux", "Enthalpy flux component 2", "W/m^2", r"$EHF_2$"),
    "EHF3": _FI("energy_flux", "Enthalpy flux component 3", "W/m^2", r"$EHF_3$"),
    # Thermodynamic (specific quantities — energy per unit mass)
    "h": _FI("specific_energy", "Specific enthalpy", "J/kg", r"$h$"),
    "h_rel": _FI(
        "specific_energy",
        "Relativistic specific enthalpy",
        "J/kg",
        r"$h_{rel}$",
    ),
    "e_int": _FI("specific_energy", "Specific internal energy", "J/kg", r"$e_{int}$"),
    # Diagnostics (spatial derivatives)
    "div_B": _FI(
        "b_field_per_length",
        "Divergence of B",
        "T/m",
        r"$\nabla \cdot B$",
    ),
    "div_E": _FI(
        "e_field_per_length",
        "Divergence of E",
        "V/m^2",
        r"$\nabla \cdot E$",
    ),
    "curl_B1": _FI(
        "b_field_per_length",
        "Curl of B component 1",
        "T/m",
        r"$(\nabla \times B)_1$",
    ),
    "curl_B2": _FI(
        "b_field_per_length",
        "Curl of B component 2",
        "T/m",
        r"$(\nabla \times B)_2$",
    ),
    "curl_B3": _FI(
        "b_field_per_length",
        "Curl of B component 3",
        "T/m",
        r"$(\nabla \times B)_3$",
    ),
    "vort1": _FI(
        "velocity_per_length",
        "Vorticity component 1",
        "1/s",
        r"$\omega_1$",
    ),
    "vort2": _FI(
        "velocity_per_length",
        "Vorticity component 2",
        "1/s",
        r"$\omega_2$",
    ),
    "vort3": _FI(
        "velocity_per_length",
        "Vorticity component 3",
        "1/s",
        r"$\omega_3$",
    ),
    "|vort|": _FI(
        "velocity_per_length",
        "Vorticity magnitude",
        "1/s",
        r"$|\omega|$",
    ),
    # Dimensionless
    "beta": _FI("dimensionless", "Plasma beta", "", r"$\beta$"),
    "beta_e": _FI("dimensionless", "Electron beta", "", r"$\beta_e$"),
    "beta_i": _FI("dimensionless", "Ion beta", "", r"$\beta_i$"),
    "M_A": _FI("dimensionless", "Alfvén Mach number", "", r"$M_A$"),
    "M_ms": _FI("dimensionless", "Magnetosonic Mach number", "", r"$M_{ms}$"),
    "s": _FI("dimensionless", "Specific entropy", "", r"$s$"),
    "s_e": _FI("dimensionless", "Electron entropy", "", r"$s_e$"),
    "s_i": _FI("dimensionless", "Ion entropy", "", r"$s_i$"),
    "s_gyro_e": _FI(
        "dimensionless",
        "Electron gyrotropic entropy",
        "",
        r"$s_{gyro,e}$",
    ),
    "s_gyro_i": _FI(
        "dimensionless",
        "Ion gyrotropic entropy",
        "",
        r"$s_{gyro,i}$",
    ),
    "agyrotropy": _FI("dimensionless", "Agyrotropy measure", "", r"$Q$"),
    "gamma_L": _FI("dimensionless", "Bulk Lorentz factor", "", r"$\gamma$"),
    "sigma": _FI("dimensionless", "Magnetization parameter", "", r"$\sigma$"),
    "gamma_eos": _FI("dimensionless", "Adiabatic index", "", r"$\gamma_{eos}$"),
    # Reconnection diagnostics
    "J_dot_E": _FI(
        "power_density",
        "Energy conversion rate",
        "W/m^3",
        r"$\mathbf{J} \cdot \mathbf{E}$",
    ),
    "E_prime_1": _FI(
        "e_field",
        "Non-ideal electric field, component 1",
        "V/m",
        r"$E'_1$",
    ),
    "E_prime_2": _FI(
        "e_field",
        "Non-ideal electric field, component 2",
        "V/m",
        r"$E'_2$",
    ),
    "E_prime_3": _FI(
        "e_field",
        "Non-ideal electric field, component 3",
        "V/m",
        r"$E'_3$",
    ),
    "E_ideal_1": _FI(
        "e_field",
        "Ideal electric field, component 1",
        "V/m",
        r"$E_{ideal,1}$",
    ),
    "E_ideal_2": _FI(
        "e_field",
        "Ideal electric field, component 2",
        "V/m",
        r"$E_{ideal,2}$",
    ),
    "E_ideal_3": _FI(
        "e_field",
        "Ideal electric field, component 3",
        "V/m",
        r"$E_{ideal,3}$",
    ),
    "E_Hall_1": _FI(
        "e_field",
        "Hall electric field, component 1",
        "V/m",
        r"$E_{Hall,1}$",
    ),
    "E_Hall_2": _FI(
        "e_field",
        "Hall electric field, component 2",
        "V/m",
        r"$E_{Hall,2}$",
    ),
    "E_Hall_3": _FI(
        "e_field",
        "Hall electric field, component 3",
        "V/m",
        r"$E_{Hall,3}$",
    ),
    "firehose": _FI("dimensionless", "Firehose parameter", "", r"$\mathcal{F}$"),
    "mirror": _FI("dimensionless", "Mirror parameter", "", r"$\mathcal{M}$"),
    "psi": _FI("b_field", "Magnetic flux function", "T", r"$\psi$"),
}


def register_field(
    name: str,
    quantity_type: QuantityType | str,
    *,
    long_name: str = "",
    si_unit: str | None = None,
    latex: str = "",
) -> None:
    """Register metadata for a custom field.

    Enables ``field_info()``, ``field_si_factor()``, ``in_si()``,
    ``in_units()``, and ``unit_label()`` for user-defined fields.

    Parameters
    ----------
    name : str
        Field name (e.g. ``"my_diagnostic"``).
    quantity_type : QuantityType | str
        Physical quantity type — must be a key in ``_QUANTITY_UNITS``
        (e.g. ``QuantityType.VELOCITY``, ``"pressure"``).
    long_name : str
        Human-readable label for plot titles.
    si_unit : str | None
        SI unit label.  If ``None``, inferred from *quantity_type*.
    latex : str
        LaTeX symbol for plot labels.

    Raises
    ------
    ValueError
        If *quantity_type* is not recognized.
    """
    if quantity_type not in _QUANTITY_UNITS:
        valid = sorted(_QUANTITY_UNITS)
        msg = f"Unknown quantity_type {quantity_type!r}. Valid: {valid}"
        raise ValueError(msg)

    if si_unit is None:
        si_unit = _QUANTITY_UNITS[quantity_type]

    info = FieldInfo(quantity_type, long_name, si_unit, latex)

    with _lock:
        if name in _FIELD_INFO:
            log.warning("Overwriting existing field metadata for %r", name)
        _FIELD_INFO[name] = info


def unregister_field(name: str) -> None:
    """Remove custom field metadata.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    with _lock:
        try:
            del _FIELD_INFO[name]
        except KeyError:
            msg = f"No field metadata registered for {name!r}"
            raise KeyError(msg) from None


# Per-species field metadata via regex matching. Each entry is
# (pattern, quantity_type, long_name_template, latex_template) where
# {N} = species index and {C} = component index (see SCHEMA.md §3).
_SPECIES_INFO_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    (
        re.compile(r"^n_s(\d+)$"),
        "density",
        "Number density (species {N})",
        r"$n_{{s{N}}}$",
    ),
    (
        re.compile(r"^rho_c_s(\d+)$"),
        "charge_density",
        "Charge density (species {N})",
        r"$\rho_{{c,s{N}}}$",
    ),
    (
        re.compile(r"^J([123])_s(\d+)$"),
        "current_density",
        "Current density component {C} (species {N})",
        r"$J_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^V([123])_s(\d+)$"),
        "velocity",
        "Velocity component {C} (species {N})",
        r"$V_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^Ve([123])_s(\d+)$"),
        "velocity",
        "Electron velocity component {C} (species {N})",
        r"$V_{{e,{C},s{N}}}$",
    ),
    (
        re.compile(r"^EF([123])_s(\d+)$"),
        "energy_flux",
        "Energy flux component {C} (species {N})",
        r"$EF_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^KEF([123])_s(\d+)$"),
        "energy_flux",
        "Kinetic energy flux component {C} (species {N})",
        r"$KEF_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^HF([123])_s(\d+)$"),
        "energy_flux",
        "Heat flux component {C} (species {N})",
        r"$HF_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^EHF([123])_s(\d+)$"),
        "energy_flux",
        "Enthalpy flux component {C} (species {N})",
        r"$EHF_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^q([123])_s(\d+)$"),
        "energy_flux",
        "Conductive heat flux component {C} (species {N})",
        r"$q_{{{C},s{N}}}$",
    ),
    (
        re.compile(r"^P(\d{0,2})_s(\d+)$"),
        "pressure",
        "Pressure {C} (species {N})",
        r"$P_{{{C},s{N}}}$",
    ),  # \d{0,2} matches both P11_s0 (tensor) and P_s0 (scalar)
    (
        re.compile(r"^rho_m_s(\d+)$"),
        "mass_density",
        "Mass density (species {N})",
        r"$\rho_{{m,s{N}}}$",
    ),
    (
        re.compile(r"^\|V\|_s(\d+)$"),
        "velocity",
        "Velocity magnitude (species {N})",
        r"$|V_{{s{N}}}|$",
    ),
    (
        re.compile(r"^e_k_s(\d+)$"),
        "energy_density",
        "Kinetic energy density (species {N})",
        r"$e_{{k,s{N}}}$",
    ),
    (
        re.compile(r"^e_th_trace_s(\d+)$"),
        "energy_density",
        "Thermal energy density tensor trace (species {N})",
        r"$e_{{th,\mathrm{{tr}},s{N}}}$",
    ),
    (
        re.compile(r"^e_th_s(\d+)$"),
        "energy_density",
        "Thermal energy density (species {N})",
        r"$e_{{th,s{N}}}$",
    ),
    (
        re.compile(r"^e_int_s(\d+)$"),
        "specific_energy",
        "Specific internal energy (species {N})",
        r"$e_{{int,s{N}}}$",
    ),
    (
        re.compile(r"^h_s(\d+)$"),
        "specific_energy",
        "Specific enthalpy (species {N})",
        r"$h_{{s{N}}}$",
    ),
    (
        re.compile(r"^T_s(\d+)$"),
        "temperature",
        "Temperature (species {N})",
        r"$T_{{s{N}}}$",
    ),
    (
        re.compile(r"^omega_p_s(\d+)$"),
        "frequency",
        "Plasma frequency (species {N})",
        r"$\omega_{{p,s{N}}}$",
    ),
    (
        re.compile(r"^omega_c_s(\d+)$"),
        "frequency",
        "Cyclotron frequency (species {N})",
        r"$\omega_{{c,s{N}}}$",
    ),
    (
        re.compile(r"^d_s(\d+)$"),
        "length",
        "Skin depth (species {N})",
        r"$d_{{s{N}}}$",
    ),
    (
        re.compile(r"^r_s(\d+)$"),
        "length",
        "Thermal gyroradius (species {N})",
        r"$r_{{s{N}}}$",
    ),
    (
        re.compile(r"^lambda_D_s(\d+)$"),
        "length",
        "Debye length (species {N})",
        r"$\lambda_{{D,s{N}}}$",
    ),
    (
        re.compile(r"^v_th_s(\d+)$"),
        "velocity",
        "Thermal speed (species {N})",
        r"$v_{{th,s{N}}}$",
    ),
    (
        re.compile(r"^beta_s(\d+)$"),
        "dimensionless",
        "Plasma beta (species {N})",
        r"$\beta_{{s{N}}}$",
    ),
    (
        re.compile(r"^s_s(\d+)$"),
        "dimensionless",
        "Entropy (species {N})",
        r"$s_{{s{N}}}$",
    ),
    (
        re.compile(r"^s_gyro_s(\d+)$"),
        "dimensionless",
        "Gyrotropic entropy (species {N})",
        r"$s_{{gyro,s{N}}}$",
    ),
    (
        re.compile(r"^agyrotropy_s(\d+)$"),
        "dimensionless",
        "Agyrotropy (species {N})",
        r"$Q_{{s{N}}}$",
    ),
]


_SPECIES_QUANTITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (pat, qtype) for pat, qtype, _, _ in _SPECIES_INFO_PATTERNS
]


def _try_species_info(name: str) -> FieldInfo | None:
    """Try to build FieldInfo from species regex patterns."""
    for pattern, qtype, name_tmpl, latex_tmpl in _SPECIES_INFO_PATTERNS:
        m = pattern.match(name)
        if m is not None:
            groups = m.groups()
            if len(groups) == 2:
                component, species_idx = groups
                if component == "":
                    # Scalar species field (e.g. P_s0 → "Pressure (species 0)")
                    long_name = f"Pressure (species {species_idx})"
                    latex = rf"$P_{{s{species_idx}}}$"
                else:
                    long_name = name_tmpl.format(C=component, N=species_idx)
                    latex = latex_tmpl.format(C=component, N=species_idx)
            else:
                species_idx = groups[0]
                long_name = name_tmpl.format(N=species_idx)
                latex = latex_tmpl.format(N=species_idx)
            return FieldInfo(qtype, long_name, _QUANTITY_UNITS[qtype], latex)
    return None


_COMPONENT_RE = re.compile(r"component ([123])")

_LATEX_AXIS_MAP: dict[str, str] = {"θ": r"\theta", "φ": r"\phi"}


def _latex_subscript(axis: str) -> str:
    """Format an axis label for LaTeX subscript position."""
    label = _LATEX_AXIS_MAP.get(axis, axis)
    return f"{{{label}}}" if len(label) > 1 else label


def _replace_latex_component(latex: str, digit: str, axis: str) -> str:
    """Replace a component digit with an axis label in a LaTeX string."""
    label = _LATEX_AXIS_MAP.get(axis, axis)
    sub = _latex_subscript(axis)
    # Simple trailing subscript: _N$ → _{sub}$ (e.g. $B_1$ → $B_x$)
    latex = latex.replace(f"_{digit}$", f"_{sub}$")
    # Digit at end of brace group: ,N} → ,label} (e.g. $V_{{e,1}}$ → $V_{{e,x}}$)
    latex = latex.replace(f",{digit}}}", f",{label}}}")
    # Digit at start of brace group: {{N, → {{label, (e.g. $J_{{1,s2}}$)
    latex = latex.replace(f"{{{digit},", f"{{{label},")
    # Digit between commas: ,N, → ,label, (e.g. $V_{{e,2,s1}}$)
    latex = latex.replace(f",{digit},", f",{label},")
    return latex


def _localize_field_info(
    info: FieldInfo, axis_names: tuple[str, str, str]
) -> FieldInfo:
    """Post-process FieldInfo to use geometry-specific axis labels."""
    m = _COMPONENT_RE.search(info.long_name)
    if m is None:
        return info
    digit = m.group(1)
    axis = axis_names[int(digit) - 1]
    long_name = info.long_name.replace(f"component {digit}", f"{axis}-component")
    latex = _replace_latex_component(info.latex, digit, axis)
    return copy.replace(info, long_name=long_name, latex=latex)


def field_info(
    name: str, *, axis_names: tuple[str, str, str] | None = None
) -> FieldInfo:
    r"""Look up metadata for a field or derived quantity.

    Resolution order:

    1. Direct lookup in the registry
    2. Compute alias resolution (``"B_mag"`` -> ``"|B|"``)
    3. Field alias fallback (``"Bx"`` -> ``"B1"``)
    4. Per-species regex patterns (``"n_s5"``, ``"omega_p_s3"``)

    Parameters
    ----------
    name : str
        Field or derived quantity name.
    axis_names : tuple[str, str, str] | None
        Coordinate axis labels.  When provided, component labels are
        localized (e.g. "component 1" -> "x-component" for Cartesian).

    Returns
    -------
    FieldInfo

    Raises
    ------
    KeyError
        If *name* cannot be resolved.

    Examples
    --------
    >>> field_info("|B|").si_unit
    'T'
    >>> field_info("beta").latex
    '$\\beta$'
    >>> field_info("B1", axis_names=("x", "y", "z")).long_name
    'Magnetic field x-component'
    """

    def _maybe_localize(info: FieldInfo) -> FieldInfo:
        if axis_names is not None:
            return _localize_field_info(info, axis_names)
        return info

    # 1. Direct lookup
    info = _FIELD_INFO.get(name)
    if info is not None:
        return _maybe_localize(info)

    # 2. Compute alias resolution (B_mag -> |B|, etc.)
    canonical = _COMPUTE_ALIASES.get(name)
    if canonical is not None:
        info = _FIELD_INFO.get(canonical)
        if info is not None:
            return _maybe_localize(info)

    # 3. Field alias fallback (Bx -> B1, P_e -> Pe, etc.)
    fallback = _get_field_alias_fallback()
    target = canonical if canonical is not None else name
    resolved = fallback.get(target, name)
    info = _FIELD_INFO.get(resolved)
    if info is not None:
        return _maybe_localize(info)

    # 4. Per-species regex patterns
    species_info = _try_species_info(target)
    if species_info is not None:
        return _maybe_localize(species_info)
    if resolved != target:
        species_info = _try_species_info(resolved)
        if species_info is not None:
            return _maybe_localize(species_info)

    msg = f"No metadata for field {name!r}"
    raise KeyError(msg)


def unit_label(name: str, *, si: bool = False) -> str:
    r"""Return a unit label string for a field, suitable for plot axes.

    Parameters
    ----------
    name : str
        Field or derived quantity name.
    si : bool
        If ``True``, return the SI unit label (e.g. ``"T"``).
        If ``False``, return ``"normalized"`` or ``""`` for dimensionless.

    Returns
    -------
    str

    Examples
    --------
    >>> unit_label("B1", si=True)
    'T'
    >>> unit_label("B1")
    'normalized'
    >>> unit_label("beta", si=True)
    ''
    >>> unit_label("beta")
    ''
    """
    info = field_info(name)
    if si:
        return info.si_unit
    return "" if info.quantity_type == "dimensionless" else "normalized"


def quantity_units(quantity_type: str) -> str:
    r"""Return the SI unit label for a physical quantity type.

    Parameters
    ----------
    quantity_type : str
        Quantity type string (e.g. ``"b_field"``, ``"pressure"``).

    Returns
    -------
    str
        SI unit label.

    Raises
    ------
    KeyError
        If *quantity_type* is unknown.

    Examples
    --------
    >>> quantity_units("b_field")
    'T'
    >>> quantity_units("dimensionless")
    ''
    """
    try:
        return _QUANTITY_UNITS[quantity_type]
    except KeyError:
        valid = sorted(_QUANTITY_UNITS)
        msg = f"Unknown quantity type {quantity_type!r}. Valid: {valid}"
        raise KeyError(msg) from None


__all__ = [
    "FieldInfo",
    "QuantityType",
    "field_info",
    "quantity_units",
    "register_field",
    "unit_label",
    "unregister_field",
]
