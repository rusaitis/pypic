"""Alias tables for field names and derived quantities.

Central location for name-resolution data used by both ``compute`` and
``fields``.  Kept separate to avoid the circular dependency that would
arise if either module imported the other at top level.
"""

from __future__ import annotations

import functools

_COMPUTE_ALIASES: dict[str, str] = {
    "curl_Bx": "curl_B1",
    "curl_By": "curl_B2",
    "curl_Bz": "curl_B3",
    "vort_x": "vort1",
    "vort_y": "vort2",
    "vort_z": "vort3",
    "Sx": "S1",
    "Sy": "S2",
    "Sz": "S3",
    "poynting_flux_x": "S1",
    "poynting_flux_y": "S2",
    "poynting_flux_z": "S3",
    # Magnitude aliases (_mag suffix)
    "B_mag": "|B|",
    "Bmag": "|B|",
    "E_mag": "|E|",
    "Emag": "|E|",
    "J_mag": "|J|",
    "Jmag": "|J|",
    "V_mag": "|V|",
    "Vmag": "|V|",
    "Ve_mag": "|Ve|",
    "Vemag": "|Ve|",
    "Vi_mag": "|V|_s1",
    "Vimag": "|V|_s1",
    "vort_mag": "|vort|",
    # Descriptive names — speeds and dimensionless numbers
    "plasma_beta": "beta",
    "alfven_speed": "v_A",
    "sound_speed": "c_s",
    "magnetosonic_speed": "v_ms",
    "ion_acoustic_speed": "c_ia",
    "thermal_speed_e": "v_th_e",
    "thermal_speed_i": "v_th_i",
    "alfven_mach": "M_A",
    "magnetosonic_mach": "M_ms",
    "lorentz_factor": "gamma_L",
    "magnetization": "sigma",
    "c_ms": "v_ms",
    # Descriptive names — species-dependent scales
    "plasma_frequency_e": "omega_pe",
    "plasma_frequency_i": "omega_pi",
    "gyrofrequency_e": "omega_ce",
    "gyrofrequency_i": "omega_ci",
    "skin_depth_e": "d_e",
    "skin_depth_i": "d_i",
    "gyroradius_e": "r_e",
    "gyroradius_i": "r_i",
    "debye_length": "lambda_D",
    "debye_length_e": "lambda_D",
    "parallel_pressure": "P_par",
    "perpendicular_pressure": "P_perp",
    "parallel_pressure_e": "P_par_e",
    "parallel_pressure_i": "P_par_i",
    "perpendicular_pressure_e": "P_perp_e",
    "perpendicular_pressure_i": "P_perp_i",
    # Descriptive names — energies and thermodynamics
    "energy_magnetic": "e_B",
    "energy_electric": "e_E",
    "energy_kinetic": "e_k",
    "energy_thermal": "e_th",
    "enthalpy": "h",
    "enthalpy_relativistic": "h_rel",
    "energy_internal": "e_int",
    # Structured species names
    "omega_p_s0": "omega_pe",
    "omega_p_s1": "omega_pi",
    "omega_c_s0": "omega_ce",
    "omega_c_s1": "omega_ci",
    "d_s0": "d_e",
    "d_s1": "d_i",
    "v_thermal_s0": "v_th_e",
    "v_thermal_s1": "v_th_i",
    "larmor_radius_s0": "r_e",
    "larmor_radius_s1": "r_i",
    "rL_s0": "r_e",
    "rL_s1": "r_i",
    "plasma_beta_s0": "beta_e",
    "plasma_beta_s1": "beta_i",
    "beta_s0": "beta_e",
    "beta_s1": "beta_i",
    "entropy_s0": "s_e",
    "entropy_s1": "s_i",
    "entropy_gyrotropic_s0": "s_gyro_e",
    "entropy_gyrotropic_s1": "s_gyro_i",
    # Per-species temperature/pressure aliases for s0/s1
    "T_s0": "Te",
    "T_s1": "Ti",
    "P_s0": "Pe",
    "P_s1": "Pi",
    "P_par_s0": "P_par_e",
    "P_par_s1": "P_par_i",
    "P_perp_s0": "P_perp_e",
    "P_perp_s1": "P_perp_i",
    "agyrotropy_s0": "agyrotropy_e",
    "agyrotropy_s1": "agyrotropy_i",
    # Structured v_th aliases
    "v_th_s0": "v_th_e",
    "v_th_s1": "v_th_i",
    "thermal_speed_s0": "v_th_e",
    "thermal_speed_s1": "v_th_i",
    # Structured r aliases
    "r_s0": "r_e",
    "r_s1": "r_i",
    # Structured lambda_D aliases
    "lambda_D_s0": "lambda_D",
    # Underscore-separated operator/component aliases
    "curl_B_1": "curl_B1",
    "curl_B_2": "curl_B2",
    "curl_B_3": "curl_B3",
    "curl_B_x": "curl_B1",
    "curl_B_y": "curl_B2",
    "curl_B_z": "curl_B3",
    "vort_1": "vort1",
    "vort_2": "vort2",
    "vort_3": "vort3",
    "S_1": "S1",
    "S_2": "S2",
    "S_3": "S3",
    "S_x": "S1",
    "S_y": "S2",
    "S_z": "S3",
    "E_prime_x": "E_prime_1",
    "E_prime_y": "E_prime_2",
    "E_prime_z": "E_prime_3",
    "E_ideal_x": "E_ideal_1",
    "E_ideal_y": "E_ideal_2",
    "E_ideal_z": "E_ideal_3",
    "E_Hall_x": "E_Hall_1",
    "E_Hall_y": "E_Hall_2",
    "E_Hall_z": "E_Hall_3",
    # Per-species velocity aliases
    "Ve1": "V1_s0",
    "Ve2": "V2_s0",
    "Ve3": "V3_s0",
    "Vi1": "V1_s1",
    "Vi2": "V2_s1",
    "Vi3": "V3_s1",
    "|Vi|": "|V|_s1",
    # Per-species mass density aliases
    "rho_m_e": "rho_m_s0",
    "rho_m_i": "rho_m_s1",
    # Per-species energy density aliases
    "e_k_e": "e_k_s0",
    "e_k_i": "e_k_s1",
    "e_th_e": "e_th_s0",
    "e_th_i": "e_th_s1",
    "e_int_e": "e_int_s0",
    "e_int_i": "e_int_s1",
    "h_e": "h_s0",
    "h_i": "h_s1",
    # Energy flux aliases — numbered components only.
    # The bare-prefix forms (EFe, EFi, KEFe, ...) live in _GROUP_ALIASES
    # below because they expand to *three* names at read time, which is
    # a different contract than scalar compute-time aliases.
    "EFe1": "EF1_s0",
    "EFe2": "EF2_s0",
    "EFe3": "EF3_s0",
    "EFi1": "EF1_s1",
    "EFi2": "EF2_s1",
    "EFi3": "EF3_s1",
    "energy_flux_x": "EF1",
    "energy_flux_y": "EF2",
    "energy_flux_z": "EF3",
    "KEFe1": "KEF1_s0",
    "KEFe2": "KEF2_s0",
    "KEFe3": "KEF3_s0",
    "KEFi1": "KEF1_s1",
    "KEFi2": "KEF2_s1",
    "KEFi3": "KEF3_s1",
    "HFe1": "HF1_s0",
    "HFe2": "HF2_s0",
    "HFe3": "HF3_s0",
    "HFi1": "HF1_s1",
    "HFi2": "HF2_s1",
    "HFi3": "HF3_s1",
    "EHFe1": "EHF1_s0",
    "EHFe2": "EHF2_s0",
    "EHFe3": "EHF3_s0",
    "EHFi1": "EHF1_s1",
    "EHFi2": "EHF2_s1",
    "EHFi3": "EHF3_s1",
    "qe1": "q1_s0",
    "qe2": "q2_s0",
    "qe3": "q3_s0",
    "qi1": "q1_s1",
    "qi2": "q2_s1",
    "qi3": "q3_s1",
}


# Vector-group shorthand: aliases that expand to a three-component group
# at read time (``read(fields=["EFe"]) → EF1_s0, EF2_s0, EF3_s0``). These
# are deliberately separate from ``_COMPUTE_ALIASES`` because their target
# is a *prefix*, not a single computable quantity — feeding ``EF_s0`` to
# ``compute()`` would fail. Reader code consults this map after the
# scalar alias map; see ``readers/_registry.py`` for the expansion logic.
_GROUP_ALIASES: dict[str, str] = {
    "EFe": "EF_s0",
    "EFi": "EF_s1",
    "KEFe": "KEF_s0",
    "KEFi": "KEF_s1",
    "HFe": "HF_s0",
    "HFi": "HF_s1",
    "EHFe": "EHF_s0",
    "EHFi": "EHF_s1",
    "qe": "q_s0",
    "qi": "q_s1",
}


def _build_field_alias_fallback() -> dict[str, str]:
    """Build a flat field alias lookup for SI conversion fallback.

    Merges all geometry alias dicts plus species and scalar aliases.
    This is safe because all B-field components map to the same SI
    quantity type regardless of which coordinate index they represent.
    """
    from pypic.grid import (
        _CARTESIAN_ALIASES,
        _CARTESIAN_UNDERSCORE_ALIASES,
        _CYLINDRICAL_ALIASES,
        _CYLINDRICAL_UNDERSCORE_ALIASES,
        _NUMBERED_UNDERSCORE_ALIASES,
        _SCALAR_UNDERSCORE_ALIASES,
        _SPECIES_ALIASES,
        _SPHERICAL_ALIASES,
        _SPHERICAL_UNDERSCORE_ALIASES,
    )

    merged: dict[str, str] = {}
    merged.update(_CARTESIAN_ALIASES)
    merged.update(_SPHERICAL_ALIASES)
    merged.update(_CYLINDRICAL_ALIASES)
    merged.update(_CARTESIAN_UNDERSCORE_ALIASES)
    merged.update(_SPHERICAL_UNDERSCORE_ALIASES)
    merged.update(_CYLINDRICAL_UNDERSCORE_ALIASES)
    merged.update(_NUMBERED_UNDERSCORE_ALIASES)
    merged.update(_SCALAR_UNDERSCORE_ALIASES)
    merged.update(_SPECIES_ALIASES)
    return merged


@functools.cache
def _get_field_alias_fallback() -> dict[str, str]:
    """Return the field alias fallback dict (cached, thread-safe)."""
    return _build_field_alias_fallback()
