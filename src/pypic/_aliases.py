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
    "vort_mag": "|vort|",
    # Descriptive names
    "plasma_beta": "beta",
    "v_Alfven": "v_A",
    "c_ms": "v_ms",
    "ion_acoustic_speed": "c_ia",
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
    # Structured v_th aliases
    "v_th_s0": "v_th_e",
    "v_th_s1": "v_th_i",
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
    # Descriptive energy flux aliases
    "energy_flux_x": "EF1",
    "energy_flux_y": "EF2",
    "energy_flux_z": "EF3",
}


def _build_field_alias_fallback() -> dict[str, str]:
    """Build a flat field alias lookup for SI conversion fallback.

    Merges all geometry alias dicts plus species and scalar aliases.
    This is safe because all B-field components map to the same SI
    quantity type regardless of which coordinate index they represent.
    """
    from pypic.readers.base import (
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
