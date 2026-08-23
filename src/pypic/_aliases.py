"""Alias tables for field names and derived quantities.

Central location for name-resolution data used by both ``compute`` and
``fields``.  Kept separate to avoid the circular dependency that would
arise if either module imported the other at top level.

The public surface (`COMPUTE_ALIASES`, `GROUP_ALIASES`,
`SPECIES_SUFFIX_RE`, `species_name_aliases`) is re-exported
through [`pypic.aliases`][pypic.aliases] for codegen consumers.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Matches the canonical ``_s<index>`` species qualifier in any
# Tier-3 position:
#   - End (scalar per-species): ``P_s0``, ``T_s1`` — no suffix.
#   - Middle followed by a component / modifier (vector, tensor, or
#     generic operator): ``V_s0_1``, ``P_s0_11``, ``P_s0_par``,
#     ``V_s0_perp_1``.
#   - Middle followed by closing pipe (per-species magnitude):
#     ``|V_s0|``, ``|J_s1|``.
#   - Middle followed by an operator + closing pipe (per-species
#     operator magnitude): ``|V_s0_perp|``.
# Captures (species, suffix); suffix is ``_<x>|`` (operator + pipe),
# ``_<x>``, ``|``, or empty.
SPECIES_SUFFIX_RE = re.compile(r"_s(\d+)(?P<suffix>_[^|]+\||_[^|]+|\|)?$")

COMPUTE_ALIASES: dict[str, str] = {
    "curl_Bx": "curl_B_1",
    "curl_By": "curl_B_2",
    "curl_Bz": "curl_B_3",
    "vort_x": "vort_1",
    "vort_y": "vort_2",
    "vort_z": "vort_3",
    "Sx": "S_1",
    "Sy": "S_2",
    "Sz": "S_3",
    "poynting_flux_x": "S_1",
    "poynting_flux_y": "S_2",
    "poynting_flux_z": "S_3",
    # Magnitude aliases (_mag suffix)
    "B_mag": "|B|",
    "Bmag": "|B|",
    "E_mag": "|E|",
    "Emag": "|E|",
    "J_mag": "|J|",
    "Jmag": "|J|",
    "V_mag": "|V|",
    "Vmag": "|V|",
    "Ve_mag": "|V_s0|",
    "V_e_mag": "|V_s0|",
    "Vi_mag": "|V_s1|",
    "V_i_mag": "|V_s1|",
    "V_s0_mag": "|V_s0|",
    "V_s1_mag": "|V_s1|",
    "vort_mag": "|vort|",
    # Descriptive names — speeds and dimensionless numbers
    "plasma_beta": "beta",
    "alfven_speed": "v_A",
    "sound_speed": "c_s",
    "magnetosonic_speed": "v_ms",
    "ion_acoustic_speed": "c_ia",
    "thermal_speed_e": "v_th_s0",
    "thermal_speed_i": "v_th_s1",
    "alfven_mach": "M_A",
    "magnetosonic_mach": "M_ms",
    "lorentz_factor": "gamma_L",
    "magnetization": "sigma",
    "c_ms": "v_ms",
    # Descriptive names — species-dependent scales
    "plasma_frequency_e": "omega_p_s0",
    "plasma_frequency_i": "omega_p_s1",
    "gyrofrequency_e": "omega_c_s0",
    "gyrofrequency_i": "omega_c_s1",
    "skin_depth_e": "d_s0",
    "skin_depth_i": "d_s1",
    "gyroradius_e": "r_s0",
    "gyroradius_i": "r_s1",
    "debye_length": "lambda_D_s0",
    "debye_length_e": "lambda_D_s0",
    "parallel_pressure": "P_par",
    "perpendicular_pressure": "P_perp",
    "parallel_pressure_e": "P_s0_par",
    "parallel_pressure_i": "P_s1_par",
    "perpendicular_pressure_e": "P_s0_perp",
    "perpendicular_pressure_i": "P_s1_perp",
    # Descriptive names — energies and thermodynamics
    "energy_magnetic": "e_B",
    "energy_electric": "e_E",
    "energy_kinetic": "e_k",
    "energy_thermal": "e_th",
    "enthalpy": "h",
    "enthalpy_relativistic": "h_rel",
    "energy_internal": "e_int",
    # Literature spellings (NRL Plasma Formulary form) — alias to Tier-3 canonical.
    # Stays the user-facing form in error messages and docs even though the
    # internal recipe ID follows the storage-name shape (schema.md §3).
    "omega_pe": "omega_p_s0",
    "omega_pi": "omega_p_s1",
    "omega_ce": "omega_c_s0",
    "omega_ci": "omega_c_s1",
    "d_e": "d_s0",
    "d_i": "d_s1",
    "v_th_e": "v_th_s0",
    "v_th_i": "v_th_s1",
    "r_e": "r_s0",
    "r_i": "r_s1",
    "lambda_D": "lambda_D_s0",
    # Long-form descriptive aliases
    "v_thermal_s0": "v_th_s0",
    "v_thermal_s1": "v_th_s1",
    "larmor_radius_s0": "r_s0",
    "larmor_radius_s1": "r_s1",
    "rL_s0": "r_s0",
    "rL_s1": "r_s1",
    "plasma_beta_e": "beta_s0",
    "plasma_beta_i": "beta_s1",
    "plasma_beta_s0": "beta_s0",
    "plasma_beta_s1": "beta_s1",
    "entropy_e": "s_s0",
    "entropy_i": "s_s1",
    "entropy_s0": "s_s0",
    "entropy_s1": "s_s1",
    "entropy_gyrotropic_e": "s_gyro_s0",
    "entropy_gyrotropic_i": "s_gyro_s1",
    "entropy_gyrotropic_s0": "s_gyro_s0",
    "entropy_gyrotropic_s1": "s_gyro_s1",
    # Two-species e/i convenience: alias to the universal _sN canonical.
    # Storage-equivalent (same data, different name) — the dataset's
    # bidirectional alias resolver handles the case where the input
    # data is stored under the e/i name and a recipe asks for _sN.
    "Pe": "P_s0",
    "Pi": "P_s1",
    "Te": "T_s0",
    "Ti": "T_s1",
    # Derived per-species: e/i convenience name, _sN canonical.
    "beta_e": "beta_s0",
    "beta_i": "beta_s1",
    "s_e": "s_s0",
    "s_i": "s_s1",
    "s_gyro_e": "s_gyro_s0",
    "s_gyro_i": "s_gyro_s1",
    "P_par_e": "P_s0_par",
    "P_par_i": "P_s1_par",
    "P_perp_e": "P_s0_perp",
    "P_perp_i": "P_s1_perp",
    "agyrotropy_e": "agyrotropy_s0",
    "agyrotropy_i": "agyrotropy_s1",
    "D_ng_e": "D_ng_s0",
    "D_ng_i": "D_ng_s1",
    "A_phi_e": "A_phi_s0",
    "A_phi_i": "A_phi_s1",
    "R_recon_e": "R_recon_s0",
    "R_recon_i": "R_recon_s1",
    # Per-species parallel velocity shorthand.
    "V_par_e": "V_s0_par",
    "V_par_i": "V_s1_par",
    # Long-form thermal-speed alias resolves to Tier-3 canonical.
    "thermal_speed_s0": "v_th_s0",
    "thermal_speed_s1": "v_th_s1",
    # Cartesian aliases for differential / EM-derived component names.
    # Numbered forms (curl_B_1 etc.) are the canonical names; only the
    # x/y/z spellings need an alias entry.
    "curl_B_x": "curl_B_1",
    "curl_B_y": "curl_B_2",
    "curl_B_z": "curl_B_3",
    "S_x": "S_1",
    "S_y": "S_2",
    "S_z": "S_3",
    "E_prime_x": "E_prime_1",
    "E_prime_y": "E_prime_2",
    "E_prime_z": "E_prime_3",
    "E_ideal_x": "E_ideal_1",
    "E_ideal_y": "E_ideal_2",
    "E_ideal_z": "E_ideal_3",
    "E_Hall_x": "E_Hall_1",
    "E_Hall_y": "E_Hall_2",
    "E_Hall_z": "E_Hall_3",
    # Per-species e/i magnitude shortcuts (math notation + explicit
    # underscore species spellings; both resolve to the Tier-3
    # pipe-bracketed canonical).
    "|Ve|": "|V_s0|",
    "|V_e|": "|V_s0|",
    "|Vi|": "|V_s1|",
    "|V_i|": "|V_s1|",
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
    # Cartesian-component aliases for the total energy flux vector
    # (the bare-prefix forms ``EF``, ``KEF``, ``HF``, ``EHF`` are
    # vector-group expansions in ``GROUP_ALIASES`` below).
    "energy_flux_x": "EF_1",
    "energy_flux_y": "EF_2",
    "energy_flux_z": "EF_3",
}


# Vector-group shorthand: aliases that expand to a three-component group
# at read time (``read(fields=["EFe"]) → EF_s0_1, EF_s0_2, EF_s0_3``). These
# are deliberately separate from ``COMPUTE_ALIASES`` because their target
# is a *prefix*, not a single computable quantity — feeding ``EF_s0`` to
# ``compute()`` would fail. Reader code consults this map after the
# scalar alias map; see ``readers/_registry.py`` for the expansion logic.
GROUP_ALIASES: dict[str, str] = {
    "EFe": "EF_s0",
    "EFi": "EF_s1",
    "KEFe": "KEF_s0",
    "KEFi": "KEF_s1",
    "HFe": "HF_s0",
    "HFi": "HF_s1",
    "EHFe": "EHF_s0",
    "EHFi": "EHF_s1",
    "q_e": "q_s0",
    "q_i": "q_s1",
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
    merged.update(_SCALAR_UNDERSCORE_ALIASES)
    merged.update(_SPECIES_ALIASES)
    return merged


@functools.cache
def _get_field_alias_fallback() -> dict[str, str]:
    """Return the field alias fallback dict (cached, thread-safe)."""
    return _build_field_alias_fallback()


def species_name_aliases(
    species_names: Sequence[str],
    available: Iterable[str],
) -> dict[str, str]:
    """Build species-name-explicit aliases.

    Resolves every canonical name in ``available`` carrying the
    ``_s<i>`` qualifier to an alias that names the species explicitly.
    Three shapes:

    - Scalar per-species: ``P_s0`` → ``P_electrons``.
    - Vector per-species: ``V_s0_1`` → ``V_electrons_1``.
    - Tensor per-species: ``P_s0_11`` → ``P_electrons_11``.

    Unrelated names pass through untouched.

    Parameters
    ----------
    species_names : Sequence[str]
        Species names in declaration order. ``species_names[i]`` is the
        name of the species addressed by the ``_s<i>`` suffix.
    available : Iterable[str]
        Canonical names actually present (e.g. ``ds.data_vars``). Only
        aliases whose target appears here are produced — keeps the alias
        map honest about what can resolve.

    Returns
    -------
    dict[str, str]
        ``{alias: canonical}`` mapping. Empty when no canonical name
        matches the species suffix or when ``species_names`` is empty.

    Examples
    --------
    >>> aliases = species_name_aliases(("electrons", "ions"), ["n_s0", "P_s1"])
    >>> sorted(aliases.items())
    [('P_ions', 'P_s1'), ('n_electrons', 'n_s0')]
    """
    aliases: dict[str, str] = {}
    if not species_names:
        return aliases
    available_set = set(available)
    for canonical in available_set:
        match = SPECIES_SUFFIX_RE.search(canonical)
        if match is None:
            continue
        idx = int(match.group(1))
        if idx >= len(species_names):
            continue
        species_name = species_names[idx].lower()
        if not species_name:
            continue
        base = canonical[: match.start()]
        # Tier-3 form: ``<base>_<species_name><component_suffix>``.
        # Scalars (``P_s0``) land as ``P_electrons``; vectors
        # (``V_s0_1``) as ``V_electrons_1``; tensors (``P_s0_11``)
        # as ``P_electrons_11``.
        component_suffix = match.group("suffix") or ""
        alias = f"{base}_{species_name}{component_suffix}"
        # Don't shadow an existing canonical or earlier alias.
        if alias in available_set or alias in aliases:
            continue
        aliases[alias] = canonical
    return aliases
