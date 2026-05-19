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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, assert_never

from scipy import constants

from pypic import derived, diagnostics
from pypic._aliases import (
    COMPUTE_ALIASES,
    GROUP_ALIASES,
    _get_field_alias_fallback,
)
from pypic.coordinates import operators
from pypic.coordinates.geometry import GeometryType
from pypic.exceptions import GeometryUnsupportedError
from pypic.fields import _FIELD_INFO, _SPECIES_QUANTITY_PATTERNS, QuantityType

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray
    from pypic.units import Normalization


class SpeciesArgs(StrEnum):
    """Describes which species parameters a dynamic recipe needs."""

    CHARGE_MASS = "charge_mass"
    MASS_ONLY = "mass_only"
    CHARGE_ONLY = "charge_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Recipe:
    """Describes how to derive one quantity from existing fields.

    Mapped from a canonical name in :data:`RECIPES`. ``func`` consumes
    the dependency arrays declared in ``fields`` (in order) and returns
    the derived array. The remaining attributes describe what extras
    the dispatcher should inject (grid, gamma, species args, …) before
    calling ``func``.
    """

    func: Callable[..., Any]
    fields: tuple[str, ...]
    species_index: int | None = None
    needs_grid: bool = False
    needs_gamma: bool = False
    needs_c: bool = False
    component: int | None = None
    species_args: SpeciesArgs | None = None
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


_PRESSURE_TENSOR_FIELDS = ("P_11", "P_22", "P_33", "P_12", "P_13", "P_23")
_PRESSURE_TENSOR_AND_B = (*_PRESSURE_TENSOR_FIELDS, "B_1", "B_2", "B_3")
# Per-species tensor names follow the Tier-3 template ``<field>_s<N>_<ij>``,
# so ``P_11`` becomes ``P_s{N}_11``.
_SPECIES_PRESSURE_TENSOR_AND_B = (
    *(f.replace("P_", "P_s{N}_") for f in _PRESSURE_TENSOR_FIELDS),
    "B_1",
    "B_2",
    "B_3",
)


def _vector_recipes(
    name_tmpl: str,
    func: Callable[..., Any],
    fields: tuple[str, ...],
    **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to Recipe
) -> dict[str, Recipe]:
    """Three component recipes for a tuple-returning func (curl, Poynting, ...).

    The same ``(func, fields)`` is shared across the three; only
    ``component`` varies.  ``name_tmpl`` uses ``{c}`` for the component
    digit (e.g. ``"S_{c}"``, ``"curl_B_{c}"``).
    """
    return {
        name_tmpl.format(c=c + 1): Recipe(func, fields, component=c, **kwargs)
        for c in range(3)
    }


def _scalar_component_recipes(
    name_tmpl: str,
    func: Callable[..., Any],
    fields_tmpl: tuple[str, ...],
    **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to Recipe
) -> dict[str, Recipe]:
    """Three per-component scalar recipes (``EHF{c}`` style).

    ``fields_tmpl`` entries containing ``{c}`` are expanded per component;
    the rest pass through unchanged.  Unlike :func:`_vector_recipes`, the
    function returns a scalar so ``component`` is not set.
    """
    return {
        name_tmpl.format(c=c): Recipe(
            func,
            tuple(f.format(c=c) if "{c}" in f else f for f in fields_tmpl),
            **kwargs,
        )
        for c in (1, 2, 3)
    }


_REGISTRY: dict[str, Recipe] = {
    # Magnitudes
    "|B|": Recipe(derived.magnetic_field_magnitude, ("B_1", "B_2", "B_3")),
    "|E|": Recipe(derived.electric_field_magnitude, ("E_1", "E_2", "E_3")),
    "|J|": Recipe(derived.current_density_magnitude, ("J_1", "J_2", "J_3")),
    "|V|": Recipe(derived.velocity_magnitude, ("V_1", "V_2", "V_3")),
    # |Ve| is registered as an alias to |V_s0| in _aliases.py — both
    # resolve through the "|V|" species template (Stage E).
    # Plasma parameters.  Per-species ``beta_s0``/``beta_s1`` are produced
    # by the species template ``"beta"``; the literature ``beta_e``/
    # ``beta_i`` spellings alias to ``_sN`` via ``COMPUTE_ALIASES``.
    "beta": Recipe(derived.plasma_beta, ("P", "|B|")),
    "v_A": Recipe(derived.alfven_speed, ("|B|", "rho_m"), supports_relativistic=True),
    "c_s": Recipe(
        derived.sound_speed,
        ("P", "rho_m"),
        needs_gamma=True,
        supports_relativistic=True,
    ),
    "c_ia": Recipe(
        derived.ion_acoustic_speed,
        ("T_s0", "T_s1"),
        species_index=1,
        species_args=SpeciesArgs.MASS_ONLY,
    ),
    "v_ms": Recipe(
        derived.magnetosonic_speed,
        ("v_A", "c_s"),
        supports_relativistic=True,
    ),
    "M_A": Recipe(derived.alfven_mach, ("|V|", "v_A")),
    "M_ms": Recipe(derived.magnetosonic_mach, ("|V|", "v_ms")),
    # Energies
    "e_B": Recipe(derived.magnetic_energy_density, ("|B|",)),
    "e_E": Recipe(derived.electric_energy_density, ("|E|",)),
    "e_k": Recipe(
        derived.kinetic_energy_density,
        ("rho_m", "|V|"),
        supports_relativistic=True,
    ),
    "e_th": Recipe(derived.thermal_energy_density, ("P",), needs_gamma=True),
    "e_th_trace": Recipe(
        derived.thermal_energy_density_trace,
        ("P_11", "P_22", "P_33"),
    ),
    # Thermodynamic
    "h": Recipe(
        derived.enthalpy,
        ("P", "rho_m"),
        needs_gamma=True,
        supports_relativistic=True,
    ),
    "h_rel": Recipe(
        derived.relativistic_enthalpy,
        ("P", "rho_m"),
        needs_gamma=True,
        needs_c=True,
    ),
    "gamma_L": Recipe(derived.lorentz_factor, ("|V|",), needs_c=True),
    "sigma": Recipe(derived.magnetization, ("|B|", "rho_m"), needs_c=True),
    "e_int": Recipe(derived.internal_energy, ("P", "rho_m"), needs_gamma=True),
    "s": Recipe(derived.entropy, ("P", "rho_m"), needs_gamma=True),
    # Per-species entropies (``s_e``, ``s_i``, ``s_gyro_e``, ``s_gyro_i``)
    # come from the ``"s"`` and ``"s_gyro"`` species templates below.
    # Poynting flux (tuple return — component selects)
    **_vector_recipes(
        "S_{c}",
        derived.poynting_flux,
        ("E_1", "E_2", "E_3", "B_1", "B_2", "B_3"),
    ),
    # Enthalpy flux (total, MHD): EHF_i = (gamma/(gamma-1)) P V_i
    **_scalar_component_recipes(
        "EHF_{c}",
        derived.enthalpy_flux_component,
        ("P", "V_{c}"),
        needs_gamma=True,
    ),
    # Species-dependent: Tier-3 canonical recipe IDs.  Literature
    # spellings (``omega_pe``, ``v_th_e``, ``lambda_D``) resolve here
    # via ``COMPUTE_ALIASES``.  Species index >= 2 falls through to
    # ``SPECIES_TEMPLATES`` for dynamic synthesis.
    "omega_p_s0": Recipe(
        derived.plasma_frequency,
        ("n_s0",),
        species_index=0,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "omega_c_s0": Recipe(
        derived.gyrofrequency,
        ("|B|",),
        species_index=0,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "d_s0": Recipe(
        derived.skin_depth,
        ("n_s0",),
        species_index=0,
        needs_c=True,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "v_th_s0": Recipe(
        derived.thermal_speed,
        ("T_s0",),
        species_index=0,
        species_args=SpeciesArgs.MASS_ONLY,
        supports_relativistic=True,
    ),
    "r_s0": Recipe(
        derived.gyroradius,
        ("T_s0", "|B|"),
        species_index=0,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "lambda_D_s0": Recipe(
        derived.debye_length,
        ("T_s0", "n_s0"),
        species_index=0,
        species_args=SpeciesArgs.CHARGE_ONLY,
    ),
    "omega_p_s1": Recipe(
        derived.plasma_frequency,
        ("n_s1",),
        species_index=1,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "omega_c_s1": Recipe(
        derived.gyrofrequency,
        ("|B|",),
        species_index=1,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "d_s1": Recipe(
        derived.skin_depth,
        ("n_s1",),
        species_index=1,
        needs_c=True,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    "v_th_s1": Recipe(
        derived.thermal_speed,
        ("T_s1",),
        species_index=1,
        species_args=SpeciesArgs.MASS_ONLY,
        supports_relativistic=True,
    ),
    "r_s1": Recipe(
        derived.gyroradius,
        ("T_s1", "|B|"),
        species_index=1,
        species_args=SpeciesArgs.CHARGE_MASS,
    ),
    # Total pressure from partial pressures.  ``P_s0``/``P_s1`` each resolve
    # via the species template ``"P"`` (trace of the diagonal tensor) when
    # not stored directly; ``Pe``/``Pi`` continue to work via the alias map.
    "P": Recipe(derived.total_pressure, ("P_s0", "P_s1")),
    # Pressure tensor decomposition (total).  Per-species (``P_par_s0``,
    # ``P_par_e``, ...) is produced by the ``"P_par"`` / ``"P_perp"`` /
    # ``"agyrotropy"`` species templates below.
    "P_par": Recipe(derived.parallel_pressure, _PRESSURE_TENSOR_AND_B),
    "P_perp": Recipe(derived.perpendicular_pressure, _PRESSURE_TENSOR_AND_B),
    "agyrotropy": Recipe(derived.agyrotropy, _PRESSURE_TENSOR_AND_B),
    # Alternative agyrotropy measures: Aunai 2013 (full N Frobenius)
    # and Scudder & Daughton 2008 (perp eigenvalue spread). Swisdak Q
    # is the canonical default; these ship for literature comparison.
    "D_ng": Recipe(derived.aunai_nongyrotropy, _PRESSURE_TENSOR_AND_B),
    "A_phi": Recipe(derived.scudder_agyrotropy, _PRESSURE_TENSOR_AND_B),
    # Field-aligned vector decomposition against b̂ = B/|B|.
    # NaN propagates from ``_unit_vector`` where |B| = 0.
    # Per-species V variants are produced by the ``"V_par"`` /
    # ``"V_perp_{c}"`` / ``"|V_perp|"`` species templates below.
    "J_par": Recipe(
        derived.parallel_component, ("J_1", "J_2", "J_3", "B_1", "B_2", "B_3")
    ),
    "V_par": Recipe(
        derived.parallel_component, ("V_1", "V_2", "V_3", "B_1", "B_2", "B_3")
    ),
    "E_par": Recipe(
        derived.parallel_component, ("E_1", "E_2", "E_3", "B_1", "B_2", "B_3")
    ),
    **_vector_recipes(
        "J_perp_{c}",
        derived.perpendicular_vector,
        ("J_1", "J_2", "J_3", "B_1", "B_2", "B_3"),
    ),
    **_vector_recipes(
        "V_perp_{c}",
        derived.perpendicular_vector,
        ("V_1", "V_2", "V_3", "B_1", "B_2", "B_3"),
    ),
    **_vector_recipes(
        "E_perp_{c}",
        derived.perpendicular_vector,
        ("E_1", "E_2", "E_3", "B_1", "B_2", "B_3"),
    ),
    # Pythagorean form: |A_perp|² = |A|² - A_par², single pass over
    # raw inputs.  Avoids the redundant ``perpendicular_vector`` call
    # and materialization that the component-path recipe would do.
    # Matches the per-species ``|V_perp|`` template below.
    "|J_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("J_1", "J_2", "J_3", "B_1", "B_2", "B_3"),
    ),
    "|V_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("V_1", "V_2", "V_3", "B_1", "B_2", "B_3"),
    ),
    "|E_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("E_1", "E_2", "E_3", "B_1", "B_2", "B_3"),
    ),
    # Non-ideal residual decomposition (E' = E + V×B). Reuses the
    # existing ``E_prime_{1,2,3}`` recipes (below) as inputs.
    # ``E_prime_par`` is the canonical reconnection-rate diagnostic.
    "E_prime_par": Recipe(
        derived.parallel_component,
        ("E_prime_1", "E_prime_2", "E_prime_3", "B_1", "B_2", "B_3"),
    ),
    **_vector_recipes(
        "E_prime_perp_{c}",
        derived.perpendicular_vector,
        ("E_prime_1", "E_prime_2", "E_prime_3", "B_1", "B_2", "B_3"),
    ),
    "|E_prime_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("E_prime_1", "E_prime_2", "E_prime_3", "B_1", "B_2", "B_3"),
    ),
    # Ideal-MHD field decomposition (E_ideal = -V×B). Analytic identity:
    # E_ideal · B = -(V×B) · B = 0, so ``E_ideal_par`` evaluates to zero
    # up to floating-point roundoff. Kept for symmetry and as a
    # cross-product numerical-precision diagnostic.
    "E_ideal_par": Recipe(
        derived.parallel_component,
        ("E_ideal_1", "E_ideal_2", "E_ideal_3", "B_1", "B_2", "B_3"),
    ),
    **_vector_recipes(
        "E_ideal_perp_{c}",
        derived.perpendicular_vector,
        ("E_ideal_1", "E_ideal_2", "E_ideal_3", "B_1", "B_2", "B_3"),
    ),
    "|E_ideal_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("E_ideal_1", "E_ideal_2", "E_ideal_3", "B_1", "B_2", "B_3"),
    ),
    # Hall-field decomposition (E_Hall = J×B / (n_e q_e)). Same identity:
    # ``E_Hall_par`` is analytically zero. Useful for verifying that
    # Hall-term implementations preserve the perpendicularity property.
    "E_Hall_par": Recipe(
        derived.parallel_component,
        ("E_Hall_1", "E_Hall_2", "E_Hall_3", "B_1", "B_2", "B_3"),
    ),
    **_vector_recipes(
        "E_Hall_perp_{c}",
        derived.perpendicular_vector,
        ("E_Hall_1", "E_Hall_2", "E_Hall_3", "B_1", "B_2", "B_3"),
    ),
    "|E_Hall_perp|": Recipe(
        derived.perpendicular_magnitude,
        ("E_Hall_1", "E_Hall_2", "E_Hall_3", "B_1", "B_2", "B_3"),
    ),
    # Grid-dependent diagnostics
    "div_B": Recipe(
        diagnostics.div_b,
        ("B_1", "B_2", "B_3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    "div_E": Recipe(
        diagnostics.div_e,
        ("E_1", "E_2", "E_3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    # Curl of B (tuple return — component selects)
    **_vector_recipes(
        "curl_B_{c}",
        operators.curl,
        ("B_1", "B_2", "B_3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    # Vorticity (tuple return — component selects)
    **_vector_recipes(
        "vort_{c}",
        operators.curl,
        ("V_1", "V_2", "V_3"),
        needs_grid=True,
        passes_geometry=True,
    ),
    # Vorticity magnitude — depends on vort_1/2/3
    "|vort|": Recipe(derived.velocity_magnitude, ("vort_1", "vort_2", "vort_3")),
    # Reconnection diagnostics
    "J_dot_E": Recipe(derived.j_dot_e, ("J_1", "J_2", "J_3", "E_1", "E_2", "E_3")),
    # Zenitani electron-frame dissipation D_e — canonical EDR localizer
    # for collisionless reconnection.  `c` is injected under
    # ``physics.relativistic = true`` for the γ_e prefactor.
    "D_e": Recipe(
        derived.electron_frame_dissipation,
        (
            "J_1",
            "J_2",
            "J_3",
            "E_1",
            "E_2",
            "E_3",
            "V_s0_1",
            "V_s0_2",
            "V_s0_3",
            "B_1",
            "B_2",
            "B_3",
            "rho_c",
        ),
        supports_relativistic=True,
    ),
    # Comisso & Bhattacharjee normalized local reconnection rate
    # |E'| / (v_A |B|).  Single-fluid V — the registry uses the total
    # bulk velocity; for kinetic analysis at electron scales call
    # ``derived.local_reconnection_rate`` directly with V_s0.
    "R_recon": Recipe(
        derived.local_reconnection_rate,
        (
            "E_1",
            "E_2",
            "E_3",
            "V_1",
            "V_2",
            "V_3",
            "B_1",
            "B_2",
            "B_3",
            "v_A",
        ),
    ),
    # Non-ideal electric field E' = E + VxB (component selects)
    **_vector_recipes(
        "E_prime_{c}",
        derived.non_ideal_electric_field,
        ("E_1", "E_2", "E_3", "V_1", "V_2", "V_3", "B_1", "B_2", "B_3"),
    ),
    # Ideal electric field E_ideal = -VxB (component selects)
    **_vector_recipes(
        "E_ideal_{c}",
        derived.ideal_electric_field,
        ("V_1", "V_2", "V_3", "B_1", "B_2", "B_3"),
    ),
    # Hall electric field E_Hall = JxB/(nq) (component selects)
    **_vector_recipes(
        "E_Hall_{c}",
        derived.hall_electric_field,
        ("J_1", "J_2", "J_3", "B_1", "B_2", "B_3", "n_s0"),
        species_index=0,
        species_args=SpeciesArgs.CHARGE_ONLY,
    ),
    # Anisotropy instability parameters
    "firehose": Recipe(derived.firehose_parameter, ("P_par", "P_perp", "|B|")),
    "mirror": Recipe(derived.mirror_parameter, ("P_par", "P_perp", "|B|")),
    # Magnetic flux function (2D only)
    "psi": Recipe(derived.magnetic_flux_function, ("B_2",), needs_grid=True),
}


@dataclass(frozen=True, slots=True)
class SpeciesTemplate:
    """Template for species-dependent derived quantities.

    Used to dynamically synthesize recipes for species index >= 2,
    where static registry entries don't exist.
    """

    func: Callable[..., Any]
    field_pattern: tuple[str, ...]
    species_args: SpeciesArgs
    needs_gamma: bool = False
    needs_c: bool = False
    # Mirrors ``Recipe.component`` — for tuple-returning funcs like
    # ``perpendicular_vector`` used in ``V_perp_{c}`` per-species
    # templates.  The synthesized ``Recipe`` carries it through to the
    # compute path so the right element of the returned tuple is
    # selected.
    component: int | None = None


_SPECIES_TEMPLATES: dict[str, SpeciesTemplate] = {
    "omega_p": SpeciesTemplate(
        derived.plasma_frequency, ("n_s{N}",), SpeciesArgs.CHARGE_MASS
    ),
    "omega_c": SpeciesTemplate(
        derived.gyrofrequency, ("|B|",), SpeciesArgs.CHARGE_MASS
    ),
    "d": SpeciesTemplate(
        derived.skin_depth, ("n_s{N}",), SpeciesArgs.CHARGE_MASS, needs_c=True
    ),
    "v_th": SpeciesTemplate(derived.thermal_speed, ("T_s{N}",), SpeciesArgs.MASS_ONLY),
    "r": SpeciesTemplate(
        derived.gyroradius, ("T_s{N}", "|B|"), SpeciesArgs.CHARGE_MASS
    ),
    "lambda_D": SpeciesTemplate(
        derived.debye_length, ("T_s{N}", "n_s{N}"), SpeciesArgs.CHARGE_ONLY
    ),
    "beta": SpeciesTemplate(derived.plasma_beta, ("P_s{N}", "|B|"), SpeciesArgs.NONE),
    "s": SpeciesTemplate(
        derived.entropy, ("P_s{N}", "n_s{N}"), SpeciesArgs.NONE, needs_gamma=True
    ),
    "s_gyro": SpeciesTemplate(
        derived.gyrotropic_entropy,
        ("P_s{N}_par", "P_s{N}_perp", "n_s{N}"),
        SpeciesArgs.NONE,
    ),
    "P_par": SpeciesTemplate(
        derived.parallel_pressure,
        _SPECIES_PRESSURE_TENSOR_AND_B,
        SpeciesArgs.NONE,
    ),
    "P_perp": SpeciesTemplate(
        derived.perpendicular_pressure,
        _SPECIES_PRESSURE_TENSOR_AND_B,
        SpeciesArgs.NONE,
    ),
    "agyrotropy": SpeciesTemplate(
        derived.agyrotropy,
        _SPECIES_PRESSURE_TENSOR_AND_B,
        SpeciesArgs.NONE,
    ),
    "D_ng": SpeciesTemplate(
        derived.aunai_nongyrotropy,
        _SPECIES_PRESSURE_TENSOR_AND_B,
        SpeciesArgs.NONE,
    ),
    "A_phi": SpeciesTemplate(
        derived.scudder_agyrotropy,
        _SPECIES_PRESSURE_TENSOR_AND_B,
        SpeciesArgs.NONE,
    ),
    # Per-species local reconnection rate. v_A is the bulk Alfvén speed
    # by design — the reference speed is a property of the plasma, not
    # the species — so all R_recon_s{N} share the same denominator.
    "R_recon": SpeciesTemplate(
        derived.local_reconnection_rate,
        (
            "E_1",
            "E_2",
            "E_3",
            "V_s{N}_1",
            "V_s{N}_2",
            "V_s{N}_3",
            "B_1",
            "B_2",
            "B_3",
            "v_A",
        ),
        SpeciesArgs.NONE,
    ),
    # Per-species field-aligned velocity decomposition.
    # V_par scalar; V_perp_{1,2,3} use a single tuple-returning function
    # with ``component=`` to pick the right element.
    "V_par": SpeciesTemplate(
        derived.parallel_component,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "B_1", "B_2", "B_3"),
        SpeciesArgs.NONE,
    ),
    "V_perp_1": SpeciesTemplate(
        derived.perpendicular_vector,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "B_1", "B_2", "B_3"),
        SpeciesArgs.NONE,
        component=0,
    ),
    "V_perp_2": SpeciesTemplate(
        derived.perpendicular_vector,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "B_1", "B_2", "B_3"),
        SpeciesArgs.NONE,
        component=1,
    ),
    "V_perp_3": SpeciesTemplate(
        derived.perpendicular_vector,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "B_1", "B_2", "B_3"),
        SpeciesArgs.NONE,
        component=2,
    ),
    "|V_perp|": SpeciesTemplate(
        derived.perpendicular_magnitude,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "B_1", "B_2", "B_3"),
        SpeciesArgs.NONE,
    ),
    "T": SpeciesTemplate(derived.temperature, ("P_s{N}", "n_s{N}"), SpeciesArgs.NONE),
    "P": SpeciesTemplate(
        derived.isotropic_pressure,
        ("P_s{N}_11", "P_s{N}_22", "P_s{N}_33"),
        SpeciesArgs.NONE,
    ),
    "V_1": SpeciesTemplate(
        derived.bulk_velocity, ("J_s{N}_1", "rho_c_s{N}"), SpeciesArgs.NONE
    ),
    "V_2": SpeciesTemplate(
        derived.bulk_velocity, ("J_s{N}_2", "rho_c_s{N}"), SpeciesArgs.NONE
    ),
    "V_3": SpeciesTemplate(
        derived.bulk_velocity, ("J_s{N}_3", "rho_c_s{N}"), SpeciesArgs.NONE
    ),
    "|V|": SpeciesTemplate(
        derived.velocity_magnitude,
        ("V_s{N}_1", "V_s{N}_2", "V_s{N}_3"),
        SpeciesArgs.NONE,
    ),
    # Per-species mass density: rho_m_s = |rho_c_s| * m / |q|
    "rho_m": SpeciesTemplate(
        derived.species_mass_density, ("rho_c_s{N}",), SpeciesArgs.CHARGE_MASS
    ),
    # Per-species energy densities and thermodynamic quantities
    "e_k": SpeciesTemplate(
        derived.kinetic_energy_density, ("rho_m_s{N}", "|V_s{N}|"), SpeciesArgs.NONE
    ),
    "e_th": SpeciesTemplate(
        derived.thermal_energy_density,
        ("P_s{N}",),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "e_th_trace": SpeciesTemplate(
        derived.thermal_energy_density_trace,
        ("P_s{N}_11", "P_s{N}_22", "P_s{N}_33"),
        SpeciesArgs.NONE,
    ),
    "e_int": SpeciesTemplate(
        derived.internal_energy,
        ("P_s{N}", "rho_m_s{N}"),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "h": SpeciesTemplate(
        derived.enthalpy,
        ("P_s{N}", "rho_m_s{N}"),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    # Kinetic energy flux: KEF_i = (1/2) n m |V|² V_i
    "KEF_1": SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V_s{N}_1", "V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "rho_c_s{N}"),
        SpeciesArgs.CHARGE_MASS,
    ),
    "KEF_2": SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V_s{N}_2", "V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "rho_c_s{N}"),
        SpeciesArgs.CHARGE_MASS,
    ),
    "KEF_3": SpeciesTemplate(
        derived.kinetic_energy_flux_component,
        ("V_s{N}_3", "V_s{N}_1", "V_s{N}_2", "V_s{N}_3", "rho_c_s{N}"),
        SpeciesArgs.CHARGE_MASS,
    ),
    # Heat flux: HF_i = EF_i - KEF_i (thermal + heat flux residual)
    "HF_1": SpeciesTemplate(
        derived.heat_flux_component, ("EF_s{N}_1", "KEF_s{N}_1"), SpeciesArgs.NONE
    ),
    "HF_2": SpeciesTemplate(
        derived.heat_flux_component, ("EF_s{N}_2", "KEF_s{N}_2"), SpeciesArgs.NONE
    ),
    "HF_3": SpeciesTemplate(
        derived.heat_flux_component, ("EF_s{N}_3", "KEF_s{N}_3"), SpeciesArgs.NONE
    ),
    # Enthalpy flux (per-species): EHF_i = (gamma/(gamma-1)) P_s V_i_s
    "EHF_1": SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V_s{N}_1"),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "EHF_2": SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V_s{N}_2"),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    "EHF_3": SpeciesTemplate(
        derived.enthalpy_flux_component,
        ("P_s{N}", "V_s{N}_3"),
        SpeciesArgs.NONE,
        needs_gamma=True,
    ),
    # Conductive heat flux: q_i = HF_i - EHF_i (non-adiabatic residual)
    "q_1": SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF_s{N}_1", "EHF_s{N}_1"),
        SpeciesArgs.NONE,
    ),
    "q_2": SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF_s{N}_2", "EHF_s{N}_2"),
        SpeciesArgs.NONE,
    ),
    "q_3": SpeciesTemplate(
        derived.conductive_heat_flux_component,
        ("HF_s{N}_3", "EHF_s{N}_3"),
        SpeciesArgs.NONE,
    ),
}

# Match the species qualifier ``_s<N>`` in any of four positions:
#   1. End (scalar per-species like ``omega_p_s2``, ``P_s0``, ``T_s1``).
#   2. Middle followed by a component / modifier suffix (vector/tensor
#      per-species or generic operator under Tier-3 canonical:
#      ``V_s0_1``, ``q_s0_1``, ``P_s0_11``, ``P_s0_par``,
#      ``P_s0_perp``, ``V_s0_perp_1``).
#   3. Middle followed by a closing pipe (per-species magnitude:
#      ``|V_s0|``, ``|J_s1|``).
#   4. Middle followed by an operator + closing pipe (per-species
#      operator magnitude: ``|V_s0_perp|``).
# The template lookup key is ``<prefix><suffix>`` — for ``V_s0_1`` it's
# ``"V_1"``, for ``P_s0_par`` it's ``"P_par"``, for ``|V_s0|`` it's
# ``"|V|"``, for ``|V_s0_perp|`` it's ``"|V_perp|"``. The ``_[^|]+\|``
# alternative comes first so the operator-plus-pipe form wins over the
# bare ``_[^|]+`` form when both could match.
_SPECIES_SUFFIX_RE = re.compile(
    r"^(?P<prefix>.+?)_s(?P<idx>\d+)(?P<suffix>_[^|]+\||_[^|]+|\|)?$"
)

# Generic operator suffixes that must sit *after* the species qualifier
# in Tier-3 canonical names (``P_s0_par`` is valid; ``P_par_s0`` is not).
# When the regex puts these in the prefix (``P_par`` + ``_s0`` + ``""``),
# reject so the legacy split-form (``P_par_s0``) raises ``KeyError``.
_INVALID_PREFIX_OPERATOR_ENDINGS: tuple[str, ...] = ("_par", "_perp", "|")


def _try_species_recipe(name: str) -> Recipe | None:
    """Try to build a recipe from species templates for names like ``omega_p_s2``.

    Returns ``None`` if the name doesn't match any template.
    """
    m = _SPECIES_SUFFIX_RE.match(name)
    if m is None:
        return None
    raw_prefix = m.group("prefix")
    raw_suffix = m.group("suffix") or ""
    # Tier-3 canonical names put generic operators *after* the species
    # qualifier.  Anchor the rule by rejecting matches whose prefix
    # ends with a generic operator and whose suffix is empty (the
    # legacy ``P_par_s0`` / ``|V|_s0`` shape).
    if not raw_suffix and raw_prefix.endswith(_INVALID_PREFIX_OPERATOR_ENDINGS):
        return None
    prefix = raw_prefix + raw_suffix
    idx_str = m.group("idx")
    species_index = int(idx_str)
    template = _SPECIES_TEMPLATES.get(prefix)
    if template is None:
        return None
    fields = tuple(f.replace("{N}", idx_str) for f in template.field_pattern)
    return Recipe(
        func=template.func,
        fields=fields,
        species_index=species_index,
        needs_gamma=template.needs_gamma,
        needs_c=template.needs_c,
        species_args=template.species_args,
        component=template.component,
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
    return COMPUTE_ALIASES.get(name, name)


def _get_recipe(name: str) -> Recipe:
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
    # Check if this is a vector group alias (e.g. "EFe" → read-time only)
    if name in GROUP_ALIASES or canonical in GROUP_ALIASES:
        group_name = name if name in GROUP_ALIASES else canonical
        msg = (
            f"{name!r} is a vector group (expands to 3 components). "
            f"Use {group_name}1/{group_name}2/{group_name}3 in compute(), "
            f'or read(fields=["{group_name}"]) to load all three.'
        )
        raise KeyError(msg) from None
    all_names = available_quantities()
    suggestions = difflib.get_close_matches(name, all_names, n=3, cutoff=0.4)
    msg = f"Unknown derived quantity {name!r}."
    if suggestions:
        msg += f" Did you mean: {suggestions}?"
    msg += " Call available_quantities() for the full list."
    raise KeyError(msg) from None


def _get_species_args(
    dataset: FieldDataset,
    recipe: Recipe,
) -> list[float]:
    """Extract charge and mass from species info for a recipe."""
    if recipe.species_index is None:
        return []
    if recipe.species_args is SpeciesArgs.NONE:
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
    kind: SpeciesArgs,
) -> None:
    """Append the right species parameters based on the descriptor."""
    match kind:
        case SpeciesArgs.CHARGE_MASS:
            args.extend(species_args)
        case SpeciesArgs.MASS_ONLY:
            args.append(species_args[1])
        case SpeciesArgs.CHARGE_ONLY:
            args.append(species_args[0])
        case SpeciesArgs.NONE:
            pass
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _execute_recipe(
    canonical: str,
    dataset: FieldDataset,
    _depth: int = 0,
) -> tuple[Recipe, Any]:
    """Build arguments for a recipe and invoke its pure function.

    Returns ``(recipe, full_result)``. ``full_result`` is the raw output
    of ``recipe.func``: a tuple for multi-output recipes (curl, gradient,
    vorticity) and a scalar array otherwise. Callers that want a single
    component should index into the result via ``recipe.component``.

    Shared between :func:`compute_field` (single-component path) and
    :meth:`FieldDataset._attach_vector_siblings` (multi-component path)
    so the two cannot drift on argument construction or geometry handling.
    Dependency resolution recurses through ``compute_field`` to benefit
    from its alias handling and cycle guard.
    """
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
            raise GeometryUnsupportedError(msg)
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
    return recipe, result


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

    recipe, result = _execute_recipe(canonical, dataset, _depth)

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
        # Resolve field aliases (Bx→B_1, B_x→B_1, P_e→Pe, etc.)
        fallback = _get_field_alias_fallback()
        canonical = fallback.get(canonical, canonical)
        info = _FIELD_INFO.get(canonical)
    quantity_type: str | None = info.quantity_type if info is not None else None
    if quantity_type is None:
        # Try regex patterns for per-species fields (n_s2, J_s3_1, etc.)
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
    return sorted(set(_REGISTRY) | set(COMPUTE_ALIASES))


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
    recipes (e.g. ``S_1/S_2/S_3``, ``curl_B_1/2/3``). Returns an empty
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

    recipe = Recipe(
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


# Public read-only view of the recipe registry. Codegen consumers
# (webpic, rustpic tooling) iterate ``RECIPES.items()`` to generate
# cross-language mirrors. The underlying ``_REGISTRY`` stays a mutable
# dict because :func:`register_recipe` / :func:`unregister_recipe`
# update it under ``_recipe_lock``; the proxy guarantees external
# callers only see the read side.
RECIPES: MappingProxyType[str, Recipe] = MappingProxyType(_REGISTRY)
SPECIES_TEMPLATES: MappingProxyType[str, SpeciesTemplate] = MappingProxyType(
    _SPECIES_TEMPLATES
)


__all__ = [
    "RECIPES",
    "SPECIES_TEMPLATES",
    "Recipe",
    "SpeciesArgs",
    "SpeciesTemplate",
    "available_quantities",
    "compute_field",
    "display_unit_factor",
    "field_dependencies",
    "field_si_factor",
    "register_recipe",
    "unregister_recipe",
]
