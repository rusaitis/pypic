"""The recipe tables behind ``pypic.compute``: what derives from what.

``_REGISTRY`` maps every static derived-quantity name to a `Recipe`;
``_SPECIES_TEMPLATES`` holds the per-species shapes that ``compute``
instantiates on demand for any species index. The dispatcher, lookup and
registration API live in ``pypic.compute``, which owns the write lock
over ``_REGISTRY``; this module is data plus the builders that keep the
literals short.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pypic import derived, diagnostics
from pypic.coordinates import operators

if TYPE_CHECKING:
    from collections.abc import Callable


class SpeciesArgs(StrEnum):
    """Describes which species parameters a dynamic recipe needs."""

    CHARGE_MASS = "charge_mass"
    MASS_ONLY = "mass_only"
    CHARGE_ONLY = "charge_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Recipe:
    """Describes how to derive one quantity from existing fields.

    Mapped from a canonical name in `RECIPES`. ``func`` consumes
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
    # When True the dataset's geometry reaches ``func`` as a ``geometry=``
    # kwarg — the operator-backed recipes (``div_B``, ``div_E``,
    # ``curl_B*``, ``vort*``).  ``psi`` opts out: ``magnetic_flux_function``
    # documents its own Cartesian-only generalization.
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
    the rest pass through unchanged.  Unlike `_vector_recipes`, the
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
    # |Ve| aliases |V_s0|; both resolve through the "|V|" species template.
    # Per-species ``beta_s0``/``beta_s1`` come from the ``"beta"`` template,
    # with ``beta_e``/``beta_i`` aliasing to ``_sN``.
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
    # Characteristic scales for species 0 and 1 (``omega_p_s0``, ``d_s1``,
    # ...) are generated from ``_SPECIES_TEMPLATES`` below the templates.
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
    # Field-aligned decomposition against b̂ = B/|B|; NaN propagates from
    # ``_unit_vector`` where |B| = 0.  Per-species V variants come from the
    # ``"V_par"`` / ``"V_perp_{c}"`` / ``"|V_perp|"`` templates below.
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
    # Pythagorean form |A_perp|² = |A|² - A_par²: one pass over raw
    # inputs, skipping the ``perpendicular_vector`` call and
    # materialization the component path would need.
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
    # E_ideal = -V×B, so E_ideal · B = 0 analytically and ``E_ideal_par``
    # is zero up to roundoff.  Kept for symmetry with the E_par family and
    # as a cross-product precision check.
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
    # Comisso & Bhattacharjee |E'| / (v_A |B|), on the total bulk V.
    # For electron-scale analysis call ``derived.local_reconnection_rate``
    # directly with V_s0.
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
    # Mirrors ``Recipe.component`` for tuple-returning funcs like
    # ``perpendicular_vector``.  The synthesized ``Recipe`` carries it
    # through so the compute path selects the right tuple element.
    component: int | None = None
    supports_relativistic: bool = False


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
    "v_th": SpeciesTemplate(
        derived.thermal_speed,
        ("T_s{N}",),
        SpeciesArgs.MASS_ONLY,
        supports_relativistic=True,
    ),
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


def _species_recipe(template: SpeciesTemplate, species_index: int) -> Recipe:
    """Instantiate *template* for one species index."""
    idx = str(species_index)
    return Recipe(
        func=template.func,
        fields=tuple(f.replace("{N}", idx) for f in template.field_pattern),
        species_index=species_index,
        needs_gamma=template.needs_gamma,
        needs_c=template.needs_c,
        species_args=template.species_args,
        component=template.component,
        supports_relativistic=template.supports_relativistic,
    )


# Species 0 and 1 of the characteristic scales are registered so that
# ``available_quantities()`` and the codegen bundle list them; higher
# indices synthesize on demand from the same templates.
_REGISTRY.update(
    {
        f"{prefix}_s{n}": _species_recipe(_SPECIES_TEMPLATES[prefix], n)
        for prefix in ("omega_p", "omega_c", "d", "v_th", "r", "lambda_D")
        for n in (0, 1)
    }
)
