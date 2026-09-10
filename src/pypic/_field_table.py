"""Every canonical field's metadata, as one literal.

``_FIELD_INFO`` maps a canonical name to its `FieldInfo`: quantity type,
SI unit label, long name and LaTeX symbol. ``pypic.fields`` reads it for
lookups, extends it through ``register_field``, and synthesizes the
per-species entries that would otherwise repeat it three times over.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldInfo:
    r"""Metadata for a single field or derived quantity.

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
    unit_dimension : tuple[int, ...] | None
        openPMD-style 7-tuple of integer SI base-unit powers
        (length, mass, time, current, temperature, amount, luminosity).
        ``None`` (the default) defers to ``quantity_dimension(quantity_type)``
        at attrs-population time; a non-``None`` value overrides the
        canonical lookup for custom-registered fields.
    """

    quantity_type: str
    long_name: str
    si_unit: str
    latex: str = ""
    unit_dimension: tuple[int, int, int, int, int, int, int] | None = None


# Helper to keep long FieldInfo constructors within 88 columns
_FI = FieldInfo


def _vec_three(
    name_tmpl: str,
    qtype: str,
    long_tmpl: str,
    si_unit: str,
    latex_tmpl: str,
    mag: tuple[str, str, str] | None = None,
) -> dict[str, FieldInfo]:
    """Expand a vector-field spec into three component entries (+ optional magnitude).

    Templates use ``{c}`` for the component digit.  LaTeX templates must
    double any literal braces (``{{``, ``}}``).  When *mag* is provided,
    it is ``(name, long_name, latex)`` for the magnitude entry.
    """
    out: dict[str, FieldInfo] = {
        name_tmpl.format(c=c): FieldInfo(
            qtype, long_tmpl.format(c=c), si_unit, latex_tmpl.format(c=c)
        )
        for c in (1, 2, 3)
    }
    if mag is not None:
        name, long_name, latex = mag
        out[name] = FieldInfo(qtype, long_name, si_unit, latex)
    return out


_FIELD_INFO: dict[str, FieldInfo] = {
    # Electromagnetic fields
    **_vec_three(
        "B_{c}",
        "b_field",
        "Magnetic field component {c}",
        "T",
        r"$B_{c}$",
        mag=("|B|", "Magnetic field magnitude", r"$|B|$"),
    ),
    **_vec_three(
        "B0_{c}",
        "b_field",
        "Background B component {c}",
        "T",
        r"$B_{{0,{c}}}$",
    ),
    **_vec_three(
        "E_{c}",
        "e_field",
        "Electric field component {c}",
        "V/m",
        r"$E_{c}$",
        mag=("|E|", "Electric field magnitude", r"$|E|$"),
    ),
    # Current density
    **_vec_three(
        "J_{c}",
        "current_density",
        "Current density component {c}",
        "A/m^2",
        r"$J_{c}$",
        mag=("|J|", "Current density magnitude", r"$|J|$"),
    ),
    # Velocities
    **_vec_three(
        "V_{c}",
        "velocity",
        "Bulk velocity component {c}",
        "m/s",
        r"$V_{c}$",
        mag=("|V|", "Bulk velocity magnitude", r"$|V|$"),
    ),
    "v_A": _FI("velocity", "Alfvén speed", "m/s", r"$v_A$"),
    "c_s": _FI("velocity", "Sound speed", "m/s", r"$c_s$"),
    "c_ia": _FI("velocity", "Ion acoustic speed", "m/s", r"$c_{ia}$"),
    "v_ms": _FI("velocity", "Fast magnetosonic speed", "m/s", r"$v_{ms}$"),
    "v_th_e": _FI("velocity", "Electron thermal speed", "m/s", r"$v_{th,e}$"),
    "v_th_i": _FI("velocity", "Ion thermal speed", "m/s", r"$v_{th,i}$"),
    # Four-velocity
    **_vec_three(
        "u_{c}",
        "four_velocity",
        "Four-velocity component {c}",
        "m/s",
        r"$u_{c}$",
    ),
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
    # Field-aligned vector decomposition (against b̂ = B/|B|).
    "J_par": _FI(
        "current_density", "Parallel current density", "A/m^2", r"$J_\parallel$"
    ),
    "J_perp_1": _FI(
        "current_density",
        "Perpendicular current density (component 1)",
        "A/m^2",
        r"$J_{\perp,1}$",
    ),
    "J_perp_2": _FI(
        "current_density",
        "Perpendicular current density (component 2)",
        "A/m^2",
        r"$J_{\perp,2}$",
    ),
    "J_perp_3": _FI(
        "current_density",
        "Perpendicular current density (component 3)",
        "A/m^2",
        r"$J_{\perp,3}$",
    ),
    "|J_perp|": _FI(
        "current_density",
        "Perpendicular current density magnitude",
        "A/m^2",
        r"$|\mathbf{J}_\perp|$",
    ),
    "V_par": _FI("velocity", "Parallel bulk velocity", "m/s", r"$V_\parallel$"),
    "V_perp_1": _FI(
        "velocity",
        "Perpendicular bulk velocity (component 1)",
        "m/s",
        r"$V_{\perp,1}$",
    ),
    "V_perp_2": _FI(
        "velocity",
        "Perpendicular bulk velocity (component 2)",
        "m/s",
        r"$V_{\perp,2}$",
    ),
    "V_perp_3": _FI(
        "velocity",
        "Perpendicular bulk velocity (component 3)",
        "m/s",
        r"$V_{\perp,3}$",
    ),
    "|V_perp|": _FI(
        "velocity",
        "Perpendicular bulk velocity magnitude",
        "m/s",
        r"$|\mathbf{V}_\perp|$",
    ),
    "E_par": _FI("e_field", "Parallel electric field", "V/m", r"$E_\parallel$"),
    "E_perp_1": _FI(
        "e_field",
        "Perpendicular electric field (component 1)",
        "V/m",
        r"$E_{\perp,1}$",
    ),
    "E_perp_2": _FI(
        "e_field",
        "Perpendicular electric field (component 2)",
        "V/m",
        r"$E_{\perp,2}$",
    ),
    "E_perp_3": _FI(
        "e_field",
        "Perpendicular electric field (component 3)",
        "V/m",
        r"$E_{\perp,3}$",
    ),
    "|E_perp|": _FI(
        "e_field",
        "Perpendicular electric field magnitude",
        "V/m",
        r"$|\mathbf{E}_\perp|$",
    ),
    "E_prime_par": _FI(
        "e_field",
        "Parallel non-ideal electric field",
        "V/m",
        r"$E'_\parallel$",
    ),
    "E_prime_perp_1": _FI(
        "e_field",
        "Perpendicular non-ideal electric field (component 1)",
        "V/m",
        r"$E'_{\perp,1}$",
    ),
    "E_prime_perp_2": _FI(
        "e_field",
        "Perpendicular non-ideal electric field (component 2)",
        "V/m",
        r"$E'_{\perp,2}$",
    ),
    "E_prime_perp_3": _FI(
        "e_field",
        "Perpendicular non-ideal electric field (component 3)",
        "V/m",
        r"$E'_{\perp,3}$",
    ),
    "|E_prime_perp|": _FI(
        "e_field",
        "Perpendicular non-ideal electric field magnitude",
        "V/m",
        r"$|\mathbf{E}'_\perp|$",
    ),
    # Ideal-MHD field decomposition.  ``E_ideal_par`` is analytically
    # zero (``E_ideal = -V×B`` is perpendicular to B by construction)
    # — kept for diagnostic symmetry with ``E_par`` / ``E_prime_par``.
    "E_ideal_par": _FI(
        "e_field",
        "Parallel ideal electric field (analytically zero)",
        "V/m",
        r"$E^{\mathrm{ideal}}_\parallel$",
    ),
    "E_ideal_perp_1": _FI(
        "e_field",
        "Perpendicular ideal electric field (component 1)",
        "V/m",
        r"$E^{\mathrm{ideal}}_{\perp,1}$",
    ),
    "E_ideal_perp_2": _FI(
        "e_field",
        "Perpendicular ideal electric field (component 2)",
        "V/m",
        r"$E^{\mathrm{ideal}}_{\perp,2}$",
    ),
    "E_ideal_perp_3": _FI(
        "e_field",
        "Perpendicular ideal electric field (component 3)",
        "V/m",
        r"$E^{\mathrm{ideal}}_{\perp,3}$",
    ),
    "|E_ideal_perp|": _FI(
        "e_field",
        "Perpendicular ideal electric field magnitude",
        "V/m",
        r"$|\mathbf{E}^{\mathrm{ideal}}_\perp|$",
    ),
    # Hall-field decomposition.  Same identity: ``E_Hall_par`` is
    # analytically zero (``J×B`` perpendicular to B).
    "E_Hall_par": _FI(
        "e_field",
        "Parallel Hall electric field (analytically zero)",
        "V/m",
        r"$E^{\mathrm{Hall}}_\parallel$",
    ),
    "E_Hall_perp_1": _FI(
        "e_field",
        "Perpendicular Hall electric field (component 1)",
        "V/m",
        r"$E^{\mathrm{Hall}}_{\perp,1}$",
    ),
    "E_Hall_perp_2": _FI(
        "e_field",
        "Perpendicular Hall electric field (component 2)",
        "V/m",
        r"$E^{\mathrm{Hall}}_{\perp,2}$",
    ),
    "E_Hall_perp_3": _FI(
        "e_field",
        "Perpendicular Hall electric field (component 3)",
        "V/m",
        r"$E^{\mathrm{Hall}}_{\perp,3}$",
    ),
    "|E_Hall_perp|": _FI(
        "e_field",
        "Perpendicular Hall electric field magnitude",
        "V/m",
        r"$|\mathbf{E}^{\mathrm{Hall}}_\perp|$",
    ),
    "P_11": _FI("pressure", "Pressure tensor P_11", "Pa", r"$P_{11}$"),
    "P_22": _FI("pressure", "Pressure tensor P_22", "Pa", r"$P_{22}$"),
    "P_33": _FI("pressure", "Pressure tensor P_33", "Pa", r"$P_{33}$"),
    "P_12": _FI("pressure", "Pressure tensor P_12", "Pa", r"$P_{12}$"),
    "P_13": _FI("pressure", "Pressure tensor P_13", "Pa", r"$P_{13}$"),
    "P_23": _FI("pressure", "Pressure tensor P_23", "Pa", r"$P_{23}$"),
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
    "e_th_trace": _FI(
        "energy_density",
        "Thermal energy density (tensor trace)",
        "J/m^3",
        r"$e_{th,\mathrm{tr}}$",
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
    **_vec_three(
        "S_{c}",
        "poynting_flux",
        "Poynting flux component {c}",
        "W/m^2",
        r"$S_{c}$",
    ),
    **_vec_three(
        "EF_{c}",
        "energy_flux",
        "Energy flux component {c}",
        "W/m^2",
        r"$EF_{c}$",
    ),
    **_vec_three(
        "EHF_{c}",
        "energy_flux",
        "Enthalpy flux component {c}",
        "W/m^2",
        r"$EHF_{c}$",
    ),
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
    **_vec_three(
        "curl_B_{c}",
        "b_field_per_length",
        "Curl of B component {c}",
        "T/m",
        r"$(\nabla \times B)_{c}$",
    ),
    **_vec_three(
        "vort_{c}",
        "velocity_per_length",
        "Vorticity component {c}",
        "1/s",
        r"$\omega_{c}$",
        mag=("|vort|", "Vorticity magnitude", r"$|\omega|$"),
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
    "agyrotropy_e": _FI("dimensionless", "Electron agyrotropy", "", r"$Q_e$"),
    "agyrotropy_i": _FI("dimensionless", "Ion agyrotropy", "", r"$Q_i$"),
    "D_ng": _FI("dimensionless", "Aunai nongyrotropy", "", r"$D_{ng}$"),
    "D_ng_e": _FI("dimensionless", "Electron Aunai nongyrotropy", "", r"$D_{ng,e}$"),
    "D_ng_i": _FI("dimensionless", "Ion Aunai nongyrotropy", "", r"$D_{ng,i}$"),
    "A_phi": _FI("dimensionless", "Scudder agyrotropy", "", r"$A_\phi$"),
    "A_phi_e": _FI("dimensionless", "Electron Scudder agyrotropy", "", r"$A_{\phi,e}$"),
    "A_phi_i": _FI("dimensionless", "Ion Scudder agyrotropy", "", r"$A_{\phi,i}$"),
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
    "D_e": _FI(
        "power_density",
        "Electron-frame dissipation",
        "W/m^3",
        r"$D_e$",
    ),
    "R_recon": _FI(
        "dimensionless",
        "Local reconnection rate",
        "",
        r"$R_{\mathrm{recon}}$",
    ),
    "R_recon_e": _FI(
        "dimensionless",
        "Electron-frame local reconnection rate",
        "",
        r"$R_{\mathrm{recon},e}$",
    ),
    "R_recon_i": _FI(
        "dimensionless",
        "Ion-frame local reconnection rate",
        "",
        r"$R_{\mathrm{recon},i}$",
    ),
    **_vec_three(
        "E_prime_{c}",
        "e_field",
        "Non-ideal electric field, component {c}",
        "V/m",
        r"$E'_{c}$",
    ),
    **_vec_three(
        "E_ideal_{c}",
        "e_field",
        "Ideal electric field, component {c}",
        "V/m",
        r"$E_{{ideal,{c}}}$",
    ),
    **_vec_three(
        "E_Hall_{c}",
        "e_field",
        "Hall electric field, component {c}",
        "V/m",
        r"$E_{{Hall,{c}}}$",
    ),
    "firehose": _FI("dimensionless", "Firehose parameter", "", r"$\mathcal{F}$"),
    "mirror": _FI("dimensionless", "Mirror parameter", "", r"$\mathcal{M}$"),
    "psi": _FI("b_field", "Magnetic flux function", "T", r"$\psi$"),
}
