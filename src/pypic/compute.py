"""String-based dispatch for derived quantities on FieldDataset.

Maps short names (``"|B|"``, ``"beta"``, ``"v_A"``, ...) to pure functions
in ``derived.py``, ``diagnostics.py``, and ``operators.py``. Never imports
FieldDataset — avoids circular dependencies. FieldDataset methods call into
this module via deferred imports.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scipy import constants

from pypic import derived, diagnostics
from pypic.coordinates import operators

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.readers.base import FieldDataset
    from pypic.types import FloatArray
    from pypic.units import Normalization


@dataclass(frozen=True, slots=True)
class _Recipe:
    func: Callable[..., Any]
    fields: tuple[str, ...]
    species_index: int | None = None
    needs_grid: bool = False
    needs_gamma: bool = False
    needs_c: bool = False
    component: int | None = None


_PRESSURE_TENSOR_FIELDS = ("P11", "P22", "P33", "P12", "P13", "P23")
_PRESSURE_TENSOR_AND_B = (*_PRESSURE_TENSOR_FIELDS, "B1", "B2", "B3")

_REGISTRY: dict[str, _Recipe] = {
    # Magnitudes
    "|B|": _Recipe(derived.magnetic_field_magnitude, ("B1", "B2", "B3")),
    "|E|": _Recipe(derived.electric_field_magnitude, ("E1", "E2", "E3")),
    "|J|": _Recipe(derived.current_density_magnitude, ("J1", "J2", "J3")),
    "|V|": _Recipe(derived.velocity_magnitude, ("V1", "V2", "V3")),
    # Plasma parameters
    "beta": _Recipe(derived.plasma_beta, ("P", "|B|")),
    "beta_e": _Recipe(derived.plasma_beta, ("Pe", "|B|")),
    "beta_i": _Recipe(derived.plasma_beta, ("Pi", "|B|")),
    "v_A": _Recipe(derived.alfven_speed, ("|B|", "rho_m")),
    "c_s": _Recipe(derived.sound_speed, ("P", "rho_m"), needs_gamma=True),
    "c_ia": _Recipe(derived.ion_acoustic_speed, ("Te", "Ti"), species_index=1),
    "v_ms": _Recipe(derived.magnetosonic_speed, ("v_A", "c_s")),
    "M_A": _Recipe(derived.alfven_mach, ("|V|", "v_A")),
    "M_ms": _Recipe(derived.magnetosonic_mach, ("|V|", "v_ms")),
    # Energies
    "e_B": _Recipe(derived.magnetic_energy_density, ("|B|",)),
    "e_E": _Recipe(derived.electric_energy_density, ("|E|",)),
    "e_k": _Recipe(derived.kinetic_energy_density, ("rho_m", "|V|")),
    "e_th": _Recipe(derived.thermal_energy_density, ("P",), needs_gamma=True),
    # Thermodynamic
    "h": _Recipe(derived.enthalpy, ("P", "rho_m"), needs_gamma=True),
    "h_rel": _Recipe(
        derived.relativistic_enthalpy,
        ("P", "rho_m"),
        needs_gamma=True,
        needs_c=True,
    ),
    "e_int": _Recipe(derived.internal_energy, ("P", "rho_m"), needs_gamma=True),
    "s": _Recipe(derived.entropy, ("P", "rho_m"), needs_gamma=True),
    "s_e": _Recipe(derived.entropy, ("Pe", "n_s0"), needs_gamma=True),
    "s_i": _Recipe(derived.entropy, ("Pi", "n_s1"), needs_gamma=True),
    "s_gyro": _Recipe(derived.gyrotropic_entropy, ("P_par", "P_perp", "n_s0")),
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
    # Species-dependent: electrons (species 0)
    "omega_pe": _Recipe(derived.plasma_frequency, ("n_s0",), species_index=0),
    "omega_ce": _Recipe(derived.gyrofrequency, ("|B|",), species_index=0),
    "d_e": _Recipe(derived.skin_depth, ("n_s0",), species_index=0, needs_c=True),
    "v_th_e": _Recipe(derived.thermal_speed, ("Te",), species_index=0),
    "r_e": _Recipe(derived.gyroradius, ("Te", "|B|"), species_index=0),
    "lambda_D": _Recipe(derived.debye_length, ("Te", "n_s0"), species_index=0),
    # Species-dependent: ions (species 1)
    "omega_pi": _Recipe(derived.plasma_frequency, ("n_s1",), species_index=1),
    "omega_ci": _Recipe(derived.gyrofrequency, ("|B|",), species_index=1),
    "d_i": _Recipe(derived.skin_depth, ("n_s1",), species_index=1, needs_c=True),
    "v_th_i": _Recipe(derived.thermal_speed, ("Ti",), species_index=1),
    "r_i": _Recipe(derived.gyroradius, ("Ti", "|B|"), species_index=1),
    # Pressure tensor
    "P_par": _Recipe(derived.parallel_pressure, _PRESSURE_TENSOR_AND_B),
    "P_perp": _Recipe(derived.perpendicular_pressure, _PRESSURE_TENSOR_AND_B),
    "agyrotropy": _Recipe(derived.agyrotropy, _PRESSURE_TENSOR_AND_B),
    # Grid-dependent diagnostics
    "div_B": _Recipe(diagnostics.div_b, ("B1", "B2", "B3"), needs_grid=True),
    "div_E": _Recipe(diagnostics.div_e, ("E1", "E2", "E3"), needs_grid=True),
    # Curl of B (tuple return — component selects)
    "curl_B1": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=0,
    ),
    "curl_B2": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=1,
    ),
    "curl_B3": _Recipe(
        operators.curl,
        ("B1", "B2", "B3"),
        needs_grid=True,
        component=2,
    ),
    # Vorticity (tuple return — component selects)
    "vort1": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=0,
    ),
    "vort2": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=1,
    ),
    "vort3": _Recipe(
        operators.curl,
        ("V1", "V2", "V3"),
        needs_grid=True,
        component=2,
    ),
    # Vorticity magnitude — depends on vort1/2/3
    "|vort|": _Recipe(derived.velocity_magnitude, ("vort1", "vort2", "vort3")),
}

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
    "s_gyro_e": "s_gyro",
}

# Maps field/derived names to physical quantity types for SI conversion
_FIELD_QUANTITY_MAP: dict[str, str] = {
    # Electromagnetic fields
    "B1": "b_field",
    "B2": "b_field",
    "B3": "b_field",
    "|B|": "b_field",
    "E1": "e_field",
    "E2": "e_field",
    "E3": "e_field",
    "|E|": "e_field",
    # Current density
    "J1": "current_density",
    "J2": "current_density",
    "J3": "current_density",
    "|J|": "current_density",
    # Velocities
    "V1": "velocity",
    "V2": "velocity",
    "V3": "velocity",
    "|V|": "velocity",
    "Ve1": "velocity",
    "Ve2": "velocity",
    "Ve3": "velocity",
    "v_A": "velocity",
    "c_s": "velocity",
    "c_ia": "velocity",
    "v_ms": "velocity",
    "v_th_e": "velocity",
    "v_th_i": "velocity",
    # Four-velocity
    "u1": "velocity",
    "u2": "velocity",
    "u3": "velocity",
    # Densities
    "rho_m": "mass_density",
    "rho_c": "charge_density",
    "n_s0": "density",
    "n_s1": "density",
    "n_e": "density",
    "n_i": "density",
    # Pressure
    "P": "pressure",
    "Pe": "pressure",
    "Pi": "pressure",
    "P_par": "pressure",
    "P_perp": "pressure",
    "P11": "pressure",
    "P22": "pressure",
    "P33": "pressure",
    "P12": "pressure",
    "P13": "pressure",
    "P23": "pressure",
    # Temperature
    "Te": "temperature",
    "Ti": "temperature",
    # Energy densities
    "e_B": "energy_density",
    "e_E": "energy_density",
    "e_k": "energy_density",
    "e_th": "energy_density",
    # Frequencies
    "omega_pe": "frequency",
    "omega_pi": "frequency",
    "omega_ce": "frequency",
    "omega_ci": "frequency",
    # Lengths
    "d_e": "length",
    "d_i": "length",
    "r_e": "length",
    "r_i": "length",
    "lambda_D": "length",
    # Poynting flux
    "S1": "poynting_flux",
    "S2": "poynting_flux",
    "S3": "poynting_flux",
    # Thermodynamic (specific quantities — energy per unit mass → velocity²)
    "h": "temperature",
    "h_rel": "temperature",
    "e_int": "temperature",
    # Diagnostics (per-length quantities)
    "div_B": "b_field",  # really b_field/length but same SI factor pattern
    "div_E": "e_field",
    "curl_B1": "b_field",
    "curl_B2": "b_field",
    "curl_B3": "b_field",
    "vort1": "frequency",
    "vort2": "frequency",
    "vort3": "frequency",
    "|vort|": "frequency",
    # Dimensionless
    "beta": "dimensionless",
    "beta_e": "dimensionless",
    "beta_i": "dimensionless",
    "M_A": "dimensionless",
    "M_ms": "dimensionless",
    "s": "dimensionless",
    "s_e": "dimensionless",
    "s_i": "dimensionless",
    "s_gyro": "dimensionless",
    "s_gyro_e": "dimensionless",
    "s_gyro_i": "dimensionless",
    "agyrotropy": "dimensionless",
}

# Display unit conversion: unit string → SI value
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
    "kg/m^3": 1.0,
    "Pa": 1.0,
    "nPa": 1e-9,
    "J/m^3": 1.0,
    "W/m^2": 1.0,
    "Hz": 1.0,
    "rad/s": 1.0,
    "m": 1.0,
    "km": 1e3,
    "RE": 6.371e6,
    "eV": constants.eV,
    "keV": 1e3 * constants.eV,
    "K": constants.k,
    "A/m^2": 1.0,
    "C/m^3": 1.0,
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
        all_names = sorted(set(_REGISTRY) | set(_COMPUTE_ALIASES))
        suggestions = difflib.get_close_matches(name, all_names, n=3, cutoff=0.4)
        msg = f"Unknown derived quantity {name!r}."
        if suggestions:
            msg += f" Did you mean: {suggestions}?"
        msg += f" Available: {all_names}"
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
    assert sp.charge is not None
    assert sp.mass is not None
    return [sp.charge, sp.mass]


def _get_gamma(dataset: FieldDataset) -> float:
    """Get the adiabatic index from physics config, default 5/3."""
    return float(dataset.physics.get("gamma", 5.0 / 3.0))


def _get_c(dataset: FieldDataset) -> float:
    """Get the speed of light from physics config, default 1.0."""
    return float(dataset.physics.get("c", 1.0))


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

    canonical = _resolve_name(name)

    # Direct field lookup (canonical or alias)
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
    # For species-dependent functions, we need to figure out
    # which extra args the function expects (charge, mass, or just mass)
    if species_args:
        func = recipe.func
        if func is derived.thermal_speed:
            # thermal_speed(temperature, mass)
            args.append(species_args[1])
        elif func is derived.ion_acoustic_speed:
            # ion_acoustic_speed(Te, Ti, mass_i, gamma_e, gamma_i)
            args.append(species_args[1])
        elif func is derived.gyrofrequency:
            # gyrofrequency(b, charge, mass)
            args.extend(species_args)
        elif func is derived.plasma_frequency:
            # plasma_frequency(density, charge, mass)
            args.extend(species_args)
        elif func is derived.skin_depth:
            # skin_depth(density, charge, mass, c)
            args.extend(species_args)
        elif func is derived.gyroradius:
            # gyroradius(temperature, b, charge, mass)
            args.extend(species_args)
        elif func is derived.debye_length:
            # debye_length(temperature, density, charge)
            args.append(species_args[0])

    # Append gamma
    if recipe.needs_gamma:
        args.append(_get_gamma(dataset))

    # Append c
    if recipe.needs_c:
        args.append(_get_c(dataset))

    # Append grid spacing
    if recipe.needs_grid:
        args.extend(dataset.grid.spacing)

    result = recipe.func(*args)

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
    # Also resolve FieldDataset aliases (Bx→B1, etc.)
    quantity_type = _FIELD_QUANTITY_MAP.get(canonical)
    if quantity_type is None:
        msg = (
            f"No SI conversion known for {name!r}. "
            f"Known fields: {sorted(_FIELD_QUANTITY_MAP)}"
        )
        raise ValueError(msg)
    # For div_B and div_E, the actual SI factor includes 1/length_ref
    if canonical in ("div_B", "div_E"):
        base = normalization.si_factor(quantity_type)
        return base / normalization.si_factor("length")
    if canonical.startswith("curl_B"):
        base = normalization.si_factor("b_field")
        return base / normalization.si_factor("length")
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
    """Return sorted list of all computable quantity names.

    Includes both canonical names and aliases.

    Returns
    -------
    list[str]
    """
    return sorted(set(_REGISTRY) | set(_COMPUTE_ALIASES))


__all__ = [
    "available_quantities",
    "compute_field",
    "display_unit_factor",
    "field_si_factor",
]
