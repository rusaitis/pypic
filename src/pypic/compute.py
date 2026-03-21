"""String-based dispatch for derived quantities on FieldDataset.

Maps short names (``"|B|"``, ``"beta"``, ``"v_A"``, ...) to pure functions
in ``derived.py``, ``diagnostics.py``, and ``operators.py``. Never imports
FieldDataset — avoids circular dependencies. FieldDataset methods call into
this module via deferred imports.
"""

from __future__ import annotations

import difflib
import functools
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from scipy import constants

from pypic import derived, diagnostics
from pypic.coordinates import operators
from pypic.coordinates.geometry import GeometryType
from pypic.fields import _FIELD_INFO

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.readers.base import FieldDataset
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


# Regex patterns for SI conversion of per-species fields
_SI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^n_s\d+$"), "density"),
    (re.compile(r"^rho_c_s\d+$"), "charge_density"),
    (re.compile(r"^J[123]_s\d+$"), "current_density"),
    (re.compile(r"^V[123e]?_s\d+$"), "velocity"),
    (re.compile(r"^Ve[123]_s\d+$"), "velocity"),
    (re.compile(r"^EF[123]_s\d+$"), "poynting_flux"),
    (re.compile(r"^P\d{0,2}_s\d+$"), "pressure"),
    (re.compile(r"^T_s\d+$"), "temperature"),
    (re.compile(r"^omega_[pc]_s\d+$"), "frequency"),
    (re.compile(r"^[dr]_s\d+$"), "length"),
    (re.compile(r"^lambda_D_s\d+$"), "length"),
    (re.compile(r"^v_th_s\d+$"), "velocity"),
    (re.compile(r"^(?:beta|s|s_gyro|agyrotropy)_s\d+$"), "dimensionless"),
]


# Maps field/derived names to physical quantity types for SI conversion.
# Derived from the canonical registry in fields.py — single source of truth.
_FIELD_QUANTITY_MAP: dict[str, str] = {
    name: info.quantity_type for name, info in _FIELD_INFO.items()
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
    if species_args:
        if recipe.species_args is not None:
            # Dynamic recipe: use explicit species_args descriptor
            _append_species_params(args, species_args, recipe.species_args)
        else:
            # Static recipe: dispatch by function identity
            func = recipe.func
            if func is derived.thermal_speed or func is derived.ion_acoustic_speed:
                args.append(species_args[1])
            elif (
                func is derived.gyrofrequency
                or func is derived.plasma_frequency
                or func is derived.skin_depth
                or func is derived.gyroradius
            ):
                args.extend(species_args)
            elif func is derived.debye_length:
                args.append(species_args[0])

    # Append gamma
    if recipe.needs_gamma:
        args.append(_get_gamma(dataset))

    # Append c
    if recipe.needs_c:
        args.append(_get_c(dataset))

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

    result = recipe.func(*args)

    if recipe.component is not None:
        return result[recipe.component]  # type: ignore[no-any-return]
    return result  # type: ignore[no-any-return]


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
    quantity_type = _FIELD_QUANTITY_MAP.get(canonical)
    if quantity_type is None:
        # Resolve field aliases (Bx→B1, B_x→B1, P_e→Pe, etc.)
        fallback = _get_field_alias_fallback()
        canonical = fallback.get(canonical, canonical)
        quantity_type = _FIELD_QUANTITY_MAP.get(canonical)
    if quantity_type is None:
        # Try regex patterns for per-species fields (n_s2, J1_s3, etc.)
        for pattern, qtype in _SI_PATTERNS:
            if pattern.match(canonical):
                quantity_type = qtype
                break
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
