"""Field metadata registry: names, units, and display labels.

Maps every canonical field name to its physical quantity type, SI unit
label, human-readable long name, and LaTeX symbol.  The literal lives in
``pypic._field_table``; this module resolves lookups through the alias
and species-pattern fallbacks, and owns ``register_field``.
``compute.field_si_factor`` reads the table directly at lookup time.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from enum import StrEnum

from pypic._aliases import COMPUTE_ALIASES, OPERATOR_SUFFIXES, _get_field_alias_fallback
from pypic._field_table import _FIELD_INFO, FieldInfo
from pypic.exceptions import UnknownFieldError

log = logging.getLogger(__name__)

_lock = threading.Lock()


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
    FOUR_VELOCITY = "four_velocity"
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
    "four_velocity": "m/s",
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


# openPMD `unitDimension` 7-tuple — powers of the SI base units in the order
# (length, mass, time, current, temperature, amount, luminosity).  Encodes the
# same dimensional content as ``_QUANTITY_UNITS`` but in machine-checkable form.
# Note: pypic stores temperature in *energy units* (J) by convention (see
# ``docs/conventions.md``), so ``"temperature"`` shares its 7-tuple with
# ``"energy_density"`` rather than carrying a Kelvin power.
_QUANTITY_DIMENSIONS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "b_field": (0, 1, -2, -1, 0, 0, 0),  # T
    "e_field": (1, 1, -3, -1, 0, 0, 0),  # V/m
    "velocity": (1, 0, -1, 0, 0, 0, 0),
    "four_velocity": (1, 0, -1, 0, 0, 0, 0),
    "length": (1, 0, 0, 0, 0, 0, 0),
    "time": (0, 0, 1, 0, 0, 0, 0),
    "density": (-3, 0, 0, 0, 0, 0, 0),  # m^-3
    "mass_density": (-3, 1, 0, 0, 0, 0, 0),  # kg/m^3
    "charge_density": (-3, 0, 1, 1, 0, 0, 0),  # C/m^3
    "current_density": (-2, 0, 0, 1, 0, 0, 0),  # A/m^2
    "pressure": (-1, 1, -2, 0, 0, 0, 0),  # Pa
    "temperature": (2, 1, -2, 0, 0, 0, 0),  # J (pypic energy units)
    "energy_density": (-1, 1, -2, 0, 0, 0, 0),  # J/m^3 = Pa
    "frequency": (0, 0, -1, 0, 0, 0, 0),
    "poynting_flux": (0, 1, -3, 0, 0, 0, 0),  # W/m^2
    "energy_flux": (0, 1, -3, 0, 0, 0, 0),  # W/m^2
    "b_field_per_length": (-1, 1, -2, -1, 0, 0, 0),  # T/m
    "e_field_per_length": (0, 1, -3, -1, 0, 0, 0),  # V/m^2
    "velocity_per_length": (0, 0, -1, 0, 0, 0, 0),  # 1/s
    "specific_energy": (2, 0, -2, 0, 0, 0, 0),  # J/kg
    "power_density": (-1, 1, -3, 0, 0, 0, 0),  # W/m^3
    "dimensionless": (0, 0, 0, 0, 0, 0, 0),
}


def quantity_dimension(
    quantity: str | QuantityType,
) -> tuple[int, int, int, int, int, int, int]:
    r"""Return the openPMD ``unitDimension`` 7-tuple for a quantity type.

    The tuple gives integer powers of the SI base units in the order
    *(length, mass, time, current, temperature, amount, luminosity)*.

    Parameters
    ----------
    quantity : str or QuantityType
        Physical quantity type — one of the keys in ``_QUANTITY_UNITS``.

    Returns
    -------
    tuple[int, ...]
        Seven-element tuple of integer dimensional powers.

    Raises
    ------
    KeyError
        If *quantity* is not a recognized quantity type.

    Examples
    --------
    >>> quantity_dimension("b_field")
    (0, 1, -2, -1, 0, 0, 0)
    >>> quantity_dimension(QuantityType.VELOCITY)
    (1, 0, -1, 0, 0, 0, 0)
    >>> quantity_dimension("dimensionless")
    (0, 0, 0, 0, 0, 0, 0)
    """
    key = quantity.value if isinstance(quantity, QuantityType) else quantity
    return _QUANTITY_DIMENSIONS[key]


def register_field(
    name: str,
    quantity_type: QuantityType | str,
    *,
    long_name: str = "",
    si_unit: str | None = None,
    latex: str = "",
    unit_dimension: tuple[int, int, int, int, int, int, int] | None = None,
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
    unit_dimension : tuple[int, ...] | None
        openPMD ``unitDimension`` 7-tuple (powers of length, mass,
        time, current, temperature, amount, luminosity).  ``None``
        (the default) uses ``quantity_dimension(quantity_type)``.
        Provide an explicit tuple only for non-canonical fields whose
        dimension differs from the standard lookup.

    Raises
    ------
    ValueError
        If *quantity_type* is not recognized, *unit_dimension* is not a
        length-7 sequence of ints, or *name* is already registered.
    """
    if quantity_type not in _QUANTITY_UNITS:
        valid = sorted(_QUANTITY_UNITS)
        msg = f"Unknown quantity_type {quantity_type!r}. Valid: {valid}"
        raise ValueError(msg)

    if si_unit is None:
        si_unit = _QUANTITY_UNITS[quantity_type]

    if unit_dimension is not None:
        if len(unit_dimension) != 7 or not all(
            isinstance(p, int) for p in unit_dimension
        ):
            msg = f"unit_dimension must be a 7-tuple of ints, got {unit_dimension!r}"
            raise ValueError(msg)
        unit_dimension = tuple(unit_dimension)  # type: ignore[assignment]

    info = FieldInfo(quantity_type, long_name, si_unit, latex, unit_dimension)

    with _lock:
        if name in _FIELD_INFO:
            msg = f"Field metadata for {name!r} is already registered"
            raise ValueError(msg)
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


# Per-species field metadata from a compact (prefix, qtype, long, latex)
# table, Tier-3 canonical ``<base>_s<N>[_<index>]``.  The prefix anchors the
# regex: a component capture (``J([123])``) yields the vector form
# ``^J_s(\d+)_([123])$``, no capture yields the scalar ``^<prefix>_s(\d+)$``.
# ``P(\d{0,2})`` is the lone tensor entry, whose pattern also matches the
# scalar ``P_s0``.  Templates use ``{C}`` for component, ``{N}`` for species.
_SPECIES_PATTERN_SPECS: list[tuple[str, str, str, str]] = [
    ("n", "density", "Number density (species {N})", r"$n_{{s{N}}}$"),
    ("rho_c", "charge_density", "Charge density (species {N})", r"$\rho_{{c,s{N}}}$"),
    (
        "J([123])",
        "current_density",
        "Current density component {C} (species {N})",
        r"$J_{{{C},s{N}}}$",
    ),
    (
        "V([123])",
        "velocity",
        "Velocity component {C} (species {N})",
        r"$V_{{{C},s{N}}}$",
    ),
    (
        "EF([123])",
        "energy_flux",
        "Energy flux component {C} (species {N})",
        r"$EF_{{{C},s{N}}}$",
    ),
    (
        "KEF([123])",
        "energy_flux",
        "Kinetic energy flux component {C} (species {N})",
        r"$KEF_{{{C},s{N}}}$",
    ),
    (
        "HF([123])",
        "energy_flux",
        "Heat flux component {C} (species {N})",
        r"$HF_{{{C},s{N}}}$",
    ),
    (
        "EHF([123])",
        "energy_flux",
        "Enthalpy flux component {C} (species {N})",
        r"$EHF_{{{C},s{N}}}$",
    ),
    (
        "q([123])",
        "energy_flux",
        "Conductive heat flux component {C} (species {N})",
        r"$q_{{{C},s{N}}}$",
    ),
    # \d{0,2} matches both P_s0_11 (tensor) and P_s0 (scalar)
    (r"P(\d{0,2})", "pressure", "Pressure {C} (species {N})", r"$P_{{{C},s{N}}}$"),
    ("rho_m", "mass_density", "Mass density (species {N})", r"$\rho_{{m,s{N}}}$"),
    (r"\|V\|", "velocity", "Velocity magnitude (species {N})", r"$|V_{{s{N}}}|$"),
    (
        "e_k",
        "energy_density",
        "Kinetic energy density (species {N})",
        r"$e_{{k,s{N}}}$",
    ),
    (
        "e_th_trace",
        "energy_density",
        "Thermal energy density tensor trace (species {N})",
        r"$e_{{th,\mathrm{{tr}},s{N}}}$",
    ),
    (
        "e_th",
        "energy_density",
        "Thermal energy density (species {N})",
        r"$e_{{th,s{N}}}$",
    ),
    (
        "e_int",
        "specific_energy",
        "Specific internal energy (species {N})",
        r"$e_{{int,s{N}}}$",
    ),
    ("h", "specific_energy", "Specific enthalpy (species {N})", r"$h_{{s{N}}}$"),
    ("T", "temperature", "Temperature (species {N})", r"$T_{{s{N}}}$"),
    ("omega_p", "frequency", "Plasma frequency (species {N})", r"$\omega_{{p,s{N}}}$"),
    (
        "omega_c",
        "frequency",
        "Cyclotron frequency (species {N})",
        r"$\omega_{{c,s{N}}}$",
    ),
    ("d", "length", "Skin depth (species {N})", r"$d_{{s{N}}}$"),
    ("r", "length", "Thermal gyroradius (species {N})", r"$r_{{s{N}}}$"),
    ("lambda_D", "length", "Debye length (species {N})", r"$\lambda_{{D,s{N}}}$"),
    ("v_th", "velocity", "Thermal speed (species {N})", r"$v_{{th,s{N}}}$"),
    ("beta", "dimensionless", "Plasma beta (species {N})", r"$\beta_{{s{N}}}$"),
    ("s", "dimensionless", "Entropy (species {N})", r"$s_{{s{N}}}$"),
    (
        "s_gyro",
        "dimensionless",
        "Gyrotropic entropy (species {N})",
        r"$s_{{gyro,s{N}}}$",
    ),
    ("P_par", "pressure", "Parallel pressure (species {N})", r"$P_{{\parallel,s{N}}}$"),
    (
        "P_perp",
        "pressure",
        "Perpendicular pressure (species {N})",
        r"$P_{{\perp,s{N}}}$",
    ),
    ("agyrotropy", "dimensionless", "Agyrotropy (species {N})", r"$Q_{{s{N}}}$"),
    (
        "D_ng",
        "dimensionless",
        "Aunai nongyrotropy (species {N})",
        r"$D_{{ng,s{N}}}$",
    ),
    (
        "A_phi",
        "dimensionless",
        "Scudder agyrotropy (species {N})",
        r"$A_{{\phi,s{N}}}$",
    ),
    (
        "R_recon",
        "dimensionless",
        "Local reconnection rate (species {N})",
        r"$R_{{\mathrm{{recon}},s{N}}}$",
    ),
    # Per-species field-aligned velocity decomposition.
    (
        "V_par",
        "velocity",
        "Parallel bulk velocity (species {N})",
        r"$V_{{\parallel,s{N}}}$",
    ),
    (
        "V_perp([123])",
        "velocity",
        "Perpendicular bulk velocity component {C} (species {N})",
        r"$V_{{\perp,{C},s{N}}}$",
    ),
    (
        r"\|V_perp\|",
        "velocity",
        "Perpendicular bulk velocity magnitude (species {N})",
        r"$|V_{{\perp,s{N}}}|$",
    ),
]


def _build_species_info_pattern(prefix: str) -> re.Pattern[str]:
    r"""Translate a Tier-3 prefix entry into its full species-name regex.

    Six flavors:
      - Vector with explicit ``([123])`` capture (e.g. ``J([123])``)
        → ``^J_s(\d+)_([123])$``.  When the base before the capture
        ends with a generic operator (``V_perp([123])``), the species
        qualifier slots in between the root and the operator:
        ``^V_s(\d+)_perp_([123])$``.
      - Tensor variant ``P(\d{0,2})`` → ``^P_s(\d+)(?:_(\d{2}))?$``
        (matches both scalar ``P_s0`` and tensor ``P_s0_11``)
      - Pipe-wrapped magnitude (``\|V\|`` → ``^\|V_s(\d+)\|$``):
        species sits inside the bars, bracketing the operand.  When the
        operand contains a generic operator (``\|V_perp\|``), the
        species again slots between the root and the operator:
        ``^\|V_s(\d+)_perp\|$``.
      - Generic-operator suffix (``P_par`` → ``^P_s(\d+)_par$``):
        species in middle, operator at end.
      - Plain scalar (``n``, ``T``, ``omega_p``, ``rho_m``, ``s_gyro``):
        ``^<prefix>_s(\d+)$`` — species at end.
    """
    if "([123])" in prefix:
        base = prefix.replace("([123])", "")
        # Vector with generic operator: ``V_perp([123])`` →
        # ``^V_s(\d+)_perp_([123])$``.
        if "_" in base:
            root, _, op = base.rstrip("_").rpartition("_")
            if op in OPERATOR_SUFFIXES:
                return re.compile(rf"^{root}_s(\d+)_{op}_([123])$")
        return re.compile(rf"^{base}_s(\d+)_([123])$")
    if r"P(\d{0,2})" in prefix:
        return re.compile(r"^P_s(\d+)(?:_(\d{2}))?$")
    if prefix.startswith(r"\|") and prefix.endswith(r"\|"):
        inner = prefix[2:-2]
        # Pipe-wrapped magnitude with generic operator:
        # ``\|V_perp\|`` → ``^\|V_s(\d+)_perp\|$``.
        if "_" in inner:
            root, _, op = inner.rpartition("_")
            if op in OPERATOR_SUFFIXES:
                return re.compile(rf"^\|{root}_s(\d+)_{op}\|$")
        return re.compile(rf"^\|{inner}_s(\d+)\|$")
    if "_" in prefix:
        root, _, suffix = prefix.rpartition("_")
        if suffix in OPERATOR_SUFFIXES:
            return re.compile(rf"^{root}_s(\d+)_{suffix}$")
    return re.compile(rf"^{prefix}_s(\d+)$")


_SPECIES_INFO_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    (_build_species_info_pattern(prefix), qtype, long_tmpl, latex_tmpl)
    for prefix, qtype, long_tmpl, latex_tmpl in _SPECIES_PATTERN_SPECS
]


_SPECIES_QUANTITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (pat, qtype) for pat, qtype, _, _ in _SPECIES_INFO_PATTERNS
]


def _try_species_info(name: str) -> FieldInfo | None:
    """Try to build FieldInfo from species regex patterns.

    Tier-3 patterns capture (species, component) in that order — the
    species qualifier sits between the field name and the component
    suffix (``V_s0_1`` → species=0, component=1).
    """
    for pattern, qtype, name_tmpl, latex_tmpl in _SPECIES_INFO_PATTERNS:
        m = pattern.match(name)
        if m is not None:
            groups = m.groups()
            if len(groups) == 2:
                species_idx, component = groups
                if not component:
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
    3. Field alias fallback (``"Bx"`` -> ``"B_1"``)
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
    >>> field_info("B_1", axis_names=("x", "y", "z")).long_name
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
    canonical = COMPUTE_ALIASES.get(name)
    if canonical is not None:
        info = _FIELD_INFO.get(canonical)
        if info is not None:
            return _maybe_localize(info)

    # 3. Field alias fallback (Bx -> B_1, P_e -> Pe, etc.)
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
    raise UnknownFieldError(msg)


_COMPONENT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<component>[123])$")


def vector_component(name: str) -> tuple[str, int] | None:
    """Split a registered vector component name into ``(base, component)``.

    The Tier-3 form ``<base>_<component>`` with a registered metadata
    entry marks a vector component: ``B_1``, ``KEF_s0_2``, ``E_prime_3``.
    Scalars, tensor components (``P_s0_11``) and unregistered names
    return ``None``.

    Parameters
    ----------
    name : str
        Field or derived quantity name.

    Returns
    -------
    tuple[str, int] | None
        Vector base name and 1-based component, or ``None``.

    Examples
    --------
    >>> vector_component("B_1")
    ('B', 1)
    >>> vector_component("KEF_s0_2")
    ('KEF_s0', 2)
    >>> vector_component("P_s0_11") is None
    True
    >>> vector_component("rho_c") is None
    True
    """
    m = _COMPONENT_SUFFIX_RE.match(name)
    if m is None:
        return None
    try:
        field_info(name)
    except UnknownFieldError:
        return None
    return m.group("base"), int(m.group("component"))


def unit_label(name: str, *, to_si: bool = False) -> str:
    r"""Return a unit label string for a field, suitable for plot axes.

    Parameters
    ----------
    name : str
        Field or derived quantity name.
    to_si : bool
        If ``True``, return the SI unit label (e.g. ``"T"``).
        If ``False``, return ``"normalized"`` or ``""`` for dimensionless.

    Returns
    -------
    str

    Examples
    --------
    >>> unit_label("B_1", to_si=True)
    'T'
    >>> unit_label("B_1")
    'normalized'
    >>> unit_label("beta", to_si=True)
    ''
    >>> unit_label("beta")
    ''
    """
    info = field_info(name)
    if to_si:
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
    "quantity_dimension",
    "quantity_units",
    "register_field",
    "unit_label",
    "unregister_field",
    "vector_component",
]
