"""BATSRUS field name mapping and unit conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    from pypic.types import FloatArray

FIELD_NAME_MAP: dict[str, str] = {
    # Core MHD
    "Rho": "rho_m",
    "Ux": "V_1",
    "Uy": "V_2",
    "Uz": "V_3",
    "Bx": "B_1",
    "By": "B_2",
    "Bz": "B_3",
    "P": "P",
    "jx": "J_1",
    "jy": "J_2",
    "jz": "J_3",
    "Pe": "Pe",
    "Te": "Te",
    "Ti": "Ti",
    # Electric field
    "Ex": "E_1",
    "Ey": "E_2",
    "Ez": "E_3",
    # Ion pressure (two-fluid)
    "Pi": "Pi",
    # Pressure tensor (MhdAnisoP module)
    "Pxx": "P_11",
    "Pxy": "P_12",
    "Pxz": "P_13",
    "Pyy": "P_22",
    "Pyz": "P_23",
    "Pzz": "P_33",
    # Anisotropic pressure (CGL)
    "Ppar": "P_par",
    "Pperp": "P_perp",
    # Number densities (multi-species)
    "Ne": "n_s0",
    "Ni": "n_s1",
    # Charge density
    "RhoC": "rho_c",
    # B0 splitting: background dipole field (BATSRUS-specific, not in schema.md)
    "b1x": "B_1",
    "b1y": "B_2",
    "b1z": "B_3",
    "b0x": "B0_1",
    "b0y": "B0_2",
    "b0z": "B0_3",
}

SKIP_FIELDS: frozenset[str] = frozenset({"Hyp"})

# Unit conversion factors: BATSRUS practical units → SI base units.
# Keys are the unit strings found in .h headers and .batl NamePlotUnit_V.
_UNIT_FACTORS: dict[str, float] = {
    "Mp/cc": 1e6 * constants.m_p,  # proton masses per cc → kg/m³
    "amu/cc": 1e6 * constants.m_u,  # atomic mass units per cc → kg/m³
    "km/s": 1e3,  # km/s → m/s
    "nT": 1e-9,  # nanoTesla → Tesla
    "nPa": 1e-9,  # nanoPascal → Pascal
    "uA/m2": 1e-6,  # microampere/m² → A/m²
    "mV/m": 1e-3,  # millivolt/m → V/m
    "K": 1.0,  # Kelvin stays as-is
    "R": 1.0,  # reference (coordinate) — no conversion
}


def unit_factor(unit_str: str) -> float:
    """Return the multiplicative factor to convert *unit_str* to SI base units.

    Returns 1.0 for unrecognized or normalized units.
    """
    return _UNIT_FACTORS.get(unit_str, 1.0)


def is_normalized(unit_string: str) -> bool:
    """Check whether the header unit string indicates normalized (code) units."""
    lower = unit_string.lower().strip()
    return "normalized" in lower or lower == ""


def convert_fields_to_si(
    fields: dict[str, FloatArray],
    var_names: tuple[str, ...],
    unit_names: tuple[str, ...],
) -> dict[str, FloatArray]:
    """Convert field arrays from BATSRUS practical units to SI.

    Parameters
    ----------
    fields
        Mapping from *canonical* field name to array.
    var_names
        BATSRUS variable names, same order as *unit_names*.
    unit_names
        Unit string per variable from the header.

    Returns
    -------
    dict[str, FloatArray]
        Fields with SI values (modified in-place for efficiency).
    """
    batsrus_to_unit: dict[str, str] = dict(zip(var_names, unit_names, strict=False))

    for batsrus_name, unit_str in batsrus_to_unit.items():
        canonical = FIELD_NAME_MAP.get(batsrus_name)
        if canonical is None or canonical not in fields:
            continue
        factor = unit_factor(unit_str)
        if factor != 1.0:
            fields[canonical] = np.asarray(fields[canonical] * factor, dtype=np.float64)
    return fields
