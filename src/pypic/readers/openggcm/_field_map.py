"""OpenGGCM field-name mapping and unit conversions.

OpenGGCM native units:
- Velocity: km/s
- Magnetic field: nT
- Number density: cm⁻³
- Pressure: pPa (pico-Pascal, 1e-12 Pa)
- Energy flux: mW/m2 (not imported -- derived E = -(VxB) instead)

Canonical pypic units are SI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy import constants

if TYPE_CHECKING:
    from pypic.types import FloatArray

# OpenGGCM name → canonical name
FIELD_NAME_MAP: dict[str, str] = {
    "vx": "V1",
    "vy": "V2",
    "vz": "V3",
    "bx1": "B1",
    "by1": "B2",
    "bz1": "B3",
    "rr": "rho_m",
    "pp": "P",
}

# Fields to skip by default (energy flux, not standard E-field)
DEFAULT_SKIP: set[str] = {"eflx", "efly", "eflz"}


def velocity_to_si(v_kms: FloatArray) -> FloatArray:
    """Convert velocity from km/s to m/s."""
    return v_kms * 1e3


def bfield_to_si(b_nt: FloatArray) -> FloatArray:
    """Convert magnetic field from nT to T."""
    return b_nt * 1e-9


def density_to_si(n_cc: FloatArray) -> FloatArray:
    r"""Convert number density from cm⁻³ to m⁻³."""
    return n_cc * 1e6


def density_to_mass_density_si(n_cc: FloatArray) -> FloatArray:
    r"""Convert number density (cm⁻³) to mass density (kg/m³).

    Assumes proton mass: $\rho_m = n \cdot m_p$.
    """
    return n_cc * 1e6 * constants.m_p


def pressure_to_si(p_ppa: FloatArray) -> FloatArray:
    """Convert pressure from pPa to Pa."""
    return p_ppa * 1e-12


def convert_fields_to_si(
    raw_fields: dict[str, FloatArray],
) -> dict[str, FloatArray]:
    """Convert OpenGGCM fields from native units to SI and canonical names.

    Also computes ``n_s0`` (ion number density in m⁻³) from the density
    field.

    Parameters
    ----------
    raw_fields : dict[str, FloatArray]
        Fields keyed by OpenGGCM name (``"vx"``, ``"bx1"``, etc.).

    Returns
    -------
    dict[str, FloatArray]
        Fields keyed by canonical name, in SI units.
    """
    si: dict[str, FloatArray] = {}

    for raw_name, canon_name in FIELD_NAME_MAP.items():
        if raw_name not in raw_fields:
            continue

        data = raw_fields[raw_name]
        if raw_name in ("vx", "vy", "vz"):
            si[canon_name] = velocity_to_si(data)
        elif raw_name in ("bx1", "by1", "bz1"):
            si[canon_name] = bfield_to_si(data)
        elif raw_name == "rr":
            si[canon_name] = density_to_mass_density_si(data)
            si["n_s0"] = density_to_si(data)
        elif raw_name == "pp":
            si[canon_name] = pressure_to_si(data)

    # Pass through unknown fields with native names, no conversion
    mapped_names = set(FIELD_NAME_MAP.keys())
    for raw_name, data in raw_fields.items():
        if raw_name not in mapped_names:
            si[raw_name] = data

    return si
