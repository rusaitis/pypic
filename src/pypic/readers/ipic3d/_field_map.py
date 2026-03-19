"""Field name mapping and Gaussian→SI-rationalized conversion for iPIC3D."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypic.types import FloatArray

FOUR_PI = 4.0 * math.pi

_FIELD_NAME_MAP: dict[str, str] = {
    "Bx": "B1",
    "By": "B2",
    "Bz": "B3",
    "Ex": "E1",
    "Ey": "E2",
    "Ez": "E3",
}

_MOMENT_COMPONENT_MAP: dict[str, str] = {
    "Jx": "J1",
    "Jy": "J2",
    "Jz": "J3",
    "rho": "rho_c",
}


def map_field_name(ipic3d_name: str) -> str:
    """Map an iPIC3D field name to its canonical equivalent.

    Parameters
    ----------
    ipic3d_name : str
        Native iPIC3D field name (e.g. ``"Bx"``).

    Returns
    -------
    str
        Canonical field name (e.g. ``"B1"``).

    Raises
    ------
    KeyError
        If the name has no known mapping.

    Examples
    --------
    >>> map_field_name("Bx")
    'B1'
    >>> map_field_name("Ez")
    'E3'
    """
    try:
        return _FIELD_NAME_MAP[ipic3d_name]
    except KeyError:
        msg = f"Unknown iPIC3D field name: {ipic3d_name!r}"
        raise KeyError(msg) from None


def per_species_canonical(component: str, species_index: int) -> str:
    """Build canonical per-species field name.

    Parameters
    ----------
    component : str
        iPIC3D moment component (``"Jx"``, ``"Jy"``, ``"Jz"``, ``"rho"``).
    species_index : int
        Zero-based species index.

    Returns
    -------
    str
        Canonical per-species name (e.g. ``"J1_s0"``, ``"rho_c_s2"``).

    Examples
    --------
    >>> per_species_canonical("Jx", 0)
    'J1_s0'
    >>> per_species_canonical("rho", 2)
    'rho_c_s2'
    """
    canonical = _MOMENT_COMPONENT_MAP[component]
    return f"{canonical}_s{species_index}"


_H5HUT_FIELD_MAP: dict[str, str] = {
    "Vfx": "V1",
    "Vfy": "V2",
    "Vfz": "V3",
    "divB": "div_B",
}

_PRESSURE_COMPONENT_MAP: dict[str, str] = {
    "Pxx": "P11",
    "Pxy": "P12",
    "Pxz": "P13",
    "Pyy": "P22",
    "Pyz": "P23",
    "Pzz": "P33",
}

_PHDF5_PRESSURE_MAP: dict[str, str] = {
    "pXX": "P11",
    "pXY": "P12",
    "pXZ": "P13",
    "pYY": "P22",
    "pYZ": "P23",
    "pZZ": "P33",
}

_PHDF5_DIAGONAL_PRESSURE = {"pXX", "pYY", "pZZ"}


def per_species_pressure_canonical(component: str, species_index: int) -> str:
    """Build canonical per-species pressure tensor field name.

    Parameters
    ----------
    component : str
        iPIC3D pressure component (``"Pxx"``, ``"Pxy"``, etc.).
    species_index : int
        Zero-based species index.

    Returns
    -------
    str
        Canonical per-species name (e.g. ``"P11_s0"``, ``"P12_s1"``).

    Examples
    --------
    >>> per_species_pressure_canonical("Pxx", 0)
    'P11_s0'
    >>> per_species_pressure_canonical("Pyz", 1)
    'P23_s1'
    """
    canonical = _PRESSURE_COMPONENT_MAP[component]
    return f"{canonical}_s{species_index}"


def expand_moment_dependencies(
    wanted: set[str],
    nspec: int,
) -> set[str]:
    """Expand wanted field set to include per-species dependencies.

    If a total field (e.g. ``"rho_c"``, ``"J1"``) is requested, the
    per-species components needed to compute it are added.

    Parameters
    ----------
    wanted : set[str]
        Canonical field names requested by the caller.
    nspec : int
        Number of particle species.

    Returns
    -------
    set[str]
        Expanded set including all per-species dependencies.

    Examples
    --------
    >>> sorted(expand_moment_dependencies({"rho_c", "B1"}, nspec=2))
    ['B1', 'rho_c', 'rho_c_s0', 'rho_c_s1']
    """
    expanded = set(wanted)
    for moment_comp, canon_total in _MOMENT_COMPONENT_MAP.items():
        if canon_total in expanded:
            for s in range(nspec):
                expanded.add(per_species_canonical(moment_comp, s))
    return expanded


def gaussian_pressure_to_si(p: FloatArray) -> FloatArray:
    r"""Convert iPIC3D Gaussian pressure tensor to SI-rationalized.

    iPIC3D stores $P_{stored} = P / (4\pi)$. Multiply by $4\pi$
    to recover the SI-rationalized value.

    Parameters
    ----------
    p : FloatArray
        Pressure tensor component in iPIC3D Gaussian normalization.

    Returns
    -------
    FloatArray
        Pressure in SI-rationalized normalization.

    Examples
    --------
    >>> import numpy as np
    >>> gaussian_pressure_to_si(np.array([1.0 / (4.0 * 3.141592653589793)]))
    array([1.])
    """
    return p * FOUR_PI


def gaussian_density_to_si(rho: FloatArray) -> FloatArray:
    r"""Convert iPIC3D Gaussian charge density to SI-rationalized.

    iPIC3D stores $\rho_{stored} = \rho / (4\pi)$. Multiply by $4\pi$
    to recover the SI-rationalized value where $\nabla \cdot E = \rho_c$.

    Parameters
    ----------
    rho : FloatArray
        Charge density in iPIC3D Gaussian normalization.

    Returns
    -------
    FloatArray
        Charge density in SI-rationalized normalization.

    Examples
    --------
    >>> import numpy as np
    >>> gaussian_density_to_si(np.array([1.0 / (4.0 * 3.141592653589793)]))
    array([1.])
    """
    return rho * FOUR_PI


def gaussian_current_to_si(j: FloatArray) -> FloatArray:
    r"""Convert iPIC3D Gaussian current density to SI-rationalized.

    iPIC3D stores $J_{stored} = J / (4\pi)$. Multiply by $4\pi$
    to recover the SI-rationalized value.

    Parameters
    ----------
    j : FloatArray
        Current density in iPIC3D Gaussian normalization.

    Returns
    -------
    FloatArray
        Current density in SI-rationalized normalization.

    Examples
    --------
    >>> import numpy as np
    >>> gaussian_current_to_si(np.array([1.0 / (4.0 * 3.141592653589793)]))
    array([1.])
    """
    return j * FOUR_PI
