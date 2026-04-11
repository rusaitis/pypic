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

_H5HUT_DIAGONAL_PRESSURE = {"Pxx", "Pyy", "Pzz"}

_EFLUX_MAP: dict[str, str] = {
    "EFx": "EF1",
    "EFy": "EF2",
    "EFz": "EF3",
}


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


def per_species_eflux_canonical(component: str, species_index: int) -> str:
    """Build canonical per-species energy flux field name.

    Parameters
    ----------
    component : str
        iPIC3D energy flux component (``"EFx"``, ``"EFy"``, ``"EFz"``).
    species_index : int
        Zero-based species index.

    Returns
    -------
    str
        Canonical per-species name (e.g. ``"EF1_s0"``, ``"EF2_s1"``).

    Examples
    --------
    >>> per_species_eflux_canonical("EFx", 0)
    'EF1_s0'
    >>> per_species_eflux_canonical("EFz", 1)
    'EF3_s1'
    """
    canonical = _EFLUX_MAP[component]
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


_CANONICAL_DIAGONAL_PRESSURE: frozenset[str] = frozenset({"P11", "P22", "P33"})


def correct_pressure_tensor_component(
    data: FloatArray,
    *,
    canonical_base: str,
    species_qom: float,
) -> FloatArray:
    r"""Convert an iPIC3D pressure tensor component to canonical form.

    Applies the three corrections every iPIC3D reader variant
    (phdf5, shdf5, H5hut) must apply to a raw pressure tensor
    component, in order:

    1. **Sign flip on diagonal components when** $q/m < 0$. iPIC3D
       deposits $\rho T$ where $\rho$ inherits the charge sign, so
       the stored diagonal is negative for electrons. Off-diagonals
       and positive-charge species are unaffected.
    2. **Gaussian→SI-rationalized conversion.** iPIC3D writes
       $P_{stored} = P / (4\pi)$; multiply by $4\pi$.
    3. **Charge-weighted → mass-weighted.** iPIC3D deposits
       $q\,n\,v\,v$ so dividing by $|q/m| = |qom|$ yields the physical
       $m\,n\,v\,v$.

    Parameters
    ----------
    data : FloatArray
        Raw pressure tensor component as stored in the iPIC3D file.
    canonical_base : str
        Canonical component name (``"P11"``, ``"P12"``, ..., ``"P33"``)
        — the per-species suffix is irrelevant for this correction.
    species_qom : float
        Charge-to-mass ratio of the species, in code units.

    Returns
    -------
    FloatArray
        Corrected pressure tensor component, ready to be stored under
        its canonical name.

    Examples
    --------
    >>> import numpy as np
    >>> # Electron diagonal: stored negative, positive after correction
    >>> raw = np.array([-1.0 / (4.0 * 3.141592653589793)])
    >>> correct_pressure_tensor_component(
    ...     raw, canonical_base="P11", species_qom=-1.0
    ... )
    array([1.])
    >>> # Off-diagonal: no sign flip, just Gaussian + mass weighting
    >>> raw = np.array([1.0 / (4.0 * 3.141592653589793)])
    >>> correct_pressure_tensor_component(
    ...     raw, canonical_base="P12", species_qom=-1.0
    ... )
    array([1.])
    """
    if canonical_base in _CANONICAL_DIAGONAL_PRESSURE and species_qom < 0:
        data = -data
    return gaussian_pressure_to_si(data) / abs(species_qom)


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


def compute_totals_and_filter(
    field_data: dict[str, FloatArray],
    nspec: int,
    expanded: set[str] | None,
    wanted: set[str] | None,
) -> dict[str, FloatArray]:
    """Sum per-species moments into totals and filter to wanted fields.

    Computes total ``rho_c``, ``J1``, ``J2``, ``J3`` by summing the
    per-species contributions already present in *field_data*.  Guards
    against missing per-species keys (e.g. when a species lacks data).

    Parameters
    ----------
    field_data : dict[str, FloatArray]
        Field arrays keyed by canonical name.  Modified in place
        (totals are added) before filtering.
    nspec : int
        Number of particle species.
    expanded : set[str] | None
        Expanded wanted set (includes per-species dependencies),
        or ``None`` when all fields are wanted.
    wanted : set[str] | None
        Originally requested canonical field names, or ``None``
        for all fields.

    Returns
    -------
    dict[str, FloatArray]
        Filtered field dict containing only *wanted* keys (or all
        keys when *wanted* is ``None``).

    Examples
    --------
    >>> import numpy as np
    >>> data = {
    ...     "rho_c_s0": np.array([1.0]),
    ...     "rho_c_s1": np.array([2.0]),
    ...     "J1_s0": np.array([0.5]),
    ...     "J1_s1": np.array([0.3]),
    ...     "J2_s0": np.array([0.1]),
    ...     "J2_s1": np.array([0.2]),
    ...     "J3_s0": np.array([0.0]),
    ...     "J3_s1": np.array([0.4]),
    ... }
    >>> result = compute_totals_and_filter(data, 2, None, None)
    >>> result["rho_c"]
    array([3.])
    >>> result["J1"]
    array([0.8])
    """
    for moment_comp, canon_total in _MOMENT_COMPONENT_MAP.items():
        if expanded is not None and canon_total not in expanded:
            continue
        first_key = per_species_canonical(moment_comp, 0)
        if first_key not in field_data:
            continue
        # Start from a copy of species 0, then += the rest. Avoids
        # allocating a fresh full-size array per species on large grids.
        total = field_data[first_key].copy()
        for s in range(1, nspec):
            key = per_species_canonical(moment_comp, s)
            if key in field_data:
                total += field_data[key]
        field_data[canon_total] = total

    if wanted is not None:
        return {k: v for k, v in field_data.items() if k in wanted}
    return field_data


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
