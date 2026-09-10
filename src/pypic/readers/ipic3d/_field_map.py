"""Field name mapping and Gaussian→SI-rationalized conversion for iPIC3D."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pypic.types import FloatArray

FOUR_PI = 4.0 * math.pi

_FIELD_NAME_MAP: dict[str, str] = {
    "Bx": "B_1",
    "By": "B_2",
    "Bz": "B_3",
    "Ex": "E_1",
    "Ey": "E_2",
    "Ez": "E_3",
}

_MOMENT_COMPONENT_MAP: dict[str, str] = {
    "Jx": "J_1",
    "Jy": "J_2",
    "Jz": "J_3",
    "rho": "rho_c",
}


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
        Canonical per-species name (e.g. ``"J_s0_1"``, ``"rho_c_s2"``).

    Examples
    --------
    >>> per_species_canonical("Jx", 0)
    'J_s0_1'
    >>> per_species_canonical("rho", 2)
    'rho_c_s2'
    """
    canonical = _MOMENT_COMPONENT_MAP[component]
    # Tier-3: vector canonicals end in ``_<component>`` (J_1, J_2, J_3) —
    # insert species before the trailing component. Scalars (rho_c) get
    # the species suffix appended.
    if canonical[-1].isdigit() and "_" in canonical:
        base, _, comp = canonical.rpartition("_")
        return f"{base}_s{species_index}_{comp}"
    return f"{canonical}_s{species_index}"


_H5HUT_FIELD_MAP: dict[str, str] = {
    "divB": "div_B",
}

# H5hut layout: capital-P with lowercase axis letters (Pxx, Pxy, ...).
_PRESSURE_COMPONENT_MAP: dict[str, str] = {
    "Pxx": "P_11",
    "Pxy": "P_12",
    "Pxz": "P_13",
    "Pyy": "P_22",
    "Pyz": "P_23",
    "Pzz": "P_33",
}

# Parallel/serial PHDF5 layout: lowercase-p with uppercase axis letters (pXX, pXY, ...).
_PHDF5_PRESSURE_MAP: dict[str, str] = {
    "pXX": "P_11",
    "pXY": "P_12",
    "pXZ": "P_13",
    "pYY": "P_22",
    "pYZ": "P_23",
    "pZZ": "P_33",
}

_EFLUX_MAP: dict[str, str] = {
    "EFx": "EF_1",
    "EFy": "EF_2",
    "EFz": "EF_3",
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
        Canonical per-species name (e.g. ``"P_s0_11"``, ``"P_s1_12"``).

    Examples
    --------
    >>> per_species_pressure_canonical("Pxx", 0)
    'P_s0_11'
    >>> per_species_pressure_canonical("Pyz", 1)
    'P_s1_23'
    >>> per_species_pressure_canonical("pYY", 1)
    'P_s1_22'
    """
    canonical = _PRESSURE_COMPONENT_MAP.get(component) or _PHDF5_PRESSURE_MAP[component]
    # canonical is "P_<ij>"; Tier-3 per-species form is "P_s<N>_<ij>".
    ij = canonical.removeprefix("P_")
    return f"P_s{species_index}_{ij}"


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
        Canonical per-species name (e.g. ``"EF_s0_1"``, ``"EF_s1_2"``).

    Examples
    --------
    >>> per_species_eflux_canonical("EFx", 0)
    'EF_s0_1'
    >>> per_species_eflux_canonical("EFz", 1)
    'EF_s1_3'
    """
    canonical = _EFLUX_MAP[component]
    # canonical is "EF_<c>"; Tier-3 per-species form is "EF_s<N>_<c>".
    c = canonical.removeprefix("EF_")
    return f"EF_s{species_index}_{c}"


def species_moment_names(
    species_index: int, pressure_map: Mapping[str, str]
) -> dict[str, str]:
    """Canonical → bare native name for one species' deposited moments.

    Covers charge density, current, number density (written by H5hut
    only), the six pressure-tensor components under *pressure_map*'s
    spelling, and the energy flux. Each layout adds its own species
    suffix or file path to the native name.

    Examples
    --------
    >>> names = species_moment_names(1, _PHDF5_PRESSURE_MAP)
    >>> names["J_s1_1"], names["P_s1_23"], names["EF_s1_3"], names["n_s1"]
    ('Jx', 'pYZ', 'EFz', 'N')
    """
    names = {per_species_canonical(c, species_index): c for c in _MOMENT_COMPONENT_MAP}
    names[f"n_s{species_index}"] = "N"
    for native in pressure_map:
        names[per_species_pressure_canonical(native, species_index)] = native
    for native in _EFLUX_MAP:
        names[per_species_eflux_canonical(native, species_index)] = native
    return names


def convert_species_moment(
    native: str,
    data: FloatArray,
    *,
    species_qom: float,
    pressure_map: Mapping[str, str],
) -> FloatArray:
    r"""Bring one raw per-species moment to canonical, SI-rationalized form.

    iPIC3D stores every deposited moment divided by $4\pi$; the
    pressure tensor additionally carries the charge sign and weight
    that `correct_pressure_tensor_component` removes. Number density
    is stored as is.

    Examples
    --------
    >>> import numpy as np
    >>> raw = np.array([1.0 / (4.0 * np.pi)])
    >>> convert_species_moment("Jx", raw, species_qom=-1.0, pressure_map={})
    array([1.])
    >>> convert_species_moment("N", raw, species_qom=-1.0, pressure_map={})
    array([0.07957747])
    """
    if native in pressure_map:
        return correct_pressure_tensor_component(
            data, canonical_base=pressure_map[native], species_qom=species_qom
        )
    match native:
        case "rho":
            return gaussian_density_to_si(data)
        case "N":
            return data
        case "Jx" | "Jy" | "Jz":
            return gaussian_current_to_si(data)
        case _ if native in _EFLUX_MAP:
            return gaussian_pressure_to_si(data)
    msg = f"Not an iPIC3D species moment: {native!r}"
    raise ValueError(msg)


def read_species_moments(
    load: Callable[[str], FloatArray | None],
    species_index: int,
    *,
    species_qom: float,
    expanded: set[str] | None,
    pressure_map: Mapping[str, str],
) -> dict[str, FloatArray]:
    """Load and convert one species' moments through *load*.

    *load* maps a bare native name (``"Jx"``, ``"pXX"``, ``"EFz"``, ...)
    to its raw array, or ``None`` when this layout does not carry it.
    Only names in *expanded* (all, when ``None``) are requested, so a
    selective read never touches unwanted moments.

    Examples
    --------
    >>> import numpy as np
    >>> stored = {"rho": np.array([-1.0 / (4.0 * np.pi)])}
    >>> read_species_moments(
    ...     stored.get, 0, species_qom=-1.0, expanded=None,
    ...     pressure_map=_PHDF5_PRESSURE_MAP,
    ... )
    {'rho_c_s0': array([-1.])}
    """
    out: dict[str, FloatArray] = {}
    for canonical, native in species_moment_names(species_index, pressure_map).items():
        if expanded is not None and canonical not in expanded:
            continue
        raw = load(native)
        if raw is not None:
            out[canonical] = convert_species_moment(
                native, raw, species_qom=species_qom, pressure_map=pressure_map
            )
    return out


def expand_moment_dependencies(
    wanted: set[str],
    nspec: int,
) -> set[str]:
    """Expand wanted field set to include per-species dependencies.

    If a total field (e.g. ``"rho_c"``, ``"J_1"``) is requested, the
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
    >>> sorted(expand_moment_dependencies({"rho_c", "B_1"}, nspec=2))
    ['B_1', 'rho_c', 'rho_c_s0', 'rho_c_s1']
    """
    expanded = set(wanted)
    for moment_comp, canon_total in _MOMENT_COMPONENT_MAP.items():
        if canon_total in expanded:
            for s in range(nspec):
                expanded.add(per_species_canonical(moment_comp, s))
    return expanded


def infer_total_fields(canonical_fields: set[str], nspec: int) -> set[str]:
    r"""Infer which total fields would be computed from per-species sums.

    ``compute_totals_and_filter`` sums per-species J and rho into
    totals (J_1, J_2, J_3, rho_c). This function predicts which totals
    would be produced given a set of per-species canonical names,
    without actually loading or summing arrays.

    Parameters
    ----------
    canonical_fields : set[str]
        Per-species canonical field names already discovered.
    nspec : int
        Number of particle species.

    Returns
    -------
    set[str]
        Total field names (subset of J_1, J_2, J_3, rho_c) whose
        per-species contributions are all present.

    Examples
    --------
    >>> infer_total_fields({"J_s0_1", "J_s1_1", "rho_c_s0"}, nspec=2)
    {'J_1'}
    """
    totals: set[str] = set()
    for native, canon_total in _MOMENT_COMPONENT_MAP.items():
        if all(
            per_species_canonical(native, s) in canonical_fields for s in range(nspec)
        ):
            totals.add(canon_total)
    return totals


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


_CANONICAL_DIAGONAL_PRESSURE: frozenset[str] = frozenset({"P_11", "P_22", "P_33"})


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
        Canonical component name (``"P_11"``, ``"P_12"``, ..., ``"P_33"``)
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
    ...     raw, canonical_base="P_11", species_qom=-1.0
    ... )
    array([1.])
    >>> # Off-diagonal: no sign flip, just Gaussian + mass weighting
    >>> raw = np.array([1.0 / (4.0 * 3.141592653589793)])
    >>> correct_pressure_tensor_component(
    ...     raw, canonical_base="P_12", species_qom=-1.0
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

    Computes total ``rho_c``, ``J_1``, ``J_2``, ``J_3`` by summing the
    per-species contributions already present in *field_data*.  A total
    whose per-species terms are only partly present raises: iPIC3D
    deposits every species, so a gap means truncated output or a wrong
    species count, and a silent partial sum would be wrong physics.

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

    Raises
    ------
    ValueError
        If some but not all species contribute to a requested total.

    Examples
    --------
    >>> import numpy as np
    >>> data = {
    ...     "rho_c_s0": np.array([1.0]),
    ...     "rho_c_s1": np.array([2.0]),
    ...     "J_s0_1": np.array([0.5]),
    ...     "J_s1_1": np.array([0.3]),
    ...     "J_s0_2": np.array([0.1]),
    ...     "J_s1_2": np.array([0.2]),
    ...     "J_s0_3": np.array([0.0]),
    ...     "J_s1_3": np.array([0.4]),
    ... }
    >>> result = compute_totals_and_filter(data, 2, None, None)
    >>> result["rho_c"]
    array([3.])
    >>> result["J_1"]
    array([0.8])
    """
    for moment_comp, canon_total in _MOMENT_COMPONENT_MAP.items():
        if expanded is not None and canon_total not in expanded:
            continue
        keys = [per_species_canonical(moment_comp, s) for s in range(nspec)]
        present = [key for key in keys if key in field_data]
        if not present:
            continue
        if len(present) != nspec:
            missing = [key for key in keys if key not in field_data]
            msg = f"Cannot total {canon_total!r}: missing per-species {missing}"
            raise ValueError(msg)
        # Start from a copy of species 0, then += the rest. Avoids
        # allocating a fresh full-size array per species on large grids.
        total = field_data[keys[0]].copy()
        for key in keys[1:]:
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
