"""Parser for iPIC3D ConservedQuantities diagnostic output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pypic.containers import TabularData

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import FloatArray


# Column positions in the three layouts this module parses. The two
# ConservedQuantities formats (described on the class below) agree only on
# the cycle column, so every constant carries its format in the prefix.
_A_CYCLE = 0
_A_ELECTRIC = 1
_A_MAGNETIC = 5
_A_KINETIC = 9
_A_TOTAL = 10
_A_MOMENTUM = 12

_B_CYCLE = 0
_B_TOTAL = 1
_B_ELECTRIC = 3
_B_KINETIC = 5
_B_MOMENTUM = 6
_B_MAGNETIC = 7
# Per-species triples (npart, charge, KE) fill the tail. The Format B header
# numbers its columns from 1; the parsed array indexes from 0.
_B_SPECIES_HEADER_COL = 12
_B_SPECIES_BASE = _B_SPECIES_HEADER_COL - 1
_B_SPECIES_STRIDE = 3

# SpeciesQuantities.txt: one row per species per cycle.
_SQ_CYCLE = 0
_SQ_SPECIES = 1
_SQ_MOMENTUM = 2
_SQ_TOTAL_KE = 3
_SQ_BULK_KE = 4
_SQ_THERMAL_KE = 5


@dataclass(frozen=True, slots=True)
class ConservedQuantities:
    """Time series of conserved quantities from an iPIC3D run.

    Two output formats exist:

    **Format A (Roman numeral header)** — single file from phdf5/shdf5 runs.
    Columns: cycle, electric energy (total, x, y, z), magnetic energy
    (total, x, y, z), kinetic energy, total energy, energy variation,
    momentum.

    **Format B (comment header)** — per-restart-segment files from H5hut
    runs. Columns: cycle, total energy, energy variation, electric energy,
    local B energy, kinetic energy, momentum, total B energy, internal B
    energy, KE removed, E removed, then per-species (npart, charge, KE).

    Parameters
    ----------
    cycle : FloatArray
        Cycle numbers (int-valued but stored as float for array uniformity).
    total_energy : FloatArray
        Total energy at each cycle.
    electric_energy : FloatArray
        Total electric field energy.
    magnetic_energy : FloatArray
        Total magnetic field energy.
    kinetic_energy : FloatArray
        Total kinetic energy (all species).
    momentum : FloatArray
        Total momentum magnitude.
    species_npart : tuple[FloatArray, ...]
        Number of particles per species at each cycle.
    species_charge : tuple[FloatArray, ...]
        Total charge per species.
    species_kinetic_energy : tuple[FloatArray, ...]
        Kinetic energy per species.
    """

    cycle: FloatArray
    total_energy: FloatArray
    electric_energy: FloatArray
    magnetic_energy: FloatArray
    kinetic_energy: FloatArray
    momentum: FloatArray
    species_npart: tuple[FloatArray, ...]
    species_charge: tuple[FloatArray, ...]
    species_kinetic_energy: tuple[FloatArray, ...]


def _is_roman_header(text: str) -> bool:
    """Detect Format A (Roman numeral header like 'I.    Cycle')."""
    for line in text.split("\n", 20):
        stripped = line.strip()
        if stripped.startswith("I."):
            return True
        if stripped.startswith("#"):
            return False
    return False


def _parse_single(path: Path) -> ConservedQuantities:
    """Parse Format A: single ConservedQuantities file (Roman numeral header).

    Columns: cycle, E_total, Ex, Ey, Ez, B_total, Bx, By, Bz,
    KE_total, total_energy, energy_variation, momentum.
    """
    # Skip the Roman numeral header, separator, and column-name lines.
    # Data lines start with whitespace followed by a digit.
    data_re = re.compile(r"^\s+\d")
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            if data_re.match(line):
                rows.append([float(x) for x in line.split()])
    data = np.array(rows, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return ConservedQuantities(
        cycle=data[:, _A_CYCLE],
        total_energy=data[:, _A_TOTAL],
        electric_energy=data[:, _A_ELECTRIC],
        magnetic_energy=data[:, _A_MAGNETIC],
        kinetic_energy=data[:, _A_KINETIC],
        momentum=data[:, _A_MOMENTUM],
        species_npart=(),
        species_charge=(),
        species_kinetic_energy=(),
    )


def _parse_multi_file(path: Path) -> tuple[FloatArray, int]:
    """Parse a single Format B file, returning (data_array, nspec).

    Drops truncated lines (fewer columns than expected) that occur at
    restart boundaries in large runs.
    """
    nspec = 0
    expected_cols = 0
    header_re = re.compile(r"#\((\d+)-> ")
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            m = header_re.match(line)
            if m:
                col = int(m.group(1))
                expected_cols = max(expected_cols, col)
                if col >= _B_SPECIES_HEADER_COL:
                    species_idx = (col - _B_SPECIES_HEADER_COL) // _B_SPECIES_STRIDE
                    nspec = max(nspec, species_idx + 1)
                continue
            if line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue
            if expected_cols > 0 and len(values) < expected_cols:
                continue  # truncated line
            rows.append(values)

    data = np.array(rows, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, nspec


def _parse_multi(directory: Path) -> ConservedQuantities:
    """Parse Format B: per-restart ConservedQuantities files.

    Reads all ``ConservedQuantities*.txt`` in *directory*, concatenates,
    sorts by cycle, and deduplicates (keeping the entry from the later
    file for overlapping cycles).

    Columns: cycle(1), total_energy(2), energy_var(3), E_energy(4),
    B_local(5), KE(6), momentum(7), B_total(8), B_internal(9),
    KE_removed(10), E_removed(11), then per-species: npart, charge, KE.
    """
    files = sorted(directory.glob("ConservedQuantities*.txt"))
    if not files:
        msg = f"No ConservedQuantities files found in {directory}"
        raise FileNotFoundError(msg)

    all_data: list[FloatArray] = []
    # Track file index for deduplication priority
    file_indices: list[FloatArray] = []
    nspec = 0
    for i, f in enumerate(files):
        data, ns = _parse_multi_file(f)
        nspec = max(nspec, ns)
        all_data.append(data)
        file_indices.append(np.full(len(data), i, dtype=np.float64))

    combined = np.concatenate(all_data, axis=0)
    priorities = np.concatenate(file_indices)

    # Sort by cycle, then by file index (later file wins ties)
    sort_order = np.lexsort((priorities, combined[:, _B_CYCLE]))
    combined = combined[sort_order]

    # Deduplicate: keep last occurrence of each cycle
    cycles = combined[:, _B_CYCLE]
    _, unique_idx = np.unique(cycles[::-1], return_index=True)
    unique_idx = len(cycles) - 1 - unique_idx
    unique_idx.sort()
    combined = combined[unique_idx]

    species_npart: list[FloatArray] = []
    species_charge: list[FloatArray] = []
    species_ke: list[FloatArray] = []
    for s in range(nspec):
        base_col = _B_SPECIES_BASE + s * _B_SPECIES_STRIDE
        species_npart.append(combined[:, base_col])
        species_charge.append(combined[:, base_col + 1])
        species_ke.append(combined[:, base_col + 2])

    return ConservedQuantities(
        cycle=combined[:, _B_CYCLE],
        total_energy=combined[:, _B_TOTAL],
        electric_energy=combined[:, _B_ELECTRIC],
        magnetic_energy=combined[:, _B_MAGNETIC],
        kinetic_energy=combined[:, _B_KINETIC],
        momentum=combined[:, _B_MOMENTUM],
        species_npart=tuple(species_npart),
        species_charge=tuple(species_charge),
        species_kinetic_energy=tuple(species_ke),
    )


def load_conserved_quantities(path: Path) -> ConservedQuantities:
    """Load conserved quantities from an iPIC3D run, auto-detecting format.

    Parameters
    ----------
    path : Path
        Either a single ``ConservedQuantities.txt`` file (Format A) or
        a directory containing ``ConservedQuantities*.txt`` files (Format B).

    Returns
    -------
    ConservedQuantities
        Parsed time series.

    Raises
    ------
    FileNotFoundError
        If no conserved quantities data is found.
    """
    if path.is_dir():
        return _parse_multi(path)

    text = path.read_text()
    if _is_roman_header(text):
        return _parse_single(path)

    # Single Format B file
    data, nspec = _parse_multi_file(path)
    species_npart: list[FloatArray] = []
    species_charge: list[FloatArray] = []
    species_ke: list[FloatArray] = []
    for s in range(nspec):
        base_col = _B_SPECIES_BASE + s * _B_SPECIES_STRIDE
        species_npart.append(data[:, base_col])
        species_charge.append(data[:, base_col + 1])
        species_ke.append(data[:, base_col + 2])

    return ConservedQuantities(
        cycle=data[:, _B_CYCLE],
        total_energy=data[:, _B_TOTAL],
        electric_energy=data[:, _B_ELECTRIC],
        magnetic_energy=data[:, _B_MAGNETIC],
        kinetic_energy=data[:, _B_KINETIC],
        momentum=data[:, _B_MOMENTUM],
        species_npart=tuple(species_npart),
        species_charge=tuple(species_charge),
        species_kinetic_energy=tuple(species_ke),
    )


def conserved_to_tabular(cq: ConservedQuantities) -> TabularData:
    """Convert a ``ConservedQuantities`` to a generic ``TabularData``.

    Scalar fields map directly.  Per-species tuples are flattened to
    ``"npart_s0"``, ``"charge_s0"``, ``"kinetic_energy_s0"``, etc.

    Parameters
    ----------
    cq : ConservedQuantities
        Typed iPIC3D conserved quantities.

    Returns
    -------
    TabularData
        Columnar representation with ``index_column="cycle"``.

    Examples
    --------
    >>> import numpy as np
    >>> cq = ConservedQuantities(
    ...     cycle=np.array([0.0, 1.0]),
    ...     total_energy=np.array([5.0, 5.1]),
    ...     electric_energy=np.array([1.0, 1.1]),
    ...     magnetic_energy=np.array([2.0, 2.0]),
    ...     kinetic_energy=np.array([2.0, 2.0]),
    ...     momentum=np.array([0.1, 0.1]),
    ...     species_npart=(np.array([100.0, 100.0]),),
    ...     species_charge=(np.array([1.0, 1.0]),),
    ...     species_kinetic_energy=(np.array([1.0, 1.0]),),
    ... )
    >>> tab = conserved_to_tabular(cq)
    >>> "npart_s0" in tab
    True
    >>> tab.index_column
    'cycle'
    """
    columns: dict[str, FloatArray] = {
        "cycle": cq.cycle,
        "total_energy": cq.total_energy,
        "electric_energy": cq.electric_energy,
        "magnetic_energy": cq.magnetic_energy,
        "kinetic_energy": cq.kinetic_energy,
        "momentum": cq.momentum,
    }
    for s, arr in enumerate(cq.species_npart):
        columns[f"npart_s{s}"] = arr
    for s, arr in enumerate(cq.species_charge):
        columns[f"charge_s{s}"] = arr
    for s, arr in enumerate(cq.species_kinetic_energy):
        columns[f"kinetic_energy_s{s}"] = arr

    return TabularData(
        name="conserved_quantities",
        columns=columns,
        index_column="cycle",
        metadata={"source": "iPIC3D ConservedQuantities"},
    )


def load_species_quantities(path: Path) -> TabularData:
    """Parse iPIC3D ``SpeciesQuantities.txt`` into a ``TabularData``.

    Format: one row per species per cycle. Columns:
    ``cycle, species, momentum, total_ke, bulk_ke, thermal_ke``.

    The output pivots per-species data into separate columns:
    ``cycle, momentum_s0, total_ke_s0, bulk_ke_s0, thermal_ke_s0, ..._s1, ...``

    Parameters
    ----------
    path : Path
        Path to ``SpeciesQuantities.txt`` file.

    Returns
    -------
    TabularData
        Columnar representation with ``index_column="cycle"``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        msg = f"SpeciesQuantities file not found: {path}"
        raise FileNotFoundError(msg)

    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if not rows:
        msg = f"No data rows found in {path}"
        raise ValueError(msg)

    data = np.array(rows, dtype=np.float64)

    species_ids = sorted(set(int(x) for x in data[:, _SQ_SPECIES]))
    cycles = sorted(set(data[:, _SQ_CYCLE]))
    n_cycles = len(cycles)
    cycle_arr = np.array(cycles, dtype=np.float64)

    # Build cycle→row-index mapping per species
    columns: dict[str, FloatArray] = {"cycle": cycle_arr}
    for s in species_ids:
        mask = data[:, _SQ_SPECIES] == s
        s_data = data[mask]
        order = np.argsort(s_data[:, _SQ_CYCLE])
        s_data = s_data[order]
        # Ensure same cycle count (truncate to common set)
        n = min(len(s_data), n_cycles)
        columns[f"momentum_s{s}"] = s_data[:n, _SQ_MOMENTUM]
        columns[f"total_ke_s{s}"] = s_data[:n, _SQ_TOTAL_KE]
        columns[f"bulk_ke_s{s}"] = s_data[:n, _SQ_BULK_KE]
        columns[f"thermal_ke_s{s}"] = s_data[:n, _SQ_THERMAL_KE]

    return TabularData(
        name="species_quantities",
        columns=columns,
        index_column="cycle",
        metadata={"source": "iPIC3D SpeciesQuantities"},
    )


def detect_conserved(path: Path) -> list[str]:
    """Check whether auxiliary diagnostic data exists at *path*.

    Parameters
    ----------
    path : Path
        Simulation output directory.

    Returns
    -------
    list[str]
        Names of available auxiliary datasets.
    """
    result: list[str] = []
    if (path / "ConservedQuantities.txt").exists():
        result.append("conserved_quantities")
    elif (path / "info-conserved").is_dir():
        cq_files = list((path / "info-conserved").glob("ConservedQuantities*.txt"))
        if cq_files:
            result.append("conserved_quantities")
    if (path / "SpeciesQuantities.txt").exists():
        result.append("species_quantities")
    return result


def load_ipic3d_auxiliary(
    path: Path,
    name: str,
) -> TabularData:
    """Load an iPIC3D auxiliary dataset by name.

    Parameters
    ----------
    path : Path
        Simulation output directory.
    name : str
        Dataset name.

    Returns
    -------
    TabularData

    Raises
    ------
    KeyError
        If *name* is not a recognized auxiliary dataset name.
    FileNotFoundError
        If *name* is recognized but the underlying files are missing
        from *path*.
    """
    if name == "conserved_quantities":
        cq_file = path / "ConservedQuantities.txt"
        cq_dir = path / "info-conserved"

        if cq_file.exists():
            cq = load_conserved_quantities(cq_file)
        elif cq_dir.is_dir():
            cq = load_conserved_quantities(cq_dir)
        else:
            msg = (
                f"No ConservedQuantities data found in {path}. "
                f"Expected ConservedQuantities.txt or info-conserved/"
            )
            raise FileNotFoundError(msg)

        return conserved_to_tabular(cq)

    if name == "species_quantities":
        sq_file = path / "SpeciesQuantities.txt"
        if not sq_file.exists():
            msg = f"No SpeciesQuantities.txt found in {path}"
            raise FileNotFoundError(msg)
        return load_species_quantities(sq_file)

    msg = f"Unknown iPIC3D auxiliary dataset {name!r}"
    raise KeyError(msg)
