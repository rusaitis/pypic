"""iPIC3D parallel HDF5 (phdf5) reader."""

from __future__ import annotations

import re
from contextlib import ExitStack
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pypic.readers.ipic3d._base import IPic3DReaderBase
from pypic.readers.ipic3d._field_map import (
    _EFLUX_MAP,
    _FIELD_NAME_MAP,
    _PHDF5_PRESSURE_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    infer_total_fields,
    read_species_moments,
    species_moment_names,
)
from pypic.readers.ipic3d._particles import detect_particle_steps, read_phdf5_particles

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.containers import ParticleData
    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray

_FIELDS_DIR_RE = re.compile(r"^Fields_(\d+)$")

# Which ``Moments_XXXXX/<kind>_species_N_XXXXX.h5`` file holds each native moment.
_MOMENT_FILE: dict[str, str] = {
    "rho": "rho",
    "Jx": "J",
    "Jy": "J",
    "Jz": "J",
    **dict.fromkeys(_PHDF5_PRESSURE_MAP, "Pressure"),
    **dict.fromkeys(_EFLUX_MAP, "E_flux"),
}


class _MomentFiles:
    """One species' moment files for one timestep, opened on first use.

    Each moment kind lives in its own file; pressure and energy flux are
    optional output, so a missing file reads as "not carried".
    """

    def __init__(self, moments_dir: Path, step_str: str, species: int) -> None:
        self._dir = moments_dir
        self._step_str = step_str
        self._species = species
        self._stack = ExitStack()
        self._groups: dict[str, h5py.Group | None] = {}

    def __enter__(self) -> _MomentFiles:
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()

    def _group(self, native: str) -> h5py.Group | None:
        kind = _MOMENT_FILE.get(native)
        if kind is None:
            return None
        if kind not in self._groups:
            file = self._dir / f"{kind}_species_{self._species}_{self._step_str}.h5"
            self._groups[kind] = (
                self._stack.enter_context(h5py.File(file, "r"))[
                    f"Moments/species_{self._species}"
                ]
                if file.exists()
                else None
            )
        return self._groups[kind]

    def has(self, native: str) -> bool:
        group = self._group(native)
        return group is not None and native in group

    def load(self, native: str) -> FloatArray | None:
        group = self._group(native)
        if group is None or native not in group:
            return None
        return np.array(group[native])


class IPic3DParallelReader(IPic3DReaderBase):
    """Read iPIC3D parallel HDF5 (phdf5) output.

    Each timestep is stored in separate ``Fields_XXXXX/`` and
    ``Moments_XXXXX/`` directories containing one ``.h5`` file per
    field group.

    Unique to this reader: scanning timestep directories and the
    one-file-per-moment layout. Field-name mapping, Gaussian-CGS unit
    conversions, pressure-tensor mass correction, and config
    translation live in `pypic.readers.ipic3d._field_map` and
    `pypic.readers.ipic3d._config`, shared with the serial and
    H5hut readers.
    """

    def available_timesteps(self, path: Path) -> list[int]:
        """Sorted timestep numbers, from the ``Fields_XXXXX`` directories."""
        return sorted(
            int(m.group(1))
            for entry in path.iterdir()
            if entry.is_dir() and (m := _FIELDS_DIR_RE.match(entry.name))
        )

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Probes HDF5 files in the ``Fields_XXXXX/`` and
        ``Moments_XXXXX/`` directories without loading arrays.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep index.

        Returns
        -------
        dict[str, str | None]
            Canonical → native name, ``None`` for computed totals.
        """
        step_str = f"{step:05d}"
        ns = self._config.ns
        mapping: dict[str, str | None] = {}

        for prefix in ("B", "E"):
            em_path = path / f"Fields_{step_str}" / f"{prefix}_{step_str}.h5"
            if em_path.exists():
                with h5py.File(em_path, "r") as f:
                    mapping.update(
                        (_FIELD_NAME_MAP[name], name)
                        for name in f["Fields"]
                        if name in _FIELD_NAME_MAP
                    )

        for s in range(ns):
            with _MomentFiles(path / f"Moments_{step_str}", step_str, s) as files:
                names = species_moment_names(s, _PHDF5_PRESSURE_MAP)
                mapping.update(
                    (canon, native)
                    for canon, native in names.items()
                    if files.has(native)
                )

        for total in infer_total_fields(set(mapping), ns):
            mapping[total] = None
        return mapping

    def read_timestep(
        self,
        path: Path,
        step: int,
        *,
        fields: Iterable[str] | None = None,
    ) -> FieldDataset:
        """Read field and moment data for a single timestep.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep index (e.g. 0, 10, 20).
        fields : Iterable[str] | None
            When given, only read these canonical field names.

        Returns
        -------
        FieldDataset
            Field data with canonical names and 4π corrections applied.
        """
        step_str = f"{step:05d}"
        ns = self._config.ns
        wanted: set[str] | None = set(fields) if fields is not None else None
        expanded: set[str] | None = (
            expand_moment_dependencies(wanted, ns) if wanted is not None else None
        )
        field_data: dict[str, FloatArray] = {}

        for prefix in ("B", "E"):
            if expanded is not None and not any(
                f"{prefix}_{i}" in expanded for i in "123"
            ):
                continue
            em_path = path / f"Fields_{step_str}" / f"{prefix}_{step_str}.h5"
            with h5py.File(em_path, "r") as f:
                for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                    if (
                        ipic_name.startswith(prefix)
                        and ipic_name in f["Fields"]
                        and (expanded is None or canon_name in expanded)
                    ):
                        field_data[canon_name] = np.array(f["Fields"][ipic_name])

        for s in range(ns):
            with _MomentFiles(path / f"Moments_{step_str}", step_str, s) as files:
                field_data.update(
                    read_species_moments(
                        files.load,
                        s,
                        species_qom=self._config.qom[s],
                        expanded=expanded,
                        pressure_map=_PHDF5_PRESSURE_MAP,
                    )
                )

        field_data = compute_totals_and_filter(field_data, ns, expanded, wanted)
        return self._finish(field_data, step=step)

    def available_particle_steps(self, path: Path) -> list[int]:
        """Return sorted timestep indices that have particle data."""
        return detect_particle_steps(path)

    def read_particles(
        self,
        path: Path,
        step: int,
        species: int,
        *,
        columns: Iterable[str] | None = None,
    ) -> ParticleData:
        """Load particle data for one species at one timestep.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep index.
        species : int
            Zero-based species index.
        columns : Iterable[str] | None
            Subset of ``{"position", "velocity"}`` to load.
            ``None`` loads all.  Per-particle ``weight`` and the scalar
            ``species_charge``/``species_mass`` are always populated
            (canonical layout, ``docs/schema.md``).

        Returns
        -------
        ParticleData
        """
        return read_phdf5_particles(path, step, species, self._config, columns=columns)
