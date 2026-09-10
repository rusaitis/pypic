"""iPIC3D serial HDF5 (shdf5) reader."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pypic.readers.ipic3d._base import IPic3DReaderBase
from pypic.readers.ipic3d._field_map import (
    _FIELD_NAME_MAP,
    _PHDF5_PRESSURE_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    infer_total_fields,
    read_species_moments,
    species_moment_names,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray

_CYCLE_RE = re.compile(r"^cycle_(\d+)$")


class IPic3DSerialReader(IPic3DReaderBase):
    """Read iPIC3D serial HDF5 (shdf5) output.

    Each MPI process writes to its own ``procN.hdf`` file containing all
    timesteps. This reader assembles the global arrays from the per-process
    local patches.

    Unique to this reader: the per-process patch reassembly. Field-name
    mapping, Gaussian-CGS unit conversions, pressure-tensor mass
    correction, and config translation live in
    `pypic.readers.ipic3d._field_map` and
    `pypic.readers.ipic3d._config`, shared with the parallel and
    H5hut readers.
    """

    def available_timesteps(self, path: Path) -> list[int]:
        """Sorted timestep numbers, from the cycle keys in ``proc0.hdf``."""
        with h5py.File(path / "proc0.hdf", "r") as f:
            return sorted(
                int(m.group(1)) for key in f["fields/Bx"] if (m := _CYCLE_RE.match(key))
            )

    @staticmethod
    def _present_moments(proc0: Path, cycle_key: str) -> set[str]:
        """``species_N/<native>`` moment groups carrying *cycle_key* in proc0."""
        with h5py.File(proc0, "r") as f:
            if "moments" not in f:
                return set()
            return {
                f"{species}/{native}"
                for species, group in f["moments"].items()
                for native, cycles in group.items()
                if cycle_key in cycles
            }

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Opens ``proc0.hdf`` and inspects HDF5 group keys without
        loading array data.

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
        proc0 = path / "proc0.hdf"
        cycle_key = f"cycle_{step}"
        ns = self._config.ns
        mapping: dict[str, str | None] = {}

        with h5py.File(proc0, "r") as f:
            if "fields" in f:
                mapping.update(
                    (_FIELD_NAME_MAP[name], name)
                    for name in f["fields"]
                    if name in _FIELD_NAME_MAP
                )
        present = self._present_moments(proc0, cycle_key)
        for s in range(ns):
            names = species_moment_names(s, _PHDF5_PRESSURE_MAP)
            mapping.update(
                (canon, native)
                for canon, native in names.items()
                if f"species_{s}/{native}" in present
            )

        for total in infer_total_fields(set(mapping), ns):
            mapping[total] = None
        return mapping

    def _assemble_field(
        self,
        proc_files: list[Path],
        group_path: str,
        cycle_key: str,
    ) -> FloatArray:
        """Assemble a global array from per-process local patches.

        Parameters
        ----------
        proc_files : list[Path]
            Paths to all proc*.hdf files.
        group_path : str
            HDF5 group path (e.g. ``"fields/Bx"``).
        cycle_key : str
            Cycle dataset name (e.g. ``"cycle_10"``).

        Returns
        -------
        FloatArray
            Assembled global array of shape ``(Nxc+1, Nyc+1, Nzc+1)``.
        """
        cfg = self._config
        global_shape = (cfg.nxc + 1, cfg.nyc + 1, cfg.nzc + 1)
        result = np.zeros(global_shape, dtype=np.float64)

        # Base local sizes and remainders for uneven MPI decompositions.
        # iPIC3D gives the first (N % P) ranks one extra cell.
        nxc_base = cfg.nxc // cfg.xlen
        nyc_base = cfg.nyc // cfg.ylen
        nzc_base = cfg.nzc // cfg.zlen
        nxc_extra = cfg.nxc % cfg.xlen
        nyc_extra = cfg.nyc % cfg.ylen
        nzc_extra = cfg.nzc % cfg.zlen

        for proc_path in proc_files:
            with h5py.File(proc_path, "r") as f:
                coords = f["topology/cartesian_coord"][()]
                ix, iy, iz = int(coords[0]), int(coords[1]), int(coords[2])

                data = np.array(f[group_path][cycle_key])
                nx_local, ny_local, nz_local = data.shape

                x0 = ix * nxc_base + min(ix, nxc_extra)
                y0 = iy * nyc_base + min(iy, nyc_extra)
                z0 = iz * nzc_base + min(iz, nzc_extra)

                result[x0 : x0 + nx_local, y0 : y0 + ny_local, z0 : z0 + nz_local] = (
                    data
                )

        return result

    def _load_moment(
        self,
        proc_files: list[Path],
        cycle_key: str,
        present: set[str],
        species: int,
        native: str,
    ) -> FloatArray | None:
        """Assemble one species moment, or ``None`` when proc0 lacks it."""
        if f"species_{species}/{native}" not in present:
            return None
        return self._assemble_field(
            proc_files, f"moments/species_{species}/{native}", cycle_key
        )

    def read_timestep(
        self,
        path: Path,
        step: int,
        *,
        fields: Iterable[str] | None = None,
    ) -> FieldDataset:
        """Read field and moment data for a single timestep.

        Assembles global arrays from per-process files, applies 4π
        correction to densities and currents, and computes totals.

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
        proc_files = sorted(path.glob("proc*.hdf"))
        if not proc_files:
            msg = f"No proc*.hdf files found in {path}"
            raise FileNotFoundError(msg)
        cycle_key = f"cycle_{step}"
        ns = self._config.ns
        wanted: set[str] | None = set(fields) if fields is not None else None
        expanded: set[str] | None = (
            expand_moment_dependencies(wanted, ns) if wanted is not None else None
        )
        field_data: dict[str, FloatArray] = {}

        for ipic_name, canon_name in _FIELD_NAME_MAP.items():
            if expanded is not None and canon_name not in expanded:
                continue
            field_data[canon_name] = self._assemble_field(
                proc_files, f"fields/{ipic_name}", cycle_key
            )

        present = self._present_moments(proc_files[0], cycle_key)
        for s in range(ns):
            field_data.update(
                read_species_moments(
                    partial(self._load_moment, proc_files, cycle_key, present, s),
                    s,
                    species_qom=self._config.qom[s],
                    expanded=expanded,
                    pressure_map=_PHDF5_PRESSURE_MAP,
                )
            )

        field_data = compute_totals_and_filter(field_data, ns, expanded, wanted)
        return self._finish(field_data, step=step)
