"""iPIC3D serial HDF5 (shdf5) reader."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from pypic.readers.base import FieldDataset, TabularData
from pypic.readers.ipic3d._config import IPic3DConfig, to_simulation_config
from pypic.readers.ipic3d._conserved import detect_conserved, load_ipic3d_auxiliary
from pypic.readers.ipic3d._field_map import (
    _FIELD_NAME_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    gaussian_current_to_si,
    gaussian_density_to_si,
    per_species_canonical,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.types import FloatArray


class IPic3DSerialReader:
    """Read iPIC3D serial HDF5 (shdf5) output.

    Each MPI process writes to its own ``procN.hdf`` file containing all
    timesteps. This reader assembles the global arrays from the per-process
    local patches.

    Parameters
    ----------
    config : IPic3DConfig
        Parsed iPIC3D configuration.
    """

    def __init__(self, config: IPic3DConfig) -> None:
        self._config = config
        self._sim_config = to_simulation_config(config)

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted list of available timestep numbers.

        Reads cycle keys from ``proc0.hdf`` fields group.

        Parameters
        ----------
        path : Path
            Simulation output directory.

        Returns
        -------
        list[int]
            Sorted timestep indices.
        """
        proc0 = path / "proc0.hdf"
        pattern = re.compile(r"^cycle_(\d+)$")
        with h5py.File(proc0, "r") as f:
            steps: list[int] = []
            for key in f["fields/Bx"]:
                m = pattern.match(key)
                if m:
                    steps.append(int(m.group(1)))
        return sorted(steps)

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

        # Base local sizes (may not divide evenly)
        nxc_base = cfg.nxc // cfg.xlen
        nyc_base = cfg.nyc // cfg.ylen
        nzc_base = cfg.nzc // cfg.zlen

        for proc_path in proc_files:
            with h5py.File(proc_path, "r") as f:
                coords = f["topology/cartesian_coord"][()]
                ix, iy, iz = int(coords[0]), int(coords[1]), int(coords[2])

                data = np.array(f[group_path][cycle_key])
                nx_local, ny_local, nz_local = data.shape

                x0 = ix * nxc_base
                y0 = iy * nyc_base
                z0 = iz * nzc_base

                result[x0 : x0 + nx_local, y0 : y0 + ny_local, z0 : z0 + nz_local] = (
                    data
                )

        return result

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
        cycle_key = f"cycle_{step}"
        ns = self._config.ns
        wanted: set[str] | None = set(fields) if fields is not None else None
        expanded: set[str] | None = (
            expand_moment_dependencies(wanted, ns) if wanted is not None else None
        )
        field_data: dict[str, FloatArray] = {}

        # Electromagnetic fields
        for ipic_name, canon_name in _FIELD_NAME_MAP.items():
            if expanded is not None and canon_name not in expanded:
                continue
            field_data[canon_name] = self._assemble_field(
                proc_files, f"fields/{ipic_name}", cycle_key
            )

        # Per-species moments
        for s in range(ns):
            for comp in ("Jx", "Jy", "Jz"):
                canon = per_species_canonical(comp, s)
                if expanded is not None and canon not in expanded:
                    continue
                raw = self._assemble_field(
                    proc_files, f"moments/species_{s}/{comp}", cycle_key
                )
                field_data[canon] = gaussian_current_to_si(raw)

            canon_rho = per_species_canonical("rho", s)
            if expanded is None or canon_rho in expanded:
                raw_rho = self._assemble_field(
                    proc_files, f"moments/species_{s}/rho", cycle_key
                )
                field_data[canon_rho] = gaussian_density_to_si(raw_rho)

        field_data = compute_totals_and_filter(field_data, ns, expanded, wanted)

        sc = self._sim_config
        return FieldDataset.from_arrays(
            field_data,
            sc.grid,
            sc.normalization,
            species=sc.species,
            physics=dict(sc.physics),
            metadata={**dict(sc.metadata), "step": step},
        )

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of auxiliary datasets at *path*."""
        return detect_conserved(path)

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset from *path*."""
        return load_ipic3d_auxiliary(path, name)
