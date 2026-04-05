"""iPIC3D serial HDF5 (shdf5) reader."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from pypic.readers.base import FieldDataset, SimulationConfig, TabularData
from pypic.readers.ipic3d._config import IPic3DConfig, to_simulation_config
from pypic.readers.ipic3d._conserved import detect_conserved, load_ipic3d_auxiliary
from pypic.readers.ipic3d._field_map import (
    _FIELD_NAME_MAP,
    _PHDF5_DIAGONAL_PRESSURE,
    _PHDF5_EFLUX_MAP,
    _PHDF5_PRESSURE_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    gaussian_current_to_si,
    gaussian_density_to_si,
    gaussian_pressure_to_si,
    per_species_canonical,
    per_species_eflux_canonical,
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

    def __init__(
        self, config: IPic3DConfig, sim_config: SimulationConfig | None = None
    ) -> None:
        self._config = config
        self._sim_config = sim_config or to_simulation_config(config)

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

            # Pressure tensor (optional — not all shdf5 runs include it)
            want_p_s = expanded is None or any(
                f"{cb}_s{s}" in expanded for cb in _PHDF5_PRESSURE_MAP.values()
            )
            if want_p_s:
                for phdf5_name, canon_base in _PHDF5_PRESSURE_MAP.items():
                    canon = f"{canon_base}_s{s}"
                    if expanded is not None and canon not in expanded:
                        continue
                    group_path = f"moments/species_{s}/{phdf5_name}"
                    # Check if the dataset exists in proc0
                    try:
                        with h5py.File(proc_files[0], "r") as f:
                            if group_path not in f or cycle_key not in f[group_path]:
                                continue
                    except (KeyError, OSError):
                        continue
                    data = self._assemble_field(proc_files, group_path, cycle_key)
                    if (
                        phdf5_name in _PHDF5_DIAGONAL_PRESSURE
                        and self._config.qom[s] < 0
                    ):
                        data = -data
                    data = gaussian_pressure_to_si(data)
                    # Charge-weighted → mass-weighted: ×(m/|q|) = ×(1/|qom|)
                    data = data / abs(self._config.qom[s])
                    field_data[canon] = data

            # Energy flux (optional)
            want_ef_s = expanded is None or any(
                f"{cb}_s{s}" in expanded for cb in _PHDF5_EFLUX_MAP.values()
            )
            if want_ef_s:
                for ef_name, _ef_canon_base in _PHDF5_EFLUX_MAP.items():
                    canon = per_species_eflux_canonical(ef_name, s)
                    if expanded is not None and canon not in expanded:
                        continue
                    group_path = f"moments/species_{s}/{ef_name}"
                    try:
                        with h5py.File(proc_files[0], "r") as f:
                            if group_path not in f or cycle_key not in f[group_path]:
                                continue
                    except (KeyError, OSError):
                        continue
                    data = self._assemble_field(proc_files, group_path, cycle_key)
                    field_data[canon] = gaussian_pressure_to_si(data)

        field_data = compute_totals_and_filter(field_data, ns, expanded, wanted)

        sc = self._sim_config
        return FieldDataset.from_arrays(
            field_data,
            sc.grid,
            sc.normalization,
            species=sc.species,
            physics=dict(sc.physics),
            metadata={**dict(sc.metadata), "step": step},
            frame=sc.frame,
            transforms=sc.transforms or None,
        )

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of auxiliary datasets at *path*."""
        return detect_conserved(path)

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset from *path*."""
        return load_ipic3d_auxiliary(path, name)
