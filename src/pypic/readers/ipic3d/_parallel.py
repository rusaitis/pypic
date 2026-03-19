"""iPIC3D parallel HDF5 (phdf5) reader."""

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
    _MOMENT_COMPONENT_MAP,
    _PHDF5_DIAGONAL_PRESSURE,
    _PHDF5_PRESSURE_MAP,
    gaussian_current_to_si,
    gaussian_density_to_si,
    gaussian_pressure_to_si,
    per_species_canonical,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import FloatArray


class IPic3DParallelReader:
    """Read iPIC3D parallel HDF5 (phdf5) output.

    Each timestep is stored in separate ``Fields_XXXXX/`` and
    ``Moments_XXXXX/`` directories containing one ``.h5`` file per
    field group.

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

        Scans for ``Fields_XXXXX`` directories under *path*.

        Parameters
        ----------
        path : Path
            Simulation output directory.

        Returns
        -------
        list[int]
            Sorted timestep indices.
        """
        steps: list[int] = []
        pattern = re.compile(r"^Fields_(\d+)$")
        for entry in path.iterdir():
            if entry.is_dir():
                m = pattern.match(entry.name)
                if m:
                    steps.append(int(m.group(1)))
        return sorted(steps)

    def read_timestep(self, path: Path, step: int) -> FieldDataset:
        """Read all field and moment data for a single timestep.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep index (e.g. 0, 10, 20).

        Returns
        -------
        FieldDataset
            Field data with canonical names and 4π corrections applied.
        """
        step_str = f"{step:05d}"
        fields: dict[str, FloatArray] = {}

        # Electromagnetic fields
        b_path = path / f"Fields_{step_str}" / f"B_{step_str}.h5"
        with h5py.File(b_path, "r") as f:
            for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                if ipic_name.startswith("B") and ipic_name in f["Fields"]:
                    fields[canon_name] = np.array(f["Fields"][ipic_name])

        e_path = path / f"Fields_{step_str}" / f"E_{step_str}.h5"
        with h5py.File(e_path, "r") as f:
            for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                if ipic_name.startswith("E") and ipic_name in f["Fields"]:
                    fields[canon_name] = np.array(f["Fields"][ipic_name])

        # Per-species moments
        ns = self._config.ns
        for s in range(ns):
            # Current density
            j_path = path / f"Moments_{step_str}" / f"J_species_{s}_{step_str}.h5"
            with h5py.File(j_path, "r") as f:
                group = f[f"Moments/species_{s}"]
                for comp in ("Jx", "Jy", "Jz"):
                    canon = per_species_canonical(comp, s)
                    fields[canon] = gaussian_current_to_si(np.array(group[comp]))

            # Charge density
            rho_path = path / f"Moments_{step_str}" / f"rho_species_{s}_{step_str}.h5"
            with h5py.File(rho_path, "r") as f:
                group = f[f"Moments/species_{s}"]
                canon = per_species_canonical("rho", s)
                fields[canon] = gaussian_density_to_si(np.array(group["rho"]))

            # Pressure tensor (optional — not all runs export pressure)
            p_path = (
                path / f"Moments_{step_str}" / f"Pressure_species_{s}_{step_str}.h5"
            )
            if p_path.exists():
                with h5py.File(p_path, "r") as f:
                    group = f[f"Moments/species_{s}"]
                    for phdf5_name, canon_base in _PHDF5_PRESSURE_MAP.items():
                        if phdf5_name in group:
                            canon = f"{canon_base}_s{s}"
                            data = np.array(group[phdf5_name])
                            if (
                                phdf5_name in _PHDF5_DIAGONAL_PRESSURE
                                and self._config.qom[s] < 0
                            ):
                                data = -data
                            fields[canon] = gaussian_pressure_to_si(data)

        # Compute totals by summing over species
        for moment_comp, canon_total in _MOMENT_COMPONENT_MAP.items():
            total = np.zeros_like(fields[per_species_canonical(moment_comp, 0)])
            for s in range(ns):
                total = total + fields[per_species_canonical(moment_comp, s)]
            fields[canon_total] = total

        sc = self._sim_config
        return FieldDataset.from_arrays(
            fields,
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
