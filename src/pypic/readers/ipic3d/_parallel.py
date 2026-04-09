"""iPIC3D parallel HDF5 (phdf5) reader."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pypic.readers.base import FieldDataset, ParticleData, SimulationConfig, TabularData
from pypic.readers.ipic3d._config import IPic3DConfig, to_simulation_config
from pypic.readers.ipic3d._conserved import detect_conserved, load_ipic3d_auxiliary
from pypic.readers.ipic3d._field_map import (
    _EFLUX_MAP,
    _FIELD_NAME_MAP,
    _PHDF5_DIAGONAL_PRESSURE,
    _PHDF5_PRESSURE_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    gaussian_current_to_si,
    gaussian_density_to_si,
    gaussian_pressure_to_si,
    per_species_canonical,
    per_species_eflux_canonical,
)
from pypic.readers.ipic3d._particles import detect_particle_steps, read_phdf5_particles

if TYPE_CHECKING:
    from collections.abc import Iterable
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

    def __init__(
        self, config: IPic3DConfig, sim_config: SimulationConfig | None = None
    ) -> None:
        self._config = config
        self._sim_config = sim_config or to_simulation_config(config)

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

        # Electromagnetic fields — skip file open if none wanted
        want_b = expanded is None or any(f"B{i}" in expanded for i in range(1, 4))
        if want_b:
            b_path = path / f"Fields_{step_str}" / f"B_{step_str}.h5"
            with h5py.File(b_path, "r") as f:
                for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                    if (
                        ipic_name.startswith("B")
                        and ipic_name in f["Fields"]
                        and (expanded is None or canon_name in expanded)
                    ):
                        field_data[canon_name] = np.array(f["Fields"][ipic_name])

        want_e = expanded is None or any(f"E{i}" in expanded for i in range(1, 4))
        if want_e:
            e_path = path / f"Fields_{step_str}" / f"E_{step_str}.h5"
            with h5py.File(e_path, "r") as f:
                for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                    if (
                        ipic_name.startswith("E")
                        and ipic_name in f["Fields"]
                        and (expanded is None or canon_name in expanded)
                    ):
                        field_data[canon_name] = np.array(f["Fields"][ipic_name])

        # Per-species moments
        for s in range(ns):
            # Current density
            want_j_s = expanded is None or any(
                per_species_canonical(c, s) in expanded for c in ("Jx", "Jy", "Jz")
            )
            if want_j_s:
                j_path = path / f"Moments_{step_str}" / f"J_species_{s}_{step_str}.h5"
                with h5py.File(j_path, "r") as f:
                    group = f[f"Moments/species_{s}"]
                    for comp in ("Jx", "Jy", "Jz"):
                        canon = per_species_canonical(comp, s)
                        if expanded is not None and canon not in expanded:
                            continue
                        field_data[canon] = gaussian_current_to_si(
                            np.array(group[comp])
                        )

            # Charge density
            rho_canon = per_species_canonical("rho", s)
            if expanded is None or rho_canon in expanded:
                rho_path = (
                    path / f"Moments_{step_str}" / f"rho_species_{s}_{step_str}.h5"
                )
                with h5py.File(rho_path, "r") as f:
                    group = f[f"Moments/species_{s}"]
                    field_data[rho_canon] = gaussian_density_to_si(
                        np.array(group["rho"])
                    )

            # Pressure tensor (optional)
            want_p_s = expanded is None or any(
                f"{cb}_s{s}" in expanded for cb in _PHDF5_PRESSURE_MAP.values()
            )
            if want_p_s:
                p_path = (
                    path / f"Moments_{step_str}" / f"Pressure_species_{s}_{step_str}.h5"
                )
                if p_path.exists():
                    with h5py.File(p_path, "r") as f:
                        group = f[f"Moments/species_{s}"]
                        for phdf5_name, canon_base in _PHDF5_PRESSURE_MAP.items():
                            canon = f"{canon_base}_s{s}"
                            if expanded is not None and canon not in expanded:
                                continue
                            if phdf5_name in group:
                                data = np.array(group[phdf5_name])
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
                f"{cb}_s{s}" in expanded for cb in _EFLUX_MAP.values()
            )
            if want_ef_s:
                ef_path = (
                    path / f"Moments_{step_str}" / f"E_flux_species_{s}_{step_str}.h5"
                )
                if ef_path.exists():
                    with h5py.File(ef_path, "r") as f:
                        group = f[f"Moments/species_{s}"]
                        for phdf5_name, _canon_base in _EFLUX_MAP.items():
                            canon = per_species_eflux_canonical(phdf5_name, s)
                            if expanded is not None and canon not in expanded:
                                continue
                            if phdf5_name in group:
                                data = np.array(group[phdf5_name])
                                field_data[canon] = gaussian_pressure_to_si(data)

        field_data = compute_totals_and_filter(field_data, ns, expanded, wanted)

        sc = self._sim_config
        return FieldDataset.from_arrays(
            field_data,
            sc.grid,
            sc.normalization,
            species=sc.species,
            physics=sc.physics,
            metadata={**dict(sc.metadata), "step": step},
            frame=sc.frame,
            transforms=sc.transforms or None,
        )

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
            ``None`` loads all.  ``charge`` is always loaded.

        Returns
        -------
        ParticleData
        """
        return read_phdf5_particles(path, step, species, self._config, columns=columns)

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of auxiliary datasets at *path*."""
        return detect_conserved(path)

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset from *path*."""
        return load_ipic3d_auxiliary(path, name)
