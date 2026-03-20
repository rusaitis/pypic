"""iPIC3D particle data reader for phdf5 format."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from pypic.readers.base import ParticleData

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.readers.ipic3d._config import IPic3DConfig


def detect_particle_steps(path: Path) -> list[int]:
    """Scan for ``Particles_XXXXX/`` directories and return sorted step list.

    Parameters
    ----------
    path : Path
        Simulation output directory.

    Returns
    -------
    list[int]
        Sorted timestep indices with particle output.
    """
    pattern = re.compile(r"^Particles_(\d+)$")
    steps: list[int] = []
    for entry in path.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def read_phdf5_particles(
    path: Path,
    step: int,
    species: int,
    config: IPic3DConfig,
    *,
    columns: Iterable[str] | None = None,
) -> ParticleData:
    r"""Read particle data from a phdf5-format iPIC3D output file.

    Parameters
    ----------
    path : Path
        Simulation output directory.
    step : int
        Timestep index.
    species : int
        Zero-based species index.
    config : IPic3DConfig
        Parsed iPIC3D configuration (for species names).
    columns : Iterable[str] | None
        Subset of ``{"position", "velocity"}`` to load.
        ``None`` loads all.  ``charge`` is always loaded.

    Returns
    -------
    ParticleData
    """
    step_str = f"{step:05d}"
    h5_path = path / f"Particles_{step_str}" / f"species_{species}_{step_str}.h5"
    group_name = f"Particles/species_{species}"

    want = set(columns) if columns is not None else {"position", "velocity"}

    with h5py.File(h5_path, "r") as f:
        group = f[group_name]

        # Determine n_particles from whichever dataset is available
        if "position" in group:
            n_particles = group["position"].shape[0]
        elif "velocity" in group:
            n_particles = group["velocity"].shape[0]
        else:
            msg = f"No position or velocity dataset in {h5_path}:{group_name}"
            raise ValueError(msg)

        position = None
        if "position" in want:
            position = np.array(group["position"])

        velocity = None
        if "velocity" in want:
            velocity = np.array(group["velocity"])

        # charge: always loaded — scalar (1,1) in phdf5, broadcast to (N,)
        q_raw = np.array(group["q"])
        q_scalar = float(q_raw.flat[0])
        charge = np.full(n_particles, q_scalar, dtype=np.float64)

        # ID: optional integer tracking ID
        particle_id = None
        if "ID" in group:
            particle_id = np.array(group["ID"], dtype=np.int64).ravel()

    species_name = f"species_{species}"

    return ParticleData(
        species_index=species,
        species_name=species_name,
        position=position,
        velocity=velocity,
        charge=charge,
        n_particles=n_particles,
        metadata={"path": str(h5_path), "format": "phdf5"},
        id=particle_id,
    )
