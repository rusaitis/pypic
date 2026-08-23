"""iPIC3D particle data reader for phdf5 format."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pypic.containers import ParticleData
from pypic.exceptions import UnknownFieldError

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


_PARTICLE_COLUMNS = frozenset({"position", "velocity"})


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
        Parsed iPIC3D configuration.  Provides per-species charge/mass via
        ``qom``.  Emits canonical form: ``weight`` (derived from native
        per-particle ``q`` as ``|q|`` since iPIC3D sets ``|q_species| = 1``)
        plus scalar ``species_charge`` and ``species_mass``.
    columns : Iterable[str] | None
        Subset of ``{"position", "velocity"}`` to load.  ``None`` loads all.

    Returns
    -------
    ParticleData

    Raises
    ------
    UnknownFieldError
        If *columns* names anything outside ``{"position", "velocity"}``.
        A typo fails here rather than silently dropping the column.
    """
    step_str = f"{step:05d}"
    h5_path = path / f"Particles_{step_str}" / f"species_{species}_{step_str}.h5"
    group_name = f"Particles/species_{species}"

    want = set(columns) if columns is not None else set(_PARTICLE_COLUMNS)
    unknown = want - _PARTICLE_COLUMNS
    if unknown:
        msg = (
            f"Unknown particle column(s) {sorted(unknown)}. "
            f"Available: {sorted(_PARTICLE_COLUMNS)}."
        )
        raise UnknownFieldError(msg)

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

        # iPIC3D stores macroparticle charge q_macro = q_s * w. |q_s| = 1
        # by convention, so weight = |q_macro|. Uniform-weight runs emit a
        # scalar/singleton; particle-splitting or non-uniform-density runs
        # emit a per-particle array.
        q_raw = np.asarray(group["q"])
        if q_raw.size == n_particles:
            weight = np.abs(q_raw.reshape(n_particles).astype(np.float64))
        elif q_raw.size == 1:
            weight = np.full(n_particles, abs(float(q_raw.flat[0])), dtype=np.float64)
        else:
            msg = (
                f"iPIC3D 'q' dataset size {q_raw.size} is neither 1 nor "
                f"n_particles={n_particles} in {h5_path}:{group_name}"
            )
            raise ValueError(msg)

        # ID: optional integer tracking ID
        particle_id = None
        if "ID" in group:
            particle_id = np.array(group["ID"], dtype=np.int64).ravel()

    species_name = f"species_{species}"

    # iPIC3D normalization: |q_species| = 1, sign(q_species) = sign(qom[s]),
    # m_species = 1 / |qom[s]|.
    qom_s = config.qom[species]
    species_charge = float(np.sign(qom_s))
    species_mass = 1.0 / abs(qom_s)

    return ParticleData(
        species_index=species,
        species_name=species_name,
        position=position,
        velocity=velocity,
        n_particles=n_particles,
        metadata={"path": str(h5_path), "format": "phdf5"},
        id=particle_id,
        weight=weight,
        species_charge=species_charge,
        species_mass=species_mass,
    )
