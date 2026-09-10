"""iPIC3D simulation readers (parallel HDF5, serial HDF5, and H5hut)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.readers.ipic3d._config import (
    IPic3DConfig,
    parse_inp,
    parse_settings_hdf,
    to_simulation_config,
)
from pypic.readers.ipic3d._conserved import (
    ConservedQuantities,
    conserved_to_tabular,
    load_conserved_quantities,
    load_species_quantities,
)
from pypic.readers.ipic3d._h5hut import IPic3DH5hutReader
from pypic.readers.ipic3d._parallel import IPic3DParallelReader
from pypic.readers.ipic3d._particles import detect_particle_steps, read_phdf5_particles
from pypic.readers.ipic3d._probe import can_read_confidence
from pypic.readers.ipic3d._serial import IPic3DSerialReader

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.containers import SimulationConfig
    from pypic.readers._protocols import SimulationReader

__all__ = [
    "ConservedQuantities",
    "IPic3DConfig",
    "IPic3DH5hutReader",
    "IPic3DParallelReader",
    "IPic3DSerialReader",
    "can_read_confidence",
    "conserved_to_tabular",
    "detect_particle_steps",
    "load_conserved_quantities",
    "load_species_quantities",
    "open_ipic3d",
    "parse_inp",
    "parse_settings_hdf",
    "read_phdf5_particles",
    "to_simulation_config",
]


def _has_h5hut_files(path: Path) -> bool:
    """Check whether the directory contains H5hut field files."""
    return any(path.glob("*-Fields_*.h5"))


def open_ipic3d(
    path: Path,
    *,
    config_path: Path | None = None,
) -> tuple[SimulationReader, SimulationConfig]:
    """Auto-detect iPIC3D format and return the appropriate reader.

    Detection priority:

    1. Parse config from ``.inp`` or ``settings.hdf``.
    2. If ``*-Fields_*.h5`` files exist → `IPic3DH5hutReader`.
    3. If ``WriteMethod == "shdf5"`` → `IPic3DSerialReader`.
    4. If ``WriteMethod == "h5hut"`` → `IPic3DH5hutReader`.
    5. Default → `IPic3DParallelReader`.

    File-based detection (step 2) takes precedence because
    ``WriteMethod`` is often commented out in H5hut runs.

    Parameters
    ----------
    path : Path
        Simulation output directory.
    config_path : Path | None
        Explicit path to an ``.inp`` or ``settings.hdf`` file.
        When ``None``, auto-detected from *path*.

    Returns
    -------
    tuple[SimulationReader, SimulationConfig]
        A (reader, config) pair ready for
        ``reader.read_timestep(path, step)``.

    Raises
    ------
    FileNotFoundError
        If no ``.inp`` or ``settings.hdf`` file is found.
    """
    if config_path is not None:
        suffix = config_path.suffix
        if suffix == ".hdf":
            cfg = parse_settings_hdf(config_path)
        else:
            cfg = parse_inp(config_path)
    elif inp_files := list(path.glob("*.inp")):
        cfg = parse_inp(inp_files[0])
    elif (path / "settings.hdf").exists():
        cfg = parse_settings_hdf(path / "settings.hdf")
    else:
        msg = f"No .inp or settings.hdf found in {path}"
        raise FileNotFoundError(msg)

    sim_config = to_simulation_config(cfg, path)

    reader: SimulationReader
    if _has_h5hut_files(path):
        reader = IPic3DH5hutReader(cfg, sim_config)
    else:
        match cfg.write_method:
            case "shdf5":
                reader = IPic3DSerialReader(cfg, sim_config)
            case "h5hut":
                reader = IPic3DH5hutReader(cfg, sim_config)
            case _:
                reader = IPic3DParallelReader(cfg, sim_config)

    return reader, sim_config
