"""OpenGGCM-UCLA MHD .3df reader.

Reads ``.3df`` field output files and ``grid.*.dat`` grid definitions
from the OpenGGCM global MHD model.  The ``.3df`` format uses WRN2
lossy compression (~12.5-bit precision via logarithmic quantization +
run-length encoding).

Quick start::

    from pypic.readers.openggcm import open_openggcm
    from pathlib import Path

    reader, cfg = open_openggcm(Path("examples/uclamhd-example-3D"))
    ds = reader.read_timestep(Path("examples/uclamhd-example-3D"), 6300)
    print(ds.field_names)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pypic.readers.base import SimulationConfig
from pypic.readers.openggcm._grid import OpenGGCMGrid, parse_grid_file
from pypic.readers.openggcm._probe import probe
from pypic.readers.openggcm._reader import OpenGGCMReader, _make_grid_info
from pypic.units import Normalization

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "OpenGGCMGrid",
    "OpenGGCMReader",
    "open_openggcm",
    "parse_grid_file",
    "probe",
]

_3DF_PATTERN = re.compile(r"^(.+)\.3df\.\d+$")


def open_openggcm(
    path: Path,
    normalization: Normalization | None = None,
    *,
    config_path: Path | None = None,
) -> tuple[OpenGGCMReader, SimulationConfig]:
    """Auto-detect OpenGGCM files and return a reader + config.

    Looks for ``grid.*.dat`` and ``*.3df.*`` files under *path*.

    Parameters
    ----------
    path : Path
        Directory containing OpenGGCM output files.
    normalization : Normalization | None
        If provided, data is normalized from SI to code units.
    config_path : Path | None
        Explicit path to a ``grid.*.dat`` file.  When ``None``,
        auto-detected from *path*.

    Returns
    -------
    reader : OpenGGCMReader
        Configured reader instance.
    config : SimulationConfig
        Simulation metadata.
    """
    # Find grid file
    if config_path is not None:
        grid_file = config_path
    else:
        grid_files = list(path.glob("grid.*.dat"))
        if not grid_files:
            msg = f"No grid.*.dat file found in {path}"
            raise FileNotFoundError(msg)
        grid_file = grid_files[0]
    grid = parse_grid_file(grid_file)
    log.info(
        "Grid: %d x %d x %d, x=[%.1f, %.1f] R_E",
        grid.nx,
        grid.ny,
        grid.nz,
        grid.x[0],
        grid.x[-1],
    )

    # Detect prefix from .3df files
    prefix = _detect_prefix(path)

    reader = OpenGGCMReader(grid, prefix, normalization)

    grid_info = _make_grid_info(grid)
    config = SimulationConfig(
        model_name="OpenGGCM",
        model_type="MHD",
        grid=grid_info,
        normalization=normalization or Normalization.identity(),
        species=(),
        physics={"gamma": 5.0 / 3.0},
        frame="GSM",
        metadata={
            "grid_file": grid_file.name,
            "prefix": prefix,
            **dict(grid.metadata),
        },
    )

    return reader, config


def _detect_prefix(path: Path) -> str:
    """Extract the filename prefix from .3df files."""
    for entry in path.iterdir():
        m = _3DF_PATTERN.match(entry.name)
        if m:
            return m.group(1)
    msg = f"No *.3df.* files found in {path}"
    raise FileNotFoundError(msg)


# Self-register with the reader registry
from pypic.readers._registry import register_reader as _register_reader  # noqa: E402

_register_reader("openggcm", probe, open_openggcm)
