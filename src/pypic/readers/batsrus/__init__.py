"""BATSRUS MHD simulation reader.

Supports three output formats:

- **Per-cell IDL** (``.h`` + ``*_pe*.idl``): raw per-processor binary
- **Merged IDL** (``.out`` / ``.outs``): postprocessed snapshot files
- **HDF5 BATL** (``.batl``): block-structured HDF5 from BATL library

Auto-detection via `open_batsrus` examines directory contents to select
the appropriate reader.
"""

from __future__ import annotations

import copy
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pypic.readers.batsrus._config import (
    BATSRUSConfig,
    extract_step_from_filename,
    parse_param_in,
    to_simulation_config,
)
from pypic.readers.batsrus._grid import assemble_uniform_hdf5, regrid_amr_hdf5
from pypic.readers.batsrus._hdf5 import read_batl
from pypic.readers.batsrus._header import BATSRUSHeader, parse_header
from pypic.readers.batsrus._probe import probe
from pypic.readers.batsrus._reader import BATSRUSReader

if TYPE_CHECKING:
    from pypic.readers.base import SimulationConfig, SimulationReader


class BATSRUSOutputFormat(StrEnum):
    """BATSRUS output file format."""

    HDF5 = "hdf5"
    IDL = "idl"
    OUT = "out"


__all__ = [
    "BATSRUSConfig",
    "BATSRUSHeader",
    "BATSRUSOutputFormat",
    "BATSRUSReader",
    "extract_step_from_filename",
    "open_batsrus",
    "parse_header",
    "parse_param_in",
    "probe",
    "to_simulation_config",
]


def open_batsrus(
    path: Path,
    *,
    config_path: Path | None = None,
) -> tuple[SimulationReader, SimulationConfig]:
    """Auto-detect BATSRUS output format and return a reader.

    Detection order:
    1. ``.batl`` files → HDF5 BATL format
    2. ``.h`` + ``*_pe*.idl`` files → per-cell IDL binary
    3. ``.out`` / ``.outs`` files → merged IDL

    Prefers 3D data over 2D slices when both are available.

    Parameters
    ----------
    path
        Directory containing BATSRUS output files.
    config_path : Path | None
        Explicit path to a ``PARAM.in`` file.  When ``None``,
        auto-detected from *path*.

    Returns
    -------
    reader : SimulationReader
        A `BATSRUSReader` instance.
    config : SimulationConfig
        Simulation configuration parsed from ``PARAM.in``
        and/or headers.

    Raises
    ------
    FileNotFoundError
        If no recognizable BATSRUS output is found.
    """
    path = Path(path)
    param_file = config_path or (path / "PARAM.in")
    batsrus_config = (
        parse_param_in(param_file) if param_file.exists() else BATSRUSConfig()
    )

    # Detect format and find the best prefix
    batl_files = sorted(path.glob("*.batl"))
    h_files = sorted(path.glob("*.h"))
    out_files = sorted(path.glob("*.out")) + sorted(path.glob("*.outs"))

    if batl_files:
        output_format = "hdf5"
        prefix = _detect_prefix_batl(batl_files)
    elif h_files:
        output_format = "idl"
        prefix = _detect_prefix_h(h_files)
    elif out_files:
        output_format = "out"
        prefix = _detect_prefix_out(out_files)
    else:
        msg = f"No BATSRUS output files found in {path}"
        raise FileNotFoundError(msg)

    # Extract geometry from header if available
    geometry = "cartesian"
    header = None
    grid = None
    if h_files:
        header = parse_header(h_files[0])
        geometry = header.geometry

    batsrus_config = copy.replace(batsrus_config, geometry=geometry)

    reader = BATSRUSReader(batsrus_config, output_format, prefix, geometry=geometry)

    if batl_files and header is None:
        batl = read_batl(batl_files[0])
        is_uniform = len(set(batl.refine_level)) <= 1
        if is_uniform:
            _, grid = assemble_uniform_hdf5(batl)
        else:
            _, grid = regrid_amr_hdf5(batl)

    sim_config = to_simulation_config(batsrus_config, header, grid=grid)
    return reader, sim_config


def _detect_prefix_batl(files: list[Path]) -> str:
    """Extract the plot file prefix from .batl filenames.

    Prefers 3D files (``3d__``) over slices (``z=0_``).
    """
    # Prefer 3d files
    for f in files:
        if f.name.startswith("3d"):
            return _prefix_before_step(f.name, ".batl")
    return _prefix_before_step(files[0].name, ".batl")


def _detect_prefix_h(files: list[Path]) -> str:
    """Extract prefix from .h header filenames."""
    return _prefix_before_step(files[0].name, ".h")


def _detect_prefix_out(files: list[Path]) -> str:
    """Extract prefix from .out filenames."""
    return _prefix_before_step(files[0].name, ".out")


def _prefix_before_step(filename: str, suffix: str) -> str:
    """Extract the part of the filename before the timestep/step pattern.

    Handles both ``prefix_n{step}`` and ``prefix_t{time}_n{step}`` patterns.
    Returns the prefix up to (but not including) the first ``_t`` or ``_n``
    timestamp marker.
    """
    stem = filename
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    # Find the earliest _t{8digits} or _n{8digits} pattern
    m = re.search(r"_[tn]\d{8}", stem)
    if m:
        return stem[: m.start() + 1]  # include trailing underscore
    return stem


# Self-register with the reader registry
from pypic.readers._registry import register_reader as _register_reader  # noqa: E402

_register_reader("batsrus", probe, open_batsrus)
