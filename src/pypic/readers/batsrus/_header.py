"""Parse BATSRUS ``.h`` header files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class BATSRUSHeader:
    """Parsed content of a BATSRUS ``.h`` output header file.

    These header files accompany per-processor ``.idl`` data files and
    contain all metadata needed to interpret the binary records.
    """

    ndim: int
    block_size: tuple[int, ...]
    n_root_blocks: tuple[int, ...]
    geometry: str
    domain_min: tuple[float, ...]
    domain_max: tuple[float, ...]
    is_periodic: tuple[bool, ...]
    n_step: int
    time: float
    n_cells: int
    cell_size_min: tuple[float, ...]
    n_param: int
    param_names: tuple[str, ...]
    param_values: tuple[float, ...]
    n_plot_var: int
    var_names: tuple[str, ...]
    unit_string: str
    output_format: str
    is_binary: bool
    n_byte_real: int


def parse_header(path: Path) -> BATSRUSHeader:
    """Parse a BATSRUS ``.h`` header file into a `BATSRUSHeader`.

    Parameters
    ----------
    path
        Path to the ``.h`` file.

    Returns
    -------
    BATSRUSHeader
        Frozen dataclass with all extracted metadata.
    """
    text = path.read_text()
    lines = text.splitlines()

    # Accumulate values by section
    section: str | None = None
    section_lines: list[str] = []
    sections: dict[str, list[str]] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if section is not None:
                sections[section] = section_lines
            section = stripped
            section_lines = []
        elif stripped:
            section_lines.append(stripped)
    if section is not None:
        sections[section] = section_lines

    # Every field below has a default, so without this guard any text file
    # yields a plausible-looking header (ndim=3, n_step=0, var_names=())
    # instead of an error. #HEADFILE and #NDIM are written by every BATSRUS
    # plot header.
    if not sections.keys() & {"#HEADFILE", "#NDIM"}:
        msg = (
            f"{path} is not a BATSRUS .h header: no #HEADFILE or #NDIM "
            f"section found (got {sorted(sections) or 'no sections'})"
        )
        raise ValueError(msg)

    head_lines = sections.get("#HEADFILE", [])
    is_binary = True
    n_byte_real = 8
    for ln in head_lines:
        val = ln.split()[0]
        if "IsBinary" in ln:
            is_binary = val == "T"
        elif "nByteReal" in ln:
            n_byte_real = int(val)

    ndim_lines = sections.get("#NDIM", [])
    ndim = int(ndim_lines[0].split()[0]) if ndim_lines else 3

    block_lines = sections.get("#GRIDBLOCKSIZE", [])
    block_size = tuple(int(ln.split()[0]) for ln in block_lines)

    root_lines = sections.get("#ROOTBLOCK", [])
    n_root_blocks = tuple(int(ln.split()[0]) for ln in root_lines)

    geo_lines = sections.get("#GRIDGEOMETRYLIMIT", [])
    geometry = "cartesian"
    domain_min: list[float] = []
    domain_max: list[float] = []
    if geo_lines:
        geometry = geo_lines[0].split()[0].lower()
        for i in range(ndim):
            domain_min.append(float(geo_lines[1 + 2 * i].split()[0]))
            domain_max.append(float(geo_lines[2 + 2 * i].split()[0]))

    periodic_lines = sections.get("#PERIODIC", [])
    is_periodic = tuple(ln.split()[0] == "T" for ln in periodic_lines)

    nstep_lines = sections.get("#NSTEP", [])
    n_step = int(nstep_lines[0].split()[0]) if nstep_lines else 0

    time_lines = sections.get("#TIMESIMULATION", [])
    time = float(time_lines[0].split()[0]) if time_lines else 0.0

    ncell_lines = sections.get("#NCELL", [])
    n_cells = int(ncell_lines[0].split()[0]) if ncell_lines else 0

    cellsize_lines = sections.get("#CELLSIZE", [])
    cell_size_min = tuple(float(ln.split()[0]) for ln in cellsize_lines)

    param_lines = sections.get("#SCALARPARAM", [])
    n_param = 0
    param_values: list[float] = []
    param_names: list[str] = []
    if param_lines:
        n_param = int(param_lines[0].split()[0])
        for ln in param_lines[1:]:
            parts = ln.split()
            param_values.append(float(parts[0]))
            if len(parts) > 1:
                param_names.append(parts[1])

    plotvar_lines = sections.get("#PLOTVARIABLE", [])
    n_plot_var = 0
    var_names_list: list[str] = []
    unit_string = ""
    if plotvar_lines:
        n_plot_var = int(plotvar_lines[0].split()[0])
        if len(plotvar_lines) > 1:
            all_names = plotvar_lines[1].split()
            var_names_list = all_names[:n_plot_var]
        if len(plotvar_lines) > 2:
            unit_string = plotvar_lines[2]

    fmt_lines = sections.get("#OUTPUTFORMAT", [])
    output_format = fmt_lines[0].split()[0] if fmt_lines else "binary"

    return BATSRUSHeader(
        ndim=ndim,
        block_size=block_size,
        n_root_blocks=n_root_blocks,
        geometry=geometry,
        domain_min=tuple(domain_min),
        domain_max=tuple(domain_max),
        is_periodic=is_periodic,
        n_step=n_step,
        time=time,
        n_cells=n_cells,
        cell_size_min=cell_size_min,
        n_param=n_param,
        param_names=tuple(param_names),
        param_values=tuple(param_values),
        n_plot_var=n_plot_var,
        var_names=tuple(var_names_list),
        unit_string=unit_string,
        output_format=output_format,
        is_binary=is_binary,
        n_byte_real=n_byte_real,
    )
