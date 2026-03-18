"""OpenGGCM grid file parser (``grid.*.dat``)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import FloatArray

# Names of the 21 grid sections in the order they appear in the file.
# First three are the primary coordinate arrays; the remaining 18 are
# staggered grids for the field components.


@dataclass(frozen=True, slots=True)
class OpenGGCMGrid:
    """Non-uniform grid definition from an OpenGGCM grid file.

    Parameters
    ----------
    nx, ny, nz : int
        Number of grid points along each axis.
    x : FloatArray
        X-coordinates (non-uniform), shape ``(nx,)``, in $R_E$.
    y : FloatArray
        Y-coordinates (non-uniform), shape ``(ny,)``, in $R_E$.
    z : FloatArray
        Z-coordinates (non-uniform), shape ``(nz,)``, in $R_E$.
    stagger : MappingProxyType[str, tuple[FloatArray, FloatArray, FloatArray]]
        Staggered grid positions keyed by field component (``"bx"``,
        ``"by"``, ``"bz"``, ``"ex"``, ``"ey"``, ``"ez"``).  Each value
        is ``(gx, gy, gz)`` for that component.
    metadata : MappingProxyType[str, str]
        Header metadata (``DIPOLETIME``, ``BASETIME``).
    """

    nx: int
    ny: int
    nz: int
    x: FloatArray
    y: FloatArray
    z: FloatArray
    stagger: MappingProxyType[str, tuple[FloatArray, FloatArray, FloatArray]]
    metadata: MappingProxyType[str, str]


def _parse_field_1d(lines: list[str], start: int) -> tuple[str, int, FloatArray, int]:
    """Parse one FIELD-1D-1 section starting at *start*.

    Returns ``(name, nx, values, next_line_index)``.
    """
    # lines[start] == "FIELD-1D-1"
    name = lines[start + 1].strip()
    # start+2 = description (skip)
    meta = lines[start + 3].strip().split()
    nx = int(meta[1])
    # start+4 = WRN2 header line (skip — grid file stores ASCII values)
    values = np.empty(nx, dtype=np.float64)
    data_start = start + 5
    for i in range(nx):
        values[i] = float(lines[data_start + i])
    next_idx = data_start + nx
    return name, nx, values, next_idx


def parse_grid_file(path: Path) -> OpenGGCMGrid:
    """Parse an OpenGGCM ASCII grid file.

    The file contains header metadata followed by 21 ``FIELD-1D-1``
    sections: primary grids (``gridx``, ``gridy``, ``gridz``) then 18
    staggered grids for the six field components (B and E, three
    directions each).

    Parameters
    ----------
    path : Path
        Path to the ``grid.*.dat`` file.

    Returns
    -------
    OpenGGCMGrid
    """
    text = path.read_text(encoding="ascii")
    lines = text.splitlines()

    # Parse header metadata
    meta: dict[str, str] = {}
    idx = 0
    while idx < len(lines) and "FIELD-1D-1" not in lines[idx]:
        line = lines[idx].strip()
        if line.startswith("DIPOLETIME:"):
            meta["DIPOLETIME"] = line.split(":", 1)[1]
        elif line.startswith("BASETIME:"):
            meta["BASETIME"] = line.split(":", 1)[1]
        idx += 1

    # Parse all FIELD-1D-1 sections
    grids: dict[str, FloatArray] = {}
    while idx < len(lines):
        if "FIELD-1D-1" in lines[idx]:
            name, _, values, idx = _parse_field_1d(lines, idx)
            grids[name] = values
        else:
            idx += 1

    # Extract primary grids
    x = grids["gridx"]
    y = grids["gridy"]
    z = grids["gridz"]

    # Build staggered grid mapping
    stagger: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {}
    for component in ("bx", "by", "bz", "ex", "ey", "ez"):
        gx_key = f"gx-{component}"
        gy_key = f"gy-{component}"
        gz_key = f"gz-{component}"
        if gx_key in grids and gy_key in grids and gz_key in grids:
            stagger[component] = (grids[gx_key], grids[gy_key], grids[gz_key])

    return OpenGGCMGrid(
        nx=len(x),
        ny=len(y),
        nz=len(z),
        x=x,
        y=y,
        z=z,
        stagger=MappingProxyType(stagger),
        metadata=MappingProxyType(meta),
    )
