"""OpenGGCM .3df field reader: header parsing + WRN2 data decoding.

Reads the entire file in a single syscall, then scans for field markers
and decodes WRN2 data in bulk using vectorized NumPy operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pypic.readers.openggcm._wrn2 import decompress_field_vectorized

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pypic.types import FloatArray

log = logging.getLogger(__name__)

_MARKER = b"FIELD-3D-1"
_WRN2 = b"WRN2"


def _parse_wrn2_header(line: bytes) -> tuple[int, float, float]:
    """Parse a WRN2 header line.

    Format (Fortran): ``a4, i8, 3e14.7, i8, a8``
    ``WRN298963200-0.2968665E+02 0.7313355E+01 0.3760000E+03    6300FUNC-3-1``

    Returns ``(count, zmin, zmax)``.
    """
    # line[0:4] is the "WRN2" tag, already matched by the caller.
    n = int(line[4:12])
    zmin = float(line[12:26])
    zmax = float(line[26:40])
    return n, zmin, zmax


def _skip_n_lines(
    newline_positions: np.ndarray, start: int, n: int, data_len: int
) -> int:
    """Advance past *n* newlines using pre-computed newline positions."""
    if n == 0:
        return start
    first = int(np.searchsorted(newline_positions, start, side="left"))
    target = first + n - 1
    if target >= len(newline_positions):
        return data_len
    return int(newline_positions[target]) + 1


def _read_line(data: bytes, start: int) -> tuple[bytes, int]:
    """Extract a line from the buffer, return (line_content, next_offset)."""
    end = data.find(b"\n", start)
    if end == -1:
        return data[start:], len(data)
    return data[start:end], end + 1


@dataclass(frozen=True, slots=True)
class _Record:
    """One ``FIELD-3D-1`` record: its header and the span of its WRN2 payload."""

    name: str
    timestep: int
    shape: tuple[int, int, int]
    count: int
    zmin: float
    zmax: float
    data_start: int
    data_end: int


def _scan_records(data: bytes) -> Iterator[_Record]:
    """Walk the ``FIELD-3D-1`` records of a ``.3df`` buffer without decoding."""
    newline_positions = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == 10)
    pos = 0
    while (idx := data.find(_MARKER, pos)) != -1:
        _, pos = _read_line(data, idx)
        name_bytes, pos = _read_line(data, pos)
        name = name_bytes.rstrip(b"\r").decode("ascii").strip()
        # Description / time info
        _, pos = _read_line(data, pos)
        dim_bytes, pos = _read_line(data, pos)
        timestep, nx, ny, nz = (int(part) for part in dim_bytes.split()[:4])

        wrn2_bytes, pos = _read_line(data, pos)
        wrn2_line = wrn2_bytes.rstrip(b"\r")
        if not wrn2_line.startswith(_WRN2):
            log.warning("Expected WRN2 header for field %s, skipping", name)
            continue
        count, zmin, zmax = _parse_wrn2_header(wrn2_line)
        if count != nx * ny * nz:
            msg = f"Field {name}: WRN2 count {count} != nx*ny*nz = {nx * ny * nz}"
            raise ValueError(msg)

        # Exact data region: n_chunks * 2 lines; a constant field has none.
        n_data_lines = 0 if zmin == zmax else ((count + 63) // 64) * 2
        data_end = _skip_n_lines(newline_positions, pos, n_data_lines, len(data))
        yield _Record(name, timestep, (nx, ny, nz), count, zmin, zmax, pos, data_end)
        pos = data_end


def read_3df_field_names(path: Path) -> list[str]:
    """Field names recorded in a ``.3df`` file, in file order, without decoding."""
    return [record.name for record in _scan_records(path.read_bytes())]


def read_3df_file(
    path: Path,
    *,
    skip: set[str] | None = None,
) -> tuple[dict[str, FloatArray], int, int, int, int]:
    r"""Read all 3D fields from an OpenGGCM ``.3df`` file.

    Reads the entire file into memory in one syscall, then scans for
    ``FIELD-3D-1`` markers.  Skipped fields advance past the exact
    number of WRN2 data lines.  Decoded fields use vectorized NumPy
    operations on the raw byte buffer.

    Parameters
    ----------
    path : Path
        Path to the ``.3df`` file.
    skip : set[str] | None
        Field names to skip (e.g. ``{"eflx", "efly", "eflz"}``).

    Returns
    -------
    fields : dict[str, FloatArray]
        Field arrays keyed by OpenGGCM name, shaped ``(nx, ny, nz)``.
    timestep : int
        Timestep index from the file header.
    nx, ny, nz : int
        Grid dimensions.
    """
    skip = skip or set()
    fields: dict[str, FloatArray] = {}
    timestep = 0
    shape = (0, 0, 0)

    data = path.read_bytes()
    for record in _scan_records(data):
        timestep, shape = record.timestep, record.shape
        if record.name in skip:
            log.debug("Skipped field %s", record.name)
            continue
        log.info("Reading field %s (%d values)", record.name, record.count)
        flat = decompress_field_vectorized(
            data,
            record.data_start,
            record.data_end - record.data_start,
            record.count,
            record.zmin,
            record.zmax,
        )
        # Fortran column-major: reshape with order='F'
        fields[record.name] = flat.reshape(shape, order="F")

    return fields, timestep, *shape
