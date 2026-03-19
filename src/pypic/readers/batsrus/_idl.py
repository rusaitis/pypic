"""Read BATSRUS IDL output files (per-cell binary and merged .out)."""

from __future__ import annotations

import struct
from typing import IO, TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.readers.batsrus._header import BATSRUSHeader
    from pypic.types import FloatArray


def read_idl_cells(
    path: Path,
    header: BATSRUSHeader,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read per-cell IDL binary records.

    Each Fortran record contains ``(dx, x, y, z, var1..varN)`` as floats
    with 4-byte length markers.

    Parameters
    ----------
    path
        Path to the ``.idl`` file.
    header
        Parsed ``.h`` header providing precision and variable count.

    Returns
    -------
    coords : ndarray, shape (n_cells, 3)
        Cell center coordinates (x, y, z).
    dx : ndarray, shape (n_cells,)
        Cell size per cell (uniform grids: constant; AMR: varying).
    state : ndarray, shape (n_cells, n_var)
        State variables in BATSRUS order.
    """
    dtype = np.float64 if header.n_byte_real == 8 else np.float32
    n_real = header.n_byte_real
    n_var = header.n_plot_var
    values_per_record = 1 + 3 + n_var  # dx + 3 coords + n_var
    record_data_bytes = values_per_record * n_real
    record_total_bytes = record_data_bytes + 8  # two 4-byte Fortran markers

    raw = path.read_bytes()
    n_cells = len(raw) // record_total_bytes

    if n_cells != header.n_cells:
        msg = (
            f"Expected {header.n_cells} cells from header, "
            f"but file contains {n_cells} records"
        )
        raise ValueError(msg)

    # Parse all records at once using stride tricks
    all_values = np.empty((n_cells, values_per_record), dtype=np.float64)
    offset = 0
    for i in range(n_cells):
        offset += 4  # skip leading marker
        vals = np.frombuffer(raw, dtype=dtype, count=values_per_record, offset=offset)
        all_values[i] = vals
        offset += record_data_bytes + 4  # data + trailing marker

    dx = all_values[:, 0]
    coords = all_values[:, 1:4]
    state = all_values[:, 4:]

    return coords, dx, state


def read_out_file(
    path: Path,
    *,
    skip: int = 0,
) -> tuple[FloatArray, FloatArray, tuple[str, ...], dict[str, Any]]:
    """Read a merged BATSRUS ``.out`` or ``.outs`` snapshot file.

    Adapted from the official ``batsrus_read_datafile.py`` reader.
    Supports ASCII, real4, and real8 binary formats with auto-detection.

    Parameters
    ----------
    path
        Path to the ``.out`` or ``.outs`` file.
    skip
        Number of snapshots to skip (for multi-snapshot ``.outs`` files).

    Returns
    -------
    coords : ndarray, shape (ndim, *dims)
        Coordinate arrays.
    state : ndarray, shape (nvar, *dims)
        State variable arrays.
    var_names : tuple[str, ...]
        Variable names extracted from the file.
    metadata : dict
        Additional metadata (step, time, ndim, dims, pars, param_names).
    """
    fmt = _detect_out_format(path)

    if fmt == "ascii":
        return _read_out_ascii(path, skip=skip)
    return _read_out_binary(path, skip=skip)


def _detect_out_format(path: Path) -> str:
    """Detect merged .out file format: 'ascii', 'real4', or 'real8'."""
    with path.open("rb") as f:
        raw = f.read(4)
        if len(raw) < 4:
            return "ascii"
        reclen = struct.unpack("<i", raw)[0]
        if reclen not in (79, 500):
            return "ascii"
        f.read(reclen + 4)  # skip header record + trailing marker
        raw2 = f.read(4)
        if len(raw2) < 4:
            return "ascii"
        reclen2 = struct.unpack("<i", raw2)[0]
        if reclen2 == 20:
            return "real4"
        if reclen2 == 24:
            return "real8"
        return "ascii"


def _read_out_ascii(
    path: Path, *, skip: int = 0
) -> tuple[FloatArray, FloatArray, tuple[str, ...], dict[str, Any]]:
    with path.open("r") as f:
        for _ in range(skip):
            _skip_ascii_snapshot(f)

        head = f.readline().rstrip()
        parts = f.readline().split()
        step_val = int(parts[0])
        time_val = float(parts[1])
        ndim_raw = int(parts[2])
        is_cart = ndim_raw > 0
        ndim = abs(ndim_raw)
        npar = int(parts[3])
        nvar = int(parts[4])

        dims = np.flip(np.array(f.readline().split(), dtype=np.int32))
        pars = (
            np.array(f.readline().split(), dtype=np.float64)
            if npar > 0
            else np.array([])
        )
        name_line = f.readline().rstrip()

        ngrid = int(np.prod(dims))
        lines_data = "".join(f.readline() for _ in range(ngrid))
        griddata = np.array(lines_data.split(), dtype=np.float64)
        griddata = griddata.reshape(ngrid, ndim + nvar).T

        coord = griddata[:ndim].reshape([ndim, *list(dims)])
        state = griddata[ndim:].reshape([nvar, *list(dims)])

    all_names = name_line.split()
    var_names = tuple(all_names[ndim : ndim + nvar])
    param_names = tuple(all_names[ndim + nvar :])

    metadata: dict[str, Any] = {
        "head": head,
        "step": step_val,
        "time": time_val,
        "ndim": ndim,
        "dims": tuple(int(d) for d in dims),
        "is_cartesian": is_cart,
    }
    if npar > 0:
        metadata["pars"] = pars
        metadata["param_names"] = param_names

    return coord, state, var_names, metadata


def _read_out_binary(
    path: Path, *, skip: int = 0
) -> tuple[FloatArray, FloatArray, tuple[str, ...], dict[str, Any]]:
    with path.open("rb") as f:
        for _ in range(skip):
            _skip_binary_snapshot(f)

        string_length = struct.unpack("<i", f.read(4))[0]
        head = f.read(string_length).decode().rstrip()
        f.read(4)  # trailing marker

        len2 = struct.unpack("<i", f.read(4))[0]
        if len2 == 20:
            nreal = 4
            dtype: type[np.floating[Any]] = np.float32
        else:
            nreal = 8
            dtype = np.float64

        step_val = struct.unpack("<i", f.read(4))[0]
        time_val = float(np.frombuffer(f.read(nreal), dtype=dtype)[0])
        ndim_raw = np.frombuffer(f.read(4), dtype=np.int32)[0]
        is_cart = ndim_raw > 0
        ndim = abs(int(ndim_raw))
        npar = struct.unpack("<i", f.read(4))[0]
        nvar = struct.unpack("<i", f.read(4))[0]
        f.read(8)  # markers

        dims = np.flip(np.frombuffer(f.read(4 * ndim), dtype=np.int32))
        f.read(8)  # markers

        pars = np.array([])
        if npar > 0:
            pars = np.frombuffer(f.read(nreal * npar), dtype=dtype).astype(np.float64)
            f.read(8)

        name_line = f.read(string_length).decode().rstrip()
        f.read(8)  # markers

        ngrid = int(np.prod(dims))
        coord = (
            np.frombuffer(f.read(nreal * ngrid * ndim), dtype=dtype)
            .reshape([ndim, *list(dims)])
            .astype(np.float64)
        )

        state = np.empty((nvar, ngrid), dtype=np.float64)
        for iv in range(nvar):
            f.read(8)  # markers
            state[iv] = np.frombuffer(f.read(nreal * ngrid), dtype=dtype).astype(
                np.float64
            )
        f.read(4)  # trailing marker

        state = state.reshape([nvar, *list(dims)])

    all_names = name_line.split()
    var_names = tuple(all_names[ndim : ndim + nvar])
    param_names = tuple(all_names[ndim + nvar :])

    metadata: dict[str, Any] = {
        "head": head,
        "step": step_val,
        "time": time_val,
        "ndim": ndim,
        "dims": tuple(int(d) for d in dims),
        "is_cartesian": is_cart,
    }
    if npar > 0:
        metadata["pars"] = pars
        metadata["param_names"] = param_names

    return coord, state, var_names, metadata


def _skip_ascii_snapshot(f: IO[str]) -> None:
    """Skip one ASCII snapshot in a multi-snapshot file."""
    f.readline()  # head
    parts = f.readline().split()
    npar = int(parts[3])
    dims = np.array(f.readline().split(), dtype=np.int32)
    if npar > 0:
        f.readline()
    f.readline()  # names
    ngrid = int(np.prod(dims))
    for _ in range(ngrid):
        f.readline()


def _skip_binary_snapshot(f: IO[bytes]) -> None:
    """Skip one binary snapshot in a multi-snapshot file."""
    string_length = struct.unpack("<i", f.read(4))[0]
    f.read(string_length + 4)

    len2 = struct.unpack("<i", f.read(4))[0]
    nreal = 4 if len2 == 20 else 8
    f.read(4)  # step
    f.read(nreal)  # time
    ndim = abs(np.frombuffer(f.read(4), dtype=np.int32)[0])
    npar = struct.unpack("<i", f.read(4))[0]
    nvar = struct.unpack("<i", f.read(4))[0]
    f.read(8)
    dims = np.frombuffer(f.read(4 * ndim), dtype=np.int32)
    f.read(8)
    if npar > 0:
        f.read(nreal * npar + 8)
    f.read(string_length + 8)
    ngrid = int(np.prod(dims))
    f.read(nreal * ngrid * ndim)
    for _ in range(nvar):
        f.read(nreal * ngrid + 8)
    f.read(4)
