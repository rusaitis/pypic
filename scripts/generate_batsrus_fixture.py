#!/usr/bin/env python3
"""Generate synthetic BATSRUS test fixtures.

Creates small self-contained datasets in each supported format:

- **idl-uniform**: 16x16 2D grid, per-cell IDL binary, code units
- **hdf5-uniform**: same physics, 4 blocks of 8x8, HDF5 BATL format
- **hdf5-amr**: 2 refinement levels (2 coarse + 8 fine blocks), HDF5 BATL
- **out-ascii** / **out-binary**: merged .out snapshots, same physics as
  idl-uniform so the two read paths can be compared directly

Physics: Harris current sheet ``Bx = B0 * tanh(y / delta)`` with
uniform density and pressure. No unit conversion (code units).
"""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "batsrus-synthetic"
)

# Physics parameters
B0 = 1.0
DELTA = 2.0  # current sheet half-width
RHO0 = 1.0
P0 = 0.1
GAMMA = 5.0 / 3.0

# Grid
NX, NY = 16, 16
XMIN, XMAX = -8.0, 8.0
YMIN, YMAX = -8.0, 8.0
DX = (XMAX - XMIN) / NX
DY = (YMAX - YMIN) / NY

VAR_NAMES = ("Rho", "Ux", "Uy", "Uz", "Bx", "By", "Bz", "Hyp", "P", "jx", "jy", "jz")
N_VAR = len(VAR_NAMES)


def harris_fields(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Harris current sheet fields at cell centers."""
    bx = B0 * np.tanh(y / DELTA)
    # jz = dBx/dy = B0 / delta * sech²(y/delta)
    jz = B0 / DELTA / np.cosh(y / DELTA) ** 2
    return {
        "Rho": np.full_like(x, RHO0),
        "Ux": np.zeros_like(x),
        "Uy": np.zeros_like(x),
        "Uz": np.zeros_like(x),
        "Bx": bx,
        "By": np.zeros_like(x),
        "Bz": np.zeros_like(x),
        "Hyp": np.zeros_like(x),
        "P": np.full_like(x, P0),
        "jx": np.zeros_like(x),
        "jy": np.zeros_like(x),
        "jz": jz,
    }


def write_header(
    path: Path, *, n_cells: int, dx_min: float, output_format: str = "binary"
) -> None:
    """Write a minimal .h header file."""
    text = f"""#HEADFILE
synthetic_batsrus_fixture.h
       1                nProc
       T                IsBinary
       8                nByteReal

#NDIM
       2                nDim

#GRIDBLOCKSIZE
       8                BlockSize1
       8                BlockSize2

#ROOTBLOCK
       2                nRootBlock1
       2                nRootBlock2

#GRIDGEOMETRYLIMIT
cartesian               TypeGeometry
 {XMIN:14.5E}           XyzMin1
 {XMAX:14.5E}           XyzMax1
 {YMIN:14.5E}           XyzMin2
 {YMAX:14.5E}           XyzMax2

#PERIODIC
       T                IsPeriodic1
       T                IsPeriodic2

#NSTEP
       0                nStep

#TIMESIMULATION
  0.0000000000E+00      TimeSimulation

#NCELL
{n_cells:>10d}              nCellPlot

#CELLSIZE
 {dx_min:14.10E}      CellSizeMin1
 {dx_min:14.10E}      CellSizeMin2

#PLOTRANGE
 {XMIN:14.10E}      CoordMin1
 {XMAX:14.10E}      CoordMax1
 {YMIN:14.10E}      CoordMin2
 {YMAX:14.10E}      CoordMax2

#PLOTRESOLUTION
  0.0000000000E+00      DxSavePlot1
  0.0000000000E+00      DxSavePlot2

#SCALARPARAM
       1                nParam
 {GAMMA:14.5E}           g

#PLOTVARIABLE
{N_VAR:>10d}                nPlotVar
{" ".join(VAR_NAMES)} g
normalized units

#OUTPUTFORMAT
{output_format}
"""
    path.write_text(text)


def write_param_in(path: Path) -> None:
    """Write a minimal PARAM.in file."""
    text = f"""Begin session: 1

#DESCRIPTION
Synthetic Harris current sheet fixture

#PLANET
NONE\t\t\tNamePlanet

#IOUNITS
NONE\t\t\tTypeIoUnit

#NORMALIZATION
NONE\t\t\tTypeNormalization

#GRID
2\t\t\tnRootBlockX
2\t\t\tnRootBlockY
1\t\t\tnRootBlockZ
{XMIN}\t\t\txMin
{XMAX}\t\t\txMax
{YMIN}\t\t\tyMin
{YMAX}\t\t\tyMax
-0.5\t\t\tzMin
 0.5\t\t\tzMax

#OUTERBOUNDARY
periodic\t\tTypeBc1
periodic\t\tTypeBc2
periodic\t\tTypeBc3
periodic\t\tTypeBc4

#SCHEME
2\t\t\tnOrder
Rusanov\t\t\tTypeFlux
minmod\t\t\tTypeLimiter

#STOP
-1\t\t\tMaxIteration
1.0\t\t\ttSimulationMax

#END
"""
    path.write_text(text)


def generate_idl_uniform() -> None:
    """Write IDL per-cell binary fixture (uniform 16x16)."""
    out_dir = OUTPUT_DIR / "idl-uniform"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    # Cell centers
    x_centers = XMIN + (np.arange(NX) + 0.5) * DX
    y_centers = YMIN + (np.arange(NY) + 0.5) * DY
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    fields = harris_fields(xx, yy)

    # Write .h header
    write_header(
        out_dir / "z=0_mhd_1_n00000000.h",
        n_cells=NX * NY,
        dx_min=DX,
    )

    # Write per-cell IDL binary: each record = (dx, x, y, z=0, var1..varN) as float64
    idl_path = out_dir / "z=0_mhd_1_n00000000_pe0000.idl"
    with idl_path.open("wb") as f:
        for ix in range(NX):
            for iy in range(NY):
                values = np.array(
                    [DX, xx[ix, iy], yy[ix, iy], 0.0]
                    + [fields[v][ix, iy] for v in VAR_NAMES],
                    dtype=np.float64,
                )
                data = values.tobytes()
                marker = struct.pack("<i", len(data))
                f.write(marker + data + marker)

    print(f"  IDL uniform: {idl_path} ({idl_path.stat().st_size} bytes)")


def _out_snapshot_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinates and state arrays shared by both .out writers."""
    x_centers = XMIN + (np.arange(NX) + 0.5) * DX
    y_centers = YMIN + (np.arange(NY) + 0.5) * DY
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    fields = harris_fields(xx, yy)
    state = np.stack([fields[v] for v in VAR_NAMES])
    return xx, yy, state


def generate_out_ascii() -> None:
    """Write a merged ASCII .out fixture (same physics as idl-uniform)."""
    out_dir = OUTPUT_DIR / "out-ascii"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    xx, yy, state = _out_snapshot_arrays()
    out_path = out_dir / "z=0_mhd_1_n00000000.out"

    lines = [
        "synthetic_batsrus_fixture normalized units",
        f"       0  {0.0:.8E}       2       1{N_VAR:>8d}",
        # dims are written fastest-axis-first; the reader flips them back.
        f"{NY:>8d}{NX:>8d}",
        f" {GAMMA:.8E}",
        f"x y {' '.join(VAR_NAMES)} g",
    ]
    # Row order must match the reader's reshape: C order over (nvar, *dims)
    # with dims = flip(on-disk dims) == (NX, NY).
    for ix in range(NX):
        for iy in range(NY):
            vals = [xx[ix, iy], yy[ix, iy], *[state[v, ix, iy] for v in range(N_VAR)]]
            # Full float64 precision: the fixture exists to prove the ASCII
            # and binary read paths agree, so text rounding must not be the
            # thing the comparison measures.
            lines.append(" ".join(f"{v:.17E}" for v in vals))
    out_path.write_text("\n".join(lines) + "\n")

    print(f"  OUT ascii: {out_path} ({out_path.stat().st_size} bytes)")


def generate_out_binary() -> None:
    """Write a merged real8 binary .out fixture (same physics as out-ascii)."""
    out_dir = OUTPUT_DIR / "out-binary"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    xx, yy, state = _out_snapshot_arrays()
    out_path = out_dir / "z=0_mhd_1_n00000000.out"

    head = "synthetic_batsrus_fixture normalized units".ljust(79)
    names = f"x y {' '.join(VAR_NAMES)} g".ljust(79)

    def record(payload: bytes) -> bytes:
        marker = struct.pack("<i", len(payload))
        return marker + payload + marker

    coord = np.stack([xx, yy]).astype(np.float64)

    with out_path.open("wb") as f:
        f.write(record(head.encode()))
        # reclen 24 selects the real8 branch in _detect_out_format
        f.write(
            record(
                struct.pack("<i", 0)
                + np.float64(0.0).tobytes()
                + np.int32(2).tobytes()
                + struct.pack("<i", 1)
                + struct.pack("<i", N_VAR)
            )
        )
        f.write(record(np.array([NY, NX], dtype=np.int32).tobytes()))
        f.write(record(np.array([GAMMA], dtype=np.float64).tobytes()))
        f.write(record(names.encode()))
        f.write(record(coord.tobytes()))
        for iv in range(N_VAR):
            f.write(record(state[iv].astype(np.float64).tobytes()))

    print(f"  OUT binary: {out_path} ({out_path.stat().st_size} bytes)")


def generate_hdf5_uniform() -> None:
    """Write HDF5 BATL fixture (uniform, 4 blocks of 8x8)."""
    out_dir = OUTPUT_DIR / "hdf5-uniform"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    block_nx, block_ny = 8, 8
    n_blocks_x, n_blocks_y = NX // block_nx, NY // block_ny
    n_blocks = n_blocks_x * n_blocks_y

    with h5py.File(out_dir / "z=0_mhd_1_n00000000.batl", "w") as f:
        # Integer Plot Metadata
        ipm = np.zeros(16, dtype=np.int32)
        ipm[0] = 1  # n_step?
        ipm[2] = 2  # ndim
        ipm[3] = 2  # ndim repeated
        ipm[4] = n_blocks
        ipm[5] = 2  # ?
        ipm[6] = 1  # ?
        ipm[7] = block_nx
        ipm[8] = block_ny
        ipm[15] = N_VAR
        f.create_dataset("Integer Plot Metadata", data=ipm)

        # Real Plot Metadata: [time, xmin, xmax, ymin, ymax, ...]
        rpm = np.zeros(7, dtype=np.float64)
        rpm[0] = 0.0  # time
        rpm[1] = XMIN
        rpm[2] = XMAX
        rpm[3] = YMIN
        rpm[4] = YMAX
        f.create_dataset("Real Plot Metadata", data=rpm)

        # Integer/Real Sim Metadata (minimal)
        f.create_dataset("Integer Sim Metadata", data=np.zeros(8, dtype=np.int32))
        f.create_dataset("Real Simulation Metadata", data=rpm.copy())

        # Variable and unit names
        var_bytes = np.array([v.ljust(11).encode() for v in VAR_NAMES], dtype="|S11")
        f.create_dataset("NamePlotVar_V", data=var_bytes)
        unit_bytes = np.array(["normalized".ljust(11).encode()] * N_VAR, dtype="|S11")
        f.create_dataset("NamePlotUnit_V", data=unit_bytes)

        # Axis Labels
        labels = np.array([b"x          ", b"y          "], dtype="|S11")
        f.create_dataset("Axis Labels", data=labels)

        # Bounding boxes, coordinates, refine level
        bbox = np.empty((n_blocks, 2, 2), dtype=np.float64)
        coord_db = np.empty((n_blocks, 2), dtype=np.float64)
        refine_level = np.zeros(n_blocks, dtype=np.int32)

        block_fields: dict[str, list[np.ndarray]] = {v: [] for v in VAR_NAMES}

        ib = 0
        for ibx in range(n_blocks_x):
            for iby in range(n_blocks_y):
                x0 = XMIN + ibx * block_nx * DX
                y0 = YMIN + iby * block_ny * DY
                x1 = x0 + block_nx * DX
                y1 = y0 + block_ny * DY
                bbox[ib, 0] = [x0, x1]
                bbox[ib, 1] = [y0, y1]
                coord_db[ib] = [(x0 + x1) / 2, (y0 + y1) / 2]

                # Cell centers within this block
                bx_centers = x0 + (np.arange(block_nx) + 0.5) * DX
                by_centers = y0 + (np.arange(block_ny) + 0.5) * DY
                bxx, byy = np.meshgrid(bx_centers, by_centers, indexing="ij")
                fields = harris_fields(bxx, byy)

                for v in VAR_NAMES:
                    # HDF5 BATL stores (nK, nJ, nI) — for 2D, add singleton nK=1
                    block_data = fields[v].T[np.newaxis, :, :]  # (1, nJ, nI)
                    block_fields[v].append(block_data)

                ib += 1

        f.create_dataset("bounding box", data=bbox)
        f.create_dataset("Coord_DB", data=coord_db)
        f.create_dataset("refine level", data=refine_level)
        f.create_dataset("Processor Number", data=np.zeros(n_blocks, dtype=np.int32))
        f.create_dataset("iCoord_DB", data=np.ones((n_blocks, 2), dtype=np.int32))

        for v in VAR_NAMES:
            arr = np.stack(block_fields[v], axis=0)
            f.create_dataset(v, data=arr)
            ext = np.column_stack(
                [
                    arr.reshape(n_blocks, -1).min(axis=1),
                    arr.reshape(n_blocks, -1).max(axis=1),
                ]
            )
            f.create_dataset(f"{v}_Ext", data=ext)

    print(f"  HDF5 uniform: {out_dir / 'z=0_mhd_1_n00000000.batl'}")


def generate_hdf5_amr() -> None:
    """Write HDF5 BATL fixture with 2 refinement levels.

    Layout: 2 coarse blocks (8x8, dx=2) covering the outer domain,
    8 fine blocks (8x8, dx=1) covering the inner region.
    The inner region [-4, 4] x [-4, 4] is at the fine level.
    """
    out_dir = OUTPUT_DIR / "hdf5-amr"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    block_n = 8  # cells per block per dimension

    # Coarse level: dx = 2.0, 2 blocks covering full y range
    # Block 0: x=[-8, 8], y=[-8, 0]  → all coarse
    # Actually, let's make it simpler:
    # Coarse blocks (level 0, dx=2.0): 1 block covers [-8,-8] to [8,8] as 8x8
    # Wait, that's 16/8 = dx=2. But to have AMR, we need some at level 1.
    # Let me use: outer region at level 0 (dx=1.0), inner at level 1 (dx=0.5)
    # That way we have the full 16x16 domain with 2 refinement levels.

    dx_coarse = 1.0
    dx_fine = 0.5

    # Coarse blocks: 2x2 = 4 blocks of 8x8 covering [-8,8]x[-8,8]
    # Fine blocks: 2x2 = 4 blocks of 8x8 covering [-4,4]x[-4,4]
    # (replaces center of coarse)
    # But coarse blocks that overlap with fine region should not exist.
    # In BATSRUS AMR, coarse blocks are replaced by finer ones.
    # So: 4 coarse blocks - 1 center block + 4 fine blocks = 7 blocks total?
    # Actually in BATSRUS, a coarse block is replaced by 4 (2D) child blocks.
    # Let's simplify: use non-overlapping blocks.

    # Approach: outer ring at coarse, inner square at fine.
    # Coarse (dx=1.0, block covers 8x8 cells → 8x8 in space):
    #   Top-left:    [-8, 0] x [0, 8]
    #   Top-right:   [0, 8] x [0, 8]
    #   Bottom-left: [-8, 0] x [-8, 0]
    #   Bottom-right: [0, 8] x [-8, 0]
    # But we want inner refinement. So let's split one coarse block into 4 fine blocks.
    # Remove the center-ish coarse block and replace with 4 fine blocks.

    # Simplest: 3 coarse blocks + 4 fine blocks = 7 total
    # Coarse blocks (dx=1.0):
    #   Block A: x=[-8, 0], y=[-8, 0]  → 8x8 cells
    #   Block B: x=[-8, 0], y=[0, 8]   → 8x8 cells
    #   Block C: x=[0, 8],  y=[0, 8]   → 8x8 cells
    # Fine blocks (dx=0.5, each covers 4x4 in space = 8x8 cells):
    #   Block D: x=[0, 4], y=[-8, -4]  → 8x8 cells
    #   Block E: x=[0, 4], y=[-4, 0]   → 8x8 cells
    #   Block F: x=[4, 8], y=[-8, -4]  → 8x8 cells
    #   Block G: x=[4, 8], y=[-4, 0]   → 8x8 cells

    blocks = [
        # (x0, y0, x1, y1, dx, level)
        (-8.0, -8.0, 0.0, 0.0, dx_coarse, 0),  # A
        (-8.0, 0.0, 0.0, 8.0, dx_coarse, 0),  # B
        (0.0, 0.0, 8.0, 8.0, dx_coarse, 0),  # C
        (0.0, -8.0, 4.0, -4.0, dx_fine, 1),  # D
        (0.0, -4.0, 4.0, 0.0, dx_fine, 1),  # E
        (4.0, -8.0, 8.0, -4.0, dx_fine, 1),  # F
        (4.0, -4.0, 8.0, 0.0, dx_fine, 1),  # G
    ]
    n_blocks = len(blocks)

    with h5py.File(out_dir / "z=0_mhd_1_n00000000.batl", "w") as f:
        ipm = np.zeros(16, dtype=np.int32)
        ipm[2] = 2  # ndim
        ipm[3] = 2
        ipm[4] = n_blocks
        ipm[7] = block_n
        ipm[8] = block_n
        ipm[15] = N_VAR
        f.create_dataset("Integer Plot Metadata", data=ipm)

        rpm = np.zeros(7, dtype=np.float64)
        rpm[1], rpm[2] = XMIN, XMAX
        rpm[3], rpm[4] = YMIN, YMAX
        f.create_dataset("Real Plot Metadata", data=rpm)
        f.create_dataset("Integer Sim Metadata", data=np.zeros(8, dtype=np.int32))
        f.create_dataset("Real Simulation Metadata", data=rpm.copy())

        var_bytes = np.array([v.ljust(11).encode() for v in VAR_NAMES], dtype="|S11")
        f.create_dataset("NamePlotVar_V", data=var_bytes)
        unit_bytes = np.array(["normalized".ljust(11).encode()] * N_VAR, dtype="|S11")
        f.create_dataset("NamePlotUnit_V", data=unit_bytes)
        f.create_dataset(
            "Axis Labels", data=np.array([b"x          ", b"y          "], dtype="|S11")
        )

        bbox = np.empty((n_blocks, 2, 2), dtype=np.float64)
        coord_db = np.empty((n_blocks, 2), dtype=np.float64)
        refine_levels = np.empty(n_blocks, dtype=np.int32)
        block_fields: dict[str, list[np.ndarray]] = {v: [] for v in VAR_NAMES}

        for ib, (x0, y0, x1, y1, bdx, level) in enumerate(blocks):
            bbox[ib, 0] = [x0, x1]
            bbox[ib, 1] = [y0, y1]
            coord_db[ib] = [(x0 + x1) / 2, (y0 + y1) / 2]
            refine_levels[ib] = level

            bx_centers = x0 + (np.arange(block_n) + 0.5) * bdx
            by_centers = y0 + (np.arange(block_n) + 0.5) * bdx
            bxx, byy = np.meshgrid(bx_centers, by_centers, indexing="ij")
            flds = harris_fields(bxx, byy)

            for v in VAR_NAMES:
                block_fields[v].append(flds[v].T[np.newaxis, :, :])

        f.create_dataset("bounding box", data=bbox)
        f.create_dataset("Coord_DB", data=coord_db)
        f.create_dataset("refine level", data=refine_levels)
        f.create_dataset("Processor Number", data=np.zeros(n_blocks, dtype=np.int32))
        f.create_dataset("iCoord_DB", data=np.ones((n_blocks, 2), dtype=np.int32))

        for v in VAR_NAMES:
            arr = np.stack(block_fields[v], axis=0)
            f.create_dataset(v, data=arr)
            ext = np.column_stack(
                [
                    arr.reshape(n_blocks, -1).min(axis=1),
                    arr.reshape(n_blocks, -1).max(axis=1),
                ]
            )
            f.create_dataset(f"{v}_Ext", data=ext)

    print(f"  HDF5 AMR: {out_dir / 'z=0_mhd_1_n00000000.batl'} ({n_blocks} blocks)")


def generate_idl_amr() -> None:
    """Write IDL per-cell binary fixture with 2 refinement levels.

    Same block layout as the HDF5 AMR fixture:
    3 coarse blocks (dx=1.0) + 4 fine blocks (dx=0.5).
    Written as flat per-cell records with varying dx.
    """
    out_dir = OUTPUT_DIR / "idl-amr"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_param_in(out_dir / "PARAM.in")

    dx_coarse = 1.0
    dx_fine = 0.5
    block_n = 8

    blocks = [
        # (x0, y0, x1, y1, dx)
        (-8.0, -8.0, 0.0, 0.0, dx_coarse),
        (-8.0, 0.0, 0.0, 8.0, dx_coarse),
        (0.0, 0.0, 8.0, 8.0, dx_coarse),
        (0.0, -8.0, 4.0, -4.0, dx_fine),
        (0.0, -4.0, 4.0, 0.0, dx_fine),
        (4.0, -8.0, 8.0, -4.0, dx_fine),
        (4.0, -4.0, 8.0, 0.0, dx_fine),
    ]

    # Collect all cells
    all_records: list[np.ndarray] = []
    for x0, y0, _x1, _y1, bdx in blocks:
        bx_centers = x0 + (np.arange(block_n) + 0.5) * bdx
        by_centers = y0 + (np.arange(block_n) + 0.5) * bdx
        bxx, byy = np.meshgrid(bx_centers, by_centers, indexing="ij")
        flds = harris_fields(bxx, byy)
        for ix in range(block_n):
            for iy in range(block_n):
                values = np.array(
                    [bdx, bxx[ix, iy], byy[ix, iy], 0.0]
                    + [flds[v][ix, iy] for v in VAR_NAMES],
                    dtype=np.float64,
                )
                all_records.append(values)

    n_cells = len(all_records)

    write_header(
        out_dir / "z=0_mhd_1_n00000000.h",
        n_cells=n_cells,
        dx_min=dx_fine,
    )

    idl_path = out_dir / "z=0_mhd_1_n00000000_pe0000.idl"
    with idl_path.open("wb") as f:
        for values in all_records:
            data = values.tobytes()
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

    print(f"  IDL AMR: {idl_path} ({n_cells} cells, {idl_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating BATSRUS synthetic fixtures...")
    generate_idl_uniform()
    generate_hdf5_uniform()
    generate_hdf5_amr()
    generate_idl_amr()
    generate_out_ascii()
    generate_out_binary()
    print("Done.")
