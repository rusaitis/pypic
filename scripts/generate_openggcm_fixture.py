"""Generate small OpenGGCM test fixtures from real data.

Reads the full-resolution example data (700x376x376), subsamples every 25th
point to produce a ~28x16x16 grid, and writes WRN2-encoded .3df files plus
an ASCII grid file.  The output (~200 KB total) is committed to
``tests/data/openggcm-small/`` for fast CI tests.

Usage::

    uv run python scripts/generate_openggcm_fixture.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

EXAMPLE_DIR = Path("examples/uclamhd-example-3D")
OUTPUT_DIR = Path("tests/data/openggcm-small")
STEP = 25  # subsample stride


def wrn2_encode_field(values: NDArray[np.float64]) -> tuple[bytes, float, float]:
    """Encode a flat float64 array to WRN2 format.

    Returns the encoded data bytes and the (zmin, zmax) log-range header
    values.  Uses all-literal encoding (no RLE) since the decoder handles
    both.

    Parameters
    ----------
    values : NDArray[np.float64]
        Flat array of values to encode.

    Returns
    -------
    data : bytes
        WRN2-encoded data lines.
    zmin : float
        Minimum log-magnitude.
    zmax : float
        Maximum log-magnitude.
    """
    abs_vals = np.abs(values)
    # Clamp tiny values to avoid log(0)
    abs_vals = np.maximum(abs_vals, 1e-30)

    log_vals = np.log(abs_vals)
    zmin = float(log_vals.min())
    zmax = float(log_vals.max())

    # Cap dynamic range at 37 e-folds (WRN2 precision limit)
    if zmax - zmin > 37.0:
        zmin = zmax - 37.0

    count = len(values)
    lines: list[bytes] = []

    if zmin == zmax:
        # Constant field: no data lines needed
        return b"", zmin, zmax

    dz = 4410.0 / (zmax - zmin)

    for k in range(0, count, 64):
        chunk = values[k : k + 64]
        nk = len(chunk)

        abs_chunk = np.abs(chunk)
        abs_chunk = np.maximum(abs_chunk, 1e-30)
        log_chunk = np.log(abs_chunk)

        i3 = np.round(dz * (log_chunk - zmin)).astype(np.int32)
        i3 = np.clip(i3, 0, 4417)

        i1 = i3 // 94
        i2 = i3 % 94

        # Negative values: shift i1
        neg = chunk < 0
        i1[neg] += 47

        # Convert to byte values (range 33-126)
        b1 = (i1 + 33).astype(np.uint8)
        b2 = (i2 + 33).astype(np.uint8)

        # Line 1: checksum + i1 data + space
        data1 = b1[:nk].tolist()
        cksum1 = 33 + sum(data1) % 92
        line1 = bytes([cksum1, *data1, 32, 10])

        # Line 2: checksum + i2 data + space
        data2 = b2[:nk].tolist()
        cksum2 = 33 + sum(data2) % 92
        line2 = bytes([cksum2, *data2, 32, 10])

        lines.append(line1)
        lines.append(line2)

    return b"".join(lines), zmin, zmax


def write_3df_file(
    path: Path,
    fields: dict[str, NDArray[np.float64]],
    timestep: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Write a .3df file with WRN2-encoded fields."""
    count = nx * ny * nz
    parts: list[bytes] = []

    for name, data in fields.items():
        flat = data.reshape(count, order="F")
        encoded_data, zmin, zmax = wrn2_encode_field(flat)

        # Marker
        parts.append(b"FIELD-3D-1\n")
        # Field name (80-char padded)
        parts.append(f"{name:<80s}\n".encode("ascii"))
        # Description line
        parts.append(f"{'time = 0.000E+00':<80s}\n".encode("ascii"))
        # Dimensions: timestep nx ny nz
        parts.append(f"{timestep:8d}{nx:8d}{ny:8d}{nz:8d}\n".encode("ascii"))
        # WRN2 header: format a4, i8, 3e14.7, i8, a8
        pad = 0.0
        wrn2_header = (
            f"WRN2{count:8d}{zmin:14.7E}{zmax:14.7E}{pad:14.7E}{timestep:8d}FUNC-3-1\n"
        )
        parts.append(wrn2_header.encode("ascii"))
        # Data lines
        parts.append(encoded_data)

    path.write_bytes(b"".join(parts))


def write_grid_file(
    path: Path,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    metadata: dict[str, str],
) -> None:
    """Write an ASCII grid file matching parse_grid_file() expectations."""
    lines: list[str] = []

    # Header metadata
    for key, value in metadata.items():
        lines.append(f"{key}:{value}")
    lines.append("")

    # Primary grids and stagger grids
    # We write the 3 primary grids plus 18 staggered grids.
    # For the small fixture, stagger grids are identical to primary grids
    # (sufficient for testing the parser).
    grid_sections = [
        ("gridx", x),
        ("gridy", y),
        ("gridz", z),
    ]

    # Add stagger grids for all 6 field components
    for comp in ("bx", "by", "bz", "ex", "ey", "ez"):
        grid_sections.append((f"gx-{comp}", x))
        grid_sections.append((f"gy-{comp}", y))
        grid_sections.append((f"gz-{comp}", z))

    for name, arr in grid_sections:
        lines.append("FIELD-1D-1")
        lines.append(f"  {name}")
        lines.append(f"  {name} description")
        lines.append(f"  0 {len(arr)}")
        # Fake WRN2-like header (grid files store raw ASCII values)
        lines.append(f"  ASCII {len(arr)}")
        for val in arr:
            lines.append(f"  {val:.10E}")

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def subsample_array(arr: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    """Subsample a 1D coordinate array, always including first and last."""
    indices = list(range(0, len(arr), step))
    if indices[-1] != len(arr) - 1:
        indices.append(len(arr) - 1)
    return arr[indices]


def main() -> None:
    """Read real OpenGGCM data, subsample, and write small fixtures."""
    if not EXAMPLE_DIR.exists():
        print(f"Example data not found at {EXAMPLE_DIR}")
        print("Download the OpenGGCM example data first.")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the real grid
    from pypic.readers.openggcm import parse_grid_file

    grid = parse_grid_file(EXAMPLE_DIR / "grid.gc012.dat")
    print(f"Source grid: {grid.nx} x {grid.ny} x {grid.nz}")

    # Subsample coordinate arrays
    x_sub = subsample_array(grid.x, STEP)
    y_sub = subsample_array(grid.y, STEP)
    z_sub = subsample_array(grid.z, STEP)
    sub_nx, sub_ny, sub_nz = len(x_sub), len(y_sub), len(z_sub)
    print(f"Subsampled grid: {sub_nx} x {sub_ny} x {sub_nz}")

    # Build index arrays for field subsampling
    xi = list(range(0, grid.nx, STEP))
    if xi[-1] != grid.nx - 1:
        xi.append(grid.nx - 1)
    yi = list(range(0, grid.ny, STEP))
    if yi[-1] != grid.ny - 1:
        yi.append(grid.ny - 1)
    zi = list(range(0, grid.nz, STEP))
    if zi[-1] != grid.nz - 1:
        zi.append(grid.nz - 1)

    # Write grid file
    grid_path = OUTPUT_DIR / "grid.gc012.dat"
    write_grid_file(
        grid_path,
        x_sub,
        y_sub,
        z_sub,
        dict(grid.metadata),
    )
    print(f"Wrote {grid_path} ({grid_path.stat().st_size:,} bytes)")

    # Process each timestep
    from pypic.readers.openggcm._field_io import read_3df_file

    for step_idx in (6300, 6310):
        src_file = EXAMPLE_DIR / f"gc012.3df.{step_idx:06d}"
        if not src_file.exists():
            print(f"Skipping {src_file} (not found)")
            continue

        print(f"\nReading {src_file.name}...")
        fields, _ts, _nx, _ny, _nz = read_3df_file(
            src_file,
            skip={"eflx", "efly", "eflz"},
        )

        # Subsample fields
        sub_fields: dict[str, NDArray[np.float64]] = {}
        for name, data in fields.items():
            sub_fields[name] = data[np.ix_(xi, yi, zi)]
            print(f"  {name}: {data.shape} -> {sub_fields[name].shape}")

        # Write subsampled .3df file
        out_path = OUTPUT_DIR / f"gc012.3df.{step_idx:06d}"
        write_3df_file(out_path, sub_fields, step_idx, sub_nx, sub_ny, sub_nz)
        print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    # Round-trip verification
    print("\nVerifying round-trip...")
    verify_grid = parse_grid_file(grid_path)
    assert verify_grid.nx == sub_nx, f"nx: {verify_grid.nx} != {sub_nx}"
    assert verify_grid.ny == sub_ny, f"ny: {verify_grid.ny} != {sub_ny}"
    assert verify_grid.nz == sub_nz, f"nz: {verify_grid.nz} != {sub_nz}"
    np.testing.assert_allclose(verify_grid.x, x_sub, rtol=1e-8)
    np.testing.assert_allclose(verify_grid.y, y_sub, rtol=1e-8)
    np.testing.assert_allclose(verify_grid.z, z_sub, rtol=1e-8)
    print(f"  Grid: OK ({verify_grid.nx}x{verify_grid.ny}x{verify_grid.nz})")

    for step_idx in (6300, 6310):
        out_path = OUTPUT_DIR / f"gc012.3df.{step_idx:06d}"
        if not out_path.exists():
            continue

        rt_fields, _rt_ts, rt_nx, rt_ny, rt_nz = read_3df_file(
            out_path,
            skip={"eflx", "efly", "eflz"},
        )
        assert rt_nx == sub_nx
        assert rt_ny == sub_ny
        assert rt_nz == sub_nz

        # Read source again for comparison
        src_file = EXAMPLE_DIR / f"gc012.3df.{step_idx:06d}"
        src_fields, *_ = read_3df_file(src_file, skip={"eflx", "efly", "eflz"})

        for name in rt_fields:
            src_sub = src_fields[name][np.ix_(xi, yi, zi)]
            # WRN2 is lossy (~12.5-bit); re-encoding with different dynamic
            # range boundaries can produce up to ~0.5% error
            np.testing.assert_allclose(
                rt_fields[name],
                src_sub,
                rtol=5e-3,
                err_msg=f"Round-trip mismatch for {name} at step {step_idx}",
            )
        print(f"  Step {step_idx}: OK ({len(rt_fields)} fields)")

    # Summary
    total = sum(f.stat().st_size for f in OUTPUT_DIR.iterdir())
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
