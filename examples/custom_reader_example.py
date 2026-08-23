#!/usr/bin/env python3
"""Example: reading custom simulation output with pypic.

This self-contained script demonstrates how to add a reader for a new
simulation code.  It:

1. Generates a small synthetic HDF5 file (stand-in for your code's output).
2. Defines a field map from native names to pypic canonical names.
3. Opens the data with ``open_simple`` / ``open_simulation``.
4. Computes a derived quantity (Alfvén speed).

To adapt this for your own simulation code:

- Replace the HDF5 generation with your actual output files.
- Adjust ``FIELD_MAP`` to match your code's field naming.
- Adjust ``file_pattern`` to match your code's filename convention.
- Provide ``grid=GridInfo(...)`` if your files lack grid metadata.
- Or drop a ``simulation.toml`` next to your data (see schema.md) and
  let ``open_simple`` pick it up automatically — no grid= needed.

Run::

    uv run python examples/custom_reader_example.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from pypic import (
    CARTESIAN,
    alfven_speed,
    magnetic_field_magnitude,
    open_simple,
)
from pypic.grid import GridInfo


def generate_synthetic_data(output_dir: Path) -> None:
    """Write two timesteps of fake MHD output.

    Mimics a code that writes fields at the HDF5 root level with
    its own naming convention: ``Bx_field``, ``density``, etc.
    """
    rng = np.random.default_rng(42)
    dims = (16, 12, 8)

    for step in (0, 100):
        filepath = output_dir / f"mycode_{step:06d}.h5"
        with h5py.File(filepath, "w") as f:
            # Magnetic field (code units)
            f.create_dataset("Bx_field", data=rng.uniform(0.5, 1.5, dims))
            f.create_dataset("By_field", data=rng.uniform(-0.3, 0.3, dims))
            f.create_dataset("Bz_field", data=rng.uniform(-0.1, 0.1, dims))
            # Mass density (code units)
            f.create_dataset("density", data=rng.uniform(0.5, 2.0, dims))
            # Velocity
            f.create_dataset("Vx_bulk", data=rng.uniform(-0.5, 0.5, dims))
            f.create_dataset("Vy_bulk", data=rng.uniform(-0.2, 0.2, dims))
            f.create_dataset("Vz_bulk", data=rng.uniform(-0.1, 0.1, dims))

            # Optional: embed grid metadata so pypic finds it
            # automatically.  If your code doesn't write this,
            # pass grid=GridInfo(...) instead.
            g = f.create_group("grid")
            g.attrs["dimensions"] = list(dims)
            g.attrs["spacing"] = [0.1, 0.1, 0.1]
            g.attrs["origin"] = [0.0, 0.0, 0.0]
            g.attrs["geometry"] = "cartesian"

            f.attrs["model"] = "MyCode"
            f.attrs["model_type"] = "MHD"
            f.attrs["step"] = step
            f.attrs["time"] = step * 0.01


# -- Field name mapping: your code's names → pypic canonical names --
#
# See schema.md for the full list of canonical field names.
# B_1/B_2/B_3 = magnetic field components, rho_m = mass density, etc.

FIELD_MAP = {
    "Bx_field": "B_1",
    "By_field": "B_2",
    "Bz_field": "B_3",
    "density": "rho_m",
    "Vx_bulk": "V_1",
    "Vy_bulk": "V_2",
    "Vz_bulk": "V_3",
}


def main() -> None:
    """Generate synthetic data, read it, compute Alfvén speed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        generate_synthetic_data(data_dir)

        # -- Option A: auto-detected grid from HDF5 metadata ----
        # field_map tells pypic which datasets to read and what
        # to call them.  Unmapped datasets are ignored.
        sim = open_simple(
            data_dir,
            file_pattern="mycode_{step:06d}.h5",
            field_map=FIELD_MAP,
        )

        print(f"Model: {sim.model_name} ({sim.model_type})")
        print(f"Grid:  {sim.grid.dimensions}")
        print(f"Steps: {sim.steps}")

        ds = sim.read(step=0)
        print(f"Fields: {sorted(ds.field_names())}")

        # Compute derived quantities
        b_mag = magnetic_field_magnitude(ds["Bx"], ds["By"], ds["Bz"])
        v_a = alfven_speed(b_mag, ds["rho_m"])
        print(f"|B| range: [{b_mag.min():.3f}, {b_mag.max():.3f}]")
        print(f"v_A range: [{v_a.min():.3f}, {v_a.max():.3f}]")

        # -- Option B: explicit grid (when HDF5 lacks metadata) -
        sim2 = open_simple(
            data_dir,
            file_pattern="mycode_{step:06d}.h5",
            field_map=FIELD_MAP,
            grid=GridInfo(
                dimensions=(16, 12, 8),
                spacing=(0.1, 0.1, 0.1),
                origin=(0.0, 0.0, 0.0),
                geometry=CARTESIAN,
            ),
        )
        ds2 = sim2.read(step=100)
        b2 = magnetic_field_magnitude(ds2["B_1"], ds2["B_2"], ds2["B_3"])
        print(f"\nStep 100 |B| mean: {b2.mean():.3f}")

        # -- Option C: custom I/O via _read_raw ---------------------
        # Override _read_raw for non-standard file layouts.
        # The parent class applies field_map and builds FieldDataset.
        from pypic.readers._simple import SimpleReader

        class RootReader(SimpleReader):
            """Read root-level datasets, promote to float64."""

            def _read_raw(
                self,
                filepath: Path,
                *,
                fields: set[str] | None = None,
            ) -> dict[str, np.ndarray]:
                with h5py.File(filepath, "r") as f:
                    return {
                        name: np.asarray(f[name], dtype=np.float64)
                        for name in f
                        if isinstance(f[name], h5py.Dataset) and f[name].ndim >= 2
                    }

        reader3 = RootReader(
            file_pattern="mycode_{step:06d}.h5",
            field_map=FIELD_MAP,
            grid=GridInfo(
                dimensions=(16, 12, 8),
                spacing=(0.1, 0.1, 0.1),
                origin=(0.0, 0.0, 0.0),
                geometry=CARTESIAN,
            ),
        )
        ds3 = reader3.read_timestep(data_dir, step=0)
        b3 = magnetic_field_magnitude(ds3["Bx"], ds3["By"], ds3["Bz"])
        print(f"\nOption C (custom _read_raw): |B| mean = {b3.mean():.3f}")

        print("\nDone. Adapt FIELD_MAP and file_pattern for your code.")


if __name__ == "__main__":
    main()
