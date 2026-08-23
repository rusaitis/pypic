"""Example 2: Write canonical HDF5 and read with open_simple.

Demonstrates writing a self-describing HDF5 file (with fields/ and grid/
groups) and loading it through the open_simple API. Grid metadata is
auto-detected from the file — no explicit GridInfo needed.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from pypic import (
    alfven_speed,
    magnetic_field_magnitude,
    open_simple,
    plasma_beta,
)

NX, NY, NZ = 8, 6, 4
DX, DY, DZ = 0.5, 0.5, 0.5

rng = np.random.default_rng(123)
b1 = rng.uniform(0.5, 1.5, (NX, NY, NZ))
b2 = rng.uniform(-0.2, 0.2, (NX, NY, NZ))
b3 = rng.uniform(-0.2, 0.2, (NX, NY, NZ))
rho_m = rng.uniform(0.5, 2.0, (NX, NY, NZ))
pressure = rng.uniform(0.1, 0.5, (NX, NY, NZ))

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = Path(tmpdir) / "output_000000.h5"
    with h5py.File(filepath, "w") as f:
        grp = f.create_group("fields")
        grp.create_dataset("B_1", data=b1)
        grp.create_dataset("B_2", data=b2)
        grp.create_dataset("B_3", data=b3)
        grp.create_dataset("rho_m", data=rho_m)
        grp.create_dataset("P", data=pressure)

        g = f.create_group("grid")
        g.attrs["dimensions"] = [NX, NY, NZ]
        g.attrs["spacing"] = [DX, DY, DZ]
        g.attrs["origin"] = [0.0, 0.0, 0.0]
        g.attrs["geometry"] = "cartesian"

        f.attrs["model"] = "custom_code"
        f.attrs["model_type"] = "MHD"
        f.attrs["time"] = 0.0
        f.attrs["step"] = 0

    sim = open_simple(Path(tmpdir))

    # Steps auto-detected from file pattern
    assert sim.steps == [0]
    print(sim.describe())

    ds = sim.read(step=0)

    # Fields match input arrays
    np.testing.assert_array_equal(ds["B_1"], b1)
    np.testing.assert_array_equal(ds["rho_m"], rho_m)
    assert ds.grid.dimensions == (NX, NY, NZ)

    # Derived quantities
    b_mag = magnetic_field_magnitude(ds["B_1"], ds["B_2"], ds["B_3"])
    v_a = alfven_speed(b_mag, ds["rho_m"])
    beta = plasma_beta(ds["P"], b_mag)

    assert v_a.shape == (NX, NY, NZ)
    assert beta.shape == (NX, NY, NZ)
    assert np.all(v_a > 0)
    assert np.all(beta > 0)

    print(f"v_A range: [{v_a.min():.4f}, {v_a.max():.4f}]")
    print(f"beta range: [{beta.min():.4f}, {beta.max():.4f}]")

print("OK")
