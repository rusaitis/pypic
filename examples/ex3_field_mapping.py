"""Example 3: Custom field names mapped to canonical via field_map.

Demonstrates loading an HDF5 file whose dataset names don't follow
the canonical schema (e.g. "mag_x" instead of "B_1"). Uses field_map,
fields_group="", and explicit GridInfo to handle the non-standard layout.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from pypic import GridInfo, magnetic_field_magnitude, open_simple, poynting_flux

NX, NY, NZ = 8, 6, 4

grid = GridInfo(dimensions=(NX, NY, NZ), spacing=(1.0, 1.0, 1.0))

rng = np.random.default_rng(456)
mag_x = rng.standard_normal((NX, NY, NZ))
mag_y = rng.standard_normal((NX, NY, NZ))
mag_z = rng.standard_normal((NX, NY, NZ))
elec_x = rng.standard_normal((NX, NY, NZ))
elec_y = rng.standard_normal((NX, NY, NZ))
elec_z = rng.standard_normal((NX, NY, NZ))

field_map = {
    "mag_x": "B_1",
    "mag_y": "B_2",
    "mag_z": "B_3",
    "elec_x": "E_1",
    "elec_y": "E_2",
    "elec_z": "E_3",
}

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = Path(tmpdir) / "output_000000.h5"
    with h5py.File(filepath, "w") as f:
        # Datasets at root level, no fields/ group
        f.create_dataset("mag_x", data=mag_x)
        f.create_dataset("mag_y", data=mag_y)
        f.create_dataset("mag_z", data=mag_z)
        f.create_dataset("elec_x", data=elec_x)
        f.create_dataset("elec_y", data=elec_y)
        f.create_dataset("elec_z", data=elec_z)

    sim = open_simple(
        Path(tmpdir),
        field_map=field_map,
        grid=grid,
        fields_group="",  # datasets at root, not under fields/
    )

    ds = sim.read(step=0)

    # Canonical names work
    np.testing.assert_array_equal(ds["B_1"], mag_x)
    np.testing.assert_array_equal(ds["E_3"], elec_z)

    # Cartesian aliases work
    assert ds.has_field("Bx")
    assert ds.has_field("Ez")
    np.testing.assert_array_equal(ds["Bx"], ds["B_1"])

    # Derived: Poynting flux S = E x B
    s1, s2, s3 = poynting_flux(
        ds["E_1"],
        ds["E_2"],
        ds["E_3"],
        ds["B_1"],
        ds["B_2"],
        ds["B_3"],
    )
    # Verify S1 = E2*B3 - E3*B2 (cross product x-component)
    expected_s1 = elec_y * mag_z - elec_z * mag_y
    np.testing.assert_allclose(s1, expected_s1, rtol=1e-14)

    b_mag = magnetic_field_magnitude(ds["B_1"], ds["B_2"], ds["B_3"])
    print(f"Fields: {sorted(ds.field_names())}")
    print(f"|B| range: [{b_mag.min():.4f}, {b_mag.max():.4f}]")
    print(f"|S1| max: {np.abs(s1).max():.4f}")

print("OK")
