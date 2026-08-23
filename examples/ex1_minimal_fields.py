"""Example 1: Arrays to FieldDataset — the bare minimum.

Demonstrates building a FieldDataset from raw NumPy arrays, accessing
fields via geometry aliases, and computing a derived quantity. No files,
no config, no reader — just the container.
"""

import numpy as np

from pypic import FieldDataset, GridInfo, magnetic_field_magnitude

NX, NY, NZ = 8, 6, 4

grid = GridInfo(dimensions=(NX, NY, NZ), spacing=(0.5, 0.5, 0.5))

rng = np.random.default_rng(42)
b1 = rng.standard_normal((NX, NY, NZ))
b2 = rng.standard_normal((NX, NY, NZ))
b3 = rng.standard_normal((NX, NY, NZ))

fields = {"B_1": b1, "B_2": b2, "B_3": b3}
ds = FieldDataset.from_arrays(fields, grid)

# Canonical access
assert ds["B_1"].shape == (NX, NY, NZ)
np.testing.assert_array_equal(ds["B_1"], b1)

# Geometry alias: Bx -> B_1 in Cartesian
assert ds.has_field("Bx")
np.testing.assert_array_equal(ds["Bx"], ds["B_1"])

# Field listing
assert sorted(ds.field_names()) == ["B_1", "B_2", "B_3"]

# Derived quantity: |B|
b_mag = magnetic_field_magnitude(ds["B_1"], ds["B_2"], ds["B_3"])
expected = np.sqrt(b1**2 + b2**2 + b3**2)
np.testing.assert_allclose(b_mag, expected, rtol=1e-14)
assert b_mag.shape == (NX, NY, NZ)

# Grid round-trip
assert ds.grid.dimensions == (NX, NY, NZ)
assert ds.grid.spacing == (0.5, 0.5, 0.5)

print(f"Grid: {ds.grid.dimensions}, fields: {ds.field_names()}")
print(f"|B| range: [{b_mag.min():.4f}, {b_mag.max():.4f}]")
print("OK")
