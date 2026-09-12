"""Example 5: your own quantities, and checking them against a run.

Three things `compute()` will not do for you, in the order you tend to
need them: attach an array you worked out yourself, teach the registry a
formula so every dataset gains it, and compare your result against the
simulation's own field.

No files and no config — the dataset is built from arrays, as in
`ex1_minimal_fields.py`.
"""

import numpy as np

from pypic import (
    FieldDataset,
    GridInfo,
    Normalization,
    l2_relative_error,
    register_recipe,
    unregister_recipe,
)

NX, NY, NZ = 16, 12, 8
SHAPE = (NX, NY, NZ)

grid = GridInfo(dimensions=SHAPE, spacing=(0.25, 0.25, 0.25))
rng = np.random.default_rng(7)

# A two-species run: electron and ion pressures, a bulk flow, and a
# magnetic field.
data = FieldDataset.from_arrays(
    {
        "B_1": rng.standard_normal(SHAPE),
        "B_2": rng.standard_normal(SHAPE),
        "B_3": rng.standard_normal(SHAPE),
        "V_1": rng.standard_normal(SHAPE),
        "V_2": rng.standard_normal(SHAPE),
        "V_3": rng.standard_normal(SHAPE),
        "P_s0": rng.random(SHAPE) + 0.5,
        "P_s1": rng.random(SHAPE) + 0.5,
        "rho_m": rng.random(SHAPE) + 1.0,
    },
    grid,
    # Declare the SI anchor, or in_si() refuses rather than returning
    # code units labelled as tesla. identity() asserts "already SI".
    Normalization.identity(),
)

# %% Attaching an array you computed yourself
#
# with_field stores quantity_type in the dataset's attrs, so in_si(),
# field_info() and the plotting labels work on it without touching the
# global registry. One-off results belong here.
total_pressure = np.asarray(data["P_s0"]) + np.asarray(data["P_s1"])
data = data.with_field(
    "P",
    total_pressure,
    "pressure",
    long_name="Total scalar pressure",
)

assert data.has_field("P")
assert data.field_info("P").si_unit == "Pa"
# Summing species scales like a pressure, so SI conversion follows.
np.testing.assert_allclose(data.in_si("P"), total_pressure, rtol=1e-12)

# Now that P exists, the registry can reach everything built on it.
beta = data.compute("beta")
np.testing.assert_allclose(
    beta, 2.0 * total_pressure / np.asarray(data.compute("|B|")) ** 2, rtol=1e-12
)

# %% Teaching the registry a formula
#
# register_recipe is the other half: a formula you want recomputed on
# every dataset, not an array attached to one. `fields` may name stored
# fields or other derived quantities — they resolve recursively.
#
# Note it takes the name and quantity_type directly. `Recipe`, also
# exported, is the internal record the registry stores; it is public for
# the cross-language codegen in `pypic.codegen`, not for registration.
register_recipe(
    "e_mag_fraction",
    func=lambda e_b, e_k: e_b / (e_b + e_k),
    fields=("e_B", "e_k"),
    quantity_type="dimensionless",
    long_name="Magnetic share of magnetic-plus-kinetic energy",
)
try:
    fraction = data.compute("e_mag_fraction")
    assert fraction.shape == SHAPE
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))
finally:
    # Registration is global. Leaving it behind would make a second run
    # in the same interpreter raise "already registered".
    unregister_recipe("e_mag_fraction")

# %% Comparing your own array against the simulation's
#
# Three paths, and the right one depends on what you are holding:
#
#   bare arrays, same grid   -> l2_relative_error / linf_error
#   two FieldDatasets        -> compare_fields (regrids, converts to SI)
#   your array onto this run -> with_field, then either of the above
#
# Here we re-derive |B| by hand and check it against the registry's.
by_hand = np.sqrt(
    np.asarray(data["B_1"]) ** 2
    + np.asarray(data["B_2"]) ** 2
    + np.asarray(data["B_3"]) ** 2
)
error = l2_relative_error(by_hand, np.asarray(data.compute("|B|")))
assert error < 1e-14, error

# A deliberately wrong version registers as a real discrepancy, so the
# check above is not passing by construction.
assert l2_relative_error(by_hand * 1.01, np.asarray(data.compute("|B|"))) > 1e-3

print(f"Fields: {sorted(data.field_names())}")
print(f"beta range: [{beta.min():.4f}, {beta.max():.4f}]")
print(f"|B| L2 error, hand-rolled vs compute(): {error:.2e}")
print("OK")
