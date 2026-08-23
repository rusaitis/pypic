"""Example 4: Full simulation.toml configuration.

Demonstrates writing a simulation.toml config file alongside HDF5 data,
then loading both through open_simple(config_path=...) and load_config().
Verifies species info, normalization, and species-dependent derived
quantities (thermal_speed, gyrofrequency).
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from pypic import (
    Normalization,
    SpeciesInfo,
    gyrofrequency,
    load_config,
    magnetic_field_magnitude,
    open_simple,
    thermal_speed,
)

NX, NY, NZ = 8, 6, 4

TOML_CONTENT = """\
[schema]
version = "1.0"

[model]
name = "test_code"
type = "PIC"

[run]
name = "ex4_demo_run"

[time]
scheme = "fixed"
dt = 0.01
t_start = 0.0
t_end = 1.0
n_steps = 100

[grid]
dimensions = [8, 6, 4]
spacing = [0.5, 0.5, 0.5]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 3.0, 2.0]

[boundary_conditions]
lower = ["periodic", "periodic", "periodic"]
upper = ["periodic", "periodic", "periodic"]

[units]
system = "PIC"
reference_density = 1.0e18
reference_mass = 9.109e-31
reference_charge = 1.602e-19
speed_of_light = 2.998e8

[coordinates]
geometry = "cartesian"
frame = "simulation"

[[species]]
name = "electrons"
charge = -1.0
mass = 1.0

[[species]]
name = "ions"
charge = 1.0
mass = 256.0
"""

rng = np.random.default_rng(789)
b1 = rng.uniform(0.5, 1.5, (NX, NY, NZ))
b2 = rng.uniform(-0.1, 0.1, (NX, NY, NZ))
b3 = rng.uniform(-0.1, 0.1, (NX, NY, NZ))
temperature_e = np.full((NX, NY, NZ), 0.04)  # electron temperature
temperature_i = np.full((NX, NY, NZ), 0.01)  # ion temperature

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = Path(tmpdir)

    # Write config
    toml_path = data_dir / "simulation.toml"
    toml_path.write_text(TOML_CONTENT)

    # Write HDF5 data
    filepath = data_dir / "output_000000.h5"
    with h5py.File(filepath, "w") as f:
        grp = f.create_group("fields")
        grp.create_dataset("B_1", data=b1)
        grp.create_dataset("B_2", data=b2)
        grp.create_dataset("B_3", data=b3)
        grp.create_dataset("Te", data=temperature_e)
        grp.create_dataset("Ti", data=temperature_i)

    # Load config independently
    cfg = load_config(toml_path)
    assert cfg.model_name == "test_code"
    assert cfg.model_type == "PIC"
    assert cfg.grid.dimensions == (NX, NY, NZ)
    assert cfg.grid.dt == 0.01
    assert cfg.frame == "simulation"
    assert len(cfg.species) == 2
    assert cfg.species[0].name == "electrons"
    assert cfg.species[1].name == "ions"
    assert cfg.species[0].charge == -1.0
    assert cfg.species[1].mass == 256.0

    # Normalization round-trip
    norm = cfg.normalization
    assert isinstance(norm, Normalization)

    # Open via open_simple (auto-detects simulation.toml in directory)
    sim = open_simple(data_dir)
    assert sim.model_name == "test_code"
    assert len(sim.species) == 2
    print(sim.describe())

    ds = sim.read(step=0)
    np.testing.assert_array_equal(ds["B_1"], b1)

    # Species info preserved on FieldDataset
    electrons = ds.species[0]
    ions = ds.species[1]
    assert isinstance(electrons, SpeciesInfo)
    assert electrons.name == "electrons"
    assert ions.mass == 256.0

    # Species-dependent derived quantities
    b_mag = magnetic_field_magnitude(ds["B_1"], ds["B_2"], ds["B_3"])

    assert electrons.mass is not None
    assert ions.mass is not None
    assert electrons.charge is not None
    assert ions.charge is not None

    v_th_e = thermal_speed(ds["Te"], electrons.mass)
    v_th_i = thermal_speed(ds["Ti"], ions.mass)
    assert v_th_e.shape == (NX, NY, NZ)
    # Electrons are faster: lighter mass, higher temperature
    assert np.all(v_th_e > v_th_i)

    omega_ce = gyrofrequency(b_mag, electrons.charge, electrons.mass)
    omega_ci = gyrofrequency(b_mag, ions.charge, ions.mass)
    assert np.all(omega_ce > 0)
    assert np.all(omega_ci > 0)
    # Electron gyrofrequency >> ion (mass ratio 256)
    np.testing.assert_allclose(
        omega_ce / omega_ci,
        256.0,
        rtol=1e-12,
    )

    print(f"v_th_e: {v_th_e.flat[0]:.4f}, v_th_i: {v_th_i.flat[0]:.4f}")
    print(f"omega_ce/omega_ci = {(omega_ce.flat[0] / omega_ci.flat[0]):.1f}")

print("OK")
