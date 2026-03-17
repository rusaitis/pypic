"""Tests for iPIC3D readers against real example data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.readers.ipic3d import (
    IPic3DParallelReader,
    IPic3DSerialReader,
    open_ipic3d,
    parse_inp,
    parse_settings_hdf,
    to_simulation_config,
)

EXAMPLE_2D = Path("examples/iPIC3D-example-2D")
EXAMPLE_3D = Path("examples/iPIC3D-example-3D")
EXAMPLE_SERIAL = Path("examples/iPIC3D-example-2D-serial")


# ---------------------------------------------------------------------------
# .inp parser
# ---------------------------------------------------------------------------


class TestParseInp:
    @pytest.fixture
    def cfg_2d(self):
        return parse_inp(EXAMPLE_2D / "example_2D.inp")

    @pytest.fixture
    def cfg_3d(self):
        return parse_inp(EXAMPLE_3D / "example_3D.inp")

    @pytest.fixture
    def cfg_serial(self):
        return parse_inp(EXAMPLE_SERIAL / "example_2D_serial.inp")

    def test_grid_dimensions(self, cfg_2d):
        assert cfg_2d.nxc == 100
        assert cfg_2d.nyc == 80
        assert cfg_2d.nzc == 1

    def test_domain_size(self, cfg_2d):
        assert cfg_2d.lx == 30.0
        assert cfg_2d.ly == 24.0
        assert cfg_2d.lz == 1.0

    def test_spacing(self, cfg_2d):
        assert_allclose(cfg_2d.dx, 0.3)
        assert_allclose(cfg_2d.dy, 0.3)
        assert_allclose(cfg_2d.dz, 1.0)

    def test_species_count(self, cfg_2d):
        assert cfg_2d.ns == 4

    def test_species_qom(self, cfg_2d):
        assert cfg_2d.qom == (-256.0, 1.0, -256.0, 1.0)

    def test_boundaries_periodic(self, cfg_2d):
        assert cfg_2d.periodic_x is True
        assert cfg_2d.periodic_y is True
        assert cfg_2d.periodic_z is True

    def test_write_method_parallel(self, cfg_2d):
        assert cfg_2d.write_method == "phdf5"

    def test_write_method_serial(self, cfg_serial):
        assert cfg_serial.write_method == "shdf5"

    def test_dt(self, cfg_2d):
        assert cfg_2d.dt == 0.125

    def test_theta(self, cfg_2d):
        assert cfg_2d.th == 0.5

    def test_speed_of_light(self, cfg_2d):
        assert cfg_2d.c == 1.0

    def test_b0(self, cfg_2d):
        assert cfg_2d.b0 == (0.097, 0.0, 0.0)

    def test_mpi_topology(self, cfg_2d):
        assert cfg_2d.xlen == 2
        assert cfg_2d.ylen == 4
        assert cfg_2d.zlen == 1

    def test_thermal_velocities(self, cfg_2d):
        assert cfg_2d.uth == (0.06, 0.0063, 0.06, 0.0063)
        assert cfg_2d.vth == (0.02, 0.0063, 0.02, 0.0063)

    def test_drift_velocities(self, cfg_2d):
        assert cfg_2d.w0 == (0.00325, -0.01624, 0.0, 0.0)

    def test_3d_grid(self, cfg_3d):
        assert cfg_3d.nzc == 10
        assert cfg_3d.lz == 3.0

    def test_field_output_cycle(self, cfg_2d):
        assert cfg_2d.field_output_cycle == 10

    def test_missing_key_raises(self, tmp_path):
        bad = tmp_path / "bad.inp"
        bad.write_text("nxc = 10\n")
        with pytest.raises(ExceptionGroup, match="Missing keys"):
            parse_inp(bad)


# ---------------------------------------------------------------------------
# settings.hdf parser
# ---------------------------------------------------------------------------


class TestParseSettingsHdf:
    @pytest.fixture
    def cfg_hdf(self):
        return parse_settings_hdf(EXAMPLE_SERIAL / "settings.hdf")

    @pytest.fixture
    def cfg_inp(self):
        return parse_inp(EXAMPLE_SERIAL / "example_2D_serial.inp")

    def test_grid_matches_inp(self, cfg_hdf, cfg_inp):
        assert cfg_hdf.nxc == cfg_inp.nxc
        assert cfg_hdf.nyc == cfg_inp.nyc
        assert cfg_hdf.nzc == cfg_inp.nzc

    def test_domain_matches_inp(self, cfg_hdf, cfg_inp):
        assert_allclose(cfg_hdf.lx, cfg_inp.lx)
        assert_allclose(cfg_hdf.ly, cfg_inp.ly)
        assert_allclose(cfg_hdf.lz, cfg_inp.lz)

    def test_species_qom_matches_inp(self, cfg_hdf, cfg_inp):
        assert cfg_hdf.qom == cfg_inp.qom

    def test_topology_matches_inp(self, cfg_hdf, cfg_inp):
        assert cfg_hdf.xlen == cfg_inp.xlen
        assert cfg_hdf.ylen == cfg_inp.ylen
        assert cfg_hdf.zlen == cfg_inp.zlen

    def test_periodicity_matches_inp(self, cfg_hdf, cfg_inp):
        assert cfg_hdf.periodic_x == cfg_inp.periodic_x
        assert cfg_hdf.periodic_y == cfg_inp.periodic_y
        assert cfg_hdf.periodic_z == cfg_inp.periodic_z

    def test_thermal_velocities_match(self, cfg_hdf, cfg_inp):
        assert_allclose(cfg_hdf.uth, cfg_inp.uth)
        assert_allclose(cfg_hdf.vth, cfg_inp.vth)
        assert_allclose(cfg_hdf.wth, cfg_inp.wth)


# ---------------------------------------------------------------------------
# to_simulation_config
# ---------------------------------------------------------------------------


class TestToSimulationConfig:
    @pytest.fixture
    def sim_cfg(self):
        return to_simulation_config(parse_inp(EXAMPLE_2D / "example_2D.inp"))

    def test_node_dimensions(self, sim_cfg):
        assert sim_cfg.grid.dimensions == (101, 81, 2)

    def test_origin_offset(self, sim_cfg):
        assert_allclose(sim_cfg.grid.origin, (-0.15, -0.15, -0.5))

    def test_coordinate_arrays_produce_node_positions(self, sim_cfg):
        x, y, z = sim_cfg.grid.coordinate_arrays()
        assert_allclose(x[0], 0.0, atol=1e-14)
        assert_allclose(x[-1], 30.0, atol=1e-14)
        assert_allclose(y[0], 0.0, atol=1e-14)
        assert_allclose(y[-1], 24.0, atol=1e-14)
        assert_allclose(z[0], 0.0, atol=1e-14)
        assert_allclose(z[-1], 1.0, atol=1e-14)

    def test_species_count(self, sim_cfg):
        assert len(sim_cfg.species) == 4

    def test_species_qom(self, sim_cfg):
        qoms = [s.charge_to_mass for s in sim_cfg.species]
        assert qoms == [-256.0, 1.0, -256.0, 1.0]

    def test_species_charge_mass_inferred(self, sim_cfg):
        electron = sim_cfg.species[0]
        assert electron.charge == -1.0
        assert_allclose(electron.mass, 1.0 / 256.0)

    def test_model_name(self, sim_cfg):
        assert sim_cfg.model_name == "iPIC3D"
        assert sim_cfg.model_type == "PIC"

    def test_physics_metadata(self, sim_cfg):
        assert sim_cfg.physics["theta"] == 0.5
        assert sim_cfg.physics["c"] == 1.0

    def test_boundary_conditions(self, sim_cfg):
        assert sim_cfg.grid.boundary == ("periodic", "periodic", "periodic")

    def test_grid_centering_metadata(self, sim_cfg):
        assert sim_cfg.metadata["grid_centering"] == "node"


# ---------------------------------------------------------------------------
# Parallel reader
# ---------------------------------------------------------------------------


class TestParallelReader:
    @pytest.fixture
    def reader_2d(self):
        cfg = parse_inp(EXAMPLE_2D / "example_2D.inp")
        return IPic3DParallelReader(cfg)

    @pytest.fixture
    def reader_3d(self):
        cfg = parse_inp(EXAMPLE_3D / "example_3D.inp")
        return IPic3DParallelReader(cfg)

    @pytest.fixture
    def ds_2d(self, reader_2d):
        return reader_2d.read_timestep(EXAMPLE_2D, 0)

    @pytest.fixture
    def ds_3d(self, reader_3d):
        return reader_3d.read_timestep(EXAMPLE_3D, 0)

    def test_available_timesteps(self, reader_2d):
        steps = reader_2d.available_timesteps(EXAMPLE_2D)
        assert steps == [0, 10, 20, 30]

    def test_available_timesteps_3d(self, reader_3d):
        steps = reader_3d.available_timesteps(EXAMPLE_3D)
        assert steps == [0, 10, 20, 30]

    def test_field_shapes_2d(self, ds_2d):
        assert ds_2d["B1"].shape == (101, 81, 2)

    def test_field_shapes_3d(self, ds_3d):
        assert ds_3d["B1"].shape == (101, 81, 11)

    def test_canonical_names_present(self, ds_2d):
        for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
            assert ds_2d.has_field(name)

    def test_cartesian_aliases(self, ds_2d):
        assert ds_2d.has_field("Bx")
        assert_allclose(ds_2d["Bx"], ds_2d["B1"])

    def test_total_fields_present(self, ds_2d):
        for name in ("rho_c", "J1", "J2", "J3"):
            assert ds_2d.has_field(name)

    def test_per_species_fields_present(self, ds_2d):
        for s in range(4):
            assert ds_2d.has_field(f"rho_c_s{s}")
            for comp in ("J1", "J2", "J3"):
                assert ds_2d.has_field(f"{comp}_s{s}")

    def test_density_4pi_corrected(self, ds_2d):
        # Background electron (species 2) should have rho ≈ -1.0
        # after 4π correction (raw stored as rho/(4π))
        rho_s2 = ds_2d["rho_c_s2"]
        assert_allclose(np.mean(rho_s2), -1.0, atol=0.05)

    def test_total_charge_is_sum(self, ds_2d):
        total = sum(ds_2d[f"rho_c_s{s}"] for s in range(4))
        assert_allclose(ds_2d["rho_c"], total)

    def test_total_current_is_sum(self, ds_2d):
        for comp in ("J1", "J2", "J3"):
            total = sum(ds_2d[f"{comp}_s{s}"] for s in range(4))
            assert_allclose(ds_2d[comp], total)

    def test_species_metadata(self, ds_2d):
        assert len(ds_2d.species) == 4

    def test_physics_metadata(self, ds_2d):
        assert ds_2d.physics["theta"] == 0.5
        assert ds_2d.physics["c"] == 1.0

    def test_grid_info(self, ds_2d):
        assert ds_2d.grid.dimensions == (101, 81, 2)
        assert_allclose(ds_2d.grid.spacing, (0.3, 0.3, 1.0))


# ---------------------------------------------------------------------------
# Serial reader
# ---------------------------------------------------------------------------


class TestSerialReader:
    @pytest.fixture
    def reader(self):
        cfg = parse_inp(EXAMPLE_SERIAL / "example_2D_serial.inp")
        return IPic3DSerialReader(cfg)

    @pytest.fixture
    def ds(self, reader):
        return reader.read_timestep(EXAMPLE_SERIAL, 0)

    def test_available_timesteps(self, reader):
        steps = reader.available_timesteps(EXAMPLE_SERIAL)
        assert steps == [0, 10, 20, 30]

    def test_field_shapes(self, ds):
        assert ds["B1"].shape == (101, 81, 2)

    def test_canonical_names_present(self, ds):
        for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
            assert ds.has_field(name)

    def test_per_species_present(self, ds):
        for s in range(4):
            assert ds.has_field(f"rho_c_s{s}")

    def test_total_charge_is_sum(self, ds):
        total = sum(ds[f"rho_c_s{s}"] for s in range(4))
        assert_allclose(ds["rho_c"], total)


# ---------------------------------------------------------------------------
# Cross-validation: serial must match parallel
# ---------------------------------------------------------------------------


class TestSerialMatchesParallel:
    @pytest.fixture
    def parallel_ds(self):
        cfg = parse_inp(EXAMPLE_2D / "example_2D.inp")
        reader = IPic3DParallelReader(cfg)
        return reader.read_timestep(EXAMPLE_2D, 0)

    @pytest.fixture
    def serial_ds(self):
        cfg = parse_inp(EXAMPLE_SERIAL / "example_2D_serial.inp")
        reader = IPic3DSerialReader(cfg)
        return reader.read_timestep(EXAMPLE_SERIAL, 0)

    def test_b_field_matches(self, parallel_ds, serial_ds):
        for comp in ("B1", "B2", "B3"):
            assert_allclose(serial_ds[comp], parallel_ds[comp], atol=0.0)

    def test_e_field_matches(self, parallel_ds, serial_ds):
        for comp in ("E1", "E2", "E3"):
            assert_allclose(serial_ds[comp], parallel_ds[comp], atol=0.0)

    def test_rho_matches(self, parallel_ds, serial_ds):
        # Species 0 shows ~0.08 differences between phdf5 and shdf5 runs
        # (different communication/smoothing paths in iPIC3D). Species 1-3
        # match to machine precision.
        for s in range(4):
            assert_allclose(
                serial_ds[f"rho_c_s{s}"],
                parallel_ds[f"rho_c_s{s}"],
                atol=0.1,
            )

    def test_rho_exact_for_background_species(self, parallel_ds, serial_ds):
        for s in (1, 2, 3):
            assert_allclose(
                serial_ds[f"rho_c_s{s}"],
                parallel_ds[f"rho_c_s{s}"],
                atol=1e-14,
            )

    def test_current_matches(self, parallel_ds, serial_ds):
        for comp in ("J1", "J2", "J3"):
            for s in range(4):
                assert_allclose(
                    serial_ds[f"{comp}_s{s}"],
                    parallel_ds[f"{comp}_s{s}"],
                    atol=0.01,
                )


# ---------------------------------------------------------------------------
# open_ipic3d convenience function
# ---------------------------------------------------------------------------


class TestOpenIpic3d:
    def test_detects_parallel(self):
        reader, config = open_ipic3d(EXAMPLE_2D)
        assert isinstance(reader, IPic3DParallelReader)
        assert config.model_name == "iPIC3D"

    def test_detects_serial(self):
        reader, config = open_ipic3d(EXAMPLE_SERIAL)
        assert isinstance(reader, IPic3DSerialReader)
        assert config.model_name == "iPIC3D"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_ipic3d(tmp_path)
