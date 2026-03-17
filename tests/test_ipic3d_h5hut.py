"""Tests for iPIC3D H5hut reader and ConservedQuantities parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.readers.ipic3d import (
    IPic3DH5hutReader,
    IPic3DParallelReader,
    load_conserved_quantities,
    open_ipic3d,
    parse_inp,
)

EXAMPLE_H5HUT = Path("examples/iPIC3D-example-3D-h5hut")
EXAMPLE_2D = Path("examples/iPIC3D-example-2D")


class TestParseInpH5hut:
    """Verify .inp parsing with inline comments (the key bug fix)."""

    @pytest.fixture
    def cfg(self):
        return parse_inp(EXAMPLE_H5HUT / "MHDUCLA.inp")

    def test_grid_dimensions(self, cfg):
        assert cfg.nxc == 460
        assert cfg.nyc == 130
        assert cfg.nzc == 320

    def test_domain_size(self, cfg):
        assert cfg.lx == 184.0
        assert cfg.ly == 52.0
        assert cfg.lz == 128.0

    def test_inline_comments_stripped(self, cfg):
        # nxc = 460  # comment → should parse as 460, not raise
        # This is the regression test for the .inp parser bug
        assert cfg.nxc == 460
        assert cfg.dt == 0.1

    def test_boundaries_not_periodic(self, cfg):
        assert cfg.periodic_x is False
        assert cfg.periodic_y is False
        assert cfg.periodic_z is False

    def test_species_count(self, cfg):
        assert cfg.ns == 2

    def test_qom(self, cfg):
        assert cfg.qom[0] == -256.0
        assert cfg.qom[1] == 1.0

    def test_simulation_name(self, cfg):
        assert cfg.simulation_name == "MHDUCLA"


class TestH5hutReader:
    """Test IPic3DH5hutReader against the 3D H5hut example data."""

    @pytest.fixture(scope="class")
    def reader(self):
        cfg = parse_inp(EXAMPLE_H5HUT / "MHDUCLA.inp")
        return IPic3DH5hutReader(cfg)

    @pytest.fixture(scope="class")
    def ds(self, reader):
        return reader.read_timestep(EXAMPLE_H5HUT, 202500)

    @pytest.fixture(scope="class")
    def ds_cycle0(self, reader):
        return reader.read_timestep(EXAMPLE_H5HUT, 0)

    def test_available_timesteps(self, reader):
        steps = reader.available_timesteps(EXAMPLE_H5HUT)
        assert 202500 in steps

    def test_field_shapes(self, ds):
        assert ds["B1"].shape == (461, 131, 321)

    def test_canonical_em_fields_present(self, ds):
        for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
            assert ds.has_field(name)

    def test_cartesian_aliases(self, ds):
        assert ds.has_field("Bx")
        assert_allclose(ds["Bx"], ds["B1"])

    def test_per_species_density(self, ds):
        assert ds.has_field("rho_c_s0")
        assert ds.has_field("rho_c_s1")

    def test_electron_charge_density_negative(self, ds):
        assert np.mean(ds["rho_c_s0"]) < 0

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_allclose(ds["rho_c"], total)

    def test_total_current_is_sum(self, ds):
        for comp in ("J1", "J2", "J3"):
            total = ds[f"{comp}_s0"] + ds[f"{comp}_s1"]
            assert_allclose(ds[comp], total)

    def test_fluid_velocity_present(self, ds):
        for name in ("V1", "V2", "V3"):
            assert ds.has_field(name)

    def test_pressure_tensor_present(self, ds):
        for comp in ("P11", "P12", "P13", "P22", "P23", "P33"):
            for s in (0, 1):
                assert ds.has_field(f"{comp}_s{s}")

    def test_pressure_diagonal_positive(self, ds):
        # After sign fix, electron diagonal pressure should be non-negative
        for comp in ("P11", "P22", "P33"):
            assert ds[f"{comp}_s0"].min() >= 0.0, f"{comp}_s0 has negative values"
            assert ds[f"{comp}_s1"].min() >= 0.0, f"{comp}_s1 has negative values"

    def test_divb_present(self, ds):
        assert ds.has_field("div_B")

    def test_rho_no_4pi_applied(self, ds):
        # rho_c_s1 (ions) mean should be ~1.14, not 14.3 (which is 1.14*4π)
        mean_rho_ion = np.mean(ds["rho_c_s1"])
        assert_allclose(mean_rho_ion, 1.14, atol=0.05)

    def test_electron_density_matches_rho_init(self, ds_cycle0):
        # rhoINIT = 1.0 for electrons; cycle 0 should be close
        mean_rho_e = np.mean(np.abs(ds_cycle0["rho_c_s0"]))
        assert_allclose(mean_rho_e, 1.0, atol=0.1)

    def test_pressure_consistent_with_init(self, ds_cycle0):
        # After 4π correction, |Pxx_0/rho_0| ≈ vth_e² = 0.0225² = 5.06e-4
        # Without correction it would be vth_e²/(4π) ≈ 4.0e-5
        pxx = ds_cycle0["P11_s0"]
        rho = ds_cycle0["rho_c_s0"]
        measured_vx2 = np.mean(np.abs(pxx / rho))
        vth_e_squared = 0.0225**2
        # Ratio should be ~1.0 (not ~0.08 = 1/(4π))
        ratio = measured_vx2 / vth_e_squared
        assert ratio > 0.5, f"Pressure too low: {ratio:.3f}"
        assert ratio < 3.0, f"Pressure too high: {ratio:.3f}"

    def test_cycle0_has_fewer_fields(self, ds_cycle0):
        # Cycle 0 has 30 fields (no N_s, EFx_s, divB, Qrem_s)
        # but reader should still load available fields without error
        assert ds_cycle0.has_field("B1")
        assert ds_cycle0.has_field("rho_c_s0")
        assert ds_cycle0.has_field("P11_s0")


class TestOpenIpic3dH5hut:
    def test_detects_h5hut(self):
        reader, config = open_ipic3d(EXAMPLE_H5HUT)
        assert isinstance(reader, IPic3DH5hutReader)
        assert config.model_name == "iPIC3D"

    def test_existing_formats_unaffected(self):
        reader, _ = open_ipic3d(EXAMPLE_2D)
        assert isinstance(reader, IPic3DParallelReader)


class TestConservedQuantitiesMulti:
    """Test Format B (per-restart-segment files)."""

    @pytest.fixture(scope="class")
    def cq(self):
        return load_conserved_quantities(EXAMPLE_H5HUT / "info-conserved")

    def test_loads_all_files(self, cq):
        assert len(cq.cycle) > 1000

    def test_cycles_sorted(self, cq):
        diffs = np.diff(cq.cycle)
        assert np.all(diffs > 0)

    def test_cycles_deduplicated(self, cq):
        assert len(np.unique(cq.cycle)) == len(cq.cycle)

    def test_species_arrays_length(self, cq):
        assert len(cq.species_npart) == 2
        assert len(cq.species_charge) == 2
        assert len(cq.species_kinetic_energy) == 2
        for arr in cq.species_npart:
            assert len(arr) == len(cq.cycle)

    def test_energies_positive_where_finite(self, cq):
        # Some restart segments write NaN for total energy during init;
        # check that valid entries are positive
        valid = np.isfinite(cq.total_energy)
        assert np.sum(valid) > 0.95 * len(cq.total_energy)
        assert np.all(cq.total_energy[valid] > 0)
        valid_b = np.isfinite(cq.magnetic_energy)
        assert np.all(cq.magnetic_energy[valid_b] > 0)

    def test_particle_counts_positive(self, cq):
        for arr in cq.species_npart:
            assert np.all(arr > 0)


class TestConservedQuantitiesSingle:
    """Test Format A (Roman numeral header, single file)."""

    @pytest.fixture(scope="class")
    def cq(self):
        return load_conserved_quantities(EXAMPLE_2D / "ConservedQuantities.txt")

    def test_loads(self, cq):
        assert len(cq.cycle) == 7

    def test_first_cycle_zero(self, cq):
        assert cq.cycle[0] == 0

    def test_cycles_sorted(self, cq):
        assert list(cq.cycle) == sorted(cq.cycle)

    def test_no_species_data(self, cq):
        # Format A doesn't include per-species columns
        assert len(cq.species_npart) == 0

    def test_energies_present(self, cq):
        assert len(cq.total_energy) == 7
        assert len(cq.electric_energy) == 7
        assert len(cq.magnetic_energy) == 7
