"""Tests for iPIC3D readers against synthetic fixtures with exact assertions.

All three reader formats (phdf5, shdf5, H5hut) are tested against
analytically known values from the fixture generator. Fixtures are
committed to ``tests/data/ipic3d-synthetic/`` (~200 KB).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.readers.base import AuxiliaryDataReader, TabularData
from pypic.readers.ipic3d import (
    IPic3DH5hutReader,
    IPic3DParallelReader,
    IPic3DSerialReader,
    conserved_to_tabular,
    load_conserved_quantities,
    open_ipic3d,
    parse_inp,
    parse_settings_hdf,
    to_simulation_config,
)

DATA = Path("tests/data/ipic3d-synthetic")
PHDF5_DIR = DATA / "phdf5"
SHDF5_DIR = DATA / "shdf5"
H5HUT_DIR = DATA / "h5hut"

FOUR_PI = 4.0 * math.pi

# Analytical reference values (must match generate_ipic3d_fixture.py)
NXC, NYC, NZC = 4, 4, 2
LX, LY, LZ = 4.0, 4.0, 2.0
DX, DY, DZ = LX / NXC, LY / NYC, LZ / NZC
NX, NY, NZ = NXC + 1, NYC + 1, NZC + 1
B0X = 0.1
QOM = (-64.0, 1.0)
UTH = (0.04, 0.005)
VTH = (0.02, 0.005)
WTH = (0.02, 0.005)
RHO_INIT = (1.0, 1.0)


def _expected_b1() -> np.ndarray:
    """Analytical B1 field: B0x * tanh((y - Ly/2) / delta)."""
    x = np.linspace(0, LX, NX)
    y = np.linspace(0, LY, NY)
    z = np.linspace(0, LZ, NZ)
    _, yy, _ = np.meshgrid(x, y, z, indexing="ij")
    delta = LY / 8.0
    return B0X * np.tanh((yy - LY / 2.0) / delta)


class TestParseInpSynthetic:
    @pytest.fixture
    def cfg(self):
        return parse_inp(PHDF5_DIR / "synthetic.inp")

    def test_grid_dimensions(self, cfg):
        assert cfg.nxc == NXC
        assert cfg.nyc == NYC
        assert cfg.nzc == NZC

    def test_domain_size(self, cfg):
        assert cfg.lx == LX
        assert cfg.ly == LY
        assert cfg.lz == LZ

    def test_spacing(self, cfg):
        assert_allclose(cfg.dx, DX)
        assert_allclose(cfg.dy, DY)
        assert_allclose(cfg.dz, DZ)

    def test_species_count(self, cfg):
        assert cfg.ns == 2

    def test_qom(self, cfg):
        assert cfg.qom == QOM

    def test_b0(self, cfg):
        assert cfg.b0 == (B0X, 0.0, 0.0)

    def test_periodic(self, cfg):
        assert cfg.periodic_x is True
        assert cfg.periodic_y is True
        assert cfg.periodic_z is True

    def test_write_method(self, cfg):
        assert cfg.write_method == "phdf5"

    def test_inline_comment_stripped(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        assert cfg.write_method == "h5hut"


class TestParseSettingsHdfSynthetic:
    @pytest.fixture
    def cfg_hdf(self):
        return parse_settings_hdf(SHDF5_DIR / "settings.hdf")

    @pytest.fixture
    def cfg_inp(self):
        return parse_inp(SHDF5_DIR / "synthetic_serial.inp")

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

    def test_topology(self, cfg_hdf):
        assert cfg_hdf.xlen == 2
        assert cfg_hdf.ylen == 1
        assert cfg_hdf.zlen == 1

    def test_thermal_velocities(self, cfg_hdf, cfg_inp):
        assert_allclose(cfg_hdf.uth, cfg_inp.uth)
        assert_allclose(cfg_hdf.vth, cfg_inp.vth)
        assert_allclose(cfg_hdf.wth, cfg_inp.wth)


class TestToSimulationConfigSynthetic:
    @pytest.fixture
    def sim_cfg(self):
        return to_simulation_config(parse_inp(PHDF5_DIR / "synthetic.inp"))

    def test_node_dimensions(self, sim_cfg):
        assert sim_cfg.grid.dimensions == (NX, NY, NZ)

    def test_origin_offset(self, sim_cfg):
        assert_allclose(sim_cfg.grid.origin, (-DX / 2, -DY / 2, -DZ / 2))

    def test_coordinate_arrays_produce_node_positions(self, sim_cfg):
        x, y, z = sim_cfg.grid.coordinate_arrays()
        assert_allclose(x[0], 0.0, atol=1e-14)
        assert_allclose(x[-1], LX, atol=1e-14)
        assert_allclose(y[0], 0.0, atol=1e-14)
        assert_allclose(y[-1], LY, atol=1e-14)
        assert_allclose(z[0], 0.0, atol=1e-14)
        assert_allclose(z[-1], LZ, atol=1e-14)

    def test_species_count(self, sim_cfg):
        assert len(sim_cfg.species) == 2

    def test_species_qom(self, sim_cfg):
        qoms = [s.charge_to_mass for s in sim_cfg.species]
        assert qoms == list(QOM)

    def test_species_charge_mass_inferred(self, sim_cfg):
        electron = sim_cfg.species[0]
        assert electron.charge == -1.0
        assert_allclose(electron.mass, 1.0 / 64.0)

    def test_model_name(self, sim_cfg):
        assert sim_cfg.model_name == "iPIC3D"
        assert sim_cfg.model_type == "PIC"

    def test_boundary_conditions(self, sim_cfg):
        assert sim_cfg.grid.boundary == ("periodic", "periodic", "periodic")

    def test_grid_centering_metadata(self, sim_cfg):
        assert sim_cfg.metadata["grid_centering"] == "node"

    def test_missing_key_raises(self, tmp_path):
        bad = tmp_path / "bad.inp"
        bad.write_text("nxc = 10\n")
        with pytest.raises(ExceptionGroup, match="Missing keys"):
            parse_inp(bad)


class TestPhdf5Reader:
    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        return reader.read_timestep(PHDF5_DIR, 0)

    def test_available_timesteps(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        assert reader.available_timesteps(PHDF5_DIR) == [0]

    def test_field_shapes(self, ds):
        assert ds["B1"].shape == (NX, NY, NZ)

    def test_b_field_exact(self, ds):
        assert_allclose(ds["B1"], _expected_b1(), atol=1e-14)

    def test_e_field_zero(self, ds):
        for comp in ("E1", "E2", "E3"):
            assert_allclose(ds[comp], 0.0, atol=1e-14)

    def test_canonical_names_present(self, ds):
        for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
            assert ds.has_field(name)

    def test_cartesian_aliases(self, ds):
        assert ds.has_field("Bx")
        assert_allclose(ds["Bx"], ds["B1"])

    def test_electron_density_exact(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-12)

    def test_ion_density_exact(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-12)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_allclose(ds["rho_c"], total)

    def test_total_current_is_sum(self, ds):
        for comp in ("J1", "J2", "J3"):
            total = ds[f"{comp}_s0"] + ds[f"{comp}_s1"]
            assert_allclose(ds[comp], total)

    def test_electron_pressure_positive(self, ds):
        for comp in ("P11", "P22", "P33"):
            p = ds[f"{comp}_s0"]
            assert np.all(p >= 0), f"{comp}_s0 has negative values"

    def test_ion_pressure_positive(self, ds):
        for comp in ("P11", "P22", "P33"):
            p = ds[f"{comp}_s1"]
            assert np.all(p >= 0), f"{comp}_s1 has negative values"

    def test_pressure_exact_electron_p11(self, ds):
        expected = RHO_INIT[0] * UTH[0] ** 2
        assert_allclose(ds["P11_s0"], expected, atol=1e-12)

    def test_pressure_exact_ion_p11(self, ds):
        expected = RHO_INIT[1] * UTH[1] ** 2
        assert_allclose(ds["P11_s1"], expected, atol=1e-12)

    def test_pressure_consistency_p_over_rho(self, ds):
        """P/|rho| = v_th² for each species."""
        for s, uth in enumerate(UTH):
            p = ds[f"P11_s{s}"]
            rho = ds[f"rho_c_s{s}"]
            ratio = np.mean(p) / np.mean(np.abs(rho))
            assert_allclose(ratio, uth**2, atol=1e-12)

    def test_off_diagonal_pressure_zero(self, ds):
        for s in range(2):
            for comp in ("P12", "P13", "P23"):
                assert_allclose(ds[f"{comp}_s{s}"], 0.0, atol=1e-12)

    def test_species_metadata(self, ds):
        assert len(ds.species) == 2

    def test_grid_info(self, ds):
        assert ds.grid.dimensions == (NX, NY, NZ)
        assert_allclose(ds.grid.spacing, (DX, DY, DZ))


class TestShdf5Reader:
    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        return reader.read_timestep(SHDF5_DIR, 0)

    def test_available_timesteps(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        assert reader.available_timesteps(SHDF5_DIR) == [0]

    def test_field_shapes(self, ds):
        assert ds["B1"].shape == (NX, NY, NZ)

    def test_b_field_exact(self, ds):
        assert_allclose(ds["B1"], _expected_b1(), atol=1e-14)

    def test_electron_density_exact(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-12)

    def test_ion_density_exact(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-12)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_allclose(ds["rho_c"], total)

    def test_per_species_present(self, ds):
        for s in range(2):
            assert ds.has_field(f"rho_c_s{s}")
            for comp in ("J1", "J2", "J3"):
                assert ds.has_field(f"{comp}_s{s}")


class TestShdf5MatchesPhdf5:
    """Cross-format validation: shdf5 assembly must match phdf5 exactly."""

    @pytest.fixture(scope="class")
    def phdf5_ds(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        return reader.read_timestep(PHDF5_DIR, 0)

    @pytest.fixture(scope="class")
    def shdf5_ds(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        return reader.read_timestep(SHDF5_DIR, 0)

    def test_b_field_matches(self, phdf5_ds, shdf5_ds):
        for comp in ("B1", "B2", "B3"):
            assert_allclose(shdf5_ds[comp], phdf5_ds[comp], atol=1e-14)

    def test_e_field_matches(self, phdf5_ds, shdf5_ds):
        for comp in ("E1", "E2", "E3"):
            assert_allclose(shdf5_ds[comp], phdf5_ds[comp], atol=1e-14)

    def test_density_matches(self, phdf5_ds, shdf5_ds):
        for s in range(2):
            assert_allclose(
                shdf5_ds[f"rho_c_s{s}"],
                phdf5_ds[f"rho_c_s{s}"],
                atol=1e-12,
            )

    def test_current_matches(self, phdf5_ds, shdf5_ds):
        for comp in ("J1", "J2", "J3"):
            for s in range(2):
                assert_allclose(
                    shdf5_ds[f"{comp}_s{s}"],
                    phdf5_ds[f"{comp}_s{s}"],
                    atol=1e-12,
                )


class TestH5hutReader:
    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        return reader.read_timestep(H5HUT_DIR, 0)

    def test_available_timesteps(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        assert reader.available_timesteps(H5HUT_DIR) == [0]

    def test_field_shapes(self, ds):
        assert ds["B1"].shape == (NX, NY, NZ)

    def test_float64_promotion(self, ds):
        """H5hut stores float32; reader must promote to float64."""
        assert ds["B1"].dtype == np.float64

    def test_b_field_after_transpose(self, ds):
        """ZYX→XYZ transpose + float32→float64 must recover correct values."""
        assert_allclose(ds["B1"], _expected_b1(), atol=1e-6)

    def test_e_field_zero(self, ds):
        for comp in ("E1", "E2", "E3"):
            assert_allclose(ds[comp], 0.0, atol=1e-6)

    def test_canonical_em_fields_present(self, ds):
        for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
            assert ds.has_field(name)

    def test_cartesian_aliases(self, ds):
        assert ds.has_field("Bx")

    def test_electron_density_4pi_corrected(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-5)

    def test_ion_density_4pi_corrected(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-5)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_allclose(ds["rho_c"], total)

    def test_total_current_is_sum(self, ds):
        for comp in ("J1", "J2", "J3"):
            total = ds[f"{comp}_s0"] + ds[f"{comp}_s1"]
            assert_allclose(ds[comp], total)

    def test_electron_pressure_positive(self, ds):
        for comp in ("P11", "P22", "P33"):
            assert ds[f"{comp}_s0"].min() >= 0.0, f"{comp}_s0 has negative values"

    def test_pressure_p_over_rho_consistency(self, ds):
        """4π on both P and rho cancels: P/|rho| = v_th²."""
        for s, uth in enumerate(UTH):
            p = ds[f"P11_s{s}"]
            rho = ds[f"rho_c_s{s}"]
            ratio = np.mean(p) / np.mean(np.abs(rho))
            assert_allclose(ratio, uth**2, atol=1e-4)

    def test_fluid_velocity_present(self, ds):
        for name in ("V1", "V2", "V3"):
            assert ds.has_field(name)

    def test_divb_present(self, ds):
        assert ds.has_field("div_B")


class TestH5hutUnknownFieldPassthrough:
    """Unknown fields in H5hut Block/ pass through with native names."""

    def test_extra_field_included(self, tmp_path: Path) -> None:
        import h5py  # type: ignore[import-untyped]

        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        # Copy the real H5hut file and inject an extra dataset
        src = next(iter(H5HUT_DIR.glob("*-Fields_000000.h5")))
        dst = tmp_path / src.name
        import shutil

        shutil.copy2(src, dst)
        with h5py.File(dst, "a") as f:
            block = f["Step#0"]["Block"]
            shape = block["Bx"]["0"].shape
            block.create_group("CustomField")
            block["CustomField"].create_dataset(
                "0", data=np.full(shape, 99.0, dtype=np.float32),
            )

        reader = IPic3DH5hutReader(cfg)
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("CustomField")
        assert_allclose(ds["CustomField"], 99.0, atol=1e-5)


class TestOpenIpic3dSynthetic:
    def test_detects_parallel(self):
        reader, config = open_ipic3d(PHDF5_DIR)
        assert isinstance(reader, IPic3DParallelReader)
        assert config.model_name == "iPIC3D"

    def test_detects_serial(self):
        reader, config = open_ipic3d(SHDF5_DIR)
        assert isinstance(reader, IPic3DSerialReader)
        assert config.model_name == "iPIC3D"

    def test_detects_h5hut(self):
        reader, config = open_ipic3d(H5HUT_DIR)
        assert isinstance(reader, IPic3DH5hutReader)
        assert config.model_name == "iPIC3D"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_ipic3d(tmp_path)


class TestConservedQuantitiesSyntheticFormatA:
    """Test Format A (Roman numeral header) from synthetic data."""

    @pytest.fixture(scope="class")
    def cq(self):
        return load_conserved_quantities(DATA / "phdf5" / "ConservedQuantities.txt")

    def test_cycle_count(self, cq):
        assert len(cq.cycle) == 3

    def test_first_cycle_zero(self, cq):
        assert cq.cycle[0] == 0

    def test_cycles_sorted(self, cq):
        assert list(cq.cycle) == sorted(cq.cycle)

    def test_no_species_data(self, cq):
        assert len(cq.species_npart) == 0

    def test_energies_present(self, cq):
        assert len(cq.total_energy) == 3
        assert len(cq.electric_energy) == 3
        assert len(cq.magnetic_energy) == 3

    def test_total_energy_values(self, cq):
        assert_allclose(cq.total_energy[0], 5.5)
        assert_allclose(cq.total_energy[1], 5.6)
        assert_allclose(cq.total_energy[2], 5.7)


class TestConservedQuantitiesSyntheticFormatB:
    """Test Format B (comment header, multi-file) from synthetic data."""

    @pytest.fixture(scope="class")
    def cq(self):
        return load_conserved_quantities(DATA / "h5hut" / "info-conserved")

    def test_cycle_count(self, cq):
        assert len(cq.cycle) == 5  # 0, 5, 10(deduped), 15, 20

    def test_cycles_sorted(self, cq):
        diffs = np.diff(cq.cycle)
        assert np.all(diffs > 0)

    def test_cycles_deduplicated(self, cq):
        assert len(np.unique(cq.cycle)) == len(cq.cycle)

    def test_dedup_keeps_later_file(self, cq):
        """Cycle 10 appears in both files; file 1 value should win."""
        idx_10 = int(np.searchsorted(cq.cycle, 10))
        assert_allclose(cq.total_energy[idx_10], 5.71)

    def test_species_count(self, cq):
        assert len(cq.species_npart) == 2
        assert len(cq.species_charge) == 2
        assert len(cq.species_kinetic_energy) == 2

    def test_species_arrays_length(self, cq):
        for arr in cq.species_npart:
            assert len(arr) == len(cq.cycle)

    def test_particle_counts(self, cq):
        for arr in cq.species_npart:
            assert np.all(arr == 1000)


class TestConservedToTabular:
    """Test conversion from ConservedQuantities to TabularData."""

    @pytest.fixture(scope="class")
    def tab(self):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        return conserved_to_tabular(cq)

    def test_is_tabular_data(self, tab):
        assert isinstance(tab, TabularData)

    def test_name(self, tab):
        assert tab.name == "conserved_quantities"

    def test_index_column(self, tab):
        assert tab.index_column == "cycle"

    def test_scalar_columns_present(self, tab):
        for col in (
            "cycle",
            "total_energy",
            "electric_energy",
            "magnetic_energy",
            "kinetic_energy",
            "momentum",
        ):
            assert col in tab

    def test_species_columns_flattened(self, tab):
        for s in range(2):
            assert f"npart_s{s}" in tab
            assert f"charge_s{s}" in tab
            assert f"kinetic_energy_s{s}" in tab

    def test_scalar_values_match(self, tab):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        assert_allclose(tab["total_energy"], cq.total_energy)
        assert_allclose(tab["electric_energy"], cq.electric_energy)
        assert_allclose(tab["momentum"], cq.momentum)

    def test_species_values_match(self, tab):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        for s in range(len(cq.species_npart)):
            assert_allclose(tab[f"npart_s{s}"], cq.species_npart[s])

    def test_row_count(self, tab):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        assert len(tab) == len(cq.cycle)


class TestIPic3DAuxiliaryProtocol:
    """Test that iPIC3D readers satisfy AuxiliaryDataReader."""

    def test_parallel_is_auxiliary_reader(self):
        assert isinstance(
            IPic3DParallelReader,
            type,
        )
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        assert isinstance(reader, AuxiliaryDataReader)

    def test_serial_is_auxiliary_reader(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        assert isinstance(reader, AuxiliaryDataReader)

    def test_h5hut_is_auxiliary_reader(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        assert isinstance(reader, AuxiliaryDataReader)


class TestIPic3DAvailableAuxiliary:
    """Test auxiliary detection on synthetic data directories."""

    def test_phdf5_detects_conserved(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        names = reader.available_auxiliary(PHDF5_DIR)
        assert "conserved_quantities" in names

    def test_h5hut_detects_conserved(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        names = reader.available_auxiliary(H5HUT_DIR)
        assert "conserved_quantities" in names

    def test_empty_dir_returns_empty(self, tmp_path):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        assert reader.available_auxiliary(tmp_path) == []


class TestIPic3DLoadAuxiliary:
    """Test loading auxiliary data through the reader interface."""

    def test_phdf5_load_conserved(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        tab = reader.load_auxiliary(PHDF5_DIR, "conserved_quantities")
        assert isinstance(tab, TabularData)
        assert tab.name == "conserved_quantities"
        assert "total_energy" in tab

    def test_h5hut_load_conserved(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        tab = reader.load_auxiliary(H5HUT_DIR, "conserved_quantities")
        assert isinstance(tab, TabularData)
        assert len(tab) > 0

    def test_unknown_name_raises(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        with pytest.raises(KeyError, match="unknown_dataset"):
            reader.load_auxiliary(PHDF5_DIR, "unknown_dataset")
