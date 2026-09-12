"""Tests for iPIC3D readers against synthetic fixtures with exact assertions.

All three reader formats (phdf5, shdf5, H5hut) are tested against
analytically known values from the fixture generator. Fixtures are
committed to ``tests/data/ipic3d-synthetic/`` (~200 KB).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pypic.containers import TabularData
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
from pypic.readers.ipic3d._conserved import load_ipic3d_auxiliary
from tests._helpers import per_species

DATA = Path(__file__).resolve().parent / "data" / "ipic3d-synthetic"
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
U0 = (0.0, 0.0)
V0 = (0.0, 0.0)
W0 = (0.001, -0.064)
RHO_INIT = (1.0, 1.0)


def _expected_b1() -> np.ndarray:
    """Analytical B_1 field: B0x * tanh((y - Ly/2) / delta)."""
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

    def test_grid_and_domain(self, cfg):
        assert cfg.nxc == NXC
        assert cfg.nyc == NYC
        assert cfg.nzc == NZC
        assert cfg.lx == LX
        assert cfg.ly == LY
        assert cfg.lz == LZ
        assert_array_equal(cfg.dx, DX)
        assert_array_equal(cfg.dy, DY)
        assert_array_equal(cfg.dz, DZ)
        assert cfg.periodic_x is True
        assert cfg.periodic_y is True
        assert cfg.periodic_z is True

    def test_species_and_physics(self, cfg):
        assert cfg.ns == 2
        assert cfg.qom == QOM
        assert cfg.b0 == (B0X, 0.0, 0.0)
        assert cfg.write_method == "phdf5"

    def test_inline_comment_stripped(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        assert cfg.write_method == "h5hut"


class TestParseSettingsHdfSynthetic:
    def test_settings_hdf_matches_inp(self):
        cfg_hdf = parse_settings_hdf(SHDF5_DIR / "settings.hdf")
        cfg_inp = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        assert cfg_hdf.nxc == cfg_inp.nxc
        assert cfg_hdf.nyc == cfg_inp.nyc
        assert cfg_hdf.nzc == cfg_inp.nzc
        assert_array_equal(cfg_hdf.lx, cfg_inp.lx)
        assert_array_equal(cfg_hdf.ly, cfg_inp.ly)
        assert_array_equal(cfg_hdf.lz, cfg_inp.lz)
        assert cfg_hdf.qom == cfg_inp.qom
        assert cfg_hdf.xlen == 2
        assert cfg_hdf.ylen == 1
        assert cfg_hdf.zlen == 1
        assert_array_equal(cfg_hdf.uth, cfg_inp.uth)
        assert_array_equal(cfg_hdf.vth, cfg_inp.vth)
        assert_array_equal(cfg_hdf.wth, cfg_inp.wth)


class TestToSimulationConfigSynthetic:
    @pytest.fixture
    def sim_cfg(self):
        return to_simulation_config(parse_inp(PHDF5_DIR / "synthetic.inp"))

    def test_grid_geometry(self, sim_cfg):
        assert sim_cfg.grid.dimensions == (NX, NY, NZ)
        assert_array_equal(sim_cfg.grid.origin, (-DX / 2, -DY / 2, -DZ / 2))
        x, y, z = sim_cfg.grid.coordinate_arrays()
        assert_allclose(x[0], 0.0, atol=1e-14)
        assert_allclose(x[-1], LX, atol=1e-14)
        assert_allclose(y[0], 0.0, atol=1e-14)
        assert_allclose(y[-1], LY, atol=1e-14)
        assert_allclose(z[0], 0.0, atol=1e-14)
        assert_allclose(z[-1], LZ, atol=1e-14)
        assert sim_cfg.grid.boundary == ("periodic", "periodic", "periodic")

    def test_species_and_model(self, sim_cfg):
        assert len(sim_cfg.species) == 2
        qoms = [s.charge_to_mass for s in sim_cfg.species]
        assert qoms == list(QOM)
        electron = sim_cfg.species[0]
        assert electron.charge == -1.0
        assert_array_equal(electron.mass, 1.0 / 64.0)
        assert sim_cfg.model_name == "iPIC3D"
        assert sim_cfg.model_type == "PIC"

    def test_stagger_metadata(self, sim_cfg):
        stagger = sim_cfg.metadata["stagger"]
        assert stagger.convention == "node"

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

    def test_available_fields_matches_read(self, ds):
        """Lightweight probe returns the same fields as a full read."""
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        probed = reader.available_fields(PHDF5_DIR, 0)
        assert probed == sorted(ds.field_names())

    def test_field_shapes(self, ds):
        assert ds["B_1"].shape == (NX, NY, NZ)

    def test_b_field_exact(self, ds):
        assert_allclose(ds["B_1"], _expected_b1(), atol=1e-14)

    def test_e_field_zero(self, ds):
        for comp in ("E_1", "E_2", "E_3"):
            assert_allclose(ds[comp], 0.0, atol=1e-14)

    def test_electron_density_exact(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-12)

    def test_ion_density_exact(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-12)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_array_equal(ds["rho_c"], total)

    def test_total_current_is_sum(self, ds):
        for comp in ("J_1", "J_2", "J_3"):
            total = ds[per_species(comp, 0)] + ds[per_species(comp, 1)]
            assert_array_equal(ds[comp], total)

    @pytest.mark.parametrize("species", [0, 1])
    def test_diagonal_pressure_positive(self, ds, species):
        for comp in ("P_11", "P_22", "P_33"):
            p = ds[per_species(comp, species)]
            assert np.all(p >= 0), f"{comp}_s{species} has negative values"

    def test_pressure_values(self, ds):
        """Exact P_11 values: physical P = n·m·v_th² = |rho_c|·v_th²/|qom|."""
        for s, (rho, uth, qom) in enumerate(zip(RHO_INIT, UTH, QOM, strict=True)):
            expected = rho * uth**2 / abs(qom)
            assert_allclose(ds[f"P_s{s}_11"], expected, atol=1e-14)
            p = ds[f"P_s{s}_11"]
            rho_c = ds[f"rho_c_s{s}"]
            ratio = np.mean(p) / np.mean(np.abs(rho_c))
            assert_allclose(ratio, uth**2 / abs(qom), atol=1e-14)

    def test_off_diagonal_pressure_zero(self, ds):
        for s in range(2):
            for comp in ("P_12", "P_13", "P_23"):
                assert_allclose(ds[per_species(comp, s)], 0.0, atol=1e-12)

    def test_species_metadata(self, ds):
        # Count-only is weak: a reader that populated garbage species (wrong
        # charge sign, swapped mass) still passes.  Pin the canonical iPIC3D
        # convention: species 0 = electrons (q=-1, m=1/|qom_0|=1/64),
        # species 1 = ions (q=+1, m=1/|qom_1|=1).
        assert len(ds.species) == 2
        assert ds.species[0].charge == -1.0
        assert_array_equal(ds.species[0].mass, 1.0 / 64.0)
        assert ds.species[1].charge == 1.0
        assert_array_equal(ds.species[1].mass, 1.0)

    def test_grid_info(self, ds):
        assert ds.grid.dimensions == (NX, NY, NZ)
        assert_array_equal(ds.grid.spacing, (DX, DY, DZ))


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

    def test_available_fields_matches_read(self, ds):
        """Lightweight probe returns the same fields as a full read."""
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        probed = reader.available_fields(SHDF5_DIR, 0)
        assert probed == sorted(ds.field_names())

    def test_field_shapes(self, ds):
        assert ds["B_1"].shape == (NX, NY, NZ)

    def test_b_field_exact(self, ds):
        assert_allclose(ds["B_1"], _expected_b1(), atol=1e-14)

    def test_electron_density_exact(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-12)

    def test_ion_density_exact(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-12)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_array_equal(ds["rho_c"], total)

    def test_per_species_present(self, ds):
        for s in range(2):
            assert ds.has_field(f"rho_c_s{s}")
            for comp in ("J_1", "J_2", "J_3"):
                assert ds.has_field(per_species(comp, s))


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
        for comp in ("B_1", "B_2", "B_3"):
            assert_allclose(shdf5_ds[comp], phdf5_ds[comp], atol=1e-14)

    def test_e_field_matches(self, phdf5_ds, shdf5_ds):
        for comp in ("E_1", "E_2", "E_3"):
            assert_allclose(shdf5_ds[comp], phdf5_ds[comp], atol=1e-14)

    def test_density_matches(self, phdf5_ds, shdf5_ds):
        for s in range(2):
            assert_allclose(
                shdf5_ds[f"rho_c_s{s}"],
                phdf5_ds[f"rho_c_s{s}"],
                atol=1e-12,
            )

    def test_current_matches(self, phdf5_ds, shdf5_ds):
        for comp in ("J_1", "J_2", "J_3"):
            for s in range(2):
                assert_allclose(
                    shdf5_ds[per_species(comp, s)],
                    phdf5_ds[per_species(comp, s)],
                    atol=1e-12,
                )


class TestH5hutSpeciesCount:
    def test_mismatch_with_config_raises(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        bad = copy.replace(cfg, ns=cfg.ns - 1)
        with pytest.raises(ValueError, match="nspec"):
            IPic3DH5hutReader(bad).read_timestep(H5HUT_DIR, 0)


class TestPartialSpeciesTotals:
    """A total summed over only some species is wrong physics; it must raise."""

    def test_missing_species_moment_raises(self):
        from pypic.readers.ipic3d._field_map import compute_totals_and_filter

        data = {"rho_c_s0": np.ones(3), "J_s0_1": np.ones(3), "J_s1_1": np.ones(3)}
        with pytest.raises(ValueError, match=r"Cannot total 'rho_c'.*rho_c_s1"):
            compute_totals_and_filter(data, 2, None, None)

    def test_phdf5_missing_species_file_surfaces(self, tmp_path: Path):
        import shutil

        shutil.copytree(PHDF5_DIR, tmp_path / "run")
        (tmp_path / "run" / "Moments_00000" / "rho_species_1_00000.h5").unlink()
        reader = IPic3DParallelReader(parse_inp(tmp_path / "run" / "synthetic.inp"))
        with pytest.raises(ValueError, match="rho_c_s1"):
            reader.read_timestep(tmp_path / "run", 0)


class TestH5hutMatchesPhdf5:
    """Cross-format validation: H5hut must match phdf5 after 4π correction."""

    @pytest.fixture(scope="class")
    def phdf5_ds(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        return reader.read_timestep(PHDF5_DIR, 0)

    @pytest.fixture(scope="class")
    def h5hut_ds(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        return reader.read_timestep(H5HUT_DIR, 0)

    def test_b_field_matches(self, phdf5_ds, h5hut_ds):
        for comp in ("B_1", "B_2", "B_3"):
            assert_allclose(h5hut_ds[comp], phdf5_ds[comp], atol=1e-6)

    def test_e_field_matches(self, phdf5_ds, h5hut_ds):
        for comp in ("E_1", "E_2", "E_3"):
            assert_allclose(h5hut_ds[comp], phdf5_ds[comp], atol=1e-6)

    def test_density_matches(self, phdf5_ds, h5hut_ds):
        for s in range(2):
            assert_allclose(
                h5hut_ds[f"rho_c_s{s}"],
                phdf5_ds[f"rho_c_s{s}"],
                atol=1e-5,
            )

    def test_current_matches(self, phdf5_ds, h5hut_ds):
        for comp in ("J_1", "J_2", "J_3"):
            for s in range(2):
                assert_allclose(
                    h5hut_ds[per_species(comp, s)],
                    phdf5_ds[per_species(comp, s)],
                    atol=1e-5,
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

    def test_available_fields_matches_read(self, ds):
        """Lightweight probe returns the same fields as a full read."""
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        probed = reader.available_fields(H5HUT_DIR, 0)
        assert probed == sorted(ds.field_names())

    def test_field_shapes(self, ds):
        assert ds["B_1"].shape == (NX, NY, NZ)

    def test_float64_promotion(self, ds):
        """H5hut stores float32; reader must promote to float64.

        Previously B_1-only — a reader that promoted B_1 but left B_2/B_3 (or E*,
        rho_c_*) as float32 still passed.  Check every loaded field: a dtype
        regression in a single branch of the promotion path is now visible.
        """
        for name in ds.field_names():
            assert ds[name].dtype == np.float64, f"{name} not promoted to float64"

    def test_b_field_after_transpose(self, ds):
        """ZYX→XYZ transpose + float32→float64 must recover correct values."""
        assert_allclose(ds["B_1"], _expected_b1(), atol=1e-6)

    def test_e_field_zero(self, ds):
        for comp in ("E_1", "E_2", "E_3"):
            assert_allclose(ds[comp], 0.0, atol=1e-6)

    def test_electron_density_4pi_corrected(self, ds):
        assert_allclose(ds["rho_c_s0"], -RHO_INIT[0], atol=1e-5)

    def test_ion_density_4pi_corrected(self, ds):
        assert_allclose(ds["rho_c_s1"], RHO_INIT[1], atol=1e-5)

    def test_total_charge_is_sum(self, ds):
        total = ds["rho_c_s0"] + ds["rho_c_s1"]
        assert_array_equal(ds["rho_c"], total)

    def test_total_current_is_sum(self, ds):
        for comp in ("J_1", "J_2", "J_3"):
            total = ds[per_species(comp, 0)] + ds[per_species(comp, 1)]
            assert_array_equal(ds[comp], total)

    def test_diagonal_pressure_positive(self, ds):
        # H5hut has no test_pressure_values_match_phdf5 for pressure (unlike
        # shdf5), and test_pressure_p_over_rho_consistency divides P by rho
        # so a common sign flip in both cancels in the ratio.  Positivity
        # is the residual discriminator — extend to ions so a sign flip in
        # the ion-specific 4π correction branch is caught.
        for s in range(2):
            for comp in ("P_11", "P_22", "P_33"):
                assert ds[per_species(comp, s)].min() >= 0.0, (
                    f"{comp}_s{s} has negative values"
                )

    def test_pressure_p_over_rho_consistency(self, ds):
        """Physical P/|rho_c| = v_th²/|qom| (mass-weighted pressure)."""
        for s, (uth, qom) in enumerate(zip(UTH, QOM, strict=True)):
            p = ds[f"P_s{s}_11"]
            rho = ds[f"rho_c_s{s}"]
            ratio = np.mean(p) / np.mean(np.abs(rho))
            assert_allclose(ratio, uth**2 / abs(qom), atol=1e-6)


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
                "0",
                data=np.full(shape, 99.0, dtype=np.float32),
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

    def test_cycle_properties(self, cq):
        # Previously only pinned cycle[0]==0 and sorted(); a parser that
        # returned [0, 0, 0] satisfies both (sorted allows duplicates) and
        # paired test_energies still passes because energies are indexed
        # positionally.  Pin each cycle explicitly against the fixture.
        assert len(cq.cycle) == 3
        assert cq.cycle[0] == 0
        assert cq.cycle[1] == 5
        assert cq.cycle[2] == 10
        assert list(cq.cycle) == sorted(cq.cycle)
        assert len(cq.species_npart) == 0

    def test_energies(self, cq):
        assert len(cq.total_energy) == 3
        assert len(cq.electric_energy) == 3
        assert len(cq.magnetic_energy) == 3
        assert_array_equal(cq.total_energy[0], 5.5)
        assert_array_equal(cq.total_energy[1], 5.6)
        assert_array_equal(cq.total_energy[2], 5.7)


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
        assert_array_equal(cq.total_energy[idx_10], 5.71)

    def test_species_count(self, cq):
        assert len(cq.species_npart) == 2
        assert len(cq.species_charge) == 2
        assert len(cq.species_kinetic_energy) == 2

    def test_species_arrays_length(self, cq):
        # Originally only species_npart was checked.  A parser bug that
        # truncates species_charge or species_kinetic_energy (e.g. skipping
        # the last row on one of the per-species columns) slipped through.
        expected = len(cq.cycle)
        for family in (
            cq.species_npart,
            cq.species_charge,
            cq.species_kinetic_energy,
        ):
            for arr in family:
                assert len(arr) == expected

    def test_particle_counts(self, cq):
        for arr in cq.species_npart:
            assert np.all(arr == 1000)


class TestConservedToTabular:
    """Test conversion from ConservedQuantities to TabularData."""

    @pytest.fixture(scope="class")
    def tab(self):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        return conserved_to_tabular(cq)

    def test_structure(self, tab):
        assert tab.name == "conserved_quantities"
        assert tab.index_column == "cycle"
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        assert len(tab) == len(cq.cycle)

    def test_columns_present(self, tab):
        for col in (
            "cycle",
            "total_energy",
            "electric_energy",
            "magnetic_energy",
            "kinetic_energy",
            "momentum",
        ):
            assert col in tab
        for s in range(2):
            assert f"npart_s{s}" in tab
            assert f"charge_s{s}" in tab
            assert f"kinetic_energy_s{s}" in tab

    def test_values_match(self, tab):
        cq = load_conserved_quantities(DATA / "h5hut" / "info-conserved")
        assert_array_equal(tab["total_energy"], cq.total_energy)
        assert_array_equal(tab["electric_energy"], cq.electric_energy)
        assert_array_equal(tab["momentum"], cq.momentum)
        for s in range(len(cq.species_npart)):
            assert_array_equal(tab[f"npart_s{s}"], cq.species_npart[s])


class TestAuxiliaryErrors:
    """Error-type contract for ``load_ipic3d_auxiliary``.

    Convention:
    - ``KeyError``        → unknown dataset name
    - ``FileNotFoundError`` → recognized name, missing files on disk
    """

    def test_unknown_name_raises_key_error(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="Unknown iPIC3D auxiliary"):
            load_ipic3d_auxiliary(tmp_path, "no_such_dataset")

    def test_missing_conserved_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ConservedQuantities"):
            load_ipic3d_auxiliary(tmp_path, "conserved_quantities")

    def test_missing_species_quantities_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="SpeciesQuantities"):
            load_ipic3d_auxiliary(tmp_path, "species_quantities")


class TestSelectiveReadPhdf5:
    def test_b_fields_only(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        ds = reader.read_timestep(PHDF5_DIR, 0, fields={"B_1", "B_2", "B_3"})
        assert sorted(ds.field_names()) == ["B_1", "B_2", "B_3"]

    def test_values_match_full_read(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        full = reader.read_timestep(PHDF5_DIR, 0)
        sub = reader.read_timestep(PHDF5_DIR, 0, fields={"B_1"})
        assert_array_equal(sub["B_1"], full["B_1"])

    def test_total_expands_dependencies(self):
        """Requesting rho_c reads per-species rho and computes total."""
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        ds = reader.read_timestep(PHDF5_DIR, 0, fields={"rho_c"})
        assert ds.has_field("rho_c")
        # Per-species fields are NOT in final result
        assert not ds.has_field("rho_c_s0")
        assert not ds.has_field("rho_c_s1")

    def test_total_value_correct(self):
        """Total computed from selective read matches full read."""
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        full = reader.read_timestep(PHDF5_DIR, 0)
        sub = reader.read_timestep(PHDF5_DIR, 0, fields={"rho_c"})
        assert_array_equal(sub["rho_c"], full["rho_c"])

    def test_per_species_without_total(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        ds = reader.read_timestep(
            PHDF5_DIR,
            0,
            fields={"rho_c_s0", "B_1"},
        )
        assert ds.has_field("rho_c_s0")
        assert ds.has_field("B_1")
        assert not ds.has_field("rho_c")


class TestSelectiveReadShdf5:
    def test_b_fields_only(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        ds = reader.read_timestep(SHDF5_DIR, 0, fields={"B_1", "B_2", "B_3"})
        assert sorted(ds.field_names()) == ["B_1", "B_2", "B_3"]

    def test_total_value_correct(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        full = reader.read_timestep(SHDF5_DIR, 0)
        sub = reader.read_timestep(SHDF5_DIR, 0, fields={"J_1"})
        assert_array_equal(sub["J_1"], full["J_1"])


class TestSelectiveReadH5hut:
    def test_b_fields_only(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        ds = reader.read_timestep(H5HUT_DIR, 0, fields={"B_1", "B_2", "B_3"})
        assert sorted(ds.field_names()) == ["B_1", "B_2", "B_3"]

    def test_total_expands_dependencies(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        ds = reader.read_timestep(H5HUT_DIR, 0, fields={"rho_c"})
        assert ds.has_field("rho_c")
        assert not ds.has_field("rho_c_s0")

    def test_total_value_correct(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        full = reader.read_timestep(H5HUT_DIR, 0)
        sub = reader.read_timestep(H5HUT_DIR, 0, fields={"rho_c"})
        assert_array_equal(sub["rho_c"], full["rho_c"])

    def test_em_and_moment_mix(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        ds = reader.read_timestep(
            H5HUT_DIR,
            0,
            fields={"B_1", "rho_c_s0", "J_s0_1"},
        )
        assert sorted(ds.field_names()) == ["B_1", "J_s0_1", "rho_c_s0"]


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


class TestShdf5PressureTensor:
    """Verify pressure tensor is read correctly from serial HDF5."""

    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader = IPic3DSerialReader(cfg)
        return reader.read_timestep(SHDF5_DIR, 0)

    def test_pressure_values_match_phdf5(self, ds):
        """shdf5 pressure should match phdf5 exactly."""
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        ds_p = reader.read_timestep(PHDF5_DIR, 0)
        for s in range(2):
            for comp in ("P_11", "P_22", "P_33", "P_12", "P_13", "P_23"):
                assert_allclose(
                    ds[per_species(comp, s)],
                    ds_p[per_species(comp, s)],
                    atol=1e-12,
                )

    def test_off_diagonal_pressure_zero(self, ds):
        for s in range(2):
            for comp in ("P_12", "P_13", "P_23"):
                assert_allclose(ds[per_species(comp, s)], 0.0, atol=1e-12)


class TestEnergyFluxPhdf5:
    """Verify energy flux reading from phdf5 format."""

    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        return reader.read_timestep(PHDF5_DIR, 0)

    def test_eflux_present(self, ds):
        for s in range(2):
            for comp in ("EF_1", "EF_2", "EF_3"):
                assert ds.has_field(per_species(comp, s))

    def test_eflux_shape(self, ds):
        assert ds["EF_s0_1"].shape == (NX, NY, NZ)

    def test_eflux_4pi_corrected(self, ds):
        """Energy flux should be 4π-corrected like other moments.

        Fixture convention: per-species EF_i is a drifting Maxwellian's
        i-th energy-flux component, which for this fixture reduces to
        ``rho * u_i * th_i**2`` (with u_i the bulk drift and th_i the
        thermal speed along axis i).  The pair (EF_1, EF_3) pins both
        the vacuous-when-drift-zero branch (U0=0) *and* the 4π factor:
        without the 4π correction, EF_3 would differ by ~12.57×.
        """
        # EF_1 vacuously zero (U0 = 0)
        expected_ef1 = RHO_INIT[0] * U0[0] * UTH[0] ** 2
        assert_allclose(ds["EF_s0_1"], expected_ef1, atol=1e-12)
        # EF_3 non-trivial: W0=0.001, WTH=0.02 → 4e-7 (species 0)
        expected_ef3_s0 = RHO_INIT[0] * W0[0] * WTH[0] ** 2
        assert_allclose(ds["EF_s0_3"], expected_ef3_s0, atol=1e-12)
        # EF_s1_3 negative drift (W0=-0.064) pins sign preservation
        expected_ef3_s1 = RHO_INIT[1] * W0[1] * WTH[1] ** 2
        assert_allclose(ds["EF_s1_3"], expected_ef3_s1, atol=1e-12)


class TestEnergyFluxShdf5:
    """Verify energy flux reading from serial HDF5."""

    def test_eflux_matches_phdf5(self):
        cfg_s = parse_inp(SHDF5_DIR / "synthetic_serial.inp")
        reader_s = IPic3DSerialReader(cfg_s)
        ds_s = reader_s.read_timestep(SHDF5_DIR, 0)

        cfg_p = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader_p = IPic3DParallelReader(cfg_p)
        ds_p = reader_p.read_timestep(PHDF5_DIR, 0)

        for s in range(2):
            for comp in ("EF_1", "EF_2", "EF_3"):
                assert_allclose(
                    ds_s[per_species(comp, s)],
                    ds_p[per_species(comp, s)],
                    atol=1e-12,
                )


class TestEnergyFluxH5hut:
    """Verify energy flux reading from H5hut format."""

    @pytest.fixture(scope="class")
    def ds(self):
        cfg = parse_inp(H5HUT_DIR / "SyntheticFixture.inp")
        reader = IPic3DH5hutReader(cfg)
        return reader.read_timestep(H5HUT_DIR, 0)

    def test_eflux_present(self, ds):
        for s in range(2):
            assert ds.has_field(f"EF_s{s}_1")

    def test_eflux_matches_phdf5(self, ds):
        cfg_p = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader_p = IPic3DParallelReader(cfg_p)
        ds_p = reader_p.read_timestep(PHDF5_DIR, 0)
        for s in range(2):
            for comp in ("EF_1", "EF_2", "EF_3"):
                assert_allclose(
                    ds[per_species(comp, s)],
                    ds_p[per_species(comp, s)],
                    atol=1e-5,
                )


class TestFieldOutputTag:
    """Verify FieldOutputTag and ParticlesOutputCycle parsing."""

    def test_field_output_tag_parsed(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        assert "pressure" in cfg.field_output_tag
        assert "E_flux" in cfg.field_output_tag

    def test_particles_output_cycle_parsed(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        assert cfg.particles_output_cycle == 10

    def test_field_output_tag_in_metadata(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        sim_cfg = to_simulation_config(cfg)
        # Presence + value: catches both dropped-key and wrong-value bugs.
        assert "pressure" in sim_cfg.metadata["field_output_tag"]
        assert "E_flux" in sim_cfg.metadata["field_output_tag"]
        assert sim_cfg.metadata["particles_output_cycle"] == 10

    def test_settings_hdf_defaults(self):
        cfg = parse_settings_hdf(SHDF5_DIR / "settings.hdf")
        assert cfg.field_output_tag == ""
        assert cfg.particles_output_cycle == 0


class TestSpeciesQuantities:
    """Verify SpeciesQuantities.txt parsing."""

    def test_load_species_quantities(self):
        from pypic.readers.ipic3d import load_species_quantities

        tab = load_species_quantities(PHDF5_DIR / "SpeciesQuantities.txt")
        assert tab.name == "species_quantities"
        assert tab.index_column == "cycle"
        assert len(tab) == 3  # 3 cycles

    def test_columns_present(self):
        from pypic.readers.ipic3d import load_species_quantities

        tab = load_species_quantities(PHDF5_DIR / "SpeciesQuantities.txt")
        for s in range(2):
            assert f"momentum_s{s}" in tab
            assert f"total_ke_s{s}" in tab
            assert f"bulk_ke_s{s}" in tab
            assert f"thermal_ke_s{s}" in tab

    def test_values_correct(self):
        from pypic.readers.ipic3d import load_species_quantities

        tab = load_species_quantities(PHDF5_DIR / "SpeciesQuantities.txt")
        assert_allclose(tab["momentum_s0"][0], 1.00e-03, rtol=1e-10)
        assert_allclose(tab["thermal_ke_s1"][0], 0.220, rtol=1e-10)

    def test_detect_species_quantities(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        names = reader.available_auxiliary(PHDF5_DIR)
        assert "species_quantities" in names

    def test_load_via_auxiliary_interface(self):
        cfg = parse_inp(PHDF5_DIR / "synthetic.inp")
        reader = IPic3DParallelReader(cfg)
        tab = reader.load_auxiliary(PHDF5_DIR, "species_quantities")
        assert isinstance(tab, TabularData)
        assert "momentum_s0" in tab


class TestProbeImprovements:
    """Verify Moments_*/Particles_* directory detection."""

    def test_phdf5_with_moments_higher_confidence(self):
        from pypic.readers.ipic3d import can_read_confidence

        # phdf5 dir has .inp (0.5) + Moments_* (+0.15) + Particles_* (+0.1).
        # The Fields_* fallback is suppressed because core score is already
        # non-zero.  Pin the exact sum — a loose >= bound wouldn't catch a
        # dropped reinforcement signal.
        score = can_read_confidence(PHDF5_DIR)
        assert score == pytest.approx(0.75)

    def test_moments_detected_in_phdf5(self, tmp_path):
        from pypic.readers.ipic3d import can_read_confidence

        # Directory with only Fields_* and Moments_* (no .inp).  The
        # Fields_ fallback (0.2) plus Moments_ reinforcement (0.15)
        # must sum to exactly 0.35 — loose bounds would accept either
        # signal alone and miss the other.
        (tmp_path / "Fields_00000").mkdir()
        (tmp_path / "Moments_00000").mkdir()
        score = can_read_confidence(tmp_path)
        assert score == pytest.approx(0.35)

    def test_particles_detected_in_phdf5(self, tmp_path):
        from pypic.readers.ipic3d import can_read_confidence

        (tmp_path / "Fields_00000").mkdir()
        (tmp_path / "Particles_00000").mkdir()
        score = can_read_confidence(tmp_path)
        # Fields_ fallback (0.2) + Particles_ reinforcement (0.1) = 0.30.
        assert score == pytest.approx(0.30)
