"""Tests for iPIC3D particle data reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pypic.containers import ParticleData
from pypic.readers._protocols import ParticleDataReader
from pypic.readers._registry import Simulation
from pypic.readers.ipic3d import (
    IPic3DParallelReader,
    detect_particle_steps,
    parse_inp,
    read_phdf5_particles,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "ipic3d-synthetic" / "phdf5"


class TestParticleData:
    """Unit tests for the ParticleData dataclass."""

    def test_construction_and_properties(self) -> None:
        pos = np.zeros((10, 3))
        vel = np.ones((10, 3))
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=pos,
            velocity=vel,
            n_particles=10,
            metadata={"fmt": "test"},
            weight=np.ones(10),
            species_charge=-1.0,
            species_mass=1.0,
        )
        assert pcl.species_index == 0
        assert pcl.species_name == "electrons"
        assert pcl.n_particles == 10

    def test_position_only(self) -> None:
        pcl = ParticleData(
            species_index=1,
            species_name="ions",
            position=np.zeros((5, 3)),
            velocity=None,
            n_particles=5,
            metadata={},
            weight=np.ones(5),
            species_charge=1.0,
            species_mass=1.0,
        )
        assert pcl.position is not None
        assert pcl.velocity is None

    def test_velocity_only(self) -> None:
        pcl = ParticleData(
            species_index=1,
            species_name="ions",
            position=None,
            velocity=np.ones((5, 3)),
            n_particles=5,
            metadata={},
        )
        assert pcl.position is None
        assert pcl.velocity is not None

    def test_xyz_vxyz_are_views(self) -> None:
        pos = np.arange(15, dtype=np.float64).reshape(5, 3)
        vel = np.arange(15, 30, dtype=np.float64).reshape(5, 3)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=pos,
            velocity=vel,
            n_particles=5,
            metadata={},
        )
        assert np.shares_memory(pcl.x, pos)
        assert np.shares_memory(pcl.y, pos)
        assert np.shares_memory(pcl.z, pos)
        assert np.shares_memory(pcl.vx, vel)
        assert np.shares_memory(pcl.vy, vel)
        assert np.shares_memory(pcl.vz, vel)
        np.testing.assert_array_equal(pcl.x, pos[:, 0])
        np.testing.assert_array_equal(pcl.vy, vel[:, 1])

    def test_access_unloaded_raises(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=None,
            velocity=np.ones((3, 3)),
            n_particles=3,
            metadata={},
        )
        with pytest.raises(ValueError, match="position was not loaded"):
            _ = pcl.x
        with pytest.raises(ValueError, match="position was not loaded"):
            _ = pcl.y

        pcl2 = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={},
        )
        with pytest.raises(ValueError, match="velocity was not loaded"):
            _ = pcl2.vx

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="position shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=np.zeros((5, 3)),
                velocity=None,
                n_particles=3,
                metadata={},
            )
        with pytest.raises(ValueError, match="velocity shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=None,
                velocity=np.ones((2, 3)),
                n_particles=3,
                metadata={},
            )
        with pytest.raises(ValueError, match="weight shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=np.zeros((3, 3)),
                velocity=None,
                n_particles=3,
                metadata={},
                weight=np.ones(5),
            )

    def test_weight_dtype_float64(self) -> None:
        with pytest.raises(ValueError, match="float64"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=np.zeros((3, 3)),
                velocity=None,
                n_particles=3,
                metadata={},
                weight=np.ones(3, dtype=np.float32),
            )

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=None,
                velocity=None,
                n_particles=3,
                metadata={},
            )

    def test_metadata_frozen(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={"key": "val"},
        )
        with pytest.raises(TypeError):
            pcl.metadata["new"] = "nope"  # type: ignore[index]

    def test_repr(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=np.zeros((10, 3)),
            velocity=None,
            n_particles=10,
            metadata={},
            weight=np.ones(10),
            species_charge=-1.0,
            species_mass=1.0,
        )
        r = repr(pcl)
        assert "electrons" in r
        assert "10" in r
        assert "position" in r
        assert "weight" in r
        assert "species_charge" in r

    def test_len(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((7, 3)),
            velocity=None,
            n_particles=7,
            metadata={},
        )
        assert len(pcl) == 7


class TestMacroProperties:
    """Tests for the macro_charge / macro_mass derived properties."""

    def test_macro_charge(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={},
            weight=np.array([1.0, 2.0, 3.0]),
            species_charge=-1.0,
        )
        np.testing.assert_array_equal(pcl.macro_charge, np.array([-1.0, -2.0, -3.0]))

    def test_macro_charge_missing_raises(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={},
        )
        with pytest.raises(ValueError, match="macro_charge"):
            _ = pcl.macro_charge

    def test_macro_mass(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="i",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={},
            weight=np.array([1.0, 2.0, 4.0]),
            species_mass=2.5,
        )
        np.testing.assert_array_equal(pcl.macro_mass, np.array([2.5, 5.0, 10.0]))

    def test_macro_mass_missing_raises(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="i",
            position=np.zeros((3, 3)),
            velocity=None,
            n_particles=3,
            metadata={},
            weight=np.full(3, 1.0),
        )
        with pytest.raises(ValueError, match="macro_mass"):
            _ = pcl.macro_mass


@pytest.fixture
def ipic3d_config():
    return parse_inp(FIXTURE_DIR / "synthetic.inp")


class TestPhdf5ParticleReader:
    """Integration tests against the synthetic fixture."""

    def test_detect_particle_steps(self) -> None:
        steps = detect_particle_steps(FIXTURE_DIR)
        assert steps == [0]

    def test_read_all_columns(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.position is not None
        assert pcl.velocity is not None
        assert pcl.weight is not None
        assert pcl.n_particles == 18

    def test_read_position_only(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(
            FIXTURE_DIR, 0, 0, ipic3d_config, columns=["position"]
        )
        assert pcl.position is not None
        assert pcl.velocity is None
        assert pcl.weight is not None

    def test_read_velocity_only(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(
            FIXTURE_DIR, 0, 0, ipic3d_config, columns=["velocity"]
        )
        assert pcl.position is None
        assert pcl.velocity is not None
        assert pcl.weight is not None

    def test_particle_count_correct(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.n_particles == 18
        assert len(pcl) == 18
        assert pcl.position is not None
        assert pcl.velocity is not None
        assert pcl.weight is not None
        assert pcl.position.shape == (18, 3)
        assert pcl.velocity.shape == (18, 3)
        assert pcl.weight.shape == (18,)

    def test_position_values(self, ipic3d_config) -> None:
        # Fixture generator lays particles on a deterministic 3x3x2 sub-grid:
        # x = linspace(0.1, 0.9, 3), y = same, z = linspace(0.1, 0.9, 2).
        # Flat [0.1, 0.9] range on every axis means a min/max-only check
        # would not catch an axis swap (all three fit the same envelope),
        # so pin the exact unique-value sets per axis.
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        np.testing.assert_allclose(np.unique(pcl.x), np.linspace(0.1, 0.9, 3))
        np.testing.assert_allclose(np.unique(pcl.y), np.linspace(0.1, 0.9, 3))
        np.testing.assert_allclose(np.unique(pcl.z), np.linspace(0.1, 0.9, 2))
        # With a 3x3x2 lattice, each x value appears 6 times, each z value 9.
        assert np.count_nonzero(pcl.x == 0.1) == 6
        assert np.count_nonzero(pcl.z == 0.1) == 9

    def test_macro_charge_reconstructed(self, ipic3d_config) -> None:
        # Canonical form gives us species_charge × weight.  Both fixture
        # species have unit weight, so macro_charge = ±1 per particle.
        pcl_e = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        np.testing.assert_array_equal(pcl_e.macro_charge, np.full(18, -1.0))
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        np.testing.assert_array_equal(pcl_i.macro_charge, np.full(18, 1.0))

    def test_weight_dtype(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.weight is not None
        assert pcl.weight.dtype == np.float64

    def test_canonical_form_no_charge_attribute(self, ipic3d_config) -> None:
        # Step 25b: canonical container does not carry per-particle charge.
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert not hasattr(pcl, "charge")

    def test_species_name(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.species_name == "species_0"
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        assert pcl_i.species_name == "species_1"

    def test_weight_unit_in_fixture(self, ipic3d_config) -> None:
        # Fixture uses uniform unit weighting.
        pcl_e = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl_e.weight is not None
        np.testing.assert_array_equal(pcl_e.weight, np.full(18, 1.0))

        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        assert pcl_i.weight is not None
        np.testing.assert_array_equal(pcl_i.weight, np.full(18, 1.0))

    def test_species_charge_populated(self, ipic3d_config) -> None:
        # iPIC3D convention: species_charge = sign(qom)
        pcl_e = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        assert pcl_e.species_charge == -1.0  # electrons
        assert pcl_i.species_charge == 1.0  # ions

    def test_species_mass_populated(self, ipic3d_config) -> None:
        # iPIC3D convention: species_mass = 1/|qom|.  Fixture QOM=(-64, 1),
        # so m_e = 1/64 (real electron) and m_i = 1.  Pins the exact
        # 1/|qom| formula — a stray sign, inverted ratio (|qom|/1), or
        # swap with |qom| itself all produce >0 values that passed the
        # pre-iter-14 positivity-only check.
        pcl_e = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        assert pcl_e.species_mass == 1.0 / 64.0
        assert pcl_i.species_mass == 1.0


class TestNonUniformWeight:
    """Per-particle q dataset (particle splitting, non-uniform plasma)."""

    def _write_fixture(self, tmp_path: Path, q_values: np.ndarray, n: int) -> Path:
        import h5py as h5  # type: ignore[import-untyped]

        particles_dir = tmp_path / "Particles_00000"
        particles_dir.mkdir()
        h5_path = particles_dir / "species_0_00000.h5"
        with h5.File(h5_path, "w") as f:
            g = f.create_group("Particles/species_0")
            g.create_dataset("position", data=np.zeros((n, 3)))
            g.create_dataset("velocity", data=np.ones((n, 3)))
            g.create_dataset("q", data=q_values)
        return h5_path

    def test_per_particle_q_round_trip(self, tmp_path) -> None:
        n = 7
        q = -np.array([1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0])
        self._write_fixture(tmp_path, q, n)

        cfg = parse_inp(FIXTURE_DIR / "synthetic.inp")
        pcl = read_phdf5_particles(tmp_path, 0, 0, cfg)

        assert pcl.n_particles == n
        assert pcl.weight is not None
        np.testing.assert_array_equal(pcl.weight, np.abs(q))
        np.testing.assert_array_equal(pcl.macro_charge, -np.abs(q))

    def test_q_size_mismatch_raises(self, tmp_path) -> None:
        n = 5
        q = np.ones(3)  # neither 1 nor n
        self._write_fixture(tmp_path, q, n)

        cfg = parse_inp(FIXTURE_DIR / "synthetic.inp")
        with pytest.raises(ValueError, match="'q' dataset size 3"):
            read_phdf5_particles(tmp_path, 0, 0, cfg)


class TestParticleIdSupport:
    """Test particle ID reading from phdf5 format."""

    def test_id_loaded(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.id is not None
        assert pcl.id.dtype == np.int64
        assert pcl.id.shape == (18,)

    def test_id_values(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        np.testing.assert_array_equal(pcl.id, np.arange(18, dtype=np.int64))

    def test_id_species_offset(self, ipic3d_config) -> None:
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        np.testing.assert_array_equal(pcl_i.id, np.arange(18, 36, dtype=np.int64))

    def test_id_in_repr(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert "id" in repr(pcl)

    def test_id_none_when_absent(self, tmp_path) -> None:
        """Files without ID dataset should yield id=None."""
        import h5py as h5  # type: ignore[import-untyped]

        particles_dir = tmp_path / "Particles_00000"
        particles_dir.mkdir()
        h5_path = particles_dir / "species_0_00000.h5"
        with h5.File(h5_path, "w") as f:
            g = f.create_group("Particles/species_0")
            g.create_dataset("position", data=np.zeros((5, 3)))
            g.create_dataset("velocity", data=np.ones((5, 3)))
            g.create_dataset("q", data=np.array([[1.0]]))

        pcl = read_phdf5_particles(
            tmp_path, 0, 0, parse_inp(FIXTURE_DIR / "synthetic.inp")
        )
        assert pcl.id is None

    def test_id_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="id shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=np.zeros((3, 3)),
                velocity=None,
                n_particles=3,
                metadata={},
                id=np.arange(5, dtype=np.int64),
            )


class TestParticleDataReaderProtocol:
    """Verify IPic3DParallelReader satisfies ParticleDataReader."""

    def test_parallel_reader_is_particle_reader(self, ipic3d_config) -> None:
        reader = IPic3DParallelReader(ipic3d_config)
        assert isinstance(reader, ParticleDataReader)

    def test_available_particle_steps(self, ipic3d_config) -> None:
        reader = IPic3DParallelReader(ipic3d_config)
        steps = reader.available_particle_steps(FIXTURE_DIR)
        assert steps == [0]

    def test_read_particles_via_reader(self, ipic3d_config) -> None:
        reader = IPic3DParallelReader(ipic3d_config)
        pcl = reader.read_particles(FIXTURE_DIR, 0, 0)
        assert pcl.n_particles == 18
        assert pcl.position is not None


class TestSimulationParticles:
    """Test Simulation.particles() and particle_steps."""

    def test_particles_via_simulation(self, ipic3d_config) -> None:
        from pypic.readers.ipic3d._config import to_simulation_config

        reader = IPic3DParallelReader(ipic3d_config)
        config = to_simulation_config(ipic3d_config)
        sim = Simulation(reader, config, FIXTURE_DIR)
        pcl = sim.particles(step=0, species=0)
        assert pcl.n_particles == 18
        assert pcl.position is not None

    def test_particle_steps_property(self, ipic3d_config) -> None:
        from pypic.readers.ipic3d._config import to_simulation_config

        reader = IPic3DParallelReader(ipic3d_config)
        config = to_simulation_config(ipic3d_config)
        sim = Simulation(reader, config, FIXTURE_DIR)
        assert sim.particle_steps == [0]

    def test_unsupported_reader_raises_typeerror(self) -> None:
        from unittest.mock import MagicMock

        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.grid import GridInfo
        from pypic.units import Normalization, SpeciesInfo

        mock_reader = MagicMock()
        # Ensure it doesn't satisfy ParticleDataReader
        del mock_reader.available_particle_steps
        del mock_reader.read_particles

        config = SimulationConfig(
            model_name="test",
            model_type="PIC",
            grid=GridInfo(
                dimensions=(4,),
                spacing=(1.0,),
                origin=(0.0,),
                geometry=CARTESIAN,
            ),
            normalization=Normalization.identity(),
            species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
        )
        sim = Simulation(mock_reader, config, "/tmp")
        assert sim.particle_steps == []
        with pytest.raises(TypeError, match="ParticleDataReader"):
            sim.particles(step=0, species=0)
