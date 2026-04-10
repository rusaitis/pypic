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

FIXTURE_DIR = Path("tests/data/ipic3d-synthetic/phdf5")


class TestParticleData:
    """Unit tests for the ParticleData dataclass."""

    def test_construction_and_properties(self) -> None:
        pos = np.zeros((10, 3))
        vel = np.ones((10, 3))
        q = np.full(10, -1.0)
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=pos,
            velocity=vel,
            charge=q,
            n_particles=10,
            metadata={"fmt": "test"},
        )
        assert pcl.species_index == 0
        assert pcl.species_name == "electrons"
        assert pcl.n_particles == 10

    def test_position_only(self) -> None:
        pos = np.zeros((5, 3))
        q = np.full(5, 1.0)
        pcl = ParticleData(
            species_index=1,
            species_name="ions",
            position=pos,
            velocity=None,
            charge=q,
            n_particles=5,
            metadata={},
        )
        assert pcl.position is not None
        assert pcl.velocity is None

    def test_velocity_only(self) -> None:
        vel = np.ones((5, 3))
        q = np.full(5, 1.0)
        pcl = ParticleData(
            species_index=1,
            species_name="ions",
            position=None,
            velocity=vel,
            charge=q,
            n_particles=5,
            metadata={},
        )
        assert pcl.position is None
        assert pcl.velocity is not None

    def test_charge_always_present(self) -> None:
        pos = np.zeros((3, 3))
        q = np.full(3, -1.0)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=pos,
            velocity=None,
            charge=q,
            n_particles=3,
            metadata={},
        )
        assert pcl.charge is not None
        np.testing.assert_array_equal(pcl.charge, np.full(3, -1.0))

    def test_charge_dtype_float64(self) -> None:
        pos = np.zeros((3, 3))
        q_f32 = np.full(3, 1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="float64"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=pos,
                velocity=None,
                charge=q_f32,
                n_particles=3,
                metadata={},
            )

    def test_xyz_vxyz_are_views(self) -> None:
        pos = np.arange(15, dtype=np.float64).reshape(5, 3)
        vel = np.arange(15, 30, dtype=np.float64).reshape(5, 3)
        q = np.full(5, 1.0)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=pos,
            velocity=vel,
            charge=q,
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
        vel = np.ones((3, 3))
        q = np.full(3, 1.0)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=None,
            velocity=vel,
            charge=q,
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
            charge=q,
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
                charge=np.full(3, 1.0),
                n_particles=3,
                metadata={},
            )
        with pytest.raises(ValueError, match="velocity shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=None,
                velocity=np.ones((2, 3)),
                charge=np.full(3, 1.0),
                n_particles=3,
                metadata={},
            )
        with pytest.raises(ValueError, match="charge shape"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=np.zeros((3, 3)),
                velocity=None,
                charge=np.full(5, 1.0),
                n_particles=3,
                metadata={},
            )

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ParticleData(
                species_index=0,
                species_name="e",
                position=None,
                velocity=None,
                charge=np.full(3, 1.0),
                n_particles=3,
                metadata={},
            )

    def test_metadata_frozen(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((3, 3)),
            velocity=None,
            charge=np.full(3, 1.0),
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
            charge=np.full(10, -1.0),
            n_particles=10,
            metadata={},
        )
        r = repr(pcl)
        assert "electrons" in r
        assert "10" in r
        assert "position" in r
        assert "velocity" not in r or "charge" in r

    def test_len(self) -> None:
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((7, 3)),
            velocity=None,
            charge=np.full(7, 1.0),
            n_particles=7,
            metadata={},
        )
        assert len(pcl) == 7


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
        assert pcl.charge is not None
        assert pcl.n_particles == 18

    def test_read_position_only(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(
            FIXTURE_DIR, 0, 0, ipic3d_config, columns=["position"]
        )
        assert pcl.position is not None
        assert pcl.velocity is None
        assert pcl.charge is not None  # always loaded

    def test_read_velocity_only(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(
            FIXTURE_DIR, 0, 0, ipic3d_config, columns=["velocity"]
        )
        assert pcl.position is None
        assert pcl.velocity is not None
        assert pcl.charge is not None  # always loaded

    def test_particle_count_correct(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.n_particles == 18
        assert len(pcl) == 18
        assert pcl.position.shape == (18, 3)
        assert pcl.velocity.shape == (18, 3)
        assert pcl.charge.shape == (18,)

    def test_position_values(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        # Positions were generated on a 3x3x2 sub-grid within first cell
        dx, dy, dz = 1.0, 1.0, 1.0  # fixture cell spacing
        assert pcl.x.min() >= 0.0
        assert pcl.x.max() < dx
        assert pcl.y.min() >= 0.0
        assert pcl.y.max() < dy
        assert pcl.z.min() >= 0.0
        assert pcl.z.max() < dz

    def test_charge_broadcast_from_scalar(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        # Electrons: q = -1.0 broadcast to all particles
        np.testing.assert_array_equal(pcl.charge, np.full(18, -1.0))

        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        # Ions: q = +1.0
        np.testing.assert_array_equal(pcl_i.charge, np.full(18, 1.0))

    def test_charge_full_precision(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.charge.dtype == np.float64

    def test_species_name(self, ipic3d_config) -> None:
        pcl = read_phdf5_particles(FIXTURE_DIR, 0, 0, ipic3d_config)
        assert pcl.species_name == "species_0"
        pcl_i = read_phdf5_particles(FIXTURE_DIR, 0, 1, ipic3d_config)
        assert pcl_i.species_name == "species_1"


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
                charge=np.full(3, 1.0),
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
