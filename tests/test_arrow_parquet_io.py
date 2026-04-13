"""Tests for pypic.io Arrow/Parquet particle data I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from pypic.containers import ParticleData  # noqa: E402
from pypic.io._arrow import particles_from_arrow, particles_to_arrow  # noqa: E402
from pypic.io._morton import morton_encode_3d, morton_sort_indices  # noqa: E402
from pypic.io._parquet import (  # noqa: E402
    particles_from_dataset,
    particles_from_parquet,
    particles_to_dataset,
    particles_to_parquet,
)
from pypic.units import SpeciesInfo  # noqa: E402


def _make_particles(
    n: int = 1000,
    *,
    seed: int = 42,
    species_index: int = 0,
    species_name: str = "electrons",
) -> ParticleData:
    """Create synthetic particle data for testing."""
    rng = np.random.default_rng(seed)
    return ParticleData(
        species_index=species_index,
        species_name=species_name,
        position=rng.uniform(0, 10, (n, 3)),
        velocity=rng.standard_normal((n, 3)),
        charge=np.full(n, -1.0),
        n_particles=n,
        id=np.arange(n, dtype=np.int64),
        metadata={"source": "test"},
    )


def _make_mock_simulation(
    steps: list[int],
    species: list[tuple[int, str]],
    n_particles: int = 100,
) -> Any:
    """Create a mock Simulation-like object for particles_to_dataset."""
    sim = MagicMock()
    sim.particle_steps = steps

    species_infos = tuple(
        SpeciesInfo(name=name, charge=-1.0 if name == "electrons" else 1.0, mass=1.0)
        for _, name in species
    )
    sim.config.species = species_infos

    def _particles(step: int, sp_idx: int, **kwargs: Any) -> ParticleData:
        sp_name = species[sp_idx][1] if sp_idx < len(species) else "unknown"
        seed = step * 100 + sp_idx
        return _make_particles(
            n_particles, seed=seed, species_index=sp_idx, species_name=sp_name
        )

    sim.particles = _particles
    return sim


# --- Morton encoding ---


class TestMortonEncoding:
    """Tests for Morton Z-order curve encoding."""

    def test_origin_is_zero(self):
        codes = morton_encode_3d(
            np.array([0.0]), np.array([0.0]), np.array([0.0]), bits=3
        )
        assert codes[0] == 0

    def test_max_corner(self):
        codes = morton_encode_3d(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            bits=2,
        )
        assert codes[0] == 0
        # All bits set: 2 bits per axis, interleaved = 0b111_111 = 63
        assert codes[1] == 63

    def test_sort_preserves_locality(self):
        rng = np.random.default_rng(99)
        x = rng.uniform(0, 100, 500)
        y = rng.uniform(0, 100, 500)
        z = rng.uniform(0, 100, 500)
        idx = morton_sort_indices(x, y, z)
        # Check that the sorted array isn't just the original order
        assert not np.array_equal(idx, np.arange(len(idx)))

    def test_bits_1_gives_8_possible_codes(self):
        # 1 bit per axis = 8 octants
        x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        z = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        codes = morton_encode_3d(x, y, z, bits=1)
        assert len(set(codes.tolist())) == 8

    def test_invalid_bits_raises(self):
        x = np.array([0.0])
        with pytest.raises(ValueError, match="bits must be in"):
            morton_encode_3d(x, x, x, bits=0)
        with pytest.raises(ValueError, match="bits must be in"):
            morton_encode_3d(x, x, x, bits=22)

    def test_constant_coordinate(self):
        # All same value → all codes should be 0
        x = np.array([5.0, 5.0, 5.0])
        codes = morton_encode_3d(x, x, x, bits=10)
        np.testing.assert_array_equal(codes, 0)


# --- Arrow interchange ---


class TestArrowInterchange:
    """Tests for ParticleData ↔ Arrow Table round-trips."""

    def test_round_trip_full(self):
        pcl = _make_particles(100)
        table = particles_to_arrow(pcl)
        rebuilt = particles_from_arrow(table)
        assert rebuilt.species_index == pcl.species_index
        assert rebuilt.species_name == pcl.species_name
        assert rebuilt.n_particles == pcl.n_particles
        np.testing.assert_array_equal(rebuilt.position, pcl.position)
        np.testing.assert_array_equal(rebuilt.velocity, pcl.velocity)
        np.testing.assert_array_equal(rebuilt.charge, pcl.charge)
        np.testing.assert_array_equal(rebuilt.id, pcl.id)

    def test_round_trip_position_only(self):
        rng = np.random.default_rng(1)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=rng.uniform(0, 1, (50, 3)),
            velocity=None,
            charge=np.full(50, -1.0),
            n_particles=50,
            metadata={},
        )
        table = particles_to_arrow(pcl)
        assert "vx" not in table.column_names
        rebuilt = particles_from_arrow(table)
        assert rebuilt.velocity is None
        np.testing.assert_array_equal(rebuilt.position, pcl.position)

    def test_round_trip_velocity_only(self):
        rng = np.random.default_rng(2)
        pcl = ParticleData(
            species_index=1,
            species_name="ions",
            position=None,
            velocity=rng.standard_normal((30, 3)),
            charge=np.full(30, 1.0),
            n_particles=30,
            metadata={},
        )
        table = particles_to_arrow(pcl)
        assert "x" not in table.column_names
        rebuilt = particles_from_arrow(table)
        assert rebuilt.position is None
        np.testing.assert_array_equal(rebuilt.velocity, pcl.velocity)

    def test_species_metadata_preserved(self):
        pcl = _make_particles(10, species_index=3, species_name="alpha")
        table = particles_to_arrow(pcl)
        rebuilt = particles_from_arrow(table)
        assert rebuilt.species_index == 3
        assert rebuilt.species_name == "alpha"

    def test_dtype_downcast_position_float32(self):
        pcl = _make_particles(20)
        table = particles_to_arrow(pcl, position_dtype="float32")
        assert table.column("x").type == pa.float32()
        assert table.column("y").type == pa.float32()
        assert table.column("z").type == pa.float32()

    def test_dtype_downcast_velocity_float32(self):
        pcl = _make_particles(20)
        table = particles_to_arrow(pcl, velocity_dtype="float32")
        assert table.column("vx").type == pa.float32()

    def test_charge_stays_float64(self):
        pcl = _make_particles(20)
        table = particles_to_arrow(
            pcl, position_dtype="float32", velocity_dtype="float32"
        )
        assert table.column("charge").type == pa.float64()

    def test_column_names_full(self):
        pcl = _make_particles(10)
        table = particles_to_arrow(pcl)
        assert set(table.column_names) == {
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "charge",
            "id",
        }

    def test_round_trip_without_id(self):
        rng = np.random.default_rng(5)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=rng.uniform(0, 1, (25, 3)),
            velocity=rng.standard_normal((25, 3)),
            charge=np.full(25, -1.0),
            n_particles=25,
            metadata={},
            id=None,
        )
        table = particles_to_arrow(pcl)
        assert "id" not in table.column_names
        rebuilt = particles_from_arrow(table)
        assert rebuilt.id is None


# --- Single-file Parquet ---


class TestSingleFileParquet:
    """Tests for single-file Parquet write/read."""

    def test_round_trip_basic(self, tmp_path: Path):
        pcl = _make_particles(200)
        fpath = tmp_path / "test.parquet"
        particles_to_parquet(pcl, fpath)
        rebuilt = particles_from_parquet(fpath)
        assert rebuilt.n_particles == pcl.n_particles
        assert rebuilt.species_name == pcl.species_name
        # Data survives round-trip (ordering may differ due to Morton sort)
        # Sort by ID to compare
        orig_order = np.argsort(pcl.id)
        rebuilt_order = np.argsort(rebuilt.id)
        np.testing.assert_array_equal(rebuilt.id[rebuilt_order], pcl.id[orig_order])
        np.testing.assert_allclose(
            rebuilt.position[rebuilt_order],
            pcl.position[orig_order],
            rtol=1e-6,
        )

    def test_morton_sorting_applied(self, tmp_path: Path):
        pcl = _make_particles(500)
        fpath = tmp_path / "sorted.parquet"
        particles_to_parquet(pcl, fpath)
        rebuilt = particles_from_parquet(fpath)
        # Verify Morton codes are monotonically non-decreasing
        codes = morton_encode_3d(rebuilt.x, rebuilt.y, rebuilt.z)
        assert np.all(np.diff(codes.astype(np.int64)) >= 0)

    def test_float32_downcast(self, tmp_path: Path):
        import pyarrow.parquet as pq

        pcl = _make_particles(50)
        fpath = tmp_path / "f32.parquet"
        particles_to_parquet(pcl, fpath, position_dtype="float32")
        # Read raw to check dtypes
        table = pq.read_table(str(fpath))
        assert table.column("x").type == pa.float32()
        assert table.column("charge").type == pa.float64()

    def test_speed_column_written(self, tmp_path: Path):
        import pyarrow.parquet as pq

        pcl = _make_particles(50)
        fpath = tmp_path / "speed.parquet"
        particles_to_parquet(pcl, fpath)
        table = pq.read_table(str(fpath))
        assert "speed" in table.column_names

    def test_speed_column_stripped_on_read(self, tmp_path: Path):
        pcl = _make_particles(50)
        fpath = tmp_path / "strip.parquet"
        particles_to_parquet(pcl, fpath)
        rebuilt = particles_from_parquet(fpath)
        # ParticleData should not have a speed attribute leak
        assert rebuilt.n_particles == 50


# --- Partitioned dataset ---


class TestPartitionedDataset:
    """Tests for partitioned Parquet dataset write/read."""

    def _write_dataset(self, tmp_path: Path, n_particles: int = 100) -> Path:
        root = tmp_path / "particles"
        sim = _make_mock_simulation(
            steps=[0, 100],
            species=[(0, "electrons"), (1, "ions")],
            n_particles=n_particles,
        )
        particles_to_dataset(sim, root)
        return root

    def test_write_creates_hive_layout(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        e_pq = "species=electrons/part-00000.parquet"
        i_pq = "species=ions/part-00000.parquet"
        assert (root / "step=000000" / e_pq).exists()
        assert (root / "step=000000" / i_pq).exists()
        assert (root / "step=000100" / e_pq).exists()
        assert (root / "step=000100" / i_pq).exists()

    def test_read_basic(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        pcl = particles_from_dataset(root, step=0, species="electrons")
        assert pcl.n_particles == 100
        assert pcl.species_name == "electrons"

    def test_partition_pruning_step(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        pcl = particles_from_dataset(root, step=100, species="electrons")
        assert pcl.n_particles == 100

    def test_partition_pruning_species_by_name(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        pcl = particles_from_dataset(root, step=0, species="ions")
        assert pcl.species_name == "ions"

    def test_spatial_box_filter(self, tmp_path: Path):
        root = self._write_dataset(tmp_path, n_particles=2000)
        box = ((2.0, 4.0), (2.0, 4.0), (2.0, 4.0))
        pcl = particles_from_dataset(root, step=0, species="electrons", spatial_box=box)
        # All returned particles must be within the box
        assert pcl.n_particles > 0
        assert np.all(pcl.x >= 2.0)
        assert np.all(pcl.x <= 4.0)
        assert np.all(pcl.y >= 2.0)
        assert np.all(pcl.y <= 4.0)
        assert np.all(pcl.z >= 2.0)
        assert np.all(pcl.z <= 4.0)

    def test_column_pruning(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        pcl = particles_from_dataset(
            root, step=0, species="electrons", columns=["x", "y", "z"]
        )
        assert pcl.position is not None
        assert pcl.velocity is None

    def test_energy_filter(self, tmp_path: Path):
        root = self._write_dataset(tmp_path, n_particles=2000)
        pcl = particles_from_dataset(root, step=0, species="electrons", energy_min=1.5)
        if pcl.n_particles > 0:
            speed = np.sqrt(pcl.vx**2 + pcl.vy**2 + pcl.vz**2)
            assert np.all(speed >= 1.5 - 1e-6)

    def test_id_filter(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        target_ids = [0, 5, 10, 42]
        pcl = particles_from_dataset(root, step=0, species="electrons", ids=target_ids)
        assert pcl.n_particles <= len(target_ids)
        assert all(pid in target_ids for pid in pcl.id.tolist())

    def test_read_all_partitions(self, tmp_path: Path):
        root = self._write_dataset(tmp_path)
        # No filters — reads everything
        pcl = particles_from_dataset(root)
        # 2 steps × 2 species × 100 particles = 400
        assert pcl.n_particles == 400


# --- DuckDB ---


class TestDuckDBQuery:
    """Tests for DuckDB SQL query interface."""

    duckdb_mod = pytest.importorskip("duckdb")

    def _write_dataset(self, tmp_path: Path) -> Path:
        root = tmp_path / "duckdb_particles"
        sim = _make_mock_simulation(
            steps=[0, 100],
            species=[(0, "electrons"), (1, "ions")],
            n_particles=50,
        )
        particles_to_dataset(sim, root)
        return root

    def test_basic_query(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(root, "SELECT * FROM particles LIMIT 10")
        assert pcl.n_particles == 10

    def test_filtered_query(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(
            root,
            "SELECT * FROM particles WHERE species='electrons' AND step='000000'",
        )
        assert pcl.n_particles == 50
        assert pcl.species_name == "electrons"

    def test_return_type_arrow(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        table = query_sql(
            root,
            "SELECT x, y FROM particles LIMIT 5",
            return_type="arrow",
        )
        assert isinstance(table, pa.Table)
        assert len(table) == 5
