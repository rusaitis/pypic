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
    weight: np.ndarray | None = None,
) -> ParticleData:
    """Create synthetic particle data for testing."""
    rng = np.random.default_rng(seed)
    return ParticleData(
        species_index=species_index,
        species_name=species_name,
        position=rng.uniform(0, 10, (n, 3)),
        velocity=rng.standard_normal((n, 3)),
        n_particles=n,
        id=np.arange(n, dtype=np.int64),
        metadata={"source": "test"},
        weight=weight if weight is not None else np.ones(n),
        species_charge=-1.0 if species_name == "electrons" else 1.0,
        species_mass=1.0,
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
        np.testing.assert_array_equal(rebuilt.weight, pcl.weight)
        np.testing.assert_array_equal(rebuilt.id, pcl.id)
        assert rebuilt.species_charge == pcl.species_charge
        assert rebuilt.species_mass == pcl.species_mass

    def test_round_trip_position_only(self):
        rng = np.random.default_rng(1)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=rng.uniform(0, 1, (50, 3)),
            velocity=None,
            n_particles=50,
            metadata={},
            weight=np.ones(50),
            species_charge=-1.0,
            species_mass=1.0,
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
            n_particles=30,
            metadata={},
            weight=np.ones(30),
            species_charge=1.0,
            species_mass=1.0,
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

    def test_numpy_scalar_metadata_round_trips(self):
        # h5py-style readers commonly hand back attrs as numpy scalars
        # (np.float32, np.int32, ...).  json.dumps rejects those, so the
        # encoder must coerce them to Python natives — same path used
        # for FieldDataset attrs.
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=np.zeros((4, 3)),
            velocity=np.zeros((4, 3)),
            n_particles=4,
            metadata={
                "time": np.float32(1.25),
                "step": np.int32(7),
                "shape": np.array([4, 3]),
            },
            weight=np.ones(4),
            species_charge=-1.0,
            species_mass=1.0,
        )
        table = particles_to_arrow(pcl)
        rebuilt = particles_from_arrow(table)
        assert rebuilt.metadata["time"] == 1.25
        assert isinstance(rebuilt.metadata["time"], float)
        assert rebuilt.metadata["step"] == 7
        assert isinstance(rebuilt.metadata["step"], int)
        assert rebuilt.metadata["shape"] == [4, 3]

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

    def test_no_charge_column(self):
        # Step 25b: canonical form never carries a per-particle charge column.
        pcl = _make_particles(10)
        table = particles_to_arrow(pcl)
        assert "charge" not in table.column_names

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
            "weight",
            "id",
        }

    def test_species_metadata_round_trip(self):
        # species_charge/species_mass travel in schema metadata
        pcl_with_meta = ParticleData(
            species_index=0,
            species_name="electrons",
            position=np.zeros((20, 3)),
            velocity=np.ones((20, 3)),
            n_particles=20,
            metadata={},
            weight=np.ones(20),
            species_charge=2.5,
            species_mass=4.0,
        )
        table = particles_to_arrow(pcl_with_meta)
        rebuilt = particles_from_arrow(table)
        assert rebuilt.species_charge == 2.5
        assert rebuilt.species_mass == 4.0

    def test_non_uniform_weight_round_trip(self):
        rng = np.random.default_rng(13)
        n = 30
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 1, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            metadata={},
            weight=rng.uniform(0.5, 2.0, n),
            species_charge=-1.0,
            species_mass=1.0,
        )
        table = particles_to_arrow(pcl)
        assert "weight" in table.column_names
        rebuilt = particles_from_arrow(table)
        assert rebuilt.weight is not None
        np.testing.assert_array_equal(rebuilt.weight, pcl.weight)
        # macro_charge reconstructs as species_charge × weight
        np.testing.assert_array_equal(rebuilt.macro_charge, -1.0 * pcl.weight)

    def test_round_trip_without_weight(self):
        # Particles without weight: no moment math available, but the
        # container still round-trips (positions-only workflows).
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=np.zeros((10, 3)),
            velocity=None,
            n_particles=10,
            metadata={},
        )
        table = particles_to_arrow(pcl)
        assert "weight" not in table.column_names
        rebuilt = particles_from_arrow(table)
        assert rebuilt.weight is None

    def test_round_trip_without_id(self):
        rng = np.random.default_rng(5)
        pcl = ParticleData(
            species_index=0,
            species_name="e",
            position=rng.uniform(0, 1, (25, 3)),
            velocity=rng.standard_normal((25, 3)),
            n_particles=25,
            metadata={},
            weight=np.ones(25),
            species_charge=-1.0,
            species_mass=1.0,
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
        assert table.column("weight").type == pa.float64()

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

    def test_sort_by_invalid_raises(self, tmp_path: Path):
        pcl = _make_particles(10)
        with pytest.raises(ValueError, match="sort_by"):
            particles_to_parquet(pcl, tmp_path / "bad.parquet", sort_by="velocity")  # type: ignore[arg-type]

    def test_canonical_round_trip(self, tmp_path: Path):
        # Canonical form: weight + species scalars round-trip
        rng = np.random.default_rng(31)
        n = 60
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 5, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            metadata={},
            weight=rng.uniform(0.5, 2.0, n),
            species_charge=-1.0,
            species_mass=1.0,
        )
        fpath = tmp_path / "canonical.parquet"
        particles_to_parquet(pcl, fpath)
        rebuilt = particles_from_parquet(fpath)
        assert rebuilt.weight is not None
        assert rebuilt.species_charge == -1.0
        assert rebuilt.species_mass == 1.0
        assert not hasattr(rebuilt, "charge")

    def test_sort_by_weight(self, tmp_path: Path):
        rng = np.random.default_rng(17)
        n = 200
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 10, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            id=np.arange(n, dtype=np.int64),
            weight=rng.uniform(0.5, 2.0, n),
            species_charge=-1.0,
            species_mass=1.0,
            metadata={},
        )
        fpath = tmp_path / "weight_sorted.parquet"
        particles_to_parquet(pcl, fpath, sort_by="weight")
        rebuilt = particles_from_parquet(fpath)
        assert rebuilt.weight is not None
        assert np.all(np.diff(rebuilt.weight) >= 0)


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

    def test_unfiltered_multispecies_read_raises(self, tmp_path: Path):
        # The container is single-species; merging electrons + ions into
        # one ParticleData would corrupt macro_charge/macro_mass since
        # they use the scalar species_charge.  Force the caller to pick.
        root = self._write_dataset(tmp_path)
        with pytest.raises(ValueError, match=r"matched 2 species"):
            particles_from_dataset(root)

    def test_string_species_resolves_correct_index(self, tmp_path: Path):
        # particles_from_dataset(species="ions") must surface the
        # original species_index (1), not the default 0 — downstream
        # consumers rely on ParticleData.species_index for routing.
        root = self._write_dataset(tmp_path)
        pcl_e = particles_from_dataset(root, step=0, species="electrons")
        pcl_i = particles_from_dataset(root, step=0, species="ions")
        assert pcl_e.species_index == 0
        assert pcl_i.species_index == 1

    def test_integer_species_uses_stored_index(self, tmp_path: Path):
        # Hive partitioning erases config order, so an integer selector
        # must consult per-fragment metadata rather than alphabetical
        # directory order.  Names chosen so alphabetical order
        # disagrees with the stored species_index ("alpha" < "zeta").
        pcl_zeta = _make_particles(20, seed=1, species_index=0, species_name="zeta")
        pcl_alpha = _make_particles(30, seed=2, species_index=1, species_name="alpha")
        root = tmp_path / "non_alpha"
        particles_to_dataset(
            [(0, "zeta", pcl_zeta), (0, "alpha", pcl_alpha)],
            root,
        )
        idx0 = particles_from_dataset(root, step=0, species=0)
        idx1 = particles_from_dataset(root, step=0, species=1)
        assert idx0.species_name == "zeta"
        assert idx0.species_index == 0
        assert idx1.species_name == "alpha"
        assert idx1.species_index == 1

    def test_sparse_part_numbering_resumes_at_next_free_index(self, tmp_path: Path):
        # If a partition already contains part-00001.parquet without
        # part-00000.parquet (deletion, manual edit, partial prior
        # write), the next chunk must land at part-00002.parquet, not
        # overwrite the existing part-00001.parquet.  File-counting
        # would have produced index=1 here.
        chunk_a = _make_particles(10, seed=11, species_name="electrons")
        chunk_b = _make_particles(20, seed=22, species_name="electrons")
        root = tmp_path / "sparse"
        particles_to_dataset([(0, "electrons", chunk_a)], root)
        part_dir = root / "step=000000/species=electrons"
        # Make the prior file sparse: drop 00000, leave only 00001.
        (part_dir / "part-00000.parquet").rename(part_dir / "part-00001.parquet")
        sentinel = (part_dir / "part-00001.parquet").read_bytes()

        particles_to_dataset([(0, "electrons", chunk_b)], root)

        assert (part_dir / "part-00002.parquet").exists()
        # Existing sparse file is untouched.
        assert (part_dir / "part-00001.parquet").read_bytes() == sentinel

    def test_iterable_chunked_writes_preserve_all_batches(self, tmp_path: Path):
        # Streaming pipelines naturally yield multiple ParticleData
        # chunks per (step, species).  Each chunk must land in its own
        # part-NNNNN.parquet — the prior fixed "part-00000.parquet"
        # filename silently dropped all but the last batch.
        chunk_a = _make_particles(40, seed=1, species_name="electrons")
        chunk_b = _make_particles(60, seed=2, species_name="electrons")
        root = tmp_path / "chunked"
        particles_to_dataset(
            [(0, "electrons", chunk_a), (0, "electrons", chunk_b)],
            root,
        )
        part_dir = root / "step=000000/species=electrons"
        parts = sorted(part_dir.glob("part-*.parquet"))
        assert [p.name for p in parts] == [
            "part-00000.parquet",
            "part-00001.parquet",
        ]
        rebuilt = particles_from_dataset(root, step=0, species="electrons")
        assert rebuilt.n_particles == 40 + 60

    def test_id_column_weight_filter(self, tmp_path: Path):
        # Non-uniform weight doubles as a particle tracking ID —
        # float64 precision makes each weight unique per macroparticle.
        rng = np.random.default_rng(11)
        n = 100
        weights = rng.uniform(0.5, 2.0, n)
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 10, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            id=np.arange(n, dtype=np.int64),
            metadata={},
            weight=weights,
            species_charge=-1.0,
            species_mass=1.0,
        )
        root = tmp_path / "by_weight"
        particles_to_dataset([(0, "electrons", pcl)], root, sort_by="weight")
        target = [float(weights[3]), float(weights[42]), float(weights[77])]
        result = particles_from_dataset(
            root, step=0, species="electrons", ids=target, id_column="weight"
        )
        assert result.n_particles == 3
        assert result.weight is not None
        assert sorted(result.weight.tolist()) == sorted(target)

    def test_iterable_source(self, tmp_path: Path):
        pcl_e = _make_particles(40, seed=1, species_name="electrons")
        pcl_i = _make_particles(60, seed=2, species_index=1, species_name="ions")
        root = tmp_path / "iterable"
        particles_to_dataset(
            [(0, "electrons", pcl_e), (0, "ions", pcl_i)],
            root,
        )
        e_path = root / "step=000000/species=electrons/part-00000.parquet"
        i_path = root / "step=000000/species=ions/part-00000.parquet"
        assert e_path.exists()
        assert i_path.exists()
        rebuilt_e = particles_from_dataset(root, step=0, species="electrons")
        assert rebuilt_e.n_particles == 40

    def test_partitioned_dataset_round_trip(self, tmp_path: Path):
        # Canonical form round-trips through the partitioned dataset:
        # weight + species_charge/species_mass all recovered.
        rng = np.random.default_rng(43)
        n = 50
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 5, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            metadata={},
            weight=rng.uniform(0.5, 2.0, n),
            species_charge=-1.0,
            species_mass=1.0,
        )
        root = tmp_path / "canonical_dataset"
        particles_to_dataset([(0, "electrons", pcl)], root)
        rebuilt = particles_from_dataset(root, step=0, species="electrons")
        assert rebuilt.weight is not None
        assert rebuilt.species_charge == -1.0
        assert rebuilt.species_mass == 1.0
        np.testing.assert_array_equal(rebuilt.macro_charge, -1.0 * rebuilt.weight)

    def test_columns_pruning(self, tmp_path: Path):
        # When user requests only x/y/z, weight should not be loaded.
        rng = np.random.default_rng(47)
        n = 40
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 5, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            metadata={},
            weight=rng.uniform(0.5, 2.0, n),
            species_charge=-1.0,
        )
        root = tmp_path / "pruned_columns"
        particles_to_dataset([(0, "electrons", pcl)], root)
        rebuilt = particles_from_dataset(
            root, step=0, species="electrons", columns=["x", "y", "z"]
        )
        assert rebuilt.position is not None
        assert rebuilt.velocity is None
        assert rebuilt.weight is None  # not requested

    def test_scalar_only_columns_raises(self, tmp_path: Path):
        # columns=['id'] omits both position and velocity triplets.
        # Failing late inside ParticleData.__post_init__ produced an
        # opaque message; fail early with a projection-aware one.
        rng = np.random.default_rng(101)
        n = 10
        pcl = ParticleData(
            species_index=0,
            species_name="electrons",
            position=rng.uniform(0, 1, (n, 3)),
            velocity=rng.standard_normal((n, 3)),
            n_particles=n,
            id=np.arange(n, dtype=np.int64),
            metadata={},
            weight=np.ones(n),
            species_charge=-1.0,
            species_mass=1.0,
        )
        root = tmp_path / "scalar_only"
        particles_to_dataset([(0, "electrons", pcl)], root)
        with pytest.raises(ValueError, match=r"full position triplet"):
            particles_from_dataset(root, step=0, species="electrons", columns=["id"])

    def test_iterable_with_steps_kwarg_raises(self, tmp_path: Path):
        pcl = _make_particles(10)
        with pytest.raises(ValueError, match="only apply when source is a Simulation"):
            particles_to_dataset(
                [(0, "electrons", pcl)],
                tmp_path / "bad",
                steps=[0],
            )

    def test_particle_metadata_round_trip(self, tmp_path: Path):
        # Reviewer regression: ``_matched_species_metadata`` already
        # recovers the full per-fragment ``pypic`` payload, but the
        # final ``inject_species_meta`` call used to drop the nested
        # ``metadata`` dict — dataset round-trips silently lost any
        # non-empty ``ParticleData.metadata``.
        pcl = _make_particles(20)  # metadata={"source": "test"}
        root = tmp_path / "with_metadata"
        particles_to_dataset([(0, "electrons", pcl)], root)
        rebuilt = particles_from_dataset(root, step=0, species="electrons")
        assert rebuilt.metadata == {"source": "test"}


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
        pcl = query_sql(
            root, "SELECT * FROM particles WHERE species='electrons' LIMIT 10"
        )
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
        # Scalar metadata must survive the DuckDB strip + recover path.
        assert pcl.species_index == 0
        assert pcl.species_charge == -1.0
        assert pcl.species_mass == 1.0
        # ``_make_particles`` stamps metadata={'source': 'test'}; the
        # recovery path must carry it back through the SQL entrypoint.
        assert pcl.metadata == {"source": "test"}

    def test_string_species_resolves_correct_index(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(root, "SELECT * FROM particles WHERE species='ions'")
        assert pcl.species_name == "ions"
        assert pcl.species_index == 1
        assert pcl.species_charge == 1.0

    def test_unfiltered_multispecies_query_raises(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        with pytest.raises(ValueError, match=r"matched 2 species"):
            query_sql(root, "SELECT * FROM particles")

    def test_empty_single_species_result(self, tmp_path: Path):
        # Well-formed single-species filter that happens to match zero
        # rows should return an empty ParticleData, not raise.
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(
            root,
            "SELECT * FROM particles WHERE species='electrons' AND 1=0",
        )
        assert pcl.n_particles == 0
        assert pcl.position is not None
        assert pcl.position.shape == (0, 3)

    def test_missing_species_column_raises(self, tmp_path: Path):
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        # SELECT x, y — drops partition columns entirely; we can't
        # reconstruct ParticleData without a species hint.
        with pytest.raises(ValueError, match=r"species.*partition column"):
            query_sql(
                root,
                "SELECT x, y FROM particles WHERE species='electrons' LIMIT 5",
            )

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

    def test_scalar_only_projection_raises(self, tmp_path: Path):
        # Reviewer regression: a projection with the `species` column
        # but no full position or velocity triplet used to fall through
        # to particles_from_arrow and raise the opaque "At least one of
        # position or velocity must be provided".  Must fail early with
        # an actionable pointer to return_type="arrow".
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        with pytest.raises(ValueError, match=r"position triplet.*velocity triplet"):
            query_sql(
                root,
                "SELECT species, id FROM particles WHERE species='electrons' LIMIT 5",
            )

    def test_empty_result_adopts_single_species_from_disk(self, tmp_path: Path):
        # Reviewer regression: a zero-row result on a single-species
        # dataset must not lose the species identity — it should adopt
        # the only on-disk species rather than fall back to
        # "unknown"/index 0/no charge.
        from pypic.io._duckdb import query_sql

        root = tmp_path / "single_species"
        sim = _make_mock_simulation(
            steps=[0, 100],
            species=[(0, "electrons")],
            n_particles=50,
        )
        particles_to_dataset(sim, root)
        pcl = query_sql(
            root,
            "SELECT * FROM particles WHERE species='electrons' AND 1=0",
        )
        assert pcl.n_particles == 0
        assert pcl.species_name == "electrons"
        assert pcl.species_index == 0
        assert pcl.species_charge == -1.0
        assert pcl.species_mass == 1.0

    def test_empty_result_multispecies_recovers_pinned_species(self, tmp_path: Path):
        # When the SQL pins exactly one on-disk species via a literal
        # ``species='NAME'``, an empty multi-species result must still
        # recover the intended identity (name, index, charge, mass)
        # rather than degrade to the ``unknown`` placeholder.
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(
            root,
            "SELECT * FROM particles WHERE species='electrons' AND 1=0",
        )
        assert pcl.n_particles == 0
        assert pcl.species_name == "electrons"
        assert pcl.species_index == 0
        assert pcl.species_charge == -1.0
        assert pcl.species_mass == 1.0

    def test_empty_result_multispecies_no_literal_stays_placeholder(
        self, tmp_path: Path
    ):
        # Without a ``species='NAME'`` literal in the SQL, an empty
        # multi-species result is genuinely ambiguous and must keep
        # the ``unknown`` placeholder.
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(
            root,
            "SELECT * FROM particles WHERE x > 1e20 AND 1=0",
        )
        assert pcl.n_particles == 0
        assert pcl.species_name == "unknown"

    def test_velocity_only_projection_succeeds(self, tmp_path: Path):
        # ParticleData requires *one* of position or velocity, not both —
        # velocity-only projections must still round-trip.
        from pypic.io._duckdb import query_sql

        root = self._write_dataset(tmp_path)
        pcl = query_sql(
            root,
            "SELECT species, vx, vy, vz FROM particles "
            "WHERE species='electrons' LIMIT 5",
        )
        assert pcl.n_particles == 5
        assert pcl.velocity is not None
        assert pcl.velocity.shape == (5, 3)
        assert pcl.position is None
