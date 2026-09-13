"""Tests for SimpleReader with synthetic HDF5 fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import constants

from pypic.containers import SimulationConfig
from pypic.coordinates.geometry import CARTESIAN
from pypic.grid import GridInfo
from pypic.readers._simple import (
    SimpleReader,
    _parse_file_pattern,
    can_read_confidence,
    open_simple,
)
from pypic.units import Normalization, PhysicsParams
from tests._helpers import make_uniform_grid

if TYPE_CHECKING:
    from pathlib import Path


DIMS = (4, 3, 2)
SPACING = (1.0, 2.0, 3.0)
ORIGIN = (0.0, 0.0, 0.0)


def _sample_grid() -> GridInfo:
    return make_uniform_grid(*DIMS, spacing=SPACING, origin=ORIGIN)


def _sample_config(
    grid: GridInfo | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        model_name="test_sim",
        model_type="MHD",
        grid=grid or _sample_grid(),
        normalization=Normalization.identity(),
        physics=PhysicsParams(gamma=5.0 / 3.0),
    )


def _write_h5(
    filepath: Path,
    fields: dict[str, np.ndarray],
    *,
    fields_group: str = "fields",
    grid_attrs: dict[str, Any] | None = None,
    model: str | None = None,
    model_type: str | None = None,
    step: int | None = None,
    time: float | None = None,
) -> None:
    """Write a synthetic HDF5 file."""
    with h5py.File(filepath, "w") as f:
        grp = f.create_group(fields_group) if fields_group else f
        for name, data in fields.items():
            grp.create_dataset(name, data=data)

        if grid_attrs is not None:
            g = f.create_group("grid")
            for k, v in grid_attrs.items():
                g.attrs[k] = v

        if model is not None:
            f.attrs["model"] = model
        if model_type is not None:
            f.attrs["model_type"] = model_type
        if step is not None:
            f.attrs["step"] = step
        if time is not None:
            f.attrs["time"] = time


def _make_fields() -> dict[str, np.ndarray]:
    """Small arrays with recognizable values."""
    rng = np.random.default_rng(42)
    return {
        "B_1": rng.standard_normal(DIMS),
        "B_2": rng.standard_normal(DIMS),
        "B_3": rng.standard_normal(DIMS),
        "rho_m": np.abs(rng.standard_normal(DIMS)) + 0.1,
    }


def _grid_attrs() -> dict[str, Any]:
    return {
        "dimensions": list(DIMS),
        "spacing": list(SPACING),
        "origin": list(ORIGIN),
        "geometry": "cartesian",
    }


@pytest.fixture
def canonical_dir(tmp_path: Path) -> Path:
    """Three timesteps with full metadata."""
    fields = _make_fields()
    for step in (0, 10, 20):
        _write_h5(
            tmp_path / f"output_{step:06d}.h5",
            fields,
            grid_attrs=_grid_attrs(),
            model="test_sim",
            model_type="MHD",
            step=step,
            time=step * 0.1,
        )
    return tmp_path


@pytest.fixture
def bare_dir(tmp_path: Path) -> Path:
    """One timestep, no metadata — just fields."""
    _write_h5(
        tmp_path / "output_000000.h5",
        _make_fields(),
    )
    return tmp_path


@pytest.fixture
def mapped_dir(tmp_path: Path) -> Path:
    """One timestep with non-canonical field names."""
    rng = np.random.default_rng(99)
    native_fields = {
        "magnetic_x": rng.standard_normal(DIMS),
        "magnetic_y": rng.standard_normal(DIMS),
        "magnetic_z": rng.standard_normal(DIMS),
        "density": np.abs(rng.standard_normal(DIMS)) + 0.1,
    }
    _write_h5(
        tmp_path / "output_000000.h5",
        native_fields,
        grid_attrs=_grid_attrs(),
        model="custom_code",
        model_type="MHD",
        step=0,
    )
    return tmp_path


class TestParseFilePattern:
    def test_default_pattern(self) -> None:
        glob_pat, regex = _parse_file_pattern(
            "output_{step:06d}.h5",
        )
        assert glob_pat == "output_*.h5"
        assert regex.match("output_000042.h5")
        assert not regex.match("other_000042.h5")

    def test_custom_pattern(self) -> None:
        glob_pat, regex = _parse_file_pattern(
            "fields_{step:04d}.hdf5",
        )
        assert glob_pat == "fields_*.hdf5"
        m = regex.match("fields_0100.hdf5")
        assert m is not None
        assert m.group(1) == "0100"

    def test_no_prefix(self) -> None:
        glob_pat, _ = _parse_file_pattern("{step:08d}.h5")
        assert glob_pat == "*.h5"


class TestAvailableTimesteps:
    def test_finds_all_steps(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        steps = reader.available_timesteps(canonical_dir)
        assert steps == [0, 10, 20]

    def test_empty_directory(self, tmp_path: Path) -> None:
        reader = SimpleReader()
        assert reader.available_timesteps(tmp_path) == []

    def test_custom_pattern(self, tmp_path: Path) -> None:
        for s in (5, 15):
            _write_h5(
                tmp_path / f"snap_{s:04d}.h5",
                _make_fields(),
                grid_attrs=_grid_attrs(),
            )
        reader = SimpleReader(
            file_pattern="snap_{step:04d}.h5",
        )
        assert reader.available_timesteps(tmp_path) == [5, 15]


class TestReadTimestepAutoDetect:
    def test_reads_fields_with_correct_values_and_grid(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0)
        assert ds.has_field("B_1")
        assert ds.has_field("B_2")
        assert ds.has_field("B_3")
        assert ds.has_field("rho_m")
        assert ds["B_1"].shape == DIMS
        expected = _make_fields()
        assert_array_equal(ds["B_1"], expected["B_1"])
        assert ds.grid.dimensions == DIMS
        assert ds.grid.spacing == SPACING
        assert ds.grid.origin == ORIGIN
        assert ds.has_field("Bx")
        assert_array_equal(ds["Bx"], ds["B_1"])

    def test_aliases_metadata_and_missing_file(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 10)
        assert ds.metadata["step"] == 10
        assert ds.metadata["time"] == pytest.approx(1.0)
        with pytest.raises(FileNotFoundError):
            reader.read_timestep(canonical_dir, 999)


class TestReadTimestepWithConfig:
    def test_config_fallback_when_no_metadata(
        self,
        bare_dir: Path,
    ) -> None:
        config = _sample_config()
        reader = SimpleReader(config=config)
        ds = reader.read_timestep(bare_dir, 0)
        assert ds.grid.dimensions == DIMS
        assert ds.has_field("B_1")

    def test_hdf5_metadata_takes_priority(
        self,
        canonical_dir: Path,
    ) -> None:
        different_grid = GridInfo(
            dimensions=(10, 10, 10),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 1.0, 1.0),
            geometry=CARTESIAN,
        )
        config = _sample_config(grid=different_grid)
        reader = SimpleReader(config=config)
        ds = reader.read_timestep(canonical_dir, 0)
        # HDF5 metadata wins
        assert ds.grid.dimensions == DIMS
        assert ds.grid.spacing == SPACING

    def test_physics_from_config(
        self,
        canonical_dir: Path,
    ) -> None:
        config = _sample_config()
        reader = SimpleReader(config=config)
        ds = reader.read_timestep(canonical_dir, 0)
        assert ds.physics.gamma == pytest.approx(5.0 / 3.0)

    def test_explicit_grid_without_config(
        self,
        bare_dir: Path,
    ) -> None:
        grid = _sample_grid()
        reader = SimpleReader(grid=grid)
        ds = reader.read_timestep(bare_dir, 0)
        assert ds.grid.dimensions == DIMS
        assert ds.has_field("B_1")

    def test_explicit_normalization(
        self,
        canonical_dir: Path,
    ) -> None:
        norm = Normalization.identity()
        reader = SimpleReader(normalization=norm)
        ds = reader.read_timestep(canonical_dir, 0)
        assert ds.normalization is norm


class TestFieldMapping:
    def test_maps_native_to_canonical(
        self,
        mapped_dir: Path,
    ) -> None:
        field_map = {
            "magnetic_x": "B_1",
            "magnetic_y": "B_2",
            "magnetic_z": "B_3",
            "density": "rho_m",
        }
        reader = SimpleReader(field_map=field_map)
        ds = reader.read_timestep(mapped_dir, 0)
        assert ds.has_field("B_1")
        assert ds.has_field("Bx")
        assert ds.has_field("rho_m")

    def test_unmapped_fields_pass_through(
        self,
        mapped_dir: Path,
    ) -> None:
        partial_map = {"magnetic_x": "B_1"}
        reader = SimpleReader(field_map=partial_map)
        ds = reader.read_timestep(mapped_dir, 0)
        assert ds.has_field("B_1")
        # Unmapped fields pass through with native names
        assert ds.has_field("density")
        assert ds.has_field("magnetic_y")
        assert ds.has_field("magnetic_z")


class TestFieldsAtRoot:
    def test_explicit_root(self, tmp_path: Path) -> None:
        _write_h5(
            tmp_path / "output_000000.h5",
            _make_fields(),
            fields_group="",
            grid_attrs=_grid_attrs(),
        )
        reader = SimpleReader(fields_group="")
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("B_1")

    def test_scalars_skipped_without_field_map(
        self,
        tmp_path: Path,
    ) -> None:
        """Scalar datasets at root are not read as fields."""
        filepath = tmp_path / "output_000000.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("B_1", data=np.ones(DIMS))
            f.create_dataset("step", data=42)  # scalar
            f.create_dataset("x", data=np.arange(4.0))  # 1-D
            g = f.create_group("grid")
            for k, v in _grid_attrs().items():
                g.attrs[k] = v
        reader = SimpleReader()
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("B_1")
        assert not ds.has_field("step")
        assert not ds.has_field("x")


class TestFieldsGroupMisconfig:
    def test_fields_group_pointing_at_dataset_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """fields_group= pointing at a Dataset (not a Group) fails with
        a clear TypeError rather than a confusing downstream crash."""
        filepath = tmp_path / "output_000000.h5"
        with h5py.File(filepath, "w") as f:
            # "fields" is a Dataset here, not a Group
            f.create_dataset("fields", data=np.ones(DIMS))
            g = f.create_group("grid")
            for k, v in _grid_attrs().items():
                g.attrs[k] = v
        reader = SimpleReader(fields_group="fields")
        with pytest.raises(TypeError, match="expected a Group"):
            reader.read_timestep(tmp_path, 0)


class TestMissingMetadataError:
    def test_no_grid_no_config_raises(
        self,
        bare_dir: Path,
    ) -> None:
        reader = SimpleReader()
        with pytest.raises(ValueError, match="grid"):
            reader.read_timestep(bare_dir, 0)


class TestOpenSimple:
    def test_with_hdf5_metadata(
        self,
        canonical_dir: Path,
    ) -> None:
        sim = open_simple(canonical_dir)
        assert sim.model_name == "test_sim"
        assert sim.model_type == "MHD"
        assert sim.grid.dimensions == DIMS

        ds = sim.read(step=0)
        assert ds.has_field("B_1")

    def test_config_sources(
        self,
        bare_dir: Path,
    ) -> None:
        config = _sample_config()
        sim = open_simple(bare_dir, config=config)
        assert sim.config is config
        ds = sim.read(step=0)
        assert ds.grid.dimensions == DIMS

        grid = _sample_grid()
        sim2 = open_simple(bare_dir, grid=grid)
        assert sim2.grid.dimensions == DIMS
        ds2 = sim2.read(step=0)
        assert ds2.has_field("B_1")

    def test_tuple_unpacking_still_works(
        self,
        canonical_dir: Path,
    ) -> None:
        reader, config = open_simple(canonical_dir)
        assert config.model_name == "test_sim"
        ds = reader.read_timestep(canonical_dir, 0)
        assert ds.has_field("B_1")

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_simple(tmp_path)

    def test_simulation_toml_auto_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        _write_h5(
            tmp_path / "output_000000.h5",
            _make_fields(),
        )
        toml_content = """\
[schema]
version = "2.0"
[model]
name = "toml_sim"
type = "MHD"
[run]
name = "simple_reader_test"
[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 10
[grid]
dimensions = [4, 3, 2]
spacing = [1.0, 2.0, 3.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 6.0, 6.0]
[units]
anchor = "si"
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "p"
charge = 1.0
mass = 1.0
"""
        (tmp_path / "simulation.toml").write_text(toml_content)
        reader, config = open_simple(tmp_path)
        assert config.model_name == "toml_sim"
        assert config.grid.dimensions == DIMS
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("B_1")

    def test_config_path_outside_data_dir(
        self,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        conf_dir = tmp_path / "configs"
        conf_dir.mkdir()

        _write_h5(data_dir / "output_000000.h5", _make_fields())

        toml_content = """\
[schema]
version = "2.0"
[model]
name = "remote_toml"
type = "PIC"
[run]
name = "remote_toml_test"
[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 10
[grid]
dimensions = [4, 3, 2]
spacing = [1.0, 2.0, 3.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 6.0, 6.0]
[units]
anchor = "si"
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "e"
charge = -1.0
mass = 1.0
"""
        toml_file = conf_dir / "simulation.toml"
        toml_file.write_text(toml_content)

        reader, config = open_simple(
            data_dir,
            config_path=toml_file,
        )
        assert config.model_name == "remote_toml"
        assert config.model_type == "PIC"
        ds = reader.read_timestep(data_dir, 0)
        assert ds.has_field("B_1")


class TestCanReadConfidence:
    """Glob-only, as the registry contract requires: no file is opened."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert can_read_confidence(tmp_path) == 0.0

    def test_any_h5_is_a_weak_signal(self, tmp_path: Path) -> None:
        (tmp_path / "data.h5").write_bytes(b"not even HDF5")
        assert can_read_confidence(tmp_path) == pytest.approx(0.2)

    def test_default_pattern_and_toml_add_up(self, tmp_path: Path) -> None:
        (tmp_path / "output_000000.h5").touch()
        (tmp_path / "simulation.toml").touch()
        assert can_read_confidence(tmp_path) == pytest.approx(0.7)


class TestCustomReadRaw:
    def test_subclass_bypasses_hdf5(
        self,
        tmp_path: Path,
    ) -> None:
        """Custom _read_raw returns arrays without reading files."""

        class InMemoryReader(SimpleReader):
            def _read_raw(
                self,
                filepath: Any,
                **kwargs: Any,
            ) -> dict[str, np.ndarray]:
                return _make_fields()

        reader = InMemoryReader(grid=_sample_grid())
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("B_1")
        assert ds.has_field("rho_m")
        assert ds.grid.dimensions == DIMS
        expected = _make_fields()
        assert_array_equal(ds["B_1"], expected["B_1"])

    def test_field_map_applied_after_read_raw(
        self,
        tmp_path: Path,
    ) -> None:
        """field_map renames native names from _read_raw."""
        rng = np.random.default_rng(77)
        native_data = {
            "mag_x": rng.standard_normal(DIMS),
            "mag_y": rng.standard_normal(DIMS),
            "mag_z": rng.standard_normal(DIMS),
        }

        class NativeReader(SimpleReader):
            def _read_raw(
                self,
                filepath: Any,
                **kwargs: Any,
            ) -> dict[str, np.ndarray]:
                return dict(native_data)

        reader = NativeReader(
            field_map={
                "mag_x": "B_1",
                "mag_y": "B_2",
                "mag_z": "B_3",
            },
            grid=_sample_grid(),
        )
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.has_field("B_1")
        assert ds.has_field("Bx")
        assert not ds.has_field("mag_x")
        assert_array_equal(ds["B_1"], native_data["mag_x"])

    def test_missing_grid_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Custom _read_raw without grid raises ValueError."""

        class NoFileReader(SimpleReader):
            def _read_raw(
                self,
                filepath: Any,
                **kwargs: Any,
            ) -> dict[str, np.ndarray]:
                return _make_fields()

        reader = NoFileReader()
        with pytest.raises(ValueError, match="grid"):
            reader.read_timestep(tmp_path, 0)

    def test_config_provides_grid(
        self,
        tmp_path: Path,
    ) -> None:
        """config can provide grid for custom _read_raw."""

        class NoFileReader(SimpleReader):
            def _read_raw(
                self,
                filepath: Any,
                **kwargs: Any,
            ) -> dict[str, np.ndarray]:
                return _make_fields()

        config = _sample_config()
        reader = NoFileReader(config=config)
        ds = reader.read_timestep(tmp_path, 0)
        assert ds.grid.dimensions == DIMS
        assert ds.physics.gamma == pytest.approx(5.0 / 3.0)

    def test_transposed_hdf5(
        self,
        tmp_path: Path,
    ) -> None:
        """Custom _read_raw can transpose arrays from HDF5."""
        rng = np.random.default_rng(42)
        original = rng.standard_normal(DIMS)
        filepath = tmp_path / "output_000000.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("B_1", data=original.T)

        class TransposedReader(SimpleReader):
            def _read_raw(
                self,
                filepath: Any,
                **kwargs: Any,
            ) -> dict[str, np.ndarray]:
                with h5py.File(filepath, "r") as f:
                    return {
                        name: np.asarray(f[name]).T
                        for name in f
                        if isinstance(f[name], h5py.Dataset)
                    }

        reader = TransposedReader(grid=_sample_grid())
        ds = reader.read_timestep(tmp_path, 0)
        assert_array_equal(ds["B_1"], original)


class TestSelectFields:
    def test_select_fields_keeps_and_resolves(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0)
        sub = ds.select_fields(["B_1", "rho_m"])
        assert sorted(sub.field_names()) == ["B_1", "rho_m"]
        # Alias resolution
        alias_sub = ds.select_fields(["Bx"])
        assert alias_sub.has_field("B_1")
        assert alias_sub.has_field("Bx")
        assert sorted(alias_sub.field_names()) == ["B_1"]
        # Values preserved
        assert_array_equal(sub["B_1"], ds["B_1"])
        # Metadata preserved
        assert sub.grid.dimensions == ds.grid.dimensions
        assert sub.normalization is ds.normalization

    def test_missing_field_raises(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0)
        with pytest.raises(KeyError, match="nonexistent"):
            ds.select_fields(["nonexistent"])

    def test_preserves_caller_order(
        self,
        canonical_dir: Path,
    ) -> None:
        """select_fields returns fields in the order the caller asked for.

        Regression for the set+sorted round-trip that imposed
        alphabetical order on output regardless of request order.
        """
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0)
        requested = ["rho_m", "B_2", "B_1"]
        sub = ds.select_fields(requested)
        assert sub.field_names() == requested

    def test_deduplicates_aliases_preserving_first_occurrence(
        self,
        canonical_dir: Path,
    ) -> None:
        """Duplicate / aliased names collapse to one entry at first position."""
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0)
        # "Bx" resolves to "B_1" → same canonical field as the later "B_1"
        sub = ds.select_fields(["rho_m", "Bx", "B_1"])
        assert sub.field_names() == ["rho_m", "B_1"]


class TestSelectiveRead:
    def test_reads_only_requested_fields(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        ds = reader.read_timestep(canonical_dir, 0, fields={"B_1", "rho_m"})
        assert sorted(ds.field_names()) == ["B_1", "rho_m"]

    def test_values_match_full_read(
        self,
        canonical_dir: Path,
    ) -> None:
        reader = SimpleReader()
        full = reader.read_timestep(canonical_dir, 0)
        sub = reader.read_timestep(canonical_dir, 0, fields={"B_1"})
        assert_array_equal(sub["B_1"], full["B_1"])

    def test_with_field_map(
        self,
        mapped_dir: Path,
    ) -> None:
        field_map = {
            "magnetic_x": "B_1",
            "magnetic_y": "B_2",
            "magnetic_z": "B_3",
            "density": "rho_m",
        }
        reader = SimpleReader(field_map=field_map)
        ds = reader.read_timestep(mapped_dir, 0, fields={"B_1", "rho_m"})
        assert sorted(ds.field_names()) == ["B_1", "rho_m"]


class TestSelectiveReadViaSimulation:
    def test_alias_resolution(
        self,
        canonical_dir: Path,
    ) -> None:
        sim = open_simple(canonical_dir)
        ds = sim.read(step=0, fields=["Bx", "rho_m"])
        assert ds.has_field("B_1")
        assert ds.has_field("rho_m")
        assert not ds.has_field("B_2")

    def test_none_reads_all(
        self,
        canonical_dir: Path,
    ) -> None:
        sim = open_simple(canonical_dir)
        ds = sim.read(step=0)
        assert len(ds.field_names()) == 4


class TestDataInSI:
    r"""``[units].data_in_si`` rescales SI arrays against the declared anchor.

    A generic HDF5 file carries no unit convention, so the deck is the
    only thing that can say its numbers are SI. Without the flag the
    arrays are read as code units and every quantity carrying $\mu_0$
    comes out wrong — which is what SI-emitting codes hit before 2.0.
    """

    B_SI = 5.0e-9  # 5 nT, solar wind
    N_SI = 5.0e6  # 5 cm^-3

    def _sim_dir(self, tmp_path: Path, *, data_in_si: bool) -> Path:
        rho = self.N_SI * constants.m_p
        with h5py.File(tmp_path / "output_000000.h5", "w") as f:
            g = f.create_group("fields")
            for index, value in enumerate((self.B_SI, 0.0, 0.0), start=1):
                g.create_dataset(f"B_{index}", data=np.full((2, 2, 2), value))
            g.create_dataset("rho_m", data=np.full((2, 2, 2), rho))
        (tmp_path / "simulation.toml").write_text(f"""\
[schema]
version = "2.0"
[model]
name = "si_sim"
type = "vlasov"
[run]
name = "solar_wind"
[time]
scheme = "fixed"
dt = 1.0
t_start = 0.0
t_end = 1.0
n_steps = 1
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
lower = [0.0, 0.0, 0.0]
upper = [2.0, 2.0, 2.0]
[units]
anchor = "explicit"
data_in_si = {str(data_in_si).lower()}
reference_length = 6.371e6
reference_mass_density = 8.35e-21
reference_b_field = 5.0e-9
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "protons"
charge = 1.0
mass = 1.0
""")
        return tmp_path

    def test_si_arrays_reach_si_quantities_unchanged(self, tmp_path: Path) -> None:
        """Round trip: SI in, rescale on load, SI back out."""
        sim = open_simple(self._sim_dir(tmp_path, data_in_si=True))
        ds = sim.read(step=0, fields=["B", "rho_m"])
        rho = self.N_SI * constants.m_p
        np.testing.assert_allclose(
            float(np.mean(ds.in_si("v_A"))),
            self.B_SI / np.sqrt(constants.mu_0 * rho),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(np.mean(ds.in_si("e_B"))),
            self.B_SI**2 / (2 * constants.mu_0),
            rtol=1e-12,
        )

    def test_without_the_flag_the_arrays_are_read_as_code_units(
        self, tmp_path: Path
    ) -> None:
        """The pre-2.0 behaviour, kept as the default."""
        sim = open_simple(self._sim_dir(tmp_path, data_in_si=False))
        ds = sim.read(step=0, fields=["B", "rho_m"])
        assert float(np.mean(ds["B_1"])) == self.B_SI
