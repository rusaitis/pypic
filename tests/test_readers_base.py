"""Tests for GridInfo, FieldDataset, SimulationConfig, and SimulationReader."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pypic import open_simulation
from pypic.coordinates import CARTESIAN, CYLINDRICAL, SPHERICAL
from pypic.dataset import FieldDataset, _default_aliases
from pypic.grid import GridInfo
from pypic.readers import ReaderBase
from pypic.readers._protocols import SimulationReader
from pypic.units import Normalization, PhysicsParams, SpeciesInfo
from tests._helpers import make_synthetic_fielddataset, make_uniform_grid
from tests._sim_fixtures import make_sim_dir

DATA = Path(__file__).parent / "data"


@pytest.fixture
def sample_grid():
    return make_uniform_grid(8, 6, 4, spacing=0.5)


@pytest.fixture
def sample_fields():
    rng = np.random.default_rng(42)
    return {
        "B_1": rng.standard_normal((8, 6, 4)),
        "B_2": rng.standard_normal((8, 6, 4)),
        "B_3": rng.standard_normal((8, 6, 4)),
        "rho_c": rng.standard_normal((8, 6, 4)),
    }


@pytest.fixture
def sample_dataset(sample_grid):
    return make_synthetic_fielddataset(sample_grid, ("B_1", "B_2", "B_3", "rho_c"))


class TestGridInfo:
    def test_coordinate_arrays_cell_centered(self, sample_grid):
        x, y, z = sample_grid.coordinate_arrays()
        assert_allclose(x, 0.0 + (np.arange(8) + 0.5) * 0.5)
        assert_allclose(y, 0.0 + (np.arange(6) + 0.5) * 0.5)
        assert_allclose(z, 0.0 + (np.arange(4) + 0.5) * 0.5)

    def test_coordinate_arrays_with_offset_origin(self):
        grid = GridInfo(
            dimensions=(3,), spacing=(2.0,), origin=(10.0,), geometry=CARTESIAN
        )
        (x,) = grid.coordinate_arrays()
        assert_allclose(x, [11.0, 13.0, 15.0])

    @pytest.mark.parametrize(
        ("dims", "spacing", "origin", "expected_ndim", "expected_len"),
        [
            ((10, 5), (1.0, 2.0), (0.0, 0.0), 2, (10, 5)),
            ((20,), (0.1,), (0.0,), 1, (20,)),
        ],
        ids=["2d", "1d"],
    )
    def test_lower_dimensional_grid(
        self, dims, spacing, origin, expected_ndim, expected_len
    ):
        grid = GridInfo(
            dimensions=dims, spacing=spacing, origin=origin, geometry=CARTESIAN
        )
        coords = grid.coordinate_arrays()
        assert len(coords) == expected_ndim
        for i, c in enumerate(coords):
            assert len(c) == expected_len[i]
        if expected_ndim == 1:
            assert_allclose(coords[0][0], origin[0] + 0.5 * spacing[0])

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            GridInfo(
                dimensions=(8, 6),
                spacing=(0.5,),
                origin=(0.0, 0.0),
                geometry=CARTESIAN,
            )

    def test_validation_zero_dimension(self):
        with pytest.raises(ValueError, match="must be > 0"):
            GridInfo(
                dimensions=(0, 6),
                spacing=(0.5, 0.5),
                origin=(0.0, 0.0),
                geometry=CARTESIAN,
            )

    @pytest.mark.parametrize("bad_spacing", [-0.5, 0.0], ids=["negative", "zero"])
    def test_validation_negative_spacing(self, bad_spacing):
        """Zero spacing is a degenerate grid and must fail like negative does.

        Guards the ``spacing[i] > 0`` check from silently degrading to
        ``>= 0`` — a zero dx silently breaks every finite-difference
        diagnostic (divide-by-zero, NaNs far downstream).
        """
        with pytest.raises(ValueError, match="must be > 0"):
            GridInfo(
                dimensions=(8,),
                spacing=(bad_spacing,),
                origin=(0.0,),
                geometry=CARTESIAN,
            )

    def test_validation_boundary_length_mismatch(self):
        with pytest.raises(ValueError, match="boundary length"):
            GridInfo(
                dimensions=(8, 6),
                spacing=(0.5, 0.5),
                origin=(0.0, 0.0),
                geometry=CARTESIAN,
                boundary=("periodic",),
            )

    def test_boundary_valid(self):
        grid = GridInfo(
            dimensions=(8, 6),
            spacing=(0.5, 0.5),
            origin=(0.0, 0.0),
            geometry=CARTESIAN,
            boundary=("periodic", "open"),
        )
        assert grid.boundary == ("periodic", "open")


class TestFieldDatasetAccess:
    def test_getitem_returns_ndarray(self, sample_dataset):
        result = sample_dataset["B_1"]
        assert isinstance(result, np.ndarray)

    def test_zero_copy(self, sample_fields, sample_grid):
        original = sample_fields["B_1"]
        ds = FieldDataset.from_arrays(
            sample_fields, sample_grid, Normalization.identity()
        )
        assert np.shares_memory(ds["B_1"], original)

    def test_alias_access_cartesian(self, sample_dataset, sample_fields):
        assert_allclose(sample_dataset["Bx"], sample_fields["B_1"])
        assert_allclose(sample_dataset["By"], sample_fields["B_2"])
        assert_allclose(sample_dataset["Bz"], sample_fields["B_3"])

    def test_has_field(self, sample_dataset):
        assert sample_dataset.has_field("B_1")
        assert sample_dataset.has_field("rho_c")
        assert sample_dataset.has_field("Bx")
        assert sample_dataset.has_field("By")
        assert not sample_dataset.has_field("pressure")

    def test_field_names_canonical_only(self, sample_dataset):
        names = sample_dataset.field_names()
        assert sorted(names) == ["B_1", "B_2", "B_3", "rho_c"]
        assert "Bx" not in names

    def test_xr_returns_dataset(self, sample_dataset):
        assert isinstance(sample_dataset.xr, xr.Dataset)

    def test_missing_key_raises_with_message(self, sample_dataset):
        with pytest.raises(KeyError, match="not found"):
            sample_dataset["nonexistent"]

    def test_did_you_mean_close_match(self, sample_dataset):
        """Typo 'rho' with 'rho_c' present should suggest close match."""
        with pytest.raises(KeyError, match="Did you mean") as exc_info:
            sample_dataset["rho"]
        assert "rho_c" in str(exc_info.value)

    def test_did_you_mean_alias_suggestion(self):
        grid = GridInfo(
            dimensions=(2,), spacing=(1.0,), origin=(0.0,), geometry=CARTESIAN
        )
        ds = FieldDataset.from_arrays(
            {"B_1": np.array([1.0, 2.0])}, grid, Normalization.identity()
        )
        with pytest.raises(KeyError, match="Did you mean") as exc_info:
            ds["Bx1"]
        msg = str(exc_info.value)
        assert "Bx" in msg or "B_1" in msg

    def test_no_suggestion_for_unrelated_key(self, sample_dataset):
        with pytest.raises(KeyError) as exc_info:
            sample_dataset["completely_wrong_name"]
        assert "Did you mean" not in str(exc_info.value)


class TestFieldDatasetSlicing:
    def test_isel_scalar_reduces_dim(self, sample_dataset):
        sliced = sample_dataset.isel(z=0)
        assert sliced["B_1"].ndim == 2
        assert sliced["B_1"].shape == (8, 6)
        assert len(sliced.grid.dimensions) == 2

    def test_isel_slice_keeps_dim(self, sample_dataset):
        sliced = sample_dataset.isel(z=slice(0, 2))
        assert sliced["B_1"].shape == (8, 6, 2)
        assert sliced.grid.dimensions == (8, 6, 2)

    def test_sel_scalar_drops_dim(self, sample_dataset):
        z_coord = sample_dataset.xr.coords["z"].values[1]
        sliced = sample_dataset.sel(z=z_coord)
        assert sliced["B_1"].ndim == 2
        assert len(sliced.grid.dimensions) == 2

    def test_sel_method_nearest(self, sample_dataset):
        sliced = sample_dataset.sel(z=0.3, method="nearest")
        assert sliced["B_1"].ndim == 2

    def test_metadata_preserved(self, sample_grid, sample_fields):
        species = (SpeciesInfo(name="e", charge=-1.0, mass=1.0),)
        ds = FieldDataset.from_arrays(
            sample_fields,
            sample_grid,
            Normalization.identity(),
            species=species,
            physics=PhysicsParams(extra={"eta": 0.01}),
            metadata={"label": "test"},
        )
        sliced = ds.isel(z=0)
        assert sliced.species == species
        assert sliced.physics.extra["eta"] == 0.01
        assert sliced.metadata == {"label": "test"}
        assert sliced.normalization is ds.normalization

    def test_dict_indexers_for_unicode_dims(self):
        """Spherical grid has θ and φ which can't be kwargs."""
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(1.0, 0.5, 1.0),
            origin=(1.0, 0.0, 0.0),
            geometry=SPHERICAL,
        )
        fields = {"B_1": np.ones((4, 3, 2))}
        ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())
        sliced = ds.isel({"θ": 0})
        assert sliced["B_1"].shape == (4, 2)

    def test_grid_origin_updated_after_slice(self, sample_dataset):
        sliced = sample_dataset.isel(x=slice(2, 6))
        assert sliced.grid.dimensions == (4, 6, 4)
        expected_origin = sample_dataset.grid.origin[0] + 2 * 0.5
        assert_allclose(sliced.grid.origin[0], expected_origin)


class TestAliases:
    @pytest.mark.parametrize(
        ("geometry", "checks"),
        [
            (CARTESIAN, {"Bx": "B_1", "By": "B_2", "Ez": "E_3"}),
            (SPHERICAL, {"Br": "B_1", "Btheta": "B_2", "Bphi": "B_3"}),
            (CYLINDRICAL, {"Br": "B_1", "Bphi": "B_2", "Bz": "B_3"}),
        ],
        ids=["cartesian", "spherical", "cylindrical"],
    )
    def test_geometry_aliases(self, geometry, checks):
        aliases = _default_aliases(geometry)
        for alias, canonical in checks.items():
            assert aliases[alias] == canonical

    def test_custom_aliases_override(self, sample_grid, sample_fields):
        ds = FieldDataset.from_arrays(
            sample_fields,
            sample_grid,
            Normalization.identity(),
            aliases={"Bperp": "B_1", "Bx": "B_2"},
        )
        # Custom alias overrides geometry default
        assert_allclose(ds["Bx"], sample_fields["B_2"])
        assert_allclose(ds["Bperp"], sample_fields["B_1"])

    def test_aliases_only_for_existing_fields(self, sample_grid):
        """Aliases for fields not in the dataset are silently dropped."""
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones((8, 6, 4))},
            sample_grid,
            Normalization.identity(),
        )
        assert ds.has_field("Bx")
        assert not ds.has_field("By")  # B_2 doesn't exist

    def test_n_e_n_i_aliases(self, sample_grid):
        """n_e and n_i resolve to n_s0 and n_s1."""
        fields = {
            "n_s0": np.full((8, 6, 4), 1e18),
            "n_s1": np.full((8, 6, 4), 1e18),
        }
        ds = FieldDataset.from_arrays(fields, sample_grid, Normalization.identity())
        assert ds.has_field("n_e")
        assert ds.has_field("n_i")
        assert_allclose(ds["n_e"], fields["n_s0"])
        assert_allclose(ds["n_i"], fields["n_s1"])

    def test_n_e_alias_inactive_without_n_s0(self, sample_grid):
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones((8, 6, 4))}, sample_grid, Normalization.identity()
        )
        assert not ds.has_field("n_e")

    def test_four_velocity_aliases(self, sample_grid):
        """ux/uy/uz resolve to u_1/u_2/u_3."""
        fields = {
            "u_1": np.ones((8, 6, 4)),
            "u_2": np.full((8, 6, 4), 2.0),
            "u_3": np.full((8, 6, 4), 3.0),
        }
        ds = FieldDataset.from_arrays(fields, sample_grid, Normalization.identity())
        assert ds.has_field("ux")
        assert ds.has_field("uy")
        assert ds.has_field("uz")
        assert_allclose(ds["ux"], fields["u_1"])
        assert_allclose(ds["uz"], fields["u_3"])


class TestSimulationReader:
    def test_protocol_satisfied(self):
        class MyReader:
            def read_timestep(self, path: Path, step: int) -> FieldDataset: ...
            def available_timesteps(self, path: Path) -> list[int]:
                return []

        assert isinstance(MyReader(), SimulationReader)

    def test_supports_selective_read(self):
        from collections.abc import Iterable

        from pypic.readers._protocols import supports_selective_read

        class WithFields:
            def read_timestep(
                self,
                path: Path,
                step: int,
                *,
                fields: Iterable[str] | None = None,
            ) -> FieldDataset: ...

            def available_timesteps(self, path: Path) -> list[int]:
                return []

        class WithoutFields:
            def read_timestep(self, path: Path, step: int) -> FieldDataset: ...

            def available_timesteps(self, path: Path) -> list[int]:
                return []

        assert supports_selective_read(WithFields())
        assert not supports_selective_read(WithoutFields())


class TestSimulationConfigImmutability:
    def test_physics_not_mutable(self):
        from pypic.containers import SimulationConfig
        from pypic.units import Normalization, PhysicsParams, SpeciesInfo

        cfg = SimulationConfig(
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
            physics=PhysicsParams(gamma=5.0 / 3.0),
            metadata={"run_id": "abc"},
        )
        with pytest.raises(AttributeError):
            cfg.physics.gamma = 999  # type: ignore[misc]
        with pytest.raises(TypeError):
            cfg.metadata["new_key"] = "bad"  # type: ignore[index]

    def test_physics_extra_not_mutable(self):
        from pypic.containers import SimulationConfig
        from pypic.units import Normalization, PhysicsParams

        cfg = SimulationConfig(
            model_name="test",
            model_type="PIC",
            grid=GridInfo(
                dimensions=(4,),
                spacing=(1.0,),
                origin=(0.0,),
                geometry=CARTESIAN,
            ),
            normalization=Normalization.identity(),
            physics=PhysicsParams(gamma=5.0 / 3.0, extra={"eta": 0.01}),
        )
        with pytest.raises(TypeError):
            cfg.physics.extra["eta"] = 999  # type: ignore[index]
        assert cfg.physics.gamma == pytest.approx(5.0 / 3.0)


class TestWithDerived:
    def test_single_field(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|")
        assert ds.has_field("|B|")
        expected = np.sqrt(
            sample_dataset["B_1"] ** 2
            + sample_dataset["B_2"] ** 2
            + sample_dataset["B_3"] ** 2
        )
        assert_allclose(ds["|B|"], expected)

    def test_multiple_fields(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|", "e_B")
        assert ds.has_field("|B|")
        assert ds.has_field("e_B")

    def test_idempotent_for_existing_fields(self, sample_dataset):
        """Fields already in the dataset are not recomputed."""
        ds = sample_dataset.with_derived("B_1")
        assert ds.has_field("B_1")
        assert_allclose(ds["B_1"], sample_dataset["B_1"])

    def test_survives_isel(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|")
        sliced = ds.isel(z=0)
        assert sliced.has_field("|B|")
        assert sliced["|B|"].shape == (8, 6)

    def test_survives_plane_selection(self, sample_dataset):
        from pypic.selections import PlaneSelection

        ds = sample_dataset.with_derived("|B|")
        sliced = PlaneSelection(normal="z").apply(ds)
        assert sliced.has_field("|B|")

    def test_unit_conversion(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|")
        si_vals = ds.in_si("|B|")
        assert si_vals.shape == ds["|B|"].shape

    def test_field_info_populated(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|")
        info = ds.field_info("|B|")
        assert info.quantity_type == "b_field"

    def test_chaining_avoids_recomputation(self):
        """with_derived("|B|", "e_B") should compute |B| once."""
        shape = (4, 3, 2)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(shape), "B_2": np.zeros(shape), "B_3": np.zeros(shape)},
            GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0)),
            Normalization.identity(),
        )
        # |B| is a dependency of e_B, so computing it first helps
        ds = ds.with_derived("|B|", "e_B")
        assert ds.has_field("|B|")
        assert ds.has_field("e_B")
        # e_B = |B|^2 / 2 = 0.5
        assert_allclose(ds["e_B"], 0.5)

    def test_vector_siblings(self):
        """Computing a vector component stores all sibling components."""
        shape = (4, 3, 2)
        ds = FieldDataset.from_arrays(
            {
                "E_1": np.ones(shape),
                "E_2": np.zeros(shape),
                "E_3": np.zeros(shape),
                "B_1": np.zeros(shape),
                "B_2": np.zeros(shape),
                "B_3": np.ones(shape),
            },
            GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0)),
            Normalization.identity(),
        )
        ds = ds.with_derived("S_1")
        # Poynting flux siblings S_2, S_3 should also be stored
        assert ds.has_field("S_1")
        assert ds.has_field("S_2")
        assert ds.has_field("S_3")


class TestWithFieldAutoFill:
    def test_known_field_auto_fills(self, sample_dataset):
        """with_field for a registered name auto-fills metadata."""
        b_mag = np.ones(sample_dataset.grid.dimensions)
        ds = sample_dataset.with_field("|B|", b_mag)
        info = ds.field_info("|B|")
        assert info.quantity_type == "b_field"
        assert info.long_name != ""

    def test_unknown_field_without_qt_raises(self, sample_dataset):
        data = np.ones(sample_dataset.grid.dimensions)
        with pytest.raises(ValueError, match="Unknown field"):
            sample_dataset.with_field("totally_unknown_xyz", data)

    def test_explicit_override(self, sample_dataset):
        """Explicit quantity_type overrides registry."""
        b_mag = np.ones(sample_dataset.grid.dimensions)
        ds = sample_dataset.with_field("|B|", b_mag, "pressure")
        info = ds.field_info("|B|")
        # attrs-level metadata takes precedence
        assert info.quantity_type == "pressure"


class TestSnapshotTime:
    def test_recorded_time_wins_over_step_times_dt(self):
        grid = GridInfo(dimensions=(2,), spacing=(1.0,), dt=0.5)
        ds = FieldDataset.from_arrays(
            {"B_1": np.zeros(2)}, grid, metadata={"step": 4, "time": 7.25}
        )
        assert ds.time == 7.25

    def test_step_times_dt_when_no_time_recorded(self):
        grid = GridInfo(dimensions=(2,), spacing=(1.0,), dt=0.5)
        ds = FieldDataset.from_arrays({"B_1": np.zeros(2)}, grid, metadata={"step": 4})
        assert ds.time == 2.0

    def test_none_without_dt(self):
        grid = GridInfo(dimensions=(2,), spacing=(1.0,))
        ds = FieldDataset.from_arrays({"B_1": np.zeros(2)}, grid, metadata={"step": 4})
        assert ds.time is None


class TestFromArraysCoords:
    def test_non_uniform_coords_replace_the_grid_axis(self):
        grid = GridInfo(dimensions=(4, 2), spacing=(1.0, 1.0))
        x = np.array([0.0, 0.1, 0.5, 2.0])
        ds = FieldDataset.from_arrays({"B_1": np.zeros((4, 2))}, grid, coords={"x": x})
        np.testing.assert_array_equal(ds.xr.coords["x"].values, x)
        np.testing.assert_array_equal(ds.xr.coords["y"].values, [0.5, 1.5])

    def test_unknown_axis_raises(self):
        grid = GridInfo(dimensions=(4, 2), spacing=(1.0, 1.0))
        with pytest.raises(ValueError, match="coords name axes the grid lacks"):
            FieldDataset.from_arrays(
                {"B_1": np.zeros((4, 2))}, grid, coords={"z": np.zeros(3)}
            )


# (label, directory, step, whether the file or config records a time)
_READER_FIXTURES = [
    ("ipic3d/phdf5", DATA / "ipic3d-synthetic" / "phdf5", 0, True),
    ("ipic3d/shdf5", DATA / "ipic3d-synthetic" / "shdf5", 0, True),
    ("ipic3d/h5hut", DATA / "ipic3d-synthetic" / "h5hut", 0, True),
    ("batsrus/idl", DATA / "batsrus-synthetic" / "idl-uniform", 0, True),
    ("batsrus/hdf5", DATA / "batsrus-synthetic" / "hdf5-uniform", 0, True),
    ("batsrus/out", DATA / "batsrus-synthetic" / "out-ascii", 0, True),
    ("openggcm", DATA / "openggcm-small", 6300, False),
]


class TestReaderContract:
    """Every built-in reader honours the `ReaderBase` contract on real fixtures."""

    def test_every_reader_stamps_step_and_lists_what_it_reads(self, tmp_path):
        cases = [*_READER_FIXTURES, ("simple", make_sim_dir(tmp_path), 1, True)]
        problems: list[str] = []
        for label, directory, step, knows_time in cases:
            sim = open_simulation(directory)
            ds = sim.read(step)
            if not isinstance(sim.reader, ReaderBase):
                problems.append(
                    f"{label}: {type(sim.reader).__name__} lacks ReaderBase"
                )
            if ds.metadata.get("step") != step:
                problems.append(f"{label}: metadata step {ds.metadata.get('step')!r}")
            if (ds.time is not None) != knows_time:
                problems.append(f"{label}: time is {ds.time!r}")
            if sim.available_fields(step) != sorted(ds.field_names()):
                problems.append(f"{label}: listing disagrees with the read")
            if "stagger" not in ds.metadata:
                problems.append(f"{label}: no stagger provenance")
        assert not problems, "\n".join(problems)
