"""Tests for GridInfo, FieldDataset, SimulationConfig, and SimulationReader."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pypic.coordinates import CARTESIAN, CYLINDRICAL, SPHERICAL
from pypic.readers.base import (
    FieldDataset,
    GridInfo,
    SimulationConfig,
    SimulationReader,
    _default_aliases,
)
from pypic.units import Normalization, SpeciesInfo


@pytest.fixture
def sample_grid():
    return GridInfo(
        dimensions=(8, 6, 4),
        spacing=(0.5, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )


@pytest.fixture
def sample_fields():
    rng = np.random.default_rng(42)
    return {
        "B1": rng.standard_normal((8, 6, 4)),
        "B2": rng.standard_normal((8, 6, 4)),
        "B3": rng.standard_normal((8, 6, 4)),
        "rho_c": rng.standard_normal((8, 6, 4)),
    }


@pytest.fixture
def sample_dataset(sample_grid, sample_fields):
    return FieldDataset.from_arrays(
        sample_fields, sample_grid, Normalization.identity()
    )


class TestGridInfo:
    def test_construction(self, sample_grid):
        assert sample_grid.dimensions == (8, 6, 4)
        assert sample_grid.spacing == (0.5, 0.5, 0.5)
        assert sample_grid.origin == (0.0, 0.0, 0.0)
        assert sample_grid.geometry is CARTESIAN

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

    def test_frozen(self, sample_grid):
        with pytest.raises(AttributeError):
            sample_grid.dimensions = (1, 1, 1)  # type: ignore[misc]

    def test_2d_grid(self):
        grid = GridInfo(
            dimensions=(10, 5),
            spacing=(1.0, 2.0),
            origin=(0.0, 0.0),
            geometry=CARTESIAN,
        )
        coords = grid.coordinate_arrays()
        assert len(coords) == 2
        assert len(coords[0]) == 10
        assert len(coords[1]) == 5

    def test_1d_grid(self):
        grid = GridInfo(
            dimensions=(20,), spacing=(0.1,), origin=(0.0,), geometry=CARTESIAN
        )
        (x,) = grid.coordinate_arrays()
        assert len(x) == 20
        assert_allclose(x[0], 0.05)

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

    def test_validation_negative_spacing(self):
        with pytest.raises(ValueError, match="must be > 0"):
            GridInfo(
                dimensions=(8,),
                spacing=(-0.5,),
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
        result = sample_dataset["B1"]
        assert isinstance(result, np.ndarray)

    def test_zero_copy(self, sample_fields, sample_grid):
        original = sample_fields["B1"]
        ds = FieldDataset.from_arrays(
            sample_fields, sample_grid, Normalization.identity()
        )
        assert np.shares_memory(ds["B1"], original)

    def test_alias_access_cartesian(self, sample_dataset, sample_fields):
        assert_allclose(sample_dataset["Bx"], sample_fields["B1"])
        assert_allclose(sample_dataset["By"], sample_fields["B2"])
        assert_allclose(sample_dataset["Bz"], sample_fields["B3"])

    def test_has_field_canonical(self, sample_dataset):
        assert sample_dataset.has_field("B1")
        assert sample_dataset.has_field("rho_c")

    def test_has_field_alias(self, sample_dataset):
        assert sample_dataset.has_field("Bx")
        assert sample_dataset.has_field("By")

    def test_has_field_missing(self, sample_dataset):
        assert not sample_dataset.has_field("pressure")

    def test_field_names_canonical_only(self, sample_dataset):
        names = sample_dataset.field_names()
        assert sorted(names) == ["B1", "B2", "B3", "rho_c"]
        assert "Bx" not in names

    def test_xr_returns_dataset(self, sample_dataset):
        assert isinstance(sample_dataset.xr, xr.Dataset)

    def test_missing_key_raises_with_message(self, sample_dataset):
        with pytest.raises(KeyError, match="not found"):
            sample_dataset["nonexistent"]


class TestFieldDatasetSlicing:
    def test_isel_scalar_reduces_dim(self, sample_dataset):
        sliced = sample_dataset.isel(z=0)
        assert sliced["B1"].ndim == 2
        assert sliced["B1"].shape == (8, 6)
        assert len(sliced.grid.dimensions) == 2

    def test_isel_slice_keeps_dim(self, sample_dataset):
        sliced = sample_dataset.isel(z=slice(0, 2))
        assert sliced["B1"].shape == (8, 6, 2)
        assert sliced.grid.dimensions == (8, 6, 2)

    def test_sel_scalar_drops_dim(self, sample_dataset):
        z_coord = sample_dataset.xr.coords["z"].values[1]
        sliced = sample_dataset.sel(z=z_coord)
        assert sliced["B1"].ndim == 2
        assert len(sliced.grid.dimensions) == 2

    def test_sel_method_nearest(self, sample_dataset):
        sliced = sample_dataset.sel(z=0.3, method="nearest")
        assert sliced["B1"].ndim == 2

    def test_metadata_preserved(self, sample_grid, sample_fields):
        species = [SpeciesInfo(name="e", charge=-1.0, mass=1.0)]
        ds = FieldDataset.from_arrays(
            sample_fields,
            sample_grid,
            Normalization.identity(),
            species=species,
            physics={"eta": 0.01},
            metadata={"run": "test"},
        )
        sliced = ds.isel(z=0)
        assert sliced.species == species
        assert sliced.physics == {"eta": 0.01}
        assert sliced.metadata == {"run": "test"}
        assert sliced.normalization is ds.normalization

    def test_dict_indexers_for_unicode_dims(self):
        """Spherical grid has θ and φ which can't be kwargs."""
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(1.0, 0.5, 1.0),
            origin=(1.0, 0.0, 0.0),
            geometry=SPHERICAL,
        )
        fields = {"B1": np.ones((4, 3, 2))}
        ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())
        sliced = ds.isel({"θ": 0})
        assert sliced["B1"].shape == (4, 2)

    def test_grid_origin_updated_after_slice(self, sample_dataset):
        sliced = sample_dataset.isel(x=slice(2, 6))
        assert sliced.grid.dimensions == (4, 6, 4)
        expected_origin = sample_dataset.grid.origin[0] + 2 * 0.5
        assert_allclose(sliced.grid.origin[0], expected_origin)


class TestAliases:
    def test_cartesian_aliases(self):
        aliases = _default_aliases(CARTESIAN)
        assert aliases["Bx"] == "B1"
        assert aliases["By"] == "B2"
        assert aliases["Ez"] == "E3"

    def test_spherical_aliases(self):
        aliases = _default_aliases(SPHERICAL)
        assert aliases["Br"] == "B1"
        assert aliases["Btheta"] == "B2"
        assert aliases["Bphi"] == "B3"

    def test_cylindrical_aliases(self):
        aliases = _default_aliases(CYLINDRICAL)
        assert aliases["Br"] == "B1"
        assert aliases["Bphi"] == "B2"
        assert aliases["Bz"] == "B3"

    def test_custom_aliases_override(self, sample_grid, sample_fields):
        ds = FieldDataset.from_arrays(
            sample_fields,
            sample_grid,
            Normalization.identity(),
            aliases={"Bperp": "B1", "Bx": "B2"},
        )
        # Custom alias overrides geometry default
        assert_allclose(ds["Bx"], sample_fields["B2"])
        assert_allclose(ds["Bperp"], sample_fields["B1"])

    def test_aliases_only_for_existing_fields(self, sample_grid):
        """Aliases for fields not in the dataset are silently dropped."""
        ds = FieldDataset.from_arrays(
            {"B1": np.ones((8, 6, 4))},
            sample_grid,
            Normalization.identity(),
        )
        assert ds.has_field("Bx")
        assert not ds.has_field("By")  # B2 doesn't exist


class TestSimulationConfig:
    def test_construction(self):
        grid = GridInfo(
            dimensions=(4,),
            spacing=(1.0,),
            origin=(0.0,),
            geometry=CARTESIAN,
        )
        cfg = SimulationConfig(
            model_name="run1",
            model_type="pic",
            grid=grid,
            normalization=Normalization.identity(),
            species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
            physics={},
            geometry=CARTESIAN,
            frame="simulation",
            metadata={},
        )
        assert cfg.model_name == "run1"
        assert cfg.species[0].name == "e"

    def test_frozen(self):
        grid = GridInfo(
            dimensions=(4,),
            spacing=(1.0,),
            origin=(0.0,),
            geometry=CARTESIAN,
        )
        cfg = SimulationConfig(
            model_name="run1",
            model_type="pic",
            grid=grid,
            normalization=Normalization.identity(),
            species=(),
            physics={},
            geometry=CARTESIAN,
            frame="simulation",
            metadata={},
        )
        with pytest.raises(AttributeError):
            cfg.model_name = "changed"  # type: ignore[misc]


class TestSimulationReader:
    def test_protocol_satisfied(self):
        class MyReader:
            def read_timestep(self, path: Path, step: int) -> FieldDataset: ...
            def available_timesteps(self, path: Path) -> list[int]:
                return []

        assert isinstance(MyReader(), SimulationReader)

    def test_protocol_not_satisfied(self):
        class NotAReader:
            pass

        assert not isinstance(NotAReader(), SimulationReader)
