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
    SimulationReader,
    _default_aliases,
)
from pypic.units import Normalization, PhysicsParams, SpeciesInfo
from tests._helpers import make_synthetic_fielddataset, make_uniform_grid


@pytest.fixture
def sample_grid():
    return make_uniform_grid(8, 6, 4, spacing=0.5)


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
def sample_dataset(sample_grid):
    return make_synthetic_fielddataset(sample_grid, ("B1", "B2", "B3", "rho_c"))


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

    def test_has_field(self, sample_dataset):
        assert sample_dataset.has_field("B1")
        assert sample_dataset.has_field("rho_c")
        assert sample_dataset.has_field("Bx")
        assert sample_dataset.has_field("By")
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
            {"B1": np.array([1.0, 2.0])}, grid, Normalization.identity()
        )
        with pytest.raises(KeyError, match="Did you mean") as exc_info:
            ds["Bx1"]
        msg = str(exc_info.value)
        assert "Bx" in msg or "B1" in msg

    def test_no_suggestion_for_unrelated_key(self, sample_dataset):
        with pytest.raises(KeyError) as exc_info:
            sample_dataset["completely_wrong_name"]
        assert "Did you mean" not in str(exc_info.value)


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
        species = (SpeciesInfo(name="e", charge=-1.0, mass=1.0),)
        ds = FieldDataset.from_arrays(
            sample_fields,
            sample_grid,
            Normalization.identity(),
            species=species,
            physics=PhysicsParams(extra={"eta": 0.01}),
            metadata={"run": "test"},
        )
        sliced = ds.isel(z=0)
        assert sliced.species == species
        assert sliced.physics.extra["eta"] == 0.01
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
    @pytest.mark.parametrize(
        ("geometry", "checks"),
        [
            (CARTESIAN, {"Bx": "B1", "By": "B2", "Ez": "E3"}),
            (SPHERICAL, {"Br": "B1", "Btheta": "B2", "Bphi": "B3"}),
            (CYLINDRICAL, {"Br": "B1", "Bphi": "B2", "Bz": "B3"}),
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
            {"B1": np.ones((8, 6, 4))}, sample_grid, Normalization.identity()
        )
        assert not ds.has_field("n_e")

    def test_four_velocity_aliases(self, sample_grid):
        """ux/uy/uz resolve to u1/u2/u3."""
        fields = {
            "u1": np.ones((8, 6, 4)),
            "u2": np.full((8, 6, 4), 2.0),
            "u3": np.full((8, 6, 4), 3.0),
        }
        ds = FieldDataset.from_arrays(fields, sample_grid, Normalization.identity())
        assert ds.has_field("ux")
        assert ds.has_field("uy")
        assert ds.has_field("uz")
        assert_allclose(ds["ux"], fields["u1"])
        assert_allclose(ds["uz"], fields["u3"])


class TestSimulationReader:
    def test_protocol_satisfied(self):
        class MyReader:
            def read_timestep(self, path: Path, step: int) -> FieldDataset: ...
            def available_timesteps(self, path: Path) -> list[int]:
                return []

        assert isinstance(MyReader(), SimulationReader)

    def test_supports_selective_read(self):
        from collections.abc import Iterable

        from pypic.readers.base import supports_selective_read

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
        from pypic.readers.base import SimulationConfig
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
        from pypic.readers.base import SimulationConfig
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
            sample_dataset["B1"] ** 2
            + sample_dataset["B2"] ** 2
            + sample_dataset["B3"] ** 2
        )
        assert_allclose(ds["|B|"], expected)

    def test_multiple_fields(self, sample_dataset):
        ds = sample_dataset.with_derived("|B|", "e_B")
        assert ds.has_field("|B|")
        assert ds.has_field("e_B")

    def test_idempotent_for_existing_fields(self, sample_dataset):
        """Fields already in the dataset are not recomputed."""
        ds = sample_dataset.with_derived("B1")
        assert ds.has_field("B1")
        assert_allclose(ds["B1"], sample_dataset["B1"])

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
            {"B1": np.ones(shape), "B2": np.zeros(shape), "B3": np.zeros(shape)},
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
                "E1": np.ones(shape),
                "E2": np.zeros(shape),
                "E3": np.zeros(shape),
                "B1": np.zeros(shape),
                "B2": np.zeros(shape),
                "B3": np.ones(shape),
            },
            GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0)),
            Normalization.identity(),
        )
        ds = ds.with_derived("S1")
        # Poynting flux siblings S2, S3 should also be stored
        assert ds.has_field("S1")
        assert ds.has_field("S2")
        assert ds.has_field("S3")


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
