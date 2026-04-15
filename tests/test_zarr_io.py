"""Tests for pypic.io Zarr export/import round-trips."""

from __future__ import annotations

import math

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from pypic.coordinates.geometry import SPHERICAL  # noqa: E402
from pypic.coordinates.transforms import FrameTransform  # noqa: E402
from pypic.dataset import FieldDataset  # noqa: E402
from pypic.grid import GridInfo  # noqa: E402
from pypic.io import from_zarr, to_zarr, to_zarr_timeseries  # noqa: E402
from pypic.io._serialize import (  # noqa: E402
    dict_to_grid,
    dict_to_normalization,
    dict_to_physics,
    dict_to_transforms,
    grid_to_dict,
    list_to_species,
    normalization_to_dict,
    physics_to_dict,
    species_to_list,
    transforms_to_dict,
)
from pypic.units import Normalization, PhysicsParams, SpeciesInfo  # noqa: E402
from tests._helpers import (  # noqa: E402
    ELECTRONS,
    IONS,
    make_test_dataset,
    make_uniform_grid,
)


class TestSerializationHelpers:
    """Unit tests for _serialize round-trip fidelity."""

    def test_grid_round_trip_basic(self):
        grid = make_uniform_grid(4, 3, 2, spacing=0.5, origin=-1.0)
        d = grid_to_dict(grid)
        rebuilt = dict_to_grid(d)
        assert rebuilt.dimensions == grid.dimensions
        assert rebuilt.spacing == grid.spacing
        assert rebuilt.origin == grid.origin
        assert rebuilt.geometry.type == grid.geometry.type
        assert rebuilt.dt is None
        assert rebuilt.boundary is None
        assert rebuilt.surviving_axes is None

    def test_grid_round_trip_with_optional_fields(self):
        grid = GridInfo(
            dimensions=(4, 2),
            spacing=(1.0, 1.0),
            origin=(0.0, 0.0),
            dt=0.01,
            boundary=("periodic", "open"),
            surviving_axes=(0, 2),
        )
        d = grid_to_dict(grid)
        rebuilt = dict_to_grid(d)
        assert rebuilt.dt == 0.01
        assert rebuilt.boundary == ("periodic", "open")
        assert rebuilt.surviving_axes == (0, 2)

    def test_normalization_round_trip(self):
        norm = Normalization.pic_electron(1e18)
        d = normalization_to_dict(norm)
        rebuilt = dict_to_normalization(d)
        assert rebuilt.length_ref == norm.length_ref
        assert rebuilt.density_ref == norm.density_ref
        assert rebuilt.mass_ref == norm.mass_ref

    def test_normalization_identity_round_trip(self):
        norm = Normalization.identity()
        d = normalization_to_dict(norm)
        rebuilt = dict_to_normalization(d)
        assert rebuilt.is_identity

    def test_species_round_trip(self):
        species = (ELECTRONS, IONS)
        lst = species_to_list(species)
        rebuilt = list_to_species(lst)
        assert len(rebuilt) == 2
        assert rebuilt[0].name == "electrons"
        assert rebuilt[0].charge == -1.0
        assert rebuilt[0].mass == ELECTRONS.mass
        assert rebuilt[1].name == "ions"

    def test_species_empty_list(self):
        rebuilt = list_to_species([])
        assert rebuilt == ()

    def test_species_with_optional_fields(self):
        sp = SpeciesInfo(
            name="protons",
            charge=1.0,
            mass=1.0,
            temperature=0.1,
            thermal_velocity=(0.01, 0.02, 0.03),
            drift_velocity=(0.5, 0.0, 0.0),
            density=2.0,
            particles_per_cell=(5, 5, 1),
        )
        lst = species_to_list((sp,))
        rebuilt = list_to_species(lst)
        assert rebuilt[0].temperature == 0.1
        assert rebuilt[0].thermal_velocity == (0.01, 0.02, 0.03)
        assert rebuilt[0].drift_velocity == (0.5, 0.0, 0.0)
        assert rebuilt[0].particles_per_cell == (5, 5, 1)

    def test_physics_round_trip(self):
        physics = PhysicsParams(
            gamma=1.4,
            c=1.0,
            relativistic=True,
            extra={"theta": 0.5},
        )
        d = physics_to_dict(physics)
        rebuilt = dict_to_physics(d)
        assert rebuilt.gamma == 1.4
        assert rebuilt.c == 1.0
        assert rebuilt.relativistic is True
        assert rebuilt.extra["theta"] == 0.5

    def test_physics_inf_c_round_trip(self):
        physics = PhysicsParams(c=math.inf)
        d = physics_to_dict(physics)
        assert d["c"] == "inf"
        rebuilt = dict_to_physics(d)
        assert math.isinf(rebuilt.c)

    def test_transforms_round_trip(self):
        rot = (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        t = FrameTransform(
            source_frame="simulation",
            target_frame="GSM",
            origin=(52.0, 26.0, 64.0),
            rotation=rot,
            scale=0.25,
            target_axis_names=("X_GSM", "Y_GSM", "Z_GSM"),
        )
        d = transforms_to_dict({"GSM": t})
        rebuilt = dict_to_transforms(d)
        assert "GSM" in rebuilt
        rt = rebuilt["GSM"]
        assert rt.source_frame == "simulation"
        assert rt.target_frame == "GSM"
        assert rt.origin == (52.0, 26.0, 64.0)
        assert rt.rotation == rot
        assert rt.scale == 0.25
        assert rt.target_axis_names == ("X_GSM", "Y_GSM", "Z_GSM")

    def test_transforms_empty(self):
        d = transforms_to_dict({})
        rebuilt = dict_to_transforms(d)
        assert rebuilt == {}

    def test_metadata_numpy_scalars_round_trip(self, tmp_path):
        # Reviewer regression: h5py returns attrs as numpy scalars
        # (np.float32, np.int64) or arrays, and xarray's Zarr attr
        # validator rejects those with
        # "Invalid attribute in Dataset.attrs".  The serializer must
        # recursively coerce to JSON-native types.
        grid = make_uniform_grid(4, 3, 2)
        metadata = {
            "time": np.float32(1.25),
            "step": np.int64(42),
            "flag": np.bool_(True),
            "seq": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner": np.int32(7)},
        }
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata=metadata,
        )
        store = tmp_path / "meta.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.metadata["time"] == 1.25
        assert isinstance(loaded.metadata["time"], float)
        assert loaded.metadata["step"] == 42
        assert isinstance(loaded.metadata["step"], int)
        assert loaded.metadata["flag"] is True
        assert loaded.metadata["seq"] == [1.0, 2.0, 3.0]
        assert loaded.metadata["nested"] == {"inner": 7}

    def test_metadata_stagger_info_round_trip(self, tmp_path):
        # Reviewer regression: every reader (openggcm, batsrus, the
        # config loader) populates metadata["stagger"] as a typed
        # StaggerInfo dataclass.  Without an explicit (de)serializer
        # the value would either crash json.dumps or come back as a
        # plain dict — the FieldDataset round-trip-fidelity contract
        # requires the original type.
        from pypic.containers import StaggerInfo

        grid = make_uniform_grid(4, 3, 2)
        stagger = StaggerInfo(
            convention="staggered",
            field_locations={"B": "face", "E": "edge"},
            interpolation_order=1,
            notes="Yee mesh",
        )
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"stagger": stagger, "run": "demo"},
        )
        store = tmp_path / "stagger.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        loaded_stagger = loaded.metadata["stagger"]
        assert isinstance(loaded_stagger, StaggerInfo)
        assert loaded_stagger.convention == "staggered"
        assert dict(loaded_stagger.field_locations) == {"B": "face", "E": "edge"}
        assert loaded_stagger.interpolation_order == 1
        assert loaded_stagger.notes == "Yee mesh"
        assert loaded.metadata["run"] == "demo"


class TestToZarrFromZarr:
    """Round-trip tests for to_zarr / from_zarr."""

    def test_round_trip_basic(self, tmp_path):
        fds = make_test_dataset(
            {"B1": np.ones((4, 3, 2)), "B2": np.zeros((4, 3, 2))},
        )
        store = tmp_path / "test.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)

        assert sorted(loaded.field_names()) == ["B1", "B2"]
        np.testing.assert_allclose(loaded["B1"], fds["B1"])
        np.testing.assert_allclose(loaded["B2"], fds["B2"])
        assert loaded.grid.dimensions == fds.grid.dimensions
        assert loaded.grid.spacing == fds.grid.spacing
        assert loaded.grid.origin == fds.grid.origin
        assert loaded.normalization.is_identity

    def test_round_trip_full_metadata(self, tmp_path):
        rot = (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        transform = FrameTransform(
            "simulation",
            "GSM",
            origin=(10.0, 5.0, 3.0),
            rotation=rot,
            scale=0.5,
        )
        physics = PhysicsParams(
            gamma=1.4,
            c=1.0,
            relativistic=True,
            extra={"theta": 0.5},
        )
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 2.0, 3.0),
            dt=0.01,
            boundary=("periodic", "open", "periodic"),
        )
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2)), "rho_m": np.full((4, 3, 2), 2.0)},
            grid,
            Normalization.pic_electron(1e18),
            species=[ELECTRONS, IONS],
            physics=physics,
            metadata={"run_name": "test_run"},
            frame="GSM",
            transforms={"GSM": transform},
        )
        store = tmp_path / "full.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)

        assert loaded.grid.dt == 0.01
        assert loaded.grid.boundary == ("periodic", "open", "periodic")
        assert len(loaded.species) == 2
        assert loaded.species[0].name == "electrons"
        assert loaded.species[1].name == "ions"
        assert loaded.physics.gamma == 1.4
        assert loaded.physics.relativistic is True
        assert loaded.physics.extra["theta"] == 0.5
        assert loaded.frame == "GSM"
        assert "GSM" in loaded.transforms
        np.testing.assert_allclose(loaded["B1"], 1.0)
        np.testing.assert_allclose(loaded["rho_m"], 2.0)

    def test_dtype_float32_downcast(self, tmp_path):
        fds = make_test_dataset(
            {"B1": np.ones((4, 3, 2), dtype=np.float64)},
        )
        store = tmp_path / "f32.zarr"
        to_zarr(fds, store, dtype="float32")
        loaded = from_zarr(store)
        assert loaded["B1"].dtype == np.float32
        np.testing.assert_allclose(loaded["B1"], 1.0, rtol=1e-6)

    def test_round_trip_2d_surviving_axes(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 2),
            spacing=(1.0, 1.0),
            origin=(0.0, 0.0),
            surviving_axes=(0, 2),
        )
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "2d.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.grid.dimensions == (4, 2)
        assert loaded.grid.surviving_axes == (0, 2)
        assert loaded.grid.surviving_axis_names == ("x", "z")

    def test_per_field_metadata_preserved(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "meta.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        b1_attrs = loaded.xr["B1"].attrs
        assert b1_attrs["quantity_type"] == "b_field"
        assert "si_unit" in b1_attrs

    def test_spherical_geometry_round_trip(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(1.0, 0.5, 1.0),
            origin=(1.0, 0.0, 0.0),
            geometry=SPHERICAL,
        )
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "sph.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.grid.geometry.type.value == "spherical"
        assert loaded.grid.geometry.axis_names == ("r", "\u03b8", "\u03c6")

    def test_no_pypic_attrs_raises(self, tmp_path):
        import xarray as xr

        ds = xr.Dataset({"B1": xr.DataArray(np.ones(4))})
        store = tmp_path / "no_meta.zarr"
        ds.to_zarr(str(store), zarr_format=3, consolidated=False)
        with pytest.raises(ValueError, match="No 'pypic' metadata"):
            from_zarr(store)


class TestToZarrTimeseries:
    """Tests for to_zarr_timeseries with iterable source."""

    def test_timeseries_basic(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        pairs = [
            (
                0.0,
                FieldDataset.from_arrays(
                    {"B1": np.full((4, 3, 2), 1.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
            (
                1.0,
                FieldDataset.from_arrays(
                    {"B1": np.full((4, 3, 2), 2.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
            (
                2.0,
                FieldDataset.from_arrays(
                    {"B1": np.full((4, 3, 2), 3.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
        ]
        store = tmp_path / "ts.zarr"
        to_zarr_timeseries(pairs, store)

        import xarray as xr

        ds = xr.open_zarr(str(store), consolidated=False)
        assert "time" in ds.dims
        assert ds.sizes["time"] == 3
        np.testing.assert_allclose(ds["B1"].sel(time=0.0).values, 1.0)
        np.testing.assert_allclose(ds["B1"].sel(time=2.0).values, 3.0)

    def test_timeseries_rejects_field_drift(self, tmp_path):
        # xarray's to_zarr(mode="a", append_dim=...) doesn't enforce
        # a consistent variable set across appends; a growing field
        # set would silently write a store unreadable by from_zarr
        # (conflicting sizes on `time`).  Fail loud at write time.
        grid = make_uniform_grid(4, 3, 2)
        step0 = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2)), "B2": np.zeros((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "drift.zarr"
        with pytest.raises(ValueError, match=r"field set.*differs.*B2"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        # And the partial store from step 0 must not survive — otherwise
        # it looks like a valid single-step export on retry.
        assert not store.exists()

    def test_timeseries_cleans_partial_store_on_append_failure(self, tmp_path):
        # Reviewer regression: shape drift between steps raises only
        # after step 0 has written to disk.  The partial store must
        # be removed so the filesystem state matches the error state.
        grid_small = make_uniform_grid(2, 2, 2)
        grid_big = make_uniform_grid(3, 2, 2)
        step0 = FieldDataset.from_arrays(
            {"B1": np.ones((2, 2, 2))},
            grid_small,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B1": np.ones((3, 2, 2))},
            grid_big,
            Normalization.identity(),
        )
        store = tmp_path / "partial.zarr"
        with pytest.raises(ValueError, match=r"different dimension sizes"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        assert not store.exists()

    def test_timeseries_preserves_metadata(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            species=[ELECTRONS],
            frame="sim",
        )
        store = tmp_path / "ts_meta.zarr"
        to_zarr_timeseries([(0.0, fds), (1.0, fds)], store)

        loaded_ds = from_zarr(store)
        assert loaded_ds.frame == "sim"
        assert len(loaded_ds.species) == 1
        assert loaded_ds.species[0].name == "electrons"
