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
from pypic.io.metadata import (  # noqa: E402
    decode_pypic_attrs,
    dict_to_normalization,
    dict_to_transforms,
    encode_pypic_attrs,
    grid_to_dict,
    list_to_species,
    normalization_to_dict,
    physics_to_dict,
    species_to_list,
    to_json_native,
    transforms_to_dict,
)
from pypic.units import (  # noqa: E402
    Normalization,
    PhysicsParams,
    SpeciesInfo,
    UnitSystem,
)
from tests._helpers import (  # noqa: E402
    ELECTRONS,
    IONS,
    make_test_dataset,
    make_uniform_grid,
)


class TestSerializationHelpers:
    """Unit tests for ``pypic.io.metadata`` round-trip fidelity."""

    def test_grid_encodes_every_axis_tuple_as_a_json_list(self):
        grid = make_uniform_grid(4, 3, 2, spacing=0.5, origin=-1.0)
        assert grid_to_dict(grid) == {
            "dimensions": [4, 3, 2],
            "spacing": [0.5, 0.5, 0.5],
            "origin": [-1.0, -1.0, -1.0],
            "geometry": {
                "type": "cartesian",
                "axis_names": ["x", "y", "z"],
                "axis_units": list(grid.geometry.axis_units),
            },
            "dt": None,
            "boundary": None,
            "surviving_axes": None,
        }

    def test_grid_encodes_the_optional_fields_when_set(self):
        grid = GridInfo(
            dimensions=(4, 2),
            spacing=(1.0, 1.0),
            origin=(0.0, 0.0),
            dt=0.01,
            boundary=("periodic", "open"),
            surviving_axes=(0, 2),
        )
        d = grid_to_dict(grid)
        assert (d["dt"], d["boundary"], d["surviving_axes"]) == (
            0.01,
            ["periodic", "open"],
            [0, 2],
        )

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

    def test_undeclared_survives_the_round_trip(self):
        """Otherwise a store hop launders undeclared into declared SI."""
        rebuilt = dict_to_normalization(
            normalization_to_dict(Normalization.undeclared())
        )
        assert rebuilt.system is None

    def test_declared_system_survives_the_round_trip(self):
        rebuilt = dict_to_normalization(
            normalization_to_dict(Normalization.pic_electron(1e18))
        )
        assert rebuilt.system is UnitSystem.PIC

    def test_a_dict_without_system_decodes_as_declared(self):
        """Stores written before the key existed keep their meaning."""
        legacy = {
            k: v
            for k, v in normalization_to_dict(Normalization.identity()).items()
            if k != "system"
        }
        assert dict_to_normalization(legacy).system is UnitSystem.CUSTOM

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

    def test_physics_encodes_the_extra_table_as_a_plain_dict(self):
        physics = PhysicsParams(
            gamma=1.4,
            c=1.0,
            relativistic=True,
            extra={"theta": 0.5},
        )
        assert physics_to_dict(physics) == {
            "gamma": 1.4,
            "c": 1.0,
            "relativistic": True,
            "extra": {"theta": 0.5},
        }

    def test_physics_encodes_infinite_c_as_a_string(self):
        # JSON has no infinity literal, and c = inf is the
        # non-relativistic MHD default.
        assert physics_to_dict(PhysicsParams(c=math.inf))["c"] == "inf"

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
            {"B_1": np.ones((4, 3, 2))},
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
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"stagger": stagger, "label": "demo"},
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
        assert loaded.metadata["label"] == "demo"


class TestToJsonNativeStrict:
    """``to_json_native`` fails loud on types it cannot coerce.

    Silent pass-through would defer the failure to ``json.dumps`` deep
    inside the Zarr writer, producing a cryptic ``Object of type X is
    not JSON serializable`` error that hides which metadata stamp
    introduced the bad value.  Raising at the coercion boundary names
    the offending type and the responsible caller.
    """

    def test_set_raises(self):
        with pytest.raises(TypeError, match="'set'"):
            to_json_native({1, 2, 3})

    def test_bytes_raises(self):
        with pytest.raises(TypeError, match="'bytes'"):
            to_json_native(b"abc")

    def test_datetime_date_raises(self):
        import datetime

        with pytest.raises(TypeError, match="'date'"):
            to_json_native(datetime.date(2026, 5, 19))

    def test_arbitrary_object_raises(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="'Custom'"):
            to_json_native(Custom())

    def test_nested_unsupported_raises(self):
        # The strict guard fires regardless of nesting depth — the
        # recursive walk surfaces the bad leaf.
        with pytest.raises(TypeError, match="'bytes'"):
            to_json_native({"outer": {"inner": [b"abc"]}})


class TestDecodeTransformsNullEdge:
    """``decode_pypic_attrs`` tolerates an explicit ``null`` transforms key.

    Pre-fix, ``coords_attrs.get("transforms") or d.get("transforms", {})``
    fell through to ``None`` when both keys were present-but-null,
    crashing ``dict_to_transforms(None)`` with ``AttributeError``.  The
    trailing ``or {}`` collapses the null branch to an empty mapping.
    """

    def test_coordinates_transforms_null_decodes_to_empty(self):
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        attrs = encode_pypic_attrs(fds)
        attrs["coordinates"]["transforms"] = None
        attrs["transforms"] = None
        _, _, _, _, _, _, transforms = decode_pypic_attrs(attrs)
        assert transforms == {}


class TestToZarrFromZarr:
    """Round-trip tests for to_zarr / from_zarr."""

    def test_undeclared_normalization_survives_a_store_hop(self, tmp_path):
        """A store round-trip must not turn "unknown" into "declared SI"."""
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))}, make_uniform_grid(4, 3, 2)
        )
        store = tmp_path / "undeclared.zarr"
        to_zarr(fds, store)
        assert from_zarr(store).normalization.system is None

    def test_declared_system_survives_a_store_hop(self, tmp_path):
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            make_uniform_grid(4, 3, 2),
            Normalization.pic_electron(1e18),
        )
        store = tmp_path / "declared.zarr"
        to_zarr(fds, store)
        assert from_zarr(store).normalization.system is UnitSystem.PIC

    def test_round_trip_basic(self, tmp_path):
        fds = make_test_dataset(
            {"B_1": np.ones((4, 3, 2)), "B_2": np.zeros((4, 3, 2))},
        )
        store = tmp_path / "test.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)

        assert sorted(loaded.field_names()) == ["B_1", "B_2"]
        np.testing.assert_array_equal(loaded["B_1"], fds["B_1"])
        np.testing.assert_array_equal(loaded["B_2"], fds["B_2"])
        assert loaded.grid.dimensions == fds.grid.dimensions
        assert loaded.grid.spacing == fds.grid.spacing
        assert loaded.grid.origin == fds.grid.origin
        assert loaded.normalization.is_identity

    def test_section_attrs_do_not_leak_onto_the_loaded_dataset(self, tmp_path):
        """Root-group sections stay typed fields, never user-visible attrs.

        The sections are written on the root group, while ``from_zarr``
        returns the ``/fields`` group, so nothing has to strip them —
        this pins that the two stay separate instead of trusting a
        hand-kept mirror of the encoder's key list.
        """
        fds = make_test_dataset({"B_1": np.ones((4, 3, 2))})
        store = tmp_path / "no_leak.zarr"
        to_zarr(fds, store)
        assert from_zarr(store).xr.attrs == {}

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
            {"B_1": np.ones((4, 3, 2)), "rho_m": np.full((4, 3, 2), 2.0)},
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
        np.testing.assert_array_equal(loaded["B_1"], 1.0)
        np.testing.assert_array_equal(loaded["rho_m"], 2.0)

    def test_dtype_float32_downcast(self, tmp_path):
        fds = make_test_dataset(
            {"B_1": np.ones((4, 3, 2), dtype=np.float64)},
        )
        store = tmp_path / "f32.zarr"
        to_zarr(fds, store, dtype="float32")
        loaded = from_zarr(store)
        assert loaded["B_1"].dtype == np.float32
        np.testing.assert_allclose(loaded["B_1"], 1.0, rtol=1e-6)

    def test_round_trip_2d_surviving_axes(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 2),
            spacing=(1.0, 1.0),
            origin=(0.0, 0.0),
            surviving_axes=(0, 2),
        )
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 2))},
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
        fds = make_test_dataset({"B_1": np.ones((4, 3, 2))})
        store = tmp_path / "meta.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        b1_attrs = loaded.xr["B_1"].attrs
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
            {"B_1": np.ones((4, 3, 2))},
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

        ds = xr.Dataset({"B_1": xr.DataArray(np.ones(4))})
        store = tmp_path / "no_meta.zarr"
        ds.to_zarr(str(store), zarr_format=3, consolidated=False)
        with pytest.raises(ValueError, match="no pypic metadata found"):
            from_zarr(store)

    def test_failed_write_cleans_up_fresh_store(self, tmp_path):
        # Non-JSON-native values (here, a ``set``) are now rejected by
        # ``to_json_native`` inside ``encode_pypic_attrs`` *before* any
        # zarr disk operation runs, so no half-written store can survive
        # — strictly stronger than the prior "cleanup after partial
        # write" contract.  Kept as a regression: if a future value
        # slipped past ``to_json_native`` and failed inside the zarr
        # attr encoder, the writer's cleanup logic would still need to
        # delete the half-store, and ``assert not store.exists()``
        # would re-catch it.
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"bad": {1, 2, 3}},
        )
        store = tmp_path / "broken.zarr"
        with pytest.raises(TypeError, match=r"does not coerce 'set'"):
            to_zarr(fds, store)
        assert not store.exists()

    def test_unit_dimension_round_trip(self, tmp_path):
        # openPMD-style 7-tuple survives Zarr write/read on canonical fields.
        fds = make_test_dataset(
            {"B_1": np.ones((4, 3, 2)), "rho_m": np.ones((4, 3, 2))},
        )
        store = tmp_path / "ud.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.xr["B_1"].attrs["unit_dimension"] == [0, 1, -2, -1, 0, 0, 0]
        assert loaded.xr["rho_m"].attrs["unit_dimension"] == [-3, 1, 0, 0, 0, 0, 0]

    def test_layout_v1_structure(self, tmp_path):
        # Validate the on-disk shape — the contract a non-pypic Zarr
        # consumer (a JS WebGPU viewer, a Rust zarrs pipeline) reads
        # against — not just that ``from_zarr`` round-trips. Each
        # top-level attrs key mirrors a §2 simulation.toml section
        # of the same name (schema.md §4.2).
        fds = make_test_dataset({"B_1": np.ones((4, 3, 2))})
        store = tmp_path / "v1.zarr"
        to_zarr(fds, store)
        root = zarr.open_group(str(store), mode="r")
        # ``schema.version`` mirrors ``simulation.toml`` ``[schema].version``
        # and is the single discriminator for both vocabulary and
        # storage shape.  The on-disk path nests under ``schema`` to
        # mirror the TOML form rather than a flat ``schema_version``.
        assert root.attrs["schema"] == {"version": "1.0"}
        assert "schema_version" not in root.attrs
        # Field arrays under /fields, not at the root.
        assert list(root.array_keys()) == []
        assert "fields" in list(root.group_keys())
        # Required metadata sections flat at root, mirroring §2.
        for key in ("grid", "normalization", "physics", "coordinates"):
            assert key in root.attrs, f"missing root attr {key!r}"
        # Coordinates owns frame and (when present) transforms — they
        # are no longer top-level keys under the §4.2 reshape.
        assert "frame" not in root.attrs
        assert "transforms" not in root.attrs
        assert root.attrs["coordinates"]["frame"] == fds.frame
        # Schema fields the encoder drops on emit (the reshape moved
        # them into typed sub-keys; see schema.md §4.2 mapping table).
        assert "dt" not in root.attrs["grid"]
        assert "boundary" not in root.attrs["grid"]
        assert "geometry" not in root.attrs["grid"]
        assert "gamma" not in root.attrs["physics"]
        assert "c" not in root.attrs["physics"]

    def test_from_zarr_rejects_unknown_schema_version(self, tmp_path):
        # A v2.0 store must not silently decode through the v1.0 path —
        # see schema.md §1 *Versioning* (single-discriminator promise).
        fds = make_test_dataset({"B_1": np.ones((4, 3, 2))})
        store = tmp_path / "future.zarr"
        to_zarr(fds, store)
        root = zarr.open_group(str(store), mode="a")
        root.attrs["schema"] = {"version": "2.0"}
        if hasattr(zarr, "consolidate_metadata"):
            zarr.consolidate_metadata(str(store))
        with pytest.raises(ValueError, match=r"schema\.version"):
            from_zarr(store)

    def test_from_zarr_reads_pre_reshape_attrs_shape(self, tmp_path):
        # Stores written before the §4.2 reshape carried ``dt`` and
        # ``boundary`` under ``grid``, ``geometry`` as a nested struct
        # under ``grid``, ``c`` and ``gamma`` under ``physics``, and
        # ``frame`` / ``transforms`` at the top level. The decoder
        # accepts that shape for one minor version so existing stores
        # don't need migration.
        physics = PhysicsParams(gamma=1.4, c=1.0, relativistic=True)
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 2.0, 3.0),
            dt=0.01,
            boundary=("periodic", "open", "periodic"),
        )
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.pic_electron(1e18),
            physics=physics,
            frame="GSM",
        )
        store = tmp_path / "old_layout.zarr"
        to_zarr(fds, store)
        root = zarr.open_group(str(store), mode="a")
        # Repaint root attrs into the pre-reshape shape so the decoder
        # exercises every fall-back path.
        old_grid = dict(root.attrs["grid"])
        old_grid["origin"] = old_grid.pop("lower")
        old_grid.pop("upper", None)
        old_grid["dt"] = root.attrs["time"]["dt"]
        old_grid["boundary"] = list(root.attrs["boundary_conditions"]["lower"])
        old_grid["geometry"] = {
            "type": root.attrs["coordinates"]["geometry"],
            "axis_names": list(root.attrs["coordinates"].get("axis_labels", [])),
            "axis_units": [],
        }
        root.attrs["grid"] = old_grid
        root.attrs["frame"] = root.attrs["coordinates"]["frame"]
        root.attrs["transforms"] = root.attrs["coordinates"].get("transforms", {})
        old_physics = dict(root.attrs["physics"])
        old_physics["gamma"] = old_physics.pop("gamma_eos")
        old_physics["c"] = root.attrs["normalization"]["speed_of_light"]
        root.attrs["physics"] = old_physics
        old_norm = dict(root.attrs["normalization"])
        old_norm.pop("speed_of_light", None)
        root.attrs["normalization"] = old_norm
        del root.attrs["time"]
        del root.attrs["boundary_conditions"]
        del root.attrs["coordinates"]
        if hasattr(zarr, "consolidate_metadata"):
            zarr.consolidate_metadata(str(store))
        loaded = from_zarr(store)
        assert loaded.frame == "GSM"
        assert loaded.grid.dt == 0.01
        assert loaded.grid.boundary == ("periodic", "open", "periodic")
        assert loaded.grid.geometry.type == fds.grid.geometry.type
        assert loaded.physics.gamma == 1.4
        assert loaded.physics.c == 1.0
        assert loaded.physics.relativistic is True

    def test_from_zarr_rejects_missing_schema_attr(self, tmp_path):
        # Missing ``schema`` root attr is rejected with the discriminator
        # error rather than a downstream KeyError on ``grid`` / etc.
        fds = make_test_dataset({"B_1": np.ones((4, 3, 2))})
        store = tmp_path / "no_schema.zarr"
        to_zarr(fds, store)
        root = zarr.open_group(str(store), mode="a")
        del root.attrs["schema"]
        if hasattr(zarr, "consolidate_metadata"):
            zarr.consolidate_metadata(str(store))
        with pytest.raises(ValueError, match=r"schema\.version"):
            from_zarr(store)


class TestRunAndSimulationTomlRoundTrip:
    """Verbatim ``simulation.toml`` and typed ``[run]`` round-trip via root attrs.

    Schema.md §4.2 promotes both to top-level root attrs so non-pypic
    consumers (webpic, Rust pipelines) see them at ``attrs.run`` /
    ``attrs.simulation_toml`` without going through the loose
    ``metadata`` bag.  On the Python side, both re-land in
    ``fds.metadata`` for caller ergonomics — these tests pin both
    contracts.
    """

    _SAMPLE_TOML = """\
[schema]
version = "1.0"

[model]
name = "TestCode"
type = "PIC"

[run]
name = "round-trip-test"
description = "fixture"
doi = "10.5281/zenodo.12345678"
license = "CC-BY-4.0"

[time]
dt = 0.05
t_start = 0.0
t_end = 1.0
n_steps = 20

[grid]
dimensions = [4, 3, 2]
spacing = [1.0, 1.0, 1.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 3.0, 2.0]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"

[[species]]
name = "electrons"
charge = -1.0
mass = 1.0
"""

    def _make_run(self):
        from pypic.schema import Run

        return Run.model_validate(
            {
                "name": "round-trip-test",
                "description": "fixture",
                "doi": "10.5281/zenodo.12345678",
                "license": "CC-BY-4.0",
            }
        )

    def _fds(self, metadata=None):
        return FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            make_uniform_grid(4, 3, 2),
            Normalization.identity(),
            metadata=metadata,
        )

    def test_run_round_trips_typed(self, tmp_path):
        from pypic.schema import Run

        run = self._make_run()
        fds = self._fds(metadata={"run": run})
        store = tmp_path / "run.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        # decode_pypic_attrs rebuilds Run via model_validate — same
        # logical content, fresh instance.
        loaded_run = loaded.metadata["run"]
        assert isinstance(loaded_run, Run)
        assert loaded_run.name == "round-trip-test"
        assert loaded_run.doi == "10.5281/zenodo.12345678"
        assert loaded_run.license == "CC-BY-4.0"
        assert loaded_run == run

    def test_simulation_toml_round_trips_text(self, tmp_path):
        fds = self._fds(metadata={"simulation_toml": self._SAMPLE_TOML})
        store = tmp_path / "toml.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.metadata["simulation_toml"] == self._SAMPLE_TOML

    def test_attrs_are_top_level_not_in_metadata_bag(self, tmp_path):
        # The whole point of the lift: cross-tool consumers read
        # ``attrs.run`` / ``attrs.simulation_toml`` at the root group,
        # not buried under ``attrs.metadata``.
        run = self._make_run()
        fds = self._fds(metadata={"run": run, "simulation_toml": self._SAMPLE_TOML})
        store = tmp_path / "toplevel.zarr"
        to_zarr(fds, store)
        root = zarr.open_group(str(store), mode="r")
        assert "run" in root.attrs
        assert "simulation_toml" in root.attrs
        assert root.attrs["simulation_toml"] == self._SAMPLE_TOML
        # And NOT nested under the open metadata bag.
        metadata_bag = root.attrs.get("metadata", {})
        assert "run" not in metadata_bag
        assert "simulation_toml" not in metadata_bag

    def test_simulation_toml_writer_kwarg_overrides_metadata(self, tmp_path):
        # FieldDataset has no simulation_toml in metadata; the writer
        # kwarg attaches it from a file path at write time only.
        toml_path = tmp_path / "src.toml"
        toml_path.write_text(self._SAMPLE_TOML, encoding="utf-8")
        fds = self._fds()
        assert "simulation_toml" not in fds.metadata
        store = tmp_path / "kwarg.zarr"
        to_zarr(fds, store, simulation_toml=toml_path)
        loaded = from_zarr(store)
        assert loaded.metadata["simulation_toml"] == self._SAMPLE_TOML
        # The source fds was not mutated.
        assert "simulation_toml" not in fds.metadata

    def test_simulation_toml_writer_kwarg_missing_path_raises(self, tmp_path):
        fds = self._fds()
        with pytest.raises(FileNotFoundError, match=r"simulation_toml"):
            to_zarr(fds, tmp_path / "x.zarr", simulation_toml=tmp_path / "nope.toml")

    def test_run_and_simulation_toml_round_trip_through_timeseries(self, tmp_path):
        from pypic.schema import Run

        run = self._make_run()
        grid = make_uniform_grid(4, 3, 2)
        pairs = [
            (
                float(t),
                FieldDataset.from_arrays(
                    {"B_1": np.full((4, 3, 2), float(t))},
                    grid,
                    Normalization.identity(),
                    metadata={"run": run, "simulation_toml": self._SAMPLE_TOML},
                ),
            )
            for t in (0.0, 1.0, 2.0)
        ]
        store = tmp_path / "ts.zarr"
        to_zarr_timeseries(pairs, store)
        loaded = from_zarr(store)
        assert isinstance(loaded.metadata["run"], Run)
        assert loaded.metadata["run"] == run
        assert loaded.metadata["simulation_toml"] == self._SAMPLE_TOML

    def test_load_config_attaches_raw_toml(self, tmp_path):
        from pypic.readers.config import load_config

        toml_path = tmp_path / "sim.toml"
        toml_path.write_text(self._SAMPLE_TOML, encoding="utf-8")
        cfg = load_config(toml_path)
        assert cfg.metadata["simulation_toml"] == self._SAMPLE_TOML
        # ``[run]`` survives onto the typed field — unchanged behaviour.
        assert cfg.run is not None
        assert cfg.run.name == "round-trip-test"
        assert cfg.run.doi == "10.5281/zenodo.12345678"


class TestToZarrTimeseries:
    """Tests for to_zarr_timeseries with iterable source."""

    def test_timeseries_basic(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        pairs = [
            (
                0.0,
                FieldDataset.from_arrays(
                    {"B_1": np.full((4, 3, 2), 1.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
            (
                1.0,
                FieldDataset.from_arrays(
                    {"B_1": np.full((4, 3, 2), 2.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
            (
                2.0,
                FieldDataset.from_arrays(
                    {"B_1": np.full((4, 3, 2), 3.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
        ]
        store = tmp_path / "ts.zarr"
        to_zarr_timeseries(pairs, store)

        import xarray as xr

        # v1 layout: fields live under /fields, not at the root.
        ds = xr.open_zarr(str(store), group="fields", consolidated="auto")
        assert "time" in ds.dims
        assert ds.sizes["time"] == 3
        np.testing.assert_array_equal(ds["B_1"].sel(time=0.0).values, 1.0)
        np.testing.assert_array_equal(ds["B_1"].sel(time=2.0).values, 3.0)

    def test_timeseries_rejects_field_drift(self, tmp_path):
        # xarray's to_zarr(mode="a", append_dim=...) doesn't enforce
        # a consistent variable set across appends; a growing field
        # set would silently write a store unreadable by from_zarr
        # (conflicting sizes on `time`).  Fail loud at write time.
        grid = make_uniform_grid(4, 3, 2)
        step0 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2)), "B_2": np.zeros((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "drift.zarr"
        with pytest.raises(ValueError, match=r"field set.*differs.*B_2"):
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
            {"B_1": np.ones((2, 2, 2))},
            grid_small,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B_1": np.ones((3, 2, 2))},
            grid_big,
            Normalization.identity(),
        )
        store = tmp_path / "partial.zarr"
        # Different shapes imply different grids — the identity check
        # rejects this with a clearer message before xarray's
        # dimension-size check would have fired.  Cleanup behavior
        # (the actual point of this test) is unchanged.
        with pytest.raises(ValueError, match=r"grid differ from the first step"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        assert not store.exists()

    def test_timeseries_cleans_stub_on_first_write_failure(self, tmp_path):
        # Reviewer regression: the cleanup guard used to key off the
        # ``first`` flag, which stays True when xarray's *first*
        # ``ds.to_zarr(mode='w')`` call fails during materialization.
        # A stub store containing only ``zarr.json`` was left behind
        # and a later ``from_zarr`` surfaced the misleading "No 'pypic'
        # metadata found" error instead of the real write failure.
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "stub.zarr"
        with pytest.raises((TypeError, ValueError)):
            to_zarr_timeseries(
                [(0.0, fds)],
                store,
                encoding={"B_1": {"dtype": "not-a-real-dtype"}},
            )
        assert not store.exists()

    def test_timeseries_preserves_metadata(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
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

    def test_timeseries_rejects_grid_drift(self, tmp_path):
        # Reviewer regression: timeseries writers used to silently
        # flatten every later step's identity (grid, frame, etc.) to
        # step 0 — succeeded but stored a misleading description of
        # the simulation.  The identity check now fails loud.
        step0 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            make_uniform_grid(4, 3, 2, spacing=1.0, origin=0.0),
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            make_uniform_grid(4, 3, 2, spacing=5.0, origin=10.0),
            Normalization.identity(),
        )
        store = tmp_path / "drift_grid.zarr"
        with pytest.raises(ValueError, match=r"grid differ from the first step"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        assert not store.exists()

    def test_timeseries_rejects_frame_drift(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        step0 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            frame="simulation",
        )
        step1 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            frame="gsm",
        )
        store = tmp_path / "drift_frame.zarr"
        with pytest.raises(ValueError, match=r"frame differ from the first step"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        assert not store.exists()

    def test_timeseries_intersects_per_step_metadata(self, tmp_path):
        # Per-step metadata divergence (e.g. iPIC3D's per-file ``time``
        # / ``step`` scalars) is normal — the writer now intersects so
        # the stored attrs reflect what is actually true everywhere
        # rather than silently keeping step 0's snapshot.
        grid = make_uniform_grid(4, 3, 2)
        step0 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"shared": "ok", "step_meta": "a"},
        )
        step1 = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"shared": "ok", "step_meta": "b"},
        )
        store = tmp_path / "ts_meta_intersect.zarr"
        to_zarr_timeseries([(0.0, step0), (1.0, step1)], store)
        loaded = from_zarr(store)
        assert loaded.metadata == {"shared": "ok"}

    def test_metadata_tuple_round_trip(self, tmp_path):
        # Reviewer regression: tuples used to be coerced to lists and
        # come back as lists, losing the type identity.  Tagging
        # preserves the round-trip.
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"tuple_val": (1, 2, 3), "list_val": [4, 5]},
        )
        store = tmp_path / "tup.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.metadata["tuple_val"] == (1, 2, 3)
        assert isinstance(loaded.metadata["tuple_val"], tuple)
        assert loaded.metadata["list_val"] == [4, 5]
        assert isinstance(loaded.metadata["list_val"], list)

    def test_metadata_non_string_keys_round_trip(self, tmp_path):
        # Reviewer regression: stringifying integer keys silently
        # collides ``{1: a, '1': b}`` and loses int-vs-str identity.
        # Tagged ``keyed_dict`` preserves both key type and uniqueness.
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"num_key_map": {1: "one", 2: "two"}},
        )
        store = tmp_path / "intkey.zarr"
        to_zarr(fds, store)
        loaded = from_zarr(store)
        assert loaded.metadata["num_key_map"] == {1: "one", 2: "two"}
        assert all(isinstance(k, int) for k in loaded.metadata["num_key_map"])
