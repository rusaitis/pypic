"""Tests for PlaneSelection and BoxSelection."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pypic.coordinates import CARTESIAN, CoordinateGeometry, GeometryType
from pypic.readers import FieldDataset, GridInfo
from pypic.selections import BoxSelection, PlaneSelection
from pypic.units import Normalization


@pytest.fixture
def cartesian_3d() -> FieldDataset:
    """8x6x4 Cartesian dataset with B1, B2, B3."""
    grid = GridInfo(
        dimensions=(8, 6, 4),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    rng = np.random.default_rng(42)
    fields = {
        "B1": rng.standard_normal((8, 6, 4)),
        "B2": rng.standard_normal((8, 6, 4)),
        "B3": rng.standard_normal((8, 6, 4)),
    }
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


@pytest.fixture
def spherical_3d() -> FieldDataset:
    """4x3x2 spherical dataset with B1."""
    geom = CoordinateGeometry(
        type=GeometryType.SPHERICAL,
        axis_names=("r", "θ", "φ"),
        axis_units=("length", "angle", "angle"),
    )
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(0.5, 0.1, 0.2),
        origin=(1.0, 0.0, 0.0),
        geometry=geom,
    )
    fields = {"B1": np.arange(24, dtype=float).reshape(4, 3, 2)}
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


class TestPlaneSelection:
    @pytest.mark.parametrize(
        ("normal", "expected_shape"),
        [("z", (8, 6)), ("y", (8, 4)), ("x", (6, 4))],
    )
    def test_shape_after_slicing(
        self, cartesian_3d: FieldDataset, normal: str, expected_shape: tuple
    ) -> None:
        idx = 0 if normal != "y" else 2
        result = PlaneSelection(normal=normal, index=idx).apply(cartesian_3d)
        assert result["B1"].shape == expected_shape

    def test_midplane_default(self, cartesian_3d: FieldDataset) -> None:
        result = PlaneSelection(normal="z").apply(cartesian_3d)
        expected = cartesian_3d.isel({"z": 2})  # 4 // 2 = 2
        assert_array_equal(result["B1"], expected["B1"])

    def test_data_matches_direct_isel(self, cartesian_3d: FieldDataset) -> None:
        idx = 3
        result = PlaneSelection(normal="y", index=idx).apply(cartesian_3d)
        expected = cartesian_3d.isel({"y": idx})
        assert_array_equal(result["B2"], expected["B2"])

    def test_grid_updated_after_plane_slice(self, cartesian_3d: FieldDataset) -> None:
        result = PlaneSelection(normal="z", index=1).apply(cartesian_3d)
        assert result.grid.dimensions == (8, 6)
        assert result.grid.spacing == (1.0, 1.0)
        assert len(result.grid.origin) == 2
        assert result.grid.origin == (0.0, 0.0)

    def test_metadata_preserved(self, cartesian_3d: FieldDataset) -> None:
        result = PlaneSelection(normal="z", index=0).apply(cartesian_3d)
        assert result.normalization is cartesian_3d.normalization
        assert result.species == cartesian_3d.species

    def test_aliases_preserved(self, cartesian_3d: FieldDataset) -> None:
        result = PlaneSelection(normal="z", index=0).apply(cartesian_3d)
        assert result.has_field("Bx")
        assert_array_equal(result["Bx"], result["B1"])

    def test_spherical_axis(self, spherical_3d: FieldDataset) -> None:
        result = PlaneSelection(normal="θ", index=1).apply(spherical_3d)
        assert result["B1"].shape == (4, 2)

    def test_invalid_axis_raises(self, cartesian_3d: FieldDataset) -> None:
        with pytest.raises(ValueError, match="not found"):
            PlaneSelection(normal="w", index=0).apply(cartesian_3d)


class TestBoxSelection:
    def test_subbox_shape(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (1, 5), "y": (0, 3)}).apply(cartesian_3d)
        assert result["B1"].shape == (4, 3, 4)

    def test_single_axis_range(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"z": (1, 3)}).apply(cartesian_3d)
        assert result["B1"].shape == (8, 6, 2)

    def test_all_axes_ranged(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (2, 6), "y": (1, 4), "z": (0, 2)}).apply(
            cartesian_3d
        )
        assert result["B1"].shape == (4, 3, 2)

    def test_empty_ranges_noop(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={}).apply(cartesian_3d)
        assert_array_equal(result["B1"], cartesian_3d["B1"])

    def test_data_matches_direct_isel(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (2, 5)}).apply(cartesian_3d)
        expected = cartesian_3d.isel({"x": slice(2, 5)})
        assert_array_equal(result["B3"], expected["B3"])

    def test_grid_dimensions_updated(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (1, 4), "z": (0, 2)}).apply(cartesian_3d)
        assert result.grid.dimensions == (3, 6, 2)

    def test_grid_origin_updated(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (2, 6)}).apply(cartesian_3d)
        np.testing.assert_allclose(result.grid.origin[0], 2.0, atol=1e-12)

    def test_grid_spacing_preserved(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (1, 5)}).apply(cartesian_3d)
        assert result.grid.spacing == (1.0, 1.0, 1.0)

    def test_metadata_preserved(self, cartesian_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"x": (0, 2)}).apply(cartesian_3d)
        assert result.normalization is cartesian_3d.normalization
        assert result.species == cartesian_3d.species

    def test_spherical_geometry(self, spherical_3d: FieldDataset) -> None:
        result = BoxSelection(ranges={"r": (1, 3), "φ": (0, 1)}).apply(spherical_3d)
        assert result["B1"].shape == (2, 3, 1)

    def test_invalid_axis_raises(self, cartesian_3d: FieldDataset) -> None:
        with pytest.raises(ValueError, match="not found"):
            BoxSelection(ranges={"q": (0, 2)}).apply(cartesian_3d)
