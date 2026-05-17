"""Tests for ``pypic.reductions.project`` (axis-reduction API)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pypic.coordinates import CARTESIAN, CoordinateGeometry, GeometryType
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.reductions import project
from pypic.selections import BoxSelection, PlaneSelection, SphereSelection
from pypic.units import Normalization


@pytest.fixture
def cartesian_3d() -> FieldDataset:
    """8x6x4 Cartesian dataset with B_1, B_2, B_3 — same shape as
    ``test_selections.cartesian_3d`` so projection composition tests
    line up with the selection conventions."""
    grid = GridInfo(
        dimensions=(8, 6, 4),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    rng = np.random.default_rng(42)
    fields = {
        "B_1": rng.standard_normal((8, 6, 4)),
        "B_2": rng.standard_normal((8, 6, 4)),
        "B_3": rng.standard_normal((8, 6, 4)),
    }
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


@pytest.fixture
def spherical_3d() -> FieldDataset:
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
    fields = {"B_1": np.arange(24, dtype=float).reshape(4, 3, 2)}
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


def test_integrate_constant_field_matches_trapezoid(
    cartesian_3d: FieldDataset,
) -> None:
    const = 3.0
    grid = cartesian_3d.grid
    fields = {
        "B_1": np.full(grid.dimensions, const),
        "B_2": np.zeros(grid.dimensions),
        "B_3": np.zeros(grid.dimensions),
    }
    ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())
    result = project(ds, "z", reduction="integrate")
    z_coord = grid.coordinate_arrays()[2]
    expected = np.trapezoid(fields["B_1"], x=z_coord, axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_integrate_random_field_matches_trapezoid(
    cartesian_3d: FieldDataset,
) -> None:
    z_coord = cartesian_3d.grid.coordinate_arrays()[2]
    expected = np.trapezoid(cartesian_3d["B_1"], x=z_coord, axis=2)
    result = project(cartesian_3d, "z", reduction="integrate")
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_sum_no_spacing_factor(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="sum")
    expected = cartesian_3d["B_1"].sum(axis=2)
    assert_array_equal(result["B_1"], expected)


@pytest.mark.parametrize("reduction", ["mean", "max", "min", "std"])
def test_mean_max_min_std_dispatch(cartesian_3d: FieldDataset, reduction: str) -> None:
    result = project(cartesian_3d, "z", reduction=reduction)  # type: ignore[arg-type]
    expected = getattr(cartesian_3d["B_1"], reduction)(axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_box_selection_composes(cartesian_3d: FieldDataset) -> None:
    box = BoxSelection(ranges={"x": (2, 5), "y": (1, 4)})
    via_kwarg = project(cartesian_3d, "z", selection=box, reduction="sum")
    via_apply = project(box.apply(cartesian_3d), "z", reduction="sum")
    assert_array_equal(via_kwarg["B_1"], via_apply["B_1"])


def test_sphere_selection_omit_drops_masked_cells(
    cartesian_3d: FieldDataset,
) -> None:
    # Keep a small inside-sphere region; outside is NaN. With skipna=True
    # the projection should be finite everywhere a column had any
    # surviving cell.
    sphere = SphereSelection(center=(4.0, 3.0, 2.0), radius=2.5, keep="inside")
    result = project(cartesian_3d, "z", selection=sphere, reduction="mean")
    # Inside the projected disk: at least one z-cell is non-NaN, so the
    # mean is finite.
    center_xy = result["B_1"][4, 3]
    assert np.isfinite(center_xy)


def test_sphere_selection_propagate_yields_nan(
    cartesian_3d: FieldDataset,
) -> None:
    sphere = SphereSelection(center=(4.0, 3.0, 2.0), radius=2.5, keep="inside")
    result = project(
        cartesian_3d,
        "z",
        selection=sphere,
        reduction="mean",
        nan_policy="propagate",
    )
    # A column at the box corner is entirely NaN under the sphere mask,
    # so propagation must yield NaN there.
    assert np.isnan(result["B_1"][0, 0])


def test_nan_policy_raise_errors_on_nan_input(
    cartesian_3d: FieldDataset,
) -> None:
    sphere = SphereSelection(center=(4.0, 3.0, 2.0), radius=2.5, keep="inside")
    with pytest.raises(ValueError, match="NaN cell"):
        project(cartesian_3d, "z", selection=sphere, nan_policy="raise")


def test_nan_policy_invalid_value(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="nan_policy must be"):
        project(cartesian_3d, "z", nan_policy="bogus")  # type: ignore[arg-type]


def test_invalid_reduction(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="reduction must be one of"):
        project(cartesian_3d, "z", reduction="median")  # type: ignore[arg-type]


def test_surviving_axes_after_project_z(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="mean")
    assert result.grid.surviving_axes == (0, 1)
    assert result.grid.surviving_axis_names == ("x", "y")
    assert result.grid.dimensions == (8, 6)


def test_surviving_axes_after_plane_then_project(
    cartesian_3d: FieldDataset,
) -> None:
    step1 = PlaneSelection(normal="y", index=2).apply(cartesian_3d)
    step2 = project(step1, "z", reduction="mean")
    assert step2.grid.surviving_axes == (0,)
    assert step2.grid.surviving_axis_names == ("x",)
    assert step2.grid.dimensions == (8,)


def test_unknown_axis_raises(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="not found"):
        project(cartesian_3d, "w")


def test_unknown_field_raises_keyerror(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(KeyError, match="nope"):
        project(cartesian_3d, "z", fields=["B_1", "nope"])


def test_fields_subset_drops_others(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", fields=["B_1"])
    assert "B_1" in result.xr.data_vars
    assert "B_2" not in result.xr.data_vars
    assert "B_3" not in result.xr.data_vars


def test_non_cartesian_raises_not_implemented(
    spherical_3d: FieldDataset,
) -> None:
    with pytest.raises(NotImplementedError, match="Cartesian"):
        project(spherical_3d, "r")


def test_aliases_preserved(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="mean")
    assert result.has_field("Bx")
    assert_array_equal(result["Bx"], result["B_1"])


def test_projection_attr_recorded(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="integrate")
    attr = result.xr["B_1"].attrs["projection"]
    assert attr == {"axis": "z", "reduction": "integrate", "length": 4.0}


def test_projection_attr_no_length_for_mean(
    cartesian_3d: FieldDataset,
) -> None:
    result = project(cartesian_3d, "z", reduction="mean")
    attr = result.xr["B_1"].attrs["projection"]
    assert attr == {"axis": "z", "reduction": "mean"}


def test_quantity_type_preserved(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="integrate")
    assert result.field_info("B_1").quantity_type == "b_field"


def test_method_form_matches_function(cartesian_3d: FieldDataset) -> None:
    via_func = project(cartesian_3d, "z", reduction="mean")
    via_method = cartesian_3d.project("z", reduction="mean")
    assert_array_equal(via_method["B_1"], via_func["B_1"])


def test_metadata_preserved(cartesian_3d: FieldDataset) -> None:
    result = project(cartesian_3d, "z", reduction="mean")
    assert result.normalization is cartesian_3d.normalization
    assert result.species == cartesian_3d.species
