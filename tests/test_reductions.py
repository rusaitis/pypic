"""Tests for ``pypic.reductions.reduce`` (axis-reduction API)."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pypic.coordinates import CARTESIAN, CoordinateGeometry, GeometryType
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.reductions import Reduction, reduce
from pypic.selections import BoxSelection, PlaneSelection, SphereSelection
from pypic.units import Normalization


@pytest.fixture
def cartesian_3d() -> FieldDataset:
    """8x6x4 Cartesian dataset with B_1, B_2, B_3 — same shape as
    ``test_selections.cartesian_3d`` so reduction composition tests
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
    result = reduce(ds, "z", reduction="integrate")
    z_coord = grid.coordinate_arrays()[2]
    expected = np.trapezoid(fields["B_1"], x=z_coord, axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_integrate_random_field_matches_trapezoid(
    cartesian_3d: FieldDataset,
) -> None:
    z_coord = cartesian_3d.grid.coordinate_arrays()[2]
    expected = np.trapezoid(cartesian_3d["B_1"], x=z_coord, axis=2)
    result = reduce(cartesian_3d, "z", reduction="integrate")
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_sum_no_spacing_factor(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="sum")
    expected = cartesian_3d["B_1"].sum(axis=2)
    assert_array_equal(result["B_1"], expected)


@pytest.mark.parametrize("reduction", ["mean", "max", "min", "std"])
def test_mean_max_min_std_dispatch(cartesian_3d: FieldDataset, reduction: str) -> None:
    result = reduce(cartesian_3d, "z", reduction=reduction)  # type: ignore[arg-type]
    expected = getattr(cartesian_3d["B_1"], reduction)(axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_box_selection_composes(cartesian_3d: FieldDataset) -> None:
    box = BoxSelection(ranges={"x": (2, 5), "y": (1, 4)})
    via_kwarg = reduce(cartesian_3d, "z", selection=box, reduction="sum")
    via_apply = reduce(box.apply(cartesian_3d), "z", reduction="sum")
    assert_array_equal(via_kwarg["B_1"], via_apply["B_1"])


def test_sphere_selection_omit_drops_masked_cells(
    cartesian_3d: FieldDataset,
) -> None:
    # Keep a small inside-sphere region; outside is NaN. With skipna=True
    # the reduction should be finite everywhere a column had any
    # surviving cell.
    sphere = SphereSelection(center=(4.0, 3.0, 2.0), radius=2.5, keep="inside")
    result = reduce(cartesian_3d, "z", selection=sphere, reduction="mean")
    # Inside the projected disk: at least one z-cell is non-NaN, so the
    # mean is finite.
    center_xy = result["B_1"][4, 3]
    assert np.isfinite(center_xy)


def test_sphere_selection_propagate_yields_nan(
    cartesian_3d: FieldDataset,
) -> None:
    sphere = SphereSelection(center=(4.0, 3.0, 2.0), radius=2.5, keep="inside")
    result = reduce(
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
        reduce(cartesian_3d, "z", selection=sphere, nan_policy="raise")


def test_nan_policy_invalid_value(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="nan_policy must be"):
        reduce(cartesian_3d, "z", nan_policy="bogus")  # type: ignore[arg-type]


def test_invalid_reduction(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="reduction must be one of"):
        reduce(cartesian_3d, "z", reduction="bogus")  # type: ignore[arg-type]


def test_surviving_axes_after_reduce_z(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="mean")
    assert result.grid.surviving_axes == (0, 1)
    assert result.grid.surviving_axis_names == ("x", "y")
    assert result.grid.dimensions == (8, 6)


def test_surviving_axes_after_plane_then_reduce(
    cartesian_3d: FieldDataset,
) -> None:
    step1 = PlaneSelection(normal="y", index=2).apply(cartesian_3d)
    step2 = reduce(step1, "z", reduction="mean")
    assert step2.grid.surviving_axes == (0,)
    assert step2.grid.surviving_axis_names == ("x",)
    assert step2.grid.dimensions == (8,)


def test_unknown_axis_raises(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="not found"):
        reduce(cartesian_3d, "w")


def test_unknown_field_raises_keyerror(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(KeyError, match="nope"):
        reduce(cartesian_3d, "z", fields=["B_1", "nope"])


def test_fields_subset_drops_others(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", fields=["B_1"])
    assert "B_1" in result.xr.data_vars
    assert "B_2" not in result.xr.data_vars
    assert "B_3" not in result.xr.data_vars


def test_non_cartesian_raises_not_implemented(
    spherical_3d: FieldDataset,
) -> None:
    with pytest.raises(NotImplementedError, match="Cartesian"):
        reduce(spherical_3d, "r")


def test_reduce_along_time_dim(cartesian_3d: FieldDataset) -> None:
    """``reduce`` accepts non-grid dims like ``time``  — useful for
    ``to_zarr_timeseries`` outputs where ``time`` is the leading dim
    on every field array.  Grid covers spatial axes only; xarray
    tracks any extra leading dims independently."""
    import xarray as xr

    grid = cartesian_3d.grid
    base = cartesian_3d.xr["B_1"]
    # Stack two snapshots along a new "time" dim, with explicit coords
    # so trapezoidal integration also works.
    stacked_b1 = xr.concat([base, base + 10.0], dim="time").assign_coords(
        time=[0.0, 1.0]
    )
    stacked = xr.Dataset({"B_1": stacked_b1})
    fds = FieldDataset(stacked, grid, cartesian_3d.normalization)

    time_mean = reduce(fds, "time", reduction="mean")
    expected = (base.values + (base.values + 10.0)) / 2
    np.testing.assert_allclose(time_mean["B_1"], expected, rtol=1e-12)
    # Grid is unchanged — time is not a spatial axis.
    assert time_mean.grid.surviving_axis_names == ("x", "y", "z")
    assert time_mean.grid.dimensions == grid.dimensions


def test_reduce_along_time_dim_non_cartesian(
    spherical_3d: FieldDataset,
) -> None:
    """Pure non-spatial reductions bypass the Cartesian gate — the
    Jacobian only matters when integrating over a spatial axis."""
    import xarray as xr

    base = spherical_3d.xr["B_1"]
    stacked_b1 = xr.concat([base, base * 2.0], dim="time").assign_coords(
        time=[0.0, 1.0]
    )
    stacked = xr.Dataset({"B_1": stacked_b1})
    fds = FieldDataset(stacked, spherical_3d.grid, spherical_3d.normalization)
    # No NotImplementedError — time is not in surviving_axis_names.
    time_mean = reduce(fds, "time", reduction="mean")
    expected = (base.values + base.values * 2.0) / 2
    np.testing.assert_allclose(time_mean["B_1"], expected, rtol=1e-12)


def test_reduce_along_time_then_spatial_still_gates_geometry(
    spherical_3d: FieldDataset,
) -> None:
    """Mixed (time, r) reduction on a spherical grid must still raise:
    once any spatial axis enters the reduction the Jacobian deferral
    kicks in."""
    import xarray as xr

    base = spherical_3d.xr["B_1"]
    stacked_b1 = xr.concat([base, base * 2.0], dim="time").assign_coords(
        time=[0.0, 1.0]
    )
    stacked = xr.Dataset({"B_1": stacked_b1})
    fds = FieldDataset(stacked, spherical_3d.grid, spherical_3d.normalization)
    with pytest.raises(NotImplementedError, match="Cartesian"):
        reduce(fds, ("time", "r"), reduction="mean")


def test_aliases_preserved(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="mean")
    assert result.has_field("Bx")
    assert_array_equal(result["Bx"], result["B_1"])


def test_reduction_attr_recorded(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="integrate")
    attr = result.xr["B_1"].attrs["reduction"]
    assert attr == {"axis": "z", "op": "integrate"}


def test_reduction_attr_no_length_key(cartesian_3d: FieldDataset) -> None:
    """``length`` was dropped (was N·dx but trapezoidal covers (N-1)·dx)."""
    result = reduce(cartesian_3d, "z", reduction="mean")
    attr = result.xr["B_1"].attrs["reduction"]
    assert "length" not in attr


def test_quantity_type_preserved(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="integrate")
    assert result.field_info("B_1").quantity_type == "b_field"


def test_method_form_matches_function(cartesian_3d: FieldDataset) -> None:
    via_func = reduce(cartesian_3d, "z", reduction="mean")
    via_method = cartesian_3d.reduce("z", reduction="mean")
    assert_array_equal(via_method["B_1"], via_func["B_1"])


def test_metadata_preserved(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="mean")
    assert result.normalization is cartesian_3d.normalization
    assert result.species == cartesian_3d.species


def test_median_matches_np_median(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="median")
    expected = np.median(cartesian_3d["B_1"], axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_var_matches_np_var(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="var")
    expected = np.var(cartesian_3d["B_1"], axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_argmax_returns_coord_value(cartesian_3d: FieldDataset) -> None:
    """``idxmax`` returns the *coord value* at the extremum, not an index."""
    result = reduce(cartesian_3d, "z", reduction="argmax")
    z_coord = cartesian_3d.grid.coordinate_arrays()[2]
    # Every result cell must be one of the cell-center z-coordinates.
    assert np.isin(result["B_1"], z_coord).all()


def test_argmin_returns_coord_value(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="argmin")
    z_coord = cartesian_3d.grid.coordinate_arrays()[2]
    assert np.isin(result["B_1"], z_coord).all()


def test_argmax_overrides_quantity_type(cartesian_3d: FieldDataset) -> None:
    """``argmax`` returns an axis position — quantity becomes ``length``."""
    result = reduce(cartesian_3d, "z", reduction="argmax")
    assert result.field_info("B_1").quantity_type == "length"


def test_argmax_clears_field_descriptors(cartesian_3d: FieldDataset) -> None:
    """``argmax`` must drop ``latex`` and ``long_name`` — the array now
    holds coordinate positions, not field values, so the original
    field-specific descriptors would mislabel the result."""
    # Sanity: the input has these set (from the field registry).
    assert cartesian_3d.xr["B_1"].attrs.get("long_name")
    result = reduce(cartesian_3d, "z", reduction="argmax")
    assert "latex" not in result.xr["B_1"].attrs
    assert "long_name" not in result.xr["B_1"].attrs


def test_argmax_stamps_result_kind(cartesian_3d: FieldDataset) -> None:
    result = reduce(cartesian_3d, "z", reduction="argmax")
    attr = result.xr["B_1"].attrs["reduction"]
    assert attr["result_kind"] == "axis_position"


def test_argmax_rejects_multi_axis(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="single axis"):
        reduce(cartesian_3d, ("y", "z"), reduction="argmax")


def test_multi_axis_mean_equals_chained(cartesian_3d: FieldDataset) -> None:
    """Tuple form must match chained single-axis calls to machine precision."""
    one_call = reduce(cartesian_3d, ("y", "z"), reduction="mean")
    chained = reduce(
        reduce(cartesian_3d, "y", reduction="mean"),
        "z",
        reduction="mean",
    )
    assert_allclose(one_call["B_1"], chained["B_1"], rtol=1e-12)


def test_multi_axis_integrate_equals_chained_trapezoid(
    cartesian_3d: FieldDataset,
) -> None:
    one_call = reduce(cartesian_3d, ("y", "z"), reduction="integrate")
    chained = reduce(
        reduce(cartesian_3d, "y", reduction="integrate"),
        "z",
        reduction="integrate",
    )
    assert_allclose(one_call["B_1"], chained["B_1"], rtol=1e-12)


def test_multi_axis_surviving_axes(cartesian_3d: FieldDataset) -> None:
    """Two-axis reduction on a 3D dataset must leave one surviving axis."""
    result = reduce(cartesian_3d, ("y", "z"), reduction="mean")
    assert result.grid.surviving_axes == (0,)
    assert result.grid.surviving_axis_names == ("x",)
    assert result.grid.dimensions == (8,)


def test_multi_axis_attr_stores_tuple(cartesian_3d: FieldDataset) -> None:
    """For multi-axis, ``attrs['reduction']['axis']`` is the input tuple."""
    result = reduce(cartesian_3d, ("y", "z"), reduction="mean")
    attr = result.xr["B_1"].attrs["reduction"]
    assert attr["axis"] == ("y", "z")


def test_multi_axis_unknown_axis_raises(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="not found"):
        reduce(cartesian_3d, ("y", "w"), reduction="mean")


def test_multi_axis_duplicate_axis_raises(cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reduce(cartesian_3d, ("z", "z"), reduction="mean")


def test_reduction_type_alias_exported() -> None:
    """``Reduction`` Literal stays in sync with ``_VALID_REDUCTIONS``."""
    from pypic.reductions import _VALID_REDUCTIONS

    assert get_args(Reduction.__value__) == _VALID_REDUCTIONS
    assert "median" in _VALID_REDUCTIONS
    assert "argmax" in _VALID_REDUCTIONS


@pytest.fixture
def cartesian_weighted_3d() -> FieldDataset:
    """4x3x2 dataset with a smooth field and a positive weight field.

    Weight is strictly positive so the weighted mean / integrate
    denominators are non-zero everywhere, and the weighted answer
    can be cross-checked against a hand-computed reference.
    """
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    rng = np.random.default_rng(7)
    fields = {
        "B_1": rng.standard_normal((4, 3, 2)),
        "rho_c": 0.5 + rng.uniform(size=(4, 3, 2)),  # > 0 everywhere
    }
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


def test_weight_mean_matches_hand(cartesian_weighted_3d: FieldDataset) -> None:
    result = reduce(cartesian_weighted_3d, "z", reduction="mean", weight="rho_c")
    f = cartesian_weighted_3d["B_1"]
    w = cartesian_weighted_3d["rho_c"]
    expected = (f * w).sum(axis=2) / w.sum(axis=2)
    assert_allclose(result["B_1"], expected, rtol=1e-12)


def test_weight_integrate_matches_hand(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    result = reduce(cartesian_weighted_3d, "z", reduction="integrate", weight="rho_c")
    f = cartesian_weighted_3d["B_1"]
    w = cartesian_weighted_3d["rho_c"]
    z = cartesian_weighted_3d.grid.coordinate_arrays()[2]
    numer = np.trapezoid(f * w, x=z, axis=2)
    denom = np.trapezoid(w, x=z, axis=2)
    assert_allclose(result["B_1"], numer / denom, rtol=1e-12)


def test_weight_none_equals_unweighted(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    """``weight=None`` must reproduce the unweighted path exactly."""
    with_kwarg = reduce(cartesian_weighted_3d, "z", reduction="mean", weight=None)
    without = reduce(cartesian_weighted_3d, "z", reduction="mean")
    assert_array_equal(with_kwarg["B_1"], without["B_1"])


def test_weight_rejected_for_max(cartesian_weighted_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="weight="):
        reduce(cartesian_weighted_3d, "z", reduction="max", weight="rho_c")


@pytest.mark.parametrize(
    "reduction",
    ["sum", "median", "min", "std", "var", "argmax", "argmin"],
)
def test_weight_rejected_for_other_reductions(
    cartesian_weighted_3d: FieldDataset, reduction: str
) -> None:
    with pytest.raises(ValueError, match="weight="):
        reduce(
            cartesian_weighted_3d,
            "z",
            reduction=reduction,  # type: ignore[arg-type]
            weight="rho_c",
        )


def test_weight_unknown_field_raises(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    with pytest.raises(KeyError, match="weight field"):
        reduce(cartesian_weighted_3d, "z", reduction="mean", weight="not_a_field")


def test_weight_stamps_provenance_attr(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    result = reduce(cartesian_weighted_3d, "z", reduction="mean", weight="rho_c")
    attr = result.xr["B_1"].attrs["reduction"]
    assert attr["weight"] == "rho_c"
    assert attr["op"] == "mean"


def test_weight_composes_with_box_selection(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    """selection.apply runs first; weight resolves against cropped data."""
    box = BoxSelection(ranges={"x": (1, 3)})
    via_kwarg = reduce(
        cartesian_weighted_3d,
        "z",
        reduction="mean",
        selection=box,
        weight="rho_c",
    )
    via_apply = reduce(
        box.apply(cartesian_weighted_3d),
        "z",
        reduction="mean",
        weight="rho_c",
    )
    assert_array_equal(via_kwarg["B_1"], via_apply["B_1"])


def test_weight_resolves_alias(cartesian_weighted_3d: FieldDataset) -> None:
    """Aliases (e.g. species-name forms) resolve through resolve_key."""
    # rho_c has no canonical-form alias here, but B_1 has the alias "Bx".
    # Build a tiny synthetic case using Bx as the weight name.
    via_alias = reduce(cartesian_weighted_3d, "z", reduction="mean", weight="Bx")
    via_canonical = reduce(cartesian_weighted_3d, "z", reduction="mean", weight="B_1")
    # NaN-safe equality: weighted-by-Bx may produce NaN where Bx≈0;
    # the canonical path produces the same NaN pattern.
    assert_array_equal(
        np.asarray(np.isnan(via_alias["B_1"])),
        np.asarray(np.isnan(via_canonical["B_1"])),
    )
    np.testing.assert_allclose(
        via_alias["B_1"],
        via_canonical["B_1"],
        rtol=1e-12,
        equal_nan=True,
    )


def test_weight_with_nan_omit_uses_joint_mask(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    """Cells where field OR weight is NaN are skipped from both integrals."""
    f = cartesian_weighted_3d.xr["B_1"].values.copy()
    w = cartesian_weighted_3d.xr["rho_c"].values.copy()
    f[0, 0, 0] = np.nan  # field NaN
    w[0, 0, 1] = np.nan  # weight NaN
    grid = cartesian_weighted_3d.grid
    nan_ds = FieldDataset.from_arrays(
        {"B_1": f, "rho_c": w}, grid, Normalization.identity()
    )
    result = reduce(nan_ds, "z", reduction="mean", weight="rho_c", nan_policy="omit")
    # Hand-compute: in column (0,0), both z=0 (field NaN) and z=1 (weight
    # NaN) are masked, leaving no valid cells → weighted mean is NaN.
    assert np.isnan(result["B_1"][0, 0])
    # In an undisturbed column (e.g. (3, 2)), result equals unweighted-by-NaN
    # weighted mean.
    expected_other = (f[3, 2, :] * w[3, 2, :]).sum() / w[3, 2, :].sum()
    assert_allclose(result["B_1"][3, 2], expected_other, rtol=1e-12)


def test_weight_with_nan_raise_checks_weight(
    cartesian_weighted_3d: FieldDataset,
) -> None:
    """``nan_policy='raise'`` must catch NaN in the weight field too.

    Use ``fields=["B_1"]`` so the weight isn't already in
    ``ds_to_reduce`` (which would trip the per-field NaN check first).
    """
    w = cartesian_weighted_3d.xr["rho_c"].values.copy()
    w[0, 0, 0] = np.nan
    grid = cartesian_weighted_3d.grid
    nan_ds = FieldDataset.from_arrays(
        {"B_1": cartesian_weighted_3d.xr["B_1"].values, "rho_c": w},
        grid,
        Normalization.identity(),
    )
    with pytest.raises(ValueError, match="weight field"):
        reduce(
            nan_ds,
            "z",
            reduction="mean",
            fields=["B_1"],
            weight="rho_c",
            nan_policy="raise",
        )
