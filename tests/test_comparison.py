"""Tests for :mod:`pypic.comparison`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from pypic.types import FloatArray

from pypic.comparison import (
    compare_fields,
    field_comparison_report,
    field_difference_dataset,
)
from pypic.dataset import FieldDataset
from pypic.diagnostics import l2_relative_error, linf_error
from pypic.regrid import align_grids, common_grid
from pypic.units import Normalization
from tests._helpers import make_uniform_grid


def _make_1d(
    n: int, dx: float, origin: float, func=None, *, normalization=None
) -> FieldDataset:
    """1D dataset with ``f(x) = 2x + 1`` and ``B1 = sin(pi x / 5)`` by default."""
    grid = make_uniform_grid(n, spacing=dx, origin=origin)
    (x,) = grid.coordinate_arrays()
    f = 2.0 * x + 1.0 if func is None else func(x)
    return FieldDataset.from_arrays(
        {"B1": f},
        grid,
        normalization if normalization is not None else Normalization.identity(),
    )


def _make_2d(
    nx: int,
    ny: int,
    *,
    dx: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    offset: float = 0.0,
) -> FieldDataset:
    """2D dataset: ``B1 = x + 2y + offset`` and ``rho_m = 1 + 0.1 x``."""
    grid = make_uniform_grid(nx, ny, spacing=dx, origin=origin)
    gx, gy = np.meshgrid(*grid.coordinate_arrays(), indexing="ij")
    return FieldDataset.from_arrays(
        {
            "B1": gx + 2.0 * gy + offset,
            "rho_m": 1.0 + 0.1 * gx,
        },
        grid,
        Normalization.identity(),
    )


# ---------------------------------------------------------------------------
# compare_fields
# ---------------------------------------------------------------------------


class TestCompareFields:
    """Tests for :func:`compare_fields`."""

    def test_identical_datasets_l2_zero(self) -> None:
        ds = _make_1d(10, 1.0, 0.0)
        assert compare_fields(ds, ds, "B1", metric="l2") == 0.0

    def test_identical_datasets_linf_zero(self) -> None:
        ds = _make_1d(10, 1.0, 0.0)
        assert compare_fields(ds, ds, "B1", metric="linf") == 0.0

    def test_uniform_offset_linf_matches(self) -> None:
        a = _make_2d(8, 8)
        b = _make_2d(8, 8, offset=0.5)
        result = compare_fields(a, b, "B1", metric="linf")
        assert_allclose(result, 0.5, atol=1e-12)

    def test_matches_manual_alignment(self) -> None:
        """``compare_fields`` equals ``align_grids`` + pure diagnostic."""
        coarse = _make_2d(6, 6, dx=2.0)
        fine = _make_2d(12, 12, dx=1.0, origin=(1.0, 1.0))
        a_aligned, b_aligned = align_grids(coarse, fine)
        expected = float(l2_relative_error(a_aligned["B1"], b_aligned["B1"]))
        result = compare_fields(coarse, fine, "B1", metric="l2")
        assert_allclose(result, expected, rtol=1e-12)

    def test_unknown_metric_raises(self) -> None:
        ds = _make_1d(4, 1.0, 0.0)
        with pytest.raises(ValueError, match="metric"):
            compare_fields(ds, ds, "B1", metric="rmse")

    def test_unknown_units_raises(self) -> None:
        ds = _make_1d(4, 1.0, 0.0)
        with pytest.raises(ValueError, match="units"):
            compare_fields(ds, ds, "B1", units="normalized")

    def test_unknown_field_raises(self) -> None:
        ds = _make_1d(4, 1.0, 0.0)
        with pytest.raises(KeyError, match="not found"):
            compare_fields(ds, ds, "missing_field")

    def test_alias_roundtrip(self) -> None:
        """Passing ``"Bx"`` equals passing the canonical ``"B1"``."""
        ds = _make_2d(6, 6)
        assert ds.has_field("Bx")  # Cartesian alias active
        via_alias = compare_fields(ds, ds, "Bx", metric="l2")
        via_canonical = compare_fields(ds, ds, "B1", metric="l2")
        assert via_alias == via_canonical == 0.0

    def test_si_vs_code_units_scale_linearly(self) -> None:
        """Linf error scales with the SI conversion factor."""
        norm = Normalization.pic_electron(1e18)
        grid = make_uniform_grid(8, spacing=1.0)
        (x,) = grid.coordinate_arrays()
        a = FieldDataset.from_arrays({"B1": np.sin(x)}, grid, norm)
        b = FieldDataset.from_arrays({"B1": np.cos(x)}, grid, norm)

        linf_code = compare_fields(a, b, "B1", metric="linf", units="code")
        linf_si = compare_fields(a, b, "B1", metric="linf", units="si")
        factor = norm.si_factor("b_field")
        assert_allclose(linf_si, linf_code * factor, rtol=1e-12)

    def test_si_vs_code_l2_identical(self) -> None:
        """Relative L2 is scale-invariant — SI and code agree."""
        norm = Normalization.pic_electron(1e18)
        grid = make_uniform_grid(8, spacing=1.0)
        (x,) = grid.coordinate_arrays()
        a = FieldDataset.from_arrays({"B1": np.sin(x)}, grid, norm)
        b = FieldDataset.from_arrays({"B1": np.cos(x)}, grid, norm)
        l2_code = compare_fields(a, b, "B1", metric="l2", units="code")
        l2_si = compare_fields(a, b, "B1", metric="l2", units="si")
        assert_allclose(l2_si, l2_code, rtol=1e-12)

    def test_ambiguous_alias_mapping_raises(self) -> None:
        """Alias resolving to different canonicals across datasets fails loud."""
        grid = make_uniform_grid(4, spacing=1.0)
        a = FieldDataset.from_arrays(
            {"B1": np.ones(4), "B2": np.zeros(4)},
            grid,
            Normalization.identity(),
            aliases={"shared": "B1"},
        )
        b = FieldDataset.from_arrays(
            {"B1": np.ones(4), "B2": np.zeros(4)},
            grid,
            Normalization.identity(),
            aliases={"shared": "B2"},
        )
        with pytest.raises(ValueError, match="different canonical names"):
            compare_fields(a, b, "shared")


# ---------------------------------------------------------------------------
# field_comparison_report
# ---------------------------------------------------------------------------


class TestFieldComparisonReport:
    """Tests for :func:`field_comparison_report`."""

    def test_default_fields_intersection(self) -> None:
        grid = make_uniform_grid(6, 6, spacing=1.0)
        a = FieldDataset.from_arrays(
            {
                "B1": np.ones((6, 6)),
                "B2": np.ones((6, 6)),
                "rho_m": np.ones((6, 6)),
            },
            grid,
            Normalization.identity(),
        )
        b = FieldDataset.from_arrays(
            {"B1": np.ones((6, 6)), "rho_m": np.ones((6, 6))},
            grid,
            Normalization.identity(),
        )
        report = field_comparison_report(a, b)
        assert set(report["fields"].keys()) == {"B1", "rho_m"}
        for entry in report["fields"].values():
            assert entry["l2"] == 0.0
            assert entry["linf"] == 0.0

    def test_explicit_field_list(self) -> None:
        ds = _make_2d(6, 6)
        report = field_comparison_report(ds, ds, fields=["rho_m"])
        assert list(report["fields"].keys()) == ["rho_m"]

    def test_report_shape(self) -> None:
        ds = _make_2d(6, 6)
        report = field_comparison_report(ds, ds)
        assert set(report.keys()) == {"fields", "grid", "units"}
        assert report["units"] == "si"
        grid = report["grid"]
        assert "common_dimensions" in grid
        assert "resolution_ratio" in grid
        assert len(grid["resolution_ratio"]) == 2

    def test_no_common_fields_raises(self) -> None:
        grid = make_uniform_grid(4, spacing=1.0)
        a = FieldDataset.from_arrays({"B1": np.ones(4)}, grid, Normalization.identity())
        b = FieldDataset.from_arrays(
            {"rho_m": np.ones(4)}, grid, Normalization.identity()
        )
        with pytest.raises(ValueError, match="No common fields"):
            field_comparison_report(a, b)

    def test_resolution_ratio_correct(self) -> None:
        a = _make_2d(6, 6, dx=2.0)
        b = _make_2d(12, 12, dx=1.0, origin=(1.0, 1.0))
        report = field_comparison_report(a, b)
        assert report["grid"]["resolution_ratio"] == (2.0, 2.0)


# ---------------------------------------------------------------------------
# field_difference_dataset
# ---------------------------------------------------------------------------


class TestFieldDifferenceDataset:
    """Tests for :func:`field_difference_dataset`."""

    def test_grid_matches_common_grid(self) -> None:
        a = _make_2d(6, 6, dx=2.0, origin=(0.0, 0.0))
        b = _make_2d(8, 8, dx=1.0, origin=(1.0, 1.0))
        diff = field_difference_dataset(a, b)
        expected = common_grid(a.grid, b.grid)
        assert diff.grid.dimensions == expected.dimensions
        assert diff.grid.spacing == expected.spacing
        assert diff.grid.origin == expected.origin

    def test_identical_datasets_all_zero(self) -> None:
        ds = _make_2d(6, 6)
        diff = field_difference_dataset(ds, ds)
        for name in diff.field_names():
            assert_allclose(diff[name], 0.0, atol=1e-12)

    def test_canonical_field_names(self) -> None:
        ds = _make_2d(6, 6)
        diff = field_difference_dataset(ds, ds)
        # Full intersection: B1 and rho_m, no aliases in the keys.
        assert set(diff.field_names()) == {"B1", "rho_m"}

    def test_field_info_preserved(self) -> None:
        ds = _make_2d(6, 6)
        diff = field_difference_dataset(ds, ds)
        assert diff.field_info("B1").quantity_type == "b_field"

    def test_metadata_records_units(self) -> None:
        ds = _make_2d(6, 6)
        diff = field_difference_dataset(ds, ds)
        assert diff.metadata["comparison"]["units"] == "si"

    def test_plottable_via_plot_field_slice(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        from pypic.plotting import plot_field_slice

        a = _make_2d(10, 10)
        b = _make_2d(10, 10, offset=0.3)
        diff = field_difference_dataset(a, b)
        fig, _ = plot_field_slice(diff, "B1")
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_explicit_field_list(self) -> None:
        ds = _make_2d(6, 6)
        diff = field_difference_dataset(ds, ds, fields=["B1"])
        assert diff.field_names() == ["B1"]


# ---------------------------------------------------------------------------
# Coarse-mismatch warning
# ---------------------------------------------------------------------------


class TestCoarseMismatchWarning:
    """Tests for the >10x spacing-ratio warning."""

    def test_no_warning_under_threshold(self) -> None:
        import warnings

        a = _make_2d(10, 10, dx=1.0)
        b = _make_2d(20, 20, dx=0.5)  # ratio 2x
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            compare_fields(a, b, "B1")
        mismatches = [w for w in record if "cross-scale" in str(w.message)]
        assert not mismatches

    def test_warns_above_threshold(self) -> None:
        a = _make_2d(10, 10, dx=1.0)
        b = _make_2d(200, 200, dx=0.05)  # ratio 20x
        with pytest.warns(UserWarning, match="cross-scale"):
            compare_fields(a, b, "B1")

    def test_warns_once_per_report(self) -> None:
        """A report over many fields still warns exactly once."""
        grid_a = make_uniform_grid(10, 10, spacing=1.0)
        grid_b = make_uniform_grid(200, 200, spacing=0.05)
        names = ["B1", "B2", "B3", "rho_m"]
        a = FieldDataset.from_arrays(
            cast(
                "dict[str, FloatArray]",
                {name: np.ones((10, 10)) for name in names},
            ),
            grid_a,
            Normalization.identity(),
        )
        b = FieldDataset.from_arrays(
            cast(
                "dict[str, FloatArray]",
                {name: np.ones((200, 200)) for name in names},
            ),
            grid_b,
            Normalization.identity(),
        )
        with pytest.warns(UserWarning, match="cross-scale") as record:
            field_comparison_report(a, b)
        mismatches = [w for w in record if "cross-scale" in str(w.message)]
        assert len(mismatches) == 1


# ---------------------------------------------------------------------------
# Structural invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grid_a_params", "grid_b_params"),
    [
        ((6, 6), (6, 6)),
        ((6, 6), (12, 12)),
        ((8, 8), (5, 5)),
    ],
)
def test_compare_fields_equivalent_to_align_plus_pure(
    grid_a_params: tuple[int, int],
    grid_b_params: tuple[int, int],
) -> None:
    """``compare_fields`` is exactly ``align_grids`` + pure diagnostic."""
    a = _make_2d(*grid_a_params, dx=1.0, origin=(0.0, 0.0))
    b = _make_2d(*grid_b_params, dx=1.0, origin=(0.5, 0.5))
    a_aligned, b_aligned = align_grids(a, b)
    expected_l2 = float(l2_relative_error(a_aligned["B1"], b_aligned["B1"]))
    expected_linf = float(linf_error(a_aligned["B1"], b_aligned["B1"]))
    assert compare_fields(a, b, "B1", metric="l2") == expected_l2
    assert compare_fields(a, b, "B1", metric="linf") == expected_linf
