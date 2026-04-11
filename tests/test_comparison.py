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
from pypic.coordinates.transforms import FrameTransform
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

    def test_custom_alias_survives_alignment(self) -> None:
        """Custom aliases passed to ``from_arrays`` resolve in the report.

        Regression for #5 — pre-fix, ``_resolve_field_list`` was called
        on the regridded datasets, whose alias tables only carry the
        Cartesian defaults; user-supplied aliases were dropped, so
        passing the alias name through ``fields=`` raised KeyError.
        """
        # Different grids → align_grids actually runs (no no-op).
        grid_a = make_uniform_grid(6, 6, spacing=1.0)
        grid_b = make_uniform_grid(12, 12, spacing=0.5)
        a = FieldDataset.from_arrays(
            {"B1": np.ones((6, 6))},
            grid_a,
            Normalization.identity(),
            aliases={"my_alias": "B1"},
        )
        b = FieldDataset.from_arrays(
            {"B1": np.ones((12, 12))},
            grid_b,
            Normalization.identity(),
            aliases={"my_alias": "B1"},
        )
        report = field_comparison_report(a, b, fields=["my_alias"])
        assert "B1" in report["fields"]

    def test_bad_field_raises_before_alignment(self) -> None:
        """Bad field name raises KeyError before paying align_grids cost."""
        a = _make_2d(6, 6, dx=1.0)
        b = _make_2d(12, 12, dx=0.5)
        with pytest.raises(KeyError, match="not found"):
            field_comparison_report(a, b, fields=["definitely_not_a_field"])


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

    def test_metadata_records_source_frames(self) -> None:
        """``source_frames`` captures the *original* frames after auto-transform."""
        grid = make_uniform_grid(6, 6, spacing=1.0)
        arr = np.ones((6, 6))
        a = FieldDataset.from_arrays(
            {"B1": arr}, grid, Normalization.identity(), frame="GSM"
        )
        # Identity GSE→GSM transform: enough to satisfy the frame check
        # without rotating the fields (the rotation math itself is tested
        # in test_transforms.py).
        b = FieldDataset.from_arrays(
            {"B1": arr},
            grid,
            Normalization.identity(),
            frame="GSE",
            transforms={"GSM": FrameTransform("GSE", "GSM")},
        )
        diff = field_difference_dataset(a, b)
        assert diff.metadata["comparison"]["source_frames"] == ("GSM", "GSE")

    def test_metadata_records_code_units(self) -> None:
        """Code-units path records the units choice in metadata."""
        norm = Normalization.pic_electron(1e18)
        grid = make_uniform_grid(6, spacing=1.0)
        a = FieldDataset.from_arrays({"B1": np.ones(6)}, grid, norm)
        b = FieldDataset.from_arrays({"B1": 0.5 * np.ones(6)}, grid, norm)
        diff = field_difference_dataset(a, b, units="code")
        assert diff.metadata["comparison"]["units"] == "code"

    def test_si_units_round_trip_no_double_conversion(self) -> None:
        """``units='si'`` result must not double-convert when ``in_si`` runs.

        Pre-fix the result kept ``a.normalization`` (PIC, code units) but
        stored SI values, so ``diff.in_si("B1")`` re-applied the SI factor
        and silently squared it. Now the result is given identity
        normalization so ``diff.in_si("B1")`` round-trips to ``diff["B1"]``.
        """
        norm = Normalization.pic_electron(1e18)
        grid = make_uniform_grid(6, spacing=1.0)
        a = FieldDataset.from_arrays({"B1": np.ones(6)}, grid, norm)
        b = FieldDataset.from_arrays({"B1": 0.5 * np.ones(6)}, grid, norm)
        diff = field_difference_dataset(a, b, units="si")
        # Identity normalization → in_si() and __getitem__ agree.
        assert_allclose(diff.in_si("B1"), diff["B1"], rtol=1e-12)
        # And the stored values equal the SI difference of the inputs.
        expected_si_diff = a.in_si("B1") - b.in_si("B1")
        assert_allclose(diff["B1"], expected_si_diff, rtol=1e-12)

    def test_code_units_keeps_source_normalization(self) -> None:
        """``units='code'`` preserves *a*'s normalization (no identity swap)."""
        norm = Normalization.pic_electron(1e18)
        grid = make_uniform_grid(6, spacing=1.0)
        a = FieldDataset.from_arrays({"B1": np.ones(6)}, grid, norm)
        b = FieldDataset.from_arrays({"B1": 0.5 * np.ones(6)}, grid, norm)
        diff = field_difference_dataset(a, b, units="code")
        # Same normalization as the inputs → in_si applies the factor.
        assert diff.normalization == norm
        assert_allclose(diff.in_si("B1"), 0.5 * norm.si_factor("b_field"), rtol=1e-12)

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
# Frame alignment — auto-transform B into A's frame before comparing
# ---------------------------------------------------------------------------


class TestFrameAlignment:
    """Tests for the frame auto-alignment behavior across all three APIs.

    Frames must agree before subtracting field values — otherwise you'd
    be computing ``Bx_GSM - Bx_GSE``, which is physically meaningless.
    The comparison module transforms B into A's frame via
    :meth:`FieldDataset.transform_to` and raises a clear ``ValueError``
    if no transform is registered.
    """

    @staticmethod
    def _gsm_dataset(value: float = 1.0) -> FieldDataset:
        grid = make_uniform_grid(6, 6, spacing=1.0)
        return FieldDataset.from_arrays(
            {"B1": np.full((6, 6), value)},
            grid,
            Normalization.identity(),
            frame="GSM",
        )

    @staticmethod
    def _gse_dataset_with_transform(value: float = 1.0) -> FieldDataset:
        grid = make_uniform_grid(6, 6, spacing=1.0)
        return FieldDataset.from_arrays(
            {"B1": np.full((6, 6), value)},
            grid,
            Normalization.identity(),
            frame="GSE",
            transforms={"GSM": FrameTransform("GSE", "GSM")},
        )

    def test_compare_fields_auto_transforms(self) -> None:
        """``compare_fields`` succeeds when B has a transform to A's frame."""
        a = self._gsm_dataset()
        b = self._gse_dataset_with_transform()
        # Identity transform → fields equal → zero L2.
        assert compare_fields(a, b, "B1") == 0.0

    def test_no_transform_raises_value_error(self) -> None:
        """Different frames + no transform → ValueError naming both frames."""
        a = self._gsm_dataset()
        grid = make_uniform_grid(6, 6, spacing=1.0)
        b = FieldDataset.from_arrays(
            {"B1": np.ones((6, 6))},
            grid,
            Normalization.identity(),
            frame="GSE",  # no transforms registered
        )
        with pytest.raises(ValueError, match=r"dataset B.*'GSE'.*'GSM'"):
            compare_fields(a, b, "B1")

    def test_explicit_frame_transforms_both(self) -> None:
        """``frame=...`` transforms *both* inputs to a third frame."""
        # A is GSM with a transform to GSE; B is GSE with a transform to
        # GSM. Compare in GSE → A gets transformed, B passes through.
        grid = make_uniform_grid(6, 6, spacing=1.0)
        arr = np.ones((6, 6))
        a = FieldDataset.from_arrays(
            {"B1": arr},
            grid,
            Normalization.identity(),
            frame="GSM",
            transforms={"GSE": FrameTransform("GSM", "GSE")},
        )
        b = FieldDataset.from_arrays(
            {"B1": arr},
            grid,
            Normalization.identity(),
            frame="GSE",
        )
        # Identity transforms → fields unchanged → zero L2.
        assert compare_fields(a, b, "B1", frame="GSE") == 0.0

    def test_explicit_frame_missing_transform_on_a_raises(self) -> None:
        """``frame=...`` raises ValueError naming A if A lacks the transform."""
        # A is GSM with no transforms; B is GSE with no transforms.
        # Asking for frame="GSE" should fail on A specifically.
        a = self._gsm_dataset()
        grid = make_uniform_grid(6, 6, spacing=1.0)
        b = FieldDataset.from_arrays(
            {"B1": np.ones((6, 6))},
            grid,
            Normalization.identity(),
            frame="GSE",
        )
        with pytest.raises(ValueError, match=r"dataset A.*'GSM'.*'GSE'"):
            compare_fields(a, b, "B1", frame="GSE")

    def test_explicit_frame_already_native_is_noop(self) -> None:
        """``frame=`` matching A's frame behaves like the default path."""
        a = self._gsm_dataset()
        b = self._gse_dataset_with_transform()
        # Both default and explicit "GSM" should give the same result
        # without requiring any transform on A.
        assert compare_fields(a, b, "B1") == compare_fields(a, b, "B1", frame="GSM")

    def test_field_difference_dataset_explicit_frame(self) -> None:
        """``field_difference_dataset(frame=...)`` puts the result in that frame."""
        grid = make_uniform_grid(6, 6, spacing=1.0)
        arr = np.ones((6, 6))
        a = FieldDataset.from_arrays(
            {"B1": arr},
            grid,
            Normalization.identity(),
            frame="GSM",
            transforms={"GSE": FrameTransform("GSM", "GSE")},
        )
        b = FieldDataset.from_arrays(
            {"B1": arr},
            grid,
            Normalization.identity(),
            frame="GSE",
        )
        diff = field_difference_dataset(a, b, frame="GSE")
        assert diff.frame == "GSE"
        # source_frames still records the originals, not the requested frame.
        assert diff.metadata["comparison"]["source_frames"] == ("GSM", "GSE")

    def test_field_difference_dataset_result_in_a_frame(self) -> None:
        """After auto-transform, the result dataset is in A's frame."""
        a = self._gsm_dataset()
        b = self._gse_dataset_with_transform()
        diff = field_difference_dataset(a, b)
        assert diff.frame == "GSM"

    def test_field_comparison_report_auto_transforms(self) -> None:
        """``field_comparison_report`` runs the same frame-alignment path."""
        a = self._gsm_dataset()
        b = self._gse_dataset_with_transform()
        report = field_comparison_report(a, b)
        assert report["fields"]["B1"]["l2"] == 0.0

    def test_same_frame_skips_transform(self) -> None:
        """Identical frames → no transform attempted (no transforms registered)."""
        # Both datasets in default "simulation" frame, neither has any
        # transforms. If _align_frames tried to transform, this would
        # raise KeyError("No transforms registered"). It must not.
        a = _make_2d(6, 6)
        b = _make_2d(6, 6)
        assert a.frame == b.frame  # sanity
        compare_fields(a, b, "B1")  # must not raise


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
# NaN handling — regression for #1a (synthetic) and #1b (real masks)
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """Regression tests for the two distinct NaN sources in cross-grid comparison.

    1. *Synthetic NaN* — fixed by tightening :func:`pypic.regrid.common_grid`
       to inclusive sample bounds (#1a). Comparing different-resolution grids
       on the same domain must not introduce boundary NaN.

    2. *Real NaN* — fixed by adding ``nan_policy='omit'`` (default) to the
       pure diagnostics (#1b). Upstream masks (sphere selections, divisions
       near nulls) must produce a useful similarity score with a warning,
       not a silently poisoned metric.
    """

    def test_synthetic_nan_eliminated_by_tight_common_grid(self) -> None:
        """10-cell vs 5-cell, same domain, constant field → exact zero metric."""
        # Pre-fix this case produced NaN: 10-cell samples [0.5..9.5] vs
        # 5-cell samples [1..9]. The OLD common_grid spanned [0, 10] with
        # dx=1, so target samples 0.5 and 9.5 fell outside the 5-cell
        # source range and became NaN. After #1a, common_grid uses
        # inclusive sample bounds → target samples land in [1..9], all
        # valid in both sources, no synthetic NaN.
        a = FieldDataset.from_arrays(
            {"B1": np.full(10, 7.0)},
            make_uniform_grid(10, spacing=1.0, origin=0.0),
            Normalization.identity(),
        )
        b = FieldDataset.from_arrays(
            {"B1": np.full(5, 7.0)},
            make_uniform_grid(5, spacing=2.0, origin=0.0),
            Normalization.identity(),
        )
        import warnings as _w

        with _w.catch_warnings(record=True) as record:
            _w.simplefilter("always")
            l2 = compare_fields(a, b, "B1", metric="l2")
            linf = compare_fields(a, b, "B1", metric="linf")
        assert l2 == 0.0
        assert linf == 0.0
        # No NaN-handling warning fired — the tight common_grid prevented
        # synthetic boundary NaN entirely.
        nan_warnings = [w for w in record if "NaN" in str(w.message)]
        assert nan_warnings == []

    def test_real_nan_returns_finite_metric_with_warning(self) -> None:
        """Upstream masks → finite L2/L∞ + UserWarning naming the count."""
        # Same grid for both → align_grids is a no-op; arrays go straight
        # to the pure diagnostic where ``nan_policy='omit'`` (default)
        # masks the NaN cells.
        grid = make_uniform_grid(10, spacing=1.0)
        values_a = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        values_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0])
        a = FieldDataset.from_arrays({"B1": values_a}, grid, Normalization.identity())
        b = FieldDataset.from_arrays({"B1": values_b}, grid, Normalization.identity())

        with pytest.warns(UserWarning, match=r"ignored 2 NaN") as record:
            l2 = compare_fields(a, b, "B1", metric="l2")
        # 8 valid cells, all identical → exact zero.
        assert l2 == 0.0
        # Exactly one warning per call (not one per cell).
        nan_warnings = [w for w in record if "NaN" in str(w.message)]
        assert len(nan_warnings) == 1

        with pytest.warns(UserWarning, match=r"ignored 2 NaN"):
            linf = compare_fields(a, b, "B1", metric="linf")
        assert linf == 0.0

    def test_real_nan_propagate_policy_returns_nan(self) -> None:
        """``nan_policy='propagate'`` opts out of masking via the public API."""
        grid = make_uniform_grid(10, spacing=1.0)
        values_a = np.ones(10)
        values_a[3] = np.nan
        values_b = np.ones(10)
        a = FieldDataset.from_arrays({"B1": values_a}, grid, Normalization.identity())
        b = FieldDataset.from_arrays({"B1": values_b}, grid, Normalization.identity())
        result = compare_fields(a, b, "B1", nan_policy="propagate")
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# method= passthrough
# ---------------------------------------------------------------------------


class TestMethodPassthrough:
    """Tests for the ``method=`` kwarg threaded through the comparison API."""

    def test_method_cubic_differs_from_linear_on_curved_field(self) -> None:
        """``method='cubic'`` should give different L2 than ``'linear'``."""
        # A is the smooth function sampled coarsely, B is sampled finely
        # so it's effectively the ground truth. Linear interpolation
        # under-samples curvature; cubic should fit it more closely.
        coarse = make_uniform_grid(10, spacing=1.0)
        fine = make_uniform_grid(40, spacing=0.25)
        (cx,) = coarse.coordinate_arrays()
        (fx,) = fine.coordinate_arrays()
        a = FieldDataset.from_arrays(
            {"B1": np.sin(cx)}, coarse, Normalization.identity()
        )
        b = FieldDataset.from_arrays({"B1": np.sin(fx)}, fine, Normalization.identity())
        l2_linear = compare_fields(a, b, "B1", method="linear")
        l2_cubic = compare_fields(a, b, "B1", method="cubic")
        # Both nonzero (10-cell vs 40-cell sin(x) — interpolation error
        # is real), and cubic is materially smaller than linear.
        assert l2_linear > 0.0
        assert l2_cubic > 0.0
        assert l2_cubic < l2_linear

    def test_method_validation_unknown_string_raises_via_scipy(self) -> None:
        """An unknown ``method`` string surfaces scipy's error promptly."""
        coarse = make_uniform_grid(6, spacing=1.0)
        fine = make_uniform_grid(12, spacing=0.5)
        a = FieldDataset.from_arrays(
            {"B1": np.ones(6)}, coarse, Normalization.identity()
        )
        b = FieldDataset.from_arrays(
            {"B1": np.ones(12)}, fine, Normalization.identity()
        )
        with pytest.raises(ValueError, match="method"):
            compare_fields(a, b, "B1", method="not_a_real_method")


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
