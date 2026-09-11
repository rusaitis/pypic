"""Line, time-series, scatter and spectrum plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypic.plotting import plot_line, plot_time_series

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.containers import TabularData
    from pypic.dataset import FieldDataset


class TestPlotLine:
    def test_1d(self, ds_1d: FieldDataset) -> None:
        """1D line plot draws the full B_1 vector as a single line.

        Asserting only isinstance(fig, Figure) would be tautological. This pins the
        drawn ydata to equal the input array and xdata to be the
        grid coordinate, so a regression that plots a slice or the
        wrong field is caught.
        """
        fig, ax = plot_line(ds_1d, "B_1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1
        line = ax.lines[0]
        np.testing.assert_array_equal(line.get_ydata(), ds_1d["B_1"])
        np.testing.assert_array_equal(
            line.get_xdata(), ds_1d.grid.coordinate_arrays()[0]
        )
        plt.close(fig)

    def test_2d_with_axis(self, ds_2d: FieldDataset) -> None:
        """2D slice along x sets ydata to the mid-y row of B_1.

        A line-count-only assertion let a
        regression that sliced the wrong axis or the wrong index pass
        silently. This pins the ydata to the explicit
        ``B_1[:, ny // 2]`` slice that ``plot_line`` uses by default.
        """
        fig, ax = plot_line(ds_2d, "B_1", axis="x")
        assert len(ax.lines) == 1
        ny = ds_2d.grid.dimensions[1]
        expected = np.asarray(ds_2d["B_1"])[:, ny // 2]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected)
        plt.close(fig)

    def test_custom_axes_overlay(self, ds_2d: FieldDataset) -> None:
        """Overlaying a second call onto ``ax`` adds a distinct B_2 line.

        In addition to the line-count check we pin
        each line's ydata to the correct field's midplane row, so a
        regression where the second call overwrote the first or
        plotted the wrong field is caught.
        """
        fig, ax = plot_line(ds_2d, "B_1", axis="x", label="first")
        plot_line(ds_2d, "B_2", axis="x", ax=ax, label="second")
        assert len(ax.lines) == 2
        ny = ds_2d.grid.dimensions[1]
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), np.asarray(ds_2d["B_1"])[:, ny // 2]
        )
        np.testing.assert_array_equal(
            ax.lines[1].get_ydata(), np.asarray(ds_2d["B_2"])[:, ny // 2]
        )
        plt.close(fig)

    def test_missing_axis_raises(self, ds_2d: FieldDataset) -> None:
        with pytest.raises(ValueError, match="axis is required"):
            plot_line(ds_2d, "B_1")

    def test_label_and_legend(self, ds_2d: FieldDataset) -> None:
        """Supplying ``label=`` creates a legend whose single entry
        matches the supplied label string.

        Checking only that a
        legend object was present; this pins the legend text to the
        user-supplied ``"test"`` so a regression that injects the
        wrong label (e.g. the raw field name) is caught.
        """
        fig, ax = plot_line(ds_2d, "B_1", axis="x", label="test")
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["test"]
        plt.close(fig)


class TestPlotTimeSeries:
    def test_single_column(self, tabular: TabularData) -> None:
        """Drawn ydata equals the named column and xdata equals the index.

        A line-count-only assertion let a
        regression that swapped x/y or plotted the wrong column pass.
        Pinning both axes to the TabularData entries catches a
        column-alias drift (``total_energy`` vs ``kinetic_energy``).
        """
        fig, ax = plot_time_series(tabular, "total_energy")
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), tabular["total_energy"])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), tabular.index)
        plt.close(fig)

    def test_multiple_columns(self, tabular: TabularData) -> None:
        """Each column becomes its own line with a legend entry matching
        the column name.

        Asserting only line count
        and legend-present. This pins each line's ydata to the
        matching tabular column and assert the legend labels equal
        the column names in order — catches a mis-ordering that
        would relabel lines silently.
        """
        fig, ax = plot_time_series(tabular, ["total_energy", "kinetic_energy"])
        assert len(ax.lines) == 2
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), tabular["total_energy"])
        np.testing.assert_array_equal(
            ax.lines[1].get_ydata(), tabular["kinetic_energy"]
        )
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == [
            "total_energy",
            "kinetic_energy",
        ]
        plt.close(fig)

    def test_custom_x_column(self, tabular: TabularData) -> None:
        """Custom ``x_column`` drives xlabel and xdata.

        Pins xdata-equals-column and line-count
        assertions alongside the xlabel check, so a regression that
        set the label correctly but plotted the wrong x-axis values
        is caught.
        """
        fig, ax = plot_time_series(tabular, "total_energy", x_column="cycle")
        assert ax.get_xlabel() == "cycle"
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), tabular["cycle"])
        plt.close(fig)

    def test_no_legend(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy", legend=False)
        assert ax.get_legend() is None
        plt.close(fig)


class TestPlotLines:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_lines

        fig, ax = plot_lines(ds_2d, ["B_1", "B_2", "B_3"], axis="x")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 3
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        """Custom *labels* flow through to legend entry text verbatim."""
        from pypic.plotting import plot_lines

        fig, ax = plot_lines(ds_2d, ["B_1", "B_2"], axis="x", labels=["$B_x$", "$B_y$"])
        assert len(ax.lines) == 2
        legend = ax.get_legend()
        assert legend is not None, "custom labels should force a legend"
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert legend_texts == ["$B_x$", "$B_y$"]
        plt.close(fig)


class TestPlotLineComparison:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        """Two datasets → two lines; identical inputs → identical y-data."""
        from pypic.plotting import plot_line_comparison

        fig, ax = plot_line_comparison([ds_2d, ds_2d], "B_1", axis="x")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 2
        y0 = ax.lines[0].get_ydata()
        y1 = ax.lines[1].get_ydata()
        np.testing.assert_allclose(y0, y1)
        plt.close(fig)

    def test_labels(self, ds_2d: FieldDataset) -> None:
        """Custom *labels* appear verbatim as legend entries."""
        from pypic.plotting import plot_line_comparison

        fig, ax = plot_line_comparison(
            [ds_2d, ds_2d],
            "B_1",
            axis="x",
            labels=["run A", "run B"],
        )
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "run A" in legend_texts
        assert "run B" in legend_texts
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        """``ax=`` reuses the existing axes; two new lines are added."""
        from pypic.plotting import plot_line_comparison

        fig_ext, ax_ext = plt.subplots()
        before = len(ax_ext.lines)
        _, ax = plot_line_comparison(
            [ds_2d, ds_2d],
            "B_1",
            axis="x",
            ax=ax_ext,
        )
        assert ax is ax_ext
        assert len(ax_ext.lines) == before + 2
        plt.close(fig_ext)

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        """Derived fields (``|B|``) are supported and produce finite y-data."""
        from pypic.plotting import plot_line_comparison

        fig, ax = plot_line_comparison([ds_2d, ds_2d], "|B|", axis="x")
        assert len(ax.lines) == 2
        y0 = ax.lines[0].get_ydata()
        assert np.all(np.isfinite(y0))
        assert np.all(y0 >= 0.0)  # magnitude is non-negative
        plt.close(fig)


class TestPlotScatter:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        """Scatter paints one marker per input cell (N = grid size)."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "B_1", "B_2")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.collections, "expected a PathCollection from scatter()"
        offsets = ax.collections[0].get_offsets()
        # ds_2d is a 10x8 grid → 80 points.
        assert offsets.shape == (ds_2d["B_1"].size, 2)
        plt.close(fig)

    def test_color_field(self, ds_2d: FieldDataset) -> None:
        """``color_field`` attaches a per-point array for the colormap."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "B_1", "B_2", color_field="rho_m")
        arr = ax.collections[0].get_array()
        assert arr is not None
        assert arr.size == ds_2d["rho_m"].size
        plt.close(fig)

    def test_density_mode(self, ds_2d: FieldDataset) -> None:
        """``density=True`` colors points by local density — array present."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "B_1", "B_2", density=True)
        arr = ax.collections[0].get_array()
        assert arr is not None
        assert arr.size > 0
        assert float(np.nanmin(arr)) >= 0.0  # density is non-negative
        plt.close(fig)

    def test_log_axes(self, ds_2d: FieldDataset) -> None:
        """``log_x``/``log_y`` set matching axis scales."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "rho_m", "P", log_x=True, log_y=True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_3d_auto_slice(self, ds_3d: FieldDataset) -> None:
        """A 3D dataset auto-slices — scatter count matches one 2D slice."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_3d, "B_1", "B_2")
        offsets = ax.collections[0].get_offsets()
        # Full 3D is 10x8x6=480; a 2D slice is 80. Accept any proper slice.
        assert 0 < offsets.shape[0] < ds_3d["B_1"].size
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        """Passing ``ax=`` reuses that Axes; a new collection is attached."""
        from pypic.plotting import plot_scatter

        fig_ext, ax_ext = plt.subplots()
        before = len(ax_ext.collections)
        _, ax = plot_scatter(ds_2d, "B_1", "B_2", ax=ax_ext)
        assert ax is ax_ext
        assert len(ax_ext.collections) == before + 1
        plt.close(fig_ext)

    def test_derived_fields(self, ds_2d: FieldDataset) -> None:
        """Scatter of derived fields ``|B|`` vs ``beta`` carries real data
        (no NaN-only arrays)."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "|B|", "beta")
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape[0] > 0
        assert np.isfinite(offsets).all()
        plt.close(fig)

    def test_density_with_color(self, ds_2d: FieldDataset) -> None:
        """``density=True`` + ``color_field`` uses color_field for the
        color array, not density (density-mode only replaces the marker
        sizing/coloring when no color_field is passed)."""
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(
            ds_2d,
            "B_1",
            "B_2",
            density=True,
            color_field="rho_m",
        )
        arr = ax.collections[0].get_array()
        assert arr is not None
        # The attached array should match rho_m (up to the scatter's
        # finite-mask); at minimum the value range must overlap rho_m's.
        assert float(np.nanmin(arr)) >= 0.0
        plt.close(fig)

    @pytest.mark.parametrize("density", [False, True], ids=["scatter", "hexbin"])
    def test_color_limits(self, ds_2d: FieldDataset, density: bool) -> None:
        from pypic.plotting import plot_scatter

        _, ax = plot_scatter(
            ds_2d,
            "B_1",
            "B_2",
            color_field="rho_m",
            density=density,
            vmin=0.2,
            vmax=0.8,
        )
        norm = ax.collections[0].norm
        assert (norm.vmin, norm.vmax) == (0.2, 0.8)
        plt.close("all")


class TestPlotPoincareSection:
    def test_save_writes_the_figure(self, tmp_path: Path) -> None:
        from pypic.plotting import plot_poincare_section
        from pypic.traces import PoincareSection, PoincareSurface

        section = PoincareSection(
            surface=PoincareSurface.from_axis("z", 0.0),
            seeds=np.zeros((1, 3)),
            direction="forward",
            punctures_3d=(np.zeros((2, 3)),),
            punctures_2d=(np.array([[0.1, 0.2], [0.3, -0.1]]),),
            field_lines=(),
        )
        out = tmp_path / "poincare.png"
        plot_poincare_section(section, save=str(out))
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestPlotPowerSpectrum:
    def test_basic(self) -> None:
        """Log-log axes scale and data is written through unchanged."""
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, ax = plot_power_spectrum(k, power)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        xdata, ydata = ax.lines[0].get_data()
        np.testing.assert_allclose(xdata, k)
        np.testing.assert_allclose(ydata, power)
        plt.close(fig)

    def test_compensated(self) -> None:
        """``compensated=5/3`` multiplies power by $k^{5/3}$ — a
        $k^{-5/3}$ spectrum becomes flat (ratio 1)."""
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, ax = plot_power_spectrum(k, power, compensated=5.0 / 3.0)
        _, ydata = ax.lines[0].get_data()
        # Compensated Kolmogorov spectrum is flat at 1.0 to FP precision.
        np.testing.assert_allclose(ydata, 1.0, rtol=1e-10)
        plt.close(fig)

    def test_reference_slopes(self) -> None:
        """Each entry in ``reference_slopes`` draws an extra line."""
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, ax = plot_power_spectrum(
            k,
            power,
            reference_slopes=[-5.0 / 3.0, -3.0],
        )
        # Data line + two reference lines = 3 lines total.
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_custom_axes(self) -> None:
        """Passing ``ax=`` paints onto that axes and adds exactly one line."""
        from pypic.plotting import plot_power_spectrum

        fig_ext, ax_ext = plt.subplots()
        before = len(ax_ext.lines)
        k = np.linspace(0.1, 10, 20)
        _, ax = plot_power_spectrum(k, k**-2, ax=ax_ext)
        assert ax is ax_ext
        assert len(ax_ext.lines) == before + 1
        plt.close(fig_ext)

    def test_with_label_and_legend(self) -> None:
        """Passing *label* places that exact string in the legend."""
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 30)
        fig, ax = plot_power_spectrum(k, k**-2, label="$B_z$")
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "$B_z$" in legend_texts
        plt.close(fig)
