"""Tests for pypic.plotting module."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pypic.plotting import (  # noqa: E402
    DARK,
    DEFAULT,
    LIGHT,
    OVERLAY_BORDER_PAD,
    PlotTheme,
    VectorLegendEntry,
    add_inset_colorbar,
    add_panel_label,
    add_status_badge,
    add_vector_legend,
    plot_comparison,
    plot_field_slice,
    plot_line,
    plot_quiver,
    plot_streamlines,
    plot_time_series,
    use_theme,
)
from pypic.plotting._badge import _format_status_text  # noqa: E402
from pypic.plotting._colormaps import is_positive_definite, symmetric_clim  # noqa: E402
from pypic.readers.base import FieldDataset, GridInfo, TabularData  # noqa: E402
from pypic.selections import PlaneSelection  # noqa: E402
from pypic.units import Normalization  # noqa: E402


@pytest.fixture
def grid_2d() -> GridInfo:
    return GridInfo(dimensions=(10, 8), spacing=(1.0, 1.0), origin=(0.0, 0.0))


@pytest.fixture
def grid_3d() -> GridInfo:
    return GridInfo(
        dimensions=(10, 8, 6), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    )


@pytest.fixture
def ds_2d(grid_2d: GridInfo) -> FieldDataset:
    rng = np.random.default_rng(42)
    return FieldDataset.from_arrays(
        {
            "B1": rng.standard_normal((10, 8)),
            "B2": rng.standard_normal((10, 8)),
            "B3": rng.standard_normal((10, 8)),
            "rho_m": np.abs(rng.standard_normal((10, 8))) + 0.1,
            "P": np.abs(rng.standard_normal((10, 8))) + 0.1,
        },
        grid_2d,
        Normalization.identity(),
    )


@pytest.fixture
def ds_3d(grid_3d: GridInfo) -> FieldDataset:
    rng = np.random.default_rng(42)
    return FieldDataset.from_arrays(
        {
            "B1": rng.standard_normal((10, 8, 6)),
            "B2": rng.standard_normal((10, 8, 6)),
            "B3": rng.standard_normal((10, 8, 6)),
            "rho_m": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
            "P": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
        },
        grid_3d,
        Normalization.identity(),
    )


class TestPlotFieldSlice:
    def test_returns_figure_and_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_3d_auto_slice(self, ds_3d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_3d, "B1")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_3d_explicit_plane(self, ds_3d: FieldDataset) -> None:
        plane = PlaneSelection(normal="x", index=3)
        fig, _ax = plot_field_slice(ds_3d, "B2", plane=plane)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_computed_field(self, ds_3d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_3d, "|B|")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        import matplotlib.pyplot as plt

        _fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_field_slice(ds_2d, "B1", ax=ax_ext)
        assert ax is ax_ext
        matplotlib.pyplot.close(fig)

    def test_custom_title_and_step(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1", title="Custom", step=5)
        assert ax.get_title() == "Custom"
        matplotlib.pyplot.close(fig)

    def test_no_colorbar(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_2d, "B1", colorbar=False)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_2d, "B1", units="nT")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_dark_theme(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_2d, "B1", theme=DARK)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_non_square_no_transpose_bug(self, ds_2d: FieldDataset) -> None:
        """10x8 grid: pcolormesh should not raise shape errors."""
        fig, _ax = plot_field_slice(ds_2d, "B1")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_positive_definite_field(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_2d, "rho_m")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_symmetric_override(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_field_slice(ds_2d, "B1", symmetric=False)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestPlotComparison:
    def test_returns_figure_and_axes_dict(self, ds_2d: FieldDataset) -> None:
        fig, axes = plot_comparison(ds_2d, ds_2d, "B1")
        assert isinstance(fig, Figure)
        assert set(axes.keys()) == {"a", "b", "diff"}
        for ax in axes.values():
            assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_3d_auto_slice(self, ds_3d: FieldDataset) -> None:
        fig, axes = plot_comparison(ds_3d, ds_3d, "B1")
        assert "diff" in axes
        matplotlib.pyplot.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        fig, axes = plot_comparison(ds_2d, ds_2d, "B1", labels=("Run1", "Run2"))
        assert axes["a"].get_title() == "Run1"
        assert axes["b"].get_title() == "Run2"
        matplotlib.pyplot.close(fig)

    def test_derived_field(self, ds_3d: FieldDataset) -> None:
        fig, _axes = plot_comparison(ds_3d, ds_3d, "|B|")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestColormapDetection:
    def test_magnitude_always_positive(self) -> None:
        data = np.array([-1.0, 0.0, 1.0])
        assert is_positive_definite("|B|", data) is True

    def test_component_field_signed(self) -> None:
        data = np.array([-1.0, 0.0, 1.0])
        assert is_positive_definite("B1", data) is False

    def test_density_positive(self) -> None:
        from pypic.fields import FieldInfo

        data = np.array([1.0, 2.0])
        info = FieldInfo("density", "Number density", "m^-3")
        assert is_positive_definite("n_e", data, info) is True

    def test_charge_density_signed(self) -> None:
        from pypic.fields import FieldInfo

        data = np.array([-1.0, 1.0])
        info = FieldInfo("charge_density", "Charge density", "C/m^3")
        assert is_positive_definite("rho_c", data, info) is False

    def test_beta_positive(self) -> None:
        data = np.array([0.5, 1.0, 2.0])
        assert is_positive_definite("beta", data) is True

    def test_data_fallback_positive(self) -> None:
        data = np.array([0.0, 1.0, 2.0])
        assert is_positive_definite("custom_field", data) is True

    def test_data_fallback_signed(self) -> None:
        data = np.array([-1.0, 1.0])
        assert is_positive_definite("custom_field", data) is False


class TestSymmetricClim:
    def test_symmetric(self) -> None:
        data = np.array([-3.0, 1.0, 2.0])
        vmin, vmax = symmetric_clim(data)
        assert vmin == -3.0
        assert vmax == 3.0

    def test_all_nan(self) -> None:
        data = np.array([np.nan, np.nan, np.nan])
        vmin, vmax = symmetric_clim(data)
        assert vmin == 0.0
        assert vmax == 0.0


class TestResolveFieldValues:
    def test_stored_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "B1", units=None)
        np.testing.assert_array_equal(values, ds_2d["B1"])

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "|B|", units=None)
        expected = ds_2d.compute("|B|")
        np.testing.assert_array_equal(values, expected)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "B1", units="nT")
        expected = ds_2d.in_units("B1", "nT")
        np.testing.assert_array_equal(values, expected)


class TestThemes:
    def test_light_rcparams_valid(self) -> None:
        import matplotlib as mpl

        for key in LIGHT.rcparams:
            assert key in mpl.rcParams, f"{key} not a valid rcParam"

    def test_dark_rcparams_valid(self) -> None:
        import matplotlib as mpl

        for key in DARK.rcparams:
            assert key in mpl.rcParams, f"{key} not a valid rcParam"

    def test_default_is_light(self) -> None:
        assert DEFAULT is LIGHT

    def test_use_theme_restores_state(self) -> None:
        import matplotlib as mpl

        original = mpl.rcParams["figure.dpi"]
        custom = PlotTheme(
            name="test",
            rcparams={"figure.dpi": 72},
            sequential_cmap="viridis",
            diverging_cmap="coolwarm",
            grid_color="0.0",
        )
        with use_theme(custom):
            assert mpl.rcParams["figure.dpi"] == 72
        assert mpl.rcParams["figure.dpi"] == original


@pytest.fixture
def ds_1d() -> FieldDataset:
    grid = GridInfo(dimensions=(20,), spacing=(0.5,), origin=(0.0,))
    rng = np.random.default_rng(99)
    return FieldDataset.from_arrays(
        {"B1": rng.standard_normal(20)},
        grid,
        Normalization.identity(),
    )


@pytest.fixture
def tabular() -> TabularData:
    cycles = np.arange(50, dtype=np.float64)
    return TabularData(
        name="diagnostics",
        columns={
            "cycle": cycles,
            "total_energy": np.exp(-cycles / 20.0),
            "kinetic_energy": 0.5 * np.exp(-cycles / 20.0),
        },
        index_column="cycle",
    )


class TestPlotLine:
    def test_1d_returns_figure_and_axes(self, ds_1d: FieldDataset) -> None:
        fig, ax = plot_line(ds_1d, "B1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_2d_with_axis(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 1
        matplotlib.pyplot.close(fig)

    def test_3d_with_axis_and_index(self, ds_3d: FieldDataset) -> None:
        fig, ax = plot_line(ds_3d, "B1", axis="x", index={"y": 2, "z": 3})
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 1
        matplotlib.pyplot.close(fig)

    def test_computed_field(self, ds_3d: FieldDataset) -> None:
        fig, _ax = plot_line(ds_3d, "|B|", axis="x")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x", units="nT")
        assert "nT" in ax.get_ylabel()
        matplotlib.pyplot.close(fig)

    def test_custom_axes_overlay(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x")
        plot_line(ds_2d, "B2", axis="x", ax=ax, label="$B_y$")
        assert len(ax.lines) == 2
        matplotlib.pyplot.close(fig)

    def test_kwargs_passthrough(self, ds_1d: FieldDataset) -> None:
        fig, _ax = plot_line(ds_1d, "B1", color="red", linestyle="--")
        matplotlib.pyplot.close(fig)

    def test_missing_axis_raises(self, ds_2d: FieldDataset) -> None:
        with pytest.raises(ValueError, match="axis is required"):
            plot_line(ds_2d, "B1")

    def test_label_and_legend(self, ds_1d: FieldDataset) -> None:
        fig, ax = plot_line(ds_1d, "B1", label="test")
        legend = ax.get_legend()
        assert legend is not None
        matplotlib.pyplot.close(fig)


class TestPlotTimeSeries:
    def test_single_column(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 1
        matplotlib.pyplot.close(fig)

    def test_multiple_columns(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, ["total_energy", "kinetic_energy"])
        assert len(ax.lines) == 2
        assert ax.get_legend() is not None
        matplotlib.pyplot.close(fig)

    def test_custom_x_column(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy", x_column="cycle")
        assert ax.get_xlabel() == "cycle"
        matplotlib.pyplot.close(fig)

    def test_no_legend(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(
            tabular, ["total_energy", "kinetic_energy"], legend=False
        )
        assert ax.get_legend() is None
        matplotlib.pyplot.close(fig)

    def test_custom_labels(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(
            tabular,
            ["total_energy", "kinetic_energy"],
            labels=["Total", "Kinetic"],
        )
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["Total", "Kinetic"]
        matplotlib.pyplot.close(fig)

    def test_kwargs_passthrough(self, tabular: TabularData) -> None:
        fig, _ax = plot_time_series(tabular, "total_energy", linestyle="--")
        matplotlib.pyplot.close(fig)


class TestPlotStreamlines:
    def test_returns_figure_and_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(ds_2d, "B")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_auto_slices_3d(self, ds_3d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_3d, "B")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_explicit_plane(self, ds_3d: FieldDataset) -> None:
        plane = PlaneSelection(normal="x", index=3)
        fig, _ax = plot_streamlines(ds_3d, "B", plane=plane)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        import matplotlib.pyplot as plt

        _fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_streamlines(ds_2d, "B", ax=ax_ext)
        assert ax is ax_ext
        matplotlib.pyplot.close(fig)

    def test_color_field(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", color_field="|B|")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_linewidth_scaling(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", linewidth=(0.3, 3.0))
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_fixed_linewidth(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", linewidth=1.5)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_dark_theme(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", theme=DARK)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_no_colorbar(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", colorbar=False)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_uniform_color(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", color="black")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_alpha(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", color="white", alpha=0.4)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_alpha_with_colormap(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", alpha=0.6)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_arrowstyle(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(ds_2d, "B", arrowstyle="->")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_kwargs_passthrough(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_streamlines(
            ds_2d, "B", integration_direction="forward", minlength=0.2
        )
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestPlotQuiver:
    def test_returns_figure_and_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_quiver(ds_2d, "B")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_auto_slices_3d(self, ds_3d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_3d, "B")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_explicit_plane(self, ds_3d: FieldDataset) -> None:
        plane = PlaneSelection(normal="x", index=3)
        fig, _ax = plot_quiver(ds_3d, "B", plane=plane)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        import matplotlib.pyplot as plt

        _fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_quiver(ds_2d, "B", ax=ax_ext)
        assert ax is ax_ext
        matplotlib.pyplot.close(fig)

    def test_color_field(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", color_field="|B|")
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_stride(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", stride=2)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_stride_tuple(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", stride=(2, 3))
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_dark_theme(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", theme=DARK)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_no_colorbar(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", colorbar=False)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_uniform_color(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", color="black", stride=2)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_alpha(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", color="gray", alpha=0.5, stride=2)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_alpha_with_colormap(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(ds_2d, "B", alpha=0.6)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_kwargs_passthrough(self, ds_2d: FieldDataset) -> None:
        fig, _ax = plot_quiver(
            ds_2d, "B", headwidth=5, headlength=6, pivot="mid"
        )
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestStatusBadge:
    """Tests for add_status_badge and _format_status_text."""

    def test_step_only(self) -> None:
        text = _format_status_text(
            step=42, time=None, time_units="", step_range=None
        )
        assert text == "step 42"

    def test_time_only(self) -> None:
        text = _format_status_text(
            step=None, time=3.14, time_units="", step_range=None
        )
        assert text == "t = 3.14"

    def test_custom_label_step(self) -> None:
        text = _format_status_text(
            step=100, time=None, time_units="", step_range=None, label="cycle"
        )
        assert text == "cycle 100"

    def test_empty_label_step(self) -> None:
        text = _format_status_text(
            step=100, time=None, time_units="", step_range=None, label=""
        )
        assert text == "100"

    def test_show_max_true(self) -> None:
        text = _format_status_text(
            step=100, time=None, time_units="", step_range=(0, 500), show_max=True
        )
        assert text == "step 100 / 500"

    def test_show_max_false(self) -> None:
        text = _format_status_text(
            step=100, time=None, time_units="", step_range=(0, 500), show_max=False
        )
        assert text == "step 100"

    def test_custom_label_with_max(self) -> None:
        text = _format_status_text(
            step=100,
            time=None,
            time_units="",
            step_range=(0, 500),
            label="cycle",
            show_max=True,
        )
        assert text == "cycle 100 / 500"

    def test_empty_label_with_max(self) -> None:
        text = _format_status_text(
            step=100, time=None, time_units="", step_range=(0, 500), label=""
        )
        assert text == "100 / 500"

    def test_time_custom_label(self) -> None:
        text = _format_status_text(
            step=None, time=3.14, time_units="", step_range=None, label="time"
        )
        assert text == "time = 3.14"

    def test_time_empty_label(self) -> None:
        text = _format_status_text(
            step=None, time=3.14, time_units="", step_range=None, label=""
        )
        assert text == "3.14"

    def test_both_step_and_time(self) -> None:
        text = _format_status_text(
            step=100, time=5.0, time_units="", step_range=None
        )
        assert text == "step 100, t = 5.00"

    def test_both_with_custom_label(self) -> None:
        """label replaces step prefix; time keeps 't'."""
        text = _format_status_text(
            step=100, time=5.0, time_units="ns", step_range=None, label="cycle"
        )
        assert text == "cycle 100, t = 5.00 ns"

    def test_no_step_no_time_raises(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="At least one"):
            add_status_badge(ax)
        plt.close(fig)

    def test_custom_bg_color(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1, bg_color="red", bg_alpha=0.5)
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (1.0, 0.0, 0.0), atol=0.01)
        np.testing.assert_allclose(fc[3], 0.5, atol=0.01)
        plt.close(fig)

    def test_auto_detect_on_light_bg(self) -> None:
        """Auto-detect on white axes: lighter shade (near white)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1)
        fc = box.patch.get_facecolor()
        # White bg + lighter shade → stays white
        np.testing.assert_allclose(fc[:3], (1.0, 1.0, 1.0), atol=0.01)
        np.testing.assert_allclose(fc[3], 0.65, atol=0.01)
        plt.close(fig)

    def test_auto_detect_on_dark_bg(self) -> None:
        """Auto-detect on dark axes: darker shade (darkened facecolor)."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        with mpl.rc_context({"axes.facecolor": "#1e1e1e", "text.color": "#e0e0e0"}):
            fig, ax = plt.subplots()
            box = add_status_badge(ax, step=1)
            fc = box.patch.get_facecolor()
            # #1e1e1e * 0.4 → darkened, not pure black
            assert all(c < 0.12 for c in fc[:3]), f"expected dark bg, got {fc[:3]}"
            assert fc[3] == pytest.approx(0.65, abs=0.01)
            plt.close(fig)

    def test_shade_darker_on_light_bg(self) -> None:
        """shade='darker' on white bg produces gray overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1, shade="darker")
        fc = box.patch.get_facecolor()
        # white * 0.4 = (0.4, 0.4, 0.4)
        np.testing.assert_allclose(fc[:3], (0.4, 0.4, 0.4), atol=0.01)
        plt.close(fig)

    def test_shade_lighter(self) -> None:
        """shade='lighter' on white bg stays white."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1, shade="lighter")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (1.0, 1.0, 1.0), atol=0.01)
        plt.close(fig)

    def test_badge_renders_on_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1")
        box = add_status_badge(ax, step=42)
        assert box in ax.artists
        matplotlib.pyplot.close(fig)


class TestInsetColorbar:
    """Tests for add_inset_colorbar."""

    @pytest.fixture
    def mesh_on_ax(self) -> tuple:
        """Create an axes with a pcolormesh for testing."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 8))
        mesh = ax.pcolormesh(data)
        return fig, ax, mesh

    def test_returns_colorbar(self, mesh_on_ax: tuple) -> None:
        from matplotlib.colorbar import Colorbar

        fig, ax, mesh = mesh_on_ax
        cb = add_inset_colorbar(ax, mesh, "test")
        assert isinstance(cb, Colorbar)
        matplotlib.pyplot.close(fig)

    def test_light_mode_default(self, mesh_on_ax: tuple) -> None:
        """Auto-detect on default (white) axes: white bg patch at alpha 0.65."""
        fig, ax, mesh = mesh_on_ax
        add_inset_colorbar(ax, mesh)
        bg_patch = ax.patches[-1]
        fc = bg_patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (1.0, 1.0, 1.0), atol=0.01)
        np.testing.assert_allclose(fc[3], 0.65, atol=0.01)
        matplotlib.pyplot.close(fig)

    def test_shade_darker(self, mesh_on_ax: tuple) -> None:
        """shade='darker' on white bg → gray overlay."""
        fig, ax, mesh = mesh_on_ax
        add_inset_colorbar(ax, mesh, shade="darker")
        bg_patch = ax.patches[-1]
        fc = bg_patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (0.4, 0.4, 0.4), atol=0.01)
        np.testing.assert_allclose(fc[3], 0.65, atol=0.01)
        matplotlib.pyplot.close(fig)

    def test_custom_colors(self, mesh_on_ax: tuple) -> None:
        from matplotlib.colors import to_rgba

        fig, ax, mesh = mesh_on_ax
        add_inset_colorbar(ax, mesh, bg_color="navy", text_color="gold")
        bg_patch = ax.patches[-1]
        fc = bg_patch.get_facecolor()
        expected_bg = to_rgba("navy")
        np.testing.assert_allclose(fc[:3], expected_bg[:3], atol=0.01)
        matplotlib.pyplot.close(fig)

    @pytest.mark.parametrize(
        "loc",
        ["upper left", "upper right", "lower left", "lower right", "upper center"],
    )
    def test_all_locs(self, loc: str, mesh_on_ax: tuple) -> None:
        fig, ax, mesh = mesh_on_ax
        cb = add_inset_colorbar(ax, mesh, loc=loc)
        assert cb is not None
        matplotlib.pyplot.close(fig)

    def test_n_ticks(self, mesh_on_ax: tuple) -> None:
        fig, ax, mesh = mesh_on_ax
        cb = add_inset_colorbar(ax, mesh, n_ticks=2)
        fig.canvas.draw()
        tick_labels = cb.ax.get_xticklabels()
        visible = [t for t in tick_labels if t.get_text()]
        assert len(visible) <= 4
        matplotlib.pyplot.close(fig)

    def test_via_plot_field_slice(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1", colorbar="inset")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_via_streamlines(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(ds_2d, "B", colorbar="inset")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)

    def test_via_quiver(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_quiver(ds_2d, "B", colorbar="inset")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        matplotlib.pyplot.close(fig)


class TestVectorLegend:
    """Tests for VectorLegendEntry and add_vector_legend."""

    def test_single_entry(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry)
        assert box in ax.artists
        plt.close(fig)

    def test_multiple_entries(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entries = [
            VectorLegendEntry(label="B", color="black"),
            VectorLegendEntry(label="V", color="red", linewidth=2.0),
        ]
        box = add_vector_legend(ax, entries)
        assert box in ax.artists
        plt.close(fig)

    def test_shade_darker(self) -> None:
        """shade='darker' on white bg → gray overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="white")
        box = add_vector_legend(ax, entry, shade="darker")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (0.4, 0.4, 0.4), atol=0.01)
        plt.close(fig)

    def test_shade_lighter(self) -> None:
        """shade='lighter' on white bg → stays white."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry, shade="lighter")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (1.0, 1.0, 1.0), atol=0.01)
        plt.close(fig)

    def test_custom_colors(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry, bg_color="navy", text_color="gold")
        fc = box.patch.get_facecolor()
        from matplotlib.colors import to_rgba

        np.testing.assert_allclose(fc[:3], to_rgba("navy")[:3], atol=0.01)
        plt.close(fig)

    def test_auto_legend_on_streamlines(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(ds_2d, "B", color="black")
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) >= 1
        matplotlib.pyplot.close(fig)

    def test_auto_legend_on_quiver(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_quiver(ds_2d, "B", color="black", stride=2)
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) >= 1
        matplotlib.pyplot.close(fig)

    def test_legend_false_disables(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(ds_2d, "B", color="black", legend=False)
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) == 0
        matplotlib.pyplot.close(fig)

    def test_custom_legend_label(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(
            ds_2d, "B", color="black", legend="Magnetic field"
        )
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) >= 1
        matplotlib.pyplot.close(fig)


class TestPanelLabel:
    """Tests for add_panel_label."""

    def test_returns_anchored_offsetbox(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredOffsetbox

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a")
        assert isinstance(box, AnchoredOffsetbox)
        assert box in ax.artists
        plt.close(fig)

    def test_shade_darker(self) -> None:
        """shade='darker' on white bg → gray overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a", shade="darker")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (0.4, 0.4, 0.4), atol=0.01)
        plt.close(fig)

    def test_shade_lighter(self) -> None:
        """shade='lighter' on white bg → stays white."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a", shade="lighter")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], (1.0, 1.0, 1.0), atol=0.01)
        plt.close(fig)

    def test_custom_colors(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "b", bg_color="red", text_color="white")
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], to_rgba("red")[:3], atol=0.01)
        plt.close(fig)

    @pytest.mark.parametrize(
        "loc",
        ["upper left", "upper right", "lower left", "lower right", "upper center"],
    )
    def test_all_locs(self, loc: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "c", loc=loc)
        assert box in ax.artists
        plt.close(fig)

    def test_multi_panel_grid(self) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3)
        labels = ["a", "b", "c"]
        for ax, label in zip(axes, labels, strict=True):
            box = add_panel_label(ax, label)
            assert box in ax.artists
        plt.close(fig)


class TestOverlayTextAlpha:
    """Text alpha propagation for all overlay elements."""

    def test_badge_default_text_alpha(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1)
        text_area = box.get_child()
        color = text_area._text.get_color()
        assert len(color) == 4
        np.testing.assert_allclose(color[3], 0.85, atol=0.01)
        plt.close(fig)

    def test_badge_explicit_text_alpha(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1, text_alpha=1.0)
        text_area = box.get_child()
        color = text_area._text.get_color()
        np.testing.assert_allclose(color[3], 1.0, atol=0.01)
        plt.close(fig)

    def test_panel_label_default_text_alpha(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a")
        text_area = box.get_child()
        color = text_area._text.get_color()
        assert len(color) == 4
        np.testing.assert_allclose(color[3], 0.85, atol=0.01)
        plt.close(fig)

    def test_vector_legend_default_text_alpha(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry)
        # Single entry: child is an HPacker containing [DrawingArea, TextArea]
        hpacker = box.get_child()
        text_area = hpacker.get_children()[1]
        color = text_area._text.get_color()
        assert len(color) == 4
        np.testing.assert_allclose(color[3], 0.85, atol=0.01)
        plt.close(fig)

    def test_inset_colorbar_default_text_alpha(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        mesh = ax.pcolormesh(rng.standard_normal((10, 8)))
        cb = add_inset_colorbar(ax, mesh, "test")
        fig.canvas.draw()
        tick_labels = cb.ax.get_xticklabels()
        visible = [t for t in tick_labels if t.get_text()]
        if visible:
            color = visible[0].get_color()
            np.testing.assert_allclose(color[3], 0.85, atol=0.01)
        plt.close(fig)


class TestOverlayBorderPad:
    """OVERLAY_BORDER_PAD used consistently across overlays."""

    def test_constant_value(self) -> None:
        assert OVERLAY_BORDER_PAD == 0.6

    def test_badge_uses_constant(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1)
        assert box.borderpad == OVERLAY_BORDER_PAD
        plt.close(fig)

    def test_panel_label_uses_constant(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a")
        assert box.borderpad == OVERLAY_BORDER_PAD
        plt.close(fig)

    def test_vector_legend_uses_constant(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry)
        assert box.borderpad == OVERLAY_BORDER_PAD
        plt.close(fig)

    def test_legend_borderaxespad_in_themes(self) -> None:
        assert LIGHT.rcparams["legend.borderaxespad"] == OVERLAY_BORDER_PAD
        assert DARK.rcparams["legend.borderaxespad"] == OVERLAY_BORDER_PAD
