"""Tests for pypic.plotting module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pypic.plotting import (  # noqa: E402
    LegendEntry,
    add_badge,
    add_inset_colorbar,
    add_label,
    add_legend,
    get_theme,
    plot_comparison,
    plot_field_slice,
    plot_line,
    plot_quiver,
    plot_streamlines,
    plot_time_series,
    set_theme,
    use_theme,
)
from pypic.plotting._badge import (  # noqa: E402
    _detect_overlay_defaults,
    _format_status_text,
)
from pypic.plotting._colormaps import is_positive_definite, symmetric_clim  # noqa: E402
from pypic.plotting.styles import _resolve_theme_arg  # noqa: E402
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


class TestPlotFieldSlice:
    def test_returns_figure_and_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_field_slice(ds_2d, "B1", ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_custom_title_and_step(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1", title="Custom", step=42)
        assert ax.get_title() == "Custom"
        plt.close(fig)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"field": "|B|"}, id="derived"),
            pytest.param({"colorbar": False}, id="no-colorbar"),
            pytest.param({"colorbar": "inset"}, id="inset-colorbar"),
            pytest.param({"units": "nT"}, id="units"),
            pytest.param({"theme": "dark"}, id="dark-theme"),
            pytest.param({"symmetric": False}, id="symmetric-override"),
            pytest.param({"extremes": "transparent"}, id="transparent-extremes"),
            pytest.param({"extremes": "semi"}, id="semi-extremes"),
            pytest.param({"extremes": "darken"}, id="darken-extremes"),
            pytest.param({"extremes": None}, id="none-extremes"),
            pytest.param({"alpha": 0.5}, id="alpha"),
        ],
    )
    def test_options(self, ds_2d: FieldDataset, kwargs: dict) -> None:
        field = kwargs.pop("field", "B1")
        fig, _ = plot_field_slice(ds_2d, field, **kwargs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_3d_auto_and_explicit(self, ds_3d: FieldDataset) -> None:
        fig1, _ = plot_field_slice(ds_3d, "B1")
        fig2, _ = plot_field_slice(
            ds_3d, "B2", plane=PlaneSelection(normal="x", index=3)
        )
        plt.close(fig1)
        plt.close(fig2)


class TestPlotComparison:
    def test_returns_figure_and_axes_dict(self, ds_2d: FieldDataset) -> None:
        fig, axes = plot_comparison(ds_2d, ds_2d, "B1")
        assert isinstance(fig, Figure)
        assert set(axes) == {"a", "b", "diff"}
        plt.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        fig, axes = plot_comparison(ds_2d, ds_2d, "B1", labels=("Run1", "Run2"))
        assert axes["a"].get_title() == "Run1"
        assert axes["b"].get_title() == "Run2"
        plt.close(fig)

    def test_3d_and_derived(self, ds_3d: FieldDataset) -> None:
        fig, _ = plot_comparison(ds_3d, ds_3d, "|B|")
        assert isinstance(fig, Figure)
        plt.close(fig)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"colorbar": "inset"}, id="inset-colorbar"),
            pytest.param({"colorbar": False}, id="no-colorbar"),
            pytest.param({"extremes": "transparent"}, id="transparent-extremes"),
            pytest.param({"extremes": "semi"}, id="semi-extremes"),
            pytest.param({"extremes": "darken"}, id="darken-extremes"),
            pytest.param({"extremes": None}, id="none-extremes"),
            pytest.param({"vmin": -1.0, "vmax": 1.0}, id="custom-clim"),
        ],
    )
    def test_options(self, ds_2d: FieldDataset, kwargs: dict) -> None:
        fig, _ = plot_comparison(ds_2d, ds_2d, "B1", **kwargs)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestColormapDetection:
    @pytest.mark.parametrize(
        ("name", "data_positive", "quantity_type", "expected"),
        [
            ("|B|", False, None, True),
            ("B1", False, None, False),
            ("rho_m", False, "density", True),
            ("rho_c", False, "charge_density", False),
            ("beta", True, None, True),
            ("unknown", True, None, True),
            ("unknown", False, None, False),
        ],
        ids=[
            "magnitude",
            "component",
            "density",
            "charge-density",
            "beta",
            "all-positive-fallback",
            "mixed-sign-fallback",
        ],
    )
    def test_detection(
        self,
        name: str,
        data_positive: bool,
        quantity_type: str | None,
        expected: bool,
    ) -> None:
        from pypic.fields import FieldInfo

        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([-1.0, 0.0, 1.0])
        data = pos if data_positive else neg
        info = (
            FieldInfo(quantity_type=quantity_type, long_name="", si_unit="")
            if quantity_type
            else None
        )
        assert is_positive_definite(name, data, info) is expected


class TestSymmetricClim:
    def test_symmetric(self) -> None:
        assert symmetric_clim(np.array([-3.0, 1.0, 2.0])) == (-3.0, 3.0)

    def test_all_nan(self) -> None:
        vmin, vmax = symmetric_clim(np.array([np.nan, np.nan]))
        assert vmin < 0 < vmax  # epsilon expansion, not degenerate (0, 0)

    def test_uniform_field(self) -> None:
        vmin, vmax = symmetric_clim(np.array([0.0, 0.0, 0.0]))
        assert vmin < 0 < vmax


class TestResolveFieldValues:
    def test_stored_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "B1", None)
        np.testing.assert_array_equal(values, ds_2d["B1"])

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "|B|", None)
        expected = np.sqrt(ds_2d["B1"] ** 2 + ds_2d["B2"] ** 2 + ds_2d["B3"] ** 2)
        np.testing.assert_allclose(values, expected)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        code = resolve_field_values(ds_2d, "B1", None)
        si = resolve_field_values(ds_2d, "B1", "nT")
        assert si.shape == code.shape


class TestThemes:
    @pytest.mark.parametrize("name", ["light", "dark"])
    def test_rcparams_valid(self, name: str) -> None:
        theme = _resolve_theme_arg(name)
        valid_keys = set(matplotlib.rcParams)
        for key in theme.rcparams:
            assert key in valid_keys, f"{key!r} not a valid rcParam"

    def test_default_is_light(self) -> None:
        assert get_theme().name == "light"

    def test_customize_rcparam(self) -> None:
        light = get_theme()
        big = light.customize(font_size=14)
        assert big.rcparams["font.size"] == 14
        assert big.sequential_cmap == light.sequential_cmap

    def test_customize_field(self) -> None:
        dark = _resolve_theme_arg("dark")
        custom = dark.customize(
            sequential_cmaps=("viridis",),
            grid_color=(0.5, 0.5, 0.5, 0.1),
        )
        assert custom.sequential_cmap == "viridis"
        assert custom.grid_color == (0.5, 0.5, 0.5, 0.1)
        assert custom.diverging_cmap == dark.diverging_cmap

    def test_customize_preserves_original(self) -> None:
        light = get_theme()
        original_size = light.rcparams["font.size"]
        _ = light.customize(font_size=20)
        assert light.rcparams["font.size"] == original_size


class TestPlotLine:
    def test_1d(self, ds_1d: FieldDataset) -> None:
        fig, ax = plot_line(ds_1d, "B1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_2d_with_axis(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x")
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_custom_axes_overlay(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x", label="first")
        plot_line(ds_2d, "B2", axis="x", ax=ax, label="second")
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_missing_axis_raises(self, ds_2d: FieldDataset) -> None:
        with pytest.raises(ValueError, match="axis is required"):
            plot_line(ds_2d, "B1")

    def test_label_and_legend(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_line(ds_2d, "B1", axis="x", label="test")
        assert ax.get_legend() is not None
        plt.close(fig)


class TestPlotTimeSeries:
    def test_single_column(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy")
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_multiple_columns(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, ["total_energy", "kinetic_energy"])
        assert len(ax.lines) == 2
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_x_column(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy", x_column="cycle")
        assert ax.get_xlabel() == "cycle"
        plt.close(fig)

    def test_no_legend(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(tabular, "total_energy", legend=False)
        assert ax.get_legend() is None
        plt.close(fig)


class TestVectorPlots:
    """Shared tests for plot_streamlines and plot_quiver."""

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_returns_figure_and_axes(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        fig, ax = plot_fn(ds_2d, "B")  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_custom_axes(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_fn(ds_2d, "B", ax=ax_ext)  # type: ignore[operator]
        assert ax is ax_ext
        plt.close(fig_ext)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_uniform_color(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        fig, _ = plot_fn(ds_2d, "B", color="black")  # type: ignore[operator]
        assert isinstance(fig, Figure)
        plt.close(fig)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_auto_legend(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        fig, ax = plot_fn(ds_2d, "B", color="black")  # type: ignore[operator]
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) >= 1
        plt.close(fig)

    def test_legend_false_disables(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_streamlines(ds_2d, "B", color="black", legend=False)
        legend_boxes = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(legend_boxes) == 0
        plt.close(fig)

    @pytest.mark.parametrize(
        ("plot_fn", "extra_kwargs"),
        [
            (plot_streamlines, {"color_field": "|B|"}),
            (plot_streamlines, {"theme": "dark"}),
            (plot_streamlines, {"colorbar": False}),
            (plot_streamlines, {"alpha": 0.4, "color": "black"}),
            (plot_streamlines, {"extremes": "transparent"}),
            (plot_streamlines, {"extremes": "semi"}),
            (plot_streamlines, {"extremes": None}),
            (plot_quiver, {"stride": 2}),
            (plot_quiver, {"stride": (2, 3)}),
            (plot_quiver, {"theme": "dark"}),
            (plot_quiver, {"colorbar": False}),
            (plot_quiver, {"extremes": "transparent"}),
            (plot_quiver, {"extremes": "semi"}),
            (plot_quiver, {"extremes": None}),
        ],
        ids=[
            "stream-color_field",
            "stream-dark",
            "stream-no-cb",
            "stream-alpha",
            "stream-transparent",
            "stream-semi",
            "stream-none",
            "quiver-stride",
            "quiver-stride-tuple",
            "quiver-dark",
            "quiver-no-cb",
            "quiver-transparent",
            "quiver-semi",
            "quiver-none",
        ],
    )
    def test_options(
        self, ds_2d: FieldDataset, plot_fn: object, extra_kwargs: dict
    ) -> None:
        fig, _ = plot_fn(ds_2d, "B", **extra_kwargs)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestStatusBadge:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"step": 42}, "step 42"),
            ({"time": 3.14}, "t = 3.14"),
            ({"step": 42, "time": 3.14}, "step 42, t = 3.14"),
            ({"step": 42, "label": "cycle"}, "cycle 42"),
            ({"step": 42, "label": ""}, "42"),
            ({"step": 100, "step_range": (0, 500)}, "step 100 / 500"),
            ({"step": 100, "step_range": (0, 500), "show_max": False}, "step 100"),
            ({"time": 0.005, "time_units": "ns"}, "t = 5.00e-03 ns"),
        ],
        ids=[
            "step",
            "time",
            "both",
            "custom-label",
            "empty-label",
            "with-max",
            "no-max",
            "time-units",
        ],
    )
    def test_format_status_text(self, kwargs: dict, expected: str) -> None:
        result = _format_status_text(
            text=None,
            step=kwargs.get("step"),
            time=kwargs.get("time"),
            time_units=kwargs.get("time_units", ""),
            step_range=kwargs.get("step_range"),
            label=kwargs.get("label"),
            show_max=kwargs.get("show_max", True),
        )
        assert result == expected

    def test_no_content_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide text"):
            add_badge(plt.subplots()[1])

    def test_custom_text(self) -> None:
        fig, ax = plt.subplots()
        box = add_badge(ax, "Harris sheet")
        assert box in ax.artists
        plt.close(fig)

    def test_time_as_string(self) -> None:
        result = _format_status_text(
            text=None,
            step=None,
            time="13:34",
            time_units="",
            step_range=None,
        )
        assert result == "t = 13:34"

    def test_custom_text_with_progress(self) -> None:
        fig, ax = plt.subplots()
        box = add_badge(ax, "Loading...", progress=0.7)
        assert box in ax.artists
        plt.close(fig)

    def test_cycle_label(self) -> None:
        result = _format_status_text(
            text=None,
            step=102312,
            time=None,
            time_units="",
            step_range=None,
            label="Cycle",
        )
        assert result == "Cycle 102312"

    def test_custom_bg_color(self) -> None:
        from matplotlib.colors import to_rgba

        fig, ax = plt.subplots()
        box = add_badge(ax, step=1, bg_color="red", bg_alpha=0.5)
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], to_rgba("red")[:3], atol=0.01)
        np.testing.assert_allclose(fc[3], 0.5, atol=0.01)
        plt.close(fig)

    def test_badge_renders_on_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1")
        box = add_badge(ax, step=42)
        assert box in ax.artists
        plt.close(fig)


class TestOverlayVariant:
    """Test the shared variant auto-detection logic once."""

    def test_default_uses_theme_color(self) -> None:
        bg, _fg, alpha = _detect_overlay_defaults(None)
        assert len(bg) == 3
        assert 0 < alpha <= 1

    def test_alt_variant_returns_alt_colors(self) -> None:
        bg, fg, alpha = _detect_overlay_defaults("alt")
        bg_default, fg_default, alpha_default = _detect_overlay_defaults(None)
        # Alt should differ from default in at least alpha
        assert alpha != alpha_default or bg != bg_default

    def test_returns_three_values(self) -> None:
        result = _detect_overlay_defaults("darker")
        assert len(result) == 3
        bg, fg, alpha = result
        assert len(bg) == 3
        assert len(fg) == 3
        assert isinstance(alpha, float)


class TestInsetColorbar:
    @pytest.fixture
    def mesh_on_ax(self) -> tuple:
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((5, 5))
        mesh = ax.pcolormesh(data)
        yield fig, ax, mesh
        plt.close(fig)

    def test_returns_colorbar(self, mesh_on_ax: tuple) -> None:
        from matplotlib.colorbar import Colorbar

        _, ax, mesh = mesh_on_ax
        cb = add_inset_colorbar(ax, mesh, "test")
        assert isinstance(cb, Colorbar)
        plt.close("all")

    @pytest.mark.parametrize(
        "loc",
        ["upper left", "upper right", "lower left", "lower right", "upper center"],
    )
    def test_all_locs(self, mesh_on_ax: tuple, loc: str) -> None:
        _, ax, mesh = mesh_on_ax
        cb = add_inset_colorbar(ax, mesh, loc=loc)
        assert cb is not None
        plt.close("all")

    @pytest.mark.parametrize(
        "plot_fn",
        [
            lambda ds: plot_field_slice(ds, "B1", colorbar="inset"),
            lambda ds: plot_streamlines(ds, "B", colorbar="inset"),
            lambda ds: plot_quiver(ds, "B", colorbar="inset"),
        ],
        ids=["slice", "streamlines", "quiver"],
    )
    def test_via_plot_functions(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        fig, _ = plot_fn(ds_2d)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVectorLegend:
    def test_single_entry(self) -> None:
        fig, ax = plt.subplots()
        entry = LegendEntry(label="B", color="black")
        box = add_legend(ax, entry)
        assert box in ax.artists
        plt.close(fig)

    def test_multiple_entries(self) -> None:
        fig, ax = plt.subplots()
        entries = [
            LegendEntry(label="B", color="black"),
            LegendEntry(label="V", color="red", linewidth=2.0),
        ]
        box = add_legend(ax, entries)
        assert box in ax.artists
        plt.close(fig)


class TestPanelLabel:
    def test_returns_anchored_offsetbox(self) -> None:
        from matplotlib.offsetbox import AnchoredOffsetbox

        fig, ax = plt.subplots()
        box = add_label(ax, "a")
        assert isinstance(box, AnchoredOffsetbox)
        assert box in ax.artists
        plt.close(fig)

    @pytest.mark.parametrize(
        "loc",
        ["upper left", "upper right", "lower left", "lower right", "upper center"],
    )
    def test_all_locs(self, loc: str) -> None:
        fig, ax = plt.subplots()
        box = add_label(ax, "a", loc=loc)
        assert box in ax.artists
        plt.close(fig)


class TestAddContours:
    def test_basic_contour(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import add_contours

        fig, ax = plot_field_slice(ds_2d, "B1")
        cs = add_contours(ax, ds_2d, "P", levels=3)
        assert cs is not None
        plt.close(fig)

    def test_contour_with_labels(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import add_contours

        fig, ax = plot_field_slice(ds_2d, "|B|")
        add_contours(ax, ds_2d, "rho_m", levels=4, labels=True, colors="white")
        plt.close(fig)


class TestLogScale:
    def test_log_scale_positive_field(self, ds_2d: FieldDataset) -> None:
        fig, _ = plot_field_slice(ds_2d, "rho_m", log_scale=True)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotFieldWithVectors:
    def test_streamlines(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_with_vectors

        fig, ax = plot_field_with_vectors(ds_2d, "|B|", "B")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_quiver(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_with_vectors

        fig, _ax = plot_field_with_vectors(
            ds_2d, "rho_m", "B", vector_style="quiver", stride=2
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_with_vectors

        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_field_with_vectors(ds_2d, "B1", "B", ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)


class TestPlotFieldGrid:
    def test_basic_grid(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B1", "|B|", "rho_m", "P"], ncols=2)
        assert isinstance(fig, Figure)
        assert len(axes) == 4
        plt.close(fig)

    def test_single_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B1"])
        assert len(axes) == 1
        plt.close(fig)

    def test_panel_labels_disabled(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, _ = plot_field_grid(ds_2d, ["B1", "P"], panel_labels=False)
        plt.close(fig)


class TestPlotCrossSection:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_cross_section

        fig, (ax_2d, ax_1d) = plot_cross_section(ds_2d, "B1", cut_axis="x")
        assert isinstance(fig, Figure)
        assert isinstance(ax_2d, Axes)
        assert isinstance(ax_1d, Axes)
        plt.close(fig)

    def test_custom_cut_index(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_cross_section

        fig, _ = plot_cross_section(ds_2d, "B1", cut_axis="y", cut_index=2)
        plt.close(fig)

    def test_invalid_cut_axis_rejects(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_cross_section

        with pytest.raises(ValueError, match="cut_axis"):
            plot_cross_section(ds_2d, "B1", cut_axis="z")


class TestComparisonShowError:
    def test_show_error(self, ds_2d: FieldDataset) -> None:
        fig, _axes = plot_comparison(ds_2d, ds_2d, "B1", show_error=True)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotLines:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_lines

        fig, ax = plot_lines(ds_2d, ["B1", "B2", "B3"], axis="x")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 3
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_lines

        fig, ax = plot_lines(ds_2d, ["B1", "B2"], axis="x", labels=["$B_x$", "$B_y$"])
        assert len(ax.lines) == 2
        plt.close(fig)


class TestSetDefaultTheme:
    def test_set_and_get(self) -> None:
        original = get_theme()
        try:
            set_theme("dark")
            assert get_theme().name == "dark"
        finally:
            set_theme(original)

    def test_badge_on_slice(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1", step=42, badge=True)
        artists = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(artists) >= 1
        plt.close(fig)

    def test_save_creates_file(self, ds_2d: FieldDataset, tmp_path: Path) -> None:
        out = tmp_path / "test.png"
        plot_field_slice(ds_2d, "B1", save=str(out))
        assert out.exists()


class TestExtremesMode:
    """Unit tests for _apply_extremes across all four modes."""

    @pytest.fixture
    def mesh(self) -> object:
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((5, 5))
        mesh = ax.pcolormesh(data, vmin=-1, vmax=1)
        yield mesh
        plt.close(fig)

    def test_semi_sets_reduced_alpha(self, mesh: object) -> None:
        from pypic.plotting._colorbar import _apply_extremes

        _apply_extremes(mesh, mode="semi")
        cmap = mesh.cmap  # type: ignore[attr-defined]
        assert cmap._rgba_under[3] == pytest.approx(0.3, abs=0.01)
        assert cmap._rgba_over[3] == pytest.approx(0.3, abs=0.01)

    def test_transparent_matches_background(self, mesh: object) -> None:
        from matplotlib.colors import to_rgba

        from pypic.plotting._colorbar import _apply_extremes

        _apply_extremes(mesh, mode="transparent")
        cmap = mesh.cmap  # type: ignore[attr-defined]
        bg = to_rgba(plt.rcParams.get("axes.facecolor", "white"))
        np.testing.assert_allclose(cmap._rgba_under, bg, atol=0.01)
        np.testing.assert_allclose(cmap._rgba_over, bg, atol=0.01)

    def test_darken_preserves_full_alpha(self, mesh: object) -> None:
        from pypic.plotting._colorbar import _apply_extremes

        _apply_extremes(mesh, mode="darken")
        cmap = mesh.cmap  # type: ignore[attr-defined]
        assert cmap._rgba_under[3] == pytest.approx(1.0, abs=0.01)
        assert cmap._rgba_over[3] == pytest.approx(1.0, abs=0.01)

    def test_none_is_noop(self, mesh: object) -> None:
        from pypic.plotting._colorbar import _apply_extremes

        original_name = mesh.cmap.name  # type: ignore[attr-defined]
        _apply_extremes(mesh, mode=None)
        assert mesh.cmap.name == original_name  # type: ignore[attr-defined]


class TestFileThemes:
    """Theme export, discovery, and file-based lookup."""

    def test_export_creates_files(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting._theme_io import _bundled_theme_dir

        result = export_themes(tmp_path)
        assert result == tmp_path
        bundled = {p.name for p in _bundled_theme_dir().glob("*.toml")}
        assert {f.name for f in tmp_path.glob("*.toml")} == bundled

    def test_export_no_overwrite(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes

        export_themes(tmp_path)
        (tmp_path / "dark.toml").write_text('name = "modified"\n')
        export_themes(tmp_path, overwrite=False)
        assert "modified" in (tmp_path / "dark.toml").read_text()

    def test_export_overwrite(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes

        export_themes(tmp_path)
        (tmp_path / "dark.toml").write_text('name = "modified"\n')
        export_themes(tmp_path, overwrite=True)
        assert "modified" not in (tmp_path / "dark.toml").read_text()

    def test_available_themes(self) -> None:
        from pypic.plotting import available_themes

        themes = available_themes()
        assert "dark" in themes
        assert "light" in themes
        assert len(themes) >= 5

    def test_available_themes_skips_malformed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import available_themes, export_themes

        export_themes(tmp_path)
        (tmp_path / "bad.toml").write_text("not valid {{{toml")
        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        themes = available_themes()
        assert "bad" not in themes

    def test_user_theme_overrides_bundled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting.styles import _resolve_theme_arg

        export_themes(tmp_path)
        text = (tmp_path / "dark.toml").read_text()
        (tmp_path / "dark.toml").write_text(text.replace("#f39c12", "#ff0000"))

        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        assert _resolve_theme_arg("dark").accent_color == "#ff0000"

    def test_resolve_by_file_path(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting.styles import _resolve_theme_arg

        export_themes(tmp_path)
        theme = _resolve_theme_arg(str(tmp_path / "dark.toml"))
        assert theme.name == "dark"

    def test_custom_theme_from_user_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import save_theme

        dark = _resolve_theme_arg("dark")
        custom = dark.customize(name="space-purple", accent_color="#9b59b6")
        save_theme(custom, tmp_path / "space-purple.toml")

        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        resolved = _resolve_theme_arg("space-purple")
        assert resolved.accent_color == "#9b59b6"

    def test_use_theme_with_string(self) -> None:
        original = matplotlib.rcParams["text.color"]
        with use_theme("dark"):
            assert matplotlib.rcParams["text.color"] != original
        assert matplotlib.rcParams["text.color"] == original


class TestLegendEntryNoneColor:
    def test_none_color_creates(self) -> None:
        entry = LegendEntry(label="B", color=None)
        assert entry.color is None

    def test_none_color_renders(self) -> None:
        fig, ax = plt.subplots()
        entry = LegendEntry(label="B", color=None)
        with use_theme("light"):
            add_legend(ax, entry)
        assert len(ax.get_children()) > 0
        plt.close(fig)


class TestOverlayAutoPlacement:
    """Overlays auto-select non-colliding corners."""

    def test_default_corners(self) -> None:
        """Each overlay type picks its preferred corner when all are free."""
        from pypic.plotting._badge import _OCCUPIED_ATTR

        fig, ax = plt.subplots()
        with use_theme("dark"):
            add_label(ax, "a")
            cbar_im = ax.imshow([[0, 1]], aspect="auto")
            add_inset_colorbar(ax, cbar_im, "test")
            add_legend(ax, LegendEntry(label="v", color="white"))
            add_badge(ax, step=1)
        occupied = getattr(ax, _OCCUPIED_ATTR)
        assert "upper left" in occupied      # label
        assert "lower right" in occupied     # colorbar
        assert "lower left" in occupied      # legend
        assert "upper right" in occupied     # badge
        plt.close(fig)

    def test_bumps_to_free_corner(self) -> None:
        """When preferred corner is taken, overlay picks the next free one."""
        from pypic.plotting._badge import _OCCUPIED_ATTR

        fig, ax = plt.subplots()
        with use_theme("dark"):
            # First label takes upper left
            add_label(ax, "a")
            # Second label prefers upper left, but it's taken → upper right
            add_label(ax, "b")
        occupied = getattr(ax, _OCCUPIED_ATTR)
        assert "upper left" in occupied
        assert "upper right" in occupied
        plt.close(fig)

    def test_explicit_loc_overrides(self) -> None:
        """Explicit loc is respected even if it collides."""
        from pypic.plotting._badge import _OCCUPIED_ATTR

        fig, ax = plt.subplots()
        with use_theme("dark"):
            add_label(ax, "a")  # claims upper left
            add_badge(ax, step=1, loc="upper left")  # explicit: same corner
        occupied = getattr(ax, _OCCUPIED_ATTR)
        assert "upper left" in occupied
        plt.close(fig)


class TestFieldGridNewParams:
    def test_shared_vmin_vmax(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, _axes = plot_field_grid(
            ds_2d, ["B1", "B2"], ncols=2, vmin=-2.0, vmax=2.0
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_cmap_dict(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(
            ds_2d,
            ["B1", "rho_m"],
            ncols=2,
            cmap={"B1": "RdBu_r", "rho_m": "inferno"},
        )
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_cmap_string(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, _axes = plot_field_grid(ds_2d, ["B1", "B2"], ncols=2, cmap="viridis")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_log_scale(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_field_grid

        fig, _axes = plot_field_grid(
            ds_2d, ["rho_m", "P"], ncols=2, log_scale=True
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestComparisonNewParams:
    def test_symmetric_false(self, ds_2d: FieldDataset) -> None:
        fig, _axes = plot_comparison(ds_2d, ds_2d, "B1", symmetric=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_alpha(self, ds_2d: FieldDataset) -> None:
        fig, _axes = plot_comparison(ds_2d, ds_2d, "B1", alpha=0.5)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_log_scale(self, ds_2d: FieldDataset) -> None:
        fig, _axes = plot_comparison(ds_2d, ds_2d, "rho_m", log_scale=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_log_scale_with_symmetric_warns(self, ds_2d: FieldDataset) -> None:
        with pytest.warns(UserWarning, match="log_scale=True ignored"):
            fig, _ = plot_comparison(
                ds_2d, ds_2d, "B1", log_scale=True, symmetric=True,
            )
        plt.close(fig)


class TestPlotKymograph:
    def test_basic(self) -> None:
        from pypic.plotting import plot_kymograph

        rng = np.random.default_rng(99)
        values = rng.standard_normal((5, 10))
        coords = np.linspace(0, 10, 10)
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fig, ax = plot_kymograph(values, coords, times)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_symmetric(self) -> None:
        from pypic.plotting import plot_kymograph

        values = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
        coords = np.array([0.0, 1.0, 2.0])
        times = np.array([0.0, 1.0])
        fig, _ax = plot_kymograph(values, coords, times, symmetric=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_log_scale(self) -> None:
        from pypic.plotting import plot_kymograph

        values = np.abs(np.random.default_rng(1).standard_normal((4, 8))) + 0.1
        coords = np.linspace(0, 8, 8)
        times = np.arange(4, dtype=float)
        fig, _ax = plot_kymograph(values, coords, times, log_scale=True)
        plt.close(fig)

    def test_custom_axes(self) -> None:
        from pypic.plotting import plot_kymograph

        fig_ext, ax_ext = plt.subplots()
        values = np.ones((3, 5))
        coords = np.arange(5, dtype=float)
        times = np.arange(3, dtype=float)
        _, ax = plot_kymograph(values, coords, times, ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_colorbar_inset(self) -> None:
        from pypic.plotting import plot_kymograph

        values = np.ones((3, 5))
        coords = np.arange(5, dtype=float)
        times = np.arange(3, dtype=float)
        fig, _ax = plot_kymograph(values, coords, times, colorbar="inset")
        plt.close(fig)

    def test_invalid_shape(self) -> None:
        from pypic.plotting import plot_kymograph

        with pytest.raises(ValueError, match="2D"):
            plot_kymograph(np.ones(10), np.ones(10), np.ones(1))

    def test_labels_and_title(self) -> None:
        from pypic.plotting import plot_kymograph

        values = np.ones((2, 4))
        coords = np.arange(4, dtype=float)
        times = np.arange(2, dtype=float)
        fig, ax = plot_kymograph(
            values, coords, times,
            xlabel="$x$ [$d_i$]", ylabel="$t$ [$\\Omega_i^{-1}$]",
            title="Bz kymograph", label="$B_z$",
        )
        assert ax.get_xlabel() == "$x$ [$d_i$]"
        assert ax.get_title() == "Bz kymograph"
        plt.close(fig)


class TestPlotScatter:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "B1", "B2")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_color_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, _ax = plot_scatter(ds_2d, "B1", "B2", color_field="rho_m")
        plt.close(fig)

    def test_density_mode(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, _ax = plot_scatter(ds_2d, "B1", "B2", density=True)
        plt.close(fig)

    def test_log_axes(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, ax = plot_scatter(ds_2d, "rho_m", "P", log_x=True, log_y=True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_3d_auto_slice(self, ds_3d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, _ax = plot_scatter(ds_3d, "B1", "B2")
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_scatter(ds_2d, "B1", "B2", ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_derived_fields(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, _ax = plot_scatter(ds_2d, "|B|", "beta")
        plt.close(fig)

    def test_density_with_color(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_scatter

        fig, _ax = plot_scatter(
            ds_2d, "B1", "B2", density=True, color_field="rho_m",
        )
        plt.close(fig)


class TestPlotPowerSpectrum:
    def test_basic(self) -> None:
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, ax = plot_power_spectrum(k, power)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_compensated(self) -> None:
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, _ax = plot_power_spectrum(k, power, compensated=5.0 / 3.0)
        plt.close(fig)

    def test_reference_slopes(self) -> None:
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 50)
        power = k ** (-5.0 / 3.0)
        fig, _ax = plot_power_spectrum(
            k, power, reference_slopes=[-5.0 / 3.0, -3.0],
        )
        plt.close(fig)

    def test_custom_axes(self) -> None:
        from pypic.plotting import plot_power_spectrum

        fig_ext, ax_ext = plt.subplots()
        k = np.linspace(0.1, 10, 20)
        _, ax = plot_power_spectrum(k, k**-2, ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_with_label_and_legend(self) -> None:
        from pypic.plotting import plot_power_spectrum

        k = np.linspace(0.1, 10, 30)
        fig, ax = plot_power_spectrum(k, k**-2, label="$B_z$")
        assert ax.get_legend() is not None
        plt.close(fig)


class TestPlotLineComparison:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_line_comparison

        fig, ax = plot_line_comparison([ds_2d, ds_2d], "B1", axis="x")
        assert isinstance(fig, Figure)
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_labels(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_line_comparison

        fig, ax = plot_line_comparison(
            [ds_2d, ds_2d], "B1", axis="x", labels=["run A", "run B"],
        )
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_line_comparison

        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_line_comparison(
            [ds_2d, ds_2d], "B1", axis="x", ax=ax_ext,
        )
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_line_comparison

        fig, _ax = plot_line_comparison([ds_2d, ds_2d], "|B|", axis="x")
        plt.close(fig)
