"""Tests for pypic.plotting module."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pypic.plotting import (  # noqa: E402
    DARK,
    DEFAULT,
    LIGHT,
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
from pypic.plotting._badge import (  # noqa: E402
    _detect_overlay_defaults,
    _format_status_text,
)
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
            pytest.param({"theme": DARK}, id="dark-theme"),
            pytest.param({"symmetric": False}, id="symmetric-override"),
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
    def test_returns_figure_and_axes_dict(
        self, ds_2d: FieldDataset
    ) -> None:
        fig, axes = plot_comparison(ds_2d, ds_2d, "B1")
        assert isinstance(fig, Figure)
        assert set(axes) == {"a", "b", "diff"}
        plt.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        fig, axes = plot_comparison(
            ds_2d, ds_2d, "B1", labels=("Run1", "Run2")
        )
        assert axes["a"].get_title() == "Run1"
        assert axes["b"].get_title() == "Run2"
        plt.close(fig)

    def test_3d_and_derived(self, ds_3d: FieldDataset) -> None:
        fig, _ = plot_comparison(ds_3d, ds_3d, "|B|")
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
            "magnitude", "component", "density", "charge-density",
            "beta", "all-positive-fallback", "mixed-sign-fallback",
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
        assert symmetric_clim(np.array([np.nan, np.nan])) == (0.0, 0.0)


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
    @pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
    def test_rcparams_valid(self, theme: PlotTheme) -> None:
        valid_keys = set(matplotlib.rcParams)
        for key in theme.rcparams:
            assert key in valid_keys, f"{key!r} not a valid rcParam"

    def test_default_is_light(self) -> None:
        assert DEFAULT is LIGHT

    def test_use_theme_restores_state(self) -> None:
        original = matplotlib.rcParams["text.color"]
        with use_theme(DARK):
            assert matplotlib.rcParams["text.color"] != original
        assert matplotlib.rcParams["text.color"] == original


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
        fig, ax = plot_time_series(
            tabular, ["total_energy", "kinetic_energy"]
        )
        assert len(ax.lines) == 2
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_custom_x_column(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(
            tabular, "total_energy", x_column="cycle"
        )
        assert ax.get_xlabel() == "cycle"
        plt.close(fig)

    def test_no_legend(self, tabular: TabularData) -> None:
        fig, ax = plot_time_series(
            tabular, "total_energy", legend=False
        )
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
    def test_custom_axes(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_fn(ds_2d, "B", ax=ax_ext)  # type: ignore[operator]
        assert ax is ax_ext
        plt.close(fig_ext)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_uniform_color(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        fig, _ = plot_fn(ds_2d, "B", color="black")  # type: ignore[operator]
        assert isinstance(fig, Figure)
        plt.close(fig)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_auto_legend(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
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
            (plot_streamlines, {"theme": DARK}),
            (plot_streamlines, {"colorbar": False}),
            (plot_streamlines, {"alpha": 0.4, "color": "black"}),
            (plot_quiver, {"stride": 2}),
            (plot_quiver, {"stride": (2, 3)}),
            (plot_quiver, {"theme": DARK}),
            (plot_quiver, {"colorbar": False}),
        ],
        ids=[
            "stream-color_field", "stream-dark", "stream-no-cb",
            "stream-alpha", "quiver-stride", "quiver-stride-tuple",
            "quiver-dark", "quiver-no-cb",
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
            "step", "time", "both", "custom-label", "empty-label",
            "with-max", "no-max", "time-units",
        ],
    )
    def test_format_status_text(self, kwargs: dict, expected: str) -> None:
        result = _format_status_text(
            step=kwargs.get("step"),
            time=kwargs.get("time"),
            time_units=kwargs.get("time_units", ""),
            step_range=kwargs.get("step_range"),
            label=kwargs.get("label"),
            show_max=kwargs.get("show_max", True),
        )
        assert result == expected

    def test_no_step_no_time_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            add_status_badge(plt.subplots()[1])

    def test_custom_bg_color(self) -> None:
        from matplotlib.colors import to_rgba

        fig, ax = plt.subplots()
        box = add_status_badge(ax, step=1, bg_color="red", bg_alpha=0.5)
        fc = box.patch.get_facecolor()
        np.testing.assert_allclose(fc[:3], to_rgba("red")[:3], atol=0.01)
        np.testing.assert_allclose(fc[3], 0.5, atol=0.01)
        plt.close(fig)

    def test_badge_renders_on_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B1")
        box = add_status_badge(ax, step=42)
        assert box in ax.artists
        plt.close(fig)


class TestOverlayShade:
    """Test the shared shade auto-detection logic once."""

    @pytest.mark.parametrize(
        ("rc_bg", "shade", "expect_dark_bg"),
        [
            ("white", None, False),
            ("#1e1e1e", None, True),
            ("white", "darker", True),
            ("white", "lighter", False),
        ],
        ids=["auto-light", "auto-dark", "explicit-darker", "explicit-lighter"],
    )
    def test_detect_overlay_defaults(
        self, rc_bg: str, shade: str | None, expect_dark_bg: bool
    ) -> None:
        import matplotlib as mpl

        with mpl.rc_context({"axes.facecolor": rc_bg, "text.color": "#e0e0e0"}):
            bg, _fg = _detect_overlay_defaults(shade)
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            if expect_dark_bg:
                assert luminance < 0.5, f"expected dark bg, got {bg}"
            else:
                assert luminance >= 0.5, f"expected light bg, got {bg}"


class TestInsetColorbar:
    @pytest.fixture
    def mesh_on_ax(self) -> tuple:
        fig, ax = plt.subplots()
        data = np.random.default_rng(0).standard_normal((5, 5))
        mesh = ax.pcolormesh(data)
        return fig, ax, mesh

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
    def test_via_plot_functions(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        fig, _ = plot_fn(ds_2d)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVectorLegend:
    def test_single_entry(self) -> None:
        fig, ax = plt.subplots()
        entry = VectorLegendEntry(label="B", color="black")
        box = add_vector_legend(ax, entry)
        assert box in ax.artists
        plt.close(fig)

    def test_multiple_entries(self) -> None:
        fig, ax = plt.subplots()
        entries = [
            VectorLegendEntry(label="B", color="black"),
            VectorLegendEntry(label="V", color="red", linewidth=2.0),
        ]
        box = add_vector_legend(ax, entries)
        assert box in ax.artists
        plt.close(fig)


class TestPanelLabel:
    def test_returns_anchored_offsetbox(self) -> None:
        from matplotlib.offsetbox import AnchoredOffsetbox

        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a")
        assert isinstance(box, AnchoredOffsetbox)
        assert box in ax.artists
        plt.close(fig)

    @pytest.mark.parametrize(
        "loc",
        ["upper left", "upper right", "lower left", "lower right", "upper center"],
    )
    def test_all_locs(self, loc: str) -> None:
        fig, ax = plt.subplots()
        box = add_panel_label(ax, "a", loc=loc)
        assert box in ax.artists
        plt.close(fig)
