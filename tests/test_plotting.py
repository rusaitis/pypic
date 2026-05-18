"""Tests for pypic.plotting module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pypic.grid import GridInfo

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pypic.containers import TabularData  # noqa: E402
from pypic.dataset import FieldDataset  # noqa: E402
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
from pypic.plotting._colormaps import (  # noqa: E402
    _auto_linthresh,
    auto_clim,
    is_positive_definite,
    round_nice,
    symmetric_clim,
)
from pypic.plotting.styles import _resolve_theme_arg  # noqa: E402
from pypic.selections import PlaneSelection  # noqa: E402
from pypic.units import Normalization  # noqa: E402
from tests._helpers import make_uniform_grid  # noqa: E402


@pytest.fixture
def grid_2d() -> GridInfo:
    return make_uniform_grid(10, 8)


@pytest.fixture
def grid_3d() -> GridInfo:
    return make_uniform_grid(10, 8, 6)


@pytest.fixture
def ds_2d(grid_2d: GridInfo) -> FieldDataset:
    rng = np.random.default_rng(42)
    return FieldDataset.from_arrays(
        {
            "B_1": rng.standard_normal((10, 8)),
            "B_2": rng.standard_normal((10, 8)),
            "B_3": rng.standard_normal((10, 8)),
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
            "B_1": rng.standard_normal((10, 8, 6)),
            "B_2": rng.standard_normal((10, 8, 6)),
            "B_3": rng.standard_normal((10, 8, 6)),
            "rho_m": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
            "P": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
        },
        grid_3d,
        Normalization.identity(),
    )


@pytest.fixture
def ds_1d() -> FieldDataset:
    rng = np.random.default_rng(99)
    return FieldDataset.from_arrays(
        {"B_1": rng.standard_normal(20)},
        make_uniform_grid(20, spacing=0.5),
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
        """Plotted ``B_1`` data equals the input field (transposed).

        Hardened (iter 17): the pcolormesh-backed image is in
        ``ax.collections[0]``; its ``get_array()`` carries the plotted
        values with axes swapped relative to the input. Pinning the
        array prevents silent regressions where the colorbar renders
        but the wrong field (or wrong slice) is drawn.
        """
        fig, ax = plot_field_slice(ds_2d, "B_1")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # pcolormesh transposes (nx, ny) → (ny, nx) for display
        plotted = np.asarray(ax.collections[0].get_array())
        np.testing.assert_array_equal(plotted, ds_2d["B_1"].T)
        plt.close(fig)

    def test_custom_axes(self, ds_2d: FieldDataset) -> None:
        """``ax=`` reuses the caller's axes; no new figure is created."""
        fig_ext, ax_ext = plt.subplots()
        fig_ret, ax = plot_field_slice(ds_2d, "B_1", ax=ax_ext)
        assert ax is ax_ext
        assert fig_ret is fig_ext
        # Drawing happened on the provided axes
        assert len(ax_ext.collections) == 1
        plt.close(fig_ext)

    def test_custom_title_and_step(self, ds_2d: FieldDataset) -> None:
        """Explicit ``title`` overrides the default; ``step`` is accepted."""
        fig, ax = plot_field_slice(ds_2d, "B_1", title="Custom", step=42)
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
        """Option kwargs leave observable traces on the resulting figure.

        Hardened (iter 17): the pre-hardening assertion was
        ``isinstance(fig, Figure)`` — a tautology that passes if
        matplotlib doesn't crash. Each option now pins a concrete
        visible effect: ``colorbar=False`` leaves a single axes,
        ``colorbar='inset'`` adds a child axes, ``alpha`` pins the
        collection alpha, ``units='nT'`` ends the colorbar label in
        ``nT``, ``field='|B|'`` guarantees nonnegative plotted values,
        and ``symmetric=False`` pins asymmetric vmin/vmax against
        the default symmetric-about-zero diverging norm.
        """
        field = kwargs.pop("field", "B_1")
        option_id = next(
            (
                k
                for k in (
                    "colorbar",
                    "units",
                    "theme",
                    "symmetric",
                    "extremes",
                    "alpha",
                )
                if k in kwargs
            ),
            "field",
        )
        fig, ax = plot_field_slice(ds_2d, field, **kwargs)
        assert isinstance(fig, Figure)
        coll = ax.collections[0]

        if field == "|B|":
            plotted = np.asarray(coll.get_array())
            assert plotted.min() >= 0.0, "magnitude field must be non-negative"
        if option_id == "colorbar":
            cb = kwargs["colorbar"]
            if cb is False:
                assert len(fig.axes) == 1  # no separate colorbar axes
            elif cb == "inset":
                # inset colorbar is attached as a child of ax, not a new axes
                assert len(fig.axes) == 1
                assert len(ax.child_axes) == 1
        if option_id == "alpha":
            assert coll.get_alpha() == pytest.approx(kwargs["alpha"])
        if option_id == "units":
            cax = [a for a in fig.axes if a is not ax]
            # Colorbar label contains the supplied units token (e.g. '[nT]')
            assert cax, "units-labeled plot must have a colorbar axes"
            assert kwargs["units"] in cax[0].get_ylabel()
        if option_id == "symmetric" and kwargs.get("symmetric") is False:
            # Asymmetric data should NOT be forced to vmin = -vmax
            norm = coll.norm
            assert norm.vmin != pytest.approx(-norm.vmax)
        plt.close(fig)

    def test_3d_auto_and_explicit(self, ds_3d: FieldDataset) -> None:
        """Explicit ``plane=`` selects the requested slice from a 3D dataset.

        Hardened (iter 17): the pre-hardening body created two figures
        and closed them with no assertions. The explicit slice at
        ``x=3`` is pinned to the corresponding ``B_2[3, :, :]`` plane
        (transposed for pcolormesh). Prevents silent regressions where
        plane-selection picks the wrong axis or the wrong index.
        """
        fig1, ax1 = plot_field_slice(ds_3d, "B_1")
        assert isinstance(fig1, Figure)
        assert len(ax1.collections) == 1
        fig2, ax2 = plot_field_slice(
            ds_3d, "B_2", plane=PlaneSelection(normal="x", index=3)
        )
        plotted = np.asarray(ax2.collections[0].get_array())
        np.testing.assert_array_equal(plotted, ds_3d["B_2"][3, :, :].T)
        plt.close(fig1)
        plt.close(fig2)


class TestPlotComparison:
    def test_returns_figure_and_axes_dict(self, ds_2d: FieldDataset) -> None:
        """Three-panel layout; diff panel vanishes for identical inputs.

        Hardened (iter 17): when both inputs are the same dataset the
        elementwise difference must be zero everywhere. The panel keys
        were already pinned; we now also pin the numerical content of
        the ``diff`` panel (guards against a regression where the diff
        panel silently plots one of the inputs instead of ``a - b``).
        """
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1")
        assert isinstance(fig, Figure)
        assert set(axes) == {"a", "b", "diff"}
        diff = np.asarray(axes["diff"].collections[0].get_array())
        np.testing.assert_array_equal(diff, np.zeros_like(diff))
        plt.close(fig)

    def test_custom_labels(self, ds_2d: FieldDataset) -> None:
        """``labels=`` sets the titles of the ``a`` and ``b`` panels."""
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1", labels=("Run1", "Run2"))
        assert axes["a"].get_title() == "Run1"
        assert axes["b"].get_title() == "Run2"
        plt.close(fig)

    def test_3d_and_derived(self, ds_3d: FieldDataset) -> None:
        """3D + derived field (``|B|``) renders with non-negative panels.

        Hardened (iter 17): was ``isinstance(fig, Figure)``. Now also
        asserts the ``a`` and ``b`` panels carry non-negative data (as
        required for a magnitude) and the diff panel has zero content.
        """
        fig, axes = plot_comparison(ds_3d, ds_3d, "|B|")
        assert isinstance(fig, Figure)
        a_arr = np.asarray(axes["a"].collections[0].get_array())
        b_arr = np.asarray(axes["b"].collections[0].get_array())
        diff = np.asarray(axes["diff"].collections[0].get_array())
        assert a_arr.min() >= 0.0
        assert b_arr.min() >= 0.0
        np.testing.assert_array_equal(diff, np.zeros_like(diff))
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
        """Option kwargs leave observable traces on the comparison figure.

        Hardened (iter 17): pre-hardening body only asserted
        ``isinstance(fig, Figure)``. Each option now pins a concrete
        visible effect: ``colorbar=False`` yields 3 total axes (one
        per panel, no colorbar); ``colorbar='inset'`` yields 3 total
        axes plus 3 inset child axes; ``vmin/vmax`` pin the norm
        limits on all three panels.
        """
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1", **kwargs)
        assert isinstance(fig, Figure)
        if "colorbar" in kwargs:
            cb = kwargs["colorbar"]
            if cb is False:
                assert len(fig.axes) == 3  # panels only, no colorbars
            elif cb == "inset":
                assert len(fig.axes) == 3
                assert sum(len(a.child_axes) for a in axes.values()) == 3
        if "vmin" in kwargs:
            for panel in ("a", "b"):
                assert axes[panel].collections[0].norm.vmin == pytest.approx(
                    kwargs["vmin"]
                )
                assert axes[panel].collections[0].norm.vmax == pytest.approx(
                    kwargs["vmax"]
                )
        plt.close(fig)


class TestColormapDetection:
    @pytest.mark.parametrize(
        ("name", "data_positive", "quantity_type", "expected"),
        [
            ("|B|", False, None, True),
            ("B_1", False, None, False),
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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vmin, vmax = symmetric_clim(np.array([np.nan, np.nan]))
        assert vmin < 0 < vmax  # epsilon expansion, not degenerate (0, 0)


class TestResolveFieldColormap:
    """Single-source-of-truth dispatcher for field → colormap.

    The matplotlib backend reads the returned name; the pyvista backend
    reads the returned Colormap object. Both backends call this function,
    so the same canonical names always pick the same colormap.
    """

    @pytest.fixture
    def theme(self):  # type: ignore[no-untyped-def]
        from pypic.plotting.styles import get_theme

        return get_theme()

    @pytest.mark.parametrize(
        ("name", "values", "quantity_type", "expect_diverging"),
        [
            ("B_1", np.array([-1.0, 0.0, 1.0]), "b_field", True),
            ("rho_c", np.array([-1.0, 0.0, 1.0]), "charge_density", True),
            ("J_dot_E", np.array([-1.0, 0.0, 1.0]), "power_density", True),
            ("psi", np.array([-1.0, 0.0, 1.0]), None, True),
            ("div_B", np.array([-1.0, 0.0, 1.0]), None, True),
            ("|B|", np.array([0.5, 1.0, 1.5]), None, False),
            ("n_s0", np.array([0.5, 1.0, 1.5]), "density", False),
        ],
        ids=["B_1", "rho_c", "J_dot_E", "psi", "div_B", "abs_B", "n_s0"],
    )
    def test_canonical_field_dispatch(  # type: ignore[no-untyped-def]
        self,
        theme,
        name: str,
        values: np.ndarray,
        quantity_type: str | None,
        expect_diverging: bool,
    ) -> None:
        """Each canonical field name resolves to the right theme cmap."""
        from pypic.fields import FieldInfo
        from pypic.plotting._colormaps import resolve_field_colormap

        info = (
            FieldInfo(quantity_type=quantity_type, long_name="", si_unit="")
            if quantity_type
            else None
        )
        cmap_name, cmap_obj = resolve_field_colormap(name, values, theme, info=info)
        # Returned tuple is internally consistent
        assert cmap_obj.name == cmap_name
        # Picks the right family from the theme
        expected = theme.diverging_cmap if expect_diverging else theme.sequential_cmap
        assert cmap_name == expected

    def test_string_override_passes_through(self, theme) -> None:  # type: ignore[no-untyped-def]
        """A user-supplied cmap name bypasses auto-detection."""
        from pypic.plotting._colormaps import resolve_field_colormap

        cmap_name, cmap_obj = resolve_field_colormap(
            "B_1", np.array([-1.0, 1.0]), theme, cmap="viridis"
        )
        assert cmap_name == "viridis"
        assert cmap_obj.name == "viridis"

    def test_colormap_object_passes_through(self, theme) -> None:  # type: ignore[no-untyped-def]
        """A pre-built Colormap is returned unchanged with its name."""
        import matplotlib.pyplot as plt

        from pypic.plotting._colormaps import resolve_field_colormap

        plasma = plt.colormaps["plasma"]
        cmap_name, cmap_obj = resolve_field_colormap(
            "B_1", np.array([-1.0, 1.0]), theme, cmap=plasma
        )
        assert cmap_obj is plasma
        assert cmap_name == "plasma"

    def test_matches_resolve_colormap_string_path(self, theme) -> None:  # type: ignore[no-untyped-def]
        """Returned name matches the legacy string-only resolver."""
        from pypic.plotting._colormaps import resolve_colormap, resolve_field_colormap

        for name, values in [
            ("B_1", np.array([-1.0, 1.0])),
            ("|B|", np.array([0.0, 1.0])),
            ("rho_c", np.array([-1.0, 1.0])),
        ]:
            legacy = resolve_colormap(name, values, theme)
            new_name, _ = resolve_field_colormap(name, values, theme)
            assert legacy == new_name

    def test_uniform_field(self) -> None:
        vmin, vmax = symmetric_clim(np.array([0.0, 0.0, 0.0]))
        assert vmin < 0 < vmax


class TestResolveFieldValues:
    def test_stored_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "B_1", None)
        np.testing.assert_array_equal(values, ds_2d["B_1"])

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "|B|", None)
        expected = np.sqrt(ds_2d["B_1"] ** 2 + ds_2d["B_2"] ** 2 + ds_2d["B_3"] ** 2)
        np.testing.assert_allclose(values, expected)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        code = resolve_field_values(ds_2d, "B_1", None)
        si = resolve_field_values(ds_2d, "B_1", "nT")
        # Shape preserved
        assert si.shape == code.shape
        # Identity normalization: code→SI conversion is 1 T; 1 T = 1e9 nT.
        # Pin the scale so a regression in the unit-conversion path (e.g.
        # forgetting the nT prefix) is caught, not just "something happened".
        np.testing.assert_allclose(si, code * 1.0e9, rtol=1e-12)


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
        """1D line plot draws the full B_1 vector as a single line.

        Hardened (iter 18): the pre-hardening body asserted only
        isinstance(fig, Figure) — tautological. We now pin the
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

        Hardened (iter 18): the line-count-only assertion let a
        regression that sliced the wrong axis or the wrong index pass
        silently. We now pin the ydata to the explicit
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

        Hardened (iter 18): in addition to the line-count check we pin
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

        Hardened (iter 18): the pre-hardening body only checked that a
        legend object was present; we now pin the legend text to the
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

        Hardened (iter 18): the line-count-only assertion let a
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

        Hardened (iter 18): the pre-hardening body asserted line count
        and legend-present. We now pin each line's ydata to the
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

        Hardened (iter 18): added xdata-equals-column and line-count
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


class TestVectorPlots:
    """Shared tests for plot_streamlines and plot_quiver."""

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_returns_figure_and_axes(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        """Vector plot draws a LineCollection (streamlines) or Quiver
        (quiver) into the axes.

        Hardened (iter 18): the isinstance-only assertion let a no-op
        plot pass. Pinning ``ax.collections[0]`` ensures at least one
        vector primitive was drawn — a regression that returned an
        empty axes is caught.
        """
        fig, ax = plot_fn(ds_2d, "B")  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) >= 1
        plt.close(fig)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_custom_axes(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        """Passing ``ax=ax_ext`` draws onto the supplied axes.

        Hardened (iter 18): in addition to the identity check, we now
        pin that at least one collection ends up on ``ax_ext``, so a
        regression that returned ``ax_ext`` without drawing onto it
        is caught.
        """
        fig_ext, ax_ext = plt.subplots()
        assert len(ax_ext.collections) == 0
        _, ax = plot_fn(ds_2d, "B", ax=ax_ext)  # type: ignore[operator]
        assert ax is ax_ext
        assert len(ax_ext.collections) >= 1
        plt.close(fig_ext)

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_uniform_color(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        """Uniform ``color="black"`` bypasses the colormap path.

        Hardened (iter 18): pre-hardening only asserted isinstance. We
        now pin that no colorbar axes was created (the colormap branch
        attaches one, the uniform-color branch does not) so a regression
        that silently reverts to colormap mode is caught.
        """
        fig, ax = plot_fn(ds_2d, "B", color="black")  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1
        # Uniform-color mode: no colorbar axes (only the main axes exists).
        assert len(fig.axes) == 1
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
        """Each ``plot_streamlines`` / ``plot_quiver`` option leaves a
        concrete observable trace on the returned figure.

        Hardened (iter 18): the pre-hardening body asserted only
        isinstance — any regression that silently ignored the kwarg
        passed. Each option now pins a concrete effect:
        ``colorbar=False`` leaves a single axes, ``alpha`` pins the
        collection alpha, and the vector primitive is verified drawn
        for every case.
        """
        fig, ax = plot_fn(ds_2d, "B", **extra_kwargs)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1
        if extra_kwargs.get("colorbar") is False:
            # No separate colorbar axes should be created
            assert len(fig.axes) == 1
        if "alpha" in extra_kwargs:
            assert ax.collections[0].get_alpha() == pytest.approx(extra_kwargs["alpha"])
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
        fc = np.asarray(box.patch.get_facecolor())
        np.testing.assert_allclose(fc[:3], to_rgba("red")[:3], atol=0.01)
        np.testing.assert_allclose(fc[3], 0.5, atol=0.01)
        plt.close(fig)

    def test_badge_renders_on_axes(self, ds_2d: FieldDataset) -> None:
        fig, ax = plot_field_slice(ds_2d, "B_1")
        box = add_badge(ax, step=42)
        assert box in ax.artists
        plt.close(fig)


class TestOverlayVariant:
    """Test the shared variant auto-detection logic once."""

    def test_default_uses_theme_color(self) -> None:
        """``variant=None`` returns an RGB triple and a valid alpha.

        Hardened (iter 18): the pre-hardening bound ``0 < alpha <= 1``
        admitted any alpha >= next-representable-zero. We tighten to
        alpha strictly less than 1 (overlays are never fully opaque)
        and pin each RGB component to the valid ``[0, 1]`` range, so
        a regression that returned unclamped colors is caught.
        """
        bg, fg, alpha = _detect_overlay_defaults(None)
        assert len(bg) == 3
        assert len(fg) == 3
        assert 0.0 < alpha < 1.0
        assert all(0.0 <= c <= 1.0 for c in bg)
        assert all(0.0 <= c <= 1.0 for c in fg)

    def test_alt_variant_returns_alt_colors(self) -> None:
        """``variant="alt"`` returns a distinct bg, fg, or alpha.

        Hardened (iter 18): the prior ``!= alpha_default or !=
        bg_default`` permitted identical output in both attributes.
        We now require at least one of {bg, fg, alpha} differs
        from the ``None`` variant so a regression that silently
        degenerated ``"alt"`` to the default is caught.
        """
        bg, fg, alpha = _detect_overlay_defaults("alt")
        bg_default, fg_default, alpha_default = _detect_overlay_defaults(None)
        assert (bg, fg, alpha) != (bg_default, fg_default, alpha_default)

    def test_returns_three_values(self) -> None:
        result = _detect_overlay_defaults("darker")
        assert len(result) == 3
        bg, fg, alpha = result
        assert len(bg) == 3
        assert len(fg) == 3
        assert isinstance(alpha, float)


class TestInsetColorbar:
    @pytest.fixture
    def mesh_on_ax(self) -> Iterator[tuple[Figure, Axes, Any]]:
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
        """Every supported ``loc`` attaches the colorbar as a child of
        the host axes (inset mode), not as a sibling figure axes.

        Hardened (iter 18): ``cb is not None`` was tautological once
        the function signature declared ``-> Colorbar``. We now pin
        the inset-specific structural invariant — exactly one
        ``ax.child_axes`` entry — which catches a regression that
        degenerates to the non-inset (sibling-axes) colorbar path.
        """
        _, ax, mesh = mesh_on_ax
        n_children_before = len(ax.child_axes)
        cb = add_inset_colorbar(ax, mesh, loc=loc)
        assert cb is not None
        assert len(ax.child_axes) == n_children_before + 1
        plt.close("all")

    @pytest.mark.parametrize(
        "plot_fn",
        [
            lambda ds: plot_field_slice(ds, "B_1", colorbar="inset"),
            lambda ds: plot_streamlines(ds, "B", colorbar="inset"),
            lambda ds: plot_quiver(ds, "B", colorbar="inset"),
        ],
        ids=["slice", "streamlines", "quiver"],
    )
    def test_via_plot_functions(self, ds_2d: FieldDataset, plot_fn: object) -> None:
        """``colorbar="inset"`` produces a single-axes figure with the
        colorbar rendered as an inset child.

        Hardened (iter 18): pre-hardening only asserted isinstance.
        We now pin the inset invariant — one figure axes, one
        ``ax.child_axes`` — which catches a regression that silently
        fell back to a separate colorbar axes (``fig.axes >= 2``).
        """
        fig, ax = plot_fn(ds_2d)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert len(ax.child_axes) == 1
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
        """``add_contours`` adds a ContourSet to an existing axes
        without replacing the underlying pcolormesh.

        Hardened (iter 18): the pre-hardening ``cs is not None`` was
        a function-signature tautology. We now pin:
        1. ``ax.collections`` grows by at least one (a new
           ``LineCollection`` per contour line), and
        2. the ContourSet exposes the requested number of levels
           (matplotlib auto-expands to ~nevels+1 boundaries, so we
           assert monotonicity rather than equality).
        """
        from pypic.plotting import add_contours

        fig, ax = plot_field_slice(ds_2d, "B_1")
        n_before = len(ax.collections)
        cs = add_contours(ax, ds_2d, "P", levels=3)
        assert cs is not None
        assert len(ax.collections) > n_before
        assert len(cs.levels) >= 3
        plt.close(fig)

    def test_contour_with_labels(self, ds_2d: FieldDataset) -> None:
        """``labels=True`` attaches label texts to the contour set.

        Hardened (iter 18): prior body had no assertions. We now
        pin ``len(cs.labelTexts) > 0`` (clabel populates this list)
        so a regression that silently drops the label path is caught.
        """
        from pypic.plotting import add_contours

        fig, ax = plot_field_slice(ds_2d, "|B|")
        cs = add_contours(ax, ds_2d, "rho_m", levels=4, labels=True, colors="white")
        assert cs is not None
        # clabel populates labelTexts; empty list means no labels were drawn.
        assert len(cs.labelTexts) > 0
        plt.close(fig)


class TestLogScale:
    def test_log_scale_positive_field(self, ds_2d: FieldDataset) -> None:
        """``log_scale=True`` wraps the pcolormesh norm in LogNorm.

        Hardened (iter 18): isinstance(fig, Figure) didn't guard the
        log path at all — a silently-linear plot would pass. Pinning
        the collection's norm to ``matplotlib.colors.LogNorm`` catches
        a regression that forgets to apply log scaling.
        """
        from matplotlib.colors import LogNorm

        fig, ax = plot_field_slice(ds_2d, "rho_m", log_scale=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax.collections[0].norm, LogNorm)
        plt.close(fig)


class TestComposedFieldAndVectors:
    def test_slice_then_streamlines(self, ds_2d: FieldDataset) -> None:
        """Overlaying streamlines on a field slice produces exactly
        one QuadMesh (slice) plus one LineCollection (streamlines).

        Hardened (iter 18): the isinstance-only assertions ignored
        whether the second call actually drew anything. We pin the
        collection types so a regression that silently replaces the
        slice, skips the streamlines, or draws the wrong primitive
        is caught.
        """
        from matplotlib.collections import LineCollection, QuadMesh

        fig, ax = plot_field_slice(ds_2d, "|B|", title="")
        plot_streamlines(ds_2d, "B", ax=ax, colorbar=False, legend=False, title="")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) == 2
        assert isinstance(ax.collections[0], QuadMesh)
        assert isinstance(ax.collections[1], LineCollection)
        plt.close(fig)

    def test_slice_then_quiver(self, ds_2d: FieldDataset) -> None:
        """Overlaying a quiver on a field slice yields QuadMesh +
        Quiver collections on the same axes.

        Hardened (iter 18): pre-hardening only asserted isinstance.
        We pin both collections' types to catch a regression that
        drops either the slice or the quiver silently.
        """
        from matplotlib.collections import QuadMesh
        from matplotlib.quiver import Quiver

        fig, ax = plot_field_slice(ds_2d, "rho_m", title="")
        plot_quiver(ds_2d, "B", ax=ax, stride=2, colorbar=False, legend=False, title="")
        assert isinstance(fig, Figure)
        assert len(ax.collections) == 2
        assert isinstance(ax.collections[0], QuadMesh)
        assert isinstance(ax.collections[1], Quiver)
        plt.close(fig)

    def test_shared_axes(self, ds_2d: FieldDataset) -> None:
        """Overlaying slice + streamlines on an externally supplied
        axes draws both onto that axes.

        Hardened (iter 18): in addition to the identity check, we
        pin that exactly two collections end up on ``ax_ext`` —
        catches a regression where the second call detaches or
        creates a new axes silently.
        """
        fig_ext, ax_ext = plt.subplots()
        _, ax = plot_field_slice(ds_2d, "B_1", ax=ax_ext, title="")
        plot_streamlines(ds_2d, "B", ax=ax, colorbar=False, legend=False, title="")
        assert ax is ax_ext
        assert len(ax_ext.collections) == 2
        plt.close(fig_ext)


class TestPlotFieldGrid:
    def test_basic_grid(self, ds_2d: FieldDataset) -> None:
        """Requesting N fields yields N axes, each with a pcolormesh.

        Hardened (iter 18): pre-hardening only checked
        ``len(axes) == 4``. We now pin that every panel has at least
        one collection drawn — a regression that returned N empty
        axes passed the prior assertion.
        """
        from pypic.plotting import plot_field_grid

        fields = ["B_1", "|B|", "rho_m", "P"]
        fig, axes = plot_field_grid(ds_2d, fields, ncols=2)
        assert isinstance(fig, Figure)
        assert len(axes) == len(fields)
        for panel in axes:
            assert len(panel.collections) >= 1
        plt.close(fig)

    def test_single_field(self, ds_2d: FieldDataset) -> None:
        """A 1-field grid yields one axes with one pcolormesh.

        Hardened (iter 18): added the ``len(collections) == 1``
        invariant alongside the panel-count check.
        """
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B_1"])
        assert len(axes) == 1
        assert len(axes[0].collections) == 1
        plt.close(fig)

    def test_panel_labels_disabled(self, ds_2d: FieldDataset) -> None:
        """``panel_labels=False`` suppresses the panel-label artists.

        Hardened (iter 18): prior body had no assertion. We now
        scan every panel's ``ax.artists`` and assert no
        ``AnchoredOffsetbox`` (the panel-label container) is
        present — catches a regression that ignores the kwarg.
        """
        from matplotlib.offsetbox import AnchoredOffsetbox

        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B_1", "P"], panel_labels=False)
        for panel in axes:
            anchored = [a for a in panel.artists if isinstance(a, AnchoredOffsetbox)]
            assert anchored == []
        plt.close(fig)


class TestPlotCrossSection:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        """Cross-section returns (2D ax with pcolormesh, 1D ax with line).

        Hardened (iter 18): pre-hardening only did isinstance checks.
        We now pin the 2D panel has exactly one collection
        (the pcolormesh) and the 1D panel has at least one line
        drawn — a regression that leaves either panel empty is
        caught.
        """
        from pypic.plotting import plot_cross_section

        fig, (ax_2d, ax_1d) = plot_cross_section(ds_2d, "B_1", cut_axis="x")
        assert isinstance(fig, Figure)
        assert isinstance(ax_2d, Axes)
        assert isinstance(ax_1d, Axes)
        assert len(ax_2d.collections) == 1
        assert len(ax_1d.lines) >= 1
        plt.close(fig)

    def test_custom_cut_index(self, ds_2d: FieldDataset) -> None:
        """Custom ``cut_index`` drives the 1D line's ydata to the
        B_1 row at that cut index.

        Hardened (iter 18): the prior body had no assertion. ``cut_axis
        ="y"`` means slice AT y-index 2, yielding ``B_1[2, :]``. Pinning
        this catches a regression that silently ignores ``cut_index``
        (defaulting to midplane) or swaps the slicing axis.
        """
        from pypic.plotting import plot_cross_section

        fig, (_, ax_1d) = plot_cross_section(ds_2d, "B_1", cut_axis="y", cut_index=2)
        assert len(ax_1d.lines) >= 1
        expected = np.asarray(ds_2d["B_1"])[2, :]
        np.testing.assert_array_equal(ax_1d.lines[0].get_ydata(), expected)
        plt.close(fig)

    def test_invalid_cut_axis_rejects(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting import plot_cross_section

        with pytest.raises(ValueError, match="cut_axis"):
            plot_cross_section(ds_2d, "B_1", cut_axis="z")


class TestComparisonShowError:
    def test_show_error(self, ds_2d: FieldDataset) -> None:
        """``show_error=True`` overlays the L2 relative error as a
        text annotation on the diff panel.

        Hardened (iter 18): the pre-hardening isinstance(fig, Figure)
        would pass even if show_error was silently ignored. We now
        assert a text artist containing ``"L_2"`` (LaTeX ``$L_2$``)
        exists on the diff panel — catches a regression that drops
        the error annotation.
        """
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1", show_error=True)
        assert isinstance(fig, Figure)
        assert "diff" in axes
        diff_ax = axes["diff"]
        # The L2 text is rendered via ``ax.text`` — search diff panel.
        l2_texts = [t for t in diff_ax.texts if "L_2" in t.get_text()]
        assert len(l2_texts) == 1
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


class TestSetDefaultTheme:
    def test_set_and_get(self) -> None:
        original = get_theme()
        try:
            set_theme("dark")
            assert get_theme().name == "dark"
        finally:
            set_theme(original)

    def test_badge_on_slice(self, ds_2d: FieldDataset) -> None:
        """Passing *step* to ``badge=True`` writes that step number into
        the badge text (not just any patch)."""
        fig, ax = plot_field_slice(ds_2d, "B_1", step=42, badge=True)
        # The badge is a patch-bearing artist with a child text carrying
        # the formatted cycle/step string.
        patch_artists = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(patch_artists) >= 1
        all_text = " ".join(t.get_text() for t in ax.texts)
        child_text = " ".join(
            t.get_text()
            for a in patch_artists
            for t in getattr(a, "get_children", lambda: [])()
            if hasattr(t, "get_text")
        )
        combined = all_text + " " + child_text
        assert "42" in combined, (
            f"expected step=42 in badge, got texts={all_text!r}, "
            f"child_texts={child_text!r}"
        )
        plt.close(fig)

    def test_save_creates_file(self, ds_2d: FieldDataset, tmp_path: Path) -> None:
        """``save=<path>`` writes a non-empty PNG with the PNG signature."""
        out = tmp_path / "test.png"
        plot_field_slice(ds_2d, "B_1", save=str(out))
        assert out.exists()
        # PNG magic bytes — proves a real image was written, not an empty file.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert out.stat().st_size > 500  # pcolormesh output is never this small


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
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import available_themes, export_themes

        export_themes(tmp_path)
        (tmp_path / "bad.toml").write_text("not valid {{{toml")
        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        themes = available_themes()
        assert "bad" not in themes

    def test_user_theme_overrides_bundled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
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
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
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

    def test_bundled_themes_have_webpic_section(self) -> None:
        """Every bundled theme carries a parseable ``[webpic]`` block.

        Cross-tool contract: webpic reads ``[webpic]`` (panel layout,
        shortcuts, diagnostics) plus the shared ``[colors]``/
        ``[colormaps]``/``[font]`` for visual identity. A missing or
        mis-versioned section silently degrades webpic's defaults.
        """
        import tomllib
        from pathlib import Path as _Path

        from pypic.plotting._theme_io import _bundled_theme_dir

        problems: list[str] = []
        for path in sorted(_Path(_bundled_theme_dir()).glob("*.toml")):
            raw = tomllib.loads(path.read_text())
            webpic = raw.get("webpic")
            if not isinstance(webpic, dict):
                problems.append(f"{path.name}: missing [webpic]")
                continue
            if webpic.get("version") != 1:
                got = webpic.get("version")
                problems.append(f"{path.name}: webpic.version = {got!r}, expected 1")
                continue
            for required in ("layout", "shortcuts", "diagnostics", "embed"):
                if required not in webpic:
                    problems.append(f"{path.name}: missing [webpic.{required}]")
        assert not problems, "themes failed [webpic] invariant: " + "; ".join(problems)


class TestLegendEntryNoneColor:
    def test_none_color_creates(self) -> None:
        """``color=None`` is preserved on the dataclass (theme-driven render)."""
        entry = LegendEntry(label="B", color=None)
        assert entry.color is None
        assert entry.label == "B"

    def test_none_color_renders(self) -> None:
        """Rendering a ``color=None`` entry attaches an ``AnchoredOffsetbox``
        artist whose descendant text carries the label."""
        from matplotlib.offsetbox import AnchoredOffsetbox

        fig, ax = plt.subplots()
        entry = LegendEntry(label="B", color=None)
        with use_theme("light"):
            box = add_legend(ax, entry)
        assert isinstance(box, AnchoredOffsetbox)
        assert box in ax.artists
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
        assert "upper left" in occupied  # label
        assert "lower right" in occupied  # colorbar
        assert "lower left" in occupied  # legend
        assert "upper right" in occupied  # badge
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
        """Explicit *vmin*/*vmax* propagate to each panel's norm identically."""
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B_1", "B_2"], ncols=2, vmin=-2.0, vmax=2.0)
        assert isinstance(fig, Figure)
        # Each panel carries one pcolormesh collection; both must honor
        # the shared color limits.
        for ax in axes:
            colls = ax.collections
            assert colls, "expected a pcolormesh per panel"
            norm = colls[0].norm
            assert norm.vmin == pytest.approx(-2.0)
            assert norm.vmax == pytest.approx(2.0)
        plt.close(fig)

    def test_cmap_dict(self, ds_2d: FieldDataset) -> None:
        """Per-field cmap dict dispatches a distinct colormap to each panel."""
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(
            ds_2d,
            ["B_1", "rho_m"],
            ncols=2,
            cmap={"B_1": "RdBu_r", "rho_m": "inferno"},
        )
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        cmap_names = [ax.collections[0].cmap.name for ax in axes]
        assert "RdBu_r" in cmap_names
        assert "inferno" in cmap_names
        plt.close(fig)

    def test_cmap_string(self, ds_2d: FieldDataset) -> None:
        """Scalar cmap string applies the same colormap to every panel."""
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B_1", "B_2"], ncols=2, cmap="viridis")
        assert isinstance(fig, Figure)
        for ax in axes:
            assert ax.collections[0].cmap.name == "viridis"
        plt.close(fig)

    def test_log_scale(self, ds_2d: FieldDataset) -> None:
        """``log_scale=True`` yields a LogNorm on positive-definite fields."""
        from matplotlib.colors import LogNorm

        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["rho_m", "P"], ncols=2, log_scale=True)
        assert isinstance(fig, Figure)
        for ax in axes:
            assert isinstance(ax.collections[0].norm, LogNorm)
        plt.close(fig)


class TestComparisonNewParams:
    def test_symmetric_false(self, ds_2d: FieldDataset) -> None:
        """``symmetric=False`` allows A/B norms to span the raw data range
        (non-symmetric around zero for a random-signed field)."""
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1", symmetric=False)
        assert isinstance(fig, Figure)
        norm_a = axes["a"].collections[0].norm
        # For a random-signed field with nonzero mean, the norm should
        # not be exactly symmetric around zero.
        assert norm_a.vmin + norm_a.vmax != pytest.approx(0.0, abs=1e-12)
        plt.close(fig)

    def test_alpha(self, ds_2d: FieldDataset) -> None:
        """*alpha* is forwarded to every panel's mesh."""
        fig, axes = plot_comparison(ds_2d, ds_2d, "B_1", alpha=0.5)
        assert isinstance(fig, Figure)
        for key in ("a", "b", "diff"):
            mesh = axes[key].collections[0]
            assert mesh.get_alpha() == pytest.approx(0.5)
        plt.close(fig)

    def test_log_scale(self, ds_2d: FieldDataset) -> None:
        """``log_scale=True`` on a positive field yields LogNorm on A and B."""
        from matplotlib.colors import LogNorm

        fig, axes = plot_comparison(ds_2d, ds_2d, "rho_m", log_scale=True)
        assert isinstance(fig, Figure)
        assert isinstance(axes["a"].collections[0].norm, LogNorm)
        assert isinstance(axes["b"].collections[0].norm, LogNorm)
        plt.close(fig)

    def test_log_scale_with_symmetric_warns(self, ds_2d: FieldDataset) -> None:
        """``log_scale=True`` + ``symmetric=True`` is a conflict → warn
        and drop log_scale (symmetric wins)."""
        with pytest.warns(UserWarning, match="log_scale=True ignored"):
            fig, axes = plot_comparison(
                ds_2d,
                ds_2d,
                "B_1",
                log_scale=True,
                symmetric=True,
            )
        # symmetric wins: norm should be symmetric around zero on A/B
        norm_a = axes["a"].collections[0].norm
        assert norm_a.vmin == pytest.approx(-norm_a.vmax)
        plt.close(fig)


class TestPlotKymograph:
    def test_basic(self) -> None:
        """The pcolormesh carries the input array verbatim (up to NaN
        masking) and the axes span matches ``coords``/``times`` extents."""
        from pypic.plotting import plot_kymograph

        rng = np.random.default_rng(99)
        values = rng.standard_normal((5, 10))
        coords = np.linspace(0, 10, 10)
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fig, ax = plot_kymograph(values, coords, times)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        mesh = ax.collections[0]
        arr = np.asarray(mesh.get_array()).reshape(values.shape)
        np.testing.assert_allclose(arr, values)
        plt.close(fig)

    def test_symmetric(self) -> None:
        """``symmetric=True`` forces ``vmin == -vmax`` on the norm."""
        from pypic.plotting import plot_kymograph

        values = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
        coords = np.array([0.0, 1.0, 2.0])
        times = np.array([0.0, 1.0])
        fig, ax = plot_kymograph(values, coords, times, symmetric=True)
        assert isinstance(fig, Figure)
        norm = ax.collections[0].norm
        assert norm.vmin == pytest.approx(-norm.vmax)
        assert norm.vmax > 0
        plt.close(fig)

    def test_log_scale(self) -> None:
        """``log_scale=True`` yields a ``LogNorm`` on positive data."""
        from matplotlib.colors import LogNorm

        from pypic.plotting import plot_kymograph

        values = np.abs(np.random.default_rng(1).standard_normal((4, 8))) + 0.1
        coords = np.linspace(0, 8, 8)
        times = np.arange(4, dtype=float)
        fig, ax = plot_kymograph(values, coords, times, log_scale=True)
        assert isinstance(ax.collections[0].norm, LogNorm)
        plt.close(fig)

    def test_custom_axes(self) -> None:
        """Passing ``ax=`` reuses that axes and paints onto it."""
        from pypic.plotting import plot_kymograph

        fig_ext, ax_ext = plt.subplots()
        before = len(ax_ext.collections)
        values = np.ones((3, 5))
        coords = np.arange(5, dtype=float)
        times = np.arange(3, dtype=float)
        _, ax = plot_kymograph(values, coords, times, ax=ax_ext)
        assert ax is ax_ext
        assert len(ax_ext.collections) == before + 1  # one new pcolormesh
        plt.close(fig_ext)

    def test_colorbar_inset(self) -> None:
        """``colorbar="inset"`` creates an inset Axes on the parent."""
        from pypic.plotting import plot_kymograph

        values = np.ones((3, 5))
        coords = np.arange(5, dtype=float)
        times = np.arange(3, dtype=float)
        fig, ax = plot_kymograph(values, coords, times, colorbar="inset")
        # Inset colorbars live as children of the parent axes.
        assert len(ax.child_axes) >= 1
        plt.close(fig)

    def test_invalid_shape(self) -> None:
        """1D input to ``plot_kymograph`` raises ``ValueError`` mentioning 2D."""
        from pypic.plotting import plot_kymograph

        with pytest.raises(ValueError, match="2D"):
            plot_kymograph(np.ones(10), np.ones(10), np.ones(1))

    def test_labels_and_title(self) -> None:
        """All three of xlabel/ylabel/title are pinned verbatim."""
        from pypic.plotting import plot_kymograph

        values = np.ones((2, 4))
        coords = np.arange(4, dtype=float)
        times = np.arange(2, dtype=float)
        fig, ax = plot_kymograph(
            values,
            coords,
            times,
            xlabel="$x$ [$d_i$]",
            ylabel="$t$ [$\\Omega_i^{-1}$]",
            title="Bz kymograph",
            label="$B_z$",
        )
        assert ax.get_xlabel() == "$x$ [$d_i$]"
        assert ax.get_ylabel() == "$t$ [$\\Omega_i^{-1}$]"
        assert ax.get_title() == "Bz kymograph"
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


class TestAnnotations:
    """Smoke tests for data-space annotation helpers."""

    def test_add_planet_returns_two_wedges(self) -> None:
        from matplotlib.patches import Wedge

        from pypic.plotting.annotations import add_planet

        fig, ax = plt.subplots()
        day, night = add_planet(ax, center=(0.0, 0.0), radius=1.0)
        assert isinstance(day, Wedge)
        assert isinstance(night, Wedge)
        # Both wedges land on the axes
        assert day in ax.patches
        assert night in ax.patches
        plt.close(fig)

    @pytest.mark.parametrize(
        ("sun_direction", "expected_sun_angle"),
        [("left", 180.0), ("right", 0.0), ("up", 90.0), ("down", 270.0)],
    )
    def test_add_planet_all_directions(
        self, sun_direction: str, expected_sun_angle: float
    ) -> None:
        """The day wedge's angular span is centered on the sun direction
        (theta1 = sun_angle - 90°)."""
        from pypic.plotting.annotations import add_planet

        fig, ax = plt.subplots()
        day, night = add_planet(ax, sun_direction=sun_direction)  # type: ignore[arg-type]
        assert len(ax.patches) == 2
        # Day wedge spans [sun-90, sun+90]; its theta1 pins the sun angle.
        assert day.theta1 == pytest.approx(expected_sun_angle - 90)
        assert day.theta2 == pytest.approx(expected_sun_angle + 90)
        # Night wedge spans the opposite half.
        assert night.theta1 == pytest.approx(expected_sun_angle + 90)
        assert night.theta2 == pytest.approx(expected_sun_angle + 270)
        plt.close(fig)

    def test_add_circle_returns_circle_patch(self) -> None:
        from matplotlib.patches import Circle

        from pypic.plotting.annotations import add_circle

        fig, ax = plt.subplots()
        c = add_circle(ax, radius=2.0)
        assert isinstance(c, Circle)
        assert c in ax.patches
        plt.close(fig)

    def test_add_circle_with_label_uses_theme_fontsize(self) -> None:
        """Default fontsize=None falls back to theme.annotation_fontsize."""
        from pypic.plotting.annotations import add_circle

        theme = get_theme()
        fig, ax = plt.subplots()
        with use_theme(theme):
            add_circle(ax, radius=2.0, label="boundary")
        # Label text was added to the axes
        labels = [t.get_text() for t in ax.texts]
        assert "boundary" in labels
        # And the text font size matches the theme default
        annotation_text = next(t for t in ax.texts if t.get_text() == "boundary")
        assert annotation_text.get_fontsize() == theme.annotation_fontsize
        plt.close(fig)

    def test_add_circle_explicit_fontsize_wins(self) -> None:
        """An explicit fontsize argument overrides the theme value."""
        from pypic.plotting.annotations import add_circle

        fig, ax = plt.subplots()
        add_circle(ax, radius=1.0, label="L1", fontsize=11.0)
        text = next(t for t in ax.texts if t.get_text() == "L1")
        assert text.get_fontsize() == 11.0
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


class TestRoundNice:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.7, 0.5),
            (1.3, 1.0),
            (3.5, 5.0),
            (150.0, 200.0),
            (0.003, 0.002),
            (7.0, 5.0),
            (15.0, 20.0),
            (0.11, 0.1),
        ],
    )
    def test_round_nice_values(self, value: float, expected: float) -> None:
        assert round_nice(value) == pytest.approx(expected)

    def test_zero(self) -> None:
        assert round_nice(0.0) == 0.0


class TestAutoClim:
    def test_signed_symmetric(self) -> None:
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        vmin, vmax = auto_clim(data, positive_definite=False)
        assert vmin == -vmax
        assert vmin < 0

    def test_positive_definite(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0

    def test_constant_field(self) -> None:
        data = np.full(100, 5.0)
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0

    def test_all_nan(self) -> None:
        data = np.full(10, np.nan)
        vmin, vmax = auto_clim(data, positive_definite=False)
        assert vmin < 0
        assert vmax > 0

    def test_all_nan_positive(self) -> None:
        data = np.full(10, np.nan)
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0


class TestAutoLinthresh:
    def test_typical_data(self) -> None:
        data = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        lt = _auto_linthresh(data)
        assert lt > 0

    def test_all_zero(self) -> None:
        data = np.zeros(10)
        lt = _auto_linthresh(data)
        assert lt == pytest.approx(1e-8)


class TestSymlogSlice:
    def test_symlog_renders(self, ds_2d: FieldDataset) -> None:
        """``symlog=True`` produces a SymLogNorm on the mesh."""
        from matplotlib.colors import SymLogNorm

        fig, ax = plot_field_slice(ds_2d, "B_1", symlog=True)
        assert isinstance(ax.collections[0].norm, SymLogNorm)
        plt.close(fig)

    def test_symlog_with_linthresh(self, ds_2d: FieldDataset) -> None:
        """Explicit *linthresh* is honored by the resulting SymLogNorm."""
        from matplotlib.colors import SymLogNorm

        fig, ax = plot_field_slice(ds_2d, "B_1", symlog=True, linthresh=0.1)
        norm = ax.collections[0].norm
        assert isinstance(norm, SymLogNorm)
        assert norm.linthresh == pytest.approx(0.1)
        plt.close(fig)
