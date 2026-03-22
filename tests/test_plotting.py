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
    PlotTheme,
    plot_comparison,
    plot_field_slice,
    use_theme,
)
from pypic.plotting._colormaps import is_positive_definite, symmetric_clim  # noqa: E402
from pypic.readers.base import FieldDataset, GridInfo  # noqa: E402
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
        )
        with use_theme(custom):
            assert mpl.rcParams["figure.dpi"] == 72
        assert mpl.rcParams["figure.dpi"] == original
