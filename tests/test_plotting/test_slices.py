"""Field slices, comparisons, grids, cross-sections and kymographs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from pypic.dataset import FieldDataset
from pypic.plotting import plot_comparison, plot_field_slice, plot_kymograph
from pypic.selections import PlaneSelection
from pypic.units import Normalization
from tests._helpers import make_uniform_grid

if TYPE_CHECKING:
    from collections.abc import Callable


class TestPlotFieldSlice:
    def test_returns_figure_and_axes(self, ds_2d: FieldDataset) -> None:
        """Plotted ``B_1`` data equals the input field (transposed).

        The pcolormesh-backed image is in
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

        Asserting only ``isinstance(fig, Figure)`` is a tautology that passes if
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

        The weaker form created two figures
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


class TestLogScale:
    def test_log_scale_positive_field(self, ds_2d: FieldDataset) -> None:
        """``log_scale=True`` wraps the pcolormesh norm in LogNorm.

        An ``isinstance(fig, Figure)`` check did not guard the
        log path at all — a silently-linear plot would pass. Pinning
        the collection's norm to ``matplotlib.colors.LogNorm`` catches
        a regression that forgets to apply log scaling.
        """
        from matplotlib.colors import LogNorm

        fig, ax = plot_field_slice(ds_2d, "rho_m", log_scale=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax.collections[0].norm, LogNorm)
        plt.close(fig)


def _as_dataset(values: np.ndarray) -> FieldDataset:
    return FieldDataset.from_arrays(
        {"B_1": values}, make_uniform_grid(*values.shape), Normalization.identity()
    )


def _slice_norm(values: np.ndarray, **options: Any) -> Normalize:
    _, ax = plot_field_slice(_as_dataset(values), "B_1", **options)
    return ax.collections[0].norm


def _comparison_norm(values: np.ndarray, **options: Any) -> Normalize:
    ds = _as_dataset(values)
    _, axes = plot_comparison(ds, ds, "B_1", **options)
    return axes["a"].collections[0].norm


def _kymograph_norm(values: np.ndarray, **options: Any) -> Normalize:
    n_times, n_x = values.shape
    coords, times = np.arange(float(n_x)), np.arange(float(n_times))
    _, ax = plot_kymograph(values, coords, times, **options)
    return ax.collections[0].norm


_SIGNED = np.random.default_rng(7).standard_normal((10, 8))
_ABSMAX = float(np.abs(_SIGNED).max())
_MOSTLY_POSITIVE = np.abs(_SIGNED) + 0.1
_MOSTLY_POSITIVE[0, 0] = -1.0
_MOSTLY_POSITIVE[1, 1] = 0.0
_POSITIVE_RANGE = (
    float(_MOSTLY_POSITIVE[_MOSTLY_POSITIVE > 0].min()),
    float(_MOSTLY_POSITIVE.max()),
)


class TestNormAgreement:
    @pytest.mark.parametrize(
        ("values", "options", "expected"),
        [
            pytest.param(
                _SIGNED, {}, (Normalize, -_ABSMAX, _ABSMAX), id="auto-symmetric"
            ),
            pytest.param(
                _SIGNED,
                {"symmetric": False},
                (Normalize, float(_SIGNED.min()), float(_SIGNED.max())),
                id="data-range",
            ),
            pytest.param(
                _SIGNED,
                {"vmin": -0.5, "vmax": 0.5},
                (Normalize, -0.5, 0.5),
                id="explicit",
            ),
            pytest.param(
                _SIGNED,
                {"vmax": 2.0},
                (Normalize, float(_SIGNED.min()), 2.0),
                id="one-sided",
            ),
            pytest.param(
                _MOSTLY_POSITIVE,
                {"log_scale": True, "symmetric": False},
                (LogNorm, *_POSITIVE_RANGE),
                id="log-skips-non-positive",
            ),
        ],
    )
    def test_same_input_gives_the_same_norm_on_every_plot(
        self,
        values: np.ndarray,
        options: dict[str, Any],
        expected: tuple[type, float, float],
    ) -> None:
        """Slice, comparison and kymograph resolve the color scale the same way."""
        sites: dict[str, Callable[..., Normalize]] = {
            "slice": _slice_norm,
            "comparison": _comparison_norm,
            "kymograph": _kymograph_norm,
        }
        observed = {}
        for site, norm_of in sites.items():
            norm = norm_of(values, **options)
            observed[site] = (type(norm), norm.vmin, norm.vmax)
        plt.close("all")
        assert set(observed.values()) == {expected}, observed


class TestSuppliedAxes:
    def test_comparison_draws_into_the_three_supplied_axes(
        self, ds_2d: FieldDataset
    ) -> None:
        _, supplied = plt.subplots(1, 3)
        _, axes = plot_comparison(ds_2d, ds_2d, "B_1", ax=tuple(supplied))
        drawn = [(panel, len(panel.collections)) for panel in axes.values()]
        assert drawn == [(panel, 1) for panel in supplied]
        plt.close("all")

    def test_cross_section_draws_into_the_supplied_pair(
        self, ds_2d: FieldDataset
    ) -> None:
        from pypic.plotting import plot_cross_section

        _, (top, bottom) = plt.subplots(1, 2)
        _, (ax_2d, ax_1d) = plot_cross_section(
            ds_2d, "B_1", cut_axis="x", ax=(top, bottom)
        )
        drawn = (ax_2d, ax_1d, len(top.collections), len(bottom.lines))
        assert drawn == (top, bottom, 1, 1)
        plt.close("all")

    def test_field_grid_draws_one_field_per_supplied_axes(
        self, ds_2d: FieldDataset
    ) -> None:
        from pypic.plotting import plot_field_grid

        _, supplied = plt.subplots(1, 2)
        _, axes = plot_field_grid(ds_2d, ["B_1", "P"], ax=list(supplied))
        drawn = [(panel, len(panel.collections)) for panel in axes]
        assert drawn == [(panel, 1) for panel in supplied]
        plt.close("all")

    def test_field_grid_rejects_a_mismatched_axes_count(
        self, ds_2d: FieldDataset
    ) -> None:
        from pypic.plotting import plot_field_grid

        _, supplied = plt.subplots(1, 3)
        with pytest.raises(ValueError, match="3 axes for 2 fields"):
            plot_field_grid(ds_2d, ["B_1", "P"], ax=list(supplied))
        plt.close("all")


class TestSymlogSlice:
    def test_symlog_centres_signed_fields_on_zero(self, ds_2d: FieldDataset) -> None:
        """Symlog keeps a signed field's zero-centred limits, where the
        diverging colormap puts its midpoint."""
        fig, ax = plot_field_slice(ds_2d, "B_1", symlog=True)
        norm = ax.collections[0].norm
        assert norm.vmin == -norm.vmax
        plt.close(fig)

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


class TestAddContours:
    def test_basic_contour(self, ds_2d: FieldDataset) -> None:
        """``add_contours`` adds a ContourSet to an existing axes
        without replacing the underlying pcolormesh.

        The weaker form ``cs is not None`` was
        a function-signature tautology. This pins:
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

    def test_returns_the_contour_set_it_drew(self, ds_2d: FieldDataset) -> None:
        from matplotlib.contour import QuadContourSet

        from pypic.plotting import add_contours

        fig, ax = plot_field_slice(ds_2d, "B_1")
        cs = add_contours(ax, ds_2d, "P")
        assert isinstance(cs, QuadContourSet)
        assert cs in ax.collections
        plt.close(fig)

    def test_contour_with_labels(self, ds_2d: FieldDataset) -> None:
        """``labels=True`` attaches label texts to the contour set.

        An earlier form had no assertions. This
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


class TestPlotComparison:
    def test_returns_figure_and_axes_dict(self, ds_2d: FieldDataset) -> None:
        """Three-panel layout; diff panel vanishes for identical inputs.

        When both inputs are the same dataset the
        elementwise difference must be zero everywhere. The panel keys
        were already pinned; this also pins the numerical content of
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

        A weaker form asserted ``isinstance(fig, Figure)``. Now also
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

        Asserting only ``isinstance(fig, Figure)`` would be tautological. Each
        option instead pins a concrete
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


class TestComparisonShowError:
    def test_show_error(self, ds_2d: FieldDataset) -> None:
        """``show_error=True`` overlays the L2 relative error as a
        text annotation on the diff panel.

        The weaker form isinstance(fig, Figure)
        would pass even if show_error was silently ignored. This
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


class TestPlotFieldGrid:
    def test_basic_grid(self, ds_2d: FieldDataset) -> None:
        """Requesting N fields yields N axes, each with a pcolormesh.

        Checking only ``len(axes) == 4`` is weak. This pins that every panel has
        at least
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

        Pins the ``len(collections) == 1``
        invariant alongside the panel-count check.
        """
        from pypic.plotting import plot_field_grid

        fig, axes = plot_field_grid(ds_2d, ["B_1"])
        assert len(axes) == 1
        assert len(axes[0].collections) == 1
        plt.close(fig)

    def test_panel_labels_disabled(self, ds_2d: FieldDataset) -> None:
        """``panel_labels=False`` suppresses the panel-label artists.

        An earlier form had no assertion. This
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


class TestPlotCrossSection:
    def test_basic(self, ds_2d: FieldDataset) -> None:
        """Cross-section returns (2D ax with pcolormesh, 1D ax with line).

        The weaker form did isinstance checks only.
        This pins that the 2D panel has exactly one collection
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

        An earlier form had no assertion. ``cut_axis
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
        np.testing.assert_array_equal(arr, values)
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

    def test_units_are_appended_to_the_colorbar_label(self) -> None:
        values = np.ones((3, 5))
        fig, ax = plot_kymograph(
            values, np.arange(5.0), np.arange(3.0), label="$B_z$", units="nT"
        )
        colorbar_ax = next(a for a in fig.axes if a is not ax)
        assert colorbar_ax.get_ylabel() == "$B_z$ [nT]"
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
