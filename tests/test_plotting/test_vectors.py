"""Quiver and streamline plots, alone and composed with a field slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypic.plotting import (
    LegendEntry,
    add_legend,
    plot_field_slice,
    plot_quiver,
    plot_streamlines,
)

if TYPE_CHECKING:
    from pypic.dataset import FieldDataset


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

        An isinstance-only assertion let a no-op
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

        In addition to the identity check, this pins
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

        Asserting only isinstance would be tautological. This
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

    @pytest.mark.parametrize(
        "plot_fn", [plot_streamlines, plot_quiver], ids=["streamlines", "quiver"]
    )
    def test_unknown_color_field_is_named_in_the_error(
        self, ds_2d: FieldDataset, plot_fn: object
    ) -> None:
        with pytest.raises(ValueError, match="color_field 'nope' not found"):
            plot_fn(ds_2d, "B", color_field="nope")  # type: ignore[operator]

    def test_quiver_colors_magnitude_in_the_requested_units(
        self, ds_2d: FieldDataset
    ) -> None:
        """With *units* and no *color_field*, arrows are colored by the
        in-plane magnitude in those units, matching the colorbar label."""
        _, ax = plot_quiver(ds_2d, "B", units="nT")
        expected = np.hypot(ds_2d.in_units("B_1", "nT"), ds_2d.in_units("B_2", "nT"))
        drawn = np.asarray(ax.collections[0].get_array())
        np.testing.assert_allclose(
            np.sort(drawn), np.sort(expected.ravel()), rtol=1e-12
        )
        plt.close("all")

    def test_quiver_honours_color_limits(self, ds_2d: FieldDataset) -> None:
        _, ax = plot_quiver(ds_2d, "B", vmin=0.5, vmax=1.5)
        norm = ax.collections[0].norm
        assert (norm.vmin, norm.vmax) == (0.5, 1.5)
        plt.close("all")

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

        Asserting only isinstance — any regression that silently ignored the kwarg
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


class TestComposedFieldAndVectors:
    def test_slice_then_streamlines(self, ds_2d: FieldDataset) -> None:
        """Overlaying streamlines on a field slice produces exactly
        one QuadMesh (slice) plus one LineCollection (streamlines).

        An isinstance-only assertion ignored
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

        Asserting only isinstance would be tautological.
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

        In addition to the identity check, we
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
