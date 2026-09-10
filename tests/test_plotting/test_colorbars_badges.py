"""Colorbars, status badges, labels, legends and annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pypic.plotting import (
    LegendEntry,
    add_badge,
    add_inset_colorbar,
    add_label,
    add_legend,
    get_theme,
    plot_field_slice,
    plot_quiver,
    plot_streamlines,
    use_theme,
)
from pypic.plotting._badge import _detect_overlay_defaults, _format_status_text

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

    from pypic.dataset import FieldDataset


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

        The looser bound ``0 < alpha <= 1``
        admitted any alpha >= next-representable-zero. Tightened to
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

        The looser ``!= alpha_default or !=
        bg_default`` permitted identical output in both attributes.
        Requires at least one of {bg, fg, alpha} to differ
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

        ``cb is not None`` was tautological once
        the function signature declared ``-> Colorbar``. This pins
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

        Asserting only isinstance would be tautological.
        This pins the inset invariant — one figure axes, one
        ``ax.child_axes`` — which catches a regression that silently
        fell back to a separate colorbar axes (``fig.axes >= 2``).
        """
        fig, ax = plot_fn(ds_2d)  # type: ignore[operator]
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert len(ax.child_axes) == 1
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
