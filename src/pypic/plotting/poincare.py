"""Scatter plot of a `PoincareSection` in plane coordinates."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.plotting.styles import ThemeArg
    from pypic.traces._poincare import PoincareSection


def plot_poincare_section(
    section: PoincareSection,
    *,
    ax: Axes | None = None,
    theme: ThemeArg = None,
    color_by_seed: bool = True,
    marker_size: float = 2.0,
    alpha: float = 0.7,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Scatter the puncture cloud of a `PoincareSection`.

    Each seed's punctures get a distinct color from the active theme's
    cycle when ``color_by_seed`` is true — co-orbital points appear as
    same-color closed curves (islands / KAM surfaces) and chaotic seeds
    fill 2D regions in a single color.

    Parameters
    ----------
    section : PoincareSection
        Output of [`pypic.traces.poincare_section`][pypic.traces.poincare_section].
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    theme : PlotTheme | None
        Plot theme. ``None`` uses the active default.
    color_by_seed : bool
        Cycle a distinct theme color per seed. When false, all punctures
        share one color (cleaner for very dense clouds).
    marker_size : float
        Marker size in points².
    alpha : float
        Point transparency.
    title : str | None
        Axes title. Defaults to ``f"Poincaré section: {surface.name}"``
        when ``section.surface.name`` is set.
    figsize : tuple[float, float] | None
        Figure size override.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import get_or_create_axes
    from pypic.plotting.styles import _resolve_theme_arg, apply_grid, use_theme

    resolved_theme = _resolve_theme_arg(theme)
    owned = ax is None

    with use_theme(resolved_theme) if owned else nullcontext():
        fig, ax = get_or_create_axes(resolved_theme, ax, figsize)
        cycle = resolved_theme.color_cycle
        n_colors = len(cycle) if cycle else 1

        if color_by_seed:
            for k, pts in enumerate(section.punctures_2d):
                if pts.shape[0] == 0:
                    continue
                color = cycle[k % n_colors] if cycle else None
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=marker_size,
                    alpha=alpha,
                    color=color,
                    edgecolors="none",
                    label=f"seed {k}" if section.n_seeds <= 8 else None,
                )
        else:
            cloud = section.all_punctures_2d
            color = cycle[0] if cycle else None
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                s=marker_size,
                alpha=alpha,
                color=color,
                edgecolors="none",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$u$")
        ax.set_ylabel("$v$")
        if title is None and section.surface.name is not None:
            title = f"Poincaré section: {section.surface.name}"
        if title is not None:
            ax.set_title(title)
        apply_grid(ax, resolved_theme)

    return fig, ax
