"""Plotting utilities for pypic field data.

Requires matplotlib (optional dependency). Install with::

    pip install pypic[plot]

This module can be imported for type checking without matplotlib installed.
Actual plotting functions call ``ensure_matplotlib()`` at entry.
"""

from pypic.plotting._badge import (
    OVERLAY_BORDER_PAD,
    BadgeLoc,
    VectorLegendEntry,
    add_panel_label,
    add_status_badge,
    add_vector_legend,
)
from pypic.plotting._colorbar import add_inset_colorbar
from pypic.plotting.comparison import plot_comparison
from pypic.plotting.lines import plot_line, plot_time_series
from pypic.plotting.lines3d import plot_field_line, plot_trajectory
from pypic.plotting.slices import plot_field_slice
from pypic.plotting.styles import (
    ANDROMEDA,
    ANUPPUCCIN_LIGHT,
    CATPPUCCIN_MOCHA,
    DARK,
    DEFAULT,
    LIGHT,
    PlotTheme,
    apply_grid,
    apply_theme_to_figure,
    style_3d_axes,
    use_theme,
)
from pypic.plotting.vectors import plot_quiver, plot_streamlines

__all__ = [
    "ANDROMEDA",
    "ANUPPUCCIN_LIGHT",
    "CATPPUCCIN_MOCHA",
    "DARK",
    "DEFAULT",
    "LIGHT",
    "OVERLAY_BORDER_PAD",
    "BadgeLoc",
    "PlotTheme",
    "VectorLegendEntry",
    "add_inset_colorbar",
    "add_panel_label",
    "add_status_badge",
    "add_vector_legend",
    "apply_grid",
    "apply_theme_to_figure",
    "plot_comparison",
    "plot_field_line",
    "plot_field_slice",
    "plot_line",
    "plot_quiver",
    "plot_streamlines",
    "plot_time_series",
    "plot_trajectory",
    "style_3d_axes",
    "use_theme",
]
