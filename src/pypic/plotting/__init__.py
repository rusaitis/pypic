"""Plotting utilities for pypic field data.

Requires matplotlib (optional dependency). Install with::

    pip install pypic[plot]

This module can be imported for type checking without matplotlib installed.
Actual plotting functions call ``ensure_matplotlib()`` at entry.
"""

from pypic.plotting._badge import (
    BadgeLoc,
    LegendEntry,
    add_badge,
    add_label,
    add_legend,
)
from pypic.plotting._colorbar import ExtremesMode, add_colorbar, add_inset_colorbar
from pypic.plotting._theme_io import (
    available_themes,
    export_themes,
    load_theme,
    save_theme,
)
from pypic.plotting.annotations import SunDirection, add_circle, add_planet
from pypic.plotting.comparison import plot_comparison
from pypic.plotting.cross_section import plot_cross_section
from pypic.plotting.kymograph import plot_kymograph
from pypic.plotting.lines import (
    plot_line,
    plot_line_comparison,
    plot_lines,
    plot_time_series,
)
from pypic.plotting.panels import plot_field_grid
from pypic.plotting.scatter import plot_scatter
from pypic.plotting.slices import add_contours, plot_field_slice
from pypic.plotting.spectral import plot_power_spectrum
from pypic.plotting.styles import (
    PlotTheme,
    ThemeArg,
    apply_grid,
    apply_theme_to_figure,
    get_active_theme,
    get_theme,
    set_theme,
    style_legend,
    use_theme,
)
from pypic.plotting.vectors import plot_quiver, plot_streamlines

__all__ = [
    "BadgeLoc",
    "ExtremesMode",
    "LegendEntry",
    "PlotTheme",
    "SunDirection",
    "ThemeArg",
    "add_badge",
    "add_circle",
    "add_colorbar",
    "add_contours",
    "add_inset_colorbar",
    "add_label",
    "add_legend",
    "add_planet",
    "apply_grid",
    "apply_theme_to_figure",
    "available_themes",
    "export_themes",
    "get_active_theme",
    "get_theme",
    "load_theme",
    "plot_comparison",
    "plot_cross_section",
    "plot_field_grid",
    "plot_field_slice",
    "plot_kymograph",
    "plot_line",
    "plot_line_comparison",
    "plot_lines",
    "plot_power_spectrum",
    "plot_quiver",
    "plot_scatter",
    "plot_streamlines",
    "plot_time_series",
    "save_theme",
    "set_theme",
    "style_legend",
    "use_theme",
]
