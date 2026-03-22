"""Plotting utilities for pypic field data.

Requires matplotlib (optional dependency). Install with::

    pip install pypic[plot]

This module can be imported for type checking without matplotlib installed.
Actual plotting functions call ``ensure_matplotlib()`` at entry.
"""

from pypic.plotting.comparison import plot_comparison
from pypic.plotting.slices import plot_field_slice
from pypic.plotting.styles import DARK, DEFAULT, LIGHT, PlotTheme, use_theme

__all__ = [
    "DARK",
    "DEFAULT",
    "LIGHT",
    "PlotTheme",
    "plot_comparison",
    "plot_field_slice",
    "use_theme",
]
