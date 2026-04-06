"""Axis triad and equatorial grid for pyvista 3D plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
    from pypic.plotting.styles import PlotTheme


def _label_values(ticks: list[float]) -> list[float]:
    """Return tick values to label, skipping near-zero and thinning.

    Mirrors the matplotlib ``style_3d_axes`` logic: skip labels within
    30 % of the tick step from zero, cap to 5 labels by thinning.
    """
    if len(ticks) < 2:
        return ticks
    zero_skip = 0.3
    max_before_thin = 5
    step = abs(ticks[1] - ticks[0])
    filtered = [v for v in ticks if abs(v) > zero_skip * step]
    if not filtered:
        return ticks
    if len(filtered) > max_before_thin:
        return filtered[::2]
    return filtered


def _rgba_to_hex(rgba: tuple[float, float, float, float]) -> str:
    """Convert an RGBA float tuple to a hex color string (ignoring alpha)."""
    r, g, b, _a = rgba
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def add_axis_triad(
    plotter: Any,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    length: float = 5.0,
    *,
    labels: tuple[str, str, str] = ("$X$", "$Y$", "$Z$"),
    font_size: int | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw axis arrows from *center* with themed colors and labels.

    Uses ``theme.axis_x_color``, ``axis_y_color``, ``axis_z_color``
    for the three axes and ``theme.font_label`` for sizing.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    center : tuple[float, float, float]
        Origin of the triad.
    length : float
        Length of each axis arrow.
    labels : tuple[str, str, str]
        Labels for the x, y, z axes.
    font_size : int or None
        Override label font size. ``None`` derives from theme.
    theme : PlotTheme or None
        Theme for colors. ``None`` uses active theme.
    """
    ensure_pyvista()
    import pyvista as pv

    t = _resolve_theme(theme)
    colors = (t.axis_x_color, t.axis_y_color, t.axis_z_color)
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    c = np.asarray(center, dtype=float)

    # Scale font: pyvista font_size is ~3x matplotlib pt for similar visual weight
    fs = font_size if font_size is not None else int(t.font_label * 3)

    for direction, color, label in zip(directions, colors, labels, strict=True):
        d = np.asarray(direction, dtype=float)
        tip = c + d * length
        shaft = pv.Line(tuple(c), tuple(tip))
        plotter.add_mesh(shaft, color=color, line_width=1.5)
        cone = pv.Cone(
            center=tuple(tip - d * 0.2),
            direction=direction,
            height=0.4,
            radius=0.1,
            resolution=16,
        )
        plotter.add_mesh(cone, color=color)
        plotter.add_point_labels(
            [tip + d * 0.5],
            [label],
            font_size=fs,
            text_color=color,
            shape=None,
            render_points_as_spheres=False,
            point_size=0,
        )


def add_equatorial_grid(
    plotter: Any,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    z: float = 0.0,
    n_lines: int = 5,
    coord_units: str = "",
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw subtle grid lines on a z-plane with coordinate labels.

    Replicates the matplotlib ``style_3d_axes`` grid: ``MaxNLocator``
    tick placement, label thinning near zero, labels at grid edges
    with the last label appending *coord_units*.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    xlim : tuple[float, float]
        X-axis range.
    ylim : tuple[float, float]
        Y-axis range.
    z : float
        Z-coordinate of the grid plane.
    n_lines : int
        Approximate number of grid lines per axis.
    coord_units : str
        Unit string appended to the last label (e.g. ``"$R_E$"``).
    theme : PlotTheme or None
        Theme for grid color and fonts. ``None`` uses active theme.
    """
    ensure_pyvista()
    import pyvista as pv

    t = _resolve_theme(theme)
    gc = t.grid_color  # (r, g, b, alpha)
    grid_hex = _rgba_to_hex(gc)
    grid_opacity = gc[3]
    grid_lw = t.grid_major_width

    label_color = _rgba_to_hex(t.secondary_text_color)
    # pyvista font_size ~3x matplotlib pt
    label_fs = int(t.font_tick * 2.5)

    from matplotlib.ticker import MaxNLocator

    # Normalize ranges so offset math works for reversed limits
    x_lo, x_hi = sorted(xlim)
    y_lo, y_hi = sorted(ylim)

    loc = MaxNLocator(nbins=n_lines)
    x_ticks = [v for v in loc.tick_values(x_lo, x_hi) if x_lo <= v <= x_hi]
    y_ticks = [v for v in loc.tick_values(y_lo, y_hi) if y_lo <= v <= y_hi]

    # Draw grid lines
    for x in x_ticks:
        line = pv.Line((x, y_lo, z), (x, y_hi, z))
        plotter.add_mesh(line, color=grid_hex, opacity=grid_opacity, line_width=grid_lw)

    for y in y_ticks:
        line = pv.Line((x_lo, y, z), (x_hi, y, z))
        plotter.add_mesh(line, color=grid_hex, opacity=grid_opacity, line_width=grid_lw)

    # Coordinate labels at grid edges
    offset_frac = 0.04
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    x_labels = _label_values(x_ticks)
    for j, x in enumerate(x_labels):
        val = f"{x:g}"
        if coord_units and j == len(x_labels) - 1:
            val += f" {coord_units}"
        plotter.add_point_labels(
            [(x, y_lo - y_range * offset_frac, z)],
            [val],
            font_size=label_fs,
            text_color=label_color,
            shape=None,
            render_points_as_spheres=False,
            point_size=0,
        )

    y_labels = _label_values(y_ticks)
    for j, y in enumerate(y_labels):
        val = f"{y:g}"
        if coord_units and j == len(y_labels) - 1:
            val += f" {coord_units}"
        plotter.add_point_labels(
            [(x_lo - x_range * offset_frac, y, z)],
            [val],
            font_size=label_fs,
            text_color=label_color,
            shape=None,
            render_points_as_spheres=False,
            point_size=0,
        )
