"""Axis triad and equatorial grid for pyvista 3D plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
    import pyvista as pv

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


def _frame_labels(data: Any) -> tuple[str, str, str]:
    """Generate axis labels from a FieldDataset's coordinate frame."""
    frame = data.frame
    if frame and frame != "simulation":
        return (f"$x_{{{frame}}}$", f"$y_{{{frame}}}$", f"$z_{{{frame}}}$")
    return ("$x$", "$y$", "$z$")


def _coord_units_from_data(data: Any) -> str:
    """Extract coordinate units from dataset metadata."""
    meta = data.metadata if hasattr(data, "metadata") else {}
    unit = meta.get("physical_extent_unit", "")
    if unit and unit != "m":
        return f"${unit}$" if "_" in unit else unit
    return ""


def add_axis_triad(
    plotter: pv.Plotter,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    length: float = 5.0,
    *,
    data: Any = None,
    labels: tuple[str, str, str] | None = None,
    font_size: int | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw axis arrows from *center* with themed colors and labels.

    When *data* (a :class:`FieldDataset`) is provided and *labels* is
    ``None``, axis labels are auto-generated from the coordinate frame
    (e.g. ``"$x_{GSM}$"``).

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    center : tuple[float, float, float]
        Origin of the triad.
    length : float
        Length of each axis arrow.
    data : FieldDataset or None
        Source dataset for auto-generating frame labels.
    labels : tuple[str, str, str] or None
        Labels for the x, y, z axes. ``None`` auto-generates from
        *data* frame or defaults to ``("$X$", "$Y$", "$Z$")``.
    font_size : int or None
        Override label font size. ``None`` derives from theme.
    theme : PlotTheme or None
        Theme for colors. ``None`` uses active theme.
    """
    if labels is None:
        labels = _frame_labels(data) if data is not None else ("$X$", "$Y$", "$Z$")
    ensure_pyvista()
    import pyvista as pv

    t = _resolve_theme(theme)
    colors = (t.axis_x_color, t.axis_y_color, t.axis_z_color)
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    c = np.asarray(center, dtype=float)

    # Scale font: pyvista font_size is ~3x matplotlib pt for similar visual weight
    fs = font_size if font_size is not None else int(t.font_label * 3.0)

    lw = t.line_width  # default 1.5
    cone_h = t.arrow_size * 0.1  # 4.0 → 0.4
    cone_r = t.arrow_size * 0.025  # 4.0 → 0.1

    for direction, color, label in zip(directions, colors, labels, strict=True):
        d = np.asarray(direction, dtype=float)
        tip = c + d * length
        shaft = pv.Line(tuple(c), tuple(tip))
        plotter.add_mesh(shaft, color=color, line_width=lw)
        cone = pv.Cone(
            center=tuple(tip - d * cone_h * 0.5),
            direction=direction,
            height=cone_h,
            radius=cone_r,
            resolution=16,
        )
        plotter.add_mesh(cone, color=color)
        plotter.add_point_labels(
            [tip + d * 0.5],
            [label],
            font_size=fs,
            text_color=color,
            shape=None,
            show_points=False,
        )


def _nice_step(data_range: float, target_n: int) -> float:
    """Pick the nearest 1-2-5 step size for *data_range* / *target_n*."""
    import math

    raw = data_range / max(target_n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    residual = raw / mag
    if residual <= 1.5:
        return mag
    if residual <= 3.5:
        return 2 * mag
    if residual <= 7.5:
        return 5 * mag
    return 10 * mag


def _uniform_ticks(lo: float, hi: float, step: float) -> list[float]:
    """Generate evenly spaced ticks at multiples of *step* within [lo, hi]."""
    import math

    first = math.ceil(lo / step) * step
    ticks: list[float] = []
    v = first
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def add_equatorial_grid(
    plotter: pv.Plotter,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    data_2d: Any = None,
    margin: float = 0.0,
    z: float = 0.0,
    step: float | None = None,
    n_lines: int = 5,
    coord_units: str | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw subtle grid lines on a z-plane with coordinate labels.

    Grid lines use a uniform spacing in 1-2-5 multiples, identical
    in both x and y directions. Labels are placed at grid edges with
    the last label appending *coord_units*.

    Limits and coordinate units can be derived automatically from a
    2D :class:`FieldDataset`.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    xlim : tuple[float, float] or None
        X-axis range. ``None`` derives from *data_2d*.
    ylim : tuple[float, float] or None
        Y-axis range. ``None`` derives from *data_2d*.
    data_2d : FieldDataset or None
        A 2D dataset to derive limits and coordinate units from.
    margin : float
        Extra extent beyond the data range in each direction
        (in data coordinates). Only used with *data_2d*.
    z : float
        Z-coordinate of the grid plane.
    step : float or None
        Explicit grid spacing. ``None`` auto-selects a 1-2-5 step
        based on the larger axis range and *n_lines*.
    n_lines : int
        Target number of grid lines per axis (used when *step* is ``None``).
    coord_units : str or None
        Unit string appended to the last label. ``None`` auto-derives
        from *data_2d* metadata when available.
    theme : PlotTheme or None
        Theme for grid color and fonts. ``None`` uses active theme.
    """
    # Derive limits and coord_units from data if not given
    if xlim is None or ylim is None:
        if data_2d is None:
            msg = "Either xlim/ylim or data_2d must be provided"
            raise ValueError(msg)
        coords = data_2d.grid.coordinate_arrays()
        if xlim is None:
            xlim = (float(coords[0][0]) - margin, float(coords[0][-1]) + margin)
        if ylim is None:
            ylim = (float(coords[1][0]) - margin, float(coords[1][-1]) + margin)
    if coord_units is None and data_2d is not None:
        coord_units = _coord_units_from_data(data_2d)
    if coord_units is None:
        coord_units = ""
    ensure_pyvista()
    import pyvista as pv

    t = _resolve_theme(theme)
    gc = t.grid_color  # (r, g, b, alpha)
    grid_hex = _rgba_to_hex(gc)
    grid_opacity = gc[3]
    grid_lw = t.grid_major_width

    label_color = _rgba_to_hex(t.secondary_text_color)
    label_fs = int(t.font_tick * 2.8)

    # Normalize ranges for reversed limits
    x_lo, x_hi = sorted(xlim)
    y_lo, y_hi = sorted(ylim)

    # Uniform 1-2-5 step from the larger range
    if step is None:
        max_range = max(x_hi - x_lo, y_hi - y_lo)
        step = _nice_step(max_range, n_lines)

    x_ticks = _uniform_ticks(x_lo, x_hi, step)
    y_ticks = _uniform_ticks(y_lo, y_hi, step)

    # Draw grid lines
    for x in x_ticks:
        line = pv.Line((x, y_lo, z), (x, y_hi, z))
        plotter.add_mesh(line, color=grid_hex, opacity=grid_opacity, line_width=grid_lw)

    for y in y_ticks:
        line = pv.Line((x_lo, y, z), (x_hi, y, z))
        plotter.add_mesh(line, color=grid_hex, opacity=grid_opacity, line_width=grid_lw)

    # Coordinate labels at grid edges
    offset_frac = 0.015
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
            show_points=False,
            bold=False,
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
            show_points=False,
            bold=False,
        )
