"""Reusable 3D meshes: planet, reference circles, equatorial surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme, resolve_cmap

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset


def add_planet(
    plotter: Any,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    sun_direction: str = "right",
    day_color: tuple[int, int, int] = (220, 220, 230),
    night_color: tuple[int, int, int] = (30, 30, 50),
    resolution: int = 32,
) -> None:
    r"""Add a day/night planet sphere.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    radius : float
        Planet radius.
    center : tuple[float, float, float]
        Planet center.
    sun_direction : str
        ``"right"`` (+x is sunlit) or ``"left"`` (-x is sunlit).
    day_color : tuple[int, int, int]
        RGB color for the dayside (0–255).
    night_color : tuple[int, int, int]
        RGB color for the nightside (0–255).
    resolution : int
        Sphere resolution (theta and phi).
    """
    ensure_pyvista()
    import pyvista as pv

    planet = pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    centers = planet.cell_centers().points
    sign = 1.0 if sun_direction == "right" else -1.0
    day_mask = (centers[:, 0] - center[0]) * sign > 0
    colors = np.where(
        day_mask[:, np.newaxis],
        np.array(day_color),
        np.array(night_color),
    )
    planet.cell_data["colors"] = colors.astype(np.uint8)
    plotter.add_mesh(planet, scalars="colors", rgb=True)


def add_reference_circles(
    plotter: Any,
    radii: list[float],
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    z: float = 0.0,
    color: str | None = None,
    width: float = 1.0,
    n_points: int = 128,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw reference circles on a z-plane.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    radii : list[float]
        List of circle radii to draw.
    center : tuple[float, float, float]
        Center of the circles.
    z : float
        Z-coordinate of the plane.
    color : str or None
        Circle color. ``None`` uses theme grid color.
    width : float
        Line width.
    n_points : int
        Number of points per circle.
    theme : PlotTheme or None
        Theme for default color.
    """
    ensure_pyvista()

    if color is None:
        t = _resolve_theme(theme)
        gc = t.grid_color[:3]
        lum = 0.299 * gc[0] + 0.587 * gc[1] + 0.114 * gc[2]
        # Near-black grid color is invisible on dark backgrounds — use grey
        color = "grey" if lum < 0.1 else f"#{int(gc[0]*255):02x}{int(gc[1]*255):02x}{int(gc[2]*255):02x}"

    theta = np.linspace(0, 2 * np.pi, n_points)
    for r in radii:
        ring = np.column_stack([
            center[0] + r * np.cos(theta),
            center[1] + r * np.sin(theta),
            np.full(n_points, z),
        ])
        plotter.add_lines(ring, color=color, width=width)


def add_equatorial_surface(
    plotter: Any,
    data_2d: FieldDataset,
    field: str,
    *,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    opacity: float = 0.8,
    z: float = 0.0,
    scalar_label: str | None = None,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_right",
    fmt: str = "%.1f",
    theme: PlotTheme | None = None,
) -> None:
    r"""Add a scalar surface from a 2D FieldDataset slice.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    data_2d : FieldDataset
        A 2D dataset (e.g. from ``PlaneSelection.apply()``).
    field : str
        Field name to plot as surface color.
    cmap : str, Colormap, or None
        Colormap. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Color limits. ``None`` for auto.
    opacity : float
        Surface opacity.
    z : float
        Z-coordinate of the surface plane.
    scalar_label : str or None
        Label for the scalar bar.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    scalar_bar_position : str
        Position hint: ``"lower_right"``, ``"lower_left"``.
    fmt : str
        Number format for scalar bar labels (e.g. ``"%.0f"``).
    theme : PlotTheme or None
        Theme for colors and fonts.
    """
    ensure_pyvista()
    import pyvista as pv

    t = _resolve_theme(theme)
    resolved_cmap = resolve_cmap(cmap, signed=True, theme=theme)

    coords = data_2d.grid.coordinate_arrays()
    x2d, y2d = np.meshgrid(coords[0], coords[1], indexing="ij")
    values = data_2d[field]

    surface = pv.StructuredGrid(x2d, y2d, np.full_like(x2d, z))
    label = scalar_label or field
    surface[label] = np.nan_to_num(values, nan=0.0).ravel(order="F")

    pos_x = 0.72 if "right" in scalar_bar_position else 0.05

    plotter.add_mesh(
        surface,
        scalars=label,
        cmap=resolved_cmap,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=dict(
            title=f"{label}\n ",
            n_labels=5,
            position_x=pos_x,
            position_y=0.03,
            width=0.22,
            height=0.035,
            title_font_size=int(t.font_label),
            label_font_size=int(t.font_tick * 1.4),
            shadow=True,
            fmt=fmt,
        ),
    )
