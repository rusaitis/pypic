"""Reusable 3D meshes: planet, reference circles, equatorial surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme, resolve_cmap

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset


def add_planet(
    plotter: pv.Plotter,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    sun_direction: str = "right",
    day_color: tuple[int, int, int] = (220, 220, 230),
    night_color: tuple[int, int, int] = (30, 30, 50),
    resolution: int = 32,
) -> pv.Actor:
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

    Returns
    -------
    pv.Actor
    """
    ensure_pyvista()
    import pyvista as _pv

    planet = _pv.Sphere(
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
    return plotter.add_mesh(planet, scalars="colors", rgb=True)


def add_reference_circles(
    plotter: pv.Plotter,
    radii: list[float],
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    z: float = 0.0,
    color: str | None = None,
    width: float | None = None,
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
    width : float or None
        Line width. ``None`` uses ``theme.line_width``.
    n_points : int
        Number of points per circle.
    theme : PlotTheme or None
        Theme for default color and width.
    """
    ensure_pyvista()

    t = _resolve_theme(theme)
    if color is None:
        gc = t.grid_color[:3]
        lum = 0.299 * gc[0] + 0.587 * gc[1] + 0.114 * gc[2]
        color = "grey" if lum < 0.1 else f"#{int(gc[0]*255):02x}{int(gc[1]*255):02x}{int(gc[2]*255):02x}"
    if width is None:
        width = t.line_width

    theta = np.linspace(0, 2 * np.pi, n_points)
    for r in radii:
        ring = np.column_stack([
            center[0] + r * np.cos(theta),
            center[1] + r * np.sin(theta),
            np.full(n_points, z),
        ])
        plotter.add_lines(ring, color=color, width=width)


def _auto_resolve(
    data_2d: FieldDataset,
    field: str,
    *,
    units: str | None,
    cmap: str | Colormap | None,
    clim: tuple[float, float] | None,
    scalar_label: str | None,
    theme: PlotTheme | None,
) -> tuple[np.ndarray, object, tuple[float, float] | None, str]:
    """Auto-resolve field values, colormap, clim, and label from metadata.

    Returns (values, resolved_cmap, clim, label).
    """
    from pypic.plotting._colormaps import is_positive_definite, symmetric_clim
    from pypic.plotting._labels import field_label
    from pypic.plotting._resolve import resolve_field_values

    t = _resolve_theme(theme)

    # Values with optional unit conversion
    values = resolve_field_values(data_2d, field, units)

    # Colormap: user override > metadata auto-select > theme default
    if cmap is None:
        info = data_2d.field_info(field)
        if is_positive_definite(field, values, info):
            resolved_cmap = resolve_cmap(None, signed=False, theme=theme)
        else:
            resolved_cmap = resolve_cmap(None, signed=True, theme=theme)
    else:
        resolved_cmap = resolve_cmap(cmap, theme=theme)

    # Clim: user override > auto (symmetric for signed, (0, max) for positive)
    if clim is None:
        info = data_2d.field_info(field)
        if is_positive_definite(field, values, info):
            vmax = float(np.nanmax(values))
            clim = (0.0, vmax if vmax > 0 else 1e-8)
        else:
            clim = symmetric_clim(values)

    # Label: user override > metadata + units
    if scalar_label is None:
        info = data_2d.field_info(field)
        if info is not None:
            scalar_label = field_label(info, unit_str=units or "")
        else:
            scalar_label = f"{field} [{units}]" if units else field

    return values, resolved_cmap, clim, scalar_label


def add_equatorial_surface(
    plotter: pv.Plotter,
    data_2d: FieldDataset,
    field: str,
    *,
    units: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    opacity: float = 0.8,
    z: float = 0.0,
    scalar_label: str | None = None,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_right",
    fmt: str | None = None,
    theme: PlotTheme | None = None,
) -> pv.Actor:
    r"""Add a scalar surface from a 2D FieldDataset slice.

    When *units* is provided, field values are automatically converted
    and the colorbar label is generated from field metadata. When *cmap*
    is ``None``, it is auto-selected (diverging for signed fields,
    sequential for positive-definite). When *clim* is ``None``, it is
    auto-computed (symmetric for signed, ``(0, max)`` for positive).

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    data_2d : FieldDataset
        A 2D dataset (e.g. from ``PlaneSelection.apply()``).
    field : str
        Field name to plot as surface color.
    units : str or None
        Display units (e.g. ``"nT"``). Converts values and auto-labels.
    cmap : str, Colormap, or None
        Colormap. ``None`` auto-selects from field metadata and theme.
    clim : tuple[float, float] or None
        Color limits. ``None`` auto-computes from field data.
    opacity : float
        Surface opacity.
    z : float
        Z-coordinate of the surface plane.
    scalar_label : str or None
        Label for the scalar bar. ``None`` auto-generates from metadata.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    scalar_bar_position : str
        Position: ``"lower_right"``, ``"lower_left"``,
        ``"upper_right"``, ``"upper_left"``.
    fmt : str or None
        Number format for scalar bar labels. ``None`` auto-selects
        (``"%.0f"`` for integer-scale values, ``"%.1f"`` otherwise).
    theme : PlotTheme or None
        Theme for colors and fonts.

    Returns
    -------
    pv.Actor
    """
    ensure_pyvista()
    import pyvista as _pv

    t = _resolve_theme(theme)

    values, resolved_cmap, clim, label = _auto_resolve(
        data_2d, field, units=units, cmap=cmap, clim=clim,
        scalar_label=scalar_label, theme=theme,
    )

    # Auto fmt: integer format if clim range is large enough
    if fmt is None:
        if clim is not None and abs(clim[1] - clim[0]) >= 5:
            fmt = "%.0f"
        else:
            fmt = "%.1f"

    coords = data_2d.grid.coordinate_arrays()
    x2d, y2d = np.meshgrid(coords[0], coords[1], indexing="ij")

    surface = _pv.StructuredGrid(x2d, y2d, np.full_like(x2d, z))
    # Use a plain array name internally; pretty label only for scalar bar title
    array_name = field
    surface[array_name] = np.nan_to_num(values, nan=0.0).ravel(order="F")

    # Styled scalar bar with themed background
    sbar_args: dict[str, object] = {}
    sbar_title = ""
    if show_scalar_bar:
        from pypic.plotting.pyvista._overlay import build_scalar_bar

        sbar_args, sbar_title = build_scalar_bar(
            plotter, label,
            position=scalar_bar_position, fmt=fmt, clim=clim, theme=theme,
        )

    actor = plotter.add_mesh(
        surface,
        scalars=array_name,
        cmap=resolved_cmap,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=sbar_args,
    )

    if show_scalar_bar:
        from pypic.plotting.pyvista._overlay import style_scalar_bar

        style_scalar_bar(plotter, sbar_title)

    return actor
