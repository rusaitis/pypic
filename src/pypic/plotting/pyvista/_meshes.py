"""Reusable 3D meshes: planet, reference circles, equatorial surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme
    from pypic.readers._field_dataset import FieldDataset


def add_planet(
    plotter: pv.Plotter,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    sun_direction: Literal["right", "left"] = "right",
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
    opacity: float | None = None,
    width: float | None = None,
    n_points: int = 128,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw reference circles on a z-plane.

    Each circle is rendered as a sequence of disjoint segments (via
    ``Plotter.add_lines``), which produces a dashed appearance — half
    the *n_points* become visual gaps.

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
        Circle color. ``None`` uses ``theme.grid_color`` (RGB only).
    opacity : float or None
        Circle opacity. ``None`` uses ``min(1.0, theme.grid_color[3] * 4)``
        so the circles read as visibly stronger reference markers than
        the much fainter regular grid lines.
    width : float or None
        Line width. ``None`` uses ``theme.line_width``.
    n_points : int
        Number of points per circle (half become visual gaps).
    theme : PlotTheme or None
        Theme for default color, opacity, and width.
    """
    ensure_pyvista()

    t = _resolve_theme(theme)
    gc = t.grid_color
    if color is None:
        color = f"#{int(gc[0] * 255):02x}{int(gc[1] * 255):02x}{int(gc[2] * 255):02x}"
    if opacity is None:
        opacity = min(1.0, gc[3] * 4)
    if width is None:
        width = t.line_width

    theta = np.linspace(0, 2 * np.pi, n_points)
    for r in radii:
        ring = np.column_stack(
            [
                center[0] + r * np.cos(theta),
                center[1] + r * np.sin(theta),
                np.full(n_points, z),
            ]
        )
        actor = plotter.add_lines(ring, color=color, width=width)
        actor.GetProperty().SetOpacity(opacity)


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
    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_field_colormap,
        symmetric_clim,
    )
    from pypic.plotting._labels import field_label
    from pypic.plotting._resolve import resolve_field_values

    # Values with optional unit conversion
    values = resolve_field_values(data_2d, field, units)
    info = data_2d.field_info(field)
    t = _resolve_theme(theme)

    # Colormap: shared dispatcher with the matplotlib backend.
    _, resolved_cmap = resolve_field_colormap(field, values, t, info=info, cmap=cmap)

    # Clim: user override > auto (symmetric for signed, (0, max) for positive)
    if clim is None:
        if is_positive_definite(field, values, info):
            vmax = float(np.nanmax(values))
            clim = (0.0, vmax if vmax > 0 else 1e-8)
        else:
            clim = symmetric_clim(values)

    # Label: user override > metadata + units
    if scalar_label is None:
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
    opacity: float | str | np.ndarray = 0.8,
    z: float = 0.0,
    rendering: Literal["smooth", "pixel"] = "smooth",
    lighting: bool = False,
    scalar_label: str | None = None,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_right",
    fmt: str | None = None,
    extremes: str | None = "semi",
    colorbar_ticks: list[float] | None = None,
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
    opacity : float, str, or ndarray
        Surface opacity. A float gives uniform alpha; a string names
        a pyvista transfer function mapping scalar value to alpha
        (``"linear"``, ``"sigmoid"``, ``"geom"``, and their ``"_r"``
        reversed variants); an ndarray supplies per-point or per-cell
        opacity. Combine ``opacity="linear"`` with a diverging cmap
        to make mid-range values fade out so strong features pop.
    z : float
        Z-coordinate of the surface plane.
    rendering : str
        ``"smooth"`` (default) renders a ``pv.StructuredGrid`` with
        point data and Gouraud-interpolated colors — smooth gradients
        between grid nodes, good for eye-friendly visualization.
        ``"pixel"`` renders a ``pv.ImageData`` with cell data and
        ``interpolate_before_map=False`` — each data sample becomes
        one flat-shaded cell at the true grid resolution, showing
        the actual sampling without interpolation. Assumes a uniform
        grid (pypic's standard); for non-uniform grids use ``"smooth"``.
    lighting : bool
        ``False`` (default) disables lighting so colors reflect the
        colormap exactly. ``True`` restores diffuse/specular lighting,
        which adds an angle-dependent brightness gradient across the
        flat plane — rarely what you want for scientific 2D slices.
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
    extremes : "semi", "transparent", "darken", or None
        How to style the colorbar's under/over extension triangles.
        ``"semi"`` (default) renders them at reduced opacity, mirroring
        :func:`pypic.plotting.plot_field_slice`. ``None`` keeps full
        opacity.
    colorbar_ticks : list[float] or None
        Explicit colorbar tick positions. ``None`` auto-generates from
        *clim*. Mirrors the ``colorbar_ticks`` parameter on
        :func:`pypic.plotting.plot_field_slice`.
    theme : PlotTheme or None
        Theme for colors and fonts.

    Returns
    -------
    pv.Actor
    """
    ensure_pyvista()
    import pyvista as _pv

    values, resolved_cmap, clim, label = _auto_resolve(
        data_2d,
        field,
        units=units,
        cmap=cmap,
        clim=clim,
        scalar_label=scalar_label,
        theme=theme,
    )

    coords = data_2d.grid.coordinate_arrays()
    array_name = field
    clean = np.nan_to_num(values, nan=0.0)

    if rendering == "pixel":
        # Uniform-grid ImageData: each data sample becomes one flat
        # cell. Dimensions are one larger than the data along each
        # axis (cells are between vertices), origin shifted by half
        # a grid spacing so cell centers land on the original sample
        # positions. Requires a uniform grid.
        nx, ny = clean.shape
        dx = float(coords[0][1] - coords[0][0]) if nx > 1 else 1.0
        dy = float(coords[1][1] - coords[1][0]) if ny > 1 else 1.0
        origin = (
            float(coords[0][0]) - dx * 0.5,
            float(coords[1][0]) - dy * 0.5,
            z,
        )
        surface = _pv.ImageData(
            dimensions=(nx + 1, ny + 1, 1),
            spacing=(dx, dy, 1.0),
            origin=origin,
        )
        surface.cell_data[array_name] = clean.ravel(order="F")
    else:
        # StructuredGrid + point data: Gouraud-interpolated smooth rendering
        x2d, y2d = np.meshgrid(coords[0], coords[1], indexing="ij")
        surface = _pv.StructuredGrid(x2d, y2d, np.full_like(x2d, z))
        surface[array_name] = clean.ravel(order="F")

    actor = plotter.add_mesh(
        surface,
        scalars=array_name,
        cmap=resolved_cmap,
        clim=clim,
        opacity=opacity,
        lighting=lighting,
        interpolate_before_map=(rendering == "smooth"),
        show_scalar_bar=False,
    )

    if show_scalar_bar:
        from pypic.plotting.pyvista._overlay import add_colorbar

        add_colorbar(
            plotter,
            resolved_cmap,
            clim,
            label=label,
            loc=scalar_bar_position,
            fmt=fmt,
            extremes=extremes,
            ticks=colorbar_ticks,
            theme=theme,
        )

    return actor
