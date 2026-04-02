"""2D vector field plotting: streamlines and quiver arrows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pypic.plotting._guard import ensure_matplotlib
from pypic.plotting._labels import axis_label, figure_title
from pypic.plotting._resolve import (
    get_or_create_axes,
    prepare_data,
    resolve_coord_units,
    resolve_field_values,
    surviving_axis_names,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import PlotTheme, ThemeArg
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection


def _resolve_plane_components(data: FieldDataset, field_prefix: str) -> tuple[str, str]:
    """Return the two in-plane component field names for a vector field.

    Maps surviving dataset axes to the corresponding numbered component
    indices using the full 3D geometry axis order. For example, on a
    z-normal slice with surviving axes ``(x, y)``, returns
    ``("B1", "B2")`` for ``field_prefix="B"``.

    Parameters
    ----------
    data : FieldDataset
        2D dataset (after plane selection).
    field_prefix : str
        Vector field prefix (e.g. ``"B"``, ``"V"``, ``"J"``).

    Returns
    -------
    tuple[str, str]
        Numbered field names for the two in-plane components.
    """
    surviving = data.grid.surviving_axis_names
    if len(surviving) != 2:
        msg = f"Expected 2D data, got {len(surviving)}D"
        raise ValueError(msg)
    all_axes = data.grid.geometry.axis_names
    comp0 = all_axes.index(surviving[0]) + 1
    comp1 = all_axes.index(surviving[1]) + 1
    return f"{field_prefix}{comp0}", f"{field_prefix}{comp1}"


def _resolve_vector_colors(
    data: FieldDataset,
    field: str,
    color_field: str | None,
    magnitude: object,
    units: str | None,
    theme: PlotTheme,
    cmap: str | Colormap | None,
) -> tuple[object, str, object]:
    """Resolve color values, colormap name, and field info for vector plots.

    Returns
    -------
    tuple
        ``(color_values, cmap_name, info)``
    """
    from pypic.plotting._colormaps import resolve_colormap

    if color_field is not None:
        try:
            color_values = resolve_field_values(data, color_field, units)
        except KeyError:
            msg = f"color_field {color_field!r} not found in dataset"
            raise ValueError(msg) from None
        color_name = color_field
    else:
        color_values = magnitude
        # Display name for in-plane magnitude (e.g. "|B_plane|")
        color_name = f"|{field}_{{plane}}|"

    info = data.field_info(color_field if color_field is not None else f"|{field}|")
    cmap_name = resolve_colormap(
        color_name,
        color_values,
        theme,
        info=info,
        cmap=cmap if isinstance(cmap, str) else None,
    )
    return color_values, cmap_name, info


def plot_streamlines(
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    color: str | None = None,
    color_field: str | None = None,
    units: str | None = None,
    coord_units: str | tuple[str, str] | None = None,
    alpha: float = 1.0,
    theme: ThemeArg = None,
    cmap: str | Colormap | None = None,
    density: float = 1.5,
    downsample: int = 1,
    smooth: float | None = None,
    linewidth: float | tuple[float, float] | None = None,
    arrowsize: float = 1.0,
    arrowstyle: str = "-|>",
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    legend: bool | str = True,
    badge: bool = False,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — streamplot passthrough
) -> tuple[Figure, Axes]:
    r"""Plot streamlines of a 2D vector field.

    By default, lines are colored by in-plane magnitude through a
    colormap. Pass *color* (e.g. ``"black"``) for uniform-color
    streamlines — useful as overlays on scalar field plots.

    If *data* is 3D and no *plane* is given, defaults to a midplane
    slice along the last axis.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    field : str
        Vector field prefix — ``"B"``, ``"V"``, ``"J"``, etc.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    color : str | None
        Uniform line color (e.g. ``"black"``, ``"white"``, ``"#3399ff"``).
        ``None`` maps color to magnitude or *color_field* via colormap.
    color_field : str | None
        Scalar field for line color (e.g. ``"|B|"``, ``"beta"``).
        ``None`` uses in-plane magnitude. Ignored when *color* is set.
    units : str | None
        Display units for the color field.
    coord_units : str, tuple[str, str], or None
        Display units for coordinate axes. A single string applies to
        both axes; a tuple ``(x_unit, y_unit)`` labels each independently.
    alpha : float
        Line and arrow transparency (0 = invisible, 1 = opaque).
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override automatic colormap selection.
    density : float
        Streamline density (passed to ``ax.streamplot``).
    downsample : int
        Downsample the vector grid by this factor before tracing
        streamlines. Values > 1 speed up ``streamplot`` significantly
        on large grids with no visible difference.
    smooth : float or None
        Gaussian smoothing sigma in grid cells, applied to the vector
        components before tracing. Suppresses grid-scale noise (useful
        for PIC moment data). ``None`` disables smoothing.
    linewidth : float | tuple[float, float] | None
        Fixed linewidth, or ``(min, max)`` tuple to scale by magnitude.
        ``None`` defaults to ``(0.5, 2.0)`` scaled by magnitude.
    arrowsize : float
        Arrow size scaling for streamplot.
    arrowstyle : str
        Arrow style string (default ``"-|>"``). Common alternatives:
        ``"->"`` (thinner), ``"fancy"``, ``"simple"``.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    colorbar : bool
        Whether to add a colorbar. Ignored when *color* is set.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
        ``"transparent"`` makes them invisible.
    legend : bool or str
        When *color* is set (uniform mode), add a vector legend overlay.
        ``True`` uses the field prefix as label, a string overrides it.
        ``False`` disables. Ignored when using colormap mode.
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.streamplot()`` (``minlength``, ``maxlength``,
        ``start_points``, ``integration_direction``,
        ``broken_streamlines``, etc.).

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    owned = ax is None
    theme = _resolve_theme_arg(theme)
    data = prepare_data(data, plane)

    comp_u, comp_v = _resolve_plane_components(data, field)
    u = resolve_field_values(data, comp_u, None)
    v = resolve_field_values(data, comp_v, None)

    magnitude = np.sqrt(u**2 + v**2)
    use_colormap = color is None

    if use_colormap:
        color_values, cmap_name, info = _resolve_vector_colors(
            data, field, color_field, magnitude, units, theme, cmap
        )

    coords = data.grid.coordinate_arrays()
    surviving_axes = surviving_axis_names(data)

    # Linewidth dispatch
    match linewidth:
        case None:
            lw_range: tuple[float, float] | None = (0.5, 2.0)
        case (lo, hi):
            lw_range = (lo, hi)
        case _:
            lw_range = None

    if lw_range is not None:
        lw_mid = 0.5 * (lw_range[0] + lw_range[1])
        mag_min, mag_max = float(np.nanmin(magnitude)), float(np.nanmax(magnitude))
        if mag_max > mag_min:
            lw_scaled = lw_range[0] + (lw_range[1] - lw_range[0]) * (
                (magnitude - mag_min) / (mag_max - mag_min)
            )
            lw_scaled = np.nan_to_num(lw_scaled, nan=lw_mid)
        else:
            lw_scaled = np.full_like(magnitude, lw_mid)
        lw_arg = lw_scaled.T
    else:
        lw_arg = linewidth  # type: ignore[assignment]

    # Smooth vector components to suppress grid-scale noise
    if smooth is not None and smooth > 0:
        from scipy.ndimage import gaussian_filter

        nan_mask = np.isnan(u) | np.isnan(v)
        u = gaussian_filter(np.nan_to_num(u, nan=0.0), sigma=smooth)
        v = gaussian_filter(np.nan_to_num(v, nan=0.0), sigma=smooth)
        u[nan_mask] = np.nan
        v[nan_mask] = np.nan

    # Downsample vector grid for faster streamline tracing
    s = downsample
    if s > 1:
        coords = (coords[0][::s], coords[1][::s])
        u = u[::s, ::s]
        v = v[::s, ::s]
        if use_colormap:
            color_values = color_values[::s, ::s]  # type: ignore[possibly-undefined]
        if isinstance(lw_arg, np.ndarray):
            lw_arg = lw_arg[::s, ::s]

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        if use_colormap:
            stream = ax.streamplot(
                coords[0],
                coords[1],
                u.T,
                v.T,
                color=color_values.T,
                cmap=cmap if not isinstance(cmap, str) else cmap_name,
                density=density,
                linewidth=lw_arg,
                arrowsize=arrowsize,
                arrowstyle=arrowstyle,
                **kwargs,
            )
        else:
            stream = ax.streamplot(
                coords[0],
                coords[1],
                u.T,
                v.T,
                color=color,
                density=density,
                linewidth=lw_arg,
                arrowsize=arrowsize,
                arrowstyle=arrowstyle,
                **kwargs,
            )

        if alpha < 1.0:
            stream.lines.set_alpha(alpha)
            # stream.arrows.set_alpha doesn't work — arrows are individual FancyArrowPatch children
            from matplotlib.patches import FancyArrowPatch

            for child in ax.get_children():
                if isinstance(child, FancyArrowPatch):
                    child.set_alpha(alpha)

        if use_colormap and colorbar:
            from pypic.plotting._colorbar import attach_colorbar
            from pypic.plotting._labels import field_label

            unit_str = units or ""
            cb_label = field_label(info, unit_str=unit_str)
            attach_colorbar(
                fig, ax, stream.lines, cb_label, colorbar, extremes=extremes
            )

        if not use_colormap and legend is not False:
            from pypic.plotting._badge import LegendEntry, add_legend

            legend_label = legend if isinstance(legend, str) else field
            from pypic.plotting.styles import _theme_val

            default_lw: float = _theme_val("line_width", 1.0)
            lw = linewidth if isinstance(linewidth, (int, float)) else default_lw
            entry = LegendEntry(
                label=legend_label,
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
            add_legend(ax, entry)

        cu_x, cu_y = resolve_coord_units(coord_units)
        ax.set_xlabel(axis_label(surviving_axes[0], unit_str=cu_x))
        ax.set_ylabel(axis_label(surviving_axes[1], unit_str=cu_y))
        ax.set_aspect("equal")
        apply_grid(ax, theme)

        if title is not None:
            ax.set_title(title)
        else:
            _info = info if use_colormap else data.field_info(f"|{field}|")
            ax.set_title(figure_title(_info, step=step, time=time))

        if badge and (step is not None or time is not None):
            from pypic.plotting._badge import add_badge

            add_badge(ax, step=step, time=time)

        if owned:
            fig.tight_layout()
            apply_rounding(ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax


def plot_quiver(
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    color: str | None = None,
    color_field: str | None = None,
    units: str | None = None,
    coord_units: str | tuple[str, str] | None = None,
    alpha: float = 1.0,
    theme: ThemeArg = None,
    cmap: str | Colormap | None = None,
    stride: int | tuple[int, int] = 1,
    scale: float | None = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    legend: bool | str = True,
    badge: bool = False,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — quiver passthrough
) -> tuple[Figure, Axes]:
    r"""Plot a quiver (arrow) field on a 2D slice.

    By default, arrows are colored by in-plane magnitude through a
    colormap. Pass *color* (e.g. ``"black"``) for uniform-color
    arrows — useful as overlays on scalar field plots.

    If *data* is 3D and no *plane* is given, defaults to a midplane
    slice along the last axis.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    field : str
        Vector field prefix — ``"B"``, ``"V"``, ``"J"``, etc.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    color : str | None
        Uniform arrow color (e.g. ``"black"``, ``"white"``).
        ``None`` maps color to magnitude or *color_field* via colormap.
    color_field : str | None
        Scalar field for arrow color. ``None`` uses in-plane magnitude.
        Ignored when *color* is set.
    units : str | None
        Display units for the color field.
    coord_units : str, tuple[str, str], or None
        Display units for coordinate axes. A single string applies to
        both axes; a tuple ``(x_unit, y_unit)`` labels each independently.
    alpha : float
        Arrow transparency (0 = invisible, 1 = opaque).
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override automatic colormap selection.
    stride : int | tuple[int, int]
        Subsample every N grid points. Scalar or ``(stride_x, stride_y)``.
    scale : float | None
        Quiver scale factor (passed to ``ax.quiver``). ``None`` for auto.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    colorbar : bool
        Whether to add a colorbar. Ignored when *color* is set.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
        ``"transparent"`` makes them invisible.
    legend : bool or str
        When *color* is set (uniform mode), add a vector legend overlay.
        ``True`` uses the field prefix as label, a string overrides it.
        ``False`` disables. Ignored when using colormap mode.
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.quiver()`` (``headwidth``, ``headlength``,
        ``headaxislength``, ``pivot``, ``minshaft``, ``minlength``,
        ``units``, ``angles``, ``width``, etc.).

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    owned = ax is None
    theme = _resolve_theme_arg(theme)
    data = prepare_data(data, plane)

    comp_u, comp_v = _resolve_plane_components(data, field)
    u = resolve_field_values(data, comp_u, None)
    v = resolve_field_values(data, comp_v, None)

    use_colormap = color is None

    if use_colormap:
        magnitude = np.sqrt(u**2 + v**2)
        color_values, cmap_name, info = _resolve_vector_colors(
            data, field, color_field, magnitude, units, theme, cmap
        )

    coords = data.grid.coordinate_arrays()
    surviving_axes = surviving_axis_names(data)

    # Stride dispatch
    match stride:
        case int():
            s0, s1 = stride, stride
        case (s0, s1):
            pass

    x_sub = coords[0][::s0]
    y_sub = coords[1][::s1]
    u_sub = u[::s0, ::s1]
    v_sub = v[::s0, ::s1]

    xx, yy = np.meshgrid(x_sub, y_sub, indexing="ij")

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        if use_colormap:
            color_sub = color_values[::s0, ::s1]
            quiv = ax.quiver(
                xx.T,
                yy.T,
                u_sub.T,
                v_sub.T,
                color_sub.T,
                cmap=cmap if not isinstance(cmap, str) else cmap_name,
                scale=scale,
                alpha=alpha,
                **kwargs,
            )
        else:
            quiv = ax.quiver(
                xx.T,
                yy.T,
                u_sub.T,
                v_sub.T,
                color=color,
                scale=scale,
                alpha=alpha,
                **kwargs,
            )

        if use_colormap and colorbar:
            from pypic.plotting._colorbar import attach_colorbar
            from pypic.plotting._labels import field_label

            unit_str = units or ""
            cb_label = field_label(info, unit_str=unit_str)
            attach_colorbar(fig, ax, quiv, cb_label, colorbar, extremes=extremes)

        if not use_colormap and legend is not False:
            from pypic.plotting._badge import LegendEntry, add_legend

            legend_label = legend if isinstance(legend, str) else field
            entry = LegendEntry(
                label=legend_label,
                color=color,
                alpha=alpha,
            )
            add_legend(ax, entry)

        cu_x, cu_y = resolve_coord_units(coord_units)
        ax.set_xlabel(axis_label(surviving_axes[0], unit_str=cu_x))
        ax.set_ylabel(axis_label(surviving_axes[1], unit_str=cu_y))
        ax.set_aspect("equal")
        apply_grid(ax, theme)

        if title is not None:
            ax.set_title(title)
        else:
            _info = info if use_colormap else data.field_info(f"|{field}|")
            ax.set_title(figure_title(_info, step=step, time=time))

        if badge and (step is not None or time is not None):
            from pypic.plotting._badge import add_badge

            add_badge(ax, step=step, time=time)

        if owned:
            fig.tight_layout()
            apply_rounding(ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax
