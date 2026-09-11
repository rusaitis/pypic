"""2D vector field plotting: streamlines and quiver arrows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pypic.plotting._badge import LegendEntry, add_legend
from pypic.plotting._colorbar import attach_colorbar
from pypic.plotting._guard import ensure_matplotlib
from pypic.plotting._labels import field_label, figure_title
from pypic.plotting._resolve import (
    finish_axes,
    get_or_create_axes,
    maybe_save,
    plane_axis_labels,
    prepare_data,
    resolve_field_values,
)
from pypic.plotting.styles import _resolve_theme_arg, _theme_val, use_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.dataset import FieldDataset
    from pypic.fields import FieldInfo
    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import PlotTheme, ThemeArg
    from pypic.selections import PlaneSelection
    from pypic.types import FloatArray


@dataclass(frozen=True, slots=True)
class _VectorInputs:
    data: FieldDataset
    components: tuple[str, str]
    u: FloatArray
    v: FloatArray
    color_values: FloatArray | None  # None when drawn in one uniform color
    colormap: Colormap | None
    info: FieldInfo  # colorbar label and default title


def _vector_prelude(
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None,
    color: str | None,
    color_field: str | None,
    units: str | None,
    theme: PlotTheme,
    cmap: str | Colormap | None,
) -> _VectorInputs:
    """Slice to the plane, look up the in-plane components, resolve the colors.

    The surviving axes map to component indices through the full geometry
    order, so a y-normal slice of ``B`` draws ``(B_1, B_3)``. Without
    *color_field* the colors are the in-plane magnitude, in *units* when given.
    """
    from pypic.plotting._colormaps import resolve_field_colormap

    data = prepare_data(data, plane)
    surviving = data.grid.surviving_axis_names
    if len(surviving) != 2:
        msg = f"Expected 2D data, got {len(surviving)}D"
        raise ValueError(msg)
    all_axes = data.grid.geometry.axis_names
    comp_u, comp_v = (f"{field}_{all_axes.index(a) + 1}" for a in surviving)
    u = resolve_field_values(data, comp_u, None)
    v = resolve_field_values(data, comp_v, None)
    # The leading "|" is what marks the default magnitude positive-definite.
    magnitude_name = f"|{field}|"
    if color is not None:
        info = data.field_info(magnitude_name)
        return _VectorInputs(data, (comp_u, comp_v), u, v, None, None, info)

    if color_field is not None:
        try:
            color_values = resolve_field_values(data, color_field, units)
        except KeyError:
            msg = f"color_field {color_field!r} not found in dataset"
            raise ValueError(msg) from None
    elif units is not None:
        color_values = np.hypot(
            resolve_field_values(data, comp_u, units),
            resolve_field_values(data, comp_v, units),
        )
    else:
        color_values = np.hypot(u, v)
    color_name = color_field if color_field is not None else magnitude_name
    info = data.field_info(color_name)
    _, colormap = resolve_field_colormap(
        color_name, color_values, theme, info=info, cmap=cmap
    )
    return _VectorInputs(data, (comp_u, comp_v), u, v, color_values, colormap, info)


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
    vmin: float | None = None,
    vmax: float | None = None,
    density: float = 1.5,
    downsample: int = 1,
    smooth: float | None = None,
    magnitude_min: float | None = None,
    magnitude_max: float | None = None,
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
    extremes : "semi", "transparent", "darken", or None
        Colorbar out-of-range indicator style. ``"semi"`` (default)
        uses semi-transparent extension colors; ``"transparent"``
        hides them; ``"darken"`` darkens the endpoint colors;
        ``None`` leaves matplotlib defaults untouched.
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
    vmin : float or None
        Lower color limit. ``None`` (default) autoscales.
    vmax : float or None
        Upper color limit. ``None`` (default) autoscales.
    magnitude_min : float or None
        Lower bound on vector magnitude; weaker vectors are
        masked out. ``None`` (default) keeps all.
    magnitude_max : float or None
        Upper bound on vector magnitude; stronger vectors are
        masked out. ``None`` (default) keeps all.
    badge : str or None
        Corner badge text (run label, timestamp, ...). ``None``
        (default) draws no badge.
    save : str or Path or None
        Path to write the figure to. When given, the figure is saved
        and closed; when ``None`` (default) it is left open for the
        caller to display or modify further.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    owns_figure = ax is None
    theme = _resolve_theme_arg(theme)
    inputs = _vector_prelude(
        data,
        field,
        plane=plane,
        color=color,
        color_field=color_field,
        units=units,
        theme=theme,
        cmap=cmap,
    )
    data, u, v, color_values = inputs.data, inputs.u, inputs.v, inputs.color_values
    magnitude = np.hypot(u, v)
    coords = data.grid.coordinate_arrays()

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
        if color_values is not None:
            color_values = color_values[::s, ::s]
        if isinstance(lw_arg, np.ndarray):
            lw_arg = lw_arg[::s, ::s]

    # Zero out vectors outside the magnitude range so streamplot skips them.
    # When units is provided, thresholds are in display units — convert to
    # code units via the same factor used by in_units().
    if magnitude_min is not None or magnitude_max is not None:
        if units is not None:
            comp_u = inputs.components[0]
            code = resolve_field_values(data, comp_u, None)
            display = resolve_field_values(data, comp_u, units)
            nonzero = np.abs(code) > 0
            if np.any(nonzero):
                scale = float(np.nanmedian(display[nonzero] / code[nonzero]))
            else:
                scale = 1.0
            if magnitude_min is not None:
                magnitude_min = magnitude_min / scale
            if magnitude_max is not None:
                magnitude_max = magnitude_max / scale

        mag = np.sqrt(u**2 + v**2)
        mask = np.ones_like(mag, dtype=bool)
        if magnitude_min is not None:
            mask &= mag >= magnitude_min
        if magnitude_max is not None:
            mask &= mag <= magnitude_max
        u = np.where(mask, u, 0.0)
        v = np.where(mask, v, 0.0)

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        if color_values is not None:
            from matplotlib.colors import Normalize

            norm = (
                Normalize(vmin=vmin, vmax=vmax)
                if vmin is not None or vmax is not None
                else None
            )
            stream = ax.streamplot(
                coords[0],
                coords[1],
                u.T,
                v.T,
                color=color_values.T,
                cmap=inputs.colormap,
                norm=norm,
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
            # stream.arrows.set_alpha doesn't work — arrows are individual
            # FancyArrowPatch children
            from matplotlib.patches import FancyArrowPatch

            for child in ax.get_children():
                if isinstance(child, FancyArrowPatch):
                    child.set_alpha(alpha)

        if color_values is not None and colorbar:
            cb_label = field_label(inputs.info, unit_str=units or "")
            attach_colorbar(
                fig, ax, stream.lines, cb_label, colorbar, extremes=extremes
            )
        if color is not None and legend is not False:
            default_lw: float = _theme_val("line_width", 1.0)
            lw = linewidth if isinstance(linewidth, (int, float)) else default_lw
            entry = LegendEntry(
                label=legend if isinstance(legend, str) else field,
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
            add_legend(ax, entry)

        if title is None:
            title = figure_title(inputs.info, step=step, time=time)
        xlabel, ylabel = plane_axis_labels(data, coord_units)
        finish_axes(
            fig,
            ax,
            theme,
            owns_figure=owns_figure,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            aspect="equal",
            badge=badge,
            step=step,
            time=time,
        )

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
    vmin: float | None = None,
    vmax: float | None = None,
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
    vmin, vmax : float or None
        Color limits for the arrow colors. ``None`` (default) autoscales.
        Ignored when *color* is set.
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
    extremes : "semi", "transparent", "darken", or None
        Colorbar out-of-range indicator style. ``"semi"`` (default)
        uses semi-transparent extension colors; ``"transparent"``
        hides them; ``"darken"`` darkens the endpoint colors;
        ``None`` leaves matplotlib defaults untouched.
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
    badge : str or None
        Corner badge text (run label, timestamp, ...). ``None``
        (default) draws no badge.
    save : str or Path or None
        Path to write the figure to. When given, the figure is saved
        and closed; when ``None`` (default) it is left open for the
        caller to display or modify further.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    owns_figure = ax is None
    theme = _resolve_theme_arg(theme)
    inputs = _vector_prelude(
        data,
        field,
        plane=plane,
        color=color,
        color_field=color_field,
        units=units,
        theme=theme,
        cmap=cmap,
    )
    data, u, v, color_values = inputs.data, inputs.u, inputs.v, inputs.color_values
    coords = data.grid.coordinate_arrays()

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

        if color_values is not None:
            from matplotlib.colors import Normalize

            quiv = ax.quiver(
                xx.T,
                yy.T,
                u_sub.T,
                v_sub.T,
                color_values[::s0, ::s1].T,
                cmap=inputs.colormap,
                norm=Normalize(vmin=vmin, vmax=vmax),
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

        if color_values is not None and colorbar:
            cb_label = field_label(inputs.info, unit_str=units or "")
            attach_colorbar(fig, ax, quiv, cb_label, colorbar, extremes=extremes)
        if color is not None and legend is not False:
            entry = LegendEntry(
                label=legend if isinstance(legend, str) else field,
                color=color,
                alpha=alpha,
            )
            add_legend(ax, entry)

        if title is None:
            title = figure_title(inputs.info, step=step, time=time)
        xlabel, ylabel = plane_axis_labels(data, coord_units)
        finish_axes(
            fig,
            ax,
            theme,
            owns_figure=owns_figure,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            aspect="equal",
            badge=badge,
            step=step,
            time=time,
        )

    maybe_save(fig, save)
    return fig, ax
