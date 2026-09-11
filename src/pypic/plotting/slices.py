"""2D field slice plotting and contour overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.dataset import FieldDataset
    from pypic.plotting._badge import OverlayVariant
    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import ThemeArg
    from pypic.selections import PlaneSelection


def plot_field_slice(
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    coord_units: str | tuple[str, str] | None = None,
    theme: ThemeArg = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    symmetric: bool | None = None,
    log_scale: bool = False,
    symlog: bool = False,
    linthresh: float | None = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    colorbar: bool | Literal["inset"] = True,
    colorbar_label: str | None = None,
    colorbar_variant: OverlayVariant | None = None,
    colorbar_ticks: list[float] | None = None,
    extremes: ExtremesMode = "semi",
    badge: bool = False,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a 2D slice of a scalar field.

    If *data* is 3D and no *plane* is given, defaults to a midplane
    slice along the last axis (``PlaneSelection(normal=axis_names[2])``).

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    field : str
        Field name — canonical, alias, or derived (e.g. ``"|B|"``).
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    units : str | None
        Display units for field values (e.g. ``"nT"``).
    coord_units : str, tuple[str, str], or None
        Display units for coordinate axes. A single string applies to
        both axes; a tuple ``(x_unit, y_unit)`` labels each independently.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override automatic colormap selection.
    vmin, vmax : float | None
        Color limits. ``None`` for auto.
    alpha : float
        Mesh transparency (0 = invisible, 1 = opaque). Useful for
        overlaying semi-transparent scalar fields.
    symmetric : bool | None
        Force symmetric color limits around zero. ``None`` auto-detects
        (``True`` when diverging colormap is selected).
    log_scale : bool
        Use logarithmic color mapping (``LogNorm``). Non-positive values
        are masked with NaN. Ignored when *symmetric* is ``True``.
    symlog : bool
        Use symmetric-log color mapping (``SymLogNorm``). Combines a
        linear region around zero with logarithmic tails — ideal for
        signed fields with large dynamic range (current density,
        vorticity). Mutually exclusive with *log_scale*.
    linthresh : float | None
        Linear threshold for symlog. Values within ``[-linthresh,
        linthresh]`` are mapped linearly; outside is logarithmic.
        ``None`` auto-detects from ``median(|nonzero values|)``.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    colorbar : bool
        Whether to add a colorbar.
    extremes : "semi", "transparent", "darken", or None
        How to style values outside ``[vmin, vmax]``.
        ``"semi"`` (default) — semi-transparent (~30% opacity).
        ``"transparent"`` — fully invisible.
        ``"darken"`` — darkened endpoint colors.
        ``None`` — matplotlib default (no modification).
    figsize : tuple[float, float] | None
        Figure size override.
    vmin : float or None
        Lower color limit. ``None`` (default) autoscales.
    vmax : float or None
        Upper color limit. ``None`` (default) autoscales.
    colorbar_label : str or None
        Override the colorbar label. ``None`` (default) builds
        one from the field's registry metadata and units.
    colorbar_variant : str or None
        Colorbar placement variant. ``None`` (default) uses
        the active theme's choice.
    colorbar_ticks : Sequence[float] or None
        Explicit colorbar tick positions. ``None`` (default)
        lets matplotlib choose.
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

    import warnings

    import numpy as np

    from pypic.plotting._colorbar import attach_colorbar
    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_field_colormap,
        resolve_norm,
    )
    from pypic.plotting._labels import field_label, figure_title
    from pypic.plotting._resolve import (
        finish_axes,
        get_or_create_axes,
        maybe_save,
        plane_axis_labels,
        prepare_data,
        resolve_field_values,
    )
    from pypic.plotting.styles import _resolve_theme_arg, use_theme

    owns_figure = ax is None
    theme = _resolve_theme_arg(theme)
    data = prepare_data(data, plane)
    values = resolve_field_values(data, field, units)

    if np.all(np.isnan(values)):
        warnings.warn(f"Field {field!r} is entirely NaN", stacklevel=2)

    info = data.field_info(field)
    coords = data.grid.coordinate_arrays()

    _, colormap = resolve_field_colormap(field, values, theme, info=info, cmap=cmap)
    if symmetric is None:
        symmetric = not is_positive_definite(field, values, info)
    norm = resolve_norm(
        values,
        symmetric=symmetric,
        log_scale=log_scale,
        symlog=symlog,
        vmin=vmin,
        vmax=vmax,
        linthresh=linthresh,
    )

    # Transparent extremes: mask out-of-range values so pcolormesh
    # renders them as truly transparent (not black)
    if extremes == "transparent" and norm.vmin is not None and norm.vmax is not None:
        in_range = (values >= norm.vmin) & (values <= norm.vmax)
        values = np.where(in_range, values, np.nan)

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)
        mesh = ax.pcolormesh(
            coords[0],
            coords[1],
            values.T,
            shading="auto",
            cmap=colormap,
            alpha=alpha,
            norm=norm,
        )

        unit_str = units or ""
        cb_label = (
            colorbar_label
            if colorbar_label is not None
            else field_label(info, unit_str=unit_str)
        )
        attach_colorbar(
            fig,
            ax,
            mesh,
            cb_label,
            colorbar,
            extremes=extremes,
            variant=colorbar_variant,
            ticks=colorbar_ticks,
        )

        if title is None:
            title = figure_title(info, step=step, time=time)
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


def add_contours(
    ax: Axes,
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    levels: int | list[float] = 5,
    colors: str | list[str] | None = None,
    linewidths: float = 0.5,
    alpha: float = 0.7,
    linestyles: str = "solid",
    labels: bool = False,
    label_fontsize: float | None = None,
    **kwargs: Any,  # noqa: ANN401 — contour passthrough
) -> object:
    r"""Add contour lines to existing axes from a scalar field.

    Works as an overlay on `plot_field_slice` or any other 2D plot.

    Parameters
    ----------
    ax : Axes
        Target axes (must already have coordinate limits set).
    data : FieldDataset
        Input dataset (2D or 3D).
    field : str
        Scalar field name for contouring.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    units : str | None
        Display units for field values.
    levels : int or list[float]
        Number of contour levels, or explicit level values.
    colors : str, list[str], or None
        Line color(s). ``None`` uses the first color from the theme's
        color cycle (falls back to ``"black"``).
    linewidths : float
        Contour line width.
    alpha : float
        Line transparency.
    linestyles : str
        Line style (``"solid"``, ``"dashed"``, ``"dotted"``).
    labels : bool
        Whether to add inline contour labels.
    label_fontsize : float or None
        Font size for contour labels (when *labels* is ``True``).
        ``None`` (default) reads from ``theme.contour_label_fontsize``.
    **kwargs
        Passed to ``ax.contour()``.

    Returns
    -------
    QuadContourSet
        The contour set added to *ax*.

    Examples
    --------
    >>> fig, ax = plot_field_slice(data, "|B|")  # doctest: +SKIP
    >>> add_contours(ax, data, "P", levels=8, colors="white")  # doctest: +SKIP
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import prepare_data, resolve_field_values
    from pypic.plotting.styles import _theme_val

    if colors is None:
        cycle = _theme_val("color_cycle", ())
        colors = cycle[0] if cycle else "black"

    data = prepare_data(data, plane)
    values = resolve_field_values(data, field, units)
    coords = data.grid.coordinate_arrays()

    cs = ax.contour(
        coords[0],
        coords[1],
        values.T,
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        alpha=alpha,
        linestyles=linestyles,
        **kwargs,
    )

    if labels:
        from pypic.plotting.styles import _theme_val

        fs = (
            label_fontsize
            if label_fontsize is not None
            else _theme_val("contour_label_fontsize", 7.0)
        )
        ax.clabel(cs, inline=True, fontsize=fs)

    return cs
