"""Line plots: spatial profiles and time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.containers import TabularData
    from pypic.dataset import FieldDataset
    from pypic.plotting.styles import ThemeArg


def plot_line(
    data: FieldDataset,
    field: str,
    *,
    axis: str | None = None,
    index: dict[str, int] | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: ThemeArg = None,
    label: str | None = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> tuple[Figure, Axes]:
    r"""Plot a 1D spatial profile of a field along one axis.

    For 2D/3D data, other axes are sliced at the given *index* values
    (default: midplane). For 1D data, the single axis is used directly.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (1D, 2D, or 3D).
    field : str
        Field name — canonical, alias, or derived (e.g. ``"|B|"``).
    axis : str | None
        Axis to plot along (e.g. ``"x"``). Required for 2D/3D data.
        ``None`` auto-selects the only axis for 1D data.
    index : dict[str, int] | None
        Fix non-plot axes at these integer indices. ``None`` uses
        midplane for each sliced axis.
    units : str | None
        Display units for field values (e.g. ``"nT"``).
    coord_units : str | None
        Display units for the coordinate axis.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    label : str | None
        Legend label. ``None`` omits legend entry.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.plot()`` (color, linestyle, linewidth, etc.).
    save : str or Path or None
        Path to write the figure to. When given, the figure is saved
        and closed; when ``None`` (default) it is left open for the
        caller to display or modify further.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._labels import axis_label, field_label, figure_title
    from pypic.plotting._resolve import (
        finish_axes,
        get_or_create_axes,
        maybe_save,
        resolve_field_values,
    )
    from pypic.plotting.styles import _resolve_theme_arg, style_legend, use_theme

    owns_figure = ax is None
    theme = _resolve_theme_arg(theme)

    ndim = len(data.grid.dimensions)
    axis_names = list(data.grid.surviving_axis_names)

    if ndim == 1:
        plot_axis = axis_names[0]
    elif axis is not None:
        if axis not in axis_names:
            msg = f"axis={axis!r} not in dataset axes {axis_names}"
            raise ValueError(msg)
        plot_axis = axis
    else:
        msg = f"axis is required for {ndim}D data (axes: {axis_names})"
        raise ValueError(msg)

    slice_axes = [a for a in axis_names if a != plot_axis]
    if slice_axes:
        indexers: dict[str, int] = {}
        for a in slice_axes:
            if index is not None and a in index:
                indexers[a] = index[a]
            else:
                dim_idx = axis_names.index(a)
                indexers[a] = data.grid.dimensions[dim_idx] // 2
        data = data.isel(indexers)

    import warnings

    import numpy as np

    values = resolve_field_values(data, field, units)

    if values.shape[0] < 2:
        warnings.warn(
            f"Field {field!r} has fewer than 2 points; line will not be visible",
            stacklevel=2,
        )

    if np.all(np.isnan(values)):
        warnings.warn(f"Field {field!r} is entirely NaN", stacklevel=2)

    info = data.field_info(field)
    coord = data.grid.coordinate_arrays()[0]
    if title is None and (step is not None or time is not None):
        title = figure_title(info, step=step, time=time)

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        ax.plot(coord, values, label=label, **kwargs)
        if label is not None:
            ax.legend()
            style_legend(ax)
        finish_axes(
            fig,
            ax,
            theme,
            owns_figure=owns_figure,
            xlabel=axis_label(plot_axis, unit_str=coord_units or ""),
            ylabel=field_label(info, unit_str=units or ""),
            title=title,
        )

    maybe_save(fig, save)
    return fig, ax


def plot_lines(
    data: FieldDataset,
    fields: list[str],
    *,
    axis: str | None = None,
    index: dict[str, int] | None = None,
    labels: list[str] | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: ThemeArg = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> tuple[Figure, Axes]:
    r"""Plot multiple fields overlaid on the same axes.

    Convenience wrapper around `plot_line` that handles axes
    reuse, automatic color cycling, and legend display.

    Parameters
    ----------
    data : FieldDataset
        Input dataset.
    fields : list[str]
        Field names to plot (one line per field).
    axis : str | None
        Axis to plot along (required for 2D/3D data).
    index : dict[str, int] | None
        Fix non-plot axes at these indices.
    labels : list[str] | None
        Legend labels. ``None`` uses field names.
    units : str | None
        Display units for field values.
    coord_units : str | None
        Display units for coordinate axis.
    theme : PlotTheme | None
        Plot theme. ``None`` uses default.
    title : str | None
        Axes title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    save : str | None
        Save figure to this path (and close).
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.plot()`` (color, linestyle, etc.).

    Returns
    -------
    tuple[Figure, Axes]
    """
    if labels is None:
        labels = fields

    fig = None
    for field_name, lbl in zip(fields, labels, strict=True):
        fig, ax = plot_line(
            data,
            field_name,
            axis=axis,
            index=index,
            units=units,
            coord_units=coord_units,
            theme=theme,
            label=lbl,
            title=title,
            step=step,
            time=time,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    assert fig is not None
    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax  # type: ignore[return-value]


def plot_line_comparison(
    datasets: list[FieldDataset],
    field: str,
    *,
    axis: str | None = None,
    index: dict[str, int] | None = None,
    labels: list[str] | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: ThemeArg = None,
    title: str | None = None,
    ax: Axes | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> tuple[Figure, Axes]:
    r"""Compare 1D profiles of the same field from multiple datasets.

    Overlays one line per dataset with automatic color cycling and
    legend. Useful for comparing simulation runs with different
    parameters or resolutions.

    Parameters
    ----------
    datasets : list[FieldDataset]
        Datasets to compare (must share compatible axes).
    field : str
        Field name to plot from each dataset.
    axis : str | None
        Axis to plot along (required for 2D/3D data).
    index : dict[str, int] | None
        Fix non-plot axes at these indices.
    labels : list[str] | None
        Legend labels (one per dataset). ``None`` uses ``"run 0"``,
        ``"run 1"``, etc.
    units : str | None
        Display units for field values.
    coord_units : str | None
        Display units for the coordinate axis.
    theme : PlotTheme | None
        Plot theme. ``None`` uses default.
    title : str | None
        Axes title.
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    save : str | None
        Save figure to this path.
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.plot()`` (linestyle, linewidth, etc.).

    Returns
    -------
    tuple[Figure, Axes]
    """
    if labels is None:
        labels = [f"run {i}" for i in range(len(datasets))]

    fig = None
    for ds, lbl in zip(datasets, labels, strict=True):
        fig, ax = plot_line(
            ds,
            field,
            axis=axis,
            index=index,
            units=units,
            coord_units=coord_units,
            theme=theme,
            label=lbl,
            title=title,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    assert fig is not None
    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax  # type: ignore[return-value]


def plot_time_series(
    data: TabularData,
    columns: str | list[str],
    *,
    x_column: str | None = None,
    theme: ThemeArg = None,
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> tuple[Figure, Axes]:
    r"""Plot one or more columns from tabular data as time series.

    Parameters
    ----------
    data : TabularData
        Tabular data source (e.g. conserved quantities).
    columns : str | list[str]
        Column name(s) to plot.
    x_column : str | None
        Column for the x-axis. ``None`` uses ``data.index``.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    labels : list[str] | None
        Legend labels. ``None`` uses column names.
    title : str | None
        Axes title.
    xlabel : str | None
        X-axis label. ``None`` uses *x_column* name or ``"Cycle"``.
    ylabel : str | None
        Y-axis label.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    figsize : tuple[float, float] | None
        Figure size override.
    legend : bool
        Whether to show a legend (default ``True``).
    **kwargs
        Passed to ``ax.plot()`` (color, linestyle, linewidth, etc.).
    save : str or Path or None
        Path to write the figure to. When given, the figure is saved
        and closed; when ``None`` (default) it is left open for the
        caller to display or modify further.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import finish_axes, get_or_create_axes, maybe_save
    from pypic.plotting.styles import _resolve_theme_arg, style_legend, use_theme

    owns_figure = ax is None
    theme = _resolve_theme_arg(theme)

    if isinstance(columns, str):
        columns = [columns]

    x = data[x_column] if x_column is not None else data.index

    if labels is None:
        labels = columns
    if xlabel is None:
        xlabel = x_column or data.index_column or "Cycle"

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        for col, lbl in zip(columns, labels, strict=True):
            ax.plot(x, data[col], label=lbl, **kwargs)
        if legend and (len(columns) > 1 or labels != columns):
            ax.legend()
            style_legend(ax)
        finish_axes(
            fig,
            ax,
            theme,
            owns_figure=owns_figure,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    maybe_save(fig, save)
    return fig, ax
