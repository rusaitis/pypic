"""Line plots: spatial profiles and time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset, TabularData


def plot_line(
    data: FieldDataset,
    field: str,
    *,
    axis: str | None = None,
    index: dict[str, int] | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: PlotTheme | None = None,
    label: str | None = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
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

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._labels import axis_label, field_label, figure_title
    from pypic.plotting._resolve import get_or_create_axes, resolve_field_values
    from pypic.plotting.styles import DEFAULT, apply_grid, style_legend, use_theme

    if theme is None:
        theme = DEFAULT

    ndim = len(data.grid.dimensions)
    axis_names = list(data.grid.geometry.axis_names[:ndim])

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

    values = resolve_field_values(data, field, units)

    info = data.field_info(field)
    coord = data.grid.coordinate_arrays()[0]

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        ax.plot(coord, values, label=label, **kwargs)
        ax.set_xlabel(axis_label(plot_axis, unit_str=coord_units or ""))
        ax.set_ylabel(field_label(info, unit_str=units or ""))
        apply_grid(ax, theme)

        if title is not None:
            ax.set_title(title)
        elif step is not None or time is not None:
            ax.set_title(figure_title(info, step=step, time=time))

        if label is not None:
            ax.legend()
            style_legend(ax)

    return fig, ax


def plot_time_series(
    data: TabularData,
    columns: str | list[str],
    *,
    x_column: str | None = None,
    theme: PlotTheme | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
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

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import get_or_create_axes
    from pypic.plotting.styles import DEFAULT, apply_grid, style_legend, use_theme

    if theme is None:
        theme = DEFAULT

    if isinstance(columns, str):
        columns = [columns]

    x = data[x_column] if x_column is not None else data.index

    if labels is None:
        labels = columns

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        for col, lbl in zip(columns, labels, strict=True):
            ax.plot(x, data[col], label=lbl, **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif x_column is not None:
            ax.set_xlabel(x_column)
        else:
            ax.set_xlabel(
                data.index_column if data.index_column is not None else "Cycle"
            )

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        apply_grid(ax, theme)

        if legend and (len(columns) > 1 or labels != columns):
            ax.legend()
            style_legend(ax)

    return fig, ax
