"""Shared field resolution, midplane defaulting, and axes helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pypic.plotting._labels import axis_label

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.dataset import FieldDataset
    from pypic.plotting.styles import PlotTheme
    from pypic.selections import PlaneSelection
    from pypic.types import FloatArray


def require_plottable_grid(data: FieldDataset) -> None:
    """Raise if the grid is too small for 2D plotting."""
    dims = data.grid.dimensions
    if any(d < 2 for d in dims[:2]):
        msg = f"Grid too small to plot: dimensions {dims}"
        raise ValueError(msg)


def resolve_coord_units(
    coord_units: str | tuple[str, str] | None,
) -> tuple[str, str]:
    """Normalize *coord_units* to a per-axis ``(x_unit, y_unit)`` pair."""
    if coord_units is None:
        return ("", "")
    if isinstance(coord_units, tuple):
        return coord_units
    return (coord_units, coord_units)


def plane_axis_labels(
    data: FieldDataset, coord_units: str | tuple[str, str] | None
) -> tuple[str, str]:
    """Label the two surviving axes of a plane plot."""
    surviving = data.grid.surviving_axis_names
    x_unit, y_unit = resolve_coord_units(coord_units)
    return (
        axis_label(surviving[0], unit_str=x_unit),
        axis_label(surviving[1], unit_str=y_unit),
    )


def resolve_field_values(
    data: FieldDataset, field: str, units: str | None
) -> FloatArray:
    """Resolve a field name to its values, with optional unit conversion.

    Handles stored fields, aliases, and derived quantities. When *units*
    is provided, delegates to ``data.in_units()`` which handles both
    stored and derived fields in a single call.

    Parameters
    ----------
    data : FieldDataset
        Source dataset.
    field : str
        Field name (canonical, alias, or derived).
    units : str | None
        Display units (e.g. ``"nT"``). ``None`` returns code units.

    Returns
    -------
    FloatArray
    """
    if units is not None:
        return data.in_units(field, units)
    if data.has_field(field):
        return data[field]
    return data.compute(field)


def get_or_create_axes(
    theme: PlotTheme,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure if *ax* is ``None``.

    When *ax* is provided, applies the theme to the existing figure so
    that axes created outside ``use_theme()`` still match the palette.

    Parameters
    ----------
    theme : PlotTheme
        Active plot theme.
    ax : Axes or None
        Existing axes, or ``None`` to create a new figure.
    figsize : tuple or None
        Figure size (only used when creating a new figure).

    Returns
    -------
    tuple[Figure, Axes]
    """
    import matplotlib.pyplot as plt

    from pypic.plotting.styles import apply_theme_to_figure

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    raw_fig = ax.get_figure()
    if raw_fig is None:
        msg = "Axes is not attached to a figure"
        raise RuntimeError(msg)
    # Narrow Figure | SubFigure → Figure: pypic only creates top-level
    # figures, never subfigures, so this is safe in practice.
    fig = cast("Figure", raw_fig)
    apply_theme_to_figure(fig, theme)
    return fig, ax


def finish_axes(
    fig: Figure,
    ax: Axes,
    theme: PlotTheme,
    *,
    owns_figure: bool,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    aspect: Literal["equal"] | None = None,
    minor_grid: bool = False,
    badge: bool = False,
    step: int | None = None,
    time: float | None = None,
) -> None:
    """Label, grid, bake and round *ax* once its artists are drawn.

    ``None`` leaves a label or title as it was, so plots layered onto shared
    axes keep each other's. Only the call that created *fig* lays it out, and
    rounding comes last because it measures the laid-out axes.
    """
    from pypic.plotting.styles import apply_grid, apply_rounding, bake_theme

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if aspect is not None:
        ax.set_aspect(aspect)
    apply_grid(ax, theme, minor=minor_grid)
    bake_theme(ax, theme)
    if badge and (step is not None or time is not None):
        from pypic.plotting._badge import add_badge

        add_badge(ax, step=step, time=time)
    if owns_figure:
        fig.tight_layout()
    apply_rounding(ax)


def default_midplane(data: FieldDataset) -> PlaneSelection | None:
    """Return a midplane ``PlaneSelection`` for 3D data, ``None`` otherwise.

    The slice is along the last axis (e.g. *z* for Cartesian).

    Parameters
    ----------
    data : FieldDataset
        Input dataset.

    Returns
    -------
    PlaneSelection | None
    """
    if len(data.grid.dimensions) == 3:
        from pypic.selections import PlaneSelection

        normal = data.grid.geometry.axis_names[2]
        return PlaneSelection(normal=normal)
    return None


def prepare_data(data: FieldDataset, plane: PlaneSelection | None) -> FieldDataset:
    """Apply default midplane (if 3D) and validate for 2D plotting.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    plane : PlaneSelection | None
        Explicit plane selection. ``None`` auto-selects the midplane
        for 3D data.

    Returns
    -------
    FieldDataset
        2D dataset ready for plotting.
    """
    if plane is None:
        plane = default_midplane(data)
    if plane is not None:
        data = plane.apply(data)
    require_plottable_grid(data)
    return data


def maybe_save(
    fig: Figure,
    save: str | None,
    *,
    dpi: int | None = None,
    fmt: str | None = None,
) -> None:
    """Save figure to *save* path and close, if *save* is not None.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    save : str | None
        Output path. ``None`` is a no-op.
    dpi : int | None
        Override DPI for the saved file.
    fmt : str | None
        Override format (e.g. ``"png"``, ``"pdf"``).
    """
    if save is not None:
        import matplotlib.pyplot as plt

        kwargs: dict[str, object] = {}
        if dpi is not None:
            kwargs["dpi"] = dpi
        if fmt is not None:
            kwargs["format"] = fmt
        fig.savefig(save, **kwargs)  # type: ignore[arg-type]
        plt.close(fig)
