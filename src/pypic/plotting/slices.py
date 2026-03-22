"""2D field slice plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection


def plot_field_slice(
    data: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: PlotTheme | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool | None = None,
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    ax: Axes | None = None,
    colorbar: bool | Literal["inset"] = True,
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
    coord_units : str | None
        Display units for coordinate axes.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override automatic colormap selection.
    vmin, vmax : float | None
        Color limits. ``None`` for auto.
    symmetric : bool | None
        Force symmetric color limits around zero. ``None`` auto-detects
        (``True`` when diverging colormap is selected).
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
    figsize : tuple[float, float] | None
        Figure size override.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()
    import matplotlib.pyplot as plt

    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_colormap,
        symmetric_clim,
    )
    from pypic.plotting._labels import axis_label, field_label, figure_title
    from pypic.plotting._resolve import default_midplane, resolve_field_values
    from pypic.plotting.styles import (
        DEFAULT,
        apply_grid,
        apply_theme_to_figure,
        use_theme,
    )

    if theme is None:
        theme = DEFAULT

    if plane is None:
        plane = default_midplane(data)
    if plane is not None:
        data = plane.apply(data)

    values = resolve_field_values(data, field, units)

    info = data.field_info(field)
    coords = data.grid.coordinate_arrays()
    surviving_axes = data.grid.geometry.axis_names[: len(data.grid.dimensions)]

    cmap_name = resolve_colormap(
        field, values, theme, info=info, cmap=cmap if isinstance(cmap, str) else None
    )

    use_symmetric = symmetric
    if use_symmetric is None:
        use_symmetric = not is_positive_definite(field, values, info)

    if vmin is None and vmax is None and use_symmetric:
        vmin, vmax = symmetric_clim(values)

    with use_theme(theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]
            apply_theme_to_figure(fig, theme)

        mesh = ax.pcolormesh(
            coords[0],
            coords[1],
            values.T,
            shading="auto",
            cmap=cmap if not isinstance(cmap, str) else cmap_name,
            vmin=vmin,
            vmax=vmax,
        )

        unit_str = units if units else ""
        if colorbar:
            cb_label = field_label(info, unit_str=unit_str)
            if colorbar == "inset":
                from pypic.plotting._colorbar import add_inset_colorbar

                add_inset_colorbar(ax, mesh, cb_label)
            else:
                from pypic.plotting._colorbar import add_colorbar

                add_colorbar(fig, ax, mesh, cb_label)

        ax.set_xlabel(axis_label(surviving_axes[0], unit_str=coord_units or ""))
        ax.set_ylabel(axis_label(surviving_axes[1], unit_str=coord_units or ""))
        ax.set_aspect("equal")
        apply_grid(ax, theme)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(figure_title(info, step=step, time=time))

        fig.tight_layout()

    return fig, ax


