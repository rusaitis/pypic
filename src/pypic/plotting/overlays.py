"""Composite plots: scalar fields with vector overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import ThemeArg
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection


def plot_field_with_vectors(
    data: FieldDataset,
    scalar_field: str,
    vector_field: str,
    *,
    vector_style: Literal["streamlines", "quiver"] = "streamlines",
    vector_color: str = "white",
    vector_alpha: float = 0.6,
    vector_linewidth: float = 0.8,
    stride: int | tuple[int, int] = 2,
    density: float = 1.5,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    coord_units: str | tuple[str, str] | None = None,
    theme: ThemeArg = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
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
) -> tuple[Figure, Axes]:
    r"""Plot a scalar field with vector field streamlines or arrows overlaid.

    Combines :func:`plot_field_slice` and :func:`plot_streamlines` (or
    :func:`plot_quiver`) in a single call, handling colorbar and legend
    placement automatically.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    scalar_field : str
        Scalar field for the color map (e.g. ``"|B|"``, ``"rho_m"``).
    vector_field : str
        Vector field prefix for overlay (e.g. ``"B"``, ``"V"``).
    vector_style : "streamlines" or "quiver"
        Vector visualization style.
    vector_color : str
        Uniform color for vector overlay lines/arrows.
    vector_alpha : float
        Transparency of the vector overlay.
    vector_linewidth : float
        Line width for streamlines or arrow shafts.
    stride : int or tuple[int, int]
        Subsampling for quiver mode (ignored for streamlines).
    density : float
        Streamline density (ignored for quiver mode).
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    units : str | None
        Display units for scalar field values.
    coord_units : str, tuple[str, str], or None
        Display units for coordinate axes.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override colormap for the scalar field.
    vmin, vmax : float | None
        Color limits for the scalar field.
    log_scale : bool
        Use logarithmic color mapping for the scalar field.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number for the title.
    time : float | None
        Simulation time for the title.
    ax : Axes | None
        Existing axes to draw on. ``None`` creates a new figure.
    colorbar : bool or "inset"
        Colorbar mode for the scalar field.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
    legend : bool or str
        Vector legend overlay. ``True`` uses the field prefix as label.
    figsize : tuple[float, float] | None
        Figure size override.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import prepare_data
    from pypic.plotting.slices import plot_field_slice
    from pypic.plotting.vectors import plot_quiver, plot_streamlines

    data = prepare_data(data, plane)

    fig, ax = plot_field_slice(
        data,
        scalar_field,
        units=units,
        coord_units=coord_units,
        theme=theme,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log_scale=log_scale,
        title=title,
        step=step,
        time=time,
        ax=ax,
        colorbar=colorbar,
        extremes=extremes,
        figsize=figsize,
    )

    # Vector overlay on already-sliced data (no plane= to avoid double-slicing)
    match vector_style:
        case "streamlines":
            plot_streamlines(
                data,
                vector_field,
                color=vector_color,
                alpha=vector_alpha,
                linewidth=vector_linewidth,
                density=density,
                theme=theme,
                ax=ax,
                colorbar=False,
                legend=legend,
            )
        case "quiver":
            plot_quiver(
                data,
                vector_field,
                color=vector_color,
                alpha=vector_alpha,
                stride=stride,
                theme=theme,
                ax=ax,
                colorbar=False,
                legend=legend,
            )

    if badge and (step is not None or time is not None):
        from pypic.plotting._badge import add_badge

        add_badge(ax, step=step, time=time)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax
