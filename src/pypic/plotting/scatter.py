"""Field-vs-field scatter and density plots for correlation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    from pypic.plotting.styles import ThemeArg
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection


def plot_scatter(
    data: FieldDataset,
    field_x: str,
    field_y: str,
    *,
    plane: PlaneSelection | None = None,
    color_field: str | None = None,
    units_x: str | None = None,
    units_y: str | None = None,
    color_units: str | None = None,
    theme: ThemeArg = None,
    cmap: str | None = None,
    alpha: float = 0.3,
    marker_size: float = 1.0,
    density: bool = False,
    bins: int = 80,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Scatter or density plot of two field quantities.

    Plots every grid point's value of *field_x* against *field_y*.
    Useful for equation-of-state analysis ($P$ vs $\rho$), anisotropy
    studies ($P_\parallel$ vs $P_\perp$), and general correlation
    exploration.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    field_x : str
        Field for the horizontal axis.
    field_y : str
        Field for the vertical axis.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    color_field : str | None
        Optional third field for point color (e.g. ``"beta"``).
        ``None`` uses uniform color in scatter mode, or count density
        in ``density=True`` mode.
    units_x : str | None
        Display units for *field_x*.
    units_y : str | None
        Display units for *field_y*.
    color_units : str | None
        Display units for *color_field*.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | None
        Colormap override for colored scatter or density mode.
    alpha : float
        Point transparency (scatter mode only).
    marker_size : float
        Marker size in points² (scatter mode only).
    density : bool
        Use a hexbin density plot instead of a scatter plot. Better
        for large grids where individual points overlap heavily.
    bins : int
        Hexbin grid size (``density=True`` only).
    log_x : bool
        Logarithmic x-axis.
    log_y : bool
        Logarithmic y-axis.
    title : str | None
        Axes title.
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    colorbar : bool
        Show a colorbar when *color_field* is set or ``density=True``.
    save : str | None
        Save figure to this path.
    figsize : tuple[float, float] | None
        Figure size override.

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._colorbar import _style_colorbar
    from pypic.plotting._labels import field_label
    from pypic.plotting._resolve import (
        get_or_create_axes,
        prepare_data,
        resolve_field_values,
    )
    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    owned = ax is None
    theme = _resolve_theme_arg(theme)
    data = prepare_data(data, plane)

    x = resolve_field_values(data, field_x, units_x).ravel()
    y = resolve_field_values(data, field_y, units_y).ravel()

    color = None
    if color_field is not None:
        color = resolve_field_values(data, color_field, color_units).ravel()

    # Filter NaN
    mask = np.isfinite(x) & np.isfinite(y)
    if color is not None:
        mask &= np.isfinite(color)
    x, y = x[mask], y[mask]
    if color is not None:
        color = color[mask]

    info_x = data.field_info(field_x)
    info_y = data.field_info(field_y)

    if cmap is None:
        cmap = theme.sequential_cmap

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        mappable: ScalarMappable | None = None
        if density:
            hb = ax.hexbin(
                x,
                y,
                C=color,
                gridsize=bins,
                cmap=cmap,
                mincnt=1,
                xscale="log" if log_x else "linear",
                yscale="log" if log_y else "linear",
            )
            mappable = hb
        else:
            if color is not None:
                sc = ax.scatter(
                    x,
                    y,
                    c=color,
                    s=marker_size,
                    alpha=alpha,
                    cmap=cmap,
                    edgecolors="none",
                )
                mappable = sc
            else:
                ax.scatter(
                    x,
                    y,
                    s=marker_size,
                    alpha=alpha,
                    edgecolors="none",
                )

        if log_x and not density:
            ax.set_xscale("log")
        if log_y and not density:
            ax.set_yscale("log")

        ax.set_xlabel(field_label(info_x, unit_str=units_x or ""))
        ax.set_ylabel(field_label(info_y, unit_str=units_y or ""))
        if title is not None:
            ax.set_title(title)
        apply_grid(ax, theme)

        if colorbar and mappable is not None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            from pypic.plotting.styles import _theme_val

            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right",
                size=_theme_val("colorbar_width", "4%"),
                pad=_theme_val("colorbar_pad", 0.05),
            )
            cb = fig.colorbar(mappable, cax=cax)
            if color_field is not None:
                info_c = data.field_info(color_field)
                cb_label = field_label(info_c, unit_str=color_units or "")
            elif density:
                cb_label = "count"
            else:
                cb_label = ""
            _style_colorbar(cb, cb_label)

        if owned:
            fig.tight_layout()
            apply_rounding(ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax
