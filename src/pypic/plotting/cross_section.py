"""Cross-section plot: 2D field slice with 1D cut profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.dataset import FieldDataset
    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import ThemeArg
    from pypic.selections import PlaneSelection


def plot_cross_section(
    data: FieldDataset,
    field: str,
    *,
    cut_axis: str,
    cut_index: int | None = None,
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
    title: str | None = None,
    step: int | None = None,
    time: float | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    cut_color: str | None = None,
    cut_linestyle: str = "--",
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    r"""Two-panel figure: 2D field slice with a 1D cut profile below.

    The top panel shows the scalar field via :func:`plot_field_slice`
    with a dashed line marking the cut location. The bottom panel
    shows the 1D profile along that cut.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    field : str
        Scalar field name.
    cut_axis : str
        Axis along which to extract the 1D profile (e.g. ``"x"``).
        The cut is taken at *cut_index* along the perpendicular axis.
    cut_index : int | None
        Index on the perpendicular axis where the cut is made.
        ``None`` uses the midplane.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    units : str | None
        Display units for field values.
    coord_units : str, tuple[str, str], or None
        Display units for coordinate axes.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override colormap.
    vmin, vmax : float | None
        Color limits.
    log_scale : bool
        Logarithmic color mapping.
    title : str | None
        Override auto-generated title.
    step : int | None
        Timestep number.
    time : float | None
        Simulation time.
    colorbar : bool or "inset"
        Colorbar mode.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
    cut_color : str or None
        Color for the cut line marker on the 2D panel. ``None`` uses
        the theme's accent color.
    cut_linestyle : str
        Line style for the cut marker.
    figsize : tuple[float, float] | None
        Figure size override. ``None`` uses ``(7, 8)``.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes]]
        Figure and ``(ax_2d, ax_1d)`` axes pair.
    """
    ensure_matplotlib()

    import matplotlib.pyplot as plt
    import numpy as np

    from pypic.plotting._labels import axis_label, field_label
    from pypic.plotting._resolve import (
        prepare_data,
        resolve_coord_units,
        resolve_field_values,
        surviving_axis_names,
    )
    from pypic.plotting.slices import plot_field_slice
    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    theme = _resolve_theme_arg(theme)
    if cut_color is None:
        cut_color = theme.accent_color
    data = prepare_data(data, plane)

    surviving = surviving_axis_names(data)
    if cut_axis not in surviving:
        msg = f"cut_axis={cut_axis!r} not in dataset axes {list(surviving)}"
        raise ValueError(msg)

    # Determine which axis is the cut direction and which is perpendicular
    cut_dim_idx = surviving.index(cut_axis)
    perp_dim_idx = 1 - cut_dim_idx  # only works for 2D

    coords = data.grid.coordinate_arrays()
    if cut_index is None:
        cut_index = data.grid.dimensions[perp_dim_idx] // 2

    dim_size = data.grid.dimensions[perp_dim_idx]
    if cut_index < 0 or cut_index >= dim_size:
        perp_name = surviving[perp_dim_idx]
        msg = (
            f"cut_index {cut_index} out of bounds for axis "
            f"{perp_name!r} with dimension {dim_size}"
        )
        raise ValueError(msg)

    # Cut position in physical coordinates
    cut_coord = float(coords[perp_dim_idx][cut_index])

    with use_theme(theme):
        fig, (ax_2d, ax_1d) = plt.subplots(
            2,
            1,
            figsize=figsize or (7, 8),
            height_ratios=[2, 1],
            sharex=(cut_dim_idx == 0),
        )

        # Top panel: 2D slice
        plot_field_slice(
            data,
            field,
            units=units,
            coord_units=coord_units,
            theme=theme,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            symmetric=symmetric,
            log_scale=log_scale,
            title=title,
            step=step,
            time=time,
            ax=ax_2d,
            colorbar=colorbar,
            extremes=extremes,
        )

        # Suppress redundant xlabel on top panel when sharing x-axis
        if cut_dim_idx == 0:
            ax_2d.set_xlabel("")

        # Mark the cut line on the 2D panel
        if cut_dim_idx == 0:
            ax_2d.axhline(cut_coord, color=cut_color, linestyle=cut_linestyle, lw=1)
        else:
            ax_2d.axvline(cut_coord, color=cut_color, linestyle=cut_linestyle, lw=1)

        # Bottom panel: 1D profile along the cut
        import warnings

        values = resolve_field_values(data, field, units)
        profile = values[:, cut_index] if cut_dim_idx == 0 else values[cut_index, :]
        cut_coords = coords[cut_dim_idx]

        if np.all(np.isnan(profile)):
            warnings.warn(
                f"Profile along {cut_axis!r} at index {cut_index} is entirely NaN",
                stacklevel=2,
            )

        ax_1d.plot(cut_coords, profile, color=cut_color)
        info = data.field_info(field)
        cu_x, cu_y = resolve_coord_units(coord_units)
        cut_unit = cu_x if cut_dim_idx == 0 else cu_y
        ax_1d.set_xlabel(axis_label(cut_axis, unit_str=cut_unit))
        ax_1d.set_ylabel(field_label(info, unit_str=units or ""))
        apply_grid(ax_1d, theme)

        if np.any(np.isfinite(profile)):
            ax_1d.set_xlim(cut_coords[0], cut_coords[-1])

        fig.tight_layout()
        apply_rounding(ax_2d)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, (ax_2d, ax_1d)
