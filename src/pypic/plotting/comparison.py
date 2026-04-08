"""Three-panel comparison plots: A | B | difference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import ThemeArg
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection


def plot_comparison(
    data_a: FieldDataset,
    data_b: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    coord_units: str | tuple[str, str] | None = None,
    theme: ThemeArg = None,
    cmap: str | Colormap | None = None,
    diff_cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    labels: tuple[str, str] = ("A", "B"),
    alpha: float = 1.0,
    symmetric: bool | None = None,
    log_scale: bool = False,
    step: int | None = None,
    time: float | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    show_error: bool = False,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    r"""Three-panel comparison: dataset A, dataset B, and their difference.

    Parameters
    ----------
    data_a, data_b : FieldDataset
        Two datasets to compare (must share compatible grids).
    field : str
        Field name — canonical, alias, or derived.
    plane : PlaneSelection | None
        Plane selection for 3D data. ``None`` auto-slices at midplane.
    units : str | None
        Display units for field values.
    coord_units : str | None
        Display units for coordinate axes.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | Colormap | None
        Override colormap for A/B panels.
    diff_cmap : str | Colormap | None
        Override colormap for difference panel. Defaults to diverging.
    vmin, vmax : float | None
        Color limits for A/B panels. ``None`` for auto.
    labels : tuple[str, str]
        Panel labels for A and B.
    alpha : float
        Mesh transparency (0 = invisible, 1 = opaque).
    symmetric : bool | None
        Force symmetric color limits on A/B panels. ``None``
        auto-detects (symmetric for signed fields). ``True`` forces
        symmetric, ``False`` disables.
    log_scale : bool
        Use logarithmic color mapping on A/B panels. The difference
        panel always uses linear scale. Ignored when *symmetric* is
        active.
    step : int | None
        Timestep number for the suptitle.
    time : float | None
        Simulation time for the suptitle.
    colorbar : bool or "inset"
        ``True`` for side colorbars, ``"inset"`` for overlay colorbars,
        ``False`` to disable.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
        ``"transparent"`` makes them invisible.
    show_error : bool
        When ``True``, display the relative L2 error on the difference
        panel as a text annotation.
    figsize : tuple[float, float] | None
        Figure size override. Defaults to ``(14, 4)``.
    title : str | None
        Override auto-generated suptitle.

    Returns
    -------
    tuple[Figure, dict[str, Axes]]
        Figure and dict with keys ``"a"``, ``"b"``, ``"diff"``.
    """
    ensure_matplotlib()

    import warnings

    import matplotlib.pyplot as plt
    import numpy as np

    from pypic.plotting._colorbar import attach_colorbar
    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_colormap,
        symmetric_clim,
    )
    from pypic.plotting._labels import axis_label, field_label, figure_title
    from pypic.plotting._resolve import (
        default_midplane,
        require_plottable_grid,
        resolve_coord_units,
        resolve_field_values,
        surviving_axis_names,
    )
    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    theme = _resolve_theme_arg(theme)

    if plane is None:
        plane = default_midplane(data_a)
    if plane is not None:
        data_a = plane.apply(data_a)
        data_b = plane.apply(data_b)
    require_plottable_grid(data_a)

    values_a = resolve_field_values(data_a, field, units)
    values_b = resolve_field_values(data_b, field, units)
    diff = values_a - values_b

    info = data_a.field_info(field)
    coords = data_a.grid.coordinate_arrays()
    surviving_axes = surviving_axis_names(data_a)

    cmap_name = resolve_colormap(
        field, values_a, theme, info=info, cmap=cmap if isinstance(cmap, str) else None
    )
    diff_cmap_name = diff_cmap if isinstance(diff_cmap, str) else theme.diverging_cmap

    # Determine whether A/B panels use symmetric color limits
    use_symmetric = symmetric
    if use_symmetric is None:
        use_symmetric = not is_positive_definite(field, values_a, info)

    # Log scale: incompatible with symmetric
    norm = None
    if log_scale and use_symmetric:
        warnings.warn(
            "log_scale=True ignored because symmetric color limits are active",
            stacklevel=2,
        )
        log_scale = False

    if vmin is not None and vmax is not None:
        combined_min, combined_max = vmin, vmax
    else:
        combined_min = float(np.nanmin([np.nanmin(values_a), np.nanmin(values_b)]))
        combined_max = float(np.nanmax([np.nanmax(values_a), np.nanmax(values_b)]))
        if use_symmetric:
            absmax = max(abs(combined_min), abs(combined_max))
            combined_min, combined_max = -absmax, absmax

    if log_scale:
        from matplotlib.colors import LogNorm

        safe_min = combined_min if combined_min > 0 else 1e-10
        norm = LogNorm(vmin=safe_min, vmax=combined_max)

    diff_vmin, diff_vmax = symmetric_clim(diff)

    unit_str = units if units else ""
    cb_label = field_label(info, unit_str=unit_str)

    with use_theme(theme):
        fig, axes_dict = plt.subplot_mosaic(
            [["a", "b", "diff"]],
            figsize=figsize or (14, 4),
        )

        diff_title = f"{labels[0]} \u2212 {labels[1]}"
        panels = [
            ("a", values_a, labels[0], combined_min, combined_max),
            ("b", values_b, labels[1], combined_min, combined_max),
            ("diff", diff, diff_title, diff_vmin, diff_vmax),
        ]

        for key, values, panel_title, panel_vmin, panel_vmax in panels:
            if key == "diff":
                panel_cmap = diff_cmap or diff_cmap_name
            elif cmap is not None and not isinstance(cmap, str):
                panel_cmap = cmap
            else:
                panel_cmap = cmap_name

            ax = axes_dict[key]
            mesh_kwargs: dict[str, Any] = {
                "shading": "auto",
                "cmap": panel_cmap,
                "alpha": alpha,
            }
            # Log norm for A/B panels only (diff is always linear)
            if norm is not None and key != "diff":
                mesh_kwargs["norm"] = norm
            else:
                mesh_kwargs["vmin"] = panel_vmin
                mesh_kwargs["vmax"] = panel_vmax
            mesh = ax.pcolormesh(coords[0], coords[1], values.T, **mesh_kwargs)
            label = f"\u0394 {cb_label}" if key == "diff" else cb_label
            attach_colorbar(fig, ax, mesh, label, colorbar, extremes=extremes)
            cu_x, cu_y = resolve_coord_units(coord_units)
            ax.set_xlabel(axis_label(surviving_axes[0], unit_str=cu_x))
            ax.set_ylabel(axis_label(surviving_axes[1], unit_str=cu_y))
            ax.set_aspect("equal")
            apply_grid(ax, theme)
            ax.set_title(panel_title)

        if show_error:
            import matplotlib as mpl

            from pypic.diagnostics import l2_relative_error

            l2 = l2_relative_error(values_a, values_b)
            diff_ax = axes_dict["diff"]
            bg = mpl.rcParams.get("axes.facecolor", "white")
            tc = mpl.rcParams.get("xtick.color", "0.4")
            diff_ax.text(
                0.02,
                0.98,
                f"$L_2$ = {l2:.2e}",
                transform=diff_ax.transAxes,
                fontsize=theme.annotation_fontsize,
                va="top",
                ha="left",
                color=tc,
                bbox={"facecolor": bg, "alpha": 0.7, "edgecolor": "none"},
            )

        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle(figure_title(info, step=step, time=time))

        fig.tight_layout()
        for ax_item in axes_dict.values():
            apply_rounding(ax_item)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, axes_dict
