"""Three-panel comparison plots: A | B | difference."""

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
    diff_vmin: float | None = None,
    diff_vmax: float | None = None,
    labels: tuple[str, str] = ("A", "B"),
    alpha: float = 1.0,
    symmetric: bool | None = None,
    log_scale: bool = False,
    symlog: bool = False,
    linthresh: float | None = None,
    step: int | None = None,
    time: float | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    show_error: bool = False,
    ax: tuple[Axes, Axes, Axes] | None = None,
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
    diff_vmin, diff_vmax : float | None
        Color limits for the difference panel. ``None`` uses symmetric
        limits from `symmetric_clim`.
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
    extremes : "semi", "transparent", "darken", or None
        Colorbar out-of-range indicator style. ``"semi"`` (default)
        uses semi-transparent extension colors; ``"transparent"``
        hides them; ``"darken"`` darkens the endpoint colors;
        ``None`` leaves matplotlib defaults untouched.
    show_error : bool
        When ``True``, display the relative L2 error on the difference
        panel as a text annotation.
    ax : tuple[Axes, Axes, Axes] or None
        Existing axes for A, B and the difference, in that order; the
        figure they belong to is then left to the caller to lay out.
        ``None`` creates a one-row, three-panel figure.
    figsize : tuple[float, float] | None
        Figure size override. Defaults to ``(14, 4)``.
    title : str | None
        Override auto-generated suptitle.
    data_a : FieldDataset
        Left-hand dataset (panel A).
    data_b : FieldDataset
        Right-hand dataset (panel B); must share A's grid.
    vmin : float or None
        Lower color limit. ``None`` (default) autoscales.
    vmax : float or None
        Upper color limit. ``None`` (default) autoscales.
    diff_vmin : float or None
        Lower color limit for the difference panel. ``None``
        (default) autoscales symmetrically about zero.
    diff_vmax : float or None
        Upper color limit for the difference panel. ``None``
        (default) autoscales symmetrically about zero.
    symlog : bool
        Use a symmetric-log color scale, for signed fields
        spanning several decades.
    linthresh : float or None
        Linear threshold for ``symlog``. ``None`` (default)
        auto-detects from the data.
    save : str or Path or None
        Path to write the figure to. When given, the figure is saved
        and closed; when ``None`` (default) it is left open for the
        caller to display or modify further.

    Returns
    -------
    tuple[Figure, dict[str, Axes]]
        Figure and dict with keys ``"a"``, ``"b"``, ``"diff"``.
    """
    ensure_matplotlib()

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    from pypic.plotting._colorbar import attach_colorbar
    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_field_colormap,
        resolve_norm,
        symmetric_clim,
    )
    from pypic.plotting._labels import field_label, figure_title
    from pypic.plotting._resolve import (
        default_midplane,
        get_or_create_axes,
        maybe_save,
        plane_axis_labels,
        require_plottable_grid,
        resolve_field_values,
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
    if values_a.shape != values_b.shape:
        msg = (
            f"Grid shape mismatch: {values_a.shape} vs {values_b.shape}. "
            f"Regrid datasets to a common grid before comparing."
        )
        raise ValueError(msg)
    diff = values_a - values_b

    info = data_a.field_info(field)
    coords = data_a.grid.coordinate_arrays()
    xlabel, ylabel = plane_axis_labels(data_a, coord_units)

    _, colormap = resolve_field_colormap(field, values_a, theme, info=info, cmap=cmap)
    if symmetric is None:
        symmetric = not is_positive_definite(field, values_a, info)
    # One norm over both inputs, so A and B share a color scale.
    norm = resolve_norm(
        np.concatenate([values_a.ravel(), values_b.ravel()]),
        symmetric=symmetric,
        log_scale=log_scale,
        symlog=symlog,
        vmin=vmin,
        vmax=vmax,
        linthresh=linthresh,
    )

    if diff_vmin is None or diff_vmax is None:
        auto_dvmin, auto_dvmax = symmetric_clim(diff)
        diff_vmin = diff_vmin if diff_vmin is not None else auto_dvmin
        diff_vmax = diff_vmax if diff_vmax is not None else auto_dvmax
    diff_norm = Normalize(vmin=diff_vmin, vmax=diff_vmax)

    unit_str = units if units else ""
    cb_label = field_label(info, unit_str=unit_str)

    with use_theme(theme):
        if ax is None:
            fig, axes_dict = plt.subplot_mosaic(
                [["a", "b", "diff"]],
                figsize=figsize or (14, 4),
            )
        else:
            fig, _ = get_or_create_axes(theme, ax[0], None)
            axes_dict = dict(zip(("a", "b", "diff"), ax, strict=True))

        diff_title = f"{labels[0]} \u2212 {labels[1]}"
        panels = [
            ("a", values_a, labels[0], colormap, norm),
            ("b", values_b, labels[1], colormap, norm),
            (
                "diff",
                diff,
                diff_title,
                diff_cmap if diff_cmap is not None else theme.diverging_cmap,
                diff_norm,
            ),
        ]

        for key, values, panel_title, panel_cmap, panel_norm in panels:
            panel_ax = axes_dict[key]
            mesh = panel_ax.pcolormesh(
                coords[0],
                coords[1],
                values.T,
                shading="auto",
                cmap=panel_cmap,
                alpha=alpha,
                norm=panel_norm,
            )
            label = f"\u0394 {cb_label}" if key == "diff" else cb_label
            attach_colorbar(fig, panel_ax, mesh, label, colorbar, extremes=extremes)
            panel_ax.set_xlabel(xlabel)
            panel_ax.set_ylabel(ylabel)
            panel_ax.set_aspect("equal")
            apply_grid(panel_ax, theme)
            panel_ax.set_title(panel_title)

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

        if ax is None:
            fig.tight_layout()
        for panel_ax in axes_dict.values():
            apply_rounding(panel_ax)

    maybe_save(fig, save)
    return fig, axes_dict
