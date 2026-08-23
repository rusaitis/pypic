"""Time-distance (kymograph) plots for wave and reconnection analysis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.plotting._colorbar import ExtremesMode
    from pypic.plotting.styles import ThemeArg
    from pypic.types import FloatArray


def plot_kymograph(
    values: FloatArray,
    coords: FloatArray,
    times: FloatArray,
    *,
    label: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    theme: ThemeArg = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool | None = None,
    log_scale: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a time-distance (kymograph) diagram.

    Displays a 1D spatial profile at each timestep as a 2D color map,
    with the spatial coordinate on the x-axis and time on the y-axis.
    Common in reconnection, wave propagation, and shock studies.

    Parameters
    ----------
    values : FloatArray
        2D array of shape ``(n_times, n_x)`` — one row per timestep.
    coords : FloatArray
        1D spatial coordinate array of length ``n_x``.
    times : FloatArray
        1D time array of length ``n_times``.
    label : str
        Colorbar label (e.g. ``"$B_z$"`` or ``"$B_z$ [nT]"``).
    xlabel : str | None
        X-axis label. ``None`` defaults to empty.
    ylabel : str | None
        Y-axis label. ``None`` defaults to ``"time"``.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str | None
        Colormap override. ``None`` auto-selects (diverging if data
        contains negative values, sequential otherwise).
    vmin, vmax : float | None
        Color limits. ``None`` for auto.
    symmetric : bool | None
        Force symmetric color limits around zero. ``None`` auto-detects.
    log_scale : bool
        Use logarithmic color mapping. Ignored when *symmetric*.
    title : str | None
        Axes title.
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    colorbar : bool or "inset"
        Colorbar mode.
    extremes : "semi", "transparent", "darken", or None
        How to style values outside ``[vmin, vmax]``.
    save : str | None
        Save figure to this path.
    figsize : tuple[float, float] | None
        Figure size override.
    vmin : float or None
        Lower color limit. ``None`` (default) autoscales.
    vmax : float or None
        Upper color limit. ``None`` (default) autoscales.

    Returns
    -------
    tuple[Figure, Axes]

    Raises
    ------
    ValueError
        If *values* is not 2D or shapes are inconsistent.
    """
    ensure_matplotlib()

    from pypic.plotting._colorbar import attach_colorbar
    from pypic.plotting._colormaps import symmetric_clim
    from pypic.plotting._resolve import get_or_create_axes
    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        use_theme,
    )

    if values.ndim != 2:
        msg = f"values must be 2D (n_times, n_x), got {values.ndim}D"
        raise ValueError(msg)
    n_times, n_x = values.shape
    if coords.shape != (n_x,):
        msg = f"coords length {coords.shape[0]} != values columns {n_x}"
        raise ValueError(msg)
    if times.shape != (n_times,):
        msg = f"times length {times.shape[0]} != values rows {n_times}"
        raise ValueError(msg)

    owned = ax is None
    theme = _resolve_theme_arg(theme)

    # Auto-detect symmetric
    use_symmetric = symmetric
    if use_symmetric is None:
        finite = values[np.isfinite(values)]
        use_symmetric = bool(finite.size > 0 and float(np.nanmin(finite)) < 0)

    # Colormap selection
    if cmap is None:
        cmap = theme.diverging_cmap if use_symmetric else theme.sequential_cmap

    # Log scale handling (same pattern as slices.py)
    norm = None
    if log_scale and use_symmetric:
        warnings.warn(
            "log_scale=True ignored because symmetric color limits are active",
            stacklevel=2,
        )
        log_scale = False

    if log_scale:
        from matplotlib.colors import LogNorm

        plot_values = np.where(values > 0, values, np.nan)
        finite = plot_values[np.isfinite(plot_values)]
        if finite.size > 0:
            auto_vmin = vmin if vmin is not None else float(np.nanmin(finite))
            auto_vmax = vmax if vmax is not None else float(np.nanmax(finite))
            if auto_vmin <= 0:
                positive = finite[finite > 0]
                auto_vmin = float(np.nanmin(positive)) if positive.size > 0 else 1e-10
            norm = LogNorm(vmin=auto_vmin, vmax=auto_vmax)
        values = plot_values
        vmin, vmax = None, None
    elif vmin is None and vmax is None and use_symmetric:
        vmin, vmax = symmetric_clim(values)

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        mesh_kwargs: dict[str, Any] = {"shading": "auto", "cmap": cmap}
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = vmin
            mesh_kwargs["vmax"] = vmax

        mesh = ax.pcolormesh(coords, times, values, **mesh_kwargs)

        attach_colorbar(fig, ax, mesh, label, colorbar, extremes=extremes)

        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel if ylabel is not None else "time")
        if title is not None:
            ax.set_title(title)
        apply_grid(ax, theme)

        if owned:
            fig.tight_layout()
            apply_rounding(ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax
