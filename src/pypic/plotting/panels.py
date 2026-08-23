"""Multi-panel layout helpers for field overview figures."""

from __future__ import annotations

import math
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


def plot_field_grid(
    data: FieldDataset,
    fields: list[str],
    *,
    ncols: int = 3,
    plane: PlaneSelection | None = None,
    units: dict[str, str] | None = None,
    coord_units: str | tuple[str, str] | None = None,
    theme: ThemeArg = None,
    cmap: str | Colormap | dict[str, str | Colormap] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool | Literal["inset"] = True,
    extremes: ExtremesMode = "semi",
    alpha: float = 1.0,
    symmetric: bool | None = None,
    log_scale: bool = False,
    step: int | None = None,
    time: float | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    panel_labels: bool = True,
    suptitle: str | None = None,
) -> tuple[Figure, list[Axes]]:
    r"""Plot multiple fields in an auto-arranged grid with panel labels.

    Creates a figure with ``ceil(len(fields) / ncols)`` rows and
    *ncols* columns. Each panel shows one field via
    `plot_field_slice`, with optional ``(a)``, ``(b)``, ``(c)``
    labels.

    Parameters
    ----------
    data : FieldDataset
        Input dataset (2D or 3D).
    fields : list[str]
        Field names to plot, one per panel.
    ncols : int
        Number of columns in the grid.
    plane : PlaneSelection | None
        Plane selection for 3D data (applied to all panels).
    units : dict[str, str] | None
        Per-field display units, keyed by field name. Fields not in the
        dict use code units.
    coord_units : str, tuple[str, str], or None
        Coordinate axis units (shared across panels).
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    cmap : str, Colormap, dict, or None
        Colormap override. A single string or Colormap applies to all
        panels; a dict maps field names to per-field colormaps.
    vmin, vmax : float | None
        Shared color limits across all panels. ``None`` for auto.
    colorbar : bool or "inset"
        Colorbar mode for each panel.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
    log_scale : bool
        Use logarithmic color mapping for all panels.
    step : int | None
        Timestep number (shown in suptitle if *suptitle* is not set).
    time : float | None
        Simulation time (shown in suptitle if *suptitle* is not set).
    figsize : tuple[float, float] | None
        Figure size override. ``None`` auto-scales based on grid size.
    panel_labels : bool
        Add ``(a)``, ``(b)``, … labels to each panel.
    suptitle : str | None
        Figure super-title. ``None`` generates from step/time.

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and flat list of axes (one per field).
    """
    ensure_matplotlib()

    import matplotlib.pyplot as plt

    from pypic.plotting._badge import add_label
    from pypic.plotting._resolve import prepare_data
    from pypic.plotting.slices import plot_field_slice
    from pypic.plotting.styles import _resolve_theme_arg, apply_rounding, use_theme

    theme = _resolve_theme_arg(theme)

    if not fields:
        msg = "fields list must not be empty"
        raise ValueError(msg)

    data = prepare_data(data, plane)

    nrows = math.ceil(len(fields) / ncols)
    if figsize is None:
        figsize = (theme.figsize_per_col * ncols, theme.figsize_per_row * nrows)

    # Scale overlay text for dense grids (harder to read at reduced size)
    label_scale = (
        theme.panel_label_scale_sparse if nrows == 1 else theme.panel_label_scale_dense
    )
    label_fontsize = theme.font_overlay * label_scale

    with use_theme(theme):
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = list(axes_arr.flat)

        for i, field_name in enumerate(fields):
            field_units = (units or {}).get(field_name)
            panel_cmap = cmap.get(field_name) if isinstance(cmap, dict) else cmap
            plot_field_slice(
                data,
                field_name,
                units=field_units,
                coord_units=coord_units,
                theme=theme,
                cmap=panel_cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                symmetric=symmetric,
                log_scale=log_scale,
                ax=axes_flat[i],
                colorbar=colorbar,
                extremes=extremes,
                step=step,
                time=time,
            )
            if panel_labels:
                add_label(axes_flat[i], chr(ord("a") + i), fontsize=label_fontsize)

        # Hide unused axes
        for j in range(len(fields), len(axes_flat)):
            axes_flat[j].set_visible(False)

        if suptitle is not None:
            fig.suptitle(suptitle)
        elif step is not None or time is not None:
            parts: list[str] = []
            if step is not None:
                parts.append(f"step {step}")
            if time is not None:
                parts.append(f"t = {time:.2f}")
            fig.suptitle(", ".join(parts))

        fig.tight_layout()
        for panel_ax in axes_flat[: len(fields)]:
            apply_rounding(panel_ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, axes_flat[: len(fields)]
