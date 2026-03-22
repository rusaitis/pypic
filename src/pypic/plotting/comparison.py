"""Three-panel comparison plots: A | B | difference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection
    from pypic.types import FloatArray


def plot_comparison(
    data_a: FieldDataset,
    data_b: FieldDataset,
    field: str,
    *,
    plane: PlaneSelection | None = None,
    units: str | None = None,
    coord_units: str | None = None,
    theme: PlotTheme | None = None,
    cmap: str | Colormap | None = None,
    diff_cmap: str | Colormap | None = None,
    labels: tuple[str, str] = ("A", "B"),
    step: int | None = None,
    time: float | None = None,
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
    labels : tuple[str, str]
        Panel labels for A and B.
    step : int | None
        Timestep number for the suptitle.
    time : float | None
        Simulation time for the suptitle.
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
    import matplotlib.pyplot as plt
    import numpy as np

    from pypic.plotting._colormaps import (
        is_positive_definite,
        resolve_colormap,
        symmetric_clim,
    )
    from pypic.plotting._labels import axis_label, colorbar_label, figure_title
    from pypic.plotting.styles import DEFAULT, use_theme

    if theme is None:
        theme = DEFAULT

    ndim_a = len(data_a.grid.dimensions)
    if ndim_a == 3 and plane is None:
        from pypic.selections import PlaneSelection

        normal = data_a.grid.geometry.axis_names[2]
        plane = PlaneSelection(normal=normal)

    if plane is not None:
        data_a = plane.apply(data_a)
        data_b = plane.apply(data_b)

    def _get_values(ds: FieldDataset) -> FloatArray:
        if ds.has_field(field):
            return ds.in_units(field, units) if units else ds[field]
        vals = ds.compute(field)
        if units:
            return ds.in_units(field, units)
        return vals

    values_a = _get_values(data_a)
    values_b = _get_values(data_b)
    diff = values_a - values_b

    info = data_a.field_info(field)
    coords = data_a.grid.coordinate_arrays()
    surviving_axes = data_a.grid.geometry.axis_names[: len(data_a.grid.dimensions)]

    cmap_name = resolve_colormap(
        field, values_a, theme, info=info, cmap=cmap if isinstance(cmap, str) else None
    )
    diff_cmap_name = diff_cmap if isinstance(diff_cmap, str) else theme.diverging_cmap

    positive = is_positive_definite(field, values_a, info)
    combined_min = float(np.nanmin([np.nanmin(values_a), np.nanmin(values_b)]))
    combined_max = float(np.nanmax([np.nanmax(values_a), np.nanmax(values_b)]))
    if not positive:
        absmax = max(abs(combined_min), abs(combined_max))
        combined_min, combined_max = -absmax, absmax

    diff_vmin, diff_vmax = symmetric_clim(diff)

    unit_str = units if units else ""
    cb_label = colorbar_label(info, unit_str=unit_str)

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

        for key, values, panel_title, vmin, vmax in panels:
            if key == "diff":
                panel_cmap = diff_cmap or diff_cmap_name
            elif cmap is not None and not isinstance(cmap, str):
                panel_cmap = cmap
            else:
                panel_cmap = cmap_name

            ax = axes_dict[key]
            mesh = ax.pcolormesh(
                coords[0],
                coords[1],
                values.T,
                shading="auto",
                cmap=panel_cmap,
                vmin=vmin,
                vmax=vmax,
            )
            label = f"\u0394 {cb_label}" if key == "diff" else cb_label
            fig.colorbar(mesh, ax=ax).set_label(label)
            ax.set_xlabel(axis_label(surviving_axes[0], unit_str=coord_units or ""))
            ax.set_ylabel(axis_label(surviving_axes[1], unit_str=coord_units or ""))
            ax.set_aspect("equal")
            ax.set_title(panel_title)

        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle(figure_title(info, step=step, time=time))

        fig.tight_layout()

    return fig, axes_dict
