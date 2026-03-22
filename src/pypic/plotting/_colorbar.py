"""Shared colorbar styling for pypic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: object,
    label: str,
    *,
    extend: str = "both",
) -> Colorbar:
    """Add a colorbar with darkened over/under extensions and subtle outline."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import to_rgba
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Darken the extreme colors for over/under extension triangles
    if isinstance(mappable, ScalarMappable):
        cmap = mappable.cmap.copy()
        factor = 0.65
        for setter, val in [("set_under", 0.0), ("set_over", 1.0)]:
            r, g, b, a = to_rgba(cmap(val))
            getattr(cmap, setter)((r * factor, g * factor, b * factor, a))
        mappable.set_cmap(cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, extend=extend)  # type: ignore[arg-type]
    cb.set_label(label)
    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor((0.5, 0.5, 0.5, 0.3))
    cb.ax.tick_params(width=0.3, length=2)
    return cb
