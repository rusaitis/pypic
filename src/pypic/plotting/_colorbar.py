"""Shared colorbar styling for pypic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    from pypic.plotting._badge import BadgeLoc, OverlayShade

ExtremesMode = Literal["darken", "transparent"]


def _apply_extremes(
    mappable: object,
    *,
    mode: ExtremesMode = "darken",
    darken_factor: float = 0.65,
) -> None:
    """Style the over/under extension colors of a ScalarMappable.

    Parameters
    ----------
    mappable : object
        A ``ScalarMappable`` (e.g. from ``pcolormesh``).
    mode : "darken" or "transparent"
        ``"darken"`` — multiply the endpoint RGB by *darken_factor*.
        ``"transparent"`` — set under/over to fully transparent.
    darken_factor : float
        RGB multiplier for ``"darken"`` mode.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import to_rgba

    if not isinstance(mappable, ScalarMappable):
        return

    cmap = mappable.cmap.copy()
    if mode == "transparent":
        cmap.set_under((0.0, 0.0, 0.0, 0.0))
        cmap.set_over((0.0, 0.0, 0.0, 0.0))
    else:
        for setter, val in [("set_under", 0.0), ("set_over", 1.0)]:
            r, g, b, a = to_rgba(cmap(val))
            getattr(cmap, setter)(
                (r * darken_factor, g * darken_factor, b * darken_factor, a)
            )
    mappable.set_cmap(cmap)


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: object,
    label: str,
    *,
    extend: str = "both",
    extremes: ExtremesMode = "darken",
) -> Colorbar:
    """Add a colorbar with styled over/under extensions and subtle outline.

    Parameters
    ----------
    extremes : "darken" or "transparent"
        ``"darken"`` darkens the extension colors. ``"transparent"`` makes
        values outside ``[vmin, vmax]`` invisible.

    Colors for the outline, ticks, and label are read from the active
    matplotlib rcParams so that custom themes propagate automatically.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgba
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _apply_extremes(mappable, mode=extremes)

    text_color = mpl.rcParams.get("text.color", "black")
    tick_color = mpl.rcParams.get("xtick.color", "0.4")
    outline_rgba = (*to_rgba(tick_color)[:3], 0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, extend=extend)  # type: ignore[arg-type]
    cb.set_label(label, color=text_color)
    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor(outline_rgba)
    cb.ax.tick_params(width=0.3, length=2, colors=tick_color, labelcolor=text_color)
    return cb


def add_inset_colorbar(
    ax: Axes,
    mappable: object,
    label: str = "",
    *,
    loc: BadgeLoc = "lower right",
    shade: OverlayShade | None = None,
    width: float = 0.3,
    height: float = 0.015,
    pad: float = 0.03,
    extend: str = "both",
    n_ticks: int = 3,
    fontsize: float = 8,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float = 0.65,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.85,
) -> Colorbar:
    """Add a horizontal colorbar rendered inside the plot axes.

    Uses ``ax.inset_axes()`` for the colorbar with a rounded background
    patch. Visually matches the status badge style from
    :func:`~pypic.plotting.add_status_badge`.

    Parameters
    ----------
    ax : Axes
        Target axes.
    mappable : object
        A ``ScalarMappable`` (e.g. from ``pcolormesh`` or ``streamplot``).
    label : str
        Colorbar label text.
    loc : BadgeLoc
        Inset placement location.
    shade : "darker", "lighter", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    width : float
        Bar width as a fraction of axes width.
    height : float
        Bar height as a fraction of axes height.
    pad : float
        Padding from axes edge as a fraction.
    extend : str
        Colorbar extension (``"both"``, ``"min"``, ``"max"``, ``"neither"``).
    n_ticks : int
        Maximum number of colorbar tick intervals.
    fontsize : float
        Font size for tick labels and colorbar label.
    bg_color : str, tuple, or None
        Background box color override.
    bg_alpha : float
        Background box opacity.
    text_color : str, tuple, or None
        Tick and label color override.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).

    Returns
    -------
    Colorbar
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator

    from pypic.plotting._badge import _detect_overlay_defaults, _resolve_rgba

    default_bg, default_fg = _detect_overlay_defaults(shade)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg)
    fg_rgba = _resolve_rgba(text_color, text_alpha, default_fg)

    # Space below the bar for tick labels, above for the label title
    tick_space = 0.03
    above_bar = 0.025 if label else 0.0
    total_height = above_bar + height + tick_space
    box_pad_x = 0.015
    box_pad_y = 0.035

    # Position: (x0, y0) is the bottom-left of the total bounding box
    if loc == "lower right":
        x0 = 1.0 - pad - box_pad_x - width
        y0 = pad + box_pad_y
    elif loc == "lower left":
        x0 = pad + box_pad_x
        y0 = pad + box_pad_y
    elif loc == "upper right":
        x0 = 1.0 - pad - box_pad_x - width
        y0 = 1.0 - pad - box_pad_y - total_height
    elif loc == "upper left":
        x0 = pad + box_pad_x
        y0 = 1.0 - pad - box_pad_y - total_height
    else:  # upper center
        x0 = 0.5 - width / 2
        y0 = 1.0 - pad - box_pad_y - total_height

    # Background patch (slightly larger than colorbar + labels)
    bg_patch = FancyBboxPatch(
        (x0 - box_pad_x, y0 - box_pad_y),
        width + 2 * box_pad_x,
        total_height + 2 * box_pad_y,
        # rounding_size in axes-fraction coords; visually matches
        # OVERLAY_BOX_STYLE (0.6 in offset-box points)
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=bg_rgba,
        edgecolor="none",
        transform=ax.transAxes,
        zorder=4.9,
    )
    ax.add_patch(bg_patch)

    # Bar sits above the tick space; label renders above via title
    bar_y = y0 + tick_space
    cax = ax.inset_axes([x0, bar_y, width, height], zorder=5)
    fig = ax.get_figure()
    cb = fig.colorbar(  # type: ignore[union-attr, arg-type]
        mappable, cax=cax, orientation="horizontal", extend=extend
    )

    cb.locator = MaxNLocator(nbins=n_ticks)
    cb.update_ticks()

    # Style ticks and label — full RGBA propagates text_alpha
    tick_color = fg_rgba
    cax.tick_params(
        labelsize=fontsize,
        colors=tick_color,
        width=0.3,
        length=2,
    )
    if label:
        cax.set_title(label, fontsize=fontsize, color=tick_color, pad=2)

    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor((*fg_rgba[:3], 0.3))
    cax.set_facecolor("none")

    _apply_extremes(mappable)

    return cb
