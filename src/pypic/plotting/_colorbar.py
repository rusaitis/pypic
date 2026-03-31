"""Shared colorbar styling for pypic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    from pypic.plotting._badge import BadgeLoc, OverlayVariant

ExtremesMode = Literal["darken", "semi", "transparent"] | None


def _apply_extremes(
    mappable: object,
    *,
    mode: ExtremesMode = "semi",
    darken_factor: float = 0.75,
    semi_alpha: float = 0.3,
) -> None:
    """Style the over/under extension colors of a ScalarMappable.

    Parameters
    ----------
    mappable : object
        A ``ScalarMappable`` (e.g. from ``pcolormesh``).
    mode : "semi", "transparent", "darken", or None
        ``"semi"`` — endpoint colors at reduced opacity (*semi_alpha*).
        ``"transparent"`` — fully transparent under/over.
        ``"darken"`` — multiply endpoint RGB by *darken_factor*.
        ``None`` — no-op (matplotlib defaults).
    darken_factor : float
        RGB multiplier for ``"darken"`` mode.
    semi_alpha : float
        Alpha for ``"semi"`` mode (default 0.3).
    """
    if mode is None:
        return

    import matplotlib as mpl
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import to_rgba

    if not isinstance(mappable, ScalarMappable):
        return

    cmap = mappable.cmap.copy()
    match mode:
        case "transparent":
            bg = to_rgba(mpl.rcParams.get("axes.facecolor", "white"))
            cmap.set_under(bg)
            cmap.set_over(bg)
        case "semi":
            for setter, val in [("set_under", 0.0), ("set_over", 1.0)]:
                r, g, b, _a = to_rgba(cmap(val))
                getattr(cmap, setter)((r, g, b, semi_alpha))
        case "darken":
            for setter, val in [("set_under", 0.0), ("set_over", 1.0)]:
                r, g, b, a = to_rgba(cmap(val))
                getattr(cmap, setter)(
                    (r * darken_factor, g * darken_factor, b * darken_factor, a)
                )
    mappable.set_cmap(cmap)


def _style_colorbar(
    cb: Colorbar,
    label: str = "",
    *,
    tick_color: str | tuple[float, ...] | None = None,
) -> None:
    """Apply theme-consistent styling to a Colorbar (outline, ticks, label)."""
    import matplotlib as mpl
    from matplotlib.colors import to_rgba

    from pypic.plotting.styles import _theme_val

    outline_w: float = _theme_val("colorbar_outline_width", 0.3)
    tick_len: float = _theme_val("colorbar_tick_length", 2.0)

    if tick_color is None:
        tick_color = mpl.rcParams.get("xtick.color", "0.4")
    outline_rgba = (*to_rgba(tick_color)[:3], 0.3)

    if label:
        cb.set_label(label, color=tick_color)
    cb.outline.set_linewidth(outline_w)
    cb.outline.set_edgecolor(outline_rgba)
    cb.ax.tick_params(
        width=outline_w, length=tick_len, colors=tick_color, labelcolor=tick_color
    )


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: object,
    label: str,
    *,
    extend: str = "both",
    extremes: ExtremesMode = "semi",
) -> Colorbar:
    """Add a colorbar with styled over/under extensions and subtle outline.

    Parameters
    ----------
    extremes : "semi", "transparent", "darken", or None
        ``"semi"`` (default) — semi-transparent extension colors.
        ``"transparent"`` — fully transparent extensions.
        ``"darken"`` — darkened endpoint colors.
        ``None`` — matplotlib defaults (no modification).

    Colors for the outline, ticks, and label are read from the active
    matplotlib rcParams so that custom themes propagate automatically.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from pypic.plotting.styles import _theme_val

    _apply_extremes(mappable, mode=extremes)
    if extremes is None:
        extend = "neither"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size=_theme_val("colorbar_width", "4%"),
        pad=_theme_val("colorbar_pad", 0.05),
    )
    cb = fig.colorbar(mappable, cax=cax, extend=extend)  # type: ignore[arg-type]
    _style_colorbar(cb, label)
    return cb


def add_inset_colorbar(
    ax: Axes,
    mappable: object,
    label: str = "",
    *,
    loc: BadgeLoc | None = None,
    variant: OverlayVariant | None = None,
    width: float = 0.3,
    height: float = 0.02,
    pad: float | None = None,
    extend: str = "both",
    extremes: ExtremesMode = "semi",
    n_ticks: int = 3,
    fontsize: float | None = None,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
) -> Colorbar:
    """Add a horizontal colorbar rendered inside the plot axes.

    Uses ``ax.inset_axes()`` for the colorbar with a rounded background
    patch. Visually matches the badge style from
    :func:`~pypic.plotting.add_badge`.

    Parameters
    ----------
    ax : Axes
        Target axes.
    mappable : object
        A ``ScalarMappable`` (e.g. from ``pcolormesh`` or ``streamplot``).
    label : str
        Colorbar label text.
    loc : BadgeLoc or None
        Inset placement. ``None`` auto-selects (prefers lower right).
    variant : "darker", "lighter", "alt", or None
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

    from pypic.plotting._badge import (
        _claim_corner,
        _detect_overlay_defaults,
        _resolve_rgba,
    )
    from pypic.plotting.styles import _theme_val

    actual_loc = _claim_corner(ax, "lower right", loc)

    default_bg, default_fg = _detect_overlay_defaults(variant)
    overlay_alpha: float = _theme_val("overlay_color", (0, 0, 0, 0.65))[3]

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg, overlay_alpha)
    fg_rgba = _resolve_rgba(text_color, text_alpha, default_fg)

    rounding: float = _theme_val("overlay_rounding", 0.6)
    if pad is None:
        pad = _theme_val("overlay_margin", 0.03) * 0.5

    box_pad = 0.015

    if fontsize is None:
        fontsize = _theme_val("font_overlay", 9.0)

    fig = ax.get_figure()

    _apply_extremes(mappable, mode=extremes)
    if extremes is None:
        extend = "neither"

    # --- Pass 1: render a provisional colorbar to measure text extents ---
    prov_cax = ax.inset_axes([0.3, 0.3, width, height], zorder=5)
    cb = fig.colorbar(  # type: ignore[union-attr]
        mappable, cax=prov_cax, orientation="horizontal", extend=extend
    )
    cb.locator = MaxNLocator(nbins=n_ticks)
    cb.update_ticks()
    _style_colorbar(cb, tick_color=fg_rgba)
    prov_cax.tick_params(labelsize=fontsize, colors=fg_rgba)
    if label:
        prov_cax.set_title(label, fontsize=fontsize, color=fg_rgba, pad=4)

    renderer = fig.canvas.get_renderer()  # type: ignore[union-attr]
    fig.draw(renderer)
    bbox_disp = prov_cax.get_tightbbox(renderer)
    bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())  # type: ignore[union-attr]

    # Overhangs: how much text extends beyond the bar on each side
    overhang_left = 0.3 - bbox_ax.x0
    overhang_right = bbox_ax.x1 - (0.3 + width)
    overhang_bottom = 0.3 - bbox_ax.y0
    overhang_top = bbox_ax.y1 - (0.3 + height)

    content_w = width + overhang_left + overhang_right
    content_h = height + overhang_top + overhang_bottom
    total_w = content_w + 2 * box_pad
    total_h = content_h + 2 * box_pad

    # Remove provisional colorbar axes
    prov_cax.remove()

    # --- Pass 2: place at the correct position ---
    if "right" in actual_loc:
        bg_x = 1.0 - pad - total_w
    elif "center" in actual_loc:
        bg_x = 0.5 - total_w / 2
    else:
        bg_x = pad
    if "upper" in actual_loc:
        bg_y = 1.0 - pad - total_h
    else:
        bg_y = pad

    bar_x = bg_x + box_pad + overhang_left
    bar_y = bg_y + box_pad + overhang_bottom
    cax = ax.inset_axes([bar_x, bar_y, width, height], zorder=5)
    cb = fig.colorbar(  # type: ignore[union-attr]
        mappable, cax=cax, orientation="horizontal", extend=extend
    )
    cb.locator = MaxNLocator(nbins=n_ticks)
    cb.update_ticks()
    _style_colorbar(cb, tick_color=fg_rgba)
    cax.tick_params(labelsize=fontsize, colors=fg_rgba)
    if label:
        cax.set_title(label, fontsize=fontsize, color=fg_rgba, pad=4)
    cax.set_facecolor("none")

    # Background patch sized to actual content
    has_bg = bg_rgba[3] >= 0.01
    bg_patch = FancyBboxPatch(
        (bg_x, bg_y),
        total_w,
        total_h,
        boxstyle=f"round,pad=0,rounding_size={rounding * 0.02:.4f}",
        facecolor=bg_rgba if has_bg else "none",
        edgecolor="none",
        transform=ax.transAxes,
        zorder=4.9,
    )
    ax.add_patch(bg_patch)

    return cb


def attach_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: object,
    label: str,
    colorbar: bool | Literal["inset"],
    *,
    extremes: ExtremesMode = "semi",
) -> None:
    """Dispatch to the appropriate colorbar function, or do nothing.

    Parameters
    ----------
    fig : Figure
        Parent figure (used for standard colorbars).
    ax : Axes
        Target axes.
    mappable : object
        A ``ScalarMappable``.
    label : str
        Colorbar label.
    colorbar : bool or "inset"
        ``True`` for a standard side colorbar, ``"inset"`` for an
        overlay colorbar, ``False`` to skip.
    extremes : "semi", "transparent", "darken", or None
        How to style over/under values. ``None`` uses matplotlib defaults.
    """
    if not colorbar:
        return
    if colorbar == "inset":
        add_inset_colorbar(ax, mappable, label, extremes=extremes)
    else:
        add_colorbar(fig, ax, mappable, label, extremes=extremes)
