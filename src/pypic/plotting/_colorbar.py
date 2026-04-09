"""Shared colorbar styling for pypic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    from pypic.plotting._badge import BadgeLoc, OverlayVariant

ExtremesMode = Literal["darken", "semi", "transparent"] | None

# Inset colorbar layout — conversion factors from theme points to axes fraction.
_ALPHA_VISIBLE = 0.01  # minimum alpha to consider "has background"
_PAD_TO_AXES_FRAC = 0.05  # overlay_padding (points) → axes-fraction padding
_ROUNDING_TO_AXES_FRAC = 0.04  # overlay_rounding (points) → FancyBboxPatch rounding


def _compact_formatter(value: float, _pos: object) -> str:
    """Format tick values compactly, keeping scientific notation on one line."""
    if value == 0:
        return "0"
    abs_val = abs(value)
    if 0.01 <= abs_val < 10000:
        return f"{value:g}"
    return f"{value:.1e}"


def _apply_extremes(
    mappable: ScalarMappable,
    *,
    mode: ExtremesMode = "semi",
    darken_factor: float = 0.75,
    semi_alpha: float = 0.3,
) -> None:
    """Style the over/under extension colors of a ScalarMappable.

    Parameters
    ----------
    mappable : ScalarMappable
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
    tick_color: str | tuple[float, float, float, float] | None = None,
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
    # cb.outline is typed Spine | None in mpl stubs; in practice always set.
    # mpl stub bug: Spine inherits from Patch but the inherited setters
    # confuse mypy ("Spine not callable"). Suppress the operator error.
    assert cb.outline is not None
    cb.outline.set_linewidth(outline_w)  # type: ignore[operator]
    cb.outline.set_edgecolor(outline_rgba)  # type: ignore[operator]
    cb.ax.tick_params(
        width=outline_w, length=tick_len, colors=tick_color, labelcolor=tick_color
    )


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
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
    cb = fig.colorbar(mappable, cax=cax, extend=extend)
    _style_colorbar(cb, label)
    return cb


def add_inset_colorbar(
    ax: Axes,
    mappable: ScalarMappable,
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
    ticks: list[float] | None = None,
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
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from pypic.plotting._badge import (
        _claim_corner,
        _detect_overlay_defaults,
        _resolve_border,
    )
    from pypic.plotting._overlay_common import resolve_rgba_override
    from pypic.plotting.styles import _theme_val

    actual_loc = _claim_corner(ax, "lower right", loc)

    default_bg, default_fg, overlay_alpha = _detect_overlay_defaults(variant)

    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, (*default_bg, overlay_alpha))
    fg_rgba = resolve_rgba_override(text_color, text_alpha, (*default_fg, 0.65))

    rounding: float = _theme_val("overlay_rounding", 0.6)
    overlay_pad: float = _theme_val("overlay_padding", 0.4)
    if pad is None:
        pad = _theme_val("overlay_margin", 0.03) * 0.5

    box_pad = overlay_pad * _PAD_TO_AXES_FRAC

    if fontsize is None:
        fontsize = _theme_val("font_overlay", 9.0)

    fig = ax.get_figure()
    if fig is None:
        msg = "axes has no parent figure"
        raise RuntimeError(msg)

    _apply_extremes(mappable, mode=extremes)
    if extremes is None:
        extend = "neither"

    # --- Pass 1: provisional colorbar at arbitrary position for measurement ---
    _prov = 0.3  # arbitrary axes-fraction origin; overwritten in pass 2
    prov_cax = ax.inset_axes((_prov, _prov, width, height), zorder=5)
    cb = fig.colorbar(mappable, cax=prov_cax, orientation="horizontal", extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.locator = MaxNLocator(nbins=n_ticks)
        cb.update_ticks()
    _style_colorbar(cb, tick_color=fg_rgba)
    prov_cax.xaxis.set_major_formatter(FuncFormatter(_compact_formatter))
    prov_cax.xaxis.get_offset_text().set_visible(False)
    tick_w: float = _theme_val("colorbar_outline_width", 0.3) * 2
    prov_cax.tick_params(
        labelsize=fontsize,
        colors=fg_rgba,
        top=True,
        bottom=False,
        labeltop=False,
        labelbottom=True,
        direction="in",
        width=tick_w,
    )
    if label:
        prov_cax.set_title(label, fontsize=fontsize, color=fg_rgba, pad=4)

    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # mpl backend stub gap
    fig.draw(renderer)
    bbox_disp = prov_cax.get_tightbbox(renderer)
    bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())  # type: ignore[union-attr]

    # Overhangs: how much text extends beyond the bar on each side
    overhang_left = _prov - bbox_ax.x0
    overhang_right = bbox_ax.x1 - (_prov + width)
    overhang_bottom = _prov - bbox_ax.y0
    overhang_top = bbox_ax.y1 - (_prov + height)

    # Ensure content width accommodates the title if it's wider than the bar.
    # get_tightbbox can underestimate title extent, so measure explicitly.
    if label and prov_cax.title.get_text():
        title_bbox = prov_cax.title.get_window_extent(renderer)
        title_w_ax = title_bbox.transformed(ax.transAxes.inverted()).width
        # Add a small buffer (half a character width) for font rendering variance
        char_w = title_w_ax / max(len(label), 1)
        title_overhang = max(0, (title_w_ax + char_w - width) / 2)
        overhang_left = max(overhang_left, title_overhang)
        overhang_right = max(overhang_right, title_overhang)

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
    bg_y = 1.0 - pad - total_h if "upper" in actual_loc else pad

    bar_x = bg_x + box_pad + overhang_left
    bar_y = bg_y + box_pad + overhang_bottom
    cax = ax.inset_axes((bar_x, bar_y, width, height), zorder=5)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.locator = MaxNLocator(nbins=n_ticks)
        cb.update_ticks()
    _style_colorbar(cb, tick_color=fg_rgba)
    cax.xaxis.set_major_formatter(FuncFormatter(_compact_formatter))
    cax.xaxis.get_offset_text().set_visible(False)
    cax.tick_params(
        labelsize=fontsize,
        colors=fg_rgba,
        top=True,
        bottom=False,
        labeltop=False,
        labelbottom=True,
        direction="in",
        width=tick_w,
    )
    if label:
        cax.set_title(label, fontsize=fontsize, color=fg_rgba, pad=4)
    cax.set_facecolor("none")

    # Background patch — initial size from provisional measurement
    has_bg = bg_rgba[3] >= _ALPHA_VISIBLE
    bg_patch = FancyBboxPatch(
        (bg_x, bg_y),
        total_w,
        total_h,
        boxstyle=f"round,pad=0,rounding_size={rounding * _ROUNDING_TO_AXES_FRAC:.4f}",
        facecolor=bg_rgba if has_bg else "none",
        edgecolor=_resolve_border(variant),
        linewidth=0.5,  # matches _badge._OVERLAY_BORDER_LW
        transform=ax.transAxes,
        zorder=4.9,
    )
    ax.add_patch(bg_patch)

    # At draw time, re-measure content and reposition both the background
    # patch and the colorbar axes so that the overlay respects the margin
    # even after set_xlim / set_aspect changes the axes layout.
    _loc_str = actual_loc
    # Store the original bar position in axes fraction for absolute
    # repositioning (avoids accumulating deltas across multiple draws).
    _orig_bar_x = bar_x
    _orig_bar_y = bar_y

    # Remove the inset locator so set_position sticks across draws.
    # mpl stubs declare locator as non-Optional but None is the documented
    # way to clear it; cast away the bogus strictness.
    cax.set_axes_locator(None)  # type: ignore[arg-type]

    def _resize_bg(event: object) -> None:
        r = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # mpl backend stub gap

        # Temporarily put cax back at its original position to get a
        # stable tightbbox measurement (avoids feedback loops).
        ax_pos = ax.get_position()
        cax.set_position(
            (
                ax_pos.x0 + _orig_bar_x * ax_pos.width,
                ax_pos.y0 + _orig_bar_y * ax_pos.height,
                width * ax_pos.width,
                height * ax_pos.height,
            )
        )

        tb = cax.get_tightbbox(r)
        if tb is None:
            return
        tb_ax = tb.transformed(ax.transAxes.inverted())
        new_w = tb_ax.width + 2 * box_pad
        new_h = tb_ax.height + 2 * box_pad

        # Where the background should be (anchored to corner with margin)
        if "right" in _loc_str:
            new_bg_x = 1.0 - pad - new_w
        elif "center" in _loc_str:
            new_bg_x = 0.5 - new_w / 2
        else:
            new_bg_x = pad
        new_bg_y = 1.0 - pad - new_h if "upper" in _loc_str else pad

        bg_patch.set_bounds(new_bg_x, new_bg_y, new_w, new_h)

        # Shift cax so content is centered in the background
        content_cx = tb_ax.x0 + tb_ax.width / 2
        content_cy = tb_ax.y0 + tb_ax.height / 2
        target_cx = new_bg_x + new_w / 2
        target_cy = new_bg_y + new_h / 2
        new_bar_x = _orig_bar_x + (target_cx - content_cx)
        new_bar_y = _orig_bar_y + (target_cy - content_cy)
        cax.set_position(
            (
                ax_pos.x0 + new_bar_x * ax_pos.width,
                ax_pos.y0 + new_bar_y * ax_pos.height,
                width * ax_pos.width,
                height * ax_pos.height,
            )
        )

    fig.canvas.mpl_connect("draw_event", _resize_bg)

    return cb


def attach_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
    label: str,
    colorbar: bool | Literal["inset"],
    *,
    extremes: ExtremesMode = "semi",
    variant: OverlayVariant | None = None,
    ticks: list[float] | None = None,
) -> None:
    """Dispatch to the appropriate colorbar function, or do nothing.

    Parameters
    ----------
    fig : Figure
        Parent figure (used for standard colorbars).
    ax : Axes
        Target axes.
    mappable : ScalarMappable
        A ``ScalarMappable``.
    label : str
        Colorbar label.
    colorbar : bool or "inset"
        ``True`` for a standard side colorbar, ``"inset"`` for an
        overlay colorbar, ``False`` to skip.
    extremes : "semi", "transparent", "darken", or None
        How to style over/under values. ``None`` uses matplotlib defaults.
    variant : "darker", "lighter", "alt", or None
        Overlay variant for inset colorbar styling.
    ticks : list[float] or None
        Explicit tick positions. ``None`` uses automatic ticks.
    """
    if not colorbar:
        return
    if colorbar == "inset":
        add_inset_colorbar(
            ax,
            mappable,
            label,
            extremes=extremes,
            variant=variant,
            ticks=ticks,
        )
    else:
        cb = add_colorbar(fig, ax, mappable, label, extremes=extremes)
        if ticks is not None:
            cb.set_ticks(ticks)
