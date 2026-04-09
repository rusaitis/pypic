"""In-plot overlay elements: status badge, vector legend, panel label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pypic.plotting._format import (
    BadgeLoc,
    _format_status_text,
    _format_time_value,  # noqa: F401  re-exported for backward compat
)
from pypic.plotting._guard import ensure_matplotlib
from pypic.plotting._overlay_common import (
    ALPHA_VISIBLE,
    contrast_ratio,
    resolve_rgba_override,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, OffsetBox

# _format_time_value and _format_status_text are re-exported so callers
# (including tests) that imported them directly from this module still
# work. The canonical home is now pypic.plotting._format.
__all__ = [
    "BadgeLoc",
    "LegendEntry",
    "add_badge",
    "add_label",
    "add_legend",
]

# Overlay layout constants — centralized so every overlay (badge, legend,
# label, progress bar) uses the same visual language.
_OVERLAY_BORDER_LW = 0.5  # edge linewidth on all overlay boxes
_BLEND_FACTOR = 0.4  # darken/lighten blend toward black/white
_CHAR_WIDTH_RATIO = 0.55  # average glyph width / font size (monospace ≈ 0.6)
_LINE_HEIGHT_RATIO = 0.4  # DrawingArea height / font size for legend lines
_ARROW_SCALE = 2.5  # arrow tip size relative to line width
_ARROW_MUTATION = 1.5  # FancyArrowPatch mutation_scale / arrow_size
_SINGLE_CHAR_PAD = 0.1  # extra h-pad so single-letter labels look square
_MIN_VISIBLE_FILL = 0.02  # skip bar fill below 2% (invisible at badge scale)
_BAR_WIDTH_MIN = 60.0  # auto-sized progress bar minimum width (points)
_BAR_WIDTH_MAX = 200.0  # auto-sized progress bar maximum width (points)
_OVERLAY_SEP = 3  # standard VPacker/HPacker separator (points)

_CORNERS: tuple[BadgeLoc, ...] = (
    "upper left",
    "upper right",
    "lower right",
    "lower left",
)

_OCCUPIED_ATTR = "_pypic_occupied_corners"


def _claim_corner(
    ax: Axes, preferred: BadgeLoc, explicit: BadgeLoc | None = None
) -> BadgeLoc:
    """Pick an overlay corner, avoiding already-occupied ones.

    If *explicit* is given it is used as-is (user override).  Otherwise
    *preferred* is tried first, then the remaining corners in clockwise
    order.  If all four are taken, *preferred* is returned anyway.
    """
    occupied: set[BadgeLoc] = getattr(ax, _OCCUPIED_ATTR, set())

    chosen: BadgeLoc = explicit if explicit is not None else preferred
    if explicit is None and chosen in occupied:
        for corner in _CORNERS:
            if corner not in occupied:
                chosen = corner
                break

    occupied.add(chosen)
    setattr(ax, _OCCUPIED_ATTR, occupied)
    return chosen


def _resolve_border(
    variant: OverlayVariant | None = None,
) -> tuple[float, float, float, float]:
    """Return the overlay border color for the given variant."""
    from pypic.plotting.styles import _theme_val

    if variant == "alt":
        return _theme_val("overlay_alt_border_color", (0.5, 0.5, 0.5, 0.3))
    return _theme_val("overlay_border_color", (0.3, 0.3, 0.3, 0.2))


def _make_overlay_box(
    ax: Axes,
    child: OffsetBox,
    loc: BadgeLoc,
    bg_rgba: tuple[float, float, float, float],
    pad: float | None = None,
    variant: OverlayVariant | None = None,
) -> AnchoredOffsetbox:
    """Create a styled overlay box and add it to *ax*."""
    from matplotlib.offsetbox import AnchoredOffsetbox

    from pypic.plotting.styles import _overlay_box_style, _theme_val

    if pad is None:
        pad = _theme_val("overlay_padding", 0.4)
    margin: float = _theme_val("overlay_margin", 0.03)

    has_bg = bg_rgba[3] >= ALPHA_VISIBLE
    box = AnchoredOffsetbox(
        loc=loc,
        child=child,
        pad=pad,
        # AnchoredOffsetbox pad unit is font-size; ×20 ≈ points
        borderpad=margin * 20,
        frameon=has_bg,
    )
    if has_bg:
        box.patch.set_boxstyle(_overlay_box_style())
        box.patch.set_facecolor(bg_rgba)
        box.patch.set_edgecolor(_resolve_border(variant))
        box.patch.set_linewidth(_OVERLAY_BORDER_LW)
    ax.add_artist(box)
    return box


OverlayVariant = Literal["darker", "lighter", "alt"]


def _detect_overlay_defaults(
    variant: OverlayVariant | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Derive overlay bg, fg colors and alpha from the active theme.

    When a :func:`~pypic.plotting.use_theme` context is active and the
    theme defines an ``overlay_color``, that color is used directly.
    Otherwise the overlay background is the axes facecolor blended toward
    black (``"darker"``) or white (``"lighter"``).

    Parameters
    ----------
    variant : "darker", "lighter", "alt", or None
        ``"alt"`` uses the theme's alternative overlay colors.
        ``None`` auto-selects based on axes facecolor luminance.

    Returns
    -------
    tuple[tuple, tuple, float]
        ``(default_bg, default_fg, bg_alpha)``
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgba

    from pypic.plotting.styles import get_active_theme, get_theme

    # Use the active theme if inside use_theme(), otherwise fall back to
    # the global default.  This ensures overlays created outside a
    # use_theme() context still pick up theme colors.
    theme = get_active_theme() or get_theme()

    # "alt" variant: use alternative overlay colors from theme
    if variant == "alt" and theme is not None:
        return (
            theme.overlay_alt_color[:3],
            theme.overlay_alt_text_color[:3],
            theme.overlay_alt_color[3],
        )

    # Primary overlay color from theme
    if theme is not None and any(c > 0 for c in theme.overlay_color[:3]):
        default_bg: tuple[float, float, float] = theme.overlay_color[:3]
        alpha = theme.overlay_color[3]
    else:
        # Fallback: darken or lighten the axes facecolor
        bg = to_rgba(mpl.rcParams.get("axes.facecolor", "white"))
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        if variant is None:
            variant = "darker" if luminance < 0.5 else "lighter"
        alpha = 0.65  # fallback when no theme — matches default overlay_color[3]
        f = _BLEND_FACTOR
        if variant == "darker":
            default_bg = (bg[0] * f, bg[1] * f, bg[2] * f)
        else:
            default_bg = (
                bg[0] + (1.0 - bg[0]) * f,
                bg[1] + (1.0 - bg[1]) * f,
                bg[2] + (1.0 - bg[2]) * f,
            )

    # Overlay text color from theme, or fall back to rcParams
    if theme is not None:
        default_fg: tuple[float, float, float] = theme.overlay_text_color[:3]
    else:
        fg_rgba = to_rgba(mpl.rcParams.get("text.color", "black"))
        default_fg = (fg_rgba[0], fg_rgba[1], fg_rgba[2])

    return default_bg, default_fg, alpha


def _pick_overlay_variant(
    content_colors: list[tuple[float, float, float]],
) -> OverlayVariant | None:
    """Choose default or alt overlay variant for best contrast.

    Compares the average contrast ratio of *content_colors* against the
    default overlay background and the alt overlay background.  Returns
    ``"alt"`` if the alt background gives better contrast, ``None``
    (default overlay) otherwise.
    """
    default_bg = _detect_overlay_defaults(None)[0]
    alt_bg = _detect_overlay_defaults("alt")[0]

    avg_default = sum(contrast_ratio(c, default_bg) for c in content_colors) / len(
        content_colors
    )
    avg_alt = sum(contrast_ratio(c, alt_bg) for c in content_colors) / len(
        content_colors
    )
    return "alt" if avg_alt > avg_default else None


def _build_progress_bar(
    fraction: float,
    bar_width: float,
    bar_height: float,
    bar_color: str,
    bar_alpha: float,
    track_color: tuple[float, ...],
    variant: OverlayVariant | None = None,
) -> DrawingArea:
    """Build a DrawingArea containing track + fill rectangles."""
    if bar_width <= 0 or bar_height <= 0:
        msg = "bar_width and bar_height must be positive"
        raise ValueError(msg)

    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import FancyBboxPatch

    from pypic.plotting.styles import _theme_val

    overlay_rounding: float = _theme_val("overlay_rounding", 0.6)
    # ×0.7 reduces rounding relative to bar height; cap at half for pill shape
    rounding = min(overlay_rounding * bar_height * 0.7, bar_height / 2)
    box_style = f"round,pad=0,rounding_size={rounding}"

    drawing = DrawingArea(bar_width, bar_height)

    border = _resolve_border(variant)
    track = FancyBboxPatch(
        (0, 0),
        bar_width,
        bar_height,
        boxstyle=box_style,
        facecolor=track_color,
        edgecolor=border,
        linewidth=_OVERLAY_BORDER_LW,
    )
    drawing.add_artist(track)

    if fraction >= _MIN_VISIBLE_FILL:
        fill_width = bar_width * fraction
        fill = FancyBboxPatch(
            (0, 0),
            fill_width,
            bar_height,
            boxstyle=box_style,
            facecolor=bar_color,
            edgecolor="none",
            alpha=bar_alpha,
        )
        drawing.add_artist(fill)

    return drawing


def add_badge(
    ax: Axes,
    text: str | None = None,
    *,
    step: int | None = None,
    time: float | str | None = None,
    time_units: str = "",
    step_range: tuple[int, int] | None = None,
    label: str | None = None,
    show_max: bool = True,
    progress: float | None = None,
    variant: OverlayVariant | None = None,
    loc: BadgeLoc | None = None,
    fontsize: float | None = None,
    bar_color: str | None = None,
    bar_alpha: float = 0.8,
    bar_width: float | None = None,
    bar_height: float | None = None,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
    track_color: str | tuple[float, ...] | None = None,
) -> AnchoredOffsetbox:
    r"""Add a status badge overlay to an axes.

    Renders simulation step, time, or custom text as a rounded box.
    A progress bar is shown when *step_range* or *progress* is set.

    Parameters
    ----------
    ax : Axes
        Target axes for the badge.
    text : str, optional
        Direct custom text. When provided, *step*/*time*/*label* are
        ignored for text generation (but *step_range*/*progress* still
        drive the progress bar).
    step : int, optional
        Current simulation step number.
    time : float or str, optional
        Simulation time. A float is auto-formatted; a string is used
        verbatim (e.g. ``"13:34"``).
    time_units : str
        Unit label appended to the time value (e.g. ``"ns"``).
    step_range : tuple[int, int], optional
        ``(start, end)`` step range for auto-computing progress.
    label : str or None
        Custom label prefix (e.g. ``"Cycle"``, ``"Step"``). ``None``
        uses auto-labels (``"step"`` / ``"t"``). ``""`` suppresses.
    show_max : bool
        When ``step_range`` is set, show ``"X / Y"`` if True.
    progress : float or None
        Explicit progress fraction (0.0 to 1.0). Overrides *step_range*
        auto-computation when both are given.
    variant : "darker", "lighter", "alt", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc or None
        Badge placement. ``None`` auto-selects (prefers upper right,
        avoids corners already claimed by other overlays).
    fontsize : float
        Font size for the status text.
    bar_color : str
        Fill color for the progress bar.
    bar_alpha : float
        Opacity of the progress bar fill.
    bar_width : float
        Width of the progress bar in points.
    bar_height : float
        Height of the progress bar in points.
    bg_color : str, tuple, or None
        Box background color override. ``None`` derives from *variant*.
    bg_alpha : float
        Box background opacity.
    text_color : str, tuple, or None
        Text color override. ``None`` uses rcParams text color.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).
    track_color : str, tuple, or None
        Progress bar track color override.

    Returns
    -------
    AnchoredOffsetbox
        The badge artist added to the axes.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> _ = add_badge(ax, step=42)
    >>> _ = add_badge(ax, step=100, label="Cycle")
    >>> _ = add_badge(ax, time="13:34")
    >>> _ = add_badge(ax, "Harris sheet, δ = 0.5 d_i")
    >>> plt.close(fig)
    """
    ensure_matplotlib()

    from matplotlib.offsetbox import TextArea, VPacker

    from pypic.plotting.styles import _theme_val

    actual_loc = _claim_corner(ax, "upper right", loc)

    default_bg, default_fg, overlay_alpha = _detect_overlay_defaults(variant)

    if fontsize is None:
        fontsize = _theme_val("font_overlay", 9.0)
    if bar_color is None:
        bar_color = _theme_val("accent_color", "#e8913a")
    auto_bar_width = bar_width is None
    if bar_width is None:
        bar_width = _theme_val("progress_bar_width", 80.0)
    if bar_height is None:
        bar_height = _theme_val("progress_bar_height", 4.0)

    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, (*default_bg, overlay_alpha))
    resolved_text = resolve_rgba_override(text_color, text_alpha, (*default_fg, 0.65))

    if track_color is None:
        track_key = "track_alt_color" if variant == "alt" else "track_color"
        track_rgba = _theme_val(track_key, (0.3, 0.3, 0.3, 0.3))
    else:
        effective_alpha = overlay_alpha if bg_alpha is None else bg_alpha
        track_rgba = resolve_rgba_override(
            track_color, effective_alpha * 0.4, (*default_fg, 0.65)
        )

    status_text = _format_status_text(
        text=text,
        step=step,
        time=time,
        time_units=time_units,
        step_range=step_range,
        label=label,
        show_max=show_max,
    )

    if not status_text:
        msg = "Provide text, step, or time for the badge"
        raise ValueError(msg)

    text_props = {"fontsize": fontsize, "color": resolved_text}
    text_area = TextArea(status_text, textprops=text_props)

    # Determine progress fraction
    fraction: float | None = progress
    if fraction is None and step_range is not None and step is not None:
        start, end = step_range
        if start > end:
            msg = f"step_range start must be <= end, got ({start}, {end})"
            raise ValueError(msg)
        fraction = max(0.0, min(1.0, (step - start) / max(end - start, 1)))

    child: OffsetBox
    if fraction is not None:
        # Scale bar width to match text length when using the default
        if auto_bar_width:
            estimated = len(status_text) * fontsize * _CHAR_WIDTH_RATIO
            bar_width = max(_BAR_WIDTH_MIN, min(_BAR_WIDTH_MAX, estimated))
        bar = _build_progress_bar(
            max(0.0, min(1.0, fraction)),
            bar_width,
            bar_height,
            bar_color,
            bar_alpha,
            track_rgba,
            variant=variant,
        )
        child = VPacker(
            children=[text_area, bar], pad=0, sep=_OVERLAY_SEP, align="left"
        )
    else:
        child = text_area

    return _make_overlay_box(ax, child, actual_loc, bg_rgba, variant=variant)


@dataclass(frozen=True, slots=True)
class LegendEntry:
    """One row in a vector legend: a sample line and its label."""

    label: str
    color: str | None = None
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float = 1.0


def add_label(
    ax: Axes,
    label: str,
    *,
    variant: OverlayVariant | None = None,
    loc: BadgeLoc | None = None,
    fontsize: float | None = None,
    fontweight: str = "bold",
    ha: str = "left",
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
) -> AnchoredOffsetbox:
    r"""Add a bold panel letter (a, b, c, ...) overlay to an axes.

    Styled with the same rounded box as :func:`add_badge`.

    Parameters
    ----------
    ax : Axes
        Target axes.
    label : str
        Panel label text (e.g. ``"a"``, ``"b"``).
    variant : "darker", "lighter", "alt", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc or None
        Placement. ``None`` auto-selects (prefers upper left).
    fontsize : float
        Label font size.
    fontweight : str
        Font weight (default ``"bold"``).
    bg_color, bg_alpha, text_color
        Color overrides; same semantics as :func:`add_badge`.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).

    Returns
    -------
    AnchoredOffsetbox
    """
    ensure_matplotlib()
    from matplotlib.offsetbox import HPacker, TextArea

    from pypic.plotting.styles import _theme_val

    actual_loc = _claim_corner(ax, "upper left", loc)

    is_phrase = " " in label
    if fontsize is None:
        fontsize = _theme_val("font_label" if is_phrase else "font_title", 12.0)
    if fontweight == "bold" and is_phrase:
        fontweight = "normal"

    default_bg, default_fg, overlay_alpha = _detect_overlay_defaults(variant)

    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, (*default_bg, overlay_alpha))
    resolved_text = resolve_rgba_override(text_color, text_alpha, (*default_fg, 0.65))

    props = {
        "fontsize": fontsize,
        "fontweight": fontweight,
        "color": resolved_text,
        "ha": ha,
    }
    text_area = TextArea(label, textprops=props, multilinebaseline=True)

    # Extra horizontal padding so the box looks square for single letters
    pad = 0 if is_phrase else fontsize * _SINGLE_CHAR_PAD
    child = HPacker(children=[text_area], pad=pad, sep=0, align="center")

    return _make_overlay_box(ax, child, actual_loc, bg_rgba, variant=variant)


def add_legend(
    ax: Axes,
    entries: str | LegendEntry | list[LegendEntry],
    *,
    color: str | None = None,
    linewidth: float = 1.0,
    entry_alpha: float = 1.0,
    variant: OverlayVariant | None = None,
    loc: BadgeLoc | None = None,
    fontsize: float | None = None,
    sample_width: float = 20,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
) -> AnchoredOffsetbox:
    r"""Add a vector legend overlay showing colored line samples with labels.

    Each entry renders as a short line segment (using the entry's color,
    linewidth, linestyle, alpha) next to its label text.  Multiple entries
    are stacked vertically.  The box style matches :func:`add_badge`.

    For a single entry, pass a string label directly::

        add_legend(ax, "J", color="white")

    Parameters
    ----------
    ax : Axes
        Target axes.
    entries : str, LegendEntry, or list[LegendEntry]
        A field label string, a single entry, or multiple entries.
    color : str or None
        Line color when *entries* is a string.
    linewidth : float
        Line width when *entries* is a string.
    entry_alpha : float
        Line opacity when *entries* is a string.
    variant : "darker", "lighter", "alt", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc or None
        Placement. ``None`` auto-selects (prefers lower left).
    fontsize : float
        Label font size.
    sample_width : float
        Width of the sample line in points.
    bg_color, bg_alpha, text_color
        Color overrides; same semantics as :func:`add_badge`.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).

    Returns
    -------
    AnchoredOffsetbox
    """
    ensure_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import (
        DrawingArea,
        HPacker,
        TextArea,
        VPacker,
    )
    from matplotlib.patches import FancyArrowPatch

    from pypic.plotting.styles import _theme_val

    actual_loc = _claim_corner(ax, "lower left", loc)

    if fontsize is None:
        fontsize = _theme_val("font_overlay", 9.0)

    if isinstance(entries, str):
        entries = [
            LegendEntry(
                label=entries,
                color=color,
                linewidth=linewidth,
                alpha=entry_alpha,
            )
        ]
    elif isinstance(entries, LegendEntry):
        entries = [entries]

    # Auto-select overlay variant for best contrast with entry colors
    if variant is None:
        from matplotlib.colors import to_rgba

        entry_rgbs = [to_rgba(e.color)[:3] for e in entries if e.color is not None]
        if entry_rgbs:
            variant = _pick_overlay_variant(entry_rgbs)

    default_bg, default_fg, overlay_alpha = _detect_overlay_defaults(variant)

    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, (*default_bg, overlay_alpha))
    resolved_text = resolve_rgba_override(text_color, text_alpha, (*default_fg, 0.65))

    rows: list[Artist] = []
    line_height = fontsize * _LINE_HEIGHT_RATIO
    for entry in entries:
        drawing = DrawingArea(sample_width, line_height)
        mid_y = line_height / 2
        base_arrow: float = _theme_val("arrow_size", 4.0)
        arrow_size = max(base_arrow, entry.linewidth * _ARROW_SCALE)
        line = Line2D(
            [0, sample_width - arrow_size],
            [mid_y, mid_y],
            color=entry.color,
            linewidth=entry.linewidth,
            linestyle=entry.linestyle,
            alpha=entry.alpha,
        )
        drawing.add_artist(line)
        arrow = FancyArrowPatch(
            (sample_width - arrow_size, mid_y),
            (sample_width, mid_y),
            arrowstyle="-|>",
            mutation_scale=arrow_size * _ARROW_MUTATION,
            color=entry.color,
            linewidth=entry.linewidth,
            alpha=entry.alpha,
        )
        drawing.add_artist(arrow)

        text = TextArea(
            entry.label,
            textprops={
                "fontsize": fontsize,
                "fontweight": "bold",
                "color": resolved_text,
            },
        )
        row = HPacker(children=[drawing, text], pad=0, sep=_OVERLAY_SEP, align="center")
        rows.append(row)

    legend_child: OffsetBox
    if len(rows) > 1:
        legend_child = VPacker(children=rows, pad=0, sep=_OVERLAY_SEP, align="left")
    else:
        legend_child = rows[0]  # type: ignore[assignment]  # HPacker is an OffsetBox

    return _make_overlay_box(
        ax,
        legend_child,
        actual_loc,
        bg_rgba,
        variant=variant,
    )
