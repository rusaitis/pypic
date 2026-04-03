"""In-plot overlay elements: status badge, vector legend, panel label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea

BadgeLoc = Literal[
    "upper left", "upper right", "lower left", "lower right", "upper center"
]

_LOC_CODES: dict[str, int] = {
    "upper right": 1,
    "upper left": 2,
    "lower left": 3,
    "lower right": 4,
    "upper center": 9,
}

_CORNERS: tuple[str, ...] = (
    "upper left",
    "upper right",
    "lower right",
    "lower left",
)

_OCCUPIED_ATTR = "_pypic_occupied_corners"


def _claim_corner(
    ax: Axes, preferred: str, explicit: BadgeLoc | None = None
) -> str:
    """Pick an overlay corner, avoiding already-occupied ones.

    If *explicit* is given it is used as-is (user override).  Otherwise
    *preferred* is tried first, then the remaining corners in clockwise
    order.  If all four are taken, *preferred* is returned anyway.
    """
    occupied: set[str] = getattr(ax, _OCCUPIED_ATTR, set())

    chosen = explicit if explicit is not None else preferred
    if explicit is None and chosen in occupied:
        for corner in _CORNERS:
            if corner not in occupied:
                chosen = corner
                break

    occupied.add(chosen)
    setattr(ax, _OCCUPIED_ATTR, occupied)
    return chosen


def _resolve_border(variant: OverlayVariant | None = None) -> tuple[float, ...]:
    """Return the overlay border color for the given variant."""
    from pypic.plotting.styles import _theme_val

    if variant == "alt":
        return _theme_val("overlay_alt_border_color", (0.5, 0.5, 0.5, 0.3))
    return _theme_val("overlay_border_color", (0.3, 0.3, 0.3, 0.2))


def _make_overlay_box(
    ax: Axes,
    child: object,
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

    loc_code = _LOC_CODES.get(loc, 1)
    has_bg = bg_rgba[3] >= 0.01
    box = AnchoredOffsetbox(
        loc=loc_code,
        child=child,
        pad=pad,
        borderpad=margin * 20,  # convert axes fraction to approx points
        frameon=has_bg,
    )
    if has_bg:
        box.patch.set_boxstyle(_overlay_box_style())
        box.patch.set_facecolor(bg_rgba)
        box.patch.set_edgecolor(_resolve_border(variant))
        box.patch.set_linewidth(0.5)
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
    alpha = 0.65
    if theme is not None and any(c > 0 for c in theme.overlay_color[:3]):
        default_bg: tuple[float, float, float] = theme.overlay_color[:3]
        alpha = theme.overlay_color[3]
    else:
        # Fallback: darken or lighten the axes facecolor
        bg = to_rgba(mpl.rcParams.get("axes.facecolor", "white"))
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        if variant is None:
            variant = "darker" if luminance < 0.5 else "lighter"
        f = 0.4
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


def _contrast_ratio(
    c1: tuple[float, float, float], c2: tuple[float, float, float]
) -> float:
    """WCAG relative-luminance contrast ratio between two RGB colors."""

    def _lum(c: tuple[float, float, float]) -> float:
        r, g, b = (
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in c
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1, l2 = _lum(c1), _lum(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


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

    avg_default = sum(_contrast_ratio(c, default_bg) for c in content_colors) / len(
        content_colors
    )
    avg_alt = sum(_contrast_ratio(c, alt_bg) for c in content_colors) / len(
        content_colors
    )
    return "alt" if avg_alt > avg_default else None


def _format_time_value(time: float, units: str) -> tuple[str, str]:
    """Format a numeric time value, returning ``(value_str, suffix_str)``.

    When *units* is ``"s"`` and *time* >= 60, produces human-readable
    durations like ``"2min 30s"`` or ``"1h 5min 12s"`` (suffix is empty
    since units are embedded). Otherwise falls back to numeric formatting.
    """
    if units == "s" and abs(time) >= 60:
        sign = "-" if time < 0 else ""
        t = abs(time)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if h > 0:
            return f"{sign}{h}h {m}min {s}s", ""
        return f"{sign}{m}min {s}s", ""
    suffix = f" {units}" if units else ""
    if abs(time) >= 1e4 or (0 < abs(time) < 0.01):
        return f"{time:.2e}", suffix
    return f"{time:.2f}", suffix


def _format_status_text(
    *,
    text: str | None,
    step: int | None,
    time: float | str | None,
    time_units: str,
    step_range: tuple[int, int] | None,
    label: str | None = None,
    show_max: bool = True,
) -> str:
    """Build the status text string from step/time/text parameters.

    Parameters
    ----------
    text : str or None
        Direct custom text. When provided, *step*/*time*/*label* are
        ignored for text generation.
    label : str or None
        Custom prefix for the step (or time-only) display.
        ``None`` uses auto-labels (``"step"`` / ``"t"``).
        ``""`` suppresses the prefix entirely.
    show_max : bool
        When ``step_range`` is set, include ``" / max"`` after the step.
    """
    if text is not None:
        return text

    parts: list[str] = []

    if step is not None:
        step_label = "step" if label is None else label
        prefix = f"{step_label} " if step_label else ""
        if step_range is not None and show_max:
            parts.append(f"{prefix}{step} / {step_range[1]}")
        else:
            parts.append(f"{prefix}{step}")

    if time is not None:
        if isinstance(time, str):
            t_str = time
            suffix = f" {time_units}" if time_units else ""
        else:
            t_str, suffix = _format_time_value(time, time_units)
        # When time-only, label replaces the "t" prefix; with step, time keeps "t"
        time_label = "t" if step is not None else ("t" if label is None else label)
        if time_label:
            parts.append(f"{time_label} = {t_str}{suffix}")
        else:
            parts.append(f"{t_str}{suffix}")

    return ", ".join(parts) if parts else ""


def _resolve_rgba(
    color: str | tuple[float, ...] | None,
    alpha: float | None,
    fallback: tuple[float, float, float],
    fallback_alpha: float = 0.65,
) -> tuple[float, float, float, float]:
    """Resolve an optional color override to RGBA, falling back to *fallback*.

    When *alpha* is ``None``, the theme's overlay alpha (*fallback_alpha*)
    is used.  Pass ``0.0`` explicitly for a transparent overlay.
    """
    a = fallback_alpha if alpha is None else alpha
    if color is None:
        return (*fallback, a)
    from matplotlib.colors import to_rgba

    return (*to_rgba(color)[:3], a)


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
        linewidth=0.5,
    )
    drawing.add_artist(track)

    if fraction >= 0.02:
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

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg, overlay_alpha)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    if track_color is None:
        track_key = "track_alt_color" if variant == "alt" else "track_color"
        track_rgba = _theme_val(track_key, (0.3, 0.3, 0.3, 0.3))
    else:
        effective_alpha = overlay_alpha if bg_alpha is None else bg_alpha
        track_rgba = _resolve_rgba(track_color, effective_alpha * 0.4, default_fg)

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

    if fraction is not None:
        # Scale bar width to match text length when using the default
        if auto_bar_width:
            estimated = len(status_text) * fontsize * 0.55
            bar_width = max(60, min(200, estimated))
        bar = _build_progress_bar(
            max(0.0, min(1.0, fraction)),
            bar_width,
            bar_height,
            bar_color,
            bar_alpha,
            track_rgba,
            variant=variant,
        )
        child = VPacker(children=[text_area, bar], pad=0, sep=3, align="left")
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

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg, overlay_alpha)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    props = {"fontsize": fontsize, "fontweight": fontweight, "color": resolved_text, "ha": ha}
    text_area = TextArea(label, textprops=props, multilinebaseline=True)

    # Extra horizontal padding so the box looks square for single letters
    pad = 0 if is_phrase else fontsize * 0.12
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
        entries = [LegendEntry(label=entries, color=color, linewidth=linewidth, alpha=entry_alpha)]
    elif isinstance(entries, LegendEntry):
        entries = [entries]

    # Auto-select overlay variant for best contrast with entry colors
    if variant is None:
        from matplotlib.colors import to_rgba

        entry_rgbs = [to_rgba(e.color)[:3] for e in entries if e.color is not None]
        if entry_rgbs:
            variant = _pick_overlay_variant(entry_rgbs)

    default_bg, default_fg, overlay_alpha = _detect_overlay_defaults(variant)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg, overlay_alpha)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    rows: list[HPacker] = []
    line_height = fontsize * 0.4
    for entry in entries:
        drawing = DrawingArea(sample_width, line_height)
        mid_y = line_height / 2
        base_arrow: float = _theme_val("arrow_size", 4.0)
        arrow_size = max(base_arrow, entry.linewidth * 2.5)
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
            mutation_scale=arrow_size * 1.5,
            color=entry.color,
            linewidth=entry.linewidth,
            alpha=entry.alpha,
        )
        drawing.add_artist(arrow)

        text = TextArea(
            entry.label,
            textprops={"fontsize": fontsize, "fontweight": "bold", "color": resolved_text},
        )
        row = HPacker(children=[drawing, text], pad=0, sep=4, align="center")
        rows.append(row)

    if len(rows) > 1:
        child = VPacker(children=rows, pad=0, sep=3, align="left")
    else:
        child = rows[0]

    return _make_overlay_box(ax, child, actual_loc, bg_rgba, variant=variant)
