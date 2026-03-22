"""In-plot overlay elements: status badge, vector legend, panel label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea

from pypic.plotting.styles import OVERLAY_BORDER_PAD

OVERLAY_BOX_STYLE = "round,pad=0.4,rounding_size=0.6"

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


def _make_overlay_box(
    ax: Axes,
    child: object,
    loc: BadgeLoc,
    bg_rgba: tuple[float, float, float, float],
) -> AnchoredOffsetbox:
    """Create a styled overlay box and add it to *ax*."""
    from matplotlib.offsetbox import AnchoredOffsetbox

    loc_code = _LOC_CODES.get(loc, 1)
    box = AnchoredOffsetbox(
        loc=loc_code,
        child=child,
        pad=0.4,
        borderpad=OVERLAY_BORDER_PAD,
        frameon=True,
    )
    box.patch.set_boxstyle(OVERLAY_BOX_STYLE)
    box.patch.set_facecolor(bg_rgba)
    box.patch.set_edgecolor("none")
    ax.add_artist(box)
    return box


OverlayShade = Literal["darker", "lighter"]

_SHADE_FACTOR = 0.4


def _detect_overlay_defaults(
    shade: OverlayShade | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Derive overlay bg and fg colors from rcParams.

    The overlay background is the axes facecolor blended toward black
    (``"darker"``) or white (``"lighter"``).  When *shade* is ``None``,
    the direction is auto-selected from the axes facecolor luminance.
    Foreground color is always read from ``rcParams["text.color"]`` so
    that custom themes propagate into overlay text automatically.

    Parameters
    ----------
    shade : "darker", "lighter", or None
        ``None`` auto-selects based on axes facecolor luminance.

    Returns
    -------
    tuple[tuple, tuple]
        ``(default_bg, default_fg)``
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgba

    bg = to_rgba(mpl.rcParams.get("axes.facecolor", "white"))
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]

    if shade is None:
        shade = "darker" if luminance < 0.5 else "lighter"

    if shade == "darker":
        f = _SHADE_FACTOR
        default_bg = (bg[0] * f, bg[1] * f, bg[2] * f)
    else:
        f = _SHADE_FACTOR
        default_bg = (
            bg[0] + (1.0 - bg[0]) * f,
            bg[1] + (1.0 - bg[1]) * f,
            bg[2] + (1.0 - bg[2]) * f,
        )

    fg_rgba = to_rgba(mpl.rcParams.get("text.color", "black"))
    default_fg: tuple[float, float, float] = (fg_rgba[0], fg_rgba[1], fg_rgba[2])

    return default_bg, default_fg


def _format_status_text(
    *,
    step: int | None,
    time: float | None,
    time_units: str,
    step_range: tuple[int, int] | None,
    label: str | None = None,
    show_max: bool = True,
) -> str:
    """Build the status text string from step/time parameters.

    Parameters
    ----------
    label : str or None
        Custom prefix for the step (or time-only) display.
        ``None`` uses auto-labels (``"step"`` / ``"t"``).
        ``""`` suppresses the prefix entirely.
    show_max : bool
        When ``step_range`` is set, include ``" / max"`` after the step.
    """
    parts: list[str] = []

    if step is not None:
        step_label = "step" if label is None else label
        prefix = f"{step_label} " if step_label else ""
        if step_range is not None and show_max:
            parts.append(f"{prefix}{step} / {step_range[1]}")
        else:
            parts.append(f"{prefix}{step}")

    if time is not None:
        if abs(time) >= 1e4 or (0 < abs(time) < 0.01):
            t_str = f"{time:.2e}"
        else:
            t_str = f"{time:.2f}"
        suffix = f" {time_units}" if time_units else ""
        # When time-only, label replaces the "t" prefix; with step, time keeps "t"
        time_label = "t" if step is not None else ("t" if label is None else label)
        if time_label:
            parts.append(f"{time_label} = {t_str}{suffix}")
        else:
            parts.append(f"{t_str}{suffix}")

    return ", ".join(parts) if parts else ""


def _resolve_rgba(
    color: str | tuple[float, ...] | None,
    alpha: float,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Resolve an optional color override to RGBA, falling back to *fallback*."""
    if color is None:
        return (*fallback, alpha)
    from matplotlib.colors import to_rgba

    return (*to_rgba(color)[:3], alpha)


def _build_progress_bar(
    fraction: float,
    bar_width: float,
    bar_height: float,
    bar_color: str,
    bar_alpha: float,
    track_color: tuple[float, ...],
) -> DrawingArea:
    """Build a DrawingArea containing track + fill rectangles."""
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import FancyBboxPatch

    drawing = DrawingArea(bar_width, bar_height)

    track = FancyBboxPatch(
        (0, 0),
        bar_width,
        bar_height,
        boxstyle="round,pad=0,rounding_size=2",
        facecolor=track_color,
        edgecolor="none",
    )
    drawing.add_artist(track)

    if fraction >= 0.02:
        fill_width = bar_width * fraction
        fill = FancyBboxPatch(
            (0, 0),
            fill_width,
            bar_height,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=bar_color,
            edgecolor="none",
            alpha=bar_alpha,
        )
        drawing.add_artist(fill)

    return drawing


def add_status_badge(
    ax: Axes,
    *,
    step: int | None = None,
    time: float | None = None,
    time_units: str = "",
    step_range: tuple[int, int] | None = None,
    label: str | None = None,
    show_max: bool = True,
    shade: OverlayShade | None = None,
    loc: BadgeLoc = "upper right",
    fontsize: float = 9,
    bar_color: str = "#e8913a",
    bar_alpha: float = 0.8,
    bar_width: float = 80,
    bar_height: float = 4,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float = 0.65,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.85,
    track_color: str | tuple[float, ...] | None = None,
) -> AnchoredOffsetbox:
    r"""Add a status badge overlay to an axes.

    Renders simulation step and/or time as a rounded box with a progress
    bar when ``step_range`` is provided. The overlay background is
    derived from the axes facecolor — darkened or lightened depending
    on the *shade* parameter (auto-detected by default). Text color
    is read from ``rcParams["text.color"]`` so that custom themes
    propagate automatically. Override individual colors with
    ``bg_color``, ``text_color``, or ``track_color``.

    Parameters
    ----------
    ax : Axes
        Target axes for the badge.
    step : int, optional
        Current simulation step number.
    time : float, optional
        Current simulation time.
    time_units : str
        Unit label appended to the time value (e.g. ``"ns"``).
    step_range : tuple[int, int], optional
        ``(start, end)`` step range for progress bar display.
    label : str or None
        Custom label prefix. ``None`` uses auto-labels (``"step"`` for
        step display, ``"t"`` for time). ``""`` suppresses the prefix.
    show_max : bool
        When ``step_range`` is set, show ``"X / Y"`` if True,
        just ``"X"`` if False.
    shade : "darker", "lighter", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc
        Badge placement location.
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
        Box background color override. ``None`` derives from *shade*.
    bg_alpha : float
        Box background opacity.
    text_color : str, tuple, or None
        Text color override. ``None`` uses rcParams text color.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).
    track_color : str, tuple, or None
        Progress bar track color override. ``None`` derives from the
        foreground color at low alpha.

    Returns
    -------
    AnchoredOffsetbox
        The badge artist added to the axes.
    """
    ensure_matplotlib()

    from matplotlib.offsetbox import TextArea, VPacker

    default_bg, default_fg = _detect_overlay_defaults(shade)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)
    track_rgba = _resolve_rgba(track_color, bg_alpha * 0.4, default_fg)

    status_text = _format_status_text(
        step=step,
        time=time,
        time_units=time_units,
        step_range=step_range,
        label=label,
        show_max=show_max,
    )

    if not status_text:
        msg = "At least one of step or time must be provided"
        raise ValueError(msg)

    text_props = {"fontsize": fontsize, "color": resolved_text}
    text_area = TextArea(status_text, textprops=text_props)

    if step_range is not None and step is not None:
        start, end = step_range
        fraction = (step - start) / max(end - start, 1)
        fraction = max(0.0, min(1.0, fraction))

        progress = _build_progress_bar(
            fraction, bar_width, bar_height, bar_color, bar_alpha, track_rgba
        )
        child = VPacker(children=[text_area, progress], pad=0, sep=3, align="left")
    else:
        child = text_area

    return _make_overlay_box(ax, child, loc, bg_rgba)


@dataclass(frozen=True, slots=True)
class VectorLegendEntry:
    """One row in a vector legend: a sample line and its label."""

    label: str
    color: str
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float = 1.0


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    shade: OverlayShade | None = None,
    loc: BadgeLoc = "upper left",
    fontsize: float = 14,
    fontweight: str = "bold",
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float = 0.65,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.85,
) -> AnchoredOffsetbox:
    r"""Add a bold panel letter (a, b, c, ...) overlay to an axes.

    Styled with the same rounded box as :func:`add_status_badge`.

    Parameters
    ----------
    ax : Axes
        Target axes.
    label : str
        Panel label text (e.g. ``"a"``, ``"b"``).
    shade : "darker", "lighter", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc
        Placement location.
    fontsize : float
        Label font size.
    fontweight : str
        Font weight (default ``"bold"``).
    bg_color, bg_alpha, text_color
        Color overrides; same semantics as :func:`add_status_badge`.
    text_alpha : float
        Text opacity (default 0.85 for subtle softening).

    Returns
    -------
    AnchoredOffsetbox
    """
    ensure_matplotlib()
    from matplotlib.offsetbox import TextArea

    default_bg, default_fg = _detect_overlay_defaults(shade)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    props = {"fontsize": fontsize, "fontweight": fontweight, "color": resolved_text}
    text_area = TextArea(label, textprops=props)

    return _make_overlay_box(ax, text_area, loc, bg_rgba)


def add_vector_legend(
    ax: Axes,
    entries: VectorLegendEntry | list[VectorLegendEntry],
    *,
    shade: OverlayShade | None = None,
    loc: BadgeLoc = "upper left",
    fontsize: float = 9,
    sample_width: float = 20,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float = 0.65,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.85,
) -> AnchoredOffsetbox:
    r"""Add a vector legend overlay showing colored line samples with labels.

    Each entry renders as a short line segment (using the entry's color,
    linewidth, linestyle, alpha) next to its label text.  Multiple entries
    are stacked vertically.  The box style matches :func:`add_status_badge`.

    Parameters
    ----------
    ax : Axes
        Target axes.
    entries : VectorLegendEntry or list[VectorLegendEntry]
        One or more legend entries.
    shade : "darker", "lighter", or None
        ``"darker"``: darken axes facecolor for overlay bg.
        ``"lighter"``: lighten it. ``None`` (default): auto-detect.
    loc : BadgeLoc
        Placement location.
    fontsize : float
        Label font size.
    sample_width : float
        Width of the sample line in points.
    bg_color, bg_alpha, text_color
        Color overrides; same semantics as :func:`add_status_badge`.
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

    if isinstance(entries, VectorLegendEntry):
        entries = [entries]

    default_bg, default_fg = _detect_overlay_defaults(shade)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    rows: list[HPacker] = []
    line_height = fontsize * 0.4
    for entry in entries:
        drawing = DrawingArea(sample_width, line_height)
        mid_y = line_height / 2
        arrow_size = max(4.0, entry.linewidth * 2.5)
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
            textprops={"fontsize": fontsize, "color": resolved_text},
        )
        row = HPacker(children=[drawing, text], pad=0, sep=4, align="center")
        rows.append(row)

    if len(rows) > 1:
        child = VPacker(children=rows, pad=0, sep=3, align="left")
    else:
        child = rows[0]

    return _make_overlay_box(ax, child, loc, bg_rgba)
