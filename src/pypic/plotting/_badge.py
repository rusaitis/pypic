"""In-plot overlay elements: status badge, vector legend, panel label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea

OVERLAY_BOX_STYLE = "round,pad=0.4,rounding_size=0.6"
OVERLAY_BORDER_PAD: float = 0.6

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
    dark_mode: bool = True,
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
    bar when ``step_range`` is provided. Two built-in modes controlled by
    ``dark_mode``: dark (black background, white text — the default) and
    light (white background, black text). Override individual colors with
    ``bg_color``, ``text_color``, or ``track_color`` when needed.

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
    dark_mode : bool
        ``True`` (default): black background, white text.
        ``False``: white background, black text.
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
        Box background color override. ``None`` uses the ``dark_mode``
        default (alpha controlled by *bg_alpha*).
    bg_alpha : float
        Box background opacity.
    text_color : str, tuple, or None
        Text color override. ``None`` uses the ``dark_mode`` default.
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

    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

    if dark_mode:
        default_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
        default_fg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    else:
        default_bg = (1.0, 1.0, 1.0)
        default_fg = (0.0, 0.0, 0.0)

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
    dark_mode: bool = False,
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
    dark_mode : bool
        ``False`` (default): white background.  ``True``: black background.
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
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

    if dark_mode:
        default_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
        default_fg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    else:
        default_bg = (1.0, 1.0, 1.0)
        default_fg = (0.0, 0.0, 0.0)

    bg_rgba = _resolve_rgba(bg_color, bg_alpha, default_bg)
    resolved_text = _resolve_rgba(text_color, text_alpha, default_fg)

    props = {"fontsize": fontsize, "fontweight": fontweight, "color": resolved_text}
    text_area = TextArea(label, textprops=props)

    loc_code = _LOC_CODES.get(loc, 2)
    box = AnchoredOffsetbox(
        loc=loc_code,
        child=text_area,
        pad=0.4,
        borderpad=OVERLAY_BORDER_PAD,
        frameon=True,
    )
    box.patch.set_boxstyle(OVERLAY_BOX_STYLE)
    box.patch.set_facecolor(bg_rgba)
    box.patch.set_edgecolor("none")

    ax.add_artist(box)
    return box


def add_vector_legend(
    ax: Axes,
    entries: VectorLegendEntry | list[VectorLegendEntry],
    *,
    dark_mode: bool = False,
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
    dark_mode : bool
        ``False`` (default): white background.  ``True``: black background.
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
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        DrawingArea,
        HPacker,
        TextArea,
        VPacker,
    )

    if isinstance(entries, VectorLegendEntry):
        entries = [entries]

    if dark_mode:
        default_bg: tuple[float, float, float] = (0.0, 0.0, 0.0)
        default_fg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    else:
        default_bg = (1.0, 1.0, 1.0)
        default_fg = (0.0, 0.0, 0.0)

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

    loc_code = _LOC_CODES.get(loc, 2)
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
