"""In-plot status badge overlay showing simulation step/time info."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea

    from pypic.plotting.styles import PlotTheme

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
) -> str:
    """Build the status text string from step/time parameters."""
    parts: list[str] = []

    if step is not None:
        if step_range is not None:
            parts.append(f"step {step} / {step_range[1]}")
        else:
            parts.append(f"step {step}")

    if time is not None:
        if abs(time) >= 1e4 or (0 < abs(time) < 0.01):
            t_str = f"{time:.2e}"
        else:
            t_str = f"{time:.2f}"
        suffix = f" {time_units}" if time_units else ""
        parts.append(f"t = {t_str}{suffix}")

    return ", ".join(parts) if parts else ""


def _build_progress_bar(
    fraction: float,
    bar_width: float,
    bar_height: float,
    bar_color: str,
    track_color: tuple[float, ...] | str,
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
    loc: BadgeLoc = "upper right",
    fontsize: float = 9,
    bar_color: str = "#e8913a",
    bar_width: float = 80,
    bar_height: float = 4,
    theme: PlotTheme | None = None,
) -> AnchoredOffsetbox:
    r"""Add a status badge overlay to an axes.

    Renders simulation step and/or time as a rounded, semi-transparent
    box matching the legend aesthetic. Optionally includes a progress bar
    when ``step_range`` is provided.

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
    loc : BadgeLoc
        Badge placement location.
    fontsize : float
        Font size for the status text.
    bar_color : str
        Fill color for the progress bar.
    bar_width : float
        Width of the progress bar in points.
    bar_height : float
        Height of the progress bar in points.
    theme : PlotTheme, optional
        Theme for styling. Falls back to LIGHT defaults.

    Returns
    -------
    AnchoredOffsetbox
        The badge artist added to the axes.
    """
    ensure_matplotlib()

    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

    from pypic.plotting.styles import LIGHT

    theme = theme or LIGHT

    text_color = theme.rcparams.get("text.color", "0.1")
    bg_color = theme.rcparams.get("legend.facecolor", (0, 0, 0, 0.06))

    status_text = _format_status_text(
        step=step, time=time, time_units=time_units, step_range=step_range
    )

    if not status_text:
        msg = "At least one of step or time must be provided"
        raise ValueError(msg)

    text_props = {"fontsize": fontsize, "color": text_color}
    text_area = TextArea(status_text, textprops=text_props)

    if step_range is not None and step is not None:
        start, end = step_range
        fraction = (step - start) / max(end - start, 1)
        fraction = max(0.0, min(1.0, fraction))

        progress = _build_progress_bar(
            fraction, bar_width, bar_height, bar_color, bg_color
        )
        child = VPacker(children=[text_area, progress], pad=0, sep=3, align="left")
    else:
        child = text_area

    loc_code = _LOC_CODES.get(loc, 1)
    box = AnchoredOffsetbox(
        loc=loc_code,
        child=child,
        pad=0.4,
        borderpad=0.6,
        frameon=True,
    )

    box.patch.set_boxstyle("round,pad=0.4,rounding_size=0.6")
    box.patch.set_facecolor(bg_color)
    box.patch.set_edgecolor("none")

    ax.add_artist(box)
    return box
