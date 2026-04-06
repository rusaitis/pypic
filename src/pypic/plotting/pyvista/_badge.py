"""Progress badge overlay for pyvista 3D plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._overlay import draw_rounded_rect, draw_rounded_rect_border
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
    import pyvista as pv

    from pypic.plotting.styles import PlotTheme


def _format_time(time: float, units: str) -> str:
    """Format time as human-readable string."""
    if units == "s" and abs(time) >= 60:
        sign = "-" if time < 0 else ""
        t = abs(time)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if h > 0:
            return f"{sign}{h}h {m}min {s}s"
        return f"{sign}{m}min {s}s"
    if abs(time) >= 1e4 or (0 < abs(time) < 0.01):
        return f"{time:.2e} {units}".strip()
    return f"{time:.2f} {units}".strip()


def _position_xy(
    position: str, box_w: float, box_h: float, edge_margin: float,
) -> tuple[float, float]:
    """Map position string to (x, y) in normalized viewport coords."""
    m = edge_margin
    positions = {
        "upper_left": (m, 1.0 - m - box_h),
        "upper_right": (1.0 - m - box_w, 1.0 - m - box_h),
        "lower_left": (m, m),
        "lower_right": (1.0 - m - box_w, m),
    }
    return positions.get(position, (m, 1.0 - m - box_h))


def add_badge(
    plotter: pv.Plotter,
    *,
    step: int | None = None,
    time: float | None = None,
    time_units: str = "",
    progress: float | None = None,
    position: str = "upper_right",
    font_size: int | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Add a progress badge with rounded background and visual progress bar.

    Renders simulation time or step number, and a graphical progress bar
    inside a themed rounded rectangle overlay.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    step : int or None
        Current simulation step. Shown only when *time* is ``None``.
    time : float or None
        Simulation time value. Takes precedence over *step*.
    time_units : str
        Units for *time* (e.g. ``"s"``).
    progress : float or None
        Progress fraction in ``[0, 1]``. Shows a visual progress bar.
    position : str
        Screen position: ``"upper_left"``, ``"upper_right"``,
        ``"lower_left"``, ``"lower_right"``.
    font_size : int or None
        Override label font size. ``None`` derives from theme.
    theme : PlotTheme or None
        Theme for colors.
    """
    ensure_pyvista()

    t = _resolve_theme(theme)

    # Build status text — time takes precedence over step
    if time is not None:
        status_text = _format_time(time, time_units)
    elif step is not None:
        status_text = f"step {step}"
    else:
        status_text = ""

    if not status_text and progress is None:
        return

    # Font size → viewport fraction: pyvista font_size is roughly pixels at
    # 1000px window height, so font_size/1000 ≈ text height in viewport coords
    fs = font_size if font_size is not None else int(t.font_overlay * 2.2)
    text_h = fs / 1000.0 * 2.0  # approximate line height with padding

    # Layout dimensions scaled from theme overlay settings
    pad = t.overlay_padding * 0.025        # inner padding
    edge_margin = t.overlay_margin          # distance from window edge
    margin = t.overlay_margin * 0.5        # inset for bar/text within box
    rounding = t.overlay_rounding * 0.02
    bar_h = t.progress_bar_height * 0.002
    box_w = t.progress_bar_width * 0.0025  # 80.0 → 0.20
    box_h = pad + (text_h if status_text else 0) + (bar_h + pad if progress is not None else 0) + pad

    origin_x, origin_y = _position_xy(position, box_w, box_h, edge_margin)

    # Background rounded rectangle with subtle border
    oc = t.overlay_color
    draw_rounded_rect(plotter, origin_x, origin_y, box_w, box_h, rounding, oc[:3], oc[3])
    bc = t.overlay_border_color
    draw_rounded_rect_border(plotter, origin_x, origin_y, box_w, box_h, rounding, bc[:3], bc[3])

    # Progress bar (track + fill)
    if progress is not None:
        fraction = max(0.0, min(1.0, progress))
        bar_x = origin_x + margin
        bar_y = origin_y + pad
        bar_w = box_w - 2 * margin
        bar_r = t.progress_bar_rounding * 0.005

        # Track
        tc = t.track_color
        draw_rounded_rect(plotter, bar_x, bar_y, bar_w, bar_h, bar_r, tc[:3], tc[3])

        # Fill
        if fraction > 0.01:
            from matplotlib.colors import to_rgb

            accent_rgb = to_rgb(t.accent_color)
            fill_w = max(bar_h, bar_w * fraction)
            draw_rounded_rect(plotter, bar_x, bar_y, fill_w, bar_h, bar_r, accent_rgb, 0.85)

    # Text label
    if status_text:
        text_color = t.overlay_text_color[:3]
        text_x = origin_x + margin
        text_y = origin_y + box_h - pad - text_h

        plotter.add_text(
            status_text,
            position=(text_x, text_y),
            font_size=fs,
            color=text_color,
            viewport=True,
        )
