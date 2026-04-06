"""Progress badge overlay for pyvista 3D plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
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


def _progress_bar(fraction: float, width: int = 20) -> str:
    """Render a text-based progress bar: [████░░░░░░] 45%."""
    filled = int(width * max(0.0, min(1.0, fraction)))
    empty = width - filled
    pct = int(fraction * 100)
    return f"[{'█' * filled}{'░' * empty}] {pct}%"


def add_badge(
    plotter: Any,
    *,
    step: int | None = None,
    time: float | None = None,
    time_units: str = "",
    progress: float | None = None,
    position: str = "upper_left",
    font_size: int | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Add a progress badge as screen-space text overlay.

    Shows simulation time, step number, and a text-based progress bar
    using theme colors. Simpler than the matplotlib badge (no rounded
    box patches) but carries the same information.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    step : int or None
        Current simulation step.
    time : float or None
        Simulation time value.
    time_units : str
        Units for *time* (e.g. ``"s"``).
    progress : float or None
        Progress fraction in ``[0, 1]``. Shows a text progress bar.
    position : str
        Screen position: ``"upper_left"``, ``"upper_right"``,
        ``"lower_left"``, ``"lower_right"``.
    theme : PlotTheme or None
        Theme for text color.
    """
    ensure_pyvista()

    t = _resolve_theme(theme)

    # Build text lines
    parts: list[str] = []
    if time is not None:
        parts.append(_format_time(time, time_units))
    if step is not None:
        parts.append(f"step {step}")
    if progress is not None:
        parts.append(_progress_bar(progress))

    if not parts:
        return

    text = "\n".join(parts)

    # Resolve text color from theme
    tc = t.overlay_text_color
    color = (tc[0], tc[1], tc[2])

    # Font size: use override or scale from theme
    font_size = font_size if font_size is not None else int(t.font_overlay * 1.5)

    plotter.add_text(
        text,
        position=position.replace("_", " "),
        font_size=font_size,
        color=color,
        shadow=True,
    )
