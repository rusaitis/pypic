"""Data-space annotations: planet markers, boundaries, landmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._guard import ensure_matplotlib
from pypic.plotting.styles import _theme_val

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Circle, Wedge

    from pypic.plotting._badge import OverlayVariant

SunDirection = Literal["left", "right", "up", "down"]

# Sun direction → angle (degrees, CCW from +x) where the day-side center points
_SUN_ANGLES: dict[str, float] = {
    "left": 180.0,
    "right": 0.0,
    "up": 90.0,
    "down": 270.0,
}


def add_planet(
    ax: Axes,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 1.0,
    *,
    sun_direction: SunDirection = "left",
    day_color: str = "#f0f0f0",
    night_color: str = "#1a1a2e",
    edgecolor: str = "#888888",
    linewidth: float = 0.5,
    zorder: int = 10,
) -> tuple[Wedge, Wedge]:
    """Draw a day/night planet marker at a data-space position.

    Parameters
    ----------
    ax : Axes
        Target axes.
    center : tuple[float, float]
        Planet center in data coordinates.
    radius : float
        Planet radius in data units.
    sun_direction : {"left", "right", "up", "down"}
        Which side of the plot the Sun is on.
    day_color, night_color : str
        Face colors for the sunlit and dark hemispheres.
    edgecolor : str
        Border color for both semicircles.
    linewidth : float
        Border width.
    zorder : int
        Drawing order (default 10, above most plot elements).

    Returns
    -------
    tuple[Wedge, Wedge]
        The (day, night) wedge patches.
    """
    ensure_matplotlib()
    from matplotlib.patches import Wedge

    sun_angle = _SUN_ANGLES[sun_direction]

    day = Wedge(
        center, radius, sun_angle - 90, sun_angle + 90,
        facecolor=day_color, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder,
    )
    night = Wedge(
        center, radius, sun_angle + 90, sun_angle + 270,
        facecolor=night_color, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(day)
    ax.add_patch(night)
    return day, night


def add_circle(
    ax: Axes,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    *,
    label: str | None = None,
    label_position: float = 45.0,
    color: str | tuple[float, float, float] | None = None,
    alpha: float = 0.6,
    linestyle: str = "--",
    linewidth: float = 0.8,
    fontsize: float | None = None,
    text_alpha: float = 0.9,
    variant: OverlayVariant | None = None,
    zorder: int = 5,
) -> Circle:
    """Draw a reference circle with an optional radius label.

    Parameters
    ----------
    ax : Axes
        Target axes.
    radius : float
        Circle radius in data units.
    center : tuple[float, float]
        Circle center in data coordinates.
    label : str, optional
        Text placed on the circle perimeter (e.g. ``"5 R_E"``).
        ``None`` suppresses the label.
    label_position : float
        Angle in degrees (CCW from +x) where the label sits.
    color : str, optional
        Line color. Defaults to the theme's grid color at higher opacity.
    alpha : float
        Line opacity.
    linestyle : str
        Line style (``"--"``, ``":"``, ``"-."``, etc.).
    linewidth : float
        Line width.
    fontsize : float | None
        Label font size. ``None`` reads from the active theme's
        ``annotation_fontsize`` field.
    text_alpha : float
        Label opacity.
    variant : {"alt"} or None
        Overlay variant for label styling. ``"alt"`` uses the alternate
        overlay colors from the active theme.
    zorder : int
        Drawing order.

    Returns
    -------
    Circle
        The circle patch.
    """
    ensure_matplotlib()
    import math

    from matplotlib.patches import Circle

    # Resolve circle color from theme grid color (with more visibility)
    if color is None:
        grid_rgba = _theme_val("grid_color", (0.0, 0.0, 0.0, 0.08))
        color = grid_rgba[:3]  # RGB only, alpha controlled separately

    circle = Circle(
        center, radius,
        facecolor="none", edgecolor=color, alpha=alpha,
        linestyle=linestyle, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(circle)

    if label is not None:
        # Resolve overlay colors based on variant
        suffix = "_alt" if variant == "alt" else ""
        bg_key = f"overlay{suffix}_color"
        fg_key = f"overlay{suffix}_text_color"
        bg_color = _theme_val(bg_key, (0.07, 0.07, 0.07, 0.65))
        fg_color = _theme_val(fg_key, (0.88, 0.88, 0.88, 0.8))

        default_fs = _theme_val("annotation_fontsize", 7.5)
        fs = fontsize if fontsize is not None else default_fs

        angle_rad = math.radians(label_position)
        tx = center[0] + radius * math.cos(angle_rad)
        ty = center[1] + radius * math.sin(angle_rad)
        ax.text(
            tx, ty, label,
            fontsize=fs, alpha=text_alpha, color=fg_color,
            ha="center", va="center",
            rotation=label_position - 90,
            rotation_mode="anchor",
            zorder=zorder,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=bg_color,
                edgecolor="none",
            ),
        )

    return circle
