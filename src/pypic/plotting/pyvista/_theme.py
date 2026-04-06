"""Apply pypic PlotTheme to a pyvista Plotter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.plotting.pyvista._guard import ensure_pyvista

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme


def _resolve_theme(theme: PlotTheme | None) -> PlotTheme:
    """Return *theme* or the currently active pypic theme."""
    if theme is not None:
        return theme
    from pypic.plotting.styles import get_active_theme

    return get_active_theme()


def apply_theme(
    plotter: Any,
    theme: PlotTheme | None = None,
) -> None:
    r"""Apply pypic :class:`PlotTheme` colors to a pyvista Plotter.

    Sets background color based on theme text luminance and hides
    the default corner orientation axes widget.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter to style.
    theme : PlotTheme or None
        Theme to apply. ``None`` uses the active pypic theme.
    """
    ensure_pyvista()
    t = _resolve_theme(theme)

    # Use theme's figure background if available, otherwise infer from text luminance
    fig_bg = t.rcparams.get("figure.facecolor") or t.rcparams.get("axes.facecolor")
    if fig_bg and fig_bg != "none":
        plotter.set_background(fig_bg)
    else:
        text_lum = 0.299 * t.text_color[0] + 0.587 * t.text_color[1] + 0.114 * t.text_color[2]
        plotter.set_background("#1e1e1e" if text_lum > 0.5 else "#fafafa")

    # Hide default corner orientation widget — use add_axis_triad() instead
    plotter.hide_axes()


def create_plotter(
    *,
    theme: PlotTheme | None = None,
    off_screen: bool = False,
    window_size: tuple[int, int] = (1600, 1000),
    terrain_style: bool = True,
    **kwargs: Any,
) -> Any:
    r"""Create a themed pyvista Plotter.

    Applies the pypic theme, hides the default corner axes widget,
    and optionally enables terrain-style interaction (orbit around
    z-axis without yaw).

    Parameters
    ----------
    theme : PlotTheme or None
        Theme to apply. ``None`` uses the active pypic theme.
    off_screen : bool
        Whether to render off-screen (for saving screenshots).
    window_size : tuple[int, int]
        Window size in pixels.
    terrain_style : bool
        Lock rotation to orbit around the z-axis (no yaw/tumble).
    **kwargs
        Passed to ``pv.Plotter()``.

    Returns
    -------
    pv.Plotter
    """
    ensure_pyvista()
    import pyvista as pv

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size, **kwargs)
    apply_theme(plotter, theme)
    if terrain_style:
        plotter.enable_terrain_style(mouse_wheel_zooms=True)
    return plotter


def set_camera(
    plotter: Any,
    *,
    focal: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float = 50.0,
    elevation: float = 22.0,
    azimuth: float = 40.0,
) -> None:
    r"""Position camera using elevation/azimuth angles.

    Uses the matplotlib ``view_init`` convention: *elevation* is degrees
    above the equatorial plane, *azimuth* is degrees around the z-axis.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    focal : tuple[float, float, float]
        Camera focal point (look-at target).
    distance : float
        Distance from focal point to camera.
    elevation : float
        Elevation angle in degrees (0 = equatorial, 90 = north pole).
    azimuth : float
        Azimuth angle in degrees (counterclockwise from +x).
    """
    import math

    elev = math.radians(elevation)
    azim = math.radians(azimuth)
    cx = focal[0] + distance * math.cos(elev) * math.cos(azim)
    cy = focal[1] + distance * math.cos(elev) * math.sin(azim)
    cz = focal[2] + distance * math.sin(elev)
    plotter.camera_position = [(cx, cy, cz), focal, (0, 0, 1)]


def resolve_cmap(
    cmap: str | Colormap | None = None,
    *,
    signed: bool = True,
    theme: PlotTheme | None = None,
) -> Colormap:
    r"""Resolve a colormap name to a matplotlib Colormap object.

    Handles pypic custom colormaps (e.g. ``"bkr"``) and theme defaults.
    pyvista accepts matplotlib ``Colormap`` objects directly.

    Parameters
    ----------
    cmap : str, Colormap, or None
        Colormap name or object. ``None`` selects from theme.
    signed : bool
        If *cmap* is ``None``, use the diverging (``True``) or
        sequential (``False``) default from the theme.
    theme : PlotTheme or None
        Theme for default selection. ``None`` uses active theme.

    Returns
    -------
    Colormap
    """
    import matplotlib.pyplot as plt

    # Ensure pypic custom colormaps are registered
    import pypic.plotting._colormaps  # noqa: F401

    if cmap is None:
        t = _resolve_theme(theme)
        name = t.diverging_cmap if signed else t.sequential_cmap
        return plt.colormaps[name]

    if isinstance(cmap, str):
        return plt.colormaps[cmap]

    return cmap
