"""Apply pypic PlotTheme to a pyvista Plotter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pypic.plotting.pyvista._guard import ensure_pyvista

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme

logger = logging.getLogger(__name__)


def _resolve_theme(theme: PlotTheme | None) -> PlotTheme:
    """Return *theme*, or the active theme, or the global default."""
    if theme is not None:
        return theme
    from pypic.plotting.styles import get_active_theme, get_theme

    return get_active_theme() or get_theme()


def apply_theme(
    plotter: pv.Plotter,
    theme: PlotTheme | None = None,
) -> None:
    r"""Apply pypic `PlotTheme` colors to a pyvista Plotter.

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
        text_lum = (
            0.299 * t.text_color[0] + 0.587 * t.text_color[1] + 0.114 * t.text_color[2]
        )
        plotter.set_background("#1e1e1e" if text_lum > 0.5 else "#fafafa")

    # Hide default corner orientation widget — use add_axis_triad() instead
    plotter.hide_axes()


def _add_themed_orientation_cube(
    plotter: pv.Plotter,
    theme: PlotTheme,
) -> None:
    """Add the camera orientation cube widget with themed colors.

    Colors the six handle rings and ±X/±Y/±Z text labels from
    ``theme.axis_x_color`` / ``axis_y_color`` / ``axis_z_color`` and
    anchors the widget to the lower-left corner.
    """
    from matplotlib.colors import to_rgb

    x_rgb = to_rgb(theme.axis_x_color)
    y_rgb = to_rgb(theme.axis_y_color)
    z_rgb = to_rgb(theme.axis_z_color)

    widget = plotter.add_camera_orientation_widget()
    rep = widget.GetRepresentation()
    rep.AnchorToLowerLeft()
    rep.SetXAxisColor(*x_rgb)
    rep.SetYAxisColor(*y_rgb)
    rep.SetZAxisColor(*z_rgb)
    # All six ±axis label texts — colored to match their axis.
    for getter_name, rgb in (
        ("GetXPlusLabelProperty", x_rgb),
        ("GetXMinusLabelProperty", x_rgb),
        ("GetYPlusLabelProperty", y_rgb),
        ("GetYMinusLabelProperty", y_rgb),
        ("GetZPlusLabelProperty", z_rgb),
        ("GetZMinusLabelProperty", z_rgb),
    ):
        getattr(rep, getter_name)().SetColor(*rgb)


def create_plotter(
    *,
    theme: PlotTheme | None = None,
    off_screen: bool = False,
    window_size: tuple[int, int] = (1600, 1000),
    terrain_style: bool = True,
    orbit_step_deg: float = 5.0,
    orientation_style: Literal["cube", "arrows"] = "cube",
    **kwargs: Any,
) -> Any:
    r"""Create a themed pyvista Plotter.

    Applies the pypic theme, hides the default corner axes widget,
    and optionally enables terrain-style interaction (orbit around
    z-axis without yaw).

    Interactive sessions also get a few navigation conveniences
    that are no-ops when *off_screen* is true:

    - **Right-click** flies the camera to the picked point and makes
      it the new focal point — i.e. the rotation pivot. The single
      most useful way to re-center rotation on something interesting.
    - A **corner orientation widget** is shown in the bottom-left,
      colored with the theme's ``axis_x_color`` / ``axis_y_color`` /
      ``axis_z_color`` so it matches `add_axis_triad` at the
      data origin. Two styles are available via *orientation_style*:
      ``"cube"`` (default) is the 3D rotation cube with ±X/±Y/±Z
      clickable face labels — handle rings and label texts are
      theme-colored; ``"arrows"`` is the simpler tri-arrow axes
      widget for minimal overhead.
    - Keys ``1`` / ``2`` / ``3`` snap to the canonical ``xy`` / ``xz``
      / ``yz`` views (overriding VTK's default ``3`` = stereo toggle).
    - **Arrow keys** orbit the camera around the focal point in fixed
      increments: ``Left`` / ``Right`` = azimuth (yaw), ``Up`` /
      ``Down`` = elevation (pitch). Step size is *orbit_step_deg*.
      Scripts that call ``plotter.add_key_event`` for these keys
      after construction override the defaults cleanly.
    - Key ``u`` levels the camera — realigns the view-up to world
      ``+Z`` without moving the focal point. Useful after heavy
      tumbling leaves a tilted horizon.

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
        Note: terrain style disables VTK's built-in ``f`` key
        (fly-to-picked-point). Use the right-click pivot instead, or
        pass ``terrain_style=False`` to get the trackball default with
        free tumble and ``f``.
    orbit_step_deg : float
        Degrees of camera orbit per arrow-key press. Default 5.
    orientation_style : str
        ``"cube"`` (default) uses the 3D camera orientation cube with
        themed handle rings and ±axis labels. ``"arrows"`` uses the
        simpler tri-arrow axes indicator. Both sit in the bottom-left
        corner and both use ``theme.axis_{x,y,z}_color``.
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
    if not off_screen:
        t = _resolve_theme(theme)
        # Right-click any point → camera flies there, making it the
        # new rotation pivot. Works under both terrain and trackball.
        plotter.enable_fly_to_right_click()
        # Corner orientation widget — rotates with the camera so the
        # current view direction is unambiguous. Both styles use the
        # theme's axis colors and sit in the bottom-left corner so the
        # corner widget and the 3D triad at the data origin stay
        # visually consistent.
        if orientation_style == "cube":
            _add_themed_orientation_cube(plotter, t)
        else:
            plotter.add_axes(
                x_color=t.axis_x_color,
                y_color=t.axis_y_color,
                z_color=t.axis_z_color,
                viewport=(0.0, 0.0, 0.2, 0.2),
            )
        # Preset views (override VTK defaults; '3' = stereo is rarely useful)
        plotter.add_key_event("1", plotter.view_xy)
        plotter.add_key_event("2", plotter.view_xz)
        plotter.add_key_event("3", plotter.view_yz)

        # Stepwise keyboard orbit around the focal point.
        # azimuth/elevation orbit the camera around the focal point,
        # which is what you want for scientific data viewing.
        # (camera.yaw/pitch would rotate the view direction around
        # the camera position — first-person look-around.)
        def _orbit(az: float = 0.0, el: float = 0.0) -> None:
            if az:
                plotter.camera.azimuth(az)
            if el:
                plotter.camera.elevation(el)
            plotter.render()

        step = orbit_step_deg
        plotter.add_key_event("Left", lambda: _orbit(az=-step))
        plotter.add_key_event("Right", lambda: _orbit(az=+step))
        plotter.add_key_event("Up", lambda: _orbit(el=+step))
        plotter.add_key_event("Down", lambda: _orbit(el=-step))

        # Level the camera — realigns view-up to world +Z without
        # moving the focal point. Non-destructive counterpart to
        # VTK's 'r' (reset camera), useful after compounded orbit
        # steps introduce roll.
        def _level_up() -> None:
            plotter.camera.up = (0.0, 0.0, 1.0)
            plotter.render()

        plotter.add_key_event("u", _level_up)
    return plotter


def show_or_save(
    plotter: pv.Plotter,
    *,
    save: bool = False,
    outfile: str | None = None,
    transparent_background: bool = True,
) -> None:
    r"""Show the interactive window or save a screenshot.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    save : bool
        If ``True``, render off-screen and save to *outfile*.
    outfile : str or None
        Output file path. Required when *save* is ``True``.
    transparent_background : bool
        Save with transparent background (PNG alpha channel).
    """
    if save:
        if outfile is None:
            msg = "outfile is required when save=True"
            raise ValueError(msg)
        plotter.show(auto_close=False)
        plotter.screenshot(str(outfile), transparent_background=transparent_background)
        plotter.close()
        logger.info("Saved to %s", outfile)
    else:
        plotter.show()


def set_camera(
    plotter: pv.Plotter,
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
