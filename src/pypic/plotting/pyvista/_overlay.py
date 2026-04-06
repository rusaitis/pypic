"""Shared 2D overlay primitives for pyvista (rounded rects, borders)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import vtk as _vtk

    from pypic.plotting.styles import PlotTheme


def _rounded_rect_points(
    x: float, y: float, w: float, h: float, rounding: float, n_arc: int = 6,
) -> list[tuple[float, float]]:
    """Compute corner-arc points for a rounded rectangle."""
    r = min(rounding, w * 0.3, h * 0.3)
    corners = [
        (x + r, y + r, -np.pi, -np.pi / 2),
        (x + w - r, y + r, -np.pi / 2, 0),
        (x + w - r, y + h - r, 0, np.pi / 2),
        (x + r, y + h - r, np.pi / 2, np.pi),
    ]
    pts: list[tuple[float, float]] = []
    for cx, cy, a0, a1 in corners:
        for ang in np.linspace(a0, a1, n_arc):
            pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    return pts


def _vtk_viewport_coord() -> _vtk.vtkCoordinate:
    """Create a normalized-viewport coordinate transform."""
    import vtk  # type: ignore[import-untyped]

    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    return coord


def draw_rounded_rect(
    plotter: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    rounding: float,
    color: tuple[float, float, float],
    opacity: float,
) -> Any:
    """Draw a filled 2D rounded rectangle in normalized viewport coordinates.

    Returns the ``vtkActor2D`` for later modification or removal.
    """
    import vtk  # type: ignore[import-untyped]

    pts = _rounded_rect_points(x, y, w, h, rounding)

    points = vtk.vtkPoints()
    for px, py in pts:
        points.InsertNextPoint(px, py, 0)

    polygon = vtk.vtkCellArray()
    polygon.InsertNextCell(len(pts))
    for i in range(len(pts)):
        polygon.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polygon)

    coord = _vtk_viewport_coord()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    plotter.renderer.AddActor2D(actor)
    return actor


def draw_rounded_rect_border(
    plotter: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    rounding: float,
    color: tuple[float, float, float],
    opacity: float,
    line_width: float = 1.0,
) -> Any:
    """Draw a border outline of a rounded rectangle.

    Returns the ``vtkActor2D`` for later modification or removal.
    """
    import vtk  # type: ignore[import-untyped]

    pts = _rounded_rect_points(x, y, w, h, rounding)

    points = vtk.vtkPoints()
    for px, py in pts:
        points.InsertNextPoint(px, py, 0)

    lines = vtk.vtkCellArray()
    n = len(pts)
    lines.InsertNextCell(n + 1)
    for i in range(n):
        lines.InsertCellPoint(i)
    lines.InsertCellPoint(0)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    coord = _vtk_viewport_coord()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(line_width)
    plotter.renderer.AddActor2D(actor)
    return actor


def build_scalar_bar(
    plotter: Any,
    title: str,
    *,
    position: str = "lower_right",
    fmt: str | None = None,
    clim: tuple[float, float] | None = None,
    theme: PlotTheme | None = None,
) -> tuple[dict[str, Any], str]:
    """Build a themed scalar bar with rounded background and return the args dict.

    Draws the background rectangle and border on *plotter*, then returns
    the ``scalar_bar_args`` dict to pass to ``plotter.add_mesh()``.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter (background rects are drawn immediately).
    title : str
        Scalar bar title text.
    position : str
        Corner position: ``"lower_right"``, ``"lower_left"``,
        ``"upper_right"``, ``"upper_left"``.
    fmt : str
        Number format for tick labels.
    theme : PlotTheme or None
        Theme for colors and fonts.

    Returns
    -------
    dict[str, Any]
        The ``scalar_bar_args`` dict for ``plotter.add_mesh()``.
    """
    from pypic.plotting.pyvista._theme import _resolve_theme

    t = _resolve_theme(theme)

    title_fs = int(t.font_label * 2.5)
    label_fs = int(t.font_tick * 2.8)
    title_h = title_fs / 1000.0 * 2.0
    tick_h = label_fs / 1000.0 * 1.5

    bar_w = 0.30 + label_fs * 0.003
    bar_h = 0.035
    em = t.overlay_margin + 0.02
    bar_x = (1.0 - em - bar_w) if "right" in position else em
    bar_y = (1.0 - em - bar_h) if "upper" in position else em

    pad_x = t.overlay_padding * 0.06
    pad_y = t.overlay_padding * 0.025
    bg_x = bar_x - pad_x
    bg_y = bar_y - pad_y - tick_h
    bg_w = bar_w + 2 * pad_x
    bg_h = tick_h + bar_h + title_h + 2 * pad_y
    rounding = min(t.overlay_rounding * 0.02, bg_w * 0.2, bg_h * 0.2)

    oc = t.overlay_color
    draw_rounded_rect(plotter, bg_x, bg_y, bg_w, bg_h, rounding, oc[:3], oc[3])
    bc = t.overlay_border_color
    draw_rounded_rect_border(plotter, bg_x, bg_y, bg_w, bg_h, rounding, bc[:3], bc[3])

    # Auto-select format from clim range if not specified
    if fmt is None:
        if clim is not None and abs(clim[1] - clim[0]) >= 5:
            fmt = "%.0f"
        elif clim is not None and abs(clim[1] - clim[0]) >= 0.5:
            fmt = "%.1f"
        else:
            fmt = "%.2f"

    return dict(
        title=title,
        n_labels=5,
        position_x=bar_x,
        position_y=bar_y,
        width=bar_w,
        height=bar_h,
        title_font_size=title_fs,
        label_font_size=label_fs,
        shadow=False,
        fmt=fmt,
        color=t.overlay_text_color[:3],
    ), title


def style_scalar_bar(plotter: Any, title: str, *, text_pad: int = 4) -> None:
    """Apply title padding to a scalar bar after creation.

    Call after ``plotter.add_mesh()`` to adjust the title-to-bar spacing.
    """
    try:
        sbar = plotter.scalar_bars[title]
        sbar.SetTextPad(text_pad)
    except (KeyError, AttributeError):
        pass
