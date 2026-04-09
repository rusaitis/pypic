"""Shared 2D overlay primitives for pyvista (rounded rects, borders, colorbar)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import vtk as _vtk
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme


# Theme padding/rounding fractions converted to normalized viewport units.
# These are the empirically-tuned constants that translate the theme's
# point-like values into viewport coords for every overlay (badge, label,
# colorbar). Centralized so a single tweak affects all overlays uniformly.
_PAD_X_FRAC = 0.06  # × theme.overlay_padding → horizontal inner padding
_PAD_Y_FRAC = 0.025  # × theme.overlay_padding → vertical inner padding
_ROUND_FRAC = 0.02  # × theme.overlay_rounding → corner radius

# Font→viewport-height conversion ratios. pyvista font_size is roughly
# pixels at 1000px window height, so font_size/1000 ≈ raw text height in
# viewport coordinates. The ratio tunes that to a comfortable line height.
_TEXT_H_RATIO = 2.0  # full-line text (titles, badge text, label text)
_TICK_H_RATIO = 1.5  # compact tick labels (colorbar)

# Colorbar gradient strip default dimensions (viewport fractions).
_STRIP_W_BASE = 0.3  # base width
_STRIP_W_FONT_SCALE = 0.003  # extra width per tick font-size unit
_STRIP_H_DEFAULT = 0.015  # height


def _rounded_rect_points(
    x: float,
    y: float,
    w: float,
    h: float,
    rounding: float,
    n_arc: int = 6,
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
    import vtk

    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    return coord


def _position_xy(
    position: str,
    box_w: float,
    box_h: float,
    edge_margin: float,
) -> tuple[float, float]:
    """Map a corner/center position string to (x, y) in normalized viewport.

    Shared by every overlay (badge, label, colorbar). *position* is one of
    ``"upper_left"``, ``"upper_right"``, ``"lower_left"``, ``"lower_right"``,
    ``"upper_center"``, ``"lower_center"``.
    """
    m = edge_margin
    if "center" in position:
        x = 0.5 - box_w * 0.5
    elif "right" in position:
        x = 1.0 - m - box_w
    else:
        x = m
    y = 1.0 - m - box_h if "upper" in position else m
    return x, y


def _overlay_layout(theme: PlotTheme) -> tuple[float, float, float]:
    """Return ``(pad_x, pad_y, rounding)`` viewport fractions for the theme.

    Centralizes the empirical multipliers so every overlay (badge, label,
    colorbar) shares identical inner padding and corner radius scaling.
    """
    return (
        theme.overlay_padding * _PAD_X_FRAC,
        theme.overlay_padding * _PAD_Y_FRAC,
        theme.overlay_rounding * _ROUND_FRAC,
    )


def _text_height(font_size: int, ratio: float = _TEXT_H_RATIO) -> float:
    """Approximate single-line text height in normalized viewport units.

    pyvista font_size is roughly pixels at a 1000px window height, so
    ``font_size / 1000`` is the raw text-cell height; *ratio* widens that
    to a comfortable line height (~2.0 for normal text, ~1.5 for compact
    tick labels).
    """
    return font_size / 1000.0 * ratio


def _resolve_overlay_colors(
    theme: PlotTheme,
    variant: str | None,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Return ``(bg, border, text, track)`` RGBA tuples for the variant.

    *variant* is ``None`` for primary or ``"alt"`` for the contrasting alt
    overlay colors. ``track`` is included even when only the badge uses it
    so all overlay functions can call this helper uniformly (other callers
    can simply discard the fourth element).
    """
    if variant == "alt":
        return (
            theme.overlay_alt_color,
            theme.overlay_alt_border_color,
            theme.overlay_alt_text_color,
            theme.track_alt_color,
        )
    return (
        theme.overlay_color,
        theme.overlay_border_color,
        theme.overlay_text_color,
        theme.track_color,
    )


def _draw_filled_polygon(
    plotter: Any,
    points_2d: list[tuple[float, float]],
    color: tuple[float, float, float],
    opacity: float,
) -> Any:
    """Draw a filled 2D polygon through *points_2d* (uniform color).

    Returns the ``vtkActor2D``. Used as the building block for
    :func:`draw_rounded_rect`, :func:`_draw_triangle`, and any other
    convex single-color filled shape in normalized viewport coordinates.
    """
    import vtk

    pts = vtk.vtkPoints()
    for px, py in points_2d:
        pts.InsertNextPoint(px, py, 0)

    polygon = vtk.vtkCellArray()
    n = len(points_2d)
    polygon.InsertNextCell(n)
    for i in range(n):
        polygon.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
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
    return _draw_filled_polygon(
        plotter,
        _rounded_rect_points(x, y, w, h, rounding),
        color,
        opacity,
    )


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
    return _draw_polyline_border(
        plotter,
        _rounded_rect_points(x, y, w, h, rounding),
        color,
        opacity,
        line_width,
    )


def _auto_fmt(clim: tuple[float, float]) -> str:
    """Pick a tick label format string from the clim range."""
    span = abs(clim[1] - clim[0])
    if span >= 5:
        return "%.0f"
    if span >= 0.5:
        return "%.1f"
    return "%.2f"


def _draw_gradient_strip(
    plotter: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    cmap: Colormap,
    *,
    n_segments: int = 128,
) -> Any:
    """Draw a horizontal gradient strip as N colored quads.

    Returns the ``vtkActor2D``.
    """
    import vtk

    dx = w / n_segments
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)

    for i in range(n_segments):
        x0 = x + i * dx
        x1 = x + (i + 1) * dx
        base = i * 4
        points.InsertNextPoint(x0, y, 0)
        points.InsertNextPoint(x1, y, 0)
        points.InsertNextPoint(x1, y + h, 0)
        points.InsertNextPoint(x0, y + h, 0)
        cells.InsertNextCell(4)
        for j in range(4):
            cells.InsertCellPoint(base + j)

        t = (i + 0.5) / n_segments
        rgba = cmap(t)
        rcol = int(rgba[0] * 255)
        gcol = int(rgba[1] * 255)
        bcol = int(rgba[2] * 255)
        colors.InsertNextTuple3(rcol, gcol, bcol)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.GetCellData().SetScalars(colors)

    coord = _vtk_viewport_coord()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    plotter.renderer.AddActor2D(actor)
    return actor


def _draw_triangle(
    plotter: Any,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> Any:
    """Draw a filled 2D triangle in normalized viewport coordinates."""
    return _draw_filled_polygon(plotter, [p0, p1, p2], color, opacity)


def _draw_polyline_border(
    plotter: Any,
    points_2d: list[tuple[float, float]],
    color: tuple[float, float, float],
    opacity: float,
    line_width: float = 1.0,
) -> Any:
    """Draw a closed 2D polyline outline through *points_2d*."""
    import vtk

    pts = vtk.vtkPoints()
    for px, py in points_2d:
        pts.InsertNextPoint(px, py, 0)

    n = len(points_2d)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n + 1)
    for i in range(n):
        lines.InsertCellPoint(i)
    lines.InsertCellPoint(0)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
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


def _add_text_actor(
    plotter: Any,
    text: str,
    x: float,
    y: float,
    font_size: int,
    color: tuple[float, float, float],
    *,
    h_align: str = "left",
    bold: bool = False,
) -> Any:
    """Add a 2D text actor with explicit alignment in normalized viewport coords.

    *h_align* is one of ``"left"``, ``"center"``, ``"right"`` and uses VTK's
    own text-bbox justification rather than character-width estimation.
    *bold* toggles ``vtkTextProperty.SetBold``.
    """
    import vtk

    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    tp = actor.GetTextProperty()
    # pyvista's add_text scales font_size by 2x internally; match that
    tp.SetFontSize(int(font_size * 2))
    tp.SetColor(*color)
    tp.SetShadow(False)
    if bold:
        tp.SetBold(1)
    if h_align == "center":
        tp.SetJustificationToCentered()
    elif h_align == "right":
        tp.SetJustificationToRight()
    else:
        tp.SetJustificationToLeft()
    tp.SetVerticalJustificationToBottom()
    coord = actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToNormalizedViewport()
    coord.SetValue(x, y)
    plotter.renderer.AddActor2D(actor)
    return actor


def _measure_text_width_px(
    plotter: Any,
    text: str,
    font_size: int,
) -> float:
    """Return the rendered width of *text* at *font_size* in pixels.

    Uses VTK's ``vtkTextActor.GetSize`` against the active renderer so the
    DPI and font metrics match what will actually be drawn. Returns ``0.0``
    when the renderer cannot yet measure the text.
    """
    import vtk

    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    tp = actor.GetTextProperty()
    tp.SetFontSize(int(font_size * 2))
    tp.SetShadow(False)
    size: list[float] = [0.0, 0.0]
    actor.GetSize(plotter.renderer, size)
    return float(size[0])


def _fit_fonts_to_strip(
    plotter: Any,
    strip_w: float,
    tick_labels: list[str],
    title: str,
    tick_fs: int,
    title_fs: int,
    *,
    label_budget: float = 0.92,
    title_budget: float = 0.95,
    min_fs: int = 7,
) -> tuple[int, int]:
    """Shrink *tick_fs* and *title_fs* if labels would overflow the strip width.

    Measures actual rendered widths via :func:`_measure_text_width_px` and
    scales each font down only when its content exceeds the allowed budget
    fraction of the strip width. Never scales above the requested sizes.
    """
    ww = plotter.window_size[0] if hasattr(plotter, "window_size") else 1600
    strip_w_px = strip_w * ww

    if tick_labels:
        widths = [_measure_text_width_px(plotter, lab, tick_fs) for lab in tick_labels]
        total = sum(widths)
        if len(widths) > 1 and tick_labels[0]:
            # Reserve roughly one extra character of breathing room per gap
            char_w = widths[0] / max(len(tick_labels[0]), 1)
            total += char_w * (len(widths) - 1)
        if total > 0 and total > strip_w_px * label_budget:
            scale = strip_w_px * label_budget / total
            tick_fs = max(min_fs, int(tick_fs * scale))

    if title:
        title_w_px = _measure_text_width_px(plotter, title, title_fs)
        if title_w_px > 0 and title_w_px > strip_w_px * title_budget:
            scale = strip_w_px * title_budget / title_w_px
            title_fs = max(min_fs, int(title_fs * scale))

    return tick_fs, title_fs


def _draw_tick_marks(
    plotter: Any,
    ticks_x: list[float],
    y_top: float,
    y_bottom: float,
    color: tuple[float, float, float],
    opacity: float,
    line_width: float = 1.0,
) -> Any:
    """Draw short vertical tick lines. Returns ``vtkActor2D``."""
    import vtk

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for i, tx in enumerate(ticks_x):
        points.InsertNextPoint(tx, y_top, 0)
        points.InsertNextPoint(tx, y_bottom, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i * 2)
        lines.InsertCellPoint(i * 2 + 1)

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


def add_colorbar(
    plotter: Any,
    cmap: str | Colormap,
    clim: tuple[float, float],
    label: str = "",
    *,
    loc: str = "lower right",
    extend: str = "both",
    extremes: str | None = "semi",
    variant: str | None = None,
    width: float | None = None,
    height: float | None = None,
    fmt: str | None = None,
    n_labels: int = 5,
    ticks: list[float] | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Draw a fully custom colorbar using VTK 2D primitives.

    Renders a gradient strip, tick marks, tick labels, and a label
    inside a themed rounded-rectangle background. All elements are
    drawn in normalized viewport coordinates for pixel-perfect
    positioning independent of window size.

    Parameters match :func:`pypic.plotting.add_inset_colorbar` (matplotlib)
    where possible: *label* aligns with matplotlib's colorbar ``label=``
    convention, and *loc* matches :func:`add_badge` / :func:`add_label`.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    cmap : str or Colormap
        Colormap (name or object) for the gradient strip.
    clim : tuple[float, float]
        Color limits ``(vmin, vmax)`` mapped to the gradient.
    label : str
        Label text above the gradient strip.
    loc : str
        Corner or center placement: ``"lower right"`` (default),
        ``"lower left"``, ``"upper right"``, ``"upper left"``,
        ``"lower center"``, ``"upper center"``. Legacy underscore
        form (``"lower_right"``) is also accepted.
    extend : str
        Triangular under/over indicators on the strip ends:
        ``"both"`` (default), ``"min"``, ``"max"``, or ``"neither"``.
    extremes : "semi", "transparent", "darken", or None
        How to style the extension triangles. ``"semi"`` (default)
        renders them at ~30% opacity; ``"darken"`` multiplies the
        endpoint RGB by 0.75; ``"transparent"`` skips the triangles
        entirely; ``None`` uses the cmap's under/over colors at full
        opacity. Mirrors :func:`pypic.plotting.add_inset_colorbar`.
    variant : str or None
        ``None`` uses the primary overlay colors; ``"alt"`` uses
        ``theme.overlay_alt_*`` (matching the matplotlib badge variant).
    width : float or None
        Override the gradient strip width in normalized viewport
        coordinates. ``None`` derives a sensible default from
        ``theme.font_tick``.
    height : float or None
        Override the gradient strip height in normalized viewport
        coordinates. ``None`` uses the default ``0.015``.
    fmt : str or None
        Number format for tick labels. ``None`` auto-selects from
        the clim range. Ignored when *ticks* is provided.
    n_labels : int
        Target number of tick labels. Ignored when *ticks* is provided.
    ticks : list[float] or None
        Explicit tick positions in data units. ``None`` (default)
        auto-generates ticks from *clim* and *n_labels*. Mirrors the
        ``ticks`` parameter on :func:`pypic.plotting.add_inset_colorbar`.
    theme : PlotTheme or None
        Theme for colors and fonts.
    """
    from pypic.plotting._format import _normalize_loc

    # Accept both "lower right" (mpl) and legacy "lower_right" (pyvista)
    loc = _normalize_loc(loc)
    from pypic.plotting.pyvista._axes import _nice_step, _uniform_ticks
    from pypic.plotting.pyvista._theme import _resolve_theme, resolve_cmap

    t = _resolve_theme(theme)

    if isinstance(cmap, str):
        cmap = resolve_cmap(cmap, theme=theme)
    if fmt is None:
        fmt = _auto_fmt(clim)

    extend_min = extend in ("both", "min")
    extend_max = extend in ("both", "max")

    # Extremes mode: "transparent" hides the triangles entirely; the
    # other modes are applied to the triangle color/opacity at draw time.
    if extremes == "transparent":
        extend_min = False
        extend_max = False
    if extremes == "semi":
        tri_rgb_factor, tri_opacity = 1.0, 0.3
    elif extremes == "darken":
        tri_rgb_factor, tri_opacity = 0.75, 1.0
    else:  # None or unrecognized → full opacity
        tri_rgb_factor, tri_opacity = 1.0, 1.0

    bg_color, border_color, text_color, _ = _resolve_overlay_colors(t, variant)

    # Initial font sizes (scaled from theme via theme-defined multipliers)
    title_fs = int(t.font_label * t.colorbar_title_font_scale)
    tick_fs = int(t.font_tick * t.colorbar_tick_font_scale)

    # Gradient strip dimensions: explicit overrides or derived defaults.
    # The default strip width grows slightly with font size so longer
    # tick labels still fit at the original tick font size.
    strip_w = (
        width if width is not None else _STRIP_W_BASE + tick_fs * _STRIP_W_FONT_SCALE
    )
    strip_h = height if height is not None else _STRIP_H_DEFAULT

    # Tick values and labels (need strip_w to know spacing budget).
    # User-supplied *ticks* override the auto-generated ones, mirroring
    # add_inset_colorbar's ``ticks`` parameter on the matplotlib side.
    lo, hi = clim
    span = hi - lo
    if ticks is not None:
        tick_values = list(ticks)
    elif span < 1e-12:
        tick_values = [lo]
    else:
        step = _nice_step(span, max(n_labels - 1, 1))
        tick_values = _uniform_ticks(lo, hi, step)
        # Thin labels when too dense
        while len(tick_values) > n_labels + 1:
            tick_values = tick_values[::2]
    tick_labels = [fmt % v for v in tick_values]

    # Triangle extensions on the strip ends (under/over indicators).
    # Width is bounded to keep them small relative to the strip.
    tri_w_full = min(strip_h * 0.7, strip_w * 0.05)
    tri_w_left = tri_w_full if extend_min else 0.0
    tri_w_right = tri_w_full if extend_max else 0.0
    gradient_w = strip_w - tri_w_left - tri_w_right

    # Auto-fit fonts. Tick labels are constrained to the gradient region;
    # the label can use the full visual strip width since it sits above.
    tick_fs, title_fs = _fit_fonts_to_strip(
        plotter,
        gradient_w,
        tick_labels,
        label,
        tick_fs,
        title_fs,
    )

    # Vertical dimension estimates (after possible font shrinkage)
    title_h = _text_height(title_fs)
    tick_h = _text_height(tick_fs, ratio=_TICK_H_RATIO)

    # Gaps and tick mark length
    label_gap = 0.008  # space between top of label and bottom of tick mark
    title_gap = 0.010  # space between top of strip and bottom of title
    tick_mark_h = 0.006

    # Padding (and rounding base) shared with badge/label via _overlay_layout
    pad_x, pad_y, base_rounding = _overlay_layout(t)

    # Background dimensions
    bg_w = strip_w + 2 * pad_x
    bg_h = (
        pad_y
        + tick_h
        + label_gap
        + tick_mark_h
        + strip_h
        + (title_gap + title_h if label else 0)
        + pad_y
    )

    # Position anchor — colorbar uses a slightly larger edge margin than
    # the badge default for visual breathing room
    bg_x, bg_y = _position_xy(loc, bg_w, bg_h, t.overlay_margin + 0.02)

    # Interior positions (bottom to top)
    strip_x = bg_x + pad_x
    strip_y = bg_y + pad_y + tick_h + label_gap + tick_mark_h
    tick_mark_top = strip_y
    tick_mark_bottom = strip_y - tick_mark_h
    tick_label_y = bg_y + pad_y
    # Label is centered horizontally over the strip via VTK justification
    title_x = strip_x + strip_w * 0.5
    title_y = strip_y + strip_h + title_gap

    # Tick x-positions span the inner gradient region only; the triangles
    # at each end represent under/over values and have no labels.
    gradient_x = strip_x + tri_w_left
    if span < 1e-12:
        ticks_x = [gradient_x + gradient_w * 0.5]
    else:
        ticks_x = [gradient_x + (v - lo) / span * gradient_w for v in tick_values]

    # --- Draw layers (back to front) ---

    # 1. Background
    rounding = min(base_rounding, bg_w * 0.2, bg_h * 0.2)
    draw_rounded_rect(
        plotter,
        bg_x,
        bg_y,
        bg_w,
        bg_h,
        rounding,
        bg_color[:3],
        bg_color[3],
    )
    draw_rounded_rect_border(
        plotter,
        bg_x,
        bg_y,
        bg_w,
        bg_h,
        rounding,
        border_color[:3],
        border_color[3],
    )

    # 2. Inner gradient + optional under/over triangle extensions + outline
    _draw_gradient_strip(
        plotter,
        gradient_x,
        strip_y,
        gradient_w,
        strip_h,
        cmap,
    )

    y_mid = strip_y + strip_h * 0.5
    # Triangles use cmap.get_under() / get_over() so that callers can
    # customize them via cmap.set_under() / set_over(). When not set,
    # these return the gradient endpoints (cmap(0) / cmap(1)). The
    # *extremes* mode then scales the rgb (darken) and/or the opacity
    # (semi) — same vocabulary as the matplotlib backend.
    if extend_min:
        under_rgba = cmap.get_under()
        _draw_triangle(
            plotter,
            (gradient_x, strip_y),
            (gradient_x, strip_y + strip_h),
            (strip_x, y_mid),
            (
                under_rgba[0] * tri_rgb_factor,
                under_rgba[1] * tri_rgb_factor,
                under_rgba[2] * tri_rgb_factor,
            ),
            tri_opacity,
        )
    if extend_max:
        over_rgba = cmap.get_over()
        _draw_triangle(
            plotter,
            (gradient_x + gradient_w, strip_y + strip_h),
            (gradient_x + gradient_w, strip_y),
            (strip_x + strip_w, y_mid),
            (
                over_rgba[0] * tri_rgb_factor,
                over_rgba[1] * tri_rgb_factor,
                over_rgba[2] * tri_rgb_factor,
            ),
            tri_opacity,
        )

    # Outline tracing the (optionally extended) strip, clockwise from top-left.
    # Each end is either a triangle apex or a flat vertical edge.
    outline_pts: list[tuple[float, float]] = [
        (gradient_x, strip_y + strip_h),
        (gradient_x + gradient_w, strip_y + strip_h),
    ]
    if extend_max:
        outline_pts.append((strip_x + strip_w, y_mid))
    else:
        outline_pts.append((gradient_x + gradient_w, strip_y + strip_h))
    outline_pts.append((gradient_x + gradient_w, strip_y))
    outline_pts.append((gradient_x, strip_y))
    if extend_min:
        outline_pts.append((strip_x, y_mid))
    _draw_polyline_border(
        plotter,
        outline_pts,
        border_color[:3],
        border_color[3],
        line_width=1.0,
    )

    # 3. Tick marks
    text_rgb = text_color[:3]
    _draw_tick_marks(
        plotter,
        ticks_x,
        tick_mark_top,
        tick_mark_bottom,
        text_rgb,
        opacity=0.85,
        line_width=1.5,
    )

    # 4. Label (centered above the strip)
    if label:
        _add_text_actor(
            plotter,
            label,
            title_x,
            title_y,
            title_fs,
            text_rgb,
            h_align="center",
        )

    # 5. Tick labels (centered on each tick)
    for tx, tick_label in zip(ticks_x, tick_labels, strict=True):
        _add_text_actor(
            plotter,
            tick_label,
            tx,
            tick_label_y,
            tick_fs,
            text_rgb,
            h_align="center",
        )
