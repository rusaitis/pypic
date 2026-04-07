"""Progress badge overlay for pyvista 3D plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pypic.plotting._format import (
    BadgeLoc,
    _format_status_text,
    _normalize_loc,
)
from pypic.plotting._overlay_common import resolve_rgba_override
from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._overlay import (
    _overlay_layout,
    _position_xy,
    _resolve_overlay_colors,
    _text_height,
    draw_rounded_rect,
    draw_rounded_rect_border,
)
from pypic.plotting.pyvista._theme import _resolve_theme

if TYPE_CHECKING:
    import pyvista as pv

    from pypic.plotting.styles import PlotTheme


def add_badge(
    plotter: pv.Plotter,
    text: str | None = None,
    *,
    step: int | None = None,
    time: float | str | None = None,
    time_units: str = "",
    step_range: tuple[int, int] | None = None,
    label: str | None = None,
    show_max: bool = True,
    progress: float | None = None,
    variant: str | None = None,
    loc: BadgeLoc | None = None,
    fontsize: float | None = None,
    bar_color: str | None = None,
    bar_alpha: float = 0.8,
    bar_width: float | None = None,
    bar_height: float | None = None,
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
    track_color: str | tuple[float, ...] | None = None,
    width: float | None = None,
    height: float | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Add a status badge overlay with optional progress bar.

    Renders simulation step, time, or custom text in a themed rounded
    rectangle overlay. A progress bar is shown when *progress* or
    *step_range* is set. Mirrors :func:`pypic.plotting.add_badge` for
    API parity between the matplotlib and pyvista paths — the parameter
    names match so backend-independent code can pass the same kwargs.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    text : str or None
        Direct custom text. When provided, *step* / *time* / *label*
        are ignored for content (but *progress* / *step_range* still
        drive the bar). Pass as a positional argument:
        ``add_badge(plotter, "Harris sheet, δ = 0.5 d_i")``.
    step : int or None
        Current simulation step number.
    time : float, str, or None
        Simulation time. A float is auto-formatted; a string is used
        verbatim (e.g. ``"13:34"``).
    time_units : str
        Units appended to numeric *time* (e.g. ``"s"``, ``"ns"``).
    step_range : tuple[int, int] or None
        ``(start, end)`` step range for auto-computing progress and
        rendering ``"step X / Y"``.
    label : str or None
        Custom prefix for the step (or time-only) display.
        ``None`` uses auto-labels (``"step"`` / ``"t"``).
        ``""`` suppresses the prefix entirely.
    show_max : bool
        When *step_range* is set, render ``"X / Y"`` if ``True``.
    progress : float or None
        Explicit progress fraction in ``[0, 1]``. Overrides
        *step_range* auto-computation when both are given.
    variant : str or None
        ``None`` uses the primary overlay colors; ``"alt"`` uses
        ``theme.overlay_alt_*`` and ``track_alt_color`` (matching the
        matplotlib badge variant). ``accent_color`` is unchanged.
    loc : BadgeLoc or None
        Corner or center placement. One of ``"upper left"``,
        ``"upper right"`` (default), ``"lower left"``, ``"lower right"``,
        ``"upper center"``. Legacy underscore form (``"upper_right"``)
        also accepted. ``None`` uses the default ``"upper right"``.
    fontsize : float or None
        Override label font size. ``None`` derives from theme via
        ``theme.badge_font_scale``.
    bar_color : str or None
        Progress bar fill color. ``None`` uses ``theme.accent_color``.
    bar_alpha : float
        Progress bar fill opacity.
    bar_width : float or None
        Override the badge box width in normalized viewport coordinates.
        ``None`` auto-grows the box to fit the status text.
    bar_height : float or None
        Override the badge box height in normalized viewport coordinates.
        ``None`` derives from font and progress-bar metrics.
    bg_color : str, tuple, or None
        Box background color override. ``None`` uses the theme/variant.
    bg_alpha : float or None
        Box background opacity override.
    text_color : str, tuple, or None
        Text color override. ``None`` uses the theme/variant.
    text_alpha : float
        Text opacity (applied on top of the variant-resolved color).
    track_color : str, tuple, or None
        Progress bar track color override. ``None`` uses the
        theme/variant track color.
    width : float or None
        Alias for *bar_width* — accepted for backward compatibility.
    height : float or None
        Alias for *bar_height* — accepted for backward compatibility.
    theme : PlotTheme or None
        Theme for colors.

    Notes
    -----
    Unlike :func:`pypic.plotting.add_badge` (matplotlib), this function
    returns ``None``. The underlying VTK actors are drawn directly into
    the plotter's renderer and are not exposed for post-hoc
    customization — re-call with the desired keyword arguments instead.
    """
    ensure_pyvista()
    from pypic.plotting.pyvista._overlay import (
        _add_text_actor,
        _measure_text_width_px,
    )

    t = _resolve_theme(theme)
    default_bg, default_border, default_text, default_track = (
        _resolve_overlay_colors(t, variant)
    )

    # Apply color overrides on top of theme/variant defaults
    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, default_bg)
    text_rgba = resolve_rgba_override(
        text_color, text_alpha, (*default_text[:3], text_alpha),
    )
    track_rgba = resolve_rgba_override(track_color, None, default_track)

    status_text = _format_status_text(
        text=text,
        step=step,
        time=time,
        time_units=time_units,
        step_range=step_range,
        label=label,
        show_max=show_max,
    )

    # Auto-compute progress from step_range when not explicitly given
    if progress is None and step_range is not None and step is not None:
        start, end = step_range
        if start > end:
            msg = f"step_range start must be <= end, got ({start}, {end})"
            raise ValueError(msg)
        progress = max(0.0, min(1.0, (step - start) / max(end - start, 1)))

    if not status_text and progress is None:
        return

    fs = (
        fontsize
        if fontsize is not None
        else int(t.font_overlay * t.badge_font_scale)
    )
    text_h = _text_height(int(fs))

    # Padding/rounding shared with colorbar/label via _overlay_layout.
    # The badge uses pad_y as its single inner padding (it's symmetric).
    _, pad, rounding = _overlay_layout(t)
    edge_margin = t.overlay_margin          # distance from window edge
    margin = t.overlay_margin * 0.5        # inset for bar/text within box
    bar_h = t.progress_bar_height * 0.002
    box_w = t.progress_bar_width * 0.0025  # 80.0 → 0.20
    box_h = (
        pad
        + (text_h if status_text else 0)
        + (bar_h + pad if progress is not None else 0)
        + pad
    )

    # Explicit overrides (new *_width/*_height names take precedence over
    # the legacy width/height aliases for symmetry with matplotlib)
    effective_width = bar_width if bar_width is not None else width
    effective_height = bar_height if bar_height is not None else height
    if effective_width is not None:
        box_w = effective_width
    if effective_height is not None:
        box_h = effective_height

    # Auto-grow the box to fit long status text (unless explicitly sized)
    if effective_width is None and status_text:
        text_px = _measure_text_width_px(plotter, status_text, int(fs))
        ww = (
            float(plotter.window_size[0])
            if hasattr(plotter, "window_size")
            else 1600.0
        )
        if ww > 0:
            text_w_norm = text_px / ww
            required_w = text_w_norm + 2 * margin
            if required_w > box_w:
                box_w = min(required_w, 0.5)

    canonical_loc = _normalize_loc(loc) if loc else "upper right"
    origin_x, origin_y = _position_xy(canonical_loc, box_w, box_h, edge_margin)

    # Background rounded rectangle with subtle border
    draw_rounded_rect(
        plotter, origin_x, origin_y, box_w, box_h, rounding,
        bg_rgba[:3], bg_rgba[3],
    )
    draw_rounded_rect_border(
        plotter, origin_x, origin_y, box_w, box_h, rounding,
        default_border[:3], default_border[3],
    )

    # Progress bar (track + fill)
    if progress is not None:
        fraction = max(0.0, min(1.0, progress))
        bar_x = origin_x + margin
        bar_y = origin_y + pad
        bar_w_draw = box_w - 2 * margin
        bar_r = t.progress_bar_rounding * 0.005

        # Track
        draw_rounded_rect(
            plotter, bar_x, bar_y, bar_w_draw, bar_h, bar_r,
            track_rgba[:3], track_rgba[3],
        )

        # Fill
        if fraction > 0.01:
            from matplotlib.colors import to_rgb

            accent_rgb = to_rgb(bar_color) if bar_color else to_rgb(t.accent_color)
            fill_w = max(bar_h, bar_w_draw * fraction)
            draw_rounded_rect(
                plotter, bar_x, bar_y, fill_w, bar_h, bar_r,
                accent_rgb, bar_alpha,
            )

    # Text label (left-aligned within the box)
    if status_text:
        text_x = origin_x + margin
        text_y = origin_y + box_h - pad - text_h
        _add_text_actor(
            plotter, status_text, text_x, text_y, int(fs), text_rgba[:3],
            h_align="left",
        )


def add_label(
    plotter: pv.Plotter,
    label: str,
    *,
    variant: str | None = None,
    loc: BadgeLoc | None = None,
    fontsize: float | None = None,
    fontweight: str = "bold",
    ha: Literal["left", "center", "right"] = "center",
    bg_color: str | tuple[float, ...] | None = None,
    bg_alpha: float | None = None,
    text_color: str | tuple[float, ...] | None = None,
    text_alpha: float = 0.8,
    width: float | None = None,
    height: float | None = None,
    theme: PlotTheme | None = None,
) -> None:
    r"""Add a panel-letter or short-phrase overlay.

    Styled with the same rounded box, border, and theme variant system
    as :func:`add_badge` and :func:`add_colorbar`. Mirrors
    :func:`pypic.plotting.add_label` for parity between the matplotlib
    and pyvista paths — parameter names match so backend-independent
    code can pass the same kwargs.

    Single-token labels (no spaces) render bold using ``font_title`` and
    are padded so the box looks square — appropriate for panel letters
    like ``"a"``, ``"b"``. Multi-word phrases render normal-weight using
    ``font_label`` when *fontweight* is left at its default of
    ``"bold"``; pass any other value to opt out of the auto-downgrade.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    label : str
        Label text (e.g. ``"a"``, ``"Run A"``).
    variant : str or None
        ``None`` uses the primary overlay colors; ``"alt"`` uses
        ``theme.overlay_alt_*``.
    loc : BadgeLoc or None
        Corner or center placement: ``"upper left"`` (default),
        ``"upper right"``, ``"lower left"``, ``"lower right"``,
        ``"upper center"``. Legacy underscore form (``"upper_left"``)
        also accepted. ``None`` uses the default ``"upper left"``.
    fontsize : float or None
        Override label font size. ``None`` auto-derives from
        ``theme.font_title`` (single token) or ``theme.font_label``
        (phrase), scaled by ``theme.badge_font_scale``.
    fontweight : str
        Font weight (e.g. ``"bold"``, ``"normal"``). Default ``"bold"``
        is auto-downgraded to ``"normal"`` for multi-word phrases to
        match :func:`pypic.plotting.add_label`.
    ha : str
        Horizontal text alignment within the box: ``"left"``,
        ``"center"`` (default), or ``"right"``. The pyvista default
        differs from matplotlib's ``"left"`` because pyvista anchors
        the text actor at the box center; visually, single-token
        labels render identically either way.
    bg_color : str, tuple, or None
        Box background color override. ``None`` uses the theme/variant.
    bg_alpha : float or None
        Box background opacity override.
    text_color : str, tuple, or None
        Text color override. ``None`` uses the theme/variant.
    text_alpha : float
        Text opacity (applied on top of the variant-resolved color).
    width : float or None
        Override the box width in normalized viewport coordinates.
        ``None`` derives from text width (with square enforcement for
        single tokens).
    height : float or None
        Override the box height. ``None`` derives from font metrics.
    theme : PlotTheme or None
        Theme for colors.

    Notes
    -----
    Unlike :func:`pypic.plotting.add_label` (matplotlib), this function
    returns ``None``. The underlying VTK actors are drawn directly into
    the plotter's renderer and are not exposed for post-hoc
    customization — re-call with the desired keyword arguments instead.
    """
    ensure_pyvista()
    from pypic.plotting.pyvista._overlay import (
        _add_text_actor,
        _measure_text_width_px,
    )

    if not label:
        return

    t = _resolve_theme(theme)
    default_bg, default_border, default_text, _ = _resolve_overlay_colors(
        t, variant,
    )

    bg_rgba = resolve_rgba_override(bg_color, bg_alpha, default_bg)
    text_rgba = resolve_rgba_override(
        text_color, text_alpha, (*default_text[:3], text_alpha),
    )

    is_phrase = " " in label
    # Auto-downgrade bold → normal for multi-word phrases (matplotlib parity)
    effective_weight = "normal" if (fontweight == "bold" and is_phrase) else fontweight
    bold = effective_weight not in ("normal", "light", "ultralight")

    if fontsize is None:
        base = t.font_label if is_phrase else t.font_title
        fontsize = int(base * t.badge_font_scale)
    fontsize_int = int(fontsize)

    # Layout shared with badge/colorbar via _overlay_layout
    text_h = _text_height(fontsize_int)
    pad_x, pad_y, base_rounding = _overlay_layout(t)

    box_h = pad_y * 2 + text_h

    # Measure rendered text width and compute the box width.
    # Single tokens get square padding so panel letters look like a chip.
    text_px = _measure_text_width_px(plotter, label, fontsize_int)
    ww = (
        float(plotter.window_size[0])
        if hasattr(plotter, "window_size")
        else 1600.0
    )
    text_w_norm = text_px / ww if ww > 0 else 0.0
    if is_phrase:
        box_w = text_w_norm + 2 * pad_x
    else:
        box_w = max(text_w_norm + 2 * pad_x, box_h)

    # Explicit overrides
    if width is not None:
        box_w = width
    if height is not None:
        box_h = height

    edge_margin = t.overlay_margin
    canonical_loc = _normalize_loc(loc) if loc else "upper left"
    origin_x, origin_y = _position_xy(canonical_loc, box_w, box_h, edge_margin)

    # Background rounded rect with subtle border
    rounding = min(base_rounding, box_w * 0.2, box_h * 0.2)
    draw_rounded_rect(
        plotter, origin_x, origin_y, box_w, box_h, rounding,
        bg_rgba[:3], bg_rgba[3],
    )
    draw_rounded_rect_border(
        plotter, origin_x, origin_y, box_w, box_h, rounding,
        default_border[:3], default_border[3],
    )

    # Label text — horizontal anchor depends on ha; vertical is always
    # centered (baseline adjusted by _text_height for glyph descent).
    if ha == "left":
        text_x = origin_x + pad_x
    elif ha == "right":
        text_x = origin_x + box_w - pad_x
    else:
        text_x = origin_x + box_w * 0.5
    text_y = origin_y + (box_h - text_h) * 0.5
    _add_text_actor(
        plotter, label, text_x, text_y, fontsize_int, text_rgba[:3],
        h_align=ha,
        bold=bold,
    )
