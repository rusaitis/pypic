"""Field line and trajectory tube rendering for pyvista."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import resolve_cmap

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme
    from pypic.traces import FieldLine, ParticleTrace


def _polyline_from_points(points: np.ndarray) -> Any:
    """Build a pyvista PolyData polyline from an (N, 3) point array."""
    import pyvista as pv

    n = points.shape[0]
    polyline = pv.PolyData(points)
    cells = np.column_stack([
        np.full(n - 1, 2, dtype=int),
        np.arange(n - 1),
        np.arange(1, n),
    ]).ravel()
    polyline.lines = cells
    return polyline


def add_field_line(
    plotter: Any,
    field_line: FieldLine,
    *,
    color: str | None = None,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    signed: bool = True,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    theme: PlotTheme | None = None,
) -> None:
    r"""Render a :class:`FieldLine` as a colored tube.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_line : FieldLine
        Field line to render.
    color : str or None
        Uniform color. Ignored if *scalar* is set.
    scalar : str or None
        Name of a scalar in ``field_line.scalars`` for per-point coloring.
    cmap : str, Colormap, or None
        Colormap for scalar coloring. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Color limits for scalar coloring. ``None`` for auto.
    signed : bool
        Whether the scalar is signed (diverging cmap) or positive-definite
        (sequential cmap). Only affects default colormap selection.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show the scalar bar.
    theme : PlotTheme or None
        Theme for defaults.
    """
    ensure_pyvista()

    pts = field_line.points
    if pts.shape[0] < 2:
        return

    polyline = _polyline_from_points(pts)

    if scalar is not None and scalar in field_line.scalars:
        resolved_cmap = resolve_cmap(cmap, signed=signed, theme=theme)
        polyline[scalar] = field_line.scalars[scalar]
        tube = polyline.tube(radius=radius)
        plotter.add_mesh(
            tube,
            scalars=scalar,
            cmap=resolved_cmap,
            clim=clim,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
        )
    else:
        tube = polyline.tube(radius=radius)
        plotter.add_mesh(
            tube, color=color or "white", opacity=opacity, show_scalar_bar=False,
        )


def add_field_lines(
    plotter: Any,
    field_lines: list[FieldLine],
    *,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    signed: bool = True,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = True,
    theme: PlotTheme | None = None,
) -> None:
    r"""Render multiple field lines with a shared color scale.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_lines : list[FieldLine]
        Field lines to render.
    scalar : str or None
        Name of a scalar in each ``field_line.scalars`` for coloring.
    cmap : str, Colormap, or None
        Colormap. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Shared color limits. ``None`` computes from all lines.
    signed : bool
        Whether the scalar is signed (diverging) or positive-definite
        (sequential). Only affects default colormap selection.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a single scalar bar for all lines.
    theme : PlotTheme or None
        Theme for defaults.
    """
    ensure_pyvista()

    if not field_lines:
        return

    # Auto-compute shared clim if needed
    if scalar is not None and clim is None:
        all_vals = []
        for fl in field_lines:
            if scalar in fl.scalars:
                all_vals.append(fl.scalars[scalar])
        if all_vals:
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            if len(valid) > 0:
                clim = (float(np.min(valid)), float(np.max(valid)))

    for i, fl in enumerate(field_lines):
        is_last = i == len(field_lines) - 1
        add_field_line(
            plotter,
            fl,
            scalar=scalar,
            cmap=cmap,
            clim=clim,
            signed=signed,
            radius=radius,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar and is_last,
            theme=theme,
        )


def add_trajectory(
    plotter: Any,
    trace: ParticleTrace,
    *,
    color: str | None = None,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    radius: float = 0.06,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    theme: PlotTheme | None = None,
) -> None:
    r"""Render a :class:`ParticleTrace` as a colored tube.

    Supports coloring by any scalar in ``trace.scalars``, or by the
    built-in ``"time"`` or ``"speed"`` keys (derived automatically).

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    trace : ParticleTrace
        Particle trajectory to render.
    color : str or None
        Uniform color. Ignored if *scalar* is set.
    scalar : str or None
        Name of a scalar for per-point coloring. ``"time"`` and
        ``"speed"`` are derived automatically if not in ``trace.scalars``.
    cmap : str, Colormap, or None
        Colormap. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Color limits. ``None`` for auto.
    radius : float
        Tube radius.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    theme : PlotTheme or None
        Theme for defaults.
    """
    ensure_pyvista()

    pts = trace.points
    if pts.shape[0] < 2:
        return

    polyline = _polyline_from_points(pts)

    # Resolve built-in scalars
    scalars = {**trace.scalars, "time": trace.time}
    speed = np.sqrt(np.sum(trace.velocity**2, axis=1))
    scalars["speed"] = speed

    if scalar is not None and scalar in scalars:
        resolved_cmap = resolve_cmap(cmap, signed=False, theme=theme)
        polyline[scalar] = scalars[scalar]
        tube = polyline.tube(radius=radius)
        plotter.add_mesh(
            tube,
            scalars=scalar,
            cmap=resolved_cmap,
            clim=clim,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
        )
    else:
        tube = polyline.tube(radius=radius)
        plotter.add_mesh(
            tube, color=color or "white", opacity=opacity, show_scalar_bar=False,
        )


def add_trajectories(
    plotter: Any,
    traces: list[ParticleTrace],
    *,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    radius: float = 0.06,
    opacity: float = 1.0,
    show_scalar_bar: bool = True,
    theme: PlotTheme | None = None,
) -> None:
    r"""Render multiple particle trajectories with a shared color scale.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    traces : list[ParticleTrace]
        Particle trajectories to render.
    scalar : str or None
        Scalar name for coloring (see :func:`add_trajectory`).
    cmap : str, Colormap, or None
        Colormap. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Shared color limits. ``None`` computes from all traces.
    radius : float
        Tube radius.
    show_scalar_bar : bool
        Whether to show a single scalar bar.
    theme : PlotTheme or None
        Theme for defaults.
    """
    ensure_pyvista()

    if not traces:
        return

    # Auto-compute shared clim
    if scalar is not None and clim is None:
        all_vals = []
        for tr in traces:
            built_in = {"time": tr.time, "speed": np.sqrt(np.sum(tr.velocity**2, axis=1))}
            scalars = {**tr.scalars, **built_in}
            if scalar in scalars:
                all_vals.append(scalars[scalar])
        if all_vals:
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            if len(valid) > 0:
                clim = (float(np.min(valid)), float(np.max(valid)))

    for i, tr in enumerate(traces):
        is_last = i == len(traces) - 1
        add_trajectory(
            plotter,
            tr,
            scalar=scalar,
            cmap=cmap,
            clim=clim,
            radius=radius,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar and is_last,
            theme=theme,
        )
