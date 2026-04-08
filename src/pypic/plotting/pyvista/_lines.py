"""Field line and trajectory tube rendering for pyvista."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme, resolve_cmap

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from pypic.plotting.styles import PlotTheme
    from pypic.readers.base import FieldDataset
    from pypic.traces import FieldLine, ParticleTrace


def _polyline_from_points(points: np.ndarray) -> pv.PolyData:
    """Build a pyvista PolyData polyline from an (N, 3) point array."""
    import pyvista as _pv

    n = points.shape[0]
    polyline = _pv.PolyData(points)
    cells = np.column_stack(
        [
            np.full(n - 1, 2, dtype=int),
            np.arange(n - 1),
            np.arange(1, n),
        ]
    ).ravel()
    polyline.lines = cells
    return polyline


def _prepare_scalar(
    field_line: FieldLine,
    scalar: str,
    data: FieldDataset | None,
    units: str | None,
) -> tuple[np.ndarray, str]:
    """Sample and convert scalar values for a field line.

    Returns (values, display_name) where display_name includes units.
    """
    # Sample from data if not already on the field line
    if scalar in field_line.scalars:
        values = field_line.scalars[scalar]
    elif data is not None:
        from pypic.traces import sample_field

        values = sample_field(data, field_line.points, scalar)
    else:
        msg = f"Scalar {scalar!r} not in field line and no data provided"
        raise KeyError(msg)

    # Unit conversion
    display_name = scalar
    if units is not None and data is not None:
        # Get the SI conversion factor from a single-value probe
        info = data.field_info(scalar)
        quantity_type = info.quantity_type if info else "b_field"
        si_val = float(data.normalization.to_si(quantity_type, 1.0))
        from pypic.compute import display_unit_factor

        unit_factor = display_unit_factor(units)
        values = values * si_val / unit_factor
        display_name = f"{scalar} ({units})"

    return values, display_name


def add_field_line(
    plotter: pv.Plotter,
    field_line: FieldLine,
    *,
    color: str | None = None,
    scalar: str | None = None,
    data: FieldDataset | None = None,
    units: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    signed: bool = True,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> pv.Actor | None:
    r"""Render a :class:`FieldLine` as a colored tube.

    When *data* is provided with *scalar*, the field is automatically
    sampled along the line via ``attach_scalars()``. When *units* is
    also set, values are converted to display units.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_line : FieldLine
        Field line to render.
    color : str or None
        Uniform color. Ignored if *scalar* is set.
    scalar : str or None
        Name of a scalar for per-point coloring. Sampled from *data*
        automatically if not already in ``field_line.scalars``.
    data : FieldDataset or None
        Source dataset for scalar sampling and unit conversion.
    units : str or None
        Display units (e.g. ``"nT"``). Requires *data*.
    cmap : str, Colormap, or None
        Colormap. ``None`` auto-selects from field metadata.
    clim : tuple[float, float] or None
        Color limits. ``None`` for auto.
    signed : bool
        Whether the scalar is signed (diverging cmap) or positive-definite.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a themed scalar bar.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    pv.Actor or None
    """
    ensure_pyvista()

    pts = field_line.points
    if pts.shape[0] < 2:
        return None

    polyline = _polyline_from_points(pts)

    if scalar is not None:
        values, display_name = _prepare_scalar(field_line, scalar, data, units)
        resolved_cmap = resolve_cmap(cmap, signed=signed, theme=theme)
        polyline[display_name] = values
        tube = polyline.tube(radius=radius)

        actor = plotter.add_mesh(
            tube,
            scalars=display_name,
            cmap=resolved_cmap,
            clim=clim,
            opacity=opacity,
            show_scalar_bar=False,
        )

        if show_scalar_bar and clim is not None:
            from pypic.plotting.pyvista._overlay import add_colorbar

            add_colorbar(
                plotter,
                resolved_cmap,
                clim,
                label=display_name,
                loc=scalar_bar_position,
                theme=theme,
            )

        return actor
    tube = polyline.tube(radius=radius)
    return plotter.add_mesh(
        tube,
        color=color or "white",
        opacity=opacity,
        show_scalar_bar=False,
    )


def add_field_lines(
    plotter: pv.Plotter,
    field_lines: list[FieldLine],
    *,
    scalar: str | None = None,
    data: FieldDataset | None = None,
    units: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    signed: bool = True,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> list[pv.Actor]:
    r"""Render multiple field lines with a shared color scale.

    When *data* is provided, scalars are automatically sampled along
    each line. When *units* is also set, values are unit-converted.
    When *cmap* and *clim* are ``None``, they are auto-derived from
    field metadata.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_lines : list[FieldLine]
        Field lines to render.
    scalar : str or None
        Field name for coloring. Sampled from *data* if not already
        on each line. ``None`` uses theme color cycle.
    data : FieldDataset or None
        Source dataset for scalar sampling and unit conversion.
    units : str or None
        Display units (e.g. ``"nT"``). Requires *data*.
    cmap : str, Colormap, or None
        Colormap. ``None`` auto-selects from field metadata.
    clim : tuple[float, float] or None
        Shared color limits. ``None`` auto-computes.
    signed : bool
        Whether the scalar is signed or positive-definite.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a single themed scalar bar.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    list[pv.Actor]
    """
    ensure_pyvista()

    if not field_lines:
        return []

    # Auto-compute shared clim from all lines
    if scalar is not None and clim is None:
        all_vals = []
        for fl in field_lines:
            try:
                values, _ = _prepare_scalar(fl, scalar, data, units)
                all_vals.append(values)
            except KeyError:
                pass
        if all_vals:
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            if len(valid) > 0:
                from pypic.plotting._colormaps import is_positive_definite

                info = data.field_info(scalar) if data is not None else None
                if is_positive_definite(scalar, combined, info):
                    clim = (0.0, float(np.max(valid)))
                else:
                    absmax = float(np.max(np.abs(valid)))
                    clim = (-absmax, absmax)

    # Auto-select signedness from field metadata so the right default
    # cmap is chosen downstream by add_field_line via resolve_cmap()
    if scalar is not None and cmap is None and data is not None:
        from pypic.plotting._colormaps import is_positive_definite

        info = data.field_info(scalar)
        signed = not is_positive_definite(scalar, np.array([0.0]), info)

    # Color cycle for uniform-colored lines
    t = _resolve_theme(theme)
    cycle = t.color_cycle if t.color_cycle else ("#1f77b4",)

    actors: list[pv.Actor] = []
    for i, fl in enumerate(field_lines):
        is_last = i == len(field_lines) - 1
        line_color = None if scalar is not None else cycle[i % len(cycle)]
        actor = add_field_line(
            plotter,
            fl,
            color=line_color,
            scalar=scalar,
            data=data,
            units=units,
            cmap=cmap,
            clim=clim,
            signed=signed,
            radius=radius,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar and is_last,
            scalar_bar_position=scalar_bar_position,
            theme=theme,
        )
        if actor is not None:
            actors.append(actor)
    return actors


def add_trajectory(
    plotter: pv.Plotter,
    trace: ParticleTrace,
    *,
    color: str | None = None,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> pv.Actor | None:
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
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a themed scalar bar via :func:`add_colorbar`.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    pv.Actor or None
    """
    ensure_pyvista()

    pts = trace.points
    if pts.shape[0] < 2:
        return None

    polyline = _polyline_from_points(pts)

    scalars = {**trace.scalars, "time": trace.time}
    speed = np.sqrt(np.sum(trace.velocity**2, axis=1))
    scalars["speed"] = speed

    if scalar is not None and scalar in scalars:
        resolved_cmap = resolve_cmap(cmap, signed=False, theme=theme)
        polyline[scalar] = scalars[scalar]
        tube = polyline.tube(radius=radius)

        actor = plotter.add_mesh(
            tube,
            scalars=scalar,
            cmap=resolved_cmap,
            clim=clim,
            opacity=opacity,
            show_scalar_bar=False,
        )

        if show_scalar_bar and clim is not None:
            from pypic.plotting.pyvista._overlay import add_colorbar

            add_colorbar(
                plotter,
                resolved_cmap,
                clim,
                label=scalar,
                loc=scalar_bar_position,
                theme=theme,
            )

        return actor
    tube = polyline.tube(radius=radius)
    return plotter.add_mesh(
        tube,
        color=color or "white",
        opacity=opacity,
        show_scalar_bar=False,
    )


def add_trajectories(
    plotter: pv.Plotter,
    traces: list[ParticleTrace],
    *,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> list[pv.Actor]:
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
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a single themed scalar bar via :func:`add_colorbar`.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    list[pv.Actor]
    """
    ensure_pyvista()

    if not traces:
        return []

    if scalar is not None and clim is None:
        all_vals = []
        for tr in traces:
            built_in = {
                "time": tr.time,
                "speed": np.sqrt(np.sum(tr.velocity**2, axis=1)),
            }
            all_scalars = {**tr.scalars, **built_in}
            if scalar in all_scalars:
                all_vals.append(all_scalars[scalar])
        if all_vals:
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            if len(valid) > 0:
                clim = (float(np.min(valid)), float(np.max(valid)))

    t = _resolve_theme(theme)
    cycle = t.color_cycle if t.color_cycle else ("#1f77b4",)

    actors: list[pv.Actor] = []
    for i, tr in enumerate(traces):
        is_last = i == len(traces) - 1
        line_color = None if scalar is not None else cycle[i % len(cycle)]
        actor = add_trajectory(
            plotter,
            tr,
            color=line_color,
            scalar=scalar,
            cmap=cmap,
            clim=clim,
            radius=radius,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar and is_last,
            scalar_bar_position=scalar_bar_position,
            theme=theme,
        )
        if actor is not None:
            actors.append(actor)
    return actors
