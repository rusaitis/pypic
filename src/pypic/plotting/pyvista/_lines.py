"""Field line and trajectory tube rendering for pyvista."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypic.plotting.pyvista._guard import ensure_pyvista
from pypic.plotting.pyvista._theme import _resolve_theme, resolve_cmap

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from pypic.dataset import FieldDataset
    from pypic.plotting.styles import PlotTheme
    from pypic.traces import FieldLine, ParticleTrace
    from pypic.types import FloatArray


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


def _trajectory_scalar(trace: ParticleTrace, scalar: str) -> FloatArray:
    if scalar in trace.scalars:
        return trace.scalars[scalar]
    if scalar == "time":
        return trace.time
    if scalar == "speed":
        return np.linalg.norm(trace.velocity, axis=1)
    available = sorted({*trace.scalars, "time", "speed"})
    msg = f"Scalar {scalar!r} not on the trace; available: {available}"
    raise KeyError(msg)


def _tube_colors(n: int, color: str | None, theme: PlotTheme | None) -> list[str]:
    if color is not None:
        return [color] * n
    cycle = _resolve_theme(theme).color_cycle or ("#1f77b4",)
    return [cycle[i % len(cycle)] for i in range(n)]


def _add_colored_tubes(
    plotter: pv.Plotter,
    paths: list[FloatArray],
    colors: list[str],
    *,
    radius: float,
    opacity: float,
) -> list[pv.Actor]:
    return [
        plotter.add_mesh(
            _polyline_from_points(points).tube(radius=radius),
            color=tube_color,
            opacity=opacity,
            show_scalar_bar=False,
        )
        for points, tube_color in zip(paths, colors, strict=True)
    ]


def _add_scalar_tubes(
    plotter: pv.Plotter,
    paths: list[FloatArray],
    values: list[FloatArray],
    *,
    name: str,
    cmap: Colormap,
    clim: tuple[float, float] | None,
    radius: float,
    opacity: float,
    show_scalar_bar: bool,
    scalar_bar_position: str,
    theme: PlotTheme | None,
) -> list[pv.Actor]:
    actors = []
    for points, tube_values in zip(paths, values, strict=True):
        polyline = _polyline_from_points(points)
        polyline[name] = tube_values
        actors.append(
            plotter.add_mesh(
                polyline.tube(radius=radius),
                scalars=name,
                cmap=cmap,
                clim=clim,
                opacity=opacity,
                show_scalar_bar=False,
            )
        )
    if show_scalar_bar and clim is not None:
        from pypic.plotting.pyvista._overlay import add_colorbar

        add_colorbar(
            plotter, cmap, clim, label=name, loc=scalar_bar_position, theme=theme
        )
    return actors


def add_field_lines(
    plotter: pv.Plotter,
    field_lines: list[FieldLine],
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
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> list[pv.Actor]:
    r"""Render field lines as tubes on one shared color scale.

    When *data* is provided, scalars are automatically sampled along
    each line. When *units* is also set, values are unit-converted.
    When *cmap* and *clim* are ``None``, they are auto-derived from
    field metadata and the sampled values: symmetric about zero for a
    signed field on the theme's diverging map, ``(0, max)`` on its
    sequential map otherwise.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_lines : list[FieldLine]
        Field lines to render.
    color : str or None
        Uniform color for every line when *scalar* is ``None``. ``None``
        cycles the theme's color cycle.
    scalar : str or None
        Field name for coloring. Sampled from *data* if not already
        on each line. ``None`` draws uniform colors.
    data : FieldDataset or None
        Source dataset for scalar sampling and unit conversion.
    units : str or None
        Display units (e.g. ``"nT"``). Requires *data*.
    cmap : str, Colormap, or None
        Colormap. ``None`` auto-selects from field metadata.
    clim : tuple[float, float] or None
        Shared color limits. ``None`` auto-computes.
    signed : bool
        Whether the scalar is signed or positive-definite; only read
        when *cmap* is ``None`` and there is no *data* to decide from.
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
        One actor per field line.
    """
    ensure_pyvista()

    if not field_lines:
        return []
    paths = [fl.points for fl in field_lines]
    if scalar is None:
        colors = _tube_colors(len(paths), color, theme)
        return _add_colored_tubes(
            plotter, paths, colors, radius=radius, opacity=opacity
        )

    from pypic.plotting._colormaps import is_positive_definite

    sampled = [_prepare_scalar(fl, scalar, data, units) for fl in field_lines]
    values = [tube_values for tube_values, _ in sampled]
    combined = np.concatenate(values)
    info = data.field_info(scalar) if data is not None else None
    is_positive = is_positive_definite(scalar, combined, info)
    if cmap is None and data is not None:
        signed = not is_positive
    finite = combined[np.isfinite(combined)]
    if clim is None and finite.size > 0:
        top = float(np.max(finite if is_positive else np.abs(finite)))
        clim = (0.0, top) if is_positive else (-top, top)
    return _add_scalar_tubes(
        plotter,
        paths,
        values,
        name=sampled[0][1],
        cmap=resolve_cmap(cmap, signed=signed, theme=theme),
        clim=clim,
        radius=radius,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_position=scalar_bar_position,
        theme=theme,
    )


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
) -> pv.Actor:
    r"""Render one `FieldLine` as a tube, exactly as `add_field_lines` would.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    field_line : FieldLine
        Field line to render.
    color : str or None
        Uniform color when *scalar* is ``None``. ``None`` uses the first
        color of the theme's cycle.
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
        Color limits. ``None`` derives them as `add_field_lines` does.
    signed : bool
        Whether the scalar is signed (diverging cmap) or positive-definite;
        only read when *cmap* is ``None`` and there is no *data*.
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
    pv.Actor
    """
    (actor,) = add_field_lines(
        plotter,
        [field_line],
        color=color,
        scalar=scalar,
        data=data,
        units=units,
        cmap=cmap,
        clim=clim,
        signed=signed,
        radius=radius,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_position=scalar_bar_position,
        theme=theme,
    )
    return actor


def add_trajectories(
    plotter: pv.Plotter,
    traces: list[ParticleTrace],
    *,
    color: str | None = None,
    scalar: str | None = None,
    cmap: str | Colormap | None = None,
    clim: tuple[float, float] | None = None,
    radius: float = 0.08,
    opacity: float = 1.0,
    show_scalar_bar: bool = True,
    scalar_bar_position: str = "lower_left",
    theme: PlotTheme | None = None,
) -> list[pv.Actor]:
    r"""Render particle trajectories as tubes on one shared color scale.

    Supports coloring by any scalar in ``trace.scalars``, or by
    ``"time"`` and ``"speed"``, derived from the trace when it does not
    carry a scalar of that name.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    traces : list[ParticleTrace]
        Particle trajectories to render.
    color : str or None
        Uniform color for every trace when *scalar* is ``None``. ``None``
        cycles the theme's color cycle.
    scalar : str or None
        Scalar name for coloring. ``None`` draws uniform colors.
    cmap : str, Colormap, or None
        Colormap. ``None`` selects the theme's sequential map.
    clim : tuple[float, float] or None
        Shared color limits. ``None`` spans the finite values of all traces.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a single themed scalar bar via `add_colorbar`.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    list[pv.Actor]
        One actor per trace.

    Raises
    ------
    KeyError
        If a trace carries no *scalar* and it is neither ``"time"`` nor
        ``"speed"``.
    """
    ensure_pyvista()

    if not traces:
        return []
    paths = [tr.points for tr in traces]
    if scalar is None:
        colors = _tube_colors(len(paths), color, theme)
        return _add_colored_tubes(
            plotter, paths, colors, radius=radius, opacity=opacity
        )

    values = [_trajectory_scalar(tr, scalar) for tr in traces]
    combined = np.concatenate(values)
    finite = combined[np.isfinite(combined)]
    if clim is None and finite.size > 0:
        clim = (float(finite.min()), float(finite.max()))
    return _add_scalar_tubes(
        plotter,
        paths,
        values,
        name=scalar,
        cmap=resolve_cmap(cmap, signed=False, theme=theme),
        clim=clim,
        radius=radius,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_position=scalar_bar_position,
        theme=theme,
    )


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
) -> pv.Actor:
    r"""Render one `ParticleTrace` as a tube, exactly as `add_trajectories` would.

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter.
    trace : ParticleTrace
        Particle trajectory to render.
    color : str or None
        Uniform color when *scalar* is ``None``. ``None`` uses the first
        color of the theme's cycle.
    scalar : str or None
        Name of a scalar for per-point coloring. ``"time"`` and
        ``"speed"`` are derived automatically if not in ``trace.scalars``.
    cmap : str, Colormap, or None
        Colormap. ``None`` selects from theme.
    clim : tuple[float, float] or None
        Color limits. ``None`` spans the trace's finite values.
    radius : float
        Tube radius.
    opacity : float
        Tube opacity (0–1).
    show_scalar_bar : bool
        Whether to show a themed scalar bar via `add_colorbar`.
    scalar_bar_position : str
        Scalar bar corner: ``"lower_left"``, ``"lower_right"``, etc.
    theme : PlotTheme or None
        Theme for defaults.

    Returns
    -------
    pv.Actor

    Raises
    ------
    KeyError
        If the trace carries no *scalar* and it is neither ``"time"`` nor
        ``"speed"``.
    """
    (actor,) = add_trajectories(
        plotter,
        [trace],
        color=color,
        scalar=scalar,
        cmap=cmap,
        clim=clim,
        radius=radius,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_position=scalar_bar_position,
        theme=theme,
    )
    return actor
