"""3D line plotting: field lines and particle trajectories with scalar coloring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

    from pypic.traces import FieldLine, ParticleTrace
    from pypic.types import FloatArray


def _colored_line_3d(
    ax: Axes,
    points: FloatArray,
    values: FloatArray,
    *,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linewidth: float = 1.2,
    alpha: float = 0.9,
) -> object:
    """Create a ``Line3DCollection`` with per-segment scalar coloring.

    Parameters
    ----------
    ax : Axes
        A 3D ``Axes3D`` instance.
    points : FloatArray
        Positions along the curve, shape ``(N, 3)``.
    values : FloatArray
        Scalar values at each point, shape ``(N,)``. The segment between
        points *i* and *i+1* is colored by ``values[i]``.
    cmap : str or Colormap or None
        Colormap name or object. ``None`` uses ``"inferno"``.
    vmin, vmax : float or None
        Color limits. ``None`` for auto.
    linewidth : float
        Line width.
    alpha : float
        Line transparency.

    Returns
    -------
    Line3DCollection
        The collection added to *ax*.
    """
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = np.concatenate(
        [points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1
    )

    if cmap is None:
        cmap = "inferno"

    norm = Normalize(
        vmin=vmin if vmin is not None else float(np.nanmin(values)),
        vmax=vmax if vmax is not None else float(np.nanmax(values)),
    )

    lc = Line3DCollection(
        segments, cmap=cmap, norm=norm, linewidths=linewidth, alpha=alpha
    )
    lc.set_array(values[:-1])
    ax.add_collection3d(lc)  # type: ignore[attr-defined]
    return lc


def plot_field_line(
    ax: Axes,
    line: FieldLine,
    *,
    color: str | FloatArray | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linewidth: float = 1.2,
    alpha: float = 0.9,
    label: str | None = None,
    colorbar: bool | Literal["inset"] = False,
    extremes: str = "darken",
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> Artist:
    r"""Plot a field line in 3D, optionally colored by a scalar quantity.

    Parameters
    ----------
    ax : Axes
        A 3D ``Axes3D`` instance.
    line : FieldLine
        The field line to plot.
    color : str, FloatArray, or None
        How to color the line:

        - **str matching a scalar name** (e.g. ``"|B|"``, ``"arc_length"``)
          — look up in ``line.scalars`` and color by value via colormap.
        - **ndarray** of shape ``(N,)`` — per-point scalar coloring.
        - **color string** (e.g. ``"white"``, ``"#ff0000"``) — uniform color.
        - ``None`` — uniform color using the next color from the prop cycle.
    cmap : str or Colormap or None
        Colormap for scalar coloring. ``None`` uses ``"inferno"``.
    vmin, vmax : float or None
        Color limits for scalar coloring. ``None`` for auto.
    linewidth : float
        Line width.
    alpha : float
        Line transparency.
    label : str or None
        Legend label for the line.
    colorbar : bool or "inset"
        Whether to add a colorbar when using scalar coloring.
        ``"inset"`` uses :func:`add_inset_colorbar`.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
        ``"transparent"`` makes them invisible.
    **kwargs
        Passed to ``ax.plot()`` (uniform mode only).

    Returns
    -------
    Artist
        The line artist (``Line3D`` for uniform, ``Line3DCollection``
        for scalar coloring).
    """
    ensure_matplotlib()

    values = _resolve_color_values(color, line.points, line.scalars)

    if values is not None:
        artist = _colored_line_3d(
            ax, line.points, values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=linewidth, alpha=alpha,
        )
        if colorbar:
            cb_label = color if isinstance(color, str) else ""
            _add_3d_colorbar(ax, artist, cb_label, colorbar, extremes=extremes)
        return artist

    pts = line.points
    lines = ax.plot(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color=color, linewidth=linewidth, alpha=alpha, label=label,
        **kwargs,
    )
    return lines[0]


def plot_trajectory(
    ax: Axes,
    trace: ParticleTrace,
    *,
    color: str | FloatArray | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linewidth: float = 1.2,
    alpha: float = 0.9,
    label: str | None = None,
    colorbar: bool | Literal["inset"] = False,
    extremes: str = "darken",
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> Artist:
    r"""Plot a particle trajectory in 3D, optionally colored by a scalar.

    Parameters
    ----------
    ax : Axes
        A 3D ``Axes3D`` instance.
    trace : ParticleTrace
        The particle trajectory to plot.
    color : str, FloatArray, or None
        How to color the line:

        - **str matching a scalar name** (e.g. ``"speed"``, ``"kinetic_energy"``)
          — look up in ``trace.scalars`` and color by value via colormap.
        - ``"time"`` — color by the trajectory's time array.
        - **ndarray** of shape ``(N,)`` — per-point scalar coloring.
        - **color string** (e.g. ``"red"``) — uniform color.
        - ``None`` — uniform color using the next prop cycle color.
    cmap : str or Colormap or None
        Colormap for scalar coloring. ``None`` uses ``"inferno"``.
    vmin, vmax : float or None
        Color limits. ``None`` for auto.
    linewidth : float
        Line width.
    alpha : float
        Line transparency.
    label : str or None
        Legend label.
    colorbar : bool or "inset"
        Whether to add a colorbar when using scalar coloring.
    extremes : "darken" or "transparent"
        How to style values outside ``[vmin, vmax]``.
    **kwargs
        Passed to ``ax.plot()`` (uniform mode only).

    Returns
    -------
    Artist
        The line artist.
    """
    ensure_matplotlib()

    # ParticleTrace has a .time array that can be used as a built-in scalar
    scalars = dict(trace.scalars)
    scalars["time"] = trace.time

    values = _resolve_color_values(color, trace.points, scalars)

    if values is not None:
        artist = _colored_line_3d(
            ax, trace.points, values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=linewidth, alpha=alpha,
        )
        if colorbar:
            cb_label = color if isinstance(color, str) else ""
            _add_3d_colorbar(ax, artist, cb_label, colorbar, extremes=extremes)
        return artist

    pts = trace.points
    lines = ax.plot(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color=color, linewidth=linewidth, alpha=alpha, label=label,
        **kwargs,
    )
    return lines[0]


def _resolve_color_values(
    color: str | FloatArray | None,
    points: FloatArray,
    scalars: Mapping[str, FloatArray],
) -> FloatArray | None:
    """Resolve *color* to a scalar array, or None for uniform coloring."""
    if color is None:
        return None

    if isinstance(color, np.ndarray):
        if color.shape == (points.shape[0],):
            return color
        msg = (
            f"color array has shape {color.shape}, "
            f"expected ({points.shape[0]},)"
        )
        raise ValueError(msg)

    if isinstance(color, str) and color in scalars:
        return scalars[color]

    # It's a color string like "white" or "#ff0000" — uniform mode
    return None


def _add_3d_colorbar(
    ax: Axes,
    mappable: object,
    label: str,
    mode: bool | Literal["inset"],
    extremes: str = "darken",
) -> None:
    """Attach a colorbar to a 3D axes."""
    from pypic.plotting._colorbar import _apply_extremes

    _apply_extremes(mappable, mode=extremes)  # type: ignore[arg-type]

    fig = ax.get_figure()
    if mode == "inset":
        from pypic.plotting._colorbar import add_inset_colorbar

        add_inset_colorbar(ax, mappable, label)
    elif fig is not None:
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=label)  # type: ignore[arg-type]
