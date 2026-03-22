"""Plot themes: PlotTheme dataclass, LIGHT/DARK presets, use_theme() context manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cycler import cycler

from pypic.plotting._badge import OVERLAY_BORDER_PAD

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_COMMON_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (7.0, 5.0),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.15,
    "legend.edgecolor": "none",
    "legend.fontsize": 10,
    "image.origin": "lower",
    "image.interpolation": "none",
    "axes.grid": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes3d.mouserotationstyle": "azel",
}


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """Immutable collection of matplotlib styling for pypic plots.

    Parameters
    ----------
    name : str
        Human-readable theme name.
    rcparams : dict[str, Any]
        Full set of matplotlib rcParams to apply.
    sequential_cmap : str
        Default colormap for positive-definite fields.
    diverging_cmap : str
        Default colormap for signed fields.
    grid_color : str
        Color for grid lines (``"0.0"`` black, ``"1.0"`` white).
    """

    name: str
    rcparams: dict[str, Any]
    sequential_cmap: str
    diverging_cmap: str
    grid_color: str


LIGHT = PlotTheme(
    name="light",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "0.1",
        "axes.edgecolor": "0.1",
        "axes.labelcolor": "0.1",
        "xtick.color": "0.4",
        "ytick.color": "0.4",
        "legend.facecolor": (0.0, 0.0, 0.0, 0.06),
        "legend.labelcolor": (0.15, 0.15, 0.15, 0.85),
        "legend.borderaxespad": OVERLAY_BORDER_PAD,
        "axes.prop_cycle": cycler("color", [
            "#1e66f5",  # blue
            "#d20f39",  # red
            "#40a02b",  # green
            "#fe640b",  # orange
            "#8839ef",  # purple
            "#179299",  # teal
            "#e64553",  # maroon
            "#df8e1d",  # yellow
        ]),
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="0.0",
)

DARK = PlotTheme(
    name="dark",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#1e1e1e",
        "savefig.facecolor": "#1e1e1e",
        "text.color": "#e0e0e0",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "legend.facecolor": (1.0, 1.0, 1.0, 0.08),
        "legend.labelcolor": (0.8, 0.8, 0.8, 0.85),
        "legend.borderaxespad": OVERLAY_BORDER_PAD,
        "axes.prop_cycle": cycler("color", [
            "#7cb7ff",  # blue
            "#f47067",  # red
            "#96e072",  # green
            "#f39c12",  # orange
            "#c74ded",  # purple
            "#00e8c6",  # cyan
            "#ff8b6a",  # coral
            "#ffe66d",  # yellow
        ]),
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="1.0",
)

CATPPUCCIN_MOCHA = PlotTheme(
    name="catppuccin-mocha",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "#1e1e2e",  # base
        "axes.facecolor": "#1e1e2e",
        "savefig.facecolor": "#1e1e2e",
        "text.color": "#cdd6f4",  # text
        "axes.edgecolor": "#cdd6f4",
        "axes.labelcolor": "#bac2de",  # subtext1
        "xtick.color": "#a6adc8",  # subtext0
        "ytick.color": "#a6adc8",
        "legend.facecolor": (0.19, 0.20, 0.27, 0.15),  # surface0
        "legend.labelcolor": (0.80, 0.84, 0.96, 0.85),  # text
        "legend.borderaxespad": OVERLAY_BORDER_PAD,
        "axes.prop_cycle": cycler("color", [
            "#89b4fa",  # blue
            "#f38ba8",  # red
            "#a6e3a1",  # green
            "#fab387",  # peach
            "#cba6f7",  # mauve
            "#94e2d5",  # teal
            "#eba0ac",  # maroon
            "#f9e2af",  # yellow
        ]),
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="#cdd6f4",
)

ANUPPUCCIN_LIGHT = PlotTheme(
    name="anuppuccin-light",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "#eff1f5",  # base
        "axes.facecolor": "#eff1f5",
        "savefig.facecolor": "#eff1f5",
        "text.color": "#4c4f69",  # text
        "axes.edgecolor": "#4c4f69",
        "axes.labelcolor": "#5c5f77",  # subtext1
        "xtick.color": "#6c6f85",  # subtext0
        "ytick.color": "#6c6f85",
        "legend.facecolor": (0.86, 0.88, 0.91, 0.15),  # mantle
        "legend.labelcolor": (0.30, 0.31, 0.41, 0.85),  # text
        "legend.borderaxespad": OVERLAY_BORDER_PAD,
        "axes.prop_cycle": cycler("color", [
            "#1e66f5",  # blue
            "#d20f39",  # red
            "#40a02b",  # green
            "#fe640b",  # peach
            "#8839ef",  # mauve
            "#179299",  # teal
            "#e64553",  # maroon
            "#df8e1d",  # yellow
        ]),
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="#4c4f69",
)

ANDROMEDA = PlotTheme(
    name="andromeda",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "#23262e",  # editor bg
        "axes.facecolor": "#23262e",
        "savefig.facecolor": "#23262e",
        "text.color": "#d5ced9",  # primary text
        "axes.edgecolor": "#d5ced9",
        "axes.labelcolor": "#d5ced9",
        "xtick.color": "#9a919c",  # dimmed text
        "ytick.color": "#9a919c",
        "legend.facecolor": (0.13, 0.14, 0.16, 0.15),  # widget bg
        "legend.labelcolor": (0.84, 0.81, 0.85, 0.85),  # text
        "legend.borderaxespad": OVERLAY_BORDER_PAD,
        "axes.prop_cycle": cycler("color", [
            "#00e8c6",  # cyan (signature)
            "#7cb7ff",  # blue
            "#f92672",  # pink
            "#ffe66d",  # yellow
            "#c74ded",  # purple
            "#96e072",  # green
            "#f39c12",  # orange
            "#ee5d43",  # red
        ]),
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="#d5ced9",
)

DEFAULT = LIGHT


@contextmanager
def use_theme(theme: PlotTheme) -> Generator[None]:
    """Temporarily apply *theme* rcParams, restoring originals on exit."""
    import matplotlib as mpl

    old = {k: mpl.rcParams[k] for k in theme.rcparams if k in mpl.rcParams}
    mpl.rcParams.update(theme.rcparams)
    try:
        yield
    finally:
        mpl.rcParams.update(old)


def apply_theme_to_figure(fig: Figure, theme: PlotTheme) -> None:
    """Set figure facecolors, text colors, and tick colors to match *theme*.

    Useful when the figure was created outside ``use_theme()``.  Also
    restyles any existing colorbar axes so that tick labels, axis labels,
    and outlines pick up the theme palette.
    """
    fc = theme.rcparams.get("figure.facecolor", "white")
    fig.set_facecolor(fc)
    afc = theme.rcparams.get("axes.facecolor", "white")
    tc = theme.rcparams.get("text.color", "black")
    tick_c = theme.rcparams.get("xtick.color", "0.4")
    label_c = theme.rcparams.get("axes.labelcolor", tc)

    for text in fig.texts:
        text.set_color(tc)

    for ax in fig.get_axes():
        ax.set_facecolor(afc)
        ax.title.set_color(tc)
        ax.xaxis.label.set_color(label_c)
        ax.yaxis.label.set_color(label_c)
        ax.tick_params(colors=tick_c, labelcolor=tick_c)


def apply_grid(ax: Axes, theme: PlotTheme, *, minor: bool = False) -> None:
    """Enable subtle grid lines styled for *theme*.

    Parameters
    ----------
    ax : Axes
        Target axes.
    theme : PlotTheme
        Theme providing ``grid_color``.
    minor : bool
        If ``True``, also draw minor grid lines.
    """
    ax.grid(which="major", linewidth=0.5, alpha=0.08, color=theme.grid_color)
    if minor:
        ax.minorticks_on()
        ax.grid(which="minor", linewidth=0.3, alpha=0.05, color=theme.grid_color)


def style_legend(ax: Axes) -> None:
    """Round the legend box corners if a legend is present."""
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.6")


def style_3d_axes(
    ax: Axes,
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis_labels: tuple[str, str, str] = ("$x$", "$y$", "$z$"),
    axis_length: float | None = None,
    grid_z: float | None = None,
    grid_count: int = 5,
    coord_units: str = "",
) -> None:
    """Style a 3D axes with a clean equatorial grid and axis triad.

    Removes matplotlib's default 3D panes, spines, ticks, and grid,
    then draws:

    - **Equatorial grid**: subtle lines on the ``z = grid_z`` plane
      with coordinate labels at the ends (no tick marks or borders).
    - **Axis triad**: three short arrows from *center* along +x, +y,
      +z with labels at their tips.

    All colors are read from rcParams so that the result matches the
    active :class:`PlotTheme`.

    Parameters
    ----------
    ax : Axes
        A 3D ``Axes3D`` instance.
    center : tuple[float, float, float]
        Origin for the axis triad.
    axis_labels : tuple[str, str, str]
        Labels for the x, y, z arrows.
    axis_length : float or None
        Length of the triad arrows. ``None`` uses 15 %% of the axes range.
    grid_z : float or None
        Z-coordinate of the grid plane. ``None`` uses ``center[2]``.
    grid_count : int
        Number of grid lines per axis direction.
    coord_units : str
        Unit string appended to grid labels (e.g. ``"R_E"``).
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgba

    text_color = mpl.rcParams.get("text.color", "#e0e0e0")
    bg_color = mpl.rcParams.get("axes.facecolor", "#1e1e1e")
    tc_rgba = to_rgba(text_color)
    tc_rgb = tc_rgba[:3]

    # --- Strip default 3D chrome ---
    ax.set_facecolor(bg_color)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore[attr-defined]
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color("none")
        axis.set_ticks([])
        axis.set_ticklabels([])
        axis.label.set_text("")
        axis._axinfo["grid"]["color"] = "none"
    ax.grid(False)

    # --- Equatorial grid ---
    from matplotlib.ticker import MaxNLocator

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    z0 = grid_z if grid_z is not None else center[2]

    locator = MaxNLocator(nbins=grid_count)
    x_ticks = [v for v in locator.tick_values(xlim[0], xlim[1])
               if xlim[0] <= v <= xlim[1]]
    y_ticks = [v for v in locator.tick_values(ylim[0], ylim[1])
               if ylim[0] <= v <= ylim[1]]

    grid_alpha = 0.06
    grid_lw = 0.5
    label_alpha = 0.5
    label_size = 8

    # Lines parallel to y
    for x in x_ticks:
        ax.plot(
            [x, x], [ylim[0], ylim[1]], [z0, z0],
            color=tc_rgb, alpha=grid_alpha, linewidth=grid_lw, zorder=0,
        )
    # Lines parallel to x
    for y in y_ticks:
        ax.plot(
            [xlim[0], xlim[1]], [y, y], [z0, z0],
            color=tc_rgb, alpha=grid_alpha, linewidth=grid_lw, zorder=0,
        )

    # --- Grid endpoint labels (skip near-zero, every other if crowded) ---
    offset_frac = 0.04
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    def _label_values(ticks: list[float]) -> list[float]:
        """Select tick values to label, skipping near-zero and thinning."""
        if len(ticks) < 2:
            return ticks
        step = abs(ticks[1] - ticks[0]) if len(ticks) > 1 else 1.0
        filtered = [v for v in ticks if abs(v) > 0.3 * step]
        return filtered[::2] if len(filtered) > 5 else filtered

    for j, x in enumerate(_label_values(x_ticks)):
        val = f"{x:g}"
        is_last = j == len(_label_values(x_ticks)) - 1
        if coord_units and is_last:
            val += f" {coord_units}"
        ax.text(
            x, ylim[0] - y_range * offset_frac, z0,
            val, color=(*tc_rgb, label_alpha),
            fontsize=label_size, ha="center", va="top", zorder=1,
        )

    for j, y in enumerate(_label_values(y_ticks)):
        val = f"{y:g}"
        is_last = j == len(_label_values(y_ticks)) - 1
        if coord_units and is_last:
            val += f" {coord_units}"
        ax.text(
            xlim[0] - x_range * offset_frac, y, z0,
            val, color=(*tc_rgb, label_alpha),
            fontsize=label_size, ha="right", va="center", zorder=1,
        )

    # --- Axis triad ---
    if axis_length is None:
        axis_length = 0.20 * min(x_range, y_range)

    arrow_alpha = 0.7
    arrow_lw = 1.0
    label_offset = 1.3  # label position as fraction of arrow length

    directions = [
        (axis_length, 0.0, 0.0),
        (0.0, axis_length, 0.0),
        (0.0, 0.0, axis_length),
    ]
    for (dx, dy, dz), lbl in zip(directions, axis_labels, strict=True):
        # Arrow shaft
        ax.plot(
            [center[0], center[0] + dx],
            [center[1], center[1] + dy],
            [center[2], center[2] + dz],
            color=tc_rgb, alpha=arrow_alpha, linewidth=arrow_lw,
            solid_capstyle="round", zorder=5,
        )
        # Label at tip
        ax.text(
            center[0] + dx * label_offset,
            center[1] + dy * label_offset,
            center[2] + dz * label_offset,
            lbl, color=(*tc_rgb, arrow_alpha),
            fontsize=10, ha="center", va="center",
            fontweight="bold", zorder=5,
        )
