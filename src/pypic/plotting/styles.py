"""Plot themes: PlotTheme dataclass, LIGHT/DARK presets, use_theme() context manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
        "legend.labelcolor": "0.25",
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
        "legend.labelcolor": "#cccccc",
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
    grid_color="1.0",
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
    """Set figure facecolors and text colors to match *theme*.

    Useful when the figure was created outside ``use_theme()``.
    """
    fc = theme.rcparams.get("figure.facecolor", "white")
    fig.set_facecolor(fc)
    afc = theme.rcparams.get("axes.facecolor", "white")
    tc = theme.rcparams.get("text.color", "black")
    for ax in fig.get_axes():
        ax.set_facecolor(afc)
    # Update any existing suptitle / figure-level text
    for text in fig.texts:
        text.set_color(tc)


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
