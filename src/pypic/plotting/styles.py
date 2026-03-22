"""Plot themes: PlotTheme dataclass, LIGHT/DARK presets, use_theme() context manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

_COMMON_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (7.0, 5.0),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "image.origin": "lower",
    "image.interpolation": "none",
    "axes.grid": False,
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
    """

    name: str
    rcparams: dict[str, Any]
    sequential_cmap: str
    diverging_cmap: str


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
        "xtick.color": "0.1",
        "ytick.color": "0.1",
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
)

DARK = PlotTheme(
    name="dark",
    rcparams={
        **_COMMON_RC,
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#1e1e1e",
        "savefig.facecolor": "#1e1e1e",
        "text.color": "#cccccc",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
    },
    sequential_cmap="inferno",
    diverging_cmap="RdBu_r",
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
