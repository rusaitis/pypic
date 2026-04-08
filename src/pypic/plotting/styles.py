"""Plot themes: PlotTheme dataclass, use_theme() context manager, file-based themes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_COMMON_RC: dict[str, Any] = {
    "font.serif": ["DejaVu Serif", "Computer Modern", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "figure.figsize": (7.0, 5.0),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
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
        Full set of matplotlib rcParams to apply (must not contain
        ``axes.prop_cycle`` — use *color_cycle* instead).
    sequential_cmaps : tuple[str, ...]
        Colormap preference list for positive-definite fields (first is default).
    diverging_cmaps : tuple[str, ...]
        Colormap preference list for signed fields (first is default).
    grid_color : str
        Color for grid lines (``"0.0"`` black, ``"1.0"`` white).
    color_cycle : tuple[str, ...]
        Hex color wheel for sequential visual elements (lines, scatter,
        categories). Built into ``axes.prop_cycle`` at theme-application
        time so that ``cycler`` is not required at import.
    """

    name: str
    rcparams: dict[str, Any]

    # Colors — RGBA tuples (r, g, b, a) with built-in opacity.
    # String colors (background, accent) are opaque and don't need alpha.
    text_color: tuple[float, float, float, float] = (0.88, 0.88, 0.88, 0.9)
    secondary_text_color: tuple[float, float, float, float] = (0.53, 0.53, 0.53, 0.8)
    grid_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.08)
    overlay_color: tuple[float, float, float, float] = (0.07, 0.07, 0.07, 0.65)
    overlay_text_color: tuple[float, float, float, float] = (0.88, 0.88, 0.88, 0.8)
    overlay_alt_color: tuple[float, float, float, float] = (0.12, 0.12, 0.12, 0.55)
    overlay_alt_text_color: tuple[float, float, float, float] = (0.88, 0.88, 0.88, 0.8)
    overlay_border_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.2)
    overlay_alt_border_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.3)
    track_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    track_alt_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.4)
    accent_color: str = "#e8913a"
    color_cycle: tuple[str, ...] = ()

    # Colormaps (preference lists; first is default)
    sequential_cmaps: tuple[str, ...] = ("inferno",)
    diverging_cmaps: tuple[str, ...] = ("RdBu_r",)

    # Font
    font_family: tuple[str, ...] = ("DejaVu Serif", "Computer Modern", "Times", "serif")
    font_title: float = 12.0
    font_label: float = 11.0
    font_tick: float = 10.0
    font_overlay: float = 9.0

    # Overlay (badges, legends, inset colorbars)
    overlay_rounding: float = 0.6
    overlay_padding: float = 0.4
    overlay_margin: float = 0.03

    # Lines & arrows
    line_width: float = 1.5
    arrow_size: float = 4.0
    arrow_style: str = "triangle"

    # Axes (3D triad)
    axis_x_color: str = "#d63031"
    axis_y_color: str = "#00b894"
    axis_z_color: str = "#0984e3"
    axis_arrows: bool = True

    # Ticks
    tick_direction: str = "in"           # "in", "out", or "inout"
    tick_major_length: float = 4.0
    tick_major_width: float = 0.6
    tick_minor_length: float = 2.0
    tick_minor_width: float = 0.4

    # Grid
    grid_major_width: float = 0.5
    grid_minor_width: float = 0.3
    grid_style: str = "solid"

    # Colorbar
    colorbar_width: str = "4%"
    colorbar_outline_width: float = 0.3
    colorbar_tick_length: float = 2.0
    colorbar_pad: float = 0.05
    colorbar_title_font_scale: float = 1.4
    colorbar_tick_font_scale: float = 1.2

    # Progress bar
    progress_bar_width: float = 80.0
    progress_bar_height: float = 4.0
    progress_bar_rounding: float = 2.0
    badge_font_scale: float = 2.2

    # Pyvista axes (3D triad and equatorial grid)
    axis_triad_font_scale: float = 3.0
    grid_label_font_scale: float = 2.8

    # Multi-panel grid layout
    figsize_per_col: float = 4.5
    figsize_per_row: float = 4.0
    # single-row grids — labels can stay small
    panel_label_scale_sparse: float = 1.55
    # multi-row grids — bump labels for legibility
    panel_label_scale_dense: float = 1.78

    # Contour overlay
    contour_label_fontsize: float = 7.0

    # Data-space annotations (reference circles, error labels, etc.)
    annotation_fontsize: float = 7.5

    # Plot area
    plot_rounding: float = 0.0

    @property
    def sequential_cmap(self) -> str:
        """Default sequential colormap (first in preference list)."""
        return self.sequential_cmaps[0]

    @property
    def diverging_cmap(self) -> str:
        """Default diverging colormap (first in preference list)."""
        return self.diverging_cmaps[0]

    def customize(self, **overrides: Any) -> PlotTheme:  # noqa: ANN401
        r"""Return a new theme with selected fields overridden.

        Pass any :class:`PlotTheme` field name as a keyword argument.
        Unknown keys with underscores are converted to matplotlib
        rcParam keys and merged into ``rcparams``
        (``figure_dpi=200`` → ``"figure.dpi": 200``).

        Returns
        -------
        PlotTheme

        Examples
        --------
        >>> t = get_theme()
        >>> big = t.customize(font_title=16.0)
        >>> big.font_title
        16.0
        >>> wide = t.customize(figsize_per_col=6.0)
        >>> wide.figsize_per_col
        6.0
        """
        import copy
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(self)}
        theme_kw: dict[str, Any] = {}
        rc_kw: dict[str, Any] = {}

        for key, value in overrides.items():
            if key in field_names:
                theme_kw[key] = value
            else:
                rc_kw[key.replace("_", ".")] = value

        if rc_kw:
            theme_kw["rcparams"] = {**self.rcparams, **rc_kw}

        return copy.replace(self, **theme_kw)


_DEFAULT_THEME_NAME = "light"
"""Name of the theme loaded as default when no ``set_theme()`` has been called."""

DEFAULT: PlotTheme | None = None
"""Current default theme.  Loaded lazily on first access."""

ThemeArg = PlotTheme | str | None
"""Accepted type for the ``theme`` parameter across all plotting functions."""


def _resolve_theme_arg(theme: ThemeArg) -> PlotTheme:
    """Normalize a theme argument to a PlotTheme instance.

    Accepts a :class:`PlotTheme` object, a theme name (looked up from
    the user and package theme directories), a ``.toml`` file path, or
    ``None`` (falls back to the current default).
    """
    if theme is None:
        return get_theme()
    if isinstance(theme, PlotTheme):
        return theme
    from pathlib import Path

    from pypic.plotting._theme_io import _find_theme, load_theme

    if theme.endswith(".toml"):
        return load_theme(Path(theme).expanduser())
    return load_theme(_find_theme(theme.lower()))


def set_theme(theme: ThemeArg) -> None:
    """Set the default theme for all pypic plot functions.

    Parameters
    ----------
    theme : ThemeArg
        Theme name, path, or object.

    Examples
    --------
    >>> set_theme("dark")
    >>> get_theme().name
    'dark'
    >>> set_theme("light")
    """
    global DEFAULT
    DEFAULT = _resolve_theme_arg(theme)


def get_theme() -> PlotTheme:
    """Return the current default theme.

    Loads ``light`` from the theme directory on first call if no
    default has been set.

    Examples
    --------
    >>> get_theme().name
    'light'
    """
    global DEFAULT
    if DEFAULT is None:
        from pypic.plotting._theme_io import _find_theme, load_theme

        DEFAULT = load_theme(_find_theme(_DEFAULT_THEME_NAME))
    return DEFAULT


_active_theme: PlotTheme | None = None


def get_active_theme() -> PlotTheme | None:
    """Return the theme set by the innermost :func:`use_theme` context, or ``None``."""
    return _active_theme


def _theme_val(attr: str, default: Any) -> Any:  # noqa: ANN401
    """Read *attr* from the active or global default theme.

    Falls back to *default* only when no theme has been loaded at all.
    """
    theme = _active_theme if _active_theme is not None else get_theme()
    return getattr(theme, attr, default)


_GENERIC_FAMILIES = frozenset({
    "serif", "sans-serif", "monospace", "cursive", "fantasy",
})


def _available_fonts(families: tuple[str, ...]) -> list[str]:
    """Filter *families* to those installed, keeping generic family names."""
    from matplotlib.font_manager import fontManager

    installed = {f.name for f in fontManager.ttflist}
    result = [f for f in families if f in installed or f in _GENERIC_FAMILIES]
    return result or ["sans-serif"]


@contextmanager
def use_theme(theme: ThemeArg) -> Generator[None]:
    """Temporarily apply *theme* rcParams, restoring originals on exit.

    Accepts a :class:`PlotTheme`, a theme name string, a ``.toml``
    file path, or ``None`` (uses the current default).
    """
    global _active_theme
    import matplotlib as mpl

    resolved = _resolve_theme_arg(theme)
    rc = dict(resolved.rcparams)

    # Inject font family and sizes from theme fields.
    # Filter to available fonts to suppress matplotlib findfont warnings.
    rc["font.family"] = _available_fonts(resolved.font_family)
    rc["font.size"] = resolved.font_label
    rc["axes.titlesize"] = resolved.font_title
    rc["axes.labelsize"] = resolved.font_label
    rc["xtick.labelsize"] = resolved.font_tick
    rc["ytick.labelsize"] = resolved.font_tick
    rc["legend.fontsize"] = resolved.font_overlay

    # Tick geometry
    rc["xtick.direction"] = resolved.tick_direction
    rc["ytick.direction"] = resolved.tick_direction
    rc["xtick.major.size"] = resolved.tick_major_length
    rc["ytick.major.size"] = resolved.tick_major_length
    rc["xtick.major.width"] = resolved.tick_major_width
    rc["ytick.major.width"] = resolved.tick_major_width
    rc["xtick.minor.size"] = resolved.tick_minor_length
    rc["ytick.minor.size"] = resolved.tick_minor_length
    rc["xtick.minor.width"] = resolved.tick_minor_width
    rc["ytick.minor.width"] = resolved.tick_minor_width

    if resolved.color_cycle:
        from cycler import cycler

        rc["axes.prop_cycle"] = cycler("color", list(resolved.color_cycle))

    # Inject RGBA colors from theme into rcParams
    rc["text.color"] = resolved.text_color
    rc["axes.edgecolor"] = resolved.text_color[:3]
    rc["axes.labelcolor"] = resolved.text_color
    rc["xtick.color"] = resolved.secondary_text_color
    rc["ytick.color"] = resolved.secondary_text_color
    rc["legend.facecolor"] = resolved.overlay_color
    rc["legend.labelcolor"] = resolved.secondary_text_color

    old = {k: mpl.rcParams[k] for k in rc if k in mpl.rcParams}
    # Always capture prop_cycle so themes without color_cycle restore it
    if "axes.prop_cycle" not in old:
        old["axes.prop_cycle"] = mpl.rcParams["axes.prop_cycle"]
    prev_theme = _active_theme
    _active_theme = resolved
    mpl.rcParams.update(rc)
    try:
        yield
    finally:
        _active_theme = prev_theme
        mpl.rcParams.update(old)


def apply_theme_to_figure(fig: Figure, theme: PlotTheme) -> None:
    """Set figure facecolors, text colors, and tick colors to match *theme*.

    Useful when the figure was created outside ``use_theme()``.  Applies
    background, text, tick, spine, and legend colors to all axes on the
    figure.
    """
    fc = theme.rcparams.get("figure.facecolor", "white")
    fig.set_facecolor(fc)
    afc = theme.rcparams.get("axes.facecolor", "white")

    tc = theme.text_color
    label_c = theme.text_color
    tick_c = theme.secondary_text_color
    legend_label_c = theme.secondary_text_color

    for text in fig.texts:
        text.set_color(tc)

    for ax in fig.get_axes():
        ax.set_facecolor(afc)
        ax.title.set_color(tc)
        ax.xaxis.label.set_color(label_c)
        ax.yaxis.label.set_color(label_c)
        ax.tick_params(colors=tick_c, labelcolor=tick_c)
        for spine_name in ("left", "right", "top", "bottom"):
            visible = theme.rcparams.get(f"axes.spines.{spine_name}", True)
            ax.spines[spine_name].set_visible(visible)
            if visible:
                edge_c = theme.rcparams.get("axes.edgecolor", "black")
                ax.spines[spine_name].set_edgecolor(edge_c)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(legend_label_c)


def bake_theme(ax: Axes, theme: ThemeArg = None) -> None:
    """Bake theme font sizes and tick geometry onto *ax*.

    Ensures theme settings persist after the ``use_theme()`` context
    manager exits (which restores rcParams to their previous values).
    """
    resolved = _resolve_theme_arg(theme)
    ax.tick_params(
        labelsize=resolved.font_tick,
        direction=resolved.tick_direction,
        which="major",
        length=resolved.tick_major_length,
        width=resolved.tick_major_width,
    )
    ax.tick_params(
        which="minor",
        direction=resolved.tick_direction,
        length=resolved.tick_minor_length,
        width=resolved.tick_minor_width,
    )
    ax.xaxis.label.set_fontsize(resolved.font_label)
    ax.yaxis.label.set_fontsize(resolved.font_label)
    if ax.get_title():
        ax.title.set_fontsize(resolved.font_title)


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
    gc = theme.grid_color
    ax.grid(
        which="major",
        linewidth=theme.grid_major_width,
        linestyle=theme.grid_style,
        alpha=gc[3],
        color=gc[:3],
    )
    if minor:
        ax.minorticks_on()
        ax.grid(
            which="minor",
            linewidth=theme.grid_minor_width,
            linestyle=theme.grid_style,
            alpha=gc[3],
            color=gc[:3],
        )


def _rounded_axes_path(aspect: float, r: float) -> object:
    """Build a rounded rectangle Path with aspect-corrected circular arcs."""
    from matplotlib.path import Path

    if aspect >= 1:
        rx, ry = min(r / aspect, 0.5), min(r, 0.5)
    else:
        rx, ry = min(r, 0.5), min(r * aspect, 0.5)

    kx, ky = 0.5523 * rx, 0.5523 * ry
    c4 = Path.CURVE4
    return Path(
        [
            (rx, 0), (1 - rx, 0),
            (1 - rx + kx, 0), (1, ry - ky), (1, ry),
            (1, 1 - ry),
            (1, 1 - ry + ky), (1 - rx + kx, 1), (1 - rx, 1),
            (rx, 1),
            (rx - kx, 1), (0, 1 - ry + ky), (0, 1 - ry),
            (0, ry),
            (0, ry - ky), (rx - kx, 0), (rx, 0),
            (rx, 0),
        ],
        [
            Path.MOVETO, Path.LINETO,
            c4, c4, c4, Path.LINETO,
            c4, c4, c4, Path.LINETO,
            c4, c4, c4, Path.LINETO,
            c4, c4, c4, Path.CLOSEPOLY,
        ],
    )


def apply_rounding(ax: Axes) -> None:
    """Clip axes content to a rounded rectangle (no-op when ``plot_rounding == 0``)."""
    rounding: float = _theme_val("plot_rounding", 0.0)
    if rounding <= 0:
        return
    from matplotlib.patches import PathPatch

    fig = ax.get_figure()
    if fig is None:
        return
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0

    path = _rounded_axes_path(aspect, rounding)
    clip = PathPatch(path, transform=ax.transAxes, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    for child in ax.get_children():
        if child is not clip:
            child.set_clip_path(clip)


def _overlay_box_style(theme: PlotTheme | None = None) -> str:
    """Build overlay box style string from theme or defaults."""
    if theme is None:
        theme = get_active_theme() or get_theme()
    rounding = theme.overlay_rounding
    if rounding <= 0:
        return "square,pad=0"
    return f"round,pad=0,rounding_size={rounding}"


def style_legend(ax: Axes) -> None:
    """Round the legend box corners if a legend is present."""
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_boxstyle(_overlay_box_style())


