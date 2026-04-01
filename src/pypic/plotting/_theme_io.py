"""TOML-based theme loading and saving.

Theme files use a cross-project schema: universal visual identity at the
top level (``[colors]``, ``[font]``, ``[overlay]``, ``[grid]``), with
app-specific details under ``[pypic]``, ``[webpic]``, etc.  Each
application reads its own section and ignores the rest.

Theme lookup order:
1. User directory (``~/.config/pypic/themes/``, or ``$PYPIC_THEME_DIR``)
2. Bundled package themes (``pypic/plotting/themes/``)
3. Explicit file path (``theme="~/my/theme.toml"``)
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from pypic.plotting.styles import _COMMON_RC, PlotTheme

_log = logging.getLogger(__name__)

RGBA = tuple[float, float, float, float]


def _parse_rgba(
    val: str | list[float], default_alpha: float = 1.0,
) -> RGBA:
    """Parse a TOML color value to an RGBA tuple.

    Accepts a hex string (``"#1e1e1e"``, alpha defaults to
    *default_alpha*) or a 3- or 4-element list (``[r, g, b]`` or
    ``[r, g, b, a]``).
    """
    if isinstance(val, str):
        from matplotlib.colors import to_rgba

        r, g, b, _a = to_rgba(val)
        return (r, g, b, default_alpha)
    if len(val) == 3:
        return (val[0], val[1], val[2], default_alpha)
    return (val[0], val[1], val[2], val[3])


def _rgba_to_toml(c: RGBA) -> str:
    """Format an RGBA tuple as a TOML inline array."""
    return f"[{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {c[3]:.2f}]"


def _default_theme_dir() -> Path:
    """Return the default user theme directory.

    Respects ``$PYPIC_THEME_DIR`` if set, otherwise
    ``$XDG_CONFIG_HOME/pypic/themes`` (defaulting to
    ``~/.config/pypic/themes``).
    """
    env = os.environ.get("PYPIC_THEME_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "pypic" / "themes"


def _bundled_theme_dir() -> Path:
    """Return the path to the bundled package theme directory."""
    return Path(str(importlib.resources.files("pypic.plotting") / "themes"))


def _find_theme(name: str) -> Path:
    """Locate a theme TOML file by name.

    Searches the user directory first, then bundled package themes.

    Parameters
    ----------
    name : str
        Theme name (without ``.toml`` extension).

    Returns
    -------
    Path

    Raises
    ------
    KeyError
        If no theme file matches *name*.
    """
    filename = f"{name}.toml"
    user_path = _default_theme_dir() / filename
    if user_path.is_file():
        return user_path
    bundled_path = _bundled_theme_dir() / filename
    if bundled_path.is_file():
        return bundled_path
    available = sorted(
        {p.stem for p in _iter_theme_files()},
    )
    msg = f"Unknown theme {name!r}. Available: {', '.join(available)}"
    raise KeyError(msg)


def _iter_theme_files() -> list[Path]:
    """Return all theme TOML files from both directories (user first)."""
    seen: dict[str, Path] = {}
    # Bundled themes first (will be overridden by user themes)
    bundled = _bundled_theme_dir()
    if bundled.is_dir():
        for p in sorted(bundled.glob("*.toml")):
            seen[p.stem.lower()] = p
    # User themes override
    user = _default_theme_dir()
    if user.is_dir():
        for p in sorted(user.glob("*.toml")):
            seen[p.stem.lower()] = p
    return list(seen.values())


def load_theme(path: str | Path) -> PlotTheme:
    r"""Load a :class:`PlotTheme` from a TOML file.

    Only ``name`` and ``[colors]`` (with at least ``background`` and
    ``text``) are required.  All other sections fall back to
    :class:`PlotTheme` defaults.  App-specific settings live under
    ``[pypic]``; other sections (``[webpic]``, etc.) are ignored.

    Parameters
    ----------
    path : str or Path
        Path to the ``.toml`` theme file.

    Returns
    -------
    PlotTheme

    Examples
    --------
    >>> from pypic.plotting._theme_io import _bundled_theme_dir, load_theme
    >>> t = load_theme(_bundled_theme_dir() / "dark.toml")
    >>> t.name
    'dark'
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    name = raw.get("name", path.stem)

    # Build rcParams from _COMMON_RC + background color
    rc: dict[str, Any] = {**_COMMON_RC}
    colors = raw.get("colors", {})
    if "background" in colors:
        bg = colors["background"]
        for key in ("figure.facecolor", "axes.facecolor", "savefig.facecolor"):
            rc[key] = bg

    # Universal sections
    overlay = raw.get("overlay", {})
    font = raw.get("font", {})
    grid = raw.get("grid", {})
    ticks = raw.get("ticks", {})
    cmaps = raw.get("colormaps", {})
    lines = raw.get("lines", {})
    axes = raw.get("axes", {})

    overlay_padding = overlay.get("padding", 0.4)
    rc["legend.borderaxespad"] = overlay_padding * 1.5

    colorbar_sec = raw.get("colorbar", {})
    progress_sec = raw.get("progress_bar", {})
    plot_sec = raw.get("plot", {})

    # Color cycle
    cycle_section = colors.get("cycle", {})
    color_cycle = tuple(cycle_section.get("values", ()))

    # Font family: accept string or list
    _default_family = ("DejaVu Serif", "Computer Modern", "Times", "serif")
    raw_family = font.get("family", _default_family)
    font_family = (raw_family,) if isinstance(raw_family, str) else tuple(raw_family)

    # Colormaps: accept string or list
    raw_seq = cmaps.get("sequential", "inferno")
    seq_cmaps = (raw_seq,) if isinstance(raw_seq, str) else tuple(raw_seq)
    raw_div = cmaps.get("diverging", "RdBu_r")
    div_cmaps = (raw_div,) if isinstance(raw_div, str) else tuple(raw_div)

    # Parse RGBA colors with appropriate default alphas
    defaults = PlotTheme.__new__(PlotTheme)
    text_color = (
        _parse_rgba(colors["text"], 0.9) if "text" in colors else defaults.text_color
    )
    sec_text = (
        _parse_rgba(colors["secondary_text"], 0.8)
        if "secondary_text" in colors
        else defaults.secondary_text_color
    )
    grid_color = (
        _parse_rgba(colors["grid"], 0.08)
        if "grid" in colors
        else defaults.grid_color
    )
    overlay_color = (
        _parse_rgba(colors["overlay"], 0.65)
        if "overlay" in colors
        else defaults.overlay_color
    )
    overlay_text = (
        _parse_rgba(colors["overlay_text"], 0.8)
        if "overlay_text" in colors
        else defaults.overlay_text_color
    )
    overlay_alt = (
        _parse_rgba(colors["overlay_alt"], 0.55)
        if "overlay_alt" in colors
        else defaults.overlay_alt_color
    )
    overlay_alt_text = (
        _parse_rgba(colors["overlay_alt_text"], 0.8)
        if "overlay_alt_text" in colors
        else defaults.overlay_alt_text_color
    )
    overlay_border = (
        _parse_rgba(colors["overlay_border"], 0.2)
        if "overlay_border" in colors
        else defaults.overlay_border_color
    )
    overlay_alt_border = (
        _parse_rgba(colors["overlay_alt_border"], 0.3)
        if "overlay_alt_border" in colors
        else (0.5, 0.5, 0.5, 0.3)
    )
    track_color = (
        _parse_rgba(colors["track"], 0.3)
        if "track" in colors
        else defaults.track_color
    )
    track_alt_color = (
        _parse_rgba(colors["track_alt"], 0.4)
        if "track_alt" in colors
        else (0.5, 0.5, 0.5, 0.4)
    )

    return PlotTheme(
        name=name,
        rcparams=rc,
        text_color=text_color,
        secondary_text_color=sec_text,
        grid_color=grid_color,
        overlay_color=overlay_color,
        overlay_text_color=overlay_text,
        overlay_alt_color=overlay_alt,
        overlay_alt_text_color=overlay_alt_text,
        overlay_border_color=overlay_border,
        overlay_alt_border_color=overlay_alt_border,
        track_color=track_color,
        track_alt_color=track_alt_color,
        accent_color=colors.get("accent", "#e8913a"),
        color_cycle=color_cycle,
        sequential_cmaps=seq_cmaps,
        diverging_cmaps=div_cmaps,
        font_family=font_family,
        font_title=font.get("title", 12.0),
        font_label=font.get("label", 11.0),
        font_tick=font.get("tick", 10.0),
        font_overlay=font.get("overlay", 9.0),
        overlay_rounding=overlay.get("rounding", 0.6),
        overlay_padding=overlay_padding,
        overlay_margin=overlay.get("margin", 0.03),
        line_width=lines.get("width", 1.5),
        arrow_size=lines.get("arrow_size", 4.0),
        arrow_style=lines.get("arrow_style", "triangle"),
        axis_x_color=axes.get("x_color", "#d63031"),
        axis_y_color=axes.get("y_color", "#00b894"),
        axis_z_color=axes.get("z_color", "#0984e3"),
        axis_arrows=axes.get("arrows", True),
        tick_direction=ticks.get("direction", "in"),
        tick_major_length=ticks.get("major_length", 4.0),
        tick_major_width=ticks.get("major_width", 0.6),
        tick_minor_length=ticks.get("minor_length", 2.0),
        tick_minor_width=ticks.get("minor_width", 0.4),
        grid_major_width=grid.get("major_width", 0.5),
        grid_minor_width=grid.get("minor_width", 0.3),
        grid_style=grid.get("style", "solid"),
        colorbar_width=colorbar_sec.get("width", "4%"),
        colorbar_outline_width=colorbar_sec.get("outline_width", 0.3),
        colorbar_tick_length=colorbar_sec.get("tick_length", 2.0),
        colorbar_pad=colorbar_sec.get("pad", 0.05),
        progress_bar_width=progress_sec.get("width", 80.0),
        progress_bar_height=progress_sec.get("height", 4.0),
        progress_bar_rounding=progress_sec.get("rounding", 2.0),
        plot_rounding=plot_sec.get("rounding", 0.0),
    )


def save_theme(theme: PlotTheme, path: str | Path) -> None:
    r"""Save a :class:`PlotTheme` to a TOML file.

    The output uses the cross-project schema with RGBA color tuples.

    Parameters
    ----------
    theme : PlotTheme
        Theme to serialize.
    path : str or Path
        Output file path.

    Examples
    --------
    >>> from pypic.plotting._theme_io import _bundled_theme_dir, load_theme, save_theme
    >>> t = load_theme(_bundled_theme_dir() / "light.toml")
    >>> save_theme(t, "/tmp/_test_light.toml")
    """
    path = Path(path)
    lines: list[str] = [f'name = "{theme.name}"', ""]

    # [colors]
    lines.append("[colors]")
    bg = theme.rcparams.get("figure.facecolor", "white")
    lines.append(f'background = "{bg}"')
    lines.append(f"text = {_rgba_to_toml(theme.text_color)}")
    lines.append(f"secondary_text = {_rgba_to_toml(theme.secondary_text_color)}")
    lines.append(f"grid = {_rgba_to_toml(theme.grid_color)}")
    lines.append(f'accent = "{theme.accent_color}"')
    lines.append(f"overlay = {_rgba_to_toml(theme.overlay_color)}")
    lines.append(f"overlay_text = {_rgba_to_toml(theme.overlay_text_color)}")
    lines.append(f"overlay_alt = {_rgba_to_toml(theme.overlay_alt_color)}")
    lines.append(
        f"overlay_alt_text = {_rgba_to_toml(theme.overlay_alt_text_color)}"
    )
    lines.append(f"overlay_border = {_rgba_to_toml(theme.overlay_border_color)}")
    lines.append(
        f"overlay_alt_border = {_rgba_to_toml(theme.overlay_alt_border_color)}"
    )
    lines.append(f"track = {_rgba_to_toml(theme.track_color)}")
    lines.append(f"track_alt = {_rgba_to_toml(theme.track_alt_color)}")

    if theme.color_cycle:
        lines.append("")
        lines.append("[colors.cycle]")
        cycle_strs = ", ".join(f'"{c}"' for c in theme.color_cycle)
        lines.append(f"values = [{cycle_strs}]")

    # [colormaps]
    lines.extend(["", "[colormaps]"])
    if len(theme.sequential_cmaps) == 1:
        lines.append(f'sequential = "{theme.sequential_cmap}"')
    else:
        seq_strs = ", ".join(f'"{c}"' for c in theme.sequential_cmaps)
        lines.append(f"sequential = [{seq_strs}]")
    if len(theme.diverging_cmaps) == 1:
        lines.append(f'diverging = "{theme.diverging_cmap}"')
    else:
        div_strs = ", ".join(f'"{c}"' for c in theme.diverging_cmaps)
        lines.append(f"diverging = [{div_strs}]")

    # [font]
    lines.extend(["", "[font]"])
    family_strs = ", ".join(f'"{f}"' for f in theme.font_family)
    lines.append(f"family = [{family_strs}]")
    lines.append(f"title = {theme.font_title}")
    lines.append(f"label = {theme.font_label}")
    lines.append(f"tick = {theme.font_tick}")
    lines.append(f"overlay = {theme.font_overlay}")

    # [overlay]
    lines.extend(["", "[overlay]"])
    lines.append(f"rounding = {theme.overlay_rounding}")
    lines.append(f"padding = {theme.overlay_padding}")
    lines.append(f"margin = {theme.overlay_margin}")

    # [lines]
    lines.extend(["", "[lines]"])
    lines.append(f"width = {theme.line_width}")
    lines.append(f"arrow_size = {theme.arrow_size}")
    lines.append(f'arrow_style = "{theme.arrow_style}"')

    # [axes]
    lines.extend(["", "[axes]"])
    lines.append(f'x_color = "{theme.axis_x_color}"')
    lines.append(f'y_color = "{theme.axis_y_color}"')
    lines.append(f'z_color = "{theme.axis_z_color}"')
    lines.append(f"arrows = {'true' if theme.axis_arrows else 'false'}")

    # [ticks]
    lines.extend(["", "[ticks]"])
    lines.append(f'direction = "{theme.tick_direction}"')
    lines.append(f"major_length = {theme.tick_major_length}")
    lines.append(f"major_width = {theme.tick_major_width}")
    lines.append(f"minor_length = {theme.tick_minor_length}")
    lines.append(f"minor_width = {theme.tick_minor_width}")

    # [grid]
    lines.extend(["", "[grid]"])
    lines.append(f"major_width = {theme.grid_major_width}")
    lines.append(f"minor_width = {theme.grid_minor_width}")
    lines.append(f'style = "{theme.grid_style}"')

    # [colorbar]
    lines.extend(["", "[colorbar]"])
    lines.append(f'width = "{theme.colorbar_width}"')
    lines.append(f"outline_width = {theme.colorbar_outline_width}")
    lines.append(f"tick_length = {theme.colorbar_tick_length}")
    lines.append(f"pad = {theme.colorbar_pad}")

    # [progress_bar]
    lines.extend(["", "[progress_bar]"])
    lines.append(f"width = {theme.progress_bar_width}")
    lines.append(f"height = {theme.progress_bar_height}")
    lines.append(f"rounding = {theme.progress_bar_rounding}")

    # [plot]
    if theme.plot_rounding > 0:
        lines.extend(["", "[plot]"])
        lines.append(f"rounding = {theme.plot_rounding}")

    lines.append("")  # trailing newline
    path.write_text("\n".join(lines))


def available_themes() -> dict[str, PlotTheme]:
    """Return all available themes from user and package directories.

    User themes override bundled themes of the same name.  Each call
    reads fresh from disk.

    Returns
    -------
    dict[str, PlotTheme]
        Mapping from theme name to loaded theme.
    """
    themes: dict[str, PlotTheme] = {}
    for path in _iter_theme_files():
        try:
            theme = load_theme(path)
        except Exception:
            _log.warning("failed to load theme from %s", path, exc_info=True)
            continue
        themes[path.stem.lower()] = theme
    return themes


def export_themes(
    directory: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy bundled themes to a directory for editing.

    Parameters
    ----------
    directory : str | Path | None
        Target directory.  ``None`` uses the default user theme
        directory (``~/.config/pypic/themes/``).
    overwrite : bool
        If ``False`` (default), skip files that already exist.

    Returns
    -------
    Path
        The directory themes were written to.
    """
    target = Path(directory) if directory is not None else _default_theme_dir()
    target.mkdir(parents=True, exist_ok=True)

    bundled = _bundled_theme_dir()
    for src in sorted(bundled.glob("*.toml")):
        dst = target / src.name
        if dst.exists() and not overwrite:
            _log.info("skipping existing %s", dst)
            continue
        shutil.copy2(src, dst)
        _log.info("wrote %s", dst)

    return target
