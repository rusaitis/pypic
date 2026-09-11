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

import dataclasses
import importlib.resources
import json
import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal, assert_never

from pypic.plotting.styles import _COMMON_RC, PlotTheme

_log = logging.getLogger(__name__)

RGBA = tuple[float, float, float, float]

type _Kind = Literal["rgba", "float", "str", "bool", "names"]

# (section, key, PlotTheme field, kind), in the order save_theme writes them.
# Fields missing here (the font and layout scale factors) are not part of the
# cross-tool file format and always take their PlotTheme default.
_THEME_FIELDS: tuple[tuple[str, str, str, _Kind], ...] = (
    ("colors", "text", "text_color", "rgba"),
    ("colors", "secondary_text", "secondary_text_color", "rgba"),
    ("colors", "grid", "grid_color", "rgba"),
    ("colors", "accent", "accent_color", "str"),
    ("colors", "overlay", "overlay_color", "rgba"),
    ("colors", "overlay_text", "overlay_text_color", "rgba"),
    ("colors", "overlay_alt", "overlay_alt_color", "rgba"),
    ("colors", "overlay_alt_text", "overlay_alt_text_color", "rgba"),
    ("colors", "overlay_border", "overlay_border_color", "rgba"),
    ("colors", "overlay_alt_border", "overlay_alt_border_color", "rgba"),
    ("colors", "track", "track_color", "rgba"),
    ("colors", "track_alt", "track_alt_color", "rgba"),
    ("colors.cycle", "values", "color_cycle", "names"),
    ("colormaps", "sequential", "sequential_cmaps", "names"),
    ("colormaps", "diverging", "diverging_cmaps", "names"),
    ("font", "family", "font_family", "names"),
    ("font", "title", "font_title", "float"),
    ("font", "label", "font_label", "float"),
    ("font", "tick", "font_tick", "float"),
    ("font", "overlay", "font_overlay", "float"),
    ("overlay", "rounding", "overlay_rounding", "float"),
    ("overlay", "padding", "overlay_padding", "float"),
    ("overlay", "margin", "overlay_margin", "float"),
    ("lines", "width", "line_width", "float"),
    ("lines", "arrow_size", "arrow_size", "float"),
    ("lines", "arrow_style", "arrow_style", "str"),
    ("axes", "x_color", "axis_x_color", "str"),
    ("axes", "y_color", "axis_y_color", "str"),
    ("axes", "z_color", "axis_z_color", "str"),
    ("axes", "arrows", "axis_arrows", "bool"),
    ("ticks", "direction", "tick_direction", "str"),
    ("ticks", "major_length", "tick_major_length", "float"),
    ("ticks", "major_width", "tick_major_width", "float"),
    ("ticks", "minor_length", "tick_minor_length", "float"),
    ("ticks", "minor_width", "tick_minor_width", "float"),
    ("grid", "major_width", "grid_major_width", "float"),
    ("grid", "minor_width", "grid_minor_width", "float"),
    ("grid", "style", "grid_style", "str"),
    ("colorbar", "width", "colorbar_width", "str"),
    ("colorbar", "outline_width", "colorbar_outline_width", "float"),
    ("colorbar", "tick_length", "colorbar_tick_length", "float"),
    ("colorbar", "pad", "colorbar_pad", "float"),
    ("progress_bar", "width", "progress_bar_width", "float"),
    ("progress_bar", "height", "progress_bar_height", "float"),
    ("progress_bar", "rounding", "progress_bar_rounding", "float"),
    ("plot", "rounding", "plot_rounding", "float"),
)

_DEFAULTS: dict[str, Any] = {f.name: f.default for f in dataclasses.fields(PlotTheme)}

# A hex string or an [r, g, b(, a)] list for colors; a bare string for a
# one-element list of names.
_TOML_TYPES: dict[_Kind, tuple[type, ...]] = {
    "rgba": (str, list),
    "float": (int, float),
    "str": (str,),
    "bool": (bool,),
    "names": (str, list),
}

_FACECOLOR_KEYS = ("figure.facecolor", "axes.facecolor", "savefig.facecolor")


def _parse_rgba(
    val: str | list[float],
    default_alpha: float = 1.0,
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
    r, g, b, *alpha = (float(c) for c in val)
    return (r, g, b, alpha[0] if alpha else default_alpha)


def _decode(kind: _Kind, value: Any, field: str) -> Any:  # noqa: ANN401 — TOML value
    match kind:
        case "rgba":
            return _parse_rgba(value, default_alpha=_DEFAULTS[field][3])
        case "float":
            return float(value)
        case "names":
            return (value,) if isinstance(value, str) else tuple(value)
        case "str" | "bool":
            return value
        case _ as unreachable:
            assert_never(unreachable)


def _toml_value(value: object) -> str:
    # JSON's strings, numbers, booleans and arrays are valid TOML, and repr
    # floats make save → load exact.
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


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
    r"""Load a `PlotTheme` from a TOML file.

    Only ``name`` and ``[colors]`` (with at least ``background`` and
    ``text``) are required.  All other keys fall back to `PlotTheme`
    defaults, and sections other tools own (``[webpic]``, ...) are
    ignored.

    Raises
    ------
    ValueError
        If a key holds the wrong TOML type (a string where a number or
        boolean belongs, ...).

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

    values: dict[str, Any] = {}
    for section, key, field, kind in _THEME_FIELDS:
        table = raw
        for part in section.split("."):
            table = table.get(part, {})
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, _TOML_TYPES[kind]) or (
            kind == "float" and isinstance(value, bool)
        ):
            msg = f"{path}: [{section}] {key} = {value!r} is not a valid {kind}"
            raise ValueError(msg)
        values[field] = _decode(kind, value, field)

    rc: dict[str, Any] = {**_COMMON_RC}
    background = raw.get("colors", {}).get("background")
    if background is not None:
        rc.update(dict.fromkeys(_FACECOLOR_KEYS, background))
    padding = values.get("overlay_padding", _DEFAULTS["overlay_padding"])
    rc["legend.borderaxespad"] = padding * 1.5
    return PlotTheme(name=raw.get("name", path.stem), rcparams=rc, **values)


def save_theme(theme: PlotTheme, path: str | Path) -> None:
    r"""Save a `PlotTheme` to a TOML file.

    The output uses the cross-project schema with RGBA color tuples.

    Parameters
    ----------
    theme : PlotTheme
        Theme to serialize.
    path : str or Path
        Output file path.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from pypic.plotting._theme_io import _bundled_theme_dir, load_theme, save_theme
    >>> theme = load_theme(_bundled_theme_dir() / "light.toml")
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     out = Path(tmp) / "light.toml"
    ...     save_theme(theme, out)
    ...     load_theme(out).name == theme.name
    True
    """
    background = theme.rcparams.get("figure.facecolor", "white")
    lines = [f"name = {_toml_value(theme.name)}"]
    section = ""
    for field_section, key, field, _kind in _THEME_FIELDS:
        if field_section != section:
            section = field_section
            lines += ["", f"[{section}]"]
            if section == "colors":
                lines.append(f"background = {_toml_value(background)}")
        lines.append(f"{key} = {_toml_value(getattr(theme, field))}")
    Path(path).write_text("\n".join(lines) + "\n")


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
