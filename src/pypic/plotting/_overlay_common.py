"""Pure-function overlay helpers shared across plotting backends.

These helpers are deliberately backend-agnostic — no matplotlib or
pyvista imports at module load. The matplotlib `pypic.plotting._badge`
and pyvista `pypic.plotting.pyvista._badge` modules both import from
here so the WCAG luminance/contrast logic and the ``(color, alpha) → RGBA``
override pattern have a single canonical implementation.
"""

from __future__ import annotations

# Shared overlay visibility threshold — any alpha below this is treated
# as "no background".  Used by both matplotlib and pyvista overlay code.
ALPHA_VISIBLE = 0.01


def _lum(c: tuple[float, float, float]) -> float:
    """Relative luminance of an sRGB color per WCAG 2.x.

    Each channel is gamma-corrected before the standard
    ``0.2126 R + 0.7152 G + 0.0722 B`` weighting.
    """
    r, g, b = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(
    c1: tuple[float, float, float], c2: tuple[float, float, float]
) -> float:
    """WCAG relative-luminance contrast ratio between two RGB colors.

    Returns a value in ``[1, 21]`` where 1 is no contrast and 21 is the
    maximum (black on white). The two arguments are interchangeable —
    the larger luminance is always taken as the numerator.

    Examples
    --------
    >>> round(contrast_ratio((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 1)
    21.0
    >>> round(contrast_ratio((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 1)
    1.0
    """
    l1, l2 = _lum(c1), _lum(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def resolve_rgba_override(
    color: str | tuple[float, ...] | None,
    alpha: float | None,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Resolve an optional ``(color, alpha)`` user override against an RGBA fallback.

    Used by overlay drawing code to combine an explicit user color/alpha
    pair with a theme-derived default. The two backends route every
    overlay color decision through this helper so they handle ``None``
    overrides identically.

    - Both ``color`` and ``alpha`` ``None`` → return *fallback* unchanged.
    - Only ``color`` ``None`` → keep fallback's RGB, use *alpha* as opacity.
    - Only ``alpha`` ``None`` → use *color*'s RGB, keep fallback's opacity.
    - Both supplied → use *color*'s RGB, use *alpha* as opacity.

    *color* may be any matplotlib-style color (string name, hex, or
    tuple of 3+ floats). The matplotlib import is local to keep this
    module importable without matplotlib at the top level.

    Examples
    --------
    >>> resolve_rgba_override(None, None, (0.1, 0.2, 0.3, 0.65))
    (0.1, 0.2, 0.3, 0.65)
    >>> resolve_rgba_override(None, 0.8, (0.1, 0.2, 0.3, 0.65))
    (0.1, 0.2, 0.3, 0.8)
    """
    if color is None and alpha is None:
        return fallback
    if color is None:
        return (fallback[0], fallback[1], fallback[2], alpha)  # type: ignore[return-value]

    from matplotlib.colors import to_rgb

    rgb = to_rgb(color) if isinstance(color, str) else (color[0], color[1], color[2])
    a = alpha if alpha is not None else fallback[3]
    return (rgb[0], rgb[1], rgb[2], a)
