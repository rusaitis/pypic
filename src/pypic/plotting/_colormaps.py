"""Colormap selection: auto-detect signed vs positive-definite fields."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.colors import Colormap, Normalize

    from pypic.fields import FieldInfo
    from pypic.plotting.styles import PlotTheme
    from pypic.types import FloatArray


def _register_custom_colormaps() -> None:
    """Register pypic's custom colormaps with matplotlib (once)."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    # Blue-black-red diverging colormap for signed fields.
    # Gradient: cyan → blue → dark → black → dark → red → orange
    # https://eltos.github.io/gradient/#0:00BEEF-25:1967F3-45:1A356B-50:1A1719-55:662423-75:DB3832-100:DB9032
    _BKR_COLORS = (  # noqa: N806
        (0.000, (0.000, 0.745, 0.937)),
        (0.250, (0.098, 0.404, 0.953)),
        (0.450, (0.102, 0.208, 0.420)),
        (0.500, (0.102, 0.090, 0.098)),
        (0.550, (0.400, 0.141, 0.137)),
        (0.750, (0.859, 0.220, 0.196)),
        (1.000, (0.859, 0.565, 0.196)),
    )
    bkr = LinearSegmentedColormap.from_list("bkr", _BKR_COLORS)
    matplotlib.colormaps.register(bkr)
    matplotlib.colormaps.register(bkr.reversed(), name="bkr_r")


_register_custom_colormaps()

_POSITIVE_QUANTITY_TYPES = frozenset(
    {
        "density",
        "mass_density",
        "pressure",
        "temperature",
        "energy_density",
        "frequency",
        "length",
        "specific_energy",
    }
)

_POSITIVE_NAMES = frozenset(
    {
        "beta",
        "beta_e",
        "beta_i",
        "M_A",
        "M_ms",
        "sigma",
        "gamma_L",
        "gamma_eos",
        "agyrotropy",
    }
)


def is_positive_definite(
    name: str, data: FloatArray, info: FieldInfo | None = None
) -> bool:
    """Determine whether a field is positive-definite for colormap selection.

    Parameters
    ----------
    name : str
        Field name.
    data : FloatArray
        Field values (used as last-resort fallback).
    info : FieldInfo | None
        Field metadata, if available.

    Returns
    -------
    bool
    """
    if name.startswith("|") and name.endswith("|"):
        return True

    if info is not None and info.quantity_type in _POSITIVE_QUANTITY_TYPES:
        return True

    if name in _POSITIVE_NAMES:
        return True

    if not np.any(np.isfinite(data)):
        return False
    return bool(np.nanmin(data) >= 0)


def resolve_colormap(
    name: str,
    data: FloatArray,
    theme: PlotTheme,
    *,
    info: FieldInfo | None = None,
    cmap: str | None = None,
) -> str:
    """Choose the appropriate colormap for a field.

    Parameters
    ----------
    name : str
        Field name.
    data : FloatArray
        Field values.
    theme : PlotTheme
        Active theme (provides default colormaps).
    info : FieldInfo | None
        Field metadata, if available.
    cmap : str | None
        User override. If provided, returned directly.

    Returns
    -------
    str
        Colormap name.
    """
    if cmap is not None:
        import matplotlib.pyplot as plt

        if cmap not in plt.colormaps():
            msg = f"Unknown colormap {cmap!r}"
            raise ValueError(msg)
        return cmap
    if is_positive_definite(name, data, info):
        return theme.sequential_cmap
    return theme.diverging_cmap


def resolve_field_colormap(
    name: str,
    data: FloatArray,
    theme: PlotTheme,
    *,
    info: FieldInfo | None = None,
    cmap: str | Colormap | None = None,
) -> tuple[str, Colormap]:
    """Choose the colormap for a field, returning both name and Colormap object.

    Single source of truth for "given a field, pick the right colormap":
    matplotlib backends consume the string name, pyvista needs the
    `Colormap` object. Wraps
    `resolve_colormap` (which only returns the name) and looks up
    the matching Colormap so both backends route every overlay through
    one dispatcher with identical positive-definite detection.

    Parameters
    ----------
    name : str
        Field name (used for positive-definite detection).
    data : FloatArray
        Field values (used as last-resort fallback for sign detection).
    theme : PlotTheme
        Active theme (provides default sequential / diverging cmaps).
    info : FieldInfo | None
        Field metadata, if available.
    cmap : str, Colormap, or None
        User override. ``Colormap`` instances pass through unchanged
        (returned with their ``.name`` attribute). String names are
        looked up directly. ``None`` triggers auto-selection from the
        theme based on positive-definite detection.

    Returns
    -------
    tuple[str, Colormap]
        ``(cmap_name, colormap_object)``. The two are always consistent.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap as _Colormap

    # Pre-resolved Colormap object: pass through, surface its registered name.
    if isinstance(cmap, _Colormap):
        return (cmap.name, cmap)

    # String name (user override) or None (auto-detect): both routes go
    # through resolve_colormap so the auto-detection logic stays in one
    # place. The result is a name; look it up to get the Colormap.
    cmap_name = resolve_colormap(name, data, theme, info=info, cmap=cmap)
    return (cmap_name, plt.colormaps[cmap_name])


def symmetric_clim(data: FloatArray) -> tuple[float, float]:
    """Compute symmetric color limits centered on zero.

    Parameters
    ----------
    data : FloatArray
        Field values.

    Returns
    -------
    tuple[float, float]
        ``(-absmax, absmax)``
    """
    absmax = float(np.nanmax(np.abs(data)))
    if not np.isfinite(absmax) or absmax == 0.0:
        return (-1e-8, 1e-8)
    return (-absmax, absmax)


def resolve_norm(
    values: FloatArray,
    *,
    symmetric: bool,
    log_scale: bool = False,
    symlog: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    linthresh: float | None = None,
) -> Normalize:
    """Build the color normalization for a pcolormesh plot.

    *values* is everything the norm must cover, every panel that shares it.
    A missing limit comes from the finite data range, or both from
    `symmetric_clim` when *symmetric* is set and neither is given. Log
    scaling needs positive limits, so it skips non-positive data and a
    non-positive *vmin*, and it yields to *symmetric* with a warning.
    """
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    if log_scale and symmetric:
        warnings.warn(
            "log_scale=True ignored because symmetric color limits are active",
            stacklevel=3,
        )
        log_scale = False
    if log_scale:
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == 0:
            return Normalize()
        low = vmin if vmin is not None and vmin > 0 else float(positive.min())
        high = vmax if vmax is not None else float(positive.max())
        return LogNorm(vmin=low, vmax=high)

    if symmetric and vmin is None and vmax is None:
        vmin, vmax = symmetric_clim(values)
    else:
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            vmin = float(finite.min()) if vmin is None else vmin
            vmax = float(finite.max()) if vmax is None else vmax
    if symlog:
        threshold = linthresh if linthresh is not None else _auto_linthresh(values)
        return SymLogNorm(linthresh=threshold, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def round_nice(value: float) -> float:
    r"""Round a positive value to the nearest 1-2-5 $\times 10^n$ step.

    The 1-2-5 sequence (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)
    produces clean colorbar tick labels and avoids awkward intermediate
    values like 3 or 7.

    Parameters
    ----------
    value : float
        Positive value to round. Zero returns 0.

    Returns
    -------
    float

    Examples
    --------
    >>> round_nice(0.7)
    0.5
    >>> round_nice(3.5)
    5.0
    >>> round_nice(150.0)
    200.0
    """
    if value <= 0:
        return 0.0
    import math

    exponent = math.floor(math.log10(value))
    mantissa = value / 10**exponent
    candidates = (1.0, 2.0, 5.0, 10.0)
    # On tie, pick the larger candidate (wider color range is safer)
    best = min(candidates, key=lambda c: (abs(c - mantissa), -c))
    return float(best * 10**exponent)


def auto_clim(
    values: FloatArray,
    *,
    positive_definite: bool,
) -> tuple[float, float]:
    r"""Compute auto color limits using 3$\sigma$ from the median.

    Positive-definite fields get ``(0, round_nice(median + 3*sigma))``.
    Signed fields get ``(-spread, spread)`` where
    ``spread = round_nice(3*sigma)``, symmetric around zero so the
    colormap's zero-crossing aligns with physical zero.

    Parameters
    ----------
    values : FloatArray
        Field values (may contain NaN).
    positive_definite : bool
        Whether the field is positive-definite.

    Returns
    -------
    tuple[float, float]
        ``(vmin, vmax)``
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1e-8, 1e-8) if not positive_definite else (0.0, 1e-8)

    median = float(np.median(finite))
    sigma = float(np.std(finite))

    if sigma == 0.0:
        if positive_definite:
            return (0.0, round_nice(abs(median)) if median != 0 else 1e-8)
        absmax = abs(median) if median != 0 else 1e-8
        return (-round_nice(absmax), round_nice(absmax))

    if positive_definite:
        vmax = round_nice(median + 3.0 * sigma)
        return (0.0, vmax if vmax > 0 else 1e-8)

    spread = round_nice(3.0 * sigma)
    return (-spread, spread)


def _auto_linthresh(values: FloatArray) -> float:
    """Auto-detect the linear threshold for symlog normalization.

    Returns the median of the absolute nonzero finite values, which
    places the linear/log transition at the typical magnitude of the
    data — keeping the linear region around zero small enough to
    resolve sign changes while letting the log tails capture the
    dynamic range.

    Parameters
    ----------
    values : FloatArray
        Field values.

    Returns
    -------
    float
        Positive threshold. Falls back to 1e-8 if no valid data.
    """
    finite = values[np.isfinite(values)]
    nonzero = np.abs(finite[finite != 0])
    if nonzero.size == 0:
        return 1e-8
    return float(np.median(nonzero))
