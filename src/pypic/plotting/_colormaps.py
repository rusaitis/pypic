"""Colormap selection: auto-detect signed vs positive-definite fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.fields import FieldInfo
    from pypic.plotting.styles import PlotTheme
    from pypic.types import FloatArray

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
        return cmap
    if is_positive_definite(name, data, info):
        return theme.sequential_cmap
    return theme.diverging_cmap


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
    return (-absmax, absmax)
