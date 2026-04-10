"""Sample gridded FieldDataset values at arbitrary 3D positions."""

from __future__ import annotations

__all__ = [
    "attach_scalars",
    "attach_scalars_to_trace",
    "sample_field",
    "sample_fields",
]

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pypic.readers._field_dataset import FieldDataset
    from pypic.traces._fieldline import FieldLine
    from pypic.traces._particletrace import ParticleTrace
    from pypic.types import FloatArray


def _nearest_indices(
    coord_arrays: tuple[Any, ...],
    points: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute nearest-neighbor grid indices and an in-bounds mask.

    Returns
    -------
    tuple
        ``(indices, mask)`` where ``indices`` has shape ``(N, ndim)``
        (dtype intp) and ``mask`` has shape ``(N,)`` (dtype bool).
    """
    ndim = len(coord_arrays)
    n = points.shape[0]
    indices = np.empty((n, ndim), dtype=np.intp)
    mask = np.ones(n, dtype=bool)
    for d in range(ndim):
        coords = coord_arrays[d]
        vals = points[:, d]
        if len(coords) == 1:
            # Degenerate axis (single grid node, e.g. 2D-in-3D layouts).
            # Only valid index is 0; bounding-box uses a heuristic margin
            # since there is no spacing to derive a half-cell from.
            idx = np.zeros(n, dtype=np.intp)
            half_dx = 0.5 * abs(float(coords[0])) + 0.5
        else:
            idx = np.searchsorted(coords, vals) - 1
            idx = np.clip(idx, 0, len(coords) - 2)
            closer_to_next = np.abs(vals - coords[idx + 1]) < np.abs(vals - coords[idx])
            idx = np.where(closer_to_next, idx + 1, idx)
            half_dx = 0.5 * (coords[1] - coords[0])
        out = (vals < coords[0] - half_dx) | (vals > coords[-1] + half_dx)
        mask &= ~out
        indices[:, d] = idx
    return indices, mask


def sample_field(
    data: FieldDataset,
    points: FloatArray,
    field: str,
    *,
    method: str = "nearest",
) -> FloatArray:
    """Sample a scalar field from a FieldDataset at arbitrary 3D positions.

    Parameters
    ----------
    data : FieldDataset
        Gridded field data.
    points : FloatArray
        Query positions, shape ``(N, 3)``.
    field : str
        Field name (canonical, alias, or computable via ``compute()``).
    method : str
        Interpolation method: ``"nearest"`` (pure NumPy) or ``"linear"``
        (uses ``scipy.interpolate.RegularGridInterpolator``).

    Returns
    -------
    FloatArray
        Sampled values, shape ``(N,)``. NaN for points outside the domain.
    """
    values: np.ndarray[Any, Any] = data[field]
    coord_arrays = data.grid.coordinate_arrays()

    if method == "nearest":
        return _sample_nearest(coord_arrays, values, points)
    if method == "linear":
        return _sample_linear(coord_arrays, values, points)
    msg = f"Unknown interpolation method {method!r}. Use 'nearest' or 'linear'."
    raise ValueError(msg)


def _sample_nearest(
    coord_arrays: tuple[Any, ...],
    values: np.ndarray[Any, Any],
    points: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Nearest-neighbor sampling (pure NumPy)."""
    indices, mask = _nearest_indices(coord_arrays, points)
    result = np.full(points.shape[0], np.nan, dtype=np.float64)
    if np.any(mask):
        idx_tuple = tuple(indices[mask, d] for d in range(len(coord_arrays)))
        result[mask] = values[idx_tuple]
    return result


def _sample_linear(
    coord_arrays: tuple[Any, ...],
    values: np.ndarray[Any, Any],
    points: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Trilinear interpolation via scipy."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        coord_arrays,
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(points[:, : len(coord_arrays)])


def sample_fields(
    data: FieldDataset,
    points: FloatArray,
    fields: list[str],
    *,
    method: str = "nearest",
) -> dict[str, FloatArray]:
    """Sample multiple fields at the same positions.

    Parameters
    ----------
    data : FieldDataset
        Gridded field data.
    points : FloatArray
        Query positions, shape ``(N, 3)``.
    fields : list[str]
        Field names to sample.
    method : str
        Interpolation method.

    Returns
    -------
    dict[str, FloatArray]
        Field name → sampled values.
    """
    return {f: sample_field(data, points, f, method=method) for f in fields}


def attach_scalars(
    field_line: FieldLine,
    data: FieldDataset,
    fields: list[str],
    *,
    method: str = "nearest",
) -> FieldLine:
    """Return a new FieldLine with sampled scalars merged in.

    Parameters
    ----------
    field_line : FieldLine
        Input field line.
    data : FieldDataset
        Gridded data to sample from.
    fields : list[str]
        Field names to sample and attach.
    method : str
        Interpolation method.

    Returns
    -------
    FieldLine
        New instance with additional scalars.
    """
    sampled = sample_fields(data, field_line.points, fields, method=method)
    merged = dict(field_line.scalars)
    merged.update(sampled)
    return copy.replace(field_line, scalars=merged)


def attach_scalars_to_trace(
    trace: ParticleTrace,
    data: FieldDataset,
    fields: list[str],
    *,
    method: str = "nearest",
) -> ParticleTrace:
    """Return a new ParticleTrace with sampled scalars merged in.

    Parameters
    ----------
    trace : ParticleTrace
        Input particle trace.
    data : FieldDataset
        Gridded data to sample from.
    fields : list[str]
        Field names to sample and attach.
    method : str
        Interpolation method.

    Returns
    -------
    ParticleTrace
        New instance with additional scalars.
    """
    sampled = sample_fields(data, trace.points, fields, method=method)
    merged = dict(trace.scalars)
    merged.update(sampled)
    return copy.replace(trace, scalars=merged)
