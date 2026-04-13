"""Zarr v3 export and import for FieldDataset.

Single-timestep persistence via ``to_zarr`` / ``from_zarr`` and
multi-timestep time-series via ``to_zarr_timeseries``.  All pypic
metadata (grid, normalization, species, physics, frame, transforms)
is serialized into ``xr.Dataset.attrs`` so the round-trip is fully
self-describing.

Requires optional dependencies: ``zarr>=3.1.0`` and ``numcodecs>=0.16.0``.
Install with ``pip install pypic[zarr]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import xarray as xr

from pypic.dataset import FieldDataset
from pypic.io._guard import ensure_zarr
from pypic.io._serialize import decode_pypic_attrs, encode_pypic_attrs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from pypic.readers._registry import Simulation

__all__ = ["from_zarr", "to_zarr", "to_zarr_timeseries"]

_log = logging.getLogger(__name__)


def _default_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Build per-variable encoding with Blosc+zstd+bitshuffle compression."""
    from zarr.codecs import BloscCodec

    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    return {str(name): {"compressors": compressor} for name in ds.data_vars}


def to_zarr(
    fds: FieldDataset,
    path: str | Path,
    *,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
) -> None:
    r"""Write a FieldDataset to a Zarr v3 store.

    All pypic metadata (grid, normalization, species, physics, frame,
    transforms) is serialized into ``xr.Dataset.attrs["pypic"]`` so
    that ``from_zarr`` can reconstruct the full object.

    Parameters
    ----------
    fds : FieldDataset
        The dataset to write.
    path : str or Path
        Destination directory (created if it does not exist).
    dtype : str or None
        When set (e.g. ``"float32"``), all field arrays are downcast
        to this dtype on write.  Default: preserve source dtype.
    encoding : dict or None
        Per-variable encoding overrides merged on top of the defaults
        (Blosc zstd + bitshuffle).  Keys are variable names, values
        are dicts passed to ``xr.Dataset.to_zarr(encoding=...)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0))
    >>> fds = FieldDataset.from_arrays(
    ...     {"B1": np.ones((4, 3, 2))}, grid, Normalization.identity(),
    ... )
    >>> # to_zarr(fds, "/tmp/test.zarr")  # writes to disk
    """
    ensure_zarr()
    ds = fds.xr.copy(deep=False)
    ds.attrs["pypic"] = encode_pypic_attrs(fds)

    enc = _default_encoding(ds)
    if dtype is not None:
        for var_name in ds.data_vars:
            enc[str(var_name)]["dtype"] = dtype
    if encoding is not None:
        for name, overrides in encoding.items():
            enc.setdefault(name, {}).update(overrides)

    ds.to_zarr(
        str(path),
        zarr_format=3,
        consolidated=False,
        mode="w",
        encoding=enc,
    )
    _log.info("Wrote %d fields to %s", len(ds.data_vars), path)


def from_zarr(path: str | Path) -> FieldDataset:
    r"""Read a FieldDataset from a Zarr v3 store.

    Returns a lazy-loading dataset by default — field arrays are read
    from disk on first access.

    Parameters
    ----------
    path : str or Path
        Path to the Zarr store directory.

    Returns
    -------
    FieldDataset
        Reconstructed dataset with full metadata.

    Examples
    --------
    >>> # fds = from_zarr("/tmp/test.zarr")
    """
    ensure_zarr()
    ds = xr.open_zarr(str(path), consolidated=False)
    pypic_attrs = ds.attrs.get("pypic")
    if pypic_attrs is None:
        msg = f"No 'pypic' metadata found in Zarr store at {path}"
        raise ValueError(msg)

    grid, normalization, species, physics, metadata, frame, transforms = (
        decode_pypic_attrs(pypic_attrs)
    )

    # Strip the pypic key from attrs so it doesn't leak into the user-facing
    # dataset.  Work on a copy to avoid mutating the cached store attrs.
    ds_attrs = dict(ds.attrs)
    ds_attrs.pop("pypic", None)
    ds.attrs = ds_attrs

    return FieldDataset(
        ds,
        grid,
        normalization,
        species=species,
        physics=physics,
        metadata=metadata,
        frame=frame,
        transforms=transforms,
    )


def to_zarr_timeseries(
    source: Simulation | Iterable[tuple[float | int, FieldDataset]],
    path: str | Path,
    *,
    steps: Sequence[int] | None = None,
    fields: Sequence[str] | None = None,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
) -> None:
    r"""Write multiple timesteps to a single Zarr v3 store.

    Each field becomes ``(nt, n1, n2, n3)`` with ``time`` as the first
    dimension, chunked so that reading one timestep is O(1).

    Parameters
    ----------
    source : Simulation or Iterable[tuple[float | int, FieldDataset]]
        Either a ``Simulation`` object (reads timesteps via
        ``source.read(step)``) or an iterable of ``(time, fds)``
        pairs.
    path : str or Path
        Destination Zarr store directory.
    steps : Sequence[int] or None
        Timestep indices to write (only used when *source* is a
        ``Simulation``).  Defaults to ``source.steps``.
    fields : Sequence[str] or None
        Field names to include (only used when *source* is a
        ``Simulation``).  Defaults to all available fields.
    dtype : str or None
        Downcast dtype (e.g. ``"float32"``).
    encoding : dict or None
        Per-variable encoding overrides.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(dimensions=(4, 3), spacing=(1.0, 1.0))
    >>> pairs = [
    ...     (0.0, FieldDataset.from_arrays(
    ...         {"B1": np.ones((4, 3))}, grid, Normalization.identity())),
    ...     (1.0, FieldDataset.from_arrays(
    ...         {"B1": np.ones((4, 3)) * 2}, grid, Normalization.identity())),
    ... ]
    >>> # to_zarr_timeseries(pairs, "/tmp/ts.zarr")
    """
    ensure_zarr()
    # Normalize source to an iterable of (time_value, FieldDataset)
    pairs: Iterable[tuple[float | int, FieldDataset]]
    from pypic.readers._registry import Simulation as _Sim

    if isinstance(source, _Sim):
        step_list = list(steps) if steps is not None else source.steps
        fields_list = list(fields) if fields is not None else None

        def _iter_sim() -> Iterable[tuple[float | int, FieldDataset]]:
            for step in step_list:
                fds = source.read(step, fields=fields_list)
                dt = fds.grid.dt
                t: float | int = step * dt if dt is not None else step
                yield t, fds

        pairs = _iter_sim()
    else:
        pairs = source

    path_str = str(path)
    first = True
    pypic_attrs: dict[str, Any] | None = None
    for time_val, fds in pairs:
        ds = fds.xr.copy(deep=False)
        ds = ds.expand_dims(time=[float(time_val)])

        if first:
            pypic_attrs = encode_pypic_attrs(fds)
            enc = _default_encoding(ds)
            if dtype is not None:
                for var_name in ds.data_vars:
                    enc[str(var_name)]["dtype"] = dtype
            if encoding is not None:
                for enc_name, overrides in encoding.items():
                    enc.setdefault(enc_name, {}).update(overrides)

            ds.to_zarr(
                path_str,
                zarr_format=3,
                consolidated=False,
                mode="w",
                encoding=enc,
            )
            first = False
        else:
            ds.to_zarr(
                path_str,
                consolidated=False,
                mode="a",
                append_dim="time",
            )

    # Write pypic metadata after all appends — xarray's append mode
    # clears dataset-level attrs, so we stamp them onto the zarr store
    # directly at the end.
    if pypic_attrs is not None:
        import zarr

        store = zarr.open_group(path_str, mode="r+")
        store.attrs["pypic"] = pypic_attrs

    _log.info("Wrote timeseries to %s", path)
