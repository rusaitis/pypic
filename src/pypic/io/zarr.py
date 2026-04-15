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


def _ds_to_field_dataset(ds: xr.Dataset, source_label: str) -> FieldDataset:
    """Decode pypic metadata from an xarray Dataset and construct a FieldDataset.

    Shared by ``from_zarr`` (plain Zarr) and ``from_zarr_icechunk``.
    """
    pypic_attrs = ds.attrs.get("pypic")
    if pypic_attrs is None:
        msg = f"No 'pypic' metadata found in {source_label}"
        raise ValueError(msg)

    grid, normalization, species, physics, metadata, frame, transforms = (
        decode_pypic_attrs(pypic_attrs)
    )

    # Strip the pypic key so it doesn't leak into the user-facing dataset.
    # Work on a copy to avoid mutating the cached store attrs.
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


def _resolve_timeseries_pairs(
    source: Simulation | Iterable[tuple[float | int, FieldDataset]],
    steps: Sequence[int] | None,
    fields: Sequence[str] | None,
) -> Iterable[tuple[float | int, FieldDataset]]:
    """Normalize a timeseries source into an iterable of (time, FieldDataset).

    Shared by ``to_zarr_timeseries`` and ``to_zarr_timeseries_icechunk``.
    """
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

        return _iter_sim()
    return source


def _default_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Build per-variable encoding with Blosc+zstd+bitshuffle compression."""
    from zarr.codecs import BloscCodec

    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    return {str(name): {"compressors": compressor} for name in ds.data_vars}


def _check_timeseries_fields(
    expected: frozenset[str],
    current: frozenset[str],
    time_val: float,
) -> None:
    """Reject timeseries steps whose field set differs from the first.

    xarray's ``to_zarr(mode="a", append_dim="time")`` does not enforce
    a consistent variable set across appends; a drift would silently
    write a store unreadable by ``from_zarr`` (conflicting sizes on
    ``time``).  Strict check matches the repo's "fail loud on
    selection mismatches" convention.
    """
    if current == expected:
        return
    extra = sorted(current - expected)
    missing = sorted(expected - current)
    msg = (
        f"Timeseries step at time={time_val}: field set {sorted(current)} "
        f"differs from the first step's {sorted(expected)} "
        f"(extra: {extra}, missing: {missing}).  All steps must share "
        "one field set — xarray append would otherwise leave the store "
        "unreadable."
    )
    raise ValueError(msg)


def _build_encoding(
    ds: xr.Dataset,
    dtype: str | None,
    encoding: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge default compression, optional dtype downcast, and user overrides."""
    enc = _default_encoding(ds)
    if dtype is not None:
        for var_name in ds.data_vars:
            enc[str(var_name)]["dtype"] = dtype
    if encoding is not None:
        for name, overrides in encoding.items():
            enc.setdefault(name, {}).update(overrides)
    return enc


def to_zarr(
    fds: FieldDataset,
    path: str | Path,
    *,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    backend: str | None = None,
    message: str | None = None,
    branch: str = "main",
) -> str | None:
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
    backend : str or None
        Storage backend.  ``None`` (default) for plain Zarr v3,
        ``"icechunk"`` for versioned Icechunk storage.
    message : str or None
        Commit message (Icechunk only).
    branch : str
        Branch to commit to (Icechunk only).  Default ``"main"``.

    Returns
    -------
    str or None
        Snapshot ID when ``backend="icechunk"``, ``None`` otherwise.

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
    if backend == "icechunk":
        from pypic.io._icechunk import to_zarr_icechunk

        return to_zarr_icechunk(
            fds,
            path,
            dtype=dtype,
            encoding=encoding,
            message=message,
            branch=branch,
        )
    if backend is not None:
        msg = f"Unknown backend: {backend!r}. Use None or 'icechunk'."
        raise ValueError(msg)
    if message is not None:
        msg = "message= requires backend='icechunk'."
        raise ValueError(msg)

    ensure_zarr()
    import shutil
    from pathlib import Path as _Path

    # Same cleanup gating as the Icechunk writers: ``ds.to_zarr`` in
    # ``mode='w'`` materializes a partial store (``zarr.json`` and any
    # chunks written before the failure) into the destination before
    # attribute serialization can reject, say, non-serializable
    # metadata.  Without cleanup a later ``from_zarr`` on that path
    # surfaces "No 'pypic' metadata found" instead of the actual
    # underlying write error, misleading operators.  A pre-existing
    # non-empty directory is left alone — that is the user's data.
    path_obj = _Path(path)
    created_new = not path_obj.exists() or (
        path_obj.is_dir() and not any(path_obj.iterdir())
    )

    ds = fds.xr.copy(deep=False)
    ds.attrs["pypic"] = encode_pypic_attrs(fds)

    try:
        ds.to_zarr(
            str(path),
            zarr_format=3,
            consolidated=False,
            mode="w",
            encoding=_build_encoding(ds, dtype, encoding),
        )
    except BaseException:
        if created_new:
            shutil.rmtree(path, ignore_errors=True)
        raise
    _log.info("Wrote %d fields to %s", len(ds.data_vars), path)
    return None


def from_zarr(
    path: str | Path,
    *,
    branch: str | None = None,
    tag: str | None = None,
    snapshot_id: str | None = None,
) -> FieldDataset:
    r"""Read a FieldDataset from a Zarr v3 store.

    Returns a lazy-loading dataset by default — field arrays are read
    from disk on first access.  Icechunk repositories are auto-detected;
    pass *branch*, *tag*, or *snapshot_id* to read a specific version.

    Parameters
    ----------
    path : str or Path
        Path to the Zarr store directory (or Icechunk repository).
    branch : str or None
        Icechunk branch to read from.
    tag : str or None
        Icechunk tag to read from.
    snapshot_id : str or None
        Icechunk snapshot ID to read from.

    Returns
    -------
    FieldDataset
        Reconstructed dataset with full metadata.

    Examples
    --------
    >>> # fds = from_zarr("/tmp/test.zarr")
    """
    has_ref = any(x is not None for x in (branch, tag, snapshot_id))

    if has_ref:
        from pypic.io._icechunk import from_zarr_icechunk

        return from_zarr_icechunk(
            path,
            branch=branch,
            tag=tag,
            snapshot_id=snapshot_id,
        )

    from pypic.io._icechunk import is_icechunk_store

    if is_icechunk_store(path):
        from pypic.io._icechunk import from_zarr_icechunk

        return from_zarr_icechunk(path)

    ensure_zarr()
    ds = xr.open_zarr(str(path), consolidated=False)
    return _ds_to_field_dataset(ds, f"Zarr store at {path}")


def to_zarr_timeseries(
    source: Simulation | Iterable[tuple[float | int, FieldDataset]],
    path: str | Path,
    *,
    steps: Sequence[int] | None = None,
    fields: Sequence[str] | None = None,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    backend: str | None = None,
    message: str | None = None,
    branch: str = "main",
) -> str | None:
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
    backend : str or None
        Storage backend.  ``None`` for plain Zarr v3,
        ``"icechunk"`` for versioned Icechunk storage.
    message : str or None
        Commit message (Icechunk only).
    branch : str
        Branch to commit to (Icechunk only).  Default ``"main"``.

    Returns
    -------
    str or None
        Snapshot ID when ``backend="icechunk"``, ``None`` otherwise.

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
    if backend == "icechunk":
        from pypic.io._icechunk import to_zarr_timeseries_icechunk

        return to_zarr_timeseries_icechunk(
            source,
            path,
            steps=steps,
            fields=fields,
            dtype=dtype,
            encoding=encoding,
            message=message,
            branch=branch,
        )
    if backend is not None:
        msg = f"Unknown backend: {backend!r}. Use None or 'icechunk'."
        raise ValueError(msg)
    if message is not None:
        msg = "message= requires backend='icechunk'."
        raise ValueError(msg)

    ensure_zarr()
    import shutil

    pairs = _resolve_timeseries_pairs(source, steps, fields)
    path_str = str(path)
    first = True
    pypic_attrs: dict[str, Any] | None = None
    expected_fields: frozenset[str] = frozenset()
    try:
        for time_val, fds in pairs:
            current_fields = frozenset(fds.field_names())
            ds = fds.xr.expand_dims(time=[float(time_val)])

            if first:
                pypic_attrs = encode_pypic_attrs(fds)
                expected_fields = current_fields
                ds.to_zarr(
                    path_str,
                    zarr_format=3,
                    consolidated=False,
                    mode="w",
                    encoding=_build_encoding(ds, dtype, encoding),
                )
                first = False
            else:
                _check_timeseries_fields(expected_fields, current_fields, time_val)
                ds.to_zarr(
                    path_str,
                    consolidated=False,
                    mode="a",
                    append_dim="time",
                )
    except BaseException:
        # Partial multi-step writes leave a store that looks like a
        # valid single-step export — delete it so the filesystem state
        # matches the error state.  BaseException covers KeyboardInterrupt
        # as well: a Ctrl-C between appends would otherwise leave the
        # same orphan store behind.
        if not first:
            shutil.rmtree(path_str, ignore_errors=True)
        raise

    if first:
        msg = "No timesteps to write — source yielded zero items."
        raise ValueError(msg)

    # Stamp pypic metadata after all appends — xarray's append mode
    # clears dataset-level attrs.
    import zarr

    store = zarr.open_group(path_str, mode="r+")
    store.attrs["pypic"] = pypic_attrs

    _log.info("Wrote timeseries to %s", path)
    return None
