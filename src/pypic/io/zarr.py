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
from pypic.io._serialize import (
    SCHEMA_VERSION,
    _to_json_native,
    decode_pypic_attrs,
    encode_pypic_attrs,
    read_simulation_toml,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from pypic.readers._registry import Simulation

__all__ = ["from_zarr", "to_zarr", "to_zarr_timeseries"]

_log = logging.getLogger(__name__)


# Root-attrs keys reserved for pypic metadata.  Stripped from any
# Dataset before it is handed to ``FieldDataset`` so layout-bookkeeping
# attrs don't leak as user-visible.
_PYPIC_ROOT_ATTR_KEYS = frozenset(
    {
        "schema",
        "grid",
        "normalization",
        "species",
        "physics",
        "frame",
        "transforms",
        "metadata",
    }
)


def _strip_pypic_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with pypic-internal attrs removed.

    Mutates ``ds.attrs`` in place and returns the same object — caller
    typically passes a fresh Dataset reference (the result of
    ``open_datatree(...)["fields"].to_dataset()``).
    """
    ds.attrs = {k: v for k, v in ds.attrs.items() if k not in _PYPIC_ROOT_ATTR_KEYS}
    return ds


def _ds_to_field_dataset(
    ds: xr.Dataset, root_attrs: dict[str, Any], source_label: str
) -> FieldDataset:
    """Build a FieldDataset from a fields Dataset and root attrs.

    *ds* must already be the field-bearing Dataset under ``/fields``.
    *root_attrs* is the dict from the root group's attrs.
    """
    try:
        grid, normalization, species, physics, metadata, frame, transforms = (
            decode_pypic_attrs(root_attrs)
        )
    except ValueError as exc:
        msg = f"No pypic metadata found in {source_label}: {exc}"
        raise ValueError(msg) from None
    return FieldDataset(
        _strip_pypic_attrs(ds),
        grid,
        normalization,
        species=species,
        physics=physics,
        metadata=metadata,
        frame=frame,
        transforms=transforms,
    )


def _open_store(
    store: Any,  # noqa: ANN401  # zarr accepts str/path/store object
    source_label: str,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Open a schema-v1.0 Zarr store; return (fields_dataset, root_attrs).

    Root group must carry the ``schema.version`` discriminator and a
    ``/fields`` child group with the field arrays.  Uses
    ``consolidated="auto"`` so consolidated stores get the one-shot
    metadata read while non-consolidated stores still load.
    """
    tree = xr.open_datatree(store, engine="zarr", consolidated="auto")
    root_attrs: dict[str, Any] = {str(k): v for k, v in tree.attrs.items()}
    schema_attrs = root_attrs.get("schema")
    if not isinstance(schema_attrs, dict) or "version" not in schema_attrs:
        msg = (
            f"{source_label}: no pypic metadata found "
            f"(expected ``schema.version`` discriminator)"
        )
        raise ValueError(msg)
    version = schema_attrs["version"]
    if version != SCHEMA_VERSION:
        msg = (
            f"{source_label}: schema.version={version!r} but this pypic "
            f"build expects {SCHEMA_VERSION!r}; cannot decode safely"
        )
        raise ValueError(msg)
    if "fields" not in tree.children:
        msg = f"{source_label}: schema.version declared but no /fields group present"
        raise ValueError(msg)
    return tree["fields"].to_dataset(), root_attrs


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


def _datatree_encoding(
    ds_encoding: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Wrap a Dataset-level encoding under the ``/fields`` group key.

    ``DataTree.to_zarr`` validates encoding keys against the tree's
    group paths exactly (``"/"``, ``"/fields"``, ...), so the leading
    slash is required.  Plain ``Dataset.to_zarr`` takes the flat
    ``{var_name: {options}}`` form.  This helper bridges the two so
    callers can build the per-variable encoding once.
    """
    return {"/fields": dict(ds_encoding)}


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


def _check_timeseries_identity(
    first_fds: FieldDataset,
    current_fds: FieldDataset,
    time_val: float,
) -> None:
    """Reject timeseries steps whose structural attrs differ from the first.

    A timeseries describes one simulation's evolution: grid,
    normalization, species, physics, frame, and frame transforms must
    stay constant.  Without this check, the writer used to silently
    flatten every later step's identity to the first step's — the data
    landed but the reconstructed FieldDataset described a different
    simulation than the one that produced it.  Per-step *metadata*
    differences are tolerated (intersected by the caller); this guard
    only fires for the identity-defining attrs.
    """
    diffs: list[str] = []
    if first_fds.grid != current_fds.grid:
        diffs.append("grid")
    if first_fds.normalization != current_fds.normalization:
        diffs.append("normalization")
    if first_fds.species != current_fds.species:
        diffs.append("species")
    if first_fds.physics != current_fds.physics:
        diffs.append("physics")
    if first_fds.frame != current_fds.frame:
        diffs.append("frame")
    if dict(first_fds.transforms) != dict(current_fds.transforms):
        diffs.append("transforms")
    if not diffs:
        return
    msg = (
        f"Timeseries step at time={time_val}: "
        f"{', '.join(diffs)} differ from the first step.  These attrs "
        "define the simulation's identity and must be constant across "
        "a timeseries write — pre-flatten the source or split into "
        "separate stores."
    )
    raise ValueError(msg)


def _intersect_encoded_metadata(
    running: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Keep only metadata keys whose encoded values match across steps.

    Per-step metadata divergence (e.g. iPIC3D's per-file ``time`` /
    ``step`` scalars) is normal and not an error, but the final stored
    attrs must reflect what is actually true everywhere — silently
    keeping step-0's values would mislead readers.  This intersection
    runs on the already JSON-encoded dicts, so equality respects the
    serializer's coercions (numpy scalars → python natives, etc.).
    """
    return {k: v for k, v in running.items() if k in current and current[k] == v}


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


def _write_timeseries_steps(
    pairs: Iterable[tuple[float | int, FieldDataset]],
    store: Any,  # noqa: ANN401  # zarr accepts str/path/store object
    *,
    dtype: str | None,
    encoding: dict[str, dict[str, Any]] | None,
    consolidated: bool = True,
) -> dict[str, Any] | None:
    """Append timeseries pairs into *store*; return the final pypic attrs.

    Step 1 is written through ``DataTree.to_zarr`` so the root group's
    flat v1 attrs land in one shot and the field arrays live under
    ``/fields``.  Subsequent steps drop to
    ``Dataset.to_zarr(group="fields", mode="a", append_dim="time")``
    — DataTree currently has no ``append_dim`` parameter, but the child
    Dataset already lives at the same on-disk path so a flat append
    extends the same arrays.

    Validates each step's field set and structural identity against
    the first; intersects per-step metadata to a cross-step common
    ground.  Returns the pypic attrs dict (with ``metadata`` set to
    the intersection) for callers to stamp back onto root after all
    appends settle, or ``None`` if *pairs* yielded nothing.
    """
    first = True
    pypic_attrs: dict[str, Any] | None = None
    first_fds: FieldDataset | None = None
    running_meta: dict[str, Any] = {}
    expected_fields: frozenset[str] = frozenset()
    for time_val, fds in pairs:
        current_fields = frozenset(fds.field_names())
        ds = fds.xr.expand_dims(time=[float(time_val)])
        if first:
            pypic_attrs = encode_pypic_attrs(fds)
            first_fds = fds
            running_meta = dict(pypic_attrs.get("metadata", {}))
            expected_fields = current_fields
            tree = xr.DataTree.from_dict({"fields": ds})
            tree.attrs = pypic_attrs
            ds_encoding = _build_encoding(ds, dtype, encoding)
            tree.to_zarr(
                store,
                mode="w",
                consolidated=consolidated,
                encoding=_datatree_encoding(ds_encoding),
                zarr_format=3,
            )
            first = False
        else:
            _check_timeseries_fields(expected_fields, current_fields, time_val)
            assert first_fds is not None
            _check_timeseries_identity(first_fds, fds, time_val)
            running_meta = _intersect_encoded_metadata(
                running_meta, _to_json_native(dict(fds.metadata))
            )
            ds.to_zarr(
                store,
                group="fields",
                consolidated=False,
                mode="a",
                append_dim="time",
            )
    if first:
        return None
    assert pypic_attrs is not None
    pypic_attrs["metadata"] = running_meta
    return pypic_attrs


def to_zarr(
    fds: FieldDataset,
    path: str | Path,
    *,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    backend: str | None = None,
    message: str | None = None,
    branch: str = "main",
    simulation_toml: str | Path | None = None,
) -> str | None:
    r"""Write a FieldDataset to a Zarr v3 store.

    All pypic metadata (grid, normalization, species, physics, frame,
    transforms) is serialized as flat keys on the root group's attrs
    alongside a ``schema.version`` discriminator (see schema.md §4.2),
    so that ``from_zarr`` — and any non-pypic consumer — can
    reconstruct the full object straight from the store.

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
    ...     {"B_1": np.ones((4, 3, 2))}, grid, Normalization.identity(),
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
            simulation_toml=simulation_toml,
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

    # Cleanup gating: ``DataTree.to_zarr(mode="w")`` materializes a
    # partial store (``zarr.json``, any chunks written before failure)
    # into the destination before attribute serialization can reject
    # — say — non-serializable metadata.  Without cleanup a later
    # ``from_zarr`` on that path surfaces "No pypic metadata found"
    # instead of the actual write error, misleading operators.  A
    # pre-existing non-empty directory is left alone — user data.
    path_obj = _Path(path)
    created_new = not path_obj.exists() or (
        path_obj.is_dir() and not any(path_obj.iterdir())
    )

    ds = fds.xr.copy(deep=False)
    tree = xr.DataTree.from_dict({"fields": ds})
    pypic_attrs = encode_pypic_attrs(fds)
    if simulation_toml is not None:
        pypic_attrs["simulation_toml"] = read_simulation_toml(simulation_toml)
    tree.attrs = pypic_attrs

    ds_encoding = _build_encoding(ds, dtype, encoding)
    try:
        tree.to_zarr(
            str(path),
            mode="w",
            consolidated=True,
            encoding=_datatree_encoding(ds_encoding),
            zarr_format=3,
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
    ds, root_attrs = _open_store(str(path), f"Zarr store at {path}")
    return _ds_to_field_dataset(ds, root_attrs, f"Zarr store at {path}")


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
    simulation_toml: str | Path | None = None,
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
    ...         {"B_1": np.ones((4, 3))}, grid, Normalization.identity())),
    ...     (1.0, FieldDataset.from_arrays(
    ...         {"B_1": np.ones((4, 3)) * 2}, grid, Normalization.identity())),
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
            simulation_toml=simulation_toml,
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

    pairs = _resolve_timeseries_pairs(source, steps, fields)
    path_str = str(path)
    # Cleanup gating mirrors the single-step + icechunk writers: if the
    # output path was fresh or empty when we started, any failure
    # inside the try is ours to clean up — including when xarray's
    # *first* ``ds.to_zarr(mode='w')`` call fails partway through and
    # leaves a stub store with just ``zarr.json``.  A pre-existing
    # non-empty directory is left alone (user data).
    path_obj = _Path(path)
    created_new = not path_obj.exists() or (
        path_obj.is_dir() and not any(path_obj.iterdir())
    )
    try:
        pypic_attrs = _write_timeseries_steps(
            pairs, path_str, dtype=dtype, encoding=encoding
        )
    except BaseException:
        # Any failure inside the loop — including a first-step
        # materialization failure that leaves only ``zarr.json`` behind
        # — reclaims the store when we owned the directory.
        # BaseException covers KeyboardInterrupt as well: a Ctrl-C
        # between appends would otherwise leave the same orphan store.
        if created_new:
            shutil.rmtree(path_str, ignore_errors=True)
        raise

    if pypic_attrs is None:
        msg = "No timesteps to write — source yielded zero items."
        raise ValueError(msg)

    if simulation_toml is not None:
        pypic_attrs["simulation_toml"] = read_simulation_toml(simulation_toml)

    # Restamp the cross-step metadata intersection on the root group.
    # ``_write_timeseries_steps`` already wrote step-1's pypic_attrs
    # via the DataTree, but those carried step-1's metadata only.
    # After all appends, the running intersection (computed during the
    # loop) is the only set of values true at every timestep, so reset
    # the metadata key — and re-consolidate so the root attrs change
    # is visible to ``consolidated="auto"`` readers.
    import zarr

    root = zarr.open_group(path_str, mode="r+")
    root.attrs.update(pypic_attrs)
    zarr.consolidate_metadata(root.store)

    _log.info("Wrote timeseries to %s", path)
    return None
