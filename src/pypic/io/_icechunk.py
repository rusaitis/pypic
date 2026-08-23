"""Icechunk storage backend for versioned Zarr v3 I/O.

Provides Git-like versioning (tags, snapshots, branches) and ACID
transactions over Zarr v3 stores via the Icechunk Rust backend.

Requires optional dependency ``icechunk>=1.1``.
Install with ``pip install pypic-plasma[icechunk]``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from pypic.io._guard import ensure_icechunk
from pypic.io.metadata import encode_pypic_attrs, read_simulation_toml
from pypic.io.zarr import (
    _build_encoding,
    _datatree_encoding,
    _ds_to_field_dataset,
    _open_store,
    _resolve_timeseries_pairs,
    _write_timeseries_steps,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pypic.dataset import FieldDataset
    from pypic.readers._registry import Simulation

# Only the repo/version-management helpers are public API; the four
# ``*_icechunk`` read/write entry points are dispatch targets of
# ``to_zarr(backend="icechunk")`` and ``from_zarr``, not names a caller
# reaches for directly.
__all__ = [
    "icechunk_ancestry",
    "icechunk_create_tag",
    "open_icechunk_repo",
]

_log = logging.getLogger(__name__)


def is_icechunk_store(path: str | Path) -> bool:
    """Check whether a local path is an Icechunk repository.

    Detects the Icechunk on-disk layout by looking for the ``repo``
    marker file.  Does not import icechunk — this is a fast filesystem
    check used by ``from_zarr`` auto-detection.

    Parameters
    ----------
    path : str or Path
        Path to probe.

    Returns
    -------
    bool
        ``True`` if *path* looks like an Icechunk repository.
    """
    p = Path(path)
    return (p / "repo").is_file() and (p / "snapshots").is_dir()


def open_icechunk_repo(
    path: str | Path,
    *,
    create: bool = False,
    authorize_virtual_chunk_access: dict[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    r"""Open (or create) a local Icechunk repository.

    Parameters
    ----------
    path : str or Path
        Directory for the repository.
    create : bool
        When ``True``, create the repository if it does not exist.
    authorize_virtual_chunk_access : dict or None
        Mapping of URL prefix → credentials (``None`` for unauthenticated
        local ``file://`` URLs).  When ``None`` (the default), the
        function auto-detects every ``VirtualChunkContainer`` registered
        with the repo and authorizes each prefix with ``None``
        credentials — local virtual stores written by
        ``to_icechunk_virtual`` round-trip without further wiring.  Pass
        an empty dict to disable virtual chunk reads, or supply explicit
        credentials for cloud (``s3://``, ``gs://``) containers.

    Returns
    -------
    icechunk.Repository
        The repository handle.
    """
    ensure_icechunk()
    import icechunk

    storage = icechunk.local_filesystem_storage(str(path))

    if authorize_virtual_chunk_access is None:
        # Two-pass open: peek at the persisted RepositoryConfig to
        # discover registered containers, then re-open with auth that
        # matches their url_prefixes.  Containers store their full
        # prefix (e.g. file:///path/to/source/), which is what
        # Icechunk requires — a generic "file://" auth does not match.
        if create:
            probe = icechunk.Repository.open_or_create(storage)
        else:
            probe = icechunk.Repository.open(storage)
        containers = probe.config.virtual_chunk_containers or {}
        authorize_virtual_chunk_access = {prefix: None for prefix in containers}

    if create:
        return icechunk.Repository.open_or_create(
            storage,
            authorize_virtual_chunk_access=authorize_virtual_chunk_access,
        )
    return icechunk.Repository.open(
        storage,
        authorize_virtual_chunk_access=authorize_virtual_chunk_access,
    )


def _ensure_branch(repo: Any, branch: str) -> None:  # noqa: ANN401
    """Create ``branch`` forked from ``main`` if it does not exist yet.

    ``Repository.writable_session`` requires the branch ref to exist;
    Icechunk only auto-creates ``main`` when the repo itself is
    created.  Idempotent.
    """
    if branch in repo.list_branches():
        return
    if branch == "main":
        # main is the initial branch — if it is missing the repo is
        # broken, not empty.  Let the downstream session call raise.
        return
    main_tip = repo.lookup_branch("main")
    repo.create_branch(branch, main_tip)


def to_zarr_icechunk(
    fds: FieldDataset,
    path: str | Path,
    *,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    message: str | None = None,
    branch: str = "main",
    simulation_toml: str | Path | None = None,
) -> str:
    r"""Write a FieldDataset to an Icechunk-managed Zarr v3 store.

    Creates (or opens) a local Icechunk repository at *path*, writes
    all field data plus pypic metadata, and commits atomically.

    Parameters
    ----------
    fds : FieldDataset
        The dataset to write.
    path : str or Path
        Directory for the Icechunk repository.
    dtype : str or None
        Downcast dtype (e.g. ``"float32"``).
    encoding : dict or None
        Per-variable encoding overrides.
    message : str or None
        Commit message.  Defaults to ``"pypic: write <n> fields"``.
    branch : str
        Branch to commit to.  Default ``"main"``.
    simulation_toml : str or Path or None
        Source ``simulation.toml`` to stamp verbatim into
        ``attrs.simulation_toml``. Carries the schema sections
        the typed FieldDataset boundary drops. ``None``
        (default) stamps whatever the dataset already knows.

    Returns
    -------
    str
        The snapshot ID of the new commit.
    """
    ensure_icechunk()
    import shutil

    # Same cleanup gating as ``to_zarr_timeseries``: a fresh-or-empty
    # output directory means the half-initialized repo is ours to remove
    # if anything between ``open_icechunk_repo(create=True)`` and
    # ``session.commit`` raises.  Without this, callers see
    # ``is_icechunk_store`` return True while ``from_zarr`` raises
    # ``GroupNotFoundError`` — non-serializable metadata or an
    # ``encode_pypic_attrs`` failure are the realistic triggers.
    path_obj = Path(path)
    created_new = not path_obj.exists() or (
        path_obj.is_dir() and not any(path_obj.iterdir())
    )

    repo = open_icechunk_repo(path, create=True)
    try:
        _ensure_branch(repo, branch)
        session = repo.writable_session(branch)

        ds = fds.xr.copy(deep=False)
        tree = xr.DataTree.from_dict({"fields": ds})
        pypic_attrs = encode_pypic_attrs(fds)
        if simulation_toml is not None:
            pypic_attrs["simulation_toml"] = read_simulation_toml(simulation_toml)
        tree.attrs = pypic_attrs

        # Icechunk doesn't support Zarr's consolidated metadata
        # (snapshots already act as the equivalent index), so leave it
        # off here even though plain-Zarr writes use ``consolidated=True``.
        ds_encoding = _build_encoding(ds, dtype, encoding)
        tree.to_zarr(
            session.store,
            mode="w",
            consolidated=False,
            encoding=_datatree_encoding(ds_encoding),
            zarr_format=3,
        )

        n_fields = len(ds.data_vars)
        if message is None:
            message = f"pypic: write {n_fields} fields"

        snapshot_id: str = session.commit(message)
    except BaseException:
        if created_new:
            shutil.rmtree(path, ignore_errors=True)
        raise
    _log.info("Wrote %d fields to %s (snapshot %s)", n_fields, path, snapshot_id)
    return snapshot_id


def from_zarr_icechunk(
    path: str | Path,
    *,
    branch: str | None = None,
    tag: str | None = None,
    snapshot_id: str | None = None,
) -> FieldDataset:
    r"""Read a FieldDataset from an Icechunk repository.

    Auto-opens the repository at *path* and creates a read-only
    session at the requested ref (branch tip, tag, or snapshot).

    Parameters
    ----------
    path : str or Path
        Path to the Icechunk repository.
    branch : str or None
        Branch to read from.  Default ``"main"`` when no ref is given.
    tag : str or None
        Tag to read from.
    snapshot_id : str or None
        Exact snapshot ID to read from.

    Returns
    -------
    FieldDataset
        Reconstructed dataset with full metadata.

    Raises
    ------
    ValueError
        If more than one of *branch*, *tag*, *snapshot_id* is specified,
        or if no pypic metadata is found.
    """
    ensure_icechunk()

    specified = sum(x is not None for x in (branch, tag, snapshot_id))
    if specified > 1:
        msg = "Specify at most one of branch, tag, or snapshot_id."
        raise ValueError(msg)

    repo = open_icechunk_repo(path)

    if tag is not None:
        session = repo.readonly_session(tag=tag)
    elif snapshot_id is not None:
        session = repo.readonly_session(snapshot_id=snapshot_id)
    else:
        session = repo.readonly_session(branch=branch or "main")

    ds, root_attrs = _open_store(session.store, f"Icechunk store at {path}")
    return _ds_to_field_dataset(ds, root_attrs, f"Icechunk store at {path}")


def to_zarr_timeseries_icechunk(
    source: Simulation | Iterable[tuple[float | int, FieldDataset]],
    path: str | Path,
    *,
    steps: Sequence[int] | None = None,
    fields: Sequence[str] | None = None,
    dtype: str | None = None,
    encoding: dict[str, dict[str, Any]] | None = None,
    message: str | None = None,
    branch: str = "main",
    simulation_toml: str | Path | None = None,
) -> str:
    r"""Write multiple timesteps to an Icechunk-managed Zarr v3 store.

    All timesteps are written within a single session and committed
    atomically — either all land or none do.

    Parameters
    ----------
    source : Simulation or Iterable[tuple[float | int, FieldDataset]]
        Either a ``Simulation`` object or an iterable of ``(time, fds)``
        pairs.
    path : str or Path
        Directory for the Icechunk repository.
    steps : Sequence[int] or None
        Timestep indices (only for ``Simulation`` source).
    fields : Sequence[str] or None
        Field names to include (only for ``Simulation`` source).
    dtype : str or None
        Downcast dtype (e.g. ``"float32"``).
    encoding : dict or None
        Per-variable encoding overrides.
    message : str or None
        Commit message.  Defaults to ``"pypic: write timeseries"``.
    branch : str
        Branch to commit to.  Default ``"main"``.
    simulation_toml : str or Path or None
        Source ``simulation.toml`` to stamp verbatim into
        ``attrs.simulation_toml``. Carries the schema sections
        the typed FieldDataset boundary drops. ``None``
        (default) stamps whatever the dataset already knows.

    Returns
    -------
    str
        The snapshot ID of the new commit.
    """
    ensure_icechunk()
    import shutil

    import zarr

    # Whether we created the repo directory on this call.  An existing
    # repo's uncommitted session is transactional — nothing to clean —
    # but ``open_icechunk_repo(create=True)`` on a fresh (or empty) path
    # persists an initial snapshot before we know whether the source is
    # usable.  If our pypic write then fails, ``is_icechunk_store``
    # returns True but ``from_zarr`` raises ``GroupNotFoundError``.
    # Remove the half-initialized directory so the filesystem state
    # matches the error state.  An empty directory pre-existing on disk
    # is treated the same as a missing one — we own its contents.
    path_obj = Path(path)
    created_new = not path_obj.exists() or (
        path_obj.is_dir() and not any(path_obj.iterdir())
    )

    pairs = _resolve_timeseries_pairs(source, steps, fields)
    repo = open_icechunk_repo(path, create=True)
    _ensure_branch(repo, branch)
    session = repo.writable_session(branch)

    try:
        pypic_attrs = _write_timeseries_steps(
            pairs,
            session.store,
            dtype=dtype,
            encoding=encoding,
            consolidated=False,
        )
        if pypic_attrs is None:
            msg = "No timesteps to write — source yielded zero items."
            raise ValueError(msg)

        if simulation_toml is not None:
            pypic_attrs["simulation_toml"] = read_simulation_toml(simulation_toml)

        # Restamp the cross-step metadata intersection on the root group;
        # ``_write_timeseries_steps`` wrote step-1's metadata via the
        # DataTree but the running intersection is the only set true at
        # every step.  Icechunk persists attribute writes within the
        # same writable session, so the commit below picks them up.
        root = zarr.open_group(session.store, mode="r+")
        for key, value in pypic_attrs.items():
            root.attrs[key] = value

        if message is None:
            message = "pypic: write timeseries"

        snapshot_id: str = session.commit(message)
    except BaseException:
        if created_new:
            shutil.rmtree(path, ignore_errors=True)
        raise
    _log.info("Wrote timeseries to %s (snapshot %s)", path, snapshot_id)
    return snapshot_id


def icechunk_create_tag(
    path: str | Path,
    tag: str,
    *,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> None:
    r"""Create a named tag in an Icechunk repository.

    Parameters
    ----------
    path : str or Path
        Path to the Icechunk repository.
    tag : str
        Tag name (e.g. ``"v1.0-paper-submission"``).
    snapshot_id : str or None
        Snapshot to tag.  Defaults to the tip of *branch*.
    branch : str
        Branch whose tip to tag (ignored when *snapshot_id* is given).
    """
    ensure_icechunk()

    repo = open_icechunk_repo(path)
    if snapshot_id is None:
        snapshot_id = repo.lookup_branch(branch)
    repo.create_tag(tag, snapshot_id)
    _log.info("Tagged snapshot %s as %r in %s", snapshot_id, tag, path)


def icechunk_ancestry(
    path: str | Path,
    *,
    branch: str = "main",
) -> list[dict[str, str]]:
    r"""Return the commit history of an Icechunk repository.

    Parameters
    ----------
    path : str or Path
        Path to the Icechunk repository.
    branch : str
        Branch whose ancestry to inspect.

    Returns
    -------
    list[dict[str, str]]
        List of ``{"id": ..., "message": ...}`` dicts, most recent
        first.
    """
    ensure_icechunk()

    repo = open_icechunk_repo(path)
    return [
        {"id": info.id, "message": info.message}
        for info in repo.ancestry(branch=branch)
    ]
