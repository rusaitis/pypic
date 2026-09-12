"""Virtual Zarr views over existing HDF5 simulation outputs.

Uses VirtualiZarr to extract byte-range metadata from HDF5 files and
persists the references through Icechunk's Zarr v3 backend (in-memory
store) so that consumers see a standard ``xr.open_zarr``-style dataset
that resolves chunks by reading byte ranges from the source HDF5 files.

Requires optional dependencies ``virtualizarr>=2.4`` and
``icechunk>=1.1``; both are installed by ``pip install "pypic-plasma[zarr]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import xarray as xr

from pypic.coordinates.geometry import CARTESIAN, GEOMETRY_BY_NAME
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.io._guard import ensure_icechunk, ensure_virtualizarr
from pypic.io._icechunk import _ensure_branch
from pypic.io.metadata import encode_pypic_attrs
from pypic.units import Normalization, UnitSystem

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pypic.containers import SimulationConfig
    from pypic.coordinates.transforms import FrameTransform

__all__ = ["open_virtual", "to_icechunk_virtual"]


def _as_text(val: object) -> str:
    """Coerce an HDF5 attr value to ``str``, decoding bytes as UTF-8.

    h5py may return string-valued attrs as ``bytes`` (or ``numpy.bytes_``)
    depending on how the source file encoded them.  Plain ``str(b"x")``
    produces ``"b'x'"`` rather than ``"x"``, which silently corrupts
    geometry/boundary/model labels.
    """
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _read_metadata_from_h5(
    path: str,
) -> tuple[GridInfo | None, Normalization | None, dict[str, Any]]:
    """Extract pypic-compatible metadata from an HDF5 file.

    Reads the ``grid/`` and ``normalization/`` groups if present
    (pypic canonical layout, schema.md Section 4).  Returns
    ``(grid, normalization, extra_attrs)`` where either may be
    ``None`` if the file lacks the relevant group.
    """
    grid: GridInfo | None = None
    norm: Normalization | None = None
    extra: dict[str, Any] = {}

    with h5py.File(path, "r") as f:
        # Grid metadata
        if "grid" in f:
            g = f["grid"]
            dims = tuple(int(x) for x in g.attrs["dimensions"])
            spacing = tuple(float(x) for x in g.attrs["spacing"])
            origin = tuple(float(x) for x in g.attrs.get("origin", np.zeros(len(dims))))
            geom_str = _as_text(g.attrs.get("geometry", "cartesian"))
            geometry = GEOMETRY_BY_NAME.get(geom_str, CARTESIAN)
            dt_val = g.attrs.get("dt")
            dt = float(dt_val) if dt_val is not None else None
            boundary_raw = g.attrs.get("boundary")
            boundary = (
                tuple(_as_text(b) for b in boundary_raw)
                if boundary_raw is not None
                else None
            )
            grid = GridInfo(
                dimensions=dims,
                spacing=spacing,
                origin=origin,
                geometry=geometry,
                dt=dt,
                boundary=boundary,
            )

        # Normalization metadata
        if "normalization" in f:
            n = f["normalization"]
            raw_system = n.attrs.get("system")
            norm = Normalization(
                length_ref=float(n.attrs.get("length_ref", 1.0)),
                time_ref=float(n.attrs.get("time_ref", 1.0)),
                velocity_ref=float(n.attrs.get("velocity_ref", 1.0)),
                b_field_ref=float(n.attrs.get("b_field_ref", 1.0)),
                e_field_ref=float(n.attrs.get("e_field_ref", 1.0)),
                density_ref=float(n.attrs.get("density_ref", 1.0)),
                mass_ref=float(n.attrs.get("mass_ref", 1.0)),
                charge_ref=float(n.attrs.get("charge_ref", 1.0)),
                # A group without the attr predates it and carried eight
                # refs someone wrote down — declared, system unknown.
                system=(
                    UnitSystem.CUSTOM
                    if raw_system is None
                    else UnitSystem(_as_text(raw_system))
                ),
            )

        # Root-level scalar metadata
        for key in ("time", "step", "model"):
            if key in f.attrs:
                val = f.attrs[key]
                if key == "step":
                    extra[key] = int(val)
                elif key == "model":
                    extra[key] = _as_text(val)
                else:
                    extra[key] = val

    return grid, norm, extra


def _container_name_for(url_prefix: str) -> str:
    """Stable, per-prefix container name.

    Icechunk requires globally unique container names within a repo;
    multiple commits from different source directories must each get a
    distinct name that derives deterministically from the prefix.
    """
    import hashlib

    digest = hashlib.sha1(url_prefix.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"local_{digest[:12]}"


def _make_virtual_chunk_container(url_prefix: str, source_dir: str) -> Any:  # noqa: ANN401
    """Build a VirtualChunkContainer for a local filesystem source dir."""
    import icechunk

    return icechunk.VirtualChunkContainer(
        name=_container_name_for(url_prefix),
        url_prefix=url_prefix,
        store=icechunk.local_filesystem_store(source_dir),
    )


def _virtual_repo_config(source_dir: str) -> tuple[Any, str]:
    """Build the Icechunk RepositoryConfig + url_prefix for a source dir.

    The returned config has a ``VirtualChunkContainer`` registered for
    ``file://{source_dir}/`` so that Icechunk knows how to resolve
    virtual chunk references back to the original HDF5 bytes.
    """
    import icechunk

    config = icechunk.RepositoryConfig.default()
    url_prefix = f"file://{source_dir}/"
    config.set_virtual_chunk_container(
        _make_virtual_chunk_container(url_prefix, source_dir)
    )
    return config, url_prefix


def _build_vds(
    source_str: str,
    fields_group: str | None,
    drop_variables: list[str] | None,
) -> xr.Dataset:
    """Extract a VirtualiZarr virtual dataset from an HDF5 file."""
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser

    url = f"file://{source_str}"
    registry = ObjectStoreRegistry()  # type: ignore[var-annotated]
    registry.register("file://", LocalStore())
    parser = HDFParser(group=fields_group)
    return open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        drop_variables=drop_variables,
        loadable_variables=[],
    )


def _virtual_to_readable(vds: xr.Dataset, *, source_dir: str) -> xr.Dataset:
    """Persist virtual references via Icechunk's Zarr v3 backend.

    Uses an in-memory Icechunk store as the v3-format home for the
    virtual refs, then opens it as a lazy ``xr.Dataset`` that resolves
    chunks by reading byte ranges from the source HDF5 files.
    """
    import icechunk

    storage = icechunk.in_memory_storage()
    config, url_prefix = _virtual_repo_config(source_dir)
    repo = icechunk.Repository.create(
        storage,
        config=config,
        authorize_virtual_chunk_access={url_prefix: None},
    )
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store)
    session.commit("Virtual refs")
    read_session = repo.readonly_session(branch="main")
    return xr.open_zarr(read_session.store, consolidated=False)  # type: ignore[no-any-return]


def open_virtual(
    path: str | Path,
    *,
    fields_group: str | None = "fields",
    drop_variables: list[str] | None = None,
    config: SimulationConfig | None = None,
) -> FieldDataset:
    r"""Create a virtual FieldDataset backed by HDF5 byte ranges.

    Uses VirtualiZarr to extract byte-range metadata from an HDF5
    file, producing a ``FieldDataset`` that lazily reads field data
    from the original file without copying.

    For files following the pypic canonical HDF5 layout (schema.md
    Section 4), grid and normalization metadata are read automatically
    from the ``grid/`` and ``normalization/`` groups.  For other
    layouts, pass a ``config`` with the required metadata.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.
    fields_group : str or None
        HDF5 group containing field datasets.  ``"fields"`` for the
        canonical layout.  ``None`` reads from the root group.
    drop_variables : list[str] or None
        HDF5 dataset names to exclude from the virtual view.
    config : SimulationConfig or None
        Explicit metadata.  When provided, overrides any metadata
        found in the HDF5 file.

    Returns
    -------
    FieldDataset
        A lazy-loading dataset backed by virtual references to the
        original HDF5 file.

    Raises
    ------
    ValueError
        If grid metadata cannot be determined from the file or config.
    ImportError
        If ``virtualizarr`` or ``icechunk`` is not installed.

    Notes
    -----
    Virtual references are persisted via Icechunk's native Zarr v3
    backend (in-memory store).  Both ``virtualizarr`` and ``icechunk``
    are installed by the ``zarr`` extra (``pip install "pypic-plasma[zarr]"``).
    """
    ensure_virtualizarr()
    ensure_icechunk()

    from pathlib import Path as _Path

    path_str = str(_Path(path).resolve())
    vds = _build_vds(path_str, fields_group, drop_variables)
    ds = _virtual_to_readable(vds, source_dir=str(_Path(path_str).parent))

    if config is not None:
        grid = config.grid
        normalization = config.normalization
        species = config.species
        physics = config.physics
        metadata: Mapping[str, Any] = config.metadata
        frame = config.frame
        transforms: Mapping[str, FrameTransform] | None = config.transforms
    else:
        h5_grid, h5_norm, h5_extra = _read_metadata_from_h5(path_str)
        if h5_grid is None:
            msg = (
                f"No 'grid' group found in {path_str} and no config provided. "
                "Pass config= with grid metadata, or use a canonical HDF5 layout."
            )
            raise ValueError(msg)
        grid = h5_grid
        normalization = h5_norm if h5_norm is not None else Normalization.undeclared()
        species = None
        physics = None
        metadata = h5_extra
        frame = "simulation"
        transforms = None

    dim_names = list(grid.surviving_axis_names)
    ds_dims = list(ds.dims)
    if len(ds_dims) != len(dim_names):
        msg = (
            f"Dimension count mismatch: HDF5 has {len(ds_dims)} dimensions "
            f"{ds_dims} but grid expects {len(dim_names)} {dim_names}."
        )
        raise ValueError(msg)
    rename_map = dict(zip(ds_dims, dim_names, strict=True))
    ds = ds.rename(rename_map)

    coord_arrays = grid.coordinate_arrays()
    coords = {dim_names[i]: coord_arrays[i] for i in range(len(dim_names))}
    ds = ds.assign_coords(coords)

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


def to_icechunk_virtual(
    source: str | Path,
    output: str | Path,
    *,
    fields_group: str | None = "fields",
    drop_variables: list[str] | None = None,
    config: SimulationConfig | None = None,
    message: str | None = None,
    branch: str = "main",
) -> str:
    r"""Persist HDF5 byte-range references to an Icechunk repository.

    Writes the virtual references for *source*'s field datasets to a
    persistent on-disk Icechunk repo at *output* and attaches pypic
    metadata as group attrs.  Subsequent reads via ``from_zarr(output)``
    resolve chunks by reading byte ranges from *source* — no field
    data is copied.

    Parameters
    ----------
    source : str or Path
        Path to the source HDF5 file.
    output : str or Path
        Destination directory for the Icechunk repository.
    fields_group : str or None
        HDF5 group containing field datasets (``"fields"`` for the
        canonical layout; ``None`` reads from the root group).
    drop_variables : list[str] or None
        HDF5 dataset names to exclude from the virtual view.
    config : SimulationConfig or None
        Explicit metadata.  When provided, overrides any metadata
        found in the HDF5 file.
    message : str or None
        Icechunk commit message.  Defaults to a generated string.
    branch : str
        Branch to commit to.  Defaults to ``"main"``.

    Returns
    -------
    str
        The Icechunk snapshot ID of the new commit.

    Raises
    ------
    ImportError
        If ``virtualizarr`` or ``icechunk`` is not installed.
    ValueError
        If grid metadata cannot be determined from *source* or *config*.

    Notes
    -----
    Both ``virtualizarr`` and ``icechunk`` are installed by the
    ``zarr`` extra (``pip install "pypic-plasma[zarr]"``).  Moving or deleting
    *source* after the write breaks the virtual refs in *output* — the
    on-disk repo is metadata only.
    """
    ensure_virtualizarr()
    ensure_icechunk()

    import shutil
    from pathlib import Path as _Path

    import icechunk
    import zarr

    source_path = _Path(source).resolve()
    source_str = str(source_path)
    source_dir = str(source_path.parent)
    output_path = _Path(output)
    # Decide *before* the mkdir whether the directory is ours to clean up.
    # Same gating as ``to_zarr``; here the realistic trigger is
    # ``open_virtual`` raising because the source HDF5 lacks ``grid/``.
    created_new = not output_path.exists() or (
        output_path.is_dir() and not any(output_path.iterdir())
    )
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        vds = _build_vds(source_str, fields_group, drop_variables)

        storage = icechunk.local_filesystem_storage(str(output_path))
        repo_config, url_prefix = _virtual_repo_config(source_dir)

        # Merge the new virtual chunk container into any config already
        # persisted for this repo.  ``open_or_create`` accepts
        # ``config=`` but does not union it with previously-saved
        # containers — passing only the current source's container
        # would silently displace every prefix from earlier commits, so
        # later reads of those refs fail with "no virtual chunk
        # container can handle the chunk location".
        try:
            persisted = icechunk.Repository.fetch_config(storage)
        except icechunk.IcechunkError:
            # Fresh repo: fetch_config raises before any commit exists.
            persisted = None
        if persisted is not None:
            existing = persisted.virtual_chunk_containers or {}
            if url_prefix not in existing:
                persisted.set_virtual_chunk_container(
                    _make_virtual_chunk_container(url_prefix, source_dir)
                )
            repo_config = persisted

        all_prefixes: dict[str, Any] = dict.fromkeys(
            repo_config.virtual_chunk_containers or {}
        )
        all_prefixes.setdefault(url_prefix, None)

        repo = icechunk.Repository.open_or_create(
            storage,
            config=repo_config,
            authorize_virtual_chunk_access=all_prefixes,
        )
        # Persist the (possibly augmented) config so ``from_zarr`` —
        # which opens the repo without knowing which containers to
        # authorize — can auto-discover every prefix this repo has
        # ever written against.
        repo.save_config()
        _ensure_branch(repo, branch)
        session = repo.writable_session(branch)
        # Clear the session's working-tree root so virtualizarr's
        # to_icechunk can create a fresh root group.  Required for (a)
        # repeat commits to the same branch and (b) new branches
        # forked from a non-empty main — both inherit the prior root
        # group from the branch tip, and virtualizarr's
        # ``Group.from_store`` raises ContainsGroupError on any
        # pre-existing node.  Prior snapshots stay intact in repo
        # history; only this commit's root is replaced.
        session.store.sync_clear()
        # Write virtual refs under ``/fields`` to match the schema-v1.0
        # Zarr layout (schema.md §4.2): field arrays live under the
        # ``/fields`` child group, pypic metadata sits flat on the root
        # group's attrs.  ``from_zarr`` enforces both.
        vds.vz.to_icechunk(session.store, group="fields")

        # Reuse open_virtual to assemble the canonical FieldDataset
        # attrs.  This re-extracts vds against an in-memory store
        # (cheap — only HDF5 metadata is read) and gives
        # encode_pypic_attrs the same FieldDataset shape from_zarr
        # will reconstruct on read.
        fds = open_virtual(
            source,
            fields_group=fields_group,
            drop_variables=drop_variables,
            config=config,
        )
        group = zarr.open_group(session.store, mode="r+")
        for key, value in encode_pypic_attrs(fds).items():
            group.attrs[key] = value

        snapshot: str = session.commit(
            message if message is not None else "pypic: virtual refs"
        )
    except BaseException:
        if created_new:
            shutil.rmtree(output_path, ignore_errors=True)
        raise
    return snapshot
