"""Parquet I/O for ParticleData — single-file and partitioned datasets.

Single-file:
    ``particles_to_parquet`` / ``particles_from_parquet``

Partitioned dataset (multi-step, multi-species, Hive layout):
    ``particles_to_dataset`` / ``particles_from_dataset``

Particles are Morton Z-order sorted before writing so that Parquet
row-group min/max statistics on ``x``/``y``/``z`` form tight spatial
bounding boxes.  This enables ``pyarrow.dataset`` predicate pushdown
to skip 95%+ of row groups for spatial box queries.

Requires optional dependency: ``pyarrow>=17.0``.
Install with ``pip install pypic[arrow]``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pypic.io._arrow import (
    inject_species_meta,
    particles_from_arrow,
    particles_to_arrow,
)
from pypic.io._guard import ensure_arrow
from pypic.io._morton import morton_sort_indices

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import pyarrow as pa

    from pypic.containers import ParticleData
    from pypic.readers._registry import Simulation

__all__ = [
    "particles_from_dataset",
    "particles_from_parquet",
    "particles_to_dataset",
    "particles_to_parquet",
]

_log = logging.getLogger(__name__)

_DEFAULT_ROW_GROUP_SIZE: int = 750_000
_PARTITION_COLS = ("step", "species")
_DERIVED_COLS = ("speed",)
_STRIP_COLS = _PARTITION_COLS + _DERIVED_COLS


def _add_speed_column(table: pa.Table) -> pa.Table:
    """Append a ``speed`` column (|v|) to the table."""
    import pyarrow.compute as pc

    if not all(c in table.column_names for c in ("vx", "vy", "vz")):
        return table
    vx = table.column("vx")
    vy = table.column("vy")
    vz = table.column("vz")
    speed = pc.sqrt(
        pc.add(pc.add(pc.multiply(vx, vx), pc.multiply(vy, vy)), pc.multiply(vz, vz))
    )
    return table.append_column("speed", speed)


def _morton_sort_table(table: pa.Table) -> pa.Table:
    """Reorder an Arrow table by Morton Z-order on x/y/z."""
    import pyarrow as pa

    if not all(c in table.column_names for c in ("x", "y", "z")):
        return table
    x = table.column("x").to_numpy(zero_copy_only=False)
    y = table.column("y").to_numpy(zero_copy_only=False)
    z = table.column("z").to_numpy(zero_copy_only=False)
    idx = morton_sort_indices(x, y, z)
    return table.take(pa.array(idx))


def _column_sort_table(table: pa.Table, column: str) -> pa.Table:
    """Reorder an Arrow table by ascending values of *column*.

    Use when the dataset will be queried by ``column`` as a particle
    tracking ID — sorting clusters identical/nearby values into the
    same row groups, making row-group min/max statistics tight enough
    for predicate pushdown to skip non-matching groups.
    """
    import pyarrow.compute as pc

    if column not in table.column_names:
        return table
    idx = pc.sort_indices(table, sort_keys=[(column, "ascending")])
    return table.take(idx)


def _strip_extra_columns(table: pa.Table) -> pa.Table:
    """Remove partition and derived columns before converting to ParticleData."""
    drop = [c for c in _STRIP_COLS if c in table.column_names]
    return table.drop(drop) if drop else table


def particles_to_parquet(
    data: ParticleData,
    path: str | Path,
    *,
    position_dtype: str | None = None,
    velocity_dtype: str | None = None,
    compression_level: int = 1,
    row_group_size: int = _DEFAULT_ROW_GROUP_SIZE,
    sort_by: Literal["position", "charge", "weight"] = "position",
) -> None:
    r"""Write a single ``ParticleData`` to a Parquet file.

    Particles are sorted before writing so that Parquet row-group
    min/max statistics enable predicate pushdown on read.  The default
    ``sort_by="position"`` uses a Morton Z-order curve, optimal for
    spatial box queries.  Use ``sort_by="charge"`` to optimize for
    particle tracking when charge serves as the per-particle ID
    (iPIC3D non-uniform plasma); spatial queries become slower in
    return.

    Parameters
    ----------
    data : ParticleData
        Source particle data (single species, single timestep).
    path : str or Path
        Destination ``.parquet`` file.
    position_dtype : str or None
        Downcast position columns (e.g. ``"float32"``).
    velocity_dtype : str or None
        Downcast velocity columns (e.g. ``"float32"``).
    compression_level : int
        zstd compression level (1 for processing, 3 for archival).
    row_group_size : int
        Target rows per row group (500K--1M recommended).
    sort_by : {"position", "charge", "weight"}
        Pre-write sort order.  ``"position"`` (default) Morton-sorts
        on x/y/z for spatial pushdown.  ``"charge"`` or ``"weight"``
        ascending-sorts on that column, enabling row-group statistics
        pushdown for particle tracking by charge or weight.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.containers import ParticleData
    >>> pcl = ParticleData(
    ...     species_index=0, species_name="e",
    ...     position=np.zeros((5, 3)), velocity=np.ones((5, 3)),
    ...     charge=np.full(5, -1.0), n_particles=5, metadata={},
    ... )
    >>> # particles_to_parquet(pcl, "/tmp/pcl.parquet")
    """
    ensure_arrow()
    import pyarrow.parquet as pq

    table = particles_to_arrow(
        data, position_dtype=position_dtype, velocity_dtype=velocity_dtype
    )
    table = _add_speed_column(table)
    if sort_by == "position":
        table = _morton_sort_table(table)
    elif sort_by in ("charge", "weight"):
        table = _column_sort_table(table, sort_by)
    else:
        msg = f"sort_by must be 'position', 'charge', or 'weight', got {sort_by!r}"
        raise ValueError(msg)

    pq.write_table(
        table,
        str(path),
        compression="zstd",
        compression_level=compression_level,
        use_byte_stream_split=True,
        row_group_size=row_group_size,
    )
    _log.info(
        "Wrote %d particles (%s) to %s", data.n_particles, data.species_name, path
    )


def particles_from_parquet(path: str | Path) -> ParticleData:
    r"""Read a single Parquet file into a ``ParticleData``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.parquet`` file.

    Returns
    -------
    ParticleData

    Examples
    --------
    >>> # pcl = particles_from_parquet("/tmp/pcl.parquet")
    """
    ensure_arrow()
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    table = _strip_extra_columns(table)
    return particles_from_arrow(table)


def _resolve_species_list(
    source: Simulation,
    species: Sequence[int | str] | None,
) -> list[tuple[int, str]]:
    """Return ``[(index, name), ...]`` for the requested species."""
    all_species = source.config.species
    if species is None:
        return [(i, sp.name) for i, sp in enumerate(all_species)]
    result: list[tuple[int, str]] = []
    for s in species:
        if isinstance(s, int):
            if s < len(all_species):
                result.append((s, all_species[s].name))
            else:
                msg = (
                    f"Species index {s} out of range (have {len(all_species)} species)"
                )
                raise IndexError(msg)
        else:
            for i, sp in enumerate(all_species):
                if sp.name == s:
                    result.append((i, sp.name))
                    break
            else:
                msg = f"Species name {s!r} not found in simulation config"
                raise KeyError(msg)
    return result


def _resolve_particle_pairs(
    source: Simulation | Iterable[tuple[int, str, ParticleData]],
    steps: Sequence[int] | None,
    species: Sequence[int | str] | None,
) -> Iterable[tuple[int, str, ParticleData]]:
    """Normalize a particle source into an iterable of (step, species_name, data).

    Mirrors ``_resolve_timeseries_pairs`` in ``zarr.py`` for fields.
    Uses duck typing on ``particle_steps`` to detect Simulation-like
    sources so test mocks and other adapters work without subclassing.
    """
    if hasattr(source, "particle_steps") and hasattr(source, "particles"):
        sim: Any = source
        step_list = list(steps) if steps is not None else sim.particle_steps
        species_list = _resolve_species_list(sim, species)

        def _iter_sim() -> Iterable[tuple[int, str, ParticleData]]:
            for step in step_list:
                for sp_idx, sp_name in species_list:
                    pcl = sim.particles(step, sp_idx)
                    if pcl.n_particles > 0:
                        yield step, sp_name, pcl

        return _iter_sim()

    if steps is not None or species is not None:
        msg = "steps= and species= only apply when source is a Simulation"
        raise ValueError(msg)
    return source


def particles_to_dataset(
    source: Simulation | Iterable[tuple[int, str, ParticleData]],
    path: str | Path,
    *,
    steps: Sequence[int] | None = None,
    species: Sequence[int | str] | None = None,
    position_dtype: str | None = None,
    velocity_dtype: str | None = None,
    compression_level: int = 1,
    row_group_size: int = _DEFAULT_ROW_GROUP_SIZE,
    sort_by: Literal["position", "charge", "weight"] = "position",
) -> None:
    r"""Write a partitioned Parquet dataset from a Simulation or iterable.

    Layout::

        {path}/step=000000/species=electrons/part-00000.parquet
        {path}/step=000000/species=ions/part-00000.parquet
        {path}/step=000100/species=electrons/part-00000.parquet
        ...

    Parameters
    ----------
    source : Simulation or Iterable[tuple[int, str, ParticleData]]
        Either a ``Simulation`` (reads via ``source.particles``) or an
        iterable of ``(step, species_name, ParticleData)`` tuples for
        custom HDF5→Parquet pipelines.  When passing an iterable,
        ``steps`` and ``species`` must be ``None``.
    path : str or Path
        Root directory for the partitioned dataset.
    steps : Sequence[int] or None
        Timestep indices to write (Simulation source only).  Defaults
        to all particle steps.
    species : Sequence[int | str] or None
        Species indices or names (Simulation source only).  Defaults
        to all species.
    position_dtype, velocity_dtype : str or None
        Downcast options.
    compression_level : int
        zstd level (1 for processing, 3 for archival).
    row_group_size : int
        Target rows per row group.
    sort_by : {"position", "charge"}
        Pre-write sort order; forwarded to ``particles_to_parquet``.

    Examples
    --------
    >>> # particles_to_dataset(sim, "/tmp/particles")
    >>> # Or with an iterable for custom pipelines:
    >>> # pairs = [(0, "electrons", pcl_e), (0, "ions", pcl_i)]
    >>> # particles_to_dataset(pairs, "/tmp/particles")
    """
    ensure_arrow()
    root = Path(path)
    pairs = _resolve_particle_pairs(source, steps, species)

    n_written = 0
    for step, sp_name, pcl in pairs:
        if pcl.n_particles == 0:
            _log.debug("Skipping empty step=%d species=%s", step, sp_name)
            continue
        part_dir = root / f"step={step:06d}" / f"species={sp_name}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_path = part_dir / "part-00000.parquet"
        particles_to_parquet(
            pcl,
            part_path,
            position_dtype=position_dtype,
            velocity_dtype=velocity_dtype,
            compression_level=compression_level,
            row_group_size=row_group_size,
            sort_by=sort_by,
        )
        n_written += 1
    _log.info("Wrote partitioned dataset (%d files) to %s", n_written, root)


def particles_from_dataset(
    path: str | Path,
    *,
    step: int | None = None,
    species: str | int | None = None,
    spatial_box: (
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    ) = None,
    ids: np.ndarray | Sequence[int] | Sequence[float] | None = None,
    id_column: str = "id",
    energy_min: float | None = None,
    columns: Sequence[str] | None = None,
) -> ParticleData:
    r"""Read from a partitioned Parquet dataset with selective loading.

    Uses ``pyarrow.dataset`` with Hive partitioning for partition
    pruning (step, species) and Parquet row-group statistics for
    predicate pushdown (spatial box, energy threshold, IDs).

    Parameters
    ----------
    path : str or Path
        Root directory of the partitioned dataset.
    step : int or None
        Timestep to load (partition pruning).
    species : str or int or None
        Species name or index (partition pruning).
    spatial_box : tuple or None
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` for
        spatial filtering via predicate pushdown.
    ids : array or Sequence or None
        Values to filter on against ``id_column``.  Default
        ``id_column="id"`` matches the integer tracking column;
        pass ``id_column="charge"`` with float64 values to track
        iPIC3D particles by their (per-particle) charge/weight.
    id_column : str
        Column name to filter ``ids`` against.  Default ``"id"``.
        For best pushdown effectiveness when ``id_column != "id"``,
        write the dataset with matching ``sort_by`` (e.g.
        ``sort_by="charge"`` for ``id_column="charge"``).
    energy_min : float or None
        Minimum speed ``|v|`` threshold for energy filtering.
    columns : Sequence[str] or None
        Column names to load (e.g. ``["x", "y", "z"]``).  ``charge``
        and ``id_column`` are always included.

    Returns
    -------
    ParticleData

    Examples
    --------
    >>> # pcl = particles_from_dataset("/tmp/particles", step=0, species="electrons")
    """
    ensure_arrow()
    # Declare partition schema explicitly so step is read as string, not int
    import pyarrow as pa
    import pyarrow.dataset as pads

    part_schema = pa.schema(
        [
            pa.field("step", pa.string()),
            pa.field("species", pa.string()),
        ]
    )
    partitioning = pads.HivePartitioning(part_schema)
    dataset = pads.dataset(str(path), format="parquet", partitioning=partitioning)

    # Partition filter
    part_filter: Any = None
    if step is not None:
        step_str = f"{step:06d}"
        part_filter = pads.field("step") == step_str
    if species is not None:
        species_str = _resolve_species_str(species, Path(path))
        sp_filter = pads.field("species") == species_str
        part_filter = sp_filter if part_filter is None else part_filter & sp_filter

    # Row-level filter (predicate pushdown on Parquet statistics)
    row_filter: Any = None
    if spatial_box is not None:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = spatial_box
        row_filter = (
            (pads.field("x") >= x_min)
            & (pads.field("x") <= x_max)
            & (pads.field("y") >= y_min)
            & (pads.field("y") <= y_max)
            & (pads.field("z") >= z_min)
            & (pads.field("z") <= z_max)
        )
    if energy_min is not None:
        speed_filter = pads.field("speed") >= energy_min
        row_filter = speed_filter if row_filter is None else row_filter & speed_filter
    if ids is not None:
        id_list = list(ids) if not isinstance(ids, list) else ids
        id_filter = pads.field(id_column).isin(id_list)
        row_filter = id_filter if row_filter is None else row_filter & id_filter

    combined_filter: Any = None
    if part_filter is not None and row_filter is not None:
        combined_filter = part_filter & row_filter
    elif part_filter is not None:
        combined_filter = part_filter
    elif row_filter is not None:
        combined_filter = row_filter

    # Column pruning — only force-include the active id_column when filtering
    read_columns: list[str] | None = None
    if columns is not None:
        read_columns = list(columns)
        if ids is not None and id_column not in read_columns:
            read_columns.append(id_column)

    table = dataset.to_table(filter=combined_filter, columns=read_columns)

    # Extract species info from partition columns before stripping
    species_name = "unknown"
    species_index = 0
    if "species" in table.column_names and len(table) > 0:
        species_name = str(table.column("species")[0].as_py())
    if isinstance(species, str):
        species_name = species
    elif isinstance(species, int):
        species_index = species

    # Recover scalar species_charge/species_mass from the per-file
    # Parquet schema metadata (pyarrow.dataset's unified schema doesn't
    # merge per-file metadata).  Read from the first matching fragment.
    species_charge_meta = None
    species_mass_meta = None
    try:
        first_frag = next(iter(dataset.get_fragments(filter=combined_filter)))
        file_meta = first_frag.physical_schema.metadata
        if file_meta and b"pypic" in file_meta:
            import json as _json

            payload = _json.loads(file_meta[b"pypic"])
            species_charge_meta = payload.get("species_charge")
            species_mass_meta = payload.get("species_mass")
    except (StopIteration, KeyError, ValueError):
        pass

    table = _strip_extra_columns(table)
    table = inject_species_meta(
        table,
        species_index,
        species_name,
        species_charge=species_charge_meta,
        species_mass=species_mass_meta,
    )
    return particles_from_arrow(table)


def _resolve_species_str(species: str | int, root: Path) -> str:
    """Convert a species argument to the string used in partition keys.

    For string arguments, returns directly.  For integer indices,
    discovers species names from the Hive directory structure
    (stable across pyarrow versions, unlike expression repr parsing).
    """
    if isinstance(species, str):
        return species
    species_dirs: set[str] = set()
    prefix = "species="
    for p in root.glob("step=*/species=*"):
        if p.is_dir() and p.name.startswith(prefix):
            species_dirs.add(p.name[len(prefix) :])
    species_sorted = sorted(species_dirs)
    if species >= len(species_sorted):
        msg = (
            f"Species index {species} out of range"
            f" (found {len(species_sorted)} species)"
        )
        raise IndexError(msg)
    return species_sorted[species]
