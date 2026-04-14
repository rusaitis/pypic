"""DuckDB SQL query interface for partitioned particle Parquet datasets.

DuckDB automatically discovers Hive-partitioned layouts and applies
predicate pushdown, partition pruning, and morsel-driven parallelism.

Requires optional dependencies: ``duckdb>=1.4`` and ``pyarrow>=17.0``.
Install with ``pip install pypic[duckdb]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pypic.io._guard import ensure_arrow, ensure_duckdb

if TYPE_CHECKING:
    import pyarrow as pa

    from pypic.containers import ParticleData

__all__ = ["query_sql"]


def _lookup_species_meta(
    path: str | Path, species_name: str
) -> tuple[int, float | None, float | None]:
    """Read species_index/charge/mass from any matching Parquet fragment.

    Returns ``(0, None, None)`` when no fragment is found — callers then
    produce an ``unknown``-tagged ParticleData.
    """
    import pyarrow.parquet as pq

    from pypic.io._arrow import _decode_species_meta

    for p in Path(path).glob(f"step=*/species={species_name}/*.parquet"):
        payload = _decode_species_meta(pq.read_metadata(str(p)).metadata)
        return (
            int(payload.get("species_index", 0)),
            payload.get("species_charge"),
            payload.get("species_mass"),
        )
    return (0, None, None)


def query_sql(
    path: str | Path,
    sql: str,
    *,
    return_type: str = "particledata",
) -> ParticleData | pa.Table:
    r"""Execute SQL over a partitioned particle Parquet dataset via DuckDB.

    The dataset is available in the query as the ``particles`` view.
    DuckDB automatically handles Hive partition discovery, predicate
    pushdown, and parallel scanning.

    Parameters
    ----------
    path : str or Path
        Root directory of the partitioned Parquet dataset.
    sql : str
        SQL query.  The dataset is available as ``particles``.
        Example: ``"SELECT * FROM particles WHERE step='000100' AND x > 5"``
    return_type : str
        ``"particledata"`` (default) converts the result to
        ``ParticleData``.  ``"arrow"`` returns a raw ``pyarrow.Table``.

    Returns
    -------
    ParticleData or pyarrow.Table

    Examples
    --------
    >>> # pcl = query_sql("/tmp/particles", "SELECT * FROM particles LIMIT 10")
    """
    ensure_duckdb()
    ensure_arrow()
    import duckdb

    from pypic.io._arrow import (
        inject_species_meta,
        particles_from_arrow,
    )
    from pypic.io._parquet import _strip_extra_columns

    glob_pattern = str(Path(path) / "**" / "*.parquet")
    escaped = glob_pattern.replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE VIEW particles AS "
            f"SELECT * FROM parquet_scan('{escaped}', "
            f"hive_partitioning=true)"
        )
        result = con.execute(sql)
        arrow_table = result.arrow().read_all()
    finally:
        con.close()

    if return_type == "arrow":
        return arrow_table

    # ParticleData is single-species; DuckDB strips Parquet schema
    # metadata, so recover species_index/charge/mass from one matching
    # Parquet fragment after confirming the SQL result hits exactly
    # one species.  The caller must keep the `species` partition column
    # in the projection (SELECT * does by default); cross-species and
    # aggregate queries should use return_type="arrow".
    if "species" not in arrow_table.column_names:
        msg = (
            "query_sql(return_type='particledata') requires the 'species' "
            "partition column in the result.  Keep `species` in the SELECT "
            "projection, or use return_type='arrow' for scalar / aggregate "
            "queries."
        )
        raise ValueError(msg)

    cols = set(arrow_table.column_names)
    has_position = {"x", "y", "z"}.issubset(cols)
    has_velocity = {"vx", "vy", "vz"}.issubset(cols)
    if not (has_position or has_velocity):
        msg = (
            "query_sql(return_type='particledata') requires a full "
            "position triplet (x, y, z) or velocity triplet (vx, vy, vz) "
            f"in the projection.  Got columns: {sorted(cols)}.  "
            "Use return_type='arrow' for scalar or aggregate queries."
        )
        raise ValueError(msg)

    species_values = {
        v for v in arrow_table.column("species").unique().to_pylist() if v is not None
    }
    if not species_values:
        # Well-formed filter that matched zero rows.  If the dataset
        # has exactly one species on disk, adopt its metadata — empty
        # reads then preserve species identity the same way
        # ``particles_from_dataset(species=...)`` does.  Multi-species
        # datasets remain genuinely ambiguous (we can't recover which
        # species the WHERE clause asked for), so fall back to the
        # placeholder so callers can still dispatch on ``n_particles``.
        on_disk = sorted(
            {p.name.split("=", 1)[1] for p in Path(path).glob("step=*/species=*")}
        )
        arrow_table = _strip_extra_columns(arrow_table)
        if len(on_disk) == 1:
            species_name = on_disk[0]
            species_index, species_charge, species_mass = _lookup_species_meta(
                path, species_name
            )
            arrow_table = inject_species_meta(
                arrow_table,
                species_index,
                species_name,
                species_charge=species_charge,
                species_mass=species_mass,
            )
        else:
            arrow_table = inject_species_meta(arrow_table, 0, "unknown")
        return particles_from_arrow(arrow_table)
    if len(species_values) > 1:
        names = sorted(species_values)
        msg = (
            f"query_sql matched {len(names)} species ({', '.join(names)}); "
            "ParticleData is a single-species container.  Filter with "
            "WHERE species='...' or use return_type='arrow'."
        )
        raise ValueError(msg)
    species_name = next(iter(species_values))
    species_index, species_charge, species_mass = _lookup_species_meta(
        path, species_name
    )

    arrow_table = _strip_extra_columns(arrow_table)
    arrow_table = inject_species_meta(
        arrow_table,
        species_index,
        species_name,
        species_charge=species_charge,
        species_mass=species_mass,
    )
    return particles_from_arrow(arrow_table)
