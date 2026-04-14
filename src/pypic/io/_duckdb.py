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
        _decode_species_meta,
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
    if "species" not in arrow_table.column_names or len(arrow_table) == 0:
        msg = (
            "query_sql(return_type='particledata') requires the 'species' "
            "partition column in the result.  Keep `species` in the SELECT "
            "projection, or use return_type='arrow' for scalar / aggregate "
            "queries."
        )
        raise ValueError(msg)

    species_values = {
        v for v in arrow_table.column("species").unique().to_pylist() if v is not None
    }
    if len(species_values) > 1:
        names = sorted(species_values)
        msg = (
            f"query_sql matched {len(names)} species ({', '.join(names)}); "
            "ParticleData is a single-species container.  Filter with "
            "WHERE species='...' or use return_type='arrow'."
        )
        raise ValueError(msg)
    species_name = next(iter(species_values))

    # Read one Parquet fragment from the matching partition for the
    # authoritative species_index/charge/mass payload.
    import pyarrow.parquet as pq

    species_index = 0
    species_charge: float | None = None
    species_mass: float | None = None
    for p in Path(path).glob(f"step=*/species={species_name}/*.parquet"):
        payload = _decode_species_meta(pq.read_metadata(str(p)).metadata)
        species_index = int(payload.get("species_index", 0))
        species_charge = payload.get("species_charge")
        species_mass = payload.get("species_mass")
        break

    arrow_table = _strip_extra_columns(arrow_table)
    arrow_table = inject_species_meta(
        arrow_table,
        species_index,
        species_name,
        species_charge=species_charge,
        species_mass=species_mass,
    )
    return particles_from_arrow(arrow_table)
