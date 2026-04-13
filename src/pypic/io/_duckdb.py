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

    from pypic.io._arrow import particles_from_arrow
    from pypic.io._parquet import _strip_extra_columns

    glob_pattern = str(Path(path) / "**" / "*.parquet")
    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE VIEW particles AS "
            f"SELECT * FROM parquet_scan('{glob_pattern}', "
            f"hive_partitioning=true)"
        )
        result = con.execute(sql)
        arrow_table = result.arrow().read_all()
    finally:
        con.close()

    if return_type == "arrow":
        return arrow_table

    # DuckDB doesn't preserve Parquet schema metadata, so extract
    # species info from partition columns before stripping them.
    import json

    species_name = "unknown"
    species_index = 0
    if "species" in arrow_table.column_names and len(arrow_table) > 0:
        species_name = str(arrow_table.column("species")[0].as_py())

    arrow_table = _strip_extra_columns(arrow_table)

    meta = arrow_table.schema.metadata or {}
    meta[b"pypic"] = json.dumps(
        {
            "species_index": species_index,
            "species_name": species_name,
            "n_particles": len(arrow_table),
        }
    ).encode("utf-8")
    arrow_table = arrow_table.replace_schema_metadata(meta)

    return particles_from_arrow(arrow_table)
