"""DuckDB SQL query interface for partitioned particle Parquet datasets.

DuckDB automatically discovers Hive-partitioned layouts and applies
predicate pushdown, partition pruning, and morsel-driven parallelism.

Requires optional dependencies: ``duckdb>=1.4`` and ``pyarrow>=17.0``.
Install with ``pip install "pypic-plasma[duckdb]"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypic.io._guard import ensure_arrow, ensure_duckdb

_SPECIES_LITERAL = re.compile(r"species\s*=\s*'([^']+)'", re.IGNORECASE)

if TYPE_CHECKING:
    import pyarrow as pa

    from pypic.containers import ParticleData

__all__ = ["query_sql"]


def _lookup_species_meta(path: str | Path, species_name: str) -> dict[str, Any]:
    """Read the full ``pypic`` species payload from any matching fragment.

    Returns the payload dict (``species_index``, optional
    ``species_charge``/``species_mass``/``metadata``) recovered from
    the first matching Parquet fragment's schema metadata.  Returns
    ``{"species_index": 0}`` when no fragment is found — callers then
    produce an ``unknown``-tagged ParticleData.
    """
    import pyarrow.parquet as pq

    from pypic.io._arrow import _decode_species_meta

    for p in Path(path).glob(f"step=*/species={species_name}/*.parquet"):
        return _decode_species_meta(pq.read_metadata(str(p)).metadata)
    return {"species_index": 0}


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

    See Also
    --------
    pypic.io.particles_to_dataset : Writes the partitioned dataset this queries.
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
        # Well-formed filter, zero rows.  Recover the intended species when
        # one species is on disk, or when the SQL pins exactly one via a
        # literal ``species='NAME'``.  Anything else is ambiguous and falls
        # through to the ``unknown`` placeholder.
        on_disk = sorted(
            {p.name.split("=", 1)[1] for p in Path(path).glob("step=*/species=*")}
        )
        arrow_table = _strip_extra_columns(arrow_table)
        recovered_species: str | None = None
        if len(on_disk) == 1:
            recovered_species = on_disk[0]
        else:
            pinned = {m.group(1) for m in _SPECIES_LITERAL.finditer(sql)}
            on_disk_pinned = pinned & set(on_disk)
            if len(on_disk_pinned) == 1:
                recovered_species = next(iter(on_disk_pinned))
        if recovered_species is not None:
            payload = _lookup_species_meta(path, recovered_species)
            # Keep species identity (index / charge / mass) but drop
            # per-step ``metadata``: the fragment scanned is *some* step's
            # schema, not the zero-row step asked about.
            arrow_table = inject_species_meta(
                arrow_table,
                int(payload.get("species_index", 0)),
                recovered_species,
                species_charge=payload.get("species_charge"),
                species_mass=payload.get("species_mass"),
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
    payload = _lookup_species_meta(path, species_name)

    arrow_table = _strip_extra_columns(arrow_table)
    arrow_table = inject_species_meta(
        arrow_table,
        int(payload.get("species_index", 0)),
        species_name,
        species_charge=payload.get("species_charge"),
        species_mass=payload.get("species_mass"),
        metadata=payload.get("metadata"),
    )
    return particles_from_arrow(arrow_table)
