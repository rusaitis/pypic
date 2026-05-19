"""Zarr v3, Parquet/Arrow, virtual HDF5, and Icechunk I/O for pypic.

Field data: ``to_zarr`` / ``from_zarr`` / ``to_zarr_timeseries`` (Zarr v3),
``open_virtual`` (VirtualiZarr), Icechunk versioning.

Particle data: ``particles_to_arrow`` / ``particles_from_arrow`` (Arrow),
``particles_to_parquet`` / ``particles_from_parquet`` (single-file Parquet),
``particles_to_dataset`` / ``particles_from_dataset`` (partitioned Parquet),
``query_sql`` (DuckDB over Parquet).

Optional dependencies per feature:
    ``pip install pypic[zarr]`` — Zarr v3, VirtualiZarr
    ``pip install pypic[icechunk]`` — Icechunk versioned storage
    ``pip install pypic[arrow]`` — Arrow/Parquet particle I/O
    ``pip install pypic[duckdb]`` — DuckDB SQL queries
"""

from pypic.io import metadata
from pypic.io._arrow import particles_from_arrow, particles_to_arrow
from pypic.io._duckdb import query_sql
from pypic.io._icechunk import (
    icechunk_ancestry,
    icechunk_create_tag,
    open_icechunk_repo,
)
from pypic.io._parquet import (
    particles_from_dataset,
    particles_from_parquet,
    particles_to_dataset,
    particles_to_parquet,
)
from pypic.io._virtual import open_virtual, to_icechunk_virtual
from pypic.io.zarr import from_zarr, to_zarr, to_zarr_timeseries

__all__ = [
    "from_zarr",
    "icechunk_ancestry",
    "icechunk_create_tag",
    "metadata",
    "open_icechunk_repo",
    "open_virtual",
    "particles_from_arrow",
    "particles_from_dataset",
    "particles_from_parquet",
    "particles_to_arrow",
    "particles_to_dataset",
    "particles_to_parquet",
    "query_sql",
    "to_icechunk_virtual",
    "to_zarr",
    "to_zarr_timeseries",
]
