# Modern I/O

Zarr v3 for field data, Parquet/Arrow for particle data, and Icechunk for
versioned storage. Every backend here is optional — each sits behind its own
extra and raises an install hint rather than an `ImportError` traceback when
the dependency is missing.

| Entry point | Extra | Purpose |
|---|---|---|
| `to_zarr` / `from_zarr` | `zarr` | Single-step field store, Zarr v3 |
| `to_zarr_timeseries` | `zarr` | Multi-step store with a leading `time` dimension |
| `open_virtual` | `zarr` | VirtualiZarr view over existing HDF5, no conversion |
| `to_icechunk_virtual`, `open_icechunk_repo`, `icechunk_create_tag`, `icechunk_ancestry` | `icechunk` | Git-like versioning and ACID commits over Zarr v3 |
| `particles_to_arrow` / `particles_from_arrow` | `arrow` | In-memory `ParticleData` ↔ Arrow table |
| `particles_to_parquet` / `particles_from_parquet` | `arrow` | Single-file particle round-trip |
| `particles_to_dataset` / `particles_from_dataset` | `arrow` | Hive-partitioned dataset, by step and species |
| `query_sql` | `duckdb` | SQL over particle Parquet, with spatial pushdown |

The on-disk layouts are specified in [Schema § 4](../schema.md#4-output-layouts):
§ 4.2 for the Zarr store this module writes, § 4.3 for the Parquet particle
layout. Within each Parquet partition, rows are Morton-ordered over `(x, y, z)`
so row-group statistics support spatial predicate pushdown.

::: pypic.io
