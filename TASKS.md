# pypic — Implementation Roadmap

Each step produces something testable. No step starts until the previous step's tests pass.

---

## Completed

- [x] **Step 1:** Project skeleton (pyproject.toml, uv, ruff, pytest, mypy, CI)
- [x] **Step 2:** Normalization class (pic_standard, pic_electron, mhd_standard, identity)
- [x] **Step 3:** PhysicsConstants, SpeciesInfo
- [x] **Step 4:** CoordinateGeometry, GeometryType, metric_factors
- [x] **Step 5:** FieldDataset, GridInfo, SimulationReader, SimulationConfig
- [x] **Step 6:** simulation.toml loader
- [x] **Step 7:** Derived part 1 — |B|, |E|, |J|, |V|, beta, v_A, Poynting, energies, entropy
- [x] **Step 8:** Derived part 2 — characteristic scales (v_th, omega_p, d_i, r_i, lambda_D, c_s, M_A)
- [x] **Step 8b:** Per-species pressure decomposition in compute registry
- [x] **Step 9:** Diagnostics — L2/Linf error, div B/E, field energy
- [x] **Step 10:** Differential operators — curl, div, grad (Cartesian)
- [x] **Step 11:** PlaneSelection, BoxSelection
- [x] **Step 12:** Readers — iPIC3D (parallel/serial/H5hut), BATSRUS (IDL/HDF5), OpenGGCM, SimpleReader + registry + auto-detection
- [x] **Step 13:** compute(), in_si(), in_units(), QuantityType, field metadata registry
- [x] **Step 14:** 2D plotting — plot_field_slice, plot_comparison, publication styles
- [x] **Step 15:** Frame transforms — ReferenceFrame, FrameTransform, chaining, transform_to
- [x] **Step 16:** SphereSelection (NaN masking) + FieldDataset.where()
- [x] **Step 17:** MkDocs documentation site
- [x] **Step 18:** Relativistic — lorentz_factor, magnetization, ~9 functions with rel. corrections
- [x] **Step 19:** Regrid — regrid(), align_grids(), common_grid() (Cartesian)
- [x] **Step 20:** Cross-grid comparison — compare_fields(), field_comparison_report(), field_difference_dataset()
- [x] **Step 21b:** available_fields() + available_fields_mapping()
- [x] **Step 21:** CLI — info, fields, stats, compare, validate (typer + rich)
- [x] **Step 22:** CLI — plot, plot-compare (themes, contours, animate, batch)
- [x] **Step 30:** Reduced geometry after slicing
- [~] **Step 31:** Remove default geometry from operators — deferred (revisit when non-Cartesian operators land)
- [x] **Step 32:** Separate four_velocity quantity type
- [x] **Step 33:** specific_energy quantity type for enthalpy (fixed dimensional bug)
- [x] **Step 34:** StaggerInfo provenance metadata

**Milestone: daily-use tool** — load data → compute derived quantities → compare runs → select subregions → convert units → make paper figures. ✅

---

## Phase 8: Modern I/O Formats

- [x] **Step 24: `pypic.io` — Zarr export/import for FieldDataset**
  Zarr v3 + xarray DataTree for chunked, self-describing field data
  storage.  Layout v1 (post-2026.05): fields under ``/fields`` (mirroring
  schema.md §4.1's HDF5 grouping), pypic metadata as flat keys at the root
  group's attrs (``grid``, ``normalization``, ``physics``, ...), with a
  ``pypic_layout: "v1"`` discriminator.  Cross-language consumers
  (JS/Rust) can read the layout without going through pypic.
  Consolidated metadata (``consolidated=True`` on writes,
  ``consolidated="auto"`` on reads) gives a one-shot metadata fetch.
  Two write modes:
  - `to_zarr(fds, path)` — single timestep. Builds an `xr.DataTree`
    from `{"fields": fds.xr}`, stamps root attrs from
    `encode_pypic_attrs(fds)`, calls `tree.to_zarr(...)`.
  - `to_zarr_timeseries(simulation, path, *, steps, fields)` — multi-timestep store with `time` as a dimension. Each field becomes `(nt, nx, ny, nz)`, chunked along `time` so reading one step is O(1). Enables time-series analysis without scanning separate files.
  `from_zarr(path) -> FieldDataset` reconstructs everything including per-field metadata. Returns lazy-loading dataset by default (`xr.open_zarr` is lazy — reading one field doesn't touch others). For multi-variable stores, async concurrent metadata fetching via `zarr.config.set({'async.concurrency': 128})` delivers up to 14× speedup.
  **Naming:** Canonical numbered names (`B_1`, `B_2`, `B_3`) in the stored format, not geometry-specific (`Bx`, `Br`). Geometry is in metadata; aliases resolve on load. Consistent with HDF5 layout (schema.md § 4).
  **Field metadata:** Already self-describing via xarray DataArray attrs (`quantity_type`, `si_unit`, `long_name`, `latex`, `units`), set by `from_arrays()` and `with_field()`. `xr.Dataset.to_zarr()` serializes attrs automatically. `from_zarr()` reconstructs `FieldDataset` including per-field metadata. No CF vocabulary (CF has no plasma physics coverage).
  **FrameTransforms:** Serialize origin, rotation matrix, and scale as arrays in metadata. Skip callable-based transforms; reconstruct on load.
  **Precision:** `dtype="float32"` kwarg on `to_zarr()` / `to_zarr_timeseries()` downcasts all field arrays to single precision on write (halves storage). Most PIC codes write single-precision dumps anyway; float64→float32 loses ~7 decimal digits, well below PIC numerical accuracy. Default: preserve source dtype. Implemented via xarray's `encoding` dict — user can also pass `encoding=` directly for per-field control.
  **Compression:** Default `BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')` — the standalone `ZstdCodec` lacks shuffle pre-filtering and compresses floats poorly. Blosc2 + zstd + bitshuffle achieves 10–300× on smooth electromagnetic field data due to high spatial correlation. The newer bytedelta filter (Blosc2 2.8+) is an emerging improvement over bitshuffle (37% better on pressure-type data) — expose as an option once stable. User-configurable via `encoding=` passthrough to xarray.
  **Sharding (cloud):** For cloud-hosted stores (S_3, GCS, R2), enable sharding to group chunks into single storage objects, avoiding the small-files problem. Shards are the minimum write unit — the entire shard must fit in memory. Dask chunks must align with shard boundaries. Expose via `shards=` kwarg.
  **Version pinning:** `zarr>=3.1.0,<4` — versions 3.0.0–3.0.7 were yanked from PyPI due to a data-loss bug (append mode silently deleted data). v3.0.8 is the first safe release; v3.1+ is recommended. Requires `numcodecs>=0.16.0` (fixes BloscCodec defaulting to `typesize=1`, which produced 10–20× larger chunks). Optional deps under `zarr` extra.

- [x] **Step 24b: `pypic.io` — VirtualiZarr for legacy HDF5**
  `open_virtual(path) -> FieldDataset` creates lightweight virtual Zarr views over existing HDF5 simulation outputs by extracting byte-range metadata, enabling `xr.open_zarr()` access that transparently reads from original files without conversion. Uses VirtualiZarr v2.4+ (`open_virtual_dataset()`, standard `xr.concat`/`merge`). Virtual references can be persisted to Icechunk (Step 24c) for repeated fast access. Limitations: inherits source file chunking (contiguous HDF5 datasets become single chunks), potential issues with non-standard HDF5 compression filters. Optional dep: `virtualizarr>=2.4` under `zarr` extra.
  **Depends on:** Step 24 (Zarr foundations).

- [x] **Step 24c: `pypic.io` — Icechunk storage backend**
  Optional Git-like versioning and ACID transactions over Zarr v3 stores via Icechunk. Rust-based I/O backend achieves 13–14 Gbps read/write throughput on cloud instances (2–10× faster than zarr + s3fs). Value for pypic: tag dataset versions for reproducibility (`repo.create_tag("v1.0-paper-submission", snapshot_id=...)`), time-travel to prior analysis states, and Rust-accelerated I/O even for non-versioned workflows. `to_zarr(..., backend="icechunk")` writes to an Icechunk-managed store; `from_zarr()` auto-detects Icechunk stores. Optional dep: `icechunk>=1.1` under `icechunk` extra.
  **Migrate `open_virtual` to Zarr v3:** Step 24b's `open_virtual` currently uses Kerchunk (Zarr v2 format) as the virtual-reference intermediary because VirtualiZarr's native Zarr v3 virtual backend is Icechunk. Once Icechunk is available, switch `open_virtual` to persist virtual refs via `vds.vz.to_icechunk()` instead of `vds.vz.to_kerchunk()`, eliminating the only Zarr v2 code path in pypic and fixing the fill-value edge case (datasets where all values equal the fill value read back incorrectly through Kerchunk).
  **Depends on:** Step 24 (Zarr foundations).

- [x] **Step 25: `pypic.io` — Parquet/Arrow for ParticleData**
  Two-tier API for particle I/O, designed for billion-particle datasets with selective reads.
  **Low-level (in-memory interchange):**
  `particles_to_arrow(data) -> pyarrow.Table` (zero-copy NumPy→Arrow), `particles_from_arrow(table) -> ParticleData`. Columnar storage: `x/y/z/vx/vy/vz/charge/id` columns (letter names — particle positions are always in the simulation Cartesian frame). Species metadata in Arrow schema metadata. Used by the Starlette server (Step 37) for Arrow IPC over WebSocket streaming.
  **High-level (partitioned dataset for large-scale I/O):**
  `particles_to_parquet(data, path)` — single species/step file. `particles_to_dataset(simulation, path, *, steps, species)` — multi-step partitioned Parquet dataset with Hive-style layout:
  ```
  particles/step=000000/species=electrons/part-00000.parquet
  ```
  **Selection support** via `particles_from_dataset(path, *, step, species, spatial_box, ids, energy_min, columns)`:
  - *Time*: partition pruning — reading step 100 opens only `step=000100/`.
  - *Species*: partition pruning — reading electrons skips ion files.
  - *Spatial region*: particles sorted by Morton (Z-order) curve within each partition. Morton confirmed over Hilbert — <5% locality difference at fine row-group granularity, 5–8× cheaper to compute (bit-interleaving via BMI2). Parquet row-group min/max statistics on x/y/z enable predicate pushdown; spatial box queries skip 95%+ of row groups.
  - *Energy*: `|v|` stored as a derived column at write time. Parquet statistics enable energy-threshold pushdown.
  - *Particle ID*: predicate pushdown on `id` column. Optional secondary index file (`id → step/row_group`) for trajectory reconstruction across timesteps.
  - *Column pruning*: `columns=["x", "y", "z"]` reads only requested columns.
  **Row groups:** 500K–1M rows per row group (balances metadata overhead vs skip granularity).
  **Compression:** zstd with shuffle pre-filter. Expect 1.5–3× lossless on particle data (noisy positions/velocities lack the spatial correlation that gives field data 10–300×). Always apply shuffle before compression — without it, LZ4 achieves ~1× and zstd only 5–8× on raw floats. Use zstd level 1 for processing, level 3 for archival.
  **Precision:** `position_dtype="float32"` and `velocity_dtype="float32"` kwargs downcast position/velocity columns to single precision on write. Charge stays float64 (full mantissa serves as unique particle identifier) and id stays int64. The `|v|` derived column follows the velocity dtype. Default: preserve source dtype.
  **DuckDB query engine (optional):** `query_sql(path, sql) -> ParticleData | pa.Table` provides SQL access to partitioned particle datasets via DuckDB. Automatic predicate pushdown, partition pruning, and morsel-driven parallelism — 45ms filtered counts where Pandas takes 7.5s, ~1.3 GB memory on 140 GB datasets. Pattern: DuckDB for interactive exploration and ad-hoc spatial queries, PyArrow dataset API for programmatic pipelines. DuckDB queries Arrow tables with zero-copy; results export as Arrow or NumPy. Optional dep: `duckdb>=1.4` under `duckdb` extra.
  `particles_from_parquet(path) -> ParticleData` for single-file full load. Optional dep: `pyarrow>=17.0` under `arrow` extra.
  **Evaluated and rejected:** Lance (1.1× compression vs Parquet's 3×+, AI/ML-focused ecosystem, no browser reader), GeoParquet (WKB encoding overhead, 2D-biased tooling, ~3× larger files than plain Parquet with spatial sorting), TileDB (immature xarray integration, lower cloud I/O throughput than Zarr+Rust backends, minimal physics/earth-science adoption).

- [x] **Step 25b: canonicalize `ParticleData` — drop per-particle `charge`, standardize on `weight` + scalars**
  Tighten the Step 25 schema before it hardens: **readers always translate native PIC layouts into one canonical form** — per-particle `weight` (array) plus scalar `species_charge` and `species_mass`. The optional per-particle `charge` field disappears entirely; storage-convention branching moves from `ParticleData` and every downstream caller into the single place where it belongs (the reader).
  **Container changes (`containers.py`):**
  - Remove `charge: FloatArray | None` from `ParticleData`.
  - `weight`, `species_charge`, `species_mass` become required-ish (validation ensures they're present together for any per-particle mass/charge computation).
  - `macro_charge` collapses to a one-liner: `species_charge × weight`. No two-branch fallback.
  - `macro_mass` unchanged: `species_mass × weight`.
  - Drop the `__post_init__` validation block for `charge`.
  **Reader responsibility:** each `SimulationReader` produces the canonical form regardless of native layout.
  - **Combined-storage codes** (iPIC3D, OSIRIS): read per-particle `q = q_s × w` from disk, split into `weight = |q| / |species_charge|` and populate `species_charge`/`species_mass` from the run config. iPIC3D already does this split today — just stop storing the redundant `charge` array.
  - **Separate-storage codes** (VPIC, WarpX, Smilei, EPOCH, PIConGPU, TRISTAN-MP): read `weight` directly, populate scalars from config. No change.
  **I/O layer (`pypic.io._arrow`, `pypic.io._parquet`):**
  - Arrow/Parquet schema loses the optional `charge` column. Always has `weight` (`(N,)` float64) + scalar metadata.
  - `particles_to_arrow` / `particles_from_arrow`: drop the `charge` branch.
  - `particles_from_dataset`: `id_column="charge"` no longer valid; particle tracking uses `id_column="weight"` (already supported — the per-particle `weight` value carries the same unique-identifier property that iPIC3D's `charge` did, since `|q_species| = 1` makes them identical up to sign).
  - `particles_to_parquet`: `sort_by="charge"` drops; `sort_by="weight"` remains and absorbs the particle-tracking use case.
  **Schema (`docs/schema.md` § Per-particle data columns):**
  - Remove `charge` from the canonical field table.
  - Rewrite the "Charge–weight conventions across PIC codes" section: mention the combined/separate split as a *reader concern only*, note that the canonical container doesn't carry it.
  - Document the single-charge-state-per-species assumption explicitly (mixed ionization states must be modeled as separate species).
  - Note the round-trip fidelity caveat: `read → write → read` reconstructs `q = species_charge × weight` bit-exactly but doesn't preserve the original disk bytes of combined-storage codes. pypic is analysis, not simulation — not a concern for restart regeneration.
  **Tests:**
  - Remove tests that assert per-particle `charge` round-trip presence.
  - Keep tests that assert `macro_charge` derives correctly and equals the pre-refactor value on iPIC3D fixtures.
  - Add a reader-boundary test: iPIC3D reader input (per-particle `q` on disk) produces canonical form (no `charge`, populated `weight` + scalars).
  **Migration:** breaking change to the Step 25 Parquet schema. Version window is open (Step 25 just shipped). Bump `__version__` minor; no deprecation shim — pypic hasn't hit 1.0.
  **Depends on:** Step 25 (Parquet/Arrow foundation), commit `c272e33` (introduced `weight`/species scalars).

- [x] **Step 26: `pypic convert` CLI subcommand — fields & particles**
  Two subcommands with pipeline-appropriate flags. Both reuse the existing
  step-range parser (`parse_steps` in `cli.py`, which already handles `N`,
  `first`, `last`, `all`, `start:stop:stride`) and the unified
  `open_simulation()` path resolver.

  **`pypic convert fields <path> --output DIR`**
  - `--step SPEC` — forwarded to `parse_steps`. Default `all`.
  - `--fields B,E_3,rho_c` — forwarded to `Simulation.read(fields=...)`;
    supports vector-group shorthand (`"B"` → `B_1,B_2,B_3`) and aliases.
  - `--box x=0:64,y=0:64,z=32:64` — `BoxSelection.apply()` at convert
    time to crop spatial extent.
  - `--plane z=mid` — `PlaneSelection.apply()` for 2D slabs (reuses the
    plane-parsing helper already in `cli.py`).
  - `--target-resolution DX` — regrid to uniform spacing via `pypic.regrid`.
  - `--to-si` — boolean; applies `in_si()` to every field before write
    and writes with `Normalization.identity()`.  Handoff path for non-
    pypic consumers (IDL, MATLAB, plain xarray readers).  Default
    (code units + full Normalization serialized) is lossless and the
    right choice for pypic-to-pypic round-trips — consumers can call
    `in_units()` at read time.
  - `--dtype float32` — forwarded to `to_zarr{,_timeseries}(dtype=...)`.
  - `--compression zstd|blosc[:level]` — built into the `encoding=` dict
    passed to `to_zarr{,_timeseries}`.
  - `--virtual` — use `open_virtual()` to build lazy HDF5-backed refs
    instead of copying data; the resulting FieldDataset is written
    through the same `to_zarr` path (persists refs, not data).
  - `--backend zarr|icechunk` — forwards to `to_zarr{,_timeseries}
    (backend=...)`. Icechunk enables Git-like versioning (Step 24c).
  - `--tag NAME` / `--message TEXT` — icechunk-only; pass `message=` to
    the writer and `repo.create_tag(NAME, snapshot_id=...)` on success.
  - `--progress / --no-progress` — rich.progress bar for multi-step runs
    (default: enabled when stderr is a TTY).
  - `--dry-run` — print the planned write list without executing.

  **`pypic convert particles <path> --output DIR`**
  - `--step SPEC` — same parser; forwarded to
    `particles_to_dataset(steps=...)`.
  - `--species electrons,ions` — forwarded to
    `particles_to_dataset(species=...)`; accepts names or indices.
  - `--columns x,y,z` — forwarded to the reader's `columns=` kwarg.
  - `--box x=0.0:10.0,y=...` — crop particles by position before write.
  - `--sort-by position|weight` — forwards to
    `particles_to_{parquet,dataset}(sort_by=...)`.
  - `--position-dtype float32` / `--velocity-dtype float32` — forwards
    to the two existing downcast kwargs.
  - `--compression-level 1|3|5` — forwards to
    `particles_to_{parquet,dataset}(compression_level=...)`.
  - `--row-group-size N` — forwards to `row_group_size=`.
  - `--progress / --no-progress` / `--dry-run` — as above.

  **Both-pipelines subcommand:** `pypic convert all <path> --output DIR`
  writes fields to `{DIR}/fields.zarr` and particles (when present) to
  `{DIR}/particles/`.  Detects particle output via `sim.particle_steps`.
  Chosen over a subcommand-less default because typer's `Context` model
  conflates callback args with subcommand args.

  **Not in scope (separate future steps):** frame transforms at convert
  time (Step 40 extension), DuckDB-query-based particle filtering
  (belongs in `query_sql` in `pypic.io._parquet`, not the convert CLI),
  cloud sharding (`shards=` passthrough — add once `to_zarr` exposes it
  as a documented kwarg).

  **Shipped:** `convert fields` with `--step`, `--output`, `--fields`,
  `--box`, `--plane`/`--plane-index`/`--plane-coord`,
  `--target-resolution`, `--to-si`, `--dtype`, `--compression
  zstd|blosc[:level]`, `--virtual`, `--backend`, `--message`, `--tag`,
  `--progress/--no-progress`, `--dry-run`. `convert particles` with
  `--step`, `--output`, `--species`, `--columns`, `--box` (position
  filter in code units), `--sort-by`, `--position-dtype`,
  `--velocity-dtype`, `--compression-level`, `--row-group-size`,
  `--progress/--no-progress`, `--dry-run`. `convert all` runs both
  with sensible defaults. Single-step writes call `to_zarr`; multi-step
  writes call `to_zarr_timeseries` (adds a leading time dim). Icechunk
  `--tag` uses `icechunk_create_tag` after a successful write. Virtual
  mode (`--virtual`) treats PATH as a single HDF5 file and persists
  byte-range refs via `open_virtual` → `to_zarr`.

  **Dropped from spec:** per-field `--units nT,km/s,...` display
  strings.  Unit metadata is always serialized via the Normalization
  object; consumers call `in_units()` at read time.  Baking display
  units into the file would break the "one coherent normalization per
  file" invariant without saving anyone a step.

  **Depends on:** Steps 24, 24b, 24c, 25, 25b — all shipped.

---

## Phase 9: Additional Readers

- [ ] **Step 23: `pypic.readers.vlasiator` — VLSV reader via analysator**
  `VLasiatorReader` implementing `SimulationReader`. Two-grid strategy: FSgrid fields (`fg_b`, `fg_e`) read directly as uniform arrays; DCCRG fields (`proton/vg_rho`, `proton/vg_v`, `proton/vg_p`) regridded to uniform at `target_resolution` (default: FSgrid resolution). DCCRG cell IDs encode position + refinement level — decode to (x, y, z, dx) then block-average/NN-repeat (like BATSRUS AMR pattern, not `pypic.regrid` which is for uniform→uniform).
  Field mapping: `fg_b` → `B_1/B_2/B_3`, `fg_e` → `E_1/E_2/E_3`, `proton/vg_rho` → `n_s0`, `proton/vg_v` → `V_1/V_2/V_3`, `proton/vg_p` (6 components) → pressure tensor. Species auto-detected from VLSV population names. Auto-detection: `.vlsv` extension + file signature. `open_vlasiator()` convenience function. Optional dep: `analysator` under `vlasiator` extra. All tests mock analysator.

- [ ] **Step 35: `pypic.readers.vpic` — VPIC reader**
  `VPICReader` implementing `SimulationReader`. VPIC writes per-rank binary files (band-interleaved by field) or HDF5 via `vpic_decks`. Field mapping: `cbx/cby/cbz` → `B_1/B_2/B_3` (cell-centered B), `ex/ey/ez` → `E_1/E_2/E_3` (Yee edge), `jfx/jfy/jfz` → `J_1/J_2/J_3`, `rhob` → `rho_c`, per-species hydro files → density, velocity, pressure tensor. Yee mesh destaggering to co-located grid (linear interpolation, `StaggerInfo(convention="staggered")`). Metadata from `info` dumps or deck header. Auto-detection: `global.vpc` or `info` file presence. `open_vpic()` convenience function. All tests use synthetic fixtures.

- [ ] **Step 36: `pypic.readers.arms` — ARMS reader**
  `ARMSReader` implementing `SimulationReader`. ARMS (Adaptively Refined MHD Solver) outputs HDF5 with block-structured AMR. Regrid to uniform grid at `target_resolution` (like BATSRUS pattern). Field mapping from ARMS native names to canonical schema. Spherical geometry support (ARMS is commonly run in spherical coordinates for coronal/heliospheric simulations). `StaggerInfo(convention="staggered")` — ARMS uses a staggered mesh (CT for divergence-free B). Auto-detection: ARMS-specific HDF5 group structure. `open_arms()` convenience function. All tests use synthetic fixtures.

- [ ] **Step 42: `pypic.readers.openpmd` — openPMD reader (WarpX, PIConGPU, Smilei, FBPIC)**
  `OpenPMDReader` implementing `SimulationReader`. One reader covers four of the most-used modern PIC codes since they all emit the openPMD standard natively (HDF5 + ADIOS2 backends). High leverage compared to one reader per code.
  **Iteration encoding:** support both `groupBased` (single file, `/data/<step>/`) and `fileBased` (one file per step, `%T` placeholder pattern). `variableBased` (ADIOS2 streaming) is out of scope for v1.
  **Field mapping:** `meshes/B/{x,y,z}` → `B_1/B_2/B_3`, `meshes/E/{x,y,z}` → `E_1/E_2/E_3`, `meshes/J/{x,y,z}` → `J_1/J_2/J_3`, `meshes/rho` → `rho_c`. Per-species moments where the code emits them.
  **Stagger:** read the per-record `position` array (0.0–1.0 offset) directly into `StaggerInfo` (depends on Tier 2 per-component stagger work). Destagger to co-located grid for the canonical `FieldDataset`.
  **Units:** read `unitDimension` 7-tuple + `unitSI` per record; preserve as field metadata. The simulation-level `[units]` block is reconstructed from ED-PIC particle records (`charge`, `mass`, `weighting`) plus reference density derivable from species moments. Codes that don't write enough metadata to reconstruct fall back to `Normalization.identity()` (treat as SI).
  **Particles:** read `particles/<species>/{position,positionOffset,momentum,charge,mass,weighting,id}` and translate to canonical `ParticleData`. Honor `macroWeighted` + `weightingPower` semantics from ED-PIC when reading; always emit canonical form (per-particle `weight` + scalar `species_charge`/`species_mass`) per Step 25b.
  **Geometry:** `cartesian` → our `cartesian`; `thetaMode` (FBPIC RZ-mode decomposition) requires an azimuthal-mode reconstruction step before destagger — likely punt to Phase 2 of this reader.
  **Run metadata:** populate `[run].name` from `software` + `softwareVersion` attrs; `[run].date` from `date` attr; resources from `machine` attr.
  **Code identification:** the openPMD `software` attribute (e.g., "WarpX", "PIConGPU") drives a small dispatch table for the few code-specific metadata quirks (path conventions, mass/charge units that don't follow ED-PIC strictly). Auto-detection: presence of `openPMD` root attribute.
  Optional dep: `openpmd-api>=0.17` under an `openpmd` extra. All tests use synthetic openPMD files generated via openpmd-api in the test fixture setup. **Depends on:** Tier 1 ED-PIC vocabulary adoption (TASKS-schema-extension.md), Tier 2 per-component stagger (`StaggerInfo.position` array), Documentation backlog openPMD mapping.

---

## Pending Extensions

- [ ] **Step 19b: spherical regridding for `pypic.regrid`**
  Extend `regrid()` / `common_grid()` / `align_grids()` to handle `GeometryType.SPHERICAL`. Metric-factor-aware interpolation on $(r, \theta, \phi)$ grids (not just tensor-product linear in the raw indices — the $\sin\theta$ Jacobian matters near the poles). Pole handling: clamp $\theta \in [\epsilon, \pi - \epsilon]$ or switch to a local Cartesian chart near each pole. Intersection grid semantics: $r$ extends like Cartesian; $\theta$ intersected in $[0, \pi]$; $\phi$ intersected modulo $2\pi$ with wrap-around support.
  Primary use case: comparing two ARMS runs at different angular resolutions (Step 36). Also unblocks Step 20b (volume-weighted comparison norms). Keeps the `NotImplementedError` branch in `_require_cartesian_grid` alive for cylindrical until that reader lands.
  Tests: spherical harmonic round-trip ($Y_\ell^m$ sampled on a coarse grid, regridded to fine, residual bounded by the truncation order), pole fidelity (analytic $\cos\theta$ field, zero error at $\theta = 0, \pi$ within interpolation tolerance), $\phi$-wrap correctness (periodic field resampled across the $\phi = 2\pi$ seam).
  **Depends on:** Step 19 (Cartesian regrid).

- [ ] **Step 20b: volume-weighted comparison norms**
  Add metric-factor integration to `compare_fields()` / `field_comparison_report()` so L2 and L∞ correctly weight each cell by $\sqrt{|g|}\,d^n x$ instead of treating every sample uniformly. Current Step 20 is correct for uniform Cartesian grids (where $\Delta V$ cancels between numerator and denominator of the L2 norm), but wrong for spherical grids where cells near the poles or near $r = 0$ cover exponentially less volume. New API: `compare_fields(..., weighted: bool = False)` — defaults preserve current Cartesian behavior, `True` switches to the properly-weighted norm via `GridInfo.geometry.metric_factors()`. L∞ unaffected (max is a pointwise statistic). Tests: volume-weighted L2 of a radial shell equals the analytic shell volume; Cartesian result unchanged (regression test against current values).
  **Depends on:** Step 19b (spherical regridding — without it there is no spherical dataset to compare and this step is vacuous).

- [x] **Step 43: `pypic.reductions` — `project()` axis reduction**
  Single primitive that collapses a `FieldDataset` along one surviving axis with a chosen reduction (`integrate`, `sum`, `mean`, `max`, `min`, `std`), optionally pre-filtered by a `BoxSelection` or `SphereSelection`. Column densities, line-of-sight integrated $\mathbf{J}\!\cdot\!\mathbf{E}$, slab-mean fields, and projected-max diagnostics all compose from this one verb — no `SlabSelection` type needed, preserving the *selections describe regions, not data* invariant. `integrate` uses xarray's trapezoidal `.integrate(coord=...)` so non-uniform 1-D coords work for free once stretched grids land. Each projected DataArray gains an `attrs["projection"]` dict (`{axis, reduction, length?}`) for downstream provenance. Returns a `FieldDataset` with `surviving_axes` correctly updated via `_wrap_sliced`, so subsequent `compute()`, `transform_to()`, `plot_field_slice()`, and `in_si()` work unchanged. Gives webpic a clean wire format: `{selection, axis, reduction}` decodes to one server-side call.
  **Deferrals** (sub-bullets, not blockers):
  - **`weight=` kwarg** for yt-style weighted reductions (density-weighted LOS averages, emission-weighted temperature). Composes poorly with multi-field projection; one-line user workaround is `mean(field*w)/mean(w)`.
  - **Spherical / cylindrical Jacobian-aware integration.** Currently hard-fails with `NotImplementedError` per `regrid.py`'s `_require_cartesian_grid` pattern; depends on Step 19b.
  - **Unit-aware projection.** After `reduction="integrate"` the SI dimension shifts by one length factor along the projected axis (m⁻³ → m⁻² for density); MVP preserves `quantity_type` / `si_unit` unchanged, so `in_si()` is off by one length-unit on integrated outputs. Users multiply by `normalization.length_si` for the correct SI. Proper fix tracked alongside Step 20b.
  - **Multi-axis reduction (3D → 1D in one call).** Chain `project(project(ds, "z"), "y")` instead.
  - **CLI subcommand and `plot_projection` helper.** Compose via existing `plot_field_slice(project(ds, "z"), ...)`.

- [ ] **Step 40: time-dependent frame transforms**
  Extend `FrameTransform` to support rotation matrices that vary per timestep. Primary use case: GSE↔GSM depends on dipole tilt angle, which changes with time. Two approaches, both supported:
  - **Parameter-driven:** `parameter = "dipole_tilt"` in `[coordinates.transforms]` names a time-varying quantity looked up per step from simulation metadata or auxiliary data. The rotation matrix is recomputed at each timestep.
  - **SPICE kernels:** Optional integration with `spiceypy` for ephemeris-based transforms (GSE↔HEE↔RTN, planetary frames). `from_spice(frame_a, frame_b, epoch)` builds a `FrameTransform` from NAIF kernels. Optional dep: `spiceypy` under `spice` extra. Useful for comparing simulation output with spacecraft observations in the correct frame at the correct epoch.
  `FieldDataset.transform_to(frame, *, epoch=None)` gains an optional epoch parameter. Static transforms (current behavior) are unchanged. Tests: round-trip GSE→GSM→GSE at known tilt angles against published rotation matrices.
  **Depends on:** Step 15 (frame transforms).

---

## Phase 10: Ecosystem Integration

- [ ] **Step 27: `pypic.interop` — yt, PlasmaPy, SpacePy adapters**
  `pypic.interop.yt`: `to_yt_dataset(fds) -> yt.StreamDataset` — maps canonical fields to yt field tuples, sets domain from GridInfo. Cartesian only.
  `pypic.interop.plasmpy`: `to_plasmpy_plasma(fds, species_index) -> dict` — extracts density, temperature, |B| as `astropy.units.Quantity` (SI via `normalization.to_si()`). Dict, not PlasmaPy Plasma object (their API is unstable). `validate_against_plasmpy()` for cross-validation of derived quantities.
  `pypic.interop.spacepy`: `from_spacepy_dm(dm, grid, normalization) -> FieldDataset` — converts SpacePy DataModel (CDF/ISTP) with user-provided grid and field map.
  Optional deps: `yt>=4.3`, `plasmapy>=2024.7` + `astropy>=6.0`, `spacepy>=0.6` — each under its own extra. Each adapter is import-guarded with helpful install message. All tests mock external libraries.

- [ ] **Step 28: ecosystem documentation page**
  `docs/ecosystem.md` — positioning guide: pypic (multi-code reader unification + normalization + derived quantities), PlasmaPy (reference formulas + constants), SpacePy (spacecraft/observational data + CDF), yt (AMR visualization + volume rendering). Code examples for each adapter. "When to use which tool" decision guide.

- [ ] **Step 29: `pypic.interop.spase` — SPASE XML metadata export**
  `to_spase_xml(fds, *, resource_id, contact, description) -> str` generates a SPASE `NumericalData` XML document from FieldDataset metadata. Maps `simulation.toml` sections to SPASE elements: `[model]` → `SimulationRun`, `[grid]` → `SpatialDescription`, `[units]` → `Units` on each Parameter, `[[species]]` → `Particle` parameters, canonical fields → `Parameter` elements with `ParameterKey`/`Name`/`Description`/`Units`. `to_spase_file(fds, path, **kwargs)` writes to disk. No external deps (stdlib `xml.etree.ElementTree`). Enables publishing pypic-processed data to CDAWEB/VHO/CCMC archives. All tests use synthetic FieldDatasets.

---

## Phase 12: Virtual Probes & Spacecraft

- [ ] **Step 41: `pypic.probes` — virtual probe sampling**
  `Probe` frozen dataclass: a named point `(x, y, z)` in the simulation domain. `ProbeArray`: collection of probes (detector arrays, virtual satellite constellations). `ProbeTrajectory`: time-varying position as `(t, x, y, z)` array — a spacecraft orbit or moving detector path. A fixed probe is a degenerate trajectory (constant position).
  Core functions:
  - `sample(probe, dataset) -> dict[str, float]` — interpolate all fields at the probe position for one timestep. Reuses `RegularGridInterpolator` from `traces/_sampling.py`.
  - `sample_timeseries(probe, simulation, steps) -> TabularData` — sample across timesteps, producing time-series columns (time, B_1, B_2, B_3, ...). Output is `TabularData` (already exists).
  - `sample_trajectory(trajectory, simulation) -> TabularData` — sample along a moving path, one position per timestep.
  - `sample_array(probes, dataset) -> TabularData` — sample all probes at one timestep, one row per probe.
  Schema: `[[probes]]` section in simulation.toml (see schema.md § 8). Probes defined in config are available via `Simulation.probes`. CLI: `pypic probe <path> --name NAME --step all --field FIELD` for quick time-series extraction.
  **iPIC3D integration:** iPIC3D outputs virtual satellite data at fixed probe locations via its own high-cadence sampling. `AuxiliaryDataReader` already loads this as `TabularData`. The probe framework can: (a) define new probes and resample from field output, (b) load iPIC3D native virtual satellite data, (c) compare the two (native has higher time resolution; resampled has all derived fields).
  **Depends on:** `traces/_sampling.py` (interpolation), `TabularData` (output container).

- [ ] **Step 41b: SPICE-driven probe trajectories**
  `ProbeTrajectory.from_spice(target, observer, frame, epochs)` builds a trajectory from NAIF SPICE kernels via `spiceypy`. Enables direct comparison: load simulation, define a probe trajectory matching MMS/Cluster/PSP orbit, sample simulated fields along the real spacecraft path, compare with CDF observations (via SpacePy adapter, Step 27). Optional dep: `spiceypy` under `spice` extra.
  **Depends on:** Step 41 (probes), Step 40 (SPICE frame transforms).

---

## Phase 13: Cross-Project Integration

> **Tier-3 canonical names (locked in pre-v1.0).** Cross-tool work
> below adopts the Tier-3 canonical name shape: `<field>[_s<N>][_<i>]`
> with the species qualifier between the field name and the index
> (`B_1`, `V_s0_1`, `P_s0_11`, `q_s0_1`). HDF5 §4.1 and Zarr §4.2
> stores must use these names — `B1`, `V1_s0`, `P11_s0` are not
> emitted by any pypic-aware tool. rustpic and webpic should wire
> directly to Tier-3 names; no migration shim needed since neither
> has shipped.

- [ ] **Step 37: `pypic.server` — Arrow IPC streaming via Starlette/FastAPI**
  Zero-copy field data serving to webpic (Three.js/WebGPU viewer). Arrow IPC over WebSocket — **not** Arrow Flight (no Flight JS client exists for browsers; gRPC-Web requires an Envoy proxy and eliminates Flight's advantages). Pipeline: `pyarrow RecordBatch → IPC stream bytes → WebSocket → tableFromIPC() → Float32Array → Three.js BufferAttribute → GPU`. WebSocket provides persistent bidirectional connections ideal for continuous simulation streaming and time-series animation.
  Selections from the viewer UI map to pypic `Selection` objects server-side. Lazy I/O via xarray/dask serves only requested slices from disk. Arrow IPC carries structured metadata (field names, coordinates, units, normalization) in a single response. Readable in JS (`apache-arrow` npm package) and Rust (`arrow-rs`), aligning all three projects on one interchange format. Derived quantities computed server-side via `compute()`, unit conversion via `in_si()` / `in_units()`. Optional dep: `fastapi`, `uvicorn`, `pyarrow`, `websockets` under `server` extra. The server is a separate entry point, not part of the library import path.
  **Depends on:** Steps 24-25 (Zarr/Arrow foundations).

- [ ] **Step 38: `pypic.readers.rustpic` — Rust PIC code reader**
  Reader for rustpic's schema.md-conformant HDF5 output. The Rust code writes the canonical HDF5 layout directly (Section 4 of schema.md), so this is essentially `SimpleReader` with rustpic-specific metadata extraction and validation. pypic serves as the **reference implementation** — validate Rust-computed derived quantities against Python results on the same problem. Cross-project integration tests: run both codes on identical initial conditions, compare via `field_comparison_report()`. `open_rustpic()` convenience function. Auto-detection via HDF5 `model` attribute = `"rustpic"`.
  **Depends on:** Step 20 (cross-grid comparison diagnostics).

- [ ] **Step 39: webpic data pipeline documentation**
  End-to-end guide for the full platform: rustpic (Rust simulation) → HDF5 → pypic (Python analysis) → Starlette/FastAPI + Arrow IPC → webpic (Three.js/WebGPU visualization). Documents the schema.md contract that keeps Python, Rust, and JavaScript in sync. Selection round-trip: viewer UI selection → server `Selection` object → `FieldDataset` slice → Arrow IPC → GPU buffer. Coordinate transform pipeline: viewer requests a frame → server calls `transform_to()` → transformed data streamed. Covers: authentication model, chunked transfer for large datasets, WebSocket option for time-series animation.

---

## Dependency Graph

```
Step 19 (regrid) ←── Step 19b (spherical) ←── Step 20b (volume-weighted norms)
Step 15 (transforms) ←── Step 40 (time-dependent transforms)
Step 5 (FieldDataset) ←── Steps 24, 25 (Zarr/Arrow) ←── Step 26 (convert CLI)
                      ←── Step 24b (VirtualiZarr) ←── Step 24
                      ←── Step 24c (Icechunk) ←── Step 24
                      ←── Step 25b (canonical ParticleData) ←── Step 25
                      ←── Steps 23, 35, 36 (additional readers)
                      ←── Step 27 (interop adapters)
```

Recommended order: 24/25 parallelizable anytime, 24b/24c after 24, 25b right after 25 (before the Parquet schema hardens), 26 shipped, 40 anytime, 23/35/36 anytime, 27–28 after API stabilizes.
