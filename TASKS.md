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
- [ ] **Step 31: Remove default geometry from operators — Deferred.** Revisit when non-Cartesian operators land.
- [x] **Step 32:** Separate four_velocity quantity type
- [x] **Step 33:** specific_energy quantity type for enthalpy (fixed dimensional bug)
- [x] **Step 34:** StaggerInfo provenance metadata
- [x] **M0-prep: webpic API readiness** — promoted compute-side name registries to public surface so cross-tool codegen (webpic, rustpic tooling) reads stable names instead of underscore-prefixed internals.
  - New `pypic.aliases` module re-exports `COMPUTE_ALIASES`, `GROUP_ALIASES`, `SPECIES_SUFFIX_RE`, `species_name_aliases`.
  - `pypic.compute` now exposes `Recipe`, `RECIPES` (`MappingProxyType` view over the mutable `_REGISTRY`), `SpeciesArgs`, `SpeciesTemplate`, `SPECIES_TEMPLATES`; `pypic` top-level re-exports `aliases`, `Recipe`, `RECIPES`.
  - `_REGISTRY` stays underscore-prefixed (mutated by `register_recipe`/`unregister_recipe` under `_recipe_lock`); the proxy guarantees external callers only see the read side.
  - CLI smoke test `tests/test_schema_export.py::test_cli_export_codegen_flags_smoke` covers the `--inline-single-use-defs --include-x-extensions` flag combination webpic codegen pins (per-flag semantics already covered by `test_inline_single_use_defs_flattens_unique_refs` / `test_include_x_extensions_annotates_extensible_objects`).
  - `[webpic]` block (version=1; layout, shortcuts, diagnostics, embed) appended to all 7 bundled theme TOMLs with three per-theme overrides — LCARS `docked-side = "left"`, synthwave `timestamp-query-overlay = true`, light `fps-overlay = false`. pypic's theme loader (`plotting/_theme_io.py`) already ignores unknown sections, so no loader change was needed.
  - New `tests/test_public_api.py` (6 aggregated invariants: identity-with-private symbols, RECIPES vs `_REGISTRY` keyset parity, proxy immutability, live `register_recipe` reflection, codegen-import smoke); `tests/test_plotting.py::TestFileThemes::test_bundled_themes_have_webpic_section` checks every theme parses with a `[webpic]` block containing the required sub-tables.

**Milestone: daily-use tool** — load data → compute derived quantities → compare runs → select subregions → convert units → make paper figures. ✅

---

## Phase 8: Modern I/O Formats

- [x] **Step 24: `pypic.io` — Zarr export/import for FieldDataset**
  Zarr v3 + xarray DataTree. Layout v1 (post-2026.05): fields under `/fields` (mirrors schema.md §4.1 HDF5 grouping), metadata as flat keys on the root group's attrs (`grid`, `normalization`, `physics`, ...), with a `schema.version` discriminator (mirroring `simulation.toml`'s `[schema].version`; see schema.md §1 *Versioning*) so JS/Rust consumers can read without going through pypic. Consolidated metadata (`consolidated=True` write, `"auto"` read) gives one-shot fetch.
  Two write modes: `to_zarr(fds, path)` (single timestep, builds `xr.DataTree`) and `to_zarr_timeseries(simulation, path, *, steps, fields)` (multi-timestep with `time` as a dimension; chunked along time so reading one step is O(1)). `from_zarr(path) -> FieldDataset` reconstructs lazily including per-field metadata. Async concurrent metadata fetching (`zarr.config.set({'async.concurrency': 128})`) gives up to 14× speedup.
  **Naming:** canonical numbered names (`B_1`, `B_2`, `B_3`) on disk — geometry-specific aliases resolve on load. **Field metadata:** xarray attrs (`quantity_type`, `si_unit`, `long_name`, `latex`, `units`) serialize automatically. **FrameTransforms:** origin, rotation, scale as arrays in metadata; callable-based transforms skipped.
  **Precision:** `dtype="float32"` kwarg downcasts on write (halves storage; PIC outputs are single-precision anyway). **Compression:** default `BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')` — 10–300× on smooth EM data. Standalone `ZstdCodec` lacks shuffle and compresses floats poorly. Bytedelta (Blosc2 2.8+) is 37% better than bitshuffle on pressure-type data — expose as option once stable. **Sharding:** `shards=` kwarg for cloud stores (S3/GCS/R2) avoids small-files problem.
  **Version pinning:** `zarr>=3.1.0,<4` (3.0.0–3.0.7 yanked for append-mode data-loss bug). Requires `numcodecs>=0.16.0` (fixes BloscCodec `typesize=1` default). Optional deps under `zarr` extra.

- [x] **Step 24b: `pypic.io` — VirtualiZarr for legacy HDF5**
  `open_virtual(path) -> FieldDataset` creates lightweight virtual Zarr views over existing HDF5 by extracting byte-range metadata — `xr.open_zarr()` reads from original files without conversion. Uses VirtualiZarr v2.4+. Refs can be persisted to Icechunk (24c). Limitations: inherits source chunking (contiguous HDF5 datasets become single chunks); non-standard HDF5 compression filters may fail. Optional dep: `virtualizarr>=2.4` under `zarr` extra.
  **Depends on:** Step 24.

- [x] **Step 24c: `pypic.io` — Icechunk storage backend**
  Optional Git-like versioning and ACID transactions over Zarr v3 via Icechunk. Rust-based I/O: 13–14 Gbps read/write on cloud (2–10× faster than zarr+s3fs). Value: version tags for reproducibility (`repo.create_tag("v1.0-paper", snapshot_id=...)`), time-travel, Rust-accelerated I/O for non-versioned workflows too. `to_zarr(..., backend="icechunk")` writes; `from_zarr()` auto-detects. Optional dep: `icechunk>=1.1` under `icechunk` extra.
  **`open_virtual` runs on Zarr v3.** Originally specced against Kerchunk (Zarr v2) because VirtualiZarr's native v3 virtual backend *is* Icechunk; the Icechunk-backed `vds.vz.to_icechunk()` path landed (`pypic/io/_virtual.py`) and is now the only virtual-store path. The Kerchunk fill-value edge case (datasets where all values equal the fill value) is gone with it.
  **Depends on:** Step 24.

- [x] **Step 25: `pypic.io` — Parquet/Arrow for ParticleData**
  Two-tier API for billion-particle datasets with selective reads.
  **Low-level (in-memory):** `particles_to_arrow(data) -> pyarrow.Table` (zero-copy NumPy→Arrow), `particles_from_arrow(table) -> ParticleData`. Columnar `x/y/z/vx/vy/vz/weight/id` (positions always Cartesian). Species metadata in Arrow schema metadata. Used by the Starlette server (Step 37) for Arrow IPC over WebSocket.
  **High-level (partitioned):** `particles_to_parquet` (single file) and `particles_to_dataset(simulation, path, *, steps, species)` (Hive layout: `particles/step=000000/species=electrons/part-00000.parquet`).
  **Selection** via `particles_from_dataset(path, *, step, species, spatial_box, ids, energy_min, columns)`:
  - *Time / species:* partition pruning.
  - *Spatial:* Morton (Z-order) sort within partitions — chosen over Hilbert (<5% locality difference, 5–8× cheaper via BMI2 bit-interleaving). Parquet row-group min/max on x/y/z skips 95%+ of row groups.
  - *Energy:* `|v|` stored as derived column; predicate pushdown.
  - *ID:* predicate pushdown on `id`. Optional secondary index (`id → step/row_group`) for trajectory reconstruction.
  - *Columns:* column pruning via `columns=`.
  **Row groups:** 500K–1M rows. **Compression:** zstd + shuffle pre-filter (1.5–3× on particle data; without shuffle, LZ4 ≈1×, zstd 5–8×). Level 1 for processing, 3 for archival. **Precision:** `position_dtype`/`velocity_dtype="float32"` kwargs; `weight` stays full precision, `id` int64.
  **DuckDB (optional):** `query_sql(path, sql)` — automatic pushdown + partition pruning + morsel-driven parallelism. 45ms filtered counts vs Pandas 7.5s on 140 GB. Pattern: DuckDB for interactive, PyArrow for pipelines. Optional dep: `duckdb>=1.4` under `duckdb` extra. Core dep: `pyarrow>=17.0` under `arrow` extra.
  **Evaluated and rejected:** Lance (1.1× compression, no browser reader), GeoParquet (WKB overhead, ~3× larger), TileDB (immature xarray, lower cloud throughput).

- [x] **Step 25b: canonicalize `ParticleData` — drop per-particle `charge`, standardize on `weight` + scalars**
  Readers always translate native PIC layouts into one canonical form — per-particle `weight` (array) plus scalar `species_charge` and `species_mass`. The optional per-particle `charge` field disappears; storage-convention branching moves into the reader.
  **Container:** drop `charge: FloatArray | None`; `weight` + scalars become required-together for any per-particle mass/charge computation. `macro_charge = species_charge × weight` (one-liner). Drop the `__post_init__` charge validation.
  **Reader responsibility:** combined-storage codes (iPIC3D, OSIRIS) split `q = q_s × w` from disk into `weight = |q|/|species_charge|` + scalars. Separate-storage codes (VPIC, WarpX, Smilei, EPOCH, PIConGPU, TRISTAN-MP) already write `weight` — no change.
  **I/O layer:** Arrow/Parquet schema loses `charge` column. Particle tracking uses `id_column="weight"` / `sort_by="weight"` — equivalent because `|q_species| = 1` makes `weight` and old `charge` identical up to sign.
  **Schema doc:** drop `charge` from canonical table; document combined/separate split as reader concern only; single-charge-state-per-species assumption (mixed ionization → separate species). Round-trip fidelity caveat: `read → write → read` reconstructs `q` bit-exactly via float64 but doesn't preserve combined-layout disk bytes (pypic is analysis, not restart regeneration).
  **Migration:** breaking change to Step 25 Parquet schema. Version window open (Step 25 just shipped). Bump minor; no deprecation shim — pre-1.0.
  **Depends on:** Step 25.

- [x] **Step 26: `pypic convert` CLI subcommand — fields & particles**
  Two subcommands reusing the existing step-range parser (`parse_steps`: `N | first | last | all | start:stop:stride`) and `open_simulation()` path resolver.
  **`convert fields <path> --output DIR`:** `--step`, `--fields` (vector-group shorthand: `"B"` → `B_1,B_2,B_3`), `--box`, `--plane`/`--plane-index`/`--plane-coord`, `--target-resolution`, `--to-si` (boolean; writes with `Normalization.identity()` for non-pypic consumers — default code-units is lossless for pypic round-trips), `--dtype float32`, `--compression zstd|blosc[:level]`, `--virtual` (uses `open_virtual()` to persist HDF5 byte-range refs), `--backend zarr|icechunk`, `--tag NAME` / `--message TEXT` (icechunk only), `--progress/--no-progress`, `--dry-run`.
  **`convert particles <path> --output DIR`:** `--step`, `--species`, `--columns`, `--box` (position filter in code units), `--sort-by position|weight`, `--position-dtype`/`--velocity-dtype float32`, `--compression-level 1|3|5`, `--row-group-size`, `--progress`, `--dry-run`.
  **`convert all`:** runs both with sensible defaults. Detects particle output via `sim.particle_steps`. Chosen over a subcommand-less default because typer's Context model conflates callback and subcommand args.
  Single-step writes call `to_zarr`; multi-step writes call `to_zarr_timeseries` (leading time dim). Icechunk `--tag` calls `icechunk_create_tag` after success.
  **Not in scope:** frame transforms at convert time (→ Step 40 extension), DuckDB-query-based particle filtering (belongs in `query_sql`), cloud sharding (add once `to_zarr` exposes `shards=`).
  **Dropped from spec:** per-field `--units` display strings — unit metadata always serialized via Normalization; consumers call `in_units()` at read time. Baking display units would break "one coherent normalization per file" without saving anyone a step.
  **Depends on:** Steps 24, 24b, 24c, 25, 25b — all shipped.

---

## Phase 9: Additional Readers

- [ ] **Step 23: `pypic.readers.vlasiator` — VLSV reader via analysator**
  `VLasiatorReader`. Two-grid strategy: FSgrid fields (`fg_b`, `fg_e`) read as uniform arrays; DCCRG fields (`proton/vg_rho`, `proton/vg_v`, `proton/vg_p`) regridded to uniform at `target_resolution` (default: FSgrid resolution). DCCRG cell IDs encode position + refinement level — decode then block-average/NN-repeat (BATSRUS AMR pattern, not `pypic.regrid`). Field map: `fg_b` → `B_*`, `fg_e` → `E_*`, `proton/vg_rho` → `n_s0`, `proton/vg_v` → `V_*`, `proton/vg_p` (6 components) → pressure tensor. Species auto-detected from VLSV population names. Auto-detect via `.vlsv` extension + signature. Optional dep: `analysator` under `vlasiator` extra. Tests mock analysator.

- [ ] **Step 35: `pypic.readers.vpic` — VPIC reader**
  `VPICReader`. VPIC writes per-rank binary (band-interleaved by field) or HDF5 via `vpic_decks`. Map: `cbx/cby/cbz` → `B_*` (cell-centered), `ex/ey/ez` → `E_*` (Yee edge), `jfx/jfy/jfz` → `J_*`, `rhob` → `rho_c`, per-species hydro → density/velocity/pressure tensor. Yee mesh destaggering (linear interp, `StaggerInfo(convention="staggered")`) — the Cartesian-Yee destagger primitive is already implemented at `src/pypic/_stagger.py` (tested, currently unwired); consume it here. Metadata from `info` dumps or deck header. Auto-detect: `global.vpc` or `info` file. Tests use synthetic fixtures.

- [ ] **Step 36: `pypic.readers.arms` — ARMS reader**
  `ARMSReader`. ARMS (Adaptively Refined MHD Solver) outputs HDF5 block-structured AMR. Regrid to uniform at `target_resolution` (BATSRUS pattern). Spherical geometry support (ARMS commonly run in spherical for coronal/heliospheric). `StaggerInfo(convention="staggered")` — CT for divergence-free B. Auto-detect via ARMS-specific HDF5 group structure. Tests use synthetic fixtures.

- [ ] **Step 42: `pypic.readers.openpmd` — openPMD reader (WarpX, PIConGPU, Smilei, FBPIC)**
  `OpenPMDReader`. One reader covers four major modern PIC codes (all emit openPMD natively, HDF5 + ADIOS2). High leverage vs per-code readers.
  **Iteration encoding:** `groupBased` (single file, `/data/<step>/`) and `fileBased` (one file per step, `%T` pattern). `variableBased` (ADIOS2 streaming) out of scope for v1.
  **Field map:** `meshes/B/{x,y,z}` → `B_*`, `meshes/E/{x,y,z}` → `E_*`, `meshes/J/{x,y,z}` → `J_*`, `meshes/rho` → `rho_c`. Per-species moments where emitted.
  **Stagger:** read per-record `position` array (0.0–1.0 offset) into `StaggerInfo` (Tier 2/3 per-component). Destagger to co-located grid via the Cartesian-Yee primitive at `src/pypic/_stagger.py` (tested, currently unwired) — wire it into the per-record `position` handling here.
  **Units:** read `unitDimension` 7-tuple + `unitSI` per record. Reconstruct `[units]` from ED-PIC particle records (`charge`, `mass`, `weighting`) + reference density from species moments. Fall back to `Normalization.identity()` (treat as SI) when insufficient metadata.
  **Particles:** `position/positionOffset/momentum/charge/mass/weighting/id` → canonical `ParticleData` (Step 25b form). Honor `macroWeighted` + `weightingPower` semantics.
  **Geometry:** `cartesian` → ours; `thetaMode` (FBPIC RZ-mode) needs azimuthal-mode reconstruction before destagger — punt to Phase 2.
  **Code dispatch:** `software` attribute drives a small table for code-specific quirks (path conventions, mass/charge unit drift from ED-PIC). Auto-detect via `openPMD` root attribute.
  Optional dep: `openpmd-api>=0.17` under `openpmd` extra. Tests use synthetic openPMD via openpmd-api. **Depends on:** Tier 1 ED-PIC vocabulary, Tier 2 per-component stagger.

---

## Pending Extensions

- [ ] **Step 19b: spherical regridding for `pypic.regrid`**
  Extend `regrid()` / `common_grid()` / `align_grids()` to handle `GeometryType.SPHERICAL`. Metric-factor-aware interpolation on $(r, \theta, \phi)$ — $\sin\theta$ Jacobian matters near the poles. Pole handling: clamp $\theta \in [\epsilon, \pi - \epsilon]$ or local Cartesian chart near each pole. Intersection: $r$ like Cartesian; $\theta$ in $[0, \pi]$; $\phi$ modulo $2\pi$ with wrap. Primary use: comparing ARMS runs at different angular resolutions (Step 36); unblocks Step 20b.
  Tests: spherical harmonic round-trip ($Y_\ell^m$, residual bounded by truncation order); pole fidelity ($\cos\theta$ field, zero error at poles within tolerance); $\phi$-wrap correctness at the $2\pi$ seam.
  **Depends on:** Step 19.

- [ ] **Step 20b: volume-weighted comparison norms**
  Add metric-factor integration to `compare_fields()` / `field_comparison_report()` so L2/L∞ weight each cell by $\sqrt{|g|}\,d^n x$. Step 20 is correct on uniform Cartesian ($\Delta V$ cancels) but wrong on spherical (poles, $r=0$). API: `compare_fields(..., weighted: bool = False)` — defaults preserve Cartesian behavior. L∞ unaffected (pointwise). Tests: radial shell L2 = analytic shell volume; Cartesian regression unchanged.
  **Depends on:** Step 19b.

- [x] **Step 43: `pypic.reductions` — `reduce()` axis reduction**
  Single primitive that collapses a `FieldDataset` along one or more surviving axes (`axis: str | tuple[str, ...]`) with `integrate | sum | mean | median | max | min | std | var | argmax | argmin`, optionally pre-filtered by a `BoxSelection`/`SphereSelection`. Column densities, LOS-integrated $\mathbf{J}\!\cdot\!\mathbf{E}$, slab means, projected-max diagnostics, peak-position maps all compose — no `SlabSelection` type needed (preserves *selections describe regions, not data*).
  `integrate` uses xarray's trapezoidal `.integrate(coord=...)` looped over axes (non-uniform 1-D coords work for free). `argmax`/`argmin` use `idxmax`/`idxmin` returning the *coordinate value* (single-axis only). `weight=<field-name>` produces yt-style density/emission-weighted averages for `mean` and `integrate` ($\sum f w / \sum w$); joint NaN-masking under `nan_policy="omit"`. Result carries `attrs["reduction"]` ({axis, op, result_kind, weight}) for provenance. `surviving_axes` updated via `_wrap_sliced` so downstream `compute()`, `transform_to()`, `plot_field_slice()`, `in_si()` work unchanged. The `Reduction` type alias is exported; `_VALID_REDUCTIONS = typing.get_args(Reduction.__value__)`. Verb is `reduce` (not `project`) to avoid colliding with Three.js `Vector3.project(camera)` on webpic.
  CLI: `pypic reduce apply <path> --axis z --reduction mean --weight rho_c --output out.zarr`, composes with `--box`, `--plane`, multi-step writes, Icechunk tags, `--dry-run`. Gives webpic a wire format: `{selection, axis, reduction, weight?}`. Docs: `docs/api/reductions.md`.
  **Carry-overs:** spherical/cylindrical Jacobian → Step 43b (no longer blocked on 19b — `metric_factors()` already implemented for all geometries); unit-aware reduction → Step 43c; `plot_reduction` is just `plot_field_slice(ds.reduce(...))`.

- [ ] **Step 43b: spherical / cylindrical Jacobian-aware `reduce(integrate)`**
  Wire `CoordinateGeometry.metric_factors(...)` into the `integrate` branch so reductions on spherical/cylindrical return $\int f \, h_i \, dx^i$ instead of raising `NotImplementedError`. Build Jacobian $J = \prod_i h_i$ over reduced axes (function of *surviving* coords — $r$ for spherical $\theta$-integration, $r\sin\theta$ for $\phi$), broadcast, multiply field by $J$ before trapezoidal. Cartesian unchanged ($h_i=1$). Tests: spherical shell volume = $\frac{4}{3}\pi(r_2^3 - r_1^3)$ at machine precision; cylindrical disc area = $\pi r^2$ to trapezoidal order; Cartesian regression unchanged. `metric_factors()` is already at `coordinates/geometry.py:73-116` for all three geometries, so no longer blocked on 19b.
  **Depends on:** Step 5, Step 43.

- [ ] **Step 43c: unit-aware `reduce()`**
  After `reduce(reduction="integrate")` along $n$ axes, the SI unit shifts by $n$ length factors (m⁻³ → m⁻² → m⁻¹). Today `quantity_type`/`si_unit` are preserved unchanged, so `in_si()` is off by one length factor per reduced axis (workaround: multiply by `normalization.length_si**n`). Options: (a) generalize the `unit_dimension` 7-tuple arithmetic so attrs carry correct post-reduction dimensions; (b) add shifted canonical names (`column_density`, `surface_brightness`); (c) hybrid. Pairs with Step 20b's volume-weighted-norms work (same unit-dim concerns).
  **Depends on:** Step 43. **Pairs with:** Step 20b (shares the `unit_dimension` 7-tuple arithmetic concerns; ships independently).

- [ ] **Step 40: time-dependent frame transforms**
  Extend `FrameTransform` to per-timestep rotations. Primary use: GSE↔GSM via dipole tilt angle. Two paths:
  - **Parameter-driven:** `parameter = "dipole_tilt"` in `[coordinates.transforms]` names a time-varying quantity; rotation recomputed each step.
  - **SPICE:** `from_spice(frame_a, frame_b, epoch)` builds a transform from NAIF kernels (GSE↔HEE↔RTN, planetary frames). Optional dep: `spiceypy` under `spice` extra. Useful for spacecraft-observation comparison.
  `FieldDataset.transform_to(frame, *, epoch=None)` gains optional epoch. Static transforms unchanged. Tests: round-trip GSE→GSM→GSE at known tilt angles against published rotation matrices.
  **Depends on:** Step 15.

- [ ] **Step 44: field-line tracer & mapping infrastructure — symplectic integrator, periodic tricubic, curvature step control, squashing factor $Q$**
  Bundle of additive upgrades to `pypic.traces` + new `pypic.maps` module. Every sub-step is opt-in and sits next to the existing `trace_field_line_adaptive` (Dormand-Prince 5(4) + trilinear + PI step control, stays default). Motivating use cases:
  - **Fusion / Poincaré-section topology** — bounded invariants on closed orbits → 44a, 44b.
  - **Solar coronal mapping** — footpoint maps over PFSS/MAS on spherical grids → 44d, 44e, 44f, 44g.
  - **Magnetospheric topology** — QSL / X-line detection from MHD → 44e, 44f, 44g.

  Infrastructure pieces (44e curvature step control, 44f endpoint-only) are direct lifts from Predictive Science's MapFL Fortran tracer; 44g ($Q$) builds on both and delivers the headline solar/heliosphere mapping currently absent.

  **44a — implicit midpoint integrator.** New `trace_field_line_symplectic` / `trace_field_lines_symplectic` (fixed `step_size`, no `atol`/`rtol`). 1-stage Gauss-Legendre RK: $\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\hat{\mathbf{B}}((\mathbf{x}_n+\mathbf{x}_{n+1})/2)$, solved by 2-3 fixed-point iterations (Newton not needed, $f$ smooth and Lipschitz away from nulls). Preserves discrete symplectic 2-form: invariants like $r^2$ on closed orbits stay *bounded* instead of secularly drifting under DP. Take 5-10× larger steps for same picture quality on Hamiltonian-like flows → net wall-clock 1.2-1.5× rather than 2-3× naive overhead. Order 4 (2-stage Gauss-Legendre) deferred until 1-stage is benchmarked. Reuses `VectorFieldInterpolator` + termination logic; only new code is the step loop. **Tests:** $r^2$ conservation on $\mathbf{B}=(-y,x,0)$ over $10^4$ steps (bounded oscillation, no drift); $\psi$-conservation on 2D analytic flux; closed-circle Poincaré collapses to single ring vs current ~5% scatter.

  **44b — tricubic interpolation kwarg.** Promote `VectorFieldInterpolator.from_dataset(..., method="cubic")` (one-line scipy passthrough). Error $O(h^2) \to O(h^4)$ at 3-5× per-eval cost. Helps sharp-gradient regions (current sheets, shocks) where linear interp swings $\hat{\mathbf{B}}$ nonphysically, and preserves $\nabla\hat{\mathbf{B}}$ smoothness across cell faces (44a is more sensitive than DP). **Tests:** convergence on $\mathbf{B}=(-y,x,0)$ at three resolutions; Harris ($\tanh(y/L)$) smoothness — no spurious $\hat{\mathbf{B}}$ flips at cell faces.

  **44c — $\mathbf{A}$-based reconstruction (optional).** When a reader exposes `A_1/A_2/A_3`, interpolate $\mathbf{A}$ cubically and compute $\mathbf{B}=\nabla\times\mathbf{A}$ analytically. Guarantees $\nabla\cdot\mathbf{B}=0$ at every interior point — matters near nulls/separatrices where FD $\nabla\cdot\mathbf{B}$ artifacts dominate topology classification. Punt until a reader writes `A_*` (BATSRUS HDF5 does, iPIC3D doesn't).

  **44d — periodic tricubic splines for $\phi$/$\theta$.** Once `pypic.regrid` and tracer gain spherical support (19b territory), natural follow-on. scipy's `RegularGridInterpolator(method="cubic")` doesn't handle periodicity: $\phi=2\pi$ seam interpolates against one-sided stencil → C⁰ (not C¹) discontinuity in $\hat{\mathbf{B}}$ that visibly bends field lines. MapFL uses periodic-spline routines (`spline_periodic_type1/2` at `mapfl.f:8885, 9028`, wrapped by `compute/evaluate_spline_3d`; `mapfl.f:7286` is the `getb` inner loop with `cubic=.true.`). Knob: `periodic_axes: tuple[int,...] = ()` on `VectorFieldInterpolator.from_dataset()` — flagged axes use `CubicSpline(..., bc_type="periodic")` per spline line, accumulated into a tensor-product cubic block. **Tests:** seamless across $\phi=2\pi$ on $\mathbf{B}=(-\sin\phi,\cos\phi,0)$ (C¹ to spline accuracy); regression vs 44b in seam-free interior (zero behavior change).
  **Depends on:** Step 19b (gates the geometry where periodicity bites); 44b (the codepath this extends).

  **44e — curvature-based step control (non-default).** `step_control: Literal["error","curvature"] = "error"` kwarg alongside the existing Gustafsson PI. Curvature path (MapFL `mapfl.f:6445`): keep $\|\hat{\mathbf{B}}_{n+1}-\hat{\mathbf{B}}_n\|\cdot h/\Delta s \approx$ `over_rc` (default 0.0025 — unit-tangent rotation per step), clamped by `local_mesh_factor × min(\Delta x_i)` so a step never overshoots its source cell. Win on long quasi-laminar traces (PFSS, dipole magnetospheres, tokamak equilibria) where absolute-tolerance PI can't distinguish smooth from chaotic and stays uniformly conservative. Auto-tightens in sharp-gradient regions because both $\|\Delta\hat{\mathbf{B}}\|$ and mesh size shrink. **Tests:** PFSS dipole — curvature wins 2-5× on smooth fields at matched endpoint tolerance; Harris — both converge to similar step counts.
  **Depends on:** `pypic.traces._tracing` (`step_size_init`, `min_step`, `max_step`, `VectorFieldInterpolator`).

  **44f — endpoint-only mode for `trace_field_lines_adaptive`.** `return_endpoints_only: bool = False` kwarg skips materializing `FieldLine.points`/`arc_lengths`, returns a `FieldLineEndpoint` namedtuple: `(start, end, arc_length, termination_reason)`. Memory: a 1000×1000 map at 20k steps is $\mathcal{O}(\text{TB})$ in full-trace mode — unworkable when only endpoints are consumed. MapFL makes trajectory storage opt-in (`xt` is `optional` at `mapfl.f:7479`) for this reason. Also the kernel of 44g ($Q$ = 5 endpoint traces per seed) and any future `pypic.maps` connectivity diagnostic. **Tests:** endpoint = full-trace endpoint within roundoff; memory $O(M)$ per-seed not $O(M\cdot N_{steps})$.
  **Depends on:** `pypic.traces._tracing`.

  **44g — `pypic.maps` module + squashing factor $Q$.** New top-level module: open/closed classification, footpoint mapping $\mathbf{r}(\theta_0,\phi_0)\mapsto\mathbf{r}(\theta_1,\phi_1)$, the Titov-Démoulin squashing factor $Q$, and the Pariat & Démoulin 2012 signed-log $\text{slog}(Q) = \mathrm{sign}(B_r)\cdot\log_{10}(Q/2 + \sqrt{(Q/2)^2-1})$ that flips sign across separatrices. Flagship: `squashing_factor(ds, seeds, *, h=1e-4, ...)` — at each seed, trace 1 central + 4 perp-pair neighbors, central-difference for the 2×2 Jacobian $D$, return $Q = \|D\|_F^2/|\det D|$. Verbatim MapFL `getq` recipe (`mapfl.f:4864`), including the $\sin\theta < 5\times10^{-3}$ pole switch to Cartesian basis. Rests on 44f ($Q$ on 1000² = 5M traces; full storage untenable) and existing termination logic. **Use cases:** solar coronal connectivity (open-field maps, coronal-hole boundaries, slow/fast wind footpoints), magnetospheric QSL mapping, pre-flare $Q$-line detection. **Tests:** $Q\approx 2$ on uniform field (analytic lower bound); $Q\to\infty$ across analytic separatrix; numerical comparison vs MapFL on published PFSS test (Titov-Démoulin flux rope, or synthetic dipole + ring current).
  **References:** Titov, Hornig & Démoulin 2002; Pariat & Démoulin 2012; MapFL `mapfl.f:4864`.
  **Depends on:** 44f. Benefits from 44d on spherical PFSS.

  **Reference:** Hairer, Lubich & Wanner, *Geometric Numerical Integration* (2006) — Ch. II.1 (implicit midpoint as 1-stage Gauss-Legendre, symplecticity), Ch. V (bounded-vs-secular drift theorem via shadow-Hamiltonian). Cite as `[@HairerLubichWanner2006]`.

  **Not in scope:** variational integrators (overkill without a Lagrangian); 2-stage order-4 Gauss-Legendre (until 1-stage benchmarked); Hamiltonian formulation in flux coordinates (Boozer/Hamada — fusion-codes do this *because they have flux coordinates*; pypic assumes Cartesian/spherical/cylindrical); full MapFL diagnostic surface beyond $Q$ (expansion factor, $K$-factor, magnetic dips — add piecewise when a concrete use case appears).

  **Depends on:** existing `pypic.traces` (`VectorFieldInterpolator`, `FieldLine`, termination logic). 19b gates 44d. No new mandatory deps — scipy ships cubic `RegularGridInterpolator` and periodic `CubicSpline`.

---

## Phase 10: Ecosystem Integration

- [ ] **Step 27: `pypic.interop` — yt, PlasmaPy, SpacePy adapters**
  `pypic.interop.yt`: `to_yt_dataset(fds) -> yt.StreamDataset` — maps canonical fields to yt field tuples, sets domain from GridInfo. Cartesian only.
  `pypic.interop.plasmpy`: `to_plasmpy_plasma(fds, species_index) -> dict` — extracts density, temperature, |B| as `astropy.units.Quantity` (SI via `normalization.to_si()`). Dict, not PlasmaPy Plasma object (API unstable). `validate_against_plasmpy()` cross-validates derived quantities.
  `pypic.interop.spacepy`: `from_spacepy_dm(dm, grid, normalization) -> FieldDataset` — converts SpacePy DataModel (CDF/ISTP) with user-supplied grid and field map.
  Optional deps: `yt>=4.3`, `plasmapy>=2024.7`+`astropy>=6.0`, `spacepy>=0.6` — each under its own extra. Import-guarded with helpful install message. Tests mock external libraries.

- [ ] **Step 28: ecosystem documentation page**
  `docs/ecosystem.md` — positioning: pypic (multi-code reader + normalization + derived), PlasmaPy (reference formulas + constants), SpacePy (spacecraft/CDF), yt (AMR + volume rendering). Code examples for each adapter. "When to use which tool" decision guide.

- [ ] **Step 29: `pypic.interop.spase` — SPASE XML metadata export**
  `to_spase_xml(fds, *, resource_id, contact, description) -> str` generates a SPASE `NumericalData` XML. Maps `simulation.toml` sections to SPASE elements: `[model]` → `SimulationRun`, `[grid]` → `SpatialDescription`, `[units]` → `Units` on each Parameter, `[[species]]` → `Particle`, canonical fields → `Parameter` (`ParameterKey`/`Name`/`Description`/`Units`). `to_spase_file(fds, path, **kwargs)` writes to disk. No external deps (stdlib `xml.etree.ElementTree`). Enables publishing to CDAWEB/VHO/CCMC. Tests use synthetic FieldDatasets.

---

## Phase 11: Virtual Probes & Spacecraft

- [ ] **Step 41: `pypic.probes` — virtual probe sampling**
  `Probe` frozen dataclass: named point `(x, y, z)`. `ProbeArray`: collection (detector arrays, satellite constellations). `ProbeTrajectory`: time-varying `(t, x, y, z)` — spacecraft orbit / moving detector. A fixed probe is a degenerate trajectory.
  Core functions:
  - `sample(probe, dataset) -> dict[str, float]` — interpolate all fields at the probe position for one timestep. Reuses `RegularGridInterpolator` from `traces/_sampling.py`.
  - `sample_timeseries(probe, simulation, steps) -> TabularData` — sample across timesteps (time, B_1, B_2, B_3, ...).
  - `sample_trajectory(trajectory, simulation) -> TabularData` — sample along a moving path.
  - `sample_array(probes, dataset) -> TabularData` — all probes at one timestep, one row per probe.
  Schema: `[[probes]]` section (schema.md § 2 — `[[probes]]`). Probes from config available via `Simulation.probes`. CLI: `pypic probe <path> --name NAME --step all --field FIELD`.
  **iPIC3D integration:** iPIC3D outputs native virtual satellite data at fixed locations (high cadence). `AuxiliaryDataReader` already loads as `TabularData`. Probes can: (a) define new and resample, (b) load native, (c) compare (native = higher time res; resampled = all derived fields).
  **Depends on:** `traces/_sampling.py`, `TabularData`.

- [ ] **Step 41b: SPICE-driven probe trajectories**
  `ProbeTrajectory.from_spice(target, observer, frame, epochs)` builds a trajectory from NAIF SPICE kernels via `spiceypy`. Direct comparison: load sim, define trajectory matching MMS/Cluster/PSP orbit, sample fields along real path, compare with CDF observations (SpacePy adapter, Step 27). Optional dep: `spiceypy` under `spice` extra.
  **Depends on:** Step 41, Step 40.

---

## Phase 12: Cross-Project Integration

> **Tier-3 canonical names (locked pre-v1.0).** Cross-tool work below uses `<field>[_s<N>][_<i>]` with species qualifier between field name and index (`B_1`, `V_s0_1`, `P_s0_11`, `q_s0_1`). HDF5 §4.1 and Zarr §4.2 stores must use these — `B1`, `V1_s0`, `P11_s0` are not emitted by any pypic-aware tool. rustpic and webpic wire directly to Tier-3; no migration shim since neither has shipped.

- [ ] **Step 37: `pypic.server` — Arrow IPC streaming via Starlette/FastAPI**
  Zero-copy field serving to webpic. Arrow IPC over WebSocket — **not** Arrow Flight (no JS Flight client for browsers; gRPC-Web needs Envoy proxy and eliminates Flight's advantages). Pipeline: `pyarrow RecordBatch → IPC bytes → WebSocket → tableFromIPC() → Float32Array → Three.js BufferAttribute → GPU`. WebSocket = persistent bidirectional for streaming + time-series animation.
  Viewer-UI selections map to pypic `Selection` server-side. Lazy I/O via xarray/dask serves only requested slices. Arrow IPC carries structured metadata (names, coords, units, normalization) in a single response. Readable in JS (`apache-arrow`) and Rust (`arrow-rs`) — aligns all three projects on one interchange format. Derived quantities via `compute()`; unit conversion via `in_si()`/`in_units()`. Optional dep: `fastapi`, `uvicorn`, `pyarrow`, `websockets` under `server` extra. Separate entry point, not on library import path.
  **Depends on:** Steps 24-25.

- [x] **Step 37a: typed server exception hierarchy**
  Foundations follow-up to Step 37's tactical S1 fix (commit `946b0ca`, which introduced one `UnknownSimulationError` and replaced a `"simulation" in str(exc)` message-sniff). One named exception per `ErrorFrame.kind` defined once, raised at the library boundary, and routed by a single dispatcher on both transports.
  **Library types** in `pypic.exceptions`: `PypicError(Exception)` base + `UnknownSimulationError`, `UnknownFieldError`, `UnknownStepError` (all `KeyError` subclasses), `GeometryUnsupportedError(NotImplementedError)`. Each subclass carries `kind: ClassVar[str]` and `status_code: ClassVar[int]` classvars matching the existing `ErrorFrame.kind` Literal and HTTP semantics (404 / 400). Multi-inheritance keeps existing `except KeyError` / `except NotImplementedError` callers unbroken.
  **Server-only** `ValidationFailedError(PypicError, ValueError)` in `pypic.server.exceptions` wraps `pydantic.ValidationError` from `SubscribeRequest.model_validate_json` and from `simulation.toml` validation at the registry boundary; preserves the original on `__cause__`.
  **Raise sites switched**: `dataset.resolve_key`, `Simulation.read` strict-fields branch, `regrid._require_cartesian_grid`, `compute._execute_recipe` geometry guard, `reductions.reduce` spatial-axes guard, `coordinates.operators._require_cartesian`.
  **Server dispatch unified**: `stream._handle_one` collapsed from 7 `except` branches to 2 (`PypicError` → `exc.kind`; `Exception` → `internal`). `_UnknownStepError` deleted (now public `UnknownStepError`). `routes.py` drops three duplicated `try/except KeyError → HTTPException(404)` blocks. `app.py` registers a single `@app.exception_handler(PypicError)` returning `JSONResponse(status_code=exc.status_code, content={"kind": exc.kind, "detail": str(exc)})` — `detail` stays back-compat with the previous `HTTPException` body, `kind` is additive to mirror the WebSocket `ErrorFrame`. `_state.SimulationRegistry.get` translates `KeyError`/`FileNotFoundError` from `open_simulation` (explicit-reader miss or auto-detect-no-match) into `UnknownSimulationError` so HTTP/WS both see a single typed error for "couldn't open this sim".
  **Tests**: new `tests/test_server_exceptions.py` (7 tests pinning `kind`/`status_code`/`isinstance` contract + `__cause__` preservation); new `test_sim_info_404_carries_typed_error_kind` in `test_server_routes.py` covering the additive HTTP `kind` field; existing `test_server_stream.py` / `test_server_routes.py` boundary tests stay green as the regression gate.
  **Deferred**: a library-level `NoReaderFoundError` (currently the `FileNotFoundError` from `_registry` auto-detect translates at the server boundary rather than at the raise site). Not blocking; doesn't surface differently to clients.
  **Depends on:** Step 37 S1 (commit `946b0ca`).

- [ ] **Step 37b: selection provenance — `attrs.selections` round-trip**
  Symmetric counterpart to `attrs["reduction"]`. Today `BoxSelection`/`PlaneSelection`/`SphereSelection` produce datasets with no recorded region — box/plane partly recoverable from post-slice `grid.lower/upper/dimensions`, but `SphereSelection` loses centre, radius, keep-direction once NaN mask lands. Webpic's `{selection, axis, reduction}` wire format round-trippable only when selection state survives `to_zarr`/`from_zarr`. Design: (a) **list** at root (`attrs.selections`), not per-field — selections apply globally and chained `Box → Sphere → reduce` needs ordered composition; (b) entries typed by `kind` (`"box"`|`"plane"`|`"sphere"`) with parameters in code units + pointer to active normalization (radii interpretable across normalizations); (c) Pydantic `[[selection]]` records in `pypic.schema._models`, JSON Schema regen + drift test; (d) replay via `Selection.from_attrs(record)` classmethods. Defer until 37 forces the wire-format contract.
  **Depends on:** Step 37.

- [ ] **Step 38: `pypic.readers.rustpic` — Rust PIC code reader**
  Reader for rustpic's schema.md-conformant HDF5 output. Rust writes the canonical layout (Section 4) directly, so essentially `SimpleReader` + rustpic-specific metadata extraction + validation. pypic serves as the **reference implementation** — validate Rust-computed derived quantities against Python on the same problem. Cross-project integration tests: identical ICs, compare via `field_comparison_report()`. Auto-detect via HDF5 `model` attribute = `"rustpic"`.
  **Depends on:** Step 20.

- [ ] **Step 39: webpic data pipeline documentation**
  End-to-end guide: rustpic (Rust sim) → HDF5 → pypic (Python analysis) → Starlette/FastAPI + Arrow IPC → webpic (Three.js/WebGPU). Documents schema.md contract syncing Python, Rust, JS. Selection round-trip: viewer UI → server `Selection` → `FieldDataset` slice → Arrow IPC → GPU buffer. Coordinate transform: viewer requests frame → server `transform_to()` → transformed data streamed. Covers auth model, chunked transfer, WebSocket option for time-series animation.

---

## Dependency Graph

```
Step 19 (regrid)         ←── 19b (spherical) ←── 20b (volume-weighted norms)  pairs-with 43c
Step 15 (transforms)     ←── 40 (time-dependent) ←── 41b (SPICE trajectory)
Step 5 (FieldDataset)    ←── 24, 25 (Zarr/Arrow)    ←── 26 (convert CLI)
                                                   ←── 24b (VirtualiZarr), 24c (Icechunk)
                                                   ←── 25b (canonical ParticleData)
                         ←── 23, 35, 36, 42 (readers)
                         ←── 27 (interop adapters)
Step 11 + Step 5         ←── 43 (reductions) ←── 43b (Jacobian), 43c (units)
Steps 24, 25             ←── 37 (Arrow IPC server) ←── 37a (typed exceptions, shipped)
                                                   ←── 37b (selection provenance), 39 (docs)
Step 20                  ←── 38 (rustpic reader)
pypic.traces             ←── 44 (field-line tracer + pypic.maps)
                              44a (implicit midpoint), 44b (tricubic), 44c (A-based)
                              44d (periodic tricubic) ←── 19b, 44b
                              44e (curvature step), 44f (endpoint-only)
                              44g (pypic.maps + Q)    ←── 44f, (44d on spherical)
traces/_sampling.py      ←── 41 (probes) ←── 41b (SPICE)
```

Recommended order: 37b after 37 (37a shipped); 38 needs a rustpic dump; 39 follows 37; 23/35/36/42 anytime; 40 unblocks 41b; 43b/43c anytime after 43; 44a-g modular (44f is the kernel for 44g); 19b precedes 20b and 44d.
