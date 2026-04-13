# pypic — Implementation Roadmap

Each step produces something testable. No step starts until the previous step's tests pass.

---

## Phase 1: Foundation

- [x] **Step 1: skeleton — project setup**
  pyproject.toml (uv, ruff, pytest `--doctest-modules`, mypy strict), CI, `py.typed`.

- [x] **Step 2: units — Normalization class**
  Frozen dataclass with reference values, `normalize(quantity, x)` / `to_si(quantity, x)` methods.
  - `pic_standard(reference_density, reference_mass, reference_charge, c)` — derives from any reference species
  - `pic_electron(n_e)` — electron-scale convenience wrapper
  - `mhd_standard(l_0, rho_0, b_0)` — Alfven-speed-based refs
  - `identity()` — all refs = 1.0

- [x] **Step 3: units — PhysicsConstants and SpeciesInfo**
  `PhysicsConstants` (PIC vs MHD normalized, `inv_c_squared()`). `SpeciesInfo` frozen dataclass with inference logic (charge + mass <-> charge_to_mass).

- [x] **Step 4: coordinates/geometry — CoordinateGeometry**
  `GeometryType` StrEnum, `CoordinateGeometry` frozen dataclass, `metric_factors()`. Pre-defined `CARTESIAN`, `SPHERICAL`, `CYLINDRICAL` instances.

- [x] **Step 5: readers/base — FieldDataset, GridInfo, SimulationReader, SimulationConfig**
  `GridInfo` (dimensions, spacing, origin, geometry, `coordinate_arrays()`). `FieldDataset` wrapping `xr.Dataset` with `__getitem__`, `has_field`, `field_names`, `sel`/`isel`, geometry-aware aliases. `SimulationReader` protocol. `SimulationConfig` frozen dataclass.

---

## Phase 2: Physics

- [x] **Step 6: readers/config — simulation.toml loader**
  `load_config(path) -> SimulationConfig` via `tomllib`. Parses all schema.md sections: `[model]` -> name/type, `[grid]` -> `GridInfo`, `[units]` -> `Normalization`, `[coordinates]` -> `CoordinateGeometry` + frame, `[[species]]` -> `list[SpeciesInfo]`, `[physics]` -> dict. Optional: `[initial_conditions]`, `[output]` -> metadata.

- [x] **Step 7: derived (part 1) — field-level quantities**
  Pure NumPy functions: `magnetic_field_magnitude`, `electric_field_magnitude`, `current_density_magnitude`, `velocity_magnitude`, `plasma_beta`, `alfven_speed`, `poynting_flux`, `magnetic_energy_density`, `electric_energy_density`, `kinetic_energy_density`, `thermal_energy_density`, `internal_energy`, `enthalpy`, `relativistic_enthalpy`, `entropy`, `gyrotropic_entropy`.

- [x] **Step 8: derived (part 2) — characteristic scales**
  Species-dependent: `thermal_speed`, `gyrofrequency`, `plasma_frequency`, `skin_depth`, `gyroradius`, `debye_length`, `sound_speed`, `ion_acoustic_speed`, `magnetosonic_speed`, `alfven_mach`, `magnetosonic_mach`, `parallel_pressure`, `perpendicular_pressure`, `agyrotropy`. Verify against NRL Formulary.

- [ ] **Step 8b: per-species pressure decomposition in compute registry**
  Wire `P_par_s0`, `P_perp_s0`, `P_par_s1`, `P_perp_s1`, `agyrotropy_s0`, `agyrotropy_s1` as compute recipes. The underlying functions (`parallel_pressure`, `perpendicular_pressure`, `agyrotropy`) already work on any tensor — the missing piece is registry plumbing: recipes that map per-species tensor components (`P11_s0`...`P33_s0`) + `B1/B2/B3` to the decomposition. Generalizes to N species via the existing `species_index` mechanism on `_Recipe`.

- [x] **Step 9: diagnostics — comparison and validation**
  `l2_relative_error`, `linf_error`, `field_difference`, `field_energy`, `div_b`, `max_div_b`, `div_e`. Cartesian central differences.

- [x] **Step 10: coordinates/operators — discrete differential operators**
  Geometry-aware `curl`, `div`, `grad`. Cartesian second-order central diffs; spherical/cylindrical raise `NotImplementedError`.

---

## Phase 3: Integration

- [x] **Step 11: selections — PlaneSelection and BoxSelection**
  Frozen dataclasses with `apply(data) -> FieldDataset`. `PlaneSelection` (normal, index; `None` = midplane). `BoxSelection` (optional index ranges per axis).

- [x] **Step 12: readers — multi-format reader system**
  Four readers implementing `SimulationReader`:
  - **iPIC3D:** `IPic3DParallelReader` (phdf5), `IPic3DSerialReader` (shdf5), `IPic3DH5hutReader` (H5hut). Parses `.inp` and `settings.hdf` configs, maps iPIC3D names to canonical, 4π Gaussian→SI-rationalized correction, node-centered grid origin, ZYX→XYZ transpose, pressure tensor sign correction, per-species fields. `ConservedQuantities` parser (Format A + B). `open_ipic3d()` auto-detects all three formats.
  - **BATSRUS:** `BATSRUSReader` for IDL per-cell and HDF5 BATL formats. AMR regridding to uniform grid, `target_resolution` parameter, `parse_param_in()`, `parse_header()`. Handles normalized and SI-unit outputs, split-B, geometry propagation.
  - **OpenGGCM:** `OpenGGCMReader` for Fortran binary 3df files with custom grid parsing.
  - **SimpleReader:** HDF5 files following the canonical schema directly.

  **Registry and auto-detection:** Confidence-based `open_simulation()` with `ProbeResult` diagnostics, factory fallback (tries next-best reader if top candidate crashes, `ExceptionGroup` if all fail), `Simulation` facade with `probe_results` introspection, `describe()`, `refresh_steps()`, `first_step`/`last_step`. Selective I/O via `fields=` parameter. `AuxiliaryDataReader` protocol for tabular data. BATSRUS `.h` probe tightened to BATSRUS timestamp pattern (avoids C header false positives). `FieldDataset._resolve_key` suggests close matches on `KeyError`.

- [x] **Step 13: FieldDataset — compute() and in_units()**
  `compute(name)` dispatches string to derived function ("|B|", "beta", "v_A", "M_A", "|vort|", "vort1"/"vort2"/"vort3", ...). `in_si(field)` for SI conversion. `in_units(field, unit_str)` for display units ("nT", "km/s"). `QuantityType` StrEnum, `_FIELD_INFO` registry with `FieldInfo` metadata (quantity_type, long_name, si_unit, latex). `register_field()` / `unregister_field()` for custom fields. `FieldDataset.with_field()` attaches fields with xarray attrs carrying metadata through slicing. Attrs-first lookup in `field_info()` / `in_si()` (xarray attrs override global registry). Geometry-aware label localization. Per-species regex patterns for auto-generated metadata.

- [x] **Step 14: plotting/slices — basic 2D visualization**
  `plot_field_slice` (plane selection, axis labels, colorbar). `plot_comparison` (three-panel: A | B | difference). Publication rcParams in `plotting/styles.py`.

**Milestone: daily-use tool** — load iPIC3D data -> compute derived quantities -> compare runs -> select subregions -> convert units -> make paper figures.

---

## Phase 4: Extensions

- [x] **Step 15: coordinates/transforms — frame transforms**
  `ReferenceFrame`, `FrameTransform` dataclasses. Load from `[coordinates.transforms]` in simulation.toml. Transform chaining (A->B + B->C = A->C). `FieldDataset.transform_to(frame_name)`.

- [x] **Step 16: selections — SphereSelection**
  Non-axis-aligned selection via `xr.where()`. Points outside sphere = NaN, grid shape preserved. NaN propagation in derived quantities and plotting. Also added `FieldDataset.where(cond)` as the general-purpose masking primitive.

- [x] **Step 18: derived (relativistic) — relativistic derived quantities**
  `lorentz_factor()` (from three-velocity or four-velocity).
  `magnetization()` ($\sigma = B^2/\rho_m c^2$).
  Extend ~9 functions with optional `lorentz_factor` parameter:
  `kinetic_energy_density`, `alfven_speed`, `sound_speed`,
  `magnetosonic_speed`, `gyrofrequency`, `plasma_frequency`,
  `skin_depth`, `gyroradius`, `thermal_speed`.
  Add `u1/u2/u3` aliases in `_build_aliases`.
  Priority: bulk-flow corrections first, thermal second.
  Tests: γ→1 recovers non-relativistic; σ→∞ gives v_A→c.

- [ ] **Step 40: time-dependent frame transforms**
  Extend `FrameTransform` to support rotation matrices that vary per timestep. Primary use case: GSE↔GSM depends on dipole tilt angle, which changes with time. Two approaches, both supported:
  - **Parameter-driven:** `parameter = "dipole_tilt"` in `[coordinates.transforms]` names a time-varying quantity looked up per step from simulation metadata or auxiliary data. The rotation matrix is recomputed at each timestep.
  - **SPICE kernels:** Optional integration with `spiceypy` for ephemeris-based transforms (GSE↔HEE↔RTN, planetary frames). `from_spice(frame_a, frame_b, epoch)` builds a `FrameTransform` from NAIF kernels. Optional dep: `spiceypy` under `spice` extra. Useful for comparing simulation output with spacecraft observations in the correct frame at the correct epoch.
  `FieldDataset.transform_to(frame, *, epoch=None)` gains an optional epoch parameter. Static transforms (current behavior) are unchanged. Tests: round-trip GSE→GSM→GSE at known tilt angles against published rotation matrices.
  **Depends on:** Step 15 (frame transforms).

---

## Phase 5: Documentation

- [x] **Step 17: MkDocs documentation site**
  Material theme + mkdocstrings + mathjax. API reference (one page per module), getting-started guide, tutorial (load -> derive -> select -> compare -> plot). `mkdocs build --strict` passes.

---

## Phase 6: Regridding & Cross-Model Comparison

- [x] **Step 19: `pypic.regrid` — uniform-to-uniform interpolation**
  `regrid(source, target_grid, *, method="linear", **kwargs) -> FieldDataset` using `scipy.interpolate.RegularGridInterpolator` (same pattern as `traces/_sampling.py`). `method` is `str` (not enum) so new interpolation strategies can be added without API changes; `**kwargs` forwarded to the interpolator for future options. `align_grids(a, b) -> (FieldDataset, FieldDataset)` regrids both to the finer grid's intersection domain. `common_grid(a, b) -> GridInfo` computes that target: intersection domain = `max(origin_a, origin_b)` to `min(extent_a, extent_b)`, spacing = `min(dx_a, dx_b)` per axis; raises if domains don't overlap. Cartesian only (raise `NotImplementedError` for spherical/cylindrical, matching `operators.py` pattern). NaN-fill outside source domain via `bounds_error=False, fill_value=np.nan`. Preserves normalization, species, physics metadata. Works for 1D, 2D, and 3D grids. No-op shortcut when source grid already matches target (avoids interpolation for same-resolution comparisons). Each field array is interpolated independently; derived fields on source are regridded as-is, not recomputed.
  - *Not* a replacement for BATSRUS AMR regridding (block-avg/NN in `batsrus/_grid.py` operates on raw AMR cell data pre-FieldDataset; this module operates on assembled uniform grids via interpolation — different problems, different algorithms).
  - *Not* responsible for destaggering. Readers destagger to co-located grids on load (see Step 34). This module operates on already-co-located `FieldDataset` grids.
  - *Not* responsible for time alignment — Step 19 is purely spatial. Temporal interpolation (comparing different codes at matching physical time when dt differs) is a separate concern.
  - *Not* conservative. `method="linear"` does not preserve volume integrals of the field (energy, mass). Conservative regridding (`method="conservative"`) is a potential future option once an energy/mass-budget validation study motivates it; non-uniform-to-uniform support (OpenGGCM's stretched grids via `scipy.interpn`) is similarly deferred until a concrete user emerges.

- [ ] **Step 19b: spherical regridding for `pypic.regrid`**
  Extend `regrid()` / `common_grid()` / `align_grids()` to handle `GeometryType.SPHERICAL`. Metric-factor-aware interpolation on $(r, \theta, \phi)$ grids (not just tensor-product linear in the raw indices — the $\sin\theta$ Jacobian matters near the poles). Pole handling: clamp $\theta \in [\epsilon, \pi - \epsilon]$ or switch to a local Cartesian chart near each pole. Intersection grid semantics: $r$ extends like Cartesian; $\theta$ intersected in $[0, \pi]$; $\phi$ intersected modulo $2\pi$ with wrap-around support.
  Primary use case: comparing two ARMS runs at different angular resolutions (Step 36). Also unblocks Step 20b (volume-weighted comparison norms). Keeps the `NotImplementedError` branch in `_require_cartesian_grid` alive for cylindrical until that reader lands.
  Tests: spherical harmonic round-trip ($Y_\ell^m$ sampled on a coarse grid, regridded to fine, residual bounded by the truncation order), pole fidelity (analytic $\cos\theta$ field, zero error at $\theta = 0, \pi$ within interpolation tolerance), $\phi$-wrap correctness (periodic field resampled across the $\phi = 2\pi$ seam).
  **Depends on:** Step 19 (Cartesian regrid).

- [x] **Step 20: cross-grid comparison diagnostics**
  `compare_fields(a, b, field, *, metric="l2", units="si") -> float` — aligns grids then computes error. Converts to SI by default before comparing (cross-model normalizations are incomparable in code units; see schema.md). `units="code"` for same-normalization runs. Resolves field aliases before matching (e.g. "Bx" in A, "B1" in B → same canonical field). Logs a warning when grid resolutions differ by more than 10× (e.g. iPIC3D kinetic-scale vs BATSRUS MHD-scale — interpolation works but the comparison may be physically meaningless). Aligns *frames* before grids: *b* is auto-transformed into *a*'s frame via `FieldDataset.transform_to`, or pass `frame="..."` to transform both inputs to a third reference frame (raises `ValueError` if a required transform is missing). `field_comparison_report(a, b, *, fields=None, units="si") -> dict[str, dict[str, float]]` — L2 + Linf for all common fields, plus grid context (domain extent, resolution ratio) for interpretability. `field_difference_dataset(a, b, *, fields=None, units="si") -> FieldDataset` — returns a FieldDataset on the common grid with difference fields, directly plottable via `plot_comparison`. Delegates to existing pure diagnostics (`l2_relative_error`, `linf_error`, `field_difference` from `diagnostics.py`) after alignment — no reimplementation. These are the only diagnostics functions that touch FieldDataset (existing ones are pure-array); justified because cross-grid comparison inherently needs grid metadata.

- [ ] **Step 20b: volume-weighted comparison norms**
  Add metric-factor integration to `compare_fields()` / `field_comparison_report()` so L2 and L∞ correctly weight each cell by $\sqrt{|g|}\,d^n x$ instead of treating every sample uniformly. Current Step 20 is correct for uniform Cartesian grids (where $\Delta V$ cancels between numerator and denominator of the L2 norm), but wrong for spherical grids where cells near the poles or near $r = 0$ cover exponentially less volume. New API: `compare_fields(..., weighted: bool = False)` — defaults preserve current Cartesian behavior, `True` switches to the properly-weighted norm via `GridInfo.geometry.metric_factors()`. L∞ unaffected (max is a pointwise statistic). Tests: volume-weighted L2 of a radial shell equals the analytic shell volume; Cartesian result unchanged (regression test against current values).
  **Depends on:** Step 19b (spherical regridding — without it there is no spherical dataset to compare and this step is vacuous).

---

## Phase 7: CLI

- [x] **Step 21b: `sim.available_fields()` — lightweight field probe**
  `available_fields(step) -> list[str]` and `available_fields_mapping(step) -> dict[str, str | None]` list canonical field names without loading arrays. Implemented on iPIC3D, BATSRUS, SimpleReader; OpenGGCM falls back to full read.

- [x] **Step 21: `pypic.cli` — core subcommands (typer)**
  Entry point `pypic = "pypic.cli:app"`, optional deps `typer>=0.12` + `rich>=13.0` under `cli` extra. Global: `--version`, `--log-level`, `-q`, `--debug`. Shared `--step` syntax: `N`, `first`/`last`, `start:stop:stride` (inclusive stop), `all`.
  - `pypic info <path> [--json]` — metadata (grid, units, species, physics, steps).
  - `pypic fields <path> [--step N] [--mapping] [--derived] [--aux] [--all] [--json]` — available/computable fields, native→canonical mapping.
  - `pypic stats <path> --field FIELD [--step last] [--units UNIT] [--json]` — min/max/mean/rms/NaN. `--field all` batch-reads every field at once. Multi-step table via `--step all`.
  - `pypic compare <path_a> <path_b> [--field FIELD] [--metric l2|linf|both] [--units si|code] [--method METHOD] [--nan-policy omit|propagate|raise] [--frame FRAME] [--json]` — cross-grid error metrics, all common fields when `--field` omitted.
  - `pypic validate <path> [--step last] [--json]` — NaN census, max |div B|, magnetic/electric field energy. Reports energy drift (total + last step) from auxiliary `conserved_quantities` time-series when available.

- [x] **Step 22: `pypic plot` and `pypic plot-compare` CLI subcommands**
  `pypic plot <path> --field FIELD` with smart defaults (last step, largest cross-section plane). Geometry-aware `--plane` accepts Cartesian shorthands (`xy`/`xz`/`yz`), axis-name pairs (`rθ`), or a single normal axis name (`z`, `φ`). `--index I` or `--coord V` for slice position. Display: `--output FILE`, `--format png|pdf|svg`, `--dpi`, `--res WxH` (downsample), `--colormap`, `--scale linear|log|symlog`, `--linthresh`, `--vmin/--vmax`, `--theme NAME`, `--contour FIELD [--contour-levels N]`. Auto color range: 3σ from median, 1-2-5 rounded (`round_nice`, `auto_clim` in `_colormaps.py`). Batch: `--step all --output "frames/{step:06d}.png" --jobs N` (ThreadPoolExecutor). Animation: `--animate out.mp4 --fps 24` (ffmpeg concat demuxer).
  `pypic plot-compare <path_a> <path_b> --field FIELD` — three-panel (A | B | diff) via `align_grids` + `plot_comparison`. `--units si|code`, `--diff-vmin/--diff-vmax`, `--method`, `--theme`.

---

## Phase 8: Modern I/O Formats

- [ ] **Step 24: `pypic.io` — Zarr export/import for FieldDataset**
  `to_zarr(fds, path)` leveraging `xr.Dataset.to_zarr()` + pypic metadata as group attrs (grid, normalization, species, physics). `from_zarr(path) -> FieldDataset` reconstructs everything. Round-trip guarantee. Default compression: zstd. Optional dep: `zarr>=3.0` under `zarr` extra.
  **Field metadata:** Already self-describing via xarray DataArray attrs (`quantity_type`, `si_unit`, `long_name`, `latex`, `units`), set by `from_arrays()` and `with_field()`. `xr.Dataset.to_zarr()` serializes attrs automatically — no separate field registry module needed. `from_zarr()` reconstructs `FieldDataset` including per-field metadata. No CF vocabulary (CF has no plasma physics coverage).

- [ ] **Step 25: `pypic.io` — Parquet/Arrow for ParticleData**
  `particles_to_parquet(data, path)`, `particles_from_parquet(path) -> ParticleData`, `particles_to_arrow(data) -> pyarrow.Table` (zero-copy), `particles_from_arrow(table, ...) -> ParticleData`. Columnar storage: x/y/z/vx/vy/vz/charge/id columns. Species metadata in Parquet footer. Optional dep: `pyarrow>=17.0` under `arrow` extra.

- [ ] **Step 26: `pypic convert` CLI subcommand**
  `pypic convert <path> --step N --output DIR [--format zarr|parquet] [--fields F1,F2] [--target-resolution DX]`. Batch mode: `--all-steps`.

---

## Phase 9: Additional Readers

- [ ] **Step 23: `pypic.readers.vlasiator` — VLSV reader via analysator**
  `VLasiatorReader` implementing `SimulationReader`. Two-grid strategy: FSgrid fields (`fg_b`, `fg_e`) read directly as uniform arrays; DCCRG fields (`proton/vg_rho`, `proton/vg_v`, `proton/vg_p`) regridded to uniform at `target_resolution` (default: FSgrid resolution). DCCRG cell IDs encode position + refinement level — decode to (x, y, z, dx) then block-average/NN-repeat (like BATSRUS AMR pattern, not `pypic.regrid` which is for uniform→uniform).
  Field mapping: `fg_b` → `B1/B2/B3`, `fg_e` → `E1/E2/E3`, `proton/vg_rho` → `n_s0`, `proton/vg_v` → `V1/V2/V3`, `proton/vg_p` (6 components) → pressure tensor. Species auto-detected from VLSV population names. Auto-detection: `.vlsv` extension + file signature. `open_vlasiator()` convenience function. Optional dep: `analysator` under `vlasiator` extra. All tests mock analysator.

- [ ] **Step 35: `pypic.readers.vpic` — VPIC reader**
  `VPICReader` implementing `SimulationReader`. VPIC writes per-rank binary files (band-interleaved by field) or HDF5 via `vpic_decks`. Field mapping: `cbx/cby/cbz` → `B1/B2/B3` (cell-centered B), `ex/ey/ez` → `E1/E2/E3` (Yee edge), `jfx/jfy/jfz` → `J1/J2/J3`, `rhob` → `rho_c`, per-species hydro files → density, velocity, pressure tensor. Yee mesh destaggering to co-located grid (linear interpolation, `StaggerInfo(convention="staggered")`). Metadata from `info` dumps or deck header. Auto-detection: `global.vpc` or `info` file presence. `open_vpic()` convenience function. All tests use synthetic fixtures.

- [ ] **Step 36: `pypic.readers.arms` — ARMS reader**
  `ARMSReader` implementing `SimulationReader`. ARMS (Adaptively Refined MHD Solver) outputs HDF5 with block-structured AMR. Regrid to uniform grid at `target_resolution` (like BATSRUS pattern). Field mapping from ARMS native names to canonical schema. Spherical geometry support (ARMS is commonly run in spherical coordinates for coronal/heliospheric simulations). `StaggerInfo(convention="staggered")` — ARMS uses a staggered mesh (CT for divergence-free B). Auto-detection: ARMS-specific HDF5 group structure. `open_arms()` convenience function. All tests use synthetic fixtures.

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

**Dependency graph:**

```
Steps 13-14 (compute/plot) ←── Step 22 (plot CLI)
                           ←── Step 20 (field_difference_dataset → plot_comparison)
                           ←── Step 21b (available_fields) ←── Step 21 (CLI core) ←── Step 26 (convert CLI)
Step 19 (regrid) ←── Step 20 (cross-grid diagnostics) ←── Step 21
              ←── Step 19b (spherical regrid) ←── Step 20b (volume-weighted norms)
Step 15 (frame transforms) ←── Step 40 (time-dependent transforms)
Step 5 (FieldDataset) ←── Steps 24, 25 (Zarr/Arrow)
                      ←── Steps 23, 35, 36 (additional readers)
                      ←── Step 27 (interop adapters)
```

Recommended implementation order: 19 → 20 → 21b → 21 → 22, with 24/25 parallelizable anytime, 26 after 21+24+25, 40 anytime after Step 15, 23/35/36 anytime after Step 12 (readers exist), 27–28 anytime after API stabilizes.

---

## Phase 11: Geometry & Type System Hardening

Design weaknesses identified during the unit/geometry audit. These are
not bugs — current behavior is correct for Cartesian data — but will
become problems as non-Cartesian geometries and relativistic workflows
grow.

- [x] **Step 30: Reduced geometry after slicing**
  After `PlaneSelection.apply()` reduces 3D→2D, the `GridInfo` keeps the
  original 3-axis `CoordinateGeometry`. Code uses `axis_names[:ndim]` to
  get surviving names, which gives the **first N** names, not the
  **surviving** names (e.g. slicing the r-axis from spherical gives
  surviving (θ, φ) but `axis_names[:2]` returns (r, θ)). Needs a concept
  of "reduced geometry" or storing surviving axis indices. Affects
  `PlaneSelection.apply()`, `BoxSelection.apply()`, and
  `_build_grid_from_dataset()` in `readers/base.py`. No impact on
  shipped readers (all Cartesian), but blocks correct spherical/cylindrical
  slicing.
  **Depends on:** Step 15 (frame transforms) or Step 16 (sphere selection).

- [~] **Step 31: Remove default geometry from operators (deferred)**
  Operators default to `GeometryType.CARTESIAN`. Removing the default
  adds verbosity to 23 Cartesian test sites with zero safety gain (non-
  Cartesian already raises `NotImplementedError`). Instead, `compute.py`
  threads geometry via `_Recipe.passes_geometry` — a no-op today but
  pre-wired for when spherical/cylindrical operators land (Step 10 ext.).
  **Revisit when:** non-Cartesian operators are implemented.

- [x] **Step 32: Separate `four_velocity` quantity type**
  `u1/u2/u3` (four-velocity, γv, unbounded) share `quantity_type="velocity"`
  with `V1/V2/V3` (three-velocity, bounded by c). SI conversion is correct
  (both have units of m/s), but the shared type prevents distinguishing them
  in validation or display contexts. A separate `"four_velocity"` type with
  the same SI factor would make the semantics explicit. Low priority — only
  matters for relativistic workflows.
  **Depends on:** Step 18 (relativistic derived quantities).

- [x] **Step 33: `specific_energy` quantity type for enthalpy**
  `h`, `h_rel`, `e_int` used `quantity_type="temperature"` with SI factor
  `mass_ref * velocity_ref²` (J). The correct SI factor for specific energy
  (energy per unit mass) is `velocity_ref²` (J/kg) — this was a dimensional
  bug masked by `identity()` normalization in tests. Fixed by adding a
  `"specific_energy"` quantity type with the correct factor.

- [x] **Step 34: `StaggerInfo` provenance metadata**
  Optional frozen dataclass recording original grid stagger convention
  before destaggering: which fields lived on faces, edges, nodes, or
  cell centers. Stored in `FieldDataset.metadata["stagger"]` by readers
  that load from staggered-mesh codes (ARMS, Athena++, BATSRUS
  face-centered). Purely informational — not used in computation or
  operators. Enables provenance tracking and documentation of
  interpolation order used during destaggering.
  **Architecture note:** Readers are responsible for destaggering to
  co-located grids. `FieldDataset` always represents a single co-located
  grid. Diagnostics like `div_b` on destaggered data measure interpolation
  error + actual divergence; for staggered codes, O(dx²) residual is
  expected and does not indicate a simulation defect. Carrying stagger
  through the pipeline (stagger-aware operators) is explicitly out of
  scope — the complexity cost outweighs the benefit for analysis workflows.
  **Depends on:** Step 12 (readers).

---

## Phase 12: Virtual Probes & Spacecraft

- [ ] **Step 41: `pypic.probes` — virtual probe sampling**
  `Probe` frozen dataclass: a named point `(x, y, z)` in the simulation domain. `ProbeArray`: collection of probes (detector arrays, virtual satellite constellations). `ProbeTrajectory`: time-varying position as `(t, x, y, z)` array — a spacecraft orbit or moving detector path. A fixed probe is a degenerate trajectory (constant position).
  Core functions:
  - `sample(probe, dataset) -> dict[str, float]` — interpolate all fields at the probe position for one timestep. Reuses `RegularGridInterpolator` from `traces/_sampling.py`.
  - `sample_timeseries(probe, simulation, steps) -> TabularData` — sample across timesteps, producing time-series columns (time, B1, B2, B3, ...). Output is `TabularData` (already exists).
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

- [ ] **Step 37: `pypic.server` — Arrow IPC streaming via Starlette/FastAPI**
  Zero-copy field data serving to webpic (Three.js viewer). Selections from the viewer UI map to pypic `Selection` objects server-side. Lazy I/O via xarray/dask serves only requested slices from disk. Arrow IPC replaces raw ArrayBuffers with structured metadata (field names, coordinates, units, normalization) in a single response. Uses `xr.Dataset` → Arrow conversion. Readable in JS (`apache-arrow`) and Rust (`arrow-rs`), aligning all three projects on one interchange format. Derived quantities computed server-side via `compute()`, unit conversion via `in_si()` / `in_units()`. Optional dep: `fastapi`, `uvicorn`, `pyarrow` under `server` extra. The server is a separate entry point, not part of the library import path.
  **Depends on:** Steps 24-25 (Zarr/Arrow foundations).

- [ ] **Step 38: `pypic.readers.rustpic` — Rust PIC code reader**
  Reader for rustpic's schema.md-conformant HDF5 output. The Rust code writes the canonical HDF5 layout directly (Section 4 of schema.md), so this is essentially `SimpleReader` with rustpic-specific metadata extraction and validation. pypic serves as the **reference implementation** — validate Rust-computed derived quantities against Python results on the same problem. Cross-project integration tests: run both codes on identical initial conditions, compare via `field_comparison_report()`. `open_rustpic()` convenience function. Auto-detection via HDF5 `model` attribute = `"rustpic"`.
  **Depends on:** Step 20 (cross-grid comparison diagnostics).

- [ ] **Step 39: webpic data pipeline documentation**
  End-to-end guide for the full platform: rustpic (Rust simulation) → HDF5 → pypic (Python analysis) → Starlette/FastAPI + Arrow IPC → webpic (Three.js/WebGPU visualization). Documents the schema.md contract that keeps Python, Rust, and JavaScript in sync. Selection round-trip: viewer UI selection → server `Selection` object → `FieldDataset` slice → Arrow IPC → GPU buffer. Coordinate transform pipeline: viewer requests a frame → server calls `transform_to()` → transformed data streamed. Covers: authentication model, chunked transfer for large datasets, WebSocket option for time-series animation.

---

## Summary

| Step | Module | Delivers | Status |
|------|--------|----------|--------|
| 1 | skeleton | CI pipeline, project config | ✅ |
| 2 | units | Normalization (PIC / MHD / SI) | ✅ |
| 3 | units | PhysicsConstants, SpeciesInfo | ✅ |
| 4 | coordinates | CoordinateGeometry, metric factors | ✅ |
| 5 | readers | FieldDataset, GridInfo, SimulationReader, SimulationConfig | ✅ |
| 6 | readers | simulation.toml loader | ✅ |
| 7 | derived | \|B\|, beta, v_A, Poynting flux, energies | ✅ |
| 8 | derived | omega_pe, d_i, r_i, lambda_D, v_th, c_s | ✅ |
| 8b | compute | Per-species P_par, P_perp, agyrotropy recipes | — |
| 9 | diagnostics | L2 error, div B, field energy | ✅ |
| 10 | coordinates | curl, div, grad (Cartesian) | ✅ |
| 11 | selections | Plane, Box | ✅ |
| 12 | readers | iPIC3D, BATSRUS, OpenGGCM, Simple readers + registry + auto-detection | ✅ |
| 13 | fields | compute(), in_units(), QuantityType, field metadata registry | ✅ |
| 14 | plotting | 2D slices, comparison | ✅ |
| **—** | **—** | **Milestone: daily-use tool** | **—** |
| 15 | coordinates | Frame transforms | ✅ |
| 16 | selections | Sphere (NaN masking) | ✅ |
| 17 | docs | MkDocs site | ✅ |
| 18 | derived | lorentz_factor, magnetization, rel. corrections | ✅ |
| 40 | coordinates | Time-dependent frame transforms (dipole tilt, SPICE) | — |
| 19 | regrid | `regrid()`, `align_grids()`, `common_grid()` (Cartesian) | ✅ |
| 19b | regrid | Spherical regridding (metric-aware, pole + $\phi$-wrap) | — |
| 20 | comparison | `compare_fields()`, `field_comparison_report()`, `field_difference_dataset()` | ✅ |
| 20b | comparison | Volume-weighted L2 norm via `weighted=True` | — |
| 21b | readers | `sim.available_fields()` + `available_fields_mapping()` | ✅ |
| 21 | cli | `info`, `fields`, `stats`, `compare`, `validate` subcommands (typer) | ✅ |
| 22 | cli | `plot`, `plot-compare` subcommands + theme, contours, animate | ✅ |
| 24 | io | Zarr export/import for FieldDataset | — |
| 25 | io | Parquet/Arrow for ParticleData | — |
| 26 | cli | `convert` subcommand | — |
| 23 | readers | VLasiator VLSV reader (FSgrid + DCCRG regrid) | — |
| 35 | readers | VPIC reader (Yee mesh destaggering) | — |
| 36 | readers | ARMS reader (block-AMR, spherical) | — |
| 27 | interop | yt, PlasmaPy, SpacePy thin adapters | — |
| 28 | docs | Ecosystem positioning page | — |
| 29 | interop | SPASE XML metadata export | — |
| 30 | readers/selections | Reduced geometry after slicing | ✅ |
| 31 | coordinates | Remove default geometry from operators (deferred — see note) | ⏸ |
| 32 | fields/units | Separate `four_velocity` quantity type | ✅ |
| 33 | fields/units | `specific_energy` quantity type for enthalpy | ✅ |
| 34 | readers | `StaggerInfo` provenance metadata | ✅ |
| 41 | probes | Virtual probe/spacecraft sampling + time-series | — |
| 41b | probes | SPICE-driven probe trajectories | — |
| 37 | server | Arrow IPC streaming via Starlette/FastAPI → webpic | — |
| 38 | readers | rustpic reader + cross-project validation | — |
| 39 | docs | webpic data pipeline end-to-end guide | — |
