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
  `load_config(path) -> SimulationConfig` via `tomllib`. Parses all SCHEMA.md sections: `[model]` -> name/type, `[grid]` -> `GridInfo`, `[units]` -> `Normalization`, `[coordinates]` -> `CoordinateGeometry` + frame, `[[species]]` -> `list[SpeciesInfo]`, `[physics]` -> dict. Optional: `[initial_conditions]`, `[output]` -> metadata.

- [x] **Step 7: derived (part 1) — field-level quantities**
  Pure NumPy functions: `magnetic_field_magnitude`, `electric_field_magnitude`, `current_density_magnitude`, `velocity_magnitude`, `plasma_beta`, `alfven_speed`, `poynting_flux`, `magnetic_energy_density`, `electric_energy_density`, `kinetic_energy_density`, `thermal_energy_density`, `internal_energy`, `enthalpy`, `relativistic_enthalpy`, `entropy`, `gyrotropic_entropy`.

- [x] **Step 8: derived (part 2) — characteristic scales**
  Species-dependent: `thermal_speed`, `gyrofrequency`, `plasma_frequency`, `skin_depth`, `gyroradius`, `debye_length`, `sound_speed`, `ion_acoustic_speed`, `magnetosonic_speed`, `alfven_mach`, `magnetosonic_mach`, `parallel_pressure`, `perpendicular_pressure`, `agyrotropy`. Verify against NRL Formulary.

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

---

## Phase 5: Documentation

- [x] **Step 17: MkDocs documentation site**
  Material theme + mkdocstrings + mathjax. API reference (one page per module), getting-started guide, tutorial (load -> derive -> select -> compare -> plot). `mkdocs build --strict` passes.

---

## Phase 6: Regridding & Cross-Model Comparison

- [ ] **Step 19: `pypic.regrid` — uniform-to-uniform interpolation**
  `regrid(source, target_grid, *, method="linear") -> FieldDataset` using `scipy.interpolate.RegularGridInterpolator`. `align_grids(a, b) -> (FieldDataset, FieldDataset)` regrids both to the finer grid's intersection domain. `common_grid(a, b) -> GridInfo` computes that target. Cartesian only (raise `NotImplementedError` for spherical/cylindrical, matching `operators.py` pattern). NaN-fill outside source domain. Preserves normalization, species, physics metadata.
  - *Not* a replacement for BATSRUS AMR regridding (block-avg/NN in `batsrus/_grid.py` operates on raw AMR cell data pre-FieldDataset; this module operates on assembled uniform grids via interpolation — different problems, different algorithms).
  - *Not* responsible for destaggering. Readers destagger to co-located grids on load (see Step 34). This module operates on already-co-located `FieldDataset` grids.

- [ ] **Step 20: cross-grid comparison diagnostics**
  `compare_fields(a, b, field, *, metric="l2") -> float` — aligns grids then computes error. `field_comparison_report(a, b, *, fields=None) -> dict[str, dict[str, float]]` — L2 + Linf for all common fields. These are the only diagnostics functions that touch FieldDataset (existing ones are pure-array); justified because cross-grid comparison inherently needs grid metadata.

---

## Phase 7: CLI

- [ ] **Step 21: `pypic.cli` — core subcommands (typer)**
  `pypic info <path>` (simulation metadata, steps, fields). `pypic fields <path> [--step N]` (canonical + alias field names). `pypic compare <path_a> <path_b> --step N --field FIELD [--metric l2|linf|both]` (numeric comparison, or table for all common fields when `--field` omitted). Entry point: `[project.scripts] pypic = "pypic.cli:app"`. Optional deps: `typer>=0.12`, `rich>=13.0` under `cli` extra.

- [ ] **Step 21b: `sim.available_fields()` — lightweight field probe**
  `Simulation.available_fields(step) -> list[str]` lists canonical field names in a timestep without loading arrays. New optional `SimulationReader` protocol method `available_fields(path, step) -> list[str]`. HDF5 readers (iPIC3D, Simple) list datasets via `h5py`; BATSRUS parses the header; others fall back to full read + `field_names()`. Prerequisite for `pypic fields` CLI command (Step 21). Also useful for selective `read(fields=...)` discovery.

- [ ] **Step 22: `pypic plot` and `pypic plot-compare` CLI subcommands**
  `pypic plot <path> --step N --field FIELD [--plane xy|xz|yz] [--index I] [--output FILE]` — reads, selects plane via `PlaneSelection`, resolves derived fields via `compute()`, calls `plot_field_slice`. `pypic plot-compare` — three-panel (A | B | difference). Plane shorthand: `--plane xy` → `PlaneSelection(normal="z")`. Diverging colormap for signed fields, sequential for positive-definite.

---

## Phase 8: VLasiator Reader

- [ ] **Step 23: `pypic.readers.vlasiator` — VLSV reader via analysator**
  `VLasiatorReader` implementing `SimulationReader`. Two-grid strategy: FSgrid fields (`fg_b`, `fg_e`) read directly as uniform arrays; DCCRG fields (`proton/vg_rho`, `proton/vg_v`, `proton/vg_p`) regridded to uniform at `target_resolution` (default: FSgrid resolution). DCCRG cell IDs encode position + refinement level — decode to (x, y, z, dx) then block-average/NN-repeat (like BATSRUS AMR pattern, not `pypic.regrid` which is for uniform→uniform).
  Field mapping: `fg_b` → `B1/B2/B3`, `fg_e` → `E1/E2/E3`, `proton/vg_rho` → `n_s0`, `proton/vg_v` → `V1/V2/V3`, `proton/vg_p` (6 components) → pressure tensor. Species auto-detected from VLSV population names. Auto-detection: `.vlsv` extension + file signature. `open_vlasiator()` convenience function. Optional dep: `analysator` under `vlasiator` extra. All tests mock analysator.

---

## Phase 9: Modern I/O Formats

- [ ] **Step 24: `pypic.io` — Zarr export/import for FieldDataset**
  `to_zarr(fds, path)` leveraging `xr.Dataset.to_zarr()` + pypic metadata as group attrs (grid, normalization, species, physics). `from_zarr(path) -> FieldDataset` reconstructs everything. Round-trip guarantee. Default compression: zstd. Optional dep: `zarr>=3.0` under `zarr` extra.
  **Field metadata:** Already self-describing via xarray DataArray attrs (`quantity_type`, `si_unit`, `long_name`, `latex`, `units`), set by `from_arrays()` and `with_field()`. `xr.Dataset.to_zarr()` serializes attrs automatically — no separate field registry module needed. `from_zarr()` reconstructs `FieldDataset` including per-field metadata. No CF vocabulary (CF has no plasma physics coverage).

- [ ] **Step 25: `pypic.io` — Parquet/Arrow for ParticleData**
  `particles_to_parquet(data, path)`, `particles_from_parquet(path) -> ParticleData`, `particles_to_arrow(data) -> pyarrow.Table` (zero-copy), `particles_from_arrow(table, ...) -> ParticleData`. Columnar storage: x/y/z/vx/vy/vz/charge/id columns. Species metadata in Parquet footer. Optional dep: `pyarrow>=17.0` under `arrow` extra.

- [ ] **Step 26: `pypic convert` CLI subcommand**
  `pypic convert <path> --step N --output DIR [--format zarr|parquet] [--fields F1,F2] [--target-resolution DX]`. Batch mode: `--all-steps`.

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
                           ←── Step 21 (CLI core) ←── Step 26 (convert CLI)
Step 19 (regrid) ←── Step 20 (cross-grid diagnostics) ←── Step 21
                 ←── Step 23 (VLasiator, for DCCRG context)
Step 5 (FieldDataset) ←── Steps 24, 25 (Zarr/Arrow)
                      ←── Step 27 (interop adapters)
```

Recommended implementation order: 19 → 20 → 21 → 22 → 23, with 24/25 parallelizable anytime, 26 after 21+24+25, 27–28 anytime after API stabilizes.

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
  `divergence()`, `curl()`, `gradient()` in `coordinates/operators.py`
  default to `GeometryType.CARTESIAN`. The original plan was to drop the
  default and force every caller to be explicit. **Deferred** after the
  audit showed there is no silent wrong behavior to guard against today
  (non-Cartesian raises `NotImplementedError` immediately) and removing
  the default would force `geometry=GeometryType.CARTESIAN` onto 23
  intentionally-Cartesian test sites — pure verbosity, zero added safety.
  Instead, the FieldDataset → recipe → operator path now threads the
  dataset's geometry through as a kwarg via `_Recipe.passes_geometry`
  in `compute.py`. Today this is a no-op (the early geometry guard at
  `compute_field` still raises for non-Cartesian) but it pre-wires the
  recipe path so spherical/cylindrical "just work" once the operators
  themselves implement them.
  **Revisit when:** spherical/cylindrical operator implementations land
  (Step 10 extension). At that point, relax the early raise in
  `compute_field` for `passes_geometry` recipes.

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

- [ ] **Step 34: `StaggerInfo` provenance metadata**
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
| 19 | regrid | `regrid()`, `align_grids()`, `common_grid()` | — |
| 20 | diagnostics | `compare_fields()`, `field_comparison_report()` | — |
| 21 | cli | `info`, `fields`, `compare` subcommands (typer) | — |
| 21b | readers | `sim.available_fields()` lightweight field probe | — |
| 22 | cli | `plot`, `plot-compare` subcommands | — |
| 23 | readers | VLasiator VLSV reader (FSgrid + DCCRG regrid) | — |
| 24 | io | Zarr export/import for FieldDataset | — |
| 25 | io | Parquet/Arrow for ParticleData | — |
| 26 | cli | `convert` subcommand | — |
| 27 | interop | yt, PlasmaPy, SpacePy thin adapters | — |
| 28 | docs | Ecosystem positioning page | — |
| 29 | interop | SPASE XML metadata export | — |
| 30 | readers/selections | Reduced geometry after slicing | ✅ |
| 31 | coordinates | Remove default geometry from operators (deferred — see note) | ⏸ |
| 32 | fields/units | Separate `four_velocity` quantity type | ✅ |
| 33 | fields/units | `specific_energy` quantity type for enthalpy | ✅ |
| 34 | readers | `StaggerInfo` provenance metadata | — |
