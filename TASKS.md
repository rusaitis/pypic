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

- [ ] **Step 13: FieldDataset — compute() and in_units()**
  `compute(name)` dispatches string to derived function ("|B|", "beta", "v_A", "M_A", "|vort|", "vort1"/"vort2"/"vort3", ...). `in_si(field)` for SI conversion. `in_units(field, unit_str)` for display units ("nT", "km/s").

- [ ] **Step 14: plotting/slices — basic 2D visualization**
  `plot_field_slice` (plane selection, axis labels, colorbar). `plot_comparison` (three-panel: A | B | difference). Publication rcParams in `plotting/styles.py`.

**Milestone: daily-use tool** — load iPIC3D data -> compute derived quantities -> compare runs -> select subregions -> convert units -> make paper figures.

---

## Phase 4: Extensions

- [ ] **Step 15: coordinates/transforms — frame transforms**
  `ReferenceFrame`, `FrameTransform` dataclasses. Load from `[coordinates.transforms]` in simulation.toml. Transform chaining (A->B + B->C = A->C). `FieldDataset.transform_to(frame_name)`.

- [ ] **Step 16: selections — SphereSelection**
  Non-axis-aligned selection via `xr.where()`. Points outside sphere = NaN, grid shape preserved. NaN propagation in derived quantities and plotting.

- [ ] **Step 18: derived (relativistic) — relativistic derived quantities**
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

- [ ] **Step 17: MkDocs documentation site**
  Material theme + mkdocstrings + mathjax. API reference (one page per module), getting-started guide, tutorial (load -> derive -> select -> compare -> plot). `mkdocs build --strict` passes.

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
| 13 | fields | compute(), in_units() | — |
| 14 | plotting | 2D slices, comparison | — |
| **—** | **—** | **Milestone: daily-use tool** | **—** |
| 15 | coordinates | Frame transforms | — |
| 16 | selections | Sphere (NaN masking) | — |
| 18 | derived | lorentz_factor, magnetization, rel. corrections | — |
| 17 | docs | MkDocs site | — |
