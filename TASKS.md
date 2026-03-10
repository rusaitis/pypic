# Plasma Analysis Toolkit: Implementation Task List

Each step produces something testable. No step starts until the previous
step's tests pass. Steps 1–14 reach a working daily-use tool. Steps 15–17
add polish.

---

## Step 1: Project skeleton

Create project structure and tooling configuration.

**Files:**
- `pyproject.toml` — uv project, ruff (NumPy docstring convention), pytest
  (`--doctest-modules`), mypy (strict)
- `src/plasma_analysis/__init__.py`
- `src/plasma_analysis/py.typed`
- `tests/__init__.py`
- `mkdocs.yml` — minimal config (build later in Step 17)
- `.github/workflows/ci.yml` — test + lint + type-check on every push

**Tests / verification:**
- `uv run pytest` passes (empty)
- `uv run ruff check src` passes
- `uv run mypy src` passes
- CI runs green on push

---

## Step 2: `units.py` — Normalization class

Core unit conversion. Mirrors the Rust `Normalization` struct.

**Implement:**
- `Normalization` frozen dataclass with all reference values
- `pic_standard(reference_density, reference_mass, reference_charge, c)`
  classmethod — derives from any reference species
- `pic_electron(n_e)` convenience classmethod — electron scales (ω_pe, d_e)
- `pic_ion(n_i, mass_i, charge_i)` convenience classmethod — ion scales (ω_pi, d_i)
- `mhd_standard(l_0, rho_0, b_0)` classmethod — derives from Alfvén speed
- `identity()` classmethod — all reference values = 1.0
- `normalize_*()` and `to_si_*()` methods (length, time, velocity, b_field,
  e_field, density) — work on both scalars and NumPy arrays

**Dependencies:** `scipy.constants`

**Tests:**
- `pic_electron(n_e).normalize_velocity(sc.c) == 1.0`
- `pic_ion(n_i, sc.m_p).normalize_velocity(sc.c) == 1.0`
- `pic_standard` with electron params matches `pic_electron`
- `pic_standard` with proton params matches `pic_ion`
- `mhd_standard`: v_A derived correctly from inputs
- Round-trip: `to_si_*(normalize_*(x)) == x` to machine precision
- `identity()`: all conversions return input unchanged
- Array inputs: conversions work on NumPy arrays, not just scalars
- Doctest examples in every method

---

## Step 3: `units.py` — PhysicsConstants and SpeciesInfo

Constants and species metadata needed by derived quantity functions.

**Implement:**
- `PhysicsConstants` dataclass (c, epsilon_0, mu_0)
- `PhysicsConstants.pic_normalized()` — c=1, ε₀=1, μ₀=1
- `PhysicsConstants.mhd_normalized()` — c=∞, ε₀=1, μ₀=1
- `PhysicsConstants.inv_c_squared()` — returns 0 for c=∞
- `SpeciesInfo` frozen dataclass (name, charge, mass, charge_to_mass,
  temperature, thermal_velocity, drift_velocity, density, particles_per_cell)
- Inference logic: charge_to_mass → charge + mass (and vice versa)

**Tests:**
- `PhysicsConstants.pic_normalized().c == 1.0`
- `PhysicsConstants.mhd_normalized().inv_c_squared() == 0.0`
- `SpeciesInfo` from charge + mass gives correct charge_to_mass property
- `SpeciesInfo` from charge_to_mass alone infers charge and mass
- Validation: error if neither charge+mass nor charge_to_mass provided

---

## Step 4: `coordinates/geometry.py` — CoordinateGeometry

Coordinate system definitions with metric scale factors.

**Implement:**
- `CoordinateGeometry` dataclass (type, axis_names, axis_units)
- Module-level constants: `CARTESIAN`, `SPHERICAL`, `CYLINDRICAL`
- `metric_factors(grid) → (h1, h2, h3)` — returns NumPy arrays of
  scale factors at each grid point

**Tests:**
- Cartesian: h1 = h2 = h3 = 1 everywhere
- Spherical: h1 = 1, h2 = r, h3 = r sin θ at known grid points
- Cylindrical: h1 = 1, h2 = r, h3 = 1 at known grid points
- Axis names match geometry (x,y,z vs r,θ,φ vs r,φ,z)

---

## Step 5: `readers/base.py` — FieldDataset and GridInfo

The universal data container. Everything else consumes this type.

**Implement:**
- `GridInfo` dataclass (dimensions, spacing, origin, dt, boundary, geometry)
- `FieldDataset` class wrapping `xr.Dataset`:
  - `__getitem__(key) → NDArray` (zero-copy via `.values`)
  - `has_field(key) → bool`
  - `field_names() → list[str]`
  - `sel(**kwargs) → FieldDataset` (coordinate-aware, wraps xr.sel)
  - `isel(**kwargs) → FieldDataset` (index-based, wraps xr.isel)
  - `.xr` property → raw xr.Dataset
  - `.grid`, `.normalization`, `.species`, `.physics`, `.metadata` attrs
- `SimulationReader` protocol (read_timestep, available_timesteps)
- `SimulationConfig` dataclass

**Dependencies:** `xarray`, `numpy`

**Tests:**
- Construct FieldDataset from synthetic NumPy arrays
- `data["B1"]` returns NumPy array (not xarray DataArray)
- `data["B1"]` is zero-copy (shares memory with underlying xr.Dataset)
- `data.sel(z=0)` returns 2D FieldDataset
- `data.isel(z=0)` returns 2D FieldDataset
- `.xr` returns xarray Dataset
- `field_names()` matches input keys
- `has_field("B1")` is True, `has_field("nonexistent")` is False

---

## Step 6: `readers/config.py` — simulation.toml loader

Parse the shared configuration format (SCHEMA.md spec).

**Implement:**
- `load_config(path) → SimulationConfig` using `tomllib`
- Parse `[units]` → `Normalization` (Approach A: named, Approach B: explicit)
- Parse `[[species]]` → `list[SpeciesInfo]`
- Parse `[grid]` → `GridInfo`
- Parse `[coordinates]` → `CoordinateGeometry` + frame info
- Parse `[physics]`, `[initial_conditions]` → dicts

**Tests:**
- Load the iPIC3D Double Harris example toml (create as test fixture)
- Grid dimensions = [100, 100, 1]
- 4 species parsed with correct charge_to_mass values
- Normalization system = "PIC"
- `[initial_conditions]` type = "double_harris"
- Missing optional sections don't crash (graceful defaults)
- Invalid toml raises clear error

---

## Step 7: `derived.py` — core derived quantities (part 1)

Pure NumPy functions for the most common field-level computations.

**Implement:**
- `magnetic_field_magnitude(b1, b2, b3) → NDArray`
- `electric_field_magnitude(e1, e2, e3) → NDArray`
- `current_density_magnitude(j1, j2, j3) → NDArray`
- `velocity_magnitude(v1, v2, v3) → NDArray`
- `plasma_beta(p, b1, b2, b3) → NDArray`
- `alfven_speed(b_mag, rho_m) → NDArray`
- `poynting_flux(e1, e2, e3, b1, b2, b3) → (s1, s2, s3)`
- `magnetic_energy_density(b1, b2, b3) → NDArray`
- `kinetic_energy_density(rho_m, v1, v2, v3) → NDArray`
- `thermal_energy_density(p, gamma) → NDArray`

Each function: NumPy-style docstring, `r"""`, LaTeX math, doctest example.

**Tests:**
- `magnetic_field_magnitude(3, 4, 0) == 5`
- `plasma_beta` of known P and B matches hand calculation
- `alfven_speed(1.0, 4.0) == 0.5` (normalized, μ₀ = 1)
- `poynting_flux` of orthogonal E, B gives correct direction
- All functions work on 1D, 2D, 3D arrays
- All doctests pass via `--doctest-modules`

---

## Step 8: `derived.py` — core derived quantities (part 2)

Species-dependent characteristic scales.

**Implement:**
- `thermal_speed(temperature, mass) → NDArray` — uses √(T/m) convention
- `gyrofrequency(b_mag, charge, mass) → NDArray`
- `plasma_frequency(density, charge, mass) → NDArray`
- `skin_depth(density, charge, mass, c) → NDArray`
- `gyroradius(temperature, b_mag, charge, mass) → NDArray`
- `debye_length(density, temperature, charge) → NDArray`
- `sound_speed(pressure, rho_m, gamma) → NDArray`
- `ion_acoustic_speed(te, ti, mass_i, gamma_i) → NDArray`
- `magnetosonic_speed(v_a, c_s) → NDArray`

**Tests:**
- `thermal_speed` uses √(T/m) (NRL convention), not √(2T/m)
- Each function verified against NRL Formulary numerical values in SI
- Each function also verified in normalized units (dimensionless check)
- `magnetosonic_speed` ≥ both v_A and c_s always

---

## Step 9: `diagnostics.py` — comparison and validation

Functions for comparing runs and validating simulation output.

**Implement:**
- `l2_relative_error(reference, computed) → float`
- `linf_error(reference, computed) → float`
- `field_difference(a, b) → NDArray`
- `field_energy(b1, b2, b3) → float` (integrated over domain)
- `div_b(b1, b2, b3, dx, dy, dz) → NDArray` (Cartesian, central diff)
- `max_div_b(b1, b2, b3, dx, dy, dz) → float`

**Tests:**
- `l2_relative_error` of identical arrays is 0.0
- `l2_relative_error` of known difference matches hand calculation
- `field_difference` of identical arrays is all zeros
- `div_b` of a curl field is zero to machine precision (~1e-14)
- `field_energy` of uniform B = 1 on 10³ grid with dx = 1: E = 500
- `max_div_b` of divergence-free field is ~machine epsilon

---

## Step 10: `coordinates/operators.py` — discrete differential operators

Geometry-aware curl, div, grad. Cartesian only for now.

**Implement:**
- `curl(f1, f2, f3, grid) → (c1, c2, c3)` — dispatches on geometry
- `div(f1, f2, f3, grid) → NDArray` — dispatches on geometry
- `grad(scalar, grid) → (g1, g2, g3)` — dispatches on geometry
- Cartesian: second-order central differences
- Spherical/cylindrical: raise `NotImplementedError` with clear message

**Tests:**
- Curl of uniform field is zero
- Curl of known analytic vector field matches analytic curl
- Div of curl is zero (to machine precision)
- Grad of linear field (f = ax + by + cz) is constant (a, b, c)
- Div of known analytic field matches analytic divergence
- Spherical geometry raises `NotImplementedError("spherical ... not yet implemented")`

---

## Step 11: `selections.py` — PlaneSelection and BoxSelection

Data subsetting that returns standard FieldDataset objects.

**Implement:**
- `PlaneSelection` frozen dataclass (normal, index)
  - `apply(data) → FieldDataset` via xarray `.isel()`
  - `index=None` defaults to midplane
- `BoxSelection` frozen dataclass (x, y, z as optional index tuples)
  - `apply(data) → FieldDataset` via xarray `.isel()`
  - `None` for any axis means full range

**Tests:**
- Plane selection on 3D (64,64,64) data returns (64,64) FieldDataset
- Box selection with x=(10,50) reduces first dimension
- None dimensions are preserved
- Midplane default: index=None → nx//2
- Composition: box then plane works correctly
- Result is a valid FieldDataset (compute, field_names, etc. all work)

---

## Step 12: `readers/ipic3d.py` — iPIC3D HDF5 reader

First real data reader. Connects the toolkit to actual simulation output.

**Implement:**
- `IPic3DReader` class implementing `SimulationReader` protocol
- `read_timestep(path, step) → FieldDataset`
  - Map iPIC3D names to canonical: Bx→B1, rho→rho_c, Jxh→J1, etc.
  - Register Cartesian aliases (Bx→B1, etc.)
  - Construct GridInfo from iPIC3D settings
  - Construct Normalization from iPIC3D qom/B0/rhoINIT
  - Parse species from iPIC3D settings into SpeciesInfo list
- `available_timesteps(path) → list[int]`
- Auto-detect and load `simulation.toml` if present alongside data

**Test fixtures:** Small synthetic HDF5 files mimicking iPIC3D layout

**Tests:**
- Read synthetic iPIC3D file, field names are canonical
- Grid dimensions match expected values
- Species count and charge_to_mass values correct
- Normalization round-trips (normalize then to_si recovers original)
- `available_timesteps` returns sorted list
- Missing fields don't crash (reader skips gracefully)

---

## Step 13: `FieldDataset.compute()` and `.in_units()`

Convenience layer connecting FieldDataset to derived.py and units.py.

**Implement:**
- `FieldDataset.compute(name) → NDArray` — dispatches string to function:
  - `"|B|"` → `magnetic_field_magnitude(data["B1"], data["B2"], data["B3"])`
  - `"|E|"`, `"|J|"`, `"|V|"` — same pattern
  - `"beta"` → `plasma_beta(data["P"], ...)`
  - `"v_A"` → `alfven_speed(...)`
  - Unknown name → clear `KeyError` with suggestion
- `FieldDataset.in_si(field) → NDArray` — normalize → SI conversion
- `FieldDataset.in_units(field, unit_str) → NDArray` — "nT", "km/s", etc.

**Tests:**
- `data.compute("|B|")` matches direct `magnetic_field_magnitude()` call
- `data.in_si("B1")` matches `normalization.to_si_b_field(data["B1"])`
- `data.in_units("B1", "nT")` = `in_si("B1") * 1e9`
- `data.in_units("B1", "code")` returns raw array unchanged
- Unknown compute name raises helpful error
- Unknown unit string raises helpful error

---

## Step 14: `plotting/slices.py` — basic 2D visualization

Publication-quality 2D slice plots.

**Implement:**
- `plot_field_slice(data, field_name, plane, slice_index, units, ax, cmap, symmetric_cbar) → Axes`
  - Uses `PlaneSelection` internally if data is 3D
  - Axis labels include coordinate names and units
  - Colorbar with label
- `plot_comparison(data_a, data_b, field_name, labels, plane, units) → Figure`
  - Three-panel: A | B | difference
  - Shared colorbar range for A and B panels
- `plotting/styles.py` — publication rcParams defaults (font sizes, dpi)

**Tests:**
- `plot_field_slice` doesn't crash on synthetic 2D and 3D data
- Axes labels contain unit string when `units != "code"`
- `plot_comparison` returns Figure with 3 axes
- Passing custom `ax` works (plots on provided axes)
- Doctest with `plt.close("all")` cleanup

---

## ── Milestone: working daily-use tool ──

After Step 14: load iPIC3D data → compute derived quantities →
compare runs → select subregions → convert units → make paper figures.
Steps 15–17 add frame transforms, spherical selections, and documentation.

---

## Step 15: `coordinates/transforms.py` — frame transforms

Data-driven coordinate frame rotations and translations.

**Implement:**
- `ReferenceFrame` dataclass (name, origin, rotation)
- `FrameTransform` dataclass (from_frame, to_frame, apply, inverse)
- Load transforms from `[coordinates.transforms]` in simulation.toml
- Transform chaining: A→B + B→C = A→C (automatic)
- `FieldDataset.transform_to(frame_name) → FieldDataset`

**Tests:**
- Identity transform returns input unchanged
- Known 90° rotation matches hand calculation
- Inverse of transform recovers original (to machine precision)
- Chain A→B→C matches direct A→C
- Unknown frame raises clear error

---

## Step 16: `selections.py` — SphereSelection

Non-axis-aligned selection using xarray masking.

**Implement:**
- `SphereSelection` frozen dataclass (center, radius)
  - `apply(data) → FieldDataset` via xarray `.where()`
  - Points outside sphere are NaN, grid shape preserved

**Tests:**
- Points at center have values, points far away are NaN
- Boundary points at exactly radius are included
- `compute("|B|")` works on masked data (NaN propagation)
- Plotting masked data doesn't crash (NaN handled gracefully)

---

## Step 17: MkDocs documentation site

Auto-generated API reference and tutorial pages.

**Implement:**
- Full `mkdocs.yml` with Material theme, mkdocstrings, mathjax
- `docs/index.md` — project overview
- `docs/getting-started.md` — install, load data, first plot
- `docs/api/` — one page per module, auto-generated via mkdocstrings
- `docs/tutorials/workflow.md` — complete example: load → derive →
  select → compare → plot
- Verify LaTeX renders in all docstrings

**Tests:**
- `mkdocs build --strict` passes with no warnings
- `mkdocs serve` shows rendered LaTeX math
- Every public function appears in API reference
- Tutorial code blocks are copy-pasteable and work

---

## Summary

| Step | Module | What it delivers | Cumulative capability |
|------|--------|------------------|-----------------------|
| 1 | skeleton | CI pipeline | Project exists, lints clean |
| 2 | units | Normalization | Unit conversion PIC ↔ MHD ↔ SI |
| 3 | units | PhysicsConstants, SpeciesInfo | Species metadata for derived quantities |
| 4 | coordinates | CoordinateGeometry | Metric factors for all three geometries |
| 5 | readers | FieldDataset, GridInfo | Universal data container with xarray |
| 6 | readers | config.py | Load simulation.toml (SCHEMA.md spec) |
| 7 | derived | Field quantities | |B|, beta, v_A, Poynting flux, energies |
| 8 | derived | Characteristic scales | ω_pe, d_i, r_i, λ_D, v_th, c_s |
| 9 | diagnostics | Comparison tools | L2 error, div B, field energy |
| 10 | coordinates | Differential operators | Curl, div, grad (Cartesian) |
| 11 | selections | Plane, Box | Subregion extraction |
| 12 | readers | iPIC3D reader | Load real simulation data |
| 13 | fields | compute(), in_units() | String-based derived quantities + unit display |
| 14 | plotting | 2D slices, comparison | Publication figures |
| **Milestone** | | **Daily-use tool** | **Load → analyze → compare → plot** |
| 15 | coordinates | Frame transforms | GSM, GSE, arbitrary rotations |
| 16 | selections | Sphere | Non-axis-aligned masking |
| 17 | docs | MkDocs site | Browsable API reference + tutorials |
