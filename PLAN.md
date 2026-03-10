# Plasma Analysis Toolkit: Python Package Plan

## Vision

A lean, modern, well-documented Python package for reading, analyzing, and
plotting plasma simulation output. Model-agnostic, coordinate-aware,
unit-aware. Usable standalone, as a FastAPI backend, or as a reference
implementation for the Rust plasma code.

## Design Principles

1. **Three independent layers:** reading → analysis → visualization.
   Each usable without the others.
2. **Normalized internally, convert at boundaries.** Same as Rust plan.
3. **xarray as container, NumPy for computation.** `FieldDataset` wraps
   `xr.Dataset` for coordinate-aware slicing, lazy I/O, and metadata.
   All computation functions take and return raw NumPy arrays — xarray
   never enters the hot path. Zero-copy access via `.values`.
4. **Coordinate geometry is data, not code.** No hardcoded GSM/GSE/etc.
   Cartesian, spherical, cylindrical supported via metric tensors.
5. **Configuration-driven.** A `simulation.toml` file (conforming to a
   shared `SCHEMA.md` spec) describes the simulation's units,
   coordinates, and frame transforms. The analysis code reads it.
6. **Every public function:** type hints, NumPy-style docstring with LaTeX,
   runnable doctest example.

---

## Package Structure

```
plasma-analysis/
├── pyproject.toml                  # uv, ruff, pytest, mypy
├── mkdocs.yml                      # MkDocs Material + mkdocstrings config
├── README.md
├── docs/
│   ├── index.md                    # Landing page
│   ├── getting-started.md          # Installation, quick example
│   ├── tutorials/                  # Narrative docs (workflows, physics)
│   └── api/                        # Auto-generated API reference
│       ├── units.md
│       ├── derived.md
│       └── ...
├── tests/
│   ├── test_units.py
│   ├── test_geometry.py
│   ├── test_transforms.py
│   ├── test_operators.py
│   ├── test_derived.py
│   ├── test_diagnostics.py
│   ├── test_selections.py
│   └── test_readers/
│       ├── test_ipic3d.py
│       └── test_batsrus.py
│
└── src/plasma_analysis/
    ├── __init__.py
    ├── py.typed
    │
    │  # ── Layer 1: Reading ──
    ├── readers/
    │   ├── __init__.py
    │   ├── base.py                 # FieldDataset, GridInfo, SimulationReader protocol
    │   ├── config.py               # simulation.toml loader
    │   ├── ipic3d.py               # iPIC3D HDF5 reader
    │   ├── batsrus.py              # BATSRUS reader
    │   └── rust_plasma.py          # [future] Rust code HDF5 reader
    │
    │  # ── Layer 2: Core ──
    ├── units.py                    # Normalization (PIC, MHD, identity), scipy.constants
    ├── coordinates/
    │   ├── __init__.py
    │   ├── geometry.py             # CoordinateGeometry, metric tensors
    │   ├── operators.py            # Geometry-aware curl, div, grad (dispatch by metric)
    │   └── transforms.py           # ReferenceFrame, frame-to-frame transforms (data-driven)
    ├── fields.py                   # FieldDataset convenience methods
    ├── selections.py               # Region descriptions: apply to FieldDataset → smaller FieldDataset
    ├── derived.py                  # Derived quantities (pure functions on arrays)
    ├── diagnostics.py              # L2 error, conservation checks, spectra
    │
    │  # ── Layer 3: Visualization (optional) ──
    └── plotting/
        ├── __init__.py
        ├── slices.py               # 2D slice plots
        ├── profiles.py             # 1D line profiles
        └── styles.py               # Matplotlib defaults, colormaps
```

---

## Layer 1: Reading

### FieldDataset — the universal internal representation

Every reader produces a `FieldDataset`. The analysis layer only sees this
type. You can also construct one directly without any reader (e.g., from
server data, synthetic tests, or a new simulation code).

Internally, `FieldDataset` wraps an `xr.Dataset`. This gives you
coordinate-aware slicing, lazy I/O via dask, built-in interpolation,
and metadata propagation — without xarray leaking into the computation
layer. String-keyed access returns raw NumPy arrays (zero-copy via
`.values`), so all derived quantity functions stay pure NumPy.

```
FieldDataset
├── _ds: xr.Dataset             # internal xarray Dataset (coordinates, attrs)
├── normalization: Normalization
├── species: list[SpeciesInfo]
├── physics: dict[str, Any]     # model-specific params (gamma, resistivity, ...)
├── metadata: dict[str, Any]    # everything else (model name, version, ...)
│
├── __getitem__(key) → NDArray  # data["Bx"] returns NumPy via .values (zero-copy)
├── has_field(key) → bool
├── field_names() → list[str]
├── sel(**kwargs) → FieldDataset  # coordinate-aware selection (wraps xr.sel)
├── isel(**kwargs) → FieldDataset # index-based selection (wraps xr.isel)
├── xr → xr.Dataset              # direct access when needed (plotting, I/O)
```

**The boundary rule:** `data["Bx"]` returns NumPy. `data.xr["Bx"]`
returns an xarray DataArray (with coordinates, for plotting or I/O).
Computation functions only ever see the NumPy side.

### SpeciesInfo — typed species parameters

Species information is needed often enough (skin depths, gyrofrequencies,
Alfvén speed from PIC data, thermal velocities) that it deserves a typed
dataclass rather than living in a generic metadata dict.

```
SpeciesInfo (frozen dataclass)
├── name: str                   # "electrons", "ions", "alpha_particles", ...
├── charge: float               # in normalized units
├── mass: float                 # in normalized units
├── particles_per_cell: int     # for PIC (0 for MHD fluid species)
├── temperature: float          # initial temperature, normalized
├── drift_velocity: tuple[float, float, float]
│
├── property charge_to_mass → float
├── property mass_ratio(reference_mass) → float
```

Derived quantity functions accept `SpeciesInfo` when needed:

```python
# Analysis functions can use species parameters directly
def ion_skin_depth(density, species, constants): ...
def gyrofrequency(b_magnitude, species, constants): ...
def thermal_velocity(species): ...
def debye_length(density, species, constants): ...
```

### SimulationConfig — parsed from simulation.toml

The full parsed configuration, available from readers or constructed
manually. The `FieldDataset` carries the parts it needs (normalization,
species, physics); the full config is available for inspection.

```
SimulationConfig
├── model: ModelInfo              # name, type ("PIC"/"MHD"/"hybrid"), version
├── grid: GridInfo                # dimensions, spacing, geometry
├── normalization: Normalization  # derived from [units] section
├── species: list[SpeciesInfo]    # from [[species]] array
├── physics: dict[str, Any]       # from [physics.pic] or [physics.mhd]
└── coordinates: CoordinateConfig # geometry + frame transforms
```

### SimulationReader — protocol, not base class

Any object with `read_timestep(path, step) → FieldDataset` works.
Each reader is a small, self-contained module. Adding a new simulation
code means adding one .py file — no registration, no plugin system.

### simulation.toml — configuration that travels with data

Describes the simulation's model type, units, species, physics parameters,
coordinate geometry, and available frame transforms. Loaded by readers
if present, or constructed manually.

The structure of `simulation.toml` conforms to a shared **`SCHEMA.md`**
specification that is referenced by all three projects (Python analysis,
Rust plasma code, Three.js viewer). This ensures canonical field names,
normalization metadata keys, species parameter names, and coordinate
system descriptions are defined once and used consistently everywhere.

```toml
[model]
name = "iPIC3D"
type = "PIC"                          # "PIC" | "MHD" | "hybrid"
version = "3.0"

[grid]
nx = 256
ny = 128
nz = 128
dx = 0.5
dy = 0.5
dz = 0.5

[units]
system = "PIC"                        # "PIC" | "MHD" | "SI" | "CGS"
electron_density = 1.0e18            # m⁻³ (reference for PIC normalization)

# ── Physics parameters (model-specific) ──

[physics.pic]                         # present only for type = "PIC"
omega_pe_over_omega_ce = 3.0         # sets B field strength relative to density
speed_of_light = 1.0                 # normalized (always 1.0 in PIC units)

[physics.mhd]                        # present only for type = "MHD"
gamma = 1.6667                       # adiabatic index (5/3)
resistivity = 0.0                    # η (0.0 = ideal MHD)
hall_term = false                    # include Hall term?

# ── Species (any number, PIC and hybrid) ──

[[species]]
name = "electrons"
charge = -1.0
mass = 1.0                           # m_e = 1 in PIC normalization
particles_per_cell = 100
temperature = 0.01                   # normalized
drift_velocity = [0.0, 0.0, 0.0]

[[species]]
name = "ions"
charge = 1.0
mass = 25.0                          # reduced mass ratio for this run
particles_per_cell = 100
temperature = 0.05                   # Ti/Te = 5
drift_velocity = [0.1, 0.0, 0.0]    # drifting ions

[[species]]                          # optional: additional species
name = "alpha_particles"
charge = 2.0
mass = 100.0                         # 4 × mass_ratio (reduced)
particles_per_cell = 50
temperature = 0.1
drift_velocity = [0.0, 0.0, 0.0]

# ── Coordinates ──

[coordinates]
geometry = "cartesian"               # "cartesian" | "spherical" | "cylindrical"
frame = "simulation"                 # native frame name (arbitrary string)

[coordinates.transforms.GSM]
type = "affine"
origin = [0.0, 0.0, 0.0]
rotation = [[1,0,0],[0,1,0],[0,0,1]]
scale = 6.371e6

[coordinates.transforms.GSE]
from_frame = "GSM"
type = "rotation"
parameter = "dipole_tilt"
```

Key design choices:

`[[species]]` uses TOML's array-of-tables — any number of species,
naturally ordered. Electron-proton, electron-positron, multi-ion all
work. Mass ratio, temperature ratio, drift velocities are all explicit.

`[physics.pic]` and `[physics.mhd]` are mutually exclusive sections.
Only the one matching `[model] type` is present. The config itself
documents what kind of simulation produced the data.

MHD simulations may still have a `[[species]]` section (for multi-fluid
MHD or to document what physical species the fluid represents), but
`particles_per_cell` would be 0.

Field names in the output follow the canonical naming convention
(see Layer 2). Each reader maps the simulation code's native names
to canonical names during loading.

---

## Layer 2: Core

### Units (`units.py`)

Mirrors the Rust `Normalization` struct exactly. Uses `scipy.constants`
for physical constant values (plain floats, zero overhead). Conversion
methods work on both scalars and NumPy arrays.

```
Normalization
├── classmethod pic_standard(reference_density, reference_mass, reference_charge, c) → Normalization
├── classmethod pic_electron(n_e) → Normalization     # convenience: electron scales (ω_pe, d_e)
├── classmethod pic_ion(n_i, mass_i, charge_i) → Normalization  # convenience: ion scales (ω_pi, d_i)
├── classmethod mhd_standard(l_0, rho_0, b_0) → Normalization
├── classmethod identity() → Normalization             # data already in SI
├── normalize_*(value) → normalized                    # SI → code
├── to_si_*(value) → si                                # code → SI
```

`pic_standard` takes a generic reference species (density, mass, charge)
and derives all normalization values such that c = 1, m_ref = 1, q_ref = 1.
`pic_electron` and `pic_ion` are thin wrappers that fill in the standard
electron or proton mass/charge. This supports iPIC3D's ion-normalized
runs and exotic plasmas (electron-positron, heavy ions) without special cases.

No `astropy.units` — too slow (10–100x overhead per operation).
`scipy.constants` provides the same CODATA values as plain floats.

### Coordinates (`coordinates/`)

Two orthogonal concepts, handled separately:

**Geometry** (`geometry.py`): Cartesian, spherical, cylindrical. Determines
how differential operators work (metric factors). This is the analog of
the Rust plan's curvilinear coordinate support.

```
CoordinateGeometry
├── type: str                   # "cartesian" | "spherical" | "cylindrical"
├── axis_names: tuple[str,str,str]
├── axis_units: tuple[str,str,str]  # "length"/"angle"
├── metric_factors(grid) → (h1, h2, h3)  # scale factors at each point
```

For Cartesian: h₁ = h₂ = h₃ = 1 (trivial).
For cylindrical (r, φ, z): h₁ = 1, h₂ = r, h₃ = 1.
For spherical (r, θ, φ): h₁ = 1, h₂ = r, h₃ = r sin θ.

The metric factors are precomputed once from the grid and cached.
Differential operators use them automatically.

**Frame transforms** (`transforms.py`): Rotations and translations between
reference frames (GSM, GSE, HEE, RTN, or any user-defined name). These
are data-driven — defined in `simulation.toml`, not hardcoded.

```
ReferenceFrame
├── name: str                   # "simulation", "GSM", "GSE", etc. (arbitrary)
├── origin: tuple[float,float,float]
├── rotation: NDArray | None    # 3x3 rotation matrix

FrameTransform
├── from_frame: str
├── to_frame: str
├── apply(x, y, z) → (x', y', z')
├── inverse() → FrameTransform
```

Transforms can chain: simulation → GSM → GSE is handled automatically
if both transforms are defined in the config. Frame names are just
strings — no hardcoded knowledge of any specific frame.

**Geometry-aware operators** (`operators.py`): curl, div, grad that
dispatch based on the geometry type and use the metric factors.

```python
def curl(fx, fy, fz, grid):
    """Geometry-aware curl using metric factors from grid.geometry."""
    h1, h2, h3 = grid.geometry.metric_factors(grid)
    # General curvilinear curl formula using h1, h2, h3
    # Reduces to standard Cartesian when all h = 1
    ...
```

Same two-layer pattern as Rust: the per-element math receives
pre-computed derivatives (doesn't know about the grid), and the
collection wrapper computes derivatives using the metric.

**Implementation order:** Cartesian first (h = 1 everywhere, trivial).
Spherical and cylindrical: data structures and dispatch present from
the start, but operator implementations can raise `NotImplementedError`
until needed. Adding them later is filling in a function body, not
restructuring.

### Derived quantities (`derived.py`)

Pure functions on NumPy arrays. No FieldDataset dependency — can be
called standalone from a notebook, server, or test.

Functions don't know about coordinates. They receive arrays and grid
spacing. The caller (or FieldDataset convenience methods) handles
coordinate-aware derivatives via the operators module.

Key quantities: |B|, |E|, |J| (from curl B), plasma β, Alfvén speed,
gyrofrequency, skin depth, Poynting flux, field-line-aligned components,
pressure tensor invariants.

### Diagnostics (`diagnostics.py`)

Comparison and validation functions: L2 relative error, field energy,
div B (should be zero), conservation checks, power spectra. Same
functions used for PIC↔MHD comparison and for validating the Rust code
against Python.

### Selections (`selections.py`)

A selection is a **description of a region** — geometry, not data.
Applying a selection to a `FieldDataset` produces a new, smaller
`FieldDataset`. This keeps one data type everywhere: every analysis
function, diagnostic, and plot works on the result without modification.

```
Selection (applied to FieldDataset → smaller FieldDataset)
│
├── PlaneSelection        # axis-aligned 2D slice
│   normal: str           # "x", "y", "z"
│   index: int | None     # None = midplane
│
├── BoxSelection          # 3D subregion (index ranges)
│   x: tuple[int,int] | None
│   y: tuple[int,int] | None
│   z: tuple[int,int] | None
│
├── SphereSelection       # spherical subregion (masked)
│   center: tuple[float,float,float]
│   radius: float
│
├── ArbitrarySliceSelection   # oblique cutting plane (not axis-aligned)
│   point: tuple[float,float,float]     # point on the plane
│   normal: tuple[float,float,float]    # plane normal vector
│
└── FieldLineSelection    # 1D profile along a magnetic field line
    seed_point: tuple[float,float,float]
    direction: str        # "forward", "backward", "both"
    max_length: float
```

All selections are frozen dataclasses — immutable, hashable, serializable.
Can be stored in config files and replayed across datasets:

```toml
# analysis.toml — reusable selection definitions
[selections.magnetopause_cut]
type = "plane"
normal = "x"
index = 128

[selections.reconnection_region]
type = "box"
x = [100, 156]
y = [50, 78]
z = [50, 78]
```

**Key design properties:**

Selections are **reusable across datasets.** Define a cut once, apply it
to PIC data, MHD data, different timesteps, different runs. The result
is always a standard `FieldDataset`.

Selections are **composable.** Apply a box, then take a plane through
the result. Each step returns a `FieldDataset`, so they chain naturally.

Selections enable **efficient comparison.** Apply the same selection to
two runs, then use standard diagnostics on the smaller results:

```python
cut = PlaneSelection(normal="y", index=64)
error = l2_relative_error(cut.apply(pic_run)["Bx"], cut.apply(mhd_run)["Bx"])
```

**xarray integration:** Selections delegate to xarray operations internally:

| Selection | xarray operation | You still write |
|-----------|-----------------|-----------------|
| PlaneSelection | `.isel()` — one line | Nothing |
| BoxSelection | `.sel()` / `.isel()` — one line | Nothing |
| SphereSelection | `.where()` for masking | Distance computation |
| ArbitrarySliceSelection | `.interp()` for point sampling | Plane geometry, sample grid |
| FieldLineSelection | `.interp()` for field at arbitrary points | ODE integrator (RK4) |

**Performance:** `PlaneSelection` and `BoxSelection` are xarray views
(zero-copy). `SphereSelection` uses xarray `.where()` masking.
`ArbitrarySliceSelection` and `FieldLineSelection` use xarray `.interp()`
for trilinear interpolation at arbitrary points — backed by scipy's
optimized routines, so no need to write your own interpolation.

**Implementation order:** `PlaneSelection` and `BoxSelection` first
(trivial xarray slicing). `SphereSelection` next (masking).
`ArbitrarySliceSelection` and `FieldLineSelection` later — they
require interpolation and integration respectively, but the interface
is defined from the start.

---

## Layer 3: Visualization (optional)

Thin matplotlib wrappers for publication-quality 2D plots. Nothing else
depends on this layer. Will be mostly superseded by the Three.js viewer,
but kept for paper figures (journals need static images).

Key functions: `plot_field_slice` (2D colormapped field on any plane),
`plot_profile` (1D line cut), `plot_comparison` (side-by-side with
difference panel). All accept a `units` parameter for axis labels
and colorbar ("code", "SI", "nT", "km/s", etc.).

---

## Tooling

| Tool | Purpose |
|------|---------|
| **uv** | Package management, venv, lockfile |
| **ruff** | Linting + formatting (replaces flake8 + black + isort) |
| **pytest** | Testing, with `--doctest-modules` for docstring examples |
| **mypy (strict)** | Type checking |
| **xarray** | Container for FieldDataset: coordinate-aware slicing, lazy I/O, interpolation |
| **dask** (optional) | Lazy/chunked computation for large datasets, via xarray |
| **MkDocs Material + mkdocstrings** | Docs with auto-generated API reference from docstrings |
| **mathjax** (via pymdownx.arithmatex) | LaTeX math rendering in docs |
| **scipy.constants** | Physical constants as plain floats |

### Documentation setup

```yaml
# mkdocs.yml
site_name: Plasma Analysis Toolkit
theme:
  name: material
  features:
    - content.code.copy
    - navigation.sections
    - search.highlight

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
  - toc:
      permalink: true

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

### Documentation standard

Every public function gets a NumPy-style docstring with `r"""` raw string,
`$...$` inline and `$$...$$` display LaTeX math, Parameters / Returns
sections with types, and a runnable `Examples` section (doubles as doctest).

MkDocs Material + mkdocstrings parses the NumPy-style docstrings, and
mathjax (via the arithmatex extension) renders LaTeX in the browser.
Live reload via `mkdocs serve` — see rendered math instantly as you write.

Example docstring:

```python
def alfven_speed(b, rho):
    r"""Compute the Alfvén speed.

    $$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

    In normalized MHD units where $\mu_0 = 1$:

    $$v_A = \frac{B}{\sqrt{\rho}}$$

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho : NDArray
        Mass density in normalized units.

    Returns
    -------
    NDArray
        Alfvén speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> alfven_speed(np.array([1.0]), np.array([4.0]))
    array([0.5])
    """
    return b / np.sqrt(rho)
```

Note `$...$` syntax instead of Sphinx's `:math:` roles — simpler, more
universal, and consistent with Markdown everywhere else (README, CLAUDE.md,
Rust docs, plan documents).

**Why MkDocs Material over Sphinx + MyST?** Both support Markdown and
LaTeX math. MkDocs Material is simpler to configure, has live reload,
produces a modern site out of the box, and the mkdocstrings plugin
covers the autodoc features needed for a scientific library. Sphinx is
more powerful for deep cross-referencing and PDF output, but those
aren't needed here. The docstring format (NumPy-style) is the same
either way, so switching later is straightforward if ever needed.

---

## Implementation Order

| Step | What | Weeks |
|------|------|-------|
| 1 | Skeleton: pyproject.toml, ruff, pytest, mypy, CI. `units.py` with full tests. | 1–2 |
| 2 | `derived.py` + `diagnostics.py`: pure NumPy functions, full doctests. | 1–2 |
| 3 | `coordinates/geometry.py` + `operators.py`: data structures for all three geometries, Cartesian operators implemented, spherical/cylindrical stubs. | 1–2 |
| 4 | `readers/`: FieldDataset, GridInfo, simulation.toml loader. iPIC3D reader. BATSRUS reader. | 2 |
| 5 | `selections.py`: PlaneSelection, BoxSelection first. SphereSelection next. ArbitrarySlice and FieldLine stubs. | 1–2 |
| 6 | `coordinates/transforms.py`: data-driven frame transforms from config. | 1 |
| 7 | `plotting/`: 2D slices, profiles, comparison plots. Publication defaults. | 1 |
| 8 | MkDocs docs: `mkdocs.yml` config, API reference via mkdocstrings, tutorial pages. | ongoing |

**Total: ~8–12 weeks to a complete, tested, documented package.**

Steps 1–2 have zero dependencies on any simulation data and can be
developed and tested entirely with synthetic arrays. Step 3 establishes
the coordinate infrastructure. Steps 4–5 connect to real data and
selections. Steps 6–7 add transforms and plotting. The package is useful
after Step 4 — everything after that is convenience.

---

## What NOT to build

- **No astropy.units in computation path.** scipy.constants + Normalization.
- **No xarray in computation functions.** derived.py, diagnostics.py, and
  operators.py take and return NumPy arrays. xarray is the container layer
  only — accessed via `data["Bx"]` which returns `.values` (zero-copy).
- **No 3D matplotlib.** Three.js viewer replaces this.
- **No plugin/extension system.** New readers = new .py files.
- **No database or caching.** Read, compute, return. (Dask handles lazy
  I/O for large files via xarray integration.)
- **No hardcoded frame names** in function signatures or module names.
- **No server in the library.** FastAPI lives in a separate project that
  imports this package. The library has no web framework dependency.
- **No GUI.** Browser viewer is the interactive interface.
- **No Selection class that owns data.** Selections describe regions;
  applying them returns standard FieldDataset objects.

---

## Future: Data Formats

**Current:** HDF5 for simulation output (iPIC3D, BATSRUS, Rust code).
Readers parse HDF5 into `xr.Dataset` → `FieldDataset`.

**Planned:** Apache Arrow IPC for the viewer data pipeline. xarray has
built-in Arrow/Parquet conversion (`ds.to_dataframe().to_arrow()` and
via the `xarray-datatree` / `arro3` ecosystem). When the FastAPI server
sends data to the Three.js viewer, Arrow IPC provides structured data
with metadata (field names, shapes, coordinates, units) in a single
zero-copy-compatible response, replacing raw ArrayBuffers with custom
headers. Arrow is also readable in JavaScript (Arrow JS) and Rust
(arrow-rs), aligning all three projects on one interchange format.

This is a future enhancement — raw ArrayBuffers are fine for v1. The
xarray foundation makes the transition straightforward when needed.

---

## How This Feeds Into the Other Projects

**→ Three.js viewer rewrite:** A separate FastAPI server project imports
this library and serves data. Initially as binary ArrayBuffers, later
via Arrow IPC. The Python analysis layer handles coordinate transforms
and unit conversions server-side, so the viewer stays simple. Selections
defined in the viewer UI map directly to Selection objects in the server.
xarray's lazy I/O means the server only reads requested slices from disk.

**→ Rust plasma code:** When the Rust code writes HDF5 output (following
the shared `SCHEMA.md` spec), add a `rust_plasma.py` reader (one file).
The rest of the analysis pipeline works unchanged. The Normalization class,
derived quantities, and diagnostics serve as the **reference
implementation** — validate Rust output by comparing against Python
results on the same problem. Cross-project integration tests use both.

**→ Comparison workflows:** Load iPIC3D PIC output and Rust MHD output
into two `FieldDataset` objects. Apply the same Selection to both.
Call `diagnostics.l2_relative_error`. Plot with `plotting.plot_comparison`.
Same code, any pair of models, any subregion. xarray coordinates ensure
grids align correctly even when resolutions differ (via `.interp()`).

**→ SCHEMA.md:** The `simulation.toml` format, canonical field names,
normalization metadata keys, and HDF5 layout are defined in a shared
schema document referenced by all three projects. This is the contract
that keeps Python, Rust, and JavaScript in sync.
