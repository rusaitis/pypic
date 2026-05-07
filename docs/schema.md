# schema.md — Shared Data Contract

## Purpose

This document defines the shared data format and configuration structure
used across the plasma simulation platform:

- **Python analysis toolkit** — reads this format, computes derived quantities
- **Rust plasma code** — writes this format as simulation output
- **Three.js viewer** — consumes this format via the Starlette/FastAPI server

Any simulation code (iPIC3D, BATSRUS, the Rust code, or future codes) can
produce data conforming to this schema. Any consumer that reads this schema
can process the data without knowing which code produced it.

The schema defines **structure**, not exhaustive parameter lists. Model-specific
parameters live in open-ended sections that each code fills as needed.

---

## 1. Configuration File: `simulation.toml` (schema v1.0)

A TOML file that travels with the simulation output. Describes what
the data contains, how it's normalized, and what coordinate system it
uses. The canonical source of truth for the v1.0 schema is the Pydantic
validator in :mod:`pypic.schema` — `validate_simulation_toml()` returns
a strictly typed `SimulationSchema`, and `load_config()` builds the
internal `SimulationConfig` from that. An annotated reference template
lives at `pypic.simulation.toml` at the repo root.

### Required sections

A valid v1.0 document must declare:

```toml
schema_version = "1.0"   # top-level bare key — lets a streaming parser
                         # identify the schema without reading everything

[schema]                 # [schema].version must match schema_version
[model]                  # code identity (name, type)
[run]                    # THIS run's identity + provenance
[time]                   # dt, t_start, t_end, n_steps, scheme
[grid]                   # dimensions, spacing, lower/upper
[units]                  # normalization + reference values
[coordinates]            # geometry, frame
[[species]]              # ≥ 1 entry required (kinetic OR fluid species)
```

### Optional sections

Listed in the order §2 walks them.

```toml
[boundary_conditions]    # per-axis BC tags (lower/upper) + per-field overrides
[physics]                # model-agnostic flags + .{pic,mhd,hybrid} sub-tables
[[bodies]]               # registry of physical objects (planets, coils, ...)
[initial_conditions]     # flat table: setup type + type-specific keys
[[drivers]]              # ongoing external coupling (magnetograms, SW inflow, ...)
[restart]                # continuation pointer (incl. multi-file from_files)
[output.*]               # five sub-sections walked together in §2:
                         #   [output.checkpoints]   lossless full-state dumps
                         #   [output.fields]        field output cadence + quantities
                         #   [output.particles]     particle output cadence + selection
                         #   [output.probes]        probe time-series cadence
                         #   [output.diagnostics]   on-the-fly derived quantities
                         #   [[output.streams]]     multi-cadence / ROI output groups
[phase_space]            # >3D phase-space grid for gyrokinetic / full Vlasov
                         #   (continuum-Vlasov sparse-block storage lives in
                         #   the [phase_space.storage] sub-table)
[[collisions]]           # per-pair collision declarations (Smilei, EPOCH, ...)
[[probes]]               # fixed or trajectory samplers
```

### Extensions

Non-portable code-specific knobs live under an `x-<code>.*` namespace
(e.g. `[x-warpx]`, `[physics.pic.x-ipic3d]`). Validators accept any
`x-*` or `x_*` key without validation. Unknown keys under
`[physics.{pic,mhd,hybrid,vlasov}]` and their `.solver` sub-tables are
also accepted — the solver vocabulary is explicitly reserved for
evolution in v1.1+.

---

## 2. Section Specifications

### [schema]

Schema version tag and creation date.

```toml
[schema]
version = "1.0"                    # REQUIRED — must equal schema_version bare key
created = 2026-04-23               # optional TOML date literal
```

### [model]

Identity of the *code* that produced the data.

```toml
[model]
name = "string"                    # REQUIRED: "iPIC3D", "ARMS", "AIKEF", ...
type = "string"                    # REQUIRED: "PIC" | "MHD" | "hybrid" |
                                   #            "vlasov" | "gyrokinetic"
version = "string"                 # optional
description = "string"             # optional
url = "string"                     # optional — code homepage
doi = "string"                     # optional — code citation DOI
license = "string"                 # optional — SPDX identifier for the code
authors = [                        # optional — inline tables
    { name = "...", orcid = "...", affiliation = "...", role = "..." },
]
```

`[model]` is the code; `[run]` below is THIS run. Their metadata
(`license`, `doi`, `authors`) are independent.

### [run]

Identity + provenance of this specific run.

```toml
[run]
name = "string"                    # REQUIRED
description = "string"             # optional
authors = [ { name = "...", orcid = "...", ... } ]   # optional
date = 2026-04-23                  # optional — TOML native date
git_sha = "ea3fdfa"                # optional — commit of the input deck
host = "stampede3.tacc.utexas.edu" # optional
license = "CC-BY-4.0"              # optional — DATA license (SPDX id)
doi = "10.5281/zenodo.12345678"    # optional — DATA DOI
funding = ["NSF-AGS-2024001"]      # optional — grant IDs
embargo = 2027-01-01               # optional — public-release date
random_seed = 42                   # optional — RNG seed for reproducible setup

[run.ensemble]                     # optional — ensemble-member identity
member_id = 3                      # 1-indexed; must satisfy member_id <= total
total     = 16

[run.resources]                    # optional — all sub-keys optional
mpi_ranks        = 4096
mpi_topology     = [16, 16, 16]
nodes            = 64
wall_clock_hours = 24.0
cpu_core_hours   = 98304
node_hours       = 1536
peak_memory_gb   = 8192
cpu_type         = "Intel Xeon Max 9480"
gpu_hours        = 0
energy_kwh       = 2400
carbon_kg_co2eq  = 340

# Heterogeneous hardware — repeatable
[[run.resources.allocations]]
type     = "cpu"                   # "cpu" | "gpu" | "tpu" | "accelerator"
hardware = "AMD EPYC Milan 7763"
count    = 8192
hours    = 393216
```

### [time]

Temporal integration controls.

```toml
[time]
scheme  = "fixed"                  # "fixed" (default) | "adaptive" | "subcycled"
                                   #   | "rk2" | "rk3" | "rk4" | "vl2"
                                   #   | "ssprk2" | "ssprk3"
                                   #   | "imex-rk2" | "imex-rk3"
dt      = 0.05                     # timestep in code units; required for every
                                   #   scheme except "adaptive" (the runtime
                                   #   derives the first step from cfl/dt_min
                                   #   /dt_max and may ignore any value given)
t_start = 0.0                      # REQUIRED
t_end   = 100.0                    # REQUIRED
n_steps = 2000                     # REQUIRED
cfl     = 0.4                      # optional — for scheme = "adaptive"
dt_min  = 1.0e-5                   # optional — for scheme = "adaptive"
dt_max  = 0.01                     # optional — for scheme = "adaptive"
dt_field       = 0.004             # optional — for scheme = "subcycled"
field_substeps = 5                 # REQUIRED when scheme = "subcycled"
splitting      = "strang"          # optional — "strang" | "lie" | "godunov"
                                   #   for operator-split MHD/multi-physics
```

### [grid]

Computational grid in **code units**. Physical realization lives in
`[coordinates].physical_extent`.

```toml
[grid]
dimensions = [nx, ny, nz]          # REQUIRED — cells per axis
spacing    = [dx, dy, dz]          # REQUIRED — cell sizes
lower      = [x_min, y_min, z_min] # REQUIRED — lower corner
upper      = [x_max, y_max, z_max] # REQUIRED — upper corner (must exceed lower)
stagger    = "cell"                # optional: "cell" (default) | "node" | "staggered"
                                   #   "cell"       — fields at cell centers
                                   #   "node"       — fields at cell vertices (iPIC3D)
                                   #   "staggered"  — Yee mesh (B faces, E edges, ...)
                                   # Single-string summary; per-component truth lives
                                   # in [grid.stagger_fields] / [grid.stagger_position]
                                   # below. Informational only — readers destagger to
                                   # co-located grids on load.
ghost_cells   = [2, 2, 2]          # optional — halo width per axis (matches dimensions)
source        = "grids/mesh.h5"    # optional — external file for complex meshes
source_format = "hdf5"             # optional

# Dynamic adaptive refinement parameters (optional)
[grid.amr]
max_level            = 4
refinement_ratio     = 2
amr_kind             = "block"     # optional — "block" (default) | "patch" | "octree"
                                   #   discriminator for BoxLib/AMReX/Chombo (block),
                                   #   refinement-region patches, or octree codes
                                   #   (RAMSES, MPI-AMRVAC) where block_size doesn't apply
level_subcycling     = false       # optional — different AMR levels advance at
                                   #   different effective dt (Athena++, AMReX)
block_size           = [8, 8, 8]
refinement_criteria  = ["current_density", "gradient_b"]
refinement_threshold = 0.1

# Static nested refinement boxes (optional, repeatable)
[[grid.refinement]]
level = 2
box   = [[-60.0, -60.0, -60.0], [60.0, 60.0, 60.0]]

# Non-uniform per-axis cell widths (optional, sparse).
# Only stretched axes appear; uniform axes inherit `[grid].spacing`.
# Length must match `[grid].dimensions[i]` and the widths must sum to
# `upper[i] - lower[i]` within `sum_rtol`.
[grid.stretched.axis_widths]
0 = [0.5, 0.6, 0.8, 1.0, 1.3]      # x stretched
2 = [0.1, 0.1, 0.2, 0.4, 0.8]      # z log-radial (ARMS, PLUTO style)

# Per-field-group stagger locations (optional). Captures the Yee-mesh
# truth that the single-string `stagger` cannot — B sits on faces,
# E sits on edges, etc. Free-form keys (canonical field-group names);
# values are "cell" | "node" | "face" | "edge".
[grid.stagger_fields]
B = "face"
E = "edge"
J = "edge"

# Per-component openPMD ED-PIC offsets (optional). Each value is a
# vector in [0.0, 1.0) giving the position of that field component
# inside the local cell along each axis. The most precise stagger
# representation; round-trips through StaggerInfo.position.
[grid.stagger_position]
B1 = [0.5, 0.0, 0.0]               # B_x on the x-face
B2 = [0.0, 0.5, 0.0]               # B_y on the y-face
B3 = [0.0, 0.0, 0.5]               # B_z on the z-face
E1 = [0.0, 0.5, 0.5]               # E_x on the x-edge
```

### [boundary_conditions]

Per-axis BC tags. Array length must match `grid.dimensions` length.

```toml
[boundary_conditions]
lower = ["periodic", "periodic", "open"]
upper = ["periodic", "periodic", "open"]

# Per-face driver foreign keys (optional, sparse).
# Keys are axis indices ("0"/"1"/"2"); values must match a declared
# [[drivers]].name entry. Faces without a driver simply don't appear.
# Unresolved names raise a validation error.
drivers_lower = { 2 = "solar_wind_inflow" }

# Per-field BC overrides (optional). Apply only to "E", "B", or
# "particles"; each entry is a full BoundaryConditions block matching
# the same axis count as the default.
[boundary_conditions.field_overrides.E]
lower = ["periodic", "periodic", "pml"]
upper = ["periodic", "periodic", "pml"]

[boundary_conditions.field_overrides.particles]
lower = ["periodic", "periodic", "absorbing"]
upper = ["periodic", "periodic", "absorbing"]
```

The tag vocabulary is free-form string — it's consumed by the reader,
not constrained by the schema (different codes use different vocabularies:
`periodic` / `reflecting` / `conducting` / `open` / `driven` / `inflow` /
`outflow` / `pole` / `inner` / `outer` / `pml` / `absorbing` /
`thermal-bath`).

### [units]

Defines the normalization, allowing conversion between code units and SI.
Two approaches: specify the normalization system and let the consumer
derive reference values, or specify reference values directly.

**Approach A: named normalization (preferred for PIC/MHD)**

```toml
[units]
system = "PIC"                     # "PIC" | "MHD" | "SI" | "custom"

# PIC normalization: derive all reference values from a reference species.
reference_species = "electrons"    # optional: "electrons" (default) | "ions" | species name
reference_density = 1.0e18         # m⁻³ (number density of reference species)
reference_mass = 9.109e-31         # kg (optional, default: electron mass)
reference_charge = 1.602e-19       # C (optional, default: elementary charge)
speed_of_light = 2.998e8           # m/s (optional, default scipy.constants.c)
scaling_factor = 10.0              # optional: informational shrink factor (no effect on computation)
scaling_description = "c/v_A reduced by 10x; mass ratio mi/me = 256 (real: 1836)"
```

For **ion-normalized PIC** (e.g. iPIC3D large-scale runs), set
`reference_species = "ions"` and use proton mass/density values.

```toml
[units]
system = "MHD"
# MHD normalization: derive all reference values from these
reference_length = 6.371e6         # meters (e.g., Earth radius)
reference_density = 1.67e-17       # kg/m³ (e.g., solar wind)
reference_b_field = 5.0e-9         # Tesla (e.g., 5 nT)
```

**Approach B: explicit reference values (for custom normalizations)**

```toml
[units]
system = "custom"

[units.reference]                  # all in SI
length = 5.31e-3                   # meters — REQUIRED for custom
time = 1.77e-11                    # optional
velocity = 2.998e8                 # optional
b_field = 1.07e-3                  # optional
e_field = 3.21e5                   # optional
density = 1.0e18                   # optional
mass = 9.109e-31                   # optional
charge = 1.602e-19                 # optional
```

If `system` is "SI", all data is already in SI and no conversion is needed
(all reference values = 1.0).

### [coordinates]

Describes the coordinate geometry and reference frame.

```toml
[coordinates]
geometry = "string"                # REQUIRED: "cartesian" | "spherical" | "cylindrical"
                                   #            | "thetaMode"
frame = "string"                   # REQUIRED: native frame name (arbitrary, e.g., "simulation")
axis_labels = ["x", "y", "z"]     # optional: override default axis names. Always
                                   #   length 3 — labels the embedding 3D coordinate
                                   #   space, even when grid.dimensions is 1D or 2D.
                                   #   Surviving labels are sliced from this 3-tuple
                                   #   to match the active grid axes.
physical_extent = [46.0, 32.0, 13.0]  # optional: domain size in target-frame physical units;
                                   #   length must match grid.dimensions (1–3, per active axis)
physical_extent_unit = "R_E"       # optional: unit for physical_extent (default "m"). Valid:
                                   #   "m", "km" (SI lengths)
                                   #   "R_E" (Earth), "R_M" (Mercury), "R_J" (Jupiter)
                                   #   "R_S" (solar radius, heliophysics convention)
                                   #   "R_sun" (unambiguous synonym for R_S)
                                   #   "AU" (astronomical unit)
                                   #   "d_i" (ion skin depth, normalization-resolved at apply time)

# Required when geometry = "thetaMode" (FBPIC azimuthal-mode RZ
# decomposition over an (r, z) grid). Forbidden otherwise.
[coordinates.modes]
n_modes      = 3                   # number of azimuthal modes stored on disk
mode_indices = [0, 1, 2]           # optional explicit mode numbers; defaults to
                                   # [0, 1, ..., n_modes-1] when omitted
```

When ``physical_extent`` is provided, the ``scale`` field of any
transform with the default ``scale=1.0`` is auto-computed as
``physical_extent / grid_extent`` (after accounting for rotation).
Can also be passed at runtime via
``open_simulation(path, physical_extent=..., physical_extent_unit=...)``.

**Scale vs shrink factor.** ``scale`` is a coordinate conversion
factor (target units per code unit). ``shrink_factor`` is a physics
diagnostic — the ratio of effective scale to the scale implied by
the normalization. ``1.0`` means the grid faithfully represents
physical distances; ``> 1`` means the domain has been compressed
relative to kinetic scales (common in reduced-mass-ratio PIC and
MHD-coupled boundaries). Stored in
``metadata["scaling"]["shrink_factor"]``. Intensive per-point
quantities (fields, densities, β, Mach numbers) are unaffected;
extensive integrals scale as shrink² or shrink³, and transit times
shorten by the shrink factor.

**Frame transforms** (optional): define how to convert from the native
frame to other reference frames. Frame names are arbitrary strings —
no hardcoded knowledge of any specific frame.

```toml
[coordinates.transforms.TARGET_FRAME]
origin = [0.0, 0.0, 0.0]          # translation (code units or physical)
rotation = [[...], [...], [...]]   # 3x3 rotation matrix (optional, must be signed permutation)
scale = 1.0                        # length scale factor (optional, auto-computed from physical_extent)
from_frame = "string"              # for chaining: transform from this frame instead of native
```

Transforms can chain: if transform A goes from "simulation" to "GSM" and
transform B goes from "GSM" to "GSE", requesting "GSE" from "simulation"
data chains both automatically.

**Time-dependent transforms.** Frames like GSE↔GSM depend on the
dipole tilt angle, which varies per timestep. A `parameter` field
names a time-varying quantity (e.g., `"dipole_tilt"`) looked up per
step to compute the rotation matrix. SPICE kernels provide an
alternative source for epoch-dependent rotations.

```toml
[coordinates.transforms.GSM]
parameter = "dipole_tilt"          # rotation recomputed per step from
                                   # the named time-varying scalar
from_frame = "GSE"                 # chains: simulation → GSE → GSM
```

### [[species]]

Describes particle species (PIC and hybrid models) or fluid species
(multi-fluid MHD). Repeatable — any number of species.

```toml
[[species]]
name = "string"                    # REQUIRED
charge = 0.0                       # code units — REQUIRED together with mass ...
mass = 0.0                         # code units — ... OR ...
charge_to_mass = 0.0               # ... REQUIRED alone (XOR with charge+mass)
particles_per_cell = [5, 5, 1]     # optional — scalar OR [nx, ny, nz]; 0 or absent = fluid
temperature = 0.0                  # optional — scalar isotropic temperature (code units)
thermal_velocity = [0.0, 0.0, 0.0] # optional — per-component thermal velocity
drift_velocity = [0.0, 0.0, 0.0]   # optional — bulk drift (code units)
density = 1.0                      # optional — number density (code units)
closure = "adiabatic"              # optional (fluid species) — "isothermal" | "adiabatic"
                                   #                           | "polytropic" | "braginskii"
                                   #                           | "cgl" | "10moment" | "14moment"
gamma_eos = 1.6666667              # optional (fluid species) — per-species adiabatic index
gamma_eos_par = 3.0                # optional (10-moment hybrids: Gkeyll, Hakim) — parallel
gamma_eos_perp = 2.0               # optional (10-moment hybrids) — perpendicular
                                   # par/perp must be present together; cannot mix with
                                   # the scalar `gamma_eos`.
inertia = 0.0                      # optional (hybrid fluid electrons) — me/mi; 0 = massless
shape = "cic"                      # optional (PIC) — "ngp" | "cic" | "tsc" | "pqs"
                                   #   (NGP=order 0, CIC=1, TSC=2, PQS=3; ED-PIC vocabulary)
tracer = false                     # optional (PIC) — tagged subset for trajectory tracking;
                                   # readers propagate the hint to particle analysis
```

The validator enforces `(charge + mass)` XOR `charge_to_mass`: provide
exactly one form. Species are indexed in declaration order — `_s0`
binds to the first entry, `_s1` to the second, etc. Reordering is a
breaking change to downstream field-name references.

Relationship: $v_{th} = \sqrt{T/m}$ (see thermal speed convention in
[Conventions](conventions.md)).

### [physics]

Model-agnostic flags at the top level; model-specific knobs under
sub-tables matching `[model].type`. v1.0 enumerates only
`relativistic` at the top level; unknown top-level keys are accepted
to leave room for v1.1+ portable additions (anticipated:
`collisional`, `radiative`, `[physics.vlasov]`). Code-specific knobs
go under `[physics.<model>.x-<code>.*]`.

```toml
[physics]
relativistic = false               # semantics identical across PIC/MHD/hybrid

[physics.pic]                      # physics-model knobs for PIC codes
omega_p_over_omega_c = 20.0        # reference-species plasma/cyclotron ratio

[physics.pic.solver]               # numerical-method knobs for PIC codes
scheme            = "semi-implicit"   # "explicit" | "semi-implicit" | "implicit"
implicitness      = 0.5               # θ-scheme weight (0.5 = Crank-Nicolson)
pusher            = "boris"           # "boris" | "vay" | "higuera-cary"
                                      #   | "llrk4" | "free-streaming"
field_solver      = "implicit-moment" # "fdtd-yee" | "pseudo-spectral"
                                      #   | "psatd" | "spectral-azimuthal"
                                      #   | "lehe" | "ck" | "ckc" | "pstd" | "gpstd"
                                      #   | "implicit-moment" | "implicit-gmres"
preconditioner    = "block-jacobi"    # "none" | "jacobi" | "block-jacobi" |
                                      # "ilu" | "amg" | "additive-schwarz"
current_smoothing = 0                 # optional — binomial filter passes per step
charge_smoothing  = 0                 # optional — binomial filter passes per step
charge_correction = "none"            # optional — "marder" | "langdon" | "boris" |
                                      #   "hyperbolic" | "spectral" | "none"
current_deposition = "esirkepov"      # optional — "esirkepov" | "zigzag" |
                                      #   "villabune" | "direct-boris" |
                                      #   "direct-morse-nielson" | "none"

[physics.mhd]
gamma       = 1.6667
resistivity = 0.0
hall_term   = false

[physics.mhd.solver]
scheme              = "fct"        # "fct" | "godunov" | "muscl-hancock" | "ppm" | "weno"
reconstruction      = "linear"     # "linear" | "plm" | "ppm" | "weno5" | "mp5"
limiter             = "zalesak"    # "zalesak" | "minmod" | "mc" | "van-leer" | "superbee"
divergence_cleaning = "ct"         # "ct" | "powell" | "dedner-glm" | "projection" | "none"
preconditioner      = "ilu"        # optional — implicit MHD: "none" | "jacobi" |
                                   #   "block-jacobi" | "ilu" | "amg" | "additive-schwarz"
# riemann = "hlld"                 # Godunov family: "roe" | "hll" | "hlle" | "hlld" | "lax-friedrichs"

[physics.hybrid]
# (physics-model knobs — currently minimal; hybrid fluid-species properties
# live on the corresponding [[species]] entry, not here.)

[physics.hybrid.solver]
scheme            = "predictor-corrector"  # "predictor-corrector" | "current-advance-method"
field_pusher      = "cyclic-leapfrog"      # "cyclic-leapfrog" | "implicit"
resistivity       = 5.0e-4
hyper_resistivity = 1.0e-6
current_smoothing = 2
preconditioner    = "block-jacobi"         # optional — for "implicit" field_pusher only:
                                           #   "none" | "jacobi" | "block-jacobi" |
                                           #   "ilu" | "amg" | "additive-schwarz"
```

Key rename: `omega_pe_over_omega_ce` (v0) → `omega_p_over_omega_c`
(v1.0) — species-agnostic, applies to whichever species is declared
in `[units].reference_species`. Unknown keys under
`[physics.{pic,mhd,hybrid}]` and their `.solver` sub-tables are
accepted without validation (vocabulary evolves in v1.1+).

### [[bodies]]

Registry of physical objects in the domain (planets, stars, coils,
exoplanet moons). Drivers and initial conditions reference bodies by
name — no coordinate duplication.

```toml
[[bodies]]
name                 = "mercury"
center               = [0.0, 0.0, 0.0]    # code units
radius               = 60.0
shape                = "sphere"           # "sphere" | "torus" | "cuboid" | "mesh"
intrinsic_dipole     = [0.0, 0.0, -190.0] # split-B analytic background
dipole_center_offset = [0.0, 0.0, 11.8]
rotation_axis        = [0.0, 0.0, 1.0]
rotation_period      = 5067360.0          # seconds
mass                 = 3.302e23           # kg
surface_absorbs_ions = true
has_atmosphere       = false
has_intrinsic_field  = true
```

### [initial_conditions]

Flat table: setup-specific keys live alongside `type`, no `.parameters`
sub-table.

```toml
[initial_conditions]
type                    = "double_harris"
B0                      = [0.05, 0.0, 0.0]
perturbation_amplitude  = 0.1
current_sheet_thickness = 0.5
```

### [[drivers]]

Ongoing external coupling (magnetograms, solar-wind inflows, pickup-ion
sources, surface absorption, MHD→PIC volume coupling). Repeatable.

```toml
[[drivers]]
name          = "photospheric_magnetogram"
type          = "magnetogram_timeseries"
coupling      = "boundary"         # "boundary" | "volume" | "source" | "sink"
direction     = "one_way"          # "one_way" (default) | "two_way"
description   = "..."
source        = "drivers/..."
columns       = ["t", "B_r", "B_theta", "B_phi"]
cadence       = 720.0              # seconds between samples
interpolation = "linear"
target_lower  = [1.0, 0.87, -0.87] # code units (optional)
target_upper  = [1.0, 2.60, 0.87]
body          = "mercury"          # optional — inherits body bounding box.
                                   #   Must match a declared [[bodies]].name;
                                   #   unresolved names raise a validation error.
```

`coupling` and `direction` are orthogonal. Target precedence:
1. `body` — defaults to that body's bounding box.
2. `target_lower` / `target_upper` — explicit box; narrows `body` when both present.
3. Neither — whole domain.

Driver-type-specific keys (`production_rate`, `fields`, etc.) are
accepted beyond the core vocabulary above.

### [restart]

Continuation pointer from a prior run. ``from`` is the path to the
restart artifact: a single file, a directory of per-rank checkpoints,
or a glob pattern. Whether it resolves to one file or many is
determined at read time by the filesystem and the code, not by the
schema. ``from_files`` is a rarely-needed escape hatch for runs whose
filenames don't follow the source code's convention.

```toml
[restart]
from = "./checkpoints/chk_000030.h5"   # REQUIRED — path to the restart artifact.
                                        #   Single file, directory of per-rank
                                        #   files, or glob — code-determined.
                                        #   (Aliased — `from` is a Python keyword.)
step = 30000                            # optional
time = 1500.0                           # optional
restore = ["fields", "particles"]       # optional — partial restart selector
                                        #   ("fields" | "particles" | "auxiliary");
                                        #   entries must be distinct. Omitting
                                        #   restores everything.
mode = "hot"                            # optional — "hot" (full state reload) |
                                        #   "cold" (re-apply IC on saved geometry)
from_files = [                          # optional escape hatch — explicit list
    "chk_000030_rank_00000.h5",         #   for runs whose per-rank filenames
    "chk_000030_rank_00001.h5",         #   don't follow the source code's
    "chk_000030_rank_00002.h5",         #   convention (e.g. relocated reruns).
]                                       #   Distinct & non-empty. When present,
                                        #   `from` should point at a directory or
                                        #   glob — not a single .h5/.bp/.zarr/.nc
                                        #   file.
```

### [output.*]

Output cadences, quantities, directories, and precisions.
Five independent sub-sections; each is optional.

```toml
[output.checkpoints]                   # lossless full-state dumps
step_interval = 5000                   # REQUIRED
dir           = "./checkpoints"        # REQUIRED
format        = "hdf5"                 # "hdf5" | "zarr" | "adios2" | "netcdf"
precision     = "f64"                  # "f32" | "f64" — checkpoints default f64
keep_last     = 3                      # optional — rolling retention
file_pattern  = "step_{step:06d}/rank_{rank:05d}.h5"  # optional — per-rank layout
files_per_step = 64                    # optional — fan-out at each output step
partition     = "by_rank"              # optional — "by_rank" | "by_field" | "monolithic"

[output.fields]                        # field output
step_interval = 50
quantities    = ["B", "E", "J", "rho_c", "V_s0"]  # vector/tensor groups auto-expand
dir           = "./fields"
format        = "hdf5"
precision     = "f32"

[output.fields.precision_overrides]    # optional — per-quantity exceptions
rho_c = "f64"                          # names must appear in `quantities` above

[output.particles]
step_interval  = 200
species        = ["electrons", "ions"]
dir            = "./particles"
format         = "hdf5"
precision      = "f32"
include_ids    = true
include_energy = true
sample         = 0.01                  # 0-1 fraction OR integer count

[output.probes]
step_interval = 1
dir           = "./probes"
format        = "hdf5"
precision     = "f64"

[output.diagnostics]                   # on-the-fly derived quantities
step_interval = 100
quantities    = ["div_B", "div_E", "|J|", "e_B", "e_E"]
dir           = "./diagnostics"
format        = "hdf5"
precision     = "f32"

# Multi-cadence / region-of-interest output (optional, repeatable).
# Each stream has a unique name. `region` optionally restricts the
# write to a sub-volume (`kind = "box"`) or plane slice
# (`kind = "plane"`).
[[output.streams]]
name          = "moments"              # unique across streams
step_interval = 10                     # 10x cadence relative to [output.fields]
quantities    = ["B", "rho_c", "V_s0"]
dir           = "./moments"
format        = "hdf5"
precision     = "f32"

[[output.streams]]
name          = "current_sheet_roi"
step_interval = 5
quantities    = ["B", "E", "J"]
dir           = "./roi"
[output.streams.region]
kind  = "box"
lower = [-1.0, -1.0, -0.5]            # code units, must align with grid axes
upper = [ 1.0,  1.0,  0.5]

[[output.streams]]
name          = "midplane"
step_interval = 1
quantities    = ["B", "E"]
dir           = "./midplane"
[output.streams.region]
kind  = "plane"
axis  = 2                              # 0/1/2 — must be < grid.dimensions length
value = 0.0
```

See `examples/ipic3d-double-harris.toml` for a complete mapping from
iPIC3D input to this schema, and `pypic.simulation.toml` at the repo
root for the annotated reference template (three scenarios: PIC, MHD,
hybrid).

### [phase_space]

Augmented phase-space dimensionality for >3D kinetic codes — gyrokinetic
(GENE, GS2, GX, Gkeyll-GK: 5D = 3 spatial + 2 velocity), full continuum
Vlasov (6D phase space). The first ``n`` dimensions must match
``[grid].dimensions`` exactly: ``[grid]`` owns the spatial sub-grid;
``[phase_space]`` extends it with velocity / extra-D axes.

```toml
[phase_space]
dimensions  = [4, 4, 4, 16, 8]         # 5D: 3 spatial + 2 velocity
                                       # First 3 must match [grid].dimensions.
axis_labels = ["x", "y", "z", "vpar", "mu"]
extents     = [                         # optional — per-axis [lower, upper]
    [0.0, 4.0], [0.0, 4.0], [0.0, 4.0],
    [-3.0, 3.0], [0.0, 5.0],
]
coordinate_system = "guiding-center"   # optional — "cartesian" (default) |
                                       #   "guiding-center" | "field-aligned" |
                                       #   "spherical-velocity"
```

**Sparse-block storage** (continuum-Vlasov codes only). Vlasiator and
Gkeyll-Vlasov-Maxwell subdivide the velocity sub-grid into blocks for
adaptive memory use, dropping blocks whose density falls below a
threshold. Optional ``[phase_space.storage]`` sub-table:

```toml
[phase_space.storage]
block_size         = [10, 10, 10]      # one entry per velocity axis;
                                       #   must evenly divide
                                       #   phase_space.dimensions[n_spatial:]
sparsity_threshold = 1.0e-15           # density floor below which blocks
                                       #   are dropped from disk
```

For 6D Vlasiator runs: the spatial part of ``dimensions`` matches
``[grid].dimensions``, the trailing 3 entries are velocity axes, and
``[phase_space.storage]`` carries the sparse-block knobs. Gyrokinetic
codes omit ``[phase_space.storage]`` because their 5-D grid is dense.

**Migration note.** v1.0 had a separate top-level ``[velocity_mesh]``
section that overlapped with ``[phase_space]`` on axis identity. v1.0.x
removes ``[velocity_mesh]`` and folds its ``block_size`` /
``sparsity_threshold`` into ``[phase_space.storage]``; the redundant
``dimensions``, ``extent``, and ``coordinate_system`` fields are
retired in favor of the ``[phase_space]`` versions. If a config still
contains ``[velocity_mesh]``, move ``block_size`` and ``sparsity_threshold``
to ``[phase_space.storage]`` and drop the rest — ``[phase_space]`` already
carries equivalent ``dimensions``, ``extents``, and ``coordinate_system``
keys.

### [[collisions]]

Per-pair collision declaration for collisional PIC codes (Smilei,
EPOCH, OSIRIS-collisional, PIConGPU). Repeatable. Both species in
``species_pair`` must match a declared ``[[species]].name`` —
unresolved names raise a validation error. Self-collisions (same
species twice) are permitted.

```toml
[[collisions]]
species_pair    = ["electrons", "ions"] # REQUIRED — both names must match [[species]]
model           = "coulomb"            # REQUIRED — "coulomb" | "bgk" | "monte-carlo"
coulomb_log     = 10.0                 # optional — Λ for Coulomb collisions
temperature_ref = 1.0                  # optional — reference T for the rate scale
description     = "..."                # optional
```

### [[probes]]

Virtual probes (detectors, spacecraft) that sample fields at specific
locations. Each probe is either **fixed** (constant `position`) or a
**trajectory** (CSV file path); provide exactly one of `position` or
`trajectory` — both or neither raises a validation error. Repeatable.

```toml
[[probes]]
name = "magnetopause_monitor"      # REQUIRED — human-readable label
position = [10.0, 0.0, 0.0]        # XOR with `trajectory` — fixed: [x, y, z] in code units
fields = ["B1", "B2", "B3", "beta"]  # optional: fields to sample.
                                     # Default (omitted): every
                                     # storage-primitive field present
                                     # in the dataset at probe time —
                                     # what `reader.available_fields()`
                                     # returns. Derived quantities like
                                     # `|B|`, `beta`, `v_A` must be
                                     # listed explicitly to opt in.

[[probes]]
name = "MMS1"                      # REQUIRED — trajectory (virtual spacecraft)
trajectory = "mms1_orbit.csv"      # XOR with `position` — CSV columns: t, x, y, z (code units)
frame = "GSM"                      # optional: transform to simulation frame
```

Output: `TabularData` time-series (one row per timestep, columns per
field). Multiple entries form probe arrays / constellations.

---

## 3. Canonical Field Names

All projects use the same field names. Readers are responsible for
mapping simulation-code-specific names to these canonical names.

For physics equations and SI conversions, see
[Equations](equations.md). For convention details, see
[Conventions](conventions.md).

### Dual naming convention

Vector field components use **numbered indices** (`B1`, `B2`, `B3`) as
the general canonical form. The coordinate geometry determines what
each index means:

| Index | Cartesian | Spherical | Cylindrical |
|-------|-----------|-----------|-------------|
| 1 | x | r | r |
| 2 | y | θ | φ |
| 3 | z | φ | z |

For Cartesian data, letter aliases (`Bx`, `By`, `Bz`) are preferred for
readability and access the same data as `B1`, `B2`, `B3`. For non-Cartesian
data, only the numbered form is canonical; the reader registers
geometry-appropriate aliases (e.g., `Br` → `B1` for spherical). The
`[coordinates] geometry` field determines which aliases are active.

### Electromagnetic fields

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `B1`, `B2`, `B3` | `Bx`, `By`, `Bz` | Magnetic field components | PIC, MHD |
| `B0_1`, `B0_2`, `B0_3` | `B0x`, `B0y`, `B0z` | Background magnetic field (split-B) | MHD (optional) |
| `E1`, `E2`, `E3` | `Ex`, `Ey`, `Ez` | Electric field components | PIC (MHD: derived) |

### Fluid / moment quantities — densities

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `n_s0`, `n_s1`, ... | Number density per species | PIC, multi-fluid MHD |
| `rho_c` | Total charge density | PIC |
| `rho_m` | Total mass density | MHD, PIC (derived) |

`rho_c` and `rho_m` are unambiguous — no overloaded `rho`. `n_e` and
`n_i` are accepted as aliases for `n_s0` and `n_s1` (see convenience
aliases table below).

**Per-species naming:** Append `_s` plus the 0-based species index
(`rho_c_s0`, `J1_s1`, `n_s3`); the component index comes before the
species suffix (`J1_s0`, `EF2_s1`). The species *name* lives in
`[[species]]`, not in the field name.

**Species-name aliases:** For any species, the canonical `<prefix>_s<index>`
form has an automatically-generated `<prefix>_<species_name>` alias when
the species name is declared in `[[species]]`. `n_s0` becomes `n_electrons`
when `species[0].name == "electrons"`; `EF1_s1` becomes `EF1_protons` when
`species[1].name == "protons"`. The alias is added at `FieldDataset`
construction time and only registered when the underlying canonical is
actually present in the dataset, so missing data produces a clean
`KeyError` rather than misdirection.

For multi-species runs (H⁺ + He²⁺ + O⁺), the species-name form is the
unambiguous way to reference per-species quantities — the integer index
depends on declaration order.

**Electron/ion convenience aliases:** For the common two-species case
(species 0 = electrons, 1 = ions), `e`/`i` suffixed names alias the
canonical `_s0`/`_s1` forms:

| Alias | Canonical | Meaning |
|-------|-----------|---------|
| `n_e`, `n_i` | `n_s0`, `n_s1` | Number density |
| `Pe`, `Pi` | `P_s0`, `P_s1` | Scalar pressure (or Tr(tensor)/3) |
| `Te`, `Ti` | `T_s0`, `T_s1` | Temperature |
| `Ve1`..`Ve3` | `V1_s0`..`V3_s0` | Electron bulk velocity |
| `Vi1`..`Vi3` | `V1_s1`..`V3_s1` | Ion bulk velocity |
| `EFe`, `EFi` | `EF_s0`, `EF_s1` | Energy flux (vector group) |
| `s_e`, `s_i` | `s_s0`, `s_s1` | Per-species entropy |
| `beta_e`, `beta_i` | `beta_s0`, `beta_s1` | Per-species plasma beta |
| `P_par_e`, `P_par_i` | `P_par_s0`, `P_par_s1` | Per-species parallel pressure |
| `P_perp_e`, `P_perp_i` | `P_perp_s0`, `P_perp_s1` | Per-species perpendicular pressure |
| `agyrotropy_e`, `agyrotropy_i` | `agyrotropy_s0`, `agyrotropy_s1` | Per-species agyrotropy |
| `s_gyro_e`, `s_gyro_i` | `s_gyro_s0`, `s_gyro_s1` | Per-species gyrotropic entropy |

The dataset's alias resolver is bidirectional for these e/i ↔ `_sN` pairs:
a reader that emits `Pe` (e.g. iPIC3D) satisfies a recipe asking for `P_s0`,
and vice versa. Storage is the same data either way; the canonical form
is a documentation choice.

This convention assumes species 0 = electrons, 1 = ions (standard in
PIC codes). For multi-species simulations (e.g. H⁺ + He²⁺ + O⁺), use
the explicit `_sN` form.

**Vector group shorthand in `read()`:** Passing a bare prefix like
`"B"` to `read(fields=...)` expands to `B1, B2, B3`. Per-species
groups also work: `"EF_s0"` expands to `EF1_s0, EF2_s0, EF3_s0`.
Derived quantities expand to their dependencies: `"Pi"` loads the
ion pressure tensor components `P11_s1`..`P33_s1`.

**Current limitations:**

- **Two-species assumption.** `P = Pe + Pi` and the `e`/`i` aliases
  hardcode species 0 = electrons, 1 = ions. For 3+ species, compute
  total pressure explicitly:
  `P = sum(data.compute(f"P_s{i}") for i in range(n_species))`.
  See `examples/advanced_calculations.py`.
- **Split-B naming.** The background field prefix `B0` ends in a digit,
  so its components use an underscore separator: `B0_1`, `B0_2`, `B0_3`
  (aliases: `B0x`, `B0y`, `B0z`). This is the only field prefix where
  the underscore is needed to avoid ambiguity with component indices.

### Fluid / moment quantities — velocities, currents, pressure

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `J1`, `J2`, `J3` | `Jx`, `Jy`, `Jz` | Current density | PIC (deposited), MHD (∇×B) |
| `V1`, `V2`, `V3` | `Vx`, `Vy`, `Vz` | Ion bulk velocity / fluid velocity | PIC (moments), MHD |
| `Ve1`, `Ve2`, `Ve3` | `Vex`, `Vey`, `Vez` | Electron bulk velocity | PIC |
| `u1`, `u2`, `u3` | `ux`, `uy`, `uz` | Four-velocity spatial components ($\gamma v^i$) | Relativistic PIC |
| `gamma_L` | — | Bulk Lorentz factor | Rel. PIC, Rel. MHD (derived) |
| `P` | — | Total scalar pressure | MHD, PIC (moments) |
| `Pe` | — | Electron scalar pressure | PIC, two-fluid MHD |
| `Pi` | — | Ion scalar pressure | PIC, two-fluid MHD |
| `Te` | — | Electron temperature (energy units) | PIC, two-fluid MHD |
| `Ti` | — | Ion temperature (energy units) | PIC, two-fluid MHD |

### Thermodynamic quantities

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `h` | Specific enthalpy (ideal gas) | MHD |
| `h_rel` | Relativistic specific enthalpy | Rel. MHD |
| `s` | Specific entropy (isotropic) | MHD, PIC |
| `s_e` | Electron entropy | PIC |
| `s_i` | Ion entropy | PIC |
| `s_gyro_e` | Electron gyrotropic entropy | PIC (anisotropic) |
| `s_gyro_i` | Ion gyrotropic entropy | PIC (anisotropic) |
| `e_int` | Specific internal energy | MHD |
| `gamma_eos` | Adiabatic index | MHD (from physics config) |

### Energy and flux quantities

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `S1`, `S2`, `S3` | `Sx`, `Sy`, `Sz` | Poynting flux | PIC, MHD |
| `EF1_s0`, `EF2_s0`, ... | `EFe`, `EFi` (vector groups) | Per-species total energy flux (3rd moment) | PIC, multi-moment MHD |
| `KEF1_s0`, `KEF2_s0`, ... | `KEFe`, `KEFi` | Kinetic energy flux (bulk flow) | PIC, MHD (derived) |
| `HF1_s0`, `HF2_s0`, ... | `HFe`, `HFi` | Total thermal flux (EF − KEF) | PIC, multi-moment MHD |
| `EHF1`, `EHF2`, `EHF3` | — | Enthalpy flux (total, fluid) | MHD, PIC (derived) |
| `EHF1_s0`, `EHF2_s0`, ... | `EHFe`, `EHFi` | Enthalpy flux (per-species) | PIC, MHD (derived) |
| `q1_s0`, `q2_s0`, ... | `qe`, `qi` | Conductive heat flux (HF − EHF) | PIC, multi-moment MHD |
| `e_B` | — | Magnetic energy density | PIC, MHD |
| `e_E` | — | Electric energy density | PIC |
| `e_k` | — | Kinetic energy density (total) | MHD, PIC (moments) |
| `e_k_s0`, `e_k_s1`, ... | `e_k_e`, `e_k_i` | Kinetic energy density (per-species) | PIC (derived) |
| `e_th` | — | Thermal energy density ($P/(\gamma-1)$) | MHD, PIC (moments) |
| `e_th_trace` | — | Thermal energy density ($\frac{1}{2}\mathrm{Tr}(\mathbf{P})$, $\gamma$-free) | PIC, multi-moment MHD |
| `e_th_s0`, `e_th_s1`, ... | `e_th_e`, `e_th_i` | Thermal energy density (per-species) | PIC (derived) |
| `rho_m_s0`, `rho_m_s1`, ... | `rho_m_e`, `rho_m_i` | Mass density (per-species) | PIC (derived) |
| `\|V\|_s0`, `\|V\|_s1`, ... | `\|Ve\|`, `\|Vi\|` | Velocity magnitude (per-species) | PIC (derived) |

### Pressure tensor

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `P_par` | Pressure parallel to B | PIC, multi-moment MHD (from tensor) |
| `P_perp` | Pressure perpendicular to B | PIC, multi-moment MHD (from tensor) |
| `Pij` | Full pressure tensor (6 independent components: P11, P12, P13, P22, P23, P33) | PIC, multi-moment MHD |
| `agyrotropy` | Agyrotropy measure (deviation from gyrotropic symmetry) | PIC, multi-moment MHD (derived) |

Any code that evolves the full pressure tensor — PIC, hybrid, 10-moment
MHD, CGL — can populate these fields. `P_par` and `P_perp` are
decomposed from the **total** pressure tensor (`P11..P33`). Per-species
decomposition (`P_par_s0`, `P_perp_s0`) uses the per-species tensors
(`P11_s0..P33_s0`).

**Storage vs derived.** Only the six tensor components (`Pij`,
`Pij_s{N}`) are storage-primitive in the canonical HDF5/Zarr layout;
`P` (trace/3), `P_par`, `P_perp`, and `agyrotropy` are derived on
demand by `compute()`. Listing `Pij` (or `Pi`/`Pe`) in
`[output.fields].quantities` auto-expands to the six components;
listing `P_par` writes the derived scalar — note that the post-hoc
decomposition is then locked to the snapshot's $\hat{b}$.

### Characteristic scales (derived)

| Canonical | Meaning | Computed from |
|-----------|---------|---------------|
| `d_e` | Electron skin depth | `n_s0`, species |
| `d_i` | Ion skin depth | `n_s1`, species |
| `r_e` | Electron thermal gyroradius | `Te`, `|B|`, species |
| `r_i` | Ion thermal gyroradius | `Ti`, `|B|`, species |
| `omega_pe` | Electron plasma frequency | `n_s0`, species |
| `omega_pi` | Ion plasma frequency | `n_s1`, species |
| `omega_ce` | Electron cyclotron frequency (positive by convention) | `|B|`, species |
| `omega_ci` | Ion cyclotron frequency (positive by convention) | `|B|`, species |
| `lambda_D` | Electron Debye length | `n_s0`, `Te`, species |
| `v_A` | Alfvén speed | `|B|`, `rho_m` |
| `v_th_e` | Electron thermal speed (NRL convention) | `Te`, species |
| `v_th_i` | Ion thermal speed (NRL convention) | `Ti`, species |
| `c_s` | Sound speed (MHD) | `P`, `rho_m`, `gamma_eos` |
| `v_ms` | Fast magnetosonic speed (perpendicular propagation) | `v_A`, `c_s` |
| `M_A` | Alfvén Mach number | `|V|`, `v_A` |
| `M_ms` | Magnetosonic Mach number | `|V|`, `v_ms` |
| `beta` | Plasma beta | `P`, `|B|` |
| `beta_e` | Electron beta | `Pe`, `|B|` |
| `beta_i` | Ion beta | `Pi`, `|B|` |
| `sigma` | Magnetization parameter | `|B|`, `rho_m`, `c` |

See the electron/ion convenience aliases table in the Densities section
for the full list of `e`/`i` shorthand names.

### Other derived quantities

| Canonical | Cartesian alias | Meaning | Computed from |
|-----------|----------------|---------|---------------|
| `\|B\|` | — | Magnetic field magnitude | B1, B2, B3 |
| `\|E\|` | — | Electric field magnitude | E1, E2, E3 |
| `\|J\|` | — | Current density magnitude | J1, J2, J3 |
| `\|V\|` | — | Bulk velocity magnitude | V1, V2, V3 |
| `\|Ve\|` | — | Electron velocity magnitude | Ve1, Ve2, Ve3 |
| `div_B` | — | Divergence of B (should be ~0) | B1, B2, B3, grid |
| `div_E` | — | Divergence of E | E1, E2, E3, grid |
| `curl_B1`, `curl_B2`, `curl_B3` | `curl_Bx`, ... | Curl of B | B1, B2, B3, grid |
| `vort1`, `vort2`, `vort3` | `vort_x`, ... | Fluid vorticity | V1, V2, V3, grid |
| `\|vort\|` | — | Vorticity magnitude | vort1, vort2, vort3 |
| `J_dot_E` | — | Energy conversion rate | J1-J3, E1-E3 |
| `E_prime_1`, `E_prime_2`, `E_prime_3` | `E_prime_x`, `E_prime_y`, `E_prime_z` | Non-ideal electric field | E1-E3, V1-V3, B1-B3 |
| `E_ideal_1`, `E_ideal_2`, `E_ideal_3` | `E_ideal_x`, `E_ideal_y`, `E_ideal_z` | Ideal electric field | V1-V3, B1-B3 |
| `E_Hall_1`, `E_Hall_2`, `E_Hall_3` | `E_Hall_x`, `E_Hall_y`, `E_Hall_z` | Hall electric field | J1-J3, B1-B3, n\_s0, species |
| `psi` | — | Magnetic flux function (2D) | B2, grid |
| `firehose` | — | Firehose instability parameter | P\_par, P\_perp, \|B\| |
| `mirror` | — | Mirror instability parameter | P\_par, P\_perp, \|B\| |

Scalar quantities (`n_s0`, `rho_m`, `P`, `Te`, `beta`, ...) use the same
name regardless of geometry.

See CLAUDE.md for naming conventions (functions, parameters, class names).

### Per-particle data columns

For PIC and hybrid codes that emit per-particle data, `ParticleData`
carries one canonical per-macroparticle representation regardless of
the source code's native storage convention: per-particle `weight`
(array) plus scalar `species_charge` and `species_mass`.  The
per-macroparticle charge and mass that enter moments and the equations
of motion are derived on demand via the :attr:`ParticleData.macro_charge`
and :attr:`macro_mass` properties.

| Field | Storage | Meaning |
|-------|---------|---------|
| `x`, `y`, `z` | per-particle (`(N,)` float) | Particle position components in the simulation Cartesian frame |
| `vx`, `vy`, `vz` | per-particle | Particle velocity components |
| `weight` | per-particle (`(N,)` float64) | Number of physical particles per macroparticle $w$ |
| `id` | per-particle (`(N,)` int64) | Integer tracking ID (when the code emits one) |
| `species_charge` | scalar (metadata) | Per-species charge $q_s$ in code units (e.g. ±1 for electrons/ions in iPIC3D normalization) |
| `species_mass` | scalar (metadata) | Per-species mass $m_s$ in code units |

The scalar `species_charge` and `species_mass` round-trip through the
Arrow/Parquet schema metadata (no per-particle storage cost).

**Native layouts (reader concern).** PIC codes split into two camps
for storing macroparticle charge on disk; readers handle the
translation, so downstream consumers always see the canonical form.

- **Combined** (iPIC3D, OSIRIS): on-disk field is $q_s w$. Readers
  split it into `weight` ($|q_s w|/|q_s|$) plus scalars from the run
  config.
- **Separate** (VPIC, WarpX, Smilei, EPOCH, PIConGPU, TRISTAN-MP):
  on-disk field is already `weight`; scalars come from config.

**Computing per-particle quantities.** `macro_charge`
(`species_charge × weight`) drives current/charge density;
`macro_mass` (`species_mass × weight`) drives kinetic energy and
mass density; $N_{\mathrm{phys}} = \sum w$. In non-uniform-weight
runs, each macroparticle's `weight` is unique at float64 precision
and can serve as a tracking ID — pass `id_column="weight"` to
`particles_from_dataset`.

**Single charge state per species.** One `[[species]]` entry carries
one scalar `species_charge`. Mixed ionization states (H⁺ + H²⁺ +
neutral H) must be modeled as separate species — every mainstream
PIC code follows this convention.

**Round-trip fidelity.** `read → write → read` reconstructs $q_s w$
bit-exactly via float64 arithmetic, but does not preserve the
original on-disk bytes of combined-storage files. pypic is analysis,
not simulation — restart regeneration is out of scope.

---

## 4. Output Layouts

Two on-disk layouts are defined: **HDF5 (§4.1)** is the canonical
cross-tool layout — what the Rust simulation code emits and what every
file-based reader translates *into*.  **Zarr (§4.2)** is what the
Python writer (`pypic.io.to_zarr` / `to_zarr_timeseries`) produces:
fields under `/fields`, metadata as flat keys on the root group's
attrs, with a `pypic_layout` discriminator.  Both use the same
**numbered canonical field names** (`B1`, `B2`, `B3`) and the same
section-level metadata vocabulary; the difference is the storage
container's group conventions (HDF5 groups vs Zarr DataTree).

### 4.1 HDF5 Output Layout

The cross-tool reference layout.  Each timestep is a separate file
(or group within a file).  Field datasets use the **numbered canonical
names** (`B1`, `B2`, `B3`), which are geometry-agnostic.

```
output_{step:06d}.h5
│
├── fields/                       # group: field data
│   ├── B1    [dataset, float64, shape (n1, n2, n3)]
│   ├── B2    [dataset, float64, shape (n1, n2, n3)]
│   ├── B3    [dataset, float64, shape (n1, n2, n3)]
│   ├── E1    [dataset, float64, shape (n1, n2, n3)]
│   ├── E2    [dataset, float64, shape (n1, n2, n3)]
│   ├── E3    [dataset, float64, shape (n1, n2, n3)]
│   ├── rho_c [dataset, float64, shape (n1, n2, n3)]  # PIC: charge density
│   ├── rho_m [dataset, float64, shape (n1, n2, n3)]  # MHD: mass density
│   ├── J1    [dataset, float64, shape (n1, n2, n3)]
│   ├── ...
│   ├── u1    [dataset, float64, shape (n1, n2, n3)]  # optional: four-velocity (rel. PIC)
│   ├── u2    [dataset, float64, shape (n1, n2, n3)]
│   ├── u3    [dataset, float64, shape (n1, n2, n3)]
│
├── grid/                         # group: grid metadata
│   ├── dimensions  [attr: (n1, n2, n3)]
│   ├── spacing     [attr: (d1, d2, d3)]
│   ├── origin      [attr: (min1, min2, min3)]
│   ├── dt          [attr: float64, code units]
│   ├── boundary    [attr: ("periodic", "periodic", "periodic")]
│   ├── geometry    [attr: "cartesian"]  # determines index interpretation
│   └── stagger     [attr: "cell"]      # original grid stagger type (provenance)
│
├── normalization/                # group: unit conversion metadata
│   ├── system      [attr: "PIC"]
│   ├── length_ref  [attr: float64, meters]
│   ├── time_ref    [attr: float64, seconds]
│   ├── b_field_ref [attr: float64, Tesla]
│   ├── velocity_ref [attr: float64, m/s]
│   └── density_ref [attr: float64, m⁻³]
│
├── time            [attr: float64, code units]
├── step            [attr: int]
└── model           [attr: "PIC"]
```

Every file contains enough metadata to convert back to SI without the
original `simulation.toml`. The HDF5 file itself always uses numbered
names; `geometry` drives alias registration in the reader (`Bx → B1`
for cartesian, `Br → B1` for spherical, etc). Existing readers (iPIC3D,
BATSRUS, ...) translate native layouts; the Rust code writes this
layout directly.

### 4.2 Zarr Output Layout

`pypic.io.to_zarr` and `to_zarr_timeseries` produce a Zarr v3 store
laid out as an xarray `DataTree`: the field arrays live under a
`/fields` child group (mirroring §4.1's `/fields/` HDF5 group); the
metadata sections sit as flat keys on the root group's attrs, with a
`pypic_layout` discriminator naming the layout version.  Consolidated
metadata is enabled — readers go through `consolidated="auto"` for
the one-shot metadata fetch when the writer left a consolidated
index, and fall back to listing for non-consolidated stores.

```
my_store.zarr/                         # Zarr v3 group root
│
├── attrs (flat root metadata):
│   ├── pypic_layout:  "v1"            # layout discriminator
│   ├── pypic_version: "0.x.y"         # writing pypic version
│   ├── grid:          { dimensions, spacing, origin, dt,
│   │                    boundary, surviving_axes,
│   │                    geometry: { type, axis_names, axis_units } }
│   ├── normalization: { length_ref, time_ref, velocity_ref,
│   │                    b_field_ref, e_field_ref, density_ref,
│   │                    mass_ref, charge_ref }
│   ├── species:       [ { name, charge, mass, ... }, ... ]
│   ├── physics:       { gamma, c, relativistic }
│   ├── frame:         "simulation"
│   ├── transforms:    { <name>: { origin, rotation, scale, ... }, ... }
│   └── metadata:      { ... reader-specific scalars,
│                         StaggerInfo as tagged dict ... }
│
└── fields/                             # /fields child group
    ├── B1, B2, B3, ...                 # field arrays
    ├── E1, E2, E3, ..., rho_c, rho_m, J1, ..., u1, u2, u3
    ├── x, y, z                         # 1-D coordinate arrays
    │                                   #   (axis names match geometry)
    └── time                            # only for multi-step writes
```

**Per-array attrs.**  Each field array under `/fields/` carries
xarray-style metadata: `long_name`, `units`, `quantity_type`,
`si_unit`, `latex`, and the openPMD-style `unit_dimension` 7-tuple
when the field has a registered SI dimension.

**Multi-step (timeseries).**  `to_zarr_timeseries` extends every
field array with a leading `time` dimension, shape `(nt, n1, n2, n3)`,
chunked so reading one timestep is O(1).  Step 1 writes the
`DataTree` with `mode="w"` (stamps root attrs); steps 2..N append
directly to `/fields` via `mode="a", append_dim="time"`.  The writer
enforces a constant field set and constant identity attrs (grid,
normalization, species, physics, frame, transforms) across appends,
and intersects per-step `metadata` to the keys whose values are
stable across all timesteps.

**Codecs.**  Default per-variable codec: `BloscCodec(cname="zstd",
clevel=5, shuffle="bitshuffle")`.  `dtype="float32"` downcasts on
write; user-supplied `encoding=` overrides per variable.

**Mapping to §4.1 (HDF5).**

| Aspect | §4.1 HDF5 | §4.2 Zarr (v1) |
|---|---|---|
| Field path | `/fields/B1` | `/fields/B1` |
| Grid metadata | `/grid/` group with attrs | root `attrs.grid` (JSON) |
| Normalization | `/normalization/` group | root `attrs.normalization` |
| Coord arrays | from grid attrs | `/fields/{x,y,z}` (1-D) |
| `time` / `step` | top-level scalar attrs | `time` dim (multi-step) |
| Files per write | one per timestep | one store, all steps |
| Layout discriminator | (none) | root `attrs.pypic_layout = "v1"` |

**Reading without pypic.**  A non-pypic consumer (JS WebGPU viewer,
Rust `zarrs` pipeline) opens the root group, reads the section dicts
straight from `attrs.grid` / `attrs.normalization` / etc., and reads
field arrays from `/fields/<name>`.  No Python or pypic library
required.  Coordinate arrays under `/fields` make the data
self-describing in CF/COARDS terms.

**Backward compatibility (v0).**  Stores written by pypic before the
layout change carry an `attrs["pypic"]` umbrella dict with field
arrays at the root group.  `from_zarr` auto-detects this layout via
the absence of `pypic_layout` and the presence of `pypic`, and
decodes transparently.  Writers always emit v1.

**Field naming invariant (both layouts).**  Stored arrays use the
numbered canonical names (`B1`, `B2`, `B3`).  Geometry- and
species-aliases are read-time conveniences resolved by
`FieldDataset`; they never appear on disk.

---

## 5. Extensibility

- **New simulation code:** Write a reader that maps native output to `FieldDataset` with canonical field names. No schema changes needed.
- **New field:** Add the name to the canonical table (this document), add to relevant readers, add derived functions if applicable.
- **New model type:** Add a `[physics.NEW_TYPE]` subsection convention, document expected fields and species.
- **New output format:** Define the layout mapping, write a reader. Everything downstream works unchanged via `FieldDataset`.
- **Code-specific knobs:** Park them under an `x-<code>` namespace (see §1 *Extensions*). Both `x-<code>` and `x_<code>` spellings are accepted; prefer the hyphenated `x-<code>` form for cross-tool portability — it matches the JSON Schema, OpenAPI, and openPMD extension conventions, and TOML parsers that emit warnings on bare keys with hyphens still parse `x-` correctly under quoting rules.

---

## 6. What This Schema Does NOT Define

- **Simulation control parameters** (solver tolerances, MPI decomposition, output intervals). Note: `dt` and `n_steps` live in `[time]` because they are essential for time-series analysis. Output cadence (`step_interval`) lives in `[output.*]`.
- **Visualization settings** (colormaps, camera angles, slice positions).
- **Exhaustive physics parameter lists.** The `[physics]` section is open-ended by design.
- **Native file layouts** of existing codes. Readers handle the translation.
- **Internal data structures** of any project. Each project maps to/from the schema at its boundaries.
- **API response format.** Defined in the Starlette/FastAPI/viewer project, not here.
