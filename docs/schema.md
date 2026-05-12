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

### Versioning

`[schema].version` is the single discriminator for both
`simulation.toml` and the on-disk output stores (HDF5 §4.1, Zarr §4.2,
each carrying the value at the root attr path `schema.version` —
the on-disk path mirrors the TOML form). v1.x commits to:

- **Additive only.** Future v1.x releases may add optional sections,
  optional keys, canonical field names, and enum values. They will
  not rename or remove existing canonical names, and will not move
  required keys. (For reference, the v0 → v1.0 rename list:
  `omega_pe_over_omega_ce` → `omega_p_over_omega_c`.)
- **Reordering `[[species]]` is breaking.** Per-species canonical
  names (`_s0`, `_s1`, ...) bind to declaration order; reordering
  changes the meaning of every per-species field on disk.
- **`x-*` extensions are unconstrained.** Code-specific knobs under
  `x-<code>.*` (or unknown sub-tables under `[physics.{pic,mhd,
  hybrid,vlasov}]`) are accepted by the validator without
  enforcement; v1.x will not break them, but also makes no
  promises about their portability.

Anything stricter than this — e.g., a removal or rename — bumps
`[schema].version` to `2.0`.

### Required sections

A valid v1.0 document must declare:

```toml
[schema]                 # version = "1.0" (REQUIRED)
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
[restart]                # continuation pointer (single file, glob, or list)
[output.*]               # five independent sub-sections plus the
                         # repeatable [[output.streams]] array, all
                         # walked together in §2:
                         #   [output.checkpoints]   lossless full-state dumps
                         #   [output.fields]        field output cadence + quantities
                         #   [output.particles]     particle output cadence + selection
                         #   [output.probes]        probe time-series cadence
                         #   [output.diagnostics]   on-the-fly derived quantities
                         #   [[output.streams]]     multi-cadence / ROI output groups
                         #                          (repeatable, optional)
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

### Strict vs open string vocabularies

Two flavours of string-valued field appear throughout §2:

- **Strict** (closed enum) — bounded structural / format primitives.
  The vocabulary is fixed by the data model itself: `[schema].version`,
  `[model].type`, `[coordinates].geometry`, `[coordinates].physical_extent_unit`,
  `[grid.amr].amr_kind`, `[grid.stagger].convention`,
  `[grid.stagger].fields` location values, `[output.*].format`,
  `[output.*].precision`, `[output.streams.region].kind`,
  `[restart].mode`, `[[drivers]].coupling`, `[[drivers]].direction`,
  `[[bodies]].shape`, `[time].splitting`,
  `[phase_space].coordinate_system`. Adding a value is a v1.x
  schema bump.
- **Open** (canonical list, unenforced) — numerical-method,
  algorithm, and closure vocabularies: `[time].scheme`,
  `[physics.*.solver].scheme`, `pusher`, `field_solver`, `limiter`,
  `riemann`, `reconstruction`, `divergence_cleaning`,
  `preconditioner`, `charge_correction`, `current_deposition`,
  `[[species]].closure`, `[[species]].shape` (PIC shape factor),
  `[[collisions]].model`, `[[drivers]].type`,
  `[boundary_conditions]` tag values. Research codes invent new
  schemes faster than the schema can enumerate them, so unknown
  strings pass validation. The lists shown next to each field are
  the v1.0 canonical values — guidance for tooling and human
  readers, not validation gates. pypic records the value as
  provenance metadata; it does not dispatch on it. Cross-field
  rules (e.g. `scheme = "subcycled"` requires `field_substeps`)
  still fire — openness applies to *value*, not *semantics*.

Per-field annotation in §2 marks each enum's category inline.

---

## 2. Section Specifications

### [schema]

Schema version tag and creation date.

```toml
[schema]
version = "1.0"                    # REQUIRED — single discriminator for the
                                   #            TOML config and on-disk stores.
                                   #            On disk the value lives at
                                   #            attrs.schema.version (Zarr §4.2)
                                   #            and /schema/version (HDF5 §4.1),
                                   #            mirroring this TOML key path.
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

`[run]` is **typed run provenance**: the validator reads it when
``simulation.toml`` is loaded, the loader stamps the result onto
`SimulationConfig.run`, and the Zarr writer lifts it to root
`attrs.run` as a JSON-mode `Run.model_dump` so cross-tool consumers
see it at the root group's attrs (see §4.2). Identity-stable across
derivations — a regridded "MMS-event-1" run is still that run. The
HDF5 §4.1 layout does not currently round-trip `[run]`; tools whose
artifacts must travel through HDF5 should ship the original
``simulation.toml`` alongside the output store, or write through
Zarr where `attrs.run` is preserved. `[run]` lives alongside
`attrs.simulation_toml` (verbatim text) — see §4.2 for the policy
on each.

### [time]

Temporal integration controls.

```toml
[time]
scheme  = "fixed"                  # open string. Canonical: "fixed" (default)
                                   #   | "adaptive" | "subcycled" | "rk2" | "rk3"
                                   #   | "rk4" | "vl2" | "ssprk2" | "ssprk3"
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
# `upper[i] - lower[i]` within `sum_rtol`. The check is the relative
# form  `abs(sum - extent) <= sum_rtol * max(abs(extent), 1.0)`, with
# `sum_rtol` defaulting to 1.0e-9. Rounding direction (toward zero,
# nearest, etc.) is intentionally unspecified — pick `sum_rtol` to
# absorb the writer's float drift, or pre-snap the widths so the sum
# matches `extent` bit-exactly.
[grid.stretched]
sum_rtol = 1.0e-9                  # optional — float drift tolerance on
                                   #   per-axis-width sum vs (upper - lower)
[grid.stretched.axis_widths]
0 = [0.5, 0.6, 0.8, 1.0, 1.3]      # x stretched
2 = [0.1, 0.1, 0.2, 0.4, 0.8]      # z log-radial (ARMS, PLUTO style)

# Stagger — three optional tiers (convention, fields, position).
# See "Stagger layering" below; readers destagger to co-located grids on load.
[grid.stagger]
convention = "staggered"           # Tier 1 — "cell" (default) | "node" | "staggered"

[grid.stagger.fields]              # Tier 2 — per-field-group locations
B = "face"                         #   values: "cell" | "node" | "face" | "edge"
E = "edge"
J = "edge"

[grid.stagger.position]            # Tier 3 — per-component ED-PIC offsets in [0.0, 1.0)
B_1 = [0.5, 0.0, 0.0]               # B_x on the x-face
B_2 = [0.0, 0.5, 0.0]               # B_y on the y-face
B_3 = [0.0, 0.0, 0.5]               # B_z on the z-face
E_1 = [0.0, 0.5, 0.5]               # E_x on the x-edge
```

**Stagger layering.** Three tiers, all optional, all additive — same
fact, increasing precision:

- **Tier 1** — `convention` is a one-word summary for human readers
  and downstream UIs ("cell-centered", "node-centered", "Yee mesh").
  Informational only; readers always destagger to a co-located grid
  on load.
- **Tier 2** — `fields` records per-group locations (`B` on faces,
  `E` on edges) for readers that consume the field-group hint when
  destaggering.
- **Tier 3** — `position` records the ED-PIC per-component offset
  inside the local cell on a $[0.0, 1.0)$ scale (the openPMD
  `position` array).  Round-trips through `StaggerInfo.position`.

The validator does **not** cross-check between tiers — a writer may
populate any subset, and a reader that consumes only Tier 1 still
works.  Readers that need an ED-PIC-precise destagger reach for Tier 3.

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

The string `reference_species` resolves against `[[species]].name`
when one matches; the three builtins `"electrons"`, `"ions"`,
`"protons"` are accepted as a fallback when no matching `[[species]]`
entry exists. This keeps legacy decks (iPIC3D bare runs that never
declare an `"ions"` species, hybrid configs that omit electrons)
parseable without forcing a downstream rename.

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

**Chain resolution.** The reader builds a directed graph from
`from_frame → frame_name` for every entry under
`[coordinates.transforms.*]` (entries without `from_frame` source
from `[coordinates].frame`). Resolving a target frame is a
breadth-first search from `[coordinates].frame`, returning the
shortest path; a cycle in the graph (`A → B → A`) raises a
validation error at apply time. When two paths share the same
length, the one declared first in the TOML wins — a stable
deterministic tie-breaker. The schema layer doesn't enforce
acyclicity (transforms are stored as a flat dict); the cycle check
fires when `transform_to()` is invoked.

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

Describes a kinetic species (PIC / hybrid kinetic) or a fluid species
(MHD / hybrid fluid). Repeatable — any number of species, any mix of
modes.

```toml
[[species]]
name = "string"                    # REQUIRED

# Identity (REQUIRED — pick one form)
charge = 0.0                       # code units — REQUIRED if mass is set (XOR with charge_to_mass)
mass = 0.0                         # code units — REQUIRED if charge is set (XOR with charge_to_mass)
charge_to_mass = 0.0               # code units — REQUIRED if charge+mass are omitted

# Shared moments / initial-condition state
temperature = 0.0                  # optional — scalar isotropic temperature (code units)
density = 1.0                      # optional — number density (code units)
drift_velocity = [0.0, 0.0, 0.0]   # optional — bulk drift (code units)
thermal_velocity = [0.0, 0.0, 0.0] # optional — per-component thermal velocity

# Kinetic-only (PIC / hybrid kinetic species)
particles_per_cell = [5, 5, 1]     # optional — scalar OR [nx, ny, nz]; 0 or absent = fluid
shape = "cic"                      # optional — "ngp" | "cic" | "tsc" | "pqs"
                                   #   (NGP=order 0, CIC=1, TSC=2, PQS=3; ED-PIC vocabulary)
tracer = false                     # optional — tagged subset for trajectory tracking;
                                   # readers propagate the hint to particle analysis

# Fluid-only (MHD / hybrid fluid species)
closure = "adiabatic"              # optional, open string. Canonical: "isothermal"
                                   #   | "adiabatic" | "polytropic" | "braginskii"
                                   #   | "cgl" | "10moment" | "14moment"
gamma_eos = 1.6666667              # optional — per-species adiabatic index
gamma_eos_par = 3.0                # optional (10-moment hybrids: Gkeyll, Hakim) — parallel
gamma_eos_perp = 2.0               # optional (10-moment hybrids) — perpendicular
                                   # par/perp must be present together; cannot mix with
                                   # the scalar `gamma_eos`.
inertia = 0.0                      # optional (hybrid fluid electrons) — me/mi; 0 = massless
```

The validator enforces `(charge + mass)` XOR `charge_to_mass`: provide
exactly one form. Species are 0-indexed in declaration order — the
binding is part of the schema contract; see §1 *Versioning*.

Temperatures are stored in **energy units** (J in SI), not Kelvin —
`T = P/n` carries no $k_B$ factor, and $v_{th} = \sqrt{T/m}$. See
[conventions.md § Temperature in Energy
Units](conventions.md#temperature-in-energy-units) for conversion to
eV/K and the openPMD impedance note.

### [physics]

Model-agnostic flags at the top level; model-specific knobs under
sub-tables matching `[model].type`. v1.0 enumerates only
`relativistic` at the top level; unknown top-level keys are accepted
to leave room for v1.1+ portable additions (anticipated:
`collisional`, `radiative`, `[physics.vlasov]`). Code-specific knobs
go under `[physics.<model>.x-<code>.*]`.

All `scheme`, `pusher`, `field_solver`, `reconstruction`, `limiter`,
`riemann`, `divergence_cleaning`, `preconditioner`, `charge_correction`,
and `current_deposition` values below are open strings — research
methods that don't appear in the canonical list still validate.
See §1 *Strict vs open string vocabularies*.

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
gamma_eos   = 1.6667                # adiabatic index ($c_p / c_v$); name
                                    #   matches the per-species form on
                                    #   [[species]] and disambiguates from
                                    #   the canonical `gamma_L` (bulk Lorentz
                                    #   factor) field used in relativistic
                                    #   contexts.
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
# Reserved for v1.1+ hybrid physics-model knobs. Empty in v1.0 —
# hybrid fluid-species properties live on the corresponding [[species]]
# entry; code-specific knobs go under [physics.hybrid.x-<code>.*].

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

`omega_p_over_omega_c` is species-agnostic — it applies to whichever
species is declared in `[units].reference_species`. Unknown keys under
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
intrinsic_dipole     = [0.0, 0.0, -190.0] # split-B analytic background; surfaces on disk as B0_*
                                          #   (see § Canonical Field Names → Electromagnetic fields)
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

**`[boundary_conditions]` ↔ `[[drivers]]` interaction.** A face listed
in `boundary_conditions.drivers_lower` / `drivers_upper` is sourced
from the named driver entry; the BC tag at that face (e.g.
`"driven"`, `"inflow"`) is the *category*, the driver is the
*supplier*. Driver geometry composes on top of the BC face:
`coupling = "boundary"` always means "values applied at the face
identified by `drivers_lower/upper`"; `target_lower / target_upper`
on the driver further narrows the face's spatial extent (e.g. only
the dayside hemisphere of an inflow face); a `body` reference on
the driver narrows again (constrains to the body's footprint on
that face). Volume / source / sink couplings ignore the face entry
in `drivers_lower/upper` and use `target_lower / target_upper` (or
`body`) to define their domain.

Driver-type-specific keys (`production_rate`, `fields`, etc.) are
accepted beyond the core vocabulary above.

### [restart]

Continuation pointer from a prior run. ``from`` is the path (or
paths) to the restart artifact:

- a single file (``./chk_000030.h5``),
- a directory of per-rank checkpoints (``./restart_30000/``),
- a glob pattern, or
- an explicit list of per-rank files for runs whose filenames don't
  follow the source code's convention (relocated reruns, mixed
  naming schemes).

Whether the path resolves to one file or many is determined at read
time by the filesystem and the code, not by the schema.

```toml
[restart]
from = "./checkpoints/chk_000030.h5"   # REQUIRED — string OR list of strings.
                                        #   (Aliased — `from` is a Python keyword.)
step = 30000                            # optional
time = 1500.0                           # optional
restore = ["fields", "particles"]       # optional — partial restart selector
                                        #   ("fields" | "particles" | "auxiliary");
                                        #   entries must be distinct. Omitting
                                        #   restores everything.
mode = "hot"                            # optional — "hot" (full state reload) |
                                        #   "cold" (re-apply IC on saved geometry)

# Escape-hatch shape — explicit per-rank file list (distinct, non-empty):
# from = [
#     "chk_000030_rank_00000.h5",
#     "chk_000030_rank_00001.h5",
#     "chk_000030_rank_00002.h5",
# ]
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
quantities    = ["B", "E", "J", "rho_c", "V_s0"]  # PIC-flavored example; vector/tensor groups
                                                  # auto-expand. MHD substitutes rho_m for rho_c
                                                  # and V for V_s0 — see pypic.simulation.toml
                                                  # Scenario B for a single-fluid MHD block.
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

### [[collisions]]

Per-pair collision declaration for collisional PIC codes (Smilei,
EPOCH, OSIRIS-collisional, PIConGPU). Repeatable. Both species in
``species_pair`` must match a declared ``[[species]].name`` —
unresolved names raise a validation error. Self-collisions (same
species twice) are permitted.

```toml
[[collisions]]
species_pair    = ["electrons", "ions"] # REQUIRED — both names must match [[species]]
model           = "coulomb"            # REQUIRED, open string. Canonical:
                                       #   "coulomb" | "bgk" | "monte-carlo"
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
fields = ["B_1", "B_2", "B_3", "beta"]  # optional: fields to sample.
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

Vector field components use **numbered indices** (`B_1`, `B_2`, `B_3`) as
the general canonical form. The coordinate geometry determines what
each index means:

| Index | Cartesian | Spherical | Cylindrical | thetaMode |
|-------|-----------|-----------|-------------|-----------|
| 1 | x | r | r | r |
| 2 | y | θ | φ | z |
| 3 | z | φ | z | mode index |

For Cartesian data, letter aliases (`Bx`, `By`, `Bz`) are preferred for
readability and access the same data as `B_1`, `B_2`, `B_3`. For non-Cartesian
data, only the numbered form is canonical; the reader registers
geometry-appropriate aliases (e.g., `Br` → `B_1` for spherical). The
`[coordinates] geometry` field determines which aliases are active.

**Alias-on-disk policy.** Aliases are read-time conveniences, derived
from `[coordinates].geometry` after the file is opened. Writers MUST
emit only the numbered canonical form (`B_1` / `B_2` / `B_3`,
`B0_1` / `B0_2` / `B0_3`, `n_s0`, `V_s1_1`, `P_s0_11`, …). Storing
both `B_1` and `Bx` for the same data is redundant and not part of
the v1.0 contract — readers that encounter both should treat the
numbered form as authoritative and ignore the duplicate.

For `geometry = "thetaMode"` (FBPIC azimuthal-mode RZ decomposition),
the spatial grid is two-dimensional `(r, z)` and the third index is
the openPMD mode number from `[coordinates.modes].mode_indices`.
Field arrays therefore have shape `(n_r, n_z, n_modes)`; the per-mode
complex amplitude is stored as a complex dtype, not as separate real
and imaginary axes.

### Electromagnetic fields

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `B_1`, `B_2`, `B_3` | `Bx`, `By`, `Bz` | Magnetic field components | PIC, MHD |
| `B0_1`, `B0_2`, `B0_3` | `B0x`, `B0y`, `B0z` | Background magnetic field (split-B) | MHD (optional) |
| `E_1`, `E_2`, `E_3` | `Ex`, `Ey`, `Ez` | Electric field components | PIC (MHD: derived) |

### Fluid / moment quantities — densities

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `n_s0`, `n_s1`, ... | Number density per species | PIC, multi-fluid MHD |
| `rho_c` | Total charge density | PIC |
| `rho_m` | Total mass density | MHD, PIC (derived) |

`rho_c` and `rho_m` are unambiguous — no overloaded `rho`.

**Per-species naming:** Append `_s` plus the 0-based species index
(`rho_c_s0`, `J_s1_1`, `n_s3`); the component index comes before the
species suffix (`J_s0_1`, `EF_s1_2`). The species *name* lives in
`[[species]]`, not in the field name.

**Species-name aliases:** For any species, the canonical `<prefix>_s<index>`
form has an automatically-generated `<prefix>_<species_name>` alias when
the species name is declared in `[[species]]`. `n_s0` becomes `n_electrons`
when `species[0].name == "electrons"`; `EF_s1_1` becomes `EF1_protons` when
`species[1].name == "protons"`. The alias is added at `FieldDataset`
construction time and only registered when the underlying canonical is
actually present in the dataset, so missing data produces a clean
`KeyError` rather than misdirection.

For multi-species runs (H⁺ + He²⁺ + O⁺), the species-name form is the
unambiguous way to reference per-species quantities — the integer index
depends on declaration order.

**Vector group shorthand in `read()`:** Passing a bare prefix like
`"B"` to `read(fields=...)` expands to `B_1, B_2, B_3`. Per-species
groups work the same way: `"EF_s0"` expands to
`EF_s0_1, EF_s0_2, EF_s0_3`. Derived quantities expand to their
dependencies: `"P_s1"` loads the six ion pressure tensor components,
`"P_par"` loads the six total-pressure tensor components plus
`B_1`..`B_3`.

pypic also ships **library-side convenience aliases** for the common
two-species electron/ion case (`Pe ↔ P_s0`, `n_i ↔ n_s1`,
`beta_e ↔ beta_s0`, ...) — see [Aliases](aliases.md). Those forms
are not part of the cross-tool schema contract; non-pypic consumers
(Rust, JS) only need the canonical names listed here.

What does *not* expand: already-resolved single-component aliases
(`"Bx"`, `"Br"`, `"E_phi"`) and names whose prefix already ends in
a digit (`"B_1"`, `"P_s1_11"`) are passed through as scalars. In
particular, requesting `"P_11"` alone loads only `P_11` — if a later
`compute("P_par")` / `"P_perp"` / `"agyrotropy"` needs the full
tensor, request the tensor explicitly via `"P_s0"` / `"P_s1"` /
`"P_sN"`, or request the derived quantity itself.

The same expansion rules apply to `[output.fields].quantities` —
listing `"B"` writes `B_1`, `B_2`, `B_3`; listing `"P_s1"` writes the
six second-species pressure tensor components.

**Current limitations:**

- **Split-B naming.** `B0` is the universal plasma-physics name for the
  background/asymptotic magnetic field, so we preserve it. Because the
  prefix already ends in a digit, components use an underscore separator:
  `B0_1`, `B0_2`, `B0_3` (Cartesian aliases: `B0x`, `B0y`, `B0z`). The
  underscore-after-digit rule generalizes to any future canonical whose
  prefix ends in a digit; `B0` is the only current customer.

### Fluid / moment quantities — velocities, currents, pressure

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `J_1`, `J_2`, `J_3` | `Jx`, `Jy`, `Jz` | Current density | PIC (deposited), MHD (∇×B) |
| `V_1`, `V_2`, `V_3` | `Vx`, `Vy`, `Vz` | Fluid bulk velocity (single-fluid MHD) | MHD |
| `V_s{N}_1`, `V_s{N}_2`, `V_s{N}_3` | — | Per-species bulk velocity | PIC, multi-fluid MHD |
| `u_1`, `u_2`, `u_3` | `ux`, `uy`, `uz` | Four-velocity spatial components ($\gamma v^i$) | Relativistic PIC |
| `gamma_L` | — | Bulk Lorentz factor | Rel. PIC, Rel. MHD (derived) |
| `P` | — | Total scalar pressure | MHD, PIC (moments) |
| `P_s{N}` | — | Per-species scalar pressure | PIC, multi-fluid MHD |
| `T_s{N}` | — | Per-species temperature (energy units) | PIC, multi-fluid MHD |

### Thermodynamic quantities

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `h` | Specific enthalpy (ideal gas) | MHD |
| `h_rel` | Relativistic specific enthalpy | Rel. MHD |
| `s` | Specific entropy (isotropic, single-fluid) | MHD, PIC |
| `s_s{N}` | Per-species specific entropy | PIC, multi-fluid MHD |
| `s_gyro_s{N}` | Per-species gyrotropic entropy | PIC (anisotropic) |
| `e_int` | Specific internal energy | MHD |

Adiabatic index `γ` is a config scalar, not a per-cell field — it
lives in `[physics.mhd].gamma_eos` (single-fluid) or
`[[species]].gamma_eos` (multi-fluid / hybrid). `compute("h")`,
`compute("e_int")`, and the relativistic enthalpy resolve it from
the dataset's physics config.

### Energy and flux quantities

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `S_1`, `S_2`, `S_3` | `Sx`, `Sy`, `Sz` | Poynting flux | PIC, MHD |
| `EF_s{N}_1`, `EF_s{N}_2`, `EF_s{N}_3` | — | Per-species total energy flux (3rd moment) | PIC, multi-moment MHD |
| `KEF_s{N}_1`, `KEF_s{N}_2`, `KEF_s{N}_3` | — | Per-species kinetic energy flux (bulk flow) | PIC, MHD (derived) |
| `HF_s{N}_1`, `HF_s{N}_2`, `HF_s{N}_3` | — | Per-species total thermal flux (EF − KEF) | PIC, multi-moment MHD |
| `EHF_1`, `EHF_2`, `EHF_3` | — | Enthalpy flux (total, fluid) | MHD, PIC (derived) |
| `EHF_s{N}_1`, `EHF_s{N}_2`, `EHF_s{N}_3` | — | Per-species enthalpy flux | PIC, MHD (derived) |
| `q_s{N}_1`, `q_s{N}_2`, `q_s{N}_3` | — | Per-species conductive heat flux (HF − EHF) | PIC, multi-moment MHD |
| `e_B` | — | Magnetic energy density | PIC, MHD |
| `e_E` | — | Electric energy density | PIC |
| `e_k` | — | Kinetic energy density (total) | MHD, PIC (moments) |
| `e_k_s{N}` | — | Per-species kinetic energy density | PIC (derived) |
| `e_th` | — | Thermal energy density ($P/(\gamma-1)$) | MHD, PIC (moments) |
| `e_th_trace` | — | Thermal energy density ($\frac{1}{2}\mathrm{Tr}(\mathbf{P})$, $\gamma$-free) | PIC, multi-moment MHD |
| `e_th_s{N}` | — | Per-species thermal energy density | PIC (derived) |
| `rho_m_s{N}` | — | Per-species mass density | PIC (derived) |
| `\|V_s{N}\|` | — | Per-species velocity magnitude | PIC (derived) |

### Pressure tensor

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `P_par` | Pressure parallel to B | PIC, multi-moment MHD (from tensor) |
| `P_perp` | Pressure perpendicular to B | PIC, multi-moment MHD (from tensor) |
| `Pij` | Full pressure tensor (6 independent components: P_11, P_12, P_13, P_22, P_23, P_33) | PIC, multi-moment MHD |
| `agyrotropy` | Swisdak $Q$ — see [equations.md § Pressure Tensor](equations.md#4-pressure-tensor) for the closed form $Q = \sqrt{1 - 4 I_2 / [(I_1 - P_\parallel)(I_1 + 3 P_\parallel)]}$, bounded $[0, 1]$ | PIC, multi-moment MHD (derived) |

Any code that evolves the full pressure tensor — PIC, hybrid, 10-moment
MHD, CGL — can populate these fields. `P_par` and `P_perp` are
decomposed from the **total** pressure tensor (`P_11..P_33`). Per-species
decomposition (`P_s0_par`, `P_s0_perp`) uses the per-species tensors
(`P_s0_11..P_s0_33`) and follows the Tier-3 species-before-modifier
template: species qualifier sits before the generic operator suffix
(`_par`, `_perp`). The two-species shorthands `P_par_e` / `P_par_i` /
`P_perp_e` / `P_perp_i` remain registered as user-facing aliases.

**Storage vs derived.** Two tiers, in preference order:

1. **Preferred** — store the six tensor components (`Pij`,
   `Pij_s{N}`).  `P` (trace/3), `P_par`, `P_perp`, and `agyrotropy`
   are then derived on demand by `compute()` and stay
   $\hat{b}$-fresh under frame transforms (the decomposition follows
   whatever $\mathbf{B}$ is in the current frame).  Listing `Pij`
   (or per-species `Pij_s0`, `Pij_s1`) in
   `[output.fields].quantities` auto-expands to the six components.
2. **Acceptable** — codes that evolve only the CGL/double-adiabatic
   decomposition (BATSRUS `MhdAnisoP`, GRMHD with anisotropic
   closure) may store `P_par` and `P_perp` directly without the
   tensor.  Frame transforms then cannot re-decompose; the stored
   values lock to the snapshot's $\hat{b}$.  Listing `P_par` in
   `[output.fields].quantities` writes the derived scalar with the
   same lock-in caveat.

`P_par`, `P_perp`, and `agyrotropy` carry no marker on disk
distinguishing "computed by the code" from "derived by `compute()`" —
the consumer treats them identically.

**Canonical interchange form.** Cross-tool readers MUST accept either
tier without negotiation. When the six-component tensor is on disk,
the reader exposes `P_par` / `P_perp` / `agyrotropy` via on-the-fly
decomposition; when only `P_par` / `P_perp` are stored, the reader
exposes them as primitives and the tensor-derived quantities are
unavailable. Writers SHOULD emit the six-component form whenever the
code's data model carries it (preferred tier) — `compute()` then
stays $\hat{b}$-fresh under frame transforms. The HDF5 layout (§4.1)
and the Zarr layout (§4.2) both show the six-component form as the
canonical naming because that's the higher-fidelity tier; the
acceptable-tier shape is the same store with `P_par` / `P_perp`
arrays in place of `P_11..P_33`.

### Characteristic scales (derived)

| Canonical | NRL alias | Meaning | Computed from |
|-----------|-----------|---------|---------------|
| `d_s0` | `d_e` | Electron skin depth | `n_s0`, species |
| `d_s1` | `d_i` | Ion skin depth | `n_s1`, species |
| `r_s0` | `r_e` | Electron thermal gyroradius | `T_s0`, `|B|`, species |
| `r_s1` | `r_i` | Ion thermal gyroradius | `T_s1`, `|B|`, species |
| `omega_p_s0` | `omega_pe` | Electron plasma frequency | `n_s0`, species |
| `omega_p_s1` | `omega_pi` | Ion plasma frequency | `n_s1`, species |
| `omega_c_s0` | `omega_ce` | Electron cyclotron frequency (positive by convention) | `|B|`, species |
| `omega_c_s1` | `omega_ci` | Ion cyclotron frequency (positive by convention) | `|B|`, species |
| `lambda_D_s0` | `lambda_D` | Electron Debye length | `n_s0`, `T_s0`, species |
| `v_A` | — | Alfvén speed | `|B|`, `rho_m` |
| `v_th_s0` | `v_th_e` | Electron thermal speed (NRL convention) | `T_s0`, species |
| `v_th_s1` | `v_th_i` | Ion thermal speed (NRL convention) | `T_s1`, species |
| `c_s` | — | Sound speed (MHD) | `P`, `rho_m`, `gamma_eos` |
| `v_ms` | — | Fast magnetosonic speed (perpendicular propagation) | `v_A`, `c_s` |
| `M_A` | — | Alfvén Mach number | `|V|`, `v_A` |
| `M_ms` | — | Magnetosonic Mach number | `|V|`, `v_ms` |
| `beta` | — | Plasma beta (auto-expands to `beta_s{N}` per species) | `P` (or `P_s{N}`), `|B|` |
| `sigma` | — | Magnetization parameter | `|B|`, `rho_m`, `c` |

The Tier-3 `<field>_s<N>` form is the canonical recipe ID — the same
template used for storage names elsewhere in the schema. NRL Plasma
Formulary spellings (`omega_pe`, `lambda_D`, `v_th_e`, ...) resolve
to the canonical via `_COMPUTE_ALIASES` and remain the human-friendly
form in error messages, doctests, and physics-literature contexts.
Multi-species runs (`omega_p_s2`, `lambda_D_s3`, ...) synthesize via
`_SPECIES_TEMPLATES` on demand.

### Other derived quantities

| Canonical | Cartesian alias | Meaning | Computed from |
|-----------|----------------|---------|---------------|
| `\|B\|` | — | Magnetic field magnitude | B_1, B_2, B_3 |
| `\|E\|` | — | Electric field magnitude | E_1, E_2, E_3 |
| `\|J\|` | — | Current density magnitude | J_1, J_2, J_3 |
| `\|V\|` | — | Bulk velocity magnitude | V_1, V_2, V_3 |
| `\|V_s{N}\|` | — | Per-species velocity magnitude | V_s{N}_1, V_s{N}_2, V_s{N}_3 |
| `div_B` | — | Divergence of B (should be ~0) | B_1, B_2, B_3, grid |
| `div_E` | — | Divergence of E | E_1, E_2, E_3, grid |
| `curl_B_1`, `curl_B_2`, `curl_B_3` | `curl_Bx`, ... | Curl of B | B_1, B_2, B_3, grid |
| `vort_1`, `vort_2`, `vort_3` | `vort_x`, ... | Fluid vorticity | V_1, V_2, V_3, grid |
| `\|vort\|` | — | Vorticity magnitude | vort_1, vort_2, vort_3 |
| `J_dot_E` | — | Energy conversion rate | J_1-J_3, E_1-E_3 |
| `E_prime_1`, `E_prime_2`, `E_prime_3` | `E_prime_x`, `E_prime_y`, `E_prime_z` | Non-ideal electric field | E_1-E_3, V_1-V_3, B_1-B_3 |
| `E_ideal_1`, `E_ideal_2`, `E_ideal_3` | `E_ideal_x`, `E_ideal_y`, `E_ideal_z` | Ideal electric field | V_1-V_3, B_1-B_3 |
| `E_Hall_1`, `E_Hall_2`, `E_Hall_3` | `E_Hall_x`, `E_Hall_y`, `E_Hall_z` | Hall electric field | J_1-J_3, B_1-B_3, n\_s0, species |
| `psi` | — | Magnetic flux function (2D) | B_2, grid |
| `firehose` | — | Firehose instability parameter | P\_par, P\_perp, \|B\| |
| `mirror` | — | Mirror instability parameter | P\_par, P\_perp, \|B\| |

Scalar quantities (`n_s0`, `rho_m`, `P`, `T_s0`, `beta`, ...) use the
same name regardless of geometry.

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

Two on-disk layouts are defined, with deliberately asymmetric roles:

- **§4.1 HDF5** is the canonical **read contract** for HDF5-emitting
  codes — the Rust simulation code, iPIC3D, BATSRUS, ARMS, OpenGGCM —
  and what every file-based reader in `pypic.readers` translates
  *into*.  pypic itself does not write this layout; there is no
  `to_hdf5` in `pypic.io`.
- **§4.2 Zarr** is what pypic itself produces via
  `pypic.io.to_zarr` / `to_zarr_timeseries`.  Fields under
  `/fields`, metadata as flat keys on the root group's attrs.

Both layouts share the same **numbered canonical field names** (`B_1`,
`B_2`, `B_3`) and the same section-level metadata vocabulary, and both
carry the schema version at the root attr path `schema.version`
(`/schema/version` for HDF5, `attrs.schema.version` for Zarr) whose
value equals `simulation.toml`'s `[schema].version` — a single
discriminator across config and on-disk storage.  The difference is
the storage container's group conventions (HDF5 groups vs Zarr
DataTree).

### 4.1 HDF5 Output Layout

The cross-tool **read contract** for HDF5-emitting codes.  pypic does
not write this layout — its readers translate native HDF5 (and IDL,
H5hut, Fortran-binary, ...) sources *into* a `FieldDataset` that
conforms to the same canonical naming and metadata shape described
here.  The Rust simulation code writes this layout directly, so the
section doubles as its output specification.

Each timestep is a separate file (or group within a file).  Field
datasets use the **numbered canonical names** (`B_1`, `B_2`, `B_3`),
which are geometry-agnostic.

```
output_{step:06d}.h5
│
├── fields/                            # group: field data (numbered canonical names)
│   ├── B_1, B_2, B_3                     # [dataset, float64, shape (n1, n2, n3)]
│   ├── E_1, E_2, E_3
│   ├── J_1, J_2, J_3
│   ├── rho_c                          # PIC: charge density (omit for MHD-only output)
│   ├── rho_m                          # MHD: mass density   (PIC may also write derived)
│   ├── n_s0, n_s1, ...                # per-species number densities
│   ├── V_s0_1, V_s0_2, V_s0_3, ...       # per-species bulk velocities
│   ├── P_s0_11, P_s0_12, ..., P_s0_33    # per-species pressure tensor (6 components)
│   └── u_1, u_2, u_3                     # optional — four-velocity (relativistic PIC)
│
├── grid/                              # group: grid metadata
│   ├── dimensions      [attr: (n1, n2, n3)]
│   ├── spacing         [attr: (d1, d2, d3)]
│   ├── origin          [attr: (min1, min2, min3)]
│   ├── dt              [attr: float64, code units]
│   ├── boundary_lower  [attr: ("periodic", "periodic", "open")]
│   ├── boundary_upper  [attr: ("periodic", "periodic", "open")]
│   ├── geometry        [attr: "cartesian"]   # drives alias registration
│   └── stagger/                              # group — three optional tiers (§2 [grid.stagger])
│       ├── convention [attr: "cell"]         # Tier 1 — "cell" | "node" | "staggered"
│       ├── fields     [attr: {"B": "face", "E": "edge", "J": "edge"}]   # Tier 2
│       └── position   [attr: {"B_1": (0.5, 0.0, 0.0), "B_2": (0.0, 0.5, 0.0), ...}]  # Tier 3 (ED-PIC)
│
├── normalization/                     # group: unit conversion metadata
│   ├── system        [attr: "PIC"]    # "PIC" | "MHD" | "SI" | "custom"
│   ├── length_ref    [attr: float64, meters]
│   ├── time_ref      [attr: float64, seconds]
│   ├── velocity_ref  [attr: float64, m/s]
│   ├── b_field_ref   [attr: float64, Tesla]
│   ├── e_field_ref   [attr: float64, V/m]
│   ├── density_ref   [attr: float64, m⁻³]
│   ├── mass_ref      [attr: float64, kg]
│   └── charge_ref    [attr: float64, C]
│
├── species/                           # group — one sub-group per declared [[species]] entry, in order
│   ├── s0/    {name: "electrons", charge: -1.0, mass: 1.0, ...}
│   ├── s1/    {name: "ions",       charge:  1.0, mass: 256.0, ...}
│   └── ...
│
├── physics/                           # group: model-agnostic flags
│   ├── relativistic  [attr: bool]
│   └── (model-specific sub-groups: pic/, mhd/, hybrid/, ...)
│
├── frame             [attr: "simulation"]   # native frame name
├── transforms/                        # optional group — one sub-group per coordinates.transforms entry
│   └── GSM/          {origin: (...), rotation: ((...), (...), (...)), scale: 1.0, from_frame: "GSE"}
│
├── time              [attr: float64, code units]
├── step              [attr: int]
├── model_name        [attr: "iPIC3D"]    # [model].name — code identity
├── model_type        [attr: "PIC"]       # [model].type — drives default vocabulary
└── schema/                               # group — schema discriminator (mirrors simulation.toml [schema])
    └── version       [attr: "1.0"]       # same value as Zarr §4.2 attrs.schema.version
```

Every file contains enough metadata to convert back to SI without the
original `simulation.toml` and to interpret per-species field names
(`n_s0`, `V_s1_1`, ...) without it.  The `schema/version` attr lets a
streaming consumer dispatch on layout vocabulary without sniffing the
rest of the file. The HDF5 file itself always uses numbered names;
`geometry` drives alias registration in the reader (`Bx → B_1` for
cartesian, `Br → B_1` for spherical, etc). Existing readers (iPIC3D,
BATSRUS, ...) translate native layouts; the Rust code writes this
layout directly.

Optional groups (`species/`, `transforms/`, `physics/<model>/`) are
omitted when the corresponding TOML section is absent — single-fluid
MHD with no transforms produces a smaller file with no `species/` or
`transforms/`.  Stagger Tiers 2 and 3 (`stagger/fields`, `stagger/
position`) are likewise omitted when the writer only knows Tier 1.

### 4.2 Zarr Output Layout

The native pypic write format.  `pypic.io.to_zarr` and
`to_zarr_timeseries` produce a Zarr v3 store laid out as an xarray
`DataTree`: the field arrays live under a `/fields` child group
(mirroring §4.1's `/fields/` HDF5 group); the metadata sections sit
as flat keys on the root group's attrs, with the schema discriminator
at the root attr path `schema.version` mirroring
`simulation.toml`'s `[schema].version` and discriminating both the
on-disk shape and the metadata vocabulary in one go.  Consolidated
metadata is enabled — readers go through `consolidated="auto"` for
the one-shot metadata fetch when the writer left a consolidated index,
and fall back to listing for non-consolidated stores.

```
my_store.zarr/                         # Zarr v3 group root
│
├── attrs (flat root metadata):
│   ├── schema:        { version: "1.0" }   # mirrors simulation.toml [schema].version
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
│   ├── run:           (optional) { name, doi, license, authors,
│   │                    git_sha, host, funding, embargo,
│   │                    resources, ensemble, ... }  # schema.Run.model_dump
│   ├── simulation_toml: (optional) "verbatim TOML text from [run] config"
│   └── metadata:      { ... reader-specific scalars,
│                         StaggerInfo as tagged dict ... }
│
└── fields/                             # /fields child group
    ├── B_1, B_2, B_3, ...                 # field arrays
    ├── E_1, E_2, E_3, ..., rho_c, rho_m, J_1, ..., u_1, u_2, u_3
    ├── x, y, z                         # 1-D coordinate arrays
    │                                   #   (axis names match geometry)
    └── time                            # only for multi-step writes
```

**Per-array attrs.**  Each field array under `/fields/` carries
xarray-style metadata: `long_name`, `units`, `quantity_type`,
`si_unit`, `latex`, and the openPMD-style `unit_dimension` 7-tuple
when the field has a registered SI dimension.

**`surviving_axes`.**  The integer tuple of grid axes still present
after any in-pypic slicing — `(0, 1, 2)` for a full 3D dataset,
`(0, 2)` for a `y`-slice produced by `PlaneSelection`, `(2,)` for
a 1-D line-out.  Indices are the original axis numbers (so a `y`-slice
preserves `0` and `2`, not `0` and `1`).  A reader can reconstruct
the full embedding by combining `surviving_axes` with `geometry.axis_names`.

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

| Aspect | §4.1 HDF5 | §4.2 Zarr |
|---|---|---|
| Field path | `/fields/B_1` | `/fields/B_1` |
| Grid metadata | `/grid/` group with attrs | root `attrs.grid` (JSON) |
| Stagger metadata | `/grid/stagger/` sub-group (`convention`, `fields`, `position`) | root `attrs.metadata` (StaggerInfo as tagged dict) |
| Normalization | `/normalization/` group | root `attrs.normalization` |
| Species | `/species/{s0,s1,...}/` sub-groups | root `attrs.species` (list) |
| Physics | `/physics/` group + sub-groups | root `attrs.physics` |
| Frame / transforms | top-level `frame` attr + `/transforms/<name>/` | root `attrs.frame` + `attrs.transforms` |
| Coord arrays | from grid attrs | `/fields/{x,y,z}` (1-D) |
| `time` / `step` | top-level scalar attrs | `time` dim (multi-step) / `metadata.time` scalar (single-step) |
| Files per write | one per timestep | one store, all steps |
| Schema version | `/schema/version` attr | root `attrs.schema.version` (mirrors `[schema].version`) |
| `[run]` provenance | not currently round-tripped (config-only) | root `attrs.run` (typed; `Run.model_dump`) |
| `simulation.toml` verbatim | not currently round-tripped | root `attrs.simulation_toml` (opaque UTF-8 text) |

`schema.version` is the on-disk layout discriminator. It carries the
**writing library's** layout version — pypic v1.0 stamps `"1.0"`,
pypic v1.1 will stamp `"1.1"`. Within a major version, it agrees
with `simulation.toml`'s `[schema].version` because v1.x library
and v1.x decks ship in lockstep. Readers reject stores whose
`schema.version` does not match the library's expected major
version. The nested root attr lets non-pypic consumers (Three.js
viewer, Rust `zarrs` pipelines) dispatch on
layout vocabulary without parsing the TOML.

**Reading without pypic.**  A non-pypic consumer (JS WebGPU viewer,
Rust `zarrs` pipeline) opens the root group, reads the section dicts
straight from `attrs.grid` / `attrs.normalization` / etc., and reads
field arrays from `/fields/<name>`.  No Python or pypic library
required.  Coordinate arrays under `/fields` make the data
self-describing in CF/COARDS terms.

**Field naming invariant.**  Stored arrays use the numbered
canonical names (`B_1`, `B_2`, `B_3`).  Geometry- and species-aliases
are read-time conveniences resolved by `FieldDataset`; they never
appear on disk.

**`attrs.run` — typed run provenance (optional).** When a
`FieldDataset` carries a `[run]` block (auto-populated by
`load_config()` from the source TOML), it is emitted as a top-level
root attr — not buried under `attrs.metadata` — so non-pypic
consumers (webpic, Rust pipelines) can read it without going
through pypic's loose metadata bag. The value is the JSON-mode dump
of `pypic.schema.Run` (`mode="json"` so dates land as ISO strings).
On read, `from_zarr` rebuilds the typed model via
`Run.model_validate(...)` and re-stuffs it into
`fds.metadata["run"]`. The block is **identity-stable across
derivations**: regrid / slice / frame-transform propagate the
original `run` unchanged, because a derivation of "MMS-event-1" is
still that run.

**`attrs.simulation_toml` — verbatim TOML text (optional).** When
the source `simulation.toml` is available (either via
`load_config()` capture or the `to_zarr(..., simulation_toml=PATH)`
writer kwarg), the whole TOML document is stamped as opaque UTF-8
text at the root group's `attrs.simulation_toml`. This carries the
sections that the typed `FieldDataset` boundary drops on the way to
Zarr — `[bodies]`, `[drivers]`, `[output.*]`, `[restart]`,
`[[probes]]`, `[[collisions]]`, `[phase_space]`, plus any `x-<code>`
extensions — losslessly, with zero per-section encoder maintenance.
Consumers that want a typed view re-validate via
`pypic.schema.validate_simulation_toml(text)`. Same derivation
policy as `attrs.run`: the original source TOML is propagated
unchanged through derived datasets (it describes the run that
produced the source, which is correct provenance for any
derivation); users republishing a derived dataset can drop the attr
explicitly.

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
