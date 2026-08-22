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
`simulation.toml` and the on-disk output stores (see §4 for the
on-disk attr path). v1.x is **additive only**: future releases may
add optional sections, optional keys, canonical field names, and
enum values, but will not rename or remove existing canonical names
and will not move required keys. (For reference, the v0 → v1.0
rename list: `omega_pe_over_omega_ce` → `omega_p_over_omega_c`.)
Anything stricter — e.g., a removal or rename — bumps
`[schema].version` to `2.0`.

**`x-*` extensions are unconstrained.** Code-specific knobs under
`x-<code>.*` (or unknown sub-tables under
`[physics.{pic,mhd,hybrid,vlasov}]`) are accepted by the validator
without enforcement; v1.x will not break them, but also makes no
promises about their portability.

**User-facing constraint: species ordering is part of the deck's
ABI.** Per-species canonical names (`_s0`, `_s1`, ...) bind to
`[[species]]` declaration order; reordering changes the meaning of
every per-species field on disk. Add new species at the end of the
list, not in the middle.

### Sections and extensions

A valid v1.0 document must declare these sections:

```toml
[schema]                 # version = "1.0" (REQUIRED)
[model]                  # code identity (name, type)
[run]                    # THIS run's identity + provenance
[time]                   # t_start, t_end, n_steps; dt (required unless scheme = "adaptive"); scheme (default "fixed")
[grid]                   # dimensions, spacing, lower/upper
[units]                  # normalization + reference values
[coordinates]            # geometry, frame
[[species]]              # ≥ 1 entry required (kinetic OR fluid species)
```

Optional sections, listed in the order §2 walks them:

```toml
[boundary_conditions]    # per-axis BC tags + per-field overrides
[physics]                # model-agnostic flags + .{pic,mhd,hybrid} sub-tables
[[bodies]]               # registry of physical objects (planets, coils, ...)
[initial_conditions]     # flat table: setup type + type-specific keys
[[drivers]]              # ongoing external coupling (magnetograms, SW inflow, ...)
[restart]                # continuation pointer
[output.checkpoints]     # lossless full-state dumps
[output.fields]          # field output cadence + quantities
[output.particles]       # particle output cadence + selection
[output.probes]          # probe time-series cadence
[output.diagnostics]     # on-the-fly derived quantities
[[output.streams]]       # multi-cadence / ROI output groups (repeatable)
[phase_space]            # >3D phase-space (gyrokinetic / continuum Vlasov)
[[collisions]]           # per-pair collision declarations
[[probes]]               # fixed or trajectory samplers
```

Non-portable code-specific knobs live under an `x-<code>.*`
namespace (e.g. `[x-warpx]`, `[physics.pic.x-ipic3d]`). Validators
accept any `x-*` or `x_*` key without validation. Unknown keys
under `[physics.{pic,mhd,hybrid,vlasov}]` and their `.solver`
sub-tables are also accepted — the solver vocabulary is explicitly
reserved for evolution in v1.1+.

### Validation rules

**Strict vs open string vocabularies.** Two flavours of
string-valued field appear in §2.

*Strict* (closed enum) — bounded structural / format primitives,
fixed by the data model itself: `[schema].version`, `[model].type`,
`[coordinates].geometry`, `[coordinates].physical_extent_unit`,
`[grid.amr].amr_kind`, `[grid.stagger].convention`,
`[grid.stagger].fields` location values, `[output.*].format`,
`[output.*].precision`, `[output.checkpoints].partition`,
`[output.streams.region].kind` (`"box"` | `"plane"`),
`[restart].mode`, `[restart].restore` array values,
`[[drivers]].coupling`, `[[drivers]].direction`,
`[[bodies]].shape`, `[phase_space].coordinate_system`,
`[run.resources.allocations].type`. Adding a value is a v1.x
schema bump. Numerical knobs (integers/floats like
`current_smoothing`, `field_substeps`) carry no vocabulary — the
open/strict distinction applies to string-valued fields only.

*Open* (canonical list, unenforced) — numerical-method, algorithm,
and closure vocabularies: `[time].scheme`, `[time].splitting`
(canonical: `"strang" | "lie" | "godunov"`),
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
still fire — openness applies to *value*, not *semantics*. Most
enum fields in §2 carry an inline canonical-values comment;
absent that, refer to the lists above.

**Foreign-key resolution.** Sections that declare named entities
(`[[species]]`, `[[bodies]]`, `[[drivers]]`) act as **registries**.
Orphan entries — a body or driver declared but never referenced
elsewhere — are accepted; the validator does not require every
registry entry to be used. References *into* a registry, however,
are validated strictly: `[[drivers]].body` → `[[bodies]].name`,
`[boundary_conditions].drivers_lower/upper` → `[[drivers]].name`,
`[[collisions]].species_pair` and `[output.particles].species` →
`[[species]].name`, `[units].reference_species` → `[[species]].name`
(with `"electrons"` / `"ions"` / `"protons"` as a builtin fallback).
Unresolved names raise a validation error at load time.

## 2. Section Specifications

### [schema]

Schema version tag and creation date.

```toml
[schema]
version = "1.0"                    # REQUIRED — see §1 Versioning
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
`simulation.toml` is loaded, the loader stamps the result onto
`SimulationConfig.run`, and the Zarr writer lifts it to root
`attrs.run` as a JSON-mode `Run.model_dump`. Identity-stable
across derivations — a regridded "MMS-event-1" run is still that
run. On-disk round-trip policy lives in §4 (Zarr round-trips
end-to-end; HDF5 has writer-side support for `/run/` but no reader
extraction yet — ship `simulation.toml` alongside HDF5 stores).

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
splitting      = "strang"          # optional, open string. Canonical:
                                   #   "strang" | "lie" | "godunov"
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

# Non-uniform per-axis cell widths (optional, sparse). Only stretched
# axes appear; uniform axes inherit `[grid].spacing`. Per-axis widths
# must sum to `upper - lower` within `sum_rtol` (default 1e-9).
[grid.stretched]
sum_rtol = 1.0e-9                  # optional — float drift tolerance
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

**Stagger layering.** Tier 1 is the semantic anchor. Tier 2 and
Tier 3 are optional augmentations that require Tier 1 to be
interpretable (a `position = [0.5, 0.0, 0.0]` is uninterpretable
until you know whether the base grid is cell-centered or
node-centered). The validator does not enforce this cross-tier
link, so writers that emit Tier 2/3 without Tier 1 produce a
parseable-but-meaningless stagger record. Readers that consume only
Tier 1 still work; readers that need ED-PIC-precise destagger reach
for Tier 3 (round-trips through `StaggerInfo.position`). Readers
always destagger to a co-located grid on load.

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
drivers_lower = { "2" = "solar_wind_inflow" }

# Per-field BC overrides (optional). Apply only to "E", "B", or
# "particles"; each entry is a full BoundaryConditions block matching
# the same axis count as the default.
[boundary_conditions.field_overrides.E]
lower = ["periodic", "periodic", "pml"]
upper = ["periodic", "periodic", "pml"]
# field_overrides.B and field_overrides.particles take the same shape.
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
`reference_species` resolves against `[[species]].name`; the
builtins `"electrons"` / `"ions"` / `"protons"` are accepted as a
fallback when no matching entry exists, so legacy decks parse
without forcing a rename.

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
# `speed_of_light` stays at top-level [units].speed_of_light
# regardless of approach; it never appears under [units.reference].
```

If `system` is "SI", all data is already in SI and no conversion is needed
(all reference values = 1.0).

**Input vs storage naming.** TOML inputs (Approach A's
`reference_*` keys, Approach B's bare keys under `[units.reference]`)
are canonicalized at read time into the eight `*_ref` storage
primitives surfaced in §4 (`length_ref`, `time_ref`, `velocity_ref`,
`b_field_ref`, `e_field_ref`, `density_ref`, `mass_ref`,
`charge_ref`). Approach A derives the missing primitives (e.g. PIC
inputs supply density/mass/charge plus `speed_of_light`; time and
velocity references are derived from those); Approach B supplies
each primitive directly. The `*_ref` form is the single name a
non-pypic reader of the on-disk store needs to know.

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

**Scale vs shrink factor.** `scale` converts code units to target
units. `shrink_factor` (stored in `metadata["scaling"]
["shrink_factor"]`) is a physics diagnostic — the ratio of effective
scale to the scale implied by normalization. `> 1` means the domain
is compressed relative to kinetic scales (common in reduced-mass-
ratio PIC). Intensive quantities are unaffected; extensive
integrals and transit times scale with it.

**Frame transforms** (optional): define how to convert from the native
frame to other reference frames. Frame names are arbitrary strings —
no hardcoded knowledge of any specific frame.

```toml
[coordinates.transforms.TARGET_FRAME]
origin = [0.0, 0.0, 0.0]          # translation (code units or physical)
rotation = [[...], [...], [...]]   # 3x3 rotation matrix (optional, orthonormal, det = ±1)
scale = 1.0                        # length scale factor (optional, auto-computed from physical_extent)
from_frame = "string"              # for chaining: transform from this frame instead of native
```

Transforms can chain: if transform A goes from "simulation" to "GSM" and
transform B goes from "GSM" to "GSE", requesting "GSE" from "simulation"
data chains both automatically.

**Chain resolution.** Resolving a target frame is a BFS from
`[coordinates].frame` over the `from_frame → frame_name` graph,
returning the shortest path. Ties break on TOML declaration order
(first wins). Cycles raise a validation error at apply time.

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
`collisional`, `radiative`, `[physics.vlasov]`,
`[physics.gyrokinetic]`). Code-specific knobs go under
`[physics.<model>.x-<code>.*]`.

Canonical values per key (all open strings; see §1 *Validation rules*):

| Key | Canonical values (open string) |
|---|---|
| `[physics.pic.solver].scheme` | `explicit` \| `semi-implicit` \| `implicit` |
| `[physics.pic.solver].pusher` | `boris` \| `vay` \| `higuera-cary` \| `llrk4` \| `free-streaming` |
| `[physics.pic.solver].field_solver` | `fdtd-yee` \| `pseudo-spectral` \| `psatd` \| `spectral-azimuthal` \| `lehe` \| `ck` \| `ckc` \| `pstd` \| `gpstd` \| `implicit-moment` \| `implicit-gmres` |
| `[physics.pic.solver].charge_correction` | `marder` \| `langdon` \| `boris` \| `hyperbolic` \| `spectral` \| `none` |
| `[physics.pic.solver].current_deposition` | `esirkepov` \| `zigzag` \| `villabune` \| `direct-boris` \| `direct-morse-nielson` \| `none` |
| `[physics.mhd.solver].scheme` | `fct` \| `godunov` \| `muscl-hancock` \| `ppm` \| `weno` |
| `[physics.mhd.solver].reconstruction` | `linear` \| `plm` \| `ppm` \| `weno5` \| `mp5` |
| `[physics.mhd.solver].limiter` | `zalesak` \| `minmod` \| `mc` \| `van-leer` \| `superbee` |
| `[physics.mhd.solver].divergence_cleaning` | `ct` \| `powell` \| `dedner-glm` \| `projection` \| `none` |
| `[physics.mhd.solver].riemann` | `roe` \| `hll` \| `hlle` \| `hlld` \| `lax-friedrichs` |
| `[physics.hybrid.solver].scheme` | `predictor-corrector` \| `current-advance-method` |
| `[physics.hybrid.solver].field_pusher` | `cyclic-leapfrog` \| `implicit` |
| `*.preconditioner` (any model, implicit solvers) | `none` \| `jacobi` \| `block-jacobi` \| `ilu` \| `amg` \| `additive-schwarz` |

```toml
[physics]
relativistic = false               # semantics identical across PIC/MHD/hybrid

[physics.pic]
omega_p_over_omega_c = 20.0        # reference-species plasma/cyclotron ratio
                                   #   — applies to [units].reference_species

[physics.pic.solver]
scheme             = "semi-implicit"
implicitness       = 0.5           # θ-scheme weight (0.5 = Crank-Nicolson)
pusher             = "boris"
field_solver       = "implicit-moment"
preconditioner     = "block-jacobi"
current_smoothing  = 0             # optional — binomial filter passes per step
charge_smoothing   = 0             # optional — binomial filter passes per step
charge_correction  = "none"        # optional
current_deposition = "esirkepov"   # optional

[physics.mhd]
gamma_eos   = 1.6667               # adiabatic index ($c_p / c_v$);
                                   #   disambiguates from canonical `gamma_L`
                                   #   (bulk Lorentz factor)
resistivity = 0.0
hall_term   = false

[physics.mhd.solver]
scheme              = "fct"
reconstruction      = "linear"
limiter             = "zalesak"
divergence_cleaning = "ct"
preconditioner      = "ilu"        # optional — implicit MHD only
# riemann           = "hlld"       # Godunov family

[physics.hybrid]
# Top-level [physics.hybrid] keys reserved for v1.1+; hybrid
# fluid-species properties live on the corresponding [[species]]
# entry. The .solver sub-table below follows the standard solver
# vocabulary.

[physics.hybrid.solver]
scheme            = "predictor-corrector"
field_pusher      = "cyclic-leapfrog"
resistivity       = 5.0e-4
hyper_resistivity = 1.0e-6
current_smoothing = 2
preconditioner    = "block-jacobi" # optional — for field_pusher = "implicit" only
```

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
                                          #   (see §3 Canonical Field Names → Electromagnetic fields)
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

Ongoing external input to this run — data file, coupled model, or
live service (magnetograms, solar-wind inflows, pickup-ion sources,
surface absorption, MHD→PIC volume coupling, magnetosphere↔ionosphere
two-way coupling, ...). Repeatable.

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
body          = "mercury"          # optional. Without a body, the driver
                                   #   falls through to target_lower/upper
                                   #   or the whole domain. Drivers do not
                                   #   require a body. When present, must
                                   #   match a declared [[bodies]].name —
                                   #   unresolved names raise a validation
                                   #   error.
```

When the upstream is a named *model* (not just a flat data file),
add a ``[drivers.model]`` sub-table to record its identity:

```toml
[[drivers]]
name      = "ionosphere_potentials"
type      = "model_coupling"
coupling  = "boundary"
direction = "two_way"             # M-I coupling: FACs down, potentials up

[drivers.model]
name        = "RIM"
type        = "ionosphere_potential_solver"   # open vocabulary
url         = "../rim_run/simulation.toml"    # any pointer; pypic-aware peer
description = "Ridley Ionosphere Model; returns potential & conductance"
# url = "https://aerospace.gov/rim/"          # alternative: homepage for non-pypic peer
```

``[drivers.model]`` carries four fields: ``name`` (required), ``type``
(required, open string — covers ``"MHD"``, ``"PIC"``,
``"ionosphere_potential_solver"``, ``"magnetic_field_extrapolation"``,
``"fluid_atmosphere"``, ``"fusion_transport"``, ...), ``url``
(optional, opaque), and ``description`` (optional). Extra keys
(``version``, ``doi``, ``git_sha``, code-specific knobs) pass through
unvalidated. ``source`` and ``[drivers.model]`` can co-exist when a
model run was checkpointed to a data file — the driver records both
the data path and the model attribution.

`coupling` and `direction` are orthogonal. Target precedence:
1. `body` — defaults to that body's bounding box.
2. `target_lower` / `target_upper` — explicit box; narrows `body` when both present.
3. Neither — whole domain.

**BC ↔ driver interaction.** For `coupling = "boundary"`, the
driver supplies values at the face named in
`boundary_conditions.drivers_lower/upper` (BC tag is the category,
driver is the supplier); `target_lower/upper` and `body` further
narrow within that face. Volume / source / sink couplings ignore
the face entry and use `target_lower/upper` or `body` for their
domain.

**Two-way semantics.** An entry describes inputs *into this run*.
For two-way coupling, the peer run owns its own ``simulation.toml``
with its own ``[[drivers]]`` entry pointing back — each side
records its half of the exchange independently. ``url`` is a single
opaque pointer: when the peer is pypic-aware, point at its
``simulation.toml``; otherwise point at the homepage, a DOI, or
whatever metadata file is best available. The schema does not
resolve ``url`` — format detection (TOML re-validation, JSON parse,
plain text) is a consumer concern. A non-standard metadata file
is better than no link at all.

Driver-type-specific keys (`production_rate`, `fields`, etc.) are
accepted beyond the core vocabulary above.

### [restart]

Continuation pointer from a prior run. `from` is a single file, a
directory of per-rank checkpoints, a glob pattern, or an explicit
list of per-rank files (for relocated reruns where filenames drift
from the source code's convention). Resolution to one or many
files happens at read time.

```toml
[restart]
from = "./checkpoints/chk_000030.h5"  # REQUIRED — string OR list of strings
step = 30000                           # optional
time = 1500.0                          # optional
restore = ["fields", "particles"]      # optional — "fields" | "particles" |
                                       #   "auxiliary"; distinct. Omit to restore all.
mode = "hot"                           # optional — "hot" (full state reload) |
                                       #   "cold" (re-apply IC on saved geometry)
```

### [output.*]

Output cadences, quantities, directories, and precisions.
Five fixed sub-sections plus the repeatable `[[output.streams]]`
array (six total); each is optional.

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

[output.fields]                        # field output (PIC example;
                                       #   MHD substitutes rho_m / V — see
                                       #   pypic.simulation.toml Scenario B)
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
| 3 | z | φ | z | φ |

The table names **vector components** (the subscripts on `B_i`,
`E_i`, `V_i`, …). For Cartesian data, letter aliases (`Bx`, `By`,
`Bz`) read the same arrays; for non-Cartesian geometries the reader
registers geometry-appropriate aliases (`Br` → `B_1` for spherical,
etc.). Particle position and velocity columns (`x`, `y`, `z`, `vx`,
`vy`, `vz`) are always Cartesian regardless of
`[coordinates].geometry` — geometry is a field-grid representation
choice, not a particle one (see *Per-particle data columns* and
§4.3).

**Alias-on-disk policy.** Writers MUST emit only the numbered
canonical form (`B_1` / `B_2` / `B_3`, `B0_1` / `B0_2` / `B0_3`,
`n_s0`, `V_s1_1`, `P_s0_11`, …). Aliases are read-time conveniences
derived from `[coordinates].geometry` after the file is opened.
Storing both `B_1` and `Bx` for the same data is not part of the
v1.0 contract; readers that encounter both treat the numbered form
as authoritative.

For `geometry = "thetaMode"` (FBPIC azimuthal-mode RZ
decomposition), the spatial grid is two-dimensional `(r, z)` and
the third *array* dimension is the openPMD mode number from
`[coordinates.modes].mode_indices`; the third *vector* component
is still the azimuthal field (`B_3 = B_φ`). Arrays have shape
`(n_r, n_z, n_modes)` with complex dtype per-mode, not separate
real/imaginary axes.

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
(`rho_c_s0`, `J_s1_1`, `n_s3`); the species suffix comes before the
component index (`J_s0_1`, `EF_s1_2`). The species *name* lives in
`[[species]]`, not in the field name.

**Species-name aliases:** For any species, the canonical
`<prefix>_s<index>` form has an automatically-generated
`<prefix>_<species_name>` alias when the species name is declared
in `[[species]]`. `n_s0` becomes `n_electrons` when
`species[0].name == "electrons"`; `EF_s1_1` becomes `EF_protons_1`
when `species[1].name == "protons"` (the species token replaces
`_s<N>`; the trailing component index, if any, stays at the end).

For multi-species runs (H⁺ + He²⁺ + O⁺), the species-name form is
the unambiguous way to reference per-species quantities — the
integer index depends on declaration order.

The same expansion rules apply to `[output.fields].quantities` —
listing `"B"` writes `B_1`, `B_2`, `B_3`; listing `"P_s1"` writes the
six second-species pressure tensor components.

> **pypic-only.** Vector group shorthand in `read()`: passing a bare
> prefix like `"B"` to `read(fields=...)` expands to `B_1, B_2, B_3`;
> `"EF_s0"` expands to `EF_s0_1, EF_s0_2, EF_s0_3`; derived
> quantities expand to their dependencies (`"Pij_s1"` loads the six
> ion tensor components; `"P_par"` loads the six total-pressure
> components plus `B_1..B_3`). `"P_s1"` (per-species *scalar*
> pressure) is distinct from `"Pij_s1"` (per-species tensor group) —
> the tensor identifier always carries the `ij` infix. Names that
> are already resolved single components (`"Bx"`, `"Br"`, `"E_phi"`)
> or whose prefix ends in a digit (`"B_1"`, `"P_s1_11"`) pass through
> as scalars; request the full tensor via `"Pij_s0"` / `"Pij_s1"` if
> a later `compute()` needs it. The species-name alias is registered
> at `FieldDataset` construction time and only when the underlying
> canonical is present, so missing data raises `KeyError` rather
> than misdirecting.

> **pypic-only.** Library-side convenience aliases for the common
> two-species electron/ion case (`Pe ↔ P_s0`, `n_i ↔ n_s1`,
> `beta_e ↔ beta_s0`, …) — see [Aliases](aliases.md). Not part of
> the cross-tool contract; non-pypic consumers (Rust, JS) only need
> the canonical names listed here.

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

`EHF` carries a total-fluid form (`EHF_1, EHF_2, EHF_3`) because
single-fluid MHD treats enthalpy flux as a primary quantity; the
other flux decompositions (`EF`, `KEF`, `HF`, `q`) appear only
per-species — sum over species at read time when a total is
needed. Volumetric scalars (`e_th`, `e_th_trace`, `e_k`) carry
both forms.

### Pressure tensor

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `P_par` | Pressure parallel to B | PIC, multi-moment MHD (derived from tensor; CGL closures may store directly — see Storage tiers below) |
| `P_perp` | Pressure perpendicular to B | PIC, multi-moment MHD (derived from tensor; CGL closures may store directly — see Storage tiers below) |
| `Pij`, `Pij_s{N}` | Full pressure tensor — 6 independent components (`P_11`, `P_12`, `P_13`, `P_22`, `P_23`, `P_33`) total or per species (`P_s0_11`, `P_s0_12`, …, `P_s0_33`) | PIC, multi-moment MHD |
| `agyrotropy` | Swisdak $Q$ — see [equations.md § Pressure Tensor](equations.md#4-pressure-tensor-and-field-aligned-decomposition) for the closed form $Q = 1 - 4 I_2 / [(I_1 - P_\parallel)(I_1 + 3 P_\parallel)]$ with full-tensor invariants, bounded $[0, 1]$ | PIC, multi-moment MHD (derived) |
| `D_ng` | Aunai's degree of nongyrotropy $D_{ng} = 2\,\|\mathbf{N}\|_F / \mathrm{Tr}(\mathbf{P})$ where $\mathbf{N}$ is the non-gyrotropic part of $\mathbf{P}$. Frame-invariant; alternative to $Q$. | PIC, multi-moment MHD (derived) |
| `A_phi` | Scudder's electron agyrotropy — perpendicular-block eigenvalue ratio, bounded $[0, 1]$. Captures only perp anisotropy (misses off-axis nongyrotropy). | PIC, multi-moment MHD (derived) |

Per-species decomposition (`P_s0_par`, `P_s0_perp`) uses the
per-species tensors (`P_s0_11..P_s0_33`); two-species shorthands
`P_par_e/i` and `P_perp_e/i` are registered as user-facing aliases.
Listing `Pij` (or per-species `Pij_s{N}`) in
`[output.fields].quantities` auto-expands to the six components.

**Storage tiers** (cross-tool readers MUST accept either):

- *Preferred* — store the six tensor components (`Pij`,
  `Pij_s{N}`). `P` / `P_par` / `P_perp` / `agyrotropy` derive on
  demand and stay $\hat{b}$-fresh under frame transforms.
- *Acceptable* — store only `P_par` and `P_perp` (BATSRUS
  `MhdAnisoP`, anisotropic GRMHD). Tensor-derived quantities are
  then unavailable and the stored values lock to the snapshot's
  $\hat{b}$.

### Field-aligned vector decomposition

Projects a vector $\mathbf{A}$ onto $\hat{b} = \mathbf{B}/|\mathbf{B}|$.
All entries are derived-on-demand (never written to disk) and return
NaN where $|B| = 0$.  Frame-transform behavior matches the *Acceptable*
storage tier above: values are $\hat{b}$-locked to the snapshot, so a
frame rotation re-expresses the perpendicular components in the new
basis.  See [conventions.md § Field-Aligned
Decomposition](conventions.md#field-aligned-decomposition) for the sign
convention and reference-vector choice.

| Canonical | Meaning | Computed from |
|-----------|---------|---------------|
| `J_par` | Field-aligned current density $J_\parallel = \mathbf{J}\cdot\hat{b}$ | `J_1`, `J_2`, `J_3`, `B_1-B_3` |
| `J_perp_1`, `J_perp_2`, `J_perp_3` | Perpendicular current density $\mathbf{J}_\perp = \mathbf{J} - J_\parallel\hat{b}$ | `J_1-J_3`, `B_1-B_3` |
| `\|J_perp\|` | Perpendicular current density magnitude | `J_1-J_3`, `B_1-B_3` |
| `V_par` | Field-aligned bulk velocity | `V_1-V_3`, `B_1-B_3` |
| `V_perp_1`, `V_perp_2`, `V_perp_3` | Perpendicular bulk velocity | `V_1-V_3`, `B_1-B_3` |
| `\|V_perp\|` | Perpendicular bulk velocity magnitude | `V_1-V_3`, `B_1-B_3` |
| `V_s{N}_par` | Per-species parallel velocity | `V_s{N}_1-V_s{N}_3`, `B_1-B_3` |
| `V_s{N}_perp_1`, `V_s{N}_perp_2`, `V_s{N}_perp_3` | Per-species perpendicular velocity | `V_s{N}_1-V_s{N}_3`, `B_1-B_3` |
| `\|V_s{N}_perp\|` | Per-species perpendicular velocity magnitude | `V_s{N}_1-V_s{N}_3`, `B_1-B_3` |
| `E_par` | Field-aligned electric field | `E_1-E_3`, `B_1-B_3` |
| `E_perp_1`, `E_perp_2`, `E_perp_3` | Perpendicular electric field | `E_1-E_3`, `B_1-B_3` |
| `\|E_perp\|` | Perpendicular electric field magnitude | `E_1-E_3`, `B_1-B_3` |
| `E_prime_par` | Field-aligned non-ideal residual $E'_\parallel = (\mathbf{E}+\mathbf{V}\times\mathbf{B})\cdot\hat{b}$ — the canonical reconnection-rate diagnostic | `E_1-E_3`, `V_1-V_3`, `B_1-B_3` |
| `E_prime_perp_1`, `E_prime_perp_2`, `E_prime_perp_3` | Perpendicular non-ideal residual | `E_1-E_3`, `V_1-V_3`, `B_1-B_3` |
| `\|E_prime_perp\|` | Perpendicular non-ideal residual magnitude | `E_1-E_3`, `V_1-V_3`, `B_1-B_3` |

`E_ideal_*` ($-\mathbf{V}\times\mathbf{B}$) and `E_Hall_*`
($\mathbf{J}\times\mathbf{B}/(n_e |q_e|)$) are registered for
symmetry with the `E_par` / `E_prime_par` family: same `_par`,
`_perp_{1,2,3}`, `|*_perp|` shape. The `_par` components are
analytically zero (both vectors are cross products with
$\mathbf{B}$, hence orthogonal to $\hat{b}$) and the `_perp`
components equal the full vector up to roundoff; useful only as a
numerical-precision check on the destaggered cross-product
implementation. Inputs: `V_1-V_3, B_1-B_3` for `E_ideal_*`;
`J_1-J_3, B_1-B_3, n_s0` for `E_Hall_*`.

Two-species shorthand aliases `V_par_e ↔ V_s0_par`, `V_par_i ↔
V_s1_par` are registered (mirrors `P_par_e`/`P_par_i`). Vector-group
expansion in `read()` resolves `"V_perp"` → `V_perp_1, V_perp_2,
V_perp_3` and `"V_s0_perp"` → `V_s0_perp_1, V_s0_perp_2, V_s0_perp_3`.

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

The `<field>_s<N>` form is the canonical recipe ID; NRL Formulary
spellings (`omega_pe`, `lambda_D`, ...) resolve as aliases.
Multi-species runs (`omega_p_s2`, `lambda_D_s3`, ...) synthesize on
demand.

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
| `D_e` | — | Electron-frame dissipation (Zenitani EDR localizer) | J_1-J_3, E_1-E_3, V_s0_1-V_s0_3, B_1-B_3, rho_c |
| `R_recon` | — | Local dimensionless reconnection rate $|\mathbf{E}'|/(v_A|B|)$ | E_1-E_3, V_1-V_3, B_1-B_3, v_A |
| `E_prime_1`, `E_prime_2`, `E_prime_3` | `E_prime_x`, `E_prime_y`, `E_prime_z` | Non-ideal electric field | E_1-E_3, V_1-V_3, B_1-B_3 |
| `E_ideal_1`, `E_ideal_2`, `E_ideal_3` | `E_ideal_x`, `E_ideal_y`, `E_ideal_z` | Ideal electric field | V_1-V_3, B_1-B_3 |
| `E_Hall_1`, `E_Hall_2`, `E_Hall_3` | `E_Hall_x`, `E_Hall_y`, `E_Hall_z` | Hall electric field | J_1-J_3, B_1-B_3, `n_s0`, species |
| `psi` | — | Magnetic flux function (2D) | B_2, grid |
| `firehose` | — | Firehose instability parameter | `P_par`, `P_perp`, `\|B\|` |
| `mirror` | — | Mirror instability parameter | `P_par`, `P_perp`, `\|B\|` |

The 3D Schindler reconnection criterion $\Xi(\mathbf{x}_0) = \int_{\mathcal{L}} E_\parallel\,d\ell$ returns one scalar per seed point and lives in `pypic.reconnection.schindler_xi`, outside the per-cell `compute()` registry. See [equations.md § 9](equations.md#9-reconnection-and-anisotropy-diagnostics).

Scalar quantities (`n_s0`, `rho_m`, `P`, `T_s0`, `beta`, ...) use
the same name regardless of geometry.

### Field-line map quantities

Outputs of `pypic.maps` (Step 44g — planned). Derived from
**field-line tracing** rather than per-cell pointwise computation,
so they live on a 2D *footpoint* / *seed* grid $(\theta_0, \phi_0)$
or an arbitrary slice surface — not the simulation cell grid.
Stored as scalar arrays in the canonical Zarr/HDF5 layout (§4.1 /
§4.2) once `pypic.maps` ships; consumers reading these names from
disk should expect a 2D mapping array, not a 3D field.

| Canonical | Description | Producer |
|-----------|-------------|----------|
| `Q_map` | Squashing factor $Q$ on a footpoint grid (Titov-Démoulin) | `pypic.maps.squashing_factor` |
| `connectivity_map` | Open/closed/disconnected classification (small-int enum) | `pypic.maps.classify_connectivity` |
| `length_map` | Total arc length of the traced field line | `pypic.maps.field_line_length` |
| `r_map_f`, `t_map_f`, `p_map_f` | Forward (outward) footpoint coordinates | `pypic.maps.footpoint_map` |
| `r_map_b`, `t_map_b`, `p_map_b` | Backward (inward) footpoint coordinates | `pypic.maps.footpoint_map` |

The footpoint maps are the raw output of field-line tracing;
everything else in the literature (signed-log $Q$, MapFL's $K$-factor
$\log_{10}|B_{r,0}/B_{r,1}|$, flux-tube expansion factor
$(B_{r,0}\,r_0^2)/(B_{r,1}\,r_1^2)$, magnetic-dip counts) derives
from these names plus the simulation $B$ field, so `pypic.maps`
exposes those as Python functions returning numpy arrays without
booking new canonical on-disk names.

The producer is **analysis, not simulation**: rustpic / webpic /
foreign readers do not emit these names; pypic writes them after
the fact, typically into a dedicated Zarr store alongside the
simulation output. Map outputs carry an `attrs["map"]` provenance
block recording the tracer settings (`integrator`,
`step_control`, `atol`/`rtol` or `over_rc`, `interpolation`),
the seed grid definition (`r0`, `t0_range`, `p0_range`, ...), and
the finite-difference $h$ used for $Q$ — sufficient for
reproducibility across re-runs. Tracer + interpolator vocabulary
(integrator scheme, step-control mode, periodic-axes flags) is
populated by `pypic.numerics` and `pypic.traces` at write time;
no `simulation.toml` section is needed because maps are a
post-processing artefact, not a simulation control parameter.

### Per-particle data columns

For PIC and hybrid codes, `ParticleData` carries one canonical
per-macroparticle representation regardless of the source code's
native layout: per-particle `weight` plus scalar `species_charge`
and `species_mass`. Per-macroparticle charge and mass for moments
and equations of motion are derived on demand
(`ParticleData.macro_charge`, `macro_mass`).

| Field | Storage | Meaning |
|-------|---------|---------|
| `x`, `y`, `z` | per-particle `(N,)` | Position in 3D ambient space — always Cartesian, even when `[coordinates].geometry` is non-Cartesian (geometry is a field-grid representation choice, not a particle one) |
| `vx`, `vy`, `vz` | per-particle `(N,)` | 3-velocity $v^i$ in code units — **not** momentum $p^i = \gamma m v^i$ and **not** four-velocity $u^i = \gamma v^i$. Relativistic readers that emit momentum or four-velocity convert at load time using `species_mass` and the per-particle Lorentz factor. |
| `weight` | per-particle `(N,)` float64 | Physical particles per macroparticle $w$ |
| `id` | per-particle `(N,)` int64 | Tracking ID (when the code emits one) |
| `species_charge` | scalar (metadata) | Per-species charge $q_s$ in code units. One charge state per `[[species]]` entry — mixed ionization → separate species. |
| `species_mass` | scalar (metadata) | Per-species mass $m_s$ in code units |

Position/velocity dtype is reader/writer choice (see §4.3 down-cast
knobs); `weight` and `id` are stored at full source precision.

Scalars round-trip through the Arrow/Parquet schema metadata (no
per-particle storage cost).

**Native layouts (reader concern).** PIC codes split into two camps
for storing macroparticle charge on disk; readers translate to the
canonical form so downstream consumers don't see the difference.

- **Combined** (iPIC3D, OSIRIS): on-disk field is $q_s w$. Readers
  split it into `weight` ($|q_s w|/|q_s|$) plus scalars from config.
- **Separate** (VPIC, WarpX, Smilei, EPOCH, PIConGPU, TRISTAN-MP):
  on-disk field is already `weight`; scalars come from config.

`read → write → read` reconstructs $q_s w$ bit-exactly via float64
arithmetic but does not preserve combined-layout disk bytes — pypic
is analysis, not restart regeneration.

**openPMD record → canonical mapping.** The four most-used codes in
the *Separate* camp (WarpX, PIConGPU, Smilei, FBPIC) emit the
openPMD ED-PIC layout, where the Phase-9 reader (TASKS Step 42)
translates each record into the canonical form above:

| openPMD record | Canonical destination |
|----------------|-----------------------|
| `particles/<sp>/position/{x,y,z}` + `positionOffset/{x,y,z}` | `x`, `y`, `z` (sum of the two, applying `unitSI`) |
| `particles/<sp>/momentum/{x,y,z}` | `vx`, `vy`, `vz` (divide by $\gamma m$, where $\gamma = \sqrt{1 + p^2/(m^2 c^2)}$) |
| `particles/<sp>/weighting` | `weight` (honor `macroWeighted` + `weightingPower` per ED-PIC) |
| `particles/<sp>/charge` (constant record) | scalar `species_charge` |
| `particles/<sp>/mass` (constant record) | scalar `species_mass` |
| `particles/<sp>/id` | `id` |

Codes that emit non-constant `charge` or `mass` records (varying
ionization state within one species) must be split into one canonical
`[[species]]` per charge state — there is no per-particle `charge` or
`mass` column in `ParticleData` to land them.

## 4. Output Layouts

Three on-disk layouts are defined, with deliberately asymmetric roles:

- **§4.1 HDF5** is the canonical **read contract** for HDF5-emitting
  codes — the Rust simulation code, iPIC3D, BATSRUS, ARMS, OpenGGCM —
  and what every file-based reader in `pypic.readers` translates
  *into*. pypic itself does not write this layout; there is no
  `to_hdf5` in `pypic.io`.
- **§4.2 Zarr** is what pypic itself produces for field data via
  `pypic.io.to_zarr` / `to_zarr_timeseries`. Fields under `/fields`,
  metadata as flat keys on the root group's attrs.
- **§4.3 Parquet** is what pypic produces for particle data via
  `pypic.io._parquet`. Hive-partitioned by step and species, with
  Morton-ordered rows for spatial pushdown.

§4.1 and §4.2 mirror §2 section names: each top-level group / attrs
key corresponds to a §2 section of the same name. They both carry
`[schema].version` at `/schema/version` (HDF5) or `attrs.schema.version`
(Zarr) and use the same canonical field names. The difference is the
storage container's group convention (HDF5 groups vs Zarr DataTree).

### 4.1 HDF5 Output Layout

The cross-tool **read contract** for HDF5-emitting codes. pypic does
not write this layout — its readers translate native HDF5 (and IDL,
H5hut, Fortran-binary, ...) sources *into* a `FieldDataset` that
conforms to the same canonical naming and metadata shape described
here. The Rust simulation code writes this layout directly, so the
section doubles as its output specification.

Each timestep is a separate file. Field datasets use the
**numbered canonical names** (`B_1`, `B_2`, `B_3`), which are
geometry-agnostic. Per the §4.2 mapping table, every §2 section
becomes a top-level `/<section>/` group whose attrs carry that
section's TOML keys verbatim; optional sections are omitted when
their TOML counterpart is absent.

```
output_{step:06d}.h5            # example name; actual pattern is set
│                               # by [output.checkpoints].file_pattern
├── fields/                     # group: field arrays (numbered canonical names)
│   ├── B_1, B_2, B_3              # dtype set by [output.fields].precision,
│   │                              #   shape (n1, n2, n3)
│   ├── E_1, E_2, E_3
│   ├── J_1, J_2, J_3
│   ├── rho_c                   # PIC: charge density (omit for MHD-only output)
│   ├── rho_m                   # MHD: mass density   (PIC may also write derived)
│   ├── n_s0, n_s1, ...
│   ├── V_s0_1, V_s0_2, V_s0_3, ...
│   ├── P_s0_11, P_s0_12, ..., P_s0_33    # per-species pressure tensor
│   │                                     #   (or P_11..P_33 for single-fluid MHD)
│   └── u_1, u_2, u_3                     # optional — four-velocity (rel. PIC)
│
├── schema/, model/, time/, grid/ (with stagger/ sub-group),
├── boundary_conditions/ (with optional field_overrides/ sub-group),
├── coordinates/ (with optional modes/ for thetaMode and transforms/
│   sub-groups), normalization/, species/{s0,s1,...}/, physics/
│   (with optional pic/ | mhd/ | hybrid/ model sub-groups),
├── run/                        # optional — writer-side contract;
│                               #   current pypic readers do not yet
│                               #   round-trip it into SimulationConfig.run
│
├── time              [attr: float64]   # current snapshot time
└── step              [attr: int]       # current snapshot step
```

**Sections not emitted to HDF5.** The §2 sections that don't survive
the FieldDataset boundary (`[bodies]`, `[drivers]`,
`[initial_conditions]`, `[output.*]`, `[restart]`, `[[collisions]]`,
`[phase_space]`, `[[probes]]`, `x-<code>` extensions) are not written
to HDF5 by any pypic-aware writer; cross-tool writers that need them
should ship `simulation.toml` alongside the HDF5 store.

Every emitted file carries enough metadata to convert back to SI
without the original `simulation.toml` and to interpret per-species
field names (`n_s0`, `V_s1_1`, ...) without it. The HDF5 file always
uses numbered names; `coordinates/geometry` drives alias registration
in the reader (`Bx → B_1` for cartesian, `Br → B_1` for spherical).
The top-level snapshot scalars `time` and `step` are always present.

### 4.2 Zarr Output Layout

The native pypic write format. `pypic.io.to_zarr` and
`to_zarr_timeseries` produce a Zarr v3 store as an xarray
`DataTree`: field arrays under `/fields`, metadata sections as
flat keys on the root group's attrs. Consolidated metadata is
enabled — readers use `consolidated="auto"` and fall back to
listing for non-consolidated stores.

Each top-level `attrs` key mirrors a §2 TOML section of the same
name. Sub-keys mirror that section's TOML keys.

```
my_store.zarr/                         # Zarr v3 group root
│
├── attrs (flat root metadata, one key per §2 section):
│   ├── schema, model, time, grid, boundary_conditions,
│   ├── coordinates, normalization, species, physics
│   ├── run                  # optional — typed Run.model_dump
│   ├── simulation_toml      # optional — verbatim TOML text
│   └── metadata             # snapshot scalars (time, step, ...);
│                            #   single-step writes only — multi-step
│                            #   writes promote `time` to a dimension
│                            #   under /fields. See mapping table below
│                            #   for sub-key contents of each section.
│
└── fields/                             # /fields child group
    ├── B_1, B_2, B_3, ...                 # field arrays
    ├── E_1, E_2, E_3, ..., rho_c, rho_m, J_1, ..., u_1, u_2, u_3
    ├── x, y, z                         # 1-D dimension coords; names track
    │                                   #   grid.surviving_axis_names
    └── time                            # only for multi-step writes
```

**Per-array metadata.** Each field array under `/fields/` carries
xarray-style attrs: `long_name`, `units`, `quantity_type`,
`si_unit`, `latex`, and the openPMD-style `unit_dimension` 7-tuple
when the field has a registered SI dimension. Arrays produced by
`pypic.reductions.reduce` additionally carry a `reduction` entry
recording `{axis, op, result_kind?, weight?, length_axes?}` for
provenance; the dict is absent on unreduced fields. Mirrors the
`attrs["map"]` provenance block that `pypic.maps` outputs carry (§3
*Field-line map quantities*). Default per-variable codec:
`BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")`.
`dtype="float32"` downcasts on write; user-supplied `encoding=`
overrides per variable.

**Slicing and append semantics.** `attrs.grid.surviving_axes` is
the integer tuple of original grid axes still present after any
in-pypic slicing — `(0, 1, 2)` for a full 3D dataset, `(0, 2)`
for a `y`-slice produced by `PlaneSelection`, `(2,)` for a 1-D
line-out. A reader reconstructs the full embedding by combining
`surviving_axes` with `attrs.coordinates.axis_labels`.

`to_zarr_timeseries` extends every field array with a leading
`time` dimension, shape `(nt, n1, n2, n3)`, chunked so reading
one timestep is O(1). Step 1 writes the `DataTree` with
`mode="w"` (stamps root attrs); steps 2..N append directly to
`/fields` via `mode="a", append_dim="time"`. The writer enforces
a constant field set and constant identity attrs (`grid`,
`normalization`, `species`, `physics`, plus `frame` and
`transforms` from `coordinates`) across appends; mismatches
raise. Other root attrs (`schema`, `model`, `time`,
`boundary_conditions`, `coordinates.*` outside frame/transforms,
`run`) flow through the per-step `metadata` intersection — keys
present in every step with identical values survive; any key
missing from one step, or differing in value, is dropped from
`attrs.metadata` after the final append. In single-step writes
the snapshot `time` lives in `attrs.metadata.time` (no `time`
dimension exists).

**Mapping to §4.1 (HDF5).** Both layouts mirror §2 section names;
the difference is the storage container's group convention (HDF5
groups vs Zarr root attrs).

| §2 section | §4.1 HDF5 | §4.2 Zarr |
|---|---|---|
| Field arrays | `/fields/<name>` | `/fields/<name>` |
| `[schema]` | `/schema/version` attr | `attrs.schema.version` |
| `[model]` | `/model/` group with attrs | `attrs.model` |
| `[time]` | `/time/` group with attrs | `attrs.time` |
| `[grid]` (without `[grid.stagger]`) | `/grid/` group with attrs | `attrs.grid` |
| `[grid.stagger]` | `/grid/stagger/` sub-group (`convention`, `fields`, `position`) | `attrs.grid.stagger` |
| `[boundary_conditions]` | `/boundary_conditions/` group + optional `field_overrides/` sub-group | `attrs.boundary_conditions` |
| `[coordinates]` (geometry, frame, axis_labels, physical_extent, modes) | `/coordinates/` group | `attrs.coordinates` |
| `[coordinates.transforms]` | `/coordinates/transforms/<name>/` sub-groups | `attrs.coordinates.transforms` |
| `[units]` | `/normalization/` group | `attrs.normalization` |
| `[[species]]` | `/species/{s0,s1,...}/` sub-groups | `attrs.species` (list, in declaration order) |
| `[physics]` | `/physics/` group + model sub-groups | `attrs.physics` |
| `[run]` (optional) | `/run/` group with attrs (writer-side contract; pypic readers do not yet round-trip it) | `attrs.run` (typed; `Run.model_dump`; round-trips end-to-end) |
| Verbatim `simulation.toml` | not currently emitted by pypic and not extracted by readers | `attrs.simulation_toml` (opaque UTF-8 text; round-trips end-to-end) |
| Coord arrays | from `/grid/` + `/coordinates/` attrs | xarray dim coords under `/fields/` (names track `grid.surviving_axis_names`) |
| Snapshot scalars `time` / `step` | top-level scalar attrs | `time`: dim under `/fields/time` (multi-step) or `attrs.metadata.time` scalar (single-step). `step`: `attrs.metadata.step` scalar in both modes. |
| Files per write | one per timestep | one store, all steps |

**Optional root attrs.**

- `attrs.run` — typed run provenance. JSON dump of
  `pypic.schema.Run` (`mode="json"`, ISO dates), surfaced as a
  top-level root attr so non-pypic consumers don't have to dig
  through `attrs.metadata`. **Identity-stable across derivations**:
  regrid / slice / frame-transform propagate the original `run`
  unchanged (a derivation of "MMS-event-1" is still that run).
- `attrs.simulation_toml` — verbatim TOML. Opaque UTF-8 text of the
  source `simulation.toml`, stamped when available via
  `load_config()` capture or the `to_zarr(..., simulation_toml=PATH)`
  kwarg. Carries the sections the typed `FieldDataset` boundary drops
  (`[bodies]`, `[drivers]`, `[output.*]`, `[restart]`, `[[probes]]`,
  `[[collisions]]`, `[phase_space]`, `x-<code>` extensions)
  losslessly. Consumers that want a typed view re-validate via
  `pypic.schema.validate_simulation_toml(text)`. Same derivation
  policy as `attrs.run`.

Readers reject stores whose `schema.version` major component doesn't
match the library's expected major. Non-pypic consumers (Three.js
viewer, Rust `zarrs` pipelines) read section dicts straight from
`attrs.*` and field arrays from `/fields/<name>` — no TOML parsing
required.

### 4.3 Particle Layout (Parquet)

Per-particle data lives in a separate Hive-partitioned Parquet
dataset, distinct from the field-grid Zarr/HDF5 stores in §4.1/§4.2.
Particles and fields are co-located in a parent directory but written
through different code paths (`pypic.io._parquet` for particles,
`pypic.io._zarr` for fields).

```
my_run/                                # parent directory (convention, not schema)
├── fields.zarr/                       # §4.2 Zarr store (or output_NNNNNN.h5 per §4.1)
└── particles/                         # Parquet dataset root
    ├── step=000000/
    │   ├── species=electrons/
    │   │   └── part-00000.parquet
    │   └── species=ions/
    │       └── part-00000.parquet
    ├── step=000200/
    │   ├── species=electrons/...
    │   └── species=ions/...
    └── ...
```

**Per-particle columns** (one Arrow column each, see §3 Per-particle
data columns): `x`, `y`, `z`, `vx`, `vy`, `vz`, `weight`, `id?`.
Position columns are always Cartesian regardless of the field grid's
`[coordinates].geometry`.

**Scalar species metadata** (Arrow schema metadata, no per-particle
storage cost): `species_charge`, `species_mass`. Round-trip through
`particles_to_arrow` / `particles_from_arrow` preserves them.

**Partitioning, ordering, and compression.** Hive partitions on
`step=NNNNNN` and `species=NAME` enable partition pruning for time
and species queries. Within each partition file, particles are sorted
by Morton (Z-order) curve over `(x, y, z)` so Parquet row-group
min/max statistics enable spatial predicate pushdown. Default codec:
zstd with shuffle pre-filter (level 1 for processing, level 3 for
archival). Row groups: 500K–1M particles each. Down-cast knobs:
`position_dtype="float32"`, `velocity_dtype="float32"`. `weight` and
`id` stay at full source precision (float64 / int64); scalar
`species_charge` and `species_mass` round-trip through Arrow schema
metadata at full precision. See TASKS.md Step 25 (Arrow/Parquet
foundation) and Step 25b (canonical `weight` + species-scalar form)
for the full I/O contract.

## 5. Extensibility

- **New simulation code:** Write a reader that maps native output to `FieldDataset` with canonical field names. No schema changes needed.
- **New field:** Add the name to the canonical table (this document), add to relevant readers, add derived functions if applicable.
- **New derived map output:** Add the name to the *Field-line map quantities* table in §3, write the producer in `pypic.maps`, persist via `pypic.io.to_zarr`. Map outputs are 2D arrays on a footpoint grid, not 3D fields — schema accommodates either without changing the layout contract.
- **New numerical method:** Land integrators / step controllers / interpolators in `pypic.numerics` (one shared kernel package — the adaptive tracer is the current consumer; future particle pushers and the Step 44 symplectic / periodic-tricubic / curvature-step paths land in the same module). The vocabulary surfaces as free-form strings in the map `attrs["map"]` provenance block.
- **New model type:** Add a `[physics.NEW_TYPE]` subsection convention, document expected fields and species.
- **New output format:** Define the layout mapping, write a reader. Everything downstream works unchanged via `FieldDataset`.
- **Code-specific knobs:** Park them under an `x-<code>` namespace — see §1 *Sections and extensions*.

## 6. What This Schema Does NOT Define

- **Simulation control parameters** (solver tolerances, MPI decomposition). Time-series essentials (`dt`, `n_steps`) live in `[time]`; output cadence (`step_interval`) lives in `[output.*]`.
- **Visualization settings** (colormaps, camera angles, slice positions).
- **Exhaustive physics parameter lists.** The `[physics]` section is open-ended by design.
- **Native file layouts** of existing codes. Readers handle the translation.
- **Internal data structures** of any project. Each project maps to/from the schema at its boundaries.
- **API response format.** Defined in the Starlette/FastAPI/viewer project, not here.
