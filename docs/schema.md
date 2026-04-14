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

## 1. Configuration File: `simulation.toml`

A TOML file that travels with the simulation output. Describes what the
data contains, how it's normalized, and what coordinate system it uses.

### Required sections

Every `simulation.toml` must have these sections:

```toml
[model]           # What produced this data
[grid]            # Grid dimensions and spacing
[units]           # How to convert between code units and SI
[coordinates]     # Coordinate geometry and reference frame
```

### Optional sections

```toml
[[species]]           # Particle species (PIC, hybrid) — repeatable
[physics]             # Model-specific physics parameters — open-ended
[initial_conditions]  # Initial field/plasma configuration — open-ended
[output]              # What fields are in the output files
[[probes]]            # Virtual probes / spacecraft — repeatable
```

---

## 2. Section Specifications

### [model]

Identifies the simulation code and model type.

```toml
[model]
name = "string"           # REQUIRED: code name ("iPIC3D", "plasma-sim", "BATSRUS", ...)
type = "string"           # REQUIRED: "PIC" | "MHD" | "hybrid" — selects normalization defaults
version = "string"        # optional: code version
description = "string"    # optional: human-readable run description
```

### [grid]

Describes the computational grid. All values are in **code units**.

```toml
[grid]
dimensions = [nx, ny, nz]          # REQUIRED: number of cells
spacing = [dx, dy, dz]             # REQUIRED: cell size in code units
origin = [x_min, y_min, z_min]     # optional, default [0, 0, 0]
dt = 0.0                           # optional: timestep in code units (essential for time-series analysis)
boundary = ["periodic", "periodic", "periodic"]  # optional: boundary type per axis
                                   #   "periodic" | "reflecting" | "conducting" | "open"
stagger = "cell"                   # optional: "cell" | "node" | "staggered"
                                   #   "cell" = all fields at cell centers (default)
                                   #   "node" = all fields at cell vertices (e.g. iPIC3D)
                                   #   "staggered" = Yee mesh (B faces, E edges, etc.)
                                   #   Readers destagger to co-located grid on load.
                                   #   Informational only — FieldDataset is always co-located.
```

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

For **ion-normalized PIC** (e.g., iPIC3D large-scale runs):

```toml
[units]
system = "PIC"
reference_species = "ions"
reference_density = 1.0e18         # m⁻³ (ion number density)
reference_mass = 1.673e-27         # kg (proton mass)
reference_charge = 1.602e-19       # C
```

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
length = 5.31e-3                   # meters
time = 1.77e-11                    # seconds
velocity = 2.998e8                 # m/s
b_field = 1.07e-3                  # Tesla
e_field = 3.21e5                   # V/m
density = 1.0e18                   # m⁻³
mass = 9.109e-31                   # kg
charge = 1.602e-19                 # C
```

If `system` is "SI", all data is already in SI and no conversion is needed
(all reference values = 1.0).

### [coordinates]

Describes the coordinate geometry and reference frame.

```toml
[coordinates]
geometry = "string"                # REQUIRED: "cartesian" | "spherical" | "cylindrical"
frame = "string"                   # REQUIRED: native frame name (arbitrary, e.g., "simulation")
axis_labels = ["x", "y", "z"]     # optional: override default axis names
physical_extent = [46.0, 32.0, 13.0]  # optional: domain size in target-frame physical units
physical_extent_unit = "R_E"       # optional: unit for physical_extent (default "m")
                                   #   valid: "m", "km", "R_E", "R_S", "AU"
```

When ``physical_extent`` is provided, the ``scale`` field of any
transform with the default ``scale=1.0`` is auto-computed as
``physical_extent / grid_extent`` (after accounting for rotation).
Can also be passed at runtime via
``open_simulation(path, physical_extent=..., physical_extent_unit=...)``.

**Scale vs shrink factor.** The ``scale`` is a coordinate conversion
factor: how many target units (e.g., R_E) per code unit (e.g., d_i).
The ``shrink_factor`` is a physics diagnostic: ratio of the effective
scale to the physical scale implied by the normalization. A shrink
factor of 1.0 means the grid faithfully represents physical distances.
A value > 1 (e.g., 3.5) means the physical domain has been compressed
relative to kinetic scales — common in PIC simulations with reduced
mass ratio or MHD-coupled boundaries. The shrink factor is logged
automatically and stored in ``metadata["scaling"]["shrink_factor"]``.
Shrinking affects only length-dependent quantities: intensive
per-point values (fields, densities, β, Mach numbers) are unchanged,
but extensive integrals over physical volume or area (total energy,
magnetic flux) scale as shrink² or shrink³. Global transit times
are compressed by the shrink factor (correct velocity, shorter path).

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

### [[species]]

Describes particle species (PIC and hybrid models) or fluid species
(multi-fluid MHD). Repeatable — any number of species.

```toml
[[species]]
name = "string"                    # REQUIRED: "electrons", "ions", "alpha", ...
charge = 0.0                       # in code units (required unless charge_to_mass is given)
mass = 0.0                         # in code units (required unless charge_to_mass is given)
charge_to_mass = 0.0               # optional: q/m ratio (alternative to separate charge/mass)
particles_per_cell = [5, 5, 1]     # PIC/hybrid: per-cell count [x, y, z] or scalar; 0 for MHD
temperature = 0.0                  # optional: scalar isotropic temperature (code units)
thermal_velocity = [0.0, 0.0, 0.0] # optional: anisotropic thermal velocities (takes precedence)
drift_velocity = [0.0, 0.0, 0.0]  # optional: initial bulk drift (code units)
density = 1.0                      # optional: initial number density (code units)
```

Each species must have either (`charge` + `mass`) or `charge_to_mass`.
Relationship: $v_{th} = \sqrt{T/m}$ (see thermal speed convention in
[Conventions](conventions.md)).

### [physics]

Open-ended section for model-specific parameters. Only the subsection
matching `[model] type` is expected to be present.

```toml
[physics.pic]                      # PIC-specific parameters
omega_pe_over_omega_ce = 3.0       # frequency ratio (sets B relative to density)
theta = 0.5                        # implicitness parameter (0.5 = Crank-Nicolson)
speed_of_light = 1.0               # normalized c (always 1.0 in standard PIC normalization)
relativistic = false               # true if the code solved relativistic equations;
                                   # derived quantities use relativistic formulas when set
# ... any other PIC parameters the code needs to document

[physics.mhd]                      # MHD-specific parameters
gamma = 1.6667                     # adiabatic index
resistivity = 0.0                  # η (0 = ideal)
hall_term = false                  # include Hall physics?
relativistic = false               # true for relativistic MHD (RMHD)
# ... any other MHD parameters
```

### [initial_conditions]

Optional section documenting the initial field and plasma configuration.
Open-ended by design; common entries shown below.

```toml
[initial_conditions]
type = "double_harris"             # human-readable label for the setup
B0 = [0.097, 0.0, 0.0]            # initial/asymptotic magnetic field (code units)

[initial_conditions.parameters]    # setup-specific parameters (open-ended)
perturbation_amplitude = 0.4
current_sheet_thickness = 0.25     # half-thickness of current sheet (code units)
```

### [output]

Optional metadata about what's in the output files.

```toml
[output]
format = "HDF5"                    # "HDF5" | "NetCDF" | "Arrow"
fields = ["B1", "B2", "B3", "E1", "E2", "E3", "rho_c", "J1", "J2", "J3"]
particle_data = false              # whether particle positions/velocities are saved
```

PIC outputs typically include `rho_c` (charge density); MHD outputs
include `rho_m` (mass density). Both may include `n_e`, `n_i`.

See `examples/ipic3d_double_harris.toml` for a complete mapping from
iPIC3D input to this schema.

### [[probes]]

Virtual probes (detectors, spacecraft) that sample fields at specific
locations. Either **fixed** (constant position) or a **trajectory**
(position varies with time). Repeatable.

```toml
[[probes]]
name = "magnetopause_monitor"      # REQUIRED: human-readable label
position = [10.0, 0.0, 0.0]       # fixed: [x, y, z] in code units
fields = ["B1", "B2", "B3", "beta"]  # optional: fields to sample (default: all)

[[probes]]
name = "MMS1"                      # trajectory (virtual spacecraft)
trajectory = "mms1_orbit.csv"      # CSV columns: t, x, y, z (code units)
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

**Per-species naming:** Append `_s` plus the species index (0-based):
`rho_c_s0`, `J1_s1`, `V1_s2`, `n_s3`. The species *name* lives in the
`[[species]]` table, not in the field name. For vector fields, the
component index comes before the species suffix: `J1_s0`, `EF2_s1`.

**Electron/ion convenience aliases:** For the common two-species case
(species 0 = electrons, species 1 = ions), short `e`/`i` suffixed names
are accepted as aliases for the canonical `_s0`/`_s1` forms:

| Alias | Canonical | Meaning |
|-------|-----------|---------|
| `n_e`, `n_i` | `n_s0`, `n_s1` | Number density |
| `Pe`, `Pi` | `P_s0`, `P_s1` | Scalar pressure (or Tr(tensor)/3) |
| `Te`, `Ti` | `T_s0`, `T_s1` | Temperature |
| `Ve1`..`Ve3` | `V1_s0`..`V3_s0` | Electron bulk velocity |
| `Vi1`..`Vi3` | `V1_s1`..`V3_s1` | Ion bulk velocity |
| `EFe`, `EFi` | `EF_s0`, `EF_s1` | Energy flux (vector group) |
| `s_e`, `s_i` | — | Per-species entropy |
| `beta_e`, `beta_i` | — | Per-species plasma beta |

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
| `\|V\|_s0`, `\|V\|_s1`, ... | `\|Vi\|` | Velocity magnitude (per-species) | PIC (derived) |

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
| `E_prime_1/2/3` | — | Non-ideal electric field | E1-E3, V1-V3, B1-B3 |
| `E_ideal_1/2/3` | — | Ideal electric field | V1-V3, B1-B3 |
| `E_Hall_1/2/3` | — | Hall electric field | J1-J3, B1-B3, n\_s0, species |
| `psi` | — | Magnetic flux function (2D) | B2, grid |
| `firehose` | — | Firehose instability parameter | P\_par, P\_perp, \|B\| |
| `mirror` | — | Mirror instability parameter | P\_par, P\_perp, \|B\| |

Scalar quantities (`n_s0`, `rho_m`, `P`, `Te`, `beta`, ...) use the same
name regardless of geometry.

See CLAUDE.md for naming conventions (functions, parameters, class names).

### Per-particle data columns

For PIC and hybrid codes that emit per-particle data, `ParticleData`
exposes the following canonical fields (used by the Arrow/Parquet
I/O layer in `pypic.io`).  All fields are **optional** — readers
populate whatever the source format provides, and the
:attr:`ParticleData.macro_charge` / :attr:`macro_mass` properties
give per-macroparticle quantities regardless of which raw fields are
populated.

| Field | Storage | Meaning |
|-------|---------|---------|
| `x`, `y`, `z` | per-particle (`(N,)` float) | Particle position components in the simulation Cartesian frame |
| `vx`, `vy`, `vz` | per-particle | Particle velocity components |
| `charge` | per-particle (`(N,)` float64) | Macroparticle charge `q_s × w`. Populated by combined-storage codes (iPIC3D, OSIRIS); `None` for separate-storage codes |
| `weight` | per-particle (`(N,)` float64) | Number of physical particles per macroparticle. Populated directly by separate-storage codes; derived from `\|charge\|/\|species_charge\|` for combined-storage codes |
| `id` | per-particle (`(N,)` int64) | Integer tracking ID (when the code emits one) |
| `species_charge` | scalar (metadata) | Per-species charge in code units (e.g. ±1 for electrons/ions in iPIC3D normalization) |
| `species_mass` | scalar (metadata) | Per-species mass in code units |

The scalar `species_charge` and `species_mass` round-trip through the
Arrow/Parquet schema metadata (no per-particle storage cost).

#### Charge–weight conventions across PIC codes

PIC codes split into two camps for how they store macroparticle charge
`q_macro = q_s × w`:

- **Combined** (iPIC3D, OSIRIS): per-particle field is `q_macro`.  Weight
  is implicit — pypic readers derive it as `\|charge\|/\|species_charge\|`.
  Both `charge` and `weight` are populated.
- **Separate** (VPIC, WarpX, Smilei, EPOCH, PIConGPU, TRISTAN-MP):
  per-particle field is `weight`; species charge is a scalar from the
  run config.  `weight`, `species_charge`, and `species_mass` are
  populated; per-particle `charge` is left `None` to avoid wasting
  ~8 GB at billion-particle scale.

#### Computing per-particle quantities

For code-agnostic analysis, use the derived properties:

- `pcl.macro_charge` — returns per-particle `charge` if loaded, else
  computes `species_charge × weight`.  Used for current density
  `J = Σ q v` and charge density `ρ_c = Σ q`.
- `pcl.macro_mass` — returns `species_mass × weight`.  Used for
  kinetic energy `KE = ½ m v²`, mass density `ρ_m = Σ m`, and
  physical particle counts `N_phys = Σ w`.

In non-uniform-weight runs (particle splitting/merging, non-uniform
initial densities), each macroparticle's `weight` is unique and can
serve as a particle tracking ID — pass `id_column="weight"` to
`particles_from_dataset` to filter by it.

---

## 4. HDF5 Output Layout

The standard layout for simulation output files. Each timestep is a
separate file (or group within a file). Field datasets use the
**numbered canonical names** (`B1`, `B2`, `B3`), which are
geometry-agnostic.

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
original `simulation.toml`. When a reader sees `geometry = "cartesian"`,
it registers `Bx → B1`, `By → B2`, `Bz → B3` as aliases; for
`geometry = "spherical"`, `Br → B1`, `Btheta → B2`, etc. The HDF5
file itself always uses numbered names.

Each reader (iPIC3D, BATSRUS, Rust code) maps its native file layout
to this canonical layout. The Rust simulation writes this layout directly.

---

## 5. Extensibility

- **New simulation code:** Write a reader that maps native output to `FieldDataset` with canonical field names. No schema changes needed.
- **New field:** Add the name to the canonical table (this document), add to relevant readers, add derived functions if applicable.
- **New model type:** Add a `[physics.NEW_TYPE]` subsection convention, document expected fields and species.
- **New output format:** Define the layout mapping, write a reader. Everything downstream works unchanged via `FieldDataset`.

---

## 6. What This Schema Does NOT Define

- **Simulation control parameters** (timestep count, solver tolerances, MPI decomposition, output intervals). Note: `dt` is in `[grid]` because it's essential for time-series analysis.
- **Visualization settings** (colormaps, camera angles, slice positions).
- **Exhaustive physics parameter lists.** The `[physics]` section is open-ended by design.
- **Native file layouts** of existing codes. Readers handle the translation.
- **Internal data structures** of any project. Each project maps to/from the schema at its boundaries.
- **API response format.** Defined in the Starlette/FastAPI/viewer project, not here.
