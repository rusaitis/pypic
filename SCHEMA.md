# SCHEMA.md — Shared Data Contract

## Purpose

This document defines the shared data format and configuration structure
used across the plasma simulation platform:

- **Python analysis toolkit** — reads this format, computes derived quantities
- **Rust plasma code** — writes this format as simulation output
- **Three.js viewer** — consumes this format via the FastAPI server

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
scaling_factor = 10.0              # optional: documents how the system is scaled (no effect on computation)
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
```

**Frame transforms** (optional): define how to convert from the native
frame to other reference frames. Frame names are arbitrary strings —
no hardcoded knowledge of any specific frame.

```toml
[coordinates.transforms.TARGET_FRAME]
type = "string"                    # "affine" | "rotation"
origin = [0.0, 0.0, 0.0]          # translation (code units or physical)
rotation = [[...], [...], [...]]   # 3x3 rotation matrix (optional)
scale = 1.0                        # length scale factor (optional)
from_frame = "string"              # for chaining: transform from this frame instead of native
parameter = "string"               # for time-dependent transforms: named parameter
```

Transforms can chain: if transform A goes from "simulation" to "GSM" and
transform B goes from "GSM" to "GSE", requesting "GSE" from "simulation"
data chains both automatically.

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
`docs/conventions.md`).

### [physics]

Open-ended section for model-specific parameters. Only the subsection
matching `[model] type` is expected to be present.

```toml
[physics.pic]                      # PIC-specific parameters
omega_pe_over_omega_ce = 3.0       # frequency ratio (sets B relative to density)
theta = 0.5                        # implicitness parameter (0.5 = Crank-Nicolson)
speed_of_light = 1.0               # normalized c (always 1.0 in standard PIC normalization)
# ... any other PIC parameters the code needs to document

[physics.mhd]                      # MHD-specific parameters
gamma = 1.6667                     # adiabatic index
resistivity = 0.0                  # η (0 = ideal)
hall_term = false                  # include Hall physics?
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

---

## 3. Canonical Field Names

All projects use the same field names. Readers are responsible for
mapping simulation-code-specific names to these canonical names.

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
| `E1`, `E2`, `E3` | `Ex`, `Ey`, `Ez` | Electric field components | PIC (MHD: derived) |

### Fluid / moment quantities — densities

| Canonical | Meaning | Units (normalized) | Present in |
|-----------|---------|-------------------|------------|
| `n_s0`, `n_s1`, ... | Number density per species | $n / n_{ref}$ | PIC, multi-fluid MHD |
| `rho_c` | Total charge density | $\sum_s n_s q_s$ | PIC |
| `rho_m` | Total mass density | $\sum_s n_s m_s$ | MHD, PIC (derived) |

`rho_c` and `rho_m` are unambiguous — no overloaded `rho`. For the common
two-species case, `n_e` and `n_i` are accepted as aliases for `n_s0` and
`n_s1`.

**Per-species naming:** Append `_s` plus the species index (0-based):
`rho_c_s0`, `J1_s1`, `V1_s2`, `n_s3`. The species *name* lives in the
`[[species]]` table, not in the field name.

### Fluid / moment quantities — velocities, currents, pressure

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `J1`, `J2`, `J3` | `Jx`, `Jy`, `Jz` | Current density $\mathbf{J} = \sum_s n_s q_s \mathbf{V}_s$ | PIC (deposited), MHD (∇×B) |
| `V1`, `V2`, `V3` | `Vx`, `Vy`, `Vz` | Ion bulk velocity (PIC: 1st moment of $f_i$) / fluid velocity (MHD) | PIC (moments), MHD |
| `Ve1`, `Ve2`, `Ve3` | `Vex`, `Vey`, `Vez` | Electron bulk velocity (PIC: 1st moment of $f_e$) | PIC |
| `P` | — | Total scalar pressure: MHD fluid pressure; PIC: $P_e + P_i$ | MHD, PIC (moments) |
| `Pe` | — | Electron scalar pressure $P_e = n_e T_e$ | PIC, two-fluid MHD |
| `Pi` | — | Ion scalar pressure $P_i = n_i T_i$ | PIC, two-fluid MHD |
| `Te` | — | Electron temperature (energy units: $T_e = P_e / n_e$) | PIC, two-fluid MHD |
| `Ti` | — | Ion temperature (energy units: $T_i = P_i / n_i$) | PIC, two-fluid MHD |

**Temperature convention:** Temperatures are in **energy units** throughout
($T = P/n$, not $T = P/(nk_B)$). To convert to Kelvin, divide by $k_B$.

### Thermodynamic quantities

| Canonical | Meaning | Definition | Present in |
|-----------|---------|-----------|------------|
| `h` | Specific enthalpy (ideal gas) | $h = \gamma P / ((\gamma - 1) \rho_m)$ | MHD |
| `h_rel` | Relativistic specific enthalpy | $h = c^2 + \gamma P / ((\gamma - 1) \rho_m)$ (constant-$\Gamma$ approx.) | Rel. MHD |
| `s` | Specific entropy (isotropic) | $s = \ln(P / \rho_m^\gamma)$ (MHD) or $\ln(P / n^\gamma)$ (PIC) | MHD, PIC |
| `s_e` | Electron entropy | $s_e = \ln(P_e / n_e^\gamma)$ | PIC |
| `s_i` | Ion entropy | $s_i = \ln(P_i / n_i^\gamma)$ | PIC |
| `s_gyro` | Gyrotropic entropy | $s = \ln(P_\parallel P_\perp^2 / n^5)$ | PIC (anisotropic) |
| `e_int` | Specific internal energy | $e_{int} = P / ((\gamma - 1) \rho_m)$ | MHD |
| `gamma_eos` | Adiabatic index | $\gamma = c_p / c_v$ | MHD (from physics config) |

PIC entropy uses $\gamma = 5/3$ (3D) by default. The gyrotropic exponent
of 5 is independent of $\gamma$ — it comes from the CGL invariants.
See `docs/conventions.md` for full derivation.

### Energy and flux quantities

| Canonical | Cartesian alias | Meaning | Present in |
|-----------|----------------|---------|------------|
| `S1`, `S2`, `S3` | `Sx`, `Sy`, `Sz` | Poynting flux $\mathbf{S} = \mathbf{E} \times \mathbf{B} / \mu_0$ | PIC, MHD |
| `e_B` | — | Magnetic energy density $B^2 / (2\mu_0)$ | PIC, MHD |
| `e_E` | — | Electric energy density $\epsilon_0 E^2 / 2$ | PIC |
| `e_k` | — | Kinetic energy density $\frac{1}{2}\rho_m V^2$ | MHD, PIC (moments) |
| `e_th` | — | Thermal energy density $P / (\gamma - 1)$ | MHD, PIC (moments) |

### Pressure tensor (PIC, anisotropic)

| Canonical | Meaning | Present in |
|-----------|---------|------------|
| `P_par` | Pressure parallel to B: $P_\parallel = \hat{b} \cdot \mathbf{P} \cdot \hat{b}$ | PIC (from tensor) |
| `P_perp` | Pressure perpendicular to B: $P_\perp = (Tr(\mathbf{P}) - P_\parallel) / 2$ | PIC (from tensor) |
| `Pij` | Full pressure tensor (6 independent components: P11, P12, P13, P22, P23, P33) | PIC |
| `agyrotropy` | Agyrotropy measure $Q$ (deviation from gyrotropic symmetry) | PIC (derived) |

### Characteristic scales (derived)

| Canonical | Meaning | Computed from |
|-----------|---------|---------------|
| `d_e` | Electron skin depth $c/\omega_{pe}$ | `n_e`, species |
| `d_i` | Ion skin depth $c/\omega_{pi}$ | `n_i`, species |
| `r_e` | Electron thermal gyroradius $v_{th,e}/\omega_{ce}$ | `Te`, `|B|`, species |
| `r_i` | Ion thermal gyroradius $v_{th,i}/\omega_{ci}$ | `Ti`, `|B|`, species |
| `omega_pe` | Electron plasma frequency $\sqrt{n_e e^2 / (\epsilon_0 m_e)}$ | `n_e`, species |
| `omega_pi` | Ion plasma frequency $\sqrt{n_i Z^2 e^2 / (\epsilon_0 m_i)}$ | `n_i`, species |
| `omega_ce` | Electron cyclotron frequency $\|e\|B / m_e$ (positive by convention) | `|B|`, species |
| `omega_ci` | Ion cyclotron frequency $ZeB / m_i$ (positive by convention) | `|B|`, species |
| `lambda_D` | Electron Debye length $\sqrt{\epsilon_0 T_e / (n_e e^2)}$ | `n_e`, `Te`, species |
| `v_A` | Alfvén speed $B/\sqrt{\mu_0 \rho_m}$ | `|B|`, `rho_m` |
| `v_th_e` | Electron thermal speed $\sqrt{T_e / m_e}$ (NRL convention) | `Te`, species |
| `v_th_i` | Ion thermal speed $\sqrt{T_i / m_i}$ (NRL convention) | `Ti`, species |
| `c_s` | Sound speed $\sqrt{\gamma P / \rho_m}$ (MHD; differs from ion acoustic $c_{ia}$) | `P`, `rho_m`, `gamma_eos` |
| `v_ms` | Fast magnetosonic speed $\sqrt{v_A^2 + c_s^2}$ (perpendicular propagation) | `v_A`, `c_s` |
| `M_A` | Alfvén Mach number $V / v_A$ | `|V|`, `v_A` |
| `M_ms` | Magnetosonic Mach number $V / v_{ms}$ | `|V|`, `v_ms` |
| `beta` | Plasma beta $2\mu_0 P / B^2$ | `P`, `|B|` |
| `beta_e` | Electron beta $2\mu_0 P_e / B^2$ | `Pe`, `|B|` |
| `beta_i` | Ion beta $2\mu_0 P_i / B^2$ | `Pi`, `|B|` |

Convention details (thermal speed, cyclotron frequency, sound speed vs
ion acoustic speed, magnetosonic dispersion) are in `docs/conventions.md`.

### Other derived quantities

| Canonical | Cartesian alias | Meaning | Computed from |
|-----------|----------------|---------|---------------|
| `\|B\|` | — | Magnetic field magnitude | B1, B2, B3 |
| `\|E\|` | — | Electric field magnitude | E1, E2, E3 |
| `\|J\|` | — | Current density magnitude | J1, J2, J3 |
| `\|V\|` | — | Bulk velocity magnitude | V1, V2, V3 |
| `div_B` | — | Divergence of B (should be ~0) | B1, B2, B3, grid |
| `div_E` | — | Divergence of E ($= \rho_c / \epsilon_0$ by Gauss's law) | E1, E2, E3, grid |
| `curl_B1`, `curl_B2`, `curl_B3` | `curl_Bx`, ... | Curl of B ($\propto \mathbf{J}$ in MHD) | B1, B2, B3, grid |
| `vort1`, `vort2`, `vort3` | `vort_x`, ... | Fluid vorticity $(\nabla \times \mathbf{V})$ | V1, V2, V3, grid |
| `\|vort\|` | — | Vorticity magnitude | vort1, vort2, vort3 |

Derived quantities are computed on demand in normalized (code) units —
physical constants vanish ($\mu_0 = 1$, $\epsilon_0 = 1$ in PIC). SI
conversion happens only at the output/visualization boundary.

Scalar quantities (`n_s0`, `rho_m`, `P`, `Te`, `beta`, ...) use the same
name regardless of geometry.

See CLAUDE.md for naming conventions (functions, parameters, class names).

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
│
├── grid/                         # group: grid metadata
│   ├── dimensions  [attr: (n1, n2, n3)]
│   ├── spacing     [attr: (d1, d2, d3)]
│   ├── origin      [attr: (min1, min2, min3)]
│   ├── dt          [attr: float64, code units]
│   ├── boundary    [attr: ("periodic", "periodic", "periodic")]
│   └── geometry    [attr: "cartesian"]  # determines index interpretation
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
- **API response format.** Defined in the FastAPI/viewer project, not here.
