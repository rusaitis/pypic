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
type = "string"           # REQUIRED: "PIC" | "MHD" | "hybrid"
version = "string"        # optional: code version
description = "string"    # optional: human-readable run description
```

`type` determines which optional sections and field names are expected.
The analysis tools use this to select appropriate defaults (e.g., PIC
normalization vs MHD normalization).

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

For non-uniform or AMR grids (e.g., BATSRUS), the reader is responsible
for regridding to a uniform grid. The `simulation.toml` describes the
regridded output, not the native AMR structure.

### [units]

Defines the normalization, allowing conversion between code units and SI.
Two approaches: specify the normalization system and let the consumer
derive reference values, or specify reference values directly.

**Approach A: named normalization (preferred for PIC/MHD)**

```toml
[units]
system = "PIC"                     # "PIC" | "MHD" | "SI" | "custom"

# PIC normalization: derive all reference values from a reference species.
# The reference species is typically electrons (standard) or ions
# (common in iPIC3D for large-scale simulations).
reference_species = "electrons"    # optional: "electrons" (default) | "ions" | species name
reference_density = 1.0e18         # m⁻³ (number density of reference species)
reference_mass = 9.109e-31         # kg (optional, default: electron mass)
reference_charge = 1.602e-19       # C (optional, default: elementary charge)
speed_of_light = 2.998e8           # m/s (optional, default scipy.constants.c)
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

The normalization derives: ω_ref = √(n_ref q_ref² / (ε₀ m_ref)),
d_ref = c / ω_ref, and all other reference values. In the resulting
normalized units: c = 1, m_ref = 1, q_ref = 1. The mass of the
*other* species appears as the mass ratio (e.g., m_e/m_i < 1 for
ion-normalized, m_i/m_e > 1 for electron-normalized).

**Scaled simulations:** PIC codes often scale the system to reduce
computational cost — e.g., reducing c/v_A, using artificial mass ratios,
or compressing the ratio between kinetic and MHD scales. The scaling is
already embedded in the normalization reference values (the reader
reconstructs them from the code's actual parameters, not the physical
ones). However, it is useful to document the scaling explicitly so
analysts know not to compare absolute physical values against
observations without accounting for it:

```toml
[units]
system = "PIC"
reference_species = "electrons"
reference_density = 1.0e18
# ...
scaling_factor = 10.0              # optional: system scaled by this factor
scaling_description = "c/v_A reduced by 10x; mass ratio mi/me = 256 (real: 1836)"
```

`scaling_factor` and `scaling_description` are purely documentation —
they don't change any computation. The normalization reference values
already reflect the scaling. These fields tell the analyst "this is a
scaled run" and by how much.

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

**Note on iPIC3D normalization:** iPIC3D does not directly provide
`reference_density` — its normalization is implicit in `qom`, `B0`, and
`rhoINIT`. The iPIC3D reader must reconstruct the reference density
and derive all normalization values from these parameters. The reader
also determines whether the run is electron- or ion-normalized from the
species definitions. This is a reader responsibility, not a schema
concern — the schema stores the *result* of this derivation.

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
charge_to_mass = 0.0               # optional: q/m ratio (alternative to separate charge/mass;
                                   #   some codes like iPIC3D define species by qom only)
particles_per_cell = [5, 5, 1]     # PIC/hybrid: per-cell count [x, y, z] or scalar
temperature = 0.0                  # optional: scalar isotropic temperature (code units)
thermal_velocity = [0.0, 0.0, 0.0] # optional: anisotropic thermal velocities [v_th1, v_th2, v_th3]
                                   #   (use instead of scalar temperature for anisotropic init)
drift_velocity = [0.0, 0.0, 0.0]  # optional: initial bulk drift (code units)
density = 1.0                      # optional: initial number density (code units)
```

**Notes on species parameters:**

Each species must have either (`charge` + `mass`) or `charge_to_mass`.
If `charge_to_mass` is given alone, the reader infers `charge` and `mass`
`qom = -256` implies $q = -1$, $m = 1/256$). The schema prefers
explicit `charge` and `mass`, but `charge_to_mass` is accepted for
compatibility with codes like iPIC3D that define species this way.

`thermal_velocity` takes precedence over `temperature` when both are
present. For isotropic initializations, use `temperature`. For
anisotropic initializations (e.g., Harris sheet with $T_\parallel \neq T_\perp$),
use `thermal_velocity` with three components. The relationship is
$v_{th} = \sqrt{T/m}$ (see thermal speed convention in Section 3).

`particles_per_cell` can be a scalar (same in all directions) or a
3-element array (iPIC3D uses per-direction counts: `npcelx, npcely, npcelz`).
For MHD species, set to 0.

### [physics]

Open-ended section for model-specific parameters. Subsections are keyed
by model type. Only the subsection matching `[model] type` is expected
to be present.

```toml
[physics.pic]                      # PIC-specific parameters
omega_pe_over_omega_ce = 3.0       # frequency ratio (sets B relative to density)
theta = 0.5                        # implicitness parameter (0.5 = Crank-Nicolson, exact energy conservation)
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
Essential for interpreting results — you can't understand a reconnection
simulation without knowing the Harris sheet parameters. Open-ended by
design; common entries shown below.

```toml
[initial_conditions]
type = "double_harris"             # human-readable label for the setup
B0 = [0.097, 0.0, 0.0]            # initial/asymptotic magnetic field (code units)

[initial_conditions.parameters]    # setup-specific parameters (open-ended)
perturbation_amplitude = 0.4       # amplitude of initial perturbation
current_sheet_thickness = 0.25     # half-thickness of current sheet (code units)
# ... any other initial condition parameters
```

The `[initial_conditions]` section is documentation — it describes what
was simulated, not how to run the simulation. Readers extract these from
the simulation code's native input file when possible.

This section is **open-ended by design**. The schema does not enumerate
all possible physics or initial condition parameters — each simulation
code adds what it needs. The structure (subsections keyed by model type
for physics, named parameters for initial conditions) is what's
standardized, not the contents.

### [output]

Optional metadata about what's in the output files.

```toml
[output]
format = "HDF5"                    # "HDF5" | "NetCDF" | "Arrow"
fields = ["B1", "B2", "B3", "E1", "E2", "E3", "rho_c", "J1", "J2", "J3"]
particle_data = false              # whether particle positions/velocities are saved
```

Note: PIC outputs typically include `rho_c` (charge density); MHD outputs
include `rho_m` (mass density). Both may include `n_e`, `n_i`.

### Example: iPIC3D Double Harris sheet → simulation.toml

How an iPIC3D input file maps to our schema (reader generates this):

```toml
[model]
name = "iPIC3D"
type = "PIC"
description = "Double Harris sheet reconnection"

[grid]
dimensions = [100, 100, 1]
spacing = [0.3, 0.3, 1.0]         # Lx/nxc, Ly/nyc, Lz/nzc
origin = [0.0, 0.0, 0.0]
dt = 0.125                         # timestep in code units
boundary = ["periodic", "periodic", "periodic"]

[units]
system = "PIC"
reference_species = "electrons"    # this run is electron-normalized
# Derived by reader from iPIC3D's qom, B0, and rhoINIT:
reference_density = 1.0e18        # m⁻³ (reconstructed)

[coordinates]
geometry = "cartesian"
frame = "simulation"

[physics.pic]
theta = 0.5
speed_of_light = 1.0

[initial_conditions]
type = "double_harris"
B0 = [0.097, 0.0, 0.0]
[initial_conditions.parameters]
perturbation_amplitude = 0.4
current_sheet_thickness = 0.25

[[species]]
name = "electrons_harris"
charge = -1.0
mass = 0.00390625                  # 1/|qom| = 1/256
charge_to_mass = -256.0            # from iPIC3D qom
particles_per_cell = [5, 5, 1]
thermal_velocity = [0.06, 0.02, 0.02]
drift_velocity = [0.0, 0.0, 0.00325]
density = 1.0

[[species]]
name = "ions_harris"
charge = 1.0
mass = 1.0
charge_to_mass = 1.0
particles_per_cell = [5, 5, 1]
thermal_velocity = [0.0063, 0.0063, 0.0063]
drift_velocity = [0.0, 0.0, -0.01624]
density = 1.0

[[species]]
name = "electrons_background"
charge = -1.0
mass = 0.00390625
charge_to_mass = -256.0
particles_per_cell = [5, 5, 1]
thermal_velocity = [0.06, 0.02, 0.02]
drift_velocity = [0.0, 0.0, 0.0]
density = 1.0

[[species]]
name = "ions_background"
charge = 1.0
mass = 1.0
charge_to_mass = 1.0
particles_per_cell = [5, 5, 1]
thermal_velocity = [0.0063, 0.0063, 0.0063]
drift_velocity = [0.0, 0.0, 0.0]
density = 1.0

[output]
format = "HDF5"
fields = ["B1", "B2", "B3", "E1", "E2", "E3", "rho_c", "J1", "J2", "J3"]
# Per-species fields use species index: rho_c_s0 .. rho_c_s3, J1_s0 .. J3_s3
# Species names are in [[species]] table, not in field names
particle_data = true
```

---

## 3. Canonical Field Names

All projects use the same field names. Readers are responsible for
mapping simulation-code-specific names to these canonical names.

### Dual naming convention: Cartesian aliases and general indices

Vector field components use **numbered indices** (`B1`, `B2`, `B3`) as
the general canonical form. The coordinate geometry determines what
each index means:

| Index | Cartesian | Spherical | Cylindrical |
|-------|-----------|-----------|-------------|
| 1 | x | r | r |
| 2 | y | θ | φ |
| 3 | z | φ | z |

For **Cartesian data** (the common case), the familiar letter aliases
(`Bx`, `By`, `Bz`) are preferred for readability and are registered
as aliases for `B1`, `B2`, `B3`. Both access the same data:

```python
data["Bx"]   # Cartesian alias — preferred when geometry is Cartesian
data["B1"]   # General index — always works regardless of geometry
```

For **non-Cartesian data**, only the numbered form is canonical.
The reader registers geometry-appropriate aliases if desired
(e.g., `Br` → `B1`, `Btheta` → `B2`, `Bphi` → `B3` for spherical).

The `[coordinates] geometry` field in `simulation.toml` determines
which aliases are active.

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

`rho_c` and `rho_m` are unambiguous — no overloaded `rho`. Number
densities per species (`n_s0`, `n_s1`, ...) are the most fundamental;
charge and mass densities are derived from them using species charge
and mass. For the common two-species case (electron-ion), `n_e` and
`n_i` are accepted as aliases for `n_s0` and `n_s1`.

**Per-species naming convention:** For per-species fields, append `_s`
plus the species index from the `[[species]]` list (0-based):
`rho_c_s0`, `J1_s1`, `V1_s2`, `n_s3`. The species *name* (e.g.,
"electrons", "ions_background") lives in the `[[species]]` table, not
in the field name. Analysis tools look up the name from the species
list when needed (for plot labels, legends, etc.).

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
(Joules in SI, or dimensionless in normalized units where $k_B = 1$).
The relationship is $T = P/n$, not $T = P/(nk_B)$. To convert to Kelvin,
divide by $k_B$. This convention is standard in PIC codes and avoids
carrying Boltzmann's constant through every formula.

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

**γ convention for PIC entropy:** PIC codes have no equation of state,
so γ in the isotropic entropy formula is chosen by the number of
velocity-space degrees of freedom: $\gamma = 5/3$ for 3D, $\gamma = 2$
for 2D, $\gamma = 3$ for 1D. The 3D value ($\gamma = 5/3$) is the
default. **Note:** The gyrotropic entropy exponent of 5 in
$P_\parallel P_\perp^2 / n^5$ is **independent** of $\gamma = 5/3$.
It arises from combining the two CGL double-adiabatic invariants:
$P_\perp / (nB) = \text{const}$ and $P_\parallel B^2 / n^3 = \text{const}$,
giving $P_\parallel P_\perp^2 / n^{3+2} = \text{const}$. The coincidence
with the numerator of 5/3 is just that — a coincidence.

PIC entropy is computed from particle velocity moments, not from an
equation of state. The isotropic form uses scalar pressure and number
density; the gyrotropic form uses the parallel and perpendicular pressure
components. Both are commonly used to study dissipation and heating in
PIC simulations of reconnection and turbulence. Species-specific
entropies (`s_e`, `s_i`) are particularly useful for tracking which
species is heated.

Entropy and enthalpy are typically derived, not stored. The adiabatic
index comes from the `[physics.mhd]` config, not from field data.

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
| `Pij` | Full pressure tensor (6 independent components of symmetric 3×3 tensor: P11, P12, P13, P22, P23, P33) | PIC |
| `agyrotropy` | Agyrotropy measure $Q$ (quantifies deviation from gyrotropic symmetry) | PIC (derived) |

### Characteristic scales (derived)

In the formulas below, `n_e`, `T_e`, `m_e` etc. refer to the appropriate
species' density, temperature, and mass from the `[[species]]` list.
For multi-species simulations, the analysis tool must be told which
species is "the electrons" and "the ions" (typically species 0 and 1).

| Canonical | Meaning | Computed from |
|-----------|---------|---------------|
| `d_e` | Electron skin depth (inertial length) $c/\omega_{pe}$ | `n_e`, species |
| `d_i` | Ion skin depth (inertial length) $c/\omega_{pi}$ | `n_i`, species |
| `r_e` | Electron thermal gyroradius $v_{th,e}/\omega_{ce}$ | `Te`, `|B|`, species |
| `r_i` | Ion thermal gyroradius $v_{th,i}/\omega_{ci}$ | `Ti`, `|B|`, species |
| `omega_pe` | Electron plasma frequency $\sqrt{n_e e^2 / (\epsilon_0 m_e)}$ | `n_e`, species |
| `omega_pi` | Ion plasma frequency $\sqrt{n_i Z^2 e^2 / (\epsilon_0 m_i)}$ | `n_i`, species |
| `omega_ce` | Electron cyclotron frequency $\|e\|B / m_e$ | `|B|`, species |
| `omega_ci` | Ion cyclotron frequency $ZeB / m_i$ | `|B|`, species |
| `lambda_D` | Electron Debye length $\sqrt{\epsilon_0 T_e / (n_e e^2)}$ | `n_e`, `Te`, species |
| `v_A` | Alfvén speed $B/\sqrt{\mu_0 \rho_m}$ | `|B|`, `rho_m` |
| `v_th_e` | Electron thermal speed $\sqrt{T_e / m_e}$ | `Te`, species |
| `v_th_i` | Ion thermal speed $\sqrt{T_i / m_i}$ | `Ti`, species |
| `c_s` | Sound speed $\sqrt{\gamma P / \rho_m}$ | `P`, `rho_m`, `gamma_eos` |
| `v_ms` | Fast magnetosonic speed $\sqrt{v_A^2 + c_s^2}$ | `v_A`, `c_s` |
| `M_A` | Alfvén Mach number $V / v_A$ | `|V|`, `v_A` |
| `M_ms` | Magnetosonic Mach number $V / v_{ms}$ | `|V|`, `v_ms` |
| `beta` | Plasma beta $2\mu_0 P / B^2$ | `P`, `|B|` |
| `beta_e` | Electron beta $2\mu_0 P_e / B^2$ | `Pe`, `|B|` |
| `beta_i` | Ion beta $2\mu_0 P_i / B^2$ | `Pi`, `|B|` |

**Thermal speed convention:** We adopt $v_{th} = \sqrt{T/m}$ (with $T$
in energy units), following the NRL Plasma Formulary (Richardson 2019)
convention. This is the 1D Maxwellian standard deviation $\sigma$ where
$f(v_x) \propto \exp(-v_x^2 / (2\sigma^2))$ with $\sigma^2 = T/m$.
**The community is split:** NRL, Chen, and kinetic theory use $\sqrt{T/m}$;
Fitzpatrick, Bellan, and some particle physics texts use $\sqrt{2T/m}$
("most probable speed"). The factor of $\sqrt{2}$ propagates into
gyroradius definitions — always state the convention explicitly. The
gyroradius $r = v_{th}/\omega_c$ uses whichever $v_{th}$ is adopted.

**Cyclotron frequency convention:** $\omega_{ce}$ and $\omega_{ci}$ are
defined as positive (magnitudes), following NRL, Chen, and Fitzpatrick.
Bellan uses the signed convention $\omega_{c\sigma} = q_\sigma B / m_\sigma$
(so $\omega_{ce} < 0$ for electrons). The signed form is needed in the
Stix cold plasma dielectric tensor. Our stored values are always positive.

**Debye length note:** `lambda_D` is specifically the *electron* Debye
length. The total plasma Debye length is
$1/\lambda_D^2 = \sum_s n_s q_s^2 / (\epsilon_0 T_s)$.

**Fast magnetosonic speed note:** $v_{ms} = \sqrt{v_A^2 + c_s^2}$ is the
maximum fast-mode phase speed, valid for propagation **perpendicular to B**
($\theta = 90°$). The general angle-dependent dispersion relation is
$v_{f,s}^2 = \frac{1}{2}[v_A^2 + c_s^2 \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta}]$
where $+$ gives the fast mode and $-$ the slow mode.

**Sound speed note:** $c_s = \sqrt{\gamma P / \rho_m}$ is the MHD sound
speed. This differs from the ion acoustic speed
$c_{ia} = \sqrt{(T_e + \gamma_i T_i) / m_i}$ commonly used in kinetic
theory (where $\gamma_e = 1$ for isothermal electrons, $\gamma_i = 3$
for 1D adiabatic ions).

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
| `vort1`, `vort2`, `vort3` | `vort_x`, ... | Fluid vorticity components $(\nabla \times \mathbf{V})$ | V1, V2, V3, grid |
| `\|vort\|` | — | Vorticity magnitude | vort1, vort2, vort3 |

Derived quantities and characteristic scales are not stored in output
files. They are computed on demand by the analysis tools using the
canonical stored fields, species parameters, and physics config.

**Computation units rule:** All derived quantities are computed in
**normalized (code) units**, never in SI. This keeps formulas clean
(physical constants vanish: $\mu_0 = 1$, $\epsilon_0 = 1$ in PIC;
$\mu_0 = 1$ in MHD), preserves numerical precision (O(1) numbers
instead of products of $10^{\pm 20}$), and avoids unnecessary
floating-point round-trips. Conversion to SI or display units (nT,
km/s, etc.) happens only at the output/visualization boundary.
Dimensionless quantities (plasma beta, Mach numbers, $\omega_{pe}/\omega_{ce}$)
are identical in any unit system and need no conversion.

Scalar quantities (`n_s0`, `rho_m`, `rho_c`, `P`, `Te`, `s`, `e_B`, `beta`, ...)
have no coordinate dependence and use the same name regardless of geometry.

### Naming conventions

**Field names (dict keys, HDF5 dataset names):**

- Vector components use numbered index: `B1`, `B2`, `B3` (canonical)
- Cartesian aliases use uppercase letter suffix: `Bx`, `By`, `Bz`
- Scalar quantities are short scientific: `rho_c`, `rho_m`, `P`, `Pe`
- Per-species fields use `_s` + index: `n_s0`, `V1_s1`, `rho_c_s2`
- Two-species aliases accepted: `n_e`=`n_s0`, `n_i`=`n_s1` (electron-ion only)
- Derived magnitudes use pipe notation in analysis: `|B|`, `|J|`
- All names are strings used as dictionary keys / HDF5 dataset names

**Function names:** Descriptive English — `magnetic_field_magnitude()`,
`plasma_beta()`, `alfven_speed()`.

**Function parameters:** Short scientific — `bx`, `rho`, `dt`, `q_over_m`.
The docstring and LaTeX equation provide the full description.

**Class/struct names:** Descriptive English — `FieldDataset`,
`Normalization`, `SpeciesInfo`, `CoordinateGeometry`.

| Context | Convention | Examples |
|---------|-----------|---------|
| Field names (dict keys) | Short scientific | `"B1"`, `"Bx"`, `"rho_m"`, `"P"` |
| Function names | Descriptive English | `magnetic_field_magnitude()` |
| Function parameters | Short scientific | `bx`, `rho`, `dt` |
| Class / struct names | Descriptive English | `FieldDataset`, `Normalization` |

This matches conventions used by NumPy, SciPy, and most scientific
Python/Rust libraries. Readable to both physicists (who recognize the
symbols) and software engineers (who read function names and docstrings).

---

## 4. HDF5 Output Layout

The standard layout for simulation output files. Each timestep is a
separate file (or group within a file). Field datasets use the
**numbered canonical names** (`B1`, `B2`, `B3`), which are
geometry-agnostic. The `geometry` attribute tells the reader what
each index means.

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

**Self-describing output:** Every file contains enough metadata to convert
back to SI without the original `simulation.toml`. The normalization
group stores all reference values. This means output files can be shared,
archived, or analyzed independently.

**Coordinate alias registration:** When a reader loads this file and sees
`geometry = "cartesian"`, it registers `Bx → B1`, `By → B2`, `Bz → B3`
as aliases in the `FieldDataset`. For `geometry = "spherical"`, it
registers `Br → B1`, `Btheta → B2`, `Bphi → B3` instead. The HDF5
file itself always uses numbered names.

**Reader responsibility:** Each reader (iPIC3D, BATSRUS, Rust code) maps
its native file layout to this canonical layout. The native files don't
need to follow this layout — the reader handles the translation.

**Rust code output:** The Rust simulation writes this layout directly.
No translation needed.

---

## 5. API Response Format (Viewer)

The FastAPI server serves data to the Three.js viewer. The initial format
is binary ArrayBuffers with metadata headers. Future: Arrow IPC.

### Current: Binary ArrayBuffer

```
GET /runs/{id}/fields/{step}/{field_name}?plane=xy&index=32&units=code

Response:
  Content-Type: application/octet-stream
  X-Shape: 256,128
  X-Dtype: float64
  X-Units: code
  X-Field: Bx
  X-Time: 42.5
  Body: raw float64 bytes
```

### Future: Arrow IPC

```
GET /runs/{id}/fields/{step}/{field_name}?plane=xy&index=32&units=code

Response:
  Content-Type: application/vnd.apache.arrow.stream
  Body: Arrow IPC stream containing:
    - field data array
    - coordinate arrays (x, y)
    - metadata (units, time, normalization refs)
```

Arrow IPC is zero-copy in JavaScript (Arrow JS) and carries metadata
alongside the data. The switch from ArrayBuffer to Arrow is transparent
to the viewer — the data access pattern is the same, just with richer
metadata.

---

## 6. Extensibility

### Adding a new simulation code

1. Write a reader (Python) that maps the code's native output to
   `FieldDataset` with canonical field names
2. Optionally write a `simulation.toml` generator for the code
3. No changes to the schema, analysis layer, or viewer

### Adding a new field

1. Add the name to the canonical field names table (this document)
2. Add it to relevant readers
3. Add derived quantity functions if applicable
4. Viewer and analysis layer pick it up via string-keyed access

### Adding a new model type

1. Add a new `[physics.NEW_TYPE]` subsection convention
2. Document expected fields and species for the new type
3. Readers and analysis tools use `[model] type` to select behavior

### Adding a new output format

1. Define the layout mapping (like HDF5 layout above)
2. Write a reader for it
3. Everything downstream works unchanged — it all goes through
   `FieldDataset`

---

## 7. What This Schema Does NOT Define

- **Simulation control parameters** (number of timesteps, solver tolerances,
  MPI domain decomposition, output intervals). Those belong in the
  simulation code's own input configuration, not in the output metadata.
  (Note: `dt` is included in `[grid]` because it is essential for
  time-series analysis, not because it's a simulation control.)

- **Visualization settings** (colormaps, camera angles, slice positions).
  Those belong in the viewer's configuration.

- **Exhaustive physics parameter lists.** The `[physics]` section is
  open-ended by design. Each code adds what it needs.

- **Native file layouts** of existing codes (iPIC3D's HDF5 structure,
  BATSRUS's .out format). Readers handle the translation.

- **Internal data structures** of any project (Rust's `FieldState`,
  Python's `xr.Dataset` internals, TypeScript types). Each project
  maps to/from the schema at its boundaries.
