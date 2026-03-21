# pypic Variable Naming Convention

Version: 2.1
Status: Draft

---

## 1. Design Principles

1. **Canonical names are programmatically constructable** — no magic strings.
2. **Ordering is strict**: `{base}[_{qualifier}][_s{k}][_{direction}]`
3. **Species before direction, direction always last** — the only hard invariant.
4. **Aliases are a separate layer**, hand-picked for common variables only.
5. **Qualifier chain is free-form** — anything between base and species is allowed.
   New physics quantities should slot in without touching the parser.

---

## 2. Name Construction

```
{base}[_{qualifier}...][_s{k}][_{direction}]
  │         │              │         │
  │         │              │         └─ Vector/tensor direction (always last)
  │         │              │            Canonical:  _1  _2  _3
  │         │              │            Cartesian:  _x  _y  _z
  │         │              │            Spherical:  _r  _theta  _phi
  │         │              │            Cylindrical: _r  _phi  _z
  │         │              │            Projected:  _par  _perp
  │         │              │
  │         │              └─ Species index (second to last)
  │         │                 _s0  _s1  _s2  ...
  │         │
  │         └─ Sub-type, variation, tensor index, etc.
  │           Free-form, multiple qualifiers allowed, underscore-separated.
  │           Examples: _magnetic, _gyrotropic, _bulk, _11, _par,
  │                     _reconnection_flow, _relativistic
  │
  └─ Physical quantity root
     Examples: B, E, n, P, V, energy, entropy, vort, omega_p,
               larmor_radius, gamma_Lorentz, ...
```

### Slot applicability (soft convention, not rigid)

Not every variable uses every slot. Common patterns:

| Pattern                        | Example                        |
|--------------------------------|--------------------------------|
| base + direction               | `B_x`, `E_1`                  |
| base + species                 | `n_s0`, `P_s1`                |
| base + species + direction     | `V_s0_x`, `u_s1_z`            |
| base + index + species         | `P_12_s0`, `P_par_s1`         |
| base + qualifier               | `energy_magnetic`, `c_s`       |
| base + qualifier + species     | `entropy_gyrotropic_s0`        |
| base + free-form + species + dir | `V_reconnection_flow_s0_x`   |

The qualifier chain is deliberately open-ended. The parser should not
reject names based on rigid per-variable signatures. New quantities may
combine slots in unforeseen ways.

---

## 3. Formatting Rules

| Rule | Correct | Wrong |
|------|---------|-------|
| Underscores everywhere | `P_e`, `T_i`, `B_x` | `Pe`, `Ti`, `Bx`* |
| `_theta` always spelled out (canonical) | `V_s0_theta` | `V_s0_th` |
| `_th` only in alias layer = "thermal" | alias `v_th_e` → `v_thermal_s0` | — |
| Bare name = total/bulk quantity | `P` = total pressure | — |
| No bare `gamma` | `gamma_Lorentz`, `gamma_ad` | `gamma` |

*\*Short forms like `Bx` exist as hand-picked aliases, not canonical names.*

---

## 4. Species System

### 4.1 Indexed canonical names

Every species-dependent quantity uses `_s{k}` where `k = 0, 1, 2, ...`
matching the simulation species array order.

```python
# These always work, regardless of configuration
ds.compute("n_s0")          # density of species 0
ds.compute("P_s2")          # pressure of species 2
ds.compute("V_s1_x")        # x-velocity of species 1
ds.compute("P_12_s0")       # off-diagonal pressure tensor, species 0
```

### 4.2 Alias registration (config-driven)

Human-readable labels are registered from simulation metadata:

```python
# Default (two-species electron-ion)
species_labels = {0: "e", 1: "i"}
# Generates: n_e -> n_s0, P_i -> P_s1, T_e -> T_s0, ...

# Three-species (electron-proton-alpha)
species_labels = {0: "e", 1: "p", 2: "alpha"}
# Generates: n_alpha -> n_s2, T_p -> T_s1, omega_c_alpha -> omega_c_s2, ...

# Multi-population (two proton beams)
species_labels = {0: "e", 1: "p_core", 2: "p_beam"}
# Generates: n_p_core -> n_s1, V_p_beam_x -> V_s2_x, ...
```

Aliases propagate automatically to all per-species variables.
If two species share a base label, alias registration should warn.

### 4.3 Bare names (no species suffix)

| Bare name | Meaning |
|-----------|---------|
| `P` | Total pressure (sum over species) |
| `V_x` | Mass-weighted bulk velocity |
| `n` | Total number density (if defined) |
| `energy_kinetic` | Total kinetic energy density |
| `entropy` | **Not defined** — always requires `_s{k}` |
| `agyrotropy` | **Not defined** — always requires `_s{k}` |

---

## 5. Variable Reference

### 5.1 Electromagnetic Fields

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `B_1`, `B_2`, `B_3` | `B_x`/`B_y`/`B_z`, `B_r`/`B_theta`/`B_phi`, `Bx`/`By`/`Bz` | stored | b_field |
| `E_1`, `E_2`, `E_3` | `E_x`/`E_y`/`E_z`, `E_r`/`E_theta`/`E_phi`, `Ex`/`Ey`/`Ez` | stored | e_field |
| `B_mag` | `\|B\|`, `Bmag` | derived | b_field |
| `E_mag` | `\|E\|`, `Emag` | derived | e_field |

Geometry aliases depend on `geometry` config (Cartesian, Spherical, Cylindrical).

```python
ds.compute("B_mag")          # magnetic field magnitude
ds.compute("Bmag")           # same (alias)
ds.compute("|B|")            # same (alias)
ds["B_x"]                    # stored Bx component (Cartesian geometry)
```

### 5.2 Current Density

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `J_1`, `J_2`, `J_3` | `J_x`/`J_y`/`J_z`, `Jx`/`Jy`/`Jz` | stored | current_density |
| `J_mag` | `\|J\|`, `Jmag` | derived | current_density |

### 5.3 Bulk Velocity

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `V_s{k}_1` | `V_s{k}_x`, `V_e_x`, `V_i_y`, ... | stored | velocity |
| `V_1` | `V_x` | derived | velocity |
| `V_mag` | `\|V\|` | derived | velocity |

`V` (no species) = mass-weighted bulk velocity.
`V_s{k}` = per-species bulk velocity.

```python
ds.compute("V_s0_x")        # electron bulk velocity, x-component
ds.compute("V_e_x")         # same (with default species labels)
ds.compute("V_x")           # total mass-weighted bulk velocity, x
ds.compute("V_mag")         # total bulk speed
```

### 5.4 Four-Velocity (relativistic)

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `u_1`, `u_2`, `u_3` | `u_x`/`u_y`/`u_z`, `u_r`/`u_theta`/`u_phi` | stored | velocity |

### 5.5 Densities

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `n_s0` | `n_e` | stored | density |
| `n_s1` | `n_i` | stored | density |
| `n_s{k}` | (from species labels) | stored | density |
| `rho_c` | — | stored | charge_density |
| `rho_m` | — | stored | mass_density |

```python
ds.compute("n_s0")           # electron number density
ds.compute("n_e")            # same
ds.compute("n_s2")           # third species density
ds.compute("n_alpha")        # same (if species_labels maps 2 -> "alpha")
```

### 5.6 Pressure & Temperature (scalar)

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `P` | — | derived | pressure |
| `P_s0` | `P_e` | stored | pressure |
| `P_s1` | `P_i` | stored | pressure |
| `T_s0` | `T_e` | stored | temperature |
| `T_s1` | `T_i` | stored | temperature |

`P` (bare) = total pressure (sum over species).

### 5.7 Pressure Tensor

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `P_11_s{k}` ... `P_33_s{k}` | `P_xx_s{k}` ... `P_zz_s{k}` | stored | pressure |
| `P_12_s{k}`, `P_13_s{k}`, `P_23_s{k}` | `P_xy_s{k}`, `P_xz_s{k}`, `P_yz_s{k}` | stored | pressure |
| `P_par_s{k}` | `P_parallel_s{k}` | derived | pressure |
| `P_perp_s{k}` | — | derived | pressure |
| `agyrotropy_s{k}` | `Q_s{k}` | derived | dimensionless |

`agyrotropy` uses the Swisdak (2016) measure.

```python
ds.compute("P_12_s0")       # off-diagonal tensor component, electrons
ds.compute("P_xy_e")        # same (geometry alias + species alias)
ds.compute("P_par_s1")      # parallel pressure, ions
ds.compute("P_parallel_i")  # same
ds.compute("agyrotropy_s0") # electron agyrotropy
ds.compute("Q_e")           # same
```

### 5.8 Plasma Parameters

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `plasma_beta` | `beta` | derived | dimensionless |
| `plasma_beta_s{k}` | `beta_e`, `beta_i`, `beta_s{k}` | derived | dimensionless |
| `v_Alfven` | `v_A` | derived | velocity |
| `c_s` | — | derived | velocity |
| `c_ia` | — | derived | velocity |
| `c_ms` | `v_ms` | derived | velocity |
| `M_A` | — | derived | dimensionless |
| `M_ms` | — | derived | dimensionless |

`c_s` = MHD sound speed. `c_ia` = ion acoustic speed.
`c_ms` = fast magnetosonic speed.

```python
ds.compute("plasma_beta")   # total plasma beta
ds.compute("beta")           # same
ds.compute("beta_e")         # electron beta
ds.compute("v_Alfven")       # Alfvén speed
ds.compute("v_A")            # same
ds.compute("c_ms")           # fast magnetosonic speed
```

### 5.9 Energy Densities

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `energy_magnetic` | `e_B`, `w_B` | derived | energy_density |
| `energy_electric` | `e_E`, `w_E` | derived | energy_density |
| `energy_kinetic` | `e_k`, `w_k` | derived | energy_density |
| `energy_thermal` | `e_th`, `w_th` | derived | energy_density |

Canonical uses `energy_` prefix to avoid `e_` / electron ambiguity.
`w_` aliases follow European convention (Energiedichte).

```python
ds.compute("energy_magnetic")  # B²/2
ds.compute("e_B")              # same
ds.compute("w_B")              # same
```

### 5.10 Thermodynamic Quantities

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `enthalpy` | `h` | derived | temperature |
| `enthalpy_relativistic` | `h_rel` | derived | temperature |
| `energy_internal` | `e_int` | derived | temperature |
| `entropy_s{k}` | `s_s{k}`, `s_e`, `s_i` | derived | dimensionless |
| `entropy_gyrotropic_s{k}` | `s_gyro_s{k}`, `s_gyro_e` | derived | dimensionless |

`entropy` without species suffix is **not defined**.
`s_gyro` without species index is **not defined**.

```python
ds.compute("enthalpy")                 # specific enthalpy
ds.compute("h")                        # same
ds.compute("enthalpy_relativistic")    # relativistic specific enthalpy
ds.compute("h_rel")                    # same
ds.compute("entropy_s0")              # electron entropy
ds.compute("s_e")                      # same
ds.compute("entropy_gyrotropic_s0")   # electron gyro entropy
ds.compute("s_gyro_e")                # same
```

### 5.11 Poynting Flux

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `S_1`, `S_2`, `S_3` | `S_x`/`S_y`/`S_z` | derived | poynting_flux |

**Case sensitivity note:** `S` (uppercase) = Poynting flux, `s` (lowercase) = entropy.
Consider using `poynting_1`/`poynting_x` as unambiguous alternatives in
case-insensitive contexts (HDF5 dataset names, filenames).

### 5.12 Species-Dependent Scales

| Canonical | Aliases | SI quantity |
|-----------|---------|-------------|
| `omega_p_s{k}` | `omega_pe`, `omega_pi` | frequency |
| `omega_c_s{k}` | `omega_ce`, `omega_ci` | frequency |
| `d_s{k}` | `d_e`, `d_i` | length |
| `v_thermal_s{k}` | `v_th_e`, `v_th_i`, `v_th_s{k}` | velocity |
| `larmor_radius_s{k}` | `rL_s{k}`, `r_e`, `r_i` | length |
| `larmor_radius_bulk_s{k}` | `rL_bulk_s{k}`, `rL_bulk_e` | length |
| `lambda_D` | — | length |

`larmor_radius` (no qualifier) = thermal gyroradius (the default).
`larmor_radius_bulk` = bulk-velocity gyroradius.
`v_thermal` follows NRL Plasma Formulary: $\sqrt{T/m}$.
`lambda_D` = Debye length (species-independent).

```python
ds.compute("omega_p_s0")            # electron plasma frequency
ds.compute("omega_pe")              # same
ds.compute("larmor_radius_s1")      # ion thermal gyroradius
ds.compute("rL_i")                  # same
ds.compute("larmor_radius_bulk_s0") # electron bulk gyroradius
ds.compute("v_thermal_s0")          # electron thermal speed
ds.compute("v_th_e")                # same
```

### 5.13 Lorentz Factor & Adiabatic Index

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `gamma_Lorentz` | `gamma_L` | derived | dimensionless |
| `gamma_Lorentz_s{k}` | `gamma_L_s{k}`, `gamma_L_e` | derived | dimensionless |
| `gamma_ad` | `adiabatic_index` | parameter | dimensionless |

`gamma_Lorentz` = bulk Lorentz factor (total).
`gamma_Lorentz_s{k}` = per-species Lorentz factor.
No bare `gamma` alias — too ambiguous.

```python
ds.compute("gamma_Lorentz")         # total bulk Lorentz factor
ds.compute("gamma_L")               # same
ds.compute("gamma_Lorentz_s0")      # electron Lorentz factor
ds.compute("gamma_L_e")             # same
```

### 5.14 Magnetization

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `sigma` | — | derived | dimensionless |

$\sigma = B^2 / \rho_m c^2$. Species-independent.

### 5.15 Differential Operators

| Canonical | Aliases | Type | SI quantity |
|-----------|---------|------|-------------|
| `div_B` | — | derived | b_field / length |
| `div_E` | — | derived | e_field / length |
| `curl_B_1`, `curl_B_2`, `curl_B_3` | `curl_B_x`/`curl_B_y`/`curl_B_z` | derived | b_field / length |
| `vort_1`, `vort_2`, `vort_3` | `vort_x`/`vort_y`/`vort_z` | derived | frequency |
| `vort_mag` | `\|vort\|` | derived | frequency |

```python
ds.compute("div_B")          # ∇·B (should be ~0, diagnostic)
ds.compute("curl_B_x")       # (∇×B)_x
ds.compute("vort_mag")       # |∇×V|
```

---

## 6. Alias Mechanisms (three layers)

### Layer 1: Geometry aliases (automatic)

Maps coordinate-letter suffixes to canonical `_1/_2/_3`.
Active set depends on `geometry` config.

| Geometry    | `_1`  | `_2`      | `_3`    |
|-------------|-------|-----------|---------|
| Cartesian   | `_x`  | `_y`      | `_z`    |
| Spherical   | `_r`  | `_theta`  | `_phi`  |
| Cylindrical | `_r`  | `_phi`    | `_z`    |

Applies to all vector quantities: `B`, `E`, `J`, `V`, `u`, `S`,
`curl_B`, `vort`, etc.

### Layer 2: Species aliases (config-driven)

Maps `_{label}` to `_s{k}` for all per-species variables.
Default: `{0: "e", 1: "i"}`. User-configurable.

### Layer 3: Hand-picked short aliases (hardcoded)

Curated list for frequently-typed variables only:

| Alias | → Canonical |
|-------|-------------|
| `Bx`, `By`, `Bz` | `B_x`, `B_y`, `B_z` |
| `Ex`, `Ey`, `Ez` | `E_x`, `E_y`, `E_z` |
| `Jx`, `Jy`, `Jz` | `J_x`, `J_y`, `J_z` |
| `Bmag`, `Emag`, `Jmag` | `B_mag`, `E_mag`, `J_mag` |
| `\|B\|`, `\|E\|`, `\|J\|` | `B_mag`, `E_mag`, `J_mag` |
| `v_A` | `v_Alfven` |
| `beta` | `plasma_beta` |
| `Q` (+species) | `agyrotropy` (+species) |
| `h` | `enthalpy` |
| `h_rel` | `enthalpy_relativistic` |
| `e_int` | `energy_internal` |
| `e_B`, `w_B` | `energy_magnetic` |
| `e_E`, `w_E` | `energy_electric` |
| `e_k`, `w_k` | `energy_kinetic` |
| `e_th`, `w_th` | `energy_thermal` |
| `gamma_L` | `gamma_Lorentz` |
| `P_parallel` (+species) | `P_par` (+species) |
| `s` (+species) | `entropy` (+species) |
| `adiabatic_index` | `gamma_ad` |
| `v_ms` | `c_ms` |

---

## 7. Edge Cases & Design Decisions

### 7.1 `_th` vs `_theta`

- `_theta` is **always spelled out** in canonical names (spherical coordinate).
- `_th` is permitted **only in the alias layer**, meaning "thermal".
- Parser rule: canonical names never contain `_th` as a token.
- Example: `v_thermal_s0` is canonical, `v_th_e` is alias.

### 7.2 Bare names (no species suffix)

- `P`, `V`, `n`, `rho_m` = total / mass-weighted bulk quantities.
- `entropy` (bare) = **not defined** (no thermodynamic total entropy).
- `agyrotropy` (bare) = **not defined** (always per-species).
- Appending `_s{k}` to species-independent variables (e.g., `B_s0`,
  `sigma_s1`) should produce a warning or error.

### 7.3 Species-independent quantities

These should **never** accept `_s{k}`:

`B`, `E`, `J`, `S` (Poynting), `div_B`, `div_E`, `curl_B`, `vort`,
`sigma`, `lambda_D`, `gamma_ad`, `v_Alfven`, `c_s`, `c_ia`, `c_ms`,
`M_A`, `M_ms`, `energy_magnetic`, `energy_electric`, `rho_c`, `rho_m`

### 7.4 Quantities with both total and per-species forms

These accept optional `_s{k}` (bare = total):

`P`, `T`, `V`, `n`, `plasma_beta`, `gamma_Lorentz`,
`energy_kinetic`, `energy_thermal`

### 7.5 Multi-population ambiguity

- Indexed names (`n_s1`, `n_s2`) are always unambiguous.
- Config labels resolve human names: `n_p_core`, `n_p_beam`.
- If two species share a base label, alias registration should warn.

### 7.6 `S` vs `s` (Poynting vs entropy)

- Case-sensitive: `S_1` = Poynting flux, `s_s0` (or `entropy_s0`) = entropy.
- In case-insensitive contexts (filenames, some HDF5 implementations),
  use `poynting_1` and `entropy_s0` as unambiguous long forms.

### 7.7 Double-digit species indices

- `n_s10`: species 10, not species 1 + trailing `0`.
- Parser resolves this because direction tokens are a known finite set
  (`_x`, `_y`, `_z`, `_1`, `_2`, `_3`, `_r`, `_theta`, `_phi`,
  `_par`, `_perp`), so `0` alone is not a valid trailing token.
- Alternative if needed: zero-pad (`_s00` ... `_s10`) for >9 species.

### 7.8 Future extensibility

The free-form qualifier chain ensures new physics quantities can be added
without parser changes. Examples of hypothetical future names that are
already valid under this convention:

```python
"V_reconnection_flow_s0_x"    # reconnection inflow velocity
"P_nongyrotropic_s0"          # non-gyrotropic pressure component
"energy_kinetic_s2"           # kinetic energy density of species 2
"entropy_relative_s0_s1"      # cross-species entropy (if ever needed)
```

The last example (`_s0_s1`) would require a parser extension for
two-species-index quantities. Not currently needed but the naming
convention does not preclude it.

---

## 8. Quick Reference Card

```python
# --- Fields ---
ds.compute("B_x")                    # Bx component
ds.compute("B_mag")                  # |B|
ds.compute("E_mag")                  # |E|

# --- Densities ---
ds.compute("n_e")                    # electron density  (= n_s0)
ds.compute("n_s2")                   # third species density
ds.compute("rho_m")                  # mass density

# --- Velocities ---
ds.compute("V_e_x")                  # electron Vx       (= V_s0_x)
ds.compute("V_x")                    # total bulk Vx

# --- Pressures ---
ds.compute("P_e")                    # electron pressure  (= P_s0)
ds.compute("P")                      # total pressure
ds.compute("P_par_s0")              # electron parallel pressure
ds.compute("P_12_s1")               # ion off-diagonal tensor

# --- Plasma parameters ---
ds.compute("beta")                   # plasma beta
ds.compute("v_A")                    # Alfvén speed
ds.compute("c_s")                    # sound speed

# --- Energy ---
ds.compute("energy_magnetic")        # B²/2  (alias: e_B)
ds.compute("energy_kinetic")         # ρV²/2 (alias: e_k)

# --- Thermodynamics ---
ds.compute("entropy_s0")            # electron entropy   (alias: s_e)
ds.compute("entropy_gyrotropic_s0") # electron gyro entropy
ds.compute("enthalpy")               # specific enthalpy  (alias: h)
ds.compute("enthalpy_relativistic")  # relativistic enthalpy (alias: h_rel)

# --- Scales ---
ds.compute("omega_pe")               # electron plasma frequency
ds.compute("larmor_radius_s1")       # ion thermal gyroradius
ds.compute("v_thermal_s0")          # electron thermal speed
ds.compute("lambda_D")              # Debye length

# --- Lorentz factor ---
ds.compute("gamma_Lorentz")          # total bulk Lorentz factor
ds.compute("gamma_L_e")             # electron Lorentz factor

# --- Diagnostics ---
ds.compute("div_B")                  # ∇·B
ds.compute("curl_B_x")             # (∇×B)_x
ds.compute("vort_mag")              # |∇×V|

# --- SI units ---
ds.in_si("B_x")                     # Bx in Tesla
ds.in_si("n_e")                     # n_e in m⁻³
ds.in_si("v_A")                     # Alfvén speed in m/s
```
