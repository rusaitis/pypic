# pypic Variable Name Reference

All names usable with `ds.compute()`, `ds.in_si()`, `ds.in_units()`,
and `ds["..."]` (for stored fields). Organized by physics category.

## Electromagnetic Fields

| Canonical | Cartesian | Spherical | Cylindrical | Type | SI quantity |
|-----------|-----------|-----------|-------------|------|-------------|
| `B1` | `Bx` | `Br` | `Br` | stored | b_field |
| `B2` | `By` | `Btheta` | `Bphi` | stored | b_field |
| `B3` | `Bz` | `Bphi` | `Bz` | stored | b_field |
| `E1` | `Ex` | `Er` | `Er` | stored | e_field |
| `E2` | `Ey` | `Etheta` | `Ephi` | stored | e_field |
| `E3` | `Ez` | `Ephi` | `Ez` | stored | e_field |
| `|B|` | — | — | — | derived | b_field |
| `|E|` | — | — | — | derived | e_field |

## Current Density

| Canonical | Cartesian | Spherical | Cylindrical | Type | SI quantity |
|-----------|-----------|-----------|-------------|------|-------------|
| `J1` | `Jx` | `Jr` | `Jr` | stored | current_density |
| `J2` | `Jy` | `Jtheta` | `Jphi` | stored | current_density |
| `J3` | `Jz` | `Jphi` | `Jz` | stored | current_density |
| `|J|` | — | — | — | derived | current_density |

## Bulk Velocity (three-velocity)

| Canonical | Cartesian | Spherical | Cylindrical | Type | SI quantity |
|-----------|-----------|-----------|-------------|------|-------------|
| `V1` | `Vx`, `vx` | `Vr`, `vr` | `Vr`, `vr` | stored | velocity |
| `V2` | `Vy`, `vy` | `Vtheta`, `vtheta` | `Vphi`, `vphi` | stored | velocity |
| `V3` | `Vz`, `vz` | `Vphi`, `vphi` | `Vz`, `vz` | stored | velocity |
| `Ve1` | `Vex` | `Ver` | `Ver` | stored | velocity |
| `Ve2` | `Vey` | `Vetheta` | `Vephi` | stored | velocity |
| `Ve3` | `Vez` | `Vephi` | `Vez` | stored | velocity |
| `|V|` | — | — | — | derived | velocity |

## Four-Velocity (relativistic)

| Canonical | Cartesian | Spherical | Cylindrical | Type | SI quantity |
|-----------|-----------|-----------|-------------|------|-------------|
| `u1` | `ux` | `ur` | `ur` | stored | velocity |
| `u2` | `uy` | `utheta` | `uphi` | stored | velocity |
| `u3` | `uz` | `uphi` | `uz` | stored | velocity |

## Densities

| Canonical | Alias | Type | SI quantity |
|-----------|-------|------|-------------|
| `n_s0` | `n_e` | stored | density |
| `n_s1` | `n_i` | stored | density |
| `rho_c` | — | stored | charge_density |
| `rho_m` | — | stored | mass_density |

`n_e`/`n_i` are convenience aliases for the common two-species case.
For more species, use `n_s2`, `n_s3`, etc. (no aliases).

## Pressure & Temperature

| Canonical | Type | SI quantity |
|-----------|------|-------------|
| `P` | stored | pressure |
| `Pe` | stored | pressure |
| `Pi` | stored | pressure |
| `Te` | stored | temperature |
| `Ti` | stored | temperature |

### Pressure Tensor (PIC)

| Canonical | Type | SI quantity |
|-----------|------|-------------|
| `P11` | stored | pressure |
| `P22` | stored | pressure |
| `P33` | stored | pressure |
| `P12` | stored | pressure |
| `P13` | stored | pressure |
| `P23` | stored | pressure |
| `P_par` | derived | pressure |
| `P_perp` | derived | pressure |
| `agyrotropy` | derived | dimensionless |

## Plasma Parameters

| Canonical | Alias | Description | SI quantity |
|-----------|-------|-------------|-------------|
| `beta` | — | Plasma beta ($2P/B^2$) | dimensionless |
| `beta_e` | — | Electron beta | dimensionless |
| `beta_i` | — | Ion beta | dimensionless |
| `v_A` | — | Alfvén speed | velocity |
| `c_s` | — | Sound speed (MHD) | velocity |
| `c_ia` | — | Ion acoustic speed | velocity |
| `v_ms` | — | Fast magnetosonic speed | velocity |
| `M_A` | — | Alfvén Mach number | dimensionless |
| `M_ms` | — | Magnetosonic Mach number | dimensionless |

## Energy Densities

| Canonical | Description | SI quantity |
|-----------|-------------|-------------|
| `e_B` | Magnetic energy density ($B^2/2$) | energy_density |
| `e_E` | Electric energy density ($E^2/2$) | energy_density |
| `e_k` | Kinetic energy density ($\rho_m V^2/2$) | energy_density |
| `e_th` | Thermal energy density ($P/(\gamma-1)$) | energy_density |

## Thermodynamic Quantities

| Canonical | Description | SI quantity |
|-----------|-------------|-------------|
| `h` | Specific enthalpy | temperature |
| `h_rel` | Relativistic specific enthalpy | temperature |
| `e_int` | Specific internal energy | temperature |
| `s` | MHD entropy ($\ln(P/\rho_m^\gamma)$) | dimensionless |
| `s_e` | Electron entropy ($\ln(P_e/n_{s0}^\gamma)$) | dimensionless |
| `s_i` | Ion entropy ($\ln(P_i/n_{s1}^\gamma)$) | dimensionless |
| `s_gyro` | Electron gyrotropic entropy | dimensionless |
| `s_gyro_i` | Ion gyrotropic entropy | dimensionless |
| `s_gyro_e` | → `s_gyro` (alias) | dimensionless |

## Poynting Flux

| Canonical | Cartesian alias | Compute alias | SI quantity |
|-----------|----------------|---------------|-------------|
| `S1` | (field alias) `Sx` | (compute alias) `Sx` | poynting_flux |
| `S2` | `Sy` | `Sy` | poynting_flux |
| `S3` | `Sz` | `Sz` | poynting_flux |

## Species-Dependent Scales

### Electrons (species index 0)

| Canonical | Description | SI quantity |
|-----------|-------------|-------------|
| `omega_pe` | Plasma frequency | frequency |
| `omega_ce` | Cyclotron frequency (positive) | frequency |
| `d_e` | Skin depth | length |
| `v_th_e` | Thermal speed (NRL: $\sqrt{T_e/m_e}$) | velocity |
| `r_e` | Thermal gyroradius | length |
| `lambda_D` | Debye length | length |

### Ions (species index 1)

| Canonical | Description | SI quantity |
|-----------|-------------|-------------|
| `omega_pi` | Plasma frequency | frequency |
| `omega_ci` | Cyclotron frequency (positive) | frequency |
| `d_i` | Skin depth | length |
| `v_th_i` | Thermal speed (NRL: $\sqrt{T_i/m_i}$) | velocity |
| `r_i` | Thermal gyroradius | length |

## Differential Operators / Diagnostics

| Canonical | Compute alias | Description | SI quantity |
|-----------|---------------|-------------|-------------|
| `div_B` | — | $\nabla \cdot \mathbf{B}$ | b_field (÷ length) |
| `div_E` | — | $\nabla \cdot \mathbf{E}$ | e_field (÷ length) |
| `curl_B1` | `curl_Bx` | $(\nabla \times \mathbf{B})_1$ | b_field (÷ length) |
| `curl_B2` | `curl_By` | $(\nabla \times \mathbf{B})_2$ | b_field (÷ length) |
| `curl_B3` | `curl_Bz` | $(\nabla \times \mathbf{B})_3$ | b_field (÷ length) |
| `vort1` | `vort_x` | Fluid vorticity component 1 | frequency |
| `vort2` | `vort_y` | Fluid vorticity component 2 | frequency |
| `vort3` | `vort_z` | Fluid vorticity component 3 | frequency |
| `|vort|` | — | Vorticity magnitude | frequency |

## Planned (not yet implemented)

| Canonical | Description | Step |
|-----------|-------------|------|
| `gamma_L` | Bulk Lorentz factor | 18 |
| `sigma` | Magnetization ($B^2/\rho_m c^2$) | 18 |

## Alias Summary

Three alias mechanisms exist, at different layers:

1. **Geometry aliases** (FieldDataset) — coordinate-letter suffixes for
   vector field components. Active set depends on `geometry`:
   Cartesian `Bx`→`B1`, Spherical `Br`→`B1`, Cylindrical `Br`→`B1`.
   Applies to prefixes: `B`, `E`, `J`, `V`, `v`, `Ve`, `S`, `u`.

2. **Species aliases** (FieldDataset) — `n_e`→`n_s0`, `n_i`→`n_s1`.
   Always active when the target field exists.

3. **Compute aliases** (compute.py) — alternative names for derived
   quantities: `curl_Bx`→`curl_B1`, `vort_x`→`vort1`, `Sx`→`S1`,
   `s_gyro_e`→`s_gyro`.
