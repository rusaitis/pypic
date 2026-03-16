# Physics Equations Reference

Complete equation reference for all derived quantities in pypic.
For canonical field names and data types, see [SCHEMA.md](../SCHEMA.md).
For detailed convention discussions, see [conventions.md](conventions.md).

**Normalization:** All computation uses code (normalized) units. In PIC
normalization, $\mu_0 = \epsilon_0 = 1$. In MHD normalization, $\mu_0 = 1$.
Physical constants vanish from the normalized forms, reappearing only at
SI conversion boundaries. Temperatures are in energy units throughout
($T = P/n$, not $T = P/(nk_B)$); divide by $k_B$ to convert to Kelvin.

## References

- **[NRL]** — NRL Plasma Formulary (Richardson 2019)
- **[Chen]** — Chen, *Introduction to Plasma Physics and Controlled Fusion*
- **[Fitz]** — Fitzpatrick, *Plasma Physics: An Introduction*
- **[CGL]** — Chew, Goldberger & Low (1956), double-adiabatic theory
- **[Jack]** — Jackson, *Classical Electrodynamics*


## 1. Densities and Moments

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `n_s` | Number density (species $s$) | $n_s / n_{ref}$ | -- |
| `rho_c` | Charge density | $\sum_s n_s q_s$ | -- |
| `rho_m` | Mass density | $\sum_s n_s m_s$ | -- |
| `J` | Current density | $\sum_s n_s q_s \mathbf{V}_s$ | -- |
| `V` | Ion bulk velocity | 1st moment of $f_i$ (PIC) / fluid velocity (MHD) | -- |
| `Ve` | Electron bulk velocity | 1st moment of $f_e$ | -- |
| `P` | Total scalar pressure | $P_e + P_i$ (PIC) / fluid pressure (MHD) | -- |
| `Pe` | Electron scalar pressure | $P_e = n_e T_e$ | -- |
| `Pi` | Ion scalar pressure | $P_i = n_i T_i$ | -- |
| `Te` | Electron temperature | $T_e = P_e / n_e$ | $T_e^{SI} = T_e \cdot m_{ref} v_{ref}^2$ |
| `Ti` | Ion temperature | $T_i = P_i / n_i$ | $T_i^{SI} = T_i \cdot m_{ref} v_{ref}^2$ |


## 2. Thermodynamic Quantities

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `h` | Specific enthalpy | $\gamma P / ((\gamma - 1) \rho_m)$ | -- |
| `h_rel` | Relativistic specific enthalpy | $c^2 + \gamma P / ((\gamma - 1) \rho_m)$ | -- |
| `s` | Specific entropy (isotropic) | $\ln(P / \rho_m^\gamma)$ (MHD), $\ln(P_s / n_s^\gamma)$ (PIC) | -- |
| `s_e` | Electron entropy | $\ln(P_e / n_e^\gamma)$ | -- |
| `s_i` | Ion entropy | $\ln(P_i / n_i^\gamma)$ | -- |
| `s_gyro` | Gyrotropic entropy | $\ln(P_{\parallel,s} P_{\perp,s}^2 / n_s^5)$ | -- |
| `e_int` | Specific internal energy | $P / ((\gamma - 1) \rho_m)$ | -- |
| `gamma_eos` | Adiabatic index | $\gamma = c_p / c_v$ | -- |

[1] MHD entropy uses mass density ($\rho_m$, fluid equation of state);
PIC entropy uses per-species number density ($n_s$, kinetic moments per
particle), with $\gamma = 5/3$ (3D) by default. The gyrotropic exponent
of 5 is independent of $\gamma$ — it arises from the CGL double-adiabatic
invariants, not from the equation of state.
See [conventions.md § γ Convention](conventions.md#γ-convention-for-pic-entropy).

[2] `h_rel` uses the constant-$\Gamma$ (Synge-type) approximation;
see Mignone & McKinney (2007) for the variable-$\Gamma$ treatment.


## 3. Energy and Flux

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `S` | Poynting flux | $\mathbf{E} \times \mathbf{B}$ | $\mathbf{E} \times \mathbf{B} / \mu_0$ |
| `e_B` | Magnetic energy density | $B^2 / 2$ | $B^2 / (2\mu_0)$ |
| `e_E` | Electric energy density | $E^2 / 2$ | $\epsilon_0 E^2 / 2$ |
| `e_k` | Kinetic energy density | $\frac{1}{2}\rho_m V^2$ | -- |
| `e_th` | Thermal energy density | $P / (\gamma - 1)$ | -- |


## 4. Pressure Tensor

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `P_par` | Parallel pressure | $P_\parallel = \hat{b} \cdot \mathbf{P} \cdot \hat{b}$ | -- |
| `P_perp` | Perpendicular pressure | $P_\perp = (\mathrm{Tr}(\mathbf{P}) - P_\parallel) / 2$ | -- |
| `Pij` | Full pressure tensor | 6 independent components: P11, P12, P13, P22, P23, P33 | -- |
| `agyrotropy` | Agyrotropy measure | $Q$ (deviation from gyrotropic symmetry) | -- |


## 5. Characteristic Scales

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `d_e` | Electron skin depth | $c / \omega_{pe}$ | -- |
| `d_i` | Ion skin depth | $c / \omega_{pi}$ | -- |
| `r_e` | Electron thermal gyroradius | $v_{th,e} / \omega_{ce}$ | -- |
| `r_i` | Ion thermal gyroradius | $v_{th,i} / \omega_{ci}$ | -- |
| `omega_pe` | Electron plasma frequency | $\sqrt{n_e q_e^2 / m_e}$ | $\sqrt{n_e e^2 / (\epsilon_0 m_e)}$ |
| `omega_pi` | Ion plasma frequency | $\sqrt{n_i q_i^2 / m_i}$ | $\sqrt{n_i Z^2 e^2 / (\epsilon_0 m_i)}$ |
| `omega_ce` | Electron cyclotron freq. [3] | $\|q_e\| B / m_e$ | $\|e\| B / m_e$ |
| `omega_ci` | Ion cyclotron freq. [3] | $|q_i| B / m_i$ | $Z e B / m_i$ |
| `lambda_D` | Electron Debye length [4] | $\sqrt{T_e / (n_e q_e^2)}$ | $\sqrt{\epsilon_0 T_e / (n_e e^2)}$ |
| `v_A` | Alfvén speed [NRL] | $B / \sqrt{\rho_m}$ | $B / \sqrt{\mu_0 \rho_m}$ |
| `v_th_e` | Electron thermal speed [5] | $\sqrt{T_e / m_e}$ | -- |
| `v_th_i` | Ion thermal speed [5] | $\sqrt{T_i / m_i}$ | -- |
| `c_s` | Sound speed [6] | $\sqrt{\gamma P / \rho_m}$ | -- |
| `v_ms` | Fast magnetosonic speed [7] | $\sqrt{v_A^2 + c_s^2}$ | -- |
| `M_A` | Alfvén Mach number | $V / v_A$ | -- |
| `M_ms` | Magnetosonic Mach number | $V / v_{ms}$ | -- |
| `beta` | Plasma beta | $2P / B^2$ | $2\mu_0 P / B^2$ |
| `beta_e` | Electron beta | $2P_e / B^2$ | $2\mu_0 P_e / B^2$ |
| `beta_i` | Ion beta | $2P_i / B^2$ | $2\mu_0 P_i / B^2$ |

[3] Cyclotron frequencies are positive by convention (magnitudes).
See [conventions.md § Cyclotron Frequency](conventions.md#cyclotron-frequency-convention). [NRL], [Chen]

[4] `lambda_D` is specifically the *electron* Debye length.
See [conventions.md § Debye Length](conventions.md#debye-length). [NRL]

[5] NRL convention: $v_{th} = \sqrt{T/m}$ (1D Maxwellian standard deviation).
The $\sqrt{2T/m}$ convention (most probable speed) differs by $\sqrt{2}$ and
propagates into gyroradius. See [conventions.md § Thermal Speed](conventions.md#thermal-speed-convention). [NRL]

[6] MHD sound speed; differs from ion acoustic speed $c_{ia}$.
See [conventions.md § Sound Speed vs Ion Acoustic](conventions.md#sound-speed-vs-ion-acoustic-speed). [Chen]

[7] Maximum fast-mode phase speed (perpendicular propagation, $\theta = 90°$).
The general dispersion relation is angle-dependent.
See [conventions.md § Fast Magnetosonic Speed](conventions.md#fast-magnetosonic-speed). [Fitz]


## 6. Magnitudes and Differential Operators

| Name | Description | Normalized |
|------|-------------|------------|
| `\|B\|` | Magnetic field magnitude | $\sqrt{B_1^2 + B_2^2 + B_3^2}$ |
| `\|E\|` | Electric field magnitude | $\sqrt{E_1^2 + E_2^2 + E_3^2}$ |
| `\|J\|` | Current density magnitude | $\sqrt{J_1^2 + J_2^2 + J_3^2}$ |
| `\|V\|` | Bulk velocity magnitude | $\sqrt{V_1^2 + V_2^2 + V_3^2}$ |
| `div_B` | Divergence of B | $\nabla \cdot \mathbf{B}$ (should be ~0) |
| `div_E` | Divergence of E | $\nabla \cdot \mathbf{E}$ ($= \rho_c$ normalized; $= \rho_c / \epsilon_0$ SI) |
| `curl_B` | Curl of B | $\nabla \times \mathbf{B}$ ($\propto \mathbf{J}$ in MHD) |
| `vort` | Fluid vorticity | $\nabla \times \mathbf{V}$ |
| `\|vort\|` | Vorticity magnitude | $\sqrt{\mathrm{vort}_1^2 + \mathrm{vort}_2^2 + \mathrm{vort}_3^2}$ |

Differential operators use second-order central finite differences
(Cartesian). Spherical and cylindrical geometries include metric factors.
