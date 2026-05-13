# Physics Equations Reference

Complete equation reference for all derived quantities in pypic.
For canonical field names and data types, see [SCHEMA](schema.md).
For detailed convention discussions, see [conventions.md](conventions.md).

**Normalization:** All computation uses code (normalized) units. In PIC
normalization, $\mu_0 = \epsilon_0 = 1$. In MHD normalization, $\mu_0 = 1$.
Physical constants vanish from the normalized forms, reappearing only at
SI conversion boundaries. Temperatures are in energy units throughout
($T = P/n$, not $T = P/(nk_B)$); divide by $k_B$ to convert to Kelvin.

### SI conversion factors

To convert a quantity from code units to SI: $x_{SI} = x_{code} \times f$,
where $f$ is the SI factor for that quantity type. The eight **primitive
references** ($n_{ref}$, $m_{ref}$, $q_{ref}$, $v_{ref}$, $l_{ref}$,
$t_{ref}$, $B_{ref}$, $E_{ref}$) are set by the normalization system
(PIC, MHD, or custom). They normalize the primitive canonical fields
($n_s$, plus per-species $m_s$/$q_s$, $\mathbf{V}$, $\mathbf{B}$,
$\mathbf{E}$). SI factors for compound canonical fields ($\rho_m$,
$\rho_c$, $\mathbf{J}$, $P$, energy and flux densities) are products
of the primitives — there is no $\rho_{ref}$ because $\rho_m$ and
$\rho_c$ are derived ($\sum_s n_s m_s$ and $\sum_s n_s q_s$), not
primitive.

| Quantity type | SI factor $f$ | SI unit | Used by |
|---------------|---------------|---------|---------|
| `density` | $n_{ref}$ | m$^{-3}$ | $n_e$, $n_i$ |
| `mass_density` | $n_{ref} m_{ref}$ | kg/m$^3$ | $\rho_m$ |
| `charge_density` | $q_{ref} n_{ref}$ | C/m$^3$ | $\rho_c$ |
| `velocity` | $v_{ref}$ | m/s | $V$, $v_A$, $v_{th}$ |
| `b_field` | $B_{ref}$ | T | $B$ |
| `e_field` | $E_{ref}$ | V/m | $E$ |
| `pressure` | $n_{ref} m_{ref} v_{ref}^2$ | Pa | $P$, $P_e$, $P_i$ [^aliases] |
| `temperature` | $m_{ref} v_{ref}^2$ | J | $T$ (energy units) |
| `energy_density` | $n_{ref} m_{ref} v_{ref}^2$ | J/m$^3$ | $e_k$, $e_{th}$, $e_B$ |
| `specific_energy` | $v_{ref}^2$ | J/kg | $h$, $e_{int}$ |
| `current_density` | $q_{ref} n_{ref} v_{ref}$ | A/m$^2$ | $J$ |
| `frequency` | $1/t_{ref}$ | rad/s | $\omega_p$, $\omega_c$ |
| `length` | $l_{ref}$ | m | $d_e$, $d_i$, $r_i$ |
| `poynting_flux` | $E_{ref} B_{ref}$ | W/m$^2$ | $S$ (EM flux) |
| `energy_flux` | $n_{ref} m_{ref} v_{ref}^3$ | W/m$^2$ | EF, KEF, HF, EHF, $q$ |
| `power_density` | $n_{ref} m_{ref} v_{ref}^2 / t_{ref}$ | W/m$^3$ | $J \cdot E$ |

`poynting_flux` and `energy_flux` share SI units (W/m$^2$) but differ
in normalization: EM flux scales with field references ($E_{ref} B_{ref}$),
particle energy flux with matter references ($n_{ref} m_{ref} v_{ref}^3$). In
code units where $\mu_0 = 1$ these are equivalent; in SI the factor of
$\mu_0$ separates them.

## 1. Densities and Moments

Per-species e/i suffix names throughout this document are pypic
library-side aliases [^aliases]. Two stylistic patterns appear,
forced by identifier readability:

- **Uppercase prefix → no underscore:** `Pe`, `Pi`, `Ve`, `Vi`,
  `Te`, `Ti`, and the multi-letter forms `EFe`, `KEFi`, `HFi`,
  `EHFe` — alias the canonical `_s0` / `_s1` per-species form.
- **Lowercase prefix → underscore mandatory:** `n_e`, `n_i`, `s_e`,
  `s_i`, `beta_e`, `beta_i`, `q_e`, `q_i`, `P_par_e`, `s_gyro_e`,
  ... — same alias relationship, written with `_e` / `_i` because
  `betae` / `se` would be unreadable.

The cross-tool canonical form uses 0-based species indices: `P_s{N}`,
`V_s{N}`, `T_s{N}`, `n_s{N}`, `s_s{N}`, `beta_s{N}`, `q_s{N}` (see
[schema.md § Per-species
naming](schema.md#fluid--moment-quantities--densities)).
See [Aliases](aliases.md) for the full registration list.

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `n_s` | Number density (species $s$) | $n_s / n_{ref}$ | -- |
| `rho_c` | Charge density | $\sum_s n_s q_s$ | -- |
| `rho_m` | Mass density | $\sum_s n_s m_s$ | -- |
| `J` | Current density | $\sum_s n_s q_s \mathbf{V}_s$ | -- |
| `V` | Ion/fluid bulk velocity | 1st moment of $f_i$ (kinetic) or fluid velocity | -- |
| `Ve` | Electron bulk velocity | 1st moment of $f_e$ | -- |
| `P` | Total scalar pressure | $P_e + P_i$ or $\frac{1}{3}\mathrm{Tr}(\mathbf{P})$ or fluid $P$ | -- |
| `Pe` | Electron scalar pressure | $P_e = n_e T_e$ | -- |
| `Pi` | Ion scalar pressure | $P_i = n_i T_i$ | -- |
| `T` | Temperature (generic) | $T = P / n$ | $T^{SI} = T \cdot m_{ref} v_{ref}^2$ |
| `Te` | Electron temperature | $T_e = P_e / n_e$ | $T_e^{SI} = T_e \cdot m_{ref} v_{ref}^2$ |
| `Ti` | Ion temperature | $T_i = P_i / n_i$ | $T_i^{SI} = T_i \cdot m_{ref} v_{ref}^2$ |

[^aliases]: pypic library-side convenience aliases bound to the
    canonical `_s{N}` names at `FieldDataset` construction time
    (`Pe ↔ P_s0`, `Pi ↔ P_s1`, `Ve ↔ V_s0`, …) for the common
    two-species electron/ion case. They are **not** part of the
    cross-tool v1.0 contract — non-pypic consumers (Rust, JS) read
    only the numbered canonical names from disk. See
    [Aliases](aliases.md) for the full registration list.


## 2. Thermodynamic Quantities

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `h` | Specific enthalpy | $\gamma P / ((\gamma - 1) \rho_m)$ | $h \cdot v_{ref}^2$ \[J/kg\] |
| `h_rel` | Relativistic specific enthalpy[^2] | $c^2 + \gamma P / ((\gamma - 1) \rho_m)$ | $h_{rel} \cdot v_{ref}^2$ \[J/kg\] |
| `s` | Specific entropy (single-fluid, isotropic)[^1] | $\ln(P / \rho_m^\gamma)$ | -- |
| `s_e` | Electron entropy[^aliases] | $\ln(P_e / n_e^\gamma)$ | -- |
| `s_i` | Ion entropy | $\ln(P_i / n_i^\gamma)$ | -- |
| `s_gyro` | Gyrotropic entropy | $\ln(P_{\parallel,s} P_{\perp,s}^2 / n_s^5)$ | -- |
| `e_int` | Specific internal energy | $P / ((\gamma - 1) \rho_m)$ | $e_{int} \cdot v_{ref}^2$ \[J/kg\] |
| `gamma_eos` | Adiabatic index | $\gamma = c_p / c_v$ | -- |

[^1]: The fluid form uses mass density ($\rho_m$, single-fluid equation of
    state); the per-species form uses number density ($n_s$, kinetic moments
    per particle), with $\gamma = 5/3$ (3D) by default. The gyrotropic exponent
    of 5 is independent of $\gamma$ — it arises from the CGL double-adiabatic
    invariants, not from the equation of state.
    See [conventions.md § γ Convention](conventions.md#gamma-convention-for-pic-entropy).

[^2]: `h_rel` uses the constant-$\Gamma$ (Synge-type) approximation;
    see [@MigMc] for the variable-$\Gamma$ treatment.


## 3. Energy and Flux

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `S` | Poynting flux | $\mathbf{E} \times \mathbf{B}$ | $\mathbf{E} \times \mathbf{B} / \mu_0$ |
| `e_B` | Magnetic energy density | $B^2 / 2$ | $B^2 / (2\mu_0)$ |
| `e_E` | Electric energy density | $E^2 / 2$ | $\epsilon_0 E^2 / 2$ |
| `e_k` | Kinetic energy density | $\frac{1}{2}\rho_m V^2$ | -- |
| `e_th` | Thermal energy density | $P / (\gamma - 1)$ | -- |
| `e_th_trace` | Thermal energy density (tensor) | $\frac{1}{2}\mathrm{Tr}(\mathbf{P}) = \frac{1}{2}(P_{11}+P_{22}+P_{33})$ | -- |

### Energy flux decomposition

The total energy flux per species (third-order velocity moment) decomposes
into kinetic, enthalpy, and conductive heat flux contributions:

$$\mathbf{EF}_s = \underbrace{\tfrac{1}{2} n_s m_s |\mathbf{V}_s|^2 \mathbf{V}_s}_{\mathbf{KEF}_s} + \underbrace{\tfrac{\gamma}{\gamma-1} P_s \mathbf{V}_s}_{\mathbf{EHF}_s} + \underbrace{\mathbf{q}_s}_{\text{heat flux}}$$

| Name | Description | Normalized | Available when |
|------|-------------|------------|----------------|
| `EF` | Total energy flux[^ef] | $\tfrac{1}{2} m \int f\, \mathbf{v}\, v^2\, d^3v$ | EF in output (PIC, multi-moment MHD) |
| `KEF` | Kinetic energy flux | $\tfrac{1}{2} n\, m\, |\mathbf{V}|^2\, \mathbf{V}$ | Any code with $n$, $m$, $\mathbf{V}$ |
| `HF` | Total thermal flux | $\mathbf{EF} - \mathbf{KEF}$ | EF in output |
| `EHF` | Enthalpy flux (adiabatic) | $\tfrac{\gamma}{\gamma-1} P\, \mathbf{V}$ | Any code with $P$, $\mathbf{V}$ |
| `q` | Conductive heat flux | $\mathbf{HF} - \mathbf{EHF}$ | EF in output |

[^ef]: iPIC3D outputs EF per species as a deposited moment. In ideal MHD
    ($\mathbf{q} = 0$), the total energy flux equals $\mathbf{KEF} + \mathbf{EHF}$.
    The conductive heat flux $\mathbf{q}$ captures non-Maxwellian and non-adiabatic
    transport — it is the physically interesting residual in reconnection exhausts,
    shocks, and turbulence. All quantities exist as both total (MHD: `EHF_1`,
    `EHF_2`, `EHF_3`) and per-species (PIC: `KEF_s0_1`, `HFi`[^aliases], `q_i`).
    **Precision notes:** `HF = EF - KEF` is exact (no closure assumption).
    `EHF` uses the scalar (isotropic) pressure; for anisotropic plasmas the
    exact enthalpy flux involves the full pressure tensor
    ($EHF_i = P_{ij} V_j + \tfrac{1}{2} P_{jj} V_i$). The isotropic form
    is a useful reference — discrepancies between `HF` and `EHF` reflect
    *both* anisotropy and heat conduction effects.


## 4. Pressure Tensor and Field-Aligned Decomposition

### 4.1 Pressure tensor

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `P` | Isotropic scalar pressure[^9] | $P = \frac{1}{3}\mathrm{Tr}(\mathbf{P}) = \frac{1}{3}(P_{11} + P_{22} + P_{33})$ | -- |
| `P_par` | Parallel pressure | $P_\parallel = \hat{b} \cdot \mathbf{P} \cdot \hat{b}$ | -- |
| `P_perp` | Perpendicular pressure | $P_\perp = (\mathrm{Tr}(\mathbf{P}) - P_\parallel) / 2$ | -- |
| `Pij` | Full pressure tensor | 6 independent components: P_11, P_12, P_13, P_22, P_23, P_33 | -- |
| `agyrotropy` | Agyrotropy measure[^Q] | $Q = \sqrt{1 - 4 I_2 / [(I_1 - P_\parallel)(I_1 + 3 P_\parallel)]}$ | -- |

### 4.2 Field-aligned vector decomposition

Projects a vector $\mathbf{A}$ onto $\hat{b} = \mathbf{B}/|\mathbf{B}|$.
Generic form, same algebra for $\mathbf{J}$, $\mathbf{V}$,
$\mathbf{E}$, and the non-ideal residual
$\mathbf{E}' = \mathbf{E} + \mathbf{V}\times\mathbf{B}$.  All returns
are NaN where $|B| = 0$.

| Name | Description | Normalized |
|------|-------------|------------|
| `J_par`, `V_par`, `E_par` | Signed parallel projection | $A_\parallel = \mathbf{A}\cdot\hat{b}$ |
| `J_perp_1/2/3`, `V_perp_1/2/3`, `E_perp_1/2/3` | Perpendicular vector components | $\mathbf{A}_\perp = \mathbf{A} - A_\parallel\hat{b}$ |
| `\|J_perp\|`, `\|V_perp\|`, `\|E_perp\|` | Perpendicular magnitude[^par_perp_id] | $\sqrt{\|\mathbf{A}\|^2 - A_\parallel^2}$ |
| `V_s{N}_par`, `V_s{N}_perp_{1,2,3}`, `\|V_s{N}_perp\|` | Per-species velocity decomposition | same form on `V_s{N}_{1,2,3}` |
| `E_prime_par` | Field-aligned non-ideal residual[^reconn_rate] | $(\mathbf{E} + \mathbf{V}\times\mathbf{B}) \cdot \hat{b}$ |
| `\|E_prime_perp\|` | Perpendicular non-ideal residual magnitude | $\sqrt{\|\mathbf{E}'\|^2 - E'^2_\parallel}$ |
| `E_ideal_par`, `E_Hall_par` | Field-aligned ideal-MHD / Hall field[^cross_perp] | $\equiv 0$ (cross product $\perp$ $\mathbf{B}$) |
| `E_ideal_perp_{1,2,3}`, `E_Hall_perp_{1,2,3}` | Perpendicular ideal-MHD / Hall field | equal the full vectors up to roundoff |
| `\|E_ideal_perp\|`, `\|E_Hall_perp\|` | Perpendicular magnitudes | $= \|\mathbf{E}^{\mathrm{ideal}}\|$, $\|\mathbf{E}^{\mathrm{Hall}}\|$ up to roundoff |

[^par_perp_id]: Pythagorean identity holds exactly:
    $\|\mathbf{A}\|^2 = A_\parallel^2 + \|\mathbf{A}_\perp\|^2$. The
    perpendicular magnitude uses this directly — half the temporaries
    of materializing the three perpendicular components.

[^reconn_rate]: $E'_\parallel$ quantifies frozen-in flux violation:
    it is zero in ideal MHD and non-zero only where the non-ideal
    terms in Ohm's law (resistivity, electron inertia, pressure
    divergence) break the frozen-in condition.  In 2D it equals
    $\partial\psi/\partial t$ at the X-point — the standard
    reconnection-rate quantity.  This identity assumes a finite
    out-of-plane guide field so $\hat{b}$ at the null points out of
    plane and $E'_\parallel = E_z$; in the strict anti-parallel limit
    $\hat{b}$ is undefined at the magnetic null and $E'_\parallel$ is
    NaN (consistent with the $|\mathbf{B}| = 0$ handling).
    Cross-reference §9 footnote 13. [@Birn; @Hesse].

[^cross_perp]: $\mathbf{E}^{\mathrm{ideal}} = -\mathbf{V}\times\mathbf{B}$
    and $\mathbf{E}^{\mathrm{Hall}} \propto \mathbf{J}\times\mathbf{B}$
    are cross products with $\mathbf{B}$, hence orthogonal to
    $\mathbf{B}$ by the scalar triple product identity.
    `E_ideal_par` and `E_Hall_par` therefore evaluate to zero up to
    floating-point roundoff.  Kept in the registry for diagnostic
    symmetry with `E_par` / `E_prime_par` and as a numerical-precision
    check on the cross-product implementation in the destaggered
    co-located grid.  The underlying scalar projection $A_\parallel =
    \mathbf{A}\cdot\hat{b}$ and rejection $\mathbf{A}_\perp =
    \mathbf{A} - A_\parallel\hat{b}$ are introduced as part of the
    single-particle gyromotion decomposition [@Chen].

[^Q]: Swisdak's gyrotropy measure [@Swisdak2016], computed from the
    first two invariants of the pressure tensor:
    $I_1 = \mathrm{Tr}(\mathbf{P}) = P_{11} + P_{22} + P_{33}$ and
    $I_2 = P_{11}P_{22} + P_{11}P_{33} + P_{22}P_{33} - P_{12}^2 -
    P_{13}^2 - P_{23}^2$. Bounded $Q \in [0, 1]$: $Q = 0$ for a
    perfectly gyrotropic plasma, $Q \to 1$ at maximal agyrotropy.
    Frame-invariant (built from tensor invariants and the magnetic-
    field-aligned scalar $P_\parallel = \hat{b} \cdot \mathbf{P}
    \cdot \hat{b}$), so the value follows whatever $\hat{b}$ is in
    the current frame. Alternatives in the literature: Scudder's
    $A\phi$ [@Scudder2008] and Aunai's $D_{ng}$ [@Aunai2013] — pypic
    standardizes on $Q$ for its closed form and bounded range, and
    also ships both alternatives as ``A_phi`` and ``D_ng``
    (per-species ``A_phi_s{N}`` / ``D_ng_s{N}``, with ``_e`` / ``_i``
    aliases) for literature comparisons. See §9.

[^9]: The trace $\mathrm{Tr}(\mathbf{P})$ is a coordinate invariant (first
    invariant of the symmetric tensor), so $P = \mathrm{Tr}(\mathbf{P})/3$
    gives the same scalar regardless of axis orientation. Identity:
    $P = (P_\parallel + 2\,P_\perp)/3$. When `P` is not directly available
    in the dataset but the pressure tensor is, `compute("P")` falls back to
    this definition. Per-species scalar pressures (canonical
    `P_s{N}`; pypic aliases `Pe`, `Pi`) use the same trace formula
    on the per-species tensor.


## 5. Characteristic Scales

The Tier-3 `<field>_s<N>` form is the canonical recipe ID; the NRL
Plasma Formulary spelling (`omega_pe`, `v_th_e`, `lambda_D`, ...)
remains a registered alias and the human-friendly form throughout
this document.[^scales-aliases]

| Canonical | NRL alias | Description | Normalized | SI |
|-----------|-----------|-------------|------------|-----|
| `d_s0` | `d_e` | Electron skin depth | $c / \omega_{pe}$ | -- |
| `d_s1` | `d_i` | Ion skin depth | $c / \omega_{pi}$ | -- |
| `r_s0` | `r_e` | Electron thermal gyroradius | $v_{th,e} / \omega_{ce}$ | -- |
| `r_s1` | `r_i` | Ion thermal gyroradius | $v_{th,i} / \omega_{ci}$ | -- |
| `omega_p_s0` | `omega_pe` | Electron plasma frequency | $\sqrt{n_e q_e^2 / m_e}$ | $\sqrt{n_e e^2 / (\epsilon_0 m_e)}$ |
| `omega_p_s1` | `omega_pi` | Ion plasma frequency | $\sqrt{n_i q_i^2 / m_i}$ | $\sqrt{n_i Z^2 e^2 / (\epsilon_0 m_i)}$ |
| `omega_c_s0` | `omega_ce` | Electron cyclotron freq.[^3] | $\|q_e\| B / m_e$ | $\|e\| B / m_e$ |
| `omega_c_s1` | `omega_ci` | Ion cyclotron freq.[^3] | $|q_i| B / m_i$ | $Z e B / m_i$ |
| `lambda_D_s0` | `lambda_D` | Electron Debye length[^4] | $\sqrt{T_e / (n_e q_e^2)}$ | $\sqrt{\epsilon_0 T_e / (n_e e^2)}$ |
| `v_A` | — | Alfvén speed [@NRL] | $B / \sqrt{\rho_m}$ | $B / \sqrt{\mu_0 \rho_m}$ |
| `v_th_s0` | `v_th_e` | Electron thermal speed[^5] | $\sqrt{T_e / m_e}$ | -- |
| `v_th_s1` | `v_th_i` | Ion thermal speed[^5] | $\sqrt{T_i / m_i}$ | -- |
| `c_s` | — | Sound speed[^6] | $\sqrt{\gamma P / \rho_m}$ | -- |
| `c_ia` | — | Ion acoustic speed[^6] | $\sqrt{(\gamma_e T_e + \gamma_i T_i) / m_i}$ | -- |
| `v_ms` | — | Fast magnetosonic speed[^7] | $\sqrt{v_A^2 + c_s^2}$ | -- |
| `M_A` | — | Alfvén Mach number | $V / v_A$ | -- |
| `M_ms` | — | Magnetosonic Mach number | $V / v_{ms}$ | -- |
| `beta` | — | Plasma beta | $2P / B^2$ | $2\mu_0 P / B^2$ |
| `beta_s0` | `beta_e`[^aliases] | Electron beta | $2P_e / B^2$ | $2\mu_0 P_e / B^2$ |
| `beta_s1` | `beta_i` | Ion beta | $2P_i / B^2$ | $2\mu_0 P_i / B^2$ |

[^scales-aliases]: Both spellings resolve to the same array:
    `compute("omega_pe")` and `compute("omega_p_s0")` are
    interchangeable. Multi-species runs (`omega_p_s2`,
    `lambda_D_s3`, ...) synthesize via `_SPECIES_TEMPLATES` on
    demand.

[^3]: Cyclotron frequencies are positive by convention (magnitudes).
    See [conventions.md § Cyclotron Frequency](conventions.md#cyclotron-frequency-convention).
    [@NRL; @Chen]

[^4]: `lambda_D` is specifically the *electron* Debye length.
    See [conventions.md § Debye Length](conventions.md#debye-length). [@NRL]

[^5]: NRL convention: $v_{th} = \sqrt{T/m}$ (1D Maxwellian standard deviation).
    The $\sqrt{2T/m}$ convention (most probable speed) differs by $\sqrt{2}$ and
    propagates into gyroradius.
    See [conventions.md § Thermal Speed](conventions.md#thermal-speed-convention). [@NRL]

[^6]: MHD sound speed; differs from ion acoustic speed $c_{ia}$.
    See [conventions.md § Sound Speed vs Ion Acoustic](conventions.md#sound-speed-vs-ion-acoustic-speed). [@Chen]

[^7]: Maximum fast-mode phase speed (perpendicular propagation, $\theta = 90°$).
    The general dispersion relation is angle-dependent.
    See [conventions.md § Fast Magnetosonic Speed](conventions.md#fast-magnetosonic-speed). [@Fitz]


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
| `grad` | Scalar field gradient | $(\nabla f)_i = \partial f / \partial x_i$ |

Differential operators use second-order central finite differences
(Cartesian)[^9]. Spherical and cylindrical geometries include metric factors.


## 7. Diagnostics and Validation

| Name | Description | Equation |
|------|-------------|----------|
| `l2_relative_error` | Discrete relative L2 norm[^8] | $\varepsilon_{L_2} = \sqrt{\sum_i (a_i - b_i)^2} / \sqrt{\sum_i b_i^2}$ |
| `linf_error` | Absolute max-norm error | $\varepsilon_{L_\infty} = \max_i \|a_i - b_i\|$ |
| `field_difference` | Pointwise signed difference | $\Delta f_i = a_i - b_i$ |
| `field_energy` | Volume integral of a scalar field | $E = \sum_{i,j,k} f_{i,j,k} \cdot \Delta V$ where $\Delta V = \prod_k \Delta x_k$ |
| `div_b` | Divergence of B | $\nabla \cdot \mathbf{B}$ (second-order central differences) |
| `max_div_b` | Maximum absolute div B | $\max \|\nabla \cdot \mathbf{B}\|$ |
| `div_e` | Divergence of E | $\nabla \cdot \mathbf{E}$ (second-order central differences) |
| `spatial_mean` | Unweighted spatial mean | $\langle f \rangle = \frac{1}{N}\sum_i f_i$ (uniform Cartesian grids) |
| `spatial_rms` | Root-mean-square amplitude | $f_{rms} = \sqrt{\langle f^2 \rangle}$ |
| `field_extrema` | Field min/max | $(\min_i f_i,\; \max_i f_i)$ ignoring NaN |

[^8]: L2 norm is discrete and unweighted — volume factors cancel on the
    same grid (LeVeque convention). L∞ is absolute, not relative, to avoid
    division near field nulls.
    See [conventions.md § Error Norms and Divergence](conventions.md#error-norms-and-divergence).

[^9]: **Richardson convergence testing.** For a $p$-th order scheme with
    grid spacing $h$, the truncation error scales as $O(h^p)$. Halving $h$
    reduces the error by a factor of $2^p$: a ratio of ~2 indicates first
    order, ~4 second order, ~8 fourth order. This is the standard method
    for verifying stencil order [@Richardson1911; @LeVeque].
    The convergence tests in `test_operators.py` use this approach with
    sinusoidal fields at three resolutions, checking interior points only
    to avoid the lower-order one-sided boundary stencils.


## 8. Relativistic Corrections

Relativistic corrections[^10] become important in magnetically dominated
plasmas ($\sigma \gg 1$), relativistic reconnection (pair plasmas,
pulsar wind nebulae), and relativistic jets. The existing non-relativistic
formulas (Sections 3, 5) are recovered when $\gamma \to 1$ and
$\sigma \to 0$. See [conventions.md § Relativistic Conventions](conventions.md#relativistic-conventions)
for velocity variable choices and numerical considerations.

[^10]: Several characteristic scales in Section 5 have relativistic
    generalizations listed here. When `physics.relativistic = true` in the
    simulation config, `compute()` (Step 13) should use these forms
    automatically. The non-relativistic limit is always recovered by setting
    $\gamma_L = 1$ (or equivalently $v \ll c$, $\sigma \ll 1$).

### 8.1 Bulk-Flow Quantities

| Name | Description | Normalized | Notes |
|------|-------------|------------|-------|
| `gamma_L` | Lorentz factor (from 3-velocity) | $\gamma = 1/\sqrt{1 - v^2/c^2}$ | bounded $[1, \infty)$ |
| `gamma_L` | Lorentz factor (from 4-velocity) | $\gamma = \sqrt{1 + u^2/c^2}$ | numerically preferred near $v \approx c$ |
| `sigma` | Magnetization [@MigBo] | $B^2 / (\rho_m c^2)$ | $\sigma \gg 1$: magnetically dominated |
| `e_k` (rel.) | Rel. kinetic energy density | $(\gamma - 1)\rho_m c^2$ | recovers $\frac{1}{2}\rho_m v^2$ for $v \ll c$ |
| `v_A` (rel.) | Rel. Alfvén speed [@Lyub] | $c\sqrt{\sigma/(1+\sigma)}$ | $\to c$ as $\sigma \to \infty$; $\to B/\sqrt{\rho_m}$ for $\sigma \ll 1$ |
| `c_s` (rel.) | Rel. sound speed [@MigMc] | $c\sqrt{\gamma_{eos} P / (\rho_m h_{rel})}$ | uses relativistic enthalpy $h_{rel}$ |
| `v_ms` (rel.) | Rel. magnetosonic speed [@MigBo] | $\sqrt{v_A^2 + c_s^2 - v_A^2 c_s^2/c^2}$ | perp. propagation; always $< c$ |

In normalized units where $c = 1$: $\sigma = B^2/\rho_m$,
$v_A = \sqrt{\sigma/(1+\sigma)}$, and the magnetosonic composition
simplifies to $v_{ms}^2 = v_A^2 + c_s^2 - v_A^2 c_s^2$.

### 8.2 Thermal Corrections

Relevant when the thermal energy approaches the rest mass energy
($T \gtrsim mc^2$), as in relativistic pair plasma reconnection
or hot accretion flows.

| Name | Rel. form | Notes |
|------|-----------|-------|
| `omega_c` | $\|q\|B / (\gamma m)$ | $\gamma$ = thermal Lorentz factor; particles gyrate slower |
| `omega_p` | $\omega_p / \sqrt{\langle\gamma\rangle}$ | mean thermal $\langle\gamma\rangle$; reduces effective plasma frequency |
| `v_th` | $v_{th}/\sqrt{1 + v_{th}^2/c^2}$ | practical cap at $c$; proper treatment uses Juttner distribution |

The thermal corrections use the mean thermal Lorentz factor
$\langle\gamma\rangle$, which for a relativistic Maxwellian (Juttner
distribution) depends on the dimensionless temperature
$\Theta = T/(mc^2)$. For $\Theta \ll 1$ the classical results are
recovered; for $\Theta \gg 1$ the ultra-relativistic limit applies.

### 8.3 Which Gamma?

Three distinct Lorentz factors arise in plasma analysis:

- **$\gamma_{bulk}$** — from the fluid (bulk) velocity $\mathbf{V}$.
  This is what `lorentz_factor()` computes from the moment velocity
  fields `V_1/V_2/V_3` or `u_1/u_2/u_3`. Used in relativistic kinetic energy,
  Alfvén speed, and Mach numbers.

- **$\langle\gamma\rangle_{thermal}$** — the mean Lorentz factor of the
  thermal distribution. Relevant for cyclotron and plasma frequency
  corrections (Section 8.2). Not directly available from fluid moments;
  approximated from temperature as $\langle\gamma\rangle \approx 1 +
  (5/2)\Theta$ for $\Theta \ll 1$.

- **$\gamma_{particle}$** — the per-particle Lorentz factor from the
  full distribution function. Only available in full particle data, not
  from the moment-based fields that pypic analyzes. Relevant for
  particle-level diagnostics (energy spectra, acceleration studies)
  but outside the scope of fluid/moment derived quantities.


## 9. Reconnection and Anisotropy Diagnostics

| Name | Description | Normalized | SI |
|------|-------------|------------|-----|
| `J_dot_E` | Energy conversion rate[^11] | $\mathbf{J} \cdot \mathbf{E}$ | $\mathbf{J} \cdot \mathbf{E}$ \[W/m³\] |
| `D_e` | Electron-frame dissipation[^De] | $\gamma_e\bigl[\mathbf{J}\cdot(\mathbf{E}+\mathbf{V}_e\times\mathbf{B}) - \rho_c\,\mathbf{V}_e\cdot\mathbf{E}\bigr]$ | same \[W/m³\] |
| `R_recon` | Local dimensionless reconnection rate[^Rrecon] | $\lvert\mathbf{E}+\mathbf{V}\times\mathbf{B}\rvert / (v_A\,\lvert\mathbf{B}\rvert)$ | dimensionless |
| `E_prime` | Non-ideal electric field[^12] | $\mathbf{E} + \mathbf{V} \times \mathbf{B}$ | same |
| `E_ideal` | Ideal (convective) E field[^12] | $-\mathbf{V} \times \mathbf{B}$ | same |
| `E_Hall` | Hall electric field[^12] | $\mathbf{J} \times \mathbf{B} / (n_e |q_e|)$ | same |
| `D_ng` | Aunai nongyrotropy[^Dng] | $2\,\|\mathbf{N}\|_F / \mathrm{Tr}(\mathbf{P})$ | dimensionless |
| `A_phi` | Scudder agyrotropy[^Aphi] | $\lvert\lambda_1^\perp - \lambda_2^\perp\rvert / (\lambda_1^\perp + \lambda_2^\perp)$ | dimensionless |
| `psi` | Magnetic flux function (2D)[^13] | $-\int B_2\, dx$ | -- |
| `firehose` | Firehose parameter[^14] | $(P_\parallel - P_\perp)/(B^2/2) - 1$ | -- |
| `mirror` | Mirror parameter[^14] | $P_\perp/P_\parallel - 1 - 1/\beta_\perp$ | -- |
| `theta_shear` | Magnetic shear angle[^17] | $\theta = \arccos(\mathbf{B}_a \cdot \mathbf{B}_b / |\mathbf{B}_a||\mathbf{B}_b|)$ | -- |
| `find_saddle_points` | X-point detection[^16] | $\det(H) = \psi_{xx}\psi_{yy} - \psi_{xy}^2 < 0$ | -- |
| `reconnection_rate` | Reconnection rate (2D)[^16] | $R = \partial\psi/\partial t\|_X$ | -- |
| `schindler_xi` | 3D reconnection criterion[^xi] | $\Xi(\mathbf{x}_0) = \int_{\mathcal{L}} E_\parallel\,d\ell$ | -- |

[^11]: Positive $\mathbf{J} \cdot \mathbf{E} > 0$ means particles
    gain energy from fields (electromagnetic-to-kinetic energy conversion).
    Per-species decomposition $\mathbf{J}_s \cdot \mathbf{E}$ identifies
    which species is energized. The energy conversion rate is the source
    term in the Poynting theorem: $\partial u_{EM}/\partial t + \nabla
    \cdot \mathbf{S} = -\mathbf{J} \cdot \mathbf{E}$.
    [@Jack] §6.8, [@Zen].

[^12]: Generalized Ohm's law: $\mathbf{E} = -\mathbf{V} \times
    \mathbf{B} + \frac{\mathbf{J} \times \mathbf{B}}{n_e q_e}
    - \frac{\nabla P_e}{n_e q_e} + \ldots$
    The non-ideal residual $\mathbf{E}' = \mathbf{E} + \mathbf{V} \times
    \mathbf{B}$ vanishes in ideal MHD and is non-zero where the frozen-in
    condition breaks down. The Hall term uses $|q_e|$ (charge magnitude,
    always positive) following the standard convention where $\mathbf{E}_{Hall}$
    points in the $\mathbf{J} \times \mathbf{B}$ direction. Dominant at
    ion skin depth scales where ion and electron motions decouple.
    [@Birn; @Hesse].

[^13]: For 2D geometry ($\partial/\partial z = 0$): $\mathbf{B} =
    \nabla\psi \times \hat{z} + B_z \hat{z}$ defines the in-plane field
    via the flux function $\psi$. Contours of $\psi$ are in-plane field
    lines. The reconnected flux is $\Delta\psi$ between the X-point and
    O-point. The reconnection rate is $\partial\psi/\partial t$ at the
    X-point, equal to the out-of-plane electric field $E_z$ there.
    Computed by cumulative integration: $\psi(x, y) = -\int_0^x B_y(x',
    y)\, dx'$ (negative sign from $B_y = -\partial\psi/\partial x$).
    [@Biskamp] §3.1.

[^14]: Firehose: unstable when $P_\parallel - P_\perp > B^2/2$
    (parallel pressure excess drives field-line bending). Mirror: unstable
    when $P_\perp/P_\parallel > 1 + 1/\beta_\perp$ (perpendicular
    pressure excess drives density compressions along field lines). Both
    arise from the CGL double-adiabatic framework and are the primary
    kinetic instabilities in collisionless plasmas with temperature
    anisotropy. [@Hellinger; @Gary; @Kunz].

[^16]: X-points (magnetic nulls in 2D) are saddle points of the flux
    function $\psi$ where $\nabla\psi = 0$ and the Hessian determinant is
    negative. The Hessian $H_{ij} = \partial^2\psi / \partial x_i \partial
    x_j$ is computed via second-order central finite differences. A negative
    determinant indicates saddle (hyperbolic) topology rather than an
    extremum (elliptic). The reconnection rate $R = \partial\psi/\partial t$
    at the X-point equals the out-of-plane electric field $E_z$ there, and
    measures the rate of magnetic flux transfer between topologically
    distinct regions. [@Biskamp] §3.1, [@Birn] Ch. 2.

[^17]: The magnetic shear angle measures the rotation of the magnetic
    field direction across a boundary (e.g. current sheet or magnetopause).
    $\theta = 0$ for parallel fields, $\pi$ for anti-parallel. Used in
    component reconnection analysis to determine whether reconnection
    geometry is anti-parallel ($\theta \approx \pi$) or component
    ($\theta < \pi$). [@Trattner].

[^De]: Zenitani's electron-frame dissipation measure: the
    Lorentz-boosted scalar $D_e = \gamma_e\bigl[\mathbf{J}\cdot
    \mathbf{E}' - \rho_c\,(\mathbf{V}_e\cdot\mathbf{E})\bigr]$ with
    $\mathbf{E}' = \mathbf{E} + \mathbf{V}_e\times\mathbf{B}$. The
    first term is the electron-frame Joule heating; the
    $\rho_c\,\mathbf{V}_e\cdot\mathbf{E}$ subtraction removes the bulk
    energy-transfer contribution to which $\mathbf{J}\cdot\mathbf{E}$
    is otherwise sensitive. Positive in the electron diffusion region,
    vanishes in ideal MHD. The canonical EDR localizer in modern
    collisionless PIC reconnection analysis. The non-relativistic
    limit is $D_e \to \mathbf{J}\cdot\mathbf{E}' -
    \rho_c\,\mathbf{V}_e\cdot\mathbf{E}$; the $\gamma_e$ prefactor
    activates when ``physics.relativistic`` is set in the dataset
    config. [@Zenitani2011].

[^Rrecon]: Dimensionless local reconnection rate normalized by the
    upstream Alfvén speed and the local field magnitude. Regions where
    $R_{\mathrm{recon}} \sim 0.1$ flag the "fast reconnection"
    plateau ubiquitous in collisionless simulations and observations.
    Per-cell scalar built from the same $\mathbf{E}'$ that drives the
    reconnection rate $\partial\psi/\partial t$ in 2D, divided by
    $v_A\,\lvert\mathbf{B}\rvert$ so the result is comparable across
    runs with different field strengths. [@ComissoBhattacharjee2016].

[^Dng]: Aunai's degree of nongyrotropy: $D_{ng} = 2\,\|\mathbf{N}\|_F
    / \mathrm{Tr}(\mathbf{P})$ where $\mathbf{N} = \mathbf{P}
    - P_\parallel\,\hat{b}\hat{b} - P_\perp(\mathbf{I} -
    \hat{b}\hat{b})$ is the non-gyrotropic part of the pressure
    tensor and $\|\cdot\|_F$ is the Frobenius norm. Frame-invariant.
    Captures both perpendicular anisotropy (eigenvalue spread in the
    perp 2×2 block) and off-axis ($\hat{b}$-coupling) nongyrotropy.
    Closed-form identity: $\|\mathbf{N}\|_F^2 = \mathrm{Tr}(\mathbf{P}^2)
    - P_\parallel^2 - 2 P_\perp^2$. pypic standardizes on Swisdak's
    $Q$ (see ``agyrotropy``) for its closed form and bounded range;
    $D_{ng}$ ships as a research alternative for literature
    comparisons. [@Aunai2013; @Swisdak2016].

[^Aphi]: Scudder's electron agyrotropy: eigenvalue-ratio of the
    perpendicular $2\times 2$ block of $\mathbf{P}$ in the
    field-aligned frame, $A_\phi = \lvert\lambda_1^\perp -
    \lambda_2^\perp\rvert / (\lambda_1^\perp + \lambda_2^\perp)$.
    Bounded $[0, 1]$, frame-invariant. **Only** captures perpendicular
    anisotropy — misses off-axis ($\hat{b}$-coupling) nongyrotropy
    that $D_{ng}$ and Swisdak's $Q$ both detect. Closed form: $A_\phi
    = \sqrt{\max(0,\;\|\Pi\|_F^2/(2 P_\perp^2) - 1)}$, where $\Pi$ is
    the double-projected perpendicular pressure tensor (same object
    computed inside ``agyrotropy``). [@Scudder2008; @Swisdak2016].

[^xi]: Schindler-Hesse-Birn 3D general-magnetic-reconnection
    criterion. Reconnection is defined by $\Xi(\mathbf{x}_0) \neq 0$
    along a magnetic field line $\mathcal{L}(\mathbf{x}_0)$ — no 3D
    null is required (unlike the 2D X-point definition). Vanishes
    identically in ideal MHD because $\mathbf{E}^{\mathrm{ideal}}
    = -\mathbf{V}\times\mathbf{B} \perp \mathbf{B}$, so non-zero $\Xi$
    isolates the non-ideal contributions in Ohm's law. Implemented as
    an RK4 field-line trace from each seed (``trace_field_line``)
    with $E_\parallel = \mathbf{E}\cdot\hat{b}$ trapezoid-integrated
    along arc length. Returns one scalar per seed — not a per-cell
    field, so dispatched outside the ``compute()`` registry as
    ``pypic.reconnection.schindler_xi``. [@Schindler1988].


## 10. Spectral Analysis

| Name | Description | Equation |
|------|-------------|----------|
| `power_spectrum_1d` | 1D power spectral density[^15] | $P(k) = \|\hat{f}(k)\|^2 \Delta x / (2\pi N)$ |
| `power_spectrum_2d` | Radially averaged 2D PSD[^15] | $P(k) = \langle \|\hat{f}\|^2 \rangle_{\|k\|=k}$ |
| `power_spectrum_3d` | Spherically averaged 3D PSD[^15] | $P(k) = \langle \|\hat{f}\|^2 \rangle_{\|k\|=k}$ |

[^15]: The PSD uses Parseval-consistent normalization: $\int P(k)\, dk$
    equals the variance of the windowed signal. Wavenumbers are in radians
    per unit length: $k = 2\pi / \lambda$. Windowing (Hann by default)
    reduces spectral leakage from finite domain boundaries at the cost of
    broadening spectral peaks. The one-sided spectrum doubles the power at
    all frequencies except DC and Nyquist to account for negative-frequency
    contributions. Common reference slopes: $k^{-5/3}$ (Kolmogorov
    inertial range), $k^{-3}$ (2D enstrophy cascade), $k^{-7/3}$
    (sub-ion Hall-MHD). [@Frisch; @Press].
