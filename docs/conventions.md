# Physics Conventions

Detailed physics conventions for derived quantities in pypic.
These notes supplement the equation tables in [equations.md](equations.md)
and the field name tables in [SCHEMA.md](../SCHEMA.md).
When TASKS.md Steps 7-8 are implemented, the relevant sections
will migrate into function docstrings in `derived.py`.

## γ Convention for PIC Entropy

PIC codes have no equation of state, so γ in the isotropic entropy
formula $s = \ln(P / n^\gamma)$ is chosen by the number of
velocity-space degrees of freedom: $\gamma = 5/3$ for 3D, $\gamma = 2$
for 2D, $\gamma = 3$ for 1D. The 3D value ($\gamma = 5/3$) is the
default.

### Gyrotropic entropy and CGL invariants

The gyrotropic entropy exponent of 5 in $P_\parallel P_\perp^2 / n^5$
is **independent** of $\gamma = 5/3$. It arises from combining the two
CGL double-adiabatic invariants:

- $P_\perp / (nB) = \text{const}$
- $P_\parallel B^2 / n^3 = \text{const}$

Eliminating $B$ gives $P_\parallel P_\perp^2 / n^{3+2} = \text{const}$.
The coincidence with the numerator of 5/3 is just that — a coincidence.

PIC entropy is computed from particle velocity moments, not from an
equation of state. The isotropic form uses scalar pressure and number
density; the gyrotropic form uses the parallel and perpendicular pressure
components. Both are commonly used to study dissipation and heating in
PIC simulations of reconnection and turbulence. Species-specific
entropies (`s_e`, `s_i`) are particularly useful for tracking which
species is heated.

## Thermal Speed Convention

We adopt $v_{th} = \sqrt{T/m}$ (with $T$ in energy units), following
the NRL Plasma Formulary (Richardson 2019) convention. This is the 1D
Maxwellian standard deviation $\sigma$ where
$f(v_x) \propto \exp(-v_x^2 / (2\sigma^2))$ with $\sigma^2 = T/m$.

**The community is split:** NRL, Chen, and kinetic theory use $\sqrt{T/m}$;
Fitzpatrick, Bellan, and some particle physics texts use $\sqrt{2T/m}$
("most probable speed"). The factor of $\sqrt{2}$ propagates into
gyroradius definitions — always state the convention explicitly. The
gyroradius $r = v_{th}/\omega_c$ uses whichever $v_{th}$ is adopted.

## Cyclotron Frequency Convention

$\omega_{ce}$ and $\omega_{ci}$ are defined as positive (magnitudes),
following NRL, Chen, and Fitzpatrick. Bellan uses the signed convention
$\omega_{c\sigma} = q_\sigma B / m_\sigma$ (so $\omega_{ce} < 0$ for
electrons). The signed form is needed in the Stix cold plasma dielectric
tensor. Our stored values are always positive.

## Debye Length

`lambda_D` is specifically the *electron* Debye length. The total plasma
Debye length is $1/\lambda_D^2 = \sum_s n_s q_s^2 / (\epsilon_0 T_s)$.

## Fast Magnetosonic Speed

$v_{ms} = \sqrt{v_A^2 + c_s^2}$ is the maximum fast-mode phase speed,
valid for propagation **perpendicular to B** ($\theta = 90°$). The
general angle-dependent dispersion relation is:

$$v_{f,s}^2 = \frac{1}{2}\left[v_A^2 + c_s^2 \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta}\right]$$

where $+$ gives the fast mode and $-$ the slow mode.

## Sound Speed vs Ion Acoustic Speed

$c_s = \sqrt{\gamma P / \rho_m}$ is the MHD sound speed. This differs
from the ion acoustic speed
$c_{ia} = \sqrt{(T_e + \gamma_i T_i) / m_i}$ commonly used in kinetic
theory (where $\gamma_e = 1$ for isothermal electrons, $\gamma_i = 3$
for 1D adiabatic ions).

## Gaussian vs SI-Rationalized Normalization

Some PIC codes (notably iPIC3D) use **Gaussian CGS** normalization where
Maxwell's equations carry explicit $4\pi$ factors: $\nabla \cdot \mathbf{E}
= 4\pi\rho_c$, energy density $= B^2/(8\pi)$, Poynting flux $\propto
\mathbf{E} \times \mathbf{B} / (4\pi)$. pypic's canonical form uses
**SI-rationalized** normalization ($\mu_0 = \epsilon_0 = 1$, no $4\pi$):
$\nabla \cdot \mathbf{E} = \rho_c$, energy density $= B^2/2$. The physics
is identical — dimensionless quantities (beta, Mach numbers, entropy) are
the same in both systems. We chose SI-rationalized as canonical because
it eliminates $4\pi$ from every derived quantity ($B^2/2$ not $B^2/(8\pi)$,
$\mathbf{E} \times \mathbf{B}$ not $\mathbf{E} \times \mathbf{B}/(4\pi)$),
keeping the pure-function physics code free of bookkeeping constants.
The conversion happens once in the reader. Key mapping: Gaussian
density $\rho_G = \rho/(4\pi)$, Gaussian energy $B^2/(8\pi) \to B^2/2$.

## Error Norms and Divergence

### L2 norm: discrete, unweighted

The relative L2 error $\varepsilon_{L_2} = \|\mathbf{a} - \mathbf{b}\|_2
/ \|\mathbf{b}\|_2$ uses discrete sums without volume weighting. When
computed and reference fields live on the same grid, $\Delta V$ appears
identically in numerator and denominator and cancels. This matches the
standard convention in convergence studies (LeVeque, *Finite Difference
Methods for Ordinary and Partial Differential Equations*). Returns
`inf` when the reference field is identically zero.

### L-infinity norm: absolute, not relative

$\varepsilon_{L_\infty} = \max |a_i - b_i|$ is absolute. A relative
L∞ norm ($\max |a_i - b_i| / |b_i|$) is misleading near field nulls
where $|b_i| \to 0$, which is common in reconnection regions and
current sheets.

### Field energy as volume integral

`field_energy` computes $\sum f_{ijk} \cdot \Delta V$ (a volume
integral), not $\sum f_{ijk}^2 \cdot \Delta V$ (an L2 norm squared).
Pass an energy density field to get total energy, or a mass density to
get total mass.

### Boundary treatment for finite differences

Divergence uses `np.gradient`, which applies second-order central
differences in the interior and second-order one-sided (forward/backward)
stencils at the first and last grid points. This is *not* periodic
wrapping — periodic domains should pad ghost cells before calling
the diagnostic functions.

### Node-centered stencil

All fields sit on the same node-centered grid (no stagger). The
divergence stencil operates on co-located field components, consistent
with the iPIC3D output convention (see Node-Centered Grid Convention
below).

## Node-Centered Grid Convention

iPIC3D outputs all fields on **nodes** (cell vertices), not cell centers.
Domain boundaries $[0, L_x]$ are node positions; cell centers sit at
half-grid offsets ($dx/2$, $3dx/2$, ...). The output array shape is
$(N_{xc}+1) \times (N_{yc}+1) \times (N_{zc}+1)$ where $N_{xc}$ is the
number of cells along each axis. No coordinate arrays are stored in the
HDF5 files — the reader reconstructs node coordinates from the grid
origin, spacing, and dimensions.

## Relativistic Conventions

These conventions apply when `physics.relativistic = true` in the
simulation config, or when analyzing output from relativistic PIC codes
(TRISTAN-MP, Zeltron, OSIRIS) or relativistic MHD codes. See
[equations.md § 8](equations.md#8-relativistic-corrections) for the
corresponding formulas.

### Three-velocity vs four-velocity

pypic supports two velocity representations:

- **Three-velocity** `V1/V2/V3` ($v^i$): bounded by $c$, the standard
  output of non-relativistic and semi-relativistic codes. Directly
  interpretable as physical speed.

- **Four-velocity** `u1/u2/u3` ($u^i = \gamma v^i$): unbounded, the
  natural output of relativistic PIC codes (TRISTAN-MP, Zeltron, OSIRIS).
  Numerically well-behaved at all speeds because there is no artificial
  upper bound. The four-velocity is the spatial part of the 4-velocity
  $u^\mu = \gamma(c, \mathbf{v})$.

When four-velocity is available, derived quantities should compute the
Lorentz factor from $\gamma = \sqrt{1 + u^2/c^2}$ and recover
three-velocity as $v^i = u^i/\gamma$ only when needed.

### Lorentz factor computation

Two equivalent forms, with different numerical properties:

- **From three-velocity:** $\gamma = 1/\sqrt{1 - v^2/c^2}$. Suffers
  from catastrophic cancellation when $v \approx c$ (the subtraction
  $1 - v^2/c^2$ loses precision). Adequate for mildly relativistic
  flows ($\gamma \lesssim 10$).

- **From four-velocity:** $\gamma = \sqrt{1 + u^2/c^2}$. No
  cancellation — numerically stable at all Lorentz factors. Always
  preferred when four-velocity data is available.

The `lorentz_factor()` function accepts either form and selects the
appropriate computation.

### Magnetization parameter

$\sigma = B^2 / (\rho_m c^2)$ (in normalized units with $\mu_0 = 1$).
This is the fundamental dimensionless parameter in relativistic plasma
physics, measuring the ratio of magnetic energy density to rest-mass
energy density.

- $\sigma \ll 1$: matter-dominated (non-relativistic MHD regime).
  The non-relativistic Alfven speed $v_A = B/\sqrt{\rho_m}$ applies.
- $\sigma \sim 1$: trans-relativistic. Full relativistic formulas needed.
- $\sigma \gg 1$: magnetically dominated (pulsar winds, jets, relativistic
  reconnection). The Alfven speed $v_A \to c$.

$\sigma$ is analogous to $1/\beta$ in that both measure the importance
of the magnetic field, but $\sigma$ compares to rest-mass energy while
$\beta$ compares to thermal energy. In a cold, highly magnetized plasma
$\sigma \gg 1$ and $\beta \ll 1$.

### Relativistic speed composition

Speeds do not add linearly in special relativity. The relativistic
magnetosonic speed uses the composition formula:

$$v_{ms}^2 = v_A^2 + c_s^2 - \frac{v_A^2 c_s^2}{c^2}$$

This ensures $v_{ms} < c$ always, unlike the non-relativistic
$v_{ms}^2 = v_A^2 + c_s^2$ which can exceed $c$ when $\sigma$ is large.
In normalized units ($c = 1$) the formula simplifies to
$v_{ms}^2 = v_A^2 + c_s^2 - v_A^2 c_s^2$. The non-relativistic limit
is recovered when $v_A, c_s \ll c$, since the correction term
$v_A^2 c_s^2 / c^2$ becomes negligible.
