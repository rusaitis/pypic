# Physics Conventions

Detailed physics conventions for derived quantities in pypic.
These notes supplement the equation tables in [equations.md](equations.md)
and the field name tables in [SCHEMA](schema.md).

## Gamma Convention for PIC Entropy

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

pypic adopts $v_{th} = \sqrt{T/m}$ (with $T$ in energy units), following
the NRL Plasma Formulary [@NRL] convention. This is the 1D Maxwellian
standard deviation $\sigma$ where
$f(v_x) \propto \exp(-v_x^2 / (2\sigma^2))$ with $\sigma^2 = T/m$.

**The community is split:** NRL [@NRL], Chen [@Chen], and kinetic theory
use $\sqrt{T/m}$; Fitzpatrick [@Fitz], Bellan, and some particle physics
texts use $\sqrt{2T/m}$ ("most probable speed"). The factor of $\sqrt{2}$
propagates into gyroradius definitions — always state the convention
explicitly. The gyroradius $r = v_{th}/\omega_c$ uses whichever $v_{th}$
is adopted.

## Temperature in Energy Units

pypic stores temperature in **energy units** (J in SI), not Kelvin.
Concretely, `T = P/n` with no Boltzmann factor; the `temperature`
quantity_type maps to the SI unit `"J"` and the openPMD
`unitDimension` 7-tuple `(2, 1, -2, 0, 0, 0, 0)` — same as
`energy_density`, not the Kelvin form `(0, 0, 0, 0, 1, 0, 0)`. This
matches NRL Formulary, Chen, Krall & Trivelpiece, and the convention
used by every production PIC code (iPIC3D, OSIRIS, VPIC, Smilei,
WarpX, TRISTAN). Plasma formulas — sound speed, Debye length, plasma
beta, gyroradii — stay free of $k_B$ as a result.

### Converting to eV or K for display

`in_si()` always returns Joules (SI by definition). For the
plasma-physics working unit eV, or for K-equivalent comparison with
thermometer data, use `in_units()`:

```python
T_J  = ds.in_si("Te")              # Joules
T_eV = ds.in_units("Te", "eV")     # ÷ scipy.constants.eV
T_K  = ds.in_units("Te", "K")      # ÷ scipy.constants.k
T_keV = ds.in_units("Te", "keV")   # convenience alias
```

Same path works for any energy-dimensional quantity (pressure,
energy_density, specific_energy), but eV is conventional only for
temperature; pressure stays in Pa (or nPa for space), energy density
in J/m³.

### openPMD impedance

When pypic eventually reads or writes openPMD data with temperature
recorded in K, the reader/writer applies a `× k_B` (or `÷ k_B`)
conversion at the boundary. The `unitDimension` attribute that pypic
emits describes its internal representation (J), not the Kelvin form
a downstream openPMD consumer might expect. Document the convention
on the file rather than the schema if you intend the data to be
consumed by tools that assume K.

## Cyclotron Frequency Convention

$\omega_{ce}$ and $\omega_{ci}$ are defined as positive (magnitudes),
following NRL [@NRL], Chen [@Chen], and Fitzpatrick [@Fitz]. Bellan uses the signed convention
$\omega_{c\sigma} = q_\sigma B / m_\sigma$ (so $\omega_{ce} < 0$ for
electrons). The signed form is needed in the Stix cold plasma dielectric
tensor. Stored values in pypic are always positive.

## Debye Length

`lambda_D` (NRL alias for canonical `lambda_D_s0`) is specifically the
*electron* Debye length. The total plasma Debye length, summed over
all species (SI form), is $1/\lambda_D^2 = \sum_s n_s q_s^2 /
(\epsilon_0 T_s)$; in pypic's normalized PIC units ($\epsilon_0 = 1$)
the $\epsilon_0$ drops out.

## Fast Magnetosonic Speed

$v_{ms} = \sqrt{v_A^2 + c_s^2}$ is the maximum fast-mode phase speed,
valid for propagation **perpendicular to B** ($\theta = 90°$). The
general angle-dependent dispersion relation is:

$$v_{f,s}^2 = \frac{1}{2}\left[v_A^2 + c_s^2 \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta}\right]$$

where $+$ gives the fast mode and $-$ the slow mode.

## Sound Speed vs Ion Acoustic Speed

$c_s = \sqrt{\gamma P / \rho_m}$ is the MHD sound speed. This differs
from the ion acoustic speed
$c_{ia} = \sqrt{(\gamma_e T_e + \gamma_i T_i) / m_i}$ commonly used in
kinetic theory, typically with $\gamma_e = 1$ (isothermal electrons)
and $\gamma_i = 3$ (1D adiabatic ions).

## Gaussian vs SI-Rationalized Normalization

Some PIC codes (notably iPIC3D) use **Gaussian CGS** normalization where
Maxwell's equations carry explicit $4\pi$ factors: $\nabla \cdot \mathbf{E}
= 4\pi\rho_c$, energy density $= B^2/(8\pi)$, Poynting flux $\propto
\mathbf{E} \times \mathbf{B} / (4\pi)$. pypic's canonical form uses
**SI-rationalized** normalization ($\mu_0 = \epsilon_0 = 1$, no $4\pi$):
$\nabla \cdot \mathbf{E} = \rho_c$, energy density $= B^2/2$. The physics
is identical — dimensionless quantities (beta, Mach numbers, entropy) are
the same in both systems. SI-rationalized is canonical here because
it eliminates $4\pi$ from every derived quantity ($B^2/2$ not $B^2/(8\pi)$,
$\mathbf{E} \times \mathbf{B}$ not $\mathbf{E} \times \mathbf{B}/(4\pi)$),
keeping the pure-function physics code free of bookkeeping constants.
The conversion happens once in the reader. Key mapping: Gaussian
charge density $\rho_{c,G} = \rho_c/(4\pi)$, Gaussian energy
$B^2/(8\pi) \to B^2/2$.

## The SI Anchor Is a Choice, Not a Recoverable Value

A PIC deck fixes only *dimensionless* ratios — $\omega_{pe}/\omega_{ce}$,
$m_i/m_e$, $c/v_A$, the grid in skin depths. Every quantity in it is
already in code units, including the ones that look physical: iPIC3D's
`rhoINIT` is a code-unit density (typically 1.0) and `qom` is a
code-unit ratio at a reduced mass ratio.

Converting to SI needs one *absolute* anchor on top of those ratios.
`Normalization.pic_standard` takes it as the reference plasma
frequency,

$$\omega_{ref} = \sqrt{\frac{n_{ref} \, q_{ref}^2}{\varepsilon_0 \, m_{ref}}}$$

from which $l_{ref}$, $t_{ref}$, $B_{ref}$ and $E_{ref}$ all follow.
Nothing in the deck supplies $n_{ref}$: the same double-Harris setup is
a laboratory plasma at $10^{18}\,\mathrm{m^{-3}}$ or the magnetotail at
$10^{6}\,\mathrm{m^{-3}}$, and which one it is, is the modeller's
interpretation of their own run. No reader can reconstruct it.

So `reference_number_density` belongs in `simulation.toml` — the deck is where
that interpretation gets recorded — and a reader that finds no
`[units]` section reports the normalization as **undeclared**
(`Normalization.undeclared()`, `system = None`) rather than guessing.
Asking such a dataset for a dimensional SI value raises
`UndeclaredNormalizationError` instead of returning code units
labelled tesla. Dimensionless quantities ($\beta$, $M_A$, the
agyrotropy measures) are exempt: they are correct under any anchor.

Three ways to supply one: ship a `simulation.toml`, pass
`normalization=` to the reader, or stay in code units.

## Massless Fluid Species

Hybrid codes close the system with an inertialess electron fluid, so
`mass = 0.0` on a `[[species]]` entry is a statement about the model,
not a missing value — `inertia` (the mass ratio $m_e/m_i$, zero when
massless) says the same thing. A species with `particles_per_cell`
still needs a positive mass: macroparticles have to be pushed.

Two consequences follow, both deliberate. `SpeciesInfo.charge_to_mass`
is left unset rather than infinite, because every consumer already
branches on its absence while an infinity would travel silently into
whatever moment consumed it next. And the per-species kinetic scales
that divide by mass — $v_{th} = \sqrt{T/m}$, $\omega_p$, $d_s$, $r_s$
— return `inf` or `0` with a NumPy divide warning. That is the correct
answer: a massless fluid has no gyroradius and no inertial length. Ask
for those quantities on the *ion* species, which carries the mass that
sets the scales a hybrid run resolves.

## Error Norms and Divergence

### L2 norm: discrete, unweighted

The relative L2 error $\varepsilon_{L_2} = \|\mathbf{a} - \mathbf{b}\|_2
/ \|\mathbf{b}\|_2$ uses discrete sums without volume weighting. When
computed and reference fields live on the same grid, $\Delta V$ appears
identically in numerator and denominator and cancels. This matches the
standard convention in convergence studies [@LeVeque]. Returns `inf`
when the reference field is identically zero.

### L-infinity norm: absolute, not relative

$\varepsilon_{L_\infty} = \max |a_i - b_i|$ is absolute. A relative
L∞ norm ($\max |a_i - b_i| / |b_i|$) is misleading near field nulls
where $|b_i| \to 0$, which is common in reconnection regions and
current sheets.

### NaN handling in diagnostics

`l2_relative_error` and `linf_error` accept a `nan_policy` keyword
with three values:

- **`"omit"`** (default) — NaN cells in either input are dropped via a
  joint mask, so the numerator and denominator of the relative L2
  error are restricted to the **same** set of valid points. Masking
  only one side would artificially shrink the relative error by
  leaving reference-field energy in the denominator that has no
  counterpart in the numerator. A `UserWarning` reports the dropped
  count every time masking occurs, so silent data loss is impossible.
- **`"propagate"`** — unaltered NumPy reduction. Any NaN in either
  input poisons the result. Use this for strict convergence and
  verification tests where a NaN anywhere is itself the bug.
- **`"raise"`** — any NaN in either input raises `ValueError` with
  the cell count. Use this in pipelines where the upstream contract
  guarantees no NaN and a violation should halt the run.

The default is `"omit"` because the routine sources of NaN in pypic
are legitimate: `SphereSelection` and `FieldDataset.where()` mask
regions with NaN by design, `pypic.regridding` fills out-of-domain cells
with NaN by design, and cross-grid comparisons against a smaller
dataset produce NaN boundary cells by construction. Propagating NaN
through these would make the common case (compare two runs after
slicing) silently useless.

The policy applies to the pure-array diagnostics in
`pypic.diagnostics`. The comparison wrappers in `pypic.comparison`
forward the same keyword unchanged — one knob, one semantics,
whether you are passing NumPy arrays or `FieldDataset` pairs.

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
the diagnostic functions, or reduce over the interior only.

The size of the difference is easy to underestimate. On an analytically
divergence-free periodic field
$\mathbf{B} = (\sin x \cos y,\ -\cos x \sin y,\ 0)$ over a $32^3$ box,
interior $\max|\nabla\cdot\mathbf{B}|$ is $1.9\times10^{-15}$ — roundoff —
while the full-grid figure is $9.5\times10^{-3}$, because 18% of cells sit
on a boundary face. Box-wide reductions are therefore dominated by the end
planes rather than by the physics, which bites spectral and turbulence runs
hardest: those are periodic by construction and their headline diagnostics
are box-wide.

When an axis is marked `periodic` — by `[boundary_conditions]`, or by the
native deck a reader parses (BATSRUS `#OUTERBOUNDARY` and the `.h`
`#PERIODIC` block, iPIC3D `PERIODICX/Y/Z`) — `compute()` warns once for the
grid-dependent quantities (`div_B`, `div_E`, `curl_B_*`, `vort_*`) rather
than returning a silently degraded edge. The warning names the axes; it is
not a statement that the result is unusable, only that its end planes are. Wrapping the stencil is a roadmap item (TASKS Step 52) —
`np.gradient` has no periodic mode, so it means `np.roll`-based central
differences on the flagged axes.

### Co-located stencil

Readers destagger to a single co-located grid on load (cell-center or
node, depending on the source code). The divergence stencil operates
on co-located field components; pypic does not provide a
staggered-mesh stencil. iPIC3D specifically emits on nodes (see
Node-Centered Grid Convention below); BATSRUS HDF5 destaggers
face-centered B to cell centers; the convention is recorded
per-dataset in `StaggerInfo` metadata.

## Field-Aligned Decomposition

`compute("J_par")`, `V_par`, `E_par`, `E_prime_par`, and their per-species
and perpendicular siblings decompose a vector $\mathbf{A}$ against
$\hat{b} = \mathbf{B}/|\mathbf{B}|$.  Formulas are in
[equations.md § 4.2](equations.md#42-field-aligned-vector-decomposition).
Five conventions apply, all matching the existing pressure-tensor
decomposition ([`P_par`](equations.md#4-pressure-tensor-and-field-aligned-decomposition)):

- **Sign of $A_\parallel$.** Signed: $A_\parallel > 0$ means $\mathbf{A}$
  is co-directional with $\mathbf{B}$, $< 0$ means anti-parallel.
  $A_\parallel = 0$ for purely perpendicular vectors.  Identity:
  $|\mathbf{A}|^2 = A_\parallel^2 + |\mathbf{A}_\perp|^2$.

- **Reference vector.** The local, *full* magnetic field $\mathbf{B}$
  from the dataset — not a background $\mathbf{B}_0$ (split-B) or a
  smoothed/averaged version.  Workflows that need decomposition
  against $\mathbf{B}_0$ should call `derived.parallel_component(a1,
  a2, a3, B0_1, B0_2, B0_3)` directly; no `J_par_B0` recipe is
  registered.

- **Behavior at $|B| = 0$.** NaN in every output component.  Inherited
  from `_unit_vector`; matches `P_par`/`P_perp`/`agyrotropy`.  In
  practice, even before exact zero $\hat{b}$ becomes ill-conditioned —
  mask cells where $|B|$ falls below a physically motivated floor
  before drawing conclusions about $\mathbf{A}_\perp$.

- **Frame transforms.** The decomposition is $\hat{b}$-locked to the
  snapshot frame.  A frame rotation that re-expresses $\mathbf{B}$ in
  a new basis re-expresses $\mathbf{A}_\perp$ in that basis too;
  $A_\parallel$ is a frame-invariant scalar up to the choice of
  $\hat{b}$.  Stored arrays reflect the snapshot frame; transform at
  read time via `FieldDataset.transform_to(...)` if a different frame
  is needed.

- **Per-species coverage.** $\mathbf{V}$ has the per-species form
  ($V_{s\{N\}\,\parallel}$, $V_{s\{N\}\,\perp,i}$, $|V_{s\{N\}\,\perp}|$).
  $\mathbf{J}$ and $\mathbf{E}$ are total-only — per-species $\mathbf{J}$
  is uncommon in output, and $\mathbf{E}$ is a field (not a species
  moment).  Add per-species recipes for either if a reader exposes the
  inputs.

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

- **Three-velocity** `V_1/V_2/V_3` ($v^i$): bounded by $c$, the standard
  output of non-relativistic and semi-relativistic codes. Directly
  interpretable as physical speed.

- **Four-velocity** `u_1/u_2/u_3` ($u^i = \gamma v^i$): unbounded, the
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
  The non-relativistic Alfvén speed $v_A = B/\sqrt{\rho_m}$ applies.
- $\sigma \sim 1$: trans-relativistic. Full relativistic formulas needed.
- $\sigma \gg 1$: magnetically dominated (pulsar winds, jets, relativistic
  reconnection). The Alfvén speed $v_A \to c$.

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
