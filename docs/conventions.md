# Physics Conventions

Detailed physics conventions for derived quantities in pypic.
These notes supplement the canonical field name tables in SCHEMA.md.
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
