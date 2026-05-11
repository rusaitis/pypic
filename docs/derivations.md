# Derivations

Fundamental derivations connecting Maxwell's equations to the fluid and
kinetic equations used throughout pypic. Each section states the key result
up front; click **Derivation** to expand the full mathematical development.

These derivations use **normalized units** ($\mu_0 = \epsilon_0 = 1$) unless
stated otherwise. See [Conventions](conventions.md) for the normalization
mapping.

## Maxwell's Equations

The foundation for all plasma physics in pypic. In SI:

$$\nabla \cdot \mathbf{E} = \frac{\rho_c}{\epsilon_0}, \qquad \nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \qquad \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

In normalized units ($\mu_0 = \epsilon_0 = 1$):

$$\nabla \cdot \mathbf{E} = \rho_c, \qquad \nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \qquad \nabla \times \mathbf{B} = \mathbf{J} + \frac{\partial \mathbf{E}}{\partial t}$$

The constraint $\nabla \cdot \mathbf{B} = 0$ is monitored by
[`div_b`](equations.md#7-diagnostics-and-validation) as a numerical
health check. The displacement current $\partial \mathbf{E}/\partial t$
is retained in PIC codes but dropped in MHD ($v \ll c$) [@Jack].

## Vlasov-Maxwell System

PIC codes solve the **Vlasov equation** for the distribution function
$f_s(\mathbf{x}, \mathbf{v}, t)$ of each species $s$:

$$\frac{\partial f_s}{\partial t} + \mathbf{v} \cdot \nabla f_s + \frac{q_s}{m_s}\left(\mathbf{E} + \mathbf{v} \times \mathbf{B}\right) \cdot \frac{\partial f_s}{\partial \mathbf{v}} = 0$$

This is a collisionless Boltzmann equation in 6D phase space. The
electromagnetic fields $(\mathbf{E}, \mathbf{B})$ evolve via Maxwell's
equations with sources computed from $f_s$:

$$n_s = \int f_s \, d^3v, \qquad n_s \mathbf{V}_s = \int \mathbf{v} \, f_s \, d^3v$$

$$\rho_c = \sum_s q_s n_s, \qquad \mathbf{J} = \sum_s q_s n_s \mathbf{V}_s$$

This self-consistent loop (particles $\to$ moments $\to$ fields $\to$
forces $\to$ particles) is the Vlasov-Maxwell system.

## Fluid Moment Equations

Taking velocity moments of the Vlasov equation yields the fluid
equations that connect kinetic physics to the macroscopic quantities
in pypic. The zeroth moment gives mass conservation; the first moment
gives momentum balance.

**Continuity** (zeroth moment):

$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{V}_s) = 0$$

**Momentum** (first moment):

$$m_s n_s \left(\frac{\partial \mathbf{V}_s}{\partial t} + \mathbf{V}_s \cdot \nabla \mathbf{V}_s\right) = n_s q_s \left(\mathbf{E} + \mathbf{V}_s \times \mathbf{B}\right) - \nabla \cdot \mathbf{P}_s$$

where the pressure tensor $\mathbf{P}_s$ is the second velocity moment
about the mean flow:

$$\mathbf{P}_s = m_s \int (\mathbf{v} - \mathbf{V}_s)(\mathbf{v} - \mathbf{V}_s) \, f_s \, d^3v$$

??? abstract "Derivation: velocity moments of the Vlasov equation"
    Multiply the Vlasov equation by a velocity-space weight $\phi(\mathbf{v})$
    and integrate over all velocities:

    $$\int \phi \frac{\partial f_s}{\partial t} d^3v + \int \phi \, \mathbf{v} \cdot \nabla f_s \, d^3v + \frac{q_s}{m_s} \int \phi \left(\mathbf{E} + \mathbf{v} \times \mathbf{B}\right) \cdot \frac{\partial f_s}{\partial \mathbf{v}} d^3v = 0$$

    **Zeroth moment** ($\phi = 1$):

    The first term gives $\partial n_s / \partial t$. The second gives
    $\nabla \cdot (n_s \mathbf{V}_s)$. The third vanishes by integration
    by parts (the Lorentz force is divergence-free in velocity space,
    and $f_s \to 0$ as $|\mathbf{v}| \to \infty$):

    $$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{V}_s) = 0$$

    **First moment** ($\phi = m_s \mathbf{v}$):

    Decompose $\mathbf{v} = \mathbf{V}_s + \mathbf{w}$ where $\mathbf{w}$
    is the random (thermal) velocity with $\langle \mathbf{w} \rangle = 0$.
    The convective term produces the pressure tensor divergence:

    $$\int m_s \mathbf{v} (\mathbf{v} \cdot \nabla f_s) \, d^3v = \nabla \cdot (m_s n_s \mathbf{V}_s \mathbf{V}_s + \mathbf{P}_s)$$

    The Lorentz term integrates by parts to give $n_s q_s (\mathbf{E} +
    \mathbf{V}_s \times \mathbf{B})$ (the magnetic force contribution from
    random velocities vanishes because $\mathbf{P}_s$ is symmetric while
    $\mathbf{v} \times \mathbf{B}$ is antisymmetric). Combining and using
    the continuity equation to simplify the convective derivative:

    $$m_s n_s \frac{d \mathbf{V}_s}{dt} = n_s q_s (\mathbf{E} + \mathbf{V}_s \times \mathbf{B}) - \nabla \cdot \mathbf{P}_s$$

    where $d/dt = \partial/\partial t + \mathbf{V}_s \cdot \nabla$ is the
    convective derivative.

    Each moment introduces the next-higher moment (the **closure problem**):
    the continuity equation involves $\mathbf{V}_s$, the momentum equation
    involves $\mathbf{P}_s$, and so on. MHD closes the hierarchy by
    assuming an equation of state ($P = P(\rho, s)$); PIC sidesteps it by
    evolving $f_s$ directly.

## Generalized Ohm's Law

The electric field in a plasma can be decomposed into physical
contributions by manipulating the electron momentum equation. The
result is the **generalized Ohm's law**:

$$\mathbf{E} = \underbrace{-\mathbf{V} \times \mathbf{B}}_{\text{ideal}} + \underbrace{\frac{\mathbf{J} \times \mathbf{B}}{n_e |q_e|}}_{\text{Hall}} - \underbrace{\frac{\nabla \cdot \mathbf{P}_e}{n_e |q_e|}}_{\text{electron pressure}} + \underbrace{\frac{m_e}{|q_e|} \frac{d\mathbf{V}_e}{dt}}_{\text{electron inertia}}$$

Each term has a characteristic length scale where it becomes important:
the ideal term dominates at MHD scales ($L \gg d_i$), the Hall term at
ion scales ($d_e \ll L \lesssim d_i$), and the pressure and inertia
terms at electron scales ($L \sim d_e$). The non-ideal electric field
$\mathbf{E}' = \mathbf{E} + \mathbf{V} \times \mathbf{B}$ vanishes in
ideal MHD and is non-zero where the frozen-in condition breaks down.

See [`E_prime`](equations.md#9-reconnection-and-anisotropy-diagnostics),
[`E_Hall`](equations.md#9-reconnection-and-anisotropy-diagnostics),
[`E_ideal`](equations.md#9-reconnection-and-anisotropy-diagnostics) in
the equations reference [@Birn; @Hesse].

??? abstract "Derivation: from the two-fluid momentum equations"
    Start with the electron momentum equation:

    $$m_e n_e \frac{d\mathbf{V}_e}{dt} = -n_e |q_e| (\mathbf{E} + \mathbf{V}_e \times \mathbf{B}) - \nabla \cdot \mathbf{P}_e$$

    (using $q_e = -|q_e|$ for electrons). Solve for $\mathbf{E}$:

    $$\mathbf{E} = -\mathbf{V}_e \times \mathbf{B} - \frac{\nabla \cdot \mathbf{P}_e}{n_e |q_e|} + \frac{m_e}{|q_e|} \frac{d\mathbf{V}_e}{dt}$$

    Now express $\mathbf{V}_e$ in terms of the bulk (ion) velocity and
    current. For a quasi-neutral plasma ($n_e \approx n_i \equiv n$):

    $$\mathbf{J} = n|q_e|(\mathbf{V}_i - \mathbf{V}_e) \quad \Longrightarrow \quad \mathbf{V}_e = \mathbf{V}_i - \frac{\mathbf{J}}{n|q_e|}$$

    Since $m_i \gg m_e$, the center-of-mass velocity $\mathbf{V} \approx
    \mathbf{V}_i$. Substituting into the $\mathbf{V}_e \times \mathbf{B}$
    term:

    $$\mathbf{V}_e \times \mathbf{B} = \left(\mathbf{V} - \frac{\mathbf{J}}{n|q_e|}\right) \times \mathbf{B} = \mathbf{V} \times \mathbf{B} - \frac{\mathbf{J} \times \mathbf{B}}{n|q_e|}$$

    Substituting back:

    $$\mathbf{E} = -\mathbf{V} \times \mathbf{B} + \frac{\mathbf{J} \times \mathbf{B}}{n|q_e|} - \frac{\nabla \cdot \mathbf{P}_e}{n|q_e|} + \frac{m_e}{|q_e|}\frac{d\mathbf{V}_e}{dt}$$

    This identifies the four terms:

    | Term | Expression | Dominant scale |
    |------|-----------|---------------|
    | Ideal (convective) | $-\mathbf{V} \times \mathbf{B}$ | $L \gg d_i$ |
    | Hall | $\mathbf{J} \times \mathbf{B} / (n\|q_e\|)$ | $d_e \ll L \lesssim d_i$ |
    | Electron pressure | $-\nabla \cdot \mathbf{P}_e / (n\|q_e\|)$ | $L \sim d_e$ |
    | Electron inertia | $(m_e/\|q_e\|) \, d\mathbf{V}_e/dt$ | $L \sim d_e$ |

    Dropping all but the first term recovers ideal MHD:
    $\mathbf{E} = -\mathbf{V} \times \mathbf{B}$.

## Ideal MHD Equations

Ideal MHD treats the plasma as a single conducting fluid. It follows
from the two-fluid equations by (1) taking the single-fluid limit
($m_e \to 0$, charge neutrality), (2) dropping the displacement current
($v \ll c$), and (3) using the ideal Ohm's law ($\mathbf{E} =
-\mathbf{V} \times \mathbf{B}$).

The closed system in normalized units ($\mu_0 = 1$):

$$\frac{\partial \rho_m}{\partial t} + \nabla \cdot (\rho_m \mathbf{V}) = 0 \qquad \text{(mass)}$$

$$\rho_m \frac{d\mathbf{V}}{dt} = (\nabla \times \mathbf{B}) \times \mathbf{B} - \nabla P \qquad \text{(momentum)}$$

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{V} \times \mathbf{B}) \qquad \text{(induction)}$$

$$\frac{d}{dt}\left(\frac{P}{\rho_m^\gamma}\right) = 0 \qquad \text{(entropy/energy)}$$

??? abstract "Derivation: from two-fluid to single-fluid MHD"
    **Single-fluid variables.** Define the mass density, bulk velocity,
    and total pressure:

    $$\rho_m = \sum_s n_s m_s \approx n_i m_i, \qquad \mathbf{V} = \frac{\sum_s n_s m_s \mathbf{V}_s}{\rho_m} \approx \mathbf{V}_i, \qquad P = \sum_s P_s$$

    **Mass conservation.** Sum the species continuity equations:

    $$\frac{\partial \rho_m}{\partial t} + \nabla \cdot (\rho_m \mathbf{V}) = 0$$

    **Momentum.** Sum the species momentum equations. The internal
    electric forces cancel by quasi-neutrality ($\sum_s n_s q_s = 0$).
    Using Ampere's law without displacement current
    ($\mathbf{J} = \nabla \times \mathbf{B}$):

    $$\rho_m \frac{d\mathbf{V}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla P = (\nabla \times \mathbf{B}) \times \mathbf{B} - \nabla P$$

    The magnetic force can be decomposed via the vector identity:

    $$(\nabla \times \mathbf{B}) \times \mathbf{B} = (\mathbf{B} \cdot \nabla)\mathbf{B} - \nabla\left(\frac{B^2}{2}\right)$$

    revealing **magnetic tension** $(\mathbf{B} \cdot \nabla)\mathbf{B}$
    (field lines resist bending) and **magnetic pressure**
    $-\nabla(B^2/2)$ (field lines resist compression). The plasma beta
    $\beta = 2P/B^2$ measures their relative importance.

    **Induction.** Substitute the ideal Ohm's law
    $\mathbf{E} = -\mathbf{V} \times \mathbf{B}$ into Faraday's law:

    $$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E} = \nabla \times (\mathbf{V} \times \mathbf{B})$$

    This is the **frozen-in flux** equation: magnetic field lines move
    with the fluid. It breaks down wherever the non-ideal terms in
    Ohm's law become significant (reconnection sites, current sheets).

    **Energy closure.** Assuming adiabatic evolution (no heat conduction
    or radiation), the entropy is conserved along fluid elements:

    $$\frac{d}{dt}\left(\frac{P}{\rho_m^\gamma}\right) = 0$$

    This is equivalent to the [entropy](equations.md#2-thermodynamic-quantities)
    $s = \ln(P/\rho_m^\gamma)$ being a Lagrangian invariant.

## Energy Conservation: Poynting's Theorem

The electromagnetic energy balance connects the field energy densities
($e_B$, $e_E$), the Poynting flux ($\mathbf{S}$), and the energy
conversion rate ($\mathbf{J} \cdot \mathbf{E}$):

$$\frac{\partial}{\partial t}\left(\frac{B^2 + E^2}{2}\right) + \nabla \cdot (\mathbf{E} \times \mathbf{B}) = -\mathbf{J} \cdot \mathbf{E}$$

Identifying: $e_B = B^2/2$, $e_E = E^2/2$, $\mathbf{S} = \mathbf{E}
\times \mathbf{B}$ (normalized Poynting flux), and $-\mathbf{J} \cdot
\mathbf{E}$ as the sink term transferring electromagnetic energy to
particle kinetic energy.

See [`e_B`](equations.md#3-energy-and-flux),
[`e_E`](equations.md#3-energy-and-flux),
[`S`](equations.md#3-energy-and-flux),
[`J_dot_E`](equations.md#9-reconnection-and-anisotropy-diagnostics) in
the equations reference.

??? abstract "Derivation: from Maxwell's equations"
    Take the dot product of Faraday's law with $\mathbf{B}$ and Ampere's
    law with $\mathbf{E}$:

    $$\mathbf{B} \cdot (\nabla \times \mathbf{E}) = -\mathbf{B} \cdot \frac{\partial \mathbf{B}}{\partial t} = -\frac{1}{2}\frac{\partial B^2}{\partial t}$$

    $$\mathbf{E} \cdot (\nabla \times \mathbf{B}) = \mathbf{E} \cdot \mathbf{J} + \mathbf{E} \cdot \frac{\partial \mathbf{E}}{\partial t} = \mathbf{J} \cdot \mathbf{E} + \frac{1}{2}\frac{\partial E^2}{\partial t}$$

    Subtract the first from the second:

    $$\mathbf{E} \cdot (\nabla \times \mathbf{B}) - \mathbf{B} \cdot (\nabla \times \mathbf{E}) = \mathbf{J} \cdot \mathbf{E} + \frac{1}{2}\frac{\partial}{\partial t}(E^2 + B^2)$$

    The left side equals $-\nabla \cdot (\mathbf{E} \times \mathbf{B})$
    by the vector identity
    $\nabla \cdot (\mathbf{A} \times \mathbf{B}) = \mathbf{B} \cdot (\nabla \times \mathbf{A}) - \mathbf{A} \cdot (\nabla \times \mathbf{B})$.
    Rearranging:

    $$\frac{\partial}{\partial t}\underbrace{\left(\frac{E^2 + B^2}{2}\right)}_{e_E + e_B} + \nabla \cdot \underbrace{(\mathbf{E} \times \mathbf{B})}_{\mathbf{S}} = -\mathbf{J} \cdot \mathbf{E}$$

    **Physical interpretation:** $\mathbf{J} \cdot \mathbf{E} > 0$ means
    the electromagnetic field loses energy to particle motion
    (acceleration) [@Jack]. In reconnection, this identifies the energy
    conversion region [@Zen]. Per-species decomposition
    $\mathbf{J}_s \cdot \mathbf{E}$ reveals which species is energized.

    In MHD, $e_E$ is negligible ($v \ll c$), and Poynting's theorem
    reduces to the magnetic energy budget:

    $$\frac{\partial}{\partial t}\left(\frac{B^2}{2}\right) + \nabla \cdot (\mathbf{E} \times \mathbf{B}) = -\mathbf{J} \cdot \mathbf{E}$$

## MHD Wave Speeds

Linearizing the ideal MHD equations around a uniform equilibrium
($\rho_0$, $P_0$, $\mathbf{B}_0 = B_0 \hat{z}$, $\mathbf{V}_0 = 0$)
yields three wave families. Their phase speeds are the characteristic
speeds used throughout pypic.

**Alfvén wave** (incompressible, transverse) [@NRL; @Chen]:

$$v_A = \frac{B_0}{\sqrt{\rho_0}}$$

**Sound wave** (compressible, longitudinal, no magnetic coupling) [@Chen]:

$$c_s = \sqrt{\frac{\gamma P_0}{\rho_0}}$$

**Fast and slow magnetosonic waves** (compressible, coupled) [@Fitz]:

$$v_{f,s}^2 = \frac{1}{2}\left[(v_A^2 + c_s^2) \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta}\right]$$

At perpendicular propagation ($\theta = 90°$), the fast speed reduces
to $v_{ms} = \sqrt{v_A^2 + c_s^2}$, which is the
[`v_ms`](equations.md#5-characteristic-scales) stored by pypic.

See [`v_A`](equations.md#5-characteristic-scales),
[`c_s`](equations.md#5-characteristic-scales),
[`v_ms`](equations.md#5-characteristic-scales) in the equations reference.

??? abstract "Derivation: linearization of ideal MHD"
    Perturb all quantities around a uniform equilibrium:
    $\rho = \rho_0 + \rho_1$, $\mathbf{V} = \mathbf{V}_1$,
    $\mathbf{B} = B_0 \hat{z} + \mathbf{B}_1$, $P = P_0 + P_1$,
    with $P_1 = c_s^2 \rho_1$ from the adiabatic equation of state.

    Substitute into the ideal MHD equations and keep only first-order
    terms. Seek plane-wave solutions
    $\propto \exp[i(\mathbf{k} \cdot \mathbf{x} - \omega t)]$ with
    wavevector $\mathbf{k}$ at angle $\theta$ to $\mathbf{B}_0$.

    The linearized momentum equation becomes:

    $$-\omega^2 \rho_0 \mathbf{V}_1 = -k^2 c_s^2 (\hat{k} \cdot \mathbf{V}_1)\hat{k} + (\mathbf{k} \times (\mathbf{k} \times (\mathbf{V}_1 \times \mathbf{B}_0))) \times \mathbf{B}_0$$

    using the linearized induction equation
    $\mathbf{B}_1 = -(\mathbf{k} \times (\mathbf{V}_1 \times \mathbf{B}_0))/\omega$
    and Ampere's law.

    **Alfvén mode.** For perturbations $\mathbf{V}_1$ perpendicular to
    both $\mathbf{k}$ and $\mathbf{B}_0$ (shear polarization), the
    pressure force vanishes ($\hat{k} \cdot \mathbf{V}_1 = 0$, so
    $\rho_1 = 0$) and the dispersion relation reduces to:

    $$\omega^2 = k_\parallel^2 v_A^2 = k^2 v_A^2 \cos^2\theta$$

    This is the **Alfvén wave**: an incompressible transverse oscillation
    of field lines, propagating only along $\mathbf{B}_0$, with phase
    speed $v_A = B_0/\sqrt{\rho_0}$.

    **Magnetosonic modes.** For perturbations in the
    $(\mathbf{k}, \mathbf{B}_0)$ plane, the pressure and magnetic forces
    couple. The dispersion relation becomes a quadratic in $\omega^2/k^2$:

    $$\left(\frac{\omega^2}{k^2}\right)^2 - (v_A^2 + c_s^2)\frac{\omega^2}{k^2} + v_A^2 c_s^2 \cos^2\theta = 0$$

    with solutions:

    $$v_{f,s}^2 = \frac{\omega^2}{k^2} = \frac{1}{2}\left[(v_A^2 + c_s^2) \pm \sqrt{(v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 \cos^2\theta}\right]$$

    The $+$ root is the **fast magnetosonic wave** (magnetic pressure and
    gas pressure reinforce). The $-$ root is the **slow magnetosonic wave**
    (they oppose). At perpendicular propagation ($\theta = 90°$,
    $\cos\theta = 0$):

    $$v_f = \sqrt{v_A^2 + c_s^2} \equiv v_{ms}, \qquad v_s = 0$$

    At parallel propagation ($\theta = 0°$): $v_f = \max(v_A, c_s)$ and
    $v_s = \min(v_A, c_s)$.

## CGL Double-Adiabatic Invariants

In a collisionless magnetized plasma, the pressure tensor is not
isotropic. The **CGL theory** [@CGL] provides closure by conserving two
adiabatic invariants along fluid elements:

$$\frac{d}{dt}\left(\frac{P_\perp}{n B}\right) = 0, \qquad \frac{d}{dt}\left(\frac{P_\parallel B^2}{n^3}\right) = 0$$

These govern the [gyrotropic entropy](equations.md#2-thermodynamic-quantities)
$s_{gyro} = \ln(P_\parallel P_\perp^2 / n^5)$ and the
[firehose/mirror instability thresholds](equations.md#9-reconnection-and-anisotropy-diagnostics).

??? abstract "Derivation: from magnetic moment conservation and flux tube geometry"
    **First invariant** ($P_\perp / nB$): the magnetic moment
    $\mu = m v_\perp^2 / (2B)$ is an adiabatic invariant for individual
    particles gyrating in a slowly varying field. For a distribution of
    particles, the perpendicular thermal energy per particle is
    $P_\perp / n$, giving:

    $$\frac{P_\perp}{nB} \propto \frac{\langle v_\perp^2 \rangle}{B} \propto \mu = \text{const}$$

    **Second invariant** ($P_\parallel B^2 / n^3$): consider a flux tube
    of cross-section $A$ and length $L$. Magnetic flux conservation gives
    $BA = \text{const}$, and particle conservation in the tube gives
    $nAL = \text{const}$, so $nL/B = \text{const}$, i.e. $L \propto B/n$.

    For the parallel degree of freedom, the particles bounce between
    mirror points. The parallel adiabatic invariant $J = \oint m v_\parallel \, dl$
    gives $v_\parallel L = \text{const}$, so
    $P_\parallel / n \propto v_\parallel^2 \propto 1/L^2 \propto n^2/B^2$.
    Therefore:

    $$\frac{P_\parallel B^2}{n^3} = \text{const}$$

    **Gyrotropic entropy.** Eliminating $B$ between the two invariants:

    - From the first: $B \propto P_\perp / n$
    - Substituting into the second: $P_\parallel (P_\perp / n)^2 / n^3 = P_\parallel P_\perp^2 / n^5 = \text{const}$

    Hence $s_{gyro} = \ln(P_\parallel P_\perp^2 / n^5)$ is conserved.
    The exponent 5 arises from the geometric constraint
    ($3$ from parallel + $2$ from perpendicular), **not** from the
    adiabatic index $\gamma = 5/3$. See
    [Conventions](conventions.md#gamma-convention-for-pic-entropy) for this
    distinction.

    **Instability thresholds.** When the CGL invariants are violated
    (collisions, wave-particle interactions), the pressure anisotropy
    drives kinetic instabilities:

    - **Firehose** ($P_\parallel > P_\perp + B^2/2$): excess parallel
      pressure bends field lines. Threshold:
      $(P_\parallel - P_\perp)/(B^2/2) > 1$ [@Hellinger].
    - **Mirror** ($P_\perp / P_\parallel > 1 + 1/\beta_\perp$): excess
      perpendicular pressure creates density compressions along field
      lines [@Gary; @Kunz].
