"""Pure NumPy functions for field-level derived quantities.

All functions operate in normalized units (μ₀ = ε₀ = 1 in PIC, μ₀ = 1 in MHD).
Arrays in, arrays out — no FieldDataset dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.types import FloatArray


def _safe_divide(
    numerator: FloatArray,
    denominator: FloatArray,
) -> FloatArray:
    """Divide, returning nan where the denominator is zero.

    What we cannot change, we let pass through.
    """
    out = np.full_like(numerator, np.nan)
    nonzero = denominator != 0
    np.divide(numerator, denominator, out=out, where=nonzero)
    return out


def _vector_magnitude(
    c1: FloatArray,
    c2: FloatArray,
    c3: FloatArray,
) -> FloatArray:
    """Compute the Euclidean magnitude of a 3-component vector field."""
    return np.sqrt(c1**2 + c2**2 + c3**2)


def magnetic_field_magnitude(
    b1: FloatArray, b2: FloatArray, b3: FloatArray
) -> FloatArray:
    r"""Magnetic field magnitude $|\mathbf{B}| = \sqrt{B_1^2 + B_2^2 + B_3^2}$.

    Parameters
    ----------
    b1, b2, b3 : NDArray
        Magnetic field components in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> magnetic_field_magnitude(np.array([3.0]), np.array([4.0]), np.array([0.0]))
    array([5.])
    """
    return _vector_magnitude(b1, b2, b3)


def electric_field_magnitude(
    e1: FloatArray, e2: FloatArray, e3: FloatArray
) -> FloatArray:
    r"""Electric field magnitude $|\mathbf{E}| = \sqrt{E_1^2 + E_2^2 + E_3^2}$.

    Parameters
    ----------
    e1, e2, e3 : NDArray
        Electric field components in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> electric_field_magnitude(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    array([1.])
    """
    return _vector_magnitude(e1, e2, e3)


def current_density_magnitude(
    j1: FloatArray, j2: FloatArray, j3: FloatArray
) -> FloatArray:
    r"""Magnitude $|\mathbf{J}| = \sqrt{J_1^2 + J_2^2 + J_3^2}$ of the current density.

    Parameters
    ----------
    j1, j2, j3 : NDArray
        Current density components in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> current_density_magnitude(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    array([1.73205081])
    """
    return _vector_magnitude(j1, j2, j3)


def velocity_magnitude(v1: FloatArray, v2: FloatArray, v3: FloatArray) -> FloatArray:
    r"""Bulk velocity magnitude $|\mathbf{V}| = \sqrt{V_1^2 + V_2^2 + V_3^2}$.

    Parameters
    ----------
    v1, v2, v3 : NDArray
        Velocity components in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> velocity_magnitude(np.array([3.0]), np.array([4.0]), np.array([0.0]))
    array([5.])
    """
    return _vector_magnitude(v1, v2, v3)


def plasma_beta(pressure: FloatArray, b: FloatArray) -> FloatArray:
    r"""Plasma beta $\beta = 2P/B^2$ (dimensionless).

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    b : NDArray
        Magnetic field magnitude in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> plasma_beta(np.array([1.0]), np.array([1.0]))
    array([2.])
    """
    return _safe_divide(2.0 * pressure, b**2)


def alfven_speed(
    b: FloatArray,
    rho_m: FloatArray,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the Alfvén speed.

    $$v_A = \frac{B}{\sqrt{\mu_0 \rho_m}}$$

    In normalized MHD units where $\mu_0 = 1$: $v_A = B / \sqrt{\rho_m}$.

    When *c* is provided, uses the relativistic form:
    $v_A = c\sqrt{\sigma / (1 + \sigma)}$ where
    $\sigma = B^2 / (\rho_m c^2)$. This approaches $c$ as
    $\sigma \to \infty$ and recovers $B/\sqrt{\rho_m}$ for
    $\sigma \ll 1$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho_m : NDArray
        Mass density in normalized units. Must be non-negative;
        negative values produce NaN (via ``sqrt``).
    c : float or None
        Speed of light. When provided, the relativistic formula is used.

    Returns
    -------
    NDArray
        Alfvén speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> alfven_speed(np.array([1.0]), np.array([4.0]))
    array([0.5])
    """
    if c is not None:
        sigma = _safe_divide(b**2, rho_m * c**2)
        return c * np.sqrt(_safe_divide(sigma, 1.0 + sigma))
    return _safe_divide(b, np.sqrt(rho_m))


def magnetic_energy_density(
    b: FloatArray,
) -> FloatArray:
    r"""Compute the magnetic energy density.

    $$e_B = \frac{B^2}{2}$$

    In SI: $e_B = B^2 / (2\mu_0)$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.

    Returns
    -------
    NDArray
        Magnetic energy density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> magnetic_energy_density(np.array([2.0]))
    array([2.])
    """
    return b**2 / 2.0


def electric_energy_density(
    e: FloatArray,
) -> FloatArray:
    r"""Compute the electric energy density.

    $$e_E = \frac{E^2}{2}$$

    In SI: $e_E = \epsilon_0 E^2 / 2$.

    Parameters
    ----------
    e : NDArray
        Electric field magnitude in normalized units.

    Returns
    -------
    NDArray
        Electric energy density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> electric_energy_density(np.array([3.0]))
    array([4.5])
    """
    return e**2 / 2.0


def kinetic_energy_density(
    rho_m: FloatArray,
    v: FloatArray,
    *,
    lorentz_factor: FloatArray | None = None,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the kinetic energy density.

    $$e_k = \frac{1}{2} \rho_m V^2$$

    When *c* is provided, uses the relativistic form:
    $e_k = (\gamma - 1)\,\rho_m\,c^2$. The Lorentz factor is computed
    from *v* unless *lorentz_factor* is given explicitly (e.g. from
    four-velocity data where $\gamma$ is more accurate).

    Parameters
    ----------
    rho_m : NDArray
        Mass density in normalized units.
    v : NDArray
        Bulk velocity magnitude in normalized units.
    lorentz_factor : NDArray or None
        Pre-computed Lorentz factor $\gamma$. When omitted and *c* is
        provided, $\gamma$ is computed from *v* and *c*.
    c : float or None
        Speed of light. When provided, the relativistic formula is used.

    Returns
    -------
    NDArray
        Kinetic energy density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> kinetic_energy_density(np.array([2.0]), np.array([3.0]))
    array([9.])
    """
    if c is not None:
        if lorentz_factor is None:
            lorentz_factor = 1.0 / np.sqrt(1.0 - v**2 / c**2)
        # Algebraically equivalent to (γ - 1) ρ c², but avoids the
        # catastrophic cancellation when v ≪ c: γ rounds to 1 at v/c <
        # √eps ≈ 1.5e-8, so the subtraction silently returns 0 instead
        # of the non-relativistic limit ½ρv². Identity: γ - 1 =
        # (γ² - 1)/(γ + 1) = (v²/c²) γ²/(γ + 1), hence (γ - 1) c² =
        # γ² v²/(γ + 1). Recovers ½ρv² as γ → 1.
        result: FloatArray = rho_m * v**2 * lorentz_factor**2 / (lorentz_factor + 1.0)
        return result
    return 0.5 * rho_m * v**2


def thermal_energy_density(
    pressure: FloatArray,
    gamma: float = 5.0 / 3.0,
) -> FloatArray:
    r"""Compute the thermal energy density.

    $$e_{th} = \frac{P}{\gamma - 1}$$

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$ (3D).

    Returns
    -------
    NDArray
        Thermal energy density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> thermal_energy_density(np.array([1.0]))
    array([1.5])
    """
    return pressure / (gamma - 1.0)


def thermal_energy_density_trace(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
) -> FloatArray:
    r"""Compute thermal energy density from the pressure tensor trace.

    $$e_{th} = \tfrac{1}{2}\mathrm{Tr}(\mathbf{P})
    = \tfrac{1}{2}(P_{11} + P_{22} + P_{33})$$

    Unlike ``thermal_energy_density`` (which uses $P/(\gamma-1)$), this
    form is exact for any dimensionality or adiabatic index — it is the
    kinetic definition of thermal energy from the second velocity moment.
    Equivalent to ``thermal_energy_density`` when $\gamma = 5/3$ (3D).

    Parameters
    ----------
    p11, p22, p33 : NDArray
        Diagonal pressure tensor components.

    Returns
    -------
    NDArray
        Thermal energy density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> thermal_energy_density_trace(
    ...     np.array([2.0]), np.array([1.0]), np.array([1.0]))
    array([2.])
    """
    return 0.5 * (p11 + p22 + p33)


def poynting_flux(
    e1: FloatArray,
    e2: FloatArray,
    e3: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    r"""Compute the Poynting flux vector.

    $$\mathbf{S} = \mathbf{E} \times \mathbf{B}$$

    In SI: $\mathbf{S} = \mathbf{E} \times \mathbf{B} / \mu_0$.

    Parameters
    ----------
    e1 : NDArray
        First component of the electric field.
    e2 : NDArray
        Second component of the electric field.
    e3 : NDArray
        Third component of the electric field.
    b1 : NDArray
        First component of the magnetic field.
    b2 : NDArray
        Second component of the magnetic field.
    b3 : NDArray
        Third component of the magnetic field.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Poynting flux components $(S_1, S_2, S_3)$.

    Examples
    --------
    >>> import numpy as np
    >>> s1, s2, s3 = poynting_flux(
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([1.0]), np.array([0.0]),
    ... )
    >>> s3.item()
    1.0
    """
    s1 = e2 * b3 - e3 * b2
    s2 = e3 * b1 - e1 * b3
    s3 = e1 * b2 - e2 * b1
    return s1, s2, s3


def internal_energy(
    pressure: FloatArray, rho_m: FloatArray, gamma: float = 5.0 / 3.0
) -> FloatArray:
    r"""Specific internal energy $e_{int} = P / ((\gamma - 1) \rho_m)$.

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$ (3D).

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> internal_energy(np.array([1.0]), np.array([1.0]))
    array([1.5])
    """
    return _safe_divide(pressure, (gamma - 1.0) * rho_m)


def enthalpy(
    pressure: FloatArray,
    rho_m: FloatArray,
    gamma: float = 5.0 / 3.0,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the specific enthalpy.

    $$h = \frac{\gamma P}{(\gamma - 1) \rho_m}$$

    When *c* is provided, uses the relativistic form (constant-$\Gamma$
    Synge-type approximation):
    $h_{rel} = c^2 + \gamma P / ((\gamma-1)\rho_m)$.

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$ (3D).
    c : float or None
        Speed of light. When provided, the relativistic rest-energy
        term $c^2$ is included.

    Returns
    -------
    NDArray
        Specific enthalpy in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> enthalpy(np.array([1.0]), np.array([1.0]))
    array([2.5])
    """
    h = _safe_divide(gamma * pressure, (gamma - 1.0) * rho_m)
    if c is not None:
        return c**2 + h
    return h


def relativistic_enthalpy(
    pressure: FloatArray,
    rho_m: FloatArray,
    gamma: float = 5.0 / 3.0,
    c: float = 1.0,
) -> FloatArray:
    r"""Compute the relativistic specific enthalpy.

    $$h_{rel} = c^2 + \frac{\gamma P}{(\gamma - 1) \rho_m}$$

    Uses the constant-$\Gamma$ (Synge-type) approximation.

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$ (3D).
    c : float
        Speed of light in normalized units. Default is 1.0.

    Returns
    -------
    NDArray
        Relativistic specific enthalpy in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> relativistic_enthalpy(np.array([1.0]), np.array([1.0]))
    array([3.5])
    """
    return enthalpy(pressure, rho_m, gamma, c=c)


def entropy(
    pressure: FloatArray,
    density: FloatArray,
    gamma: float = 5.0 / 3.0,
) -> FloatArray:
    r"""Compute the specific entropy.

    $$s = \ln\!\left(\frac{P}{\rho^\gamma}\right)$$

    For MHD, pass mass density $\rho_m$. For PIC per-species entropy, pass
    number density $n_s$.

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    density : NDArray
        Mass density (MHD) or number density (PIC) in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$ (3D).

    Returns
    -------
    NDArray
        Specific entropy (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> entropy(np.array([1.0]), np.array([1.0]))
    array([0.])
    """
    ratio = _safe_divide(pressure, density**gamma)
    return np.where(ratio > 0, np.log(ratio), np.nan)


def gyrotropic_entropy(
    p_par: FloatArray,
    p_perp: FloatArray,
    density: FloatArray,
) -> FloatArray:
    r"""Compute the gyrotropic entropy from CGL double-adiabatic invariants.

    $$s_{gyro} = \ln\!\left(\frac{P_\parallel P_\perp^2}{n^5}\right)$$

    The exponent 5 arises from combining the two CGL invariants
    ($P_\perp / nB$ and $P_\parallel B^2 / n^3$) and is independent
    of the adiabatic index $\gamma$.

    Parameters
    ----------
    p_par : NDArray
        Pressure parallel to the magnetic field.
    p_perp : NDArray
        Pressure perpendicular to the magnetic field.
    density : NDArray
        Number density.

    Returns
    -------
    NDArray
        Gyrotropic entropy (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> gyrotropic_entropy(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    array([0.])
    """
    ratio = _safe_divide(p_par * p_perp**2, density**5)
    return np.where(ratio > 0, np.log(ratio), np.nan)


def _unit_vector(
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the unit vector of a 3-component vector field.

    Returns NaN where the magnitude is zero. These NaN values propagate
    into downstream consumers (``parallel_pressure``, ``perpendicular_pressure``,
    ``agyrotropy``) in zero-field regions.
    """
    mag = _vector_magnitude(b1, b2, b3)
    return _safe_divide(b1, mag), _safe_divide(b2, mag), _safe_divide(b3, mag)


def temperature(pressure: FloatArray, density: FloatArray) -> FloatArray:
    r"""Temperature $T = P / n$ (energy units; divide by $k_B$ for Kelvin).

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    density : NDArray
        Number density in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> temperature(np.array([2.0]), np.array([4.0]))
    array([0.5])
    """
    return _safe_divide(pressure, density)


def thermal_speed(
    temperature: FloatArray,
    mass: float,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the thermal speed (NRL convention).

    $$v_{th} = \sqrt{T / m}$$

    This is the 1D Maxwellian standard deviation $\sigma$ where
    $f(v_x) \propto \exp(-v_x^2 / (2\sigma^2))$ with $\sigma^2 = T/m$.

    When *c* is provided, caps the result at $c$:
    $v_{th,rel} = v_{th} / \sqrt{1 + v_{th}^2 / c^2}$.

    Parameters
    ----------
    temperature : NDArray
        Temperature in energy units (normalized).
    mass : float
        Particle mass in normalized units.
    c : float or None
        Speed of light. When provided, the relativistic cap is applied.

    Returns
    -------
    NDArray
        Thermal speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> thermal_speed(np.array([4.0]), mass=1.0)
    array([2.])
    """
    v_th: FloatArray = np.sqrt(temperature / mass)
    if c is not None:
        return v_th / np.sqrt(1.0 + v_th**2 / c**2)  # type: ignore[no-any-return]
    return v_th


def gyrofrequency(
    b: FloatArray,
    charge: float,
    mass: float,
    *,
    lorentz_factor: FloatArray | None = None,
) -> FloatArray:
    r"""Compute the cyclotron (gyro) frequency.

    $$\omega_c = \frac{|q| B}{m}$$

    Positive by convention (magnitude of charge is used).

    When *lorentz_factor* ($\gamma$) is provided, uses the relativistic
    form: $\omega_c = |q| B / (\gamma m)$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    charge : float
        Particle charge in normalized units (sign is stripped).
    mass : float
        Particle mass in normalized units.
    lorentz_factor : NDArray or None
        Lorentz factor (thermal or bulk). When provided, particles
        gyrate slower by the factor $1/\gamma$.

    Returns
    -------
    NDArray
        Cyclotron frequency in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> gyrofrequency(np.array([2.0]), charge=-1.0, mass=1.0)
    array([2.])
    """
    omega = np.abs(charge) * b / mass
    if lorentz_factor is not None:
        return _safe_divide(omega, lorentz_factor)
    return omega  # type: ignore[no-any-return]


def plasma_frequency(
    density: FloatArray,
    charge: float,
    mass: float,
    *,
    lorentz_factor: FloatArray | None = None,
) -> FloatArray:
    r"""Compute the plasma frequency.

    $$\omega_p = \sqrt{\frac{n q^2}{m}}$$

    In SI: $\omega_p = \sqrt{n e^2 / (\epsilon_0 m)}$.

    When *lorentz_factor* ($\gamma$) is provided, uses the relativistic
    form: $\omega_{p,rel} = \omega_p / \sqrt{\gamma}$.

    Parameters
    ----------
    density : NDArray
        Number density in normalized units.
    charge : float
        Particle charge in normalized units.
    mass : float
        Particle mass in normalized units.
    lorentz_factor : NDArray or None
        Mean thermal Lorentz factor $\langle\gamma\rangle$. When
        provided, reduces the effective plasma frequency.

    Returns
    -------
    NDArray
        Plasma frequency in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> plasma_frequency(np.array([1.0]), charge=1.0, mass=1.0)
    array([1.])
    """
    omega: FloatArray = np.sqrt(density * charge**2 / mass)
    if lorentz_factor is not None:
        return omega / np.sqrt(lorentz_factor)
    return omega


def skin_depth(
    density: FloatArray,
    charge: float,
    mass: float,
    c: float = 1.0,
    *,
    lorentz_factor: FloatArray | None = None,
) -> FloatArray:
    r"""Compute the skin depth (inertial length).

    $$d = \frac{c}{\omega_p}$$

    When *lorentz_factor* is provided, uses the relativistically
    corrected plasma frequency: $d_{rel} = c / \omega_{p,rel}$.

    Parameters
    ----------
    density : NDArray
        Number density in normalized units.
    charge : float
        Particle charge in normalized units.
    mass : float
        Particle mass in normalized units.
    c : float
        Speed of light in normalized units. Default is 1.0.
    lorentz_factor : NDArray or None
        Mean thermal Lorentz factor. Passed through to
        :func:`plasma_frequency`.

    Returns
    -------
    NDArray
        Skin depth in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> skin_depth(np.array([1.0]), charge=1.0, mass=1.0)
    array([1.])
    """
    return c / plasma_frequency(density, charge, mass, lorentz_factor=lorentz_factor)


def gyroradius(
    temperature: FloatArray,
    b: FloatArray,
    charge: float,
    mass: float,
    *,
    lorentz_factor: FloatArray | None = None,
) -> FloatArray:
    r"""Compute the thermal gyroradius (Larmor radius).

    $$r = \frac{v_{th}}{\omega_c} = \frac{\sqrt{m T}}{|q| B}$$

    Uses the NRL thermal speed convention $v_{th} = \sqrt{T/m}$.

    When *lorentz_factor* ($\gamma$) is provided, the relativistic
    cyclotron frequency $\omega_c / \gamma$ is used, giving
    $r_{rel} = \gamma \cdot r$.

    Parameters
    ----------
    temperature : NDArray
        Temperature in energy units (normalized).
    b : NDArray
        Magnetic field magnitude in normalized units.
    charge : float
        Particle charge in normalized units (sign is stripped).
    mass : float
        Particle mass in normalized units.
    lorentz_factor : NDArray or None
        Lorentz factor (thermal or bulk). When provided, the gyroradius
        increases by a factor of $\gamma$.

    Returns
    -------
    NDArray
        Thermal gyroradius in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> gyroradius(np.array([1.0]), np.array([1.0]), charge=1.0, mass=1.0)
    array([1.])
    """
    r = _safe_divide(np.sqrt(mass * temperature), np.abs(charge) * b)
    if lorentz_factor is not None:
        return r * lorentz_factor
    return r


def debye_length(
    temperature: FloatArray,
    density: FloatArray,
    charge: float,
) -> FloatArray:
    r"""Compute the electron Debye length.

    $$\lambda_D = \sqrt{\frac{T}{n q^2}}$$

    In SI: $\lambda_D = \sqrt{\epsilon_0 T / (n e^2)}$.

    Parameters
    ----------
    temperature : NDArray
        Electron temperature in energy units (normalized).
    density : NDArray
        Electron number density in normalized units.
    charge : float
        Particle charge in normalized units.

    Returns
    -------
    NDArray
        Debye length in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> debye_length(np.array([1.0]), np.array([1.0]), charge=1.0)
    array([1.])
    """
    return np.sqrt(_safe_divide(temperature, density * charge**2))


def sound_speed(
    pressure: FloatArray,
    rho_m: FloatArray,
    gamma: float = 5.0 / 3.0,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the MHD sound speed.

    $$c_s = \sqrt{\frac{\gamma P}{\rho_m}}$$

    When *c* is provided, uses the relativistic form:
    $c_s = c\sqrt{\gamma P / (\rho_m h_{rel})}$ where
    $h_{rel} = c^2 + \gamma P / ((\gamma-1)\rho_m)$.

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$.
    c : float or None
        Speed of light. When provided, the relativistic formula is used.

    Returns
    -------
    NDArray
        Sound speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> sound_speed(np.array([3.0]), np.array([5.0]))
    array([1.])
    """
    if c is not None:
        h_rel = enthalpy(pressure, rho_m, gamma, c=c)
        return c * np.sqrt(_safe_divide(gamma * pressure, rho_m * h_rel))
    return np.sqrt(_safe_divide(gamma * pressure, rho_m))


def ion_acoustic_speed(
    te: FloatArray,
    ti: FloatArray,
    mass: float,
    gamma_e: float = 1.0,
    gamma_i: float = 3.0,
) -> FloatArray:
    r"""Compute the ion acoustic speed.

    $$c_{ia} = \sqrt{\frac{\gamma_e T_e + \gamma_i T_i}{m_i}}$$

    Uses $\gamma_e = 1$ (isothermal electrons) and $\gamma_i = 3$ (1D
    adiabatic ions) by default, following kinetic theory convention.

    Parameters
    ----------
    te : NDArray
        Electron temperature in energy units (normalized).
    ti : NDArray
        Ion temperature in energy units (normalized).
    mass : float
        Ion mass in normalized units.
    gamma_e : float
        Electron adiabatic index. Default is 1.0 (isothermal).
    gamma_i : float
        Ion adiabatic index. Default is 3.0 (1D adiabatic).

    Returns
    -------
    NDArray
        Ion acoustic speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> ion_acoustic_speed(np.array([1.0]), np.array([0.0]), mass=1.0)
    array([1.])
    """
    return np.sqrt((gamma_e * te + gamma_i * ti) / mass)


def magnetosonic_speed(
    v_a: FloatArray,
    c_s: FloatArray,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the fast magnetosonic speed (perpendicular propagation).

    $$v_{ms} = \sqrt{v_A^2 + c_s^2}$$

    This is the maximum fast-mode phase speed at $\theta = 90°$.

    When *c* is provided, uses the relativistic composition:
    $v_{ms}^2 = v_A^2 + c_s^2 - v_A^2 c_s^2 / c^2$, which
    guarantees $v_{ms} < c$.

    Parameters
    ----------
    v_a : NDArray
        Alfvén speed in normalized units.
    c_s : NDArray
        Sound speed in normalized units.
    c : float or None
        Speed of light. When provided, the relativistic composition
        formula is used.

    Returns
    -------
    NDArray
        Fast magnetosonic speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> magnetosonic_speed(np.array([3.0]), np.array([4.0]))
    array([5.])
    """
    if c is not None:
        # You cannae change the laws of physics — v_ms < c, always.
        return np.sqrt(v_a**2 + c_s**2 - v_a**2 * c_s**2 / c**2)
    return np.sqrt(v_a**2 + c_s**2)


def alfven_mach(v: FloatArray, v_a: FloatArray) -> FloatArray:
    r"""Alfvén Mach number $M_A = V/v_A$ (dimensionless).

    Parameters
    ----------
    v : NDArray
        Bulk velocity magnitude in normalized units.
    v_a : NDArray
        Alfvén speed in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> alfven_mach(np.array([2.0]), np.array([1.0]))
    array([2.])
    """
    return _safe_divide(v, v_a)


def magnetosonic_mach(v: FloatArray, v_ms: FloatArray) -> FloatArray:
    r"""Magnetosonic Mach number $M_{ms} = V/v_{ms}$ (dimensionless).

    Parameters
    ----------
    v : NDArray
        Bulk velocity magnitude in normalized units.
    v_ms : NDArray
        Magnetosonic speed in normalized units.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> magnetosonic_mach(np.array([5.0]), np.array([5.0]))
    array([1.])
    """
    return _safe_divide(v, v_ms)


def bulk_velocity(j: FloatArray, rho_c: FloatArray) -> FloatArray:
    r"""Bulk velocity $V_s = J_s / \rho_{c,s}$ (per-component, per-species).

    Uses charge density directly (consistent with the current-density moment)
    rather than $n \cdot q$, which may have a different normalization.

    Parameters
    ----------
    j : NDArray
        Current density component ($J_1$, $J_2$, or $J_3$) for a species.
    rho_c : NDArray
        Charge density of the species.

    Returns
    -------
    NDArray

    Examples
    --------
    >>> import numpy as np
    >>> bulk_velocity(np.array([0.5]), np.array([2.0]))
    array([0.25])
    """
    return _safe_divide(j, rho_c)


def kinetic_energy_flux_component(
    v_comp: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    v3: FloatArray,
    rho_c: FloatArray,
    charge: float,
    mass: float,
) -> FloatArray:
    r"""Compute one component of the kinetic energy flux.

    $$KEF_i = \frac{1}{2} n\, m\, |\mathbf{V}|^2\, V_i$$

    where $n = |\rho_c / q|$.

    Parameters
    ----------
    v_comp : NDArray
        Velocity component ($V_1$, $V_2$, or $V_3$).
    v1, v2, v3 : NDArray
        All three velocity components.
    rho_c : NDArray
        Charge density of the species.
    charge : float
        Species charge in normalized units.
    mass : float
        Species mass in normalized units.

    Returns
    -------
    NDArray
        Kinetic energy flux component in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([2.0])
    >>> z = np.array([0.0])
    >>> kinetic_energy_flux_component(v, v, z, z, np.array([1.0]), 1.0, 1.0)
    array([4.])
    """
    number_density = np.abs(rho_c) / abs(charge)
    velocity_sq = v1**2 + v2**2 + v3**2
    return 0.5 * number_density * mass * velocity_sq * v_comp


def heat_flux_component(
    ef_comp: FloatArray,
    kef_comp: FloatArray,
) -> FloatArray:
    r"""Compute one component of the heat flux (thermal energy flux residual).

    $$HF_i = EF_i - KEF_i$$

    The residual captures the enthalpy flux $(5/2) P V_i$ and the heat
    flux vector $q_i$ from non-Maxwellian features of the distribution.

    Parameters
    ----------
    ef_comp : NDArray
        Total energy flux component (from particle moments).
    kef_comp : NDArray
        Kinetic energy flux component (bulk flow contribution).

    Returns
    -------
    NDArray
        Heat flux component in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> heat_flux_component(np.array([10.0]), np.array([4.0]))
    array([6.])
    """
    return ef_comp - kef_comp


def enthalpy_flux_component(
    pressure: FloatArray,
    v_comp: FloatArray,
    gamma: float = 5.0 / 3.0,
) -> FloatArray:
    r"""Compute one component of the enthalpy flux.

    $$EHF_i = \frac{\gamma}{\gamma - 1}\, P\, V_i$$

    This is the adiabatic (fluid) enthalpy flux. It captures the
    $P\,dV$ work and internal energy transport but not the heat flux
    vector $\mathbf{q}$ from non-Maxwellian features.

    Works for both MHD (total $P$, fluid $V$) and PIC (per-species
    $P_s$, $V_s$).

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure (total or per-species).
    v_comp : NDArray
        Velocity component ($V_1$, $V_2$, or $V_3$).
    gamma : float
        Adiabatic index (default 5/3).

    Returns
    -------
    NDArray
        Enthalpy flux component in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> enthalpy_flux_component(np.array([1.0]), np.array([2.0]), 5/3)
    array([5.])
    """
    return (gamma / (gamma - 1.0)) * pressure * v_comp


def conductive_heat_flux_component(
    hf_comp: FloatArray,
    ehf_comp: FloatArray,
) -> FloatArray:
    r"""Compute one component of the conductive heat flux vector.

    $$q_i = HF_i - EHF_i = (EF_i - KEF_i) - \frac{\gamma}{\gamma-1} P V_i$$

    The residual captures non-adiabatic energy transport: heat conduction
    and non-Maxwellian contributions from the full distribution function.
    Vanishes for a drifting Maxwellian in ideal MHD.

    Requires the total energy flux moment (EF) from the simulation
    output — available from PIC codes and multi-moment MHD.

    Parameters
    ----------
    hf_comp : NDArray
        Total thermal flux component ($HF_i = EF_i - KEF_i$).
    ehf_comp : NDArray
        Enthalpy flux component ($EHF_i$).

    Returns
    -------
    NDArray
        Conductive heat flux component in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> conductive_heat_flux_component(np.array([6.0]), np.array([5.0]))
    array([1.])
    """
    return hf_comp - ehf_comp


def species_mass_density(
    rho_c: FloatArray,
    charge: float,
    mass: float,
) -> FloatArray:
    r"""Compute per-species mass density from charge density.

    $$\rho_{m,s} = \frac{|\rho_{c,s}|}{|q_s|}\, m_s = n_s\, m_s$$

    Parameters
    ----------
    rho_c : NDArray
        Charge density of the species.
    charge : float
        Species charge in normalized units.
    mass : float
        Species mass in normalized units.

    Returns
    -------
    NDArray
        Mass density in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> species_mass_density(np.array([-2.0]), -1.0, 0.5)
    array([1.])
    """
    return np.abs(rho_c) * mass / abs(charge)


def total_pressure(p_e: FloatArray, p_i: FloatArray) -> FloatArray:
    r"""Compute total scalar pressure from electron and ion partial pressures.

    $$P = P_e + P_i$$

    Parameters
    ----------
    p_e : NDArray
        Electron scalar pressure.
    p_i : NDArray
        Ion scalar pressure.

    Returns
    -------
    NDArray
        Total scalar pressure in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> total_pressure(np.array([2.0]), np.array([3.0]))
    array([5.])
    """
    return p_e + p_i


def isotropic_pressure(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
) -> FloatArray:
    r"""Compute the isotropic scalar pressure from the pressure tensor trace.

    $$P_{iso} = \frac{1}{3}\mathrm{Tr}(\mathbf{P})
              = \frac{1}{3}(P_{11} + P_{22} + P_{33})
              = \frac{P_\parallel + 2\,P_\perp}{3}$$

    The trace is a coordinate invariant — this gives the same result
    regardless of the orientation of the coordinate axes.

    Parameters
    ----------
    p11 : NDArray
        Pressure tensor component $P_{11}$.
    p22 : NDArray
        Pressure tensor component $P_{22}$.
    p33 : NDArray
        Pressure tensor component $P_{33}$.

    Returns
    -------
    NDArray
        Isotropic scalar pressure in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> isotropic_pressure(np.array([3.0]), np.array([6.0]), np.array([9.0]))
    array([6.])
    """
    return (p11 + p22 + p33) / 3


def parallel_pressure(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
    p12: FloatArray,
    p13: FloatArray,
    p23: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> FloatArray:
    r"""Compute the pressure parallel to the magnetic field.

    $$P_\parallel = \hat{b} \cdot \mathbf{P} \cdot \hat{b}$$

    Parameters
    ----------
    p11 : NDArray
        Pressure tensor component $P_{11}$.
    p22 : NDArray
        Pressure tensor component $P_{22}$.
    p33 : NDArray
        Pressure tensor component $P_{33}$.
    p12 : NDArray
        Pressure tensor component $P_{12}$.
    p13 : NDArray
        Pressure tensor component $P_{13}$.
    p23 : NDArray
        Pressure tensor component $P_{23}$.
    b1 : NDArray
        First component of the magnetic field.
    b2 : NDArray
        Second component of the magnetic field.
    b3 : NDArray
        Third component of the magnetic field.

    Returns
    -------
    NDArray
        Parallel pressure in normalized units.
        Returns NaN where $|B| = 0$ (undefined magnetic direction).

    Examples
    --------
    >>> import numpy as np
    >>> parallel_pressure(
    ...     np.array([1.0]), np.array([2.0]), np.array([3.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([1.0]),
    ... )
    array([3.])
    """
    bhat_1, bhat_2, bhat_3 = _unit_vector(b1, b2, b3)
    result: FloatArray = (
        bhat_1**2 * p11
        + bhat_2**2 * p22
        + bhat_3**2 * p33
        + 2.0 * (bhat_1 * bhat_2 * p12 + bhat_1 * bhat_3 * p13 + bhat_2 * bhat_3 * p23)
    )
    return result


def perpendicular_pressure(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
    p12: FloatArray,
    p13: FloatArray,
    p23: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> FloatArray:
    r"""Compute the pressure perpendicular to the magnetic field.

    $$P_\perp = \frac{\mathrm{Tr}(\mathbf{P}) - P_\parallel}{2}$$

    Parameters
    ----------
    p11 : NDArray
        Pressure tensor component $P_{11}$.
    p22 : NDArray
        Pressure tensor component $P_{22}$.
    p33 : NDArray
        Pressure tensor component $P_{33}$.
    p12 : NDArray
        Pressure tensor component $P_{12}$.
    p13 : NDArray
        Pressure tensor component $P_{13}$.
    p23 : NDArray
        Pressure tensor component $P_{23}$.
    b1 : NDArray
        First component of the magnetic field.
    b2 : NDArray
        Second component of the magnetic field.
    b3 : NDArray
        Third component of the magnetic field.

    Returns
    -------
    NDArray
        Perpendicular pressure in normalized units.
        Returns NaN where $|B| = 0$ (undefined magnetic direction).

    Examples
    --------
    >>> import numpy as np
    >>> perpendicular_pressure(
    ...     np.array([1.0]), np.array([2.0]), np.array([3.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([1.0]),
    ... )
    array([1.5])
    """
    p_par = parallel_pressure(p11, p22, p33, p12, p13, p23, b1, b2, b3)
    trace = p11 + p22 + p33
    result: FloatArray = (trace - p_par) / 2.0
    return result


def agyrotropy(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
    p12: FloatArray,
    p13: FloatArray,
    p23: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> FloatArray:
    r"""Compute the agyrotropy measure (Swisdak 2016).

    $$Q = 1 - \frac{4 I_2}{I_1^2}$$

    where $I_1 = \mathrm{Tr}(\mathbf{P}) - P_\parallel$ and
    $I_2 = (I_1^2 - \|\boldsymbol{\Pi}\|_F^2) / 2$, with
    $\boldsymbol{\Pi} = (\mathbf{I} - \hat{b}\hat{b}) \cdot \mathbf{P}
    \cdot (\mathbf{I} - \hat{b}\hat{b})$ the double-projected
    perpendicular pressure tensor (Swisdak, J. Geophys. Res. Space
    Physics, 121, 5549-5565, 2016).

    Bounded $[0, 1]$: 0 is gyrotropic, 1 is maximally agyrotropic.
    Returns NaN where $|B| = 0$ (undefined magnetic direction).

    Parameters
    ----------
    p11 : NDArray
        Pressure tensor component $P_{11}$.
    p22 : NDArray
        Pressure tensor component $P_{22}$.
    p33 : NDArray
        Pressure tensor component $P_{33}$.
    p12 : NDArray
        Pressure tensor component $P_{12}$.
    p13 : NDArray
        Pressure tensor component $P_{13}$.
    p23 : NDArray
        Pressure tensor component $P_{23}$.
    b1 : NDArray
        First component of the magnetic field.
    b2 : NDArray
        Second component of the magnetic field.
    b3 : NDArray
        Third component of the magnetic field.

    Returns
    -------
    NDArray
        Agyrotropy $Q \in [0, 1]$ (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> agyrotropy(
    ...     np.array([1.0]), np.array([1.0]), np.array([1.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([1.0]),
    ... )
    array([0.])
    """
    bhat_1, bhat_2, bhat_3 = _unit_vector(b1, b2, b3)

    p_par = (
        bhat_1**2 * p11
        + bhat_2**2 * p22
        + bhat_3**2 * p33
        + 2.0 * (bhat_1 * bhat_2 * p12 + bhat_1 * bhat_3 * p13 + bhat_2 * bhat_3 * p23)
    )

    invariant_1 = p11 + p22 + p33 - p_par

    # Double-projected perpendicular tensor: Π = (I-b̂b̂)·P·(I-b̂b̂)
    # Π_ij = P_ij - (Pb̂)_i b̂_j - b̂_i (Pb̂)_j + P_∥ b̂_i b̂_j
    p_dot_bhat_1 = p11 * bhat_1 + p12 * bhat_2 + p13 * bhat_3
    p_dot_bhat_2 = p12 * bhat_1 + p22 * bhat_2 + p23 * bhat_3
    p_dot_bhat_3 = p13 * bhat_1 + p23 * bhat_2 + p33 * bhat_3

    perp_11 = p11 - 2.0 * p_dot_bhat_1 * bhat_1 + p_par * bhat_1**2
    perp_22 = p22 - 2.0 * p_dot_bhat_2 * bhat_2 + p_par * bhat_2**2
    perp_33 = p33 - 2.0 * p_dot_bhat_3 * bhat_3 + p_par * bhat_3**2
    perp_12 = (
        p12 - p_dot_bhat_1 * bhat_2 - bhat_1 * p_dot_bhat_2 + p_par * bhat_1 * bhat_2
    )
    perp_13 = (
        p13 - p_dot_bhat_1 * bhat_3 - bhat_1 * p_dot_bhat_3 + p_par * bhat_1 * bhat_3
    )
    perp_23 = (
        p23 - p_dot_bhat_2 * bhat_3 - bhat_2 * p_dot_bhat_3 + p_par * bhat_2 * bhat_3
    )

    frobenius_norm_sq = (
        perp_11**2
        + perp_22**2
        + perp_33**2
        + 2.0 * (perp_12**2 + perp_13**2 + perp_23**2)
    )

    invariant_2 = (invariant_1**2 - frobenius_norm_sq) / 2.0

    result: FloatArray = 1.0 - _safe_divide(4.0 * invariant_2, invariant_1**2)
    return result


def j_dot_e(
    j1: FloatArray,
    j2: FloatArray,
    j3: FloatArray,
    e1: FloatArray,
    e2: FloatArray,
    e3: FloatArray,
) -> FloatArray:
    r"""Compute the electromagnetic energy conversion rate.

    $$\mathbf{J} \cdot \mathbf{E} = J_1 E_1 + J_2 E_2 + J_3 E_3$$

    Positive values indicate electromagnetic-to-kinetic energy transfer
    (particles gaining energy from fields). [Jack] §6.8,
    [Zenitani & Hoshino 2001].

    Parameters
    ----------
    j1 : NDArray
        First component of current density.
    j2 : NDArray
        Second component of current density.
    j3 : NDArray
        Third component of current density.
    e1 : NDArray
        First component of electric field.
    e2 : NDArray
        Second component of electric field.
    e3 : NDArray
        Third component of electric field.

    Returns
    -------
    NDArray
        Energy conversion rate (energy density per unit time).

    Examples
    --------
    >>> import numpy as np
    >>> j_dot_e(
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([2.0]), np.array([0.0]), np.array([0.0]),
    ... )
    array([2.])
    """
    return j1 * e1 + j2 * e2 + j3 * e3


def ideal_electric_field(
    v1: FloatArray,
    v2: FloatArray,
    v3: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the ideal (convective) electric field.

    $$\mathbf{E}_{ideal} = -\mathbf{V} \times \mathbf{B}$$

    The ideal Ohm's law contribution. In perfect ideal MHD, the total
    electric field equals this term. [Chen] §4.3, [NRL].

    Parameters
    ----------
    v1 : NDArray
        First component of bulk velocity.
    v2 : NDArray
        Second component of bulk velocity.
    v3 : NDArray
        Third component of bulk velocity.
    b1 : NDArray
        First component of magnetic field.
    b2 : NDArray
        Second component of magnetic field.
    b3 : NDArray
        Third component of magnetic field.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Ideal electric field components.

    Examples
    --------
    >>> import numpy as np
    >>> e1, e2, e3 = ideal_electric_field(
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([1.0]),
    ... )
    >>> e2.item()
    1.0
    """
    return (
        -(v2 * b3 - v3 * b2),
        -(v3 * b1 - v1 * b3),
        -(v1 * b2 - v2 * b1),
    )


def non_ideal_electric_field(
    e1: FloatArray,
    e2: FloatArray,
    e3: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    v3: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the non-ideal electric field (frozen-in violation).

    $$\mathbf{E}' = \mathbf{E} + \mathbf{V} \times \mathbf{B}$$

    Zero in ideal MHD — resistance is futile. Non-zero where the
    frozen-in condition breaks down (reconnection sites, resistive
    regions). The generalized Ohm's law decomposes this into Hall,
    pressure gradient, and inertial terms. [Birn & Priest 2007],
    [Hesse et al. 2011].

    Parameters
    ----------
    e1 : NDArray
        First component of total electric field.
    e2 : NDArray
        Second component of total electric field.
    e3 : NDArray
        Third component of total electric field.
    v1 : NDArray
        First component of bulk velocity.
    v2 : NDArray
        Second component of bulk velocity.
    v3 : NDArray
        Third component of bulk velocity.
    b1 : NDArray
        First component of magnetic field.
    b2 : NDArray
        Second component of magnetic field.
    b3 : NDArray
        Third component of magnetic field.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Non-ideal electric field components.

    Examples
    --------
    >>> import numpy as np
    >>> e1, e2, e3 = non_ideal_electric_field(
    ...     np.array([0.0]), np.array([0.0]), np.array([0.5]),
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([1.0]), np.array([0.0]),
    ... )
    >>> e3.item()
    1.5
    """
    return (
        e1 + (v2 * b3 - v3 * b2),
        e2 + (v3 * b1 - v1 * b3),
        e3 + (v1 * b2 - v2 * b1),
    )


def hall_electric_field(
    j1: FloatArray,
    j2: FloatArray,
    j3: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
    n: FloatArray,
    charge: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the Hall electric field.

    $$\mathbf{E}_{Hall} = \frac{\mathbf{J} \times \mathbf{B}}{n |q|}$$

    The Hall term in the generalized Ohm's law, using the charge
    magnitude $|q|$ (always positive). Dominant at ion skin depth
    scales where ion and electron motions decouple.
    [Birn & Priest 2007], [Hesse et al. 2011].

    Parameters
    ----------
    j1 : NDArray
        First component of current density.
    j2 : NDArray
        Second component of current density.
    j3 : NDArray
        Third component of current density.
    b1 : NDArray
        First component of magnetic field.
    b2 : NDArray
        Second component of magnetic field.
    b3 : NDArray
        Third component of magnetic field.
    n : NDArray
        Number density of the charge-carrying species.
    charge : float
        Charge of the species (in code units). The absolute value
        is used — sign does not affect the result.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Hall electric field components.

    Examples
    --------
    >>> import numpy as np
    >>> e1, e2, e3 = hall_electric_field(
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([1.0]),
    ...     np.array([2.0]), -1.0,
    ... )
    >>> e2.item()
    -0.5
    """
    charge_density_abs = n * abs(charge)
    return (
        _safe_divide(j2 * b3 - j3 * b2, charge_density_abs),
        _safe_divide(j3 * b1 - j1 * b3, charge_density_abs),
        _safe_divide(j1 * b2 - j2 * b1, charge_density_abs),
    )


def firehose_parameter(
    p_par: FloatArray,
    p_perp: FloatArray,
    b: FloatArray,
) -> FloatArray:
    r"""Compute the firehose instability parameter.

    $$\mathcal{F} = \frac{P_\parallel - P_\perp}{B^2 / 2} - 1$$

    Unstable when $\mathcal{F} > 0$ (parallel pressure excess drives
    field-line bending). [Hellinger et al. 2006], [Gary 1993].

    Parameters
    ----------
    p_par : NDArray
        Parallel pressure.
    p_perp : NDArray
        Perpendicular pressure.
    b : NDArray
        Magnetic field magnitude.

    Returns
    -------
    NDArray
        Firehose parameter (dimensionless). Positive = unstable.

    Examples
    --------
    >>> import numpy as np
    >>> firehose_parameter(np.array([3.0]), np.array([1.0]), np.array([1.0]))
    array([3.])
    """
    return _safe_divide(p_par - p_perp, 0.5 * b**2) - 1.0


def mirror_parameter(
    p_par: FloatArray,
    p_perp: FloatArray,
    b: FloatArray,
) -> FloatArray:
    r"""Compute the mirror instability parameter.

    $$\mathcal{M} = \frac{P_\perp}{P_\parallel} - 1 - \frac{1}{\beta_\perp}$$

    where $\beta_\perp = 2 P_\perp / B^2$. Unstable when $\mathcal{M} > 0$
    (perpendicular pressure excess drives density compressions).
    [Hellinger et al. 2006], [Kunz et al. 2014].

    Parameters
    ----------
    p_par : NDArray
        Parallel pressure.
    p_perp : NDArray
        Perpendicular pressure.
    b : NDArray
        Magnetic field magnitude.

    Returns
    -------
    NDArray
        Mirror parameter (dimensionless). Positive = unstable.

    Examples
    --------
    >>> import numpy as np
    >>> mirror_parameter(np.array([1.0]), np.array([2.0]), np.array([1.0]))
    array([0.75])
    """
    beta_perp = _safe_divide(2.0 * p_perp, b**2)
    return _safe_divide(p_perp, p_par) - 1.0 - _safe_divide(np.ones_like(b), beta_perp)


def magnetic_shear_angle(
    b1_a: FloatArray,
    b2_a: FloatArray,
    b3_a: FloatArray,
    b1_b: FloatArray,
    b2_b: FloatArray,
    b3_b: FloatArray,
) -> FloatArray:
    r"""Compute the angle between two magnetic field vectors.

    $$\theta = \arccos\left(\frac{\mathbf{B}_a \cdot \mathbf{B}_b}
    {|\mathbf{B}_a|\,|\mathbf{B}_b|}\right)$$

    Used for current sheet characterization and component reconnection
    analysis. [Trattner et al. 2007].

    Parameters
    ----------
    b1_a : NDArray
        First component of magnetic field A.
    b2_a : NDArray
        Second component of magnetic field A.
    b3_a : NDArray
        Third component of magnetic field A.
    b1_b : NDArray
        First component of magnetic field B.
    b2_b : NDArray
        Second component of magnetic field B.
    b3_b : NDArray
        Third component of magnetic field B.

    Returns
    -------
    NDArray
        Shear angle in radians, in $[0, \pi]$.

    Examples
    --------
    >>> import numpy as np
    >>> angle = magnetic_shear_angle(
    ...     np.array([1.0]), np.array([0.0]), np.array([0.0]),
    ...     np.array([0.0]), np.array([1.0]), np.array([0.0]),
    ... )
    >>> np.testing.assert_allclose(angle, np.pi / 2, atol=1e-15)
    """
    dot = b1_a * b1_b + b2_a * b2_b + b3_a * b3_b
    mag_a = np.sqrt(b1_a**2 + b2_a**2 + b3_a**2)
    mag_b = np.sqrt(b1_b**2 + b2_b**2 + b3_b**2)
    cos_theta = _safe_divide(dot, mag_a * mag_b)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def magnetic_flux_function(
    b2: FloatArray,
    dx: float,
    dy: float,
) -> FloatArray:
    r"""Compute the magnetic flux function for 2D geometry.

    $$\psi(x, y) = -\int_0^x B_2(x', y)\, dx'$$

    where $\mathbf{B} = \nabla\psi \times \hat{z}$, giving
    $B_1 = \partial\psi/\partial x_2$ and
    $B_2 = -\partial\psi/\partial x_1$. Contours of $\psi$ are
    in-plane magnetic field lines. The reconnection rate equals
    $\partial\psi/\partial t$ at the X-point. [Biskamp 2000] §3.1.

    Assumes Cartesian geometry. In cylindrical axisymmetric (r-z plane),
    the flux function generalizes to $\psi = -\int r B_z\, dr$ with the
    metric factor $r$.

    Parameters
    ----------
    b2 : NDArray
        Second component of the magnetic field (perpendicular to the
        integration direction). Shape ``(nx, ny)`` for 2D data.
    dx : float
        Grid spacing along the first axis.
    dy : float
        Grid spacing along the second axis (unused, accepted for
        compatibility with the grid-dependent dispatch).

    Returns
    -------
    NDArray
        Flux function $\psi$ (same shape as *b2*).

    Raises
    ------
    ValueError
        If *b2* is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> b2 = np.ones((4, 3))
    >>> psi = magnetic_flux_function(b2, 0.5, 1.0)
    >>> np.testing.assert_allclose(psi[0, :], -0.5)
    """
    if b2.ndim != 2:
        msg = f"magnetic_flux_function requires 2D data, got {b2.ndim}D"
        raise ValueError(msg)
    return -np.cumsum(b2 * dx, axis=0)


def lorentz_factor(
    v: FloatArray,
    c: float = 1.0,
) -> FloatArray:
    r"""Compute the bulk Lorentz factor from three-velocity magnitude.

    $$\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$$

    Bounded $[1, \infty)$. Suffers from catastrophic cancellation when
    $v \approx c$; prefer :func:`lorentz_factor_from_four_velocity` when
    four-velocity data is available.

    Parameters
    ----------
    v : NDArray
        Bulk velocity magnitude $|\mathbf{V}|$ in normalized units.
    c : float
        Speed of light in normalized units. Default is 1.0.

    Returns
    -------
    NDArray
        Lorentz factor $\gamma \geq 1$.

    Examples
    --------
    >>> import numpy as np
    >>> lorentz_factor(np.array([0.0]), c=1.0)
    array([1.])
    >>> np.testing.assert_allclose(
    ...     lorentz_factor(np.array([0.6]), c=1.0), 1.25, rtol=1e-15)
    """
    return 1.0 / np.sqrt(1.0 - v**2 / c**2)  # type: ignore[no-any-return]


def lorentz_factor_from_four_velocity(
    u: FloatArray,
    c: float = 1.0,
) -> FloatArray:
    r"""Compute the Lorentz factor from four-velocity magnitude.

    $$\gamma = \sqrt{1 + u^2/c^2}$$

    where $u = \gamma v$ is the spatial part of the four-velocity.
    Numerically stable at all speeds — no catastrophic cancellation
    near $v \approx c$. Preferred when four-velocity data is available
    (TRISTAN-MP, Zeltron, OSIRIS).

    Parameters
    ----------
    u : NDArray
        Four-velocity magnitude $|\mathbf{u}| = \gamma |\mathbf{v}|$.
    c : float
        Speed of light in normalized units. Default is 1.0.

    Returns
    -------
    NDArray
        Lorentz factor $\gamma \geq 1$.

    Examples
    --------
    >>> import numpy as np
    >>> lorentz_factor_from_four_velocity(np.array([0.0]), c=1.0)
    array([1.])
    """
    return np.sqrt(1.0 + u**2 / c**2)  # type: ignore[no-any-return]


def magnetization(
    b: FloatArray,
    rho_m: FloatArray,
    c: float = 1.0,
) -> FloatArray:
    r"""Compute the magnetization parameter.

    $$\sigma = \frac{B^2}{\rho_m c^2}$$

    Measures the ratio of magnetic energy density to rest-mass energy
    density. $\sigma \ll 1$: matter-dominated (non-relativistic MHD).
    $\sigma \gg 1$: magnetically dominated (pulsar winds, jets).

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    c : float
        Speed of light in normalized units. Default is 1.0.

    Returns
    -------
    NDArray
        Magnetization parameter $\sigma$ (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> magnetization(np.array([1.0]), np.array([1.0]), c=1.0)
    array([1.])
    >>> magnetization(np.array([2.0]), np.array([1.0]), c=2.0)
    array([1.])
    """
    return _safe_divide(b**2, rho_m * c**2)


__all__ = [
    "agyrotropy",
    "alfven_mach",
    "alfven_speed",
    "bulk_velocity",
    "conductive_heat_flux_component",
    "current_density_magnitude",
    "debye_length",
    "electric_energy_density",
    "electric_field_magnitude",
    "enthalpy",
    "enthalpy_flux_component",
    "entropy",
    "firehose_parameter",
    "gyrofrequency",
    "gyroradius",
    "gyrotropic_entropy",
    "hall_electric_field",
    "heat_flux_component",
    "ideal_electric_field",
    "internal_energy",
    "ion_acoustic_speed",
    "isotropic_pressure",
    "j_dot_e",
    "kinetic_energy_density",
    "kinetic_energy_flux_component",
    "lorentz_factor",
    "lorentz_factor_from_four_velocity",
    "magnetic_energy_density",
    "magnetic_field_magnitude",
    "magnetic_flux_function",
    "magnetic_shear_angle",
    "magnetization",
    "magnetosonic_mach",
    "magnetosonic_speed",
    "mirror_parameter",
    "non_ideal_electric_field",
    "parallel_pressure",
    "perpendicular_pressure",
    "plasma_beta",
    "plasma_frequency",
    "poynting_flux",
    "relativistic_enthalpy",
    "skin_depth",
    "sound_speed",
    "species_mass_density",
    "temperature",
    "thermal_energy_density",
    "thermal_energy_density_trace",
    "thermal_speed",
    "total_pressure",
    "velocity_magnitude",
]
