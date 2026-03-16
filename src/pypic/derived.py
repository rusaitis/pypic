"""Pure NumPy functions for field-level derived quantities.

All functions operate in normalized units (μ₀ = ε₀ = 1 in PIC, μ₀ = 1 in MHD).
Arrays in, arrays out — no FieldDataset dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def _vector_magnitude(
    c1: NDArray[np.floating[Any]],
    c2: NDArray[np.floating[Any]],
    c3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the Euclidean magnitude of a 3-component vector field."""
    return np.sqrt(c1**2 + c2**2 + c3**2)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def magnetic_field_magnitude(
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the magnetic field magnitude.

    $$|\mathbf{B}| = \sqrt{B_1^2 + B_2^2 + B_3^2}$$

    Parameters
    ----------
    b1 : NDArray
        First component of the magnetic field.
    b2 : NDArray
        Second component of the magnetic field.
    b3 : NDArray
        Third component of the magnetic field.

    Returns
    -------
    NDArray
        Magnetic field magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> magnetic_field_magnitude(np.array([3.0]), np.array([4.0]), np.array([0.0]))
    array([5.])
    """
    return _vector_magnitude(b1, b2, b3)


def electric_field_magnitude(
    e1: NDArray[np.floating[Any]],
    e2: NDArray[np.floating[Any]],
    e3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the electric field magnitude.

    $$|\mathbf{E}| = \sqrt{E_1^2 + E_2^2 + E_3^2}$$

    Parameters
    ----------
    e1 : NDArray
        First component of the electric field.
    e2 : NDArray
        Second component of the electric field.
    e3 : NDArray
        Third component of the electric field.

    Returns
    -------
    NDArray
        Electric field magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> electric_field_magnitude(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    array([1.])
    """
    return _vector_magnitude(e1, e2, e3)


def current_density_magnitude(
    j1: NDArray[np.floating[Any]],
    j2: NDArray[np.floating[Any]],
    j3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the current density magnitude.

    $$|\mathbf{J}| = \sqrt{J_1^2 + J_2^2 + J_3^2}$$

    Parameters
    ----------
    j1 : NDArray
        First component of the current density.
    j2 : NDArray
        Second component of the current density.
    j3 : NDArray
        Third component of the current density.

    Returns
    -------
    NDArray
        Current density magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> current_density_magnitude(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    array([1.73205081])
    """
    return _vector_magnitude(j1, j2, j3)


def velocity_magnitude(
    v1: NDArray[np.floating[Any]],
    v2: NDArray[np.floating[Any]],
    v3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the bulk velocity magnitude.

    $$|\mathbf{V}| = \sqrt{V_1^2 + V_2^2 + V_3^2}$$

    Parameters
    ----------
    v1 : NDArray
        First component of the velocity.
    v2 : NDArray
        Second component of the velocity.
    v3 : NDArray
        Third component of the velocity.

    Returns
    -------
    NDArray
        Velocity magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> velocity_magnitude(np.array([3.0]), np.array([4.0]), np.array([0.0]))
    array([5.])
    """
    return _vector_magnitude(v1, v2, v3)


def plasma_beta(
    pressure: NDArray[np.floating[Any]],
    b_magnitude: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the plasma beta.

    $$\beta = \frac{2P}{B^2}$$

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    b_magnitude : NDArray
        Magnetic field magnitude in normalized units.

    Returns
    -------
    NDArray
        Plasma beta (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> plasma_beta(np.array([1.0]), np.array([1.0]))
    array([2.])
    """
    return 2.0 * pressure / b_magnitude**2


def alfven_speed(
    b: NDArray[np.floating[Any]],
    rho_m: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the Alfvén speed.

    $$v_A = \frac{B}{\sqrt{\mu_0 \rho_m}}$$

    In normalized MHD units where $\mu_0 = 1$: $v_A = B / \sqrt{\rho_m}$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho_m : NDArray
        Mass density in normalized units.

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
    return b / np.sqrt(rho_m)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def magnetic_energy_density(
    b_magnitude: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the magnetic energy density.

    $$e_B = \frac{B^2}{2}$$

    In SI: $e_B = B^2 / (2\mu_0)$.

    Parameters
    ----------
    b_magnitude : NDArray
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
    return b_magnitude**2 / 2.0


def electric_energy_density(
    e_magnitude: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the electric energy density.

    $$e_E = \frac{E^2}{2}$$

    In SI: $e_E = \epsilon_0 E^2 / 2$.

    Parameters
    ----------
    e_magnitude : NDArray
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
    return e_magnitude**2 / 2.0


def kinetic_energy_density(
    rho_m: NDArray[np.floating[Any]],
    v_magnitude: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the kinetic energy density.

    $$e_k = \frac{1}{2} \rho_m V^2$$

    Parameters
    ----------
    rho_m : NDArray
        Mass density in normalized units.
    v_magnitude : NDArray
        Bulk velocity magnitude in normalized units.

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
    return 0.5 * rho_m * v_magnitude**2


def thermal_energy_density(
    pressure: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
) -> NDArray[np.floating[Any]]:
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


def poynting_flux(
    e1: NDArray[np.floating[Any]],
    e2: NDArray[np.floating[Any]],
    e3: NDArray[np.floating[Any]],
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
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


def internal_energy_density(
    pressure: NDArray[np.floating[Any]],
    rho_m: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
) -> NDArray[np.floating[Any]]:
    r"""Compute the specific internal energy.

    $$e_{int} = \frac{P}{(\gamma - 1) \rho_m}$$

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
        Specific internal energy in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> internal_energy_density(np.array([1.0]), np.array([1.0]))
    array([1.5])
    """
    return pressure / ((gamma - 1.0) * rho_m)


def enthalpy(
    pressure: NDArray[np.floating[Any]],
    rho_m: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
) -> NDArray[np.floating[Any]]:
    r"""Compute the specific enthalpy.

    $$h = \frac{\gamma P}{(\gamma - 1) \rho_m}$$

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
        Specific enthalpy in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> enthalpy(np.array([1.0]), np.array([1.0]))
    array([2.5])
    """
    return gamma * pressure / ((gamma - 1.0) * rho_m)


def relativistic_enthalpy(
    pressure: NDArray[np.floating[Any]],
    rho_m: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
    c: float = 1.0,
) -> NDArray[np.floating[Any]]:
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
    return c**2 + enthalpy(pressure, rho_m, gamma)


def entropy(
    pressure: NDArray[np.floating[Any]],
    density: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
) -> NDArray[np.floating[Any]]:
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
    return np.log(pressure / density**gamma)


def gyrotropic_entropy(
    p_parallel: NDArray[np.floating[Any]],
    p_perpendicular: NDArray[np.floating[Any]],
    n: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the gyrotropic entropy from CGL double-adiabatic invariants.

    $$s_{gyro} = \ln\!\left(\frac{P_\parallel P_\perp^2}{n^5}\right)$$

    The exponent 5 arises from combining the two CGL invariants
    ($P_\perp / nB$ and $P_\parallel B^2 / n^3$) and is independent
    of the adiabatic index $\gamma$.

    Parameters
    ----------
    p_parallel : NDArray
        Pressure parallel to the magnetic field.
    p_perpendicular : NDArray
        Pressure perpendicular to the magnetic field.
    n : NDArray
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
    return np.log(p_parallel * p_perpendicular**2 / n**5)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def _unit_vector(
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    """Compute the unit vector of a 3-component vector field."""
    mag = _vector_magnitude(b1, b2, b3)
    return b1 / mag, b2 / mag, b3 / mag


def thermal_speed(
    temperature: NDArray[np.floating[Any]],
    mass: float,
) -> NDArray[np.floating[Any]]:
    r"""Compute the thermal speed (NRL convention).

    $$v_{th} = \sqrt{T / m}$$

    This is the 1D Maxwellian standard deviation $\sigma$ where
    $f(v_x) \propto \exp(-v_x^2 / (2\sigma^2))$ with $\sigma^2 = T/m$.

    Parameters
    ----------
    temperature : NDArray
        Temperature in energy units (normalized).
    mass : float
        Particle mass in normalized units.

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
    return np.sqrt(temperature / mass)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def gyrofrequency(
    b_magnitude: NDArray[np.floating[Any]],
    charge: float,
    mass: float,
) -> NDArray[np.floating[Any]]:
    r"""Compute the cyclotron (gyro) frequency.

    $$\omega_c = \frac{|q| B}{m}$$

    Positive by convention (magnitude of charge is used).

    Parameters
    ----------
    b_magnitude : NDArray
        Magnetic field magnitude in normalized units.
    charge : float
        Particle charge in normalized units (sign is stripped).
    mass : float
        Particle mass in normalized units.

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
    return np.abs(charge) * b_magnitude / mass  # type: ignore[no-any-return]  # numpy arithmetic returns NDArray


def plasma_frequency(
    density: NDArray[np.floating[Any]],
    charge: float,
    mass: float,
) -> NDArray[np.floating[Any]]:
    r"""Compute the plasma frequency.

    $$\omega_p = \sqrt{\frac{n q^2}{m}}$$

    In SI: $\omega_p = \sqrt{n e^2 / (\epsilon_0 m)}$.

    Parameters
    ----------
    density : NDArray
        Number density in normalized units.
    charge : float
        Particle charge in normalized units.
    mass : float
        Particle mass in normalized units.

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
    return np.sqrt(density * charge**2 / mass)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def skin_depth(
    density: NDArray[np.floating[Any]],
    charge: float,
    mass: float,
    c: float = 1.0,
) -> NDArray[np.floating[Any]]:
    r"""Compute the skin depth (inertial length).

    $$d = \frac{c}{\omega_p}$$

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
    return c / plasma_frequency(density, charge, mass)


def gyroradius(
    temperature: NDArray[np.floating[Any]],
    b_magnitude: NDArray[np.floating[Any]],
    charge: float,
    mass: float,
) -> NDArray[np.floating[Any]]:
    r"""Compute the thermal gyroradius (Larmor radius).

    $$r = \frac{v_{th}}{\omega_c} = \frac{\sqrt{m T}}{|q| B}$$

    Uses the NRL thermal speed convention $v_{th} = \sqrt{T/m}$.

    Parameters
    ----------
    temperature : NDArray
        Temperature in energy units (normalized).
    b_magnitude : NDArray
        Magnetic field magnitude in normalized units.
    charge : float
        Particle charge in normalized units (sign is stripped).
    mass : float
        Particle mass in normalized units.

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
    return np.sqrt(mass * temperature) / (np.abs(charge) * b_magnitude)  # type: ignore[no-any-return]  # numpy ufunc returns NDArray


def debye_length(
    temperature: NDArray[np.floating[Any]],
    density: NDArray[np.floating[Any]],
    charge: float,
) -> NDArray[np.floating[Any]]:
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
    return np.sqrt(temperature / (density * charge**2))


def sound_speed(
    pressure: NDArray[np.floating[Any]],
    rho_m: NDArray[np.floating[Any]],
    gamma: float = 5.0 / 3.0,
) -> NDArray[np.floating[Any]]:
    r"""Compute the MHD sound speed.

    $$c_s = \sqrt{\frac{\gamma P}{\rho_m}}$$

    Parameters
    ----------
    pressure : NDArray
        Scalar pressure in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    gamma : float
        Adiabatic index. Default is $5/3$.

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
    return np.sqrt(gamma * pressure / rho_m)


def ion_acoustic_speed(
    temperature_e: NDArray[np.floating[Any]],
    temperature_i: NDArray[np.floating[Any]],
    mass_i: float,
    gamma_e: float = 1.0,
    gamma_i: float = 3.0,
) -> NDArray[np.floating[Any]]:
    r"""Compute the ion acoustic speed.

    $$c_{ia} = \sqrt{\frac{\gamma_e T_e + \gamma_i T_i}{m_i}}$$

    Uses $\gamma_e = 1$ (isothermal electrons) and $\gamma_i = 3$ (1D
    adiabatic ions) by default, following kinetic theory convention.

    Parameters
    ----------
    temperature_e : NDArray
        Electron temperature in energy units (normalized).
    temperature_i : NDArray
        Ion temperature in energy units (normalized).
    mass_i : float
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
    >>> ion_acoustic_speed(np.array([1.0]), np.array([0.0]), mass_i=1.0)
    array([1.])
    """
    return np.sqrt((gamma_e * temperature_e + gamma_i * temperature_i) / mass_i)


def magnetosonic_speed(
    v_alfven: NDArray[np.floating[Any]],
    c_sound: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the fast magnetosonic speed (perpendicular propagation).

    $$v_{ms} = \sqrt{v_A^2 + c_s^2}$$

    This is the maximum fast-mode phase speed at $\theta = 90°$.

    Parameters
    ----------
    v_alfven : NDArray
        Alfvén speed in normalized units.
    c_sound : NDArray
        Sound speed in normalized units.

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
    return np.sqrt(v_alfven**2 + c_sound**2)


def alfven_mach(
    v_magnitude: NDArray[np.floating[Any]],
    v_alfven: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the Alfvén Mach number.

    $$M_A = \frac{V}{v_A}$$

    Parameters
    ----------
    v_magnitude : NDArray
        Bulk velocity magnitude in normalized units.
    v_alfven : NDArray
        Alfvén speed in normalized units.

    Returns
    -------
    NDArray
        Alfvén Mach number (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> alfven_mach(np.array([2.0]), np.array([1.0]))
    array([2.])
    """
    return v_magnitude / v_alfven


def magnetosonic_mach(
    v_magnitude: NDArray[np.floating[Any]],
    v_magnetosonic: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the magnetosonic Mach number.

    $$M_{ms} = \frac{V}{v_{ms}}$$

    Parameters
    ----------
    v_magnitude : NDArray
        Bulk velocity magnitude in normalized units.
    v_magnetosonic : NDArray
        Magnetosonic speed in normalized units.

    Returns
    -------
    NDArray
        Magnetosonic Mach number (dimensionless).

    Examples
    --------
    >>> import numpy as np
    >>> magnetosonic_mach(np.array([5.0]), np.array([5.0]))
    array([1.])
    """
    return v_magnitude / v_magnetosonic


def parallel_pressure(
    p11: NDArray[np.floating[Any]],
    p22: NDArray[np.floating[Any]],
    p33: NDArray[np.floating[Any]],
    p12: NDArray[np.floating[Any]],
    p13: NDArray[np.floating[Any]],
    p23: NDArray[np.floating[Any]],
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
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
    bh1, bh2, bh3 = _unit_vector(b1, b2, b3)
    result: NDArray[np.floating[Any]] = (
        bh1**2 * p11
        + bh2**2 * p22
        + bh3**2 * p33
        + 2.0 * (bh1 * bh2 * p12 + bh1 * bh3 * p13 + bh2 * bh3 * p23)
    )
    return result


def perpendicular_pressure(
    p11: NDArray[np.floating[Any]],
    p22: NDArray[np.floating[Any]],
    p33: NDArray[np.floating[Any]],
    p12: NDArray[np.floating[Any]],
    p13: NDArray[np.floating[Any]],
    p23: NDArray[np.floating[Any]],
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
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
    result: NDArray[np.floating[Any]] = (trace - p_par) / 2.0
    return result


def agyrotropy(
    p11: NDArray[np.floating[Any]],
    p22: NDArray[np.floating[Any]],
    p33: NDArray[np.floating[Any]],
    p12: NDArray[np.floating[Any]],
    p13: NDArray[np.floating[Any]],
    p23: NDArray[np.floating[Any]],
    b1: NDArray[np.floating[Any]],
    b2: NDArray[np.floating[Any]],
    b3: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the agyrotropy measure (Swisdak 2016).

    $$Q = 1 - \frac{4 I_2}{I_1^2}$$

    where $I_1 = \mathrm{Tr}(\mathbf{P}) - P_\parallel$ and
    $I_2 = (I_1^2 - \|\mathbf{P}_\perp\|_F^2) / 2$, with
    $\mathbf{P}_\perp = \mathbf{P} - P_\parallel \hat{b}\hat{b}$.

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
    bh1, bh2, bh3 = _unit_vector(b1, b2, b3)

    p_par = (
        bh1**2 * p11
        + bh2**2 * p22
        + bh3**2 * p33
        + 2.0 * (bh1 * bh2 * p12 + bh1 * bh3 * p13 + bh2 * bh3 * p23)
    )

    i_1 = p11 + p22 + p33 - p_par

    # Double-projected perpendicular tensor: Π = (I-b̂b̂)·P·(I-b̂b̂)
    # Π_ij = P_ij - (Pb̂)_i b̂_j - b̂_i (Pb̂)_j + P_∥ b̂_i b̂_j
    pb1 = p11 * bh1 + p12 * bh2 + p13 * bh3
    pb2 = p12 * bh1 + p22 * bh2 + p23 * bh3
    pb3 = p13 * bh1 + p23 * bh2 + p33 * bh3

    pi11 = p11 - 2.0 * pb1 * bh1 + p_par * bh1**2
    pi22 = p22 - 2.0 * pb2 * bh2 + p_par * bh2**2
    pi33 = p33 - 2.0 * pb3 * bh3 + p_par * bh3**2
    pi12 = p12 - pb1 * bh2 - bh1 * pb2 + p_par * bh1 * bh2
    pi13 = p13 - pb1 * bh3 - bh1 * pb3 + p_par * bh1 * bh3
    pi23 = p23 - pb2 * bh3 - bh2 * pb3 + p_par * bh2 * bh3

    n_f = pi11**2 + pi22**2 + pi33**2 + 2.0 * (pi12**2 + pi13**2 + pi23**2)

    i_2 = (i_1**2 - n_f) / 2.0

    result: NDArray[np.floating[Any]] = 1.0 - 4.0 * i_2 / i_1**2
    return result


__all__ = [
    "agyrotropy",
    "alfven_mach",
    "alfven_speed",
    "current_density_magnitude",
    "debye_length",
    "electric_energy_density",
    "electric_field_magnitude",
    "enthalpy",
    "entropy",
    "gyrofrequency",
    "gyroradius",
    "gyrotropic_entropy",
    "internal_energy_density",
    "ion_acoustic_speed",
    "kinetic_energy_density",
    "magnetic_energy_density",
    "magnetic_field_magnitude",
    "magnetosonic_mach",
    "magnetosonic_speed",
    "parallel_pressure",
    "perpendicular_pressure",
    "plasma_beta",
    "plasma_frequency",
    "poynting_flux",
    "relativistic_enthalpy",
    "skin_depth",
    "sound_speed",
    "thermal_energy_density",
    "thermal_speed",
    "velocity_magnitude",
]
