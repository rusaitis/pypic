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


__all__ = [
    "alfven_speed",
    "current_density_magnitude",
    "electric_energy_density",
    "electric_field_magnitude",
    "enthalpy",
    "entropy",
    "gyrotropic_entropy",
    "internal_energy_density",
    "kinetic_energy_density",
    "magnetic_energy_density",
    "magnetic_field_magnitude",
    "plasma_beta",
    "poynting_flux",
    "relativistic_enthalpy",
    "thermal_energy_density",
    "velocity_magnitude",
]
