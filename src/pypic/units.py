"""Normalization systems for PIC and MHD plasma simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import constants

type Numeric = float | np.floating[Any] | NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class Normalization:
    r"""Map between simulation (code) units and SI.

    Every simulation uses a set of reference quantities to non-dimensionalize
    the equations. This class stores those reference values (all in SI) and
    provides ``normalize_*`` / ``to_si_*`` method pairs for each physical
    quantity.

    Use the classmethods to construct standard normalizations:

    - `pic_electron` / `pic_ion` / `pic_standard` for PIC codes
    - `mhd_standard` for MHD codes
    - `identity` for data already in SI or dimensionless tests

    Parameters
    ----------
    length_ref : float
        Reference length in meters ($d_e$ for PIC, $l_0$ for MHD).
    time_ref : float
        Reference time in seconds ($1/\omega_{ref}$ for PIC, $l_0/v_A$ for MHD).
    velocity_ref : float
        Reference velocity in m/s ($c$ for PIC, $v_A$ for MHD).
    b_field_ref : float
        Reference magnetic field in Tesla.
    e_field_ref : float
        Reference electric field in V/m.
    density_ref : float
        Reference number density in m$^{-3}$.
    mass_ref : float
        Reference particle mass in kg.
    charge_ref : float
        Reference charge in Coulombs.

    Examples
    --------
    >>> norm = Normalization.identity()
    >>> norm.normalize_length(3.0)
    3.0
    """

    length_ref: float
    time_ref: float
    velocity_ref: float
    b_field_ref: float
    e_field_ref: float
    density_ref: float
    mass_ref: float
    charge_ref: float

    @classmethod
    def pic_standard(
        cls,
        reference_density: float,
        reference_mass: float,
        reference_charge: float,
        c: float = constants.c,
    ) -> Normalization:
        r"""Construct PIC normalization for an arbitrary reference species.

        Derives all reference quantities from the plasma frequency of the
        reference species:

        $$\omega_{ref} = \sqrt{\frac{n_{ref} \, q_{ref}^2}{\varepsilon_0 \, m_{ref}}}$$

        Parameters
        ----------
        reference_density : float
            Number density of the reference species in m$^{-3}$.
        reference_mass : float
            Mass of the reference species in kg.
        reference_charge : float
            Charge of the reference species in Coulombs.
        c : float, optional
            Speed of light in m/s. Defaults to ``scipy.constants.c``.

        Returns
        -------
        Normalization
            PIC normalization instance with all fields in SI.

        Examples
        --------
        >>> from scipy.constants import m_e, e, c
        >>> norm = Normalization.pic_standard(1e18, m_e, e, c)
        >>> bool(np.isclose(norm.normalize_velocity(c), 1.0, rtol=1e-12))
        True
        """
        omega_ref = np.sqrt(
            reference_density
            * reference_charge**2
            / (constants.epsilon_0 * reference_mass)
        )
        length_ref = c / omega_ref
        time_ref = 1.0 / omega_ref
        b_field_ref = reference_mass * omega_ref / reference_charge
        e_field_ref = c * b_field_ref

        return cls(
            length_ref=float(length_ref),
            time_ref=float(time_ref),
            velocity_ref=float(c),
            b_field_ref=float(b_field_ref),
            e_field_ref=float(e_field_ref),
            density_ref=float(reference_density),
            mass_ref=float(reference_mass),
            charge_ref=float(reference_charge),
        )

    @classmethod
    def pic_electron(cls, n_e: float) -> Normalization:
        r"""PIC normalization using electron parameters.

        Convenience wrapper around `pic_standard` with $m_e$, $e$, $c$ from
        ``scipy.constants``.

        Parameters
        ----------
        n_e : float
            Electron number density in m$^{-3}$.

        Returns
        -------
        Normalization
            Electron-scale PIC normalization.

        Examples
        --------
        >>> norm = Normalization.pic_electron(1e18)
        >>> bool(np.isclose(norm.length_ref, constants.c / np.sqrt(
        ...     1e18 * constants.e**2 / (constants.epsilon_0 * constants.m_e)
        ... ), rtol=1e-12))
        True
        """
        return cls.pic_standard(n_e, constants.m_e, constants.e)

    @classmethod
    def pic_ion(cls, n_i: float, mass_i: float, charge_i: float) -> Normalization:
        r"""PIC normalization using ion parameters.

        Convenience wrapper around `pic_standard` with the given ion mass and
        charge, using ``scipy.constants.c`` for the speed of light.

        Parameters
        ----------
        n_i : float
            Ion number density in m$^{-3}$.
        mass_i : float
            Ion mass in kg.
        charge_i : float
            Ion charge in Coulombs.

        Returns
        -------
        Normalization
            Ion-scale PIC normalization.

        Examples
        --------
        >>> from scipy.constants import m_p, e
        >>> norm = Normalization.pic_ion(1e18, m_p, e)
        >>> norm.density_ref
        1e+18
        """
        return cls.pic_standard(n_i, mass_i, charge_i)

    @classmethod
    def mhd_standard(cls, l_0: float, rho_0: float, b_0: float) -> Normalization:
        r"""MHD normalization from macroscopic reference quantities.

        Derives the Alfvén speed:

        $$v_A = \frac{B_0}{\sqrt{\mu_0 \, \rho_0}}$$

        Parameters
        ----------
        l_0 : float
            Reference length in meters.
        rho_0 : float
            Reference mass density in kg/m$^3$.
        b_0 : float
            Reference magnetic field in Tesla.

        Returns
        -------
        Normalization
            MHD normalization instance.

        Examples
        --------
        >>> norm = Normalization.mhd_standard(1e6, 1e-12, 1e-9)
        >>> norm.length_ref
        1000000.0
        """
        v_a = b_0 / np.sqrt(constants.mu_0 * rho_0)

        return cls(
            length_ref=float(l_0),
            time_ref=float(l_0 / v_a),
            velocity_ref=float(v_a),
            b_field_ref=float(b_0),
            e_field_ref=float(v_a * b_0),
            density_ref=float(rho_0 / constants.m_p),
            mass_ref=float(constants.m_p),
            charge_ref=float(constants.e),
        )

    @classmethod
    def identity(cls) -> Normalization:
        r"""Return normalization where all reference values are unity.

        Useful for data already in SI or for dimensionless tests.

        Returns
        -------
        Normalization
            Identity normalization (all refs = 1.0).

        Examples
        --------
        >>> norm = Normalization.identity()
        >>> norm.length_ref
        1.0
        """
        return cls(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )

    # -- Conversion methods --------------------------------------------------

    def normalize_length(self, x: Numeric) -> Numeric:
        r"""Convert length from SI to code units.

        $$\hat{x} = x / x_{ref}$$

        Parameters
        ----------
        x : Numeric
            Length in meters.

        Returns
        -------
        Numeric
            Length in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_length(5.0)
        5.0
        """
        return x / self.length_ref

    def to_si_length(self, x: Numeric) -> Numeric:
        r"""Convert length from code units to SI.

        $$x = \hat{x} \cdot x_{ref}$$

        Parameters
        ----------
        x : Numeric
            Length in code units.

        Returns
        -------
        Numeric
            Length in meters.

        Examples
        --------
        >>> Normalization.identity().to_si_length(5.0)
        5.0
        """
        return x * self.length_ref

    def normalize_time(self, x: Numeric) -> Numeric:
        r"""Convert time from SI to code units.

        $$\hat{t} = t / t_{ref}$$

        Parameters
        ----------
        x : Numeric
            Time in seconds.

        Returns
        -------
        Numeric
            Time in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_time(2.0)
        2.0
        """
        return x / self.time_ref

    def to_si_time(self, x: Numeric) -> Numeric:
        r"""Convert time from code units to SI.

        $$t = \hat{t} \cdot t_{ref}$$

        Parameters
        ----------
        x : Numeric
            Time in code units.

        Returns
        -------
        Numeric
            Time in seconds.

        Examples
        --------
        >>> Normalization.identity().to_si_time(2.0)
        2.0
        """
        return x * self.time_ref

    def normalize_velocity(self, x: Numeric) -> Numeric:
        r"""Convert velocity from SI to code units.

        $$\hat{v} = v / v_{ref}$$

        Parameters
        ----------
        x : Numeric
            Velocity in m/s.

        Returns
        -------
        Numeric
            Velocity in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_velocity(10.0)
        10.0
        """
        return x / self.velocity_ref

    def to_si_velocity(self, x: Numeric) -> Numeric:
        r"""Convert velocity from code units to SI.

        $$v = \hat{v} \cdot v_{ref}$$

        Parameters
        ----------
        x : Numeric
            Velocity in code units.

        Returns
        -------
        Numeric
            Velocity in m/s.

        Examples
        --------
        >>> Normalization.identity().to_si_velocity(10.0)
        10.0
        """
        return x * self.velocity_ref

    def normalize_b_field(self, x: Numeric) -> Numeric:
        r"""Convert magnetic field from SI to code units.

        $$\hat{B} = B / B_{ref}$$

        Parameters
        ----------
        x : Numeric
            Magnetic field in Tesla.

        Returns
        -------
        Numeric
            Magnetic field in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_b_field(0.5)
        0.5
        """
        return x / self.b_field_ref

    def to_si_b_field(self, x: Numeric) -> Numeric:
        r"""Convert magnetic field from code units to SI.

        $$B = \hat{B} \cdot B_{ref}$$

        Parameters
        ----------
        x : Numeric
            Magnetic field in code units.

        Returns
        -------
        Numeric
            Magnetic field in Tesla.

        Examples
        --------
        >>> Normalization.identity().to_si_b_field(0.5)
        0.5
        """
        return x * self.b_field_ref

    def normalize_e_field(self, x: Numeric) -> Numeric:
        r"""Convert electric field from SI to code units.

        $$\hat{E} = E / E_{ref}$$

        Parameters
        ----------
        x : Numeric
            Electric field in V/m.

        Returns
        -------
        Numeric
            Electric field in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_e_field(100.0)
        100.0
        """
        return x / self.e_field_ref

    def to_si_e_field(self, x: Numeric) -> Numeric:
        r"""Convert electric field from code units to SI.

        $$E = \hat{E} \cdot E_{ref}$$

        Parameters
        ----------
        x : Numeric
            Electric field in code units.

        Returns
        -------
        Numeric
            Electric field in V/m.

        Examples
        --------
        >>> Normalization.identity().to_si_e_field(100.0)
        100.0
        """
        return x * self.e_field_ref

    def normalize_density(self, x: Numeric) -> Numeric:
        r"""Convert number density from SI to code units.

        $$\hat{n} = n / n_{ref}$$

        Parameters
        ----------
        x : Numeric
            Number density in m$^{-3}$.

        Returns
        -------
        Numeric
            Number density in code units.

        Examples
        --------
        >>> Normalization.identity().normalize_density(1e18)
        1e+18
        """
        return x / self.density_ref

    def to_si_density(self, x: Numeric) -> Numeric:
        r"""Convert number density from code units to SI.

        $$n = \hat{n} \cdot n_{ref}$$

        Parameters
        ----------
        x : Numeric
            Number density in code units.

        Returns
        -------
        Numeric
            Number density in m$^{-3}$.

        Examples
        --------
        >>> Normalization.identity().to_si_density(1e18)
        1e+18
        """
        return x * self.density_ref
