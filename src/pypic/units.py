"""Normalization systems for PIC and MHD plasma simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import constants

type Numeric = float | np.floating[Any] | NDArray[np.floating[Any]]
type Vector3 = tuple[float, float, float]


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


@dataclass(frozen=True, slots=True)
class PhysicsConstants:
    r"""Simulation-frame physical constants in code units.

    Stores the speed of light, vacuum permittivity, and vacuum permeability
    as used inside the simulation. PIC codes typically normalize all three
    to unity; MHD codes set $c = \infty$ to eliminate displacement current.

    Parameters
    ----------
    c : float
        Speed of light in code units.
    epsilon_0 : float
        Vacuum permittivity in code units.
    mu_0 : float
        Vacuum permeability in code units.

    Examples
    --------
    >>> PhysicsConstants.pic_normalized().c
    1.0
    >>> PhysicsConstants.mhd_normalized().inv_c_squared()
    0.0
    """

    c: float
    epsilon_0: float
    mu_0: float

    @classmethod
    def pic_normalized(cls) -> PhysicsConstants:
        r"""Return PIC normalization where $c = \varepsilon_0 = \mu_0 = 1$.

        Returns
        -------
        PhysicsConstants
            PIC-normalized constants.

        Examples
        --------
        >>> pc = PhysicsConstants.pic_normalized()
        >>> pc.c, pc.epsilon_0, pc.mu_0
        (1.0, 1.0, 1.0)
        """
        return cls(c=1.0, epsilon_0=1.0, mu_0=1.0)

    @classmethod
    def mhd_normalized(cls) -> PhysicsConstants:
        r"""MHD normalization where $c = \infty$, eliminating displacement current.

        Returns
        -------
        PhysicsConstants
            MHD-normalized constants.

        Examples
        --------
        >>> pc = PhysicsConstants.mhd_normalized()
        >>> math.isinf(pc.c)
        True
        """
        return cls(c=float("inf"), epsilon_0=1.0, mu_0=1.0)

    def inv_c_squared(self) -> float:
        r"""Return $1/c^2$, guarded against infinite $c$.

        Returns ``0.0`` when $c = \infty$ (MHD limit) to avoid
        ``inf * 0 = nan`` in displacement-current terms.

        Returns
        -------
        float
            $1/c^2$, or ``0.0`` if $c$ is infinite.

        Examples
        --------
        >>> PhysicsConstants.pic_normalized().inv_c_squared()
        1.0
        >>> PhysicsConstants(c=10.0, epsilon_0=1.0, mu_0=1.0).inv_c_squared()
        0.01
        """
        if math.isinf(self.c):
            return 0.0
        return 1.0 / self.c**2


@dataclass(frozen=True, slots=True)
class SpeciesInfo:
    r"""Per-species metadata for a plasma simulation.

    At minimum, provide either ``charge`` + ``mass`` or ``charge_to_mass``.
    The missing quantities are inferred automatically:

    - From charge + mass: $q/m$ is computed directly.
    - From $q/m$ alone: convention is $|q| = 1$, $m = 1/|q/m|$.
    - If all three are given, consistency is validated.

    Parameters
    ----------
    name : str
        Species label (e.g. ``"e"``, ``"ion"``).
    charge : float | None
        Charge in code units.
    mass : float | None
        Mass in code units.
    charge_to_mass : float | None
        Charge-to-mass ratio in code units.
    temperature : float | None
        Temperature in code units.
    thermal_velocity : float | Vector3 | None
        Scalar (isotropic) or per-component thermal velocity.
    drift_velocity : Vector3 | None
        Bulk drift velocity $(v_x, v_y, v_z)$.
    density : float | None
        Number density in code units.
    particles_per_cell : int | tuple[int, int, int] | None
        Particles per cell, uniform or per-direction.

    Examples
    --------
    >>> e = SpeciesInfo(name="e", charge=-1.0, mass=1/256)
    >>> e.charge_to_mass
    -256.0
    >>> ion = SpeciesInfo(name="ion", charge_to_mass=1.0)
    >>> ion.charge, ion.mass
    (1.0, 1.0)
    """

    name: str
    charge: float | None = None
    mass: float | None = None
    charge_to_mass: float | None = None
    temperature: float | None = None
    thermal_velocity: float | Vector3 | None = None
    drift_velocity: Vector3 | None = None
    density: float | None = None
    particles_per_cell: int | tuple[int, int, int] | None = None

    def __post_init__(self) -> None:  # noqa: D105 — inference logic, not a public API
        has_charge = self.charge is not None
        has_mass = self.mass is not None
        has_qom = self.charge_to_mass is not None

        if has_charge and has_mass and not has_qom:
            # Infer q/m from charge and mass
            object.__setattr__(
                self,
                "charge_to_mass",
                self.charge / self.mass,  # type: ignore[operator]  # narrowed above
            )
        elif has_qom and not has_charge and not has_mass:
            assert self.charge_to_mass is not None  # narrowed by has_qom
            qom = self.charge_to_mass
            if qom == 0.0:
                msg = (
                    "Cannot decompose charge_to_mass=0 into charge and mass. "
                    "Specify charge=0 and mass explicitly."
                )
                raise ValueError(msg)
            object.__setattr__(self, "charge", math.copysign(1.0, qom))
            object.__setattr__(self, "mass", 1.0 / abs(qom))
        elif has_charge and has_mass and has_qom:
            expected = self.charge / self.mass  # type: ignore[operator]
            if not math.isclose(expected, self.charge_to_mass, rel_tol=1e-12):  # type: ignore[arg-type]
                msg = (
                    f"Inconsistent species parameters: "
                    f"charge/mass={expected} != charge_to_mass={self.charge_to_mass}"
                )
                raise ValueError(msg)
        elif not has_charge and not has_mass and not has_qom:
            msg = "Must provide charge+mass or charge_to_mass (or all three)."
            raise ValueError(msg)
        else:
            # Only one of charge/mass provided
            msg = "Must provide both charge and mass, not just one."
            raise ValueError(msg)
