"""Normalization systems for PIC and MHD plasma simulations."""

from __future__ import annotations

__all__ = [
    "Normalization",
    "PhysicsConstants",
    "PhysicsParams",
    "SpeciesInfo",
    "UnitSystem",
]

import math
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from scipy import constants

from pypic.exceptions import UndeclaredNormalizationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypic.types import Numeric, Vector3

# The eight storage primitives, in the order `_SIFactor.exponents` indexes.
_REFERENCE_ORDER = (
    "length",
    "time",
    "velocity",
    "b_field",
    "e_field",
    "density",
    "mass",
    "charge",
)
_QUANTITIES = frozenset(_REFERENCE_ORDER)

# Correct under any anchor, so exempt from the undeclared-normalization guard.
_DIMENSIONLESS = "dimensionless"

# Display unit conversion: unit string → SI value.
# Hand-curated rather than parsed from SI prefixes: the table is small,
# closed, and a wrong factor here is a silent physics error.
_DISPLAY_UNITS: dict[str, float] = {
    "T": 1.0,
    "nT": 1e-9,
    "mT": 1e-3,
    "V/m": 1.0,
    "mV/m": 1e-3,
    "m/s": 1.0,
    "km/s": 1e3,
    "m^-3": 1.0,
    "cm^-3": 1e6,
    "/cc": 1e6,  # informal alias for cm^-3
    "Mp/cc": 1e6 * constants.m_p,  # proton masses per cc (BATSRUS)
    "amu/cc": 1e6 * constants.m_u,  # atomic mass units per cc (BATSRUS)
    "kg/m^3": 1.0,
    "Pa": 1.0,
    "nPa": 1e-9,
    "J/m^3": 1.0,
    "W/m^2": 1.0,
    "mW/m^2": 1e-3,
    "W/m^3": 1.0,
    "Hz": 1.0,
    "rad/s": 1.0,
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "m": 1.0,
    "km": 1e3,
    "R_E": 6.371e6,  # NASA NSSDCA Planetary Fact Sheet
    "eV": constants.eV,
    "keV": 1e3 * constants.eV,
    "K": constants.k,
    "A/m^2": 1.0,
    "uA/m^2": 1e-6,
    "nA/m^2": 1e-9,
    "uA/m2": 1e-6,  # BATSRUS header alias (no caret)
    "C/m^3": 1.0,
    "J/kg": 1.0,
    "MeV": 1e6 * constants.eV,
    "G": 1e-4,
    "mG": 1e-7,
    "pPa": 1e-12,
    "cm/s": 1e-2,
    "Mm": 1e6,
    "RE": 6.371e6,  # alias for R_E
    "R_S": 6.957e8,  # IAU 2015 nominal solar radius
    "R_Moon": 1.7374e6,  # NASA NSSDCA Moon Fact Sheet
    "R_Mercury": 2.4397e6,  # NASA NSSDCA Planetary Fact Sheet
    "R_Mars": 3.3895e6,  # NASA NSSDCA Planetary Fact Sheet
    "R_J": 6.9911e7,  # NASA NSSDCA Planetary Fact Sheet (volumetric mean)
    "R_Saturn": 5.8232e7,  # NASA NSSDCA Planetary Fact Sheet (volumetric mean)
    "AU": constants.au,
    "normalized": 1.0,
}


class UnitSystem(StrEnum):
    """How a `Normalization`'s eight references were anchored.

    The vocabulary of ``[units].anchor`` in ``simulation.toml``
    (Schema § 2).  It names a *derivation*, not a code type — which
    code produced the data is ``[model].type``, a separate field.
    `Normalization.system` is ``None`` when no ``[units]`` section
    declared an anchor at all; see `Normalization.undeclared`.

    ``FROM_SPECIES`` derives the length unit from a reference species'
    plasma frequency; ``EXPLICIT`` takes it as given; ``SI`` is the
    identity anchor of data already in SI.

    Examples
    --------
    >>> UnitSystem.FROM_SPECIES == "from_species"
    True
    """

    FROM_SPECIES = "from_species"
    EXPLICIT = "explicit"
    SI = "si"


@dataclass(frozen=True, slots=True)
class Normalization:
    r"""Map between simulation (code) units and SI.

    Every simulation uses a set of reference quantities to non-dimensionalize
    the equations. This class stores those reference values (all in SI) and
    provides ``normalize`` / ``to_si`` methods for each physical quantity.

    Use the classmethods to construct standard normalizations:

    - `pic_electron` / `pic_standard` for PIC codes
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
    system : UnitSystem or None
        Which normalization system these references came from, or
        ``None`` when nothing declared one. Defaults to
        ``UnitSystem.EXPLICIT``, the honest reading of eight
        hand-supplied references. ``None`` makes every dimensional
        `si_factor` raise rather than silently return 1.0 — see
        `undeclared`.

    Examples
    --------
    >>> norm = Normalization.identity()
    >>> norm.normalize("length", 3.0)
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
    system: UnitSystem | None = UnitSystem.EXPLICIT

    def __post_init__(self) -> None:
        for attr in (
            "length_ref",
            "time_ref",
            "velocity_ref",
            "b_field_ref",
            "e_field_ref",
            "density_ref",
            "mass_ref",
            "charge_ref",
        ):
            val = getattr(self, attr)
            if val <= 0:
                msg = f"{attr} must be positive, got {val}"
                raise ValueError(msg)

    @classmethod
    def pic_standard(
        cls,
        reference_density: float,
        reference_mass: float,
        reference_charge: float,
        c: float = constants.c,
        reference_velocity: float | None = None,
    ) -> Normalization:
        r"""Construct PIC normalization for an arbitrary reference species.

        Derives all reference quantities from the plasma frequency of the
        reference species:

        $$\omega_{ref} = \sqrt{\frac{n_{ref} \, q_{ref}^2}{\varepsilon_0 \, m_{ref}}}$$

        The skin depth $l_{ref} = c / \omega_{ref}$ anchors length, and
        the remaining references follow from the velocity unit:
        $t_{ref} = l_{ref} / v_{ref}$,
        $B_{ref} = v_{ref}\sqrt{\mu_0 n_{ref} m_{ref}}$, and
        $E_{ref} = v_{ref} B_{ref}$.  At the default $v_{ref} = c$ these
        reduce to $1/\omega_{ref}$ and $m_{ref}\omega_{ref}/q_{ref}$.

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
        reference_velocity : float or None, optional
            Velocity unit in m/s. ``None`` (the default) uses *c*, the
            PIC convention.  Hybrid codes normalize to the Alfvén speed
            instead; passing it makes $B_{ref}$ the field at which $v_A$
            equals that speed, which lands $t_{ref}$ on the inverse ion
            cyclotron frequency — the hybrid time unit.

        Returns
        -------
        Normalization
            PIC normalization instance with all fields in SI.

        Examples
        --------
        >>> from scipy.constants import m_e, e, c
        >>> norm = Normalization.pic_standard(1e18, m_e, e, c)
        >>> bool(np.isclose(norm.normalize("velocity", c), 1.0, rtol=1e-12))
        True
        """
        omega_ref = np.sqrt(
            reference_density
            * reference_charge**2
            / (constants.epsilon_0 * reference_mass)
        )
        length_ref = c / omega_ref
        if reference_velocity is None:
            velocity_ref = c
            # Algebraically equal to the general forms below at v = c;
            # kept verbatim so the default path stays bit-identical.
            time_ref = 1.0 / omega_ref
            b_field_ref = reference_mass * omega_ref / reference_charge
        else:
            velocity_ref = reference_velocity
            time_ref = length_ref / velocity_ref
            b_field_ref = velocity_ref * np.sqrt(
                constants.mu_0 * reference_density * reference_mass
            )
        # From E = -v x B.
        e_field_ref = velocity_ref * b_field_ref

        return cls(
            length_ref=float(length_ref),
            time_ref=float(time_ref),
            velocity_ref=float(velocity_ref),
            b_field_ref=float(b_field_ref),
            e_field_ref=float(e_field_ref),
            density_ref=float(reference_density),
            mass_ref=float(reference_mass),
            charge_ref=float(reference_charge),
            system=UnitSystem.FROM_SPECIES,
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
    def mhd_standard(
        cls,
        reference_length: float,
        reference_density: float,
        reference_b_field: float,
    ) -> Normalization:
        r"""MHD normalization from macroscopic reference quantities.

        Derives the Alfvén speed:

        $$v_A = \frac{B_0}{\sqrt{\mu_0 \, \rho_0}}$$

        Parameters
        ----------
        reference_length : float
            Reference length $l_0$ in meters.
        reference_density : float
            Reference **mass** density $\rho_0$ in kg/m$^3$. The TOML
            spelling is ``reference_mass_density``, which names its unit;
            this parameter keeps the older name for callers that already
            pass it positionally.
        reference_b_field : float
            Reference magnetic field $B_0$ in Tesla.

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
        v_a = reference_b_field / np.sqrt(constants.mu_0 * reference_density)

        return cls(
            length_ref=float(reference_length),
            time_ref=float(reference_length / v_a),
            velocity_ref=float(v_a),
            b_field_ref=float(reference_b_field),
            e_field_ref=float(v_a * reference_b_field),
            # Convert mass density → number density (reference species: proton)
            density_ref=float(reference_density / constants.m_p),
            mass_ref=float(constants.m_p),
            charge_ref=float(constants.e),
            system=UnitSystem.EXPLICIT,
        )

    @classmethod
    def identity(cls) -> Normalization:
        r"""Return normalization where all reference values are unity.

        Declares the data to be **already SI** — the runtime form of
        ``[units] system = "SI"``. Use `undeclared` instead when no
        unit system is known; the two carry the same eight references
        and differ only in `system`.

        Returns
        -------
        Normalization
            Identity normalization (all refs = 1.0, ``system = SI``).

        Examples
        --------
        >>> norm = Normalization.identity()
        >>> norm.length_ref
        1.0
        >>> norm.si_factor("b_field")
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
            system=UnitSystem.SI,
        )

    @classmethod
    def undeclared(cls) -> Normalization:
        r"""Return the normalization of data whose unit system is unknown.

        What a reader falls back to when no ``simulation.toml``
        accompanies the output. The eight references are unity so the
        arrays are left alone, but `system` is ``None``, so asking for
        a dimensional SI conversion raises
        `UndeclaredNormalizationError` instead of returning code units
        labelled tesla. Dimensionless quantities still convert.

        The absent reference is not recoverable: a PIC deck fixes only
        dimensionless ratios ($\omega_{pe}/\omega_{ce}$, $m_i/m_e$,
        $c/v_A$), so the SI anchor is the modeller's interpretation
        and belongs in ``[units]``.

        Returns
        -------
        Normalization
            All refs = 1.0, ``system = None``.

        Examples
        --------
        >>> norm = Normalization.undeclared()
        >>> norm.si_factor("dimensionless")  # correct under any anchor
        1.0
        >>> norm.si_factor("b_field")
        Traceback (most recent call last):
        pypic.exceptions.UndeclaredNormalizationError: ...
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
            system=None,
        )

    @property
    def is_identity(self) -> bool:
        """True if all reference values are 1.0 (no unit conversion).

        Examples
        --------
        >>> Normalization.identity().is_identity
        True
        """
        return all(
            getattr(self, f"{q}_ref") == 1.0
            for q in (
                "length",
                "time",
                "velocity",
                "b_field",
                "e_field",
                "density",
                "mass",
                "charge",
            )
        )

    @property
    def rationalization_ratio(self) -> float:
        r"""How far these references sit from SI-rationalized code units.

        $$\frac{B_{ref}^2}{\mu_0 \, n_{ref} \, m_{ref} \, v_{ref}^2}$$

        `pypic.derived` computes in SI-rationalized units throughout —
        $e_B = B^2/2$, $\nabla\cdot\mathbf{E} = \rho_c$ — which holds
        exactly when this ratio is 1.  Every EM quantity is wrong by a
        power of it when it is not, so its **value names the
        convention** rather than grading it:

        ============================  ===================
        Ratio                          Convention
        ============================  ===================
        1                              SI-rationalized
        $1/\mu_0$                      data already in SI
        $(c_{SI}/c_{ref})^2$           a reduced speed of light
        $2/\beta$                      gyrokinetic (gyro-Bohm)
        ============================  ===================

        A diagnostic, not a gate: the last two are deliberate physics.

        Examples
        --------
        >>> from scipy import constants
        >>> n = Normalization.pic_electron(1e18)
        >>> bool(np.isclose(n.rationalization_ratio, 1.0, rtol=1e-9))
        True
        >>> bool(np.isclose(
        ...     Normalization.identity().rationalization_ratio,
        ...     1.0 / constants.mu_0,
        ... ))
        True
        """
        return self.b_field_ref**2 / (
            constants.mu_0 * self.density_ref * self.mass_ref * self.velocity_ref**2
        )

    def summary(self) -> str:
        r"""One-line description of the unit system and its SI anchor.

        Shared by ``Simulation.describe()`` and ``pypic info`` so the
        two report a dataset's units identically. The undeclared case
        is spelled out rather than shown as unit references, since
        those read as SI when nothing established the anchor.

        The ``si`` anchor gets the rationalization annotation too, and
        needs it most: all eight references are 1.0, so its ratio is
        $1/\mu_0$ and every quantity carrying a vacuum constant is wrong
        by a power of it. Naming ``data_in_si`` there points at the fix
        rather than only at the symptom.

        Examples
        --------
        >>> Normalization.undeclared().summary()
        'undeclared (code units; no [units] section)'
        >>> Normalization.identity().summary()[:14]
        'SI (identity) '
        """
        if self.system is None:
            return "undeclared (code units; no [units] section)"
        if self.system is UnitSystem.SI and self.is_identity:
            return (
                "SI (identity) [not SI-rationalized: "
                f"B^2/(mu_0 n m v^2) = {self.rationalization_ratio:.4g}; "
                "derived quantities assume mu_0 = 1, so declare a real anchor "
                "with [units].data_in_si to compute on this data]"
            )
        line = (
            f"{self.system}: l={self.length_ref:.4g} m, "
            f"v={self.velocity_ref:.4g} m/s, "
            f"B={self.b_field_ref:.4g} T, n={self.density_ref:.4g} m^-3"
        )
        # The one property of a reference set that its numbers do not show.
        ratio = self.rationalization_ratio
        if not math.isclose(ratio, 1.0, rel_tol=1e-2):
            line += f" [not SI-rationalized: B^2/(mu_0 n m v^2) = {ratio:.4g}]"
        return line

    def _reference_value(self, quantity: str) -> float:
        """Look up the reference value for *quantity*, or raise ValueError."""
        if quantity not in _QUANTITIES:
            msg = f"Unknown quantity {quantity!r}. Valid: {sorted(_QUANTITIES)}"
            raise ValueError(msg)
        ref: float = getattr(self, f"{quantity}_ref")
        return ref

    def si_factor(self, quantity: str) -> float:
        r"""Return the SI conversion factor for a physical quantity.

        Handles both base quantities (``"length"``, ``"b_field"``, etc.) and
        compound quantities (``"pressure"``, ``"frequency"``, etc.) that are
        products of base reference values.

        Parameters
        ----------
        quantity : str
            Physical quantity name — base or compound.

        Returns
        -------
        float
            Multiplicative factor: ``value_si = value_code * si_factor``.

        Raises
        ------
        UndeclaredNormalizationError
            When `system` is ``None`` and *quantity* is dimensional.
            This is the single chokepoint under `FieldDataset.in_si`,
            `FieldDataset.in_units` and
            [`field_si_factor`][pypic.field_si_factor].
        ValueError
            When *quantity* names neither a base nor a compound
            quantity.

        Examples
        --------
        >>> Normalization.identity().si_factor("velocity")
        1.0
        >>> Normalization.identity().si_factor("dimensionless")
        1.0
        """
        if self.system is None and quantity != _DIMENSIONLESS:
            msg = (
                f"Cannot convert {quantity!r} to SI: no unit system was declared "
                "for this data, so the SI anchor is unknown and code units would "
                "be returned labelled as SI. Ship a simulation.toml with a "
                "[units] section, pass normalization= when opening the data, or "
                'ask for code units (units="code").'
            )
            raise UndeclaredNormalizationError(msg)
        try:
            spec = _SI_FACTORS[quantity]
        except KeyError:
            msg = f"Unknown quantity {quantity!r}. Valid: {sorted(_SI_FACTORS)}"
            raise ValueError(msg) from None
        factor = math.prod(
            getattr(self, f"{name}_ref") ** power
            for name, power in zip(_REFERENCE_ORDER, spec.exponents, strict=True)
            if power
        )
        if spec.mu_0_power:
            factor *= constants.mu_0**spec.mu_0_power
        return float(factor)

    def normalize(self, quantity: str, x: Numeric) -> Numeric:
        r"""Convert a physical quantity from SI to code units.

        $$\hat{x} = x / x_{ref}$$

        Parameters
        ----------
        quantity : str
            One of the eight storage primitives: ``"length"``, ``"time"``,
            ``"velocity"``, ``"b_field"``, ``"e_field"``, ``"density"``,
            ``"mass"``, ``"charge"``.
        x : Numeric
            Value in SI units.

        Returns
        -------
        Numeric
            Value in code units.

        Examples
        --------
        >>> Normalization.identity().normalize("length", 5.0)
        5.0
        """
        return x / self._reference_value(quantity)

    def to_si(self, quantity: str, x: Numeric) -> Numeric:
        r"""Convert a physical quantity from code units to SI.

        $$x = \hat{x} \cdot x_{ref}$$

        Parameters
        ----------
        quantity : str
            One of the eight storage primitives: ``"length"``, ``"time"``,
            ``"velocity"``, ``"b_field"``, ``"e_field"``, ``"density"``,
            ``"mass"``, ``"charge"``.
        x : Numeric
            Value in code units.

        Returns
        -------
        Numeric
            Value in SI units.

        Examples
        --------
        >>> Normalization.identity().to_si("length", 5.0)
        5.0
        """
        return x * self._reference_value(quantity)


class _SIFactor(NamedTuple):
    r"""How one quantity type's SI factor is built from the references.

    *exponents* are integer powers of the eight primitives in
    `_REFERENCE_ORDER`; every factor pypic needs is a monomial in them.
    *mu_0_power* is the power of $\mu_0$ that pypic's SI-rationalized
    code units leave out — non-zero only for `poynting_flux`, where
    `derived.poynting_flux` returns a bare $\mathbf{E}\times\mathbf{B}$.

    Recording $\mu_0$ separately rather than folding it in as a plain
    number is what lets `tests/test_units.py` compose these against
    each reference's own SI dimension and check the result against
    `pypic.fields._QUANTITY_DIMENSIONS` — two tables that encode the
    same physics and previously disagreed, unnoticed, on this entry.
    """

    exponents: tuple[int, int, int, int, int, int, int, int]
    mu_0_power: int = 0


#                      l   t   v   B   E   n   m   q
_SI_FACTORS: dict[str, _SIFactor] = {
    "dimensionless": _SIFactor((0, 0, 0, 0, 0, 0, 0, 0)),
    "length": _SIFactor((1, 0, 0, 0, 0, 0, 0, 0)),
    "time": _SIFactor((0, 1, 0, 0, 0, 0, 0, 0)),
    "velocity": _SIFactor((0, 0, 1, 0, 0, 0, 0, 0)),
    "b_field": _SIFactor((0, 0, 0, 1, 0, 0, 0, 0)),
    "e_field": _SIFactor((0, 0, 0, 0, 1, 0, 0, 0)),
    "density": _SIFactor((0, 0, 0, 0, 0, 1, 0, 0)),
    "mass": _SIFactor((0, 0, 0, 0, 0, 0, 1, 0)),
    "charge": _SIFactor((0, 0, 0, 0, 0, 0, 0, 1)),
    "four_velocity": _SIFactor((0, 0, 1, 0, 0, 0, 0, 0)),
    "pressure": _SIFactor((0, 0, 2, 0, 0, 1, 1, 0)),
    "temperature": _SIFactor((0, 0, 2, 0, 0, 0, 1, 0)),
    "energy_density": _SIFactor((0, 0, 2, 0, 0, 1, 1, 0)),
    "specific_energy": _SIFactor((0, 0, 2, 0, 0, 0, 0, 0)),
    "energy_flux": _SIFactor((0, 0, 3, 0, 0, 1, 1, 0)),
    "power_density": _SIFactor((0, -1, 2, 0, 0, 1, 1, 0)),
    "current_density": _SIFactor((0, 0, 1, 0, 0, 1, 0, 1)),
    "mass_density": _SIFactor((0, 0, 0, 0, 0, 1, 1, 0)),
    "charge_density": _SIFactor((0, 0, 0, 0, 0, 1, 0, 1)),
    "frequency": _SIFactor((0, -1, 0, 0, 0, 0, 0, 0)),
    "b_field_per_length": _SIFactor((-1, 0, 0, 1, 0, 0, 0, 0)),
    "e_field_per_length": _SIFactor((-1, 0, 0, 0, 1, 0, 0, 0)),
    "velocity_per_length": _SIFactor((-1, 0, 1, 0, 0, 0, 0, 0)),
    "poynting_flux": _SIFactor((0, 0, 0, 1, 1, 0, 0, 0), mu_0_power=-1),
}


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
        return cls(c=math.inf, epsilon_0=1.0, mu_0=1.0)

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

    def __post_init__(self) -> None:
        if self.mass is not None and self.mass < 0:
            msg = f"mass must not be negative, got {self.mass}"
            raise ValueError(msg)

        if (
            self.charge is not None
            and self.mass is not None
            and self.charge_to_mass is None
        ):
            # A massless fluid species has no finite charge-to-mass ratio.
            # Left unset rather than infinite: consumers already handle the
            # absent case, and an inf would propagate silently into moments.
            if self.mass != 0.0:
                object.__setattr__(self, "charge_to_mass", self.charge / self.mass)
        elif (
            self.charge_to_mass is not None
            and self.charge is None
            and self.mass is None
        ):
            if self.charge_to_mass == 0.0:
                msg = (
                    "Cannot decompose charge_to_mass=0 into charge and mass. "
                    "Specify charge=0 and mass explicitly."
                )
                raise ValueError(msg)
            # Convention: |q| = 1, sign from q/m, mass = 1/|q/m|
            object.__setattr__(self, "charge", math.copysign(1.0, self.charge_to_mass))
            object.__setattr__(self, "mass", 1.0 / abs(self.charge_to_mass))
        elif (
            self.charge is not None
            and self.mass is not None
            and self.charge_to_mass is not None
        ):
            if self.mass == 0.0:
                msg = (
                    "A massless species has no charge_to_mass; provide "
                    "charge and mass alone."
                )
                raise ValueError(msg)
            expected = self.charge / self.mass
            if not math.isclose(expected, self.charge_to_mass, rel_tol=1e-12):
                msg = (
                    f"Inconsistent species parameters: "
                    f"charge/mass={expected} != charge_to_mass={self.charge_to_mass}"
                )
                raise ValueError(msg)
        elif self.charge is None and self.mass is None and self.charge_to_mass is None:
            msg = "Must provide charge+mass or charge_to_mass (or all three)."
            raise ValueError(msg)
        else:
            provided = [
                name
                for name, val in [
                    ("charge", self.charge),
                    ("mass", self.mass),
                    ("charge_to_mass", self.charge_to_mass),
                ]
                if val is not None
            ]
            msg = (
                f"Incomplete species parameters: got {', '.join(provided)}. "
                f"Provide charge+mass, charge_to_mass alone, or all three."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PhysicsParams:
    r"""Typed physics parameters consumed by the compute pipeline.

    Known fields have defaults matching the standard non-relativistic
    PIC/MHD conventions.  Reader-specific parameters (iPIC3D theta,
    BATSRUS divb_method, etc.) go in *extra*.

    Parameters
    ----------
    gamma : float
        Adiabatic index ($\gamma = c_p / c_v$).
    c : float
        Speed of light in normalized units.
    relativistic : bool
        Use relativistic formulas for derived quantities.
    extra : dict[str, Any]
        Open-ended reader-specific parameters.

    Examples
    --------
    >>> p = PhysicsParams()
    >>> p.gamma
    1.6666666666666667
    >>> p.c
    1.0
    >>> p = PhysicsParams(gamma=1.4, extra={"eta": 0.01})
    >>> p.extra["eta"]
    0.01
    """

    gamma: float = 5.0 / 3.0
    c: float = 1.0
    relativistic: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.extra, MappingProxyType):
            object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
