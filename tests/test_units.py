import math

import numpy as np
import pytest
from scipy import constants

from pypic.exceptions import UndeclaredNormalizationError
from pypic.fields import _QUANTITY_DIMENSIONS
from pypic.units import (
    _REFERENCE_ORDER,
    _SI_FACTORS,
    Normalization,
    PhysicsConstants,
    SpeciesInfo,
    UnitSystem,
)

N_REF = 1e18


class TestPicElectron:
    def setup_method(self):
        self.norm = Normalization.pic_electron(N_REF)

    def test_velocity_ref_is_c(self):
        assert np.isclose(self.norm.normalize("velocity", constants.c), 1.0, rtol=1e-12)

    def test_length_ref_is_skin_depth(self):
        omega_pe = np.sqrt(
            N_REF * constants.e**2 / (constants.epsilon_0 * constants.m_e)
        )
        d_e = constants.c / omega_pe
        np.testing.assert_allclose(self.norm.length_ref, d_e, rtol=1e-12)

    def test_time_ref_is_inverse_omega(self):
        omega_pe = np.sqrt(
            N_REF * constants.e**2 / (constants.epsilon_0 * constants.m_e)
        )
        np.testing.assert_allclose(self.norm.time_ref, 1.0 / omega_pe, rtol=1e-12)


class TestMhd:
    def setup_method(self):
        self.l_0 = 1e6
        self.rho_0 = 1e-12
        self.b_0 = 1e-9
        self.norm = Normalization.mhd_standard(self.l_0, self.rho_0, self.b_0)

    def test_alfven_speed(self):
        v_a = self.b_0 / np.sqrt(constants.mu_0 * self.rho_0)
        np.testing.assert_allclose(self.norm.velocity_ref, v_a, rtol=1e-12)

    def test_time_ref(self):
        v_a = self.b_0 / np.sqrt(constants.mu_0 * self.rho_0)
        np.testing.assert_allclose(self.norm.time_ref, self.l_0 / v_a, rtol=1e-12)

    def test_density_is_number_density(self):
        np.testing.assert_allclose(
            self.norm.density_ref, self.rho_0 / constants.m_p, rtol=1e-12
        )


QUANTITIES = ["length", "time", "velocity", "b_field", "e_field", "density"]


@pytest.fixture(params=["pic", "mhd"], ids=["pic", "mhd"])
def norm(request):
    if request.param == "pic":
        return Normalization.pic_electron(N_REF)
    return Normalization.mhd_standard(1e6, 1e-12, 1e-9)


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_round_trip_scalar(norm, quantity):
    x = 42.0
    normalized = norm.normalize(quantity, x)
    np.testing.assert_allclose(norm.to_si(quantity, normalized), x, rtol=1e-15)


def test_invalid_quantity_raises():
    norm = Normalization.identity()
    with pytest.raises(ValueError, match="Unknown quantity"):
        norm.normalize("lenght", 1.0)
    with pytest.raises(ValueError, match="Unknown quantity"):
        norm.to_si("pressure", 1.0)


class TestPhysicsConstants:
    def test_mhd_normalized(self):
        pc = PhysicsConstants.mhd_normalized()
        assert math.isinf(pc.c)
        assert pc.epsilon_0 == 1.0
        assert pc.mu_0 == 1.0

    def test_inv_c_squared_mhd(self):
        assert PhysicsConstants.mhd_normalized().inv_c_squared() == 0.0

    def test_electromagnetic_identity(self):
        pc = PhysicsConstants.pic_normalized()
        np.testing.assert_allclose(pc.c**2 * pc.mu_0 * pc.epsilon_0, 1.0, rtol=1e-15)


class TestSpeciesInfoFromQom:
    def test_negative_qom(self):
        e = SpeciesInfo(name="e", charge_to_mass=-256.0)
        assert e.charge == -1.0
        np.testing.assert_allclose(e.mass, 1.0 / 256, rtol=1e-15)

    def test_positive_qom(self):
        ion = SpeciesInfo(name="ion", charge_to_mass=256.0)
        assert ion.charge == 1.0
        np.testing.assert_allclose(ion.mass, 1.0 / 256, rtol=1e-15)

    def test_fractional_qom(self):
        s = SpeciesInfo(name="heavy", charge_to_mass=0.5)
        assert s.charge == 1.0
        assert s.mass == 2.0


class TestSpeciesInfoValidation:
    def test_no_params_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            SpeciesInfo(name="x")

    def test_only_charge_raises(self):
        with pytest.raises(ValueError, match="Incomplete species"):
            SpeciesInfo(name="x", charge=1.0)

    def test_only_mass_raises(self):
        with pytest.raises(ValueError, match="Incomplete species"):
            SpeciesInfo(name="x", mass=1.0)

    def test_zero_qom_raises(self):
        with pytest.raises(ValueError, match="Cannot decompose"):
            SpeciesInfo(name="neutral", charge_to_mass=0.0)

    def test_inconsistent_all_three_raises(self):
        with pytest.raises(ValueError, match="Inconsistent"):
            SpeciesInfo(name="bad", charge=1.0, mass=1.0, charge_to_mass=99.0)

    def test_consistent_all_three_ok(self):
        s = SpeciesInfo(name="ok", charge=2.0, mass=1.0, charge_to_mass=2.0)
        assert s.charge_to_mass == 2.0

    def test_massless_species_has_no_charge_to_mass(self):
        """Massless fluid electrons are the standard hybrid closure.

        The ratio is left unset rather than infinite — consumers
        already branch on its absence, where an inf would travel
        silently into whatever moment used it next.
        """
        electrons = SpeciesInfo(name="electrons", charge=-1.0, mass=0.0)
        assert electrons.charge_to_mass is None

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="mass must not be negative"):
            SpeciesInfo(name="bad", charge=-1.0, mass=-1.0)


class TestCompoundSiFactors:
    """Normalization.si_factor() handles compound quantity types correctly."""

    @pytest.mark.parametrize(
        ("quantity", "expected_fn"),
        [
            ("dimensionless", lambda n: 1.0),
            ("pressure", lambda n: n.density_ref * n.mass_ref * n.velocity_ref**2),
            ("temperature", lambda n: n.mass_ref * n.velocity_ref**2),
            (
                "energy_density",
                lambda n: n.density_ref * n.mass_ref * n.velocity_ref**2,
            ),
            (
                "current_density",
                lambda n: n.charge_ref * n.density_ref * n.velocity_ref,
            ),
            ("frequency", lambda n: 1.0 / n.time_ref),
            ("mass_density", lambda n: n.density_ref * n.mass_ref),
            ("charge_density", lambda n: n.charge_ref * n.density_ref),
        ],
    )
    def test_compound_factor(self, norm, quantity, expected_fn):
        result = norm.si_factor(quantity)
        expected = expected_fn(norm)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_poynting_flux_converts_to_the_si_cross_product(self, norm):
        # Asserted against S = E x B / mu_0 built from SI inputs rather than
        # against the reference product, which would restate the implementation
        # and is how the missing mu_0 survived.
        e_code, b_code = 0.1, 0.2
        expected = (e_code * norm.e_field_ref) * (b_code * norm.b_field_ref)
        expected /= constants.mu_0

        s_code = e_code * b_code  # derived.poynting_flux, mu_0 = 1
        assert s_code * norm.si_factor("poynting_flux") == pytest.approx(
            expected, rel=1e-12
        )

    def test_unknown_compound_raises(self, norm):
        with pytest.raises(ValueError, match="Unknown quantity"):
            norm.si_factor("flux_capacitance")


# SI dimension of each storage primitive as integer powers of
# (length, mass, time, current, temperature, amount, luminosity) — written
# from the SI definitions rather than copied from either table, so the test
# below is a genuine cross-check and not a restatement of one of them.
_REFERENCE_DIMENSIONS = {
    "length": (1, 0, 0, 0, 0, 0, 0),  # m
    "time": (0, 0, 1, 0, 0, 0, 0),  # s
    "velocity": (1, 0, -1, 0, 0, 0, 0),  # m/s
    "b_field": (0, 1, -2, -1, 0, 0, 0),  # T = kg/(A s^2)
    "e_field": (1, 1, -3, -1, 0, 0, 0),  # V/m = kg m/(A s^3)
    "density": (-3, 0, 0, 0, 0, 0, 0),  # m^-3
    "mass": (0, 1, 0, 0, 0, 0, 0),  # kg
    "charge": (0, 0, 1, 1, 0, 0, 0),  # C = A s
}
_MU_0_DIMENSION = (1, 1, -2, -2, 0, 0, 0)  # H/m = kg m/(A^2 s^2)


class TestSIFactorDimensions:
    """The SI factors and the openPMD dimensions must encode one physics."""

    def test_every_factor_carries_its_registered_dimension(self) -> None:
        """Compose each factor's exponents and match `_QUANTITY_DIMENSIONS`.

        These two tables say the same thing in two notations and were
        maintained independently, with nothing relating them — which is
        how `poynting_flux` kept a factor that disagreed with its own
        registered W/m^2 for as long as it did.
        """
        mismatched = {}
        for quantity, dimension in _QUANTITY_DIMENSIONS.items():
            spec = _SI_FACTORS[quantity]
            composed = [0] * 7
            for name, power in zip(_REFERENCE_ORDER, spec.exponents, strict=True):
                for axis, base in enumerate(_REFERENCE_DIMENSIONS[name]):
                    composed[axis] += power * base
            for axis, base in enumerate(_MU_0_DIMENSION):
                composed[axis] += spec.mu_0_power * base
            if tuple(composed) != dimension:
                mismatched[quantity] = (tuple(composed), dimension)

        assert not mismatched, (
            "SI factor exponents disagree with the registered openPMD "
            f"dimension (composed, registered): {mismatched}"
        )

    def test_every_field_quantity_type_has_a_dimension_and_a_factor(self) -> None:
        """Neither table may grow an entry the other does not know."""
        assert set(_QUANTITY_DIMENSIONS) <= _SI_FACTORS.keys()


class TestRationalizationRatio:
    """The ratio names the code-unit convention a reference set implies."""

    @pytest.mark.parametrize(
        ("name", "norm", "expected"),
        [
            ("pic_electron", Normalization.pic_electron(N_REF), 1.0),
            (
                "pic_standard, ion-referenced",
                Normalization.pic_standard(1e6, constants.m_p, constants.e),
                1.0,
            ),
            (
                "pic_standard, hybrid velocity anchor",
                Normalization.pic_standard(
                    1e6, constants.m_p, constants.e, constants.c, 1.0e5
                ),
                1.0,
            ),
            ("mhd_standard", Normalization.mhd_standard(1e6, 1e-12, 1e-9), 1.0),
            ("identity", Normalization.identity(), 1.0 / constants.mu_0),
        ],
    )
    def test_builtin_constructors_declare_their_convention(self, name, norm, expected):
        np.testing.assert_allclose(
            norm.rationalization_ratio,
            expected,
            rtol=1e-9,
            err_msg=f"{name} is not the convention derived.py computes in",
        )

    def test_a_reduced_speed_of_light_shows_up_as_c_ratio_squared(self):
        # The measurable symptom of the trap: B_ref is c-independent while
        # velocity_ref scales, so the ratio tracks (c_SI / c_ref)**2.
        norm = Normalization.pic_standard(
            N_REF, constants.m_e, constants.e, constants.c / 10
        )
        np.testing.assert_allclose(norm.rationalization_ratio, 100.0, rtol=1e-9)

    def test_the_si_anchor_reports_its_ratio_like_every_other(self):
        """schema.md promises `pypic info` shows the ratio whenever it is not 1.

        The SI anchor is the row that promise most needs to cover — all
        eight references are 1.0, so every EM quantity is off by a power of
        mu_0 — and it was the one case the summary returned early on.
        """
        assert "7.958e+05" in Normalization.identity().summary()

    def test_the_si_anchor_summary_names_the_way_out(self):
        assert "data_in_si" in Normalization.identity().summary()


class TestUnitSystemProvenance:
    """An undeclared normalization is distinguishable from declared SI."""

    def test_every_constructor_stamps_its_own_system(self):
        built = {
            "identity": (Normalization.identity(), UnitSystem.SI),
            "undeclared": (Normalization.undeclared(), None),
            "pic_electron": (
                Normalization.pic_electron(N_REF),
                UnitSystem.FROM_SPECIES,
            ),
            "pic_standard": (
                Normalization.pic_standard(N_REF, constants.m_e, constants.e),
                UnitSystem.FROM_SPECIES,
            ),
            "mhd_standard": (
                Normalization.mhd_standard(1e6, 1e-12, 1e-9),
                UnitSystem.EXPLICIT,
            ),
            "eight refs by hand": (
                Normalization(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                UnitSystem.EXPLICIT,
            ),
        }
        wrong = {
            name: (norm.system, expected)
            for name, (norm, expected) in built.items()
            if norm.system is not expected
        }
        assert not wrong, f"constructors stamped the wrong system: {wrong}"

    def test_undeclared_carries_the_same_references_as_identity(self):
        undeclared = Normalization.undeclared()
        assert undeclared.is_identity

    def test_undeclared_refuses_a_dimensional_conversion(self):
        with pytest.raises(UndeclaredNormalizationError, match="no unit system"):
            Normalization.undeclared().si_factor("b_field")

    def test_undeclared_still_converts_dimensionless_quantities(self):
        assert Normalization.undeclared().si_factor("dimensionless") == 1.0

    def test_declared_si_converts_unchanged(self):
        assert Normalization.identity().si_factor("b_field") == 1.0

    def test_the_error_names_every_escape_hatch(self):
        """The message has to say what to do, not just that it failed."""
        with pytest.raises(UndeclaredNormalizationError) as exc:
            Normalization.undeclared().si_factor("pressure")
        detail = exc.value.detail
        missing = [
            hatch
            for hatch in ("simulation.toml", "normalization=", 'units="code"')
            if hatch not in detail
        ]
        assert not missing, f"escape hatches absent from the message: {missing}"

    def test_undeclared_is_not_equal_to_declared_si(self):
        """The whole point: the two must not compare equal."""
        assert Normalization.undeclared() != Normalization.identity()


class TestNormalizationValidation:
    def test_zero_ref_raises(self):
        with pytest.raises(ValueError, match="length_ref must be positive"):
            Normalization(
                length_ref=0.0,
                time_ref=1.0,
                velocity_ref=1.0,
                b_field_ref=1.0,
                e_field_ref=1.0,
                density_ref=1.0,
                mass_ref=1.0,
                charge_ref=1.0,
            )

    def test_negative_ref_raises(self):
        with pytest.raises(ValueError, match="density_ref must be positive"):
            Normalization(
                length_ref=1.0,
                time_ref=1.0,
                velocity_ref=1.0,
                b_field_ref=1.0,
                e_field_ref=1.0,
                density_ref=-1.0,
                mass_ref=1.0,
                charge_ref=1.0,
            )
