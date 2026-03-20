import math

import numpy as np
import pytest
from scipy import constants

from pypic.units import Normalization, PhysicsConstants, SpeciesInfo

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
