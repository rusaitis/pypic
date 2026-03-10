import numpy as np
import pytest
from scipy import constants

from pypic.units import Normalization

N_REF = 1e18


class TestPicElectron:
    def setup_method(self):
        self.norm = Normalization.pic_electron(N_REF)

    def test_velocity_ref_is_c(self):
        assert np.isclose(self.norm.normalize_velocity(constants.c), 1.0, rtol=1e-12)

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


class TestPicIon:
    def setup_method(self):
        self.norm = Normalization.pic_ion(N_REF, constants.m_p, constants.e)

    def test_velocity_ref_is_c(self):
        assert np.isclose(self.norm.normalize_velocity(constants.c), 1.0, rtol=1e-12)

    def test_length_ref_is_ion_skin_depth(self):
        omega_pi = np.sqrt(
            N_REF * constants.e**2 / (constants.epsilon_0 * constants.m_p)
        )
        d_i = constants.c / omega_pi
        np.testing.assert_allclose(self.norm.length_ref, d_i, rtol=1e-12)


class TestPicStandardEquivalence:
    def test_matches_pic_electron(self):
        from_standard = Normalization.pic_standard(N_REF, constants.m_e, constants.e)
        from_electron = Normalization.pic_electron(N_REF)
        for field in Normalization.__dataclass_fields__:
            np.testing.assert_allclose(
                getattr(from_standard, field),
                getattr(from_electron, field),
                rtol=1e-15,
            )

    def test_matches_pic_ion(self):
        from_standard = Normalization.pic_standard(N_REF, constants.m_p, constants.e)
        from_ion = Normalization.pic_ion(N_REF, constants.m_p, constants.e)
        for field in Normalization.__dataclass_fields__:
            np.testing.assert_allclose(
                getattr(from_standard, field),
                getattr(from_ion, field),
                rtol=1e-15,
            )


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
    normalize = getattr(norm, f"normalize_{quantity}")
    to_si = getattr(norm, f"to_si_{quantity}")
    np.testing.assert_allclose(to_si(normalize(x)), x, rtol=1e-15)


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_round_trip_array(norm, quantity):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    normalize = getattr(norm, f"normalize_{quantity}")
    to_si = getattr(norm, f"to_si_{quantity}")
    result = to_si(normalize(x))
    np.testing.assert_allclose(result, x, rtol=1e-15)
    assert result.shape == x.shape


class TestIdentity:
    def setup_method(self):
        self.norm = Normalization.identity()

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_normalize_is_noop(self, quantity):
        x = 7.5
        normalize = getattr(self.norm, f"normalize_{quantity}")
        assert normalize(x) == x

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_to_si_is_noop(self, quantity):
        x = 7.5
        to_si = getattr(self.norm, f"to_si_{quantity}")
        assert to_si(x) == x

    def test_all_refs_are_one(self):
        for field in Normalization.__dataclass_fields__:
            assert getattr(self.norm, field) == 1.0


class TestArrayInputs:
    def test_preserves_shape_1d(self):
        norm = Normalization.pic_electron(N_REF)
        x = np.linspace(0, 1, 50)
        result = norm.normalize_length(x)
        assert result.shape == (50,)

    def test_preserves_shape_2d(self):
        norm = Normalization.pic_electron(N_REF)
        x = np.ones((3, 4))
        result = norm.to_si_b_field(x)
        assert result.shape == (3, 4)
