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


class TestPhysicsConstants:
    def test_pic_normalized(self):
        pc = PhysicsConstants.pic_normalized()
        assert pc.c == 1.0
        assert pc.epsilon_0 == 1.0
        assert pc.mu_0 == 1.0

    def test_mhd_normalized(self):
        pc = PhysicsConstants.mhd_normalized()
        assert math.isinf(pc.c)
        assert pc.epsilon_0 == 1.0
        assert pc.mu_0 == 1.0

    def test_inv_c_squared_mhd(self):
        assert PhysicsConstants.mhd_normalized().inv_c_squared() == 0.0

    def test_inv_c_squared_pic(self):
        assert PhysicsConstants.pic_normalized().inv_c_squared() == 1.0

    def test_inv_c_squared_custom(self):
        pc = PhysicsConstants(c=10.0, epsilon_0=1.0, mu_0=1.0)
        np.testing.assert_allclose(pc.inv_c_squared(), 0.01, rtol=1e-15)

    def test_electromagnetic_identity(self):
        pc = PhysicsConstants.pic_normalized()
        np.testing.assert_allclose(pc.c**2 * pc.mu_0 * pc.epsilon_0, 1.0, rtol=1e-15)

    def test_frozen(self):
        pc = PhysicsConstants.pic_normalized()
        with pytest.raises(AttributeError):
            pc.c = 2.0  # type: ignore[misc]


class TestSpeciesInfoFromChargeMass:
    def test_electron(self):
        e = SpeciesInfo(name="e", charge=-1.0, mass=1 / 256)
        assert e.charge_to_mass == -256.0

    def test_ion(self):
        ion = SpeciesInfo(name="ion", charge=1.0, mass=1.0)
        assert ion.charge_to_mass == 1.0


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
        with pytest.raises(ValueError, match="both charge and mass"):
            SpeciesInfo(name="x", charge=1.0)

    def test_only_mass_raises(self):
        with pytest.raises(ValueError, match="both charge and mass"):
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


class TestSpeciesInfoOptionalFields:
    def test_defaults_are_none(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0)
        assert s.temperature is None
        assert s.thermal_velocity is None
        assert s.drift_velocity is None
        assert s.density is None
        assert s.particles_per_cell is None

    def test_temperature_preserved(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, temperature=0.1)
        assert s.temperature == 0.1

    def test_density_preserved(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, density=1.0)
        assert s.density == 1.0

    def test_drift_velocity_preserved(self):
        v = (0.1, 0.0, 0.0)
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, drift_velocity=v)
        assert s.drift_velocity == v

    def test_thermal_velocity_scalar(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, thermal_velocity=0.01)
        assert s.thermal_velocity == 0.01

    def test_thermal_velocity_vector3(self):
        vth = (0.01, 0.02, 0.03)
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, thermal_velocity=vth)
        assert s.thermal_velocity == vth

    def test_particles_per_cell_int(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, particles_per_cell=64)
        assert s.particles_per_cell == 64

    def test_particles_per_cell_tuple(self):
        ppc = (8, 8, 8)
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0, particles_per_cell=ppc)
        assert s.particles_per_cell == ppc

    def test_frozen(self):
        s = SpeciesInfo(name="e", charge=-1.0, mass=1.0)
        with pytest.raises(AttributeError):
            s.name = "ion"  # type: ignore[misc]
