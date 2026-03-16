import numpy as np
import pytest

from pypic.derived import (
    alfven_speed,
    current_density_magnitude,
    electric_energy_density,
    electric_field_magnitude,
    enthalpy,
    entropy,
    gyrotropic_entropy,
    internal_energy_density,
    kinetic_energy_density,
    magnetic_energy_density,
    magnetic_field_magnitude,
    plasma_beta,
    poynting_flux,
    relativistic_enthalpy,
    thermal_energy_density,
    velocity_magnitude,
)

MAGNITUDE_FUNCTIONS = [
    magnetic_field_magnitude,
    electric_field_magnitude,
    current_density_magnitude,
    velocity_magnitude,
]


class TestMagnitudes:
    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_345_triangle(self, func):
        result = func(np.array([3.0]), np.array([4.0]), np.array([0.0]))
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_unit_vector(self, func):
        result = func(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_all_ones(self, func):
        result = func(np.array([1.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(result, np.sqrt(3.0), rtol=1e-15)

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_zeros(self, func):
        result = func(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_multidimensional(self, func):
        c1 = np.array([[3.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        c2 = np.array([[4.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        c3 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        result = func(c1, c2, c3)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0, 0], 5.0, rtol=1e-15)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[0, 2], np.sqrt(3.0), rtol=1e-15)

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_nan_propagation(self, func):
        result = func(np.array([np.nan]), np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])

    @pytest.mark.parametrize("func", MAGNITUDE_FUNCTIONS)
    def test_empty_array(self, func):
        empty = np.array([], dtype=np.float64)
        result = func(empty, empty, empty)
        assert result.shape == (0,)


class TestPlasmaBeta:
    def test_unit_values(self):
        np.testing.assert_allclose(
            plasma_beta(np.array([1.0]), np.array([1.0])), 2.0, rtol=1e-15
        )

    def test_low_beta(self):
        np.testing.assert_allclose(
            plasma_beta(np.array([0.5]), np.array([2.0])), 0.25, rtol=1e-15
        )

    def test_beta_one(self):
        b = np.array([2.0])
        pressure = b**2 / 2.0
        np.testing.assert_allclose(plasma_beta(pressure, b), 1.0, rtol=1e-15)

    def test_zero_b_gives_inf(self):
        result = plasma_beta(np.array([1.0]), np.array([0.0]))
        assert np.isinf(result[0])


class TestAlfvenSpeed:
    def test_claudemd_example(self):
        np.testing.assert_allclose(
            alfven_speed(np.array([1.0]), np.array([4.0])), 0.5, rtol=1e-15
        )

    def test_unit_values(self):
        np.testing.assert_allclose(
            alfven_speed(np.array([1.0]), np.array([1.0])), 1.0, rtol=1e-15
        )

    def test_zero_density_gives_inf(self):
        result = alfven_speed(np.array([1.0]), np.array([0.0]))
        assert np.isinf(result[0])


class TestEnergyDensities:
    def test_magnetic_energy(self):
        np.testing.assert_allclose(
            magnetic_energy_density(np.array([2.0])), 2.0, rtol=1e-15
        )

    def test_electric_energy(self):
        np.testing.assert_allclose(
            electric_energy_density(np.array([3.0])), 4.5, rtol=1e-15
        )

    def test_kinetic_energy(self):
        np.testing.assert_allclose(
            kinetic_energy_density(np.array([2.0]), np.array([3.0])), 9.0, rtol=1e-15
        )

    def test_thermal_energy_default_gamma(self):
        np.testing.assert_allclose(
            thermal_energy_density(np.array([1.0])), 1.5, rtol=1e-15
        )

    def test_thermal_energy_gamma_2(self):
        np.testing.assert_allclose(
            thermal_energy_density(np.array([1.0]), gamma=2.0), 1.0, rtol=1e-15
        )


class TestPoyntingFlux:
    def test_e_along_x_b_along_y(self):
        s1, s2, s3 = poynting_flux(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(s1, 0.0, atol=1e-15)
        np.testing.assert_allclose(s2, 0.0, atol=1e-15)
        np.testing.assert_allclose(s3, 1.0, rtol=1e-15)

    def test_cyclic_e_along_y_b_along_z(self):
        s1, s2, s3 = poynting_flux(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(s1, 1.0, rtol=1e-15)
        np.testing.assert_allclose(s2, 0.0, atol=1e-15)
        np.testing.assert_allclose(s3, 0.0, atol=1e-15)

    def test_cyclic_e_along_z_b_along_x(self):
        s1, s2, s3 = poynting_flux(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(s1, 0.0, atol=1e-15)
        np.testing.assert_allclose(s2, 1.0, rtol=1e-15)
        np.testing.assert_allclose(s3, 0.0, atol=1e-15)

    def test_antisymmetry(self):
        e = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        b = (np.array([4.0]), np.array([5.0]), np.array([6.0]))
        s1, s2, s3 = poynting_flux(*e, *b)
        # Swap E and B: ExB -> BxE = -(ExB)
        r1, r2, r3 = poynting_flux(*b, *e)
        np.testing.assert_allclose(r1, -s1, rtol=1e-15)
        np.testing.assert_allclose(r2, -s2, rtol=1e-15)
        np.testing.assert_allclose(r3, -s3, rtol=1e-15)

    def test_multidimensional(self):
        shape = (2, 3)
        e1 = np.ones(shape)
        e2 = np.zeros(shape)
        e3 = np.zeros(shape)
        b1 = np.zeros(shape)
        b2 = np.ones(shape)
        b3 = np.zeros(shape)
        _s1, _s2, s3 = poynting_flux(e1, e2, e3, b1, b2, b3)
        assert s3.shape == shape
        np.testing.assert_allclose(s3, 1.0, rtol=1e-15)


class TestThermodynamics:
    def test_internal_energy_default_gamma(self):
        np.testing.assert_allclose(
            internal_energy_density(np.array([1.0]), np.array([1.0])),
            1.5,
            rtol=1e-15,
        )

    def test_enthalpy_is_gamma_times_internal(self):
        p = np.array([2.5])
        rho = np.array([1.3])
        gamma = 5.0 / 3.0
        h = enthalpy(p, rho, gamma)
        e_int = internal_energy_density(p, rho, gamma)
        np.testing.assert_allclose(h, gamma * e_int, rtol=1e-15)

    def test_enthalpy_equals_internal_plus_p_over_rho(self):
        p = np.array([3.0])
        rho = np.array([2.0])
        h = enthalpy(p, rho)
        e_int = internal_energy_density(p, rho)
        np.testing.assert_allclose(h, e_int + p / rho, rtol=1e-15)

    def test_relativistic_enthalpy_c1(self):
        p = np.array([1.0])
        rho = np.array([1.0])
        h = enthalpy(p, rho)
        np.testing.assert_allclose(
            relativistic_enthalpy(p, rho, c=1.0), 1.0 + h, rtol=1e-15
        )

    def test_relativistic_enthalpy_c10(self):
        p = np.array([1.0])
        rho = np.array([1.0])
        h = enthalpy(p, rho)
        np.testing.assert_allclose(
            relativistic_enthalpy(p, rho, c=10.0), 100.0 + h, rtol=1e-15
        )

    def test_entropy_unit_values(self):
        np.testing.assert_allclose(
            entropy(np.array([1.0]), np.array([1.0])), 0.0, atol=1e-15
        )

    def test_entropy_e_over_1(self):
        np.testing.assert_allclose(
            entropy(np.array([np.e]), np.array([1.0])), 1.0, rtol=1e-15
        )

    def test_gyrotropic_entropy_unit_values(self):
        np.testing.assert_allclose(
            gyrotropic_entropy(np.array([1.0]), np.array([1.0]), np.array([1.0])),
            0.0,
            atol=1e-15,
        )

    def test_gyrotropic_entropy_exponent_is_5(self):
        n = np.array([2.0])
        # With P_par=P_perp=1: log(1 * 1 / n^5) = -5 * log(n)
        expected = -5.0 * np.log(n)
        result = gyrotropic_entropy(np.array([1.0]), np.array([1.0]), n)
        np.testing.assert_allclose(result, expected, rtol=1e-15)


class TestDefaultParameters:
    def test_thermal_energy_uses_default_gamma(self):
        p = np.array([1.0])
        explicit = thermal_energy_density(p, gamma=5.0 / 3.0)
        implicit = thermal_energy_density(p)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_internal_energy_uses_default_gamma(self):
        p, rho = np.array([1.0]), np.array([1.0])
        explicit = internal_energy_density(p, rho, gamma=5.0 / 3.0)
        implicit = internal_energy_density(p, rho)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_enthalpy_uses_default_gamma(self):
        p, rho = np.array([1.0]), np.array([1.0])
        explicit = enthalpy(p, rho, gamma=5.0 / 3.0)
        implicit = enthalpy(p, rho)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_relativistic_enthalpy_uses_default_c(self):
        p, rho = np.array([1.0]), np.array([1.0])
        explicit = relativistic_enthalpy(p, rho, c=1.0)
        implicit = relativistic_enthalpy(p, rho)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_entropy_uses_default_gamma(self):
        p, rho = np.array([2.0]), np.array([3.0])
        explicit = entropy(p, rho, gamma=5.0 / 3.0)
        implicit = entropy(p, rho)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)


class TestEdgeCases:
    def test_single_element_array(self):
        result = magnetic_field_magnitude(
            np.array([3.0]), np.array([4.0]), np.array([0.0])
        )
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 5.0, rtol=1e-15)

    def test_nan_in_energy(self):
        assert np.isnan(magnetic_energy_density(np.array([np.nan]))[0])
        assert np.isnan(kinetic_energy_density(np.array([np.nan]), np.array([1.0]))[0])
        assert np.isnan(thermal_energy_density(np.array([np.nan]))[0])

    def test_nan_in_thermodynamics(self):
        assert np.isnan(entropy(np.array([np.nan]), np.array([1.0]))[0])
        assert np.isnan(enthalpy(np.array([np.nan]), np.array([1.0]))[0])
        assert np.isnan(
            gyrotropic_entropy(np.array([np.nan]), np.array([1.0]), np.array([1.0]))[0]
        )
