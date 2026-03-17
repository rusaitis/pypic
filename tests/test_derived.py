import numpy as np
import pytest

from pypic.derived import (
    agyrotropy,
    alfven_mach,
    alfven_speed,
    current_density_magnitude,
    debye_length,
    electric_energy_density,
    electric_field_magnitude,
    enthalpy,
    entropy,
    gyrofrequency,
    gyroradius,
    gyrotropic_entropy,
    internal_energy,
    ion_acoustic_speed,
    kinetic_energy_density,
    magnetic_energy_density,
    magnetic_field_magnitude,
    magnetosonic_mach,
    magnetosonic_speed,
    parallel_pressure,
    perpendicular_pressure,
    plasma_beta,
    plasma_frequency,
    poynting_flux,
    relativistic_enthalpy,
    skin_depth,
    sound_speed,
    thermal_energy_density,
    thermal_speed,
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
            internal_energy(np.array([1.0]), np.array([1.0])),
            1.5,
            rtol=1e-15,
        )

    def test_enthalpy_is_gamma_times_internal(self):
        p = np.array([2.5])
        rho = np.array([1.3])
        gamma = 5.0 / 3.0
        h = enthalpy(p, rho, gamma)
        e_int = internal_energy(p, rho, gamma)
        np.testing.assert_allclose(h, gamma * e_int, rtol=1e-15)

    def test_enthalpy_equals_internal_plus_p_over_rho(self):
        p = np.array([3.0])
        rho = np.array([2.0])
        h = enthalpy(p, rho)
        e_int = internal_energy(p, rho)
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
    @pytest.mark.parametrize(
        ("explicit_call", "implicit_call"),
        [
            (
                lambda: thermal_energy_density(np.array([1.0]), gamma=5.0 / 3.0),
                lambda: thermal_energy_density(np.array([1.0])),
            ),
            (
                lambda: internal_energy(
                    np.array([1.0]), np.array([1.0]), gamma=5.0 / 3.0
                ),
                lambda: internal_energy(np.array([1.0]), np.array([1.0])),
            ),
            (
                lambda: enthalpy(np.array([1.0]), np.array([1.0]), gamma=5.0 / 3.0),
                lambda: enthalpy(np.array([1.0]), np.array([1.0])),
            ),
            (
                lambda: relativistic_enthalpy(np.array([1.0]), np.array([1.0]), c=1.0),
                lambda: relativistic_enthalpy(np.array([1.0]), np.array([1.0])),
            ),
            (
                lambda: entropy(np.array([2.0]), np.array([3.0]), gamma=5.0 / 3.0),
                lambda: entropy(np.array([2.0]), np.array([3.0])),
            ),
        ],
        ids=[
            "thermal_energy_density",
            "internal_energy",
            "enthalpy",
            "relativistic_enthalpy",
            "entropy",
        ],
    )
    def test_explicit_equals_implicit(self, explicit_call, implicit_call):
        np.testing.assert_allclose(explicit_call(), implicit_call(), rtol=1e-15)


class TestEdgeCases:
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


class TestThermalSpeed:
    def test_t4_m1(self):
        np.testing.assert_allclose(
            thermal_speed(np.array([4.0]), mass=1.0), 2.0, rtol=1e-15
        )

    def test_mass_scaling(self):
        np.testing.assert_allclose(
            thermal_speed(np.array([4.0]), mass=4.0), 1.0, rtol=1e-15
        )


class TestGyrofrequency:
    def test_negative_charge_gives_positive(self):
        np.testing.assert_allclose(
            gyrofrequency(np.array([2.0]), charge=-1.0, mass=1.0), 2.0, rtol=1e-15
        )

    def test_mass_scaling(self):
        np.testing.assert_allclose(
            gyrofrequency(np.array([1.0]), charge=1.0, mass=4.0), 0.25, rtol=1e-15
        )


class TestPlasmaFrequency:
    def test_negative_charge_invariant(self):
        pos = plasma_frequency(np.array([2.0]), charge=1.0, mass=1.0)
        neg = plasma_frequency(np.array([2.0]), charge=-1.0, mass=1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-15)

    def test_density_scaling(self):
        np.testing.assert_allclose(
            plasma_frequency(np.array([4.0]), charge=1.0, mass=1.0), 2.0, rtol=1e-15
        )


class TestSkinDepth:
    def test_identity_d_times_omega_p_equals_c(self):
        n = np.array([3.7])
        q, m, c = 1.5, 2.3, 10.0
        d = skin_depth(n, charge=q, mass=m, c=c)
        omega = plasma_frequency(n, charge=q, mass=m)
        np.testing.assert_allclose(d * omega, c, rtol=1e-15)

    def test_custom_c(self):
        np.testing.assert_allclose(
            skin_depth(np.array([1.0]), charge=1.0, mass=1.0, c=3.0), 3.0, rtol=1e-15
        )


class TestGyroradius:
    def test_negative_charge_gives_same_result(self):
        pos = gyroradius(np.array([2.0]), np.array([3.0]), charge=1.0, mass=1.0)
        neg = gyroradius(np.array([2.0]), np.array([3.0]), charge=-1.0, mass=1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-15)

    def test_equals_vth_over_omega_c(self):
        t = np.array([2.5])
        b = np.array([3.7])
        q, m = 1.5, 2.3
        r = gyroradius(t, b, charge=q, mass=m)
        vth = thermal_speed(t, mass=m)
        omega = gyrofrequency(b, charge=q, mass=m)
        np.testing.assert_allclose(r, vth / omega, rtol=1e-14)


class TestDebyeLength:
    def test_negative_charge_invariant(self):
        pos = debye_length(np.array([2.0]), np.array([3.0]), charge=1.0)
        neg = debye_length(np.array([2.0]), np.array([3.0]), charge=-1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-15)

    def test_known_value(self):
        # T=4, n=1, q=1 → sqrt(4) = 2
        np.testing.assert_allclose(
            debye_length(np.array([4.0]), np.array([1.0]), charge=1.0), 2.0, rtol=1e-15
        )


class TestSoundSpeed:
    def test_default_gamma(self):
        explicit = sound_speed(np.array([2.0]), np.array([3.0]), gamma=5.0 / 3.0)
        implicit = sound_speed(np.array([2.0]), np.array([3.0]))
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_gamma_2(self):
        # gamma=2, P=2, rho=1 → sqrt(4) = 2
        np.testing.assert_allclose(
            sound_speed(np.array([2.0]), np.array([1.0]), gamma=2.0), 2.0, rtol=1e-15
        )


class TestIonAcousticSpeed:
    def test_cold_ions(self):
        # Te=1, Ti=0, m_i=1, gamma_e=1 -> c_ia = 1
        np.testing.assert_allclose(
            ion_acoustic_speed(np.array([1.0]), np.array([0.0]), mass_i=1.0),
            1.0,
            rtol=1e-15,
        )

    def test_defaults(self):
        te = np.array([2.0])
        ti = np.array([0.5])
        explicit = ion_acoustic_speed(te, ti, mass_i=1.0, gamma_e=1.0, gamma_i=3.0)
        implicit = ion_acoustic_speed(te, ti, mass_i=1.0)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_known_value(self):
        # gamma_e*Te + gamma_i*Ti = 1*4 + 3*2 = 10, m_i=10 -> sqrt(1) = 1
        np.testing.assert_allclose(
            ion_acoustic_speed(np.array([4.0]), np.array([2.0]), mass_i=10.0),
            1.0,
            rtol=1e-15,
        )


class TestMagnetosonicSpeed:
    def test_pythagorean(self):
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([3.0]), np.array([4.0])), 5.0, rtol=1e-15
        )

    def test_zero_sound_speed(self):
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([5.0]), np.array([0.0])), 5.0, rtol=1e-15
        )

    def test_zero_alfven_speed(self):
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([0.0]), np.array([7.0])), 7.0, rtol=1e-15
        )


class TestMachNumbers:
    def test_alfven_mach_value(self):
        np.testing.assert_allclose(
            alfven_mach(np.array([6.0]), np.array([3.0])), 2.0, rtol=1e-15
        )

    def test_alfven_mach_zero_va_gives_inf(self):
        result = alfven_mach(np.array([1.0]), np.array([0.0]))
        assert np.isinf(result[0])

    def test_magnetosonic_mach_value(self):
        np.testing.assert_allclose(
            magnetosonic_mach(np.array([10.0]), np.array([5.0])), 2.0, rtol=1e-15
        )

    def test_magnetosonic_mach_zero_vms_gives_inf(self):
        result = magnetosonic_mach(np.array([1.0]), np.array([0.0]))
        assert np.isinf(result[0])


# Shared fixtures for pressure tensor tests
ZEROS = np.array([0.0])
ONES = np.array([1.0])
B_ALONG_Z = (ZEROS, ZEROS, ONES)


class TestParallelPressure:
    def test_isotropic_any_b_direction(self):
        """Isotropic tensor P*δ_ij gives P_∥ = P for any B direction."""
        p = np.array([5.0])
        for b_dir in [(ONES, ZEROS, ZEROS), (ZEROS, ONES, ZEROS), B_ALONG_Z]:
            result = parallel_pressure(p, p, p, ZEROS, ZEROS, ZEROS, *b_dir)
            np.testing.assert_allclose(result, 5.0, rtol=1e-14)

    def test_b_along_z_diagonal(self):
        """B along z: P_∥ = P33."""
        result = parallel_pressure(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 3.0, rtol=1e-15)

    def test_b_along_111(self):
        """B along (1,1,1)/√3 with diagonal tensor diag(1,2,3)."""
        b = np.array([1.0])
        result = parallel_pressure(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            b,
            b,
            b,
        )
        # P_∥ = (1 + 2 + 3) / 3 = 2.0
        np.testing.assert_allclose(result, 2.0, rtol=1e-14)


class TestPerpendicularPressure:
    def test_isotropic_any_b_direction(self):
        """Isotropic tensor P*δ_ij gives P_⊥ = P for any B direction."""
        p = np.array([5.0])
        result = perpendicular_pressure(p, p, p, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        np.testing.assert_allclose(result, 5.0, rtol=1e-14)

    def test_b_along_z_diagonal(self):
        """B along z, diag(1,2,3): P_⊥ = (1+2)/2 = 1.5."""
        result = perpendicular_pressure(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 1.5, rtol=1e-15)

    def test_identity_trace(self):
        """Tr(P) = P_∥ + 2·P_⊥ for arbitrary tensor and B."""
        p11, p22, p33 = np.array([3.0]), np.array([5.0]), np.array([7.0])
        p12, p13, p23 = np.array([0.5]), np.array([-0.3]), np.array([0.2])
        b = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        args = (p11, p22, p33, p12, p13, p23, *b)
        p_par = parallel_pressure(*args)
        p_perp = perpendicular_pressure(*args)
        trace = p11 + p22 + p33
        np.testing.assert_allclose(trace, p_par + 2.0 * p_perp, rtol=1e-14)


class TestAgyrotropy:
    def test_isotropic_is_zero(self):
        """Isotropic tensor → Q = 0 (gyrotropic)."""
        p = np.array([3.0])
        result = agyrotropy(p, p, p, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_gyrotropic_diagonal_is_zero(self):
        """Gyrotropic but anisotropic: diag(2,2,5) with B along z → Q = 0."""
        result = agyrotropy(
            np.array([2.0]),
            np.array([2.0]),
            np.array([5.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_agyrotropic_example(self):
        """B along z, diag(3,1,1): P_∥=1, I₁=4, known Q=0.25."""
        # P_∥ = 1 (P33), Tr(P) = 5, I₁ = 5 - 1 = 4
        # P_⊥ tensor = diag(3,1,0), Frobenius² = 9 + 1 + 0 = 10
        # I₂ = (16 - 10) / 2 = 3
        # Q = 1 - 12/16 = 0.25
        result = agyrotropy(
            np.array([3.0]),
            np.array([1.0]),
            np.array([1.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 0.25, rtol=1e-14)

    def test_bounded_zero_one(self):
        """Q should be in [0, 1] for a range of tensors."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            # Generate a symmetric positive-definite tensor
            a = rng.standard_normal((3, 3))
            p_matrix = a @ a.T + 0.1 * np.eye(3)
            b_vec = rng.standard_normal(3)
            b_vec = b_vec / np.linalg.norm(b_vec)
            q = agyrotropy(
                np.array([p_matrix[0, 0]]),
                np.array([p_matrix[1, 1]]),
                np.array([p_matrix[2, 2]]),
                np.array([p_matrix[0, 1]]),
                np.array([p_matrix[0, 2]]),
                np.array([p_matrix[1, 2]]),
                np.array([b_vec[0]]),
                np.array([b_vec[1]]),
                np.array([b_vec[2]]),
            )
            assert q[0] >= -1e-10, f"Q = {q[0]} < 0"
            assert q[0] <= 1.0 + 1e-10, f"Q = {q[0]} > 1"


SCALAR_FUNCTIONS_1ARG = [
    lambda a: thermal_speed(a, mass=1.0),
    lambda a: plasma_frequency(a, charge=1.0, mass=1.0),
    lambda a: skin_depth(a, charge=1.0, mass=1.0),
]
SCALAR_FUNCTIONS_2ARG = [
    lambda a, b: gyrofrequency(a, charge=1.0, mass=1.0),
    lambda a, b: gyroradius(a, b, charge=1.0, mass=1.0),
    lambda a, b: debye_length(a, b, charge=1.0),
    lambda a, b: sound_speed(a, b),
    lambda a, b: magnetosonic_speed(a, b),
    lambda a, b: alfven_mach(a, b),
    lambda a, b: magnetosonic_mach(a, b),
]


class TestCharacteristicScalesEdgeCases:
    @pytest.mark.parametrize("func", SCALAR_FUNCTIONS_1ARG)
    def test_empty_array_1arg(self, func):
        empty = np.array([], dtype=np.float64)
        result = func(empty)
        assert result.shape == (0,)

    @pytest.mark.parametrize("func", SCALAR_FUNCTIONS_2ARG)
    def test_empty_array_2arg(self, func):
        empty = np.array([], dtype=np.float64)
        result = func(empty, empty)
        assert result.shape == (0,)

    @pytest.mark.parametrize("func", SCALAR_FUNCTIONS_1ARG)
    def test_nan_propagation_1arg(self, func):
        result = func(np.array([np.nan]))
        assert np.isnan(result[0])

    @pytest.mark.parametrize("func", SCALAR_FUNCTIONS_2ARG)
    def test_nan_propagation_2arg(self, func):
        result = func(np.array([np.nan]), np.array([1.0]))
        assert np.isnan(result[0])

    def test_pressure_tensor_nan_propagation(self):
        nan = np.array([np.nan])
        result = parallel_pressure(nan, ONES, ONES, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        assert np.isnan(result[0])

    def test_pressure_tensor_empty_arrays(self):
        empty = np.array([], dtype=np.float64)
        result = parallel_pressure(
            empty, empty, empty, empty, empty, empty, empty, empty, empty
        )
        assert result.shape == (0,)

    def test_agyrotropy_nan_propagation(self):
        nan = np.array([np.nan])
        result = agyrotropy(nan, ONES, ONES, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        assert np.isnan(result[0])
