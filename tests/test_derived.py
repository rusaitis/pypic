from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pypic.derived import (
    agyrotropy,
    alfven_mach,
    alfven_speed,
    aunai_nongyrotropy,
    current_density_magnitude,
    debye_length,
    electric_energy_density,
    electric_field_magnitude,
    electron_frame_dissipation,
    enthalpy,
    entropy,
    firehose_parameter,
    gyrofrequency,
    gyroradius,
    gyrotropic_entropy,
    hall_electric_field,
    ideal_electric_field,
    internal_energy,
    ion_acoustic_speed,
    isotropic_pressure,
    j_dot_e,
    kinetic_energy_density,
    local_reconnection_rate,
    lorentz_factor,
    lorentz_factor_from_four_velocity,
    magnetic_energy_density,
    magnetic_field_magnitude,
    magnetic_flux_function,
    magnetic_shear_angle,
    magnetization,
    magnetosonic_mach,
    magnetosonic_speed,
    mirror_parameter,
    non_ideal_electric_field,
    parallel_component,
    parallel_pressure,
    perpendicular_magnitude,
    perpendicular_pressure,
    perpendicular_vector,
    plasma_beta,
    plasma_frequency,
    poynting_flux,
    relativistic_enthalpy,
    scudder_agyrotropy,
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
    def test_known_values(self):
        # P=1, B=1 → beta = 2
        np.testing.assert_allclose(
            plasma_beta(np.array([1.0]), np.array([1.0])), 2.0, rtol=1e-15
        )
        # Low beta: P=0.5, B=2 → beta = 0.25
        np.testing.assert_allclose(
            plasma_beta(np.array([0.5]), np.array([2.0])), 0.25, rtol=1e-15
        )
        # Beta = 1 when P = B^2/2
        b = np.array([2.0])
        np.testing.assert_allclose(plasma_beta(b**2 / 2.0, b), 1.0, rtol=1e-15)

    def test_zero_b_gives_nan(self):
        result = plasma_beta(np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])


class TestAlfvenSpeed:
    def test_known_values(self):
        np.testing.assert_allclose(
            alfven_speed(np.array([1.0]), np.array([4.0])), 0.5, rtol=1e-15
        )
        np.testing.assert_allclose(
            alfven_speed(np.array([1.0]), np.array([1.0])), 1.0, rtol=1e-15
        )

    def test_invalid_density_gives_nan(self):
        with np.errstate(invalid="ignore"):
            assert np.isnan(alfven_speed(np.array([1.0]), np.array([0.0]))[0])
            assert np.isnan(alfven_speed(np.array([1.0]), np.array([-1.0]))[0])


class TestEnergyDensities:
    def test_known_values(self):
        # B=2 → e_B = B^2/2 = 2
        np.testing.assert_allclose(
            magnetic_energy_density(np.array([2.0])), 2.0, rtol=1e-15
        )
        # E=3 → e_E = E^2/2 = 4.5
        np.testing.assert_allclose(
            electric_energy_density(np.array([3.0])), 4.5, rtol=1e-15
        )
        # rho=2, V=3 → e_k = 0.5*2*9 = 9
        np.testing.assert_allclose(
            kinetic_energy_density(np.array([2.0]), np.array([3.0])), 9.0, rtol=1e-15
        )

    def test_thermal_energy(self):
        np.testing.assert_allclose(
            thermal_energy_density(np.array([1.0])), 1.5, rtol=1e-15
        )
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

    def test_relativistic_enthalpy(self):
        p = np.array([1.0])
        rho = np.array([1.0])
        h = enthalpy(p, rho)
        np.testing.assert_allclose(
            relativistic_enthalpy(p, rho, c=1.0), 1.0 + h, rtol=1e-15
        )
        np.testing.assert_allclose(
            relativistic_enthalpy(p, rho, c=10.0), 100.0 + h, rtol=1e-15
        )

    def test_entropy_values(self):
        np.testing.assert_allclose(
            entropy(np.array([1.0]), np.array([1.0])), 0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            entropy(np.array([np.e]), np.array([1.0])), 1.0, rtol=1e-15
        )

    def test_gyrotropic_entropy(self):
        np.testing.assert_allclose(
            gyrotropic_entropy(np.array([1.0]), np.array([1.0]), np.array([1.0])),
            0.0,
            atol=1e-15,
        )
        # With P_par=P_perp=1: log(1 * 1 / n^5) = -5 * log(n)
        n = np.array([2.0])
        expected = -5.0 * np.log(n)
        result = gyrotropic_entropy(np.array([1.0]), np.array([1.0]), n)
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_negative_pressure_gives_nan(self):
        with np.errstate(invalid="ignore"):
            assert np.isnan(entropy(np.array([-1.0]), np.array([1.0]))[0])
            assert np.isnan(
                gyrotropic_entropy(np.array([-1.0]), np.array([1.0]), np.array([1.0]))[
                    0
                ]
            )


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
    def test_skin_depth(self):
        # Identity: d * omega_p = c
        n = np.array([3.7])
        q, m, c = 1.5, 2.3, 10.0
        d = skin_depth(n, charge=q, mass=m, c=c)
        omega = plasma_frequency(n, charge=q, mass=m)
        np.testing.assert_allclose(d * omega, c, rtol=1e-15)
        # Known value: n=1, q=1, m=1, c=3 → d = 3
        np.testing.assert_allclose(
            skin_depth(np.array([1.0]), charge=1.0, mass=1.0, c=3.0), 3.0, rtol=1e-15
        )


class TestGyroradius:
    def test_gyroradius(self):
        # Charge sign invariance
        pos = gyroradius(np.array([2.0]), np.array([3.0]), charge=1.0, mass=1.0)
        neg = gyroradius(np.array([2.0]), np.array([3.0]), charge=-1.0, mass=1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-15)
        # Identity: r = v_th / omega_c
        t = np.array([2.5])
        b = np.array([3.7])
        q, m = 1.5, 2.3
        r = gyroradius(t, b, charge=q, mass=m)
        vth = thermal_speed(t, mass=m)
        omega = gyrofrequency(b, charge=q, mass=m)
        np.testing.assert_allclose(r, vth / omega, rtol=1e-14)


class TestDebyeLength:
    def test_debye_length(self):
        # Charge sign invariance
        pos = debye_length(np.array([2.0]), np.array([3.0]), charge=1.0)
        neg = debye_length(np.array([2.0]), np.array([3.0]), charge=-1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-15)
        # Known value: T=4, n=1, q=1 → sqrt(4) = 2
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
            ion_acoustic_speed(np.array([1.0]), np.array([0.0]), mass=1.0),
            1.0,
            rtol=1e-15,
        )

    def test_defaults(self):
        te = np.array([2.0])
        ti = np.array([0.5])
        explicit = ion_acoustic_speed(te, ti, mass=1.0, gamma_e=1.0, gamma_i=3.0)
        implicit = ion_acoustic_speed(te, ti, mass=1.0)
        np.testing.assert_allclose(explicit, implicit, rtol=1e-15)

    def test_known_value(self):
        # gamma_e*Te + gamma_i*Ti = 1*4 + 3*2 = 10, m_i=10 -> sqrt(1) = 1
        np.testing.assert_allclose(
            ion_acoustic_speed(np.array([4.0]), np.array([2.0]), mass=10.0),
            1.0,
            rtol=1e-15,
        )


class TestMagnetosonicSpeed:
    def test_pythagorean(self):
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([3.0]), np.array([4.0])), 5.0, rtol=1e-15
        )

    def test_limiting_cases(self):
        # Zero sound speed → v_ms = v_A
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([5.0]), np.array([0.0])), 5.0, rtol=1e-15
        )
        # Zero Alfvén speed → v_ms = c_s
        np.testing.assert_allclose(
            magnetosonic_speed(np.array([0.0]), np.array([7.0])), 7.0, rtol=1e-15
        )


class TestMachNumbers:
    def test_known_values(self):
        np.testing.assert_allclose(
            alfven_mach(np.array([6.0]), np.array([3.0])), 2.0, rtol=1e-15
        )
        np.testing.assert_allclose(
            magnetosonic_mach(np.array([10.0]), np.array([5.0])), 2.0, rtol=1e-15
        )

    def test_zero_denominator_gives_nan(self):
        assert np.isnan(alfven_mach(np.array([1.0]), np.array([0.0]))[0])
        assert np.isnan(magnetosonic_mach(np.array([1.0]), np.array([0.0]))[0])


# Shared fixtures for pressure tensor tests
ZEROS = np.array([0.0])
ONES = np.array([1.0])
B_ALONG_Z = (ZEROS, ZEROS, ONES)


def _tensor_b_args(p: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, ...]:
    """Pack a 3x3 symmetric P and 3-vector B into the (p11, p22, p33,
    p12, p13, p23, b1, b2, b3) argument layout used by the pressure-
    tensor diagnostics."""
    return (
        np.array([p[0, 0]]),
        np.array([p[1, 1]]),
        np.array([p[2, 2]]),
        np.array([p[0, 1]]),
        np.array([p[0, 2]]),
        np.array([p[1, 2]]),
        np.array([b[0]]),
        np.array([b[1]]),
        np.array([b[2]]),
    )


def _random_rotation(*, seed: int) -> np.ndarray:
    """Build a deterministic 3x3 proper rotation matrix via QR of
    a seeded random normal matrix.  Used to verify rotational
    invariance of tensor diagnostics."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestIsotropicPressure:
    def test_trace_divided_by_three(self):
        """P_iso = (P_11 + P_22 + P_33) / 3."""
        p11, p22, p33 = np.array([3.0]), np.array([6.0]), np.array([9.0])
        np.testing.assert_allclose(isotropic_pressure(p11, p22, p33), 6.0, rtol=1e-15)

    def test_isotropic_tensor_returns_diagonal(self):
        """Isotropic tensor P*δ_ij → P_iso = P."""
        p = np.array([5.0])
        np.testing.assert_allclose(isotropic_pressure(p, p, p), 5.0, rtol=1e-15)

    def test_consistent_with_par_perp(self):
        """P_iso = (P_∥ + 2·P_⊥) / 3 for arbitrary tensor and B."""
        p11, p22, p33 = np.array([3.0]), np.array([5.0]), np.array([7.0])
        p12, p13, p23 = np.array([0.5]), np.array([-0.3]), np.array([0.2])
        b = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        args = (p11, p22, p33, p12, p13, p23, *b)
        p_par = parallel_pressure(*args)
        p_perp = perpendicular_pressure(*args)
        p_iso = isotropic_pressure(p11, p22, p33)
        np.testing.assert_allclose(p_iso, (p_par + 2.0 * p_perp) / 3, rtol=1e-14)

    def test_nan_propagation(self):
        """NaN in any diagonal component propagates."""
        result = isotropic_pressure(
            np.array([np.nan]), np.array([1.0]), np.array([1.0])
        )
        assert np.isnan(result[0])


class TestParallelPressure:
    def test_isotropic_any_b_direction(self):
        """Isotropic tensor P*δ_ij gives P_∥ = P for any B direction."""
        p = np.array([5.0])
        for b_dir in [(ONES, ZEROS, ZEROS), (ZEROS, ONES, ZEROS), B_ALONG_Z]:
            result = parallel_pressure(p, p, p, ZEROS, ZEROS, ZEROS, *b_dir)
            np.testing.assert_allclose(result, 5.0, rtol=1e-14)

    def test_b_along_z_diagonal(self):
        """B along z: P_∥ = P_33."""
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


class TestParallelComponent:
    def test_b_along_z_picks_z(self):
        """B along z extracts the z-component of A."""
        a = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        result = parallel_component(*a, *B_ALONG_Z)
        np.testing.assert_allclose(result, 3.0, rtol=1e-14)

    def test_a_parallel_to_b(self):
        """A ∥ B → A_∥ = |A| (signed positive)."""
        a = (ZEROS, ZEROS, np.array([7.0]))
        result = parallel_component(*a, *B_ALONG_Z)
        np.testing.assert_allclose(result, 7.0, rtol=1e-14)

    def test_a_perpendicular_to_b(self):
        """A ⊥ B → A_∥ = 0."""
        a = (np.array([1.0]), np.array([0.0]), ZEROS)
        result = parallel_component(*a, *B_ALONG_Z)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_antiparallel_negative(self):
        """A = -B direction → A_∥ = -|A| (sign sanity)."""
        a = (ZEROS, ZEROS, np.array([-4.0]))
        result = parallel_component(*a, *B_ALONG_Z)
        np.testing.assert_allclose(result, -4.0, rtol=1e-14)

    def test_zero_b_gives_nan(self):
        """|B| = 0 → NaN (undefined b̂)."""
        a = (np.array([1.0]), np.array([1.0]), np.array([1.0]))
        result = parallel_component(*a, ZEROS, ZEROS, ZEROS)
        assert np.isnan(result[0])


class TestPerpendicularVector:
    def test_b_along_z_drops_z(self):
        """B along z → A_⊥ = (A_x, A_y, 0)."""
        a = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        a_perp = perpendicular_vector(*a, *B_ALONG_Z)
        np.testing.assert_allclose(a_perp[0], 1.0, rtol=1e-14)
        np.testing.assert_allclose(a_perp[1], 2.0, rtol=1e-14)
        np.testing.assert_allclose(a_perp[2], 0.0, atol=1e-15)

    def test_a_parallel_to_b_gives_zero(self):
        """A ∥ B → A_⊥ = 0 in all components."""
        a = (ZEROS, ZEROS, np.array([7.0]))
        a_perp = perpendicular_vector(*a, *B_ALONG_Z)
        for c in a_perp:
            np.testing.assert_allclose(c, 0.0, atol=1e-15)

    def test_a_perpendicular_to_b_passes_through(self):
        """A ⊥ B → A_⊥ = A (entire vector survives)."""
        a = (np.array([2.5]), np.array([-1.5]), ZEROS)
        a_perp = perpendicular_vector(*a, *B_ALONG_Z)
        for c, expected in zip(a_perp, a, strict=True):
            np.testing.assert_allclose(c, expected, rtol=1e-14)

    def test_zero_b_gives_nan_in_all_components(self):
        """|B| = 0 → NaN in every output component."""
        a = (np.array([1.0]), np.array([1.0]), np.array([1.0]))
        a_perp = perpendicular_vector(*a, ZEROS, ZEROS, ZEROS)
        for c in a_perp:
            assert np.isnan(c[0])

    def test_pythagorean_identity(self):
        """|A|² = A_∥² + |A_⊥|² for arbitrary A, B (structural invariant)."""
        rng = np.random.default_rng(seed=42)
        a1, a2, a3 = rng.normal(size=(3, 100))
        b1, b2, b3 = rng.normal(size=(3, 100))
        a_par = parallel_component(a1, a2, a3, b1, b2, b3)
        a_perp = perpendicular_vector(a1, a2, a3, b1, b2, b3)
        perp_sq = a_perp[0] ** 2 + a_perp[1] ** 2 + a_perp[2] ** 2
        a_sq = a1**2 + a2**2 + a3**2
        np.testing.assert_allclose(a_sq, a_par**2 + perp_sq, rtol=1e-12)


class TestPerpendicularMagnitude:
    def test_matches_explicit_components(self):
        """|A_⊥| from Pythagorean identity matches sqrt of summed squares."""
        rng = np.random.default_rng(seed=7)
        a1, a2, a3 = rng.normal(size=(3, 100))
        b1, b2, b3 = rng.normal(size=(3, 100))
        from_identity = perpendicular_magnitude(a1, a2, a3, b1, b2, b3)
        a_perp = perpendicular_vector(a1, a2, a3, b1, b2, b3)
        from_components = np.sqrt(a_perp[0] ** 2 + a_perp[1] ** 2 + a_perp[2] ** 2)
        np.testing.assert_allclose(from_identity, from_components, rtol=1e-12)

    def test_b_along_z(self):
        """|A_⊥| for A=(1,2,3), B along z → sqrt(5)."""
        a = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        result = perpendicular_magnitude(*a, *B_ALONG_Z)
        np.testing.assert_allclose(result, np.sqrt(5.0), rtol=1e-14)

    def test_zero_b_gives_nan(self):
        """|B| = 0 → NaN."""
        a = (np.array([1.0]), np.array([1.0]), np.array([1.0]))
        result = perpendicular_magnitude(*a, ZEROS, ZEROS, ZEROS)
        assert np.isnan(result[0])


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
        """B along z, diag(3,1,1): Swisdak Q = 1/8 (Eq. A8 hand-calc)."""
        # I_1 = Tr(P) = 5, P_∥ = P_33 = 1.
        # I_2 = 3·1 + 3·1 + 1·1 - 0 - 0 - 0 = 7.
        # (I_1 - P_∥)(I_1 + 3 P_∥) = 4 · 8 = 32.
        # Q = 1 - 4·7 / 32 = 1 - 28/32 = 4/32 = 1/8.
        result = agyrotropy(
            np.array([3.0]),
            np.array([1.0]),
            np.array([1.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 1.0 / 8.0, rtol=1e-14)

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

    def test_off_axis_nongyrotropy_caught(self):
        """Swisdak Q detects off-axis ($\\hat{b}$-coupling) nongyrotropy
        that Scudder's A_phi misses.

        For ``P = [[2,0,1],[0,2,0],[1,0,5]]`` with B = ẑ:

        - The perp 2x2 block is diag(2, 2), so A_phi = 0
          (see ``TestScudderAgyrotropy::test_off_axis_nongyrotropy_missed``).
        - But the full tensor carries off-axis structure via P_13 = 1.
          Hand-calc: I_1 = 9, P_par = 5, I_2 = 2·2 + 2·5 + 2·5 - 0 - 1 - 0 = 23,
          (I_1 - P_par)(I_1 + 3 P_par) = 4·24 = 96, Q = 1 - 92/96 = 1/24.

        Pins the formula against the historical bug where this function
        silently returned A_phi² (which would give zero here).
        """
        p11 = np.array([2.0])
        p22 = np.array([2.0])
        p33 = np.array([5.0])
        p12 = ZEROS
        p13 = np.array([1.0])
        p23 = ZEROS
        result = agyrotropy(p11, p22, p33, p12, p13, p23, *B_ALONG_Z)
        np.testing.assert_allclose(result, 1.0 / 24.0, rtol=1e-14)


class TestAunaiNongyrotropy:
    def test_isotropic_is_zero(self):
        """Isotropic tensor → D_ng = 0 (gyrotropic)."""
        p = np.array([3.0])
        result = aunai_nongyrotropy(p, p, p, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_gyrotropic_diagonal_is_zero(self):
        """Gyrotropic but anisotropic: diag(2,2,5) with B along z → D_ng = 0."""
        result = aunai_nongyrotropy(
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
        """B along z, diag(3,1,1): closed-form D_ng = 2 sqrt(2) / 5."""
        # P_∥ = P_33 = 1, P_⊥ = (3+1)/2 = 2, Tr(P) = 5
        # ||N||_F² = (P_11-P_⊥)² + (P_22-P_⊥)² + 0 + 2*(off-diag squared, all zero)
        # ||N||_F² = (3-2)² + (1-2)² = 1 + 1 = 2
        # D_ng = 2 * sqrt(2) / 5 ≈ 0.5657
        result = aunai_nongyrotropy(
            np.array([3.0]),
            np.array([1.0]),
            np.array([1.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 2.0 * np.sqrt(2.0) / 5.0, rtol=1e-14)

    def test_nan_propagation_b_zero(self):
        """D_ng returns NaN where |B| = 0."""
        result = aunai_nongyrotropy(
            np.array([2.0]),
            np.array([1.0]),
            np.array([3.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
        )
        assert np.isnan(result[0])

    def test_non_negative_random(self):
        """D_ng ≥ 0 for symmetric positive-definite tensors."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            a = rng.standard_normal((3, 3))
            p = a @ a.T + 0.1 * np.eye(3)
            b_vec = rng.standard_normal(3)
            b_vec = b_vec / np.linalg.norm(b_vec)
            d_ng = aunai_nongyrotropy(
                np.array([p[0, 0]]),
                np.array([p[1, 1]]),
                np.array([p[2, 2]]),
                np.array([p[0, 1]]),
                np.array([p[0, 2]]),
                np.array([p[1, 2]]),
                np.array([b_vec[0]]),
                np.array([b_vec[1]]),
                np.array([b_vec[2]]),
            )
            assert d_ng[0] >= -1e-12, f"D_ng = {d_ng[0]} < 0"

    def test_frame_invariance(self):
        """D_ng is built from tensor invariants and P_∥, so it is unchanged
        under any rotation applied jointly to P and B."""
        # P with both perp anisotropy (diag spread) and off-axis P_xz coupling.
        p = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 5.0]])
        b = np.array([0.0, 0.0, 1.0])
        d_ng_lab = aunai_nongyrotropy(*_tensor_b_args(p, b))

        r = _random_rotation(seed=42)
        d_ng_rot = aunai_nongyrotropy(*_tensor_b_args(r @ p @ r.T, r @ b))
        np.testing.assert_allclose(d_ng_rot, d_ng_lab, rtol=1e-13)

    def test_off_axis_nongyrotropy_caught(self):
        """P with off-axis (P_∥-coupling) component but no perp-eigenvalue
        spread: D_ng must detect it even though A_phi misses it.

        For ``P = [[2,0,1],[0,2,0],[1,0,5]]`` with B = ẑ:

        - P_par = P_33 = 5, P_perp = (Tr(P) - P_par)/2 = 2.
        - The non-gyrotropic part is N = P - P_par b̂b̂ - P_perp(I-b̂b̂)
          = [[0,0,1],[0,0,0],[1,0,0]], so ‖N‖_F² = 2.
        - D_ng = 2√2 / Tr(P) = 2√2 / 9 ≈ 0.3143.

        The companion ``A_phi`` test on the same tensor expects zero —
        this pair validates the §9 footnote claim that A_phi misses
        off-axis ($\\hat{b}$-coupling) nongyrotropy that D_ng catches.
        """
        p11 = np.array([2.0])
        p22 = np.array([2.0])
        p33 = np.array([5.0])
        p12 = ZEROS
        p13 = np.array([1.0])
        p23 = ZEROS
        result = aunai_nongyrotropy(p11, p22, p33, p12, p13, p23, *B_ALONG_Z)
        np.testing.assert_allclose(result, 2.0 * np.sqrt(2.0) / 9.0, rtol=1e-14)


class TestScudderAgyrotropy:
    def test_isotropic_is_zero(self):
        """Isotropic tensor → A_phi = 0 (gyrotropic)."""
        p = np.array([3.0])
        result = scudder_agyrotropy(p, p, p, ZEROS, ZEROS, ZEROS, *B_ALONG_Z)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_gyrotropic_diagonal_is_zero(self):
        """Gyrotropic but anisotropic: diag(2,2,5) with B along z → A_phi = 0."""
        # Both perp eigenvalues equal 2 → no perp spread.
        result = scudder_agyrotropy(
            np.array([2.0]),
            np.array([2.0]),
            np.array([5.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_perp_anisotropy_nonzero(self):
        """B along z, diag(3,1,5): perp eigenvalues 3,1 → A_phi = |3-1|/4 = 0.5."""
        result = scudder_agyrotropy(
            np.array([3.0]),
            np.array([1.0]),
            np.array([5.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            *B_ALONG_Z,
        )
        np.testing.assert_allclose(result, 0.5, rtol=1e-14)

    def test_bounded_zero_one(self):
        """A_phi ∈ [0, 1] for random SPD tensors."""
        rng = np.random.default_rng(13)
        for _ in range(100):
            a = rng.standard_normal((3, 3))
            p = a @ a.T + 0.1 * np.eye(3)
            b_vec = rng.standard_normal(3)
            b_vec = b_vec / np.linalg.norm(b_vec)
            a_phi = scudder_agyrotropy(
                np.array([p[0, 0]]),
                np.array([p[1, 1]]),
                np.array([p[2, 2]]),
                np.array([p[0, 1]]),
                np.array([p[0, 2]]),
                np.array([p[1, 2]]),
                np.array([b_vec[0]]),
                np.array([b_vec[1]]),
                np.array([b_vec[2]]),
            )
            assert a_phi[0] >= -1e-10, f"A_phi = {a_phi[0]} < 0"
            assert a_phi[0] <= 1.0 + 1e-10, f"A_phi = {a_phi[0]} > 1"

    def test_nan_propagation_b_zero(self):
        """A_phi returns NaN where |B| = 0."""
        result = scudder_agyrotropy(
            np.array([3.0]),
            np.array([1.0]),
            np.array([2.0]),
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
            ZEROS,
        )
        assert np.isnan(result[0])

    def test_frame_invariance(self):
        """A_phi depends only on tensor invariants of the perp 2x2 block,
        so a rotation applied jointly to P and B leaves it unchanged."""
        # diag(3, 1, 5) with B = ẑ has known A_phi = 0.5 (existing test).
        p = np.diag([3.0, 1.0, 5.0])
        b = np.array([0.0, 0.0, 1.0])
        a_phi_lab = scudder_agyrotropy(*_tensor_b_args(p, b))
        np.testing.assert_allclose(a_phi_lab, 0.5, rtol=1e-14)

        r = _random_rotation(seed=42)
        a_phi_rot = scudder_agyrotropy(*_tensor_b_args(r @ p @ r.T, r @ b))
        np.testing.assert_allclose(a_phi_rot, a_phi_lab, rtol=1e-13)

    def test_off_axis_nongyrotropy_missed(self):
        """A_phi misses off-axis ($\\hat{b}$-coupling) nongyrotropy.

        ``P = [[2,0,1],[0,2,0],[1,0,5]]`` with B = ẑ has:

        - Equal perp eigenvalues (the perp 2x2 block is diag(2, 2)),
          so A_phi must be exactly zero.
        - But the off-diagonal P_xz = 1 makes the tensor non-gyrotropic.
          Both ``aunai_nongyrotropy`` (D_ng = 2√2/9) and ``agyrotropy``
          (Swisdak Q = 1/24) flag it — see the companion tests
          ``TestAunaiNongyrotropy::test_off_axis_nongyrotropy_caught``
          and ``TestAgyrotropy::test_off_axis_nongyrotropy_caught``.

        Together these three tests validate the §9 footnote claim that
        A_phi captures only perp anisotropy and misses the
        $\\hat{b}$-coupling that the two full-tensor measures catch.
        """
        p11 = np.array([2.0])
        p22 = np.array([2.0])
        p33 = np.array([5.0])
        p12 = ZEROS
        p13 = np.array([1.0])
        p23 = ZEROS
        result = scudder_agyrotropy(p11, p22, p33, p12, p13, p23, *B_ALONG_Z)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)


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


class TestJDotE:
    def test_hand_calculation(self):
        """J=(1,2,0), E=(3,0,1) → J·E = 1*3 + 2*0 + 0*1 = 3."""
        result = j_dot_e(
            np.array([1.0]),
            np.array([2.0]),
            np.array([0.0]),
            np.array([3.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(result, 3.0, rtol=1e-15)

    def test_orthogonal_gives_zero(self):
        """J perpendicular to E → J·E = 0."""
        result = j_dot_e(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_negative_means_fields_gain_energy(self):
        result = j_dot_e(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([-2.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        assert result[0] < 0

    def test_nan_propagation(self):
        result = j_dot_e(
            np.array([np.nan]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        assert np.isnan(result[0])


class TestElectronFrameDissipation:
    """Zenitani 2011 electron-frame dissipation D_e."""

    def test_ideal_mhd_is_zero(self):
        """E = -V_e × B, rho_c = 0 → D_e = J · 0 - 0 = 0 for any J."""
        ve = (np.array([1.5]), np.array([-0.3]), np.array([0.2]))
        b = (np.array([0.1]), np.array([0.4]), np.array([0.9]))
        e = (
            -(ve[1] * b[2] - ve[2] * b[1]),
            -(ve[2] * b[0] - ve[0] * b[2]),
            -(ve[0] * b[1] - ve[1] * b[0]),
        )
        j = (np.array([0.5]), np.array([-0.2]), np.array([0.7]))
        rho_c = np.array([0.0])
        result = electron_frame_dissipation(*j, *e, *ve, *b, rho_c)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_zero_velocity_recovers_j_dot_e(self):
        """V_e = 0, rho_c = 0 → D_e = J · E."""
        ve = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, ONES)
        j = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        e = (np.array([4.0]), np.array([5.0]), np.array([6.0]))
        rho_c = np.array([0.0])
        result = electron_frame_dissipation(*j, *e, *ve, *b, rho_c)
        np.testing.assert_allclose(result, 1 * 4 + 2 * 5 + 3 * 6, rtol=1e-15)

    def test_rho_c_subtraction(self):
        """V_e = 0 cross product still zero, but rho_c * V_e · E nonzero."""
        ve = (np.array([1.0]), np.array([0.0]), np.array([0.0]))
        b = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        j = (ZEROS, ZEROS, ZEROS)
        e = (np.array([2.0]), np.array([0.0]), np.array([0.0]))
        rho_c = np.array([3.0])
        # E + V_e × B = E.  J · E' = 0 (J = 0).
        # D_e = 0 - rho_c * V_e · E = -3 * 1 * 2 = -6.
        result = electron_frame_dissipation(*j, *e, *ve, *b, rho_c)
        np.testing.assert_allclose(result, -6.0, rtol=1e-15)

    def test_relativistic_gamma_prefactor(self):
        """c provided → multiplies by γ_e = 1/sqrt(1 - V_e²/c²)."""
        ve = (np.array([0.6]), ZEROS, ZEROS)  # V_e = 0.6 c if c = 1
        b = (ZEROS, ZEROS, ZEROS)
        j = (np.array([1.0]), ZEROS, ZEROS)
        e = (np.array([2.0]), ZEROS, ZEROS)
        rho_c = np.array([0.0])
        # E + V_e × B = E.  D_e_NR = J · E = 2.
        # γ_e = 1/sqrt(1 - 0.36) = 1/0.8 = 1.25.
        result = electron_frame_dissipation(*j, *e, *ve, *b, rho_c, c=1.0)
        np.testing.assert_allclose(result, 1.25 * 2.0, rtol=1e-14)

    def test_nan_propagation(self):
        """NaN in any input propagates."""
        nan = np.array([np.nan])
        ve = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, ONES)
        j = (nan, ZEROS, ZEROS)
        e = (ONES, ZEROS, ZEROS)
        rho_c = np.array([0.0])
        result = electron_frame_dissipation(*j, *e, *ve, *b, rho_c)
        assert np.isnan(result[0])

    def test_non_relativistic_limit(self):
        """As $v_e/c \\to 0$ the relativistic branch must converge to the
        non-relativistic result.  Verifies the γ_e prefactor reduces to
        the identity smoothly — guards against a sign or normalization
        slip in the c-aware code path."""
        # Slow electrons at v_e/c = 1e-6, so gamma_e - 1 ~ 5e-13.
        ve = (np.array([1.0e-6]), ZEROS, ZEROS)
        b = (ZEROS, ZEROS, np.array([0.3]))  # arbitrary non-zero B
        j = (np.array([0.4]), np.array([-0.5]), np.array([0.2]))
        e = (np.array([0.1]), np.array([0.7]), np.array([-0.3]))
        rho_c = np.array([0.6])  # exercises the rho_c · V_e · E subtraction
        d_e_nonrel = electron_frame_dissipation(*j, *e, *ve, *b, rho_c)
        d_e_rel = electron_frame_dissipation(*j, *e, *ve, *b, rho_c, c=1.0)
        np.testing.assert_allclose(d_e_rel, d_e_nonrel, rtol=1e-10)


class TestLocalReconnectionRate:
    """Comisso & Bhattacharjee normalized local rate |E'|/(v_A |B|)."""

    def test_ideal_mhd_is_zero(self):
        """E = -V × B → R_recon = 0."""
        v = (np.array([0.7]), np.array([-0.1]), np.array([0.3]))
        b = (np.array([0.2]), np.array([0.5]), np.array([0.8]))
        e = (
            -(v[1] * b[2] - v[2] * b[1]),
            -(v[2] * b[0] - v[0] * b[2]),
            -(v[0] * b[1] - v[1] * b[0]),
        )
        v_a = np.array([1.5])
        result = local_reconnection_rate(*e, *v, *b, v_a)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_pure_parallel_e_field(self):
        """V = 0, E parallel to B → R_recon = |E| / (v_A |B|)."""
        v = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, np.array([2.0]))
        e = (ZEROS, ZEROS, np.array([3.0]))
        v_a = np.array([4.0])
        # |E + 0| = 3; |B| = 2; R = 3 / (4 * 2) = 0.375.
        result = local_reconnection_rate(*e, *v, *b, v_a)
        np.testing.assert_allclose(result, 0.375, rtol=1e-15)

    def test_nan_at_zero_b(self):
        """|B| = 0 → NaN."""
        v = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, ZEROS)
        e = (np.array([1.0]), ZEROS, ZEROS)
        v_a = np.array([1.0])
        result = local_reconnection_rate(*e, *v, *b, v_a)
        assert np.isnan(result[0])

    def test_nan_at_zero_va(self):
        """v_A = 0 → NaN."""
        v = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, ONES)
        e = (np.array([1.0]), ZEROS, ZEROS)
        v_a = np.array([0.0])
        result = local_reconnection_rate(*e, *v, *b, v_a)
        assert np.isnan(result[0])

    def test_nan_propagation(self):
        """NaN in any input propagates."""
        nan = np.array([np.nan])
        v = (ZEROS, ZEROS, ZEROS)
        b = (ZEROS, ZEROS, ONES)
        e = (nan, ZEROS, ZEROS)
        v_a = ONES
        result = local_reconnection_rate(*e, *v, *b, v_a)
        assert np.isnan(result[0])


class TestIdealElectricField:
    def test_hand_calculation(self):
        """V=(1,0,0), B=(0,0,1) -> -VxB = -(0,-1,0) = (0,1,0)."""
        e1, e2, e3 = ideal_electric_field(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(e1, 0.0, atol=1e-15)
        np.testing.assert_allclose(e2, 1.0, rtol=1e-15)
        np.testing.assert_allclose(e3, 0.0, atol=1e-15)

    def test_zero_velocity_gives_zero(self):
        e1, e2, e3 = ideal_electric_field(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        )
        np.testing.assert_allclose(e1, 0.0, atol=1e-15)
        np.testing.assert_allclose(e2, 0.0, atol=1e-15)
        np.testing.assert_allclose(e3, 0.0, atol=1e-15)


class TestNonIdealElectricField:
    def test_ideal_case_gives_zero(self):
        """When E = -VxB exactly, E' = E + VxB = 0 (frozen-in)."""
        # V=(1,0,0), B=(0,1,0), VxB = (0,0,1), E_ideal = (0,0,-1)
        e1, e2, e3 = non_ideal_electric_field(
            np.array([0.0]),
            np.array([0.0]),
            np.array([-1.0]),  # E = -VxB
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),  # V
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),  # B
        )
        np.testing.assert_allclose(e1, 0.0, atol=1e-15)
        np.testing.assert_allclose(e2, 0.0, atol=1e-15)
        np.testing.assert_allclose(e3, 0.0, atol=1e-15)

    def test_hand_calculation(self):
        """E=(0,0,0.5), V=(1,0,0), B=(0,1,0), VxB=(0,0,1), E'=(0,0,1.5)."""
        _e1, _e2, e3 = non_ideal_electric_field(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.5]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(e3, 1.5, rtol=1e-15)

    def test_consistency_with_ideal(self):
        """E' = E - E_ideal (by definition)."""
        v = np.array([1.0, 0.5, -0.3])
        b = np.array([0.2, -0.4, 0.8])
        e = np.array([0.1, 0.2, 0.3])
        e_id1, e_id2, e_id3 = ideal_electric_field(
            v[:1],
            v[1:2],
            v[2:],
            b[:1],
            b[1:2],
            b[2:],
        )
        ep1, ep2, ep3 = non_ideal_electric_field(
            e[:1],
            e[1:2],
            e[2:],
            v[:1],
            v[1:2],
            v[2:],
            b[:1],
            b[1:2],
            b[2:],
        )
        np.testing.assert_allclose(ep1, e[:1] - e_id1, rtol=1e-14)
        np.testing.assert_allclose(ep2, e[1:2] - e_id2, rtol=1e-14)
        np.testing.assert_allclose(ep3, e[2:] - e_id3, rtol=1e-14)


class TestHallElectricField:
    def test_hand_calculation(self):
        """J=(1,0,0), B=(0,0,1), n=2, |q|=1, JxB=(0,-1,0), E_Hall=(0,-0.5,0)."""
        e1, e2, e3 = hall_electric_field(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([2.0]),
            1.0,
        )
        np.testing.assert_allclose(e1, 0.0, atol=1e-15)
        np.testing.assert_allclose(e2, -0.5, rtol=1e-15)
        np.testing.assert_allclose(e3, 0.0, atol=1e-15)

    def test_charge_sign_independence(self):
        """Result is the same for positive and negative charge (abs used)."""
        args = (
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
        pos = hall_electric_field(*args, 1.0)
        neg = hall_electric_field(*args, -1.0)
        for p, n in zip(pos, neg, strict=True):
            np.testing.assert_allclose(p, n, rtol=1e-15)

    def test_zero_density_gives_nan(self):
        _e1, e2, _e3 = hall_electric_field(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            1.0,
        )
        assert np.isnan(e2[0])


class TestFirehoseParameter:
    def test_isotropic_is_stable(self):
        """Equal pressures: F = (P-P)/(B²/2) - 1 = -1."""
        result = firehose_parameter(
            np.array([2.0]),
            np.array([2.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(result, -1.0, rtol=1e-15)

    def test_marginal_stability(self):
        """P_par - P_perp = B²/2 → F = 0."""
        # B=2 → B²/2 = 2, so P_par - P_perp = 2
        result = firehose_parameter(
            np.array([3.0]),
            np.array([1.0]),
            np.array([2.0]),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_unstable(self):
        """Large parallel excess: F = (10-1)/(1²/2) - 1 = 18 - 1 = 17."""
        result = firehose_parameter(
            np.array([10.0]),
            np.array([1.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(result, 17.0, rtol=1e-15)

    def test_zero_b_gives_nan(self):
        result = firehose_parameter(
            np.array([1.0]),
            np.array([0.5]),
            np.array([0.0]),
        )
        assert np.isnan(result[0])


class TestMirrorParameter:
    def test_isotropic_is_stable(self):
        """P_par = P_perp: M = 1 - 1 - 1/β_perp = -1/β_perp < 0."""
        # P=1, B=1 → β_perp = 2*1/1 = 2, M = 1 - 1 - 0.5 = -0.5
        result = mirror_parameter(
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(result, -0.5, rtol=1e-14)

    def test_zero_b_gives_nan(self):
        result = mirror_parameter(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
        )
        assert np.isnan(result[0])

    def test_large_beta_limit(self):
        """When β_perp → ∞, M → P_perp/P_par - 1."""
        # B very small → β_perp very large → 1/β_perp ≈ 0
        result = mirror_parameter(
            np.array([1.0]),
            np.array([3.0]),
            np.array([1e-6]),
        )
        np.testing.assert_allclose(result, 2.0, atol=0.01)


class TestMagneticShearAngle:
    def test_parallel(self):
        angle = magnetic_shear_angle(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(angle, 0.0, atol=1e-15)

    def test_antiparallel(self):
        angle = magnetic_shear_angle(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([-1.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(angle, np.pi, rtol=1e-14)

    def test_perpendicular(self):
        angle = magnetic_shear_angle(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(angle, np.pi / 2, rtol=1e-14)

    def test_45_degrees(self):
        angle = magnetic_shear_angle(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(angle, np.pi / 4, rtol=1e-14)


class TestMagneticFluxFunction:
    def test_uniform_field(self):
        """Uniform B_2=1: ψ = -∫B_2 dx = -x*dx (negative cumsum)."""
        b2 = np.ones((10, 5))
        dx = 0.5
        psi = magnetic_flux_function(b2, dx, 1.0)
        # First row: psi[0,:] = -1*0.5 = -0.5
        np.testing.assert_allclose(psi[0, :], -0.5, rtol=1e-15)
        # Should decrease along x
        assert psi[-1, 0] < psi[0, 0]

    def test_sign_convention(self):
        r"""Verify B_2 = -∂ψ/∂x (standard convention)."""
        nx, ny = 64, 32
        dx = 0.1
        b2 = np.ones((nx, ny)) * 2.0
        psi = magnetic_flux_function(b2, dx, 1.0)
        # Reconstruct: -∂ψ/∂x should ≈ B_2
        dpsi_dx = np.gradient(psi, dx, axis=0)
        np.testing.assert_allclose(-dpsi_dx[2:-2, :], 2.0, rtol=0.01)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            magnetic_flux_function(np.ones((4, 3, 2)), 1.0, 1.0)

    @staticmethod
    def _flux_max_error(nx: int) -> float:
        r"""Max-norm error of $\psi$ for $B_2(x) = \cos(k x)$ vs analytic.

        Analytic flux: $\psi(x) = -\int_0^x \cos(k x')\,dx' = -\sin(k x)/k$.
        """
        length = 2.0 * np.pi
        dx = length / nx
        ny = 4
        k = 1.0
        x = np.arange(nx) * dx
        b2 = np.broadcast_to(np.cos(k * x)[:, None], (nx, ny))
        psi_num = magnetic_flux_function(np.ascontiguousarray(b2), dx, 1.0)
        # Interior only to skip endpoint bias.
        psi_exact = -np.sin(k * x) / k
        return float(np.max(np.abs(psi_num[2:-2, :] - psi_exact[2:-2, None])))

    def test_convergence_is_first_order(self) -> None:
        r"""``magnetic_flux_function`` uses ``np.cumsum * dx`` (left-rect rule).

        Left-rectangle quadrature on $\int_0^x f\,dx'$ is first-order:
        halving $dx$ should reduce the max error by ~2, not the ~4 of a
        second-order scheme. Documenting the actual order — a future
        upgrade to a trapezoidal cumsum would let this test relax to a
        second-order assertion.
        """
        e_coarse = self._flux_max_error(64)
        e_fine = self._flux_max_error(128)
        ratio = e_coarse / e_fine
        # First-order observed ratio ≈ 2.0; assert > 1.8 to catch
        # accidental regressions to zeroth order while tolerating
        # pre-asymptotic sag.
        assert 1.8 < ratio < 2.5, (
            f"magnetic_flux_function convergence ratio {ratio:.3f} "
            f"outside the first-order band [1.8, 2.5]; "
            f"coarse={e_coarse:.4g}, fine={e_fine:.4g}"
        )


# ---------------------------------------------------------------------------
# Unit 12 — edge-case sweep across families that previous tests skipped
# ---------------------------------------------------------------------------
#
# Each test below is one structural invariant ("every callable in this list
# satisfies property P"). Failures are aggregated into a descriptive
# message rather than fanned out across N parametrized cases — the
# project's CLAUDE.md guidance prefers aggregation for invariant checks.
#
# The lists are kept in this section so adding a new derived function only
# requires one append, not edits across multiple test classes.

# (label, callable_taking_one_array) — magnitudes and 1-arg energy/scale funcs
_FAMILY_1ARG: list[tuple[str, Callable[..., Any]]] = [
    ("magnetic_field_magnitude", lambda a: magnetic_field_magnitude(a, a, a)),
    ("electric_field_magnitude", lambda a: electric_field_magnitude(a, a, a)),
    ("current_density_magnitude", lambda a: current_density_magnitude(a, a, a)),
    ("velocity_magnitude", lambda a: velocity_magnitude(a, a, a)),
    ("magnetic_energy_density", magnetic_energy_density),
    ("electric_energy_density", electric_energy_density),
    ("thermal_energy_density", thermal_energy_density),
]

# (label, callable_taking_two_arrays)
_FAMILY_2ARG: list[tuple[str, Callable[..., Any]]] = [
    ("plasma_beta", plasma_beta),
    ("alfven_speed", alfven_speed),
    ("kinetic_energy_density", kinetic_energy_density),
]

# (label, callable_taking_(rho_array,)) — density-dependent funcs that
# should return NaN (not raise) for negative density input.
_DENSITY_DEPENDENT: list[tuple[str, Callable[..., Any]]] = [
    ("alfven_speed", lambda n: alfven_speed(np.array([1.0]), n)),
    ("plasma_frequency", lambda n: plasma_frequency(n, charge=1.0, mass=1.0)),
    ("skin_depth", lambda n: skin_depth(n, charge=1.0, mass=1.0)),
    ("debye_length", lambda n: debye_length(np.array([1.0]), n, charge=1.0)),
    ("sound_speed", lambda n: sound_speed(np.array([1.0]), n)),
]

# (label, q+ callable, q- callable) — charge-dependent funcs whose
# physical magnitudes must be invariant under charge sign flip.
# Includes skin_depth which the existing per-class tests miss.
_CHARGE_INVARIANT: list[tuple[str, Callable[..., Any], Callable[..., Any]]] = [
    (
        "gyrofrequency",
        lambda: gyrofrequency(np.array([2.0]), charge=1.0, mass=1.5),
        lambda: gyrofrequency(np.array([2.0]), charge=-1.0, mass=1.5),
    ),
    (
        "plasma_frequency",
        lambda: plasma_frequency(np.array([2.0]), charge=1.0, mass=1.5),
        lambda: plasma_frequency(np.array([2.0]), charge=-1.0, mass=1.5),
    ),
    (
        "skin_depth",
        lambda: skin_depth(np.array([2.0]), charge=1.0, mass=1.5, c=1.0),
        lambda: skin_depth(np.array([2.0]), charge=-1.0, mass=1.5, c=1.0),
    ),
    (
        "gyroradius",
        lambda: gyroradius(np.array([2.0]), np.array([3.0]), charge=1.0, mass=1.5),
        lambda: gyroradius(np.array([2.0]), np.array([3.0]), charge=-1.0, mass=1.5),
    ),
    (
        "debye_length",
        lambda: debye_length(np.array([2.0]), np.array([3.0]), charge=1.0),
        lambda: debye_length(np.array([2.0]), np.array([3.0]), charge=-1.0),
    ),
]


class TestEdgeCaseInvariants:
    """Aggregated edge-case sweep — Unit 12 of the cleanup."""

    def test_empty_arrays_propagate(self) -> None:
        """Empty input → empty output, preserving shape (0,)."""
        empty = np.array([], dtype=np.float64)
        failures: list[str] = []
        for label, func in _FAMILY_1ARG:
            try:
                result = func(empty)
            except Exception as e:
                failures.append(f"{label}: raised {type(e).__name__}: {e}")
                continue
            if result.shape != (0,):
                failures.append(f"{label}: shape={result.shape}, expected (0,)")
        for label, func in _FAMILY_2ARG:
            try:
                result = func(empty, empty)
            except Exception as e:
                failures.append(f"{label}: raised {type(e).__name__}: {e}")
                continue
            if result.shape != (0,):
                failures.append(f"{label}: shape={result.shape}, expected (0,)")
        assert not failures, "Empty-array regressions:\n  - " + "\n  - ".join(failures)

    def test_single_element_preserves_shape(self) -> None:
        """Length-1 input → length-1 output (no scalar special-casing)."""
        one = np.array([1.0])
        failures: list[str] = []
        for label, func in _FAMILY_1ARG:
            result = func(one)
            if result.shape != (1,):
                failures.append(f"{label}: shape={result.shape}, expected (1,)")
        for label, func in _FAMILY_2ARG:
            result = func(one, one)
            if result.shape != (1,):
                failures.append(f"{label}: shape={result.shape}, expected (1,)")
        assert not failures, "Single-element shape regressions:\n  - " + "\n  - ".join(
            failures
        )

    def test_negative_density_returns_nan(self) -> None:
        """Negative density → NaN, never an exception.

        Many derived quantities take ``sqrt(rho)`` or ``1/rho``. They must
        propagate NaN for invalid input rather than raising — callers
        often pass entire fields where a few cells may be unphysical.
        The numpy ``invalid value`` warnings from ``sqrt(<0)`` are
        expected and silenced via ``np.errstate``.
        """
        neg = np.array([-1.0])
        failures: list[str] = []
        with np.errstate(invalid="ignore"):
            for label, func in _DENSITY_DEPENDENT:
                try:
                    result = func(neg)
                except Exception as e:
                    failures.append(f"{label}: raised {type(e).__name__}: {e}")
                    continue
                if not np.isnan(result[0]):
                    failures.append(f"{label}: returned {result[0]!r}, expected NaN")
        assert not failures, "Negative-density NaN regressions:\n  - " + "\n  - ".join(
            failures
        )

    def test_charge_sign_invariance(self) -> None:
        r"""Magnitudes (gyrofreq, plasma freq, skin depth, gyroradius, $\lambda_D$)
        depend on $|q|$ or $q^2$ — flipping the sign must not change them.

        Catches accidental refactors that drop ``abs()`` or replace ``q**2``
        with ``q``. Includes ``skin_depth`` which previously had no
        dedicated charge-sign test.
        """
        failures: list[str] = []
        for label, pos_call, neg_call in _CHARGE_INVARIANT:
            pos = pos_call()
            neg = neg_call()
            if not np.allclose(pos, neg, rtol=1e-15):
                failures.append(f"{label}: q=+1 → {pos}, q=-1 → {neg}")
        assert not failures, (
            "Charge sign invariance regressions:\n  - " + "\n  - ".join(failures)
        )


class TestDebyeLengthScaling:
    r"""Functional-form regression tests for $\lambda_D = \sqrt{T / (n q^2)}$.

    These complement the existing known-value test (``T=4, n=1, q=1 → 2``)
    by checking the *dependence* on each input. Hand-coded values can
    survive a refactor that swaps the formula for one with the same
    output at a single point; scaling laws cannot.
    """

    @pytest.mark.parametrize(
        ("t_factor", "expected_factor"),
        [
            (1.0, 1.0),
            (2.0, np.sqrt(2.0)),
            (4.0, 2.0),
            (0.25, 0.5),
        ],
    )
    def test_temperature_scaling(
        self,
        t_factor: float,
        expected_factor: float,
    ) -> None:
        r"""$\lambda_D \propto \sqrt{T}$."""
        base = debye_length(np.array([1.0]), np.array([1.0]), charge=1.0)
        scaled = debye_length(
            np.array([t_factor]),
            np.array([1.0]),
            charge=1.0,
        )
        np.testing.assert_allclose(
            scaled,
            expected_factor * base,
            rtol=1e-15,
        )

    @pytest.mark.parametrize(
        ("n_factor", "expected_factor"),
        [
            (1.0, 1.0),
            (4.0, 0.5),
            (0.25, 2.0),
            (16.0, 0.25),
        ],
    )
    def test_density_scaling(
        self,
        n_factor: float,
        expected_factor: float,
    ) -> None:
        r"""$\lambda_D \propto 1/\sqrt{n}$."""
        base = debye_length(np.array([1.0]), np.array([1.0]), charge=1.0)
        scaled = debye_length(
            np.array([1.0]),
            np.array([n_factor]),
            charge=1.0,
        )
        np.testing.assert_allclose(
            scaled,
            expected_factor * base,
            rtol=1e-15,
        )


class TestLorentzFactor:
    def test_stationary(self):
        np.testing.assert_allclose(
            lorentz_factor(np.array([0.0]), c=1.0), 1.0, rtol=1e-15
        )

    def test_known_value(self):
        # v = 0.6c → γ = 1/√(1 - 0.36) = 1/√0.64 = 1/0.8 = 1.25
        np.testing.assert_allclose(
            lorentz_factor(np.array([0.6]), c=1.0), 1.25, rtol=1e-15
        )

    def test_nonunit_c(self):
        # v = 6, c = 10 → v/c = 0.6 → γ = 1.25
        np.testing.assert_allclose(
            lorentz_factor(np.array([6.0]), c=10.0), 1.25, rtol=1e-15
        )

    def test_four_velocity_stationary(self):
        np.testing.assert_allclose(
            lorentz_factor_from_four_velocity(np.array([0.0]), c=1.0),
            1.0,
            rtol=1e-15,
        )

    def test_three_and_four_velocity_agree(self):
        v = np.array([0.6, 0.8, 0.9, 0.99])
        gamma_from_v = lorentz_factor(v, c=1.0)
        u = gamma_from_v * v  # four-velocity = γv
        gamma_from_u = lorentz_factor_from_four_velocity(u, c=1.0)
        np.testing.assert_allclose(gamma_from_v, gamma_from_u, rtol=1e-14)


class TestMagnetization:
    def test_zero_field(self):
        np.testing.assert_allclose(
            magnetization(np.array([0.0]), np.array([1.0]), c=1.0),
            0.0,
            atol=1e-15,
        )

    def test_unit_sigma(self):
        # B² = ρ c² → σ = 1
        np.testing.assert_allclose(
            magnetization(np.array([1.0]), np.array([1.0]), c=1.0),
            1.0,
            rtol=1e-15,
        )

    def test_nonunit_c(self):
        # B=2, ρ=1, c=2 → σ = 4/(1·4) = 1
        np.testing.assert_allclose(
            magnetization(np.array([2.0]), np.array([1.0]), c=2.0),
            1.0,
            rtol=1e-15,
        )


class TestRelativisticLimits:
    """Non-relativistic limit (v ≪ c) recovers classical formulas."""

    def test_kinetic_energy_density_nonrel_limit(self):
        # For v ≪ c: (γ-1)ρc² = ½ρv² + O(v⁴/c²). Leading correction is
        # 3ρv⁴/(8c²); at v=1e-3, c=1 → rel err ≈ 7.5e-7. rtol=1e-6 gives
        # ~1.3× margin over the Taylor residual.
        rho = np.array([2.0])
        v = np.array([0.001])  # v ≪ c
        c = 1.0
        gamma = lorentz_factor(v, c)
        rel = kinetic_energy_density(rho, v, lorentz_factor=gamma, c=c)
        nonrel = kinetic_energy_density(rho, v)
        np.testing.assert_allclose(rel, nonrel, rtol=1e-6)

    def test_alfven_speed_nonrel_limit(self):
        # σ = B²/(ρc²) ≪ 1: v_A_rel/v_A_nonrel = √(1/(1+σ)) ≈ 1 - σ/2.
        # With σ = 1e-4, residual ≈ 5e-5. rtol=1e-4 gives 2× margin.
        b = np.array([0.01])
        rho = np.array([1.0])
        rel = alfven_speed(b, rho, c=1.0)
        nonrel = alfven_speed(b, rho)
        np.testing.assert_allclose(rel, nonrel, rtol=1e-4)

    def test_alfven_speed_approaches_c(self):
        # σ → ∞: v_A → c
        b = np.array([1e6])
        rho = np.array([1.0])
        c = 1.0
        v_a = alfven_speed(b, rho, c=c)
        np.testing.assert_allclose(v_a, c, rtol=1e-6)

    def test_magnetosonic_less_than_c(self):
        # Relativistic v_ms is always < c
        v_a = np.array([0.8])
        c_s = np.array([0.7])
        c = 1.0
        v_ms = magnetosonic_speed(v_a, c_s, c=c)
        assert float(v_ms[0]) < c

    def test_magnetosonic_nonrel_limit(self):
        # Relativistic v_ms² = v_A² + c_s² - v_A²c_s²/c². With
        # v_A = c_s = 1e-3, c=1, the coupling term is 1e-12 vs 2e-6 sum →
        # rel err ≈ 2.5e-7. rtol=1e-6 gives 4× margin.
        v_a = np.array([0.001])
        c_s = np.array([0.001])
        rel = magnetosonic_speed(v_a, c_s, c=1.0)
        nonrel = magnetosonic_speed(v_a, c_s)
        np.testing.assert_allclose(rel, nonrel, rtol=1e-6)

    def test_sound_speed_nonrel_limit(self):
        # Rel. c_s = √(γP/(ρ h_rel/c²)). With h_rel ≈ c² + γP/((γ-1)ρ),
        # the correction is γP/((γ-1)ρc²) at leading order. At P=1e-6,
        # ρ=1, c=1, γ=5/3 → residual ≈ 1.25e-6. rtol=1e-5 gives 8× margin.
        p = np.array([1e-6])
        rho = np.array([1.0])
        rel = sound_speed(p, rho, c=1.0)
        nonrel = sound_speed(p, rho)
        np.testing.assert_allclose(rel, nonrel, rtol=1e-5)

    def test_gyrofrequency_with_lorentz_factor(self):
        b = np.array([2.0])
        gamma = np.array([2.0])
        omega = gyrofrequency(b, charge=1.0, mass=1.0)
        omega_rel = gyrofrequency(b, charge=1.0, mass=1.0, lorentz_factor=gamma)
        np.testing.assert_allclose(omega_rel, omega / 2.0, rtol=1e-15)

    def test_plasma_frequency_with_lorentz_factor(self):
        n = np.array([4.0])
        gamma = np.array([4.0])
        omega = plasma_frequency(n, charge=1.0, mass=1.0)
        omega_rel = plasma_frequency(n, charge=1.0, mass=1.0, lorentz_factor=gamma)
        # ω_rel = ω / √γ = ω / 2
        np.testing.assert_allclose(omega_rel, omega / 2.0, rtol=1e-15)

    def test_thermal_speed_capped_at_c(self):
        # Very hot plasma: v_th → c
        t_hot = np.array([1e10])
        v_rel = thermal_speed(t_hot, mass=1.0, c=1.0)
        assert float(v_rel[0]) < 1.0

    def test_thermal_speed_nonrel_limit(self):
        # v_th_rel = v_th / √(1 + v_th²/c²). With T=1e-6, m=1, c=1 →
        # v_th = 1e-3, residual ≈ v_th²/(2c²) = 5e-7. rtol=1e-6 gives
        # 2× margin over the Taylor truncation.
        t_cold = np.array([1e-6])
        rel = thermal_speed(t_cold, mass=1.0, c=1.0)
        nonrel = thermal_speed(t_cold, mass=1.0)
        np.testing.assert_allclose(rel, nonrel, rtol=1e-6)

    def test_gyroradius_with_lorentz_factor(self):
        t = np.array([1.0])
        b = np.array([1.0])
        gamma = np.array([3.0])
        r = gyroradius(t, b, charge=1.0, mass=1.0)
        r_rel = gyroradius(t, b, charge=1.0, mass=1.0, lorentz_factor=gamma)
        np.testing.assert_allclose(r_rel, 3.0 * r, rtol=1e-15)

    def test_skin_depth_with_lorentz_factor(self):
        n = np.array([1.0])
        gamma = np.array([4.0])
        d = skin_depth(n, charge=1.0, mass=1.0)
        d_rel = skin_depth(n, charge=1.0, mass=1.0, lorentz_factor=gamma)
        # d_rel = c / (ω_p/√γ) = d·√γ = d·2
        np.testing.assert_allclose(d_rel, 2.0 * d, rtol=1e-15)
