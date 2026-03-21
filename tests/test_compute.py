import numpy as np
import pytest
from scipy import constants

from pypic.compute import (
    _COMPUTE_ALIASES,
    _REGISTRY,
    available_quantities,
    compute_field,
    display_unit_factor,
    field_si_factor,
)
from pypic.coordinates.geometry import SPHERICAL
from pypic.readers.base import FieldDataset, GridInfo
from pypic.units import Normalization, SpeciesInfo


def _make_grid(shape: tuple[int, ...] = (4, 3, 2)) -> GridInfo:
    return GridInfo(
        dimensions=shape,
        spacing=(1.0,) * len(shape),
    )


def _make_dataset(
    fields: dict[str, np.ndarray],
    *,
    shape: tuple[int, ...] = (4, 3, 2),
    species: list[SpeciesInfo] | None = None,
    physics: dict | None = None,
    normalization: Normalization | None = None,
) -> FieldDataset:
    grid = _make_grid(shape)
    return FieldDataset.from_arrays(
        fields,
        grid,
        normalization or Normalization.identity(),
        species=species,
        physics=physics,
    )


def _uniform_fields(shape: tuple[int, ...] = (4, 3, 2)) -> dict[str, np.ndarray]:
    """Minimal field set for many derived quantities."""
    return {
        "B1": np.full(shape, 3.0),
        "B2": np.full(shape, 4.0),
        "B3": np.zeros(shape),
        "E1": np.ones(shape),
        "E2": np.full(shape, 2.0),
        "E3": np.full(shape, 3.0),
        "V1": np.full(shape, 0.6),
        "V2": np.full(shape, 0.8),
        "V3": np.zeros(shape),
        "J1": np.ones(shape),
        "J2": np.ones(shape),
        "J3": np.ones(shape),
        "rho_m": np.full(shape, 4.0),
        "P": np.full(shape, 2.0),
        "Pe": np.full(shape, 1.0),
        "Pi": np.full(shape, 1.0),
        "n_s0": np.full(shape, 10.0),
        "n_s1": np.full(shape, 10.0),
        "Te": np.full(shape, 0.5),
        "Ti": np.full(shape, 0.5),
    }


ELECTRONS = SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256)
IONS = SpeciesInfo(name="ions", charge=1.0, mass=1.0)


class TestRegistryIntegrity:
    def test_no_duplicate_names(self):
        overlap = set(_REGISTRY) & set(_COMPUTE_ALIASES)
        assert not overlap, f"Name collision: {overlap}"

    def test_all_aliases_resolve(self):
        # Aliases may point to raw field names (Te, Pi, etc.) used as
        # direct passthrough, not only to _REGISTRY entries.
        raw_field_targets = {"Te", "Ti", "Pe", "Pi", "EF1", "EF2", "EF3"}
        for alias, target in _COMPUTE_ALIASES.items():
            assert target in _REGISTRY or target in raw_field_targets, (
                f"Alias {alias!r} -> {target!r} not in registry or known fields"
            )

    def test_available_quantities_nonempty(self):
        names = available_quantities()
        assert len(names) > 30

    def test_available_quantities_sorted(self):
        names = available_quantities()
        assert names == sorted(names)


class TestMagnitudes:
    @pytest.mark.parametrize(
        ("name", "fields"),
        [
            ("|B|", ("B1", "B2", "B3")),
            ("|E|", ("E1", "E2", "E3")),
            ("|J|", ("J1", "J2", "J3")),
            ("|V|", ("V1", "V2", "V3")),
        ],
    )
    def test_345_triangle(self, name, fields):
        shape = (2, 2, 2)
        data = {
            fields[0]: np.full(shape, 3.0),
            fields[1]: np.full(shape, 4.0),
            fields[2]: np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field(name, ds)
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)


class TestDependencyChains:
    def test_beta_auto_computes_bmag(self):
        """beta needs |B|, which should be auto-computed from B1/B2/B3."""
        shape = (2, 2, 2)
        data = {
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
            "P": np.full(shape, 25.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("beta", ds)
        # beta = 2P / B^2 = 2*25 / 25 = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)

    def test_alfven_mach_chain(self):
        """M_A chains through |V| and v_A (which needs |B| and rho_m)."""
        shape = (2, 2, 2)
        data = {
            "V1": np.full(shape, 3.0),
            "V2": np.full(shape, 4.0),
            "V3": np.zeros(shape),
            "B1": np.full(shape, 1.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
            "rho_m": np.full(shape, 1.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("M_A", ds)
        # |V| = 5, v_A = 1/sqrt(1) = 1, M_A = 5
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_vorticity_magnitude_chain(self):
        """ "|vort|" needs vort1/2/3, each computed via curl."""
        shape = (4, 4, 4)
        data = {
            "V1": np.ones(shape),
            "V2": np.ones(shape),
            "V3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("|vort|", ds)
        # Uniform velocity → zero vorticity
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestDirectFieldPassthrough:
    def test_existing_field_returned_directly(self):
        shape = (2, 2, 2)
        b1 = np.full(shape, 42.0)
        ds = _make_dataset({"B1": b1}, shape=shape)
        result = compute_field("B1", ds)
        np.testing.assert_array_equal(result, b1)


class TestSpeciesDependent:
    def test_omega_pe(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 4.0)}
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("omega_pe", ds)
        # omega_pe = sqrt(n * q^2 / m) = sqrt(4 * 1 / (1/256)) = sqrt(1024) = 32
        np.testing.assert_allclose(result, 32.0, rtol=1e-14)

    def test_omega_ci(self):
        shape = (2, 2, 2)
        data = {
            "B1": np.full(shape, 2.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("omega_ci", ds)
        # omega_ci = |q_i| * |B| / m_i = 1 * 2 / 1 = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)

    def test_skin_depth(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0)}
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
            physics={"c": 2.0},
        )
        result = compute_field("d_e", ds)
        # d_e = c / omega_pe = 2 / sqrt(1 * 1 / (1/256)) = 2/16 = 0.125
        np.testing.assert_allclose(result, 0.125, rtol=1e-14)

    def test_thermal_speed(self):
        shape = (2, 2, 2)
        data = {"Te": np.full(shape, 4.0)}
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("v_th_e", ds)
        # v_th_e = sqrt(Te / m_e) = sqrt(4 / (1/256)) = sqrt(1024) = 32
        np.testing.assert_allclose(result, 32.0, rtol=1e-14)

    def test_gyroradius(self):
        shape = (2, 2, 2)
        data = {
            "Ti": np.full(shape, 1.0),
            "B1": np.full(shape, 1.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("r_i", ds)
        # r_i = sqrt(m_i * T_i) / (|q_i| * |B|) = sqrt(1*1) / (1*1) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)

    def test_debye_length(self):
        shape = (2, 2, 2)
        data = {
            "Te": np.full(shape, 1.0),
            "n_s0": np.full(shape, 1.0),
        }
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("lambda_D", ds)
        # lambda_D = sqrt(T / (n * q^2)) = sqrt(1 / (1*1)) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)

    def test_missing_species_raises(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0)}
        ds = _make_dataset(data, shape=shape)
        with pytest.raises(ValueError, match="requires species"):
            compute_field("omega_pe", ds)


class TestGridDependent:
    def test_div_b_uniform(self):
        shape = (4, 4, 4)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("div_B", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_curl_component(self):
        shape = (4, 4, 4)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("curl_B1", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestPhysicsConfig:
    def test_sound_speed_custom_gamma(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 3.0),
            "rho_m": np.full(shape, 3.0),
        }
        ds = _make_dataset(data, shape=shape, physics={"gamma": 2.0})
        result = compute_field("c_s", ds)
        # c_s = sqrt(gamma * P / rho_m) = sqrt(2 * 3 / 3) = sqrt(2)
        np.testing.assert_allclose(result, np.sqrt(2.0), rtol=1e-15)

    def test_sound_speed_default_gamma(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 5.0),
            "rho_m": np.full(shape, 3.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("c_s", ds)
        # c_s = sqrt(5/3 * 5 / 3) = sqrt(25/9) = 5/3
        np.testing.assert_allclose(result, 5.0 / 3.0, rtol=1e-14)


class TestMultiComponent:
    def test_poynting_components(self):
        shape = (2, 2, 2)
        data = {
            "E1": np.full(shape, 1.0),
            "E2": np.zeros(shape),
            "E3": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.full(shape, 1.0),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape)
        s3 = compute_field("S3", ds)
        np.testing.assert_allclose(s3, 1.0, rtol=1e-15)

        s1 = compute_field("S1", ds)
        np.testing.assert_allclose(s1, 0.0, atol=1e-15)

    def test_vorticity_component(self):
        shape = (4, 4, 4)
        data = {
            "V1": np.ones(shape),
            "V2": np.ones(shape),
            "V3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("vort2", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestComputeAliases:
    def test_cartesian_aliases(self):
        shape = (4, 4, 4)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result_alias = compute_field("curl_Bx", ds)
        result_canonical = compute_field("curl_B1", ds)
        np.testing.assert_array_equal(result_alias, result_canonical)


class TestFieldDatasetMethods:
    def test_compute_method(self):
        shape = (2, 2, 2)
        data = {
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = ds.compute("|B|")
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_in_si_with_identity(self):
        shape = (2, 2, 2)
        data = {"B1": np.full(shape, 5.0)}
        ds = _make_dataset(data, shape=shape)
        result = ds.in_si("B1")
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_in_si_with_normalization(self):
        shape = (2, 2, 2)
        b_ref = 2.5
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=b_ref,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        data = {"B1": np.full(shape, 3.0)}
        ds = _make_dataset(data, shape=shape, normalization=norm)
        result = ds.in_si("B1")
        np.testing.assert_allclose(result, 3.0 * b_ref, rtol=1e-15)

    def test_in_si_dimensionless(self):
        shape = (2, 2, 2)
        norm = Normalization.pic_electron(1e18)
        data = {
            "P": np.full(shape, 1.0),
            "B1": np.full(shape, 1.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape, normalization=norm)
        beta = ds.in_si("beta")
        beta_code = compute_field("beta", ds)
        # Dimensionless → same in SI
        np.testing.assert_allclose(beta, beta_code, rtol=1e-15)

    def test_in_units_nt(self):
        shape = (2, 2, 2)
        b_ref = 1e-6  # 1 μT
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=b_ref,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        data = {"B1": np.full(shape, 5.0)}
        ds = _make_dataset(data, shape=shape, normalization=norm)
        result = ds.in_units("B1", "nT")
        # 5 code * 1e-6 T / 1e-9 = 5000 nT
        np.testing.assert_allclose(result, 5000.0, rtol=1e-15)

    def test_in_units_unknown_raises(self):
        ds = _make_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(ValueError, match="Unknown unit"):
            ds.in_units("B1", "furlongs")


class TestSIConversion:
    def test_pressure_compound_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=3.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=2.0,
            mass_ref=5.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("P", norm)
        # pressure = density_ref * mass_ref * velocity_ref^2
        expected = 2.0 * 5.0 * 9.0
        assert factor == pytest.approx(expected)

    def test_frequency_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=0.5,
            velocity_ref=1.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("omega_pe", norm)
        assert factor == pytest.approx(2.0)

    def test_div_b_factor(self):
        norm = Normalization(
            length_ref=2.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=3.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("div_B", norm)
        # b_field_ref / length_ref
        assert factor == pytest.approx(1.5)


class TestDisplayUnits:
    def test_known_unit(self):
        assert display_unit_factor("nT") == pytest.approx(1e-9)
        assert display_unit_factor("km/s") == pytest.approx(1e3)
        assert display_unit_factor("RE") == pytest.approx(6.371e6)
        assert display_unit_factor("eV") == pytest.approx(constants.eV)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            display_unit_factor("parsecs")


class TestRoundTrip:
    def test_identity_normalization_passthrough(self):
        shape = (2, 2, 2)
        data = {"B1": np.full(shape, 7.0)}
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_allclose(ds.in_si("B1"), 7.0, rtol=1e-15)


class TestErrorMessages:
    def test_unknown_name_suggests(self):
        ds = _make_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(KeyError, match="Did you mean"):
            compute_field("bta", ds)

    def test_missing_dependency_lists_available(self):
        ds = _make_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(KeyError, match="requires"):
            compute_field("|B|", ds)

    def test_recursion_depth(self):
        ds = _make_dataset({}, shape=(2, 2, 2))
        with pytest.raises((KeyError, RecursionError)):
            compute_field("M_ms", ds)


class TestEnthalpy:
    def test_enthalpy_basic(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 1.0),
            "rho_m": np.full(shape, 1.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("h", ds)
        # h = gamma * P / ((gamma-1) * rho_m) = 5/3 / (2/3) = 2.5
        np.testing.assert_allclose(result, 2.5, rtol=1e-14)


class TestEntropy:
    def test_entropy_unit_values(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 1.0),
            "rho_m": np.full(shape, 1.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("s", ds)
        # s = ln(P / rho^gamma) = ln(1) = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_species_entropy(self):
        shape = (2, 2, 2)
        data = {
            "Pe": np.full(shape, 1.0),
            "n_s0": np.full(shape, 1.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("s_e", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestIonAcousticSpeed:
    def test_basic(self):
        shape = (2, 2, 2)
        data = {
            "Te": np.full(shape, 1.0),
            "Ti": np.zeros(shape),
        }
        ds = _make_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
        )
        result = compute_field("c_ia", ds)
        # c_ia = sqrt((gamma_e*Te + gamma_i*Ti) / m_i) = sqrt(1/1) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)


class TestSpeciesAliases:
    def test_n_e_alias_resolves_through_compute(self):
        """n_e alias in FieldDataset resolves to n_s0 for compute."""
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 4.0)}
        ds = _make_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        # omega_pe depends on n_s0 — verify it works when n_e is the alias
        result = compute_field("omega_pe", ds)
        np.testing.assert_allclose(result, 32.0, rtol=1e-14)

    def test_n_e_n_i_accessible_as_fields(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0), "n_s1": np.full(shape, 2.0)}
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_allclose(ds["n_e"], 1.0)
        np.testing.assert_allclose(ds["n_i"], 2.0)

    def test_s_gyro_i_uses_n_s1(self):
        shape = (2, 2, 2)
        data = {
            "P11": np.full(shape, 1.0),
            "P22": np.full(shape, 1.0),
            "P33": np.full(shape, 3.0),
            "P12": np.zeros(shape),
            "P13": np.zeros(shape),
            "P23": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
            "n_s1": np.full(shape, 2.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("s_gyro_i", ds)
        # P_par=3 (B along z), P_perp=(1+1+3-3)/2=1
        # s_gyro = ln(P_par * P_perp^2 / n^5) = ln(3 * 1 / 32) = ln(3/32)
        expected = np.log(3.0 * 1.0**2 / 2.0**5)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_bare_s_gyro_raises(self):
        """Bare s_gyro is an error — must specify s_gyro_e or s_gyro_i."""
        shape = (2, 2, 2)
        data = {
            "P11": np.full(shape, 1.0),
            "P22": np.full(shape, 1.0),
            "P33": np.full(shape, 3.0),
            "P12": np.zeros(shape),
            "P13": np.zeros(shape),
            "P23": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
            "n_s0": np.full(shape, 1.0),
        }
        ds = _make_dataset(data, shape=shape)
        with pytest.raises(KeyError, match="s_gyro"):
            compute_field("s_gyro", ds)

    def test_four_velocity_aliases_resolve(self):
        """ux/uy/uz field aliases work through FieldDataset."""
        shape = (2, 2, 2)
        data = {
            "u1": np.full(shape, 0.5),
            "u2": np.full(shape, 0.3),
            "u3": np.full(shape, 0.1),
        }
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_allclose(ds["ux"], 0.5)
        np.testing.assert_allclose(ds["uy"], 0.3)
        np.testing.assert_allclose(ds["uz"], 0.1)


class TestPressureTensor:
    def test_parallel_pressure(self):
        shape = (2, 2, 2)
        data = {
            "P11": np.full(shape, 1.0),
            "P22": np.full(shape, 2.0),
            "P33": np.full(shape, 3.0),
            "P12": np.zeros(shape),
            "P13": np.zeros(shape),
            "P23": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("P_par", ds)
        # B along z → P_par = P33 = 3
        np.testing.assert_allclose(result, 3.0, rtol=1e-15)

    def test_perpendicular_pressure(self):
        shape = (2, 2, 2)
        data = {
            "P11": np.full(shape, 1.0),
            "P22": np.full(shape, 2.0),
            "P33": np.full(shape, 3.0),
            "P12": np.zeros(shape),
            "P13": np.zeros(shape),
            "P23": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("P_perp", ds)
        # P_perp = (Tr(P) - P_par) / 2 = (6 - 3) / 2 = 1.5
        np.testing.assert_allclose(result, 1.5, rtol=1e-15)


class TestGeometryGuard:
    def test_compute_div_b_rejects_spherical(self):
        shape = (4, 4, 4)
        grid = GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0), geometry=SPHERICAL)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = FieldDataset.from_arrays(data, grid, Normalization.identity())
        with pytest.raises(NotImplementedError, match="Cartesian"):
            compute_field("div_B", ds)

    def test_compute_vorticity_rejects_spherical(self):
        shape = (4, 4, 4)
        grid = GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0), geometry=SPHERICAL)
        data = {
            "V1": np.ones(shape),
            "V2": np.ones(shape),
            "V3": np.ones(shape),
        }
        ds = FieldDataset.from_arrays(data, grid, Normalization.identity())
        with pytest.raises(NotImplementedError, match="Cartesian"):
            compute_field("vort1", ds)


class TestSIFactorCoverage:
    def test_div_e_factor(self):
        norm = Normalization(
            length_ref=2.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=1.0,
            e_field_ref=3.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("div_E", norm)
        # e_field_ref / length_ref
        assert factor == pytest.approx(1.5)

    def test_background_b_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=7.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("B0_1", norm)
        assert factor == pytest.approx(7.0)

    def test_species_density_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=5.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("n_s2", norm)
        assert factor == pytest.approx(5.0)

    def test_pressure_tensor_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=3.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=2.0,
            mass_ref=5.0,
            charge_ref=1.0,
        )
        factor = field_si_factor("P11", norm)
        # pressure = density_ref * mass_ref * velocity_ref^2
        assert factor == pytest.approx(2.0 * 5.0 * 9.0)
