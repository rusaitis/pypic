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
    register_recipe,
    unregister_recipe,
)
from pypic.coordinates.geometry import SPHERICAL
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.units import Normalization, PhysicsParams
from tests._helpers import ELECTRONS, IONS, make_test_dataset


class TestRegistryIntegrity:
    def test_no_duplicate_names(self):
        overlap = set(_REGISTRY) & set(_COMPUTE_ALIASES)
        assert not overlap, f"Name collision: {overlap}"

    def test_all_aliases_resolve(self):
        # Aliases may point to raw field names (Te, Pi, etc.) used as
        # direct passthrough, or to dynamic species template recipes.
        from pypic.compute import _try_species_recipe

        raw_field_targets = {
            "Te",
            "Ti",
            "Pe",
            "Pi",
            "EF1",
            "EF2",
            "EF3",
            "EF_s0",
            "EF_s1",
            "EF1_s0",
            "EF2_s0",
            "EF3_s0",
            "EF1_s1",
            "EF2_s1",
            "EF3_s1",
            "KEF_s0",
            "KEF_s1",
            "HF_s0",
            "HF_s1",
            "EHF_s0",
            "EHF_s1",
            "q_s0",
            "q_s1",
        }
        for alias, target in _COMPUTE_ALIASES.items():
            in_registry = target in _REGISTRY
            in_raw = target in raw_field_targets
            in_species = _try_species_recipe(target) is not None
            assert in_registry or in_raw or in_species, (
                f"Alias {alias!r} -> {target!r} not in registry or known fields"
            )

    def test_available_quantities_nonempty(self):
        names = available_quantities()
        assert len(names) > 30

    def test_available_quantities_sorted(self):
        names = available_quantities()
        assert names == sorted(names)

    def test_species_recipes_have_species_args(self):
        """Every recipe with species_index must have a species_args descriptor."""
        for name, recipe in _REGISTRY.items():
            if recipe.species_index is not None:
                assert recipe.species_args is not None, (
                    f"Recipe {name!r} has species_index={recipe.species_index} "
                    f"but species_args is None"
                )


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
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("beta", ds)
        # beta = 2P / B^2 = 2*25 / 25 = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)

    def test_P_from_species_tensors(self):
        """P = Pe + Pi, each from per-species tensor trace."""
        shape = (2, 2, 2)
        data = {
            "P11_s0": np.full(shape, 3.0),
            "P22_s0": np.full(shape, 3.0),
            "P33_s0": np.full(shape, 3.0),
            "P11_s1": np.full(shape, 6.0),
            "P22_s1": np.full(shape, 6.0),
            "P33_s1": np.full(shape, 6.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("P", ds)
        # Pe = 3, Pi = 6, P = 9
        np.testing.assert_allclose(result, 9.0, rtol=1e-15)

    def test_P_raw_takes_precedence(self):
        """Raw scalar P in dataset takes priority over compute chain."""
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 42.0),
            "P11_s0": np.full(shape, 1.0),
            "P22_s0": np.full(shape, 1.0),
            "P33_s0": np.full(shape, 1.0),
            "P11_s1": np.full(shape, 1.0),
            "P22_s1": np.full(shape, 1.0),
            "P33_s1": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("P", ds)
        np.testing.assert_allclose(result, 42.0, rtol=1e-15)

    def test_beta_from_species_tensors(self):
        """beta chains through P → Pe + Pi → per-species tensor traces."""
        shape = (2, 2, 2)
        data = {
            "P11_s0": np.full(shape, 5.0),
            "P22_s0": np.full(shape, 5.0),
            "P33_s0": np.full(shape, 5.0),
            "P11_s1": np.full(shape, 5.0),
            "P22_s1": np.full(shape, 5.0),
            "P33_s1": np.full(shape, 5.0),
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("beta", ds)
        # Pe = 5, Pi = 5, P = 10, |B| = 5, beta = 2*10/25 = 0.8
        np.testing.assert_allclose(result, 0.8, rtol=1e-15)

    def test_Pe_from_species_tensor(self):
        """Pe falls back to Tr(electron tensor)/3."""
        shape = (2, 2, 2)
        data = {
            "P11_s0": np.full(shape, 3.0),
            "P22_s0": np.full(shape, 6.0),
            "P33_s0": np.full(shape, 9.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("Pe", ds)
        np.testing.assert_allclose(result, 6.0, rtol=1e-15)

    def test_Pi_from_species_tensor(self):
        """Pi falls back to Tr(ion tensor)/3."""
        shape = (2, 2, 2)
        data = {
            "P11_s1": np.full(shape, 6.0),
            "P22_s1": np.full(shape, 12.0),
            "P33_s1": np.full(shape, 18.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("Pi", ds)
        np.testing.assert_allclose(result, 12.0, rtol=1e-15)

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
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("|vort|", ds)
        # Uniform velocity → zero vorticity
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_existing_field_returned_directly(self):
        shape = (2, 2, 2)
        b1 = np.full(shape, 42.0)
        ds = make_test_dataset({"B1": b1}, shape=shape)
        result = compute_field("B1", ds)
        np.testing.assert_array_equal(result, b1)


class TestSpeciesDependent:
    def test_omega_pe(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 4.0)}
        ds = make_test_dataset(
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
        ds = make_test_dataset(
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
        ds = make_test_dataset(
            data,
            shape=shape,
            species=[ELECTRONS, IONS],
            physics=PhysicsParams(c=2.0),
        )
        result = compute_field("d_e", ds)
        # d_e = c / omega_pe = 2 / sqrt(1 * 1 / (1/256)) = 2/16 = 0.125
        np.testing.assert_allclose(result, 0.125, rtol=1e-14)

    def test_thermal_speed(self):
        shape = (2, 2, 2)
        data = {"Te": np.full(shape, 4.0)}
        ds = make_test_dataset(
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
        ds = make_test_dataset(
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
        ds = make_test_dataset(
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
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("div_B", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_curl_component(self):
        shape = (4, 4, 4)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("curl_B1", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestPhysicsConfig:
    def test_sound_speed_custom_gamma(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 3.0),
            "rho_m": np.full(shape, 3.0),
        }
        ds = make_test_dataset(data, shape=shape, physics=PhysicsParams(gamma=2.0))
        result = compute_field("c_s", ds)
        # c_s = sqrt(gamma * P / rho_m) = sqrt(2 * 3 / 3) = sqrt(2)
        np.testing.assert_allclose(result, np.sqrt(2.0), rtol=1e-15)

    def test_sound_speed_default_gamma(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 5.0),
            "rho_m": np.full(shape, 3.0),
        }
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("vort2", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestFieldDatasetMethods:
    def test_compute_method(self):
        shape = (2, 2, 2)
        data = {
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = ds.compute("|B|")
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_in_si_with_identity(self):
        shape = (2, 2, 2)
        data = {"B1": np.full(shape, 5.0)}
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape, normalization=norm)
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
        ds = make_test_dataset(data, shape=shape, normalization=norm)
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
        ds = make_test_dataset(data, shape=shape, normalization=norm)
        result = ds.in_units("B1", "nT")
        # 5 code * 1e-6 T / 1e-9 = 5000 nT
        np.testing.assert_allclose(result, 5000.0, rtol=1e-15)

    def test_in_units_unknown_raises(self):
        ds = make_test_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(ValueError, match="Unknown unit"):
            ds.in_units("B1", "furlongs")

    def test_cartesian_compute_aliases(self):
        shape = (4, 4, 4)
        data = {"B1": np.ones(shape), "B2": np.ones(shape), "B3": np.ones(shape)}
        ds = make_test_dataset(data, shape=shape)
        result_alias = compute_field("curl_Bx", ds)
        result_canonical = compute_field("curl_B1", ds)
        np.testing.assert_array_equal(result_alias, result_canonical)

    def test_identity_normalization_passthrough(self):
        shape = (2, 2, 2)
        ds = make_test_dataset({"B1": np.full(shape, 7.0)}, shape=shape)
        np.testing.assert_allclose(ds.in_si("B1"), 7.0, rtol=1e-15)


class TestSIFactors:
    @pytest.mark.parametrize(
        ("field", "norm_kwargs", "expected"),
        [
            # Compound: pressure = density_ref * mass_ref * velocity_ref^2
            ("P", {"velocity_ref": 3.0, "density_ref": 2.0, "mass_ref": 5.0}, 90.0),
            # Frequency = 1/time_ref
            ("omega_pe", {"time_ref": 0.5}, 2.0),
            # Background B uses b_field_ref
            ("B0_1", {"b_field_ref": 7.0}, 7.0),
            # Species density uses density_ref
            ("n_s2", {"density_ref": 5.0}, 5.0),
            # Pressure tensor = same as pressure
            ("P11", {"velocity_ref": 3.0, "density_ref": 2.0, "mass_ref": 5.0}, 90.0),
            # Dimensionless quantities
            ("gamma_L", {}, 1.0),
            ("sigma", {}, 1.0),
            # Vorticity = velocity_per_length = velocity_ref / length_ref
            ("vort1", {"velocity_ref": 4.0, "length_ref": 2.0}, 2.0),
            # Specific energy = velocity_ref^2 (NOT mass_ref * velocity_ref^2)
            ("h", {"velocity_ref": 3.0, "mass_ref": 5.0}, 9.0),
        ],
        ids=[
            "pressure",
            "frequency",
            "B0",
            "n_s2",
            "P11",
            "gamma_L",
            "sigma",
            "vort1",
            "enthalpy",
        ],
    )
    def test_si_factor(self, field, norm_kwargs, expected):
        defaults = dict(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=1.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        defaults.update(norm_kwargs)
        norm = Normalization(**defaults)
        assert field_si_factor(field, norm) == pytest.approx(expected)

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
        assert field_si_factor("div_B", norm) == pytest.approx(1.5)

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
        assert field_si_factor("div_E", norm) == pytest.approx(1.5)


class TestDisplayUnits:
    def test_known_unit(self):
        assert display_unit_factor("nT") == pytest.approx(1e-9)
        assert display_unit_factor("km/s") == pytest.approx(1e3)
        assert display_unit_factor("R_E") == pytest.approx(6.371e6)
        assert display_unit_factor("eV") == pytest.approx(constants.eV)
        assert display_unit_factor("normalized") == pytest.approx(1.0)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            display_unit_factor("parsecs")


class TestErrorMessages:
    def test_unknown_name_suggests(self):
        ds = make_test_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(KeyError, match="Did you mean"):
            compute_field("bta", ds)

    def test_missing_dependency_lists_available(self):
        ds = make_test_dataset({"B1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(KeyError, match="requires"):
            compute_field("|B|", ds)

    def test_recursion_depth(self):
        ds = make_test_dataset({}, shape=(2, 2, 2))
        with pytest.raises((KeyError, RecursionError)):
            compute_field("M_ms", ds)


class TestThermodynamicCompute:
    def test_enthalpy(self):
        shape = (2, 2, 2)
        data = {"P": np.full(shape, 1.0), "rho_m": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("h", ds)
        # h = gamma * P / ((gamma-1) * rho_m) = 5/3 / (2/3) = 2.5
        np.testing.assert_allclose(result, 2.5, rtol=1e-14)

    def test_entropy_unit_values(self):
        shape = (2, 2, 2)
        data = {"P": np.full(shape, 1.0), "rho_m": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("s", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_species_entropy(self):
        shape = (2, 2, 2)
        data = {"Pe": np.full(shape, 1.0), "n_s0": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("s_e", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_ion_acoustic_speed(self):
        shape = (2, 2, 2)
        data = {"Te": np.full(shape, 1.0), "Ti": np.zeros(shape)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("c_ia", ds)
        # c_ia = sqrt((gamma_e*Te + gamma_i*Ti) / m_i) = sqrt(1/1) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)


class TestSpeciesAliases:
    def test_n_e_alias_resolves_through_compute(self):
        """n_e alias in FieldDataset resolves to n_s0 for compute."""
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 4.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        # omega_pe depends on n_s0 — verify it works when n_e is the alias
        result = compute_field("omega_pe", ds)
        np.testing.assert_allclose(result, 32.0, rtol=1e-14)

    def test_n_e_n_i_accessible_as_fields(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0), "n_s1": np.full(shape, 2.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_allclose(ds["n_e"], 1.0)
        np.testing.assert_allclose(ds["n_i"], 2.0)

    def test_s_gyro_i_uses_per_species_pressure(self):
        shape = (2, 2, 2)
        data = {
            "P11_s1": np.full(shape, 1.0),
            "P22_s1": np.full(shape, 1.0),
            "P33_s1": np.full(shape, 3.0),
            "P12_s1": np.zeros(shape),
            "P13_s1": np.zeros(shape),
            "P23_s1": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
            "n_s1": np.full(shape, 2.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("s_gyro_i", ds)
        # P_par_i=3 (B along z), P_perp_i=(1+1+3-3)/2=1
        # s_gyro = ln(P_par * P_perp^2 / n^5) = ln(3 * 1 / 32) = ln(3/32)
        expected = np.log(3.0 * 1.0**2 / 2.0**5)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_bare_s_gyro_raises(self):
        """Bare s_gyro is an error — must specify s_gyro_e or s_gyro_i."""
        shape = (2, 2, 2)
        data = {
            "P11_s0": np.full(shape, 1.0),
            "P22_s0": np.full(shape, 1.0),
            "P33_s0": np.full(shape, 3.0),
            "P12_s0": np.zeros(shape),
            "P13_s0": np.zeros(shape),
            "P23_s0": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
            "n_s0": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
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
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_allclose(ds["ux"], 0.5)
        np.testing.assert_allclose(ds["uy"], 0.3)
        np.testing.assert_allclose(ds["uz"], 0.1)


class TestPressureTensor:
    def test_parallel_and_perpendicular_pressure(self):
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
        ds = make_test_dataset(data, shape=shape)
        # B along z → P_par = P33 = 3
        np.testing.assert_allclose(compute_field("P_par", ds), 3.0, rtol=1e-15)
        # P_perp = (Tr(P) - P_par) / 2 = (6 - 3) / 2 = 1.5
        np.testing.assert_allclose(compute_field("P_perp", ds), 1.5, rtol=1e-15)


class TestPerSpeciesPressureDecomposition:
    """Per-species P_par, P_perp, agyrotropy via compute registry."""

    def _make_species_tensor_dataset(self):
        """Two species with different diagonal tensors, B along z."""
        shape = (2, 2, 2)
        data = {
            # Species 0 (electrons): P_diag = (1, 1, 3)
            "P11_s0": np.full(shape, 1.0),
            "P22_s0": np.full(shape, 1.0),
            "P33_s0": np.full(shape, 3.0),
            "P12_s0": np.zeros(shape),
            "P13_s0": np.zeros(shape),
            "P23_s0": np.zeros(shape),
            # Species 1 (ions): P_diag = (2, 4, 6)
            "P11_s1": np.full(shape, 2.0),
            "P22_s1": np.full(shape, 4.0),
            "P33_s1": np.full(shape, 6.0),
            "P12_s1": np.zeros(shape),
            "P13_s1": np.zeros(shape),
            "P23_s1": np.zeros(shape),
            # Total tensor = sum of per-species
            "P11": np.full(shape, 3.0),
            "P22": np.full(shape, 5.0),
            "P33": np.full(shape, 9.0),
            "P12": np.zeros(shape),
            "P13": np.zeros(shape),
            "P23": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
        }
        return make_test_dataset(data, shape=shape)

    def test_per_species_parallel_pressure(self):
        ds = self._make_species_tensor_dataset()
        # B along z → P_par = P33
        np.testing.assert_allclose(compute_field("P_par_e", ds), 3.0, rtol=1e-15)
        np.testing.assert_allclose(compute_field("P_par_i", ds), 6.0, rtol=1e-15)

    def test_per_species_perpendicular_pressure(self):
        ds = self._make_species_tensor_dataset()
        # P_perp = (Tr(P) - P_par) / 2
        # s0: (1+1+3 - 3)/2 = 1.0
        # s1: (2+4+6 - 6)/2 = 3.0
        np.testing.assert_allclose(compute_field("P_perp_e", ds), 1.0, rtol=1e-15)
        np.testing.assert_allclose(compute_field("P_perp_i", ds), 3.0, rtol=1e-15)

    def test_per_species_agyrotropy_isotropic(self):
        shape = (2, 2, 2)
        p = np.full(shape, 2.0)
        data = {
            "P11_s0": p,
            "P22_s0": p,
            "P33_s0": p,
            "P12_s0": np.zeros(shape),
            "P13_s0": np.zeros(shape),
            "P23_s0": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        # Isotropic tensor → Q = 0
        np.testing.assert_allclose(compute_field("agyrotropy_e", ds), 0.0, atol=1e-15)

    def test_alias_resolution(self):
        ds = self._make_species_tensor_dataset()
        # P_par_s0 (alias) and P_par_e (static) give same result
        result_alias = compute_field("P_par_s0", ds)
        result_static = compute_field("P_par_e", ds)
        np.testing.assert_array_equal(result_alias, result_static)

    def test_total_equals_sum_of_per_species(self):
        """P_par = P_par_s0 + P_par_s1 (linear in tensor components)."""
        ds = self._make_species_tensor_dataset()
        total = compute_field("P_par", ds)
        per_species_sum = compute_field("P_par_e", ds) + compute_field("P_par_i", ds)
        np.testing.assert_allclose(total, per_species_sum, rtol=1e-15)

        total_perp = compute_field("P_perp", ds)
        perp_sum = compute_field("P_perp_e", ds) + compute_field("P_perp_i", ds)
        np.testing.assert_allclose(total_perp, perp_sum, rtol=1e-15)

    def test_dynamic_species_without_species_metadata(self):
        """P_par_s2 works when tensor fields exist but species[2] is not."""
        shape = (2, 2, 2)
        data = {
            "P11_s2": np.full(shape, 1.0),
            "P22_s2": np.full(shape, 2.0),
            "P33_s2": np.full(shape, 4.0),
            "P12_s2": np.zeros(shape),
            "P13_s2": np.zeros(shape),
            "P23_s2": np.zeros(shape),
            "B1": np.zeros(shape),
            "B2": np.zeros(shape),
            "B3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        # B along z → P_par = P33 = 4
        np.testing.assert_allclose(compute_field("P_par_s2", ds), 4.0, rtol=1e-15)


class TestGeometryGuard:
    @pytest.mark.parametrize(
        ("field", "components"),
        [
            ("div_B", {"B1", "B2", "B3"}),
            ("vort1", {"V1", "V2", "V3"}),
        ],
    )
    def test_compute_rejects_spherical(self, field, components):
        shape = (4, 4, 4)
        grid = GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0), geometry=SPHERICAL)
        data = {name: np.ones(shape) for name in components}
        ds = FieldDataset.from_arrays(data, grid, Normalization.identity())
        with pytest.raises(NotImplementedError, match="Cartesian"):
            compute_field(field, ds)


class TestRegisterRecipe:
    def test_register_and_compute(self):
        """Custom recipe is computable via compute_field."""
        register_recipe(
            "e_mag_ratio",
            func=lambda eb, ee: eb / (eb + ee),
            fields=("e_B", "e_E"),
            quantity_type="dimensionless",
            long_name="Magnetic-to-total EM energy ratio",
        )
        try:
            ds = make_test_dataset(
                {
                    "B1": np.ones((4, 3, 2)),
                    "B2": np.zeros((4, 3, 2)),
                    "B3": np.zeros((4, 3, 2)),
                    "E1": np.ones((4, 3, 2)),
                    "E2": np.zeros((4, 3, 2)),
                    "E3": np.zeros((4, 3, 2)),
                },
            )
            result = compute_field("e_mag_ratio", ds)
            assert result.shape == (4, 3, 2)
            assert "e_mag_ratio" in available_quantities()
        finally:
            unregister_recipe("e_mag_ratio")

    def test_unregister_removes_recipe_and_metadata(self):
        register_recipe(
            "_test_tmp",
            func=lambda b: b * 2,
            fields=("|B|",),
            quantity_type="b_field",
        )
        assert "_test_tmp" in _REGISTRY
        unregister_recipe("_test_tmp")
        assert "_test_tmp" not in _REGISTRY

    def test_duplicate_name_raises(self):
        register_recipe(
            "_test_dup",
            func=lambda b: b,
            fields=("|B|",),
            quantity_type="dimensionless",
        )
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_recipe(
                    "_test_dup",
                    func=lambda b: b,
                    fields=("|B|",),
                    quantity_type="dimensionless",
                )
        finally:
            unregister_recipe("_test_dup")

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(KeyError, match="No recipe"):
            unregister_recipe("_nonexistent_recipe_xyz")

    def test_si_conversion_works(self):
        """Registered recipe quantity_type enables SI conversion."""
        register_recipe(
            "_test_si",
            func=lambda b: b * 2,
            fields=("|B|",),
            quantity_type="b_field",
        )
        try:
            factor = field_si_factor("_test_si", Normalization.identity())
            assert factor == 1.0
        finally:
            unregister_recipe("_test_si")
