import numpy as np
import pytest
from scipy import constants

from pypic.compute import (
    _REGISTRY,
    COMPUTE_ALIASES,
    available_quantities,
    compute_field,
    display_unit_factor,
    field_si_factor,
    register_recipe,
    unregister_recipe,
)
from pypic.coordinates.geometry import SPHERICAL
from pypic.dataset import FieldDataset
from pypic.exceptions import UnknownFieldError
from pypic.grid import GridInfo
from pypic.units import Normalization, PhysicsParams, SpeciesInfo
from tests._helpers import ELECTRONS, IONS, make_test_dataset


class TestRegistryIntegrity:
    def test_no_duplicate_names(self):
        overlap = set(_REGISTRY) & set(COMPUTE_ALIASES)
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
            "EF_1",
            "EF_2",
            "EF_3",
            "EF_s0",
            "EF_s1",
            "EF_s0_1",
            "EF_s0_2",
            "EF_s0_3",
            "EF_s1_1",
            "EF_s1_2",
            "EF_s1_3",
            "KEF_s0",
            "KEF_s1",
            "HF_s0",
            "HF_s1",
            "EHF_s0",
            "EHF_s1",
            "q_s0",
            "q_s1",
        }
        for alias, target in COMPUTE_ALIASES.items():
            in_registry = target in _REGISTRY
            in_raw = target in raw_field_targets
            in_species = _try_species_recipe(target) is not None
            assert in_registry or in_raw or in_species, (
                f"Alias {alias!r} -> {target!r} not in registry or known fields"
            )

    def test_available_quantities_nonempty(self):
        names = available_quantities()
        # ``> 30`` alone would keep passing after an accidental registry wipe
        # that leaves a handful of entries.  Pin a small set of core
        # quantities whose absence would be a real regression.
        assert len(names) > 30
        for required in ("|B|", "beta", "v_A", "div_B"):
            assert required in names, f"core quantity {required!r} missing"

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
            ("|B|", ("B_1", "B_2", "B_3")),
            ("|E|", ("E_1", "E_2", "E_3")),
            ("|J|", ("J_1", "J_2", "J_3")),
            ("|V|", ("V_1", "V_2", "V_3")),
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
        """beta needs |B|, which should be auto-computed from B_1/B_2/B_3."""
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 3.0),
            "B_2": np.full(shape, 4.0),
            "B_3": np.zeros(shape),
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
            "P_s0_11": np.full(shape, 3.0),
            "P_s0_22": np.full(shape, 3.0),
            "P_s0_33": np.full(shape, 3.0),
            "P_s1_11": np.full(shape, 6.0),
            "P_s1_22": np.full(shape, 6.0),
            "P_s1_33": np.full(shape, 6.0),
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
            "P_s0_11": np.full(shape, 1.0),
            "P_s0_22": np.full(shape, 1.0),
            "P_s0_33": np.full(shape, 1.0),
            "P_s1_11": np.full(shape, 1.0),
            "P_s1_22": np.full(shape, 1.0),
            "P_s1_33": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("P", ds)
        np.testing.assert_allclose(result, 42.0, rtol=1e-15)

    def test_beta_from_species_tensors(self):
        """beta chains through P → Pe + Pi → per-species tensor traces."""
        shape = (2, 2, 2)
        data = {
            "P_s0_11": np.full(shape, 5.0),
            "P_s0_22": np.full(shape, 5.0),
            "P_s0_33": np.full(shape, 5.0),
            "P_s1_11": np.full(shape, 5.0),
            "P_s1_22": np.full(shape, 5.0),
            "P_s1_33": np.full(shape, 5.0),
            "B_1": np.full(shape, 3.0),
            "B_2": np.full(shape, 4.0),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("beta", ds)
        # Pe = 5, Pi = 5, P = 10, |B| = 5, beta = 2*10/25 = 0.8
        np.testing.assert_allclose(result, 0.8, rtol=1e-15)

    def test_Pe_from_species_tensor(self):
        """Pe falls back to Tr(electron tensor)/3."""
        shape = (2, 2, 2)
        data = {
            "P_s0_11": np.full(shape, 3.0),
            "P_s0_22": np.full(shape, 6.0),
            "P_s0_33": np.full(shape, 9.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("Pe", ds)
        np.testing.assert_allclose(result, 6.0, rtol=1e-15)

    def test_Pi_from_species_tensor(self):
        """Pi falls back to Tr(ion tensor)/3."""
        shape = (2, 2, 2)
        data = {
            "P_s1_11": np.full(shape, 6.0),
            "P_s1_22": np.full(shape, 12.0),
            "P_s1_33": np.full(shape, 18.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("Pi", ds)
        np.testing.assert_allclose(result, 12.0, rtol=1e-15)

    def test_alfven_mach_chain(self):
        """M_A chains through |V| and v_A (which needs |B| and rho_m)."""
        shape = (2, 2, 2)
        data = {
            "V_1": np.full(shape, 3.0),
            "V_2": np.full(shape, 4.0),
            "V_3": np.zeros(shape),
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
            "rho_m": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("M_A", ds)
        # |V| = 5, v_A = 1/sqrt(1) = 1, M_A = 5
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_vorticity_magnitude_chain(self):
        """ "|vort|" needs vort_1/2/3, each computed via curl."""
        shape = (4, 4, 4)
        data = {
            "V_1": np.ones(shape),
            "V_2": np.ones(shape),
            "V_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("|vort|", ds)
        # Uniform velocity → zero vorticity
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_existing_field_returned_directly(self):
        shape = (2, 2, 2)
        b1 = np.full(shape, 42.0)
        ds = make_test_dataset({"B_1": b1}, shape=shape)
        result = compute_field("B_1", ds)
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
            "B_1": np.full(shape, 2.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
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
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
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


class TestSpeciesTemplateRelativistic:
    def test_species_two_gets_relativistic_cap_like_species_zero(self):
        shape = (2, 2, 2)
        alphas = SpeciesInfo(name="alphas", charge=2.0, mass=4.0)
        ds = make_test_dataset(
            {"T_s0": np.ones(shape), "T_s2": np.full(shape, 4.0)},
            shape=shape,
            species=[ELECTRONS, IONS, alphas],
            physics=PhysicsParams(c=1.0, relativistic=True),
        )
        # v_th = sqrt(T/m) capped as v_th / sqrt(1 + v_th^2 / c^2)
        v0 = np.sqrt(1.0 / ELECTRONS.mass)
        v2 = np.sqrt(4.0 / alphas.mass)
        np.testing.assert_allclose(
            compute_field("v_th_s0", ds), v0 / np.sqrt(1 + v0**2), rtol=1e-12
        )
        np.testing.assert_allclose(
            compute_field("v_th_s2", ds), v2 / np.sqrt(1 + v2**2), rtol=1e-12
        )


class TestGridDependent:
    def test_div_b_uniform(self):
        shape = (4, 4, 4)
        data = {
            "B_1": np.ones(shape),
            "B_2": np.ones(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("div_B", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_curl_component(self):
        shape = (4, 4, 4)
        data = {
            "B_1": np.ones(shape),
            "B_2": np.ones(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("curl_B_1", ds)
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
            "E_1": np.full(shape, 1.0),
            "E_2": np.zeros(shape),
            "E_3": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.full(shape, 1.0),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        s3 = compute_field("S_3", ds)
        np.testing.assert_allclose(s3, 1.0, rtol=1e-15)

        s1 = compute_field("S_1", ds)
        np.testing.assert_allclose(s1, 0.0, atol=1e-15)

    def test_vorticity_component(self):
        shape = (4, 4, 4)
        data = {
            "V_1": np.ones(shape),
            "V_2": np.ones(shape),
            "V_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("vort_2", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


class TestFieldDatasetMethods:
    def test_compute_method(self):
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 3.0),
            "B_2": np.full(shape, 4.0),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = ds.compute("|B|")
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    def test_in_si_with_identity(self):
        shape = (2, 2, 2)
        data = {"B_1": np.full(shape, 5.0)}
        ds = make_test_dataset(data, shape=shape)
        result = ds.in_si("B_1")
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
        data = {"B_1": np.full(shape, 3.0)}
        ds = make_test_dataset(data, shape=shape, normalization=norm)
        result = ds.in_si("B_1")
        np.testing.assert_allclose(result, 3.0 * b_ref, rtol=1e-15)

    def test_in_si_dimensionless(self):
        shape = (2, 2, 2)
        norm = Normalization.pic_electron(1e18)
        data = {
            "P": np.full(shape, 1.0),
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
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
        data = {"B_1": np.full(shape, 5.0)}
        ds = make_test_dataset(data, shape=shape, normalization=norm)
        result = ds.in_units("B_1", "nT")
        # 5 code * 1e-6 T / 1e-9 = 5000 nT
        np.testing.assert_allclose(result, 5000.0, rtol=1e-15)

    def test_in_units_unknown_raises(self):
        ds = make_test_dataset({"B_1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(ValueError, match="Unknown unit"):
            ds.in_units("B_1", "furlongs")

    def test_in_units_temperature_eV_and_K(self):
        # pypic stores T in energy units (J).  in_units must convert to
        # the plasma working unit (eV) and to K via the Boltzmann factor.
        shape = (2, 2, 2)
        # Identity normalization → code value equals SI value (J).
        ds = make_test_dataset(
            {"Te": np.full(shape, 1.602e-18)},  # ~10 eV electrons
            shape=shape,
        )
        np.testing.assert_allclose(ds.in_si("Te"), 1.602e-18, rtol=1e-12)
        # 1 eV = constants.eV J → 1.602e-18 J / 1.602e-19 (J/eV) ≈ 10 eV
        np.testing.assert_allclose(
            ds.in_units("Te", "eV"), 1.602e-18 / constants.eV, rtol=1e-12
        )
        # 1 K = constants.k J → 1.602e-18 J / 1.381e-23 (J/K) ≈ 1.16e5 K
        np.testing.assert_allclose(
            ds.in_units("Te", "K"), 1.602e-18 / constants.k, rtol=1e-12
        )
        # keV / MeV scale linearly
        np.testing.assert_allclose(
            ds.in_units("Te", "keV"), ds.in_units("Te", "eV") / 1e3, rtol=1e-12
        )

    def test_cartesian_compute_aliases(self):
        shape = (4, 4, 4)
        data = {"B_1": np.ones(shape), "B_2": np.ones(shape), "B_3": np.ones(shape)}
        ds = make_test_dataset(data, shape=shape)
        result_alias = compute_field("curl_Bx", ds)
        result_canonical = compute_field("curl_B_1", ds)
        np.testing.assert_array_equal(result_alias, result_canonical)


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
            ("P_11", {"velocity_ref": 3.0, "density_ref": 2.0, "mass_ref": 5.0}, 90.0),
            # Dimensionless quantities
            ("gamma_L", {}, 1.0),
            ("sigma", {}, 1.0),
            # Vorticity = velocity_per_length = velocity_ref / length_ref
            ("vort_1", {"velocity_ref": 4.0, "length_ref": 2.0}, 2.0),
            # Specific energy = velocity_ref^2 (NOT mass_ref * velocity_ref^2)
            ("h", {"velocity_ref": 3.0, "mass_ref": 5.0}, 9.0),
        ],
        ids=[
            "pressure",
            "frequency",
            "B0",
            "n_s2",
            "P_11",
            "gamma_L",
            "sigma",
            "vort_1",
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
        ds = make_test_dataset({"B_1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(KeyError, match="Did you mean"):
            compute_field("bta", ds)

    def test_missing_dependency_names_request_and_leaf(self):
        ds = make_test_dataset({"B_1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(UnknownFieldError, match=r"Cannot compute '\|B\|'.*'B_2'"):
            compute_field("|B|", ds)

    def test_nested_missing_dependency_keeps_suggestions(self):
        ds = make_test_dataset({"B_1": np.ones((2, 2, 2))}, shape=(2, 2, 2))
        with pytest.raises(
            UnknownFieldError, match=r"Cannot compute 'beta'.*Did you mean"
        ):
            compute_field("beta", ds)

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
            "P_s1_11": np.full(shape, 1.0),
            "P_s1_22": np.full(shape, 1.0),
            "P_s1_33": np.full(shape, 3.0),
            "P_s1_12": np.zeros(shape),
            "P_s1_13": np.zeros(shape),
            "P_s1_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
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
            "P_s0_11": np.full(shape, 1.0),
            "P_s0_22": np.full(shape, 1.0),
            "P_s0_33": np.full(shape, 3.0),
            "P_s0_12": np.zeros(shape),
            "P_s0_13": np.zeros(shape),
            "P_s0_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "n_s0": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
        with pytest.raises(KeyError, match="s_gyro"):
            compute_field("s_gyro", ds)

    def test_four_velocity_aliases_resolve(self):
        """ux/uy/uz field aliases work through FieldDataset."""
        shape = (2, 2, 2)
        data = {
            "u_1": np.full(shape, 0.5),
            "u_2": np.full(shape, 0.3),
            "u_3": np.full(shape, 0.1),
        }
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_allclose(ds["ux"], 0.5)
        np.testing.assert_allclose(ds["uy"], 0.3)
        np.testing.assert_allclose(ds["uz"], 0.1)


class TestPressureTensor:
    def test_parallel_and_perpendicular_pressure(self):
        shape = (2, 2, 2)
        data = {
            "P_11": np.full(shape, 1.0),
            "P_22": np.full(shape, 2.0),
            "P_33": np.full(shape, 3.0),
            "P_12": np.zeros(shape),
            "P_13": np.zeros(shape),
            "P_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        # B along z → P_par = P_33 = 3
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
            "P_s0_11": np.full(shape, 1.0),
            "P_s0_22": np.full(shape, 1.0),
            "P_s0_33": np.full(shape, 3.0),
            "P_s0_12": np.zeros(shape),
            "P_s0_13": np.zeros(shape),
            "P_s0_23": np.zeros(shape),
            # Species 1 (ions): P_diag = (2, 4, 6)
            "P_s1_11": np.full(shape, 2.0),
            "P_s1_22": np.full(shape, 4.0),
            "P_s1_33": np.full(shape, 6.0),
            "P_s1_12": np.zeros(shape),
            "P_s1_13": np.zeros(shape),
            "P_s1_23": np.zeros(shape),
            # Total tensor = sum of per-species
            "P_11": np.full(shape, 3.0),
            "P_22": np.full(shape, 5.0),
            "P_33": np.full(shape, 9.0),
            "P_12": np.zeros(shape),
            "P_13": np.zeros(shape),
            "P_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
        }
        return make_test_dataset(data, shape=shape)

    def test_per_species_parallel_pressure(self):
        ds = self._make_species_tensor_dataset()
        # B along z → P_par = P_33
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
            "P_s0_11": p,
            "P_s0_22": p,
            "P_s0_33": p,
            "P_s0_12": np.zeros(shape),
            "P_s0_13": np.zeros(shape),
            "P_s0_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        # Isotropic tensor → Q = 0
        np.testing.assert_allclose(compute_field("agyrotropy_e", ds), 0.0, atol=1e-15)

    def test_alias_resolution(self):
        ds = self._make_species_tensor_dataset()
        # Tier-3 canonical (``P_s0_par``) and the e/i shorthand alias
        # (``P_par_e``) give the same result.
        result_canonical = compute_field("P_s0_par", ds)
        result_alias = compute_field("P_par_e", ds)
        np.testing.assert_array_equal(result_canonical, result_alias)

    def test_total_equals_sum_of_per_species(self):
        """P_par = P_s0_par + P_s1_par (linear in tensor components)."""
        ds = self._make_species_tensor_dataset()
        total = compute_field("P_par", ds)
        per_species_sum = compute_field("P_s0_par", ds) + compute_field("P_s1_par", ds)
        np.testing.assert_allclose(total, per_species_sum, rtol=1e-15)

        total_perp = compute_field("P_perp", ds)
        perp_sum = compute_field("P_s0_perp", ds) + compute_field("P_s1_perp", ds)
        np.testing.assert_allclose(total_perp, perp_sum, rtol=1e-15)

    def test_dynamic_species_without_species_metadata(self):
        """P_s2_par works when tensor fields exist but species[2] is not."""
        shape = (2, 2, 2)
        data = {
            "P_s2_11": np.full(shape, 1.0),
            "P_s2_22": np.full(shape, 2.0),
            "P_s2_33": np.full(shape, 4.0),
            "P_s2_12": np.zeros(shape),
            "P_s2_13": np.zeros(shape),
            "P_s2_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        # B along z → P_par = P_33 = 4
        np.testing.assert_allclose(compute_field("P_s2_par", ds), 4.0, rtol=1e-15)


class TestFieldAlignedVectorDecomposition:
    """Total J/V/E parallel-perpendicular split via compute registry."""

    def _make_vector_dataset(self):
        """B along z; J, V, E each set to distinct (1,2,3)-style vectors."""
        shape = (2, 2, 2)
        data = {
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "J_1": np.full(shape, 1.0),
            "J_2": np.full(shape, 2.0),
            "J_3": np.full(shape, 3.0),
            "V_1": np.full(shape, 4.0),
            "V_2": np.full(shape, 5.0),
            "V_3": np.full(shape, 6.0),
            "E_1": np.full(shape, 7.0),
            "E_2": np.full(shape, 8.0),
            "E_3": np.full(shape, 9.0),
        }
        return make_test_dataset(data, shape=shape)

    def test_parallel_picks_z_component(self):
        ds = self._make_vector_dataset()
        # B along z: A_par = A_3
        np.testing.assert_allclose(compute_field("J_par", ds), 3.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("V_par", ds), 6.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("E_par", ds), 9.0, rtol=1e-14)

    def test_perpendicular_components(self):
        ds = self._make_vector_dataset()
        # B along z: A_perp = (A_1, A_2, 0)
        np.testing.assert_allclose(compute_field("J_perp_1", ds), 1.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("J_perp_2", ds), 2.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("J_perp_3", ds), 0.0, atol=1e-15)
        np.testing.assert_allclose(compute_field("V_perp_1", ds), 4.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("E_perp_3", ds), 0.0, atol=1e-15)

    def test_perpendicular_magnitudes(self):
        ds = self._make_vector_dataset()
        # |J_perp| = sqrt(1+4) = sqrt(5), |V_perp| = sqrt(16+25) = sqrt(41),
        # |E_perp| = sqrt(49+64) = sqrt(113)
        np.testing.assert_allclose(
            compute_field("|J_perp|", ds), np.sqrt(5.0), rtol=1e-14
        )
        np.testing.assert_allclose(
            compute_field("|V_perp|", ds), np.sqrt(41.0), rtol=1e-14
        )
        np.testing.assert_allclose(
            compute_field("|E_perp|", ds), np.sqrt(113.0), rtol=1e-14
        )

    def test_non_ideal_residual_parallel(self):
        ds = self._make_vector_dataset()
        # V=(4,5,6), B=(0,0,1) → V×B = (5,-4,0); E' = E + V×B = (12, 4, 9)
        # B along z → E'_par = E'_3 = 9.0
        np.testing.assert_allclose(compute_field("E_prime_par", ds), 9.0, rtol=1e-14)
        # |E'_perp| = sqrt(12^2 + 4^2) = sqrt(160)
        np.testing.assert_allclose(
            compute_field("|E_prime_perp|", ds), np.sqrt(160.0), rtol=1e-14
        )

    @pytest.mark.parametrize("prefix", ["J", "V", "E", "E_prime", "E_ideal", "E_Hall"])
    def test_zero_b_propagates_nan(self, prefix):
        shape = (2, 2, 2)
        data = {
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
            "J_1": np.full(shape, 1.0),
            "J_2": np.full(shape, 2.0),
            "J_3": np.full(shape, 3.0),
            "V_1": np.full(shape, 4.0),
            "V_2": np.full(shape, 5.0),
            "V_3": np.full(shape, 6.0),
            "E_1": np.full(shape, 7.0),
            "E_2": np.full(shape, 8.0),
            "E_3": np.full(shape, 9.0),
            "n_s0": np.full(shape, 1.0),  # required by E_Hall
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS])
        assert np.all(np.isnan(compute_field(f"{prefix}_par", ds)))
        assert np.all(np.isnan(compute_field(f"|{prefix}_perp|", ds)))


class TestPerSpeciesFieldAlignedDecomposition:
    """Per-species V_par, V_perp_{1,2,3}, |V_perp| via species template."""

    def _make_dataset(self):
        shape = (2, 2, 2)
        data = {
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "V_s0_1": np.full(shape, 1.0),
            "V_s0_2": np.full(shape, 2.0),
            "V_s0_3": np.full(shape, 3.0),
            "V_s1_1": np.full(shape, 10.0),
            "V_s1_2": np.full(shape, 20.0),
            "V_s1_3": np.full(shape, 30.0),
        }
        return make_test_dataset(data, shape=shape)

    def test_per_species_parallel(self):
        ds = self._make_dataset()
        # B along z → V_par_s{N} = V_s{N}_3
        np.testing.assert_allclose(compute_field("V_s0_par", ds), 3.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("V_s1_par", ds), 30.0, rtol=1e-14)

    def test_per_species_perp_components(self):
        ds = self._make_dataset()
        np.testing.assert_allclose(compute_field("V_s0_perp_1", ds), 1.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("V_s0_perp_2", ds), 2.0, rtol=1e-14)
        np.testing.assert_allclose(compute_field("V_s0_perp_3", ds), 0.0, atol=1e-15)
        np.testing.assert_allclose(compute_field("V_s1_perp_1", ds), 10.0, rtol=1e-14)

    def test_per_species_perp_magnitude(self):
        ds = self._make_dataset()
        np.testing.assert_allclose(
            compute_field("|V_s0_perp|", ds), np.sqrt(5.0), rtol=1e-14
        )
        np.testing.assert_allclose(
            compute_field("|V_s1_perp|", ds), np.sqrt(500.0), rtol=1e-14
        )

    def test_alias_resolution(self):
        ds = self._make_dataset()
        # ``V_par_e`` and ``V_s0_par`` resolve to the same array.
        np.testing.assert_array_equal(
            compute_field("V_par_e", ds), compute_field("V_s0_par", ds)
        )
        np.testing.assert_array_equal(
            compute_field("V_par_i", ds), compute_field("V_s1_par", ds)
        )

    def test_dynamic_species_index(self):
        """V_s2_par synthesizes for species[2] without metadata."""
        shape = (2, 2, 2)
        data = {
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "V_s2_1": np.full(shape, 1.0),
            "V_s2_2": np.full(shape, 2.0),
            "V_s2_3": np.full(shape, 7.0),
        }
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_allclose(compute_field("V_s2_par", ds), 7.0, rtol=1e-14)
        np.testing.assert_allclose(
            compute_field("|V_s2_perp|", ds), np.sqrt(5.0), rtol=1e-14
        )


class TestIdealAndHallDecomposition:
    """Analytic identities for E_ideal and E_Hall decomposition.

    Both fields are cross products with B (E_ideal = -V×B,
    E_Hall ∝ J×B), so the parallel component is identically zero up
    to floating-point roundoff and the perpendicular vector equals
    the full field.
    """

    def _make_dataset(self):
        # Mix V, J, B so the cross products are non-trivial in every
        # component and the parallel-component test is non-vacuous.
        # ``E_Hall`` requires the electron species charge from species[0].
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 0.3),
            "B_2": np.full(shape, -0.4),
            "B_3": np.full(shape, 0.5),
            "V_1": np.full(shape, 1.0),
            "V_2": np.full(shape, 2.0),
            "V_3": np.full(shape, -3.0),
            "J_1": np.full(shape, 0.1),
            "J_2": np.full(shape, -0.2),
            "J_3": np.full(shape, 0.05),
            "n_s0": np.full(shape, 1.0),
        }
        return make_test_dataset(data, shape=shape, species=[ELECTRONS])

    def test_e_ideal_par_is_zero(self):
        ds = self._make_dataset()
        result = compute_field("E_ideal_par", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_e_hall_par_is_zero(self):
        ds = self._make_dataset()
        result = compute_field("E_Hall_par", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_e_ideal_perp_equals_e_ideal(self):
        """A_perp == A when A is perpendicular to B by construction."""
        ds = self._make_dataset()
        for c in (1, 2, 3):
            full = compute_field(f"E_ideal_{c}", ds)
            perp = compute_field(f"E_ideal_perp_{c}", ds)
            np.testing.assert_allclose(perp, full, rtol=1e-14, atol=1e-14)

    def test_e_hall_perp_equals_e_hall(self):
        ds = self._make_dataset()
        for c in (1, 2, 3):
            full = compute_field(f"E_Hall_{c}", ds)
            perp = compute_field(f"E_Hall_perp_{c}", ds)
            np.testing.assert_allclose(perp, full, rtol=1e-14, atol=1e-14)

    def test_perp_magnitude_equals_full_magnitude(self):
        """|A_perp| == |A| when A_par == 0.  ``|E_ideal|`` and ``|E_Hall|``
        are not registered as recipes; compute the full magnitude inline."""
        ds = self._make_dataset()

        def _full_mag(prefix: str) -> np.ndarray:
            a1 = compute_field(f"{prefix}_1", ds)
            a2 = compute_field(f"{prefix}_2", ds)
            a3 = compute_field(f"{prefix}_3", ds)
            return np.sqrt(a1**2 + a2**2 + a3**2)

        np.testing.assert_allclose(
            compute_field("|E_ideal_perp|", ds),
            _full_mag("E_ideal"),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            compute_field("|E_Hall_perp|", ds),
            _full_mag("E_Hall"),
            rtol=1e-14,
            atol=1e-14,
        )


class TestFieldAlignedPythagoreanIdentity:
    """Registry-level $|X_\\perp|^2 + X_\\parallel^2 = |X|^2$ cross-check.

    The function-level identity is exercised in ``test_derived.py``;
    this test guards the recipe wiring in ``compute.py`` so a future
    rewire of ``|X_perp|`` against the wrong inputs (or wrong
    magnitude function) surfaces here. B is intentionally not axis-
    aligned so the parallel/perpendicular split is non-trivial in
    every component. ``E_ideal`` and ``E_Hall`` are excluded — their
    parallel projection is analytically zero, so the identity
    degenerates to ``|X_perp| = |X|`` already covered at
    ``TestIdealAndHallDecomposition``.
    """

    @pytest.mark.parametrize("prefix", ["J", "V", "E", "E_prime"])
    def test_pythagorean_identity(self, prefix):
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 0.3),
            "B_2": np.full(shape, -0.4),
            "B_3": np.full(shape, 0.5),
            "J_1": np.full(shape, 1.0),
            "J_2": np.full(shape, 2.0),
            "J_3": np.full(shape, 3.0),
            "V_1": np.full(shape, 4.0),
            "V_2": np.full(shape, -5.0),
            "V_3": np.full(shape, 6.0),
            "E_1": np.full(shape, 7.0),
            "E_2": np.full(shape, 8.0),
            "E_3": np.full(shape, -9.0),
        }
        ds = make_test_dataset(data, shape=shape)
        a_par = compute_field(f"{prefix}_par", ds)
        a_perp_mag = compute_field(f"|{prefix}_perp|", ds)
        a1 = compute_field(f"{prefix}_1", ds)
        a2 = compute_field(f"{prefix}_2", ds)
        a3 = compute_field(f"{prefix}_3", ds)
        a_sq = a1**2 + a2**2 + a3**2
        np.testing.assert_allclose(
            a_par**2 + a_perp_mag**2, a_sq, rtol=1e-12, atol=1e-12
        )


class TestGeometryGuard:
    @pytest.mark.parametrize(
        ("field", "components"),
        [
            ("div_B", {"B_1", "B_2", "B_3"}),
            ("vort_1", {"V_1", "V_2", "V_3"}),
        ],
    )
    def test_compute_rejects_spherical(self, field, components):
        shape = (4, 4, 4)
        grid = GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0), geometry=SPHERICAL)
        data = {name: np.ones(shape) for name in components}
        ds = FieldDataset.from_arrays(data, grid, Normalization.identity())
        with pytest.raises(NotImplementedError, match="Cartesian"):
            compute_field(field, ds)

    @pytest.mark.parametrize(
        ("field", "components"),
        [
            ("div_B", {"B_1", "B_2", "B_3"}),
            ("div_E", {"E_1", "E_2", "E_3"}),
            ("curl_B_1", {"B_1", "B_2", "B_3"}),
            ("vort_1", {"V_1", "V_2", "V_3"}),
        ],
    )
    def test_compute_rejects_2d_grid(self, field, components):
        shape = (4, 4)
        data = {name: np.ones(shape) for name in components}
        ds = make_test_dataset(data, shape=shape)
        with pytest.raises(NotImplementedError, match=rf"{field}.*2D"):
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
                    "B_1": np.ones((4, 3, 2)),
                    "B_2": np.zeros((4, 3, 2)),
                    "B_3": np.zeros((4, 3, 2)),
                    "E_1": np.ones((4, 3, 2)),
                    "E_2": np.zeros((4, 3, 2)),
                    "E_3": np.zeros((4, 3, 2)),
                },
            )
            result = compute_field("e_mag_ratio", ds)
            assert result.shape == (4, 3, 2)
            assert "e_mag_ratio" in available_quantities()
        finally:
            unregister_recipe("e_mag_ratio")

    def test_unregister_removes_recipe_and_metadata(self):
        from pypic.fields import field_info

        register_recipe(
            "_test_tmp",
            func=lambda b: b * 2,
            fields=("|B|",),
            quantity_type="b_field",
        )
        assert "_test_tmp" in _REGISTRY
        # Metadata is present while registered.
        assert field_info("_test_tmp").quantity_type == "b_field"
        unregister_recipe("_test_tmp")
        assert "_test_tmp" not in _REGISTRY
        # Test name promises "and metadata" — verify the metadata side too,
        # so a regression that forgets to unregister the field info is caught.
        with pytest.raises(KeyError):
            field_info("_test_tmp")

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
            # Under identity normalization every quantity_type gives 1.0, so
            # that alone cannot show the registered type was honoured.  Use a
            # normalization where b_field differs from density to prove the
            # factor actually came from quantity_type="b_field".
            norm = Normalization(
                length_ref=1.0,
                time_ref=1.0,
                velocity_ref=1.0,
                b_field_ref=7.0,
                e_field_ref=1.0,
                density_ref=3.0,
                mass_ref=1.0,
                charge_ref=1.0,
            )
            assert field_si_factor("_test_si", norm) == pytest.approx(7.0)
        finally:
            unregister_recipe("_test_si")


class TestReconnectionDiagnostics:
    """Registry-level dispatch for the Phase-1 reconnection additions."""

    def _ideal_mhd_dataset(self, shape=(4, 3, 2)):
        """Field set with E = -V × B identically (ideal MHD)."""
        rng = np.random.default_rng(2026)
        v = rng.standard_normal((3, *shape))
        b = rng.standard_normal((3, *shape))
        e = np.stack(
            [
                -(v[1] * b[2] - v[2] * b[1]),
                -(v[2] * b[0] - v[0] * b[2]),
                -(v[0] * b[1] - v[1] * b[0]),
            ]
        )
        data = {
            "V_1": v[0],
            "V_2": v[1],
            "V_3": v[2],
            "B_1": b[0],
            "B_2": b[1],
            "B_3": b[2],
            "E_1": e[0],
            "E_2": e[1],
            "E_3": e[2],
            "J_1": rng.standard_normal(shape),
            "J_2": rng.standard_normal(shape),
            "J_3": rng.standard_normal(shape),
            "V_s0_1": v[0],
            "V_s0_2": v[1],
            "V_s0_3": v[2],
            "rho_c": np.zeros(shape),
            "rho_m": np.ones(shape),
        }
        return make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])

    def test_R_recon_zero_in_ideal_mhd(self):
        """E = -V × B → |E'| = 0 → R_recon = 0 everywhere."""
        ds = self._ideal_mhd_dataset()
        result = compute_field("R_recon", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_D_e_zero_in_ideal_mhd_with_zero_rho_c(self):
        """V_e = V, ρ_c = 0 → D_e = J · (E + V × B) - 0 = 0."""
        ds = self._ideal_mhd_dataset()
        result = compute_field("D_e", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-13)

    def test_D_ng_zero_on_gyrotropic_dataset(self):
        """Diagonal pressure tensor aligned with B → D_ng = 0."""
        shape = (3, 2, 2)
        b = np.zeros((3, *shape))
        b[2] = 1.0  # B along z
        data = {
            "B_1": b[0],
            "B_2": b[1],
            "B_3": b[2],
            "P_11": np.full(shape, 2.0),
            "P_22": np.full(shape, 2.0),
            "P_33": np.full(shape, 5.0),
            "P_12": np.zeros(shape),
            "P_13": np.zeros(shape),
            "P_23": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("D_ng", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_A_phi_zero_on_gyrotropic_dataset(self):
        """Diagonal pressure tensor with perp eigenvalues equal → A_phi = 0."""
        shape = (3, 2, 2)
        b = np.zeros((3, *shape))
        b[2] = 1.0
        data = {
            "B_1": b[0],
            "B_2": b[1],
            "B_3": b[2],
            "P_11": np.full(shape, 2.0),
            "P_22": np.full(shape, 2.0),
            "P_33": np.full(shape, 5.0),
            "P_12": np.zeros(shape),
            "P_13": np.zeros(shape),
            "P_23": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("A_phi", ds)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_per_species_D_ng_alias(self):
        """D_ng_e and D_ng_s0 both dispatch to the species template."""
        shape = (2, 2, 2)
        b = np.zeros((3, *shape))
        b[2] = 1.0
        data = {
            "B_1": b[0],
            "B_2": b[1],
            "B_3": b[2],
            "P_s0_11": np.full(shape, 3.0),
            "P_s0_22": np.full(shape, 1.0),
            "P_s0_33": np.full(shape, 1.0),
            "P_s0_12": np.zeros(shape),
            "P_s0_13": np.zeros(shape),
            "P_s0_23": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        from_alias = compute_field("D_ng_e", ds)
        from_canonical = compute_field("D_ng_s0", ds)
        np.testing.assert_allclose(from_alias, from_canonical, rtol=1e-15)
        # Closed-form value: P_par=1, P_perp=2, ||N||_F²=2, D_ng=2√2/5.
        np.testing.assert_allclose(from_canonical, 2.0 * np.sqrt(2.0) / 5.0, rtol=1e-14)

    def test_per_species_A_phi_alias(self):
        """A_phi_e and A_phi_s0 both dispatch through the species template."""
        shape = (2, 2, 2)
        b = np.zeros((3, *shape))
        b[2] = 1.0
        data = {
            "B_1": b[0],
            "B_2": b[1],
            "B_3": b[2],
            "P_s0_11": np.full(shape, 3.0),
            "P_s0_22": np.full(shape, 1.0),
            "P_s0_33": np.full(shape, 5.0),
            "P_s0_12": np.zeros(shape),
            "P_s0_13": np.zeros(shape),
            "P_s0_23": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        from_alias = compute_field("A_phi_e", ds)
        from_canonical = compute_field("A_phi_s0", ds)
        np.testing.assert_allclose(from_alias, from_canonical, rtol=1e-15)
        # Perp eigenvalues 3, 1 → A_phi = 2/4 = 0.5.
        np.testing.assert_allclose(from_canonical, 0.5, rtol=1e-14)

    def test_per_species_R_recon_alias(self):
        """R_recon_e and R_recon_s0 both dispatch through the species template.

        Uses the ideal-MHD fixture, in which V_s0 equals V identically, so
        the electron-frame rate also evaluates to zero.  The literal-vs-
        species mix (E_, B_, v_A literal; V_s{N}_* substituted) exercises
        the same template shape used by V_par / V_perp.
        """
        ds = self._ideal_mhd_dataset()
        from_alias = compute_field("R_recon_e", ds)
        from_canonical = compute_field("R_recon_s0", ds)
        np.testing.assert_allclose(from_alias, from_canonical, rtol=1e-15)
        np.testing.assert_allclose(from_canonical, 0.0, atol=1e-14)


def _rotation_from_e1_to(bhat: np.ndarray) -> np.ndarray:
    r"""Return the 3x3 rotation $R$ such that $R\hat{e}_1 = \hat{b}$.

    Rodrigues' formula about the axis $\hat{e}_1 \times \hat{b}$. Handles
    the parallel ($\hat{b} = \pm\hat{e}_1$) edge cases. Used only for
    test-fixture construction; no `src/` analogue.
    """
    bhat = bhat / np.linalg.norm(bhat)
    e1 = np.array([1.0, 0.0, 0.0])
    if np.allclose(bhat, e1):
        return np.eye(3)
    if np.allclose(bhat, -e1):
        return np.diag([-1.0, -1.0, 1.0])
    axis = np.cross(e1, bhat)
    axis_n = axis / np.linalg.norm(axis)
    cos_t = float(np.dot(e1, bhat))
    sin_t = float(np.linalg.norm(axis))
    k_mat = np.array(
        [
            [0.0, -axis_n[2], axis_n[1]],
            [axis_n[2], 0.0, -axis_n[0]],
            [-axis_n[1], axis_n[0], 0.0],
        ]
    )
    return np.eye(3) + sin_t * k_mat + (1.0 - cos_t) * (k_mat @ k_mat)  # type: ignore[no-any-return]


def _diag_align(p11: float, p22: float, p33: float) -> np.ndarray:
    r"""Diagonal field-aligned tensor with $\hat{b} = \hat{e}_1$."""
    return np.diag([p11, p22, p33])


def _offaxis_align(delta: float) -> np.ndarray:
    r"""$\mathbf{I} + \delta\,(\hat{e}_1\hat{e}_3^T + \hat{e}_3\hat{e}_1^T)$.

    Pure off-axis nongyrotropy: equal perpendicular eigenvalues, off-axis
    $P_{13}$ coupling. $A_\phi = 0$ in closed form; $Q$ and $D_{ng}$ both
    detect the coupling.
    """
    p = np.eye(3)
    p[0, 2] = delta
    p[2, 0] = delta
    return p


def _diag_plus_offaxis(delta: float) -> np.ndarray:
    """Combine perp anisotropy with off-axis coupling."""
    p = np.diag([1.0, 2.0, 0.5])
    p[0, 2] = delta
    p[2, 0] = delta
    return p


def _build_fixture(
    p_align: np.ndarray, r_mat: np.ndarray, bhat_lab: np.ndarray
) -> FieldDataset:
    """Build a (2,2,2) FieldDataset with spatially-uniform P (rotated to
    lab frame via R) and uniform B = bhat_lab.

    Asserts the rotated tensor stays symmetric as a construction guard.
    """
    p_lab = r_mat @ p_align @ r_mat.T
    np.testing.assert_allclose(
        p_lab, p_lab.T, atol=1e-15, err_msg="rotation lost symmetry"
    )
    shape = (2, 2, 2)
    data = {
        "P_11": np.full(shape, p_lab[0, 0]),
        "P_22": np.full(shape, p_lab[1, 1]),
        "P_33": np.full(shape, p_lab[2, 2]),
        "P_12": np.full(shape, p_lab[0, 1]),
        "P_13": np.full(shape, p_lab[0, 2]),
        "P_23": np.full(shape, p_lab[1, 2]),
        "B_1": np.full(shape, bhat_lab[0]),
        "B_2": np.full(shape, bhat_lab[1]),
        "B_3": np.full(shape, bhat_lab[2]),
    }
    return make_test_dataset(data, shape=shape)


def _build_per_species_fixture(
    p_align: np.ndarray, r_mat: np.ndarray, bhat_lab: np.ndarray
) -> FieldDataset:
    """Like `_build_fixture` but with `P_s0_*` names and a species list."""
    p_lab = r_mat @ p_align @ r_mat.T
    np.testing.assert_allclose(
        p_lab, p_lab.T, atol=1e-15, err_msg="rotation lost symmetry"
    )
    shape = (2, 2, 2)
    data = {
        "P_s0_11": np.full(shape, p_lab[0, 0]),
        "P_s0_22": np.full(shape, p_lab[1, 1]),
        "P_s0_33": np.full(shape, p_lab[2, 2]),
        "P_s0_12": np.full(shape, p_lab[0, 1]),
        "P_s0_13": np.full(shape, p_lab[0, 2]),
        "P_s0_23": np.full(shape, p_lab[1, 2]),
        "B_1": np.full(shape, bhat_lab[0]),
        "B_2": np.full(shape, bhat_lab[1]),
        "B_3": np.full(shape, bhat_lab[2]),
    }
    return make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])


class TestAgyrotropyManufacturedFixtures:
    r"""Closed-form values for Q, D_ng, A_phi on manufactured pressure-
    tensor configurations with a tilted $\hat{b}$, asserted through the
    ``compute()`` registry path.

    Three configurations exercise three distinct pieces of the algebra:

    - **C1** perp anisotropy only: $\mathbf{P}_{\mathrm{align}} =
      \mathrm{diag}(1, 2, 0.5)$. $Q = 9/65$, $D_{ng} = 3\sqrt{2}/7$,
      $A_\phi = 3/5$.
    - **C2** off-axis coupling only: $\mathbf{I} + 0.3\,(\hat{e}_1
      \hat{e}_3^T + \hat{e}_3\hat{e}_1^T)$. $Q = 3/100$, $D_{ng} =
      \sqrt{2}/5$, $A_\phi = 0$. Catches a regression where
      ``agyrotropy`` returns $A_\phi^2$ instead of Swisdak's $Q$
      (the bug fixed by ``b8ecb40``).
    - **C3** combined: $\mathrm{diag}(1, 2, 0.5) + 0.2\,(\hat{e}_1
      \hat{e}_3^T + \hat{e}_3\hat{e}_1^T)$. $Q = 241/1625$, $D_{ng} =
      \sqrt{482}/35$, $A_\phi = 3/5$ — same as C1 because $A_\phi$ is
      blind to off-axis terms by construction (double-projection
      strips them).

    References
    ----------
    - $Q$: [@Swisdak2016].
    - $D_{ng}$: [@Aunai2013].
    - $A_\phi$: [@Scudder2008].
    """

    _BHAT_LAB = np.array([0.5, 0.0, np.sqrt(3.0) / 2.0])

    def _rotation(self) -> np.ndarray:
        return _rotation_from_e1_to(self._BHAT_LAB)

    def test_compute_path_matches_closed_form_on_tilted_b(self) -> None:
        """All three agyrotropy metrics match closed-form values through
        compute() on tilted-b̂ configurations. Aggregated failures so a
        regression in one config/metric reports its row.

        Closed-form derivations
        -----------------------
        C1: P_align = diag(1, 2, 0.5), b̂_align = ê_1.
            I_1 = 3.5, I_2 = 2 + 0.5 + 1 = 3.5,
            denom = (I_1 - P_∥)(I_1 + 3 P_∥) = 2.5 · 6.5 = 16.25,
            Q = 1 - 4·3.5/16.25 = 9/65.
            P_⊥ = 1.25, N = diag(0, 0.75, -0.75), ||N||_F^2 = 9/8,
            D_ng = 2√(9/8)/3.5 = 3√2/7.
            Perp eigenvalues (2, 0.5), A_φ = 1.5/2.5 = 3/5.
        C2: P_align = I + δ(ê_1 ê_3^T + ê_3 ê_1^T), δ = 0.3.
            I_1 = 3, I_2 = 3 - δ^2 = 2.91, denom = 12,
            Q = δ^2/3 = 3/100.
            P_∥ = P_⊥ = 1, ||N||_F^2 = 2δ^2,
            D_ng = 2√(2δ^2)/3 = √2/5.
            Perp 2×2 block = diag(1, 1) exactly, A_φ = 0.
        C3: P_align = diag(1, 2, 0.5) + δ(ê_1 ê_3^T + ê_3 ê_1^T), δ = 0.2.
            I_1 = 3.5, I_2 = 3.5 - δ^2 = 3.46, denom = 16.25,
            Q = 1 - 4·3.46/16.25 = 241/1625.
            P_⊥ = 1.25, ||N||_F^2 = 9/8 + 2δ^2 = 1.205,
            D_ng = 2√1.205/3.5 = √482/35.
            Perp double-projection strips off-axis term → diag(2, 0.5),
            A_φ = 3/5 (same as C1, on purpose — A_φ is blind to δ).
        """
        bhat = self._BHAT_LAB
        r_mat = self._rotation()
        configs = [
            (
                "C1 perp",
                _diag_align(1.0, 2.0, 0.5),
                9.0 / 65.0,
                3.0 * np.sqrt(2.0) / 7.0,
                3.0 / 5.0,
            ),
            (
                "C2 offaxis",
                _offaxis_align(0.3),
                3.0 / 100.0,
                np.sqrt(2.0) / 5.0,
                0.0,
            ),
            (
                "C3 combined",
                _diag_plus_offaxis(0.2),
                241.0 / 1625.0,
                np.sqrt(482.0) / 35.0,
                3.0 / 5.0,
            ),
        ]
        failures: list[str] = []
        for label, p_align, q_exact, dng_exact, aphi_exact in configs:
            ds = _build_fixture(p_align, r_mat, bhat)
            q = float(compute_field("agyrotropy", ds).flat[0])
            dng = float(compute_field("D_ng", ds).flat[0])
            aphi = float(compute_field("A_phi", ds).flat[0])
            # Q and D_ng: closed-form-arithmetic precision (1e-13).
            # A_phi: clamp-then-sqrt in scudder_agyrotropy lifts the
            # gyrotropic-roundoff floor to ~1e-8, so loosen atol for
            # the A_phi = 0 row of C2.
            if not np.isclose(q, q_exact, rtol=1e-13, atol=1e-13):
                failures.append(f"{label}/Q: got {q!r}, want {q_exact!r}")
            if not np.isclose(dng, dng_exact, rtol=1e-13, atol=1e-13):
                failures.append(f"{label}/D_ng: got {dng!r}, want {dng_exact!r}")
            if not np.isclose(aphi, aphi_exact, rtol=1e-12, atol=5e-8):
                failures.append(f"{label}/A_phi: got {aphi!r}, want {aphi_exact!r}")
        assert not failures, "\n".join(failures)

    def test_agyrotropy_is_not_aphi_squared(self) -> None:
        """Direct b8ecb40 sentinel: `compute("agyrotropy")` returns
        Swisdak's $Q$, not $A_\\phi^2$.

        On the off-axis-only fixture (C2), $Q = 3/100$ but $A_\\phi
        \\approx 0$. If the historical bug returned, `agyrotropy`
        would degenerate to $A_\\phi^2 \\approx 0$ and this assertion
        would fail. Tight enough threshold to catch the substitution;
        loose enough that floating-point noise around $A_\\phi^2 \\sim
        10^{-16}$ never trips it.
        """
        ds = _build_fixture(_offaxis_align(0.3), self._rotation(), self._BHAT_LAB)
        q = float(compute_field("agyrotropy", ds).flat[0])
        aphi = float(compute_field("A_phi", ds).flat[0])
        assert abs(q - aphi**2) > 1e-3, (
            f"agyrotropy appears to be returning A_phi^2: "
            f"Q = {q!r}, A_phi^2 = {aphi**2!r}"
        )

    def test_per_species_template_on_combined_fixture(self) -> None:
        """The `SpeciesTemplate` path delivers the same closed-form
        values on a non-trivial tensor.

        Existing `test_per_species_agyrotropy_isotropic` only exercises
        the trivial $Q = 0$ case; this is the per-species analogue of
        the C3 row of the joint test.
        """
        ds = _build_per_species_fixture(
            _diag_plus_offaxis(0.2), self._rotation(), self._BHAT_LAB
        )
        q = float(compute_field("agyrotropy_s0", ds).flat[0])
        dng = float(compute_field("D_ng_s0", ds).flat[0])
        aphi = float(compute_field("A_phi_s0", ds).flat[0])
        np.testing.assert_allclose(q, 241.0 / 1625.0, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(dng, np.sqrt(482.0) / 35.0, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(aphi, 3.0 / 5.0, rtol=1e-12, atol=5e-8)
