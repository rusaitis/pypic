"""Tests for structured (underscore-separated) field and compute aliases."""

import numpy as np
import pytest

from pypic.compute import (
    _COMPUTE_ALIASES,
    compute_field,
    field_si_factor,
)
from pypic.dataset import _default_aliases
from pypic.units import Normalization, SpeciesInfo
from tests._helpers import ELECTRONS, IONS, make_test_dataset

ALPHAS = SpeciesInfo(name="alphas", charge=2.0, mass=4.0)


class TestUnderscoreFieldAliases:
    """Underscore-separated names resolve through FieldDataset."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("B_x", "B_1"),
            ("B_y", "B_2"),
            ("B_z", "B_3"),
            ("E_x", "E_1"),
            ("V_y", "V_2"),
            ("J_z", "J_3"),
        ],
    )
    def test_cartesian_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 7.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("B_1", "B_1"), ("E_2", "E_2"), ("V_3", "V_3"), ("J_1", "J_1")],
    )
    def test_numbered_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 3.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("P_e", "Pe"),
            ("P_i", "Pi"),
            ("T_e", "Te"),
            ("T_i", "Ti"),
            ("P_11", "P_11"),
            ("P_12", "P_12"),
            ("P_33", "P_33"),
        ],
    )
    def test_scalar_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 5.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    def test_has_field_with_underscore_alias(self):
        shape = (2, 2, 2)
        ds = make_test_dataset({"B_1": np.ones(shape)}, shape=shape)
        assert ds.has_field("B_x")
        assert ds.has_field("B_1")


class TestMagnitudeComputeAliases:
    """_mag suffix aliases resolve through compute_field."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("B_mag", "|B|"),
            ("Bmag", "|B|"),
            ("E_mag", "|E|"),
            ("Emag", "|E|"),
            ("J_mag", "|J|"),
            ("Jmag", "|J|"),
            ("V_mag", "|V|"),
            ("Vmag", "|V|"),
        ],
    )
    def test_magnitude_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        prefix = canonical[1]  # B, E, J, or V
        data = {
            f"{prefix}1": np.full(shape, 3.0),
            f"{prefix}2": np.full(shape, 4.0),
            f"{prefix}3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result_alias = compute_field(alias, ds)
        result_canonical = compute_field(canonical, ds)
        np.testing.assert_array_equal(result_alias, result_canonical)


class TestDescriptiveComputeAliases:
    """Descriptive names resolve to canonical compute entries."""

    def test_plasma_beta(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 25.0),
            "B_1": np.full(shape, 3.0),
            "B_2": np.full(shape, 4.0),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("plasma_beta", ds)
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)

    def test_v_alfven(self):
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
            "rho_m": np.full(shape, 4.0),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("alfven_speed", ds)
        np.testing.assert_allclose(result, 0.5, rtol=1e-15)

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("energy_magnetic", "e_B"),
            ("energy_electric", "e_E"),
            ("energy_kinetic", "e_k"),
            ("energy_thermal", "e_th"),
            ("enthalpy", "h"),
            ("energy_internal", "e_int"),
            ("c_ms", "v_ms"),
        ],
    )
    def test_descriptive_aliases_resolve(self, alias, canonical):
        assert _COMPUTE_ALIASES[alias] == canonical


class TestStructuredSpeciesComputeAliases:
    """Species-indexed structured aliases resolve correctly."""

    def test_omega_p_s0(self):
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 4.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("omega_p_s0", ds)
        expected = compute_field("omega_pe", ds)
        np.testing.assert_array_equal(result, expected)

    def test_d_s1(self):
        shape = (2, 2, 2)
        data = {"n_s1": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("d_s1", ds)
        expected = compute_field("d_i", ds)
        np.testing.assert_array_equal(result, expected)

    def test_larmor_radius_s0(self):
        shape = (2, 2, 2)
        data = {
            "Te": np.full(shape, 1.0),
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("larmor_radius_s0", ds)
        expected = compute_field("r_e", ds)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            # Species-coupled scales: Tier-3 ``_sN`` form is canonical;
            # the literature spellings (NRL Plasma Formulary) alias to it.
            ("omega_pe", "omega_p_s0"),
            ("omega_pi", "omega_p_s1"),
            ("omega_ce", "omega_c_s0"),
            ("omega_ci", "omega_c_s1"),
            ("d_e", "d_s0"),
            ("d_i", "d_s1"),
            ("v_th_e", "v_th_s0"),
            ("v_th_i", "v_th_s1"),
            ("r_e", "r_s0"),
            ("r_i", "r_s1"),
            ("lambda_D", "lambda_D_s0"),
            # Long-form descriptive aliases also resolve to the Tier-3 canonical.
            ("v_thermal_s0", "v_th_s0"),
            ("v_thermal_s1", "v_th_s1"),
            ("rL_s0", "r_s0"),
            ("rL_s1", "r_s1"),
            ("larmor_radius_s0", "r_s0"),
            ("larmor_radius_s1", "r_s1"),
            # Moment quantities: _sN is canonical; the e/i convenience
            # name aliases down to it.
            ("Pe", "P_s0"),
            ("Pi", "P_s1"),
            ("Te", "T_s0"),
            ("Ti", "T_s1"),
            ("beta_e", "beta_s0"),
            ("beta_i", "beta_s1"),
            ("s_e", "s_s0"),
            ("s_i", "s_s1"),
        ],
    )
    def test_species_aliases_resolve(self, alias, canonical):
        assert _COMPUTE_ALIASES[alias] == canonical


class TestOperatorAliases:
    """Underscore-separated operator/component aliases resolve correctly."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("curl_B_x", "curl_B_1"),
            ("curl_B_y", "curl_B_2"),
            ("curl_B_z", "curl_B_3"),
            ("S_x", "S_1"),
            ("S_z", "S_3"),
        ],
    )
    def test_operator_aliases_resolve(self, alias, canonical):
        # Tier-3 canonicals (curl_B_1, vort_2, S_3, ...) are direct
        # registry entries — they resolve through ``_REGISTRY``, not
        # ``_COMPUTE_ALIASES``. Only the x/y/z spellings are aliases.
        assert _COMPUTE_ALIASES[alias] == canonical

    def test_curl_b_x_computes(self):
        shape = (4, 4, 4)
        data = {
            "B_1": np.ones(shape),
            "B_2": np.ones(shape),
            "B_3": np.ones(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("curl_B_x", ds)
        expected = compute_field("curl_B_1", ds)
        np.testing.assert_array_equal(result, expected)

    def test_s_x_computes(self):
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
        result = compute_field("S_x", ds)
        expected = compute_field("S_1", ds)
        np.testing.assert_array_equal(result, expected)


class TestSIConversionThroughAliases:
    """field_si_factor resolves aliases before SI lookup."""

    def test_bx_si_factor(self):
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
        assert field_si_factor("Bx", norm) == pytest.approx(b_ref)

    def test_b_x_underscore_si_factor(self):
        b_ref = 3.0
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
        assert field_si_factor("B_x", norm) == pytest.approx(b_ref)

    def test_p_e_underscore_si_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=2.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=3.0,
            mass_ref=4.0,
            charge_ref=1.0,
        )
        factor_canonical = field_si_factor("Pe", norm)
        factor_alias = field_si_factor("P_e", norm)
        assert factor_alias == pytest.approx(factor_canonical)

    def test_in_si_through_field_alias(self):
        shape = (2, 2, 2)
        b_ref = 2.0
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
        fields = {"B_1": np.full(shape, 5.0)}
        ds = make_test_dataset(fields, shape=shape, normalization=norm)
        result = ds.in_si("Bx")
        np.testing.assert_allclose(result, 5.0 * b_ref, rtol=1e-15)

    def test_in_si_through_underscore_alias(self):
        shape = (2, 2, 2)
        b_ref = 2.0
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
        fields = {"B_1": np.full(shape, 5.0)}
        ds = make_test_dataset(fields, shape=shape, normalization=norm)
        result = ds.in_si("B_x")
        np.testing.assert_allclose(result, 5.0 * b_ref, rtol=1e-15)

    def test_in_si_magnitude_alias(self):
        shape = (2, 2, 2)
        b_ref = 2.0
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
        data = {
            "B_1": np.full(shape, 3.0),
            "B_2": np.full(shape, 4.0),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, normalization=norm)
        result = ds.in_si("B_mag")
        np.testing.assert_allclose(result, 5.0 * b_ref, rtol=1e-15)


class TestFieldAliasIntegrity:
    """No collisions between field aliases and canonical names."""

    def test_no_underscore_field_alias_collides_with_canonical(self):
        from pypic.coordinates import CARTESIAN

        aliases = _default_aliases(CARTESIAN)
        # Per v1.0.x, ``_sN`` is canonical for moments (P_s0, T_s0, ...).
        # The bare-component names below are universally canonical.
        canonical_fields = {"B_1", "B_2", "B_3", "E_1", "E_2", "E_3"}
        overlap = set(aliases) & canonical_fields
        assert not overlap, f"Alias collision with canonical: {overlap}"

    def test_underscore_aliases_present_in_defaults(self):
        from pypic.coordinates import CARTESIAN

        aliases = _default_aliases(CARTESIAN)
        # Tier-3 alias spellings — canonical RHS is B_1 / P_s0 / T_s1.
        # B_1 / P_11 themselves are canonical, not aliases, so they
        # don't appear as alias LHS.
        assert "B_x" in aliases
        assert "P_e" in aliases
        assert "T_i" in aliases
        # Legacy short forms still resolve.
        assert "Bx" in aliases
        assert "B1" in aliases


class TestMultiSpeciesDynamicRecipes:
    """Dynamic species-template recipes for species index >= 2."""

    def test_omega_p_s2_computes(self):
        shape = (2, 2, 2)
        data = {"n_s2": np.full(shape, 4.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("omega_p_s2", ds)
        # omega_p = sqrt(n * q^2 / m) = sqrt(4 * 4 / 4) = sqrt(4) = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-14)

    def test_omega_c_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "B_1": np.full(shape, 2.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("omega_c_s2", ds)
        # omega_c = |q| * |B| / m = 2 * 2 / 4 = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-15)

    def test_d_s2_computes(self):
        shape = (2, 2, 2)
        data = {"n_s2": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("d_s2", ds)
        # d = c / omega_p = 1 / sqrt(1 * 4 / 4) = 1 / 1 = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)

    def test_v_th_s2_computes(self):
        shape = (2, 2, 2)
        # T_s2 is not raw data; it must be computed from P_s2 and n_s2
        data = {
            "P_s2": np.full(shape, 8.0),
            "n_s2": np.full(shape, 2.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("v_th_s2", ds)
        # T_s2 = P/n = 4, v_th = sqrt(T/m) = sqrt(4/4) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)

    def test_r_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "P_s2": np.full(shape, 4.0),
            "n_s2": np.full(shape, 1.0),
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("r_s2", ds)
        # T_s2 = 4, r = sqrt(m*T) / (|q|*B) = sqrt(4*4) / (2*1) = 4/2 = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-14)

    def test_lambda_d_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "P_s2": np.full(shape, 4.0),
            "n_s2": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("lambda_D_s2", ds)
        # T_s2 = 4, lambda_D = sqrt(T / (n * q^2)) = sqrt(4 / (1*4)) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)

    def test_beta_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "P_s2": np.full(shape, 5.0),
            "B_1": np.full(shape, 1.0),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("beta_s2", ds)
        # beta = 2*P / B^2 = 2*5 / 1 = 10
        np.testing.assert_allclose(result, 10.0, rtol=1e-15)

    def test_s_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "P_s2": np.full(shape, 1.0),
            "n_s2": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("s_s2", ds)
        # s = ln(P / n^gamma) = ln(1) = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_temperature_s2_computes(self):
        shape = (2, 2, 2)
        data = {
            "P_s2": np.full(shape, 6.0),
            "n_s2": np.full(shape, 3.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        result = compute_field("T_s2", ds)
        # T = P / n = 2
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)


class TestMultiSpeciesSIConversion:
    """SI factor resolution for per-species fields."""

    @pytest.mark.parametrize(
        ("field", "norm_kwargs", "expected"),
        [
            ("n_s2", {"density_ref": 5.0}, 5.0),
            ("omega_p_s2", {"time_ref": 0.5}, 2.0),
            ("v_th_s2", {"velocity_ref": 3.0}, 3.0),
            ("beta_s2", {}, 1.0),
        ],
        ids=["density", "frequency", "velocity", "dimensionless"],
    )
    def test_species_si_factor(self, field, norm_kwargs, expected):
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


class TestSpeciesNameAliases:
    """Species-name aliases generated from config species list."""

    def test_n_alphas_resolves_to_n_s2(self):
        shape = (2, 2, 2)
        data = {"n_s2": np.full(shape, 7.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        np.testing.assert_array_equal(ds["n_alphas"], ds["n_s2"])

    def test_n_e_n_i_aliases_take_priority(self):
        """Hardcoded n_e/n_i must not be overwritten by species-name logic."""
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0), "n_s1": np.full(shape, 2.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        np.testing.assert_allclose(ds["n_e"], 1.0)
        np.testing.assert_allclose(ds["n_i"], 2.0)

    def test_species_alias_not_added_without_data(self):
        """n_alphas alias not created when n_s2 is absent."""
        shape = (2, 2, 2)
        data = {"n_s0": np.full(shape, 1.0)}
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS, ALPHAS])
        assert not ds.has_field("n_alphas")

    def test_scalar_per_species_aliases_for_pressure(self):
        """The generic mechanism also produces P_<species_name>."""
        shape = (2, 2, 2)
        data = {
            "P_s0": np.full(shape, 3.0),
            "P_s1": np.full(shape, 4.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        np.testing.assert_array_equal(ds["P_electrons"], ds["P_s0"])
        np.testing.assert_array_equal(ds["P_ions"], ds["P_s1"])

    def test_vector_component_per_species_aliases(self):
        shape = (2, 2, 2)
        data = {
            "V_s0_1": np.full(shape, 1.0),
            "V_s0_2": np.full(shape, 2.0),
            "EF_s1_1": np.full(shape, 9.0),
        }
        ds = make_test_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        # Tier-3 species-name alias: ``V_<species>_<component>`` mirrors
        # the canonical ``V_s<N>_<component>`` shape.
        np.testing.assert_array_equal(ds["V_electrons_1"], ds["V_s0_1"])
        np.testing.assert_array_equal(ds["V_electrons_2"], ds["V_s0_2"])
        np.testing.assert_array_equal(ds["EF_ions_1"], ds["EF_s1_1"])


class TestAuditIssue2SGyroRename:
    """s_gyro_e/s_gyro_i resolve via the species template; bare s_gyro is an error."""

    def test_s_gyro_e_resolves(self):
        # Post v1.0.x: per-species entropies come from the ``"s_gyro"``
        # species template (``s_gyro_s0``); ``s_gyro_e`` aliases down to it.
        assert _COMPUTE_ALIASES["s_gyro_e"] == "s_gyro_s0"

    def test_s_gyro_i_resolves(self):
        assert _COMPUTE_ALIASES["s_gyro_i"] == "s_gyro_s1"

    def test_bare_s_gyro_not_in_aliases(self):
        assert "s_gyro" not in _COMPUTE_ALIASES

    def test_bare_s_gyro_raises_with_suggestions(self):
        """Bare s_gyro raises KeyError with suggestion to use s_gyro_e."""
        shape = (2, 2, 2)
        data = {
            "P_11": np.full(shape, 1.0),
            "P_22": np.full(shape, 1.0),
            "P_33": np.full(shape, 3.0),
            "P_12": np.zeros(shape),
            "P_13": np.zeros(shape),
            "P_23": np.zeros(shape),
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "n_s0": np.full(shape, 1.0),
        }
        ds = make_test_dataset(data, shape=shape)
        with pytest.raises(KeyError, match="s_gyro_e"):
            compute_field("s_gyro", ds)


class TestAuditIssue6ElectronVelocityMagnitude:
    """|Ve| compute and aliases."""

    def test_ve_magnitude_345(self):
        shape = (2, 2, 2)
        data = {
            "Ve1": np.full(shape, 3.0),
            "Ve2": np.full(shape, 4.0),
            "Ve3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("|Ve|", ds)
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)

    @pytest.mark.parametrize("alias", ["Ve_mag", "Vemag"])
    def test_ve_magnitude_aliases(self, alias):
        shape = (2, 2, 2)
        data = {
            "Ve1": np.full(shape, 3.0),
            "Ve2": np.full(shape, 4.0),
            "Ve3": np.zeros(shape),
        }
        ds = make_test_dataset(data, shape=shape)
        result = compute_field(alias, ds)
        np.testing.assert_allclose(result, 5.0, rtol=1e-15)


class TestAuditSIFactorGaps:
    """SI factor resolution for fields added by audit."""

    @pytest.mark.parametrize("name", ["s_gyro_e", "s_gyro_i"])
    def test_gyrotropic_entropy_si_factor(self, name):
        norm = Normalization.identity()
        assert field_si_factor(name, norm) == pytest.approx(1.0)

    @pytest.mark.parametrize("name", ["EF_1", "EF_2", "EF_3"])
    def test_ef_si_factor(self, name):
        norm = Normalization.identity()
        factor = field_si_factor(name, norm)
        assert factor == pytest.approx(1.0)

    @pytest.mark.parametrize("name", ["B0_1", "B0_2", "B0_3"])
    def test_b0_si_factor(self, name):
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
        assert field_si_factor(name, norm) == pytest.approx(b_ref)

    def test_ef1_s0_si_factor(self):
        norm = Normalization.identity()
        factor = field_si_factor("EF_s0_1", norm)
        assert factor == pytest.approx(1.0)

    def test_ve_1_si_factor(self):
        # Tier-3: electron velocity component 1 is ``Ve_1`` (Ve already
        # implies electron). The redundant ``Ve1_s0`` form is gone.
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=3.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        assert field_si_factor("Ve_1", norm) == pytest.approx(3.0)

    def test_ve_magnitude_si_factor(self):
        norm = Normalization(
            length_ref=1.0,
            time_ref=1.0,
            velocity_ref=3.0,
            b_field_ref=1.0,
            e_field_ref=1.0,
            density_ref=1.0,
            mass_ref=1.0,
            charge_ref=1.0,
        )
        assert field_si_factor("|Ve|", norm) == pytest.approx(3.0)


class TestAuditFieldPrefixAliases:
    """EF and B0 Cartesian aliases via _FIELD_PREFIX_PAIRS."""

    def test_efx_alias_resolves(self):
        shape = (2, 2, 2)
        data = {"EF_1": np.full(shape, 7.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds["EFx"], ds["EF_1"])

    def test_b0x_alias_resolves(self):
        # Canonical split-B name is "B0_1" (schema.md § "Split-B naming":
        # the B0 prefix ends in a digit, so components use an underscore
        # separator). The Cartesian alias B0x must resolve to it.
        shape = (2, 2, 2)
        data = {"B0_1": np.full(shape, 3.0)}
        ds = make_test_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds["B0x"], ds["B0_1"])

    def test_ion_acoustic_speed_alias(self):
        assert _COMPUTE_ALIASES["ion_acoustic_speed"] == "c_ia"


class TestEnergyFluxAliases:
    """Descriptive energy_flux_x/y/z aliases for EF_1/EF_2/EF_3."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("energy_flux_x", "EF_1"),
            ("energy_flux_y", "EF_2"),
            ("energy_flux_z", "EF_3"),
        ],
    )
    def test_energy_flux_alias_resolves(self, alias, canonical):
        assert _COMPUTE_ALIASES[alias] == canonical

    def test_energy_flux_x_passthrough(self):
        shape = (2, 2, 2)
        data = {"EF_1": np.full(shape, 3.14)}
        ds = make_test_dataset(data, shape=shape)
        result = compute_field("energy_flux_x", ds)
        np.testing.assert_allclose(result, 3.14)
