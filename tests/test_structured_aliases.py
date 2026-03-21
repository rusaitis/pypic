"""Tests for structured (underscore-separated) field and compute aliases."""

import numpy as np
import pytest

from pypic.compute import (
    _COMPUTE_ALIASES,
    _REGISTRY,
    compute_field,
    field_si_factor,
)
from pypic.readers.base import FieldDataset, GridInfo, _default_aliases
from pypic.units import Normalization, SpeciesInfo

ELECTRONS = SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256)
IONS = SpeciesInfo(name="ions", charge=1.0, mass=1.0)


def _make_dataset(
    fields: dict[str, np.ndarray],
    *,
    shape: tuple[int, ...] = (4, 3, 2),
    species: list[SpeciesInfo] | None = None,
    physics: dict | None = None,
    normalization: Normalization | None = None,
) -> FieldDataset:
    grid = GridInfo(dimensions=shape, spacing=(1.0,) * len(shape))
    return FieldDataset.from_arrays(
        fields,
        grid,
        normalization or Normalization.identity(),
        species=species,
        physics=physics,
    )


class TestUnderscoreFieldAliases:
    """Underscore-separated names resolve through FieldDataset."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("B_x", "B1"),
            ("B_y", "B2"),
            ("B_z", "B3"),
            ("E_x", "E1"),
            ("V_y", "V2"),
            ("J_z", "J3"),
        ],
    )
    def test_cartesian_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 7.0)}
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("B_1", "B1"), ("E_2", "E2"), ("V_3", "V3"), ("J_1", "J1")],
    )
    def test_numbered_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 3.0)}
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("P_e", "Pe"),
            ("P_i", "Pi"),
            ("T_e", "Te"),
            ("T_i", "Ti"),
            ("P_11", "P11"),
            ("P_12", "P12"),
            ("P_33", "P33"),
        ],
    )
    def test_scalar_underscore_aliases(self, alias, canonical):
        shape = (2, 2, 2)
        data = {canonical: np.full(shape, 5.0)}
        ds = _make_dataset(data, shape=shape)
        np.testing.assert_array_equal(ds[alias], ds[canonical])

    def test_has_field_with_underscore_alias(self):
        shape = (2, 2, 2)
        ds = _make_dataset({"B1": np.ones(shape)}, shape=shape)
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
        ds = _make_dataset(data, shape=shape)
        result_alias = compute_field(alias, ds)
        result_canonical = compute_field(canonical, ds)
        np.testing.assert_array_equal(result_alias, result_canonical)


class TestDescriptiveComputeAliases:
    """Descriptive names resolve to canonical compute entries."""

    def test_plasma_beta(self):
        shape = (2, 2, 2)
        data = {
            "P": np.full(shape, 25.0),
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("plasma_beta", ds)
        np.testing.assert_allclose(result, 2.0, rtol=1e-15)

    def test_v_alfven(self):
        shape = (2, 2, 2)
        data = {
            "B1": np.full(shape, 1.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
            "rho_m": np.full(shape, 4.0),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("v_Alfven", ds)
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
        ds = _make_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("omega_p_s0", ds)
        expected = compute_field("omega_pe", ds)
        np.testing.assert_array_equal(result, expected)

    def test_d_s1(self):
        shape = (2, 2, 2)
        data = {"n_s1": np.full(shape, 1.0)}
        ds = _make_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("d_s1", ds)
        expected = compute_field("d_i", ds)
        np.testing.assert_array_equal(result, expected)

    def test_larmor_radius_s0(self):
        shape = (2, 2, 2)
        data = {
            "Te": np.full(shape, 1.0),
            "B1": np.full(shape, 1.0),
            "B2": np.zeros(shape),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape, species=[ELECTRONS, IONS])
        result = compute_field("larmor_radius_s0", ds)
        expected = compute_field("r_e", ds)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("omega_p_s0", "omega_pe"),
            ("omega_p_s1", "omega_pi"),
            ("omega_c_s0", "omega_ce"),
            ("omega_c_s1", "omega_ci"),
            ("d_s0", "d_e"),
            ("d_s1", "d_i"),
            ("v_thermal_s0", "v_th_e"),
            ("v_thermal_s1", "v_th_i"),
            ("rL_s0", "r_e"),
            ("rL_s1", "r_i"),
            ("beta_s0", "beta_e"),
            ("beta_s1", "beta_i"),
            ("entropy_s0", "s_e"),
            ("entropy_s1", "s_i"),
        ],
    )
    def test_species_aliases_resolve(self, alias, canonical):
        assert _COMPUTE_ALIASES[alias] == canonical


class TestOperatorAliases:
    """Underscore-separated operator/component aliases resolve correctly."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("curl_B_x", "curl_B1"),
            ("curl_B_y", "curl_B2"),
            ("curl_B_z", "curl_B3"),
            ("curl_B_1", "curl_B1"),
            ("curl_B_2", "curl_B2"),
            ("curl_B_3", "curl_B3"),
            ("vort_1", "vort1"),
            ("vort_2", "vort2"),
            ("vort_3", "vort3"),
            ("S_1", "S1"),
            ("S_x", "S1"),
            ("S_z", "S3"),
        ],
    )
    def test_operator_aliases_resolve(self, alias, canonical):
        assert _COMPUTE_ALIASES[alias] == canonical

    def test_curl_b_x_computes(self):
        shape = (4, 4, 4)
        data = {
            "B1": np.ones(shape),
            "B2": np.ones(shape),
            "B3": np.ones(shape),
        }
        ds = _make_dataset(data, shape=shape)
        result = compute_field("curl_B_x", ds)
        expected = compute_field("curl_B1", ds)
        np.testing.assert_array_equal(result, expected)

    def test_s_x_computes(self):
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
        result = compute_field("S_x", ds)
        expected = compute_field("S1", ds)
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
        ds = _make_dataset({"B1": np.full(shape, 5.0)}, shape=shape, normalization=norm)
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
        ds = _make_dataset({"B1": np.full(shape, 5.0)}, shape=shape, normalization=norm)
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
            "B1": np.full(shape, 3.0),
            "B2": np.full(shape, 4.0),
            "B3": np.zeros(shape),
        }
        ds = _make_dataset(data, shape=shape, normalization=norm)
        result = ds.in_si("B_mag")
        np.testing.assert_allclose(result, 5.0 * b_ref, rtol=1e-15)


class TestRegistryIntegrity:
    """No collisions between alias dicts and registry."""

    def test_no_compute_alias_collides_with_registry(self):
        overlap = set(_COMPUTE_ALIASES) & set(_REGISTRY)
        assert not overlap, f"Compute alias collision with registry: {overlap}"

    def test_all_compute_aliases_resolve_to_registry(self):
        for alias, target in _COMPUTE_ALIASES.items():
            assert target in _REGISTRY, f"Alias {alias!r} -> {target!r} not in registry"

    def test_no_underscore_field_alias_collides_with_canonical(self):
        from pypic.coordinates import CARTESIAN

        aliases = _default_aliases(CARTESIAN)
        # No alias should have the same key as a common canonical field
        canonical_fields = {"B1", "B2", "B3", "E1", "E2", "E3", "Pe", "Pi", "Te", "Ti"}
        overlap = set(aliases) & canonical_fields
        assert not overlap, f"Alias collision with canonical: {overlap}"

    def test_underscore_aliases_present_in_defaults(self):
        from pypic.coordinates import CARTESIAN

        aliases = _default_aliases(CARTESIAN)
        assert "B_x" in aliases
        assert "B_1" in aliases
        assert "P_e" in aliases
        assert "T_i" in aliases
        assert "P_11" in aliases
