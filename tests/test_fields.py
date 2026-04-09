import numpy as np
import pytest

from pypic.compute import field_si_factor
from pypic.fields import (
    _FIELD_INFO,
    _QUANTITY_UNITS,
    _SPECIES_INFO_PATTERNS,
    QuantityType,
    field_info,
    quantity_units,
    register_field,
    unit_label,
    unregister_field,
)
from pypic.readers.base import FieldDataset, GridInfo
from pypic.selections import PlaneSelection
from pypic.units import _COMPOUND_FACTORS, _QUANTITIES, Normalization, SpeciesInfo


def _make_dataset(fields: dict[str, np.ndarray]) -> FieldDataset:
    grid = GridInfo(dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0))
    return FieldDataset.from_arrays(
        fields,
        grid,
        Normalization.identity(),
        species=[
            SpeciesInfo(name="electrons", charge=-1.0, mass=1.0),
            SpeciesInfo(name="ions", charge=1.0, mass=256.0),
        ],
    )


class TestRegistryIntegrity:
    """Every entry must have a valid quantity_type and consistent si_unit."""

    def test_all_entries_consistent(self) -> None:
        norm = Normalization.identity()
        for name, info in _FIELD_INFO.items():
            factor = norm.si_factor(info.quantity_type)
            assert isinstance(factor, float), f"{name}: si_factor returned non-float"

            expected_unit = _QUANTITY_UNITS[info.quantity_type]
            assert info.si_unit == expected_unit, (
                f"{name}: si_unit={info.si_unit!r} but "
                f"_QUANTITY_UNITS[{info.quantity_type!r}]={expected_unit!r}"
            )

    def test_all_fields_have_valid_quantity_type(self) -> None:
        for name, info in _FIELD_INFO.items():
            assert info.quantity_type in _QUANTITY_UNITS, (
                f"{name}: quantity_type {info.quantity_type!r} not in _QUANTITY_UNITS"
            )


class TestFieldInfoLookup:
    def test_canonical_name(self) -> None:
        info = field_info("|B|")
        assert info.quantity_type == "b_field"
        assert info.si_unit == "T"
        assert info.long_name == "Magnetic field magnitude"
        assert info.latex == r"$|B|$"

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("B_mag", "|B|"),
            ("plasma_beta", "beta"),
            ("Bx", "B1"),
            ("P_e", "Pe"),
            ("B_1", "B1"),
        ],
    )
    def test_alias_resolution(self, alias: str, canonical: str) -> None:
        assert field_info(alias) == field_info(canonical)

    @pytest.mark.parametrize(
        ("name", "quantity_type", "long_name"),
        [
            ("n_s5", "density", "Number density (species 5)"),
            ("omega_p_s3", "frequency", "Plasma frequency (species 3)"),
            ("v_th_s2", "velocity", "Thermal speed (species 2)"),
            ("T_s4", "temperature", "Temperature (species 4)"),
            ("J1_s2", "current_density", "Current density component 1 (species 2)"),
        ],
    )
    def test_species_patterns(
        self, name: str, quantity_type: str, long_name: str
    ) -> None:
        info = field_info(name)
        assert info.quantity_type == quantity_type
        assert info.long_name == long_name

    def test_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="No metadata"):
            field_info("nonexistent_field_xyz")


@pytest.mark.parametrize(
    ("name", "si", "expected"),
    [
        ("B1", True, "T"),
        ("B1", False, "normalized"),
        ("beta", True, ""),
        ("beta", False, ""),
        ("v_A", True, "m/s"),
    ],
)
def test_unit_label(name: str, si: bool, expected: str) -> None:
    assert unit_label(name, si=si) == expected


class TestQuantityUnits:
    def test_known_type(self) -> None:
        assert quantity_units("b_field") == "T"
        assert quantity_units("pressure") == "Pa"
        assert quantity_units("dimensionless") == ""

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown quantity type"):
            quantity_units("invalid_type")


class TestFieldDatasetFieldInfo:
    def test_resolves_canonical_and_alias(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        assert ds.field_info("beta").latex == r"$\beta$"
        info = ds.field_info("Bx")
        assert info.long_name == "Magnetic field x-component"
        assert info.latex == r"$B_x$"


class TestXarrayAttrs:
    def test_known_field_has_attrs(self) -> None:
        ds = _make_dataset(
            {
                "B1": np.ones((4, 3, 2)),
                "rho_m": np.ones((4, 3, 2)),
            }
        )
        b1_attrs = ds.xr["B1"].attrs
        assert b1_attrs["long_name"] == "Magnetic field x-component"
        assert b1_attrs["units"] == "normalized"
        assert ds.xr["rho_m"].attrs["long_name"] == "Mass density"

    def test_unknown_field_no_attrs(self) -> None:
        ds = _make_dataset({"custom_field": np.ones((4, 3, 2))})
        assert "long_name" not in ds.xr["custom_field"].attrs
        assert "units" not in ds.xr["custom_field"].attrs


class TestFieldSiFactorRegression:
    """field_si_factor unchanged after _FIELD_QUANTITY_MAP derivation."""

    def test_all_canonical_names_return_float(self) -> None:
        norm = Normalization.identity()
        for name in _FIELD_INFO:
            factor = field_si_factor(name, norm)
            assert isinstance(factor, float), f"{name}: expected float"

    def test_alias_matches_canonical(self) -> None:
        norm = Normalization.identity()
        assert field_si_factor("Bx", norm) == field_si_factor("B1", norm)

    def test_species_pattern(self) -> None:
        norm = Normalization.identity()
        factor = field_si_factor("n_s5", norm)
        assert isinstance(factor, float)


class TestGeometryAwareLabels:
    """field_info() localizes component labels when axis_names is given."""

    CARTESIAN = ("x", "y", "z")
    SPHERICAL = ("r", "θ", "φ")
    CYLINDRICAL = ("r", "φ", "z")

    @pytest.mark.parametrize(
        ("name", "axes", "expected_long", "expected_latex"),
        [
            ("B1", ("x", "y", "z"), "Magnetic field x-component", r"$B_x$"),
            ("B2", ("x", "y", "z"), "Magnetic field y-component", r"$B_y$"),
            ("B1", ("r", "θ", "φ"), "Magnetic field r-component", r"$B_r$"),
            (
                "B2",
                ("r", "θ", "φ"),
                "Magnetic field θ-component",
                r"$B_{\theta}$",
            ),
            (
                "V2",
                ("r", "φ", "z"),
                "Bulk velocity φ-component",
                r"$V_{\phi}$",
            ),
        ],
    )
    def test_vector_components(
        self,
        name: str,
        axes: tuple[str, str, str],
        expected_long: str,
        expected_latex: str,
    ) -> None:
        info = field_info(name, axis_names=axes)
        assert info.long_name == expected_long
        assert info.latex == expected_latex

    def test_electron_velocity(self) -> None:
        info = field_info("Ve1", axis_names=self.CARTESIAN)
        assert info.long_name == "Electron velocity x-component"
        assert info.latex == r"$V_{e,x}$"

    def test_background_b(self) -> None:
        info = field_info("B0_1", axis_names=self.CARTESIAN)
        assert info.long_name == "Background B x-component"
        assert info.latex == r"$B_{0,x}$"

    def test_curl_component(self) -> None:
        info = field_info("curl_B1", axis_names=self.CARTESIAN)
        assert info.long_name == "Curl of B x-component"
        assert info.latex == r"$(\nabla \times B)_x$"

    def test_species_component(self) -> None:
        info = field_info("J1_s2", axis_names=self.CARTESIAN)
        assert info.long_name == "Current density x-component (species 2)"
        assert info.latex == r"$J_{x,s2}$"

    def test_no_geometry_unchanged(self) -> None:
        info = field_info("B1")
        assert info.long_name == "Magnetic field component 1"
        assert info.latex == r"$B_1$"

    def test_scalar_unaffected(self) -> None:
        assert field_info("|B|", axis_names=self.CARTESIAN) == field_info("|B|")
        assert field_info("beta", axis_names=self.CARTESIAN) == field_info("beta")
        assert field_info("rho_m", axis_names=self.CARTESIAN) == field_info("rho_m")

    def test_vorticity_component(self) -> None:
        info = field_info("vort1", axis_names=self.CARTESIAN)
        assert info.long_name == "Vorticity x-component"
        assert info.latex == r"$\omega_x$"

    def test_poynting_flux_spherical(self) -> None:
        info = field_info("S2", axis_names=self.SPHERICAL)
        assert info.long_name == "Poynting flux θ-component"
        assert info.latex == r"$S_{\theta}$"

    def test_species_electron_velocity_component(self) -> None:
        info = field_info("Ve2_s1", axis_names=self.CARTESIAN)
        assert info.long_name == "Electron velocity y-component (species 1)"
        assert info.latex == r"$V_{e,y,s1}$"


class TestNewFieldEntries:
    """Verify gamma_L, sigma, gamma_eos registry entries."""

    @pytest.mark.parametrize(
        ("name", "latex"),
        [
            ("gamma_L", r"$\gamma$"),
            ("sigma", r"$\sigma$"),
            ("gamma_eos", r"$\gamma_{eos}$"),
        ],
    )
    def test_dimensionless_entry(self, name: str, latex: str) -> None:
        info = field_info(name)
        assert info.quantity_type == "dimensionless"
        assert info.si_unit == ""
        assert info.latex == latex

    def test_high_species_index(self) -> None:
        info = field_info("n_s99")
        assert info.quantity_type == "density"
        assert info.long_name == "Number density (species 99)"

    def test_pressure_scalar_species_s0_alias(self) -> None:
        """P_s0 resolves to Pe via compute alias."""
        info = field_info("P_s0")
        assert info.quantity_type == "pressure"
        assert info.long_name == "Electron pressure"

    def test_pressure_scalar_species_s3(self) -> None:
        """P_s3 hits the regex path (no alias for species >= 2)."""
        info = field_info("P_s3")
        assert info.quantity_type == "pressure"
        assert info.long_name == "Pressure (species 3)"
        assert info.latex == r"$P_{s3}$"

    def test_pressure_tensor_species(self) -> None:
        info = field_info("P11_s2")
        assert info.quantity_type == "pressure"
        assert info.long_name == "Pressure 11 (species 2)"
        assert info.latex == r"$P_{11,s2}$"


class TestQuantityTypeCoverage:
    """Structural tests ensuring quantity-type registries stay synchronized."""

    def test_quantity_types_complete(self) -> None:
        """Every quantity type with an SI factor must have a unit label."""
        si_factor_types = _QUANTITIES | _COMPOUND_FACTORS.keys()
        unit_label_types = set(_QUANTITY_UNITS)
        assert si_factor_types == unit_label_types

    def test_species_patterns_have_known_quantity_types(self) -> None:
        """Every quantity_type in _SPECIES_INFO_PATTERNS has a _QUANTITY_UNITS entry."""
        for pattern, qtype, _, _ in _SPECIES_INFO_PATTERNS:
            assert qtype in _QUANTITY_UNITS, (
                f"Pattern {pattern.pattern!r} uses quantity_type={qtype!r} "
                f"which is missing from _QUANTITY_UNITS"
            )


class TestFieldRegistration:
    """register_field / unregister_field public API."""

    def test_register_and_lookup(self) -> None:
        name = "_test_reg_lookup"
        try:
            register_field(name, "velocity", long_name="Test speed", latex=r"$v_t$")
            info = field_info(name)
            assert info.quantity_type == "velocity"
            assert info.long_name == "Test speed"
            assert info.latex == r"$v_t$"
        finally:
            unregister_field(name)

    def test_register_enables_si_factor(self) -> None:
        name = "_test_reg_si"
        norm = Normalization.identity()
        try:
            register_field(name, "b_field")
            factor = field_si_factor(name, norm)
            assert isinstance(factor, float)
        finally:
            unregister_field(name)

    def test_register_auto_fills_si_unit(self) -> None:
        name = "_test_reg_auto_unit"
        try:
            register_field(name, "pressure")
            assert field_info(name).si_unit == "Pa"
        finally:
            unregister_field(name)

    def test_register_custom_si_unit(self) -> None:
        name = "_test_reg_custom_unit"
        try:
            register_field(name, "pressure", si_unit="nPa")
            assert field_info(name).si_unit == "nPa"
        finally:
            unregister_field(name)

    def test_register_invalid_quantity_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown quantity_type"):
            register_field("_test_bad_qtype", "nonexistent_type")

    def test_unregister(self) -> None:
        name = "_test_unreg"
        register_field(name, "dimensionless")
        unregister_field(name)
        with pytest.raises(KeyError, match="No metadata"):
            field_info(name)

    def test_unregister_nonexistent(self) -> None:
        with pytest.raises(KeyError, match="No field metadata"):
            unregister_field("_test_does_not_exist_xyz")

    def test_overwrite_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        name = "_test_overwrite"
        try:
            register_field(name, "velocity")
            with caplog.at_level("WARNING", logger="pypic.fields"):
                register_field(name, "pressure")
            assert "Overwriting" in caplog.text
            assert field_info(name).quantity_type == "pressure"
        finally:
            unregister_field(name)

    def test_register_visible_to_si_factor(self) -> None:
        name = "_test_si_visible"
        norm = Normalization.identity()
        try:
            register_field(name, "energy_density")
            factor = field_si_factor(name, norm)
            assert isinstance(factor, float)
        finally:
            unregister_field(name)

    def test_register_with_enum(self) -> None:
        name = "_test_reg_enum"
        try:
            register_field(name, QuantityType.VELOCITY, long_name="Enum speed")
            info = field_info(name)
            assert info.quantity_type == "velocity"
            assert info.long_name == "Enum speed"
        finally:
            unregister_field(name)


class TestQuantityType:
    """QuantityType StrEnum covers all _QUANTITY_UNITS keys."""

    def test_covers_all_quantity_units(self) -> None:
        enum_values = {member.value for member in QuantityType}
        assert enum_values == set(_QUANTITY_UNITS)

    def test_strenum_equality_with_strings(self) -> None:
        assert QuantityType.B_FIELD == "b_field"
        assert QuantityType.DIMENSIONLESS == "dimensionless"
        assert QuantityType.VELOCITY == "velocity"

    def test_usable_as_dict_key(self) -> None:
        assert _QUANTITY_UNITS[QuantityType.PRESSURE] == "Pa"


class TestWithField:
    """FieldDataset.with_field() — attach custom fields with metadata."""

    def test_basic(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        data = np.full((4, 3, 2), 0.42)
        ds2 = ds.with_field("R_rec", data, QuantityType.DIMENSIONLESS)
        np.testing.assert_array_equal(ds2["R_rec"], data)

    def test_in_si_via_attrs(self) -> None:
        norm = Normalization.pic_electron(n_e=1.0e18)
        grid = GridInfo(dimensions=(2,), spacing=(1.0,))
        ds = FieldDataset.from_arrays({"B1": np.array([1.0, 2.0])}, grid, norm)
        data = np.array([3.0, 4.0])
        ds2 = ds.with_field("custom_v", data, QuantityType.VELOCITY)
        si_vals = ds2.in_si("custom_v")
        expected_factor = norm.si_factor("velocity")
        np.testing.assert_allclose(si_vals, data * expected_factor)

    def test_field_info_from_attrs(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        ds2 = ds.with_field(
            "R_rec",
            np.ones((4, 3, 2)),
            QuantityType.DIMENSIONLESS,
            long_name="Reconnection rate",
            latex=r"$R_{rec}$",
        )
        info = ds2.field_info("R_rec")
        assert info.quantity_type == "dimensionless"
        assert info.long_name == "Reconnection rate"
        assert info.si_unit == ""
        assert info.latex == r"$R_{rec}$"

    def test_survives_isel(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        ds2 = ds.with_field("diag", np.ones((4, 3, 2)), QuantityType.PRESSURE)
        sliced = ds2.isel(z=0)
        info = sliced.field_info("diag")
        assert info.quantity_type == "pressure"
        assert info.si_unit == "Pa"

    def test_survives_plane_selection(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        ds2 = ds.with_field(
            "diag",
            np.ones((4, 3, 2)),
            QuantityType.VELOCITY,
            long_name="My diagnostic",
        )
        plane = PlaneSelection(normal="z").apply(ds2)
        info = plane.field_info("diag")
        assert info.quantity_type == "velocity"
        assert info.long_name == "My diagnostic"

    def test_invalid_quantity_type(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        with pytest.raises(ValueError, match="Unknown quantity_type"):
            ds.with_field("bad", np.ones((4, 3, 2)), "nonexistent_type")

    def test_immutable_original(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        original_names = ds.field_names()
        ds.with_field("extra", np.ones((4, 3, 2)), QuantityType.DENSITY)
        assert ds.field_names() == original_names

    def test_string_quantity_type(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        ds2 = ds.with_field("test_f", np.ones((4, 3, 2)), "b_field")
        info = ds2.field_info("test_f")
        assert info.quantity_type == "b_field"
        assert info.si_unit == "T"


class TestAttrsOverrideRegistry:
    """xarray attrs take priority over global registry for field_info/in_si."""

    def test_field_info_uses_attrs_over_registry(self) -> None:
        """with_field() attrs override global _FIELD_INFO for same name."""
        ds = _make_dataset({"B1": np.ones((4, 3, 2))})
        # Override B1 as if it were a pressure field
        ds2 = ds.with_field(
            "B1",
            np.full((4, 3, 2), 2.0),
            QuantityType.PRESSURE,
            long_name="Custom pressure",
            latex=r"$P_{custom}$",
        )
        info = ds2.field_info("B1")
        assert info.quantity_type == "pressure"
        assert info.long_name == "Custom pressure"
        assert info.latex == r"$P_{custom}$"

    def test_in_si_uses_attrs_over_registry(self) -> None:
        """in_si() picks quantity_type from attrs, not global registry."""
        norm = Normalization.pic_electron(n_e=1.0e18)
        grid = GridInfo(dimensions=(2,), spacing=(1.0,))
        ds = FieldDataset.from_arrays({"B1": np.array([1.0, 2.0])}, grid, norm)
        # Attach "B1" with velocity quantity_type (overriding b_field)
        ds2 = ds.with_field("B1", np.array([5.0, 6.0]), QuantityType.VELOCITY)
        si = ds2.in_si("B1")
        expected = np.array([5.0, 6.0]) * norm.si_factor("velocity")
        np.testing.assert_allclose(si, expected)


class TestFromArraysQuantityTypeAttr:
    """from_arrays() stores quantity_type in DataArray attrs."""

    def test_canonical_field_has_quantity_type(self) -> None:
        ds = _make_dataset({"B1": np.ones((4, 3, 2)), "rho_m": np.ones((4, 3, 2))})
        assert ds.xr["B1"].attrs["quantity_type"] == "b_field"
        assert ds.xr["B1"].attrs["si_unit"] == "T"
        assert ds.xr["rho_m"].attrs["quantity_type"] == "mass_density"
        assert ds.xr["rho_m"].attrs["si_unit"] == "kg/m^3"

    def test_unknown_field_no_quantity_type(self) -> None:
        ds = _make_dataset({"custom_xyz": np.ones((4, 3, 2))})
        assert "quantity_type" not in ds.xr["custom_xyz"].attrs
