import numpy as np
import pytest

from pypic.compute import _FIELD_QUANTITY_MAP, field_si_factor
from pypic.fields import (
    _FIELD_INFO,
    _QUANTITY_UNITS,
    field_info,
    quantity_units,
    unit_label,
)
from pypic.readers.base import FieldDataset, GridInfo
from pypic.units import Normalization, SpeciesInfo


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

    def test_field_quantity_map_derived_correctly(self) -> None:
        for name, info in _FIELD_INFO.items():
            assert name in _FIELD_QUANTITY_MAP
            assert _FIELD_QUANTITY_MAP[name] == info.quantity_type


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
