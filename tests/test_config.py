"""Tests for simulation.toml config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import constants

from pypic.readers.config import load_config

EXAMPLE_TOML = (
    Path(__file__).resolve().parent.parent / "examples" / "ipic3d_double_harris.toml"
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "simulation.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Integration test: iPIC3D double Harris sheet
# ---------------------------------------------------------------------------


class TestIPIC3DDoubleHarris:
    @pytest.fixture
    def cfg(self):
        return load_config(EXAMPLE_TOML)

    def test_model_name(self, cfg):
        assert cfg.model_name == "iPIC3D"

    def test_model_type(self, cfg):
        assert cfg.model_type == "PIC"

    def test_grid_dimensions(self, cfg):
        assert cfg.grid.dimensions == (100, 100, 1)

    def test_grid_spacing(self, cfg):
        np.testing.assert_allclose(cfg.grid.spacing, (0.3, 0.3, 1.0), rtol=1e-12)

    def test_grid_dt(self, cfg):
        assert cfg.grid.dt == 0.125

    def test_normalization_velocity_ref(self, cfg):
        np.testing.assert_allclose(
            cfg.normalization.velocity_ref, constants.c, rtol=1e-10
        )

    def test_species_count(self, cfg):
        assert len(cfg.species) == 4

    def test_first_species_charge(self, cfg):
        assert cfg.species[0].charge == -1.0

    def test_first_species_mass(self, cfg):
        np.testing.assert_allclose(cfg.species[0].mass, 1 / 256, rtol=1e-10)

    def test_physics(self, cfg):
        assert cfg.physics == {"pic": {"theta": 0.5, "speed_of_light": 1.0}}

    def test_frame(self, cfg):
        assert cfg.frame == "simulation"

    def test_metadata_has_initial_conditions(self, cfg):
        assert "initial_conditions" in cfg.metadata

    def test_metadata_has_output(self, cfg):
        assert "output" in cfg.metadata

    def test_metadata_description(self, cfg):
        assert cfg.metadata["description"] == "Double Harris sheet reconnection"


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------


class TestParseModel:
    def test_minimal(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.model_name == "test"

    def test_with_extras(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "MHD"
version = "2.0"
description = "A test run"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.metadata["version"] == "2.0"

    def test_missing_name(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("'name'" in str(e) for e in exc_info.value.exceptions)

    def test_missing_type(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("'type'" in str(e) for e in exc_info.value.exceptions)


# ---------------------------------------------------------------------------
# Grid section
# ---------------------------------------------------------------------------


class TestParseGrid:
    def test_full(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [10, 20, 30]
spacing = [0.1, 0.2, 0.3]
origin = [1.0, 2.0, 3.0]
dt = 0.01
boundary = ["periodic", "open", "reflecting"]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.grid.dimensions == (10, 20, 30)
        np.testing.assert_allclose(cfg.grid.origin, (1.0, 2.0, 3.0), rtol=1e-12)

    def test_default_origin(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [4, 4, 4]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.grid.origin == (0.0, 0.0, 0.0)

    def test_missing_dimensions(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("'dimensions'" in str(e) for e in exc_info.value.exceptions)

    def test_missing_spacing(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("'spacing'" in str(e) for e in exc_info.value.exceptions)


# ---------------------------------------------------------------------------
# Units section
# ---------------------------------------------------------------------------


class TestParseUnits:
    def test_pic_electron_default(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "PIC"
reference_density = 1.0e18
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        np.testing.assert_allclose(
            cfg.normalization.velocity_ref, constants.c, rtol=1e-10
        )
        np.testing.assert_allclose(
            cfg.normalization.mass_ref, constants.m_e, rtol=1e-10
        )

    def test_pic_ion_reference(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "PIC"
reference_species = "ions"
reference_density = 1.0e18
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        np.testing.assert_allclose(
            cfg.normalization.mass_ref, constants.m_p, rtol=1e-10
        )

    def test_pic_custom_species_requires_mass_charge(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "PIC"
reference_species = "alpha"
reference_density = 1.0e18
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("reference_mass" in str(e) for e in exc_info.value.exceptions)

    def test_pic_custom_species_with_explicit(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "PIC"
reference_species = "alpha"
reference_density = 1.0e18
reference_mass = 6.644e-27
reference_charge = 3.204e-19
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        np.testing.assert_allclose(cfg.normalization.mass_ref, 6.644e-27, rtol=1e-10)

    def test_mhd(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "MHD"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "MHD"
reference_length = 6.371e6
reference_density = 1.67e-17
reference_b_field = 5.0e-9
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.normalization.length_ref == 6.371e6

    def test_si_identity(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.normalization.length_ref == 1.0

    def test_custom(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "custom"
[units.reference]
length = 5.31e-3
time = 1.77e-11
velocity = 2.998e8
b_field = 1.07e-3
e_field = 3.21e5
density = 1.0e18
mass = 9.109e-31
charge = 1.602e-19
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        np.testing.assert_allclose(cfg.normalization.length_ref, 5.31e-3, rtol=1e-10)

    def test_custom_missing_ref(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "custom"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("reference" in str(e) for e in exc_info.value.exceptions)

    def test_unknown_system(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "CGS"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any("CGS" in str(e) for e in exc_info.value.exceptions)

    def test_scaling_metadata(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
scaling_factor = 10.0
scaling_description = "reduced c/v_A"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.metadata["scaling"]["scaling_factor"] == 10.0


# ---------------------------------------------------------------------------
# Coordinates section
# ---------------------------------------------------------------------------


class TestParseCoordinates:
    @pytest.mark.parametrize(
        ("geom_str", "expected_names"),
        [
            ("cartesian", ("x", "y", "z")),
            ("spherical", ("r", "θ", "φ")),
            ("cylindrical", ("r", "φ", "z")),
        ],
    )
    def test_geometry_lookup(self, tmp_path, geom_str, expected_names):
        cfg = load_config(
            _write_toml(
                tmp_path,
                f"""
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "{geom_str}"
frame = "sim"
""",
            )
        )
        assert cfg.grid.geometry.axis_names == expected_names

    def test_custom_axis_labels(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
axis_labels = ["X", "Y", "Z"]
""",
            )
        )
        assert cfg.grid.geometry.axis_names == ("X", "Y", "Z")

    def test_unknown_geometry(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "toroidal"
frame = "sim"
""",
                )
            )
        assert any("toroidal" in str(e) for e in exc_info.value.exceptions)

    def test_missing_geometry(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
frame = "sim"
""",
                )
            )
        assert any("'geometry'" in str(e) for e in exc_info.value.exceptions)

    def test_missing_frame(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
""",
                )
            )
        assert any("'frame'" in str(e) for e in exc_info.value.exceptions)


# ---------------------------------------------------------------------------
# Species section
# ---------------------------------------------------------------------------


class TestParseSpecies:
    def test_single_species(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "e"
charge = -1.0
mass = 0.004
""",
            )
        )
        assert len(cfg.species) == 1
        assert cfg.species[0].name == "e"

    def test_charge_to_mass_only(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "e"
charge_to_mass = -256.0
""",
            )
        )
        assert cfg.species[0].charge == -1.0
        np.testing.assert_allclose(cfg.species[0].mass, 1.0 / 256, rtol=1e-10)

    @pytest.mark.parametrize(
        ("field_name", "toml_value", "attr", "expected"),
        [
            (
                "thermal_velocity",
                "[0.06, 0.02, 0.02]",
                "thermal_velocity",
                (0.06, 0.02, 0.02),
            ),
            (
                "drift_velocity",
                "[0.0, 0.0, 0.01]",
                "drift_velocity",
                (0.0, 0.0, 0.01),
            ),
            (
                "particles_per_cell",
                "[5, 5, 1]",
                "particles_per_cell",
                (5, 5, 1),
            ),
        ],
    )
    def test_optional_vector_field(
        self, tmp_path, field_name, toml_value, attr, expected
    ):
        cfg = load_config(
            _write_toml(
                tmp_path,
                f"""
[model]
name = "test"
type = "PIC"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
[[species]]
name = "e"
charge = -1.0
mass = 0.004
{field_name} = {toml_value}
""",
            )
        )
        assert getattr(cfg.species[0], attr) == expected

    def test_no_species_section(self, tmp_path):
        cfg = load_config(
            _write_toml(
                tmp_path,
                """
[model]
name = "test"
type = "MHD"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
            )
        )
        assert cfg.species == ()


# ---------------------------------------------------------------------------
# ExceptionGroup: multiple broken sections
# ---------------------------------------------------------------------------


class TestExceptionGroup:
    def test_multiple_errors(self, tmp_path):
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    """
[model]
name = "test"
[coordinates]
frame = "sim"
[units]
system = "CGS"
""",
                )
            )
        assert len(exc_info.value.exceptions) >= 3

    def test_path_in_message(self, tmp_path):
        path = _write_toml(
            tmp_path,
            """
[model]
name = "test"
[grid]
dimensions = [2, 2, 2]
spacing = [1.0, 1.0, 1.0]
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
        )
        with pytest.raises(ExceptionGroup, match=str(path)):
            load_config(path)
