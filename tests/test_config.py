"""Tests for simulation.toml config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import constants

from pypic.readers.config import LENGTH_UNITS, apply_physical_extent, load_config
from pypic.units import PhysicsParams

EXAMPLE_TOML = (
    Path(__file__).resolve().parent.parent / "examples" / "ipic3d-double-harris.toml"
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "simulation.toml"
    p.write_text(content, encoding="utf-8")
    return p


class TestIPIC3DDoubleHarris:
    @pytest.fixture
    def cfg(self):
        return load_config(EXAMPLE_TOML)

    def test_model_and_metadata(self, cfg):
        assert cfg.model_name == "iPIC3D"
        assert cfg.model_type == "PIC"
        assert cfg.physics.c == 1.0
        assert cfg.physics.gamma == pytest.approx(5.0 / 3.0)
        assert cfg.physics.extra["theta"] == 0.5
        assert cfg.metadata["description"] == "Double Harris sheet reconnection"

    def test_grid(self, cfg):
        assert cfg.grid.dimensions == (100, 100, 1)
        np.testing.assert_allclose(cfg.grid.spacing, (0.3, 0.3, 1.0), rtol=1e-12)
        assert cfg.grid.dt == 0.125

    def test_species(self, cfg):
        assert len(cfg.species) == 4
        assert cfg.species[0].charge == -1.0
        np.testing.assert_allclose(cfg.species[0].mass, 1 / 256, rtol=1e-10)

    def test_normalization(self, cfg):
        np.testing.assert_allclose(
            cfg.normalization.velocity_ref, constants.c, rtol=1e-10
        )


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

    @pytest.mark.parametrize("missing_key", ["dimensions", "spacing"])
    def test_missing_grid_key(self, tmp_path, missing_key):
        if missing_key == "dimensions":
            present = "spacing = [1.0, 1.0, 1.0]"
        else:
            present = "dimensions = [2, 2, 2]"
        with pytest.raises(ExceptionGroup) as exc_info:
            load_config(
                _write_toml(
                    tmp_path,
                    f"""
[model]
name = "test"
type = "PIC"
[grid]
{present}
[units]
system = "SI"
[coordinates]
geometry = "cartesian"
frame = "sim"
""",
                )
            )
        assert any(f"'{missing_key}'" in str(e) for e in exc_info.value.exceptions)


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

    def test_custom_labels_and_unknown_geometry(self, tmp_path):
        # Custom axis labels
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

        # Unknown geometry raises
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


_SCALING_TOML = """\
[model]
name = "test"
type = "PIC"

[grid]
dimensions = [460, 130, 320]
spacing = [0.4, 0.4, 0.4]

[coordinates]
geometry = "cartesian"
frame = "simulation"
physical_extent = [46.0, 32.0, 13.0]
physical_extent_unit = "R_E"

[coordinates.transforms.GSM]
origin = [52.0, 26.0, 64.0]
rotation = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
axis_labels = ["x_GSM", "y_GSM", "z_GSM"]

[units]
system = "PIC"
reference_species = "ions"
reference_density = 0.25e6
"""


class TestPhysicalExtent:
    def test_auto_scale_from_toml(self, tmp_path: Path) -> None:
        cfg = load_config(_write_toml(tmp_path, _SCALING_TOML))
        np.testing.assert_allclose(cfg.transforms["GSM"].scale, 0.25, rtol=1e-6)

    def test_shrink_factor_computed(self, tmp_path: Path) -> None:
        cfg = load_config(_write_toml(tmp_path, _SCALING_TOML))
        shrink = cfg.metadata["scaling"]["shrink_factor"]
        np.testing.assert_allclose(shrink, 3.5, rtol=0.01)

    def test_physical_extent_in_metadata(self, tmp_path: Path) -> None:
        cfg = load_config(_write_toml(tmp_path, _SCALING_TOML))
        assert cfg.metadata["physical_extent"] == (46.0, 32.0, 13.0)
        assert cfg.metadata["physical_extent_unit"] == "R_E"

    def test_explicit_scale_validated(self, tmp_path: Path) -> None:
        """Explicit scale that matches physical_extent passes without error."""
        toml = _SCALING_TOML.replace(
            "axis_labels = [",
            "scale = 0.25\naxis_labels = [",
        )
        cfg = load_config(_write_toml(tmp_path, toml))
        np.testing.assert_allclose(cfg.transforms["GSM"].scale, 0.25)

    def test_explicit_scale_mismatch_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Explicit scale that disagrees with physical_extent logs a warning."""
        toml = _SCALING_TOML.replace(
            "axis_labels = [",
            "scale = 0.30\naxis_labels = [",
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="pypic.readers.config"):
            cfg = load_config(_write_toml(tmp_path, toml))
        assert cfg.transforms["GSM"].scale == 0.30  # explicit wins
        assert "Scale mismatch" in caplog.text

    def test_unknown_unit_raises(self, tmp_path: Path) -> None:
        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.grid import GridInfo
        from pypic.units import Normalization

        cfg = SimulationConfig(
            model_name="t",
            model_type="PIC",
            grid=GridInfo((10,), (1.0,), (0.0,), CARTESIAN),
            normalization=Normalization.identity(),
            species=(),
            physics=PhysicsParams(),
        )
        with pytest.raises(ValueError, match="Unknown physical_extent_unit"):
            apply_physical_extent(cfg, (5.0,), "parsec")

    def test_non_uniform_scale_raises(self, tmp_path: Path) -> None:
        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.coordinates.transforms import FrameTransform
        from pypic.grid import GridInfo
        from pypic.units import Normalization

        t = FrameTransform("sim", "phys")
        cfg = SimulationConfig(
            model_name="t",
            model_type="PIC",
            grid=GridInfo((100, 100, 100), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), CARTESIAN),
            normalization=Normalization.identity(),
            species=(),
            physics=PhysicsParams(),
            transforms={"phys": t},
        )
        with pytest.raises(ValueError, match="non-uniform"):
            apply_physical_extent(cfg, (50.0, 30.0, 50.0), "m")

    def test_length_units_table(self) -> None:
        assert LENGTH_UNITS["m"] == 1.0
        assert LENGTH_UNITS["km"] == 1e3
        assert LENGTH_UNITS["R_E"] == 6.371e6
        assert len(LENGTH_UNITS) >= 5


class TestMergeSimulationToml:
    """Unit tests for the shared ``readers._config_helpers.merge_simulation_toml``."""

    @pytest.fixture
    def base_config(self):  # type: ignore[no-untyped-def]
        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.grid import GridInfo
        from pypic.units import Normalization

        return SimulationConfig(
            model_name="ReaderX",
            model_type="MHD",
            grid=GridInfo(
                dimensions=(4, 3, 2),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                geometry=CARTESIAN,
            ),
            normalization=Normalization.identity(),
            frame="GSM",
            metadata={"reader_only": "kept"},
        )

    def test_no_sim_dir_returns_base_unchanged(self, base_config) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        result = merge_simulation_toml(None, base_config)
        assert result is base_config

    def test_missing_toml_returns_base_unchanged(
        self, tmp_path: Path, base_config
    ) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        result = merge_simulation_toml(tmp_path, base_config)
        assert result is base_config

    def test_toml_overrides_normalization_frame_metadata(
        self, tmp_path: Path, base_config
    ) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        _write_toml(
            tmp_path,
            """
[model]
name = "test"
type = "MHD"

[grid]
dimensions = [10, 10, 10]
spacing = [0.5, 0.5, 0.5]

[units]
system = "MHD"
reference_length = 6.371e6
reference_density = 1.67e-17
reference_b_field = 5.0e-9

[coordinates]
geometry = "cartesian"
frame = "simulation"

[output]
fields = ["B1"]
""",
        )
        result = merge_simulation_toml(tmp_path, base_config)
        # Frame override
        assert result.frame == "simulation"
        # Reader-side metadata preserved
        assert result.metadata["reader_only"] == "kept"
        # Toml metadata merged in (output goes into metadata)
        assert "output" in result.metadata
        # Reader-side fields preserved
        assert result.model_name == "ReaderX"
        assert result.grid.dimensions == (4, 3, 2)
        # Normalization actually replaced (not identity any more)
        assert result.normalization.length_ref == 6.371e6

    def test_toml_metadata_wins_on_key_conflict(
        self, tmp_path: Path, base_config
    ) -> None:
        # Re-create base with a metadata key that the toml will also set
        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.grid import GridInfo
        from pypic.readers._config_helpers import merge_simulation_toml
        from pypic.units import Normalization

        base = SimulationConfig(
            model_name="X",
            model_type="MHD",
            grid=GridInfo(
                dimensions=(4,), spacing=(1.0,), origin=(0.0,), geometry=CARTESIAN
            ),
            normalization=Normalization.identity(),
            metadata={"description": "from-reader", "reader_only": "kept"},
        )
        _write_toml(
            tmp_path,
            """
[model]
name = "test"
type = "MHD"
description = "from-toml"

[grid]
dimensions = [10, 10, 10]
spacing = [0.5, 0.5, 0.5]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"
""",
        )
        result = merge_simulation_toml(tmp_path, base)
        # Toml's description wins on conflict; reader_only is preserved
        assert result.metadata["description"] == "from-toml"
        assert result.metadata["reader_only"] == "kept"
