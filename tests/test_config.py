"""Tests for ``load_config`` — the Pydantic-backed v1.0 TOML loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import constants

from pypic.readers.config import LENGTH_UNITS, apply_physical_extent, load_config
from pypic.units import PhysicsParams

EXAMPLE_TOML = (
    Path(__file__).resolve().parent.parent / "examples" / "ipic3d-double-harris.toml"
)


def _shell(
    *,
    model: str = '[model]\nname = "test"\ntype = "PIC"',
    time: str = (
        '[time]\nscheme = "fixed"\ndt = 0.1\nt_start = 0.0\nt_end = 1.0\nn_steps = 10'
    ),
    grid: str = (
        "[grid]\ndimensions = [2, 2, 2]\nspacing = [1.0, 1.0, 1.0]\n"
        "lower = [0.0, 0.0, 0.0]\nupper = [2.0, 2.0, 2.0]"
    ),
    units: str = '[units]\nsystem = "SI"',
    coordinates: str = '[coordinates]\ngeometry = "cartesian"\nframe = "sim"',
    species: str = '[[species]]\nname = "e"\ncharge = -1.0\nmass = 0.004',
    extra: str = "",
) -> str:
    """Assemble a valid v1.0 TOML doc, with per-section overrides."""
    return (
        "\n\n".join(
            [
                'schema_version = "1.0"',
                '[schema]\nversion = "1.0"',
                model,
                '[run]\nname = "r0"',
                time,
                grid,
                units,
                coordinates,
                species,
                extra,
            ]
        ).strip()
        + "\n"
    )


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "simulation.toml"
    p.write_text(content, encoding="utf-8")
    return p


class TestLiveExample:
    @pytest.fixture
    def cfg(self) -> Any:
        return load_config(EXAMPLE_TOML)

    def test_model_and_metadata(self, cfg: Any) -> None:
        assert cfg.model_name == "iPIC3D"
        assert cfg.model_type == "PIC"
        assert cfg.physics.c == 1.0
        assert cfg.physics.gamma == pytest.approx(5.0 / 3.0)
        assert cfg.metadata["description"] == "Double Harris sheet reconnection"

    def test_physics_pic_branch_present(self, cfg: Any) -> None:
        assert "pic" in cfg.physics.extra
        assert cfg.physics.extra["pic"]["omega_p_over_omega_c"] == 10.0

    def test_grid(self, cfg: Any) -> None:
        assert cfg.grid.dimensions == (100, 100, 1)
        np.testing.assert_allclose(cfg.grid.spacing, (0.3, 0.3, 1.0), rtol=1e-12)
        assert cfg.grid.dt == 0.125
        assert cfg.grid.boundary == ("periodic", "periodic", "periodic")

    def test_species(self, cfg: Any) -> None:
        assert len(cfg.species) == 4
        assert cfg.species[0].charge == -1.0
        np.testing.assert_allclose(cfg.species[0].mass, 1 / 256, rtol=1e-10)

    def test_normalization(self, cfg: Any) -> None:
        np.testing.assert_allclose(
            cfg.normalization.velocity_ref, constants.c, rtol=1e-10
        )


class TestMinimal:
    def test_minimal_doc_loads(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _shell()))
        assert cfg.model_name == "test"

    def test_missing_model_section_raises(self, tmp_path: Path) -> None:
        toml = _shell().replace('[model]\nname = "test"\ntype = "PIC"\n\n', "")
        with pytest.raises(ValidationError, match="model"):
            load_config(_write(tmp_path, toml))

    def test_missing_type_raises(self, tmp_path: Path) -> None:
        toml = _shell(model='[model]\nname = "test"')
        with pytest.raises(ValidationError, match="type"):
            load_config(_write(tmp_path, toml))


class TestGrid:
    def test_full(self, tmp_path: Path) -> None:
        grid = (
            "[grid]\ndimensions = [10, 20, 30]\nspacing = [0.1, 0.2, 0.3]\n"
            "lower = [1.0, 2.0, 3.0]\nupper = [2.0, 6.0, 12.0]"
        )
        bcs = (
            "[boundary_conditions]\n"
            'lower = ["periodic", "open", "reflecting"]\n'
            'upper = ["periodic", "open", "reflecting"]'
        )
        cfg = load_config(_write(tmp_path, _shell(grid=grid, extra=bcs)))
        assert cfg.grid.dimensions == (10, 20, 30)
        np.testing.assert_allclose(cfg.grid.origin, (1.0, 2.0, 3.0), rtol=1e-12)
        assert cfg.grid.boundary == ("periodic", "open", "reflecting")

    def test_lower_defaults_origin_for_grid_info(self, tmp_path: Path) -> None:
        # [grid].lower in v1.0 → GridInfo.origin internally
        cfg = load_config(_write(tmp_path, _shell()))
        assert cfg.grid.origin == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("missing_key", ["dimensions", "spacing"])
    def test_missing_required_grid_key(self, tmp_path: Path, missing_key: str) -> None:
        if missing_key == "dimensions":
            grid = (
                "[grid]\nspacing = [1.0, 1.0, 1.0]\n"
                "lower = [0.0, 0.0, 0.0]\nupper = [2.0, 2.0, 2.0]"
            )
        else:
            grid = (
                "[grid]\ndimensions = [2, 2, 2]\n"
                "lower = [0.0, 0.0, 0.0]\nupper = [2.0, 2.0, 2.0]"
            )
        with pytest.raises(ValidationError, match=missing_key):
            load_config(_write(tmp_path, _shell(grid=grid)))


class TestUnits:
    def test_pic_electron_default(self, tmp_path: Path) -> None:
        units = '[units]\nsystem = "PIC"\nreference_density = 1.0e18'
        cfg = load_config(_write(tmp_path, _shell(units=units)))
        np.testing.assert_allclose(
            cfg.normalization.velocity_ref, constants.c, rtol=1e-10
        )
        np.testing.assert_allclose(
            cfg.normalization.mass_ref, constants.m_e, rtol=1e-10
        )

    def test_pic_ion_reference(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "PIC"\nreference_species = "ions"\n'
            "reference_density = 1.0e18"
        )
        cfg = load_config(_write(tmp_path, _shell(units=units)))
        np.testing.assert_allclose(
            cfg.normalization.mass_ref, constants.m_p, rtol=1e-10
        )

    def test_pic_unknown_species_requires_explicit_mass(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "PIC"\nreference_species = "alpha"\n'
            "reference_density = 1.0e18"
        )
        with pytest.raises(ValidationError, match="reference_species"):
            load_config(_write(tmp_path, _shell(units=units)))

    def test_pic_unknown_species_with_explicit_mass(self, tmp_path: Path) -> None:
        species = '[[species]]\nname = "alpha"\ncharge = 2.0\nmass = 4.0'
        units = (
            '[units]\nsystem = "PIC"\nreference_species = "alpha"\n'
            "reference_density = 1.0e18\n"
            "reference_mass = 6.644e-27\n"
            "reference_charge = 3.204e-19"
        )
        cfg = load_config(_write(tmp_path, _shell(units=units, species=species)))
        np.testing.assert_allclose(cfg.normalization.mass_ref, 6.644e-27, rtol=1e-10)

    def test_mhd(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "MHD"\nreference_length = 6.371e6\n'
            "reference_density = 1.67e-17\nreference_b_field = 5.0e-9"
        )
        cfg = load_config(
            _write(
                tmp_path,
                _shell(
                    model='[model]\nname = "t"\ntype = "MHD"',
                    units=units,
                ),
            )
        )
        assert cfg.normalization.length_ref == 6.371e6

    def test_si_identity(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _shell()))
        assert cfg.normalization.is_identity

    def test_custom(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "custom"\n'
            "[units.reference]\nlength = 5.31e-3\ntime = 1.77e-11\n"
            "velocity = 2.998e8\nb_field = 1.07e-3\ne_field = 3.21e5\n"
            "density = 1.0e18\nmass = 9.109e-31\ncharge = 1.602e-19"
        )
        cfg = load_config(_write(tmp_path, _shell(units=units)))
        np.testing.assert_allclose(cfg.normalization.length_ref, 5.31e-3, rtol=1e-10)

    def test_custom_missing_reference_subtable(self, tmp_path: Path) -> None:
        units = '[units]\nsystem = "custom"'
        with pytest.raises(ValidationError, match="reference"):
            load_config(_write(tmp_path, _shell(units=units)))

    def test_unknown_system(self, tmp_path: Path) -> None:
        units = '[units]\nsystem = "CGS"'
        with pytest.raises(ValidationError, match="CGS"):
            load_config(_write(tmp_path, _shell(units=units)))

    def test_scaling_metadata(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "SI"\nscaling_factor = 10.0\n'
            'scaling_description = "reduced c/v_A"'
        )
        cfg = load_config(_write(tmp_path, _shell(units=units)))
        assert cfg.metadata["scaling"]["scaling_factor"] == 10.0


class TestCoordinates:
    @pytest.mark.parametrize(
        ("geom_str", "expected_names"),
        [
            ("cartesian", ("x", "y", "z")),
            ("spherical", ("r", "θ", "φ")),
            ("cylindrical", ("r", "φ", "z")),
        ],
    )
    def test_geometry_lookup(
        self,
        tmp_path: Path,
        geom_str: str,
        expected_names: tuple[str, ...],
    ) -> None:
        coords = f'[coordinates]\ngeometry = "{geom_str}"\nframe = "sim"'
        cfg = load_config(_write(tmp_path, _shell(coordinates=coords)))
        assert cfg.grid.geometry.axis_names == expected_names

    def test_custom_axis_labels(self, tmp_path: Path) -> None:
        coords = (
            '[coordinates]\ngeometry = "cartesian"\nframe = "sim"\n'
            'axis_labels = ["X", "Y", "Z"]'
        )
        cfg = load_config(_write(tmp_path, _shell(coordinates=coords)))
        assert cfg.grid.geometry.axis_names == ("X", "Y", "Z")

    def test_unknown_geometry_raises(self, tmp_path: Path) -> None:
        coords = '[coordinates]\ngeometry = "toroidal"\nframe = "sim"'
        with pytest.raises(ValidationError, match="toroidal"):
            load_config(_write(tmp_path, _shell(coordinates=coords)))

    def test_missing_geometry_raises(self, tmp_path: Path) -> None:
        coords = '[coordinates]\nframe = "sim"'
        with pytest.raises(ValidationError, match="geometry"):
            load_config(_write(tmp_path, _shell(coordinates=coords)))


class TestSpecies:
    def test_single_species_charge_and_mass(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _shell()))
        assert len(cfg.species) == 1
        assert cfg.species[0].name == "e"

    def test_charge_to_mass_only(self, tmp_path: Path) -> None:
        species = '[[species]]\nname = "e"\ncharge_to_mass = -256.0'
        cfg = load_config(_write(tmp_path, _shell(species=species)))
        assert cfg.species[0].charge == -1.0
        assert cfg.species[0].mass is not None
        np.testing.assert_allclose(cfg.species[0].mass, 1.0 / 256, rtol=1e-10)

    def test_both_forms_rejected(self, tmp_path: Path) -> None:
        species = (
            '[[species]]\nname = "e"\ncharge = -1.0\nmass = 0.004\n'
            "charge_to_mass = -256.0"
        )
        with pytest.raises(ValidationError, match="choose one"):
            load_config(_write(tmp_path, _shell(species=species)))

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
        self,
        tmp_path: Path,
        field_name: str,
        toml_value: str,
        attr: str,
        expected: tuple[float, ...],
    ) -> None:
        species = (
            f'[[species]]\nname = "e"\ncharge = -1.0\nmass = 0.004\n'
            f"{field_name} = {toml_value}"
        )
        cfg = load_config(_write(tmp_path, _shell(species=species)))
        assert getattr(cfg.species[0], attr) == expected

    def test_zero_species_rejected(self, tmp_path: Path) -> None:
        toml = _shell(species="")
        with pytest.raises(ValidationError, match="species"):
            load_config(_write(tmp_path, toml))


class TestValidationAggregation:
    def test_multiple_errors_in_one_exception(self, tmp_path: Path) -> None:
        # Deliberately break model, units, and coordinates. Pydantic
        # reports all errors in a single ValidationError (not ExceptionGroup).
        toml = _shell(
            model='[model]\nname = "t"\ntype = "not_a_type"',
            units='[units]\nsystem = "CGS"',
            coordinates='[coordinates]\ngeometry = "toroidal"\nframe = "sim"',
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(_write(tmp_path, toml))
        # Pydantic gathers all violations per model_validate call
        assert exc_info.value.error_count() >= 3


_SCALING_TOML = _shell(
    model='[model]\nname = "test"\ntype = "PIC"',
    grid=(
        "[grid]\ndimensions = [460, 130, 320]\nspacing = [0.4, 0.4, 0.4]\n"
        "lower = [0.0, 0.0, 0.0]\nupper = [184.0, 52.0, 128.0]"
    ),
    coordinates=(
        '[coordinates]\ngeometry = "cartesian"\nframe = "simulation"\n'
        'physical_extent = [46.0, 32.0, 13.0]\nphysical_extent_unit = "R_E"\n'
        "[coordinates.transforms.GSM]\n"
        "origin = [52.0, 26.0, 64.0]\n"
        "rotation = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]\n"
        'axis_labels = ["x_GSM", "y_GSM", "z_GSM"]'
    ),
    units=(
        '[units]\nsystem = "PIC"\nreference_species = "ions"\n'
        "reference_density = 0.25e6"
    ),
    species='[[species]]\nname = "ions"\ncharge = 1.0\nmass = 1.0',
)


class TestPhysicalExtent:
    def test_auto_scale_from_toml(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _SCALING_TOML))
        np.testing.assert_allclose(cfg.transforms["GSM"].scale, 0.25, rtol=1e-6)

    def test_shrink_factor_computed(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _SCALING_TOML))
        shrink = cfg.metadata["scaling"]["shrink_factor"]
        np.testing.assert_allclose(shrink, 3.5, rtol=0.01)

    def test_physical_extent_in_metadata(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _SCALING_TOML))
        assert cfg.metadata["physical_extent"] == (46.0, 32.0, 13.0)
        assert cfg.metadata["physical_extent_unit"] == "R_E"

    def test_explicit_scale_validated(self, tmp_path: Path) -> None:
        toml = _SCALING_TOML.replace(
            "axis_labels = [",
            "scale = 0.25\naxis_labels = [",
        )
        cfg = load_config(_write(tmp_path, toml))
        np.testing.assert_allclose(cfg.transforms["GSM"].scale, 0.25)

    def test_explicit_scale_mismatch_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        toml = _SCALING_TOML.replace(
            "axis_labels = [",
            "scale = 0.30\naxis_labels = [",
        )
        with caplog.at_level(logging.WARNING, logger="pypic.readers.config"):
            cfg = load_config(_write(tmp_path, toml))
        assert cfg.transforms["GSM"].scale == 0.30
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
    def base_config(self) -> Any:
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

    def test_no_sim_dir_returns_base_unchanged(self, base_config: Any) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        result = merge_simulation_toml(None, base_config)
        assert result is base_config

    def test_missing_toml_returns_base_unchanged(
        self, tmp_path: Path, base_config: Any
    ) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        result = merge_simulation_toml(tmp_path, base_config)
        assert result is base_config

    def test_toml_overrides_normalization_frame_metadata(
        self, tmp_path: Path, base_config: Any
    ) -> None:
        from pypic.readers._config_helpers import merge_simulation_toml

        _write(
            tmp_path,
            _shell(
                model='[model]\nname = "test"\ntype = "MHD"',
                grid=(
                    "[grid]\ndimensions = [10, 10, 10]\n"
                    "spacing = [0.5, 0.5, 0.5]\n"
                    "lower = [0.0, 0.0, 0.0]\nupper = [5.0, 5.0, 5.0]"
                ),
                units=(
                    '[units]\nsystem = "MHD"\nreference_length = 6.371e6\n'
                    "reference_density = 1.67e-17\nreference_b_field = 5.0e-9"
                ),
                coordinates=(
                    '[coordinates]\ngeometry = "cartesian"\nframe = "simulation"'
                ),
                species='[[species]]\nname = "p"\ncharge = 1.0\nmass = 1.0',
                extra=(
                    "[output.fields]\nstep_interval = 10\n"
                    'quantities = ["B"]\ndir = "./fields"'
                ),
            ),
        )
        result = merge_simulation_toml(tmp_path, base_config)
        assert result.frame == "simulation"
        assert result.metadata["reader_only"] == "kept"
        assert result.output is not None
        assert result.output.fields is not None
        assert result.output.fields.step_interval == 10
        assert result.model_name == "ReaderX"
        assert result.grid.dimensions == (4, 3, 2)
        assert result.normalization.length_ref == 6.371e6

    def test_toml_metadata_wins_on_key_conflict(
        self, tmp_path: Path, base_config: Any
    ) -> None:
        from pypic.containers import SimulationConfig
        from pypic.coordinates.geometry import CARTESIAN
        from pypic.grid import GridInfo
        from pypic.readers._config_helpers import merge_simulation_toml
        from pypic.units import Normalization

        base = SimulationConfig(
            model_name="X",
            model_type="MHD",
            grid=GridInfo(
                dimensions=(4,),
                spacing=(1.0,),
                origin=(0.0,),
                geometry=CARTESIAN,
            ),
            normalization=Normalization.identity(),
            metadata={"description": "from-reader", "reader_only": "kept"},
        )
        _write(
            tmp_path,
            _shell(
                model=(
                    '[model]\nname = "test"\ntype = "MHD"\ndescription = "from-toml"'
                ),
                grid=(
                    "[grid]\ndimensions = [10, 10, 10]\n"
                    "spacing = [0.5, 0.5, 0.5]\n"
                    "lower = [0.0, 0.0, 0.0]\nupper = [5.0, 5.0, 5.0]"
                ),
                coordinates=(
                    '[coordinates]\ngeometry = "cartesian"\nframe = "simulation"'
                ),
                species='[[species]]\nname = "p"\ncharge = 1.0\nmass = 1.0',
            ),
        )
        result = merge_simulation_toml(tmp_path, base)
        assert result.metadata["description"] == "from-toml"
        assert result.metadata["reader_only"] == "kept"


class TestInitialConditionsAndOutput:
    """Schema-validated `[initial_conditions]` and `[output]` reach the
    typed attributes on SimulationConfig (E2 — no longer dict-dumped into
    metadata).
    """

    def test_initial_conditions_typed(self, tmp_path: Path) -> None:
        ic = (
            "[initial_conditions]\n"
            'type = "double_harris"\n'
            "B0 = [0.05, 0.0, 0.0]\n"
            "current_sheet_thickness = 0.5"
        )
        cfg = load_config(_write(tmp_path, _shell(extra=ic)))
        assert cfg.initial_conditions is not None
        assert cfg.initial_conditions.type == "double_harris"
        # Extra (setup-specific) keys reach the model via _ExtensibleBase
        assert cfg.initial_conditions.model_dump()["B0"] == [0.05, 0.0, 0.0]

    def test_initial_conditions_absent(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, _shell()))
        assert cfg.initial_conditions is None

    def test_output_typed(self, tmp_path: Path) -> None:
        out = (
            "[output.fields]\n"
            "step_interval = 50\n"
            'quantities = ["B", "E"]\n'
            'dir = "./fields"\n'
            "[output.checkpoints]\n"
            "step_interval = 5000\n"
            'dir = "./chk"\n'
        )
        cfg = load_config(_write(tmp_path, _shell(extra=out)))
        assert cfg.output is not None
        assert cfg.output.fields is not None
        assert cfg.output.fields.step_interval == 50
        assert cfg.output.fields.quantities == ["B", "E"]
        assert cfg.output.checkpoints is not None
        assert cfg.output.checkpoints.step_interval == 5000


class TestBodiesDriversRestart:
    """Validated `[[bodies]]`, `[[drivers]]`, `[restart]` survive translation."""

    def test_bodies_typed(self, tmp_path: Path) -> None:
        body = (
            "[[bodies]]\n"
            'name = "mercury"\n'
            "center = [0.0, 0.0, 0.0]\n"
            "radius = 60.0\n"
            "intrinsic_dipole = [0.0, 0.0, -190.0]\n"
            "rotation_axis = [0.0, 0.0, 1.0]\n"
        )
        cfg = load_config(_write(tmp_path, _shell(extra=body)))
        assert len(cfg.bodies) == 1
        assert cfg.bodies[0].name == "mercury"
        assert cfg.bodies[0].radius == 60.0

    def test_drivers_typed(self, tmp_path: Path) -> None:
        drv = (
            "[[drivers]]\n"
            'name = "magnetogram"\n'
            'type = "magnetogram_timeseries"\n'
            'coupling = "boundary"\n'
            "cadence = 720.0\n"
            "target_lower = [1.0, 0.87, -0.87]\n"
            "target_upper = [1.0, 2.60, 0.87]\n"  # thin photospheric sheet
        )
        cfg = load_config(_write(tmp_path, _shell(extra=drv)))
        assert len(cfg.drivers) == 1
        assert cfg.drivers[0].name == "magnetogram"
        assert cfg.drivers[0].direction == "one_way"  # default

    def test_driver_inverted_box_rejected(self, tmp_path: Path) -> None:
        drv = (
            "[[drivers]]\n"
            'name = "bad"\n'
            'type = "x"\n'
            'coupling = "boundary"\n'
            "target_lower = [0.0, 0.0, 0.0]\n"
            "target_upper = [-1.0, 1.0, 1.0]\n"
        )
        with pytest.raises(ValidationError, match=">="):
            load_config(_write(tmp_path, _shell(extra=drv)))

    def test_restart_typed(self, tmp_path: Path) -> None:
        rs = '[restart]\nfrom = "./chk/chk_000030.h5"\nstep = 30000\ntime = 1500.0\n'
        cfg = load_config(_write(tmp_path, _shell(extra=rs)))
        assert cfg.restart is not None
        assert cfg.restart.from_ == "./chk/chk_000030.h5"
        assert cfg.restart.step == 30000


class TestForwardedSections:
    """Sections previously dropped at the translator boundary now survive.

    Probes, collisions, phase_space, and the full ``[run]`` provenance
    record reach ``SimulationConfig`` as raw schema objects — same
    pattern as bodies/drivers/restart.
    """

    def test_run_provenance_typed(self, tmp_path: Path) -> None:
        run = (
            '[run]\nname = "harris-r2"\n'
            'doi = "10.5281/zenodo.12345"\n'
            'license = "CC-BY-4.0"\n'
            'funding = ["NSF-AGS-2024001"]\n'
            "random_seed = 42\n"
        )
        cfg = load_config(
            _write(
                tmp_path, _shell(extra="").replace('[run]\nname = "r0"', run.rstrip())
            )
        )
        assert cfg.run is not None
        assert cfg.run.doi == "10.5281/zenodo.12345"
        assert cfg.run.license == "CC-BY-4.0"
        assert cfg.run.funding == ["NSF-AGS-2024001"]
        assert cfg.run.random_seed == 42

    def test_probes_typed(self, tmp_path: Path) -> None:
        prb = (
            "[[probes]]\n"
            'name = "magnetopause"\n'
            "position = [10.0, 0.0, 0.0]\n"
            'fields = ["B1", "B2", "B3"]\n'
        )
        cfg = load_config(_write(tmp_path, _shell(extra=prb)))
        assert len(cfg.probes) == 1
        assert cfg.probes[0].name == "magnetopause"
        assert cfg.probes[0].position == [10.0, 0.0, 0.0]

    def test_collisions_typed(self, tmp_path: Path) -> None:
        col = (
            '[[species]]\nname = "i"\ncharge = 1.0\nmass = 1.0\n'
            "[[collisions]]\n"
            'species_pair = ["e", "i"]\n'
            'model = "coulomb"\n'
            "coulomb_log = 10.0\n"
        )
        cfg = load_config(_write(tmp_path, _shell(extra=col)))
        assert len(cfg.collisions) == 1
        assert cfg.collisions[0].species_pair == ["e", "i"]
        assert cfg.collisions[0].coulomb_log == 10.0

    def test_phase_space_storage_typed(self, tmp_path: Path) -> None:
        ps = (
            "[phase_space]\n"
            "dimensions = [2, 2, 2, 50, 50, 50]\n"
            'axis_labels = ["x", "y", "z", "vx", "vy", "vz"]\n'
            "[phase_space.storage]\n"
            "block_size = [10, 10, 10]\n"
            "sparsity_threshold = 1.0e-15\n"
        )
        cfg = load_config(_write(tmp_path, _shell(extra=ps)))
        assert cfg.phase_space is not None
        assert cfg.phase_space.storage is not None
        assert cfg.phase_space.storage.block_size == [10, 10, 10]
        assert cfg.phase_space.storage.sparsity_threshold == 1.0e-15

    def test_phase_space_typed(self, tmp_path: Path) -> None:
        ps = (
            "[phase_space]\n"
            "dimensions = [2, 2, 2, 16, 8]\n"
            'axis_labels = ["x", "y", "z", "vpar", "mu"]\n'
            'coordinate_system = "guiding-center"\n'
        )
        cfg = load_config(_write(tmp_path, _shell(extra=ps)))
        assert cfg.phase_space is not None
        assert cfg.phase_space.dimensions == [2, 2, 2, 16, 8]
        assert cfg.phase_space.coordinate_system == "guiding-center"

    def test_stagger_per_component_round_trips_into_stagger_info(
        self, tmp_path: Path
    ) -> None:
        # ``[grid.stagger_fields]`` and ``[grid.stagger_position]`` survive
        # translation as ``StaggerInfo.field_locations`` and
        # ``StaggerInfo.position`` on ``cfg.metadata['stagger']``.
        grid_with_stagger = (
            "[grid]\n"
            "dimensions = [2, 2, 2]\nspacing = [1.0, 1.0, 1.0]\n"
            "lower = [0.0, 0.0, 0.0]\nupper = [2.0, 2.0, 2.0]\n"
            'stagger = "staggered"\n'
            "[grid.stagger_fields]\n"
            'B = "face"\n'
            'E = "edge"\n'
            "[grid.stagger_position]\n"
            "B1 = [0.5, 0.0, 0.0]\n"
            "E1 = [0.0, 0.5, 0.5]\n"
        )
        cfg = load_config(_write(tmp_path, _shell(grid=grid_with_stagger)))
        stagger = cfg.metadata["stagger"]
        assert stagger.convention == "staggered"
        assert stagger.field_locations is not None
        assert stagger.field_locations["B"] == "face"
        assert stagger.position is not None
        assert stagger.position["B1"] == (0.5, 0.0, 0.0)


class TestPhysicsBranches:
    """Translator handles MHD and hybrid physics branches symmetrically."""

    def test_mhd_branch_into_extra(self, tmp_path: Path) -> None:
        units = (
            '[units]\nsystem = "MHD"\nreference_length = 6.371e6\n'
            "reference_density = 1.67e-17\nreference_b_field = 5.0e-9"
        )
        physics = (
            "[physics]\nrelativistic = false\n"
            "[physics.mhd]\ngamma = 1.4\nresistivity = 1.0e-5\nhall_term = true\n"
            '[physics.mhd.solver]\nscheme = "godunov"\nlimiter = "minmod"'
        )
        cfg = load_config(
            _write(
                tmp_path,
                _shell(
                    model='[model]\nname = "t"\ntype = "MHD"',
                    units=units,
                    extra=physics,
                ),
            )
        )
        assert cfg.physics.gamma == 1.4
        assert "mhd" in cfg.physics.extra
        assert cfg.physics.extra["mhd"]["resistivity"] == 1.0e-5
        assert cfg.physics.extra["mhd"]["solver"]["scheme"] == "godunov"

    def test_hybrid_branch_into_extra(self, tmp_path: Path) -> None:
        physics = (
            "[physics]\n[physics.hybrid]\n"
            '[physics.hybrid.solver]\nscheme = "predictor-corrector"\n'
            "resistivity = 5.0e-4\ncurrent_smoothing = 2"
        )
        cfg = load_config(
            _write(
                tmp_path,
                _shell(
                    model='[model]\nname = "t"\ntype = "hybrid"',
                    extra=physics,
                ),
            )
        )
        assert "hybrid" in cfg.physics.extra
        assert cfg.physics.extra["hybrid"]["solver"]["scheme"] == "predictor-corrector"

    def test_physics_branch_must_match_model_type(self, tmp_path: Path) -> None:
        physics = (
            "[physics]\n[physics.pic]\nomega_p_over_omega_c = 10.0\n"
            '[physics.pic.solver]\nscheme = "explicit"'
        )
        with pytest.raises(ValidationError, match=r"model\.type"):
            load_config(
                _write(
                    tmp_path,
                    _shell(
                        model='[model]\nname = "t"\ntype = "MHD"',
                        units=(
                            '[units]\nsystem = "MHD"\nreference_length = 1e6\n'
                            "reference_density = 1.0e-15\nreference_b_field = 1e-9"
                        ),
                        extra=physics,
                    ),
                )
            )


class TestGridAMR:
    """`[grid.amr]` and `[[grid.refinement]]` reach the validated schema
    object; pypic does not yet expose them on SimulationConfig but they
    must validate cleanly."""

    def test_amr_validates(self, tmp_path: Path) -> None:
        grid = (
            "[grid]\ndimensions = [16, 16, 16]\nspacing = [1.0, 1.0, 1.0]\n"
            "lower = [0.0, 0.0, 0.0]\nupper = [16.0, 16.0, 16.0]\n"
            "[grid.amr]\nmax_level = 4\nrefinement_ratio = 2\n"
            "block_size = [8, 8, 8]\nrefinement_threshold = 0.1\n"
            "[[grid.refinement]]\n"
            "level = 2\nbox = [[-60.0, -60.0, -60.0], [60.0, 60.0, 60.0]]"
        )
        cfg = load_config(_write(tmp_path, _shell(grid=grid)))
        assert cfg.grid.dimensions == (16, 16, 16)


class TestCoordinateTransforms:
    """Translation of `[coordinates.transforms.*]` covers chaining and
    parameter-driven (time-dependent) transforms."""

    def test_chain_via_from_frame(self, tmp_path: Path) -> None:
        coords = (
            '[coordinates]\ngeometry = "cartesian"\nframe = "simulation"\n'
            "[coordinates.transforms.GSE]\norigin = [0.0, 0.0, 0.0]\n"
            '[coordinates.transforms.GSM]\nfrom_frame = "GSE"\n'
            "origin = [0.0, 0.0, 0.0]"
        )
        cfg = load_config(_write(tmp_path, _shell(coordinates=coords)))
        assert "GSE" in cfg.transforms
        assert "GSM" in cfg.transforms
        # GSM transform's source_frame is the chained "GSE", not the native frame
        assert cfg.transforms["GSM"].source_frame == "GSE"
        assert cfg.transforms["GSE"].source_frame == "simulation"

    def test_parameter_driven_transform(self, tmp_path: Path) -> None:
        # Time-dependent transforms carry a `parameter` name; the FrameTransform
        # itself doesn't yet act on it, but the schema field must round-trip.
        coords = (
            '[coordinates]\ngeometry = "cartesian"\nframe = "simulation"\n'
            "[coordinates.transforms.GSM]\n"
            'parameter = "dipole_tilt"\norigin = [0.0, 0.0, 0.0]'
        )
        cfg = load_config(_write(tmp_path, _shell(coordinates=coords)))
        assert "GSM" in cfg.transforms


class TestRoundTrip:
    """A full v1.0 doc → schema → SimulationConfig pipeline does not lose
    structural information for the typed attributes pypic now exposes."""

    def test_roundtrip_preserves_typed_sections(self, tmp_path: Path) -> None:
        doc = _shell(
            extra=(
                "[initial_conditions]\n"
                'type = "harris"\nB0 = [0.1, 0.0, 0.0]\n'
                "[output.fields]\n"
                "step_interval = 25\n"
                'quantities = ["B"]\n'
                'dir = "./out"\n'
                "[[bodies]]\n"
                'name = "earth"\ncenter = [0.0, 0.0, 0.0]\nradius = 1.0\n'
                "[[drivers]]\n"
                'name = "sw"\ntype = "solar_wind"\ncoupling = "boundary"\n'
                "[restart]\n"
                'from = "./chk.h5"\n'
            )
        )
        cfg = load_config(_write(tmp_path, doc))
        assert cfg.initial_conditions is not None
        assert cfg.initial_conditions.type == "harris"
        assert cfg.output is not None
        assert cfg.output.fields is not None
        assert cfg.output.fields.step_interval == 25
        assert len(cfg.bodies) == 1
        assert cfg.bodies[0].name == "earth"
        assert len(cfg.drivers) == 1
        assert cfg.drivers[0].coupling == "boundary"
        assert cfg.restart is not None
        assert cfg.restart.from_ == "./chk.h5"
