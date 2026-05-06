"""Tests for ``pypic.schema`` — the v1.0 simulation.toml validator."""

from __future__ import annotations

import tomllib
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from pypic.schema import (
    SimulationSchema,
    UnitsMHD,
    UnitsPIC,
    validate_simulation_toml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_TEMPLATE = REPO_ROOT / "pypic.simulation.toml"


def _minimal_doc(**overrides: str) -> str:
    """Build a minimal valid v1.0 doc, with targeted overrides."""
    base = (
        dedent(
            """
        schema_version = "1.0"
        [schema]
        version = "1.0"
        [model]
        name = "demo"
        type = "PIC"
        [run]
        name = "r0"
        [time]
        scheme = "fixed"
        dt = 0.1
        t_start = 0.0
        t_end = 1.0
        n_steps = 10
        [grid]
        dimensions = [4, 4, 4]
        spacing = [1.0, 1.0, 1.0]
        lower = [0.0, 0.0, 0.0]
        upper = [4.0, 4.0, 4.0]
        [units]
        system = "SI"
        [coordinates]
        geometry = "cartesian"
        frame = "sim"
        [[species]]
        name = "electrons"
        charge = -1.0
        mass = 1.0
        """
        ).strip()
        + "\n"
    )
    for marker, replacement in overrides.items():
        base = base.replace(marker, replacement)
    return base


def _extract_scenario_toml(full_text: str, scenario: str) -> str:
    """Extract Scenario B or C from the live template by uncommenting.

    The template uses a disciplined pattern: Scenarios B and C are
    commented with a single leading ``# `` per TOML line, and any
    *prose* inside the scenario is double-commented (``# # ``). Strip
    one level and the result round-trips through ``tomllib``.
    """
    markers = {
        "B": "Scenario B — solar eruption MHD (ARMS-style active-region",
        "C": "Scenario C — heliospheric hybrid (Mercury magnetosphere",
    }
    start_needle = markers[scenario]
    lines = full_text.splitlines()
    idx_start = next(i for i, ln in enumerate(lines) if start_needle in ln)
    idx_model = next(
        i
        for i, ln in enumerate(lines[idx_start:], idx_start)
        if ln.strip() == "# [model]"
    )
    idx_end = len(lines)
    for i in range(idx_model + 1, len(lines)):
        if lines[i].startswith("# =======") or (
            i + 1 < len(lines) and lines[i + 1].startswith("# =======")
        ):
            idx_end = i
            break
    extracted: list[str] = []
    for ln in lines[idx_model:idx_end]:
        if ln.startswith("# "):
            extracted.append(ln[2:])
        elif ln == "#":
            extracted.append("")
        else:
            extracted.append(ln)
    # Scenarios B/C lack the bare `schema_version` top-level key.
    return 'schema_version = "1.0"\n' + "\n".join(extracted) + "\n"


class TestLiveTemplate:
    """The live template file must validate as-is (Scenario A active)."""

    def test_scenario_a_validates(self) -> None:
        result = validate_simulation_toml(LIVE_TEMPLATE)
        assert isinstance(result, SimulationSchema)

    def test_scenario_a_identity(self) -> None:
        s = validate_simulation_toml(LIVE_TEMPLATE)
        assert s.model.name == "iPIC3D"
        assert s.model.type == "PIC"
        assert s.schema_version == "1.0"

    def test_scenario_a_species_order(self) -> None:
        s = validate_simulation_toml(LIVE_TEMPLATE)
        assert [sp.name for sp in s.species] == ["electrons", "ions"]

    def test_scenario_a_units_discriminated(self) -> None:
        s = validate_simulation_toml(LIVE_TEMPLATE)
        assert isinstance(s.units, UnitsPIC)
        assert s.units.reference_species == "ions"

    def test_scenario_a_pic_solver_block_present(self) -> None:
        s = validate_simulation_toml(LIVE_TEMPLATE)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.scheme == "semi-implicit"
        assert s.physics.pic.solver.preconditioner == "block-jacobi"


class TestScenarioBMHD:
    """Validate Scenario B (ARMS-like spherical MHD) after uncommenting."""

    def test_scenario_b_validates(self) -> None:
        toml = _extract_scenario_toml(LIVE_TEMPLATE.read_text(), "B")
        s = validate_simulation_toml(toml)
        assert s.model.type == "MHD"
        assert s.coordinates.geometry == "spherical"
        assert isinstance(s.units, UnitsMHD)
        assert s.time.scheme == "adaptive"
        assert s.physics is not None
        assert s.physics.mhd is not None
        assert s.physics.mhd.solver is not None
        assert s.physics.mhd.solver.scheme == "fct"


class TestScenarioCHybrid:
    """Validate Scenario C (AIKEF-like Mercury hybrid)."""

    def test_scenario_c_validates(self) -> None:
        toml = _extract_scenario_toml(LIVE_TEMPLATE.read_text(), "C")
        s = validate_simulation_toml(toml)
        assert s.model.type == "hybrid"
        assert s.physics is not None
        assert s.physics.hybrid is not None
        assert s.physics.hybrid.solver is not None
        assert s.physics.hybrid.solver.scheme == "predictor-corrector"
        assert len(s.bodies) == 1
        assert s.bodies[0].name == "mercury"


class TestMinimalDoc:
    def test_roundtrips(self) -> None:
        s = validate_simulation_toml(_minimal_doc())
        assert s.model.name == "demo"

    def test_rejects_missing_required(self) -> None:
        doc = _minimal_doc().replace('[model]\nname = "demo"\ntype = "PIC"\n', "")
        with pytest.raises(ValidationError, match="model"):
            validate_simulation_toml(doc)

    def test_rejects_unknown_top_level_without_x_prefix(self) -> None:
        doc = _minimal_doc() + '\n[not_a_real_section]\nfoo = "bar"\n'
        with pytest.raises(ValidationError, match="extension"):
            validate_simulation_toml(doc)

    def test_accepts_x_dash_extension_section(self) -> None:
        doc = _minimal_doc() + '\n[x-warpx]\ndeposition = "esirkepov"\n'
        s = validate_simulation_toml(doc)
        assert s.model_extra is not None
        assert "x-warpx" in s.model_extra

    def test_accepts_x_underscore_extension_section(self) -> None:
        doc = _minimal_doc() + "\n[x_custom]\nfoo = 1\n"
        validate_simulation_toml(doc)

    def test_schema_version_mismatch(self) -> None:
        doc = _minimal_doc().replace('schema_version = "1.0"', 'schema_version = "1.1"')
        with pytest.raises(ValidationError, match="must match"):
            validate_simulation_toml(doc)

    def test_rejects_non_v1_schema(self) -> None:
        doc = dedent("""
            schema_version = "2.0"
            [schema]
            version = "2.0"
            [model]
            name = "d"
            type = "PIC"
            [run]
            name = "r"
            [time]
            dt = 0.1
            t_start = 0.0
            t_end = 1.0
            n_steps = 10
            [grid]
            dimensions = [4,4,4]
            spacing = [1.0,1.0,1.0]
            lower = [0.0,0.0,0.0]
            upper = [4.0,4.0,4.0]
            [units]
            system = "SI"
            [coordinates]
            geometry = "cartesian"
            frame = "sim"
            [[species]]
            name = "e"
            charge = -1
            mass = 1
        """).strip()
        with pytest.raises(ValidationError, match=r"1\.x"):
            validate_simulation_toml(doc)


class TestCrossSectionInvariants:
    def test_model_type_pic_forbids_mhd_block(self) -> None:
        doc = _minimal_doc() + "\n[physics]\n[physics.mhd]\ngamma = 1.5\n"
        with pytest.raises(ValidationError, match=r"model\.type"):
            validate_simulation_toml(doc)

    def test_model_type_mhd_permits_only_mhd_block(self) -> None:
        doc = (
            _minimal_doc(**{'type = "PIC"': 'type = "MHD"'})
            + "\n[physics]\n[physics.mhd]\ngamma = 1.5\n"
        )
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.mhd is not None

    def test_grid_axis_length_mismatch(self) -> None:
        doc = _minimal_doc(**{"spacing = [1.0, 1.0, 1.0]": "spacing = [1.0, 1.0]"})
        with pytest.raises(ValidationError, match="spacing"):
            validate_simulation_toml(doc)

    def test_grid_upper_not_above_lower(self) -> None:
        doc = _minimal_doc(**{"upper = [4.0, 4.0, 4.0]": "upper = [4.0, 0.0, 4.0]"})
        with pytest.raises(ValidationError, match="upper"):
            validate_simulation_toml(doc)

    def test_boundary_conditions_axis_mismatch(self) -> None:
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["periodic", "periodic"]
            upper = ["periodic", "periodic"]
        """)
        with pytest.raises(ValidationError, match="boundary_conditions"):
            validate_simulation_toml(doc)

    def test_reference_species_must_exist(self) -> None:
        doc = _minimal_doc(
            **{
                'system = "SI"': dedent("""
                    system = "PIC"
                    reference_species = "ghost_species"
                    reference_density = 1.0e6
                """).strip(),
            }
        )
        with pytest.raises(ValidationError, match="reference_species"):
            validate_simulation_toml(doc)

    def test_time_subcycled_requires_field_substeps(self) -> None:
        doc = _minimal_doc(**{'scheme = "fixed"': 'scheme = "subcycled"'})
        with pytest.raises(ValidationError, match="subcycled"):
            validate_simulation_toml(doc)


class TestSpecies:
    def test_species_charge_plus_mass_accepted(self) -> None:
        validate_simulation_toml(_minimal_doc())

    def test_species_charge_to_mass_only_accepted(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": "charge_to_mass = -1.0",
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].charge_to_mass == -1.0

    def test_species_both_rejected(self) -> None:
        both = "charge = -1.0\nmass = 1.0\ncharge_to_mass = -1.0"
        doc = _minimal_doc(**{"charge = -1.0\nmass = 1.0": both})
        with pytest.raises(ValidationError, match="choose one"):
            validate_simulation_toml(doc)

    def test_species_neither_rejected(self) -> None:
        doc = _minimal_doc(**{"charge = -1.0\nmass = 1.0": "density = 1.0"})
        with pytest.raises(ValidationError, match="charge"):
            validate_simulation_toml(doc)

    def test_missing_species_rejected(self) -> None:
        doc = dedent("""
            schema_version = "1.0"
            [schema]
            version = "1.0"
            [model]
            name = "d"
            type = "PIC"
            [run]
            name = "r"
            [time]
            dt = 0.1
            t_start = 0.0
            t_end = 1.0
            n_steps = 10
            [grid]
            dimensions = [4,4,4]
            spacing = [1.0,1.0,1.0]
            lower = [0.0,0.0,0.0]
            upper = [4.0,4.0,4.0]
            [units]
            system = "SI"
            [coordinates]
            geometry = "cartesian"
            frame = "sim"
        """).strip()
        with pytest.raises(ValidationError):
            validate_simulation_toml(doc)


class TestProbe:
    def test_probe_position_only(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [0.0, 0.0, 0.0]
        """)
        s = validate_simulation_toml(doc)
        assert s.probes[0].name == "p1"

    def test_probe_trajectory_only(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "sat1"
            trajectory = "orbits/sat1.csv"
        """)
        s = validate_simulation_toml(doc)
        assert s.probes[0].trajectory == "orbits/sat1.csv"

    def test_probe_both_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [0.0, 0.0, 0.0]
            trajectory = "file.csv"
        """)
        with pytest.raises(ValidationError, match="exactly one"):
            validate_simulation_toml(doc)

    def test_probe_neither_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
        """)
        with pytest.raises(ValidationError, match="exactly one"):
            validate_simulation_toml(doc)


class TestOutputFields:
    def test_precision_overrides_subset(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.fields]
            step_interval = 10
            quantities = ["B", "E"]
            dir = "./fields"
            [output.fields.precision_overrides]
            B = "f64"
        """)
        validate_simulation_toml(doc)

    def test_precision_overrides_not_in_quantities_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.fields]
            step_interval = 10
            quantities = ["B", "E"]
            dir = "./fields"
            [output.fields.precision_overrides]
            rho_c = "f64"
        """)
        with pytest.raises(ValidationError, match="precision_overrides"):
            validate_simulation_toml(doc)


class TestLoaderEntryPoint:
    def test_accepts_path_object(self) -> None:
        validate_simulation_toml(LIVE_TEMPLATE)

    def test_accepts_str_path(self) -> None:
        validate_simulation_toml(str(LIVE_TEMPLATE))

    def test_accepts_dict(self) -> None:
        with LIVE_TEMPLATE.open("rb") as f:
            data = tomllib.load(f)
        validate_simulation_toml(data)

    def test_accepts_bytes(self) -> None:
        validate_simulation_toml(LIVE_TEMPLATE.read_bytes())

    def test_accepts_text_blob(self) -> None:
        validate_simulation_toml(_minimal_doc())

    def test_accepts_generic_pathlike(self) -> None:
        """``os.PathLike[str]`` other than ``Path`` is part of the contract."""

        class _CustomPath:
            def __init__(self, p: Path) -> None:
                self._p = str(p)

            def __fspath__(self) -> str:
                return self._p

        validate_simulation_toml(_CustomPath(LIVE_TEMPLATE))

    def test_invalid_toml_syntax_propagates(self) -> None:
        """``tomllib.TOMLDecodeError`` should surface unwrapped."""
        with pytest.raises(tomllib.TOMLDecodeError):
            validate_simulation_toml("[unclosed_section\nkey = 1")


class TestRoundTrip:
    """Catch one-way coercions silently introduced by future schema edits."""

    def test_dump_reparse_roundtrip(self) -> None:
        s1 = validate_simulation_toml(LIVE_TEMPLATE)
        s2 = SimulationSchema.model_validate(
            s1.model_dump(by_alias=True, exclude_none=True)
        )
        assert s1 == s2


class TestAdaptiveScheme:
    """Adaptive scheme is permissive in v1.0 — vocabulary still settling."""

    def test_adaptive_scheme_minimal(self) -> None:
        doc = _minimal_doc().replace('scheme = "fixed"', 'scheme = "adaptive"')
        s = validate_simulation_toml(doc)
        assert s.time.scheme == "adaptive"
        assert s.time.cfl is None
        assert s.time.dt_min is None
        assert s.time.dt_max is None


class TestDriverBodyReference:
    """``[[drivers]].body`` must resolve to a declared ``[[bodies]].name``."""

    def test_driver_body_matches_declared(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[bodies]]
            name = "earth"
            center = [0.0, 0.0, 0.0]
            radius = 1.0

            [[drivers]]
            name = "magnetogram"
            type = "magnetogram_timeseries"
            coupling = "boundary"
            body = "earth"
        """)
        s = validate_simulation_toml(doc)
        assert s.drivers[0].body == "earth"

    def test_driver_body_unknown_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "magnetogram"
            type = "magnetogram_timeseries"
            coupling = "boundary"
            body = "no_such_body"
        """)
        with pytest.raises(ValidationError, match="no_such_body"):
            validate_simulation_toml(doc)


class TestProbeFrame:
    """``frame=`` is meaningful only on trajectory probes."""

    def test_frame_on_fixed_probe_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [0.0, 0.0, 0.0]
            frame = "GSM"
        """)
        with pytest.raises(ValidationError, match="frame"):
            validate_simulation_toml(doc)

    def test_frame_on_trajectory_probe_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "MMS1"
            trajectory = "orbits/mms1.csv"
            frame = "GSM"
        """)
        s = validate_simulation_toml(doc)
        assert s.probes[0].frame == "GSM"


# ---------------------------------------------------------------------------
# v1.0.x additive batch coverage
# ---------------------------------------------------------------------------


class TestModelTypeAdditions:
    """``vlasov`` and ``gyrokinetic`` admitted as model types."""

    def test_vlasov_accepted(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "vlasov"'})
        s = validate_simulation_toml(doc)
        assert s.model.type == "vlasov"

    def test_gyrokinetic_accepted(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'})
        s = validate_simulation_toml(doc)
        assert s.model.type == "gyrokinetic"

    def test_vlasov_skips_typed_branch_check(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "vlasov"'})
        # No typed [physics.vlasov] sub-table at v1.0; vocabulary lives in
        # extras. A bare [physics] block must validate.
        s = validate_simulation_toml(doc + "\n[physics]\nrelativistic = false\n")
        assert s.physics is not None


class TestPICSolverVocabulary:
    """ED-PIC vocabulary additions on PICSolver."""

    @pytest.mark.parametrize(
        "field_solver",
        ["psatd", "spectral-azimuthal", "lehe", "ck", "ckc", "pstd", "gpstd"],
    )
    def test_field_solver_literals(self, field_solver: str) -> None:
        doc = _minimal_doc() + dedent(f"""
            [physics]
            [physics.pic.solver]
            scheme = "explicit"
            field_solver = "{field_solver}"
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.field_solver == field_solver

    @pytest.mark.parametrize("pusher", ["llrk4", "free-streaming"])
    def test_pusher_literals(self, pusher: str) -> None:
        doc = _minimal_doc() + dedent(f"""
            [physics]
            [physics.pic.solver]
            scheme = "explicit"
            pusher = "{pusher}"
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.pusher == pusher

    def test_smoothing_knobs(self) -> None:
        doc = _minimal_doc() + dedent("""
            [physics]
            [physics.pic.solver]
            scheme = "explicit"
            current_smoothing = 4
            charge_smoothing = 2
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.current_smoothing == 4
        assert s.physics.pic.solver.charge_smoothing == 2

    def test_charge_correction_and_deposition(self) -> None:
        doc = _minimal_doc() + dedent("""
            [physics]
            [physics.pic.solver]
            scheme = "explicit"
            charge_correction = "marder"
            current_deposition = "esirkepov"
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.charge_correction == "marder"
        assert s.physics.pic.solver.current_deposition == "esirkepov"


class TestSpeciesAdditions:
    """``shape`` and ``tracer`` flags on ``[[species]]``."""

    @pytest.mark.parametrize("shape", ["ngp", "cic", "tsc", "pqs"])
    def test_particle_shape_literals(self, shape: str) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    f'charge = -1.0\nmass = 1.0\nshape = "{shape}"'
                )
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].shape == shape

    def test_tracer_default_false(self) -> None:
        s = validate_simulation_toml(_minimal_doc())
        assert s.species[0].tracer is False

    def test_tracer_true(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    "charge = -1.0\nmass = 1.0\ntracer = true"
                )
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].tracer is True


class TestTimeSchemeAdditions:
    """RK-family + IMEX schemes and ``splitting`` on ``[time]``."""

    @pytest.mark.parametrize(
        "scheme",
        ["rk2", "rk3", "rk4", "vl2", "ssprk2", "ssprk3", "imex-rk2", "imex-rk3"],
    )
    def test_rk_imex_schemes(self, scheme: str) -> None:
        doc = _minimal_doc(**{'scheme = "fixed"': f'scheme = "{scheme}"'})
        s = validate_simulation_toml(doc)
        assert s.time.scheme == scheme

    @pytest.mark.parametrize("splitting", ["strang", "lie", "godunov"])
    def test_splitting_literal(self, splitting: str) -> None:
        doc = _minimal_doc().replace(
            "n_steps = 10\n", f'n_steps = 10\nsplitting = "{splitting}"\n'
        )
        s = validate_simulation_toml(doc)
        assert s.time.splitting == splitting

    def test_rk_scheme_requires_dt(self) -> None:
        doc = _minimal_doc(**{'scheme = "fixed"': 'scheme = "rk3"'}).replace(
            "dt = 0.1", "dt = 0.0"
        )
        with pytest.raises(ValidationError, match="rk3"):
            validate_simulation_toml(doc)


class TestRunEnsemble:
    """``random_seed`` and ``ensemble`` on ``[run]``."""

    def test_random_seed(self) -> None:
        doc = _minimal_doc().replace('name = "r0"\n', 'name = "r0"\nrandom_seed = 42\n')
        s = validate_simulation_toml(doc)
        assert s.run.random_seed == 42

    def test_ensemble(self) -> None:
        doc = _minimal_doc() + dedent("""
            [run.ensemble]
            member_id = 3
            total = 16
        """)
        s = validate_simulation_toml(doc)
        assert s.run.ensemble is not None
        assert s.run.ensemble.member_id == 3
        assert s.run.ensemble.total == 16

    def test_ensemble_member_out_of_range_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [run.ensemble]
            member_id = 17
            total = 16
        """)
        with pytest.raises(ValidationError, match="member_id"):
            validate_simulation_toml(doc)


class TestGridAdditions:
    """``ghost_cells`` and ``[grid.amr]`` discriminator + subcycling."""

    def test_ghost_cells_axis_match(self) -> None:
        doc = _minimal_doc().replace(
            "upper = [4.0, 4.0, 4.0]",
            "upper = [4.0, 4.0, 4.0]\nghost_cells = [2, 2, 2]",
        )
        s = validate_simulation_toml(doc)
        assert s.grid.ghost_cells == [2, 2, 2]

    def test_ghost_cells_axis_mismatch_rejected(self) -> None:
        doc = _minimal_doc().replace(
            "upper = [4.0, 4.0, 4.0]",
            "upper = [4.0, 4.0, 4.0]\nghost_cells = [2, 2]",
        )
        with pytest.raises(ValidationError, match="ghost_cells"):
            validate_simulation_toml(doc)

    @pytest.mark.parametrize("kind", ["block", "patch", "octree"])
    def test_amr_kind_literals(self, kind: str) -> None:
        doc = _minimal_doc() + dedent(f"""
            [grid.amr]
            max_level = 4
            amr_kind = "{kind}"
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.amr is not None
        assert s.grid.amr.amr_kind == kind

    def test_amr_kind_default_block(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.amr]
            max_level = 4
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.amr is not None
        assert s.grid.amr.amr_kind == "block"
        assert s.grid.amr.level_subcycling is False

    def test_amr_level_subcycling(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.amr]
            max_level = 4
            level_subcycling = true
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.amr is not None
        assert s.grid.amr.level_subcycling is True


class TestClosureAdditions:
    """Anisotropic closure literals: cgl, 10moment, 14moment."""

    @pytest.mark.parametrize("closure", ["cgl", "10moment", "14moment"])
    def test_anisotropic_closure_literals(self, closure: str) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    f'charge = -1.0\nmass = 1.0\nclosure = "{closure}"'
                )
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].closure == closure


class TestRestartGranularity:
    """``restore`` and ``mode`` additive fields on ``[restart]``."""

    def test_restore_partial(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = "./checkpoints/last.h5"
            restore = ["fields", "auxiliary"]
            mode = "hot"
        """)
        s = validate_simulation_toml(doc)
        assert s.restart is not None
        assert s.restart.restore == ["fields", "auxiliary"]
        assert s.restart.mode == "hot"

    def test_restore_duplicates_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = "./chk.h5"
            restore = ["fields", "fields"]
        """)
        with pytest.raises(ValidationError, match="distinct"):
            validate_simulation_toml(doc)

    def test_restore_empty_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = "./chk.h5"
            restore = []
        """)
        with pytest.raises(ValidationError, match="non-empty"):
            validate_simulation_toml(doc)


class TestThetaModeGeometry:
    """``thetaMode`` geometry + required ``[coordinates.modes]`` block."""

    def test_thetamode_with_modes(self) -> None:
        doc = _minimal_doc(
            **{'geometry = "cartesian"': 'geometry = "thetaMode"'}
        ) + dedent("""
            [coordinates.modes]
            n_modes = 3
            mode_indices = [0, 1, 2]
        """)
        s = validate_simulation_toml(doc)
        assert s.coordinates.geometry == "thetaMode"
        assert s.coordinates.modes is not None
        assert s.coordinates.modes.n_modes == 3
        assert s.coordinates.modes.mode_indices == [0, 1, 2]

    def test_thetamode_without_modes_rejected(self) -> None:
        doc = _minimal_doc(**{'geometry = "cartesian"': 'geometry = "thetaMode"'})
        with pytest.raises(ValidationError, match="modes"):
            validate_simulation_toml(doc)

    def test_modes_without_thetamode_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [coordinates.modes]
            n_modes = 3
        """)
        with pytest.raises(ValidationError, match="thetaMode"):
            validate_simulation_toml(doc)

    def test_mode_indices_count_mismatch_rejected(self) -> None:
        doc = _minimal_doc(
            **{'geometry = "cartesian"': 'geometry = "thetaMode"'}
        ) + dedent("""
            [coordinates.modes]
            n_modes = 3
            mode_indices = [0, 1]
        """)
        with pytest.raises(ValidationError, match="mode_indices"):
            validate_simulation_toml(doc)


class TestOutputMultiFileLayout:
    """Per-rank / multi-file output layout fields on ``_OutputBase``."""

    def test_multi_file_layout(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.fields]
            step_interval = 50
            quantities = ["B"]
            dir = "./fields"
            file_pattern = "step_{step:06d}/rank_{rank:05d}.h5"
            files_per_step = 64
            partition = "by_rank"
        """)
        s = validate_simulation_toml(doc)
        assert s.output is not None
        assert s.output.fields is not None
        assert s.output.fields.partition == "by_rank"
        assert s.output.fields.files_per_step == 64
        assert s.output.fields.file_pattern == "step_{step:06d}/rank_{rank:05d}.h5"

    def test_partition_literal_rejection(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.fields]
            step_interval = 50
            quantities = ["B"]
            dir = "./fields"
            partition = "ad-hoc"
        """)
        with pytest.raises(ValidationError, match="partition"):
            validate_simulation_toml(doc)
