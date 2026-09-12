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
    return "\n".join(extracted) + "\n"


class TestLiveTemplate:
    """The live template file must validate as-is (Scenario A active)."""

    def test_scenario_a_validates(self) -> None:
        result = validate_simulation_toml(LIVE_TEMPLATE)
        assert isinstance(result, SimulationSchema)

    def test_scenario_a_identity(self) -> None:
        s = validate_simulation_toml(LIVE_TEMPLATE)
        assert s.model.name == "iPIC3D"
        assert s.model.type == "PIC"
        assert s.schema_.version == "1.0"

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

    def test_rejects_non_v1_schema(self) -> None:
        doc = dedent("""
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
        doc = _minimal_doc() + "\n[physics]\n[physics.mhd]\ngamma_eos = 1.5\n"
        with pytest.raises(ValidationError, match=r"model\.type"):
            validate_simulation_toml(doc)

    def test_model_type_mhd_permits_only_mhd_block(self) -> None:
        doc = (
            _minimal_doc(**{'type = "PIC"': 'type = "MHD"'})
            + "\n[physics]\n[physics.mhd]\ngamma_eos = 1.5\n"
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

    def test_adaptive_scheme_omits_dt(self) -> None:
        # Adaptive runs derive the first step from CFL bookkeeping; ``dt``
        # may be omitted entirely.
        doc = _minimal_doc().replace(
            'scheme = "fixed"\ndt = 0.1', 'scheme = "adaptive"'
        )
        s = validate_simulation_toml(doc)
        assert s.time.scheme == "adaptive"
        assert s.time.dt is None

    def test_fixed_scheme_requires_dt(self) -> None:
        doc = _minimal_doc().replace('scheme = "fixed"\ndt = 0.1', 'scheme = "fixed"')
        with pytest.raises(ValidationError, match="requires dt > 0"):
            validate_simulation_toml(doc)


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


class TestDriverModelSubtable:
    """``[drivers.model]`` — optional coupled-system identity sub-table."""

    def test_driver_model_subtable_round_trips(self) -> None:
        """Minimal [drivers.model] with name + type validates."""
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "ionosphere"
            type = "model_coupling"
            coupling = "boundary"
            direction = "two_way"

            [drivers.model]
            name = "RIM"
            type = "ionosphere_potential_solver"
        """)
        s = validate_simulation_toml(doc)
        assert s.drivers[0].model is not None
        assert s.drivers[0].model.name == "RIM"
        assert s.drivers[0].model.type == "ionosphere_potential_solver"

    def test_driver_model_full_subtable_round_trips(self) -> None:
        """All four DriverModel fields round-trip; url is opaque text."""
        url = "../rim_run/simulation.toml"
        doc = _minimal_doc() + dedent(f"""
            [[drivers]]
            name = "ionosphere"
            type = "model_coupling"
            coupling = "boundary"
            direction = "two_way"

            [drivers.model]
            name = "RIM"
            type = "ionosphere_potential_solver"
            url = "{url}"
            description = "Ridley Ionosphere Model"
        """)
        s = validate_simulation_toml(doc)
        model = s.drivers[0].model
        assert model is not None
        assert model.url == url  # verbatim, no resolution
        assert model.description == "Ridley Ionosphere Model"

    def test_driver_model_type_is_open(self) -> None:
        """DriverModel.type accepts any string — not Literal-constrained."""
        for type_value in (
            "ionosphere_potential_solver",
            "fluid_atmosphere",
            "fusion_transport",
            "magnetic_field_extrapolation",
            "completely_made_up_research_code",
        ):
            doc = _minimal_doc() + dedent(f"""
                [[drivers]]
                name = "d"
                type = "model_coupling"
                coupling = "boundary"
                [drivers.model]
                name = "Peer"
                type = "{type_value}"
            """)
            s = validate_simulation_toml(doc)
            assert s.drivers[0].model is not None
            assert s.drivers[0].model.type == type_value

    def test_driver_model_extras_accepted(self) -> None:
        """_ExtensibleBase lets coupling-specific extras pass through."""
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "d"
            type = "model_coupling"
            coupling = "boundary"

            [drivers.model]
            name = "RIM"
            type = "ionosphere_potential_solver"
            version = "3.1"
            doi = "10.1029/example"
            git_sha = "abc1234"
        """)
        s = validate_simulation_toml(doc)
        dumped = s.drivers[0].model.model_dump()  # type: ignore[union-attr]
        assert dumped["version"] == "3.1"
        assert dumped["doi"] == "10.1029/example"
        assert dumped["git_sha"] == "abc1234"

    def test_driver_with_source_and_model_coexist(self) -> None:
        """source (data file) and [drivers.model] (attribution) co-exist."""
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "checkpointed_mhd"
            type = "mhd_field_coupling"
            coupling = "volume"
            direction = "one_way"
            source = "drivers/batsrus_checkpoint.h5"

            [drivers.model]
            name = "BATSRUS"
            type = "MHD"
            url = "https://clasp.engin.umich.edu/batsrus/"
            description = "Global MHD checkpoint loaded as PIC background"
        """)
        s = validate_simulation_toml(doc)
        driver = s.drivers[0]
        assert driver.source == "drivers/batsrus_checkpoint.h5"
        assert driver.model is not None
        assert driver.model.name == "BATSRUS"

    def test_driver_model_name_required(self) -> None:
        """[drivers.model] without ``name`` raises ValidationError."""
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "d"
            type = "model_coupling"
            coupling = "boundary"
            [drivers.model]
            type = "ionosphere_potential_solver"
        """)
        with pytest.raises(ValidationError, match="name"):
            validate_simulation_toml(doc)

    def test_driver_model_type_required(self) -> None:
        """[drivers.model] without ``type`` raises ValidationError."""
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "d"
            type = "model_coupling"
            coupling = "boundary"
            [drivers.model]
            name = "RIM"
        """)
        with pytest.raises(ValidationError, match="type"):
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


class TestProbeResolveFields:
    """``Probe.resolve_fields`` narrows the omitted-default to primitives."""

    def test_omitted_fields_resolves_to_all_primitives(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [0.0, 0.0, 0.0]
        """)
        s = validate_simulation_toml(doc)
        primitives = ["B_1", "B_2", "B_3", "rho_c", "n_s0"]
        assert s.probes[0].resolve_fields(primitives) == primitives

    def test_explicit_fields_pass_through(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [0.0, 0.0, 0.0]
            fields = ["B_1", "beta"]
        """)
        s = validate_simulation_toml(doc)
        # Explicit list passes through verbatim — `beta` is derived, not
        # a primitive, but the schema doesn't reject it; the sampler
        # dispatches via `compute()`.
        assert s.probes[0].resolve_fields(["B_1", "B_2"]) == ["B_1", "beta"]


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


class TestOpenVocabularies:
    """Algorithm/method/closure fields accept research-extensible strings.

    The canonical v1.0 vocabularies (see ``_models.py``) are guidance, not
    enforcement: pypic records these values as provenance and does not
    dispatch on them, so unknown strings — typical for in-progress
    research methods — must validate cleanly.
    """

    def test_custom_time_scheme(self) -> None:
        doc = _minimal_doc(**{'scheme = "fixed"': 'scheme = "amortized-imex-rk4"'})
        s = validate_simulation_toml(doc)
        assert s.time.scheme == "amortized-imex-rk4"

    def test_custom_pic_pusher_and_field_solver(self) -> None:
        doc = _minimal_doc() + dedent("""
            [physics]
            [physics.pic.solver]
            scheme = "explicit"
            pusher = "novel-symplectic-2026"
            field_solver = "fourier-hermite-mixed"
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.pic is not None
        assert s.physics.pic.solver is not None
        assert s.physics.pic.solver.pusher == "novel-symplectic-2026"
        assert s.physics.pic.solver.field_solver == "fourier-hermite-mixed"

    def test_custom_mhd_solver_scheme(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "MHD"'}) + dedent("""
            [physics]
            [physics.mhd.solver]
            scheme = "in-house-positivity-preserving"
            limiter = "custom-tvd-3"
        """)
        s = validate_simulation_toml(doc)
        assert s.physics is not None
        assert s.physics.mhd is not None
        assert s.physics.mhd.solver is not None
        assert s.physics.mhd.solver.scheme == "in-house-positivity-preserving"
        assert s.physics.mhd.solver.limiter == "custom-tvd-3"

    def test_custom_closure(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    'charge = -1.0\nmass = 1.0\nclosure = "kinetic-bgk-2026"'
                )
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].closure == "kinetic-bgk-2026"


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


class TestGridStagger:
    """``[grid.stagger]`` sub-table — convention/fields/position tiers."""

    def test_stagger_default_convention(self) -> None:
        s = validate_simulation_toml(_minimal_doc())
        assert s.grid.stagger.convention == "cell"
        assert s.grid.stagger.fields is None
        assert s.grid.stagger.position is None

    def test_stagger_convention_explicit(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger]
            convention = "node"
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.stagger.convention == "node"

    def test_stagger_fields_per_group(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger.fields]
            B = "face"
            E = "edge"
            J = "edge"
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.stagger.fields == {"B": "face", "E": "edge", "J": "edge"}

    def test_stagger_position_offsets(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger.position]
            B_1 = [0.5, 0.0, 0.0]
            E_1 = [0.0, 0.5, 0.5]
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.stagger.position is not None
        assert s.grid.stagger.position["B_1"] == [0.5, 0.0, 0.0]

    def test_stagger_position_offset_out_of_range_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger.position]
            B_1 = [1.0, 0.0, 0.0]
        """)
        with pytest.raises(ValidationError, match=r"\[0.0, 1.0\)"):
            validate_simulation_toml(doc)

    def test_stagger_position_axis_count_mismatch_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger.position]
            B_1 = [0.5, 0.0]
        """)
        with pytest.raises(ValidationError, match="dimensions has 3"):
            validate_simulation_toml(doc)

    def test_stagger_field_invalid_location_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stagger.fields]
            B = "centroid"
        """)
        with pytest.raises(ValidationError):
            validate_simulation_toml(doc)

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


class TestPhaseSpaceStorage:
    """``[phase_space.storage]`` — sparse-block velocity-grid storage.

    ``block_size`` is validated against the velocity sub-axes of
    ``phase_space.dimensions`` by the root validator (it needs to know
    how many spatial axes ``[grid]`` claims first).
    """

    @staticmethod
    def _vlasov_doc(extra: str) -> str:
        return (
            _minimal_doc(**{'type = "PIC"': 'type = "vlasov"'})
            + dedent(
                """
            [phase_space]
            dimensions = [4, 4, 4, 50, 50, 50]
            axis_labels = ["x", "y", "z", "vx", "vy", "vz"]
            extents = [
                [0.0, 4.0], [0.0, 4.0], [0.0, 4.0],
                [-2e6, 2e6], [-2e6, 2e6], [-2e6, 2e6],
            ]
            """
            )
            + dedent(extra)
        )

    def test_minimal_phase_space(self) -> None:
        s = validate_simulation_toml(self._vlasov_doc(""))
        assert s.phase_space is not None
        assert s.phase_space.dimensions == [4, 4, 4, 50, 50, 50]
        assert s.phase_space.coordinate_system == "cartesian"
        assert s.phase_space.storage is None

    def test_storage_block_size(self) -> None:
        doc = self._vlasov_doc("""
            [phase_space.storage]
            block_size = [10, 10, 10]
            sparsity_threshold = 1.0e-15
        """)
        s = validate_simulation_toml(doc)
        assert s.phase_space is not None
        assert s.phase_space.storage is not None
        assert s.phase_space.storage.block_size == [10, 10, 10]
        assert s.phase_space.storage.sparsity_threshold == 1.0e-15

    def test_block_size_axis_mismatch_rejected(self) -> None:
        # block_size has 2 entries, but the velocity sub-grid has 3 axes
        # (phase_space.dimensions[3:]).
        doc = self._vlasov_doc("""
            [phase_space.storage]
            block_size = [10, 10]
        """)
        with pytest.raises(ValidationError, match=r"phase_space\.storage\.block_size"):
            validate_simulation_toml(doc)

    def test_block_size_must_divide_velocity_dimensions(self) -> None:
        doc = self._vlasov_doc("""
            [phase_space.storage]
            block_size = [7, 10, 10]
        """)
        with pytest.raises(ValidationError, match="block_size"):
            validate_simulation_toml(doc)


class TestRestartFromList:
    """``Restart.from`` accepts either a string or a list of per-rank files."""

    def test_from_string_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = "./chk.h5"
        """)
        s = validate_simulation_toml(doc)
        assert s.restart is not None
        assert s.restart.from_ == "./chk.h5"

    def test_from_list_accepted(self) -> None:
        # Escape-hatch shape: explicit per-rank file list for runs whose
        # filenames don't follow the source code's convention.
        doc = _minimal_doc() + dedent("""
            [restart]
            from = ["chk_00000.h5", "chk_00001.h5", "chk_00002.h5"]
        """)
        s = validate_simulation_toml(doc)
        assert s.restart is not None
        assert s.restart.from_ == [
            "chk_00000.h5",
            "chk_00001.h5",
            "chk_00002.h5",
        ]

    def test_from_list_empty_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = []
        """)
        with pytest.raises(ValidationError, match="non-empty"):
            validate_simulation_toml(doc)

    def test_from_list_duplicates_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [restart]
            from = ["a.h5", "a.h5"]
        """)
        with pytest.raises(ValidationError, match="distinct"):
            validate_simulation_toml(doc)


class TestAnisotropicGammaEOS:
    """``Species.gamma_eos_par`` / ``gamma_eos_perp`` for 10-moment hybrids."""

    def test_par_perp_pair_accepted(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    "charge = -1.0\nmass = 1.0\n"
                    "gamma_eos_par = 3.0\ngamma_eos_perp = 2.0"
                )
            }
        )
        s = validate_simulation_toml(doc)
        assert s.species[0].gamma_eos_par == 3.0
        assert s.species[0].gamma_eos_perp == 2.0
        assert s.species[0].gamma_eos is None

    def test_par_alone_rejected(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    "charge = -1.0\nmass = 1.0\ngamma_eos_par = 3.0"
                )
            }
        )
        with pytest.raises(ValidationError, match="gamma_eos_par"):
            validate_simulation_toml(doc)

    def test_scalar_plus_anisotropic_rejected(self) -> None:
        doc = _minimal_doc(
            **{
                "charge = -1.0\nmass = 1.0": (
                    "charge = -1.0\nmass = 1.0\n"
                    "gamma_eos = 1.6667\n"
                    "gamma_eos_par = 3.0\n"
                    "gamma_eos_perp = 2.0"
                )
            }
        )
        with pytest.raises(ValidationError, match="anisotropic"):
            validate_simulation_toml(doc)


class TestCollisions:
    """``[[collisions]]`` — per-pair collision declarations."""

    def test_minimal_collision_pair(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[species]]
            name = "ions"
            charge = 1.0
            mass = 1836.0

            [[collisions]]
            species_pair = ["electrons", "ions"]
            model = "coulomb"
            coulomb_log = 10.0
        """)
        s = validate_simulation_toml(doc)
        assert len(s.collisions) == 1
        assert s.collisions[0].species_pair == ["electrons", "ions"]
        assert s.collisions[0].model == "coulomb"
        assert s.collisions[0].coulomb_log == 10.0

    @pytest.mark.parametrize("model", ["coulomb", "bgk", "monte-carlo"])
    def test_collision_model_literals(self, model: str) -> None:
        doc = _minimal_doc() + dedent(f"""
            [[collisions]]
            species_pair = ["electrons", "electrons"]
            model = "{model}"
        """)
        s = validate_simulation_toml(doc)
        assert s.collisions[0].model == model

    def test_self_collisions_permitted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[collisions]]
            species_pair = ["electrons", "electrons"]
            model = "bgk"
        """)
        s = validate_simulation_toml(doc)
        assert s.collisions[0].species_pair == ["electrons", "electrons"]

    def test_unknown_species_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[collisions]]
            species_pair = ["electrons", "ghost"]
            model = "coulomb"
        """)
        with pytest.raises(ValidationError, match="ghost"):
            validate_simulation_toml(doc)

    def test_pair_size_strict(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[collisions]]
            species_pair = ["electrons"]
            model = "coulomb"
        """)
        with pytest.raises(ValidationError, match="species_pair"):
            validate_simulation_toml(doc)


class TestPhaseSpace:
    """``[phase_space]`` — kinetic phase-space dimensions for >3D codes."""

    def test_5d_gyrokinetic(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'}) + dedent("""
            [phase_space]
            dimensions = [4, 4, 4, 16, 8]
            axis_labels = ["x", "y", "z", "vpar", "mu"]
            coordinate_system = "guiding-center"
        """)
        s = validate_simulation_toml(doc)
        assert s.phase_space is not None
        assert s.phase_space.dimensions == [4, 4, 4, 16, 8]
        assert s.phase_space.coordinate_system == "guiding-center"

    def test_6d_full_vlasov(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "vlasov"'}) + dedent("""
            [phase_space]
            dimensions = [4, 4, 4, 32, 32, 32]
            extents = [
                [0, 4], [0, 4], [0, 4],
                [-2e6, 2e6], [-2e6, 2e6], [-2e6, 2e6],
            ]
        """)
        s = validate_simulation_toml(doc)
        assert s.phase_space is not None
        assert len(s.phase_space.dimensions) == 6

    def test_spatial_mismatch_rejected(self) -> None:
        # phase_space.dimensions[:3] must equal grid.dimensions
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'}) + dedent("""
            [phase_space]
            dimensions = [8, 8, 8, 16, 8]
        """)
        with pytest.raises(ValidationError, match=r"phase_space\.dimensions"):
            validate_simulation_toml(doc)

    def test_must_extend_grid(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'}) + dedent("""
            [phase_space]
            dimensions = [4, 4]
        """)
        with pytest.raises(ValidationError, match="extend"):
            validate_simulation_toml(doc)

    def test_no_extension_rejected(self) -> None:
        # dimensions match the spatial grid exactly — no velocity / extra-D
        # axes declared. Must be rejected: phase_space exists to extend.
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'}) + dedent("""
            [phase_space]
            dimensions = [4, 4, 4]
        """)
        with pytest.raises(ValidationError, match="strictly extend"):
            validate_simulation_toml(doc)

    def test_extents_axis_count_must_match(self) -> None:
        doc = _minimal_doc(**{'type = "PIC"': 'type = "gyrokinetic"'}) + dedent("""
            [phase_space]
            dimensions = [4, 4, 4, 16, 8]
            extents = [[0, 4], [0, 4], [0, 4]]
        """)
        with pytest.raises(ValidationError, match=r"phase_space\.extents"):
            validate_simulation_toml(doc)


class TestGridStretched:
    """``[grid.stretched]`` — non-uniform per-axis cell widths (ARMS)."""

    def test_single_stretched_axis(self) -> None:
        # Cell widths: 4 entries summing to 4.0 to match upper-lower.
        doc = _minimal_doc() + dedent("""
            [grid.stretched.axis_widths]
            0 = [0.4, 0.8, 1.2, 1.6]
        """)
        s = validate_simulation_toml(doc)
        assert s.grid.stretched is not None
        assert "0" in s.grid.stretched.axis_widths
        assert s.grid.stretched.axis_widths["0"] == [0.4, 0.8, 1.2, 1.6]

    def test_widths_count_mismatch_rejected(self) -> None:
        # dimensions[0] = 4 but only 3 widths.
        doc = _minimal_doc() + dedent("""
            [grid.stretched.axis_widths]
            0 = [0.5, 1.0, 2.5]
        """)
        with pytest.raises(ValidationError, match="dimensions"):
            validate_simulation_toml(doc)

    def test_widths_sum_mismatch_rejected(self) -> None:
        # extent = upper - lower = 4.0, sum = 5.0.
        doc = _minimal_doc() + dedent("""
            [grid.stretched.axis_widths]
            0 = [0.5, 1.0, 1.5, 2.0]
        """)
        with pytest.raises(ValidationError, match="sums"):
            validate_simulation_toml(doc)

    def test_axis_index_out_of_range_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stretched.axis_widths]
            5 = [1.0, 1.0, 1.0, 1.0]
        """)
        with pytest.raises(ValidationError, match="axis_widths"):
            validate_simulation_toml(doc)

    def test_empty_axis_widths_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [grid.stretched]
            axis_widths = {}
        """)
        with pytest.raises(ValidationError, match="at least one axis"):
            validate_simulation_toml(doc)


class TestOutputStreams:
    """``[[output.streams]]`` — multi-cadence / ROI output."""

    def test_basic_stream(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 10
            dir = "./moments"
            name = "moments"
            quantities = ["B", "rho_c"]
        """)
        s = validate_simulation_toml(doc)
        assert s.output is not None
        assert len(s.output.streams) == 1
        assert s.output.streams[0].name == "moments"
        assert s.output.streams[0].region is None

    def test_box_region(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 5
            dir = "./roi"
            name = "current_sheet"
            quantities = ["B", "E", "J"]
            [output.streams.region]
            kind = "box"
            lower = [1.0, 1.0, 1.0]
            upper = [3.0, 3.0, 3.0]
        """)
        s = validate_simulation_toml(doc)
        stream = s.output.streams[0]
        assert stream.region is not None
        # Pydantic discriminator yields the concrete BoxRegion.
        assert stream.region.kind == "box"

    def test_plane_region(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 1
            dir = "./midplane"
            name = "midplane"
            quantities = ["B"]
            [output.streams.region]
            kind = "plane"
            axis = 2
            value = 2.0
        """)
        s = validate_simulation_toml(doc)
        stream = s.output.streams[0]
        assert stream.region is not None
        assert stream.region.kind == "plane"

    def test_duplicate_stream_names_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 10
            dir = "./a"
            name = "fast"
            quantities = ["B"]

            [[output.streams]]
            step_interval = 100
            dir = "./b"
            name = "fast"
            quantities = ["E"]
        """)
        with pytest.raises(ValidationError, match="distinct names"):
            validate_simulation_toml(doc)

    def test_box_region_axis_mismatch_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 5
            dir = "./roi"
            name = "bad"
            quantities = ["B"]
            [output.streams.region]
            kind = "box"
            lower = [1.0, 1.0]
            upper = [3.0, 3.0]
        """)
        with pytest.raises(ValidationError, match="axes"):
            validate_simulation_toml(doc)

    def test_plane_region_axis_out_of_range_rejected(self) -> None:
        # PlaneRegion.axis bound is 0..2 at the model level, but
        # cross-check against grid.dimensions runs at root.
        doc = _minimal_doc(**{"dimensions = [4, 4, 4]": "dimensions = [4, 4]"})
        doc = doc.replace("spacing = [1.0, 1.0, 1.0]", "spacing = [1.0, 1.0]")
        doc = doc.replace("lower = [0.0, 0.0, 0.0]", "lower = [0.0, 0.0]")
        doc = doc.replace("upper = [4.0, 4.0, 4.0]", "upper = [4.0, 4.0]")
        doc += dedent("""
            [[output.streams]]
            step_interval = 5
            dir = "./roi"
            name = "bad"
            quantities = ["B"]
            [output.streams.region]
            kind = "plane"
            axis = 2
            value = 1.0
        """)
        with pytest.raises(ValidationError, match="plane"):
            validate_simulation_toml(doc)

    def test_precision_overrides_subset(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[output.streams]]
            step_interval = 10
            dir = "./moments"
            name = "moments"
            quantities = ["B"]
            [output.streams.precision_overrides]
            E = "f64"
        """)
        with pytest.raises(ValidationError, match="precision_overrides"):
            validate_simulation_toml(doc)


class TestPerFieldBoundaryConditions:
    """``BoundaryConditions.field_overrides`` and ``drivers_*`` foreign keys."""

    def test_default_unchanged(self) -> None:
        # Existing BC docs without overrides keep working.
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["periodic", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
        """)
        s = validate_simulation_toml(doc)
        assert s.boundary_conditions is not None
        assert s.boundary_conditions.field_overrides is None

    def test_field_overrides(self) -> None:
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["periodic", "periodic", "open"]
            upper = ["periodic", "periodic", "open"]
            [boundary_conditions.field_overrides.E]
            lower = ["periodic", "periodic", "pml"]
            upper = ["periodic", "periodic", "pml"]
            [boundary_conditions.field_overrides.particles]
            lower = ["periodic", "periodic", "absorbing"]
            upper = ["periodic", "periodic", "absorbing"]
        """)
        s = validate_simulation_toml(doc)
        bcs = s.boundary_conditions
        assert bcs is not None
        assert bcs.field_overrides is not None
        assert bcs.field_overrides["E"].lower[2] == "pml"
        assert bcs.field_overrides["particles"].lower[2] == "absorbing"

    def test_field_override_axis_mismatch_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["periodic", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
            [boundary_conditions.field_overrides.E]
            lower = ["periodic", "periodic"]
            upper = ["periodic", "periodic"]
        """)
        with pytest.raises(ValidationError, match="field_overrides"):
            validate_simulation_toml(doc)

    def test_field_override_unknown_group_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["periodic", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
            [boundary_conditions.field_overrides.bogus]
            lower = ["periodic", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
        """)
        with pytest.raises(ValidationError, match="bogus"):
            validate_simulation_toml(doc)

    def test_driver_foreign_keys(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "solar_wind_inflow"
            type = "solar_wind_timeseries"
            coupling = "boundary"

            [boundary_conditions]
            lower = ["driven", "periodic", "periodic"]
            upper = ["open", "periodic", "periodic"]
            drivers_lower = { 0 = "solar_wind_inflow" }
        """)
        s = validate_simulation_toml(doc)
        bcs = s.boundary_conditions
        assert bcs is not None
        assert bcs.drivers_lower == {"0": "solar_wind_inflow"}
        assert bcs.drivers_upper is None

    def test_driver_foreign_key_unknown_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [boundary_conditions]
            lower = ["driven", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
            drivers_lower = { 0 = "does_not_exist" }
        """)
        with pytest.raises(ValidationError, match="does_not_exist"):
            validate_simulation_toml(doc)

    def test_drivers_axis_index_out_of_range_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[drivers]]
            name = "x"
            type = "x"
            coupling = "boundary"

            [boundary_conditions]
            lower = ["periodic", "periodic", "periodic"]
            upper = ["periodic", "periodic", "periodic"]
            drivers_lower = { 5 = "x" }
        """)
        with pytest.raises(ValidationError, match="drivers_lower"):
            validate_simulation_toml(doc)


class TestBodyNameUniqueness:
    """``[[bodies]].name`` entries must be distinct (foreign-key target)."""

    def test_distinct_body_names_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[bodies]]
            name = "earth"
            center = [0.0, 0.0, 0.0]
            radius = 1.0

            [[bodies]]
            name = "moon"
            center = [10.0, 0.0, 0.0]
            radius = 0.27
        """)
        s = validate_simulation_toml(doc)
        assert {b.name for b in s.bodies} == {"earth", "moon"}

    def test_duplicate_body_names_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[bodies]]
            name = "earth"
            center = [0.0, 0.0, 0.0]
            radius = 1.0

            [[bodies]]
            name = "earth"
            center = [10.0, 0.0, 0.0]
            radius = 1.0
        """)
        with pytest.raises(ValidationError, match="distinct names"):
            validate_simulation_toml(doc)


class TestOutputParticlesSpeciesReference:
    """``[output.particles].species`` must resolve to declared ``[[species]]``."""

    def test_known_species_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.particles]
            step_interval = 100
            species       = ["electrons"]
            dir           = "./p"
            format        = "hdf5"
            precision     = "f32"
        """)
        s = validate_simulation_toml(doc)
        assert s.output is not None
        assert s.output.particles is not None
        assert s.output.particles.species == ["electrons"]

    def test_unknown_species_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [output.particles]
            step_interval = 100
            species       = ["electron"]
            dir           = "./p"
            format        = "hdf5"
            precision     = "f32"
        """)
        with pytest.raises(ValidationError, match=r"output\.particles\.species"):
            validate_simulation_toml(doc)

    def test_empty_species_list_accepted(self) -> None:
        # An empty list opts out of all species — no name to resolve.
        doc = _minimal_doc() + dedent("""
            [output.particles]
            step_interval = 100
            species       = []
            dir           = "./p"
            format        = "hdf5"
            precision     = "f32"
        """)
        s = validate_simulation_toml(doc)
        assert s.output is not None
        assert s.output.particles is not None
        assert s.output.particles.species == []


class TestReferenceSpeciesBuiltins:
    """``[units].reference_species`` falls back to PIC builtins."""

    @pytest.mark.parametrize("builtin", ["electrons", "ions", "protons"])
    def test_builtin_accepted_without_declaration(self, builtin: str) -> None:
        # The reference_species names "electrons", "ions", "protons" must
        # validate even when no [[species]] entry of that name exists —
        # the Approach-A fallback for legacy decks. The minimal doc declares
        # only "electrons" as a [[species]]; "ions"/"protons" are pure
        # builtin fallback resolutions.
        doc = _minimal_doc().replace(
            'system = "SI"',
            f'system = "PIC"\n'
            f'reference_species = "{builtin}"\n'
            f"reference_density = 1.0e6",
        )
        s = validate_simulation_toml(doc)
        assert isinstance(s.units, UnitsPIC)
        assert s.units.reference_species == builtin

    def test_unknown_reference_species_rejected(self) -> None:
        doc = _minimal_doc().replace(
            'system = "SI"',
            'system = "PIC"\nreference_species = "muons"\nreference_density = 1.0e6',
        )
        with pytest.raises(ValidationError, match="reference_species"):
            validate_simulation_toml(doc)


class TestRotationMatrix:
    """``coordinates.transforms.rotation`` must be a 3x3 signed permutation."""

    def test_signed_permutation_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.flipped]
            rotation = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        """)
        s = validate_simulation_toml(doc)
        assert s.coordinates.transforms["flipped"].rotation == [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def test_axis_swap_accepted(self) -> None:
        # GSE↔GSM-style relabel: x and z swap with sign flips.
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.swapped]
            rotation = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        """)
        s = validate_simulation_toml(doc)
        assert s.coordinates.transforms["swapped"].rotation is not None

    def test_wrong_row_count_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.bad]
            rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        """)
        with pytest.raises(ValidationError, match="3x3"):
            validate_simulation_toml(doc)

    def test_wrong_column_count_rejected(self) -> None:
        # Vec3Float catches the row width mismatch (min_length=3, max_length=3).
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.bad]
            rotation = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        """)
        with pytest.raises(ValidationError):
            validate_simulation_toml(doc)

    def test_non_unit_entry_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.bad]
            rotation = [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """)
        with pytest.raises(ValidationError, match="signed unit"):
            validate_simulation_toml(doc)

    def test_two_nonzeros_in_row_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.bad]
            rotation = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        """)
        with pytest.raises(ValidationError, match="signed unit"):
            validate_simulation_toml(doc)

    def test_repeated_column_rejected(self) -> None:
        # Two rows pick the same column — not a permutation.
        doc = _minimal_doc() + dedent("""
            [coordinates.transforms.bad]
            rotation = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        """)
        with pytest.raises(ValidationError, match="not a permutation"):
            validate_simulation_toml(doc)


class TestProbePositionLength:
    """``[[probes]].position`` must be 3D when present."""

    def test_three_d_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [1.0, 2.0, 3.0]
        """)
        s = validate_simulation_toml(doc)
        assert s.probes[0].position == [1.0, 2.0, 3.0]

    def test_one_d_rejected(self) -> None:
        # Schema commits to [x, y, z] regardless of grid dimensionality —
        # probes live in physical 3-space, the embedding the trajectory CSV
        # is also expressed in. Catching short positions here is cheaper
        # than letting them break at sample time.
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [1.0]
        """)
        with pytest.raises(ValidationError):
            validate_simulation_toml(doc)

    def test_four_d_rejected(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[probes]]
            name = "p1"
            position = [1.0, 2.0, 3.0, 4.0]
        """)
        with pytest.raises(ValidationError):
            validate_simulation_toml(doc)


class TestBodyCenterAxisCount:
    """``[[bodies]].center`` axis count must match ``grid.dimensions``."""

    def test_matching_axis_count_accepted(self) -> None:
        doc = _minimal_doc() + dedent("""
            [[bodies]]
            name = "earth"
            center = [0.0, 0.0, 0.0]
            radius = 1.0
        """)
        s = validate_simulation_toml(doc)
        assert s.bodies[0].center == [0.0, 0.0, 0.0]

    def test_axis_count_mismatch_rejected(self) -> None:
        # Minimal doc declares a 3D grid; a 2-element center is the same
        # alignment violation that boundary_conditions.lower and
        # coordinates.physical_extent already catch.
        doc = _minimal_doc() + dedent("""
            [[bodies]]
            name = "earth"
            center = [0.0, 0.0]
            radius = 1.0
        """)
        with pytest.raises(ValidationError, match=r"grid\.dimensions"):
            validate_simulation_toml(doc)
