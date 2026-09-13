"""Pydantic v2 validator for the pypic simulation.toml v2.0 schema.

Designed to be decoupled from pypic itself — the only imports are
stdlib and pydantic, so this subpackage can be lifted into a
standalone distribution without modification.

Entry point: `validate_simulation_toml`, which accepts a path,
a TOML text blob, or a pre-parsed dict and returns a fully typed
`SimulationSchema` root model.

Examples
--------
>>> from pypic.schema import validate_simulation_toml
>>> doc = '''
... [schema]
... version = "2.0"
... [model]
... name = "demo"
... type = "PIC"
... [run]
... name = "r0"
... [time]
... scheme = "fixed"
... dt = 0.1
... t_start = 0.0
... t_end = 1.0
... n_steps = 10
... [grid]
... dimensions = [4, 4, 4]
... spacing = [1.0, 1.0, 1.0]
... lower = [0.0, 0.0, 0.0]
... upper = [4.0, 4.0, 4.0]
... [units]
... anchor = "si"
... [coordinates]
... geometry = "cartesian"
... frame = "sim"
... [[species]]
... name = "electrons"
... charge = -1.0
... mass = 1.0
... '''
>>> s = validate_simulation_toml(doc)
>>> s.model.type
'PIC'
"""

from pypic.schema._diff import schema_structured_diff, schema_text_diff
from pypic.schema._export import build_schema, dump_schema, get_schema_path
from pypic.schema._loader import ValidationError, validate_simulation_toml
from pypic.schema._models import (
    SCHEMA_VERSION,
    Allocation,
    Author,
    Body,
    BoundaryConditions,
    BoundaryConditionsBase,
    BoxRegion,
    Collision,
    Coordinates,
    CoordinatesModes,
    CoordinateTransform,
    Driver,
    DriverModel,
    Ensemble,
    Grid,
    GridAMR,
    GridRefinementBox,
    GridStretched,
    HybridSolver,
    InitialConditions,
    MHDSolver,
    Model,
    Output,
    OutputCheckpoints,
    OutputDiagnostics,
    OutputFields,
    OutputParticles,
    OutputProbes,
    OutputStream,
    PhaseSpace,
    PhaseSpaceStorage,
    Physics,
    PhysicsHybrid,
    PhysicsMHD,
    PhysicsPIC,
    PICSolver,
    PlaneRegion,
    Probe,
    Region,
    Restart,
    Run,
    RunResources,
    SchemaMeta,
    SimulationSchema,
    Species,
    Time,
    Units,
    UnitsExplicit,
    UnitsFromSpecies,
    UnitsSI,
)

__all__ = [
    "SCHEMA_VERSION",
    "Allocation",
    "Author",
    "Body",
    "BoundaryConditions",
    "BoundaryConditionsBase",
    "BoxRegion",
    "Collision",
    "CoordinateTransform",
    "Coordinates",
    "CoordinatesModes",
    "Driver",
    "DriverModel",
    "Ensemble",
    "Grid",
    "GridAMR",
    "GridRefinementBox",
    "GridStretched",
    "HybridSolver",
    "InitialConditions",
    "MHDSolver",
    "Model",
    "Output",
    "OutputCheckpoints",
    "OutputDiagnostics",
    "OutputFields",
    "OutputParticles",
    "OutputProbes",
    "OutputStream",
    "PICSolver",
    "PhaseSpace",
    "PhaseSpaceStorage",
    "Physics",
    "PhysicsHybrid",
    "PhysicsMHD",
    "PhysicsPIC",
    "PlaneRegion",
    "Probe",
    "Region",
    "Restart",
    "Run",
    "RunResources",
    "SchemaMeta",
    "SimulationSchema",
    "Species",
    "Time",
    "Units",
    "UnitsExplicit",
    "UnitsFromSpecies",
    "UnitsSI",
    "ValidationError",
    "build_schema",
    "dump_schema",
    "get_schema_path",
    "schema_structured_diff",
    "schema_text_diff",
    "validate_simulation_toml",
]
