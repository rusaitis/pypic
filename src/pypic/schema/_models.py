"""Pydantic v2 models for the pypic simulation.toml v1.0 schema.

Zero pypic-internal imports: this module depends only on the standard
library and pydantic. That decoupling is deliberate — the validator is
designed to be lifted into a standalone package without rewrites.

Extension policy
----------------
Unknown top-level keys are accepted only when prefixed ``x-`` or ``x_``
(the v1.0 extension namespace for non-portable knobs). Unknown keys
under ``[physics.{pic,mhd,hybrid,vlasov}]`` and their ``.solver``
sub-tables are accepted without validation (v1.0 spec: validators MUST
accept unknown sub-tables here and MAY warn). Everywhere else,
``extra="forbid"`` catches typos.
"""

from __future__ import annotations

from datetime import date as _date  # noqa: TC003  (pydantic needs it at runtime)
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

Precision = Literal["f32", "f64"]
ModelType = Literal["PIC", "MHD", "hybrid"]
Geometry = Literal["cartesian", "spherical", "cylindrical"]
StaggerKind = Literal["cell", "node", "staggered"]
TimeScheme = Literal["fixed", "adaptive", "subcycled"]
DriverCoupling = Literal["boundary", "volume", "source", "sink"]
DriverDirection = Literal["one_way", "two_way"]
Closure = Literal["isothermal", "adiabatic", "polytropic", "braginskii"]
PICSolverScheme = Literal["explicit", "semi-implicit", "implicit"]
PICPusher = Literal["boris", "vay", "higuera-cary"]
PICFieldSolver = Literal[
    "fdtd-yee", "pseudo-spectral", "implicit-moment", "implicit-gmres"
]
Preconditioner = Literal[
    "none", "jacobi", "block-jacobi", "ilu", "amg", "additive-schwarz"
]
MHDSolverScheme = Literal["fct", "godunov", "muscl-hancock", "ppm", "weno"]
MHDReconstruction = Literal["linear", "plm", "ppm", "weno5", "mp5"]
MHDLimiter = Literal["zalesak", "minmod", "mc", "van-leer", "superbee"]
DivergenceCleaning = Literal["ct", "powell", "dedner-glm", "projection", "none"]
MHDRiemann = Literal["roe", "hll", "hlle", "hlld", "lax-friedrichs"]
HybridSolverScheme = Literal["predictor-corrector", "current-advance-method"]
HybridFieldPusher = Literal["cyclic-leapfrog", "implicit"]
FormatLiteral = Literal["hdf5", "zarr", "adios2", "netcdf"]
ShapeLiteral = Literal["sphere", "torus", "cuboid", "mesh"]
PhysicalExtentUnit = Literal[
    "m", "km", "R_E", "R_S", "R_sun", "R_M", "R_J", "AU", "d_i"
]


def _is_extension_key(key: str) -> bool:
    return key.startswith(("x-", "x_"))


class _StrictBase(BaseModel):
    """Reject unknown keys — used for tables where typos are a bug."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _ExtensibleBase(BaseModel):
    """Accept unknown keys. Used where v1.0 reserves extension room.

    In the physics sub-trees and on the root, unknown keys either carry
    the ``x-`` namespace (portable across all tables) or are code-specific
    experimental knobs that v1.0 spec explicitly permits.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Author(_StrictBase):
    """Person entry in ``[model].authors`` or ``[run].authors``."""

    name: str
    orcid: str | None = None
    affiliation: str | None = None
    role: str | None = None


class Model(_StrictBase):
    """``[model]`` — identity of the code that produced the data."""

    name: str
    type: ModelType
    version: str | None = None
    description: str | None = None
    url: str | None = None
    doi: str | None = None
    license: str | None = None
    authors: list[Author] = Field(default_factory=list)


class Allocation(_StrictBase):
    """One entry in ``[[run.resources.allocations]]`` (heterogeneous HW)."""

    type: Literal["cpu", "gpu", "tpu", "accelerator"]
    hardware: str
    count: PositiveInt
    hours: NonNegativeFloat


class RunResources(_StrictBase):
    """``[run.resources]`` — runtime accounting (all keys optional)."""

    mpi_ranks: PositiveInt | None = None
    mpi_topology: list[PositiveInt] | None = None
    nodes: PositiveInt | None = None
    wall_clock_hours: NonNegativeFloat | None = None
    cpu_core_hours: NonNegativeFloat | None = None
    node_hours: NonNegativeFloat | None = None
    peak_memory_gb: NonNegativeFloat | None = None
    cpu_type: str | None = None
    gpu_hours: NonNegativeFloat | None = None
    energy_kwh: NonNegativeFloat | None = None
    carbon_kg_co2eq: NonNegativeFloat | None = None
    allocations: list[Allocation] = Field(default_factory=list)


class Run(_StrictBase):
    """``[run]`` — identity + provenance of THIS run."""

    name: str
    description: str | None = None
    authors: list[Author] = Field(default_factory=list)
    date: _date | None = None
    git_sha: str | None = None
    host: str | None = None
    license: str | None = None
    doi: str | None = None
    funding: list[str] = Field(default_factory=list)
    embargo: _date | None = None
    resources: RunResources | None = None


class Time(_StrictBase):
    """``[time]`` — temporal integration controls.

    Scheme-specific keys (``cfl``, ``dt_min``, ``dt_max`` for adaptive;
    ``dt_field``, ``field_substeps`` for subcycled) are validated after
    the fact rather than modeled as discriminated unions — the vocabulary
    is still settling and the cost of a wrong shape here is low.
    """

    scheme: TimeScheme = "fixed"
    dt: NonNegativeFloat
    t_start: float
    t_end: float
    n_steps: PositiveInt
    cfl: float | None = None
    dt_min: float | None = None
    dt_max: float | None = None
    dt_field: float | None = None
    field_substeps: PositiveInt | None = None

    @model_validator(mode="after")
    def _check_time_range(self) -> Time:
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        if self.scheme == "subcycled" and self.field_substeps is None:
            raise ValueError(
                "scheme='subcycled' requires field_substeps (and usually dt_field)"
            )
        # `dt` is a placeholder for adaptive schemes (the actual step is
        # CFL-driven), but for fixed/subcycled it is the integration step.
        if self.scheme in ("fixed", "subcycled") and self.dt == 0.0:
            raise ValueError(f"scheme={self.scheme!r} requires dt > 0")
        return self


class GridAMR(_StrictBase):
    """``[grid.amr]`` — dynamic adaptive refinement parameters."""

    max_level: NonNegativeInt
    refinement_ratio: PositiveInt = 2
    block_size: list[PositiveInt] | None = None
    refinement_criteria: list[str] = Field(default_factory=list)
    refinement_threshold: NonNegativeFloat | None = None


class GridRefinementBox(_StrictBase):
    """One entry in ``[[grid.refinement]]`` (static nested refinement box)."""

    level: NonNegativeInt
    box: list[list[float]] = Field(..., min_length=2, max_length=2)


class Grid(_StrictBase):
    """``[grid]`` — computational grid in code units."""

    dimensions: list[PositiveInt] = Field(..., min_length=1, max_length=3)
    spacing: list[PositiveFloat] = Field(..., min_length=1, max_length=3)
    lower: list[float] = Field(..., min_length=1, max_length=3)
    upper: list[float] = Field(..., min_length=1, max_length=3)
    stagger: StaggerKind = "cell"
    source: str | None = None
    source_format: str | None = None
    amr: GridAMR | None = None
    refinement: list[GridRefinementBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_axis_consistency(self) -> Grid:
        n = len(self.dimensions)
        for field_name in ("spacing", "lower", "upper"):
            arr = getattr(self, field_name)
            if len(arr) != n:
                raise ValueError(
                    f"{field_name} has {len(arr)} entries but dimensions has {n}"
                )
        for i, (lo, up) in enumerate(zip(self.lower, self.upper, strict=True)):
            if up <= lo:
                raise ValueError(f"upper[{i}] ({up}) must be > lower[{i}] ({lo})")
        return self


class BoundaryConditions(_StrictBase):
    """``[boundary_conditions]`` — per-face tag arrays."""

    lower: list[str] = Field(..., min_length=1, max_length=3)
    upper: list[str] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="after")
    def _check_same_length(self) -> BoundaryConditions:
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"boundary_conditions.lower ({len(self.lower)}) and .upper "
                f"({len(self.upper)}) must have equal length"
            )
        return self


class _UnitsBase(_StrictBase):
    scaling_factor: float | None = None
    scaling_description: str | None = None


class UnitsPIC(_UnitsBase):
    """``[units]`` with ``system = "PIC"``."""

    system: Literal["PIC"]
    reference_species: str = "electrons"
    reference_density: PositiveFloat
    reference_mass: PositiveFloat | None = None
    reference_charge: float | None = None
    speed_of_light: PositiveFloat | None = None


class UnitsMHD(_UnitsBase):
    """``[units]`` with ``system = "MHD"``."""

    system: Literal["MHD"]
    reference_length: PositiveFloat
    reference_density: PositiveFloat
    reference_b_field: PositiveFloat


class UnitsSI(_UnitsBase):
    """``[units]`` with ``system = "SI"`` — already SI; no refs needed."""

    system: Literal["SI"]


class UnitsReferenceTable(_StrictBase):
    """``[units.reference]`` sub-table for ``system = "custom"``."""

    length: PositiveFloat
    time: PositiveFloat | None = None
    velocity: PositiveFloat | None = None
    b_field: PositiveFloat | None = None
    e_field: PositiveFloat | None = None
    density: PositiveFloat | None = None
    mass: PositiveFloat | None = None
    charge: float | None = None


class UnitsCustom(_UnitsBase):
    """``[units]`` with ``system = "custom"``."""

    system: Literal["custom"]
    reference: UnitsReferenceTable


Units = Annotated[
    UnitsPIC | UnitsMHD | UnitsSI | UnitsCustom,
    Field(discriminator="system"),
]


class CoordinateTransform(_StrictBase):
    """One entry under ``[coordinates.transforms.<frame>]``."""

    origin: list[float] | None = Field(None, min_length=1, max_length=3)
    rotation: list[list[float]] | None = None
    scale: float | None = None
    from_frame: str | None = None
    parameter: str | None = None
    axis_labels: list[str] | None = Field(None, min_length=3, max_length=3)


class Coordinates(_StrictBase):
    """``[coordinates]`` — geometry + reference frame."""

    geometry: Geometry
    frame: str
    axis_labels: list[str] | None = Field(None, min_length=3, max_length=3)
    physical_extent: list[PositiveFloat] | None = Field(
        None, min_length=1, max_length=3
    )
    physical_extent_unit: PhysicalExtentUnit | None = None
    transforms: dict[str, CoordinateTransform] = Field(default_factory=dict)


class PICSolver(_ExtensibleBase):
    """``[physics.pic.solver]``. Extra keys accepted — vocabulary evolves."""

    scheme: PICSolverScheme
    implicitness: float | None = None
    pusher: PICPusher | None = None
    field_solver: PICFieldSolver | None = None
    preconditioner: Preconditioner | None = None


class PhysicsPIC(_ExtensibleBase):
    """``[physics.pic]`` — PIC physics-model knobs."""

    omega_p_over_omega_c: PositiveFloat | None = None
    solver: PICSolver | None = None


class MHDSolver(_ExtensibleBase):
    """``[physics.mhd.solver]``. Extra keys accepted."""

    scheme: MHDSolverScheme
    reconstruction: MHDReconstruction | None = None
    limiter: MHDLimiter | None = None
    divergence_cleaning: DivergenceCleaning | None = None
    riemann: MHDRiemann | None = None
    preconditioner: Preconditioner | None = None


class PhysicsMHD(_ExtensibleBase):
    """``[physics.mhd]`` — MHD physics-model knobs."""

    gamma: PositiveFloat | None = None
    resistivity: NonNegativeFloat | None = None
    hall_term: bool | None = None
    solver: MHDSolver | None = None


class HybridSolver(_ExtensibleBase):
    """``[physics.hybrid.solver]``. Extra keys accepted."""

    scheme: HybridSolverScheme
    field_pusher: HybridFieldPusher | None = None
    resistivity: NonNegativeFloat | None = None
    hyper_resistivity: NonNegativeFloat | None = None
    current_smoothing: NonNegativeInt | None = None
    preconditioner: Preconditioner | None = None


class PhysicsHybrid(_ExtensibleBase):
    """``[physics.hybrid]`` — hybrid physics-model knobs."""

    solver: HybridSolver | None = None


class Physics(_ExtensibleBase):
    """``[physics]`` — flags whose semantics are identical across models.

    Extra top-level keys under ``[physics]`` are accepted to leave room
    for v1.1+ additions (``[physics.vlasov]`` in particular).
    """

    relativistic: bool = False
    pic: PhysicsPIC | None = None
    mhd: PhysicsMHD | None = None
    hybrid: PhysicsHybrid | None = None


class Body(_StrictBase):
    """One entry in ``[[bodies]]`` — registry of physical objects."""

    name: str
    center: list[float] = Field(..., min_length=1, max_length=3)
    radius: PositiveFloat | None = None
    shape: ShapeLiteral = "sphere"
    intrinsic_dipole: list[float] | None = Field(None, min_length=3, max_length=3)
    dipole_center_offset: list[float] | None = Field(None, min_length=3, max_length=3)
    rotation_axis: list[float] | None = Field(None, min_length=3, max_length=3)
    rotation_period: NonNegativeFloat | None = None
    mass: PositiveFloat | None = None
    surface_absorbs_ions: bool | None = None
    has_atmosphere: bool | None = None
    has_intrinsic_field: bool | None = None


class InitialConditions(_ExtensibleBase):
    """``[initial_conditions]`` — flat table, setup-specific keys vary."""

    type: str


class Driver(_ExtensibleBase):
    """One entry in ``[[drivers]]``.

    Drivers have a core set of v1.0 fields plus driver-type-specific
    keys — extra keys allowed so individual driver types (magnetogram,
    solar_wind_timeseries, pickup_ion_source, ...) don't need a model
    per type in v1.0.
    """

    name: str
    type: str
    coupling: DriverCoupling
    direction: DriverDirection = "one_way"
    description: str | None = None
    source: str | None = None
    columns: list[str] | None = None
    cadence: NonNegativeFloat | None = None
    interpolation: str | None = None
    target_lower: list[float] | None = Field(None, min_length=1, max_length=3)
    target_upper: list[float] | None = Field(None, min_length=1, max_length=3)
    body: str | None = None

    @model_validator(mode="after")
    def _check_target_box(self) -> Driver:
        lo, up = self.target_lower, self.target_upper
        if lo is None and up is None:
            return self
        if (lo is None) != (up is None):
            raise ValueError(
                f"driver '{self.name}': target_lower and target_upper must "
                f"both be present or both absent"
            )
        assert lo is not None  # narrowed by the (lo is None) != (up is None) check
        assert up is not None  # narrowed by the (lo is None) != (up is None) check
        if len(lo) != len(up):
            raise ValueError(
                f"driver '{self.name}': target_lower has {len(lo)} entries "
                f"but target_upper has {len(up)}"
            )
        # Allow per-axis equality (thin sheet / line / surface drivers like
        # photospheric magnetograms); reject only an axis where upper < lower.
        for i, (lov, upv) in enumerate(zip(lo, up, strict=True)):
            if upv < lov:
                raise ValueError(
                    f"driver '{self.name}': target_upper[{i}] ({upv}) must be "
                    f">= target_lower[{i}] ({lov})"
                )
        return self


class Restart(_StrictBase):
    """``[restart]`` — continuation pointer from a prior run."""

    from_: str = Field(..., alias="from")
    step: NonNegativeInt | None = None
    time: NonNegativeFloat | None = None


class Species(_StrictBase):
    """One entry in ``[[species]]``.

    Provide either ``(charge, mass)`` OR ``charge_to_mass``, not both.
    Validated after the fact.

    Species are indexed in declaration order — ``_s0`` binds to the
    first ``[[species]]`` entry, ``_s1`` to the second, etc. Reordering
    entries is a breaking change to any downstream field-name reference
    that uses the ``_sN`` suffix.
    """

    name: str
    charge: float | None = None
    mass: PositiveFloat | None = None
    charge_to_mass: float | None = None
    # 0 (or absent) marks a fluid species — see [[species]] in schema.md.
    particles_per_cell: (
        NonNegativeInt
        | Annotated[list[NonNegativeInt], Field(min_length=3, max_length=3)]
        | None
    ) = None
    temperature: NonNegativeFloat | None = None
    thermal_velocity: (
        Annotated[list[float], Field(min_length=3, max_length=3)] | None
    ) = None
    drift_velocity: Annotated[list[float], Field(min_length=3, max_length=3)] | None = (
        None
    )
    density: NonNegativeFloat | None = None
    closure: Closure | None = None
    gamma_eos: PositiveFloat | None = None
    inertia: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _check_mass_charge(self) -> Species:
        has_qm = self.charge is not None and self.mass is not None
        has_qom = self.charge_to_mass is not None
        if not (has_qm or has_qom):
            raise ValueError(
                f"species '{self.name}' must specify either "
                f"(charge + mass) or charge_to_mass"
            )
        if has_qm and has_qom:
            raise ValueError(
                f"species '{self.name}' has both (charge + mass) and "
                f"charge_to_mass — choose one"
            )
        return self


class OutputCheckpoints(_StrictBase):
    """``[output.checkpoints]`` — lossless full-state dumps."""

    step_interval: PositiveInt
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f64"
    keep_last: PositiveInt | None = None


class OutputFields(_StrictBase):
    """``[output.fields]`` — field output cadence + quantities."""

    step_interval: PositiveInt
    quantities: list[str]
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f32"
    precision_overrides: dict[str, Precision] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_overrides_subset(self) -> OutputFields:
        unknown = set(self.precision_overrides).difference(self.quantities)
        if unknown:
            raise ValueError(
                "output.fields.precision_overrides names not in quantities: "
                f"{sorted(unknown)}"
            )
        return self


class OutputParticles(_StrictBase):
    """``[output.particles]`` — particle output cadence + selection."""

    step_interval: PositiveInt
    species: list[str]
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f32"
    include_ids: bool = False
    include_energy: bool = False
    sample: float | int | None = None


class OutputProbes(_StrictBase):
    """``[output.probes]`` — probe time-series cadence."""

    step_interval: PositiveInt
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f64"


class OutputDiagnostics(_StrictBase):
    """``[output.diagnostics]`` — on-the-fly derived quantities."""

    step_interval: PositiveInt
    quantities: list[str]
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f32"


class Output(_StrictBase):
    """``[output]`` — umbrella for all write-side configuration."""

    checkpoints: OutputCheckpoints | None = None
    fields: OutputFields | None = None
    particles: OutputParticles | None = None
    probes: OutputProbes | None = None
    diagnostics: OutputDiagnostics | None = None


class Probe(_StrictBase):
    """One entry in ``[[probes]]`` — fixed or trajectory sampler."""

    name: str
    position: list[float] | None = None
    trajectory: str | None = None
    fields: list[str] | None = None
    frame: str | None = None

    @model_validator(mode="after")
    def _check_position_xor_trajectory(self) -> Probe:
        has_pos = self.position is not None
        has_traj = self.trajectory is not None
        if has_pos == has_traj:
            raise ValueError(
                f"probe '{self.name}' must have exactly one of "
                f"`position` or `trajectory`"
            )
        return self


class SchemaMeta(_StrictBase):
    """``[schema]`` — schema version + creation date."""

    version: str
    created: _date | None = None


class SimulationSchema(_ExtensibleBase):
    """Root model for a pypic simulation.toml v1.0 document.

    Required top-level sections per v1.0:
        [schema], [model], [run], [time], [grid], [units],
        [coordinates], and at least one [[species]] entry.

    Optional sections may be omitted entirely.

    Extension policy at root: unknown keys must start with ``x-`` or
    ``x_``. Known optional sections that are present are strictly
    validated.
    """

    schema_version: str
    schema_: SchemaMeta = Field(..., alias="schema")
    model: Model
    run: Run
    time: Time
    grid: Grid
    units: Units
    coordinates: Coordinates
    species: list[Species] = Field(..., min_length=1)

    boundary_conditions: BoundaryConditions | None = None
    physics: Physics | None = None
    bodies: list[Body] = Field(default_factory=list)
    initial_conditions: InitialConditions | None = None
    drivers: list[Driver] = Field(default_factory=list)
    restart: Restart | None = None
    output: Output | None = None
    probes: list[Probe] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_root_invariants(self) -> SimulationSchema:
        if self.schema_version != self.schema_.version:
            raise ValueError(
                f"schema_version bare key ('{self.schema_version}') and "
                f"[schema].version ('{self.schema_.version}') must match"
            )
        if not self.schema_version.startswith("1."):
            raise ValueError(
                f"this validator implements schema v1.x; got '{self.schema_version}'"
            )
        n = len(self.grid.dimensions)
        if (
            self.boundary_conditions is not None
            and len(self.boundary_conditions.lower) != n
        ):
            raise ValueError(
                f"boundary_conditions has "
                f"{len(self.boundary_conditions.lower)} axes but "
                f"grid.dimensions has {n}"
            )
        if (
            self.coordinates.physical_extent is not None
            and len(self.coordinates.physical_extent) != n
        ):
            raise ValueError(
                f"coordinates.physical_extent has "
                f"{len(self.coordinates.physical_extent)} entries but "
                f"grid.dimensions has {n}"
            )
        if self.physics is not None:
            self._check_physics_matches_model_type()
        self._check_reference_species_exists()
        self._check_extras_are_extensions()
        return self

    def _check_physics_matches_model_type(self) -> None:
        assert self.physics is not None
        expected = {"PIC": "pic", "MHD": "mhd", "hybrid": "hybrid"}[self.model.type]
        other_branches = {"pic", "mhd", "hybrid"} - {expected}
        for branch in other_branches:
            if getattr(self.physics, branch) is not None:
                raise ValueError(
                    f"model.type = '{self.model.type}' but "
                    f"[physics.{branch}] is set; only "
                    f"[physics.{expected}] is permitted for this model"
                )

    def _check_reference_species_exists(self) -> None:
        ref = getattr(self.units, "reference_species", None)
        if ref is None:
            return
        names = {s.name for s in self.species}
        builtins = {"electrons", "ions", "protons"}
        if ref not in names and ref not in builtins:
            raise ValueError(
                f"units.reference_species = '{ref}' does not match any "
                f"[[species]].name entry: {sorted(names)}"
            )

    def _check_extras_are_extensions(self) -> None:
        extra: dict[str, Any] = self.__pydantic_extra__ or {}
        non_ext = [k for k in extra if not _is_extension_key(k)]
        if non_ext:
            raise ValueError(
                f"unknown top-level keys (extensions must be prefixed 'x-' "
                f"or 'x_'): {sorted(non_ext)}"
            )
