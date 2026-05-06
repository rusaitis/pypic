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
ModelType = Literal["PIC", "MHD", "hybrid", "vlasov", "gyrokinetic"]
Geometry = Literal["cartesian", "spherical", "cylindrical", "thetaMode"]
StaggerKind = Literal["cell", "node", "staggered"]
TimeScheme = Literal[
    "fixed",
    "adaptive",
    "subcycled",
    # RK substages and SSP-RK schemes (Athena++, PLUTO, FLASH, AMReX).
    "rk2",
    "rk3",
    "rk4",
    "vl2",
    "ssprk2",
    "ssprk3",
    # IMEX-RK for stiff source terms (radiation, cooling, chemistry).
    "imex-rk2",
    "imex-rk3",
]
TimeSplitting = Literal["strang", "lie", "godunov"]
DriverCoupling = Literal["boundary", "volume", "source", "sink"]
DriverDirection = Literal["one_way", "two_way"]
Closure = Literal[
    "isothermal",
    "adiabatic",
    "polytropic",
    "braginskii",
    # Anisotropic / multi-moment closures.
    "cgl",
    "10moment",
    "14moment",
]
PICSolverScheme = Literal["explicit", "semi-implicit", "implicit"]
# Pusher: Boris and friends + ED-PIC additions (Lobatto-IIIA RK4, free-streaming).
PICPusher = Literal["boris", "vay", "higuera-cary", "llrk4", "free-streaming"]
# Field solver: FDTD-Yee + pseudo-spectral family + ED-PIC stencils (Lehe,
# Cole-Karkkainen, PSTD, GPSTD, FBPIC's spectral-azimuthal RZ-mode).
PICFieldSolver = Literal[
    "fdtd-yee",
    "pseudo-spectral",
    "psatd",
    "spectral-azimuthal",
    "lehe",
    "ck",
    "ckc",
    "pstd",
    "gpstd",
    "implicit-moment",
    "implicit-gmres",
]
# ED-PIC charge-correction and current-deposition vocabularies.
ChargeCorrection = Literal[
    "marder", "langdon", "boris", "hyperbolic", "spectral", "none"
]
CurrentDeposition = Literal[
    "esirkepov",
    "zigzag",
    "villabune",
    "direct-boris",
    "direct-morse-nielson",
    "none",
]
# Particle shape factor (NGP=0, CIC=1, TSC=2, PQS=3).
ParticleShape = Literal["ngp", "cic", "tsc", "pqs"]
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
OutputPartition = Literal["by_rank", "by_field", "monolithic"]
RestartMode = Literal["hot", "cold"]
RestoreKind = Literal["fields", "particles", "auxiliary"]
AMRKind = Literal["block", "patch", "octree"]
# Collisional PIC models (Smilei, EPOCH, OSIRIS-collisional, PIConGPU).
CollisionModel = Literal["coulomb", "bgk", "monte-carlo"]
# Phase-space coordinate system (gyrokinetic vs continuum-Vlasov).
PhaseSpaceCoordSystem = Literal[
    "cartesian", "guiding-center", "field-aligned", "spherical-velocity"
]
# Per-field BC scope. Drivers can supply boundary values per face per field.
BCFieldGroup = Literal["E", "B", "particles"]
# Region selection for multi-cadence / ROI output streams.
RegionKind = Literal["box", "plane"]
PhysicalExtentUnit = Literal[
    "m", "km", "R_E", "R_S", "R_sun", "R_M", "R_J", "AU", "d_i"
]

# Length-constrained list aliases used by every shape-checked vector field.
# `Vec3*` = strictly 3D. `Axis*` = 1..3 axes (axis count is enforced against
# `grid.dimensions` by the root validator).
type Vec3Float = Annotated[list[float], Field(min_length=3, max_length=3)]
type Vec3Str = Annotated[list[str], Field(min_length=3, max_length=3)]
type AxisFloat = Annotated[list[float], Field(min_length=1, max_length=3)]
type AxisInt = Annotated[list[PositiveInt], Field(min_length=1, max_length=3)]
type AxisPosFloat = Annotated[list[PositiveFloat], Field(min_length=1, max_length=3)]


def _is_extension_key(key: str) -> bool:
    return key.startswith(("x-", "x_"))


class _StrictBase(BaseModel):
    """Reject unknown keys — used for tables where typos are a bug.

    ``populate_by_name`` is left at the Pydantic v2 default (False): aliased
    fields (``[restart].from``, ``[schema]`` itself) accept only the spec
    spelling, never the Python field name (``from_``, ``schema_``). The
    spec is the contract; we don't quietly admit a second name for it.
    """

    model_config = ConfigDict(extra="forbid")


class _ExtensibleBase(BaseModel):
    """Accept unknown keys. Used where v1.0 reserves extension room.

    In the physics sub-trees and on the root, unknown keys either carry
    the ``x-`` namespace (portable across all tables) or are code-specific
    experimental knobs that v1.0 spec explicitly permits.
    """

    model_config = ConfigDict(extra="allow")


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


class Ensemble(_StrictBase):
    """``[run.ensemble]`` — ensemble-member identity for stochastic runs.

    Required for cosmological PIC, turbulence realizations, and any other
    setup where multiple runs share initial conditions modulo a random
    seed. ``member_id`` is 1-indexed and bounded by ``total``; the
    validator enforces ``1 <= member_id <= total``.
    """

    member_id: PositiveInt
    total: PositiveInt

    @model_validator(mode="after")
    def _check_member_id_in_range(self) -> Ensemble:
        if self.member_id > self.total:
            raise ValueError(
                f"ensemble.member_id ({self.member_id}) must be "
                f"<= ensemble.total ({self.total})"
            )
        return self


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
    # Stochasticity / ensemble metadata. Optional — readers fall back to
    # the `x-` extension namespace for codes that don't expose either.
    random_seed: int | None = None
    ensemble: Ensemble | None = None


class Time(_StrictBase):
    """``[time]`` — temporal integration controls.

    Scheme-specific keys are deliberately *not* modelled as a discriminated
    union — the vocabulary is still settling and the cost of a wrong shape
    here is low. v1.0 enforces only the cross-field invariants that are
    unambiguous:

    - fixed/subcycled require ``dt > 0`` (the integration step);
    - subcycled requires ``field_substeps`` (``dt_field`` is recommended
      but advisory, not enforced);
    - adaptive accepts any combination of (``cfl``, ``dt_min``,
      ``dt_max``) including all-None — the runtime owns CFL bookkeeping.

    A future v1.1 may formalise these as a discriminated union once codes
    converge on a common spelling.
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
    splitting: TimeSplitting | None = None

    @model_validator(mode="after")
    def _check_time_range(self) -> Time:
        if self.t_end < self.t_start:
            raise ValueError(f"t_end ({self.t_end}) < t_start ({self.t_start})")
        if self.scheme == "subcycled" and self.field_substeps is None:
            raise ValueError(
                "scheme='subcycled' requires field_substeps (and usually dt_field)"
            )
        # `dt` is a placeholder for adaptive schemes (the actual step is
        # CFL-driven). Every other scheme — fixed, subcycled, and the RK /
        # SSP-RK / IMEX-RK variants — uses ``dt`` as the integration step
        # (or the base step that CFL may further bound from above).
        if self.scheme != "adaptive" and self.dt == 0.0:
            raise ValueError(f"scheme={self.scheme!r} requires dt > 0")
        return self


class GridAMR(_StrictBase):
    """``[grid.amr]`` — dynamic adaptive refinement parameters.

    ``amr_kind`` discriminates block/patch (BoxLib / AMReX / Chombo / FLASH)
    from octree codes (RAMSES, MPI-AMRVAC); octree leaves are single cells
    so ``block_size`` does not apply. ``level_subcycling`` records whether
    different AMR levels advance at different effective timesteps (Athena++
    and AMReX-based codes); a per-level ``dt_factor`` array would be a
    future v1.1 add if needed.
    """

    max_level: NonNegativeInt
    refinement_ratio: PositiveInt = 2
    block_size: list[PositiveInt] | None = None
    refinement_criteria: list[str] = Field(default_factory=list)
    refinement_threshold: NonNegativeFloat | None = None
    amr_kind: AMRKind = "block"
    level_subcycling: bool = False


class GridRefinementBox(_StrictBase):
    """One entry in ``[[grid.refinement]]`` (static nested refinement box)."""

    level: NonNegativeInt
    box: list[list[float]] = Field(..., min_length=2, max_length=2)


class GridStretched(_StrictBase):
    """``[grid.stretched]`` — non-uniform per-axis cell widths.

    Sparse, keyed by axis index as a string (``"0"`` / ``"1"`` / ``"2"``).
    Only stretched axes appear; uniform axes inherit ``Grid.spacing``.
    Per-axis cell widths must sum to ``upper[i] - lower[i]`` and the
    list length must equal ``dimensions[i]``; both invariants are
    enforced by ``Grid._check_axis_consistency``.

    Adopted to describe ARMS spherical-r stretches, PLUTO log-radial
    grids, Athena++ stretched grids, and FLASH per-block-non-uniform
    layouts losslessly. Kept additive so uniform documents validate
    unchanged. Metric-aware operators are out of scope for v1.0.x —
    consumers that need them should guard explicitly.
    """

    # Required; the section is meaningful only when at least one axis is
    # declared stretched. The validator additionally enforces non-empty
    # so an explicit ``axis_widths = {}`` is rejected with a clear message.
    axis_widths: dict[str, list[PositiveFloat]]
    # Relative-tolerance for the cell-width sum vs (upper - lower) check.
    # Defaults are intentionally loose to accommodate float drift in
    # hand-written TOML; tighten via this knob when authoring tests.
    sum_rtol: NonNegativeFloat = 1.0e-9

    @model_validator(mode="after")
    def _check_axis_keys(self) -> GridStretched:
        # Keys must be parseable as small non-negative integers in
        # [0, 2]. The full axis-count cross-check (against
        # ``Grid.dimensions``) lives on ``Grid``.
        if not self.axis_widths:
            raise ValueError(
                "grid.stretched.axis_widths must contain at least one axis"
            )
        for raw_key in self.axis_widths:
            try:
                idx = int(raw_key)
            except ValueError as exc:
                raise ValueError(
                    f"grid.stretched.axis_widths key {raw_key!r} is not a "
                    f"non-negative integer"
                ) from exc
            if idx < 0 or idx > 2:
                raise ValueError(
                    f"grid.stretched.axis_widths key {raw_key!r} is out of range [0, 2]"
                )
        return self


class Grid(_StrictBase):
    """``[grid]`` — computational grid in code units.

    ``ghost_cells`` records the number of ghost cells per axis used by the
    code's halo exchange. Optional metadata for downstream edge-derivative
    analysis; readers strip ghosts before populating ``FieldDataset``.

    ``stretched`` describes per-axis non-uniform cell widths. Sparse —
    only axes with non-uniform spacing appear in
    ``[grid.stretched.axis_widths]``; the remaining axes inherit
    ``spacing``. The validator enforces that any listed axis widths sum
    to ``upper[i] - lower[i]`` and that the list length equals
    ``dimensions[i]``.
    """

    dimensions: AxisInt
    spacing: AxisPosFloat
    lower: AxisFloat
    upper: AxisFloat
    stagger: StaggerKind = "cell"
    ghost_cells: (
        Annotated[list[NonNegativeInt], Field(min_length=1, max_length=3)] | None
    ) = None
    source: str | None = None
    source_format: str | None = None
    amr: GridAMR | None = None
    refinement: list[GridRefinementBox] = Field(default_factory=list)
    stretched: GridStretched | None = None

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
        if self.ghost_cells is not None and len(self.ghost_cells) != n:
            raise ValueError(
                f"ghost_cells has {len(self.ghost_cells)} entries but "
                f"dimensions has {n}"
            )
        if self.stretched is not None:
            for raw_key, widths in self.stretched.axis_widths.items():
                idx = int(raw_key)
                if idx >= n:
                    raise ValueError(
                        f"grid.stretched.axis_widths['{raw_key}'] references "
                        f"axis {idx}, but grid.dimensions has {n} axes"
                    )
                if len(widths) != self.dimensions[idx]:
                    raise ValueError(
                        f"grid.stretched.axis_widths['{raw_key}'] has "
                        f"{len(widths)} entries, expected dimensions[{idx}]"
                        f" = {self.dimensions[idx]}"
                    )
                expected_extent = self.upper[idx] - self.lower[idx]
                actual_extent = float(sum(widths))
                tol = self.stretched.sum_rtol * max(abs(expected_extent), 1.0)
                if abs(actual_extent - expected_extent) > tol:
                    raise ValueError(
                        f"grid.stretched.axis_widths['{raw_key}'] sums to "
                        f"{actual_extent}, expected upper - lower = "
                        f"{expected_extent} (within rtol "
                        f"{self.stretched.sum_rtol})"
                    )
        return self


class BoundaryConditionsBase(_StrictBase):
    """Reusable per-face BC vector + optional driver foreign keys.

    ``lower`` / ``upper`` carry one tag per axis (``"periodic"``,
    ``"reflecting"``, ``"open"``, ``"driven"``, ...). The vocabulary is
    free-form: validators don't constrain the strings, so different code
    families coexist without an upstream enum cut.

    ``drivers_lower`` / ``drivers_upper`` link individual faces to a
    declared ``[[drivers]].name`` entry. Each is sparse: a dict keyed by
    axis index as a string (``"0"``, ``"1"``, ``"2"``), valued with the
    driver name. Faces without a driver simply don't appear. The root
    validator enforces foreign-key resolution against ``[[drivers]]``
    and the axis-index range against ``grid.dimensions``.
    """

    lower: list[str] = Field(..., min_length=1, max_length=3)
    upper: list[str] = Field(..., min_length=1, max_length=3)
    drivers_lower: dict[str, str] | None = None
    drivers_upper: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_face_alignment(self) -> BoundaryConditionsBase:
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"boundary_conditions.lower ({len(self.lower)}) and .upper "
                f"({len(self.upper)}) must have equal length"
            )
        for face_name in ("drivers_lower", "drivers_upper"):
            drivers = getattr(self, face_name)
            if drivers is None:
                continue
            for raw_key in drivers:
                try:
                    idx = int(raw_key)
                except ValueError as exc:
                    raise ValueError(
                        f"boundary_conditions.{face_name} key {raw_key!r} "
                        f"is not a non-negative integer"
                    ) from exc
                if idx < 0 or idx >= len(self.lower):
                    raise ValueError(
                        f"boundary_conditions.{face_name} key {raw_key!r} "
                        f"out of range for {len(self.lower)} axes"
                    )
        return self


class BoundaryConditions(BoundaryConditionsBase):
    """``[boundary_conditions]`` — per-face tag arrays.

    The default form: one tag per face per axis applied uniformly to
    every field. ``field_overrides`` lets PIC PML simulations and
    solar-wind-driven runs declare *different* BCs for E (PML), B (PML),
    and particles (reflecting / absorbing / thermal-bath) at the same
    face. Each entry is a :class:`BoundaryConditionsBase` matching the
    same axis count as the default.
    """

    field_overrides: dict[BCFieldGroup, BoundaryConditionsBase] | None = None

    @model_validator(mode="after")
    def _check_overrides_axis_count(self) -> BoundaryConditions:
        if self.field_overrides is None:
            return self
        n = len(self.lower)
        for field_group, override in self.field_overrides.items():
            if len(override.lower) != n:
                raise ValueError(
                    f"boundary_conditions.field_overrides[{field_group!r}] "
                    f"has {len(override.lower)} axes but the default has {n}"
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

    origin: AxisFloat | None = None
    rotation: list[list[float]] | None = None
    scale: float | None = None
    from_frame: str | None = None
    parameter: str | None = None
    axis_labels: Vec3Str | None = None


class CoordinatesModes(_StrictBase):
    """``[coordinates.modes]`` — azimuthal-mode decomposition for FBPIC RZ.

    Required when ``geometry = "thetaMode"``. ``n_modes`` counts the
    azimuthal modes stored on disk (typically 1–3, with mode 0 cylindrically
    symmetric). ``mode_indices`` is an optional explicit list of mode
    numbers — when omitted, modes are assumed to run ``0, 1, ..., n_modes-1``.
    """

    n_modes: PositiveInt
    mode_indices: list[NonNegativeInt] | None = None

    @model_validator(mode="after")
    def _check_indices_match_count(self) -> CoordinatesModes:
        if self.mode_indices is not None and len(self.mode_indices) != self.n_modes:
            raise ValueError(
                f"coordinates.modes.mode_indices has "
                f"{len(self.mode_indices)} entries but n_modes = {self.n_modes}"
            )
        return self


class Coordinates(_StrictBase):
    """``[coordinates]`` — geometry + reference frame.

    ``physical_extent`` is metadata at the schema level: the validator
    only checks shape and positivity. Post-translation, the reader-side
    helper ``pypic.readers.config.apply_physical_extent`` consumes it to
    auto-compute transform scale factors and a shrink-factor diagnostic
    (see ``docs/schema.md`` § Coordinates).

    ``modes`` is required when ``geometry = "thetaMode"`` (FBPIC azimuthal-
    mode decomposition over an (r, z) grid) and forbidden otherwise.
    """

    geometry: Geometry
    frame: str
    axis_labels: Vec3Str | None = None
    physical_extent: AxisPosFloat | None = None
    physical_extent_unit: PhysicalExtentUnit | None = None
    transforms: dict[str, CoordinateTransform] = Field(default_factory=dict)
    modes: CoordinatesModes | None = None

    @model_validator(mode="after")
    def _check_modes_geometry(self) -> Coordinates:
        if self.geometry == "thetaMode" and self.modes is None:
            raise ValueError(
                "coordinates.geometry = 'thetaMode' requires "
                "[coordinates.modes] (n_modes at minimum)"
            )
        if self.geometry != "thetaMode" and self.modes is not None:
            raise ValueError(
                f"coordinates.modes is only valid for geometry = 'thetaMode' "
                f"(got '{self.geometry}')"
            )
        return self


class PICSolver(_ExtensibleBase):
    """``[physics.pic.solver]``. Extra keys accepted — vocabulary evolves.

    ``current_smoothing`` / ``charge_smoothing`` count the binomial /
    compensator filter passes per step (standard in WarpX, Smilei,
    PIConGPU, OSIRIS — already exposed on ``HybridSolver``).
    ``charge_correction``, ``current_deposition`` adopt the openPMD
    ED-PIC vocabulary; see :class:`ChargeCorrection`,
    :class:`CurrentDeposition`.
    """

    scheme: PICSolverScheme
    implicitness: float | None = None
    pusher: PICPusher | None = None
    field_solver: PICFieldSolver | None = None
    preconditioner: Preconditioner | None = None
    current_smoothing: NonNegativeInt | None = None
    charge_smoothing: NonNegativeInt | None = None
    charge_correction: ChargeCorrection | None = None
    current_deposition: CurrentDeposition | None = None


class PhysicsPIC(_ExtensibleBase):
    """``[physics.pic]`` — PIC physics-model knobs. Extra keys accepted."""

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
    """``[physics.mhd]`` — MHD physics-model knobs. Extra keys accepted."""

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
    """``[physics.hybrid]`` — hybrid physics-model knobs. Extra keys accepted."""

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
    center: AxisFloat
    radius: PositiveFloat | None = None
    shape: ShapeLiteral = "sphere"
    intrinsic_dipole: Vec3Float | None = None
    dipole_center_offset: Vec3Float | None = None
    rotation_axis: Vec3Float | None = None
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
    target_lower: AxisFloat | None = None
    target_upper: AxisFloat | None = None
    body: str | None = None

    @model_validator(mode="after")
    def _check_target_box(self) -> Driver:
        lo, up = self.target_lower, self.target_upper
        if lo is None and up is None:
            return self
        if lo is None or up is None:
            raise ValueError(
                f"driver '{self.name}': target_lower and target_upper must "
                f"both be present or both absent"
            )
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
    """``[restart]`` — continuation pointer from a prior run.

    ``restore`` enables partial restart — restarting only the listed
    state categories rather than all of them (e.g., fields-only
    continuation that re-initializes particles from a fresh distribution).
    ``mode`` distinguishes a full state reload (``"hot"``) from a
    setup-restart that re-applies initial conditions on top of the
    saved geometry (``"cold"``). Both optional; a missing ``restore``
    means the full state is restored.

    ``from_files`` carries an explicit per-rank file list for codes that
    write one checkpoint per MPI rank (VPIC, certain AMReX builds). When
    present, ``from`` typically points at a manifest while ``from_files``
    enumerates the actual files; readers may consult either or both.
    Validator: distinct & non-empty when present.
    """

    from_: str = Field(..., alias="from")
    step: NonNegativeInt | None = None
    time: NonNegativeFloat | None = None
    restore: list[RestoreKind] | None = None
    mode: RestartMode | None = None
    from_files: list[str] | None = None

    @model_validator(mode="after")
    def _check_restore_distinct(self) -> Restart:
        if self.restore is not None and len(set(self.restore)) != len(self.restore):
            raise ValueError(
                f"restart.restore entries must be distinct, got {self.restore}"
            )
        if self.restore is not None and not self.restore:
            raise ValueError("restart.restore must be non-empty when present")
        if self.from_files is not None:
            if not self.from_files:
                raise ValueError("restart.from_files must be non-empty when present")
            if len(set(self.from_files)) != len(self.from_files):
                raise ValueError(
                    f"restart.from_files entries must be distinct, got "
                    f"{self.from_files}"
                )
        return self


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
    thermal_velocity: Vec3Float | None = None
    drift_velocity: Vec3Float | None = None
    density: NonNegativeFloat | None = None
    closure: Closure | None = None
    gamma_eos: PositiveFloat | None = None
    # Anisotropic adiabatic indices for two-fluid + 10-moment hybrids
    # (Gkeyll, Hakim) where the species pressure splits into parallel
    # and perpendicular channels. Both must be set together; cannot
    # combine with the scalar ``gamma_eos``. Validators below enforce
    # the (par AND perp) XOR (gamma_eos) discipline.
    gamma_eos_par: PositiveFloat | None = None
    gamma_eos_perp: PositiveFloat | None = None
    inertia: NonNegativeFloat | None = None
    # Particle shape factor (NGP / CIC / TSC / PQS) per ED-PIC. Optional —
    # codes that pin a global default leave this unset; codes that vary the
    # shape per species (WarpX, Smilei) populate it explicitly.
    shape: ParticleShape | None = None
    # Tracer flag. PIC codes routinely write a tagged subset for trajectory
    # tracking; the reader is responsible for propagating this hint to
    # downstream particle-trajectory analysis. Test-particle vs tagged-tracer
    # semantics may need a follow-up `tracer_kind` field; the boolean is the
    # additive starting point.
    tracer: bool = False

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
        has_par = self.gamma_eos_par is not None
        has_perp = self.gamma_eos_perp is not None
        if has_par != has_perp:
            raise ValueError(
                f"species '{self.name}': gamma_eos_par and gamma_eos_perp "
                f"must both be present or both absent"
            )
        if has_par and self.gamma_eos is not None:
            raise ValueError(
                f"species '{self.name}': cannot set scalar gamma_eos and "
                f"anisotropic (gamma_eos_par, gamma_eos_perp) simultaneously"
            )
        return self


class _OutputBase(_StrictBase):
    """Common fields shared by every ``[output.*]`` sub-table.

    Subclasses override ``precision`` for write modes that should default
    to ``f64`` (lossless checkpoints, probe time-series), and add their
    own type-specific fields.

    ``file_pattern``, ``files_per_step``, and ``partition`` describe the
    on-disk layout: VPIC writes one band-interleaved binary per MPI rank
    per dump (``partition = "by_rank"``); WarpX/openPMD writes per-process
    files plus an index. All optional — single-file readers ignore them.
    """

    step_interval: PositiveInt
    dir: str
    format: FormatLiteral = "hdf5"
    precision: Precision = "f32"
    file_pattern: str | None = None
    files_per_step: PositiveInt | None = None
    partition: OutputPartition | None = None


class OutputCheckpoints(_OutputBase):
    """``[output.checkpoints]`` — lossless full-state dumps."""

    precision: Precision = "f64"
    keep_last: PositiveInt | None = None


class OutputFields(_OutputBase):
    """``[output.fields]`` — field output cadence + quantities."""

    quantities: list[str]
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


class OutputParticles(_OutputBase):
    """``[output.particles]`` — particle output cadence + selection."""

    species: list[str]
    include_ids: bool = False
    include_energy: bool = False
    sample: float | int | None = None


class OutputProbes(_OutputBase):
    """``[output.probes]`` — probe time-series cadence."""

    precision: Precision = "f64"


class OutputDiagnostics(_OutputBase):
    """``[output.diagnostics]`` — on-the-fly derived quantities."""

    quantities: list[str]


class BoxRegion(_StrictBase):
    """``[output.streams.<n>.region]`` — axis-aligned box selection.

    Coordinates in code units (matches ``[grid].lower`` / ``[grid].upper``
    convention). ``upper`` must be strictly greater than ``lower`` per
    axis; the per-axis count must align with ``[grid].dimensions`` —
    enforced at the root level.
    """

    kind: Literal["box"] = "box"
    lower: AxisFloat
    upper: AxisFloat

    @model_validator(mode="after")
    def _check_box(self) -> BoxRegion:
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"output stream box region: lower ({len(self.lower)}) and "
                f"upper ({len(self.upper)}) must align"
            )
        for i, (lo, up) in enumerate(zip(self.lower, self.upper, strict=True)):
            if up <= lo:
                raise ValueError(
                    f"output stream box region: upper[{i}] ({up}) must be > "
                    f"lower[{i}] ({lo})"
                )
        return self


class PlaneRegion(_StrictBase):
    """``[output.streams.<n>.region]`` — single-axis plane slice.

    ``axis`` selects a coordinate axis (``0``/``1``/``2``).
    ``value`` is the plane position in code units; the reader is
    responsible for resolving it to the nearest grid index.
    """

    kind: Literal["plane"] = "plane"
    axis: NonNegativeInt = Field(..., le=2)
    value: float


# Tagged union: TOML doesn't carry the tag itself, so the validator
# discriminates on the ``kind`` literal. Either shape works.
Region = Annotated[BoxRegion | PlaneRegion, Field(discriminator="kind")]


class OutputStream(_OutputBase):
    """One entry in ``[[output.streams]]`` — multi-cadence / ROI output.

    The repeatable form alongside the existing singleton ``[output.fields]``.
    Use this for runs that write moments at 10x the cadence of full
    distributions, or ROI slabs at 10x the cadence of the global volume,
    or that simply need named output groups for downstream pipelines.

    ``name`` is required and must be unique across streams. ``region``
    optionally restricts the write to a sub-volume or plane;
    ``precision_overrides`` re-uses the ``[output.fields]`` semantics
    (per-field-name dtype overrides; names must appear in ``quantities``).
    """

    name: str
    quantities: list[str]
    region: Region | None = None
    precision_overrides: dict[str, Precision] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_overrides_subset(self) -> OutputStream:
        unknown = set(self.precision_overrides).difference(self.quantities)
        if unknown:
            raise ValueError(
                f"output.streams[{self.name!r}].precision_overrides names "
                f"not in quantities: {sorted(unknown)}"
            )
        return self


class Output(_StrictBase):
    """``[output]`` — umbrella for all write-side configuration."""

    checkpoints: OutputCheckpoints | None = None
    fields: OutputFields | None = None
    particles: OutputParticles | None = None
    probes: OutputProbes | None = None
    diagnostics: OutputDiagnostics | None = None
    streams: list[OutputStream] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_stream_names_unique(self) -> Output:
        names = [s.name for s in self.streams]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"output.streams entries must have distinct names; "
                f"duplicates: {duplicates}"
            )
        return self


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
        # `frame` is only meaningful for trajectory probes (it names the
        # frame the trajectory file is expressed in). A fixed probe sits
        # in `[coordinates].frame` by definition; carrying a `frame=` on
        # it is silently ambiguous, so we reject it.
        if has_pos and self.frame is not None:
            raise ValueError(
                f"probe '{self.name}': `frame` is only valid with "
                f"`trajectory` (fixed probes inherit [coordinates].frame)"
            )
        return self


class VelocityMesh(_StrictBase):
    r"""``[velocity_mesh]`` — continuum-Vlasov velocity-grid metadata.

    Vlasiator stores per-cell distribution functions on a 3-D Cartesian
    velocity grid with sparse-block storage. ``dimensions`` and
    ``extent`` describe the velocity-space mesh; ``block_size`` records
    the sparse-block factor for codes that subdivide the velocity grid
    for adaptive memory use; ``sparsity_threshold`` records the
    density floor below which blocks are dropped from disk.

    Recommended when ``[model].type = "vlasov"`` and the run writes
    VDFs; optional for moment-only output. The schema cannot tell from
    the TOML alone whether VDFs are emitted, so this is advisory rather
    than enforced. The validator only checks shape consistency; the
    spatial / phase-space coupling check lives on the root model.

    Relationship to ``[phase_space]``: ``[velocity_mesh]`` describes
    the *storage layout* of the velocity grid (sparse-block structure,
    sparsity threshold), while ``[phase_space]`` describes the
    *coordinate-system identity* of the augmented grid (Cartesian vs
    guiding-center vs field-aligned). Both can coexist for sparse-block
    continuum-Vlasov runs.
    """

    dimensions: list[PositiveInt] = Field(..., min_length=1, max_length=3)
    extent: list[list[float]] = Field(..., min_length=1, max_length=3)
    block_size: list[PositiveInt] | None = None
    sparsity_threshold: NonNegativeFloat | None = None
    coordinate_system: Literal["cartesian", "spherical-velocity"] = "cartesian"

    @model_validator(mode="after")
    def _check_extent_shape(self) -> VelocityMesh:
        if len(self.extent) != len(self.dimensions):
            raise ValueError(
                f"velocity_mesh.extent has {len(self.extent)} axes but "
                f"dimensions has {len(self.dimensions)}"
            )
        for i, axis_extent in enumerate(self.extent):
            if len(axis_extent) != 2:
                raise ValueError(
                    f"velocity_mesh.extent[{i}] must have 2 entries "
                    f"[v_min, v_max], got {len(axis_extent)}"
                )
            v_min, v_max = axis_extent
            if v_max <= v_min:
                raise ValueError(
                    f"velocity_mesh.extent[{i}]: v_max ({v_max}) must be > "
                    f"v_min ({v_min})"
                )
        if self.block_size is not None:
            if len(self.block_size) != len(self.dimensions):
                raise ValueError(
                    f"velocity_mesh.block_size has {len(self.block_size)} "
                    f"axes but dimensions has {len(self.dimensions)}"
                )
            for i, (block, dim) in enumerate(
                zip(self.block_size, self.dimensions, strict=True)
            ):
                if dim % block != 0:
                    raise ValueError(
                        f"velocity_mesh.block_size[{i}] ({block}) must "
                        f"divide dimensions[{i}] ({dim}) evenly"
                    )
        return self


class PhaseSpace(_StrictBase):
    r"""``[phase_space]`` — kinetic phase-space dimensions for >3D codes.

    Gyrokinetic codes (GENE, GS2, GX, Gkeyll-GK) run on 5-D grids
    (3 spatial + 2 velocity); continuum-Vlasov codes use 6-D phase
    space (3 spatial + 3 velocity). This block describes the
    *augmented* phase-space dimensionality without lifting the
    ``Grid.dimensions`` cap on the spatial side.

    Required when the simulation operates on a phase-space grid larger
    than ``[grid].dimensions``. Optional otherwise. The root validator
    enforces consistency: ``[phase_space].dimensions[:n_spatial]``
    must match ``[grid].dimensions`` when both are present.
    """

    dimensions: list[PositiveInt] = Field(..., min_length=2, max_length=6)
    axis_labels: list[str] | None = None
    extents: list[list[float]] | None = None
    coordinate_system: PhaseSpaceCoordSystem = "cartesian"

    @model_validator(mode="after")
    def _check_axis_labels_and_extents(self) -> PhaseSpace:
        n = len(self.dimensions)
        if self.axis_labels is not None and len(self.axis_labels) != n:
            raise ValueError(
                f"phase_space.axis_labels has {len(self.axis_labels)} "
                f"entries but dimensions has {n}"
            )
        if self.extents is not None:
            if len(self.extents) != n:
                raise ValueError(
                    f"phase_space.extents has {len(self.extents)} axes but "
                    f"dimensions has {n}"
                )
            for i, axis_extent in enumerate(self.extents):
                if len(axis_extent) != 2:
                    raise ValueError(
                        f"phase_space.extents[{i}] must have 2 entries "
                        f"[lower, upper], got {len(axis_extent)}"
                    )
                lo, up = axis_extent
                if up <= lo:
                    raise ValueError(
                        f"phase_space.extents[{i}]: upper ({up}) must be > lower ({lo})"
                    )
        return self


class Collision(_ExtensibleBase):
    r"""One entry in ``[[collisions]]`` — inter-species collision model.

    Collisional PIC codes (Smilei, EPOCH, OSIRIS-collisional, PIConGPU)
    declare per-pair Coulomb collisions, BGK relaxation, or
    Monte-Carlo scattering. The schema captures the *declaration* of
    each pair; numerical parameters live on the entry.

    ``species_pair`` lists the two species names participating; the
    root validator enforces that both names exist in ``[[species]]``.
    Self-collisions (same species twice) are permitted.

    Extra keys are accepted (``_ExtensibleBase``) so code-specific
    knobs — cross-section table paths, BGK relaxation-rate models,
    Monte-Carlo scattering-table identifiers — can land here without
    forcing every collisional reader through the top-level ``x-``
    namespace. Portable additions (when a vocabulary stabilises across
    codes) become typed fields in a future version.
    """

    species_pair: list[str] = Field(..., min_length=2, max_length=2)
    model: CollisionModel
    coulomb_log: PositiveFloat | None = None
    temperature_ref: PositiveFloat | None = None
    description: str | None = None


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

    Note on ``schema_version`` vs ``[schema].version``: both are
    required and ``_check_root_invariants`` enforces that they match.
    The duplication is intentional — the bare top-level
    ``schema_version`` lets a streaming parser identify the schema
    version without descending into any table, while ``[schema]``
    is the structured home for related metadata (``created`` date,
    future provenance keys).
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
    velocity_mesh: VelocityMesh | None = None
    phase_space: PhaseSpace | None = None
    collisions: list[Collision] = Field(default_factory=list)

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
        self._check_driver_body_references()
        self._check_collision_species_references()
        self._check_bc_driver_references()
        self._check_output_stream_axis_count(n)
        self._check_phase_space_consistency(n)
        self._check_extras_are_extensions()
        return self

    def _check_physics_matches_model_type(self) -> None:
        assert self.physics is not None
        # Only PIC/MHD/hybrid have a typed sub-table at v1.0; the new
        # vlasov/gyrokinetic model types route through the [physics]
        # extras namespace until typed sub-tables land in v1.1+. For
        # those, no typed-branch cross-check is possible (or needed).
        typed_branch = {"PIC": "pic", "MHD": "mhd", "hybrid": "hybrid"}.get(
            self.model.type
        )
        if typed_branch is None:
            return
        other_branches = {"pic", "mhd", "hybrid"} - {typed_branch}
        for branch in other_branches:
            if getattr(self.physics, branch) is not None:
                raise ValueError(
                    f"model.type = '{self.model.type}' but "
                    f"[physics.{branch}] is set; only "
                    f"[physics.{typed_branch}] is permitted for this model"
                )

    def _check_reference_species_exists(self) -> None:
        """Ensure ``[units].reference_species`` resolves.

        Either matches a declared ``[[species]].name`` entry, or names one
        of the always-valid PIC builtins ``{electrons, ions, protons}``.
        The builtin fallback covers legacy iPIC3D-style configs where the
        normalization references a canonical species the run hasn't
        explicitly declared in ``[[species]]``.
        """
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

    def _check_driver_body_references(self) -> None:
        """Ensure every ``[[drivers]].body`` references a declared body.

        Drivers that delegate their target box to a body (``body = "..."``)
        can only resolve if that body exists in ``[[bodies]]``. Same
        string-as-foreign-key discipline as ``reference_species``.
        """
        body_names = {b.name for b in self.bodies}
        for driver in self.drivers:
            if driver.body is not None and driver.body not in body_names:
                raise ValueError(
                    f"driver '{driver.name}': body = '{driver.body}' does "
                    f"not match any [[bodies]].name entry: "
                    f"{sorted(body_names) or '(none declared)'}"
                )

    def _check_collision_species_references(self) -> None:
        """Ensure every ``[[collisions]].species_pair`` resolves.

        Both species names in each pair must match a declared
        ``[[species]].name`` entry. Self-collisions (same species twice)
        are permitted — codes that need them include them as
        ``species_pair = ["e", "e"]``.
        """
        if not self.collisions:
            return
        names = {s.name for s in self.species}
        for collision in self.collisions:
            for species_name in collision.species_pair:
                if species_name not in names:
                    raise ValueError(
                        f"collisions.species_pair references unknown species "
                        f"'{species_name}'; declared species: {sorted(names)}"
                    )

    def _check_bc_driver_references(self) -> None:
        """Ensure every ``boundary_conditions.drivers_*`` resolves.

        Per-face driver names (in the default ``BoundaryConditions`` and
        in any ``field_overrides`` entry) must match a declared
        ``[[drivers]].name`` entry. ``None`` slots are allowed and mean
        "no driver supplies this face."
        """
        if self.boundary_conditions is None:
            return
        driver_names = {d.name for d in self.drivers}

        def _check_block(block: BoundaryConditionsBase, scope: str) -> None:
            for face_name in ("drivers_lower", "drivers_upper"):
                drivers = getattr(block, face_name)
                if drivers is None:
                    continue
                for axis_key, name in drivers.items():
                    if name not in driver_names:
                        raise ValueError(
                            f"{scope}.{face_name}['{axis_key}'] = '{name}' "
                            f"does not match any [[drivers]].name entry: "
                            f"{sorted(driver_names) or '(none declared)'}"
                        )

        _check_block(self.boundary_conditions, "boundary_conditions")
        if self.boundary_conditions.field_overrides is not None:
            for group, override in self.boundary_conditions.field_overrides.items():
                _check_block(
                    override, f"boundary_conditions.field_overrides[{group!r}]"
                )

    def _check_output_stream_axis_count(self, n: int) -> None:
        """Ensure each ``[[output.streams]].region`` aligns with the grid."""
        if self.output is None or not self.output.streams:
            return
        for stream in self.output.streams:
            region = stream.region
            if region is None:
                continue
            if isinstance(region, BoxRegion):
                if len(region.lower) != n:
                    raise ValueError(
                        f"output.streams[{stream.name!r}].region (box) has "
                        f"{len(region.lower)} axes but grid.dimensions has {n}"
                    )
            elif isinstance(region, PlaneRegion) and region.axis >= n:
                raise ValueError(
                    f"output.streams[{stream.name!r}].region (plane) "
                    f"axis {region.axis} out of range for grid.dimensions"
                    f" ({n} axes)"
                )

    def _check_phase_space_consistency(self, n: int) -> None:
        """Ensure ``[phase_space]`` agrees with ``[grid]`` on the spatial part.

        When ``[phase_space]`` is present, its first ``n`` dimensions must
        match ``[grid].dimensions`` exactly — the spatial sub-grid is
        owned by ``[grid]``; the velocity / extra-D extension lives in
        ``[phase_space]``. The remaining ``len(phase_space.dimensions) - n``
        entries describe the kinetic side (e.g. 2 velocity dims for
        gyrokinetic, 3 for full Vlasov phase space).
        """
        if self.phase_space is None:
            return
        ps_dims = self.phase_space.dimensions
        if len(ps_dims) <= n:
            raise ValueError(
                f"phase_space.dimensions has {len(ps_dims)} axes but "
                f"grid.dimensions has {n} (phase space must strictly "
                f"extend the spatial grid; declare velocity / extra-D "
                f"dimensions beyond the {n} spatial ones)"
            )
        for i in range(n):
            if ps_dims[i] != self.grid.dimensions[i]:
                raise ValueError(
                    f"phase_space.dimensions[{i}] ({ps_dims[i]}) must match "
                    f"grid.dimensions[{i}] ({self.grid.dimensions[i]})"
                )

    def _check_extras_are_extensions(self) -> None:
        extra: dict[str, Any] = self.__pydantic_extra__ or {}
        non_ext = [k for k in extra if not _is_extension_key(k)]
        if non_ext:
            raise ValueError(
                f"unknown top-level keys (extensions must be prefixed 'x-' "
                f"or 'x_'): {sorted(non_ext)}"
            )
