"""Pydantic v2 models for the pypic simulation.toml v2.0 schema.

Zero pypic-internal imports: this module depends only on the standard
library and pydantic. That decoupling is deliberate — the validator is
designed to be lifted into a standalone package without rewrites.

Extension policy
----------------
Unknown top-level keys are accepted only when prefixed ``x-`` or ``x_``
(the v2.0 extension namespace for non-portable knobs). Unknown keys
under ``[physics.{pic,mhd,hybrid,vlasov}]`` and their ``.solver``
sub-tables are accepted without validation (v2.0 spec: validators MUST
accept unknown sub-tables here and MAY warn). Everywhere else,
``extra="forbid"`` catches typos.
"""

from __future__ import annotations

from datetime import date as _date  # noqa: TC003  (pydantic needs it at runtime)
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

# Single source of truth for the schema version. Mirrors the value the
# validator enforces on ``[schema].version`` and pins the filename of the
# exported JSON Schema (``simulation.schema.v{SCHEMA_VERSION}.json``).
SCHEMA_VERSION: Final[str] = "2.0"

# Two flavours of string-typed fields:
#
#   * **Strict** (``Literal[...]``) — bounded structural / format
#     primitives. The vocabulary is fixed by the data model itself
#     (Yee-mesh positions, IEEE precisions, on-disk container formats,
#     coordinate geometries). Adding a value is a v2.x schema bump.
#
#   * **Open** (plain ``str`` with a documented canonical list) —
#     numerical-method, algorithm, and closure vocabularies. Research
#     codes invent new schemes faster than the schema can enumerate
#     them, so unknown strings pass validation. The canonical lists
#     below are guidance for tooling and human readers; pypic does
#     not dispatch on these values, it records them as provenance.

# Strict vocabularies: structural / format primitives, closed enums.
Precision = Literal["f32", "f64"]
ModelType = Literal["PIC", "MHD", "hybrid", "vlasov", "gyrokinetic"]
Geometry = Literal["cartesian", "spherical", "cylindrical", "thetaMode"]
StaggerKind = Literal["cell", "node", "staggered"]
# Per-field-group stagger locations. ``"face"`` and ``"edge"`` are the
# Yee-mesh positions (B on faces, E on edges); ``"cell"`` and ``"node"``
# carry the same meaning as the top-level ``StaggerKind`` summary tag.
StaggerLocation = Literal["cell", "node", "face", "edge"]
DriverCoupling = Literal["boundary", "volume", "source", "sink"]
DriverDirection = Literal["one_way", "two_way"]
FormatLiteral = Literal["hdf5", "zarr", "adios2", "netcdf"]
ShapeLiteral = Literal["sphere", "torus", "cuboid", "mesh"]
OutputPartition = Literal["by_rank", "by_field", "monolithic"]
RestartMode = Literal["hot", "cold"]
RestoreKind = Literal["fields", "particles", "auxiliary"]
AMRKind = Literal["block", "patch", "octree"]
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

# Open vocabularies (see above): aliased to ``str``, with the v2.0 canonical
# values listed per alias.  Cross-field rules — ``scheme = "subcycled"``
# requires ``field_substeps``, say — still fire in the validators below;
# the openness is on *value*, not on *semantics*.
#
# Time integrator: "fixed", "adaptive", "subcycled", "rk2", "rk3", "rk4",
#   "vl2", "ssprk2", "ssprk3", "imex-rk2", "imex-rk3"
TimeScheme = str
# Operator splitting: "strang", "lie", "godunov"
TimeSplitting = str
# Fluid closure: "isothermal", "adiabatic", "polytropic", "braginskii",
#   "cgl", "10moment", "14moment"
Closure = str
# PIC time-integration class: "explicit", "semi-implicit", "implicit"
PICSolverScheme = str
# PIC pusher (Boris and friends + ED-PIC additions): "boris", "vay",
#   "higuera-cary", "llrk4", "free-streaming"
PICPusher = str
# PIC field solver (FDTD-Yee + pseudo-spectral family + ED-PIC stencils):
#   "fdtd-yee", "pseudo-spectral", "psatd", "spectral-azimuthal", "lehe",
#   "ck", "ckc", "pstd", "gpstd", "implicit-moment", "implicit-gmres"
PICFieldSolver = str
# ED-PIC charge correction: "marder", "langdon", "boris", "hyperbolic",
#   "spectral", "none"
ChargeCorrection = str
# ED-PIC current deposition: "esirkepov", "zigzag", "villabune",
#   "direct-boris", "direct-morse-nielson", "none"
CurrentDeposition = str
# Particle shape factor: "ngp" (order 0), "cic" (1), "tsc" (2), "pqs" (3)
ParticleShape = str
# Preconditioner family: "none", "jacobi", "block-jacobi", "ilu", "amg",
#   "additive-schwarz"
Preconditioner = str
# MHD solver scheme: "fct", "godunov", "muscl-hancock", "ppm", "weno"
MHDSolverScheme = str
# MHD reconstruction: "linear", "plm", "ppm", "weno5", "mp5"
MHDReconstruction = str
# MHD slope limiter: "zalesak", "minmod", "mc", "van-leer", "superbee"
MHDLimiter = str
# Divergence cleaning: "ct", "powell", "dedner-glm", "projection", "none"
DivergenceCleaning = str
# MHD Riemann solver: "roe", "hll", "hlle", "hlld", "lax-friedrichs"
MHDRiemann = str
# Hybrid solver scheme: "predictor-corrector", "current-advance-method"
HybridSolverScheme = str
# Hybrid field pusher: "cyclic-leapfrog", "implicit"
HybridFieldPusher = str
# Collision model (Smilei, EPOCH, OSIRIS-collisional, PIConGPU):
#   "coulomb", "bgk", "monte-carlo"
CollisionModel = str

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
    """Accept unknown keys. Used where v2.0 reserves extension room.

    In the physics sub-trees and on the root, unknown keys either carry
    the ``x-`` namespace (portable across all tables) or are code-specific
    experimental knobs that v2.0 spec explicitly permits.
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


class RunReference(_StrictBase):
    """``[[run.references]]`` — one result published from this run.

    Results metadata is what makes an archived run findable by the
    science it produced rather than only by its settings. Each entry
    needs at least one of ``doi`` / ``url`` / ``citation`` — an entry
    naming only a ``kind`` identifies nothing.
    """

    doi: str | None = None
    url: str | None = None
    citation: str | None = None
    # Open string. Canonical: "publication" | "preprint" | "presentation"
    # | "poster" | "dataset" | "thesis" | "report".
    kind: str | None = None

    @model_validator(mode="after")
    def _check_identifiable(self) -> RunReference:
        if self.doi is None and self.url is None and self.citation is None:
            raise ValueError(
                "run.references entry needs at least one of 'doi', 'url' or 'citation'"
            )
        return self


class Run(_StrictBase):
    """``[run]`` — identity + provenance of THIS run.

    ``id`` is the stable, globally unique identifier for the run, and is
    the key every derived product carries back to its source. ``name``
    is a human label and carries no uniqueness contract; ``doi`` is
    scarce by design, since archives do not mint one per run. No format
    is enforced — CCMC run IDs, ULIDs and per-lab conventions all differ,
    and the schema's job here is to record an identifier, not to police
    its shape.

    ``references`` is *results* metadata: what was published from this
    run. Distinct from ``doi`` (this run's data) and ``[model].doi``
    (the code).

    ``idealized`` is deliberately tri-state. ``True`` means the run does
    not correspond to a real time period — artificial drivers, artificial
    internal settings, or a 2D reduction of a 3D system — so harvesting
    real-event characteristics from it would produce wrong metadata.
    ``False`` asserts the conditions are real. ``None`` (the default)
    means unstated, which is what every deck written before this key
    existed actually means; a plain ``bool`` default would make all of
    them silently claim to be real events.

    ``epoch`` anchors code time ``t = 0`` to a UTC instant, so a run of a
    real event can say which event. It lives here rather than on
    ``[time]`` because ``[time]`` does not survive to the typed
    in-memory config — only ``dt`` reaches a `FieldDataset` — whereas
    ``[run]`` round-trips end to end. It is also identity: "which real
    period is this" is a search-and-discovery question, not an
    integrator setting.
    """

    name: str
    id: str | None = None
    description: str | None = None
    idealized: bool | None = None
    authors: list[Author] = Field(default_factory=list)
    date: _date | None = None
    epoch: AwareDatetime | None = None
    git_sha: str | None = None
    host: str | None = None
    license: str | None = None
    doi: str | None = None
    references: list[RunReference] = Field(default_factory=list)
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
    here is low. v2.0 enforces only the cross-field invariants that are
    unambiguous:

    - fixed/subcycled require ``dt > 0`` (the integration step);
    - subcycled requires ``field_substeps`` (``dt_field`` is recommended
      but advisory, not enforced);
    - adaptive omits ``dt`` (or accepts any non-negative value as a hint);
      the runtime derives the first step from ``cfl`` / ``dt_min`` /
      ``dt_max`` and owns CFL bookkeeping thereafter.

    A future v2.1 may formalise these as a discriminated union once codes
    converge on a common spelling.
    """

    scheme: TimeScheme = "fixed"
    dt: NonNegativeFloat | None = None
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
        # Adaptive runs derive the first step from CFL bookkeeping and may
        # omit ``dt`` entirely. Every other scheme — fixed, subcycled, and
        # the RK / SSP-RK / IMEX-RK variants — uses ``dt`` as the integration
        # step (or the base step that CFL may further bound from above).
        if self.scheme != "adaptive" and (self.dt is None or self.dt == 0.0):
            raise ValueError(f"scheme={self.scheme!r} requires dt > 0")
        return self


class GridAMR(_StrictBase):
    """``[grid.amr]`` — dynamic adaptive refinement parameters.

    ``amr_kind`` discriminates block/patch (BoxLib / AMReX / Chombo / FLASH)
    from octree codes (RAMSES, MPI-AMRVAC); octree leaves are single cells
    so ``block_size`` does not apply. ``level_subcycling`` records whether
    different AMR levels advance at different effective timesteps (Athena++
    and AMReX-based codes); a per-level ``dt_factor`` array would be a
    future v2.1 add if needed.
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
    unchanged. Metric-aware operators are out of scope for v2.0.x —
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


class GridStagger(_StrictBase):
    """``[grid.stagger]`` — where fields live inside each cell.

    Three tiers, all optional and additive — same fact at increasing
    precision:

    1. ``convention`` — single-string summary (``"cell"`` / ``"node"``
       / ``"staggered"``). Defaults to ``"cell"``.
    2. ``fields`` — per-field-group location map, e.g.
       ``{B = "face", E = "edge"}``. Captures the Yee-mesh truth that
       a single string cannot.
    3. ``position`` — per-component openPMD ED-PIC offsets in
       ``[0, 1)`` along each axis, e.g. ``{B_1 = [0.5, 0.0, 0.0]}``.
       Lossless representation of the source mesh; consumed by readers
       that need ED-PIC-precise destaggering.

    All three are informational — readers destagger to co-located grids
    on load. They round-trip through `StaggerInfo` in
    ``pypic.containers``. The validator does not cross-check between
    tiers; a writer may populate any subset.
    """

    convention: StaggerKind = "cell"
    fields: dict[str, StaggerLocation] | None = None
    position: dict[str, list[float]] | None = None


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

    ``stagger`` consolidates the three field-placement-inside-cell
    tiers (``convention``, ``fields``, ``position``) under a single
    sub-table — see `GridStagger`.
    """

    dimensions: AxisInt
    spacing: AxisPosFloat
    lower: AxisFloat
    upper: AxisFloat
    stagger: GridStagger = Field(default_factory=GridStagger)
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
        if self.stagger.position is not None:
            for name, offsets in self.stagger.position.items():
                if len(offsets) != n:
                    raise ValueError(
                        f"grid.stagger.position['{name}'] has {len(offsets)} "
                        f"entries, expected dimensions has {n}"
                    )
                for offset in offsets:
                    if not 0.0 <= offset < 1.0:
                        raise ValueError(
                            f"grid.stagger.position['{name}'] = {offsets} — "
                            f"each offset must be in [0.0, 1.0)"
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
    face. Each entry is a `BoundaryConditionsBase` matching the
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
    r"""Fields every ``[units]`` anchor form carries.

    ``data_in_si`` says the stored arrays are in SI and asks the reader
    to rescale them against this anchor on load.  It is the second,
    orthogonal axis: the anchor says how the eight references were
    closed, ``data_in_si`` says what the numbers on disk are measured
    in.  They were one field before 2.0, which is why SI data had no way
    to name an anchor and so no way to be normalized correctly.
    """

    speed_of_light: PositiveFloat | None = None
    data_in_si: bool = False
    scaling_factor: float | None = None
    scaling_description: str | None = None


class UnitsFromSpecies(_UnitsBase):
    r"""``[units]`` with ``anchor = "from_species"`` — length from microphysics.

    The form for codes that resolve a kinetic scale: PIC, hybrid, and
    anything else whose grid is measured in skin depths.  The length
    unit is *derived*, $l_{ref} = c/\omega_{ref}$, from the reference
    species' plasma frequency — which is what distinguishes this form
    from `UnitsExplicit`, where length is given.

    The velocity unit defaults to *c*, the PIC convention.  Hybrid codes
    normalize to the Alfvén speed instead and set `reference_velocity`
    to it, which makes $B_{ref}$ the field at which $v_A$ equals that
    speed and lands the time unit on the inverse ion cyclotron
    frequency, via $d_i \Omega_{ci} = v_A$.
    """

    anchor: Literal["from_species"]
    reference_species: str = "electrons"
    reference_number_density: PositiveFloat
    reference_mass: PositiveFloat | None = None
    # A magnitude, matching the builtin defaults (electrons carry +e).
    # Signed input used to survive here and die downstream complaining
    # about b_field_ref, a key the deck never mentioned.
    reference_charge: PositiveFloat | None = None
    reference_velocity: PositiveFloat | None = None


class UnitsExplicit(_UnitsBase):
    r"""``[units]`` with ``anchor = "explicit"`` — the length unit is given.

    Covers MHD, PLUTO/Athena++-class codes, gyrokinetics, and any
    hand-built reference set.  Three relations close the eight
    primitives:

    $$v = l/t, \qquad E = vB, \qquad B = v\sqrt{\mu_0 n m}$$

    The third ties the field, velocity and density scales together, so
    a deck supplies `reference_length` plus **any two** of:

    - a velocity — `reference_velocity` or `reference_time`
    - a density — `reference_number_density` or `reference_mass_density`
    - a field — `reference_b_field` or `reference_e_field`

    and the third follows.  Supplying all three is legal: the relations
    then describe rather than derive, which is what gyrokinetic decks
    need, where $v \neq l/t$ by $\rho_*$ and $B^2 \neq \mu_0 n m v^2$ by
    $2/\beta$ on purpose.  `Normalization.rationalization_ratio` reports
    the result.

    **Duplicate spellings of one primitive are rejected; redundant
    distinct primitives are accepted.**  Both density keys together is
    an error, because which unit a number carries would be ambiguous.
    `reference_b_field` with `reference_velocity` is fine — those are
    two different quantities, and a tolerance gate could not tell a
    deliberate $2/\beta$ from a typo anyway.

    `reference_mass` defaults to the proton mass whichever density
    spelling is used.  Keying the default off the spelling would make
    two decks describing one plasma disagree by $m_p/m_e$ in silence,
    which is the trap this form exists to remove.
    """

    anchor: Literal["explicit"]
    reference_length: PositiveFloat
    reference_time: PositiveFloat | None = None
    reference_velocity: PositiveFloat | None = None
    reference_b_field: PositiveFloat | None = None
    reference_e_field: PositiveFloat | None = None
    reference_number_density: PositiveFloat | None = None
    reference_mass_density: PositiveFloat | None = None
    reference_mass: PositiveFloat | None = None
    reference_charge: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_determined(self) -> UnitsExplicit:
        if (
            self.reference_number_density is not None
            and self.reference_mass_density is not None
        ):
            raise ValueError(
                "[units] names the density twice: 'reference_number_density' "
                "(m^-3) and 'reference_mass_density' (kg/m^3). Give exactly "
                "one — which unit the number carries is otherwise ambiguous."
            )
        scales = {
            "a velocity ('reference_velocity' or 'reference_time')": (
                self.reference_velocity is not None or self.reference_time is not None
            ),
            ("a density ('reference_number_density' or 'reference_mass_density')"): (
                self.reference_number_density is not None
                or self.reference_mass_density is not None
            ),
            "a field ('reference_b_field' or 'reference_e_field')": (
                self.reference_b_field is not None or self.reference_e_field is not None
            ),
        }
        missing = [name for name, given in scales.items() if not given]
        if len(missing) > 1:
            raise ValueError(
                "[units] anchor = 'explicit' is underdetermined. With "
                "'reference_length' given, supply any two of the three "
                "scales; B = v*sqrt(mu_0*n*m) closes the third. Missing: "
                + "; ".join(missing)
            )
        return self


class UnitsSI(_UnitsBase):
    r"""``[units]`` with ``anchor = "si"`` — SI data with no code-unit anchor.

    All eight references are 1.0, so conversion is a no-op.  That makes
    electromagnetic quantities unsafe: `pypic.derived` computes in
    SI-rationalized units where $\mu_0 = 1$, and SI data has
    $\mu_0 = 1.2566\times10^{-6}$, so anything carrying a vacuum
    constant is wrong by a power of it.

    A deck that wants its SI arrays computed on correctly declares a
    real anchor and sets ``data_in_si`` instead.
    """

    anchor: Literal["si"]

    @model_validator(mode="after")
    def _check_no_rescale(self) -> UnitsSI:
        if self.data_in_si:
            raise ValueError(
                "[units] anchor = 'si' with data_in_si = true would "
                "normalize SI arrays against identity references, which is a "
                "no-op. Declare the anchor to rescale against — "
                "anchor = 'explicit' with reference_length, a density and a "
                "field or velocity."
            )
        return self


Units = Annotated[
    UnitsFromSpecies | UnitsExplicit | UnitsSI,
    Field(discriminator="anchor"),
]


class CoordinateTransform(_StrictBase):
    """One entry under ``[coordinates.transforms.<frame>]``.

    ``rotation``, when present, must be a 3×3 *signed permutation*
    matrix: exactly one ±1 per row and per column with every other
    entry zero. That covers all axis-relabeling / handedness flips
    pypic's frame transforms support today (GSE↔GSM-style rotations
    with continuous angles are time-dependent and route through
    ``parameter`` instead — see ``coordinates.transforms``). Catching
    a malformed matrix here is cheaper than letting it slip into
    ``pypic.coordinates._frames`` and surface as a runtime error at
    apply time.
    """

    origin: AxisFloat | None = None
    rotation: list[Vec3Float] | None = None
    scale: float | None = None
    from_frame: str | None = None
    parameter: str | None = None
    axis_labels: Vec3Str | None = None

    @model_validator(mode="after")
    def _check_rotation_signed_permutation(self) -> CoordinateTransform:
        if self.rotation is None:
            return self
        rows = self.rotation
        if len(rows) != 3:
            raise ValueError(
                f"coordinates.transforms.rotation must be a 3x3 matrix; "
                f"got {len(rows)} rows"
            )
        # Signed permutation: each row and each column has exactly one
        # entry equal to +/-1, all others zero. Equivalent statement —
        # R R^T == I with entries drawn from {-1, 0, +1}.
        col_used = [False, False, False]
        for i, row in enumerate(rows):
            nonzero = [(j, v) for j, v in enumerate(row) if v != 0.0]
            if len(nonzero) != 1 or nonzero[0][1] not in (1.0, -1.0):
                raise ValueError(
                    f"coordinates.transforms.rotation row {i} = {row} is "
                    f"not a signed unit vector (need exactly one +/-1 entry, "
                    f"all others 0)"
                )
            j = nonzero[0][0]
            if col_used[j]:
                raise ValueError(
                    f"coordinates.transforms.rotation column {j} is used "
                    f"twice; matrix is not a permutation"
                )
            col_used[j] = True
        return self


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
    ED-PIC vocabulary; see `ChargeCorrection`,
    `CurrentDeposition`.
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

    gamma_eos: PositiveFloat | None = None
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
    for v2.1+ additions (``[physics.vlasov]`` in particular).
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


class DriverModel(_ExtensibleBase):
    """``[drivers.model]`` — identity of a coupled external system.

    Used when a driver entry refers to a *named model* rather than (or
    in addition to) a flat data file. The type vocabulary is open:
    ``"MHD"`` and ``"PIC"`` cover pypic-aware peers, but
    ``"ionosphere_potential_solver"`` (RIM, Weimer), ``"magnetic_field_
    extrapolation"`` (PFSS, NLFFF), ``"fluid_atmosphere"`` (GITM, TIE-
    GCM), ``"fusion_transport"`` (ASTRA, JETTO), and any other category
    are equally valid.

    ``url`` is a single opaque pointer. It may be the peer's
    ``simulation.toml`` cross-link (when pypic-aware), a homepage URL,
    a DOI, or any other machine- or human-readable metadata reference.
    The schema does not resolve or sniff the format — that is a consumer
    concern, in the same spirit as ``restart.from`` and the top-level
    ``[model].url``. A non-standard metadata file is better than no
    link at all.

    ``_ExtensibleBase`` lets coupling-specific keys (``version``,
    ``doi``, ``git_sha``, model-specific knobs) ride along without
    bloating the typed surface.
    """

    name: str
    type: str
    url: str | None = None
    description: str | None = None


class Driver(_ExtensibleBase):
    """One entry in ``[[drivers]]``.

    Drivers have a core set of v2.0 fields plus driver-type-specific
    keys — extra keys allowed so individual driver types (magnetogram,
    solar_wind_timeseries, pickup_ion_source, ...) don't need a model
    per type in v2.0.

    An entry describes the *external input from this run's
    perspective*. For two-way coupling (``direction = "two_way"``),
    the asymmetry is in information flow, not in physics: the peer
    run, if pypic-aware, owns its own ``simulation.toml`` with its
    own ``[[drivers]]`` entry pointing back. ``[drivers.model]`` is
    where the peer's identity (name, type, url, description) lives;
    pure data drivers (CSV, HDF5 timeseries) can omit the sub-table
    and use ``source`` alone.
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
    model: DriverModel | None = None

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

    ``from`` is the path (or paths) to the restart artifact:

    * a single file (``./chk_000030.h5``),
    * a directory of per-rank checkpoints (``./restart_30000/``),
    * a glob pattern, or
    * an explicit list of per-rank files for runs whose names don't
      follow the source code's convention (relocated reruns, mixed
      naming schemes).

    Whether the path resolves to one file or many is determined at
    read time by the filesystem and the code, not by the schema.

    ``restore`` enables partial restart — restarting only the listed
    state categories rather than all of them (e.g., fields-only
    continuation that re-initializes particles from a fresh
    distribution). ``mode`` distinguishes a full state reload
    (``"hot"``) from a setup-restart that re-applies initial
    conditions on top of the saved geometry (``"cold"``). Both
    optional; a missing ``restore`` means the full state is restored.
    """

    from_: str | list[str] = Field(..., alias="from")
    step: NonNegativeInt | None = None
    time: NonNegativeFloat | None = None
    restore: list[RestoreKind] | None = None
    mode: RestartMode | None = None

    @model_validator(mode="after")
    def _check_restart(self) -> Restart:
        if self.restore is not None and len(set(self.restore)) != len(self.restore):
            raise ValueError(
                f"restart.restore entries must be distinct, got {self.restore}"
            )
        if self.restore is not None and not self.restore:
            raise ValueError("restart.restore must be non-empty when present")
        if isinstance(self.from_, list):
            if not self.from_:
                raise ValueError("restart.from must be non-empty when given as a list")
            if len(set(self.from_)) != len(self.from_):
                raise ValueError(
                    f"restart.from entries must be distinct, got {self.from_}"
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
    # Zero is legal for a fluid species only — massless electrons are the
    # standard hybrid closure, and `inertia` (me/mi, 0 = massless) has
    # always accepted it. `_check_mass_charge` enforces the restriction.
    mass: NonNegativeFloat | None = None
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

    def _is_kinetic(self) -> bool:
        """Whether the deck asks for macroparticles rather than a fluid."""
        ppc = self.particles_per_cell
        if ppc is None:
            return False
        return any(ppc) if isinstance(ppc, list) else ppc > 0

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
        if self.mass == 0.0 and self._is_kinetic():
            raise ValueError(
                f"species '{self.name}' carries particles_per_cell, so it is "
                f"kinetic and needs mass > 0; mass = 0 describes a massless "
                f"fluid species"
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


def _validate_precision_overrides(
    overrides: Mapping[str, Precision],
    quantities: list[str],
    *,
    where: str,
) -> None:
    """Reject precision_overrides keys that don't appear in ``quantities``.

    Shared between `OutputFields` and `OutputStream` — the
    semantics are identical; only the error prefix differs.
    """
    unknown = set(overrides).difference(quantities)
    if unknown:
        raise ValueError(
            f"{where}.precision_overrides names not in quantities: {sorted(unknown)}"
        )


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
        _validate_precision_overrides(
            self.precision_overrides, self.quantities, where="output.fields"
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
        _validate_precision_overrides(
            self.precision_overrides,
            self.quantities,
            where=f"output.streams[{self.name!r}]",
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
    """One entry in ``[[probes]]`` — fixed or trajectory sampler.

    ``fields`` controls which dataset fields the probe samples. Two modes:

    - ``fields = None`` (omitted) — sample every **storage-primitive**
      field present in the dataset at probe time. Storage primitives are
      what readers expose via ``available_fields()`` — moments and EM
      fields actually on disk, never derived quantities like ``|B|`` or
      ``beta``. Users who want derived quantities listed must request
      them explicitly. The narrow default keeps probe time-series cheap
      on big runs.
    - ``fields = [...]`` — sample exactly this list. Names that don't
      resolve at sample time should fail loudly rather than be
      silently dropped; resolution against ``available_fields()``
      happens in the probe sampler, not here.
    """

    name: str
    position: Vec3Float | None = None
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

    def resolve_fields(self, available: Iterable[str]) -> list[str]:
        """Return the concrete field list this probe samples.

        Parameters
        ----------
        available
            Storage-primitive field names present in the dataset, as
            returned by ``reader.available_fields()``.

        Returns
        -------
        list[str]
            ``list(available)`` when ``fields is None``; otherwise
            ``self.fields`` after validating that every name resolves —
            either as a storage primitive in ``available`` or as a
            derived quantity (any name not in ``available`` is left for
            the sampler to dispatch through ``compute()`` and is *not*
            rejected here, since the schema layer has no compute
            registry).
        """
        available_list = list(available)
        if self.fields is None:
            return available_list
        # Explicit list passes through verbatim; the sampler is
        # responsible for raising KeyError on names that resolve neither
        # as primitives nor as derived quantities.
        return list(self.fields)


class PhaseSpaceStorage(_StrictBase):
    r"""``[phase_space.storage]`` — sparse-block velocity-grid storage.

    Continuum-Vlasov codes (Vlasiator, Gkeyll-Vlasov-Maxwell) subdivide
    the velocity sub-grid into blocks for adaptive memory use, dropping
    blocks whose distribution-function density falls below a threshold.
    ``block_size`` records the per-velocity-axis block factor;
    ``sparsity_threshold`` records the density floor.

    Optional sub-table on `PhaseSpace`. Gyrokinetic codes
    (GENE, GS2, GX, Gkeyll-GK) emit dense 5-D grids and omit this
    block entirely. The validator checks that ``block_size`` length
    matches the velocity sub-axes of the parent ``dimensions`` —
    enforced on the parent `PhaseSpace` once the spatial
    dimension count is known at the root level.
    """

    block_size: list[PositiveInt] | None = None
    sparsity_threshold: NonNegativeFloat | None = None


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

    ``storage`` is the sparse-block sub-table for continuum-Vlasov
    codes.  Gyrokinetic runs omit it.
    """

    dimensions: list[PositiveInt] = Field(..., min_length=2, max_length=6)
    axis_labels: list[str] | None = None
    extents: list[list[float]] | None = None
    coordinate_system: PhaseSpaceCoordSystem = "cartesian"
    storage: PhaseSpaceStorage | None = None

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
    """Root model for a pypic simulation.toml v2.0 document.

    Required top-level sections per v2.0:
        [schema], [model], [run], [time], [grid], [units],
        [coordinates], and at least one [[species]] entry.

    Optional sections may be omitted entirely.

    Extension policy at root: unknown keys must start with ``x-`` or
    ``x_``. Known optional sections that are present are strictly
    validated.

    ``[schema].version`` is the single discriminator for both the
    TOML config and the on-disk vocabulary it describes; v2.x is
    additive-only per ``schema.md`` §1 *Versioning*.
    """

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
    phase_space: PhaseSpace | None = None
    collisions: list[Collision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_root_invariants(self) -> SimulationSchema:
        if not self.schema_.version.startswith("2."):
            raise ValueError(
                f"this validator implements schema v2.x; got "
                f"'{self.schema_.version}'. v2.0 renamed [units].system to "
                f"[units].anchor and replaced the PIC/MHD/SI/custom vocabulary "
                f"with from_species/explicit/si — see docs/schema.md section 1."
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
        for body in self.bodies:
            if len(body.center) != n:
                raise ValueError(
                    f"body '{body.name}': center has {len(body.center)} "
                    f"entries but grid.dimensions has {n}"
                )
        if self.physics is not None:
            self._check_physics_matches_model_type()
        self._check_reference_species_exists()
        self._check_body_names_unique()
        self._check_driver_body_references()
        self._check_collision_species_references()
        self._check_bc_driver_references()
        self._check_output_particles_species_references()
        self._check_output_stream_axis_count(n)
        self._check_phase_space_consistency(n)
        self._check_extras_are_extensions()
        return self

    def _check_physics_matches_model_type(self) -> None:
        assert self.physics is not None
        # Only PIC/MHD/hybrid have a typed sub-table at v2.0; the new
        # vlasov/gyrokinetic model types route through the [physics]
        # extras namespace until typed sub-tables land in v2.1+. For
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

    def _check_body_names_unique(self) -> None:
        """Ensure ``[[bodies]].name`` entries are distinct.

        Drivers and initial-condition entries reference bodies by name
        (``[[drivers]].body``, ``[initial_conditions].dipole_bodies``).
        Duplicate names would silently route the foreign key to whichever
        entry the reader happens to encounter first. Catching it at
        validation time keeps body-name reference resolution unambiguous.
        """
        names = [b.name for b in self.bodies]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"bodies entries must have distinct names; duplicates: {duplicates}"
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

    def _check_output_particles_species_references(self) -> None:
        """Ensure every ``[output.particles].species`` resolves.

        Each name in the ``[output.particles].species`` list must match a
        declared ``[[species]].name`` entry. Without this check, a typo
        (``species = ["electron"]``) is silently accepted by the schema
        layer and only surfaces as an empty-write or KeyError at output
        time. Mirrors the foreign-key discipline used by
        ``[[collisions]].species_pair`` and ``[[drivers]].body``.
        """
        if self.output is None or self.output.particles is None:
            return
        requested = self.output.particles.species
        if not requested:
            return
        names = {s.name for s in self.species}
        for species_name in requested:
            if species_name not in names:
                raise ValueError(
                    f"output.particles.species references unknown species "
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
        # ``phase_space.storage.block_size`` describes sparse-block storage
        # of the velocity sub-grid. Its length must equal the number of
        # velocity axes (``len(ps_dims) - n``), and each block must evenly
        # divide the corresponding velocity dimension.
        if self.phase_space.storage is None:
            return
        block_size = self.phase_space.storage.block_size
        if block_size is None:
            return
        n_velocity = len(ps_dims) - n
        if len(block_size) != n_velocity:
            raise ValueError(
                f"phase_space.storage.block_size has {len(block_size)} axes "
                f"but phase_space has {n_velocity} velocity dimensions "
                f"(phase_space.dimensions[{n}:])"
            )
        for i, (block, dim) in enumerate(zip(block_size, ps_dims[n:], strict=True)):
            if dim % block != 0:
                raise ValueError(
                    f"phase_space.storage.block_size[{i}] ({block}) must "
                    f"divide phase_space.dimensions[{n + i}] ({dim}) evenly"
                )

    def _check_extras_are_extensions(self) -> None:
        extra: dict[str, Any] = self.__pydantic_extra__ or {}
        non_ext = [k for k in extra if not _is_extension_key(k)]
        if non_ext:
            raise ValueError(
                f"unknown top-level keys (extensions must be prefixed 'x-' "
                f"or 'x_'): {sorted(non_ext)}"
            )
