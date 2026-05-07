"""Immutable data containers: SimulationConfig, TabularData, ParticleData."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.coordinates.geometry import CARTESIAN  # noqa: F401 — used in doctests
from pypic.grid import GridInfo  # noqa: TC001 — used in doctests
from pypic.units import PhysicsParams

if TYPE_CHECKING:
    from pypic.coordinates.transforms import FrameTransform
    from pypic.schema import (
        Body,
        Collision,
        Driver,
        InitialConditions,
        Output,
        PhaseSpace,
        Probe,
        Restart,
        Run,
    )
    from pypic.types import FloatArray, ModelType
    from pypic.units import Normalization, SpeciesInfo


@dataclass(frozen=True, slots=True)
class StaggerInfo:
    r"""Provenance record of the original grid stagger convention.

    Readers destagger to co-located grids on load.  ``StaggerInfo``
    documents what the grid looked like *before* destaggering — purely
    informational, never used in computation or operators.

    Parameters
    ----------
    convention : str
        Overall grid type: ``"node"`` (all fields on vertices),
        ``"cell"`` (all fields at cell centers), or ``"staggered"``
        (Yee mesh — B on faces, E on edges, etc.).
    field_locations : dict[str, str] | None
        Per-field-group stagger locations, e.g.
        ``{"B": "face", "E": "edge"}``.  Only meaningful for the
        ``"staggered"`` convention; ``None`` otherwise.  Frozen to
        ``MappingProxyType`` after construction.
    position : dict[str, tuple[float, ...]] | None
        Per-component stagger offsets in ``[0.0, 1.0)``, one tuple per
        canonical field component (e.g. ``{"B1": (0.5, 0.0, 0.0),
        "E1": (0.0, 0.5, 0.5)}``).  Adopts the openPMD ED-PIC
        ``position`` semantics so Yee-mesh PIC, BATSRUS face-centered B,
        and any future co-located write-out can describe their native
        stagger losslessly even after the reader has destaggered.
        Only meaningful for the ``"staggered"`` convention; ``None``
        otherwise.  Each tuple is converted to a ``tuple[float, ...]``
        on construction so the whole structure is hashable.
    interpolation_order : int | None
        Order of interpolation used during destaggering (1 = linear,
        2 = quadratic).  ``None`` if no destaggering was performed.
    notes : str | None
        Free-text provenance (e.g. ``"Yee mesh, B on faces"``).

    Examples
    --------
    >>> si = StaggerInfo(convention="node")
    >>> si.convention
    'node'
    >>> si.field_locations is None
    True

    >>> si = StaggerInfo(
    ...     convention="staggered",
    ...     field_locations={"B": "face", "E": "edge"},
    ...     notes="Yee mesh",
    ... )
    >>> si.field_locations["B"]
    'face'

    >>> si = StaggerInfo(
    ...     convention="staggered",
    ...     position={"B1": (0.5, 0.0, 0.0), "E1": (0.0, 0.5, 0.5)},
    ... )
    >>> si.position["B1"]
    (0.5, 0.0, 0.0)
    """

    convention: str
    field_locations: dict[str, str] | None = None
    position: dict[str, tuple[float, ...]] | None = None
    interpolation_order: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.field_locations is not None:
            object.__setattr__(
                self,
                "field_locations",
                MappingProxyType(dict(self.field_locations)),
            )
        if self.position is not None:
            normalized: dict[str, tuple[float, ...]] = {}
            for name, offsets in self.position.items():
                offset_tuple = tuple(float(x) for x in offsets)
                if not all(0.0 <= x < 1.0 for x in offset_tuple):
                    raise ValueError(
                        f"StaggerInfo.position[{name!r}] = {offset_tuple} — "
                        f"each offset must be in [0.0, 1.0)"
                    )
                normalized[name] = offset_tuple
            object.__setattr__(self, "position", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parsed simulation configuration from a TOML config file.

    Parameters
    ----------
    model_name : str
        Human-readable name for the simulation run.
    model_type : Literal["PIC", "MHD", "hybrid", "vlasov", "gyrokinetic"]
        Simulation type identifier — uppercase for fluid/PIC families,
        lowercase for kinetic continuum codes. Matches ``[model].type`` in
        :doc:`/docs/schema` and the Pydantic ``Model.type`` Literal. The
        ``vlasov`` / ``gyrokinetic`` values land in v1.0.x ahead of the
        readers that consume them — see ``TASKS-schema-extension.md``.
    grid : GridInfo
        Grid metadata (includes coordinate geometry).
    normalization : Normalization
        Unit system.
    species : tuple[SpeciesInfo, ...]
        Species definitions (tuple for immutability).
    physics : PhysicsParams
        Physics parameters (frozen dataclass).
    frame : str
        Reference frame label (e.g. ``"GSM"``, ``"simulation"``).
    initial_conditions : InitialConditions | None
        Validated ``[initial_conditions]`` object from the v1.0 schema,
        or ``None`` when the section is absent.
    output : Output | None
        Validated ``[output]`` umbrella object from the v1.0 schema, or
        ``None`` when the section is absent.
    run : Run | None
        Validated ``[run]`` provenance object — authors, DOI, license,
        funding, embargo, ensemble, resource accounting. ``None`` only
        when this config was assembled by hand outside the schema path.
    probes : tuple[Probe, ...]
        Validated ``[[probes]]`` entries; empty when no probes declared.
    collisions : tuple[Collision, ...]
        Validated ``[[collisions]]`` entries; empty when no collision
        pairs declared.
    phase_space : PhaseSpace | None
        Validated ``[phase_space]`` block for >3D kinetic codes.
        Continuum-Vlasov sparse-block storage knobs live under
        ``phase_space.storage``.
    metadata : dict[str, Any]
        Free-form annotations from readers (stagger, scaling, version,
        description, ...). Schema-typed sections live on dedicated
        attributes above rather than as opaque dict entries here.

    Examples
    --------
    >>> from pypic.units import Normalization, SpeciesInfo
    >>> cfg = SimulationConfig(
    ...     model_name="test", model_type="PIC",
    ...     grid=GridInfo(
    ...         dimensions=(4,), spacing=(1.0,), origin=(0.0,),
    ...         geometry=CARTESIAN,
    ...     ),
    ...     normalization=Normalization.identity(),
    ...     species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
    ... )
    >>> cfg.model_name
    'test'
    """

    model_name: str
    model_type: ModelType
    grid: GridInfo
    normalization: Normalization
    species: tuple[SpeciesInfo, ...] = ()
    physics: PhysicsParams = field(default_factory=lambda: PhysicsParams())
    frame: str = "simulation"
    transforms: dict[str, FrameTransform] = field(default_factory=dict)
    initial_conditions: InitialConditions | None = None
    output: Output | None = None
    bodies: tuple[Body, ...] = ()
    drivers: tuple[Driver, ...] = ()
    restart: Restart | None = None
    run: Run | None = None
    probes: tuple[Probe, ...] = ()
    collisions: tuple[Collision, ...] = ()
    phase_space: PhaseSpace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # frozen via __post_init__

    def __post_init__(self) -> None:
        # Wrap mutable dicts in read-only proxies to enforce true immutability.
        # Callers pass plain dicts; frozen assignment uses object.__setattr__.
        object.__setattr__(
            self,
            "transforms",
            MappingProxyType(dict(self.transforms)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class TabularData:
    r"""Generic columnar container for auxiliary time-series data.

    Stores named 1-D arrays sharing a common length, with optional
    index column designation.  Used for conserved quantities, solver
    diagnostics, virtual satellite probes, etc.

    Parameters
    ----------
    name : str
        Dataset label (e.g. ``"conserved_quantities"``).
    columns : dict[str, FloatArray]
        Column name → 1-D array mapping.  All arrays must have
        the same length.
    index_column : str | None
        Which column serves as the index (e.g. ``"cycle"``).
        ``None`` means row-indexed.
    metadata : dict[str, Any]
        Source info (reader name, file path, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> tab = TabularData(
    ...     name="diagnostics",
    ...     columns={"cycle": np.array([0.0, 1.0, 2.0]),
    ...              "energy": np.array([1.0, 0.9, 0.8])},
    ...     index_column="cycle",
    ... )
    >>> tab["energy"]
    array([1. , 0.9, 0.8])
    >>> len(tab)
    3
    >>> "cycle" in tab
    True
    >>> tab.column_names
    ['cycle', 'energy']
    """

    name: str
    columns: dict[str, FloatArray]  # frozen at runtime via __post_init__
    index_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # frozen at runtime

    def __post_init__(self) -> None:
        # Validate before freezing
        if self.columns:
            lengths = {k: len(v) for k, v in self.columns.items()}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                msg = f"All columns must have equal length, got {lengths}"
                raise ValueError(msg)
        if self.index_column is not None and self.index_column not in self.columns:
            msg = (
                f"index_column {self.index_column!r} not found "
                f"in columns: {sorted(self.columns)}"
            )
            raise ValueError(msg)
        # Wrap mutable dicts in read-only proxies
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __getitem__(self, key: str) -> FloatArray:
        """Return a column by name.

        Parameters
        ----------
        key : str
            Column name.

        Returns
        -------
        FloatArray

        Raises
        ------
        KeyError
            If *key* is not a column name.
        """
        try:
            return self.columns[key]
        except KeyError:
            msg = f"Column {key!r} not found. Available: {sorted(self.columns)}"
            raise KeyError(msg) from None

    def __contains__(self, key: object) -> bool:
        """Check whether *key* is a column name."""
        return key in self.columns

    def __len__(self) -> int:
        """Return the number of rows (common array length)."""
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @property
    def column_names(self) -> list[str]:
        """Sorted list of column names."""
        return sorted(self.columns)

    @property
    def index(self) -> FloatArray:
        """Index array: the designated index column, or ``np.arange(len)``."""
        if self.index_column is not None:
            return self.columns[self.index_column]
        return np.arange(len(self), dtype=np.float64)


def _check_particle_shape(
    arr: np.ndarray | None, name: str, expected: tuple[int, ...]
) -> None:
    """Shape validator for per-particle arrays in ``ParticleData``."""
    if arr is None or arr.shape == expected:
        return
    n = expected[0]
    if len(expected) == 2:
        pretty = f"(n_particles, {expected[1]}) = ({n}, {expected[1]})"
    else:
        pretty = f"n_particles ({n},)"
    msg = f"{name} shape {arr.shape} does not match {pretty}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ParticleData:
    r"""Container for particle data from a single species at one timestep.

    Canonical per-macroparticle representation: every macroparticle carries
    a ``weight`` (number of physical particles it represents); the species
    as a whole carries scalar ``species_charge`` and ``species_mass``. The
    per-macroparticle charge and mass that enter moments and the equations
    of motion are derived on demand via :attr:`macro_charge` and
    :attr:`macro_mass`. This shape is code-agnostic — readers for combined-
    storage codes (iPIC3D, OSIRIS) split the native ``q_s × w`` column into
    ``weight`` + scalars on load; separate-storage codes (VPIC, WarpX,
    Smilei, PIConGPU, TRISTAN-MP) populate the same fields directly.

    Parameters
    ----------
    species_index : int
        Zero-based species index.
    species_name : str
        Human-readable species name (e.g. ``"electrons"``).
    position : FloatArray | None
        Particle positions, shape ``(N, 3)``. ``None`` if not loaded.
    velocity : FloatArray | None
        Particle velocities, shape ``(N, 3)``. ``None`` if not loaded.
    n_particles : int
        Total particle count.
    id : np.ndarray | None
        Integer particle tracking IDs, shape ``(N,)``. ``None`` if not
        available or not requested.
    weight : FloatArray | None
        Per-macroparticle weight $w$ (number of physical particles per
        macroparticle), shape ``(N,)``, float64. Required for
        :attr:`macro_charge` and :attr:`macro_mass`.
    species_charge : float | None
        Scalar species charge $q_s$ in code units (e.g. ``-1.0`` for
        electrons in iPIC3D normalization).
    species_mass : float | None
        Scalar species mass $m_s$ in code units.
    metadata : dict[str, Any]
        Source info (file path, format, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> pcl = ParticleData(
    ...     species_index=0, species_name="electrons",
    ...     position=np.zeros((10, 3)),
    ...     velocity=np.ones((10, 3)),
    ...     n_particles=10, metadata={},
    ...     weight=np.ones(10), species_charge=-1.0, species_mass=1.0,
    ... )
    >>> pcl.x.shape
    (10,)
    >>> len(pcl)
    10
    """

    species_index: int
    species_name: str
    position: FloatArray | None
    velocity: FloatArray | None
    n_particles: int
    metadata: dict[str, Any]  # frozen at runtime via __post_init__
    id: np.ndarray | None = None
    weight: FloatArray | None = None
    species_charge: float | None = None
    species_mass: float | None = None

    def __post_init__(self) -> None:
        if self.position is None and self.velocity is None:
            msg = "At least one of position or velocity must be provided"
            raise ValueError(msg)
        n = self.n_particles
        _check_particle_shape(self.position, "position", (n, 3))
        _check_particle_shape(self.velocity, "velocity", (n, 3))
        _check_particle_shape(self.id, "id", (n,))
        _check_particle_shape(self.weight, "weight", (n,))
        if self.weight is not None and self.weight.dtype != np.float64:
            msg = f"weight must be float64, got {self.weight.dtype}"
            raise ValueError(msg)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def _component(self, attr: str, idx: int) -> FloatArray:
        """Return one column of *attr*, raising if it was not loaded."""
        arr: FloatArray | None = getattr(self, attr)
        if arr is None:
            msg = f"{attr} was not loaded"
            raise ValueError(msg)
        return arr[:, idx]

    @property
    def x(self) -> FloatArray:
        """X positions (view into ``position[:, 0]``)."""
        return self._component("position", 0)

    @property
    def y(self) -> FloatArray:
        """Y positions (view into ``position[:, 1]``)."""
        return self._component("position", 1)

    @property
    def z(self) -> FloatArray:
        """Z positions (view into ``position[:, 2]``)."""
        return self._component("position", 2)

    @property
    def vx(self) -> FloatArray:
        """X velocities (view into ``velocity[:, 0]``)."""
        return self._component("velocity", 0)

    @property
    def vy(self) -> FloatArray:
        """Y velocities (view into ``velocity[:, 1]``)."""
        return self._component("velocity", 1)

    @property
    def vz(self) -> FloatArray:
        """Z velocities (view into ``velocity[:, 2]``)."""
        return self._component("velocity", 2)

    @property
    def macro_charge(self) -> FloatArray:
        r"""Per-macroparticle charge $q_s w$."""
        if self.species_charge is None or self.weight is None:
            msg = "Cannot compute macro_charge: need both 'species_charge' and 'weight'"
            raise ValueError(msg)
        return self.species_charge * self.weight

    @property
    def macro_mass(self) -> FloatArray:
        r"""Per-macroparticle mass $m_s w$."""
        if self.species_mass is None or self.weight is None:
            msg = "Cannot compute macro_mass: need both 'species_mass' and 'weight'"
            raise ValueError(msg)
        return self.species_mass * self.weight

    def __len__(self) -> int:
        return self.n_particles

    def __repr__(self) -> str:
        loaded = []
        if self.position is not None:
            loaded.append("position")
        if self.velocity is not None:
            loaded.append("velocity")
        if self.id is not None:
            loaded.append("id")
        if self.weight is not None:
            loaded.append("weight")
        if self.species_charge is not None:
            loaded.append(f"species_charge={self.species_charge:g}")
        if self.species_mass is not None:
            loaded.append(f"species_mass={self.species_mass:g}")
        return (
            f"ParticleData({self.species_name!r}, "
            f"n={self.n_particles:,}, "
            f"loaded=[{', '.join(loaded)}])"
        )
