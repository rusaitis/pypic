"""Reader registry for auto-detection and unified simulation opening."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pypic._aliases import COMPUTE_ALIASES, GROUP_ALIASES, _default_aliases
from pypic.compute import field_dependencies
from pypic.exceptions import UnknownFieldError
from pypic.readers._protocols import supports_selective_read

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from pypic.containers import ParticleData, SimulationConfig, TabularData
    from pypic.dataset import FieldDataset
    from pypic.grid import GridInfo
    from pypic.readers._protocols import SimulationReader
    from pypic.units import Normalization, PhysicsParams, SpeciesInfo

    type CanReadFunction = Callable[[Path], float]
    type ReaderFactory = Callable[..., tuple[SimulationReader, SimulationConfig]]

log = logging.getLogger(__name__)

_lock = threading.Lock()


def _expand_requested(name: str, alias_map: dict[str, str]) -> set[str]:
    """Return the canonical names one requested field name may load.

    Resolves geometry aliases (``Bx`` → ``B_1``), then compute aliases
    (``energy_flux_x`` → ``EF_1``), then vector-group aliases (``EFe`` →
    ``EF_s0``); adds the three Tier-3 components for vector prefixes
    (``B`` → ``B_1``..``B_3``, ``J_s0`` → ``J_s0_1``..``J_s0_3``) and every
    compute dependency (``P_par`` → the six tensor components plus B).
    Names already resolved as aliases or ending in a component suffix
    (``B_1``, ``P_s0_11``) stay scalar.
    """
    resolved = alias_map.get(
        name, COMPUTE_ALIASES.get(name, GROUP_ALIASES.get(name, name))
    )
    expanded = {resolved}
    ends_with_species = re.search(r"_s\d+$", resolved) is not None
    ends_with_component = (
        re.search(r"_\d+$", resolved) is not None and not ends_with_species
    )
    if name not in alias_map and not ends_with_component:
        expanded.update(f"{resolved}_{c}" for c in "123")
    return expanded | field_dependencies(resolved)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Result of a single reader's format probe.

    Parameters
    ----------
    name : str
        Reader name.
    confidence : float
        Confidence score in ``[0.0, 1.0]``.
    error : str | None
        Error message if the probe or factory raised, else ``None``.
    """

    name: str
    confidence: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReaderEntry:
    """A registered reader with its format detector and factory.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"ipic3d"``, ``"batsrus"``).
    can_read_confidence : CanReadFunction
        Returns a confidence score in ``[0.0, 1.0]`` that *path*
        contains data readable by this reader.
    factory : ReaderFactory
        Callable that opens a simulation directory.
    """

    name: str
    can_read_confidence: CanReadFunction
    factory: ReaderFactory


_REGISTRY: dict[str, ReaderEntry] = {}


def register_reader(
    name: str,
    can_read_confidence: CanReadFunction,
    factory: ReaderFactory,
) -> None:
    """Register a reader for auto-detection.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"ipic3d"``).
    can_read_confidence : CanReadFunction
        Returns confidence in ``[0.0, 1.0]`` that *path* contains
        data readable by this reader.  Must be lightweight
        (filesystem glob only, no actual I/O).
    factory : ReaderFactory
        ``factory(path, **kwargs) -> (reader, config)``.

    Raises
    ------
    ValueError
        If *name* is already registered; call `unregister_reader` first.
    """
    with _lock:
        if name in _REGISTRY:
            msg = f"Reader {name!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[name] = ReaderEntry(
            name=name,
            can_read_confidence=can_read_confidence,
            factory=factory,
        )


def unregister_reader(name: str) -> None:
    """Remove a reader from the registry.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    with _lock:
        try:
            del _REGISTRY[name]
        except KeyError:
            msg = f"No reader registered with name {name!r}"
            raise KeyError(msg) from None


def registered_readers() -> MappingProxyType[str, ReaderEntry]:
    """Return a read-only view of all registered readers."""
    return MappingProxyType(_REGISTRY)


class Simulation:
    """Ergonomic wrapper around a simulation reader, config, and path.

    Returned by `open_simulation`.  Provides direct access to config
    properties and reads timesteps without repeating the data path.

    Also unpacks as a ``(reader, config)`` tuple::

        sim = open_simulation(path)            # preferred
        reader, config = open_simulation(path) # still works

    **Cross-model comparison workflow:**

    1. Open both simulations via ``open_simulation()``.
    2. Read matching timesteps from each.
    3. Regrid to a common grid via [`align_grids`][pypic.regrid.align_grids]
       when the two runs do not already share one.
    4. Compare fields: use ``in_si()`` for cross-model comparison
       (different normalizations make code units incomparable), or
       compare in code units for same-model parameter studies (identical
       normalization). Dimensionless quantities (beta, Mach number,
       entropy) need no conversion.

    Known limitations: no automatic timestep alignment across simulations
    (different codes use different step numbering and output cadences).

    Parameters
    ----------
    reader : SimulationReader
        The underlying reader instance.
    config : SimulationConfig
        Parsed simulation metadata.
    path : Path
        Data directory (remembered for ``read`` / ``steps``).

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> r = MagicMock()
    >>> r.available_timesteps.return_value = [0, 10]
    >>> from pypic.containers import SimulationConfig
    >>> from pypic.grid import GridInfo
    >>> from pypic.coordinates.geometry import CARTESIAN
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
    >>> sim = Simulation(r, cfg, path="/tmp")
    >>> sim.model_name
    'test'
    >>> sim.steps
    [0, 10]
    """

    def __init__(
        self,
        reader: SimulationReader,
        config: SimulationConfig,
        path: Path | str,
        *,
        probe_results: tuple[ProbeResult, ...] | None = None,
    ) -> None:
        self._reader = reader
        self._config = config
        self._path = Path(path)
        self._steps: list[int] | None = None
        self._probe_results = probe_results

    @property
    def reader(self) -> SimulationReader:
        """The underlying reader instance."""
        return self._reader

    @property
    def config(self) -> SimulationConfig:
        """Full simulation configuration."""
        return self._config

    @property
    def path(self) -> Path:
        """Data directory."""
        return self._path

    @property
    def model_name(self) -> str:
        """Simulation code name (e.g. ``"iPIC3D"``)."""
        return self._config.model_name

    @property
    def model_type(self) -> str:
        """Model type (e.g. ``"PIC"``, ``"MHD"``)."""
        return self._config.model_type

    @property
    def grid(self) -> GridInfo:
        """Grid metadata."""
        return self._config.grid

    @property
    def normalization(self) -> Normalization:
        """Unit normalization."""
        return self._config.normalization

    @property
    def species(self) -> tuple[SpeciesInfo, ...]:
        """Species definitions."""
        return self._config.species

    @property
    def physics(self) -> PhysicsParams:
        """Physics parameters."""
        return self._config.physics

    @property
    def shrink_factor(self) -> float:
        """Domain shrink factor (1.0 when unscaled)."""
        scaling = self._config.metadata.get("scaling", {})
        return float(scaling.get("shrink_factor", 1.0))

    @property
    def probe_results(self) -> tuple[ProbeResult, ...] | None:
        """Auto-detection probe results, or ``None`` if reader was explicit."""
        return self._probe_results

    @property
    def steps(self) -> list[int]:
        """Available timestep indices (cached after first access)."""
        if self._steps is None:
            self._steps = self._reader.available_timesteps(self._path)
        return self._steps

    def refresh_steps(self) -> list[int]:
        """Re-scan for available timesteps, clearing the cache."""
        self._steps = None
        return self.steps

    @property
    def first_step(self) -> int:
        """First available timestep index."""
        return self.steps[0]

    @property
    def last_step(self) -> int:
        """Last available timestep index."""
        return self.steps[-1]

    def read(
        self,
        step: int,
        *,
        fields: Iterable[str] | None = None,
        strict_fields: bool = True,
        **kwargs: Any,  # noqa: ANN401 — reader-specific params (e.g. target_resolution)
    ) -> FieldDataset:
        """Read field data for a single timestep.

        Parameters
        ----------
        step : int
            Timestep index.
        fields : Iterable[str] | None
            When given, only these fields are read.  Accepts canonical
            names (``"B_1"``) and geometry aliases (``"Bx"``).  Readers
            that support selective I/O skip unwanted datasets; others
            read all fields then filter.
        strict_fields : bool
            When ``True`` (default), raise
            [`UnknownFieldError`][pypic.exceptions.UnknownFieldError] if any name in
            *fields* matches no loaded field — the project's fail-loud
            rule for selection APIs.  Pass ``False`` only for
            exploratory scripts where some requested names are
            optional; missing names are then logged as warnings.
        **kwargs
            Forwarded to readers that accept extra parameters
            (e.g. ``target_resolution`` for BATSRUS).

        Returns
        -------
        FieldDataset

        Raises
        ------
        KeyError
            If *strict_fields* is true (the default) and any requested
            field name yielded nothing.
        """
        if fields is None and not kwargs:
            return self._attach_config_provenance(
                self._reader.read_timestep(self._path, step)
            )

        canonical: set[str] | None = None
        alias_map: dict[str, str] = {}
        if fields is not None:
            alias_map = _default_aliases(self._config.grid.geometry)
            canonical = set()
            for name in fields:
                canonical |= _expand_requested(name, alias_map)

        if supports_selective_read(self._reader):
            ds = self._reader.read_timestep(  # type: ignore[call-arg]
                self._path,
                step,
                fields=canonical,
                **kwargs,
            )
        else:
            if kwargs:
                msg = (
                    f"{type(self._reader).__name__} takes no read options; "
                    f"got {sorted(kwargs)}"
                )
                raise TypeError(msg)
            ds = self._reader.read_timestep(self._path, step)
            if canonical is not None:
                # Filter to names actually present before narrowing the
                # dataset; the post-read check below reports unmatched
                # names uniformly for both reader paths.
                available = set(ds.field_names())
                ds = ds.select_fields(canonical & available)

        # Warn (or raise, if strict_fields) for any user-requested name
        # that yielded no loaded fields.
        if fields is not None:
            loaded = set(ds.field_names())
            missing = [
                name
                for name in fields
                if not loaded & _expand_requested(name, alias_map)
            ]
            if missing:
                if strict_fields:
                    msg = (
                        f"fields={list(fields)!r}: "
                        f"{missing!r} matched no fields in the dataset. "
                        f"Available: {sorted(loaded)!r}"
                    )
                    raise UnknownFieldError(msg)
                for name in missing:
                    log.warning(
                        "fields=%r: %r matched no fields in the dataset",
                        list(fields),
                        name,
                    )

        return self._attach_config_provenance(ds)

    def _attach_config_provenance(self, fds: FieldDataset) -> FieldDataset:
        """Stamp ``[run]`` and verbatim ``simulation.toml`` onto ``fds.metadata``.

        Lifted to top-level ``attrs.run`` / ``attrs.simulation_toml`` at
        Zarr write time by ``encode_pypic_attrs`` (schema.md §4.2).
        No-op when the config carries neither (legacy/synthetic configs
        assembled outside the schema path).  Existing reader-supplied
        values win — readers may have stamped a code-specific run object
        that pypic shouldn't overwrite.
        """
        cfg = self._config
        raw_toml = cfg.metadata.get("simulation_toml")
        run = cfg.run
        if run is None and raw_toml is None:
            return fds
        new_meta = dict(fds.metadata)
        if run is not None:
            new_meta.setdefault("run", run)
        if raw_toml is not None:
            new_meta.setdefault("simulation_toml", raw_toml)
        # Internal mutation: read_timestep returns a fresh FieldDataset
        # each call, so the caller has no prior reference to invalidate.
        # Cheaper than a full reconstruction (alias resolution etc.).
        fds._metadata = new_meta
        return fds

    def available_fields(self, step: int) -> list[str]:
        """List canonical field names at *step* without loading arrays.

        Uses the reader's lightweight probe when available (via
        `FieldListingReader`);
        otherwise falls back to a full `read` and extracts
        [`field_names`][pypic.dataset.FieldDataset.field_names].

        Parameters
        ----------
        step : int
            Timestep index.

        Returns
        -------
        list[str]
            Sorted canonical field names.
        """
        from pypic.readers._protocols import FieldListingReader

        if isinstance(self._reader, FieldListingReader):
            return self._reader.available_fields(self._path, step)
        return sorted(self.read(step).field_names())

    def available_fields_mapping(self, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Uses the reader's lightweight probe when available; otherwise
        falls back to `available_fields` with ``None`` for all
        native names (native mapping unknown without reader support).

        Parameters
        ----------
        step : int
            Timestep index.

        Returns
        -------
        dict[str, str | None]
            Canonical name → native name, or ``None`` for computed
            fields or when the reader has no mapping support.
        """
        from pypic.readers._protocols import FieldListingReader

        if isinstance(self._reader, FieldListingReader):
            return self._reader.available_fields_mapping(self._path, step)
        return dict.fromkeys(self.available_fields(step))

    @property
    def auxiliary_names(self) -> list[str]:
        """Names of available auxiliary datasets, or ``[]`` if unsupported."""
        from pypic.readers._protocols import AuxiliaryDataReader

        if isinstance(self._reader, AuxiliaryDataReader):
            return self._reader.available_auxiliary(self._path)
        return []

    @property
    def particle_steps(self) -> list[int]:
        """Timesteps with particle data, or ``[]`` if unsupported."""
        from pypic.readers._protocols import ParticleDataReader

        if isinstance(self._reader, ParticleDataReader):
            return self._reader.available_particle_steps(self._path)
        return []

    def particles(
        self,
        step: int,
        species: int,
        *,
        columns: Iterable[str] | None = None,
    ) -> ParticleData:
        """Load particle data for a species at a timestep.

        Parameters
        ----------
        step : int
            Timestep index.
        species : int
            Zero-based species index.
        columns : Iterable[str] | None
            Subset of ``{"position", "velocity"}`` to load.
            ``None`` loads all.  Per-particle ``weight`` and the scalar
            ``species_charge``/``species_mass`` are always populated
            (canonical layout, ``docs/schema.md``).

        Returns
        -------
        ParticleData

        Raises
        ------
        TypeError
            If the reader does not support particle data.
        UnknownFieldError
            If *columns* names a column the reader does not recognize.
        """
        from pypic.readers._protocols import ParticleDataReader

        if isinstance(self._reader, ParticleDataReader):
            return self._reader.read_particles(
                self._path, step, species, columns=columns
            )
        msg = (
            f"Reader {type(self._reader).__name__!r} does not support "
            f"particle data (missing ParticleDataReader protocol)"
        )
        raise TypeError(msg)

    def auxiliary(self, name: str) -> TabularData:
        """Load a named auxiliary dataset.

        Parameters
        ----------
        name : str
            Dataset name (e.g. ``"conserved_quantities"``).

        Returns
        -------
        TabularData

        Raises
        ------
        TypeError
            If the reader does not support auxiliary data.
        """
        from pypic.readers._protocols import AuxiliaryDataReader

        if isinstance(self._reader, AuxiliaryDataReader):
            return self._reader.load_auxiliary(self._path, name)
        msg = (
            f"Reader {type(self._reader).__name__!r} does not support "
            f"auxiliary data (missing AuxiliaryDataReader protocol)"
        )
        raise TypeError(msg)

    def describe(self) -> str:
        """Multi-line summary of the simulation (no I/O).

        Returns
        -------
        str
        """
        dims = " x ".join(str(d) for d in self.grid.dimensions)
        spacing = " x ".join(f"{s:.2f}" for s in self.grid.spacing)
        geom = self.grid.geometry.type.value
        lines = [
            f"Simulation: {self.model_name} ({self.model_type})",
            f"  Path:    {self._path}",
            f"  Grid:    {dims} ({geom})",
            f"  Spacing: {spacing}",
        ]
        if self._config.species:
            species_parts = []
            for sp in self._config.species:
                if sp.charge_to_mass is not None:
                    species_parts.append(f"{sp.name} (q/m={sp.charge_to_mass})")
                else:
                    species_parts.append(sp.name)
            lines.append(f"  Species: {', '.join(species_parts)}")
        if self._steps is not None:
            if self._steps:
                step_range = f"{self._steps[0]}..{self._steps[-1]}"
            else:
                step_range = "empty"
            lines.append(f"  Steps:   {len(self._steps)} [{step_range}]")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[SimulationReader | SimulationConfig]:
        """Support ``reader, config = open_simulation(path)``."""
        yield self._reader
        yield self._config

    def __repr__(self) -> str:
        dims = "x".join(str(d) for d in self.grid.dimensions)
        n_steps = len(self.steps) if self._steps is not None else "?"
        return (
            f"Simulation({self.model_name!r}, {self.model_type}, "
            f"grid={dims}, steps={n_steps})"
        )


def open_simulation(
    path: Path | str,
    *,
    reader: str | ReaderFactory | None = None,
    physical_extent: tuple[float, ...] | None = None,
    physical_extent_unit: str = "m",
    **kwargs: Any,  # noqa: ANN401 — reader-specific kwargs
) -> Simulation:
    """Open a simulation directory, auto-detecting the format.

    Returns a `Simulation` object that remembers the data path::

        sim = open_simulation(path)
        sim.model_name          # "iPIC3D"
        sim.grid.dimensions     # (128, 64, 64)
        sim.steps               # [0, 100, 200, ...]
        ds = sim.read(step=100)

    Also unpacks as a ``(reader, config)`` tuple::

        reader, config = open_simulation(path)

    Parameters
    ----------
    path : Path | str
        Simulation output directory (or file).
    reader : str | ReaderFactory | None
        How to select the reader:

        - ``None`` (default) — query all registered readers and pick
          the highest-confidence match.
        - ``str`` — look up a registered reader by name
          (e.g. ``"ipic3d"``).
        - callable — call it directly as
          ``reader(path, **kwargs) -> (reader, config)``.
    physical_extent : tuple[float, ...] | None
        Physical domain size per axis in *physical_extent_unit*.
        When provided, auto-computes the spatial scale factor for
        frame transforms. Overrides ``physical_extent`` from TOML.
    physical_extent_unit : str
        Length unit for *physical_extent* (default ``"m"``).
        Valid: ``"m"``, ``"km"``, ``"R_E"``, ``"AU"``, ``"R_S"``.
    **kwargs
        Forwarded to the reader factory (e.g. ``normalization=...``
        for OpenGGCM).

    Returns
    -------
    Simulation
        Wraps the reader, config, and path.

    Raises
    ------
    KeyError
        If *reader* is a string not found in the registry.
    FileNotFoundError
        If auto-detection finds no matching reader.
    ExceptionGroup
        If every candidate reader was tried and each one failed — the
        usual outcome for a corrupt or ambiguous directory. The group
        carries one sub-exception per candidate.
    """
    path = Path(path)

    def _maybe_apply_extent(config: SimulationConfig) -> SimulationConfig:
        if physical_extent is not None:
            from pypic.readers.config import apply_physical_extent

            return apply_physical_extent(config, physical_extent, physical_extent_unit)
        return config

    result: tuple[SimulationReader, SimulationConfig]

    # Explicit callable override
    if callable(reader) and not isinstance(reader, str):
        result = reader(path, **kwargs)
        return Simulation(result[0], _maybe_apply_extent(result[1]), path)

    # Explicit name lookup
    if isinstance(reader, str):
        entry = _REGISTRY.get(reader)
        if entry is None:
            available = sorted(_REGISTRY)
            msg = f"No reader registered with name {reader!r}. Available: {available}"
            raise KeyError(msg)
        result = entry.factory(path, **kwargs)
        return Simulation(result[0], _maybe_apply_extent(result[1]), path)

    # Auto-detect: sensors indicate multiple possible formats
    with _lock:
        registry_snapshot = dict(_REGISTRY)
    probe_results: list[ProbeResult] = []
    scores: list[tuple[float, str]] = []
    for name, entry in sorted(registry_snapshot.items()):
        try:
            confidence = entry.can_read_confidence(path)
        except Exception as exc:
            probe_results.append(ProbeResult(name, 0.0, f"{type(exc).__name__}: {exc}"))
            log.debug(
                "can_read_confidence %r raised, skipping",
                name,
                exc_info=True,
            )
            continue
        probe_results.append(ProbeResult(name, confidence))
        if confidence > 0.0:
            scores.append((confidence, name))

    frozen_probes = tuple(probe_results)

    if not scores:
        lines = [f"No registered reader recognized {path}.", "", "Probe results:"]
        for pr in sorted(frozen_probes, key=lambda p: p.name):
            detail = f"  {pr.name}: {pr.confidence:.2f}"
            if pr.error:
                detail += f"  ({pr.error})"
            lines.append(detail)
        raise FileNotFoundError("\n".join(lines))

    # Highest confidence wins; alphabetical tiebreak for determinism
    scores.sort(key=lambda pair: (-pair[0], pair[1]))

    errors: list[Exception] = []
    for confidence, name in scores:
        try:
            log.info(
                "Trying reader %r (confidence %.2f) for %s",
                name,
                confidence,
                path,
            )
            result = registry_snapshot[name].factory(path, **kwargs)
            return Simulation(
                result[0],
                _maybe_apply_extent(result[1]),
                path,
                probe_results=frozen_probes,
            )
        except Exception as exc:
            log.warning("Reader %r (confidence=%.2f) failed: %s", name, confidence, exc)
            errors.append(exc)

    raise ExceptionGroup(
        f"All candidate readers failed for {path}",
        errors,
    )
