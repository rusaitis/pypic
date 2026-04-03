"""Reader registry for auto-detection and unified simulation opening."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from pypic.readers.base import (
        FieldDataset,
        GridInfo,
        ParticleData,
        SimulationConfig,
        SimulationReader,
        TabularData,
    )
    from pypic.units import Normalization, SpeciesInfo

    type CanReadFunction = Callable[[Path], float]
    type ReaderFactory = Callable[..., tuple[SimulationReader, SimulationConfig]]

log = logging.getLogger(__name__)

_lock = threading.Lock()


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

    If *name* is already registered, the existing entry is overwritten
    and a warning is logged.

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
    """
    with _lock:
        if name in _REGISTRY:
            log.warning("Overwriting existing reader %r", name)
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

    Supports tuple unpacking for backwards compatibility::

        sim = open_simulation(path)            # preferred
        reader, config = open_simulation(path) # still works

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
    >>> from pypic.readers.base import SimulationConfig, GridInfo
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
    def physics(self) -> MappingProxyType[str, Any]:
        """Physics parameters (read-only view)."""
        return self._config.physics  # type: ignore[return-value]  # MappingProxyType at runtime

    @property
    def shrink_factor(self) -> float:
        """Domain shrink factor (1.0 when unscaled)."""
        return self._config.metadata.get("scaling", {}).get("shrink_factor", 1.0)

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

    @staticmethod
    def _expanded_names(
        name: str, alias_map: dict[str, str], canonical: set[str]
    ) -> set[str]:
        """Return the canonical names that *name* could have expanded to."""
        from pypic._aliases import _COMPUTE_ALIASES

        resolved = alias_map.get(name, _COMPUTE_ALIASES.get(name, name))
        candidates = {resolved}
        # Vector expansion
        import re

        _sp = re.match(r"^(.+?)(_s\d+)$", resolved)
        prefix = _sp.group(1) if _sp else resolved
        species = _sp.group(2) if _sp else ""
        if not prefix[-1:].isdigit():
            for c in ("1", "2", "3"):
                candidates.add(f"{prefix}{c}{species}")
        # Compute deps
        from pypic.compute import field_dependencies

        candidates |= field_dependencies(resolved)
        return candidates & canonical

    def read(
        self,
        step: int,
        *,
        fields: Iterable[str] | None = None,
        **kwargs: Any,  # noqa: ANN401 — reader-specific params (e.g. target_resolution)
    ) -> FieldDataset:
        """Read field data for a single timestep.

        Parameters
        ----------
        step : int
            Timestep index.
        fields : Iterable[str] | None
            When given, only these fields are read.  Accepts canonical
            names (``"B1"``) and geometry aliases (``"Bx"``).  Readers
            that support selective I/O skip unwanted datasets; others
            read all fields then filter.
        **kwargs
            Forwarded to readers that accept extra parameters
            (e.g. ``target_resolution`` for BATSRUS).

        Returns
        -------
        FieldDataset
        """
        from pypic.readers.base import _default_aliases, supports_selective_read

        if fields is None and not kwargs:
            return self._reader.read_timestep(self._path, step)

        canonical: set[str] | None = None
        if fields is not None:
            import re

            from pypic._aliases import _COMPUTE_ALIASES
            from pypic.compute import field_dependencies

            alias_map = _default_aliases(self._config.grid.geometry)
            expanded: set[str] = set()
            for name in fields:
                # Resolve geometry aliases (Bx→B1) and compute aliases (EFe→EF_s0)
                resolved = alias_map.get(name, _COMPUTE_ALIASES.get(name, name))
                expanded.add(resolved)
                # Expand vector group shorthand:
                #   "B"    → "B1","B2","B3"
                #   "J_s0" → "J1_s0","J2_s0","J3_s0" (component before species)
                # Skip names already resolved as aliases (e.g. "Bx") and
                # names whose prefix already ends in a digit (e.g. "P11_s1").
                if name not in alias_map:
                    _sp = re.match(r"^(.+?)(_s\d+)$", resolved)
                    prefix = _sp.group(1) if _sp else resolved
                    species = _sp.group(2) if _sp else ""
                    if not prefix[-1:].isdigit():
                        for c in ("1", "2", "3"):
                            expanded.add(f"{prefix}{c}{species}")
                # Expand compute dependencies: "Pi" → "P11_s1","P22_s1","P33_s1"
                expanded |= field_dependencies(resolved)

            # If any diagonal tensor components were requested (P11, P22, P33
            # or P11_sN etc.), also include the off-diagonals so that
            # P_par/P_perp/agyrotropy can be computed from the same load.
            _diag_re = re.compile(r"^P(11|22|33)(_s\d+)?$")
            suffixes: set[str] = set()
            for f in list(expanded):
                m = _diag_re.match(f)
                if m:
                    suffixes.add(m.group(2) or "")
            for s in suffixes:
                for ij in ("11", "12", "13", "22", "23", "33"):
                    expanded.add(f"P{ij}{s}")

            canonical = expanded

        if supports_selective_read(self._reader):
            ds = self._reader.read_timestep(  # type: ignore[call-arg]
                self._path,
                step,
                fields=canonical,
                **kwargs,
            )
        else:
            ds = self._reader.read_timestep(self._path, step)
            if canonical is not None:
                ds = ds.select_fields(canonical)

        # Warn for any user-requested name that yielded no loaded fields
        if fields is not None:
            loaded = set(ds.field_names())
            for name in fields:
                # A request is satisfied if any expansion of it was loaded
                if not loaded & self._expanded_names(name, alias_map, canonical or set()):
                    log.warning(
                        "fields=%r: %r matched no fields in the dataset",
                        list(fields), name,
                    )

        return ds

    @property
    def auxiliary_names(self) -> list[str]:
        """Names of available auxiliary datasets, or ``[]`` if unsupported."""
        from pypic.readers.base import AuxiliaryDataReader

        if isinstance(self._reader, AuxiliaryDataReader):
            return self._reader.available_auxiliary(self._path)
        return []

    @property
    def particle_steps(self) -> list[int]:
        """Timesteps with particle data, or ``[]`` if unsupported."""
        from pypic.readers.base import ParticleDataReader

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
            ``None`` loads all.  ``charge`` is always loaded.

        Returns
        -------
        ParticleData

        Raises
        ------
        TypeError
            If the reader does not support particle data.
        """
        from pypic.readers.base import ParticleDataReader

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
        from pypic.readers.base import AuxiliaryDataReader

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

    Also supports tuple unpacking for backwards compatibility::

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

    # Auto-detect: probe all readers, try factories in descending confidence
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
                result[0], _maybe_apply_extent(result[1]), path,
                probe_results=frozen_probes,
            )
        except Exception as exc:
            log.warning("Reader %r (confidence=%.2f) failed: %s", name, confidence, exc)
            errors.append(exc)

    raise ExceptionGroup(
        f"All candidate readers failed for {path}",
        errors,
    )
