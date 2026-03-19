"""Reader registry for auto-detection and unified simulation opening."""

from __future__ import annotations

import inspect
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
    ...     physics={}, frame="sim", metadata={},
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
    ) -> None:
        self._reader = reader
        self._config = config
        self._path = Path(path)
        self._steps: list[int] | None = None

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
        return MappingProxyType(self._config.physics)

    @property
    def steps(self) -> list[int]:
        """Available timestep indices (cached after first access)."""
        if self._steps is None:
            self._steps = self._reader.available_timesteps(self._path)
        return self._steps

    def read(
        self,
        step: int,
        *,
        fields: Iterable[str] | None = None,
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

        Returns
        -------
        FieldDataset
        """
        if fields is None:
            return self._reader.read_timestep(self._path, step)

        from pypic.readers.base import _default_aliases

        alias_map = _default_aliases(self._config.grid.geometry)
        canonical: set[str] = {alias_map.get(name, name) for name in fields}

        sig = inspect.signature(self._reader.read_timestep)
        if "fields" in sig.parameters:
            return self._reader.read_timestep(  # type: ignore[call-arg]
                self._path,
                step,
                fields=canonical,
            )

        ds = self._reader.read_timestep(self._path, step)
        return ds.select_fields(canonical)

    @property
    def auxiliary_names(self) -> list[str]:
        """Names of available auxiliary datasets, or ``[]`` if unsupported."""
        from pypic.readers.base import AuxiliaryDataReader

        if isinstance(self._reader, AuxiliaryDataReader):
            return self._reader.available_auxiliary(self._path)
        return []

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

    result: tuple[SimulationReader, SimulationConfig]

    # Explicit callable override
    if callable(reader) and not isinstance(reader, str):
        result = reader(path, **kwargs)
        return Simulation(result[0], result[1], path)

    # Explicit name lookup
    if isinstance(reader, str):
        entry = _REGISTRY.get(reader)
        if entry is None:
            available = sorted(_REGISTRY)
            msg = f"No reader registered with name {reader!r}. Available: {available}"
            raise KeyError(msg)
        result = entry.factory(path, **kwargs)
        return Simulation(result[0], result[1], path)

    # Auto-detect: pick highest confidence
    scores: list[tuple[float, str]] = []
    for name, entry in sorted(_REGISTRY.items()):
        try:
            confidence = entry.can_read_confidence(path)
        except Exception:
            log.debug(
                "can_read_confidence %r raised, skipping",
                name,
                exc_info=True,
            )
            continue
        if confidence > 0.0:
            scores.append((confidence, name))

    if not scores:
        registered = sorted(_REGISTRY)
        msg = (
            f"No registered reader recognized {path}. Registered readers: {registered}"
        )
        raise FileNotFoundError(msg)

    # Highest confidence wins; alphabetical tiebreak for determinism
    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    best_confidence, best_name = scores[0]
    log.info(
        "Auto-detected reader %r (confidence %.2f) for %s",
        best_name,
        best_confidence,
        path,
    )
    result = _REGISTRY[best_name].factory(path, **kwargs)
    return Simulation(result[0], result[1], path)
