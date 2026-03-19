"""Reader registry for auto-detection and unified simulation opening."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pypic.readers.base import SimulationConfig, SimulationReader

    type ProbeFunction = Callable[[Path], float]
    type ReaderFactory = Callable[..., tuple[SimulationReader, SimulationConfig]]

log = logging.getLogger(__name__)

_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class ReaderEntry:
    """A registered reader with its detection probe and factory.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"ipic3d"``, ``"batsrus"``).
    probe : ProbeFunction
        Callable that returns a confidence score in ``[0.0, 1.0]``.
    factory : ReaderFactory
        Callable that opens a simulation directory.
    """

    name: str
    probe: ProbeFunction
    factory: ReaderFactory


_REGISTRY: dict[str, ReaderEntry] = {}


def register_reader(
    name: str,
    probe: ProbeFunction,
    factory: ReaderFactory,
) -> None:
    """Register a reader for auto-detection.

    If *name* is already registered, the existing entry is overwritten
    and a warning is logged.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"ipic3d"``).
    probe : ProbeFunction
        Returns confidence in ``[0.0, 1.0]`` that *path* contains
        data readable by this reader.  Must be lightweight (filesystem
        glob only, no actual I/O).
    factory : ReaderFactory
        ``factory(path, **kwargs) -> (reader, config)``.
    """
    with _lock:
        if name in _REGISTRY:
            log.warning("Overwriting existing reader %r", name)
        _REGISTRY[name] = ReaderEntry(name=name, probe=probe, factory=factory)


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


def open_simulation(
    path: Path | str,
    *,
    reader: str | ReaderFactory | None = None,
    **kwargs: Any,  # noqa: ANN401 — reader-specific kwargs
) -> tuple[SimulationReader, SimulationConfig]:
    """Open a simulation directory, auto-detecting the format.

    Parameters
    ----------
    path : Path | str
        Simulation output directory (or file).
    reader : str | ReaderFactory | None
        How to select the reader:

        - ``None`` (default) — probe all registered readers and pick
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
    tuple[SimulationReader, SimulationConfig]
        A ``(reader, config)`` pair ready for
        ``reader.read_timestep(path, step)``.

    Raises
    ------
    KeyError
        If *reader* is a string not found in the registry.
    FileNotFoundError
        If auto-detection finds no matching reader.
    """
    from pathlib import Path as _Path

    path = _Path(path)

    # Explicit callable override
    if callable(reader) and not isinstance(reader, str):
        return reader(path, **kwargs)

    # Explicit name lookup
    if isinstance(reader, str):
        entry = _REGISTRY.get(reader)
        if entry is None:
            available = sorted(_REGISTRY)
            msg = f"No reader registered with name {reader!r}. Available: {available}"
            raise KeyError(msg)
        return entry.factory(path, **kwargs)

    # Auto-detect: run all probes, pick highest confidence
    scores: list[tuple[float, str]] = []
    for name, entry in sorted(_REGISTRY.items()):
        try:
            confidence = entry.probe(path)
        except Exception:
            log.debug("Probe %r raised an exception, skipping", name, exc_info=True)
            continue
        if confidence > 0.0:
            scores.append((confidence, name))

    if not scores:
        registered = sorted(_REGISTRY)
        msg = (
            f"No registered reader recognized {path}. "
            f"Registered readers: {registered}"
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
    return _REGISTRY[best_name].factory(path, **kwargs)
