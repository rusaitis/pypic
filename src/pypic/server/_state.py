"""Process-wide simulation registry shared by HTTP and WebSocket routes.

Multiple readers / large outputs can be expensive to open repeatedly,
so the registry caches one `Simulation`
per simulation name.  Discovery is by directory listing under ``root``;
anything containing a ``simulation.toml`` qualifies.  Open succeeds
lazily on first access and is then memoized.

The registry is single-process; horizontal scaling would put a
shared cache (Redis, a forked-worker store) in front of it.  That's
explicitly out of the foundations scope.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from pypic.readers._registry import open_simulation
from pypic.server.exceptions import UnknownSimulationError

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.readers._registry import Simulation

__all__ = ["SimulationRegistry", "UnknownSimulationError"]


class SimulationRegistry:
    """Lazy, thread-safe cache of opened simulations under one root.

    Parameters
    ----------
    root : Path
        Directory whose immediate subdirectories are candidate
        simulations.  A subdirectory qualifies when it contains a file
        named ``simulation.toml`` (the v1.0 schema's canonical config
        file name).

    Notes
    -----
    The lock guards the cache dict only — `open_simulation`
    itself can run concurrently for different names.  For the same
    name, the first request opens and others wait, which is the
    right behavior given that opening is typically I/O-bound (file
    parsing, HDF5 metadata, ...) and idempotent.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[str, Simulation] = {}
        self._opening: dict[str, Lock] = {}
        self._lock = Lock()

    @property
    def root(self) -> Path:
        """The configured root directory."""
        return self._root

    def names(self) -> list[str]:
        """Return sorted simulation names discoverable under ``root``.

        A subdirectory of ``root`` is a simulation iff it contains a
        file named ``simulation.toml``.  Cheap directory listing —
        does not open the readers.
        """
        if not self._root.is_dir():
            return []
        out: list[str] = []
        for entry in self._root.iterdir():
            if entry.is_dir() and (entry / "simulation.toml").is_file():
                out.append(entry.name)
        return sorted(out)

    def get(self, name: str) -> Simulation:
        """Return the opened `Simulation` named *name*.

        Opens on the first call and caches for subsequent calls.
        Concurrent first-time opens for the same name serialize on
        a per-name lock so the work is done once.

        Raises
        ------
        UnknownSimulationError
            *name* is not present under ``root``, or its directory
            lacks a ``simulation.toml`` file. Subclass of
            `KeyError`.
        """
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            per_name = self._opening.setdefault(name, Lock())

        with per_name:
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            path = self._root / name
            if not (path.is_dir() and (path / "simulation.toml").is_file()):
                msg = (
                    f"No simulation named {name!r} under {self._root!s} "
                    "(directory missing or no simulation.toml)"
                )
                raise UnknownSimulationError(msg)
            try:
                sim = open_simulation(path)
            except (KeyError, FileNotFoundError) as exc:
                # Explicit-reader miss or auto-detect found no matching
                # format. From the server's vantage point these are both
                # "we couldn't open this simulation" — the underlying
                # cause is preserved via ``__cause__`` for debugging.
                msg = f"Could not open simulation {name!r}: {exc}"
                raise UnknownSimulationError(msg) from exc
            self._cache[name] = sim
            return sim

    def invalidate(self, name: str | None = None) -> None:
        """Drop cached simulations.

        With *name* set, evict that one entry; with *name* ``None``,
        clear the whole cache.  Useful for tests and for picking up
        new timesteps without a server restart (when paired with
        `Simulation.refresh_steps`, callers can avoid evicting
        the whole reader).
        """
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)
