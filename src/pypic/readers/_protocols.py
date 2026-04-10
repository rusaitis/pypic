"""Reader protocols and probe utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from pypic.readers._containers import ParticleData, TabularData
    from pypic.readers._field_dataset import FieldDataset


@runtime_checkable
class SimulationReader(Protocol):
    """Protocol for simulation-specific file readers.

    Any class with ``read_timestep`` and ``available_timesteps`` methods
    satisfies this protocol — no inheritance required.
    """

    def read_timestep(self, path: Path, step: int) -> FieldDataset:
        """Read field data for a single timestep."""
        ...

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted list of available timestep indices."""
        ...


@runtime_checkable
class ParticleDataReader(Protocol):
    """Opt-in protocol for readers that provide particle data.

    Readers implement this alongside ``SimulationReader`` to advertise
    and load per-species particle arrays (position, velocity, charge).
    """

    def available_particle_steps(self, path: Path) -> list[int]:
        """Return sorted timestep indices that have particle data."""
        ...

    def read_particles(
        self,
        path: Path,
        step: int,
        species: int,
        *,
        columns: Iterable[str] | None = None,
    ) -> ParticleData:
        """Load particle data for one species at one timestep.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep index.
        species : int
            Zero-based species index.
        columns : Iterable[str] | None
            Subset of ``{"position", "velocity"}`` to load.
            ``None`` loads all.  ``charge`` is always loaded.
        """
        ...


@runtime_checkable
class AuxiliaryDataReader(Protocol):
    """Opt-in protocol for readers that provide auxiliary tabular data.

    Readers implement this alongside ``SimulationReader`` to advertise
    and load non-field data (conserved quantities, diagnostics, probes).
    """

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of auxiliary datasets discoverable at *path*."""
        ...

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset from *path*."""
        ...


def supports_selective_read(reader: SimulationReader) -> bool:
    """Check whether *reader* accepts a ``fields`` keyword on ``read_timestep``.

    Inspects the method signature once at dispatch time.  This is more
    reliable than ``@runtime_checkable`` protocols (which only check
    method names, not parameter signatures) and clearer than calling
    ``inspect.signature`` inline at the call site.

    Examples
    --------
    >>> class Selective:
    ...     def read_timestep(self, path, step, *, fields=None): ...
    ...     def available_timesteps(self, path): return []
    >>> supports_selective_read(Selective())
    True
    >>> class Basic:
    ...     def read_timestep(self, path, step): ...
    ...     def available_timesteps(self, path): return []
    >>> supports_selective_read(Basic())
    False
    """
    import inspect

    sig = inspect.signature(reader.read_timestep)
    return "fields" in sig.parameters


def score_signals(path: Path, signals: Sequence[tuple[str, float]]) -> float:
    """Sum weights of glob patterns that match entries under *path*.

    For each ``(pattern, weight)`` pair the helper checks whether
    ``path.glob(pattern)`` yields at least one entry and, if so, adds
    ``weight`` to the running score. Intended for reader probe functions
    (``can_read_confidence``) so the glob-and-accumulate boilerplate does
    not get duplicated across every reader.

    Signals that require reading file contents, filtering matches by
    regex, or distinguishing files from directories should be evaluated
    by the caller and added on top of the returned score. The caller is
    responsible for any conditional logic beyond "pattern present → add
    weight".

    Parameters
    ----------
    path : Path
        Directory to scan. Non-directories return ``0.0`` immediately.
    signals : Sequence[tuple[str, float]]
        Pairs of ``(glob_pattern, weight)`` to test against *path*.

    Returns
    -------
    float
        Sum of matching weights, clamped to ``[0.0, 1.0]``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = Path(d)
    ...     (p / "config.toml").touch()
    ...     (p / "data.h5").touch()
    ...     score_signals(p, [("*.toml", 0.5), ("*.h5", 0.3), ("*.nc", 0.9)])
    0.8
    """
    if not path.is_dir():
        return 0.0
    score = 0.0
    for pattern, weight in signals:
        if next(path.glob(pattern), None) is not None:
            score += weight
    return min(score, 1.0)
