"""Shared reader scaffolding: config-carrying base with one dataset exit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.dataset import FieldDataset

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pypic.containers import SimulationConfig, TabularData
    from pypic.grid import GridInfo
    from pypic.types import FloatArray


class ReaderBase:
    """Base class for readers that turn a run's files into `FieldDataset`.

    Subclasses implement ``available_timesteps``, ``read_timestep`` and
    ``available_fields_mapping``; everything else a `Simulation` may ask
    for has a default here. The contract every reader honours:

    - ``read_timestep`` builds its dataset through `_finish`, the one
      place arrays become a `FieldDataset`. It stamps ``metadata["step"]``
      and, when the file records one, ``metadata["time"]``, and takes
      normalization, species, physics, frame and transforms from the
      merged `SimulationConfig`, so a ``simulation.toml`` reaches the data.
    - Arrays handed to `_finish` are in code units on a co-located grid.
      Destaggering, when a reader gains it, happens before that call.
    - ``available_fields`` is the sorted key set of
      ``available_fields_mapping``: readers list, they do not load.

    Parameters
    ----------
    sim_config : SimulationConfig | None
        Merged run configuration. A reader that only learns its grid
        from each file (`SimpleReader`) passes ``None`` here and the
        resolved config to `_finish` instead.
    """

    def __init__(self, sim_config: SimulationConfig | None = None) -> None:
        self._sim_config = sim_config

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*."""
        raise NotImplementedError

    def available_fields(self, path: Path, step: int) -> list[str]:
        """Sorted canonical field names at *step*, without loading arrays."""
        return sorted(self.available_fields_mapping(path, step))

    def available_auxiliary(self, path: Path) -> list[str]:
        """Names of auxiliary datasets at *path*; none unless overridden."""
        return []

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset; raises unless overridden."""
        msg = f"No auxiliary dataset {name!r}"
        raise KeyError(msg)

    def _require_config(self) -> SimulationConfig:
        """Return the merged config; readers that resolve it per file pass theirs."""
        if self._sim_config is None:
            msg = f"{type(self).__name__} has no SimulationConfig to build from"
            raise ValueError(msg)
        return self._sim_config

    def _finish(
        self,
        fields: Mapping[str, FloatArray],
        *,
        step: int,
        time: float | None = None,
        grid: GridInfo | None = None,
        coords: Mapping[str, FloatArray] | None = None,
        extra: Mapping[str, Any] | None = None,
        config: SimulationConfig | None = None,
    ) -> FieldDataset:
        """Wrap code-unit *fields* in a dataset carrying the run's config.

        Parameters
        ----------
        fields : Mapping[str, FloatArray]
            Canonical-named arrays in code units.
        step : int
            Timestep index; always stamped into ``metadata``.
        time : float | None
            Snapshot time in code units when the file records one.
            `FieldDataset.time` derives ``step * grid.dt`` otherwise.
        grid : GridInfo | None
            Grid the arrays live on; defaults to the config's. Readers
            that rebuild the grid per read (AMR regridding) pass it.
        coords : Mapping[str, FloatArray] | None
            True coordinate arrays for non-uniform meshes, per axis name.
        extra : Mapping[str, Any] | None
            Reader-specific metadata (format, stagger provenance, ...).
            Config metadata comes first, then *extra*, then step and time.
        config : SimulationConfig | None
            Overrides the config given at construction, for readers
            that resolve it per file.
        """
        sc = config if config is not None else self._require_config()
        metadata: dict[str, Any] = {**sc.metadata, **(extra or {}), "step": step}
        if time is not None:
            metadata["time"] = time
        return FieldDataset.from_arrays(
            fields,
            grid if grid is not None else sc.grid,
            sc.normalization,
            species=sc.species,
            physics=sc.physics,
            metadata=metadata,
            frame=sc.frame,
            transforms=sc.transforms or None,
            coords=coords,
            strict_fields=False,
        )
