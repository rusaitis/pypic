"""Core data containers: GridInfo, FieldDataset, SimulationReader, SimulationConfig."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from xarray import Dataset

from pypic.coordinates.geometry import (
    CARTESIAN,  # noqa: F401 — used in doctests
    CYLINDRICAL,  # noqa: F401 — used in doctests
    SPHERICAL,  # noqa: F401 — used in doctests
    CoordinateGeometry,
    GeometryType,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pypic.units import Normalization, SpeciesInfo


@dataclass(frozen=True, slots=True)
class GridInfo:
    r"""Structured grid metadata for 1D/2D/3D simulation domains.

    Parameters
    ----------
    dimensions : tuple[int, ...]
        Number of cells along each axis.
    spacing : tuple[float, ...]
        Cell size along each axis in code units.
    origin : tuple[float, ...]
        Lower-left corner coordinate of the domain.
    geometry : CoordinateGeometry
        Coordinate system (Cartesian, spherical, cylindrical).
    dt : float | None
        Timestep size in code units, if known.
    boundary : tuple[str, ...] | None
        Boundary condition per axis (e.g. ``("periodic", "open", "periodic")``).

    Examples
    --------
    >>> grid = GridInfo(
    ...     dimensions=(4,), spacing=(0.5,), origin=(0.0,),
    ...     geometry=CARTESIAN,
    ... )
    >>> grid.coordinate_arrays()[0]
    array([0.25, 0.75, 1.25, 1.75])
    """

    dimensions: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    geometry: CoordinateGeometry
    dt: float | None = None
    boundary: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        ndim = len(self.dimensions)
        if len(self.spacing) != ndim or len(self.origin) != ndim:
            msg = (
                f"Length mismatch: dimensions({ndim}), "
                f"spacing({len(self.spacing)}), origin({len(self.origin)})"
            )
            raise ValueError(msg)
        for i, d in enumerate(self.dimensions):
            if d <= 0:
                raise ValueError(f"dimensions[{i}] must be > 0, got {d}")
        for i, s in enumerate(self.spacing):
            if s <= 0:
                raise ValueError(f"spacing[{i}] must be > 0, got {s}")
        if self.boundary is not None and len(self.boundary) != ndim:
            msg = (
                f"boundary length ({len(self.boundary)}) "
                f"must match dimensions length ({ndim})"
            )
            raise ValueError(msg)

    def coordinate_arrays(self) -> tuple[NDArray[np.floating[Any]], ...]:
        r"""Cell-centered coordinate arrays for each axis.

        Returns
        -------
        tuple[NDArray[np.floating[Any]], ...]
            One 1-D array per axis: ``origin[i] + (arange(n) + 0.5) * dx[i]``.

        Examples
        --------
        >>> grid = GridInfo(
        ...     dimensions=(3, 2), spacing=(1.0, 2.0), origin=(0.0, 0.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> x, y = grid.coordinate_arrays()
        >>> x
        array([0.5, 1.5, 2.5])
        >>> y
        array([1., 3.])
        """
        return tuple(
            self.origin[i] + (np.arange(self.dimensions[i]) + 0.5) * self.spacing[i]
            for i in range(len(self.dimensions))
        )


_CARTESIAN_ALIASES: dict[str, str] = {
    "Bx": "B1",
    "By": "B2",
    "Bz": "B3",
    "Ex": "E1",
    "Ey": "E2",
    "Ez": "E3",
    "Jx": "J1",
    "Jy": "J2",
    "Jz": "J3",
    "vx": "v1",
    "vy": "v2",
    "vz": "v3",
}

_SPHERICAL_ALIASES: dict[str, str] = {
    "Br": "B1",
    "Btheta": "B2",
    "Bphi": "B3",
    "Er": "E1",
    "Etheta": "E2",
    "Ephi": "E3",
    "Jr": "J1",
    "Jtheta": "J2",
    "Jphi": "J3",
    "vr": "v1",
    "vtheta": "v2",
    "vphi": "v3",
}

_CYLINDRICAL_ALIASES: dict[str, str] = {
    "Br": "B1",
    "Bphi": "B2",
    "Bz": "B3",
    "Er": "E1",
    "Ephi": "E2",
    "Ez": "E3",
    "Jr": "J1",
    "Jphi": "J2",
    "Jz": "J3",
    "vr": "v1",
    "vphi": "v2",
    "vz": "v3",
}


def _default_aliases(geometry: CoordinateGeometry) -> dict[str, str]:
    """Return geometry-specific field name aliases."""
    match geometry.type:
        case GeometryType.CARTESIAN:
            return dict(_CARTESIAN_ALIASES)
        case GeometryType.SPHERICAL:
            return dict(_SPHERICAL_ALIASES)
        case GeometryType.CYLINDRICAL:
            return dict(_CYLINDRICAL_ALIASES)
        case _:
            return {}


def _build_grid_from_dataset(old_grid: GridInfo, new_ds: xr.Dataset) -> GridInfo:
    """Derive a reduced GridInfo from a sliced xr.Dataset."""
    axis_names = old_grid.geometry.axis_names[: len(old_grid.dimensions)]
    surviving: list[tuple[int, str]] = []

    for i, name in enumerate(axis_names):
        if name in new_ds.dims:
            surviving.append((i, name))

    new_boundary = None
    if old_grid.boundary is not None:
        new_boundary = tuple(old_grid.boundary[i] for i, _ in surviving)

    return copy.replace(
        old_grid,
        dimensions=tuple(int(new_ds.sizes[name]) for _, name in surviving),
        spacing=tuple(old_grid.spacing[i] for i, _ in surviving),
        origin=tuple(
            float(new_ds.coords[name].values[0]) - 0.5 * old_grid.spacing[i]
            for i, name in surviving
        ),
        boundary=new_boundary,
    )


class FieldDataset:
    r"""Universal container for simulation field data.

    Wraps an ``xr.Dataset`` with grid metadata, normalization info, species
    definitions, and geometry-aware field aliases (e.g. ``"Bx"`` → ``"B1"``).

    Use `from_arrays` to construct from raw NumPy arrays.

    Parameters
    ----------
    dataset : xr.Dataset
        The underlying xarray dataset.
    grid : GridInfo
        Grid metadata.
    normalization : Normalization
        Unit normalization for this data.
    species : list[SpeciesInfo] | None
        Species definitions, if applicable.
    physics : dict[str, Any] | None
        Physics parameters (e.g. resistivity, viscosity).
    metadata : dict[str, Any] | None
        Arbitrary metadata (run name, code version, etc.).
    aliases : dict[str, str] | None
        Extra field-name aliases merged with geometry defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.units import Normalization
    >>> grid = GridInfo(
    ...     dimensions=(4, 3), spacing=(1.0, 1.0), origin=(0.0, 0.0),
    ...     geometry=CARTESIAN,
    ... )
    >>> fields = {"B1": np.ones((4, 3)), "rho_c": np.zeros((4, 3))}
    >>> ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())
    >>> ds["B1"].shape
    (4, 3)
    >>> ds.has_field("Bx")
    True
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        grid: GridInfo,
        normalization: Normalization,
        *,
        species: list[SpeciesInfo] | None = None,
        physics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._ds = dataset
        self._grid = grid
        self._normalization = normalization
        self._species = species if species is not None else []
        self._physics = physics if physics is not None else {}
        self._metadata = metadata if metadata is not None else {}

        merged = _default_aliases(grid.geometry)
        if aliases:
            merged.update(aliases)
        # Only keep aliases whose canonical target exists
        self._aliases = {k: v for k, v in merged.items() if v in self._ds.data_vars}

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, NDArray[np.floating[Any]]],
        grid: GridInfo,
        normalization: Normalization,
        *,
        species: list[SpeciesInfo] | None = None,
        physics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
    ) -> FieldDataset:
        r"""Build a FieldDataset from a dict of NumPy arrays.

        Parameters
        ----------
        fields : dict[str, NDArray]
            Mapping of field names to arrays. Shapes must match
            ``grid.dimensions``.
        grid : GridInfo
            Grid metadata.
        normalization : Normalization
            Unit normalization.
        species : list[SpeciesInfo] | None
            Species definitions, if applicable.
        physics : dict[str, Any] | None
            Physics parameters.
        metadata : dict[str, Any] | None
            Arbitrary metadata.
        aliases : dict[str, str] | None
            Extra field-name aliases.

        Returns
        -------
        FieldDataset

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(2,), spacing=(1.0,), origin=(0.0,),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B1": np.array([1.0, 2.0])}, grid, Normalization.identity(),
        ... )
        >>> ds["B1"]
        array([1., 2.])
        """
        ndim = len(grid.dimensions)
        dim_names = list(grid.geometry.axis_names[:ndim])
        coord_arrays = grid.coordinate_arrays()
        coords = {dim_names[i]: coord_arrays[i] for i in range(ndim)}

        data_vars = {
            name: xr.DataArray(data=arr, dims=dim_names) for name, arr in fields.items()
        }
        dataset = xr.Dataset(data_vars, coords=coords)
        return cls(
            dataset,
            grid,
            normalization,
            species=species,
            physics=physics,
            metadata=metadata,
            aliases=aliases,
        )

    @property
    def grid(self) -> GridInfo:
        """Grid metadata."""
        return self._grid

    @property
    def normalization(self) -> Normalization:
        """Unit normalization."""
        return self._normalization

    @property
    def species(self) -> list[SpeciesInfo]:
        """Species definitions."""
        return self._species

    @property
    def physics(self) -> dict[str, Any]:
        """Physics parameters."""
        return self._physics

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata."""
        return self._metadata

    @property
    def aliases(self) -> dict[str, str]:
        """Active field-name aliases."""
        return self._aliases

    @property
    def xr(self) -> xr.Dataset:
        """Raw xarray Dataset."""
        return self._ds

    def _wrap_sliced(self, new_ds: Dataset) -> FieldDataset:
        """Wrap a sliced xr.Dataset in a new FieldDataset, preserving metadata."""
        new_grid = _build_grid_from_dataset(self._grid, new_ds)
        return FieldDataset(
            new_ds,
            new_grid,
            self._normalization,
            species=self._species,
            physics=self._physics,
            metadata=self._metadata,
            aliases={k: v for k, v in self._aliases.items() if v in new_ds.data_vars},
        )

    def _resolve_key(self, key: str) -> str:
        """Resolve a field key through aliases to the canonical name."""
        if key in self._ds.data_vars:
            return key
        canonical = self._aliases.get(key)
        if canonical is not None:
            return canonical
        available = sorted(self._ds.data_vars, key=str)
        alias_keys = sorted(self._aliases)
        msg = f"Field {key!r} not found. Available: {available}. Aliases: {alias_keys}."
        raise KeyError(msg)

    def __getitem__(self, key: str) -> NDArray[np.floating[Any]]:
        """Return field data as a NumPy array (zero-copy when possible).

        Parameters
        ----------
        key : str
            Canonical field name or alias.

        Returns
        -------
        NDArray
        """
        resolved = self._resolve_key(key)
        return self._ds[resolved].values

    def has_field(self, key: str) -> bool:
        """Check whether a field exists (canonical or alias).

        Parameters
        ----------
        key : str
            Field name to check.

        Returns
        -------
        bool

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(2,), spacing=(1.0,), origin=(0.0,),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B1": np.array([1.0, 2.0])}, grid, Normalization.identity(),
        ... )
        >>> ds.has_field("B1"), ds.has_field("Bx"), ds.has_field("rho")
        (True, True, False)
        """
        return key in self._ds.data_vars or key in self._aliases

    def field_names(self) -> list[str]:
        """Return canonical field names (no aliases).

        Returns
        -------
        list[str]

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(2,), spacing=(1.0,), origin=(0.0,),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B1": np.array([1.0, 2.0]), "rho_c": np.array([0.5, 0.5])},
        ...     grid, Normalization.identity(),
        ... )
        >>> sorted(ds.field_names())
        ['B1', 'rho_c']
        """
        return list(self._ds.data_vars)  # type: ignore[arg-type]  # xarray types Hashable, always str

    def sel(
        self,
        indexers: dict[str, Any] | None = None,
        *,
        method: str | None = None,
        **kwargs: Any,  # noqa: ANN401 — xarray passthrough
    ) -> FieldDataset:
        """Label-based selection, returning a new FieldDataset.

        Accepts ``indexers`` as a dict (useful for Unicode dim names like
        ``"θ"`` that can't be keyword arguments) and/or ``**kwargs``.

        Parameters
        ----------
        indexers : dict[str, Any] | None
            Dimension-name → label mapping.
        method : str | None
            Passed to ``xr.Dataset.sel`` (e.g. ``"nearest"``).
        **kwargs
            Additional dimension selections.

        Returns
        -------
        FieldDataset
        """
        merged = dict(indexers) if indexers else {}
        merged.update(kwargs)
        return self._wrap_sliced(self._ds.sel(merged, method=method))

    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 — xarray passthrough
    ) -> FieldDataset:
        """Integer-index selection, returning a new FieldDataset.

        Accepts ``indexers`` as a dict and/or ``**kwargs``.

        Parameters
        ----------
        indexers : dict[str, Any] | None
            Dimension-name → integer index or slice mapping.
        **kwargs
            Additional dimension selections.

        Returns
        -------
        FieldDataset
        """
        merged = dict(indexers) if indexers else {}
        merged.update(kwargs)
        return self._wrap_sliced(self._ds.isel(merged))


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


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parsed simulation configuration from a TOML config file.

    Parameters
    ----------
    model_name : str
        Human-readable name for the simulation run.
    model_type : str
        Simulation type identifier (e.g. ``"pic"``, ``"mhd"``).
    grid : GridInfo
        Grid metadata (includes coordinate geometry).
    normalization : Normalization
        Unit system.
    species : tuple[SpeciesInfo, ...]
        Species definitions (tuple for immutability).
    physics : dict[str, Any]
        Physics parameters.
    frame : str
        Reference frame label (e.g. ``"GSM"``, ``"simulation"``).
    metadata : dict[str, Any]
        Additional configuration data.

    Examples
    --------
    >>> from pypic.units import Normalization, SpeciesInfo
    >>> cfg = SimulationConfig(
    ...     model_name="test", model_type="pic",
    ...     grid=GridInfo(
    ...         dimensions=(4,), spacing=(1.0,), origin=(0.0,),
    ...         geometry=CARTESIAN,
    ...     ),
    ...     normalization=Normalization.identity(),
    ...     species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
    ...     physics={}, frame="simulation", metadata={},
    ... )
    >>> cfg.model_name
    'test'
    """

    model_name: str
    model_type: str
    grid: GridInfo
    normalization: Normalization
    species: tuple[SpeciesInfo, ...]
    physics: dict[str, Any]
    frame: str
    metadata: dict[str, Any]
