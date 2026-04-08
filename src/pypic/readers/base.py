"""Core data containers: GridInfo, FieldDataset, SimulationReader, SimulationConfig."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, assert_never, runtime_checkable

import numpy as np
import xarray as xr

from pypic.coordinates.geometry import (
    CARTESIAN,
    CYLINDRICAL,  # noqa: F401 — used in doctests
    SPHERICAL,  # noqa: F401 — used in doctests
    CoordinateGeometry,
    GeometryType,
)
from pypic.coordinates.transforms import FrameTransform

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from xarray import Dataset

    from pypic.fields import FieldInfo, QuantityType
    from pypic.types import FloatArray
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
    origin: tuple[float, ...] = ()
    geometry: CoordinateGeometry = CARTESIAN
    dt: float | None = None
    boundary: tuple[str, ...] | None = None
    surviving_axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        ndim = len(self.dimensions)
        if not self.origin:
            object.__setattr__(self, "origin", (0.0,) * ndim)
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
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.boundary is not None and len(self.boundary) != ndim:
            msg = (
                f"boundary length ({len(self.boundary)}) "
                f"must match dimensions length ({ndim})"
            )
            raise ValueError(msg)
        if self.surviving_axes is not None and len(self.surviving_axes) != ndim:
            msg = (
                f"surviving_axes length ({len(self.surviving_axes)}) "
                f"must match dimensions length ({ndim})"
            )
            raise ValueError(msg)

    @property
    def surviving_axis_names(self) -> tuple[str, ...]:
        """Axis names for the current dimensions.

        After slicing, returns only the names of axes that survived
        (e.g. ``("x", "z")`` after removing the y-axis). When no
        slicing has occurred, returns the first *ndim* names from
        the geometry.

        Examples
        --------
        >>> grid = GridInfo(
        ...     dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> grid.surviving_axis_names
        ('x', 'y', 'z')
        >>> import copy
        >>> sliced = copy.replace(
        ...     grid, dimensions=(4, 2), spacing=(1.0, 1.0),
        ...     origin=(0.0, 0.0), surviving_axes=(0, 2),
        ... )
        >>> sliced.surviving_axis_names
        ('x', 'z')
        """
        if self.surviving_axes is not None:
            return tuple(self.geometry.axis_names[i] for i in self.surviving_axes)
        return self.geometry.axis_names[: len(self.dimensions)]

    def coordinate_arrays(self) -> tuple[FloatArray, ...]:
        r"""Cell-centered coordinate arrays for each axis.

        Returns
        -------
        tuple[FloatArray, ...]
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


_FIELD_PREFIX_PAIRS = (
    ("B", "B"),
    ("B0", "B0"),  # split-B background field (BATSRUS)
    ("E", "E"),
    ("EF", "EF"),  # per-species energy flux (iPIC3D H5hut)
    ("J", "J"),
    ("V", "V"),
    ("v", "V"),  # lowercase convenience alias
    ("Ve", "Ve"),
    ("S", "S"),
    ("u", "u"),  # four-velocity
)


def _build_aliases(
    suffixes: tuple[str, str, str], *, separator: str = ""
) -> dict[str, str]:
    """Generate field name aliases for a coordinate system.

    Parameters
    ----------
    suffixes : tuple[str, str, str]
        Coordinate suffixes (e.g. ``("x", "y", "z")``).
    separator : str
        Separator between prefix and suffix. ``""`` gives ``Bx``,
        ``"_"`` gives ``B_x``.
    """
    aliases: dict[str, str] = {}
    for alias_prefix, canonical_prefix in _FIELD_PREFIX_PAIRS:
        for i, suffix in enumerate(suffixes, 1):
            aliases[f"{alias_prefix}{separator}{suffix}"] = f"{canonical_prefix}{i}"
    return aliases


_CARTESIAN_ALIASES = _build_aliases(("x", "y", "z"))
_SPHERICAL_ALIASES = _build_aliases(("r", "theta", "phi"))
_CYLINDRICAL_ALIASES = _build_aliases(("r", "phi", "z"))

_CARTESIAN_UNDERSCORE_ALIASES = _build_aliases(("x", "y", "z"), separator="_")
_SPHERICAL_UNDERSCORE_ALIASES = _build_aliases(("r", "theta", "phi"), separator="_")
_CYLINDRICAL_UNDERSCORE_ALIASES = _build_aliases(("r", "phi", "z"), separator="_")

# Numbered underscore aliases (B_1→B1, E_2→E2, etc.) — geometry-independent
_NUMBERED_UNDERSCORE_ALIASES: dict[str, str] = {}
for _alias_pfx, _canon_pfx in _FIELD_PREFIX_PAIRS:
    for _i in (1, 2, 3):
        _NUMBERED_UNDERSCORE_ALIASES[f"{_alias_pfx}_{_i}"] = f"{_canon_pfx}{_i}"

# Scalar underscore aliases (P_e→Pe, T_i→Ti, pressure tensor components)
_SCALAR_UNDERSCORE_ALIASES: dict[str, str] = {
    "P_e": "Pe",
    "P_i": "Pi",
    "T_e": "Te",
    "T_i": "Ti",
    "P_11": "P11",
    "P_12": "P12",
    "P_13": "P13",
    "P_22": "P22",
    "P_23": "P23",
    "P_33": "P33",
}

# Species-convenience aliases (geometry-independent)
_SPECIES_ALIASES: dict[str, str] = {
    "n_e": "n_s0",
    "n_i": "n_s1",
}


def _default_aliases(geometry: CoordinateGeometry) -> dict[str, str]:
    """Return geometry-specific field name aliases plus species aliases."""
    match geometry.type:
        case GeometryType.CARTESIAN:
            aliases = dict(_CARTESIAN_ALIASES)
            aliases.update(_CARTESIAN_UNDERSCORE_ALIASES)
        case GeometryType.SPHERICAL:
            aliases = dict(_SPHERICAL_ALIASES)
            aliases.update(_SPHERICAL_UNDERSCORE_ALIASES)
        case GeometryType.CYLINDRICAL:
            aliases = dict(_CYLINDRICAL_ALIASES)
            aliases.update(_CYLINDRICAL_UNDERSCORE_ALIASES)
        case _ as unreachable:
            assert_never(unreachable)
    aliases.update(_NUMBERED_UNDERSCORE_ALIASES)
    aliases.update(_SCALAR_UNDERSCORE_ALIASES)
    aliases.update(_SPECIES_ALIASES)
    return aliases


def _build_grid_from_dataset(old_grid: GridInfo, new_ds: Dataset) -> GridInfo:
    """Derive a reduced GridInfo from a sliced xr.Dataset."""
    current_names = old_grid.surviving_axis_names
    surviving: list[tuple[int, str]] = []  # (local_index, name)

    for local_idx, name in enumerate(current_names):
        if name in new_ds.dims:
            surviving.append((local_idx, name))

    # Map local indices back to original 3D geometry axis indices
    if old_grid.surviving_axes is not None:
        new_surviving = tuple(old_grid.surviving_axes[li] for li, _ in surviving)
    else:
        new_surviving = tuple(li for li, _ in surviving)

    new_boundary = None
    if old_grid.boundary is not None:
        new_boundary = tuple(old_grid.boundary[i] for i, _ in surviving)

    return copy.replace(
        old_grid,
        dimensions=tuple(int(new_ds.sizes[name]) for _, name in surviving),
        spacing=tuple(old_grid.spacing[i] for i, _ in surviving),
        origin=tuple(
            # invert cell-center formula: coord[0] = origin + 0.5*spacing
            float(new_ds.coords[name].values[0]) - 0.5 * old_grid.spacing[i]
            for i, name in surviving
        ),
        boundary=new_boundary,
        surviving_axes=new_surviving,
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
    species : Sequence[SpeciesInfo] | None
        Species definitions, if applicable.
    physics : dict[str, Any] | None
        Physics parameters (e.g. resistivity, viscosity).
    metadata : dict[str, Any] | None
        Arbitrary metadata (run name, code version, etc.).
    aliases : dict[str, str] | None
        Extra field-name aliases merged with geometry defaults.
        Aliases whose canonical target is absent from the dataset
        are silently dropped (they become inactive).

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
        species: Sequence[SpeciesInfo] | None = None,
        physics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
        frame: str = "simulation",
        transforms: dict[str, FrameTransform] | None = None,
    ) -> None:
        self._ds = dataset
        self._grid = grid
        self._normalization = normalization
        self._species = tuple(species) if species is not None else ()
        self._physics = physics if physics is not None else {}
        self._metadata = metadata if metadata is not None else {}
        self._frame = frame
        self._transforms = dict(transforms) if transforms is not None else {}

        merged = _default_aliases(grid.geometry)
        if aliases:
            merged.update(aliases)
        # Generate species-name aliases (e.g. n_electrons→n_s0) from config
        for i, sp in enumerate(self._species):
            candidate = f"n_{sp.name.lower()}"
            target = f"n_s{i}"
            if candidate not in merged and target in self._ds.data_vars:
                merged[candidate] = target
        # Only keep aliases whose canonical target exists
        self._aliases = {k: v for k, v in merged.items() if v in self._ds.data_vars}

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, FloatArray],
        grid: GridInfo,
        normalization: Normalization | None = None,
        *,
        species: Sequence[SpeciesInfo] | None = None,
        physics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
        frame: str = "simulation",
        transforms: dict[str, FrameTransform] | None = None,
    ) -> FieldDataset:
        r"""Build a FieldDataset from a dict of NumPy arrays.

        Parameters
        ----------
        fields : dict[str, FloatArray]
            Mapping of field names to arrays. Shapes must match
            ``grid.dimensions``.
        grid : GridInfo
            Grid metadata.
        normalization : Normalization | None
            Unit normalization.  Defaults to ``Normalization.identity()``.
        species : Sequence[SpeciesInfo] | None
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
        >>> grid = GridInfo(dimensions=(2,), spacing=(1.0,))
        >>> ds = FieldDataset.from_arrays({"B1": np.array([1.0, 2.0])}, grid)
        >>> ds["B1"]
        array([1., 2.])
        """
        if normalization is None:
            from pypic.units import Normalization as _Norm

            normalization = _Norm.identity()
        dim_names = list(grid.surviving_axis_names)
        coord_arrays = grid.coordinate_arrays()
        coords = {dim_names[i]: coord_arrays[i] for i in range(len(dim_names))}

        from pypic.fields import field_info as _field_info

        data_vars: dict[str, xr.DataArray] = {}
        for var_name, arr in fields.items():
            da = xr.DataArray(data=arr, dims=dim_names)
            try:
                info = _field_info(var_name, axis_names=grid.geometry.axis_names)
                da.attrs["long_name"] = info.long_name
                da.attrs["units"] = "normalized"
                da.attrs["quantity_type"] = info.quantity_type
                da.attrs["si_unit"] = info.si_unit
                if info.latex:
                    da.attrs["latex"] = info.latex
            except KeyError:
                pass
            data_vars[var_name] = da
        dataset = xr.Dataset(data_vars, coords=coords)
        return cls(
            dataset,
            grid,
            normalization,
            species=species,
            physics=physics,
            metadata=metadata,
            aliases=aliases,
            frame=frame,
            transforms=transforms,
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
    def species(self) -> tuple[SpeciesInfo, ...]:
        """Species definitions."""
        return self._species

    @property
    def physics(self) -> MappingProxyType[str, Any]:
        """Physics parameters (read-only view)."""
        return MappingProxyType(self._physics)

    @property
    def metadata(self) -> MappingProxyType[str, Any]:
        """Arbitrary metadata (read-only view)."""
        return MappingProxyType(self._metadata)

    @property
    def aliases(self) -> MappingProxyType[str, str]:
        """Active field-name aliases (read-only view)."""
        return MappingProxyType(self._aliases)

    @property
    def frame(self) -> str:
        """Current reference frame label."""
        return self._frame

    @property
    def transforms(self) -> MappingProxyType[str, FrameTransform]:
        """Registered frame transforms (read-only view)."""
        return MappingProxyType(self._transforms)

    @property
    def available_frames(self) -> list[str]:
        """Frame names reachable via registered transforms."""
        frames: set[str] = {self._frame}
        for t in self._transforms.values():
            frames.add(t.source_frame)
            frames.add(t.target_frame)
        return sorted(frames)

    @property
    def xr(self) -> xr.Dataset:
        """Raw xarray Dataset."""
        return self._ds

    def transform_to(self, target: str | FrameTransform) -> FieldDataset:
        r"""Transform this dataset to a different coordinate reference frame.

        Applies an affine transformation: translates the grid origin,
        scales coordinates, and rotates vector field components and
        pressure tensors. Scalar fields pass through unchanged. The
        array memory layout is not transposed.

        Parameters
        ----------
        target : str or FrameTransform
            Target frame name (looked up in the transform registry) or
            a `FrameTransform` instance applied directly.

        Returns
        -------
        FieldDataset
            New dataset in the target frame.

        Raises
        ------
        ValueError
            If no transform path exists from the current frame.
        KeyError
            If *target* is a string and no transforms are registered.
        """
        from pypic.coordinates.transforms import (
            find_pressure_tensor_groups,
            find_vector_triplets,
            resolve_transform,
            rotate_pressure_tensor,
            rotate_vector_components,
        )

        if isinstance(target, FrameTransform):
            transform = target
            target_frame = transform.target_frame
        else:
            target_frame = target
            if target_frame == self._frame:
                return self
            if not self._transforms:
                msg = (
                    f"No transforms registered; cannot transform "
                    f"from {self._frame!r} to {target_frame!r}"
                )
                raise KeyError(msg)
            transform = resolve_transform(self._frame, target_frame, self._transforms)
        rotation = transform.rotation_matrix
        dim_names = list(self._grid.surviving_axis_names)

        # Start with a copy of all data variables
        new_vars: dict[str, xr.DataArray] = {}
        rotated_fields: set[str] = set()

        # Rotate vector triplets
        for n1, n2, n3 in find_vector_triplets(self.field_names()):
            v1, v2, v3 = self[n1], self[n2], self[n3]
            r1, r2, r3 = rotate_vector_components(v1, v2, v3, rotation)
            for name, arr in [(n1, r1), (n2, r2), (n3, r3)]:
                new_vars[name] = xr.DataArray(
                    data=arr, dims=dim_names, attrs=dict(self._ds[name].attrs)
                )
                rotated_fields.add(name)

        # Rotate pressure tensors
        for p11, p22, p33, p12, p13, p23 in find_pressure_tensor_groups(
            self.field_names()
        ):
            rp = rotate_pressure_tensor(
                self[p11],
                self[p22],
                self[p33],
                self[p12],
                self[p13],
                self[p23],
                rotation,
            )
            for name, arr in zip([p11, p22, p33, p12, p13, p23], rp, strict=True):
                new_vars[name] = xr.DataArray(
                    data=arr, dims=dim_names, attrs=dict(self._ds[name].attrs)
                )
                rotated_fields.add(name)

        # Copy scalar fields unchanged
        for raw_name in self._ds.data_vars:
            name = str(raw_name)
            if name not in rotated_fields:
                new_vars[name] = self._ds[name]

        # Transform grid: permute axes, flip reversed ones, shift origin
        if self._grid.surviving_axes is not None:
            indices = list(self._grid.surviving_axes)
        else:
            indices = list(range(len(self._grid.dimensions)))
        ndim = len(indices)

        # The grid transformation (transpose + flip) only works for
        # signed permutation matrices (axis swaps and reflections).
        # General rotations would require interpolation onto a new grid.
        r_sub = rotation[np.ix_(indices, indices)]
        abs_r = np.abs(r_sub)
        if not (
            np.allclose(np.sum(abs_r, axis=1), 1.0, atol=1e-6)
            and np.allclose(np.sum(abs_r, axis=0), 1.0, atol=1e-6)
        ):
            raise NotImplementedError(
                "transform_to() only supports axis-swap/reflection "
                "rotation matrices (signed permutations). General "
                "rotations require grid interpolation (not yet implemented)."
            )

        t_origin = np.array(transform.origin, dtype=np.float64)
        dx_scale = transform.scale

        # Determine axis permutation and sign from rotation matrix.
        # For each TARGET axis (row of R), find which SOURCE axis
        # it draws from (the single nonzero column) and its sign.
        perm: list[int] = []  # perm[target_i] = source local index
        signs: list[float] = []  # sign of the mapping
        for target_i in indices:
            row = rotation[target_i, :]
            source_orig = int(np.argmax(np.abs(row[indices])))
            perm.append(source_orig)
            signs.append(float(np.sign(row[indices[source_orig]])))

        # Transpose + flip arrays to match target axis ordering. By this
        # point every entry in new_vars is a DataArray (rotated above or
        # copied from self._ds), so .values and .attrs are always present.
        needs_transpose = perm != list(range(ndim))
        flip_axes = [i for i, s in enumerate(signs) if s < 0]
        if needs_transpose or flip_axes:
            for name, da in list(new_vars.items()):
                arr = da.values
                if needs_transpose:
                    arr = np.transpose(arr, perm)
                for ax in flip_axes:
                    arr = np.flip(arr, axis=ax)
                new_vars[name] = xr.DataArray(
                    data=np.ascontiguousarray(arr),
                    attrs=dict(da.attrs),
                )

        # Compute new origin, spacing, dimensions from permuted source
        old_coords = self._grid.coordinate_arrays()
        new_origin_list: list[float] = []
        new_spacing_list: list[float] = []
        new_dims_list: list[int] = []
        for src_i, sign in zip(perm, signs, strict=True):
            src_coords = old_coords[src_i]
            t_first = dx_scale * sign * (src_coords[0] - t_origin[indices[src_i]])
            t_last = dx_scale * sign * (src_coords[-1] - t_origin[indices[src_i]])
            dx = dx_scale * self._grid.spacing[src_i]
            new_origin_list.append(float(min(t_first, t_last)) - 0.5 * dx)
            new_spacing_list.append(dx)
            new_dims_list.append(self._grid.dimensions[src_i])

        # Map surviving_axes through the permutation
        new_surviving = None
        if self._grid.surviving_axes is not None:
            new_surviving = tuple(
                self._grid.surviving_axes[perm[i]] for i in range(ndim)
            )

        # Update geometry axis names if specified
        new_geometry = self._grid.geometry
        if transform.target_axis_names is not None:
            new_geometry = copy.replace(
                new_geometry, axis_names=transform.target_axis_names
            )

        new_grid = copy.replace(
            self._grid,
            dimensions=tuple(new_dims_list),
            origin=tuple(new_origin_list),
            spacing=tuple(new_spacing_list),
            geometry=new_geometry,
            surviving_axes=new_surviving,
        )

        # Rebuild xr.Dataset with transformed coordinates
        new_dim_names = list(new_grid.surviving_axis_names)
        coord_arrays = new_grid.coordinate_arrays()
        coords = {new_dim_names[i]: coord_arrays[i] for i in range(len(new_dim_names))}
        rebuilt_vars: dict[str, xr.DataArray] = {}
        for name, da in new_vars.items():
            rebuilt_vars[name] = xr.DataArray(
                data=da.values,
                dims=new_dim_names,
                attrs=dict(da.attrs),
            )
        new_ds = xr.Dataset(rebuilt_vars, coords=coords)

        return FieldDataset(
            new_ds,
            new_grid,
            self._normalization,
            species=self._species,
            physics=self._physics,
            metadata=self._metadata,
            frame=target_frame,
            transforms=self._transforms,
        )

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
            frame=self._frame,
            transforms=self._transforms,
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
        import difflib

        candidates = [str(v) for v in self._ds.data_vars] + list(self._aliases)
        suggestions = difflib.get_close_matches(key, candidates, n=3, cutoff=0.5)
        if suggestions:
            msg += f" Did you mean: {suggestions}?"
        raise KeyError(msg)

    def __getitem__(self, key: str) -> FloatArray:
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

    def select_fields(self, names: Iterable[str]) -> FieldDataset:
        """Return a new FieldDataset containing only the specified fields.

        Parameters
        ----------
        names : Iterable[str]
            Field names to keep (canonical or alias).

        Returns
        -------
        FieldDataset

        Raises
        ------
        KeyError
            If any name is not found.

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(2,), spacing=(1.0,), origin=(0.0,),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B1": np.array([1.0, 2.0]), "B2": np.array([3.0, 4.0]),
        ...      "rho_c": np.array([0.5, 0.5])},
        ...     grid, Normalization.identity(),
        ... )
        >>> sub = ds.select_fields(["B1", "rho_c"])
        >>> sorted(sub.field_names())
        ['B1', 'rho_c']
        """
        resolved: set[str] = set()
        for name in names:
            resolved.add(self._resolve_key(name))

        new_ds = self._ds[sorted(resolved)]
        return FieldDataset(
            new_ds,
            self._grid,
            self._normalization,
            species=self._species,
            physics=self._physics,
            metadata=self._metadata,
            aliases={k: v for k, v in self._aliases.items() if v in resolved},
            frame=self._frame,
            transforms=self._transforms,
        )

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

    def compute(self, name: str) -> FloatArray:
        """Compute a derived quantity by name. Returns code units.

        Parameters
        ----------
        name : str
            Derived quantity name (e.g. ``"|B|"``, ``"beta"``, ``"v_A"``).
            See ``pypic.compute.available_quantities()`` for the full list.

        Returns
        -------
        FloatArray
            Computed array in code units.

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B1": np.full((4,3,2), 3.0),
        ...      "B2": np.full((4,3,2), 4.0),
        ...      "B3": np.zeros((4,3,2))},
        ...     grid, Normalization.identity(),
        ... )
        >>> ds.compute("|B|")[0, 0, 0]
        np.float64(5.0)
        """
        from pypic.compute import compute_field

        return compute_field(name, self)

    def with_field(
        self,
        name: str,
        data: FloatArray,
        quantity_type: QuantityType | str,
        *,
        long_name: str = "",
        latex: str = "",
    ) -> FieldDataset:
        """Return a new FieldDataset with an additional custom field.

        The field's ``quantity_type`` is stored in xarray attrs, so
        ``in_si()``, ``field_info()``, and unit conversion work without
        global ``register_field()`` calls.

        Parameters
        ----------
        name : str
            Field name.
        data : FloatArray
            Array matching the grid dimensions.
        quantity_type : QuantityType | str
            Physical quantity type (e.g. ``QuantityType.VELOCITY``).
        long_name : str
            Human-readable label for plot titles.
        latex : str
            LaTeX symbol for plot labels.

        Returns
        -------
        FieldDataset
            New dataset with the field added.

        Raises
        ------
        ValueError
            If *quantity_type* is not recognized.
        """
        from pypic.fields import _QUANTITY_UNITS

        expected = tuple(self._grid.dimensions)
        if data.shape != expected:
            msg = f"Array shape {data.shape} doesn't match grid dimensions {expected}"
            raise ValueError(msg)

        qt = str(quantity_type)
        if qt not in _QUANTITY_UNITS:
            valid = sorted(_QUANTITY_UNITS)
            msg = f"Unknown quantity_type {qt!r}. Valid: {valid}"
            raise ValueError(msg)

        si_unit = _QUANTITY_UNITS[qt]
        dim_names = list(self._grid.surviving_axis_names)
        da = xr.DataArray(data=data, dims=dim_names)
        da.attrs["quantity_type"] = qt
        da.attrs["si_unit"] = si_unit
        da.attrs["units"] = "normalized"
        da.attrs["long_name"] = long_name
        da.attrs["latex"] = latex

        new_ds = self._ds.assign({name: da})
        return FieldDataset(
            new_ds,
            self._grid,
            self._normalization,
            species=self._species,
            physics=self._physics,
            metadata=self._metadata,
            aliases=dict(self._aliases),
            frame=self._frame,
            transforms=self._transforms,
        )

    def field_info(self, name: str) -> FieldInfo:
        """Return metadata for a field or derived quantity.

        Checks xarray DataArray attrs first (set by ``with_field()``
        or ``from_arrays()``), then falls back to the global registry.

        Parameters
        ----------
        name : str
            Field or derived quantity name (canonical or alias).

        Returns
        -------
        FieldInfo
        """
        from pypic.fields import FieldInfo as _FieldInfo
        from pypic.fields import field_info as _field_info

        if self.has_field(name):
            resolved = self._resolve_key(name)
            attrs = self._ds[resolved].attrs
            qt = attrs.get("quantity_type")
            if qt is not None:
                return _FieldInfo(
                    quantity_type=qt,
                    long_name=attrs.get("long_name", ""),
                    si_unit=attrs.get("si_unit", ""),
                    latex=attrs.get("latex", ""),
                )
        return _field_info(name, axis_names=self._grid.geometry.axis_names)

    def in_si(self, name: str) -> FloatArray:
        """Return a field or derived quantity in SI units.

        Checks xarray DataArray attrs first (set by ``with_field()``
        or ``from_arrays()``), then falls back to the global registry.

        Parameters
        ----------
        name : str
            Field or derived quantity name.

        Returns
        -------
        FloatArray
            Values in SI units.
        """
        from pypic.compute import compute_field, field_si_factor

        if self.has_field(name):
            resolved = self._resolve_key(name)
            data = self._ds[resolved].values
            qt = self._ds[resolved].attrs.get("quantity_type")
            if qt is not None:
                factor = self._normalization.si_factor(qt)
                return data if factor == 1.0 else data * factor
            # No quantity_type attr — fall through to global registry
        else:
            data = compute_field(name, self)

        # Fall back to global registry
        factor = field_si_factor(name, self._normalization)
        return data if factor == 1.0 else data * factor

    def in_units(self, name: str, unit_str: str) -> FloatArray:
        """Return a field or derived quantity in display units.

        Parameters
        ----------
        name : str
            Field or derived quantity name.
        unit_str : str
            Target unit (e.g. ``"nT"``, ``"km/s"``, ``"cm^-3"``).

        Returns
        -------
        FloatArray
            Values in the requested units.
        """
        from pypic.compute import display_unit_factor

        return self.in_si(name) / display_unit_factor(unit_str)

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

    def where(self, cond: np.ndarray, other: float = np.nan) -> FieldDataset:
        r"""Mask fields where *cond* is ``False``.

        Returns a new :class:`FieldDataset` with the same grid shape.
        Points where *cond* is ``False`` are set to *other* (default
        ``NaN``).  Useful for spatial masks (spherical cutouts, boundary
        regions) without reducing dimensions.

        Parameters
        ----------
        cond : np.ndarray
            Boolean array with shape matching the grid dimensions.
            ``True`` keeps the value, ``False`` replaces with *other*.
        other : float
            Fill value for masked points (default ``NaN``).

        Returns
        -------
        FieldDataset
        """
        axis_names = list(self._grid.surviving_axis_names)
        mask_da = xr.DataArray(cond, dims=axis_names)
        return self._wrap_sliced(self._ds.where(mask_da, other=other))


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
        Physics parameters (immutable after construction).
    frame : str
        Reference frame label (e.g. ``"GSM"``, ``"simulation"``).
    metadata : dict[str, Any]
        Additional configuration data (immutable after construction).

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
    ... )
    >>> cfg.model_name
    'test'
    """

    model_name: str
    model_type: str
    grid: GridInfo
    normalization: Normalization
    species: tuple[SpeciesInfo, ...] = ()
    physics: dict[str, Any] = field(default_factory=dict)  # frozen via __post_init__
    frame: str = "simulation"
    transforms: dict[str, FrameTransform] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)  # frozen via __post_init__

    def __post_init__(self) -> None:
        # Wrap mutable dicts in read-only proxies to enforce true immutability.
        # Callers pass plain dicts; frozen assignment uses object.__setattr__.
        object.__setattr__(self, "physics", MappingProxyType(dict(self.physics)))
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


@dataclass(frozen=True, slots=True)
class ParticleData:
    r"""Container for particle data from a single species at one timestep.

    Stores position and velocity as ``(N, 3)`` arrays with selective
    loading: either can be ``None`` if not requested.  The ``charge``
    array is **always** loaded because per-particle $q$ doubles as a
    unique particle identifier (each particle's weight is unique at
    full float64 precision in restart files).

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
    charge : FloatArray
        Per-particle charge/weight, shape ``(N,)``. Always float64.
    n_particles : int
        Total particle count.
    id : np.ndarray | None
        Integer particle tracking IDs, shape ``(N,)``. ``None`` if not
        available or not requested.
    metadata : dict[str, Any]
        Source info (file path, format, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> pcl = ParticleData(
    ...     species_index=0, species_name="electrons",
    ...     position=np.zeros((10, 3)),
    ...     velocity=np.ones((10, 3)),
    ...     charge=np.full(10, -1.0),
    ...     n_particles=10, metadata={},
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
    charge: FloatArray
    n_particles: int
    metadata: dict[str, Any]  # frozen at runtime via __post_init__
    id: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.position is None and self.velocity is None:
            msg = "At least one of position or velocity must be provided"
            raise ValueError(msg)
        if self.charge.dtype != np.float64:
            msg = (
                f"charge must be float64 (particle ID precision), "
                f"got {self.charge.dtype}"
            )
            raise ValueError(msg)
        if self.charge.shape != (self.n_particles,):
            msg = (
                f"charge shape {self.charge.shape} does not match "
                f"n_particles ({self.n_particles},)"
            )
            raise ValueError(msg)
        if self.position is not None and self.position.shape != (self.n_particles, 3):
            msg = (
                f"position shape {self.position.shape} does not match "
                f"(n_particles, 3) = ({self.n_particles}, 3)"
            )
            raise ValueError(msg)
        if self.velocity is not None and self.velocity.shape != (self.n_particles, 3):
            msg = (
                f"velocity shape {self.velocity.shape} does not match "
                f"(n_particles, 3) = ({self.n_particles}, 3)"
            )
            raise ValueError(msg)
        if self.id is not None and self.id.shape != (self.n_particles,):
            msg = (
                f"id shape {self.id.shape} does not match "
                f"n_particles ({self.n_particles},)"
            )
            raise ValueError(msg)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def x(self) -> FloatArray:
        """X positions (view into ``position[:, 0]``)."""
        if self.position is None:
            msg = "position was not loaded"
            raise ValueError(msg)
        return self.position[:, 0]

    @property
    def y(self) -> FloatArray:
        """Y positions (view into ``position[:, 1]``)."""
        if self.position is None:
            msg = "position was not loaded"
            raise ValueError(msg)
        return self.position[:, 1]

    @property
    def z(self) -> FloatArray:
        """Z positions (view into ``position[:, 2]``)."""
        if self.position is None:
            msg = "position was not loaded"
            raise ValueError(msg)
        return self.position[:, 2]

    @property
    def vx(self) -> FloatArray:
        """X velocities (view into ``velocity[:, 0]``)."""
        if self.velocity is None:
            msg = "velocity was not loaded"
            raise ValueError(msg)
        return self.velocity[:, 0]

    @property
    def vy(self) -> FloatArray:
        """Y velocities (view into ``velocity[:, 1]``)."""
        if self.velocity is None:
            msg = "velocity was not loaded"
            raise ValueError(msg)
        return self.velocity[:, 1]

    @property
    def vz(self) -> FloatArray:
        """Z velocities (view into ``velocity[:, 2]``)."""
        if self.velocity is None:
            msg = "velocity was not loaded"
            raise ValueError(msg)
        return self.velocity[:, 2]

    def __len__(self) -> int:
        return self.n_particles

    def __repr__(self) -> str:
        loaded = []
        if self.position is not None:
            loaded.append("position")
        if self.velocity is not None:
            loaded.append("velocity")
        loaded.append("charge")
        if self.id is not None:
            loaded.append("id")
        return (
            f"ParticleData({self.species_name!r}, "
            f"n={self.n_particles:,}, "
            f"loaded=[{', '.join(loaded)}])"
        )


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
