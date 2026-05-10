"""FieldDataset and re-exports from _grid, _containers, _protocols."""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from pypic.coordinates.geometry import (
    CARTESIAN,  # noqa: F401 — used in doctests
    CYLINDRICAL,  # noqa: F401 — used in doctests
    SPHERICAL,  # noqa: F401 — used in doctests
)
from pypic.coordinates.transforms import FrameTransform
from pypic.grid import (
    GridInfo,
    _build_grid_from_dataset,
    _default_aliases,
)
from pypic.units import PhysicsParams

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from xarray import Dataset

    from pypic.fields import FieldInfo, QuantityType
    from pypic.types import FloatArray
    from pypic.units import Normalization, SpeciesInfo


class FieldDataset:
    r"""Universal container for simulation field data.

    Wraps an ``xr.Dataset`` with grid metadata, normalization info, species
    definitions, and geometry-aware field aliases (e.g. ``"Bx"`` → ``"B_1"``).

    The full alias hierarchy — geometry/Cartesian aliases, species-name
    aliases (``n_electrons``→``n_s0``), and the e/i library-convenience
    forms (``Pe``↔``P_s0``) — is documented at ``docs/aliases.md``. The
    e/i shortcuts are pypic-only ergonomics and are **not** part of the
    cross-tool schema contract in ``docs/schema.md``.

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
    physics : PhysicsParams | None
        Physics parameters (adiabatic index, speed of light, etc.).
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
    >>> fields = {"B_1": np.ones((4, 3)), "rho_c": np.zeros((4, 3))}
    >>> ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())
    >>> ds["B_1"].shape
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
        physics: PhysicsParams | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
        frame: str = "simulation",
        transforms: dict[str, FrameTransform] | None = None,
    ) -> None:
        self._ds = dataset
        self._grid = grid
        self._normalization = normalization
        self._species = tuple(species) if species is not None else ()
        self._physics = physics if physics is not None else PhysicsParams()
        self._metadata = dict(metadata) if metadata is not None else {}
        self._frame = frame
        self._transforms = dict(transforms) if transforms is not None else {}

        merged = _default_aliases(grid.geometry)
        if aliases:
            merged.update(aliases)
        # Generate species-name aliases for every per-species canonical
        # actually in the dataset (n_electrons→n_s0, P_ions→P_s1, etc.).
        from pypic._aliases import species_name_aliases as _species_name_aliases

        merged.update(
            _species_name_aliases(
                tuple(sp.name for sp in self._species),
                self._ds.data_vars,
            )
        )
        # Bidirectional alias filter.  First pass — forward direction:
        # ``alias→canonical`` when the canonical is stored.  Reverse:
        # ``canonical→alias`` when the data is stored under what's now
        # considered the alias (e.g. ``Pe`` after the v1.0 cleanup made
        # ``P_s0`` canonical).  Without the reverse direction, a recipe
        # asking for the ``_sN`` form on a dataset that stores the
        # ``e/i`` form would miss.
        data_vars_set = set(self._ds.data_vars)
        filtered: dict[str, str] = {}
        for alias_name, canonical_name in merged.items():
            if canonical_name in data_vars_set:
                filtered[alias_name] = canonical_name
            elif alias_name in data_vars_set:
                filtered.setdefault(canonical_name, alias_name)
        # Second pass — transitive chains: ``P_e → P_s0 → Pe`` collapses
        # to ``P_e → Pe`` when only ``Pe`` is stored.  Iterates to a
        # fixed point in O(|merged|) per round; chain depth is bounded
        # by the alias graph (≤2 in practice).
        for _round in range(3):
            changed = False
            for alias_name, canonical_name in merged.items():
                if alias_name in filtered or alias_name in data_vars_set:
                    continue
                if canonical_name in filtered:
                    filtered[alias_name] = filtered[canonical_name]
                    changed = True
            if not changed:
                break
        self._aliases = filtered

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, FloatArray],
        grid: GridInfo,
        normalization: Normalization | None = None,
        *,
        species: Sequence[SpeciesInfo] | None = None,
        physics: PhysicsParams | None = None,
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
        physics : PhysicsParams | None
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
        >>> ds = FieldDataset.from_arrays({"B_1": np.array([1.0, 2.0])}, grid)
        >>> ds["B_1"]
        array([1., 2.])
        """
        if normalization is None:
            from pypic.units import Normalization as _Norm

            normalization = _Norm.identity()
        dim_names = list(grid.surviving_axis_names)
        coord_arrays = grid.coordinate_arrays()
        coords = {dim_names[i]: coord_arrays[i] for i in range(len(dim_names))}

        from pypic.fields import field_info as _field_info
        from pypic.fields import quantity_dimension as _quantity_dimension

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
                ud = info.unit_dimension or _quantity_dimension(info.quantity_type)
                da.attrs["unit_dimension"] = list(ud)
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
    def physics(self) -> PhysicsParams:
        """Physics parameters."""
        return self._physics

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
        pressure tensors. Scalar values pass through unchanged. When the
        rotation includes an axis swap or reflection, field arrays are
        copied to a contiguous buffer in the new axis order; pure
        translations and identity rotations only touch metadata.

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

        # Determine which axes survive and derive the axis permutation
        # and sign map from the rotation matrix up front. All field
        # arrays then flow through a single reorient → construct pass.
        if self._grid.surviving_axes is not None:
            indices = list(self._grid.surviving_axes)
        else:
            indices = list(range(len(self._grid.dimensions)))
        ndim = len(indices)

        # The grid transformation (transpose + flip) only works for
        # signed permutation matrices (axis swaps and reflections).
        # General rotations would require interpolation onto a new grid.
        rotation_sub = rotation[np.ix_(indices, indices)]
        abs_rotation = np.abs(rotation_sub)
        if not (
            np.allclose(np.sum(abs_rotation, axis=1), 1.0, atol=1e-6)
            and np.allclose(np.sum(abs_rotation, axis=0), 1.0, atol=1e-6)
        ):
            raise NotImplementedError(
                "transform_to() only supports axis-swap/reflection "
                "rotation matrices (signed permutations). General "
                "rotations require grid interpolation (not yet implemented)."
            )

        # Axis permutation (target_i → source local index) and sign.
        axis_permutation: list[int] = []
        axis_signs: list[float] = []
        for target_i in indices:
            row = rotation[target_i, :]
            source_orig = int(np.argmax(np.abs(row[indices])))
            axis_permutation.append(source_orig)
            axis_signs.append(float(np.sign(row[indices[source_orig]])))

        needs_transpose = axis_permutation != list(range(ndim))
        flip_axes = [i for i, s in enumerate(axis_signs) if s < 0]

        def _reorient(arr: FloatArray) -> FloatArray:
            if needs_transpose:
                arr = np.transpose(arr, axis_permutation)
            for ax in flip_axes:
                arr = np.flip(arr, axis=ax)
            if needs_transpose or flip_axes:
                arr = np.ascontiguousarray(arr)
            return arr

        # Accumulate raw numpy arrays — no intermediate DataArray wrapping.
        new_arrays: dict[str, FloatArray] = {}
        rotated: set[str] = set()

        for n1, n2, n3 in find_vector_triplets(self.field_names()):
            r1, r2, r3 = rotate_vector_components(
                self[n1], self[n2], self[n3], rotation
            )
            for name, arr in ((n1, r1), (n2, r2), (n3, r3)):
                new_arrays[name] = _reorient(arr)
                rotated.add(name)

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
            for name, arr in zip((p11, p22, p33, p12, p13, p23), rp, strict=True):
                new_arrays[name] = _reorient(arr)
                rotated.add(name)

        for raw_name in self._ds.data_vars:
            name = str(raw_name)
            if name not in rotated:
                new_arrays[name] = _reorient(self._ds[name].values)

        # Compute new origin, spacing, dimensions from permuted source
        transform_origin = np.array(transform.origin, dtype=np.float64)
        dx_scale = transform.scale
        old_coords = self._grid.coordinate_arrays()
        new_origin_list: list[float] = []
        new_spacing_list: list[float] = []
        new_dims_list: list[int] = []
        for src_i, sign in zip(axis_permutation, axis_signs, strict=True):
            src_coords = old_coords[src_i]
            coord_first = (
                dx_scale * sign * (src_coords[0] - transform_origin[indices[src_i]])
            )
            coord_last = (
                dx_scale * sign * (src_coords[-1] - transform_origin[indices[src_i]])
            )
            dx = dx_scale * self._grid.spacing[src_i]
            new_origin_list.append(float(min(coord_first, coord_last)) - 0.5 * dx)
            new_spacing_list.append(dx)
            new_dims_list.append(self._grid.dimensions[src_i])

        new_surviving = None
        if self._grid.surviving_axes is not None:
            new_surviving = tuple(
                self._grid.surviving_axes[axis_permutation[i]] for i in range(ndim)
            )

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

        # Single Dataset construction: each field is wrapped exactly once.
        new_dim_names = list(new_grid.surviving_axis_names)
        coord_arrays = new_grid.coordinate_arrays()
        coords = {new_dim_names[i]: coord_arrays[i] for i in range(len(new_dim_names))}
        new_ds = xr.Dataset(
            data_vars={
                name: (new_dim_names, arr, dict(self._ds[name].attrs))
                for name, arr in new_arrays.items()
            },
            coords=coords,
        )

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

    def resolve_key(self, key: str) -> str:
        """Resolve a field key through aliases to the canonical name.

        Returns the canonical (data-vars) name for *key*, which may itself
        be a canonical name or an alias from this dataset's alias table.
        Raises :class:`KeyError` (with close-match suggestions) if *key*
        matches neither.

        Parameters
        ----------
        key : str
            Canonical field name or alias.

        Returns
        -------
        str
            Canonical name in ``self._ds.data_vars``.

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(2,), spacing=(1.0,), origin=(0.0,),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B_1": np.ones(2)}, grid, Normalization.identity()
        ... )
        >>> ds.resolve_key("Bx")  # Cartesian alias
        'B_1'
        >>> ds.resolve_key("B_1")  # canonical
        'B_1'
        """
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

    # Backwards-compatible internal alias for code that already imports
    # the underscore name from outside this module. New code should call
    # ``resolve_key`` directly.
    _resolve_key = resolve_key

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
        resolved = self.resolve_key(key)
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
        ...     {"B_1": np.array([1.0, 2.0])}, grid, Normalization.identity(),
        ... )
        >>> ds.has_field("B_1"), ds.has_field("Bx"), ds.has_field("rho")
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
        ...     {"B_1": np.array([1.0, 2.0]), "rho_c": np.array([0.5, 0.5])},
        ...     grid, Normalization.identity(),
        ... )
        >>> sorted(ds.field_names())
        ['B_1', 'rho_c']
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
        ...     {"B_1": np.array([1.0, 2.0]), "B_2": np.array([3.0, 4.0]),
        ...      "rho_c": np.array([0.5, 0.5])},
        ...     grid, Normalization.identity(),
        ... )
        >>> sub = ds.select_fields(["rho_c", "B_1"])
        >>> sub.field_names()
        ['rho_c', 'B_1']
        """
        # Dict-as-ordered-set: preserves caller insertion order while
        # deduplicating on the resolved canonical name. Alphabetical
        # sorting hides both request order and dataset order from the
        # caller, so we keep the order the caller asked for.
        resolved: dict[str, None] = {}
        for name in names:
            resolved[self.resolve_key(name)] = None

        new_ds = self._ds[list(resolved)]
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
        ...     {"B_1": np.full((4,3,2), 3.0),
        ...      "B_2": np.full((4,3,2), 4.0),
        ...      "B_3": np.zeros((4,3,2))},
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
        quantity_type: QuantityType | str | None = None,
        *,
        long_name: str = "",
        latex: str = "",
    ) -> FieldDataset:
        """Return a new FieldDataset with an additional custom field.

        The field's ``quantity_type`` is stored in xarray attrs, so
        ``in_si()``, ``field_info()``, and unit conversion work without
        global ``register_field()`` calls.

        When *quantity_type* is ``None``, metadata is looked up from the
        field registry automatically. For unregistered fields, provide
        *quantity_type* explicitly.

        Parameters
        ----------
        name : str
            Field name.
        data : FloatArray
            Array matching the grid dimensions.
        quantity_type : QuantityType | str | None
            Physical quantity type (e.g. ``QuantityType.VELOCITY``).
            When ``None``, auto-filled from the field registry.
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
            If *quantity_type* is not recognized, or is ``None`` and
            the field name is not in the registry.
        """
        from pypic.fields import _QUANTITY_UNITS
        from pypic.fields import field_info as _field_info
        from pypic.fields import quantity_dimension as _quantity_dimension

        expected = tuple(self._grid.dimensions)
        if data.shape != expected:
            msg = f"Array shape {data.shape} doesn't match grid dimensions {expected}"
            raise ValueError(msg)

        info_ud: tuple[int, int, int, int, int, int, int] | None = None
        if quantity_type is None:
            try:
                info = _field_info(name, axis_names=self._grid.geometry.axis_names)
            except KeyError:
                msg = f"Unknown field {name!r}; provide quantity_type explicitly"
                raise ValueError(msg) from None
            qt = info.quantity_type
            if not long_name:
                long_name = info.long_name
            if not latex:
                latex = info.latex
            info_ud = info.unit_dimension
        else:
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
        da.attrs["unit_dimension"] = list(info_ud or _quantity_dimension(qt))

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

    def with_derived(self, *names: str) -> FieldDataset:
        """Return a new dataset with derived fields computed and stored.

        For each name, computes the field (if not already present),
        looks up metadata from the field registry, and attaches it
        with full metadata. Fields already in the dataset are skipped.

        When a vector-component recipe is encountered (e.g. ``"S_1"``
        from Poynting flux), all sibling components (``"S_2"``, ``"S_3"``)
        are computed from a single function call and stored together.

        Parameters
        ----------
        *names : str
            Derived quantity names (e.g. ``"|B|"``, ``"beta"``).

        Returns
        -------
        FieldDataset
            New dataset with the computed fields attached.

        Examples
        --------
        >>> import numpy as np
        >>> from pypic.units import Normalization
        >>> grid = GridInfo(
        ...     dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0),
        ...     geometry=CARTESIAN,
        ... )
        >>> ds = FieldDataset.from_arrays(
        ...     {"B_1": np.full((4,3,2), 3.0),
        ...      "B_2": np.full((4,3,2), 4.0),
        ...      "B_3": np.zeros((4,3,2))},
        ...     grid, Normalization.identity(),
        ... )
        >>> ds = ds.with_derived("|B|")
        >>> ds.has_field("|B|")
        True
        >>> ds["|B|"][0, 0, 0]
        np.float64(5.0)
        """
        from pypic.compute import (
            _find_sibling_components,
            compute_field,
        )

        result = self
        for name in names:
            if result.has_field(name):
                continue

            # Check for vector-component siblings (e.g. S_1→S_2,S_3)
            siblings = _find_sibling_components(name)
            if siblings:
                # Compute the full vector result once, store all components
                result = result._attach_vector_siblings(name, siblings)
            else:
                data = compute_field(name, result)
                result = result.with_field(name, data)
        return result

    def _attach_vector_siblings(
        self,
        trigger_name: str,
        siblings: dict[str, int],
    ) -> FieldDataset:
        """Compute a tuple-returning function once, store all components.

        Delegates to the shared ``_execute_recipe`` helper so argument
        construction, dependency resolution, and the Cartesian-only
        geometry guard cannot drift from the single-component path in
        :func:`compute_field`.
        """
        from pypic.compute import _execute_recipe, _resolve_name

        canonical = _resolve_name(trigger_name)
        _recipe, full_result = _execute_recipe(canonical, self)

        result = self
        for sibling_name, component_index in siblings.items():
            if not result.has_field(sibling_name):
                result = result.with_field(sibling_name, full_result[component_index])
        return result

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
            resolved = self.resolve_key(name)
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
            resolved = self.resolve_key(name)
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

        Notes
        -----
        Temperature is stored in **energy units** (J), so use
        ``in_units(name, "eV")`` (or ``"keV"``, ``"MeV"``) for the
        plasma working unit, or ``in_units(name, "K")`` for the
        Boltzmann-factor-converted form. Magnetic field defaults to
        T in SI; ``in_units(name, "nT")`` is the space-physics
        idiom. See ``_DISPLAY_UNITS`` in ``pypic.compute`` for the
        full vocabulary.
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
