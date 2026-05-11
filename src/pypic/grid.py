"""Grid metadata and geometry-aware field name aliases."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np

from pypic.coordinates.geometry import (
    CARTESIAN,
    CYLINDRICAL,  # noqa: F401 — used in doctests
    SPHERICAL,  # noqa: F401 — used in doctests
    CoordinateGeometry,
    GeometryType,
)

if TYPE_CHECKING:
    from xarray import Dataset

    from pypic.types import FloatArray


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
    ("S", "S"),
    ("u", "u"),  # four-velocity
)


def _build_aliases(
    suffixes: tuple[str, str, str], *, separator: str = ""
) -> dict[str, str]:
    """Generate field name aliases for a coordinate system.

    The canonical RHS is always the Tier-3 form ``<prefix>_<component>``
    (`B_1`, `V_2`, `B0_1`). The alias LHS uses *separator* between
    prefix and coordinate suffix: ``""`` produces the compact form
    ``Bx`` and ``"_"`` produces the underscored form ``B_x``; both
    are user-facing convenience aliases for the same canonical.

    Parameters
    ----------
    suffixes : tuple[str, str, str]
        Coordinate suffixes (e.g. ``("x", "y", "z")``).
    separator : str
        Separator between alias prefix and suffix.
    """
    aliases: dict[str, str] = {}
    for alias_prefix, canonical_prefix in _FIELD_PREFIX_PAIRS:
        for i, suffix in enumerate(suffixes, 1):
            aliases[f"{alias_prefix}{separator}{suffix}"] = (
                f"{canonical_prefix}_{i}"
            )
    return aliases


# Short-form geometry aliases (``Bx → B_1``, ``Br → B_1``).
_CARTESIAN_ALIASES = _build_aliases(("x", "y", "z"))
_SPHERICAL_ALIASES = _build_aliases(("r", "theta", "phi"))
_CYLINDRICAL_ALIASES = _build_aliases(("r", "phi", "z"))

# Underscored geometry aliases (``B_x → B_1``, ``B_r → B_1``).
_CARTESIAN_UNDERSCORE_ALIASES = _build_aliases(("x", "y", "z"), separator="_")
_SPHERICAL_UNDERSCORE_ALIASES = _build_aliases(("r", "theta", "phi"), separator="_")
_CYLINDRICAL_UNDERSCORE_ALIASES = _build_aliases(("r", "phi", "z"), separator="_")

# Scalar underscore aliases (e.g. ``P_e`` is an alternate spelling of
# ``Pe``).  The e/i form is what carries the rich electron/ion-specific
# field metadata in ``_FIELD_INFO``; the universal-canonical flip from
# v1.0.x lives at the compute layer (``_COMPUTE_ALIASES``).  These
# entries keep underscore-spellings working for both labeling and
# storage lookup.
_SCALAR_UNDERSCORE_ALIASES: dict[str, str] = {
    "P_e": "Pe",
    "P_i": "Pi",
    "T_e": "Te",
    "T_i": "Ti",
    "P_11": "P_11",
    "P_12": "P_12",
    "P_13": "P_13",
    "P_22": "P_22",
    "P_23": "P_23",
    "P_33": "P_33",
}

# Species-convenience aliases (geometry-independent).
# Storage-equivalent: same data under two names.  ``Pe`` and ``P_s0`` (etc.)
# refer to the same array — readers that emit one form (e.g. iPIC3D writes
# ``Pe``/``Pi``) satisfy recipes that ask for the other via the dataset's
# bidirectional resolver.
_SPECIES_ALIASES: dict[str, str] = {
    "n_e": "n_s0",
    "n_i": "n_s1",
    "Pe": "P_s0",
    "Pi": "P_s1",
    "Te": "T_s0",
    "Ti": "T_s1",
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
        new_surviving = tuple(
            old_grid.surviving_axes[local_idx] for local_idx, _ in surviving
        )
    else:
        new_surviving = tuple(local_idx for local_idx, _ in surviving)

    new_boundary = None
    if old_grid.boundary is not None:
        new_boundary = tuple(old_grid.boundary[local_idx] for local_idx, _ in surviving)

    return copy.replace(
        old_grid,
        dimensions=tuple(int(new_ds.sizes[name]) for _, name in surviving),
        spacing=tuple(old_grid.spacing[local_idx] for local_idx, _ in surviving),
        origin=tuple(
            # invert cell-center formula: coord[0] = origin + 0.5*spacing
            float(new_ds.coords[name].values[0]) - 0.5 * old_grid.spacing[local_idx]
            for local_idx, name in surviving
        ),
        boundary=new_boundary,
        surviving_axes=new_surviving,
    )
