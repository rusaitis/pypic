"""Shared test fixtures and synthetic-data factories.

These helpers eliminate boilerplate when constructing ``GridInfo`` and
``FieldDataset`` instances from tests. The four factories below cover
the common patterns; reach for them before writing yet another inline
``GridInfo(...)`` + ``FieldDataset.from_arrays(...)`` pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pypic.readers.base import FieldDataset, GridInfo
from pypic.units import Normalization, PhysicsParams, SpeciesInfo

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.types import FloatArray

ELECTRONS = SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256)
IONS = SpeciesInfo(name="ions", charge=1.0, mass=1.0)


def make_test_dataset(
    fields: dict[str, np.ndarray],
    *,
    shape: tuple[int, ...] = (4, 3, 2),
    species: list[SpeciesInfo] | None = None,
    physics: PhysicsParams | None = None,
    normalization: Normalization | None = None,
) -> FieldDataset:
    """Build a FieldDataset with unit spacing for tests."""
    grid = GridInfo(dimensions=shape, spacing=(1.0,) * len(shape))
    return FieldDataset.from_arrays(
        fields,
        grid,
        normalization or Normalization.identity(),
        species=species,
        physics=physics,
    )


def make_uniform_grid(
    *dimensions: int,
    spacing: float | tuple[float, ...] = 1.0,
    origin: float | tuple[float, ...] = 0.0,
) -> GridInfo:
    """Build a uniform Cartesian ``GridInfo`` from positional dimensions.

    ``spacing`` and ``origin`` accept either a scalar (broadcast to every
    axis) or a per-axis tuple. Default geometry is whatever ``GridInfo``
    defaults to (Cartesian).

    Examples
    --------
    >>> g = make_uniform_grid(4, 3, 2, spacing=(1.0, 2.0, 3.0))
    >>> g.dimensions, g.spacing
    ((4, 3, 2), (1.0, 2.0, 3.0))
    """
    ndim = len(dimensions)
    sp = (float(spacing),) * ndim if isinstance(spacing, (int, float)) else spacing
    org = (float(origin),) * ndim if isinstance(origin, (int, float)) else origin
    return GridInfo(
        dimensions=tuple(dimensions),
        spacing=tuple(float(s) for s in sp),
        origin=tuple(float(o) for o in org),
    )


def make_synthetic_fielddataset(
    grid: GridInfo,
    field_names: Iterable[str] = ("B1", "B2", "B3"),
    *,
    seed: int = 42,
    species: list[SpeciesInfo] | None = None,
) -> FieldDataset:
    """Build a ``FieldDataset`` with deterministic standard-normal fields.

    Each named field is filled from a single ``default_rng(seed)`` so the
    output is byte-for-byte reproducible. Use this for plotting tests and
    reader tests where the *values* don't matter, only the *structure*.

    Examples
    --------
    >>> ds = make_synthetic_fielddataset(make_uniform_grid(4, 3, 2))
    >>> sorted(ds.field_names())
    ['B1', 'B2', 'B3']
    """
    rng = np.random.default_rng(seed)
    fields = {name: rng.standard_normal(grid.dimensions) for name in field_names}
    return FieldDataset.from_arrays(
        cast("dict[str, FloatArray]", fields),
        grid,
        Normalization.identity(),
        species=species,
    )


def make_harris_dataset(y_center: float = 7.5) -> FieldDataset:
    r"""Double-Harris current-sheet fixture used by visual plot tests.

    A 40x30x20 uniform Cartesian grid filled with **B**, **E**, **V**
    vectors plus mass density and pressure. The magnetic profile is
    $B_x = \tanh((y - y_0)/2)$ with the standard $\mathrm{sech}^2$
    density enhancement at the midplane. ``y_center`` shifts the sheet,
    convenient for two-run comparison plots.

    Examples
    --------
    >>> ds = make_harris_dataset()
    >>> ds.has_field("B1") and ds.has_field("rho_m")
    True
    """
    x, y, z = np.meshgrid(
        np.linspace(0, 19.5, 40),
        np.linspace(0, 14.5, 30),
        np.linspace(0, 9.5, 20),
        indexing="ij",
    )
    cosh_sq = np.cosh((y - y_center) / 2.0) ** 2
    fields = {
        "B1": np.tanh((y - y_center) / 2.0),
        "B2": 0.1 * np.sin(2 * np.pi * x / 20.0),
        "B3": 0.05 * np.cos(2 * np.pi * z / 10.0),
        "E1": 0.01 * np.sin(np.pi * y / 15.0),
        "E2": -0.02 * np.cos(np.pi * x / 20.0),
        "E3": 0.005 * np.ones_like(x),
        "V1": 0.1 * np.tanh((y - y_center) / 3.0),
        "V2": 0.05 * np.sin(2 * np.pi * x / 20.0),
        "V3": 0.02 * np.cos(np.pi * z / 10.0),
        "rho_m": 1.0 + 0.5 / cosh_sq,
        "P": 0.5 + 0.3 / cosh_sq,
    }
    grid = GridInfo(dimensions=(40, 30, 20), spacing=(0.5, 0.5, 0.5))
    return FieldDataset.from_arrays(
        cast("dict[str, FloatArray]", fields), grid, Normalization.identity()
    )


def make_dipole_dataset(
    *,
    n_cells: int = 80,
    domain_half: float = 6.0,
    moment: float = 1.0,
    planet_radius: float = 1.0,
) -> FieldDataset:
    r"""Analytic 3D magnetic dipole on a centered cubic grid.

    The grid spans $[-L, L]^3$ with $N^3$ cells; cells inside the planet
    ($r < r_p$) are zeroed. Defaults match the pyvista field-line visual
    test (80³ cube, $L = 6$, $r_p = 1$). Uses cell-centered coordinates
    via ``grid.coordinate_arrays()``.

    Examples
    --------
    >>> ds = make_dipole_dataset(n_cells=10, domain_half=2.0)
    >>> ds.grid.dimensions
    (10, 10, 10)
    """
    dx = 2.0 * domain_half / n_cells
    grid = GridInfo(
        dimensions=(n_cells, n_cells, n_cells),
        spacing=(dx, dx, dx),
        origin=(-domain_half, -domain_half, -domain_half),
    )
    x, y, z = np.meshgrid(*grid.coordinate_arrays(), indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r > 0, r, 1.0)
    r5 = r_safe**5
    bx = 3.0 * moment * x * z / r5
    by = 3.0 * moment * y * z / r5
    bz = moment * (3.0 * z**2 - r_safe**2) / r5
    inside = r < planet_radius
    bx[inside] = 0.0
    by[inside] = 0.0
    bz[inside] = 0.0
    return FieldDataset.from_arrays(
        {"B1": bx, "B2": by, "B3": bz}, grid, Normalization.identity()
    )
