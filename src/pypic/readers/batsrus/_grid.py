"""Grid assembly and AMR regridding for BATSRUS data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypic.coordinates import CARTESIAN, CoordinateGeometry
from pypic.grid import GridInfo
from pypic.readers.batsrus._field_map import FIELD_NAME_MAP, SKIP_FIELDS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypic.readers.batsrus._hdf5 import BATLData
    from pypic.types import FloatArray


def _canonical_var_pairs(
    var_names: tuple[str, ...],
) -> list[tuple[int, str]]:
    """Map BATSRUS var names to (index, canonical_name), skipping non-field vars."""
    return [
        (iv, FIELD_NAME_MAP.get(vname, vname))
        for iv, vname in enumerate(var_names)
        if vname not in SKIP_FIELDS
    ]


def _canonical_hdf5_names(
    var_names: tuple[str, ...],
    available: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Map BATSRUS var names to (raw_name, canonical_name) for available HDF5 fields."""
    return [
        (vname, FIELD_NAME_MAP.get(vname, vname))
        for vname in var_names
        if vname not in SKIP_FIELDS and vname in available
    ]


def assemble_uniform_idl(
    coords: FloatArray,
    dx: FloatArray,
    state: FloatArray,
    var_names: tuple[str, ...],
    ndim: int,
    *,
    geometry: CoordinateGeometry = CARTESIAN,
) -> tuple[dict[str, FloatArray], GridInfo]:
    """Assemble a uniform IDL grid into rectangular arrays.

    Parameters
    ----------
    coords
        Cell center coordinates, shape ``(n_cells, 3)``.
    dx
        Cell sizes, shape ``(n_cells,)`` (constant for uniform).
    state
        State variables, shape ``(n_cells, n_var)``.
    var_names
        BATSRUS variable names.
    ndim
        Number of spatial dimensions.
    geometry
        Coordinate geometry for the grid.

    Returns
    -------
    fields : dict[str, FloatArray]
        Canonical field name → rectangular array.
    grid : GridInfo
    """
    cell_dx = float(dx[0])
    axes_min = np.array([coords[:, d].min() for d in range(ndim)])
    axes_max = np.array([coords[:, d].max() for d in range(ndim)])

    dims = tuple(round((axes_max[d] - axes_min[d]) / cell_dx) + 1 for d in range(ndim))
    origin = tuple(float(axes_min[d] - cell_dx / 2) for d in range(ndim))
    spacing = tuple(cell_dx for _ in range(ndim))

    # Compute integer indices for each cell
    indices = tuple(
        np.round((coords[:, d] - float(axes_min[d])) / cell_dx).astype(int)
        for d in range(ndim)
    )

    fields: dict[str, FloatArray] = {}
    for iv, canonical in _canonical_var_pairs(var_names):
        arr = np.full(dims, np.nan, dtype=np.float64)
        arr[indices] = state[:, iv]
        fields[canonical] = arr

    grid = GridInfo(
        dimensions=dims,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
    )
    return fields, grid


def _snap_to_level(unique_dx: FloatArray, target: float) -> float:
    """Snap a target cell size to the nearest AMR level present in the data."""
    idx = int(np.argmin(np.abs(unique_dx - target)))
    return float(unique_dx[idx])


def regrid_amr_idl(
    coords: FloatArray,
    dx: FloatArray,
    state: FloatArray,
    var_names: tuple[str, ...],
    ndim: int,
    *,
    target_dx: float | None = None,
    geometry: CoordinateGeometry = CARTESIAN,
) -> tuple[dict[str, FloatArray], GridInfo]:
    """Regrid AMR IDL data to a uniform rectangular grid.

    By default, regrids to the finest cell size. When ``target_dx`` is
    set, regrids to the nearest AMR level: coarse cells are filled with
    nearest-neighbor repetition, fine cells are block-averaged.

    Parameters
    ----------
    coords, dx, state, var_names, ndim
        Same as `assemble_uniform_idl`.
    target_dx
        Target cell size in code units. When ``None``, uses the finest
        cell size. Snapped to the nearest AMR level present in the data.
    geometry
        Coordinate geometry for the output grid.

    Returns
    -------
    fields : dict[str, FloatArray]
        Canonical field name → rectangular array.
    grid : GridInfo
    """
    if target_dx is not None:
        out_dx = _snap_to_level(np.unique(dx), target_dx)
    else:
        out_dx = float(dx.min())

    # Compute domain boundaries from cell centers and their individual sizes
    cell_lo = coords[:, :ndim] - dx[:, np.newaxis] / 2
    cell_hi = coords[:, :ndim] + dx[:, np.newaxis] / 2
    global_min = np.array([float(cell_lo[:, d].min()) for d in range(ndim)])
    global_max = np.array([float(cell_hi[:, d].max()) for d in range(ndim)])

    dims = tuple(round((global_max[d] - global_min[d]) / out_dx) for d in range(ndim))
    origin = tuple(float(global_min[d]) for d in range(ndim))
    spacing = tuple(out_dx for _ in range(ndim))

    has_fine = bool(np.any(dx < out_dx * (1 - 1e-10)))

    fields: dict[str, FloatArray] = {}

    var_pairs = _canonical_var_pairs(var_names)

    if not has_fine:
        # Fast path: no cells finer than output — fill directly
        for _iv, canonical in var_pairs:
            fields[canonical] = np.full(dims, np.nan, dtype=np.float64)

        for ic in range(len(dx)):
            cell_dx = dx[ic]
            ratio = round(cell_dx / out_dx)
            idx_start = tuple(
                max(
                    0,
                    min(
                        round((coords[ic, d] - cell_dx / 2 - global_min[d]) / out_dx),
                        dims[d] - 1,
                    ),
                )
                for d in range(ndim)
            )
            slices = tuple(
                slice(idx_start[d], min(idx_start[d] + ratio, dims[d]))
                for d in range(ndim)
            )
            for iv, canonical in var_pairs:
                fields[canonical][slices] = state[ic, iv]
    else:
        # Accumulation path: average fine cells, repeat coarse
        sums: dict[str, FloatArray] = {}
        counts = np.zeros(dims, dtype=np.float64)
        for _iv, canonical in var_pairs:
            sums[canonical] = np.zeros(dims, dtype=np.float64)

        for ic in range(len(dx)):
            cell_dx = dx[ic]
            if cell_dx >= out_dx * (1 - 1e-10):
                # Coarse or equal: nearest-neighbor fill
                ratio = round(cell_dx / out_dx)
                idx_start = tuple(
                    max(
                        0,
                        min(
                            round(
                                (coords[ic, d] - cell_dx / 2 - global_min[d]) / out_dx
                            ),
                            dims[d] - 1,
                        ),
                    )
                    for d in range(ndim)
                )
                slices = tuple(
                    slice(idx_start[d], min(idx_start[d] + ratio, dims[d]))
                    for d in range(ndim)
                )
                for iv, canonical in var_pairs:
                    sums[canonical][slices] += state[ic, iv]
                counts[slices] += 1.0
            else:
                # Fine cell: accumulate into containing output cell
                idx = tuple(
                    min(
                        int((coords[ic, d] - global_min[d]) / out_dx),
                        dims[d] - 1,
                    )
                    for d in range(ndim)
                )
                for iv, canonical in var_pairs:
                    sums[canonical][idx] += state[ic, iv]
                counts[idx] += 1.0

        mask = counts > 0
        for canonical, s in sums.items():
            result = np.full(dims, np.nan, dtype=np.float64)
            result[mask] = s[mask] / counts[mask]
            fields[canonical] = result

    grid = GridInfo(
        dimensions=dims,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
    )
    return fields, grid


def assemble_uniform_hdf5(
    batl: BATLData,
    *,
    geometry: CoordinateGeometry = CARTESIAN,
) -> tuple[dict[str, FloatArray], GridInfo]:
    """Assemble uniform-resolution HDF5 blocks into rectangular arrays.

    Parameters
    ----------
    batl
        Parsed ``.batl`` data.
    geometry
        Coordinate geometry for the output grid.

    Returns
    -------
    fields : dict[str, FloatArray]
        Canonical field name → rectangular array.
    grid : GridInfo
    """
    return _assemble_hdf5_blocks(batl, geometry=geometry)


def regrid_amr_hdf5(
    batl: BATLData,
    *,
    target_dx: float | None = None,
    geometry: CoordinateGeometry = CARTESIAN,
) -> tuple[dict[str, FloatArray], GridInfo]:
    """Regrid AMR HDF5 blocks to a uniform rectangular grid.

    By default, regrids to the finest cell size. When ``target_dx`` is
    set, regrids to the nearest AMR level: finer blocks are downsampled
    via block averaging, coarser blocks are upsampled via nearest-neighbor
    repetition.

    Parameters
    ----------
    batl
        Parsed ``.batl`` data with varying refine levels.
    target_dx
        Target cell size in code units. When ``None``, uses the finest
        cell size. Snapped to the nearest AMR level present in the data.
    geometry
        Coordinate geometry for the output grid.

    Returns
    -------
    fields : dict[str, FloatArray]
        Canonical field name → rectangular array.
    grid : GridInfo
    """
    return _assemble_hdf5_blocks(batl, target_dx=target_dx, geometry=geometry)


def _assemble_hdf5_blocks(
    batl: BATLData,
    *,
    target_dx: float | None = None,
    geometry: CoordinateGeometry = CARTESIAN,
) -> tuple[dict[str, FloatArray], GridInfo]:
    """Shared implementation for uniform and AMR HDF5 assembly.

    All blocks are placed into a single rectangular output grid.
    By default, uses the finest resolution. When ``target_dx`` is set,
    uses the nearest AMR level: finer blocks are downsampled via block
    averaging, coarser blocks are upsampled via nearest-neighbor repetition.
    """
    ndim = batl.ndim
    bbox = batl.bounding_box  # (n_blocks, ndim, 2)
    n_blocks = bbox.shape[0]

    if n_blocks == 0:
        dims = batl.block_size[:ndim]
        origin = batl.domain_min
        spacing = tuple(
            (mx - mn) / d
            for mn, mx, d in zip(batl.domain_min, batl.domain_max, dims, strict=True)
        )
        fields: dict[str, FloatArray] = {}
        for _raw, canonical in _canonical_hdf5_names(batl.var_names, batl.fields):
            fields[canonical] = np.full(dims, np.nan, dtype=np.float64)
        grid = GridInfo(
            dimensions=dims, spacing=spacing, origin=origin, geometry=geometry
        )
        return fields, grid

    # Compute cell size per block from bounding boxes and block sizes
    block_dx = np.empty((n_blocks, ndim), dtype=np.float64)
    for d in range(ndim):
        block_dx[:, d] = (bbox[:, d, 1] - bbox[:, d, 0]) / batl.block_size[d]

    if target_dx is not None:
        unique = np.unique(block_dx[:, 0])
        snapped = _snap_to_level(unique, target_dx)
        out_dx = np.full(ndim, snapped)
    else:
        out_dx = np.array([block_dx[:, d].min() for d in range(ndim)])

    global_min = np.array(batl.domain_min, dtype=np.float64)
    global_max = np.array(batl.domain_max, dtype=np.float64)
    dims = tuple(
        round((global_max[d] - global_min[d]) / out_dx[d]) for d in range(ndim)
    )
    origin = tuple(float(global_min[d]) for d in range(ndim))
    spacing = tuple(float(out_dx[d]) for d in range(ndim))

    hdf5_pairs = _canonical_hdf5_names(batl.var_names, batl.fields)

    fields = {}
    for _raw, canonical in hdf5_pairs:
        fields[canonical] = np.full(dims, np.nan, dtype=np.float64)

    # Place each block into the output grid
    for ib in range(n_blocks):
        # Ratios for upsampling (block coarser than output)
        up_ratios = tuple(round(block_dx[ib, d] / out_dx[d]) for d in range(ndim))
        # Ratios for downsampling (block finer than output)
        ds_ratios = tuple(round(out_dx[d] / block_dx[ib, d]) for d in range(ndim))

        is_coarse = any(r > 1 for r in up_ratios)
        is_fine = any(r > 1 for r in ds_ratios)

        # Starting output index for this block
        idx_start = tuple(
            round((bbox[ib, d, 0] - global_min[d]) / out_dx[d]) for d in range(ndim)
        )

        for raw, canonical in hdf5_pairs:
            block_data = batl.fields[raw][ib]

            # HDF5 always stores (nK, nJ, nI) even for 2D (nK=1).
            # Remove the collapsed axis explicitly, then reverse
            # Fortran order to (x, y[, z]).
            if ndim == 3:
                block_data = block_data.transpose(2, 1, 0)
            elif ndim == 2:
                block_data = np.squeeze(block_data, axis=0).T

            if is_coarse:
                # Upsample: nearest-neighbor repeat
                block_data = np.repeat(block_data, up_ratios[0], axis=0)
                if ndim >= 2:
                    block_data = np.repeat(block_data, up_ratios[1], axis=1)
                if ndim >= 3:
                    block_data = np.repeat(block_data, up_ratios[2], axis=2)
            elif is_fine:
                # Downsample: block average
                new_shape: list[int] = []
                mean_axes: list[int] = []
                for d in range(ndim):
                    r = ds_ratios[d]
                    new_shape.extend([block_data.shape[d] // r, r])
                    mean_axes.append(2 * d + 1)
                block_data = block_data.reshape(new_shape).mean(axis=tuple(mean_axes))

            out_slices = tuple(
                slice(idx_start[d], idx_start[d] + block_data.shape[d])
                for d in range(ndim)
            )
            fields[canonical][out_slices] = block_data

    grid = GridInfo(
        dimensions=dims,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
    )
    return fields, grid


def is_uniform_idl(dx: FloatArray) -> bool:
    """Check whether all cells have the same size (uniform grid)."""
    return bool(np.allclose(dx, dx[0], rtol=1e-10))
