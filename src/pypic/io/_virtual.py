"""Virtual Zarr views over existing HDF5 simulation outputs.

Uses VirtualiZarr to extract byte-range metadata from HDF5 files,
creating lightweight references that enable ``xr.open_zarr()``-style
access without copying data.  The original HDF5 files are read
transparently via their byte ranges.

Requires optional dependency ``virtualizarr>=2.4``.
Install with ``pip install pypic[zarr]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import xarray as xr

from pypic.coordinates.geometry import CARTESIAN, GEOMETRY_BY_NAME
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.io._guard import ensure_virtualizarr
from pypic.units import Normalization

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.containers import SimulationConfig
    from pypic.coordinates.transforms import FrameTransform

__all__ = ["open_virtual"]


def _read_metadata_from_h5(
    path: str,
) -> tuple[GridInfo | None, Normalization | None, dict[str, Any]]:
    """Extract pypic-compatible metadata from an HDF5 file.

    Reads the ``grid/`` and ``normalization/`` groups if present
    (pypic canonical layout, schema.md Section 4).  Returns
    ``(grid, normalization, extra_attrs)`` where either may be
    ``None`` if the file lacks the relevant group.
    """
    grid: GridInfo | None = None
    norm: Normalization | None = None
    extra: dict[str, Any] = {}

    with h5py.File(path, "r") as f:
        # Grid metadata
        if "grid" in f:
            g = f["grid"]
            dims = tuple(int(x) for x in g.attrs["dimensions"])
            spacing = tuple(float(x) for x in g.attrs["spacing"])
            origin = tuple(float(x) for x in g.attrs.get("origin", np.zeros(len(dims))))
            geom_str = str(g.attrs.get("geometry", "cartesian"))
            geometry = GEOMETRY_BY_NAME.get(geom_str, CARTESIAN)
            dt_val = g.attrs.get("dt")
            dt = float(dt_val) if dt_val is not None else None
            boundary_raw = g.attrs.get("boundary")
            boundary = (
                tuple(str(b) for b in boundary_raw)
                if boundary_raw is not None
                else None
            )
            grid = GridInfo(
                dimensions=dims,
                spacing=spacing,
                origin=origin,
                geometry=geometry,
                dt=dt,
                boundary=boundary,
            )

        # Normalization metadata
        if "normalization" in f:
            n = f["normalization"]
            norm = Normalization(
                length_ref=float(n.attrs.get("length_ref", 1.0)),
                time_ref=float(n.attrs.get("time_ref", 1.0)),
                velocity_ref=float(n.attrs.get("velocity_ref", 1.0)),
                b_field_ref=float(n.attrs.get("b_field_ref", 1.0)),
                e_field_ref=float(n.attrs.get("e_field_ref", 1.0)),
                density_ref=float(n.attrs.get("density_ref", 1.0)),
                mass_ref=float(n.attrs.get("mass_ref", 1.0)),
                charge_ref=float(n.attrs.get("charge_ref", 1.0)),
            )

        # Root-level scalar metadata
        for key in ("time", "step", "model"):
            if key in f.attrs:
                val = f.attrs[key]
                extra[key] = int(val) if key == "step" else val

    return grid, norm, extra


def _virtual_to_readable(
    vds: xr.Dataset,
) -> xr.Dataset:
    """Convert a VirtualiZarr virtual dataset to a lazy-readable one.

    Persists virtual references as an in-memory Kerchunk dict and
    opens them via fsspec's reference filesystem, giving a normal
    ``xr.Dataset`` that reads from the original files on demand.
    """
    import fsspec

    refs = vds.vz.to_kerchunk(format="dict")
    fs = fsspec.filesystem("reference", fo=refs)
    mapper = fs.get_mapper("")
    return xr.open_dataset(mapper, engine="zarr", consolidated=False)


def open_virtual(
    path: str | Path,
    *,
    fields_group: str | None = "fields",
    drop_variables: list[str] | None = None,
    config: SimulationConfig | None = None,
) -> FieldDataset:
    r"""Create a virtual FieldDataset backed by HDF5 byte ranges.

    Uses VirtualiZarr to extract byte-range metadata from an HDF5
    file, producing a ``FieldDataset`` that lazily reads field data
    from the original file without copying.

    For files following the pypic canonical HDF5 layout (schema.md
    Section 4), grid and normalization metadata are read automatically
    from the ``grid/`` and ``normalization/`` groups.  For other
    layouts, pass a ``config`` with the required metadata.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.
    fields_group : str or None
        HDF5 group containing field datasets.  ``"fields"`` for the
        canonical layout.  ``None`` reads from the root group.
    drop_variables : list[str] or None
        HDF5 dataset names to exclude from the virtual view.
    config : SimulationConfig or None
        Explicit metadata.  When provided, overrides any metadata
        found in the HDF5 file.

    Returns
    -------
    FieldDataset
        A lazy-loading dataset backed by virtual references to the
        original HDF5 file.

    Raises
    ------
    ValueError
        If grid metadata cannot be determined from the file or config.

    Notes
    -----
    Virtual references are persisted via in-memory Kerchunk dicts,
    which use Zarr v2 format internally.  This is the only Zarr v2
    code path in pypic — all other I/O uses Zarr v3.  The Kerchunk
    intermediary is required because VirtualiZarr's native Zarr v3
    virtual-reference backend is Icechunk (Step 24c).  HDF5 datasets
    whose values are all equal to the fill value (typically 0.0) may
    read back incorrectly through this path.
    """
    ensure_virtualizarr()

    from pathlib import Path as _Path

    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser

    path_str = str(_Path(path).resolve())
    url = f"file://{path_str}"

    registry = ObjectStoreRegistry()  # type: ignore[var-annotated]
    registry.register("file://", LocalStore())

    parser = HDFParser(group=fields_group)
    vds = open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        drop_variables=drop_variables,
        loadable_variables=[],
    )

    ds = _virtual_to_readable(vds)

    if config is not None:
        grid = config.grid
        normalization = config.normalization
        species = config.species
        physics = config.physics
        metadata: dict[str, Any] = config.metadata
        frame = config.frame
        transforms: dict[str, FrameTransform] | None = config.transforms
    else:
        h5_grid, h5_norm, h5_extra = _read_metadata_from_h5(path_str)
        if h5_grid is None:
            msg = (
                f"No 'grid' group found in {path_str} and no config provided. "
                "Pass config= with grid metadata, or use a canonical HDF5 layout."
            )
            raise ValueError(msg)
        grid = h5_grid
        normalization = h5_norm if h5_norm is not None else Normalization.identity()
        species = None
        physics = None
        metadata = h5_extra
        frame = "simulation"
        transforms = None

    dim_names = list(grid.surviving_axis_names)
    ds_dims = list(ds.dims)
    if len(ds_dims) != len(dim_names):
        msg = (
            f"Dimension count mismatch: HDF5 has {len(ds_dims)} dimensions "
            f"{ds_dims} but grid expects {len(dim_names)} {dim_names}."
        )
        raise ValueError(msg)
    rename_map = {old: new for old, new in zip(ds_dims, dim_names, strict=True)}
    ds = ds.rename(rename_map)

    coord_arrays = grid.coordinate_arrays()
    coords = {dim_names[i]: coord_arrays[i] for i in range(len(dim_names))}
    ds = ds.assign_coords(coords)

    return FieldDataset(
        ds,
        grid,
        normalization,
        species=species,
        physics=physics,
        metadata=metadata,
        frame=frame,
        transforms=transforms,
    )
