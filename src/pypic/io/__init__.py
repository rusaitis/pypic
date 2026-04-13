"""Zarr v3 export/import and virtual HDF5 views for FieldDataset.

Requires optional dependencies (``zarr``, ``numcodecs``, ``virtualizarr``).
Install with ``pip install pypic[zarr]``.
"""

from pypic.io._virtual import open_virtual
from pypic.io.zarr import from_zarr, to_zarr, to_zarr_timeseries

__all__ = ["from_zarr", "open_virtual", "to_zarr", "to_zarr_timeseries"]
