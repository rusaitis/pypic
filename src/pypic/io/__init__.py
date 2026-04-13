"""Zarr v3 export/import, virtual HDF5 views, and Icechunk versioning.

Requires optional dependencies (``zarr``, ``numcodecs``, ``virtualizarr``).
Install with ``pip install pypic[zarr]``.

For Icechunk versioned storage, install ``pip install pypic[icechunk]``.
"""

from pypic.io._icechunk import (
    icechunk_ancestry,
    icechunk_create_tag,
    open_icechunk_repo,
)
from pypic.io._virtual import open_virtual
from pypic.io.zarr import from_zarr, to_zarr, to_zarr_timeseries

__all__ = [
    "from_zarr",
    "icechunk_ancestry",
    "icechunk_create_tag",
    "open_icechunk_repo",
    "open_virtual",
    "to_zarr",
    "to_zarr_timeseries",
]
