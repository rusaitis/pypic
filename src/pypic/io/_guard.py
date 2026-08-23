"""Optional-dependency guards for I/O backends.

Covers zarr, numcodecs, virtualizarr, icechunk, pyarrow, and duckdb.
"""

from __future__ import annotations

from pypic._optional import require


def ensure_zarr() -> None:
    """Raise ``ImportError`` with install hint if zarr or numcodecs are missing."""
    require("zarr", "numcodecs", extra="zarr", feature="pypic.io")


def ensure_virtualizarr() -> None:
    """Raise ``ImportError`` with install hint if virtualizarr is missing."""
    require("virtualizarr", extra="zarr", feature="pypic.io.open_virtual")


def ensure_icechunk() -> None:
    """Raise ``ImportError`` with install hint if icechunk is missing."""
    require("icechunk", extra="icechunk", feature="The Icechunk storage backend")


def ensure_arrow() -> None:
    """Raise ``ImportError`` with install hint if pyarrow is missing."""
    require("pyarrow", extra="arrow", feature="pypic.io particle I/O")


def ensure_duckdb() -> None:
    """Raise ``ImportError`` with install hint if duckdb is missing."""
    require("duckdb", extra="duckdb", feature="pypic.io.query_sql")
