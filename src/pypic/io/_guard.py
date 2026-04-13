"""Optional-dependency guards for zarr, numcodecs, virtualizarr, and icechunk."""

from __future__ import annotations

_HAS_ZARR: bool | None = None
_HAS_VIRTUALIZARR: bool | None = None
_HAS_ICECHUNK: bool | None = None


def ensure_zarr() -> None:
    """Raise ``ImportError`` with install hint if zarr or numcodecs are missing."""
    global _HAS_ZARR
    if _HAS_ZARR is True:
        return
    if _HAS_ZARR is False:
        msg = "Missing dependencies for pypic.io. Install with: pip install pypic[zarr]"
        raise ImportError(msg) from None
    missing: list[str] = []
    try:
        import zarr  # noqa: F401
    except ImportError:
        missing.append("zarr>=3.1.0")
    try:
        import numcodecs  # noqa: F401
    except ImportError:
        missing.append("numcodecs>=0.16.0")
    if missing:
        _HAS_ZARR = False
        msg = (
            f"Missing dependencies for pypic.io: {', '.join(missing)}. "
            "Install with: pip install pypic[zarr]"
        )
        raise ImportError(msg) from None
    _HAS_ZARR = True


def ensure_virtualizarr() -> None:
    """Raise ``ImportError`` with install hint if virtualizarr is missing."""
    global _HAS_VIRTUALIZARR
    if _HAS_VIRTUALIZARR is True:
        return
    if _HAS_VIRTUALIZARR is False:
        msg = (
            "Missing virtualizarr for pypic.io.open_virtual. "
            "Install with: pip install pypic[zarr]"
        )
        raise ImportError(msg) from None
    try:
        import virtualizarr  # noqa: F401
    except ImportError:
        _HAS_VIRTUALIZARR = False
        msg = (
            "virtualizarr is required for pypic.io.open_virtual. "
            "Install with: pip install pypic[zarr]"
        )
        raise ImportError(msg) from None
    _HAS_VIRTUALIZARR = True


def ensure_icechunk() -> None:
    """Raise ``ImportError`` with install hint if icechunk is missing."""
    global _HAS_ICECHUNK
    if _HAS_ICECHUNK is True:
        return
    if _HAS_ICECHUNK is False:
        msg = (
            "Missing icechunk for Icechunk backend. "
            "Install with: pip install pypic[icechunk]"
        )
        raise ImportError(msg) from None
    try:
        import icechunk  # noqa: F401
    except ImportError:
        _HAS_ICECHUNK = False
        msg = (
            "icechunk is required for the Icechunk storage backend. "
            "Install with: pip install pypic[icechunk]"
        )
        raise ImportError(msg) from None
    _HAS_ICECHUNK = True


def has_icechunk() -> bool:
    """Return ``True`` if icechunk is importable (non-raising check)."""
    global _HAS_ICECHUNK
    if _HAS_ICECHUNK is not None:
        return _HAS_ICECHUNK
    try:
        import icechunk  # noqa: F401
    except ImportError:
        _HAS_ICECHUNK = False
        return False
    _HAS_ICECHUNK = True
    return True
