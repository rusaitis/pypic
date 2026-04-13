"""Optional-dependency guard for zarr and numcodecs."""

from __future__ import annotations

_HAS_ZARR: bool | None = None


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
