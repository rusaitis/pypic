"""Import guard for pyvista (optional dependency)."""

from __future__ import annotations

_HAS_PYVISTA: bool | None = None

_MISSING_MSG = (
    "pyvista is required for 3D rendering. "
    "Install it with: pip install pypic-plasma[3d]"
)


def ensure_pyvista() -> None:
    """Raise ``ImportError`` with install hint if pyvista is missing."""
    global _HAS_PYVISTA
    if _HAS_PYVISTA is True:
        return
    if _HAS_PYVISTA is False:
        raise ImportError(_MISSING_MSG) from None
    try:
        import pyvista  # noqa: F401

        _HAS_PYVISTA = True
    except ImportError:
        _HAS_PYVISTA = False
        raise ImportError(_MISSING_MSG) from None
