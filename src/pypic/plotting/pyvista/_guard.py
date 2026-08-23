"""Import guard for pyvista (optional dependency)."""

from __future__ import annotations

from pypic._optional import require


def ensure_pyvista() -> None:
    """Raise ``ImportError`` with install hint if pyvista is missing."""
    require("pyvista", extra="3d", feature="3D rendering")
