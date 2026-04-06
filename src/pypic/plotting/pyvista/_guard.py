"""Import guard for pyvista (optional dependency)."""

_HAS_PYVISTA: bool | None = None


def ensure_pyvista() -> None:
    """Raise ``ImportError`` with install hint if pyvista is missing."""
    global _HAS_PYVISTA  # noqa: PLW0603
    if _HAS_PYVISTA is True:
        return
    try:
        import pyvista  # noqa: F401

        _HAS_PYVISTA = True
    except ImportError:
        _HAS_PYVISTA = False
        msg = (
            "pyvista is required for 3D rendering. "
            "Install it with: pip install pypic[3d]"
        )
        raise ImportError(msg) from None
