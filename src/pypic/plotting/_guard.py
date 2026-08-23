"""Optional-dependency guard for matplotlib."""

from __future__ import annotations

_HAS_MATPLOTLIB: bool | None = None

_MISSING_MSG = (
    "matplotlib is required for pypic.plotting. "
    "Install it with: pip install pypic-plasma[plot]"
)


def ensure_matplotlib() -> None:
    """Raise ``ImportError`` with install hint if matplotlib is missing."""
    global _HAS_MATPLOTLIB
    if _HAS_MATPLOTLIB is True:
        return
    if _HAS_MATPLOTLIB is False:
        raise ImportError(_MISSING_MSG) from None
    try:
        import matplotlib  # noqa: F401

        _HAS_MATPLOTLIB = True
    except ImportError:
        _HAS_MATPLOTLIB = False
        raise ImportError(_MISSING_MSG) from None
