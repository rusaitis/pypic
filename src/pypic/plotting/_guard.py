"""Optional-dependency guard for matplotlib."""

from __future__ import annotations

from pypic._optional import require


def ensure_matplotlib() -> None:
    """Raise ``ImportError`` with install hint if matplotlib is missing."""
    require("matplotlib", extra="plot", feature="pypic.plotting")
