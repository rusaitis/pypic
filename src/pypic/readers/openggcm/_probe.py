"""Lightweight format detection for OpenGGCM output directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def probe(path: Path) -> float:
    """Estimate confidence that *path* contains OpenGGCM output.

    Detection signals (additive, capped at 1.0):

    - ``grid.*.dat`` grid file: +0.5
    - ``*.3df.*`` field files: +0.5

    Parameters
    ----------
    path : Path
        Directory to probe.

    Returns
    -------
    float
        Confidence in ``[0.0, 1.0]``.
    """
    if not path.is_dir():
        return 0.0

    score = 0.0

    if next(path.glob("grid.*.dat"), None) is not None:
        score += 0.5

    if next(path.glob("*.3df.*"), None) is not None:
        score += 0.5

    return min(score, 1.0)
