"""Lightweight format detection for OpenGGCM output directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.readers.base import score_signals

if TYPE_CHECKING:
    from pathlib import Path

_SIGNALS: list[tuple[str, float]] = [
    ("grid.*.dat", 0.5),
    ("*.3df.*", 0.5),
]


def can_read_confidence(path: Path) -> float:
    """Estimate confidence that *path* contains OpenGGCM output.

    Detection signals (additive, capped at 1.0):

    - ``grid.*.dat`` grid file: +0.5
    - ``*.3df.*`` field files: +0.5

    Parameters
    ----------
    path : Path
        Directory to check.

    Returns
    -------
    float
        Confidence in ``[0.0, 1.0]``.
    """
    return score_signals(path, _SIGNALS)
