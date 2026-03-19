"""Lightweight format detection for BATSRUS output directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def probe(path: Path) -> float:
    """Estimate confidence that *path* contains BATSRUS output.

    Detection signals (additive, capped at 1.0):

    - ``PARAM.in``: +0.3
    - ``.batl`` files (HDF5 BATL): +0.5
    - ``.h`` header files: +0.3
    - ``*_pe*.idl`` per-cell files: +0.2
    - ``.out`` / ``.outs`` merged files: +0.3

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

    if (path / "PARAM.in").exists():
        score += 0.3

    if next(path.glob("*.batl"), None) is not None:
        score += 0.5

    if next(path.glob("*.h"), None) is not None:
        score += 0.3

    if next(path.glob("*_pe*.idl"), None) is not None:
        score += 0.2

    if score == 0.0:
        # Only check .out/.outs if nothing else matched
        has_out = (
            next(path.glob("*.out"), None) is not None
            or next(path.glob("*.outs"), None) is not None
        )
        if has_out:
            score += 0.3

    return min(score, 1.0)
