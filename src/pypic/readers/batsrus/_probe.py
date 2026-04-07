"""Lightweight format detection for BATSRUS output directories."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pypic.readers.base import score_signals

if TYPE_CHECKING:
    from pathlib import Path

_BATSRUS_H_RE = re.compile(r"_[tn]\d{8}")

# Pure-glob signals handled by the shared helper.
_CORE_SIGNALS: list[tuple[str, float]] = [
    ("PARAM.in", 0.3),
    ("*.batl", 0.5),
    ("*_pe*.idl", 0.2),
]


def can_read_confidence(path: Path) -> float:
    """Estimate confidence that *path* contains BATSRUS output.

    Detection signals (additive, capped at 1.0):

    - ``PARAM.in``: +0.3
    - ``.batl`` files (HDF5 BATL): +0.5
    - ``*_pe*.idl`` per-cell files: +0.2
    - ``.h`` header files with BATSRUS timestamp pattern: +0.3
    - ``.out`` / ``.outs`` merged files: +0.3 (only if nothing else matched)

    Parameters
    ----------
    path : Path
        Directory to check.

    Returns
    -------
    float
        Confidence in ``[0.0, 1.0]``.
    """
    if not path.is_dir():
        return 0.0

    score = score_signals(path, _CORE_SIGNALS)

    # Filter .h files by BATSRUS timestamp pattern to avoid C header
    # false positives (reader.h, config.h, ...).
    if any(_BATSRUS_H_RE.search(f.name) for f in path.glob("*.h")):
        score += 0.3

    # Merged .out / .outs fall back only if nothing else matched, and
    # contribute a single 0.3 regardless of which variant is present.
    if score == 0.0 and (
        next(path.glob("*.out"), None) is not None
        or next(path.glob("*.outs"), None) is not None
    ):
        score += 0.3

    return min(score, 1.0)
