"""Lightweight format detection for iPIC3D output directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.readers.base import score_signals

if TYPE_CHECKING:
    from pathlib import Path

# Pure-glob signals handled by the shared helper.
_CORE_SIGNALS: list[tuple[str, float]] = [
    ("*.inp", 0.5),
    ("settings.hdf", 0.4),
    ("*-Fields_*.h5", 0.3),
]


def _has_subdir_prefix(path: Path, prefix: str) -> bool:
    """Check whether *path* contains a subdirectory starting with *prefix*."""
    return any(
        d.is_dir() and d.name.startswith(prefix) for d in path.iterdir()
    )


def can_read_confidence(path: Path) -> float:
    """Estimate confidence that *path* contains iPIC3D output.

    Detection signals (additive, capped at 1.0):

    - ``*.inp`` config file: +0.5
    - ``settings.hdf``: +0.4
    - ``*-Fields_*.h5`` (H5hut files): +0.3
    - ``Fields_*`` subdirectories (phdf5): +0.2 (fallback only)
    - ``proc*.hdf`` files (shdf5): +0.2 (fallback only)
    - ``Moments_*`` subdirectories: +0.15 (reinforcement, only if 0 < score < 0.8)
    - ``Particles_*`` subdirectories: +0.1 (reinforcement, only if 0 < score < 0.8)

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

    # Weaker fallback signals for partial-output directories.
    if score == 0.0:
        if _has_subdir_prefix(path, "Fields_"):
            score += 0.2
        if next(path.glob("proc*.hdf"), None) is not None:
            score += 0.2

    # Moments_/Particles_ subdirs reinforce a positive core score but
    # don't bump a near-full score any further.
    if 0.0 < score < 0.8:
        if _has_subdir_prefix(path, "Moments_"):
            score += 0.15
        if _has_subdir_prefix(path, "Particles_"):
            score += 0.1

    return min(score, 1.0)
