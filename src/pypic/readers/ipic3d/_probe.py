"""Lightweight format detection for iPIC3D output directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def probe(path: Path) -> float:
    """Estimate confidence that *path* contains iPIC3D output.

    Detection signals (additive, capped at 1.0):

    - ``.inp`` config file: +0.5
    - ``settings.hdf``: +0.4
    - ``*-Fields_*.h5`` (H5hut files): +0.3
    - ``Fields_*`` directories (phdf5): +0.2
    - ``proc*.hdf`` files (shdf5): +0.2

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

    if next(path.glob("*.inp"), None) is not None:
        score += 0.5

    if (path / "settings.hdf").exists():
        score += 0.4

    if next(path.glob("*-Fields_*.h5"), None) is not None:
        score += 0.3

    if score == 0.0:
        # Only check heavier patterns if no strong signal yet
        has_fields_dir = next(
            (d for d in path.iterdir()
             if d.is_dir() and d.name.startswith("Fields_")),
            None,
        )
        if has_fields_dir is not None:
            score += 0.2

        if next(path.glob("proc*.hdf"), None) is not None:
            score += 0.2

    return min(score, 1.0)
