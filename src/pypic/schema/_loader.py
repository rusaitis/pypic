"""Entry points for validating simulation.toml against the v1.0 schema."""

from __future__ import annotations

import tomllib
from os import PathLike  # noqa: TC003  (used in runtime isinstance check)
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pypic.schema._models import SimulationSchema


def validate_simulation_toml(
    source: str | bytes | PathLike[str] | dict[str, Any],
) -> SimulationSchema:
    """Validate a pypic v1.0 simulation.toml document.

    Parameters
    ----------
    source
        Either a filesystem path to a .toml file, a str/bytes blob of
        TOML text, or a pre-parsed dict (the output of ``tomllib.loads``).

    Returns
    -------
    SimulationSchema
        The validated, fully-typed root model. All cross-section
        invariants have already been checked.

    Raises
    ------
    pydantic.ValidationError
        When any field fails validation. The error object carries
        precise dotted paths (e.g. ``grid.spacing.0``) for every
        violation, plus the contextual message from any cross-section
        validator.
    FileNotFoundError
        When ``source`` is a path that does not exist.
    tomllib.TOMLDecodeError
        When ``source`` is a path or text that is not valid TOML.

    Examples
    --------
    >>> doc = '''
    ... [schema]
    ... version = "1.0"
    ... [model]
    ... name = "demo"
    ... type = "PIC"
    ... [run]
    ... name = "r0"
    ... [time]
    ... scheme = "fixed"
    ... dt = 0.1
    ... t_start = 0.0
    ... t_end = 1.0
    ... n_steps = 10
    ... [grid]
    ... dimensions = [4, 4, 4]
    ... spacing = [1.0, 1.0, 1.0]
    ... lower = [0.0, 0.0, 0.0]
    ... upper = [4.0, 4.0, 4.0]
    ... [units]
    ... system = "SI"
    ... [coordinates]
    ... geometry = "cartesian"
    ... frame = "sim"
    ... [[species]]
    ... name = "electrons"
    ... charge = -1.0
    ... mass = 1.0
    ... '''
    >>> s = validate_simulation_toml(doc)
    >>> s.model.name
    'demo'
    """
    data = _load_toml(source)
    return SimulationSchema.model_validate(data)


def _load_toml(
    source: str | bytes | PathLike[str] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, bytes):
        return tomllib.loads(source.decode("utf-8"))
    if isinstance(source, str):
        # Ambiguity: is this a path or TOML text? Heuristic: a string that
        # contains a newline or '=' is text; otherwise treat as a path.
        # A pathological filename containing '=' would be misread as text;
        # callers in that situation should pass ``Path(...)`` directly to
        # bypass the heuristic.
        if "\n" in source or "=" in source:
            return tomllib.loads(source)
        return _load_path(Path(source))
    return _load_path(Path(source))


def _load_path(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


__all__ = ["ValidationError", "validate_simulation_toml"]
