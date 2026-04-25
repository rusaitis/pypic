"""Shared post-processing helpers for reader ``to_simulation_config()`` builders.

Each reader assembles its own ``SimulationConfig`` from native config files
(``.inp``, ``PARAM.in``, OpenGGCM grid files, ...). The grid construction,
species inference, and physics dicts are reader-specific and intentionally
*not* abstracted. The one piece they can all share is the optional
``simulation.toml`` override layer: every reader directory may carry a
``simulation.toml`` that supplies normalization, frame, transforms, and
extra metadata that the native config doesn't encode.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.containers import SimulationConfig


def merge_simulation_toml(
    sim_dir: Path | None, base: SimulationConfig
) -> SimulationConfig:
    """Merge ``simulation.toml`` overrides into *base* if present.

    If *sim_dir* contains a ``simulation.toml`` file, parse it and override
    the ``normalization``, ``frame``, and ``transforms`` fields of *base*
    with the values from the TOML, and merge the TOML's ``metadata`` into
    *base.metadata* (TOML keys win on conflict). All other fields
    (``model_name``, ``model_type``, ``grid``, ``species``, ``physics``)
    are taken from *base* unchanged.

    Returns *base* unmodified if *sim_dir* is ``None`` or has no
    ``simulation.toml``.

    Parameters
    ----------
    sim_dir : Path | None
        Directory to scan for ``simulation.toml``. ``None`` skips the merge.
    base : SimulationConfig
        Reader-built SimulationConfig to enrich.

    Returns
    -------
    SimulationConfig
        New ``SimulationConfig`` with merged fields, or *base* unchanged.
    """
    if sim_dir is None:
        return base
    toml_path = sim_dir / "simulation.toml"
    if not toml_path.exists():
        return base

    # Local import keeps this module free of circular import risk: config.py
    # depends on base.py, base.py depends on nothing reader-specific.
    from pypic.readers.config import load_config

    toml_config = load_config(toml_path)
    merged_metadata = {**dict(base.metadata), **dict(toml_config.metadata)}
    return copy.replace(
        base,
        normalization=toml_config.normalization,
        frame=toml_config.frame,
        transforms=dict(toml_config.transforms),
        initial_conditions=toml_config.initial_conditions
        if toml_config.initial_conditions is not None
        else base.initial_conditions,
        output=toml_config.output if toml_config.output is not None else base.output,
        bodies=toml_config.bodies if toml_config.bodies else base.bodies,
        drivers=toml_config.drivers if toml_config.drivers else base.drivers,
        restart=toml_config.restart
        if toml_config.restart is not None
        else base.restart,
        metadata=merged_metadata,
    )
