"""Shared helpers for readers: the ``simulation.toml`` merge and SI normalization.

Each reader assembles its own ``SimulationConfig`` from native config files
(``.inp``, ``PARAM.in``, OpenGGCM grid files, ...). The grid construction,
species inference, and physics dicts are reader-specific and intentionally
*not* abstracted. What they all share is the optional ``simulation.toml``
override layer and the step from SI-valued arrays to code units.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING

from pypic.containers import SimulationConfig
from pypic.exceptions import UnknownFieldError
from pypic.fields import field_info

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import FloatArray
    from pypic.units import Normalization

# Owned by the native config files; ``simulation.toml`` never overrides them.
READER_OWNED_FIELDS = frozenset(
    {"model_name", "model_type", "grid", "species", "physics"}
)


def merge_simulation_toml(
    sim_dir: Path | None, base: SimulationConfig
) -> SimulationConfig:
    """Merge ``simulation.toml`` overrides into *base* if present.

    Every ``SimulationConfig`` field outside `READER_OWNED_FIELDS` is taken
    from the TOML when it is set there (``normalization``, ``frame``,
    ``transforms``, ``run``, ``probes``, ...); ``metadata`` is merged with
    TOML keys winning on conflict. Reader-owned fields come from *base*
    unchanged, so the native config stays authoritative for the grid and
    species the data was actually produced with.

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
    # depends on the core containers, which depend on nothing reader-specific.
    from pypic.readers.config import load_config

    toml_config = load_config(toml_path)
    overrides: dict[str, object] = {
        "metadata": {**dict(base.metadata), **dict(toml_config.metadata)}
    }
    for spec in dataclasses.fields(SimulationConfig):
        if spec.name in READER_OWNED_FIELDS or spec.name == "metadata":
            continue
        value = getattr(toml_config, spec.name)
        if value is not None and value != () and value != {}:
            overrides[spec.name] = value
    return copy.replace(base, **overrides)


def normalize_fields(
    fields: dict[str, FloatArray], normalization: Normalization
) -> dict[str, FloatArray]:
    """Convert SI-valued canonical fields to code units.

    Each field is divided by the SI factor of its registered quantity
    type. Identity normalization returns *fields* unchanged.

    Raises
    ------
    UnknownFieldError
        If a field has no registered metadata and *normalization* is not
        the identity — its unit is unknown, so it cannot be normalized.
    """
    if normalization.is_identity:
        return fields
    out: dict[str, FloatArray] = {}
    for name, data in fields.items():
        try:
            quantity = field_info(name).quantity_type
        except UnknownFieldError:
            msg = f"Cannot normalize {name!r}: no registered quantity type"
            raise UnknownFieldError(msg) from None
        out[name] = data / normalization.si_factor(quantity)
    return out
