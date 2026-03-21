"""Shared test utilities for pypic tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypic.readers.base import FieldDataset, GridInfo
from pypic.units import Normalization, SpeciesInfo

if TYPE_CHECKING:
    import numpy as np

ELECTRONS = SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256)
IONS = SpeciesInfo(name="ions", charge=1.0, mass=1.0)


def make_test_dataset(
    fields: dict[str, np.ndarray],
    *,
    shape: tuple[int, ...] = (4, 3, 2),
    species: list[SpeciesInfo] | None = None,
    physics: dict | None = None,
    normalization: Normalization | None = None,
) -> FieldDataset:
    """Build a FieldDataset with unit spacing for tests."""
    grid = GridInfo(dimensions=shape, spacing=(1.0,) * len(shape))
    return FieldDataset.from_arrays(
        fields,
        grid,
        normalization or Normalization.identity(),
        species=species,
        physics=physics,
    )
