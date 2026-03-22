"""Shared field resolution and midplane defaulting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypic.readers.base import FieldDataset
    from pypic.selections import PlaneSelection
    from pypic.types import FloatArray


def resolve_field_values(
    data: FieldDataset, field: str, units: str | None
) -> FloatArray:
    """Resolve a field name to its values, with optional unit conversion.

    Handles stored fields, aliases, and derived quantities. When *units*
    is provided, delegates to ``data.in_units()`` which handles both
    stored and derived fields in a single call.

    Parameters
    ----------
    data : FieldDataset
        Source dataset.
    field : str
        Field name (canonical, alias, or derived).
    units : str | None
        Display units (e.g. ``"nT"``). ``None`` returns code units.

    Returns
    -------
    FloatArray
    """
    if units is not None:
        return data.in_units(field, units)
    if data.has_field(field):
        return data[field]
    return data.compute(field)


def default_midplane(data: FieldDataset) -> PlaneSelection | None:
    """Return a midplane ``PlaneSelection`` for 3D data, ``None`` otherwise.

    The slice is along the last axis (e.g. *z* for Cartesian).

    Parameters
    ----------
    data : FieldDataset
        Input dataset.

    Returns
    -------
    PlaneSelection | None
    """
    if len(data.grid.dimensions) == 3:
        from pypic.selections import PlaneSelection

        normal = data.grid.geometry.axis_names[2]
        return PlaneSelection(normal=normal)
    return None
