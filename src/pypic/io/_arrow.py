"""In-memory Arrow interchange for ParticleData.

Zero-copy (where possible) conversion between ``ParticleData`` and
``pyarrow.Table``.  Species metadata travels in the Arrow schema
metadata under key ``b"pypic"``.

Requires optional dependency: ``pyarrow>=17.0``.
Install with ``pip install pypic-plasma[arrow]``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.containers import ParticleData
from pypic.io._guard import ensure_arrow

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["particles_from_arrow", "particles_to_arrow"]

_POSITION_COLS = ("x", "y", "z")
_VELOCITY_COLS = ("vx", "vy", "vz")


def _encode_species_meta(data: ParticleData) -> bytes:
    """Serialize species metadata to JSON bytes for Arrow schema."""
    from pypic.io.metadata import to_json_native

    meta: dict[str, Any] = {
        "species_index": data.species_index,
        "species_name": data.species_name,
        "n_particles": data.n_particles,
    }
    if data.species_charge is not None:
        meta["species_charge"] = data.species_charge
    if data.species_mass is not None:
        meta["species_mass"] = data.species_mass
    raw_metadata = to_json_native(dict(data.metadata))
    if raw_metadata:
        meta["metadata"] = raw_metadata
    return json.dumps(meta).encode("utf-8")


def _decode_species_meta(schema_meta: dict[bytes, bytes] | None) -> dict[str, Any]:
    """Deserialize species metadata from Arrow schema metadata.

    Returns sensible defaults when the ``pypic`` key is absent (e.g.
    tables produced by DuckDB or manual Arrow construction).
    """
    if schema_meta is None:
        return {"species_index": 0, "species_name": "unknown", "n_particles": 0}
    raw = schema_meta.get(b"pypic")
    if raw is None:
        return {"species_index": 0, "species_name": "unknown", "n_particles": 0}
    return json.loads(raw)  # type: ignore[no-any-return]


def inject_species_meta(
    table: pa.Table,
    species_index: int,
    species_name: str,
    *,
    species_charge: float | None = None,
    species_mass: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> pa.Table:
    """Inject species metadata into an Arrow table's schema metadata.

    Used by ``particles_from_dataset`` and ``query_sql`` to attach
    species info extracted from partition columns before calling
    ``particles_from_arrow``.  Optional ``species_charge``,
    ``species_mass``, and ``metadata`` round-trip alongside the
    per-particle data — ``metadata`` preserves the original
    ``ParticleData.metadata`` that was written by the writer.
    """
    payload: dict[str, Any] = {
        "species_index": species_index,
        "species_name": species_name,
        "n_particles": len(table),
    }
    if species_charge is not None:
        payload["species_charge"] = species_charge
    if species_mass is not None:
        payload["species_mass"] = species_mass
    if metadata:
        payload["metadata"] = metadata
    meta = table.schema.metadata or {}
    meta[b"pypic"] = json.dumps(payload).encode("utf-8")
    return table.replace_schema_metadata(meta)


def particles_to_arrow(
    data: ParticleData,
    *,
    position_dtype: str | None = None,
    velocity_dtype: str | None = None,
) -> pa.Table:
    r"""Convert a ``ParticleData`` to a PyArrow Table.

    Column layout: ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``,
    ``weight``, ``id``.  Columns for unloaded fields (e.g. velocity
    when ``data.velocity is None``) are omitted.  Scalar
    ``species_charge``/``species_mass`` travel in schema metadata.

    Species metadata is stored in ``table.schema.metadata[b"pypic"]``
    as a JSON dict.

    Parameters
    ----------
    data : ParticleData
        Source particle data.
    position_dtype : str or None
        Downcast position columns (e.g. ``"float32"``).
        Default: preserve source dtype.
    velocity_dtype : str or None
        Downcast velocity columns (e.g. ``"float32"``).
        Default: preserve source dtype.

    Returns
    -------
    pyarrow.Table

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.containers import ParticleData
    >>> pcl = ParticleData(
    ...     species_index=0, species_name="electrons",
    ...     position=np.zeros((5, 3)), velocity=np.ones((5, 3)),
    ...     n_particles=5, metadata={},
    ...     weight=np.ones(5), species_charge=-1.0, species_mass=1.0,
    ... )
    >>> table = particles_to_arrow(pcl)
    >>> table.num_rows
    5
    """
    ensure_arrow()
    import pyarrow as pa

    arrays: dict[str, pa.Array] = {}

    if data.position is not None:
        for i, col_name in enumerate(_POSITION_COLS):
            col = data.position[:, i]
            if position_dtype is not None and col.dtype != np.dtype(position_dtype):
                col = col.astype(position_dtype)
            arrays[col_name] = pa.array(col)

    if data.velocity is not None:
        for i, col_name in enumerate(_VELOCITY_COLS):
            col = data.velocity[:, i]
            if velocity_dtype is not None and col.dtype != np.dtype(velocity_dtype):
                col = col.astype(velocity_dtype)
            arrays[col_name] = pa.array(col)

    if data.id is not None:
        arrays["id"] = pa.array(data.id)

    if data.weight is not None:
        arrays["weight"] = pa.array(data.weight)

    table = pa.table(arrays)
    meta = table.schema.metadata or {}
    meta[b"pypic"] = _encode_species_meta(data)
    table = table.replace_schema_metadata(meta)
    return table


def particles_from_arrow(table: pa.Table) -> ParticleData:
    r"""Reconstruct a ``ParticleData`` from a PyArrow Table.

    Expects the column layout produced by ``particles_to_arrow``.
    Species metadata is read from ``table.schema.metadata[b"pypic"]``.

    Parameters
    ----------
    table : pyarrow.Table

    Returns
    -------
    ParticleData

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.containers import ParticleData
    >>> pcl = ParticleData(
    ...     species_index=0, species_name="electrons",
    ...     position=np.zeros((5, 3)), velocity=np.ones((5, 3)),
    ...     n_particles=5, metadata={},
    ...     weight=np.ones(5), species_charge=-1.0, species_mass=1.0,
    ... )
    >>> back = particles_from_arrow(particles_to_arrow(pcl))
    >>> (back.n_particles, back.species_charge, back.species_mass)
    (5, -1.0, 1.0)
    """
    ensure_arrow()

    meta = _decode_species_meta(table.schema.metadata or {})

    position: np.ndarray | None = None
    if all(c in table.column_names for c in _POSITION_COLS):
        pos_arrays = [
            table.column(c).to_numpy(zero_copy_only=False) for c in _POSITION_COLS
        ]
        position = np.column_stack(pos_arrays)

    velocity: np.ndarray | None = None
    if all(c in table.column_names for c in _VELOCITY_COLS):
        vel_arrays = [
            table.column(c).to_numpy(zero_copy_only=False) for c in _VELOCITY_COLS
        ]
        velocity = np.column_stack(vel_arrays)

    particle_id: np.ndarray | None = None
    if "id" in table.column_names:
        particle_id = table.column("id").to_numpy(zero_copy_only=False)

    weight: np.ndarray | None = None
    if "weight" in table.column_names:
        weight_raw = table.column("weight").to_numpy(zero_copy_only=False)
        weight = (
            weight_raw
            if weight_raw.dtype == np.float64
            else weight_raw.astype(np.float64)
        )

    n_particles = len(table)

    species_charge_val = meta.get("species_charge")
    species_mass_val = meta.get("species_mass")

    from pypic.io.metadata import from_json_native

    return ParticleData(
        species_index=meta["species_index"],
        species_name=meta["species_name"],
        position=position,
        velocity=velocity,
        n_particles=n_particles,
        metadata=from_json_native(meta.get("metadata", {})),
        id=particle_id,
        weight=weight,
        species_charge=(
            float(species_charge_val) if species_charge_val is not None else None
        ),
        species_mass=(
            float(species_mass_val) if species_mass_val is not None else None
        ),
    )
