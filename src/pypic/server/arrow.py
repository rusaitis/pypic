"""Arrow IPC encoding of a :class:`~pypic.dataset.FieldDataset`.

Produces a self-contained Arrow IPC byte stream that webpic (or any
Arrow-aware consumer in JS / Rust / Python) can decode with one call:

* Python: ``pa.ipc.open_stream(bytes).read_all()``
* JavaScript: ``tableFromIPC(bytes)`` (apache-arrow npm)
* Rust: ``arrow_ipc::reader::StreamReader::try_new(&bytes[..], None)``

Each call returns a single ``RecordBatch`` with one column per field
(flattened to a 1-D buffer) plus one coordinate column per surviving
axis.  Original N-D shape, axis names, normalization, and per-field
attrs travel in the schema metadata under the ``b"pypic"`` key as
JSON — mirroring the convention used by
:mod:`pypic.io._arrow` for particles.

Foundation scope: single ``RecordBatch`` per call.  Chunked /
progressive transfer is a TASKS Step 37 follow-up that drops in
without changing the schema metadata shape.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.exceptions import UnknownFieldError
from pypic.io._guard import ensure_arrow
from pypic.io.metadata import (
    SCHEMA_VERSION,
    grid_to_dict,
    normalization_to_dict,
    species_to_list,
    to_json_native,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.dataset import FieldDataset

__all__ = [
    "decode_field_dataset_ipc",
    "field_dataset_to_arrow_ipc",
]

# Schema metadata key.  Same byte-string the particle encoder uses,
# so consumers that already speak the pypic Arrow convention pick it
# up without a second branch.
_PYPIC_META_KEY = b"pypic"


def field_dataset_to_arrow_ipc(
    fds: FieldDataset,
    *,
    fields: Iterable[str] | None = None,
    units: str = "code",
) -> bytes:
    r"""Encode a FieldDataset as an Arrow IPC stream.

    Parameters
    ----------
    fds : FieldDataset
        Source dataset.  May be 1-D, 2-D, or 3-D after slicing /
        reduction — the encoder reads the surviving shape from
        ``fds.grid.dimensions`` and ``fds.xr.dims``.
    fields : iterable of str, optional
        Subset of canonical field names to encode.  ``None`` encodes
        every data variable on the dataset.  Unknown names raise
        :class:`KeyError`.
    units : {"code", "si"}
        ``"code"`` (default) preserves the dataset's stored numeric
        values (code units, the round-trippable form).  ``"si"`` walks
        each requested field, calls :meth:`FieldDataset.in_si` to
        produce SI values, and stamps ``units="si"`` on the schema
        metadata so the consumer knows not to multiply by the
        normalization references again.

    Returns
    -------
    bytes
        Complete Arrow IPC stream (schema + RecordBatch + EOS marker).
        Decodable in one shot by any Arrow consumer.

    Raises
    ------
    ImportError
        ``pyarrow`` is not installed (install with
        ``pip install pypic-plasma[server]``).
    UnknownFieldError
        Any name in *fields* is absent from the dataset.
        Subclasses ``KeyError``.
    ValueError
        ``units`` is not one of ``"code"`` or ``"si"``.
    """
    if units not in ("code", "si"):
        msg = f"units must be 'code' or 'si', got {units!r}"
        raise ValueError(msg)

    ensure_arrow()
    import pyarrow as pa

    selected = _resolve_fields(fds, fields)
    dims = tuple(str(d) for d in fds.xr.dims)

    columns: dict[str, pa.Array] = {}
    field_attrs: dict[str, dict[str, Any]] = {}

    for name in selected:
        da = fds.xr[name]
        values = fds.in_si(name) if units == "si" else da.values
        # Flatten to 1-D so the RecordBatch sees a single buffer per
        # field.  Original N-D shape is recovered from schema metadata.
        columns[name] = pa.array(values.ravel(order="C"))
        field_attrs[name] = _serialize_field_attrs(dict(da.attrs))

    # Coordinate arrays as additional named columns.  Each surviving
    # axis becomes one column.  Length differs from the field columns
    # (which are flattened products of dims), so they go into a
    # separate batch — but to keep the foundations API single-shot,
    # we pad coordinate columns into one schema metadata slot rather
    # than emitting two batches.
    #
    # Coordinates must be finite: the schema_meta payload is later
    # ``json.dumps``-ed, and NaN/inf would either emit non-strict JSON
    # (``"NaN"``, ``"Infinity"``) or silently mislead JS/Rust clients.
    coord_arrays: dict[str, list[float]] = {}
    for dim in dims:
        if dim in fds.xr.coords:
            values = fds.xr.coords[dim].values
            if not np.isfinite(values).all():
                msg = (
                    f"Coordinate {dim!r} contains non-finite values; "
                    "the Arrow IPC wire format requires finite coordinates."
                )
                raise ValueError(msg)
            coord_arrays[dim] = [float(v) for v in values]

    schema_meta = {
        _PYPIC_META_KEY: json.dumps(
            _build_schema_metadata(
                fds=fds,
                dims=dims,
                field_attrs=field_attrs,
                coord_arrays=coord_arrays,
                units=units,
            )
        ).encode("utf-8")
    }

    arrays = list(columns.values())
    names = list(columns.keys())
    schema = pa.schema(
        [pa.field(n, a.type) for n, a in zip(names, arrays, strict=True)],
        metadata=schema_meta,
    )
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return bytes(sink.getvalue().to_pybytes())


def decode_field_dataset_ipc(ipc_bytes: bytes) -> dict[str, Any]:
    """Decode an IPC stream produced by :func:`field_dataset_to_arrow_ipc`.

    Convenience helper for tests and Python consumers.  Returns a dict
    with reconstructed N-D field arrays, the schema metadata, and the
    coordinate arrays.  Production consumers (JS / Rust) parse the
    IPC bytes directly via their own Arrow stack.

    Parameters
    ----------
    ipc_bytes : bytes
        Output of :func:`field_dataset_to_arrow_ipc`.

    Returns
    -------
    dict
        Keys: ``fields`` (mapping name → N-D NumPy array reshaped to
        the original ``shape``), ``coords`` (mapping dim → 1-D array),
        ``metadata`` (the parsed JSON from schema metadata).
    """
    ensure_arrow()
    import pyarrow as pa

    reader = pa.ipc.open_stream(ipc_bytes)
    table = reader.read_all()

    raw_meta = (table.schema.metadata or {}).get(_PYPIC_META_KEY)
    if raw_meta is None:
        msg = "Arrow IPC stream missing pypic schema metadata"
        raise ValueError(msg)
    meta = json.loads(raw_meta)

    shape = tuple(meta["shape"])
    fields_out: dict[str, np.ndarray] = {}
    for name in table.column_names:
        flat = table.column(name).to_numpy(zero_copy_only=False)
        fields_out[name] = flat.reshape(shape)

    coords_out: dict[str, np.ndarray] = {
        dim: np.asarray(arr, dtype=np.float64)
        for dim, arr in meta.get("coords", {}).items()
    }

    return {"fields": fields_out, "coords": coords_out, "metadata": meta}


def _resolve_fields(
    fds: FieldDataset,
    fields: Iterable[str] | None,
) -> list[str]:
    """Resolve a user field list to canonical names; KeyError on misses.

    Honors the same aliasing rules as ``fds.resolve_key`` so callers
    can request ``"Bx"`` and receive the underlying ``"B_1"`` column.
    """
    if fields is None:
        return [str(n) for n in fds.xr.data_vars]
    canonical: list[str] = []
    unresolved: list[str] = []
    for name in fields:
        try:
            canonical.append(fds.resolve_key(name))
        except KeyError:
            unresolved.append(name)
    if unresolved:
        # UnknownFieldError, not bare KeyError: it subclasses KeyError, so
        # callers are unaffected, but it keeps the 404 / "unknown_field"
        # routing that app.py and stream.py dispatch on.
        msg = f"Unknown field(s) in Arrow encoder: {unresolved!r}"
        raise UnknownFieldError(msg)
    return canonical


def _serialize_field_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single field's attrs into a JSON-safe dict.

    Drops xarray-internal keys that have no cross-tool meaning (e.g.
    chunk encoding state) and passes the documented metadata through
    via the shared :func:`pypic.io.metadata.to_json_native` coercer so
    NumPy scalars, tuples, and the ``reduction`` provenance dict
    round-trip cleanly.
    """
    keep = {
        k: v
        for k, v in attrs.items()
        if k
        in {
            "quantity_type",
            "si_unit",
            "latex",
            "long_name",
            "unit_dimension",
            "reduction",
        }
    }
    return {k: to_json_native(v) for k, v in keep.items()}


def _build_schema_metadata(
    *,
    fds: FieldDataset,
    dims: tuple[str, ...],
    field_attrs: dict[str, dict[str, Any]],
    coord_arrays: dict[str, list[float]],
    units: str,
) -> dict[str, Any]:
    """Assemble the JSON payload that goes under the ``b"pypic"`` key."""
    return {
        "schema_version": SCHEMA_VERSION,
        "shape": list(fds.grid.dimensions),
        "dims": list(dims),
        "units": units,
        "grid": grid_to_dict(fds.grid),
        "normalization": normalization_to_dict(fds.normalization),
        "species": species_to_list(fds.species),
        "coords": coord_arrays,
        "fields": field_attrs,
    }
