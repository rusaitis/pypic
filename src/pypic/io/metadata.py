"""JSON-compatible metadata encoders/decoders for pypic stores.

The public surface here is the shared contract between three internal
consumers: the Zarr writer ([`pypic.io.zarr`][pypic.io.zarr]), the Arrow IPC server
([`pypic.server.arrow`][pypic.server.arrow]), and the HTTP discovery routes
([`pypic.server.routes`][pypic.server.routes]). Each emits or reads the same JSON shapes
documented in :doc:`schema.md` §4.2, so the encode/decode pair lives in
one place to keep round-trip fidelity tight and the storage layout
discriminator (``SCHEMA_VERSION``) single-sourced.

Round-trip fidelity is the design goal: every ``encode`` → ``decode``
cycle must reconstruct an identical object. `to_json_native` and
`from_json_native` handle the value-level coercions (NumPy
scalars, tuples, typed ``StaggerInfo``, non-string-keyed dicts) that
sit underneath the named typed encoders.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from pypic.coordinates.geometry import GEOMETRY_BY_NAME
from pypic.coordinates.transforms import FrameTransform
from pypic.grid import GridInfo
from pypic.units import Normalization, PhysicsParams, SpeciesInfo

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.dataset import FieldDataset


# Storage layout discriminator, written to the root attr ``schema.version``
# and mirroring ``[schema].version`` from ``simulation.toml``: one version
# governs both vocabulary and storage shape (schema.md §4.2).
SCHEMA_VERSION = "1.0"


__all__ = [
    "SCHEMA_VERSION",
    "decode_pypic_attrs",
    "dict_to_grid",
    "dict_to_normalization",
    "dict_to_physics",
    "dict_to_transforms",
    "encode_pypic_attrs",
    "from_json_native",
    "grid_to_dict",
    "list_to_species",
    "normalization_to_dict",
    "physics_to_dict",
    "pop_reserved_metadata",
    "read_simulation_toml",
    "species_to_list",
    "to_json_native",
    "transforms_to_dict",
]


def grid_to_dict(grid: GridInfo) -> dict[str, Any]:
    """Serialize a GridInfo to a JSON-compatible dict."""
    return {
        "dimensions": list(grid.dimensions),
        "spacing": list(grid.spacing),
        "origin": list(grid.origin),
        "geometry": {
            "type": grid.geometry.type.value,
            "axis_names": list(grid.geometry.axis_names),
            "axis_units": list(grid.geometry.axis_units),
        },
        "dt": grid.dt,
        "boundary": list(grid.boundary) if grid.boundary is not None else None,
        "surviving_axes": (
            list(grid.surviving_axes) if grid.surviving_axes is not None else None
        ),
    }


def dict_to_grid(d: dict[str, Any]) -> GridInfo:
    """Reconstruct a GridInfo from a serialized dict."""
    geom_d = d["geometry"]
    geometry = GEOMETRY_BY_NAME[geom_d["type"]]
    # If axis_names differ from the singleton default, we still use the
    # canonical singleton — custom axis names come from frame transforms
    # and are not part of the grid's own geometry identity.
    return GridInfo(
        dimensions=tuple(d["dimensions"]),
        spacing=tuple(d["spacing"]),
        origin=tuple(d["origin"]),
        geometry=geometry,
        dt=d.get("dt"),
        boundary=tuple(d["boundary"]) if d.get("boundary") is not None else None,
        surviving_axes=(
            tuple(d["surviving_axes"]) if d.get("surviving_axes") is not None else None
        ),
    )


def normalization_to_dict(norm: Normalization) -> dict[str, float]:
    """Serialize a Normalization to a JSON-compatible dict."""
    return {
        "length_ref": norm.length_ref,
        "time_ref": norm.time_ref,
        "velocity_ref": norm.velocity_ref,
        "b_field_ref": norm.b_field_ref,
        "e_field_ref": norm.e_field_ref,
        "density_ref": norm.density_ref,
        "mass_ref": norm.mass_ref,
        "charge_ref": norm.charge_ref,
    }


def dict_to_normalization(d: dict[str, float]) -> Normalization:
    """Reconstruct a Normalization from a serialized dict."""
    return Normalization(
        length_ref=d["length_ref"],
        time_ref=d["time_ref"],
        velocity_ref=d["velocity_ref"],
        b_field_ref=d["b_field_ref"],
        e_field_ref=d["e_field_ref"],
        density_ref=d["density_ref"],
        mass_ref=d["mass_ref"],
        charge_ref=d["charge_ref"],
    )


def _species_to_dict(sp: SpeciesInfo) -> dict[str, Any]:
    """Serialize a single SpeciesInfo to a JSON-compatible dict."""
    tv: Any = sp.thermal_velocity
    if isinstance(tv, tuple):
        tv = list(tv)
    dv: Any = sp.drift_velocity
    if isinstance(dv, tuple):
        dv = list(dv)
    ppc: Any = sp.particles_per_cell
    if isinstance(ppc, tuple):
        ppc = list(ppc)
    return {
        "name": sp.name,
        "charge": sp.charge,
        "mass": sp.mass,
        "charge_to_mass": sp.charge_to_mass,
        "temperature": sp.temperature,
        "thermal_velocity": tv,
        "drift_velocity": dv,
        "density": sp.density,
        "particles_per_cell": ppc,
    }


def _dict_to_species(d: dict[str, Any]) -> SpeciesInfo:
    """Reconstruct a single SpeciesInfo from a serialized dict."""
    tv = d.get("thermal_velocity")
    if isinstance(tv, list):
        tv = tuple(tv)
    dv = d.get("drift_velocity")
    if isinstance(dv, list):
        dv = tuple(dv)
    ppc = d.get("particles_per_cell")
    if isinstance(ppc, list):
        ppc = tuple(ppc)
    return SpeciesInfo(
        name=d["name"],
        charge=d.get("charge"),
        mass=d.get("mass"),
        charge_to_mass=d.get("charge_to_mass"),
        temperature=d.get("temperature"),
        thermal_velocity=tv,
        drift_velocity=dv,
        density=d.get("density"),
        particles_per_cell=ppc,
    )


def species_to_list(species: tuple[SpeciesInfo, ...]) -> list[dict[str, Any]]:
    """Serialize a tuple of SpeciesInfo to a JSON-compatible list."""
    return [_species_to_dict(sp) for sp in species]


def list_to_species(lst: list[dict[str, Any]]) -> tuple[SpeciesInfo, ...]:
    """Reconstruct a tuple of SpeciesInfo from a serialized list."""
    return tuple(_dict_to_species(d) for d in lst)


def _encode_c(c: float) -> float | str:
    """Encode the speed of light for JSON: ``math.inf`` → ``"inf"``."""
    return "inf" if math.isinf(c) else c


def _decode_c(raw: Any) -> float:  # noqa: ANN401
    """Decode the speed of light: ``"inf"`` (or missing) → ``math.inf``."""
    return math.inf if raw == "inf" or raw is None else float(raw)


def physics_to_dict(physics: PhysicsParams) -> dict[str, Any]:
    """Serialize PhysicsParams to a JSON-compatible dict.

    ``math.inf`` (used for *c* in non-relativistic MHD) is stored as
    the string ``"inf"`` since JSON has no infinity literal.
    """
    return {
        "gamma": physics.gamma,
        "c": _encode_c(physics.c),
        "relativistic": physics.relativistic,
        "extra": dict(physics.extra),
    }


def dict_to_physics(d: dict[str, Any]) -> PhysicsParams:
    """Reconstruct PhysicsParams from a serialized dict."""
    return PhysicsParams(
        gamma=d["gamma"],
        c=_decode_c(d["c"]),
        relativistic=d["relativistic"],
        extra=d.get("extra", {}),
    )


def _transform_to_dict(t: FrameTransform) -> dict[str, Any]:
    """Serialize a single FrameTransform to a JSON-compatible dict."""
    return {
        "source_frame": t.source_frame,
        "target_frame": t.target_frame,
        "origin": list(t.origin),
        "rotation": [list(row) for row in t.rotation],
        "scale": t.scale,
        "target_axis_names": (
            list(t.target_axis_names) if t.target_axis_names is not None else None
        ),
    }


def _dict_to_transform(d: dict[str, Any]) -> FrameTransform:
    """Reconstruct a single FrameTransform from a serialized dict."""
    return FrameTransform(
        source_frame=d["source_frame"],
        target_frame=d["target_frame"],
        origin=tuple(d["origin"]),
        rotation=tuple(  # type: ignore[arg-type]
            tuple(row) for row in d["rotation"]
        ),
        scale=d.get("scale", 1.0),
        target_axis_names=(
            tuple(d["target_axis_names"])
            if d.get("target_axis_names") is not None
            else None
        ),
    )


def transforms_to_dict(
    transforms: dict[str, FrameTransform],
) -> dict[str, dict[str, Any]]:
    """Serialize frame transforms to a JSON-compatible dict."""
    return {key: _transform_to_dict(t) for key, t in transforms.items()}


def dict_to_transforms(
    d: dict[str, dict[str, Any]],
) -> dict[str, FrameTransform]:
    """Reconstruct frame transforms from a serialized dict."""
    return {key: _dict_to_transform(td) for key, td in d.items()}


def _stagger_to_dict(stagger: Any) -> dict[str, Any]:  # noqa: ANN401
    """Serialize a StaggerInfo to a tagged JSON dict.

    Tagged with ``__pypic_class__`` so ``from_json_native`` can spot
    it during decode and rebuild the dataclass instead of leaving a
    plain dict in ``metadata``.
    """
    field_locations = stagger.field_locations
    position = stagger.position
    return {
        "__pypic_class__": "StaggerInfo",
        "convention": stagger.convention,
        "field_locations": (
            dict(field_locations) if field_locations is not None else None
        ),
        "position": (
            {k: list(v) for k, v in position.items()} if position is not None else None
        ),
        "interpolation_order": stagger.interpolation_order,
        "notes": stagger.notes,
    }


def to_json_native(obj: Any) -> Any:  # noqa: ANN401
    """Recursively coerce arbitrary Python values to JSON-native equivalents.

    HDF5 readers (h5py) commonly hand back attrs as ``numpy.float32`` /
    ``numpy.int64`` / ``numpy.ndarray`` which Zarr's attr validator
    rejects, so user-supplied metadata must be normalized before
    serialization.  Python-native JSON values pass through unchanged.

    Round-trip-preserving tags (``__pypic_class__``) wrap values that
    JSON cannot represent natively without information loss:

    * ``StaggerInfo`` — the typed dataclass readers stamp into metadata.
    * ``tuple`` — distinct from ``list`` in Python; a plain
      ``[a, b]`` round-trip would lose the tuple identity.
    * Dicts with non-string keys — JSON has only string keys, so plain
      stringification silently collides ``{1: ..., "1": ...}``.

    Decode is the inverse via `from_json_native`.
    """
    from pypic.containers import StaggerInfo

    if isinstance(obj, StaggerInfo):
        return _stagger_to_dict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return {
            "__pypic_class__": "tuple",
            "items": [to_json_native(v) for v in obj],
        }
    if isinstance(obj, dict):
        if any(not isinstance(k, str) for k in obj):
            return {
                "__pypic_class__": "keyed_dict",
                "items": [
                    [to_json_native(k), to_json_native(v)] for k, v in obj.items()
                ],
            }
        return {k: to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_native(v) for v in obj]
    # ``bool`` is checked explicitly even though it is a subclass of
    # ``int`` — readers don't pass it specially, but the listing keeps
    # the JSON-native set self-documenting.
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    msg = (
        f"to_json_native does not coerce {type(obj).__name__!r}; "
        f"convert at the caller boundary (ISO string for dates, "
        f"list for sets, etc.) or extend metadata.py"
    )
    raise TypeError(msg)


def _dict_to_stagger(d: dict[str, Any]) -> Any:  # noqa: ANN401
    """Reconstruct a StaggerInfo from a tagged ``to_json_native`` dict."""
    from pypic.containers import StaggerInfo

    raw_position = d.get("position")
    position: dict[str, tuple[float, ...]] | None
    if raw_position is None:
        position = None
    else:
        position = {k: tuple(float(x) for x in v) for k, v in raw_position.items()}
    return StaggerInfo(
        convention=d["convention"],
        field_locations=d.get("field_locations"),
        position=position,
        interpolation_order=d.get("interpolation_order"),
        notes=d.get("notes"),
    )


def from_json_native(obj: Any) -> Any:  # noqa: ANN401
    """Inverse of `to_json_native`: rebuild tagged typed values.

    Walks dicts/lists recursively.  Plain JSON values pass through
    unchanged.  Only the ``__pypic_class__`` marker triggers
    reconstruction.
    """
    if isinstance(obj, dict):
        cls = obj.get("__pypic_class__")
        if cls == "StaggerInfo":
            return _dict_to_stagger(obj)
        if cls == "tuple":
            return tuple(from_json_native(v) for v in obj.get("items", []))
        if cls == "keyed_dict":
            return {
                from_json_native(k): from_json_native(v)
                for k, v in obj.get("items", [])
            }
        return {k: from_json_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_json_native(v) for v in obj]
    return obj


# Metadata keys lifted out of the open ``metadata`` bag into root attrs on
# write, so non-pypic consumers can read them directly (schema.md §4.2);
# ``stagger`` lands under ``grid.stagger``.  Read puts them all back into
# ``metadata``, keeping the Python-side API a single bag.
_RESERVED_METADATA_KEYS: tuple[str, ...] = (
    "model",
    "run",
    "simulation_toml",
    "stagger",
)


def pop_reserved_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Pop the reserved keys (``model``/``run``/``simulation_toml``/``stagger``).

    Returned dict contains the popped values keyed by name; the input
    is mutated in place so the caller is left with the open-bag
    portion safe to pass through `to_json_native`.  Used by both
    `encode_pypic_attrs` and the Zarr timeseries writer so the
    two paths agree on which keys are JSON-typed-elsewhere.
    """
    out: dict[str, Any] = {}
    for key in _RESERVED_METADATA_KEYS:
        value = metadata.pop(key, None)
        if value is not None:
            out[key] = value
    return out


def _encode_run(run: Any) -> dict[str, Any]:  # noqa: ANN401
    """Serialize a ``schema.Run`` (Pydantic model) to a JSON-native dict.

    Accepts either a ``Run`` instance (the canonical form produced by
    the TOML loader) or an already-decoded plain dict (round-trip after
    a previous ``decode_pypic_attrs``).  Returns ``mode="json"`` shape
    so dates land as ISO strings, which Zarr's JSON attr serializer
    accepts.
    """
    from pypic.schema import Run

    if isinstance(run, Run):
        return run.model_dump(mode="json")
    if isinstance(run, dict):
        # Round-trip through model_validate to normalise and re-emit.
        # Cheap and gives the same shape regardless of input dict
        # provenance.
        return Run.model_validate(run).model_dump(mode="json")
    msg = (
        f"metadata['run'] must be a pypic.schema.Run or dict, got {type(run).__name__}"
    )
    raise TypeError(msg)


def _decode_run(d: dict[str, Any]) -> Any:  # noqa: ANN401
    """Rebuild a ``schema.Run`` from a JSON-mode dict.

    Strict by default: a non-conforming ``attrs.run`` is a v1.0 schema
    violation and surfaces as ``pydantic.ValidationError``.  Callers
    that need leniency can catch and fall back to keeping the raw dict.
    """
    from pypic.schema import Run

    return Run.model_validate(d)


def read_simulation_toml(source: str | Path) -> str:
    """Read raw TOML text from a path for the ``simulation_toml=`` writer kwarg.

    Stricter than the loader: validates that *source* points to an
    existing file (the writer would otherwise stamp a misleading
    file-path-shaped string into ``attrs.simulation_toml``).  Returns
    the file's text content; the writer caller stamps it on the
    encoded attrs dict.
    """
    from pathlib import Path as _Path

    p = _Path(source)
    if not p.is_file():
        msg = (
            f"simulation_toml={source!r} is not an existing file. "
            f"Pass a path to a TOML file or attach the text manually "
            f"to fds.metadata['simulation_toml']."
        )
        raise FileNotFoundError(msg)
    return p.read_text(encoding="utf-8")


def encode_pypic_attrs(fds: FieldDataset) -> dict[str, Any]:
    """Assemble FieldDataset metadata into the schema-v1.0 root-group attrs dict.

    The returned dict is stamped as ``xr.DataTree.attrs`` (which writes
    it onto the Zarr root group's attrs) when writing. Each top-level
    key mirrors a §2 ``simulation.toml`` section of the same name —
    ``schema``, ``model``, ``time``, ``grid``, ``boundary_conditions``,
    ``coordinates``, ``normalization``, ``species``, ``physics``,
    ``run``, ``simulation_toml`` — so cross-language consumers can read
    e.g. ``store.attrs["grid"]`` without going through any pypic-
    specific umbrella. See schema.md §4.2 for the full on-disk
    mapping table.

    The ``schema`` root attr carries the version at ``schema.version``,
    matching ``simulation.toml``'s ``[schema].version`` form, and
    discriminates both vocabulary and storage layout in one go.

    Sections derived from in-memory state:
    * ``time`` — ``{ dt }`` only when ``grid.dt`` is set.
    * ``boundary_conditions`` — ``{ lower, upper }`` mirrored from
      ``grid.boundary`` (in-memory carries one tuple per axis; emitted
      as both faces).
    * ``coordinates`` — consolidates ``geometry``, ``frame``, optional
      ``axis_labels``, and ``transforms`` (previously split across
      ``grid``, top-level ``frame``, top-level ``transforms``).
    * ``physics`` — emits ``relativistic`` plus ``gamma_eos`` (from
      ``PhysicsParams.gamma``); ``c`` moves to
      ``normalization.speed_of_light``.

    Sections lifted from ``fds.metadata`` if present (popped, not
    duplicated): ``model``, ``run``, ``simulation_toml``. ``run`` is
    re-encoded through [`pypic.schema.Run`][pypic.schema.Run] to a JSON-mode dump.
    """
    metadata = dict(fds.metadata)
    reserved = pop_reserved_metadata(metadata)
    lifted: dict[str, Any] = {}

    # No reader sets ``metadata['model']`` today; the branch keeps the
    # encoding in one place.  See ``_encode_model``.
    if "model" in reserved:
        lifted["model"] = _encode_model(reserved["model"])
    if "run" in reserved:
        lifted["run"] = _encode_run(reserved["run"])
    if "simulation_toml" in reserved:
        raw_toml = reserved["simulation_toml"]
        if not isinstance(raw_toml, str):
            msg = (
                f"metadata['simulation_toml'] must be a str, "
                f"got {type(raw_toml).__name__}"
            )
            raise TypeError(msg)
        lifted["simulation_toml"] = raw_toml

    # Stagger lives in fds.metadata as a typed dataclass instance.
    # Promote it to ``grid.stagger`` so the on-disk shape mirrors the
    # ``[grid.stagger]`` TOML section and §4.1's ``/grid/stagger/``
    # group, eliminating the only ``metadata`` sub-key with a typed
    # schema home (schema.md §4.2 finding 9).
    grid_attrs = _grid_to_attrs(fds.grid, stagger=reserved.get("stagger"))
    out: dict[str, Any] = {
        "schema": {"version": SCHEMA_VERSION},
        "grid": grid_attrs,
        "coordinates": _coordinates_to_attrs(fds.grid, fds.frame, dict(fds.transforms)),
        "normalization": _normalization_to_attrs(fds.normalization, fds.physics),
        "species": species_to_list(fds.species),
        "physics": _physics_to_attrs(fds.physics),
        "metadata": to_json_native(metadata),
    }
    time_attrs = _time_to_attrs(fds.grid)
    if time_attrs:
        out["time"] = time_attrs
    bc_attrs = _boundary_conditions_to_attrs(fds.grid)
    if bc_attrs:
        out["boundary_conditions"] = bc_attrs
    out.update(lifted)
    return out


def decode_pypic_attrs(
    d: dict[str, Any],
) -> tuple[
    GridInfo,
    Normalization,
    tuple[SpeciesInfo, ...],
    PhysicsParams,
    dict[str, Any],
    str,
    dict[str, FrameTransform],
]:
    """Unpack root-group attrs into the components needed by FieldDataset.

    Expects schema-v1.0 flat keys at the top level with a
    ``schema.version`` discriminator at ``d["schema"]["version"]``.

    Accepts both the v1.0 schema-mirror layout (current) and the
    pre-mirror layout (one transitional shape with ``dt`` /
    ``boundary`` / ``geometry`` under ``grid``, ``frame`` /
    ``transforms`` at top level, ``c`` / ``gamma`` under ``physics``).
    Older stores written before the §4.2 reshape decode through the
    fallback path; new stores written after decode through the typed
    path. Both reconstruct an identical in-memory FieldDataset.

    Returns
    -------
    tuple
        (grid, normalization, species, physics, metadata, frame, transforms)

    Notes
    -----
    Optional top-level ``model``, ``run``, and ``simulation_toml``
    attrs are re-stuffed into the returned ``metadata`` dict under
    their reserved keys, so callers see the same shape they would
    after constructing a FieldDataset from a SimulationConfig via the
    reader registry. Stagger (under ``grid.stagger`` in the new
    layout, or ``metadata.stagger`` in the old) is also re-stuffed
    into ``metadata`` so the in-memory shape stays the same.
    """
    schema_attrs = d.get("schema")
    if not isinstance(schema_attrs, dict) or "version" not in schema_attrs:
        msg = (
            "No pypic metadata found in store attrs "
            "(expected ``schema.version`` discriminator)"
        )
        raise ValueError(msg)
    version = schema_attrs["version"]
    if version != SCHEMA_VERSION:
        msg = (
            f"schema.version={version!r} but this pypic build expects "
            f"{SCHEMA_VERSION!r}; cannot decode safely"
        )
        raise ValueError(msg)

    coords_attrs = d.get("coordinates", {}) or {}
    time_attrs = d.get("time", {}) or {}
    bc_attrs = d.get("boundary_conditions", {}) or {}
    grid = _attrs_to_grid(d["grid"], coords_attrs, time_attrs, bc_attrs)
    normalization = dict_to_normalization(d["normalization"])
    species = list_to_species(d.get("species", []))
    physics = _attrs_to_physics(d.get("physics", {}), d.get("normalization", {}))

    metadata: dict[str, Any] = from_json_native(d.get("metadata", {}))

    # Promote stagger from new-shape ``grid.stagger`` back into
    # ``metadata.stagger`` so the in-memory FieldDataset shape stays
    # unchanged. Old-shape stores keep stagger inside ``metadata``
    # already — from_json_native rebuilt the StaggerInfo above.
    raw_stagger = d["grid"].get("stagger") if isinstance(d.get("grid"), dict) else None
    if raw_stagger is not None and "stagger" not in metadata:
        metadata["stagger"] = _dict_to_stagger(raw_stagger)

    raw_model = d.get("model")
    if raw_model is not None:
        metadata["model"] = raw_model
    raw_run = d.get("run")
    if raw_run is not None:
        metadata["run"] = _decode_run(raw_run)
    raw_toml = d.get("simulation_toml")
    if raw_toml is not None:
        metadata["simulation_toml"] = raw_toml

    # Frame and transforms live under ``coordinates``; the top-level
    # fallback covers old-layout stores.  Trailing ``or {}`` guards an
    # explicit ``null``, which would crash ``dict_to_transforms``.
    frame = coords_attrs.get("frame") or d.get("frame", "simulation")
    raw_transforms = coords_attrs.get("transforms") or d.get("transforms", {}) or {}
    transforms = dict_to_transforms(raw_transforms)
    return grid, normalization, species, physics, metadata, frame, transforms


def _encode_model(model: Any) -> dict[str, Any]:  # noqa: ANN401
    """Serialize a model dict to JSON-native form.

    No typed model container exists in pypic yet (readers store ad-hoc
    strings under ``metadata['model_name']`` / ``metadata['model_type']``);
    this hook accepts a plain dict and normalizes it. When a typed
    ``schema.Model`` lands, swap this for ``Model.model_dump(mode='json')``
    by analogy with `_encode_run`.
    """
    if not isinstance(model, dict):
        msg = f"metadata['model'] must be a dict, got {type(model).__name__}"
        raise TypeError(msg)
    out: dict[str, Any] = to_json_native(model)
    return out


def _grid_to_attrs(grid: GridInfo, *, stagger: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """Encode the grid section per schema.md §4.2 ``attrs.grid``.

    Drops ``dt`` (now under ``attrs.time``), ``boundary`` (now under
    ``attrs.boundary_conditions``), and ``geometry`` (now under
    ``attrs.coordinates``). ``origin`` is replaced by
    ``lower``/``upper`` to mirror §2 [grid].lower / [grid].upper.
    Stagger lifts in from ``fds.metadata['stagger']`` when present.
    """
    lower = list(grid.origin)
    upper = [
        float(grid.origin[i]) + float(grid.spacing[i]) * float(grid.dimensions[i])
        for i in range(len(grid.dimensions))
    ]
    out: dict[str, Any] = {
        "dimensions": list(grid.dimensions),
        "spacing": list(grid.spacing),
        "lower": lower,
        "upper": upper,
    }
    if grid.surviving_axes is not None:
        out["surviving_axes"] = list(grid.surviving_axes)
    if stagger is not None:
        out["stagger"] = _stagger_to_dict(stagger)
    return out


def _attrs_to_grid(
    grid_d: dict[str, Any],
    coords_d: dict[str, Any],
    time_d: dict[str, Any],
    bc_d: dict[str, Any],
) -> GridInfo:
    """Reconstruct a GridInfo from the new-shape attrs sections.

    Reads ``geometry`` from ``coords_d``, ``dt`` from ``time_d``,
    ``boundary`` from ``bc_d``, and the rest from ``grid_d``. Falls
    back to old-shape locations (``grid_d['dt']`` /
    ``grid_d['boundary']`` / ``grid_d['geometry']`` and ``origin``)
    for stores written before the §4.2 reshape.
    """
    # Coordinate origin: prefer new ``lower`` (mirrors §2 [grid].lower);
    # fall back to old ``origin``.
    if "lower" in grid_d:
        origin = tuple(grid_d["lower"])
    else:
        origin = tuple(grid_d.get("origin", ()))

    # Geometry: prefer ``coords_d['geometry']`` (new); fall back to
    # ``grid_d['geometry']`` (old, was a nested dict).
    geom_value = coords_d.get("geometry")
    if geom_value is None:
        geom_value = grid_d.get("geometry")
    if isinstance(geom_value, dict):
        geom_name = geom_value.get("type", "cartesian")
    else:
        geom_name = geom_value or "cartesian"
    geometry = GEOMETRY_BY_NAME[geom_name]

    # dt: prefer ``time_d['dt']`` (new); fall back to ``grid_d['dt']``.
    dt = time_d.get("dt", grid_d.get("dt"))

    # Boundary: prefer ``bc_d['lower']`` (new — single tuple suffices
    # since the in-memory GridInfo carries one tuple per axis); fall
    # back to ``grid_d['boundary']``.
    raw_bound = bc_d.get("lower")
    if raw_bound is None:
        raw_bound = grid_d.get("boundary")
    boundary = tuple(raw_bound) if raw_bound is not None else None

    # surviving_axes is in the same place in both layouts.
    raw_surv = grid_d.get("surviving_axes")

    return GridInfo(
        dimensions=tuple(grid_d["dimensions"]),
        spacing=tuple(grid_d["spacing"]),
        origin=origin,
        geometry=geometry,
        dt=dt,
        boundary=boundary,
        surviving_axes=tuple(raw_surv) if raw_surv is not None else None,
    )


def _time_to_attrs(grid: GridInfo) -> dict[str, Any]:
    """Encode the time section per schema.md §4.2 ``attrs.time``.

    Only ``dt`` round-trips: ``t_start`` / ``t_end`` / ``n_steps`` /
    ``scheme`` live in the source ``simulation.toml`` and never reach
    the typed in-memory FieldDataset.
    """
    out: dict[str, Any] = {}
    if grid.dt is not None:
        out["dt"] = grid.dt
    return out


def _boundary_conditions_to_attrs(grid: GridInfo) -> dict[str, Any]:
    """Encode the boundary_conditions section per schema.md §4.2.

    The in-memory GridInfo carries one boundary tag per axis; both
    faces are emitted with the same value; asymmetric per-face cases
    are not representable.
    """
    if grid.boundary is None:
        return {}
    return {
        "lower": list(grid.boundary),
        "upper": list(grid.boundary),
    }


def _coordinates_to_attrs(
    grid: GridInfo,
    frame: str,
    transforms: dict[str, FrameTransform],
) -> dict[str, Any]:
    """Encode the coordinates section per schema.md §4.2 ``attrs.coordinates``.

    Consolidates ``geometry`` (was in grid), ``frame`` (was top-level),
    and ``transforms`` (was top-level) under one umbrella matching §2
    [coordinates].
    """
    out: dict[str, Any] = {
        "geometry": grid.geometry.type.value,
        "frame": frame,
    }
    out["axis_labels"] = list(grid.geometry.axis_names)
    if transforms:
        out["transforms"] = transforms_to_dict(transforms)
    return out


def _normalization_to_attrs(
    norm: Normalization, physics: PhysicsParams
) -> dict[str, Any]:
    """Encode normalization with ``speed_of_light`` lifted from physics.

    Per schema.md §2 [units].speed_of_light, the speed of light is a
    normalization reference, not a physics flag. ``math.inf`` (the
    non-relativistic sentinel) round-trips as ``"inf"`` since JSON
    has no infinity literal.
    """
    out: dict[str, Any] = dict(normalization_to_dict(norm))
    out["speed_of_light"] = _encode_c(physics.c)
    return out


def _physics_to_attrs(physics: PhysicsParams) -> dict[str, Any]:
    """Encode physics without ``c`` (now in normalization) or top-level ``gamma``.

    ``gamma`` becomes ``gamma_eos`` (the canonical schema name), a
    recognized open-vocabulary key under [physics] per §1. ``extra``
    carries arbitrary code-specific knobs.
    """
    return {
        "relativistic": physics.relativistic,
        "gamma_eos": physics.gamma,
        "extra": dict(physics.extra),
    }


def _attrs_to_physics(
    physics_d: dict[str, Any], norm_d: dict[str, Any]
) -> PhysicsParams:
    """Reconstruct PhysicsParams from new-shape attrs.

    Reads ``c`` from ``norm_d['speed_of_light']`` (new) with a
    fall-back to ``physics_d['c']`` (old). Reads ``gamma`` from
    ``physics_d['gamma_eos']`` (new) with a fall-back to
    ``physics_d['gamma']`` (old). Defaults to PhysicsParams's own
    defaults (``gamma=5/3``, ``c=math.inf``) when neither location
    carries the value — covers minimal stores from non-pypic writers.
    """
    raw_c: Any = norm_d.get("speed_of_light", physics_d.get("c"))
    raw_gamma: Any = physics_d.get("gamma_eos", physics_d.get("gamma"))
    gamma_val: float = 5.0 / 3.0 if raw_gamma is None else float(raw_gamma)
    return PhysicsParams(
        gamma=gamma_val,
        c=_decode_c(raw_c),
        relativistic=physics_d.get("relativistic", False),
        extra=physics_d.get("extra", {}),
    )
