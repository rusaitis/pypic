"""Serialize/deserialize pypic metadata to JSON-compatible dicts.

These converters bridge the gap between pypic's frozen dataclasses and
the JSON-compatible attribute dicts that xarray stores in Zarr metadata.
Round-trip fidelity is the primary design goal: every ``encode`` →
``decode`` cycle must reconstruct an identical object.
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


def physics_to_dict(physics: PhysicsParams) -> dict[str, Any]:
    """Serialize PhysicsParams to a JSON-compatible dict.

    ``math.inf`` (used for *c* in non-relativistic MHD) is stored as
    the string ``"inf"`` since JSON has no infinity literal.
    """
    c_val: float | str = physics.c
    if math.isinf(physics.c):
        c_val = "inf"
    return {
        "gamma": physics.gamma,
        "c": c_val,
        "relativistic": physics.relativistic,
        "extra": dict(physics.extra),
    }


def dict_to_physics(d: dict[str, Any]) -> PhysicsParams:
    """Reconstruct PhysicsParams from a serialized dict."""
    c_val = d["c"]
    if c_val == "inf":
        c_val = math.inf
    return PhysicsParams(
        gamma=d["gamma"],
        c=float(c_val),
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

    Tagged with ``__pypic_class__`` so ``_from_json_native`` can spot
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


def _to_json_native(obj: Any) -> Any:  # noqa: ANN401
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

    Decode is the inverse via ``_from_json_native``.
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
            "items": [_to_json_native(v) for v in obj],
        }
    if isinstance(obj, dict):
        if any(not isinstance(k, str) for k in obj):
            return {
                "__pypic_class__": "keyed_dict",
                "items": [
                    [_to_json_native(k), _to_json_native(v)] for k, v in obj.items()
                ],
            }
        return {k: _to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_native(v) for v in obj]
    return obj


def _dict_to_stagger(d: dict[str, Any]) -> Any:  # noqa: ANN401
    """Reconstruct a StaggerInfo from a tagged dict written by ``_to_json_native``."""
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


def _from_json_native(obj: Any) -> Any:  # noqa: ANN401
    """Inverse of ``_to_json_native``: rebuild tagged typed values.

    Walks dicts/lists recursively.  Plain JSON values pass through
    unchanged.  Only the ``__pypic_class__`` marker triggers
    reconstruction.
    """
    if isinstance(obj, dict):
        cls = obj.get("__pypic_class__")
        if cls == "StaggerInfo":
            return _dict_to_stagger(obj)
        if cls == "tuple":
            return tuple(_from_json_native(v) for v in obj.get("items", []))
        if cls == "keyed_dict":
            return {
                _from_json_native(k): _from_json_native(v)
                for k, v in obj.get("items", [])
            }
        return {k: _from_json_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json_native(v) for v in obj]
    return obj


# Storage layout discriminator.  Mirrors ``[schema].version`` from
# ``simulation.toml`` — the storage layout (Zarr DataTree with fields
# under ``/fields`` and metadata flat at root) is part of the schema
# v1.0 contract documented in schema.md §4.2.  Bumping the schema
# version is the single coordinated way to evolve both vocabulary and
# storage shape together.  On disk the value sits at the root attr
# path ``schema.version`` so the key path mirrors the TOML form.
SCHEMA_VERSION = "1.0"


# Reserved metadata keys lifted to top-level root attrs on write.
# ``run`` and ``simulation_toml`` are emitted as siblings of ``schema``,
# ``grid``, etc. so cross-tool consumers (webpic, Rust) can read them
# without going through pypic's open ``metadata`` bag.  On read, both
# are re-stuffed into ``metadata`` so the Python-side API stays a
# single bag (no new typed FieldDataset fields required).  See
# schema.md §4.2 for the on-disk contract.


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
    it onto the Zarr root group's attrs) when writing.  Each section is
    a top-level key so cross-language consumers can read e.g.
    ``store.attrs["grid"]`` without going through any pypic-specific
    umbrella.

    The ``schema`` root attr carries the version at ``schema.version``,
    matching ``simulation.toml``'s ``[schema].version`` form, and
    discriminates both vocabulary and storage layout in one go.  See
    schema.md §1 *Versioning* for the additive-only policy and §4.2
    for the on-disk mapping table that this dict materialises.  Other
    keys are the section dicts produced by the per-section encoders
    (``grid_to_dict``, ``normalization_to_dict``, ...).

    ``run`` (typed ``[run]`` provenance: ``schema.Run`` model dump) and
    ``simulation_toml`` (verbatim source TOML text) are lifted out of
    :attr:`FieldDataset.metadata` and emitted as top-level keys when
    present, so non-pypic consumers see them at ``attrs.run`` /
    ``attrs.simulation_toml`` instead of buried in the loose metadata
    bag.  Both keys are optional.
    """
    metadata = dict(fds.metadata)
    lifted: dict[str, Any] = {}
    raw_run = metadata.pop("run", None)
    if raw_run is not None:
        lifted["run"] = _encode_run(raw_run)
    raw_toml = metadata.pop("simulation_toml", None)
    if raw_toml is not None:
        if not isinstance(raw_toml, str):
            msg = (
                f"metadata['simulation_toml'] must be a str, "
                f"got {type(raw_toml).__name__}"
            )
            raise TypeError(msg)
        lifted["simulation_toml"] = raw_toml
    return {
        "schema": {"version": SCHEMA_VERSION},
        "grid": grid_to_dict(fds.grid),
        "normalization": normalization_to_dict(fds.normalization),
        "species": species_to_list(fds.species),
        "physics": physics_to_dict(fds.physics),
        "metadata": _to_json_native(metadata),
        "frame": fds.frame,
        "transforms": transforms_to_dict(dict(fds.transforms)),
        **lifted,
    }


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

    Returns
    -------
    tuple
        (grid, normalization, species, physics, metadata, frame, transforms)

    Notes
    -----
    Optional top-level ``run`` and ``simulation_toml`` attrs (see
    :func:`encode_pypic_attrs`) are re-stuffed into the returned
    ``metadata`` dict under their reserved keys, so callers see the
    same shape they would after constructing a FieldDataset from a
    SimulationConfig via the reader registry.
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
    grid = dict_to_grid(d["grid"])
    normalization = dict_to_normalization(d["normalization"])
    species = list_to_species(d.get("species", []))
    physics = dict_to_physics(d["physics"])
    metadata: dict[str, Any] = _from_json_native(d.get("metadata", {}))
    raw_run = d.get("run")
    if raw_run is not None:
        metadata["run"] = _decode_run(raw_run)
    raw_toml = d.get("simulation_toml")
    if raw_toml is not None:
        metadata["simulation_toml"] = raw_toml
    frame = d.get("frame", "simulation")
    transforms = dict_to_transforms(d.get("transforms", {}))
    return grid, normalization, species, physics, metadata, frame, transforms
