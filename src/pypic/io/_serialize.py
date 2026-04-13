"""Serialize/deserialize pypic metadata to JSON-compatible dicts.

These converters bridge the gap between pypic's frozen dataclasses and
the JSON-compatible attribute dicts that xarray stores in Zarr metadata.
Round-trip fidelity is the primary design goal: every ``encode`` →
``decode`` cycle must reconstruct an identical object.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pypic.coordinates.geometry import GEOMETRY_BY_NAME
from pypic.coordinates.transforms import FrameTransform
from pypic.grid import GridInfo
from pypic.units import Normalization, PhysicsParams, SpeciesInfo

if TYPE_CHECKING:
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


def encode_pypic_attrs(fds: FieldDataset) -> dict[str, Any]:
    """Assemble all FieldDataset metadata into a JSON-compatible dict.

    The returned dict is stored as ``xr.Dataset.attrs["pypic"]`` when
    writing to Zarr, preserving everything needed to reconstruct the
    FieldDataset on read.
    """
    return {
        "pypic_version": "0.1.0",
        "grid": grid_to_dict(fds.grid),
        "normalization": normalization_to_dict(fds.normalization),
        "species": species_to_list(fds.species),
        "physics": physics_to_dict(fds.physics),
        "metadata": dict(fds.metadata),
        "frame": fds.frame,
        "transforms": transforms_to_dict(dict(fds.transforms)),
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
    """Unpack a pypic attrs dict into the components needed by FieldDataset.

    Returns
    -------
    tuple
        (grid, normalization, species, physics, metadata, frame, transforms)
    """
    grid = dict_to_grid(d["grid"])
    normalization = dict_to_normalization(d["normalization"])
    species = list_to_species(d.get("species", []))
    physics = dict_to_physics(d["physics"])
    metadata = d.get("metadata", {})
    frame = d.get("frame", "simulation")
    transforms = dict_to_transforms(d.get("transforms", {}))
    return grid, normalization, species, physics, metadata, frame, transforms
