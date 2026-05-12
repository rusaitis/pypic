"""Load simulation configuration from a TOML file.

The Pydantic validator in :mod:`pypic.schema` is the authoritative
source of the v1.0 schema — this module is a thin translator from a
validated :class:`~pypic.schema.SimulationSchema` to the internal
dataclasses (:class:`~pypic.containers.SimulationConfig`, :class:`GridInfo`,
:class:`Normalization`, :class:`SpeciesInfo`). All shape validation
happens in the Pydantic layer; this module only maps fields.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import constants

from pypic.containers import SimulationConfig, StaggerInfo
from pypic.coordinates.geometry import GEOMETRY_BY_NAME, CoordinateGeometry
from pypic.coordinates.transforms import FrameTransform
from pypic.grid import GridInfo
from pypic.schema import (
    SimulationSchema,
    UnitsCustom,
    UnitsMHD,
    UnitsPIC,
    UnitsSI,
    validate_simulation_toml,
)
from pypic.units import Normalization, PhysicsParams, SpeciesInfo

if TYPE_CHECKING:
    from pypic.schema import (
        BoundaryConditions,
        Coordinates,
        Grid,
        Physics,
        Species,
        Time,
        Units,
    )

log = logging.getLogger(__name__)

_DEFAULT_SPECIES_PARAMS: dict[str, tuple[float, float]] = {
    "electrons": (constants.m_e, constants.e),
    "ions": (constants.m_p, constants.e),
    "protons": (constants.m_p, constants.e),
}

LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1e3,
    "R_E": 6.371e6,
    "R_S": 6.957e8,  # solar radius (heliophysics convention)
    "R_sun": 6.957e8,  # unambiguous synonym for R_S
    "R_M": 2.4397e6,  # Mercury radius
    "R_J": 6.9911e7,  # Jupiter radius
    "AU": constants.astronomical_unit,
    # "d_i" (ion skin depth) is normalization-dependent, not a constant —
    # resolved against ``Normalization.length_ref`` at apply time.
}


def apply_physical_extent(
    config: SimulationConfig,
    physical_extent: tuple[float, ...],
    physical_extent_unit: str = "m",
) -> SimulationConfig:
    r"""Auto-compute transform scale factors from physical domain extent.

    When a simulation represents a physical domain of known size (e.g.,
    46 R_E across), this function computes the **scale** — the coordinate
    conversion factor from code units to target units (e.g., 0.25 R_E/d_i).
    For transforms with the default ``scale=1.0``, the computed scale is
    applied. For transforms with an explicit scale, consistency is validated.

    If the normalization is not identity, also computes the **shrink factor**
    — how much the physical domain is compressed relative to what the
    normalization implies (e.g., 3.5× for reduced mass ratio PIC).
    A shrink factor of 1.0 means no spatial rescaling.

    Parameters
    ----------
    config : SimulationConfig
        Original simulation configuration.
    physical_extent : tuple[float, ...]
        Domain size per target-frame axis, in *physical_extent_unit*.
    physical_extent_unit : str
        Length unit name. See ``LENGTH_UNITS`` for valid values.

    Returns
    -------
    SimulationConfig
        New config with computed scale factors and metadata.

    Raises
    ------
    ValueError
        If the unit is unknown or the implied scale is not uniform.
    """
    if physical_extent_unit == "d_i":
        # d_i is the natural PIC length unit and equals the simulation's
        # length_ref by construction. Identity normalization makes this a
        # no-op (length_ref = 1), which is fine for grids already in d_i.
        unit_factor = config.normalization.length_ref
    else:
        unit_factor_lookup = LENGTH_UNITS.get(physical_extent_unit)
        if unit_factor_lookup is None:
            valid = ", ".join(sorted(LENGTH_UNITS))
            raise ValueError(
                f"Unknown physical_extent_unit {physical_extent_unit!r}. "
                f"Valid: {valid}, d_i"
            )
        unit_factor = unit_factor_lookup

    grid = config.grid
    grid_extent = tuple(
        d * s for d, s in zip(grid.dimensions, grid.spacing, strict=True)
    )
    computed_scale: float | None = None

    new_transforms: dict[str, FrameTransform] = {}
    for name, transform in config.transforms.items():
        r_abs = np.abs(transform.rotation_matrix)
        rotated = r_abs @ np.array(grid_extent[:3])
        n = min(len(physical_extent), len(rotated))
        scales = np.array(physical_extent[:n]) / rotated[:n]

        computed_scale = float(np.mean(scales))
        for s in scales:
            if abs(s - computed_scale) / abs(computed_scale) > 0.05:  # 5% tolerance
                raise ValueError(
                    f"physical_extent implies non-uniform scale for "
                    f"transform {name!r}: per-axis ratios {scales.tolist()}"
                )

        if transform.scale == 1.0:
            new_transforms[name] = copy.replace(transform, scale=computed_scale)
            log.info(
                "Spatial scaling: %s d_i -> %s %s (scale=%.4f)",
                " x ".join(f"{e:.0f}" for e in grid_extent),
                " x ".join(f"{e:.0f}" for e in physical_extent),
                physical_extent_unit,
                computed_scale,
            )
        else:
            new_transforms[name] = transform
            rel_diff = abs(transform.scale - computed_scale) / abs(computed_scale)
            if rel_diff > 0.05:
                log.warning(
                    "Scale mismatch: transform %r has scale=%.4f, but "
                    "physical_extent implies scale=%.4f (%.1f%% difference)",
                    name,
                    transform.scale,
                    computed_scale,
                    rel_diff * 100,
                )

    new_metadata = dict(config.metadata)
    new_metadata["physical_extent"] = physical_extent
    new_metadata["physical_extent_unit"] = physical_extent_unit

    if computed_scale is not None and not config.normalization.is_identity:
        norm_scale = config.normalization.length_ref / unit_factor
        shrink_factor = computed_scale / norm_scale
        new_metadata.setdefault("scaling", {})["shrink_factor"] = round(
            shrink_factor, 4
        )
        log.info(
            "Shrink factor: %.2fx (d_i = %.1f km, 1 %s = %.1f d_i physical, "
            "%.1f d_i effective)",
            shrink_factor,
            config.normalization.length_ref / 1e3,
            physical_extent_unit,
            unit_factor / config.normalization.length_ref,
            1.0 / computed_scale,
        )

    return copy.replace(
        config,
        transforms=new_transforms,
        metadata=new_metadata,
    )


def load_config(path: Path) -> SimulationConfig:
    """Parse a ``simulation.toml`` file into a SimulationConfig.

    Validates the file against the v1.0 schema
    (:mod:`pypic.schema`) and builds the internal :class:`SimulationConfig`
    from the result.  The raw TOML text is captured and attached to
    ``metadata["simulation_toml"]`` so downstream FieldDataset writers
    can round-trip it verbatim into ``attrs.simulation_toml`` (schema.md
    §4.2) — losslessly preserving sections (``[bodies]``, ``[drivers]``,
    ``[output]``, ``[restart]``, ``[probes]``, ...) that the typed
    SimulationConfig drops on the way to FieldDataset.

    Parameters
    ----------
    path : Path
        Path to a TOML file conforming to the v1.0 schema.

    Returns
    -------
    SimulationConfig
        Fully typed configuration with grid, normalization, species,
        physics, frame, and transforms populated.

    Raises
    ------
    pydantic.ValidationError
        If the document fails schema validation. Dotted field paths in
        the error message point to every violation.
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    schema = validate_simulation_toml(raw_text)
    config = _from_schema(schema)
    # Re-stamp metadata with the verbatim TOML text.  ``_from_schema``
    # already populated typed sections; this is purely additive.
    new_metadata = dict(config.metadata)
    new_metadata["simulation_toml"] = raw_text
    return copy.replace(config, metadata=new_metadata)


def _from_schema(schema: SimulationSchema) -> SimulationConfig:
    geometry = _build_geometry(schema.coordinates)
    grid = _build_grid(schema.grid, schema.time, schema.boundary_conditions, geometry)
    normalization = _build_normalization(schema.units)
    species = tuple(_build_species(s) for s in schema.species)
    physics = _build_physics(schema.physics)
    transforms = _build_transforms(schema.coordinates, schema.coordinates.frame)
    metadata = _build_metadata(schema)

    config = SimulationConfig(
        model_name=schema.model.name,
        model_type=schema.model.type,
        grid=grid,
        normalization=normalization,
        species=species,
        physics=physics,
        frame=schema.coordinates.frame,
        transforms=transforms,
        initial_conditions=schema.initial_conditions,
        output=schema.output,
        bodies=tuple(schema.bodies),
        drivers=tuple(schema.drivers),
        restart=schema.restart,
        run=schema.run,
        probes=tuple(schema.probes),
        collisions=tuple(schema.collisions),
        phase_space=schema.phase_space,
        metadata=metadata,
    )

    if schema.coordinates.physical_extent is not None:
        phys_ext = tuple(float(x) for x in schema.coordinates.physical_extent)
        phys_unit = schema.coordinates.physical_extent_unit or "m"
        config = apply_physical_extent(config, phys_ext, phys_unit)

    return config


def _build_geometry(coords: Coordinates) -> CoordinateGeometry:
    # ``thetaMode`` (FBPIC azimuthal-mode RZ decomposition) describes the
    # *storage* layout; the post-reconstruction physical grid is cylindrical.
    # Mode metadata travels separately on ``[coordinates.modes]`` and is
    # consumed by code-specific readers.
    geometry_key = "cylindrical" if coords.geometry == "thetaMode" else coords.geometry
    geometry = GEOMETRY_BY_NAME[geometry_key]
    if coords.axis_labels is not None:
        # Length-3 invariant is enforced by the Pydantic Coordinates model.
        labels: tuple[str, str, str] = (
            coords.axis_labels[0],
            coords.axis_labels[1],
            coords.axis_labels[2],
        )
        geometry = copy.replace(geometry, axis_names=labels)
    return geometry


def _build_grid(
    grid: Grid,
    time: Time,
    bcs: BoundaryConditions | None,
    geometry: CoordinateGeometry,
) -> GridInfo:
    dimensions = tuple(int(d) for d in grid.dimensions)
    spacing = tuple(float(s) for s in grid.spacing)
    origin = tuple(float(x) for x in grid.lower)
    dt = float(time.dt) if time.dt is not None else None
    boundary: tuple[str, ...] | None = None
    if bcs is not None:
        boundary = tuple(str(b) for b in bcs.lower)
    return GridInfo(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
        dt=dt,
        boundary=boundary,
    )


def _build_normalization(units: Units) -> Normalization:
    if isinstance(units, UnitsSI):
        return Normalization.identity()
    if isinstance(units, UnitsPIC):
        return _pic_norm(units)
    if isinstance(units, UnitsMHD):
        return Normalization.mhd_standard(
            l_0=float(units.reference_length),
            rho_0=float(units.reference_density),
            b_0=float(units.reference_b_field),
        )
    if isinstance(units, UnitsCustom):
        ref = units.reference
        return Normalization(
            length_ref=float(ref.length),
            time_ref=float(ref.time) if ref.time is not None else 0.0,
            velocity_ref=float(ref.velocity) if ref.velocity is not None else 0.0,
            b_field_ref=float(ref.b_field) if ref.b_field is not None else 0.0,
            e_field_ref=float(ref.e_field) if ref.e_field is not None else 0.0,
            density_ref=float(ref.density) if ref.density is not None else 0.0,
            mass_ref=float(ref.mass) if ref.mass is not None else 0.0,
            charge_ref=float(ref.charge) if ref.charge is not None else 0.0,
        )
    raise TypeError(f"unsupported units variant: {type(units).__name__}")


def _pic_norm(units: UnitsPIC) -> Normalization:
    species_name = units.reference_species.lower()
    defaults = _DEFAULT_SPECIES_PARAMS.get(species_name)
    default_mass, default_charge = defaults if defaults is not None else (None, None)

    if units.reference_mass is not None:
        mass = float(units.reference_mass)
    elif default_mass is not None:
        mass = default_mass
    else:
        raise ValueError(
            f"[units] PIC: unknown reference species {species_name!r} — "
            f"'reference_mass' is required"
        )

    if units.reference_charge is not None:
        charge = float(units.reference_charge)
    elif default_charge is not None:
        charge = default_charge
    else:
        raise ValueError(
            f"[units] PIC: unknown reference species {species_name!r} — "
            f"'reference_charge' is required"
        )

    c = float(units.speed_of_light) if units.speed_of_light is not None else constants.c
    return Normalization.pic_standard(float(units.reference_density), mass, charge, c)


def _build_species(sp: Species) -> SpeciesInfo:
    kwargs: dict[str, Any] = {"name": sp.name}
    if sp.charge is not None:
        kwargs["charge"] = float(sp.charge)
    if sp.mass is not None:
        kwargs["mass"] = float(sp.mass)
    if sp.charge_to_mass is not None:
        kwargs["charge_to_mass"] = float(sp.charge_to_mass)
    if sp.temperature is not None:
        kwargs["temperature"] = float(sp.temperature)
    if sp.density is not None:
        kwargs["density"] = float(sp.density)
    if sp.thermal_velocity is not None:
        kwargs["thermal_velocity"] = tuple(float(v) for v in sp.thermal_velocity)
    if sp.drift_velocity is not None:
        kwargs["drift_velocity"] = tuple(float(v) for v in sp.drift_velocity)
    if sp.particles_per_cell is not None:
        ppc = sp.particles_per_cell
        if isinstance(ppc, list):
            kwargs["particles_per_cell"] = tuple(int(p) for p in ppc)
        else:
            kwargs["particles_per_cell"] = int(ppc)
    return SpeciesInfo(**kwargs)


def _build_physics(physics: Physics | None) -> PhysicsParams:
    # PhysicsParams.c is the speed of light in *normalized* units. For the
    # PIC normalization (velocity_ref = c_SI) it is 1.0 by construction;
    # `[units].speed_of_light` is an SI value already consumed by
    # `_pic_norm` to set velocity_ref, not a code-unit override. MHD/hybrid
    # would want c_SI / v_A here, but the v1.0 schema does not yet expose
    # that knob — left at the default until a future schema field lands.
    if physics is None:
        return PhysicsParams()

    extra: dict[str, Any] = {}
    relativistic = bool(physics.relativistic)

    # `gamma_eos` is only a typed field on PhysicsMHD. Reading it from a
    # stray `[physics.pic]` extra would silently let a misplaced key win.
    gamma = (
        float(physics.mhd.gamma_eos)
        if physics.mhd is not None and physics.mhd.gamma_eos is not None
        else 5.0 / 3.0
    )

    for branch_name in ("pic", "mhd", "hybrid"):
        branch = getattr(physics, branch_name, None)
        if branch is None:
            continue
        extra[branch_name] = branch.model_dump(exclude_none=True)

    for key, value in (physics.__pydantic_extra__ or {}).items():
        extra[key] = value

    return PhysicsParams(gamma=gamma, relativistic=relativistic, extra=extra)


def _build_transforms(
    coords: Coordinates, default_frame: str
) -> dict[str, FrameTransform]:
    result: dict[str, FrameTransform] = {}
    for target_name, spec in coords.transforms.items():
        origin_source = spec.origin if spec.origin is not None else (0.0, 0.0, 0.0)
        origin = tuple(float(x) for x in origin_source)
        kwargs: dict[str, Any] = {}
        if spec.rotation is not None:
            kwargs["rotation"] = tuple(
                tuple(float(x) for x in row) for row in spec.rotation
            )
        scale = float(spec.scale) if spec.scale is not None else 1.0
        source = spec.from_frame if spec.from_frame is not None else default_frame
        if spec.axis_labels is not None:
            kwargs["target_axis_names"] = tuple(spec.axis_labels)
        result[target_name] = FrameTransform(
            source_frame=source,
            target_frame=target_name,
            origin=origin,  # type: ignore[arg-type]
            scale=scale,
            **kwargs,
        )
    return result


def _build_metadata(schema: SimulationSchema) -> dict[str, Any]:
    """Build the free-form ``metadata`` dict for ``SimulationConfig``.

    ``initial_conditions`` and ``output`` are *not* duplicated here —
    they live as typed attributes on ``SimulationConfig`` directly.
    Callers that previously read ``cfg.metadata["output"]`` should now
    use ``cfg.output`` (a validated ``Output`` model) instead.
    """
    metadata: dict[str, Any] = {}
    if schema.model.version is not None:
        metadata["version"] = schema.model.version
    if schema.model.description is not None:
        metadata["description"] = schema.model.description
    # ``[grid.stagger]`` consolidates the three Tier 1/2/3 keys
    # (``convention``, ``fields``, ``position``) under one sub-table.
    # All three round-trip through ``StaggerInfo`` so consumers see one
    # shape regardless of which tiers the source TOML populated.
    stagger = schema.grid.stagger
    position: dict[str, tuple[float, ...]] | None = None
    if stagger.position is not None:
        position = {
            name: tuple(float(x) for x in offsets)
            for name, offsets in stagger.position.items()
        }
    metadata["stagger"] = StaggerInfo(
        convention=str(stagger.convention),
        field_locations=dict(stagger.fields) if stagger.fields is not None else None,
        position=position,
    )
    scaling: dict[str, Any] = {}
    scaling_factor = getattr(schema.units, "scaling_factor", None)
    if scaling_factor is not None:
        scaling["scaling_factor"] = scaling_factor
    scaling_description = getattr(schema.units, "scaling_description", None)
    if scaling_description is not None:
        scaling["scaling_description"] = scaling_description
    if scaling:
        metadata["scaling"] = scaling
    return metadata
