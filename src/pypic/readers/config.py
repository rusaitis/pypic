"""Load simulation configuration from a TOML file."""

from __future__ import annotations

import copy
import logging
import tomllib
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    from pathlib import Path

from pypic.coordinates.geometry import GEOMETRY_BY_NAME, CoordinateGeometry
from pypic.coordinates.transforms import FrameTransform
from pypic.readers.base import GridInfo, SimulationConfig
from pypic.units import Normalization, SpeciesInfo

log = logging.getLogger(__name__)

_DEFAULT_SPECIES_PARAMS: dict[str, tuple[float, float]] = {
    "electrons": (constants.m_e, constants.e),
    "ions": (constants.m_p, constants.e),
}

LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1e3,
    "R_E": 6.371e6,
    "R_S": 6.957e8,
    "AU": constants.astronomical_unit,
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
    unit_factor = LENGTH_UNITS.get(physical_extent_unit)
    if unit_factor is None:
        valid = ", ".join(sorted(LENGTH_UNITS))
        raise ValueError(
            f"Unknown physical_extent_unit {physical_extent_unit!r}. Valid: {valid}"
        )

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

    # Compute shrink factor if normalization carries real physics
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

    Parameters
    ----------
    path : Path
        Path to a TOML configuration file conforming to SCHEMA.md.

    Returns
    -------
    SimulationConfig
        Fully typed configuration with grid, normalization, and species.

    Raises
    ------
    ExceptionGroup
        If one or more sections contain validation errors.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    errors: list[Exception] = []

    model_name = ""
    model_type = ""
    model_extras: dict[str, Any] = {}
    try:
        model_name, model_type, model_extras = _parse_model(raw.get("model", {}))
    except (ValueError, KeyError) as exc:
        errors.append(exc)

    geometry: CoordinateGeometry | None = None
    frame = ""
    try:
        geometry, frame = _parse_coordinates(raw.get("coordinates", {}))
    except (ValueError, KeyError) as exc:
        errors.append(exc)

    grid: GridInfo | None = None
    if geometry is not None:
        try:
            grid = _parse_grid(raw.get("grid", {}), geometry)
        except (ValueError, KeyError) as exc:
            errors.append(exc)
    else:
        errors.append(ValueError("[grid] skipped: [coordinates] failed"))

    normalization: Normalization | None = None
    try:
        normalization = _parse_units(raw.get("units", {}))
    except (ValueError, KeyError) as exc:
        errors.append(exc)

    species: tuple[SpeciesInfo, ...] = ()
    try:
        species = _parse_species(raw.get("species", []))
    except (ValueError, KeyError) as exc:
        errors.append(exc)

    if errors:
        raise ExceptionGroup(f"Invalid simulation config: {path}", errors)

    assert grid is not None
    assert normalization is not None

    metadata: dict[str, Any] = {}
    if model_extras:
        metadata.update(model_extras)
    if "initial_conditions" in raw:
        metadata["initial_conditions"] = raw["initial_conditions"]
    if "output" in raw:
        metadata["output"] = raw["output"]

    units_raw = raw.get("units", {})
    scaling = {}
    if "scaling_factor" in units_raw:
        scaling["scaling_factor"] = units_raw["scaling_factor"]
    if "scaling_description" in units_raw:
        scaling["scaling_description"] = units_raw["scaling_description"]
    if scaling:
        metadata["scaling"] = scaling

    transforms = _parse_transforms(raw.get("coordinates", {}), frame)

    config = SimulationConfig(
        model_name=model_name,
        model_type=model_type,
        grid=grid,
        normalization=normalization,
        species=species,
        physics=raw.get("physics", {}),
        frame=frame,
        transforms=transforms,
        metadata=metadata,
    )

    coords_raw = raw.get("coordinates", {})
    if "physical_extent" in coords_raw:
        phys_ext = tuple(float(x) for x in coords_raw["physical_extent"])
        phys_unit = str(coords_raw.get("physical_extent_unit", "m"))
        config = apply_physical_extent(config, phys_ext, phys_unit)

    return config


def _parse_transforms(
    raw: dict[str, Any], default_frame: str
) -> dict[str, FrameTransform]:
    """Parse ``[coordinates.transforms.*]`` into FrameTransform objects.

    Parameters
    ----------
    raw : dict
        The ``[coordinates]`` section from the TOML file.
    default_frame : str
        Native frame name (from ``[coordinates] frame``).

    Returns
    -------
    dict[str, FrameTransform]
        Mapping of target frame name to transform.
    """
    transforms_section = raw.get("transforms", {})
    if not transforms_section:
        return {}

    result: dict[str, FrameTransform] = {}
    for target_name, spec in transforms_section.items():
        origin = tuple(float(x) for x in spec.get("origin", [0.0, 0.0, 0.0]))
        rotation_raw = spec.get("rotation", None)
        kwargs: dict[str, Any] = {}
        if rotation_raw is not None:
            kwargs["rotation"] = tuple(
                tuple(float(x) for x in row) for row in rotation_raw
            )
        scale = float(spec.get("scale", 1.0))
        source = spec.get("from_frame", default_frame)
        axis_labels = spec.get("axis_labels", None)
        if axis_labels:
            kwargs["target_axis_names"] = tuple(axis_labels)

        result[target_name] = FrameTransform(
            source_frame=source,
            target_frame=target_name,
            origin=origin,  # type: ignore[arg-type]
            scale=scale,
            **kwargs,
        )
    return result


def _parse_model(raw: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extract model name, type, and optional extras from ``[model]``."""
    if not raw:
        raise ValueError("[model] section is missing or empty")
    if "name" not in raw:
        raise ValueError("[model] missing required key 'name'")
    if "type" not in raw:
        raise ValueError("[model] missing required key 'type'")

    extras: dict[str, Any] = {}
    if "version" in raw:
        extras["version"] = raw["version"]
    if "description" in raw:
        extras["description"] = raw["description"]
    return raw["name"], raw["type"], extras


def _parse_coordinates(
    raw: dict[str, Any],
) -> tuple[CoordinateGeometry, str]:
    """Build a CoordinateGeometry and frame from ``[coordinates]``.

    Defaults to Cartesian geometry with ``"simulation"`` frame when
    the section is missing or empty.
    """
    if not raw:
        return GEOMETRY_BY_NAME["cartesian"], "simulation"
    if "geometry" not in raw:
        raise ValueError("[coordinates] missing required key 'geometry'")
    if "frame" not in raw:
        raise ValueError("[coordinates] missing required key 'frame'")

    geom_str = raw["geometry"].lower()
    geometry = GEOMETRY_BY_NAME.get(geom_str)
    if geometry is None:
        valid = ", ".join(sorted(GEOMETRY_BY_NAME))
        raise ValueError(f"Unknown geometry {raw['geometry']!r}. Valid: {valid}")

    if "axis_labels" in raw:
        labels = tuple(raw["axis_labels"])
        if len(labels) != 3:
            raise ValueError(
                f"axis_labels must have exactly 3 elements, got {len(labels)}"
            )
        geometry = copy.replace(geometry, axis_names=labels)

    return geometry, raw["frame"]


def _parse_grid(raw: dict[str, Any], geometry: CoordinateGeometry) -> GridInfo:
    """Build a GridInfo from ``[grid]``."""
    if not raw:
        raise ValueError("[grid] section is missing or empty")
    if "dimensions" not in raw:
        raise ValueError("[grid] missing required key 'dimensions'")
    if "spacing" not in raw:
        raise ValueError("[grid] missing required key 'spacing'")

    dimensions = tuple(int(d) for d in raw["dimensions"])
    spacing = tuple(float(s) for s in raw["spacing"])
    ndim = len(dimensions)
    if "origin" in raw:
        origin = tuple(float(o) for o in raw["origin"])
    else:
        origin = (0.0,) * ndim
    dt = float(raw["dt"]) if "dt" in raw else None
    boundary = tuple(str(b) for b in raw["boundary"]) if "boundary" in raw else None

    return GridInfo(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
        dt=dt,
        boundary=boundary,
    )


def _parse_units(raw: dict[str, Any]) -> Normalization:
    """Build a Normalization from ``[units]``.

    Defaults to identity normalization when the section is missing or empty.
    """
    if not raw:
        return Normalization.identity()
    if "system" not in raw:
        raise ValueError("[units] missing required key 'system'")

    system = raw["system"].upper()

    match system:
        case "PIC":
            return _parse_units_pic(raw)
        case "MHD":
            return _parse_units_mhd(raw)
        case "SI":
            return Normalization.identity()
        case "CUSTOM":
            return _parse_units_custom(raw)
        case _:
            raise ValueError(
                f"Unknown unit system {raw['system']!r}. Valid: PIC, MHD, SI, custom"
            )


def _parse_units_pic(raw: dict[str, Any]) -> Normalization:
    """PIC normalization from reference species parameters."""
    if "reference_density" not in raw:
        raise ValueError("[units] PIC requires 'reference_density'")

    reference_density = float(raw["reference_density"])
    species_name = raw.get("reference_species", "electrons").lower()

    defaults = _DEFAULT_SPECIES_PARAMS.get(species_name)
    if defaults is not None:
        default_mass, default_charge = defaults
    else:
        default_mass, default_charge = None, None

    if "reference_mass" in raw:
        mass = float(raw["reference_mass"])
    elif default_mass is not None:
        mass = default_mass
    else:
        raise ValueError(
            f"[units] PIC: unknown reference species {species_name!r} — "
            f"'reference_mass' is required"
        )

    if "reference_charge" in raw:
        charge = float(raw["reference_charge"])
    elif default_charge is not None:
        charge = default_charge
    else:
        raise ValueError(
            f"[units] PIC: unknown reference species {species_name!r} — "
            f"'reference_charge' is required"
        )

    c = float(raw.get("speed_of_light", constants.c))

    return Normalization.pic_standard(reference_density, mass, charge, c)


def _parse_units_mhd(raw: dict[str, Any]) -> Normalization:
    """MHD normalization from macroscopic reference quantities."""
    missing = [
        k
        for k in ("reference_length", "reference_density", "reference_b_field")
        if k not in raw
    ]
    if missing:
        raise ValueError(f"[units] MHD requires: {', '.join(missing)}")
    return Normalization.mhd_standard(
        l_0=float(raw["reference_length"]),
        rho_0=float(raw["reference_density"]),
        b_0=float(raw["reference_b_field"]),
    )


def _parse_units_custom(raw: dict[str, Any]) -> Normalization:
    """Parse explicit reference values for custom normalization."""
    ref = raw.get("reference")
    if ref is None:
        raise ValueError("[units] custom requires a [units.reference] sub-table")

    required_keys = (
        "length",
        "time",
        "velocity",
        "b_field",
        "e_field",
        "density",
        "mass",
        "charge",
    )
    missing = [k for k in required_keys if k not in ref]
    if missing:
        raise ValueError(f"[units.reference] missing keys: {', '.join(missing)}")

    return Normalization(
        length_ref=float(ref["length"]),
        time_ref=float(ref["time"]),
        velocity_ref=float(ref["velocity"]),
        b_field_ref=float(ref["b_field"]),
        e_field_ref=float(ref["e_field"]),
        density_ref=float(ref["density"]),
        mass_ref=float(ref["mass"]),
        charge_ref=float(ref["charge"]),
    )


def _parse_species(raw: list[dict[str, Any]]) -> tuple[SpeciesInfo, ...]:
    """Build SpeciesInfo tuple from ``[[species]]`` entries."""
    result: list[SpeciesInfo] = []
    for i, entry in enumerate(raw):
        if "name" not in entry:
            raise ValueError(f"[[species]][{i}] missing required key 'name'")

        kwargs: dict[str, Any] = {"name": entry["name"]}

        if "charge" in entry:
            kwargs["charge"] = float(entry["charge"])
        if "mass" in entry:
            kwargs["mass"] = float(entry["mass"])
        if "charge_to_mass" in entry:
            kwargs["charge_to_mass"] = float(entry["charge_to_mass"])
        if "temperature" in entry:
            kwargs["temperature"] = float(entry["temperature"])
        if "density" in entry:
            kwargs["density"] = float(entry["density"])

        if "thermal_velocity" in entry:
            tv = entry["thermal_velocity"]
            if isinstance(tv, list):
                kwargs["thermal_velocity"] = tuple(float(v) for v in tv)
            else:
                kwargs["thermal_velocity"] = float(tv)

        if "drift_velocity" in entry:
            kwargs["drift_velocity"] = tuple(float(v) for v in entry["drift_velocity"])

        if "particles_per_cell" in entry:
            ppc = entry["particles_per_cell"]
            if isinstance(ppc, list):
                kwargs["particles_per_cell"] = tuple(int(p) for p in ppc)
            else:
                kwargs["particles_per_cell"] = int(ppc)

        result.append(SpeciesInfo(**kwargs))

    return tuple(result)
