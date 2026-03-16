"""pypic: read, analyze, and plot plasma simulation output."""

__version__ = "0.1.0"

from pypic.coordinates import (
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    CoordinateGeometry,
    GeometryType,
)
from pypic.derived import (
    alfven_speed,
    current_density_magnitude,
    electric_energy_density,
    electric_field_magnitude,
    enthalpy,
    entropy,
    gyrotropic_entropy,
    internal_energy_density,
    kinetic_energy_density,
    magnetic_energy_density,
    magnetic_field_magnitude,
    plasma_beta,
    poynting_flux,
    relativistic_enthalpy,
    thermal_energy_density,
    velocity_magnitude,
)
from pypic.readers import FieldDataset, GridInfo, load_config
from pypic.units import Normalization, PhysicsConstants, SpeciesInfo

__all__ = [
    "CARTESIAN",
    "CYLINDRICAL",
    "SPHERICAL",
    "CoordinateGeometry",
    "FieldDataset",
    "GeometryType",
    "GridInfo",
    "Normalization",
    "PhysicsConstants",
    "SpeciesInfo",
    "alfven_speed",
    "current_density_magnitude",
    "electric_energy_density",
    "electric_field_magnitude",
    "enthalpy",
    "entropy",
    "gyrotropic_entropy",
    "internal_energy_density",
    "kinetic_energy_density",
    "load_config",
    "magnetic_energy_density",
    "magnetic_field_magnitude",
    "plasma_beta",
    "poynting_flux",
    "relativistic_enthalpy",
    "thermal_energy_density",
    "velocity_magnitude",
]
