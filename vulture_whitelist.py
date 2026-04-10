"""Vulture whitelist — suppress false positives from dynamic access and doctests.

Vulture cannot trace: getattr-based access, doctest usage, protocol
definitions, or dataclass fields used only in pre-built instances.

Known unsuppressable (vulture cannot whitelist function parameters):
  - geometry.py:64  x3 in metric_factors — general signature needs all 3 coords
  - base.py:519     path, step in SimulationReader.read_timestep — protocol params
  - base.py:523     path in SimulationReader.available_timesteps — protocol param
"""

from pypic.coordinates.geometry import CoordinateGeometry
from pypic.readers._field_dataset import (
    FieldDataset,
    GridInfo,
    SimulationConfig,
    SimulationReader,
)
from pypic.units import Normalization, PhysicsConstants, SpeciesInfo

# --- geometry.py ---
# Dataclass field used in CARTESIAN/SPHERICAL/CYLINDRICAL instances + doctests
CoordinateGeometry.axis_units
# Core method needed for non-Cartesian differential operators (Step 10)
# x3 parameter: required in general signature even when unused (Cartesian/cylindrical)
CoordinateGeometry.metric_factors

# --- base.py ---
# Schema field for timestep metadata
GridInfo.dt
# Classmethod used in 4+ doctests
FieldDataset.from_arrays
# Methods used in doctests
FieldDataset.has_field
FieldDataset.field_names
# Protocol interface — implemented by concrete readers (e.g. IPic3DReader)
SimulationReader.read_timestep
SimulationReader.available_timesteps
# SimulationConfig schema fields
SimulationConfig.model_name
SimulationConfig.model_type
SimulationConfig.frame

# --- units.py ---
# Accessed dynamically via getattr(self, f"{quantity}_ref") in normalize/to_si
Normalization.velocity_ref
Normalization.density_ref
Normalization.mass_ref
Normalization.charge_ref
# Public API classmethods (some used in doctests)
Normalization.pic_electron
Normalization.mhd_standard
Normalization.identity
# Methods used in doctests
Normalization.normalize
Normalization.to_si
# Methods used in doctests
PhysicsConstants.pic_normalized
PhysicsConstants.mhd_normalized
PhysicsConstants.inv_c_squared
# SpeciesInfo optional dataclass fields for species metadata
SpeciesInfo.temperature
SpeciesInfo.thermal_velocity
SpeciesInfo.drift_velocity
SpeciesInfo.density
SpeciesInfo.particles_per_cell
