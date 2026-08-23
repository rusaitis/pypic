"""Vulture whitelist — suppress false positives from dynamic access and doctests.

Vulture cannot trace: getattr-based access, doctest usage, protocol
definitions, or dataclass fields used only in pre-built instances.

Known unsuppressable — vulture cannot whitelist function *parameters*,
only names, so these stay reported:

  - ``x3`` in ``CoordinateGeometry.metric_factors``: the general signature
    takes all three coordinates even where the geometry ignores one.
  - ``path`` / ``step`` in ``SimulationReader.read_timestep`` and ``path``
    in ``SimulationReader.available_timesteps``: protocol parameters, named
    for implementers rather than used here.

Run with ``--min-confidence 80`` to see only the high-confidence findings;
the default 60 reports a large tail of dynamic-access false positives that
this file cannot fully suppress.
"""

from pypic.containers import SimulationConfig
from pypic.coordinates.geometry import CoordinateGeometry
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.readers._protocols import SimulationReader
from pypic.units import Normalization, PhysicsConstants, SpeciesInfo

# pypic.coordinates.geometry
# Dataclass field used in CARTESIAN/SPHERICAL/CYLINDRICAL instances + doctests
CoordinateGeometry.axis_units
# Needed for non-Cartesian differential operators; the x3 parameter is
# required by the general signature even where a geometry ignores it.
CoordinateGeometry.metric_factors

# pypic.dataset / pypic.grid / pypic.containers / pypic.readers._protocols
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

# pypic.units
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
