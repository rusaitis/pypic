"""pypic: read, analyze, and plot plasma simulation output."""

__version__ = "0.1.0"

from pypic.coordinates import (
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    CoordinateGeometry,
    GeometryType,
)
from pypic.readers import FieldDataset, GridInfo
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
]
