"""Coordinate geometry definitions and discrete differential operators."""

from pypic.coordinates.geometry import (
    CARTESIAN,
    CYLINDRICAL,
    GEOMETRY_BY_NAME,
    SPHERICAL,
    CoordinateGeometry,
    GeometryType,
)
from pypic.coordinates.operators import curl, divergence, gradient
from pypic.coordinates.transforms import (
    FrameTransform,
    compose_transforms,
    find_pressure_tensor_groups,
    find_vector_triplets,
    identity_transform,
    resolve_transform,
    rotate_pressure_tensor,
    rotate_vector_components,
)

__all__ = [
    "CARTESIAN",
    "CYLINDRICAL",
    "GEOMETRY_BY_NAME",
    "SPHERICAL",
    "CoordinateGeometry",
    "FrameTransform",
    "GeometryType",
    "compose_transforms",
    "curl",
    "divergence",
    "find_pressure_tensor_groups",
    "find_vector_triplets",
    "gradient",
    "identity_transform",
    "resolve_transform",
    "rotate_pressure_tensor",
    "rotate_vector_components",
]
