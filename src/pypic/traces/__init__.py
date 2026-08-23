"""Curve containers and analysis for field lines and particle trajectories."""

from pypic.traces._analysis import (
    arc_length_cumulative,
    arc_length_total,
    closest_approach,
    curvature,
    displacement,
    drift_velocity,
    equatorial_crossings,
    gyroradius_estimate,
    kinetic_energy,
    mirror_points,
    plane_crossings,
    resample_by_arc_length,
    speed,
    tangent_vectors,
)
from pypic.traces._fieldline import FieldLine, TraceDirection
from pypic.traces._particletrace import ParticleTrace
from pypic.traces._poincare import (
    PoincareSection,
    PoincareSurface,
    poincare_section,
)
from pypic.traces._sampling import (
    attach_scalars,
    attach_scalars_to_trace,
    sample_field,
    sample_fields,
)
from pypic.traces._tracing import (
    TerminationReason,
    VectorFieldInterpolator,
    estimate_tracing_error,
    trace_field_line,
    trace_field_line_adaptive,
    trace_field_lines_adaptive,
)

__all__ = [
    "FieldLine",
    "ParticleTrace",
    "PoincareSection",
    "PoincareSurface",
    "TerminationReason",
    "TraceDirection",
    "VectorFieldInterpolator",
    "arc_length_cumulative",
    "arc_length_total",
    "attach_scalars",
    "attach_scalars_to_trace",
    "closest_approach",
    "curvature",
    "displacement",
    "drift_velocity",
    "equatorial_crossings",
    "estimate_tracing_error",
    "gyroradius_estimate",
    "kinetic_energy",
    "mirror_points",
    "plane_crossings",
    "poincare_section",
    "resample_by_arc_length",
    "sample_field",
    "sample_fields",
    "speed",
    "tangent_vectors",
    "trace_field_line",
    "trace_field_line_adaptive",
    "trace_field_lines_adaptive",
]
