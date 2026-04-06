"""PyVista 3D rendering utilities for pypic.

Provides theme-aware building blocks for interactive 3D visualization
using pyvista/VTK. Fully independent from the matplotlib plotting
modules — can be removed without affecting core pypic or 2D plotting.

Requires: ``pip install pypic[3d]`` (installs pyvista).
"""

from pypic.plotting.pyvista._axes import add_axis_triad, add_equatorial_grid
from pypic.plotting.pyvista._badge import add_badge
from pypic.plotting.pyvista._lines import (
    add_field_line,
    add_field_lines,
    add_trajectory,
    add_trajectories,
)
from pypic.plotting.pyvista._meshes import (
    add_equatorial_surface,
    add_planet,
    add_reference_circles,
)
from pypic.plotting.pyvista._theme import (
    apply_theme,
    create_plotter,
    resolve_cmap,
    set_camera,
)

__all__ = [
    "add_badge",
    "add_axis_triad",
    "add_equatorial_grid",
    "add_equatorial_surface",
    "add_field_line",
    "add_field_lines",
    "add_planet",
    "add_reference_circles",
    "add_trajectory",
    "add_trajectories",
    "apply_theme",
    "create_plotter",
    "resolve_cmap",
    "set_camera",
]
