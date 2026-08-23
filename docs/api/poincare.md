# Poincaré Sections

Surface-of-section diagnostic for field-line topology. Each seed is
traced with the adaptive Dormand-Prince integrator; every accepted-step
segment that crosses a transverse plane $\Sigma$ contributes one
puncture, located via linear interpolation. The resulting 2D scatter
pattern makes topology visually obvious:

- **closed curves** ⇒ magnetic islands / O-points
- **finite point sets** ⇒ rational flux surfaces
- **densely-filled 1D fills** ⇒ KAM surfaces (good confinement)
- **2D blobs** ⇒ chaotic / stochastic regions

A classical dynamical-systems diagnostic; in 3D plasma the same
construction underpins fusion poloidal sections, X-line separatrix
mapping in magnetotail reconnection, mirror-mode magnetic-hole
topology, and stellarator divertor footprint analysis
[@Frerichs2024].

## Tokamak poloidal section (closed orbits)

```python
import numpy as np
from pypic import PoincareSurface, poincare_section, open_simulation
from pypic.plotting import plot_poincare_section

sim = open_simulation("/path/to/run")
ds = sim.read(sim.steps[-1])

# Φ = 0 cut: poloidal plane normal is the toroidal direction (ŷ here)
surf = PoincareSurface.from_axis("y", 0.0, name="φ = 0 poloidal")

# Seed a radial fan across the minor radius
seeds = np.array([[r, 0.0, 0.0] for r in np.linspace(0.2, 0.9, 12)])

section = poincare_section(
    ds, seeds, surf,
    max_steps=20_000,           # ~100 poloidal transits at this resolution
    direction="forward",
)

fig, ax = plot_poincare_section(section, color_by_seed=True)
ax.set_xlabel(r"$Z$ [m]")
ax.set_ylabel(r"$R$ [m]")
fig.savefig("poincare_phi_0.png", dpi=200)
```

Inside an island chain, several seeds will produce the same nested
closed-curve pattern (same-color punctures forming concentric ovals).
Seeds on KAM surfaces give 1D filled curves. Seeds in the ergodic
boundary layer fill 2D regions.

## Magnetotail X-line geometry

```python
import numpy as np
from pypic import PoincareSurface, poincare_section, open_simulation

ds = open_simulation("/path/to/mhd_run").read(step=120)

# x-z plane in GSM (the standard reconnection-geometry view)
surf = PoincareSurface.from_axis("y", 0.0, name="meridional (GSM)")

# Seed a fan straddling the expected X-line
seeds = np.array([
    [-15.0, 0.0, z] for z in np.linspace(-3.0, 3.0, 21)
])

section = poincare_section(
    ds, seeds, surf,
    max_steps=10_000,
    direction="both",  # capture both inflow regions
)
```

The separatrix appears as the boundary between qualitatively different
puncture patterns: seeds on closed plasmoid loops trace ovals,
inflow-region seeds give open punctures that exit the domain on the
opposite side.

## Re-puncturing without re-integrating

Tracing is the cost; puncturing is essentially free. Keep the
underlying `FieldLine` traces in the returned section and re-puncture
against a different surface:

```python
from pypic.traces import plane_crossings

section_phi0 = poincare_section(ds, seeds, PoincareSurface.from_axis("y", 0.0))

# Re-puncture on a different toroidal angle without retracing
surf_phi_pi_2 = PoincareSurface.from_axis("x", 0.0, name="φ = π/2")
new_punctures = [
    plane_crossings(fl.points, surf_phi_pi_2.normal, surf_phi_pi_2.offset)
    for fl in section_phi0.field_lines
]
```

## Arbitrary plane normals

Tilted current sheets, oblique X-lines, and stellarator Boozer-angle
cuts don't align with the world axes:

```python
# Plane through the origin with normal (1, 1, 0)/√2
surf = PoincareSurface(normal=(1.0, 1.0, 0.0), point=(0.0, 0.0, 0.0))
```

The Gram--Schmidt basis is built against the world axis least
parallel to $\hat{\mathbf{n}}$ for numerical stability — the standard
oblique-section convention shared by FLARE [@Frerichs2024] and most
field-mapping tools.

## Implementation notes

`poincare_section` forces `loop_tol=None` on the adaptive tracer.
The default auto closed-loop detector would otherwise terminate the
very orbits of interest after their first revolution, leaving a
single puncture per seed instead of an entire ring.

Punctures are extracted **post-hoc** via
[`plane_crossings`][pypic.traces.plane_crossings] — linear
interpolation between accepted trace points. The interpolation is
$O(h^2/r)$ accurate, where $h$ is the local step size and $r$ is the
orbit radius. For sharp topology rendering, set
`max_step <= circumference / 50`.

::: pypic.traces._poincare
    options:
      # Rendered from the private module because that is where the code
      # lives, but every name below is re-exported from `pypic` and
      # `pypic.traces` — import from those, not from this path.
      show_root_heading: false
