# Numerics

Pure-numerics kernels: ODE integrators, step controllers, and
interpolators. Today the only consumer is the adaptive field-line
tracer — Dormand-Prince 5(4) [@DormandPrince1980] with FSAL re-use,
an elementary order-$p$ (I) step controller in the convention of
[@HairerWanner1993] §II.4, and `scipy.interpolate.RegularGridInterpolator`
for trilinear field evaluation. Future consumers (particle pushers,
splitting helpers, higher-order quadrature) land here. A true PI
controller [@Gustafsson1988] is queued behind an `err_prev` kwarg
already wired through the public signatures.

## Planned additions (TASKS Step 44)

- **Implicit-midpoint integrator** (44a) — single-stage Gauss-Legendre
  Runge-Kutta for symplectic, bounded-drift field-line tracing.
- **Tricubic interpolation kwarg** (44b) — non-periodic
  `RegularGridInterpolator(method="cubic")` passthrough.
- **Periodic tricubic splines** (44d) — `periodic_axes=` kwarg using
  `scipy.interpolate.CubicSpline(..., bc_type="periodic")` per spline
  line; needed for seamless $\phi$-wrap on spherical PFSS grids.
- **Curvature-based step control** (44e) — `step_control="curvature"`
  alternative to the PI controller, keeping the unit-tangent rotation
  per step bounded by `over_rc` and clamped by the local mesh size.

The first downstream consumer of the full bundle will be
`pypic.maps` (Step 44g — squashing factor $Q$, footpoint maps,
open-field classification). See [schema.md § Field-line map
quantities](../schema.md#field-line-map-quantities) for the canonical
names of map outputs on disk.

::: pypic.numerics
