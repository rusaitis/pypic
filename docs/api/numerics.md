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

## What's tested

The kernel's contract is pinned in `tests/test_numerics.py`:

- **5th-order convergence** — halving $h$ on $y'' + 4y = 0$ reduces
  the global error by a factor in $(20, 60)$, ruling out 4th- and
  6th-order rounding accidents.
- **FSAL identity and savings** — the last stage of an accepted step
  equals $f(y_{n+1})$; passing it as `k0` saves exactly one RHS
  evaluation on the next step and produces bit-for-bit identical
  results.
- **Batched ↔ scalar bit-for-bit equivalence** — at N=1 and across
  N=4 with mixed states, the batched kernel matches a per-seed
  scalar loop to `atol=1e-15`.
- **Per-seed failure isolation** — one batched seed failing at any
  Butcher stage leaves the other seeds' `y_new` unaffected.
- **Adaptive-loop integration** — a minimal driver wiring DP × error
  norm × I controller solves $y''+4y=0$ to tighter and looser
  tolerances; the tighter run achieves a strictly smaller global
  error than the looser one.
- **Controller monotonicity** — `h_new` is non-increasing in
  `err_norm` across a sweep that spans the unclamped middle and
  both growth clamps.

## Planned additions

- **Implicit-midpoint integrator** — single-stage Gauss-Legendre
  Runge-Kutta for symplectic, bounded-drift field-line tracing.
- **Tricubic interpolation kwarg** — non-periodic
  `RegularGridInterpolator(method="cubic")` passthrough.
- **Periodic tricubic splines** — `periodic_axes=` kwarg using
  `scipy.interpolate.CubicSpline(..., bc_type="periodic")` per spline
  line; needed for seamless $\phi$-wrap on spherical PFSS grids.
- **Curvature-based step control** — `step_control="curvature"`
  alternative to the error-norm controller, keeping the unit-tangent rotation
  per step bounded by `over_rc` and clamped by the local mesh size.

The first downstream consumer of the full bundle will be a planned
`pypic.maps` module — squashing factor $Q$, footpoint maps, and
open-field classification. See [schema.md § Field-line map
quantities](../schema.md#field-line-map-quantities) for the canonical
names of map outputs on disk.

::: pypic.numerics
