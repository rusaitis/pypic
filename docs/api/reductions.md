# Reductions

`pypic.reduce` collapses a `FieldDataset` along one or more axes
using a chosen reduction operation. Column densities, line-of-sight
integrals, slab averages, projected-max diagnostics, and peak-position
maps all compose from this single verb — paired with an optional
`BoxSelection` or `SphereSelection` for region restriction.

The verb is named `reduce` (not `project`) to avoid colliding with
Three.js `Vector3.project(camera)` (camera/screen-space projection) on
the webpic client side. The pypic server reduces; the viewer projects.

## Reductions table

| Reduction | xarray dispatch | Multi-axis? | Metadata behavior |
|-----------|-----------------|-------------|---|
| `integrate` | `ds.integrate(coord=ax)` looped over axes | yes (sequential trapezoidal) | preserved |
| `sum` | `ds.sum(dim=axis, skipna=...)` | yes | preserved |
| `mean` | `ds.mean(dim=axis, skipna=...)` | yes | preserved |
| `median` | `ds.median(dim=axis, skipna=...)` | yes | preserved |
| `max` / `min` | `ds.{max,min}(dim=axis, skipna=...)` | yes | preserved |
| `std` / `var` | `ds.{std,var}(dim=axis, skipna=...)` | yes | preserved |
| `argmax` / `argmin` | `ds.{idxmax,idxmin}(dim=axis, ...)` | **single axis only** | overridden to `quantity_type="length"` |

`argmax` / `argmin` use xarray's `idxmax` / `idxmin` to return the
*coordinate value* of the extremum (e.g. the z-position where `|J|`
peaks) — not a raw integer index. Multi-axis search has no single
coord-value to return, so reject with `ValueError`.

## Single-axis vs multi-axis

```python
# Single axis: 3D → 2D
column = pypic.reduce(ds, "z", reduction="integrate")

# Tuple: 3D → 1D in one call (sequential trapezoidal under the hood)
line_out = pypic.reduce(ds, ("y", "z"), reduction="mean")
```

The single-axis and tuple forms are functionally identical for the
non-`integrate` reductions; for `integrate` the multi-axis form runs a
sequential trapezoidal that matches the chained two-call form to
machine precision.

## Weight semantics

`weight=<field-name>` produces yt-style density- or emission-weighted
averages. Only `reduction="mean"` and `reduction="integrate"` accept a
weight — other reductions raise `ValueError`.

- **Weighted mean**: $\langle f \rangle_w = \sum_i f_i w_i / \sum_i w_i$
  over the reduced axes.
- **Weighted integrate**: $\int f w \, dx / \int w \, dx$ — the column
  weighted average along the line of sight (yt's `weight_field`
  convention).

```python
# Density-weighted temperature column average
T_avg = pypic.reduce(ds, "z", reduction="integrate",
                     weight="rho_c", fields=["T_s0"])
```

Provenance: `T_avg.xr["T_s0"].attrs["reduction"]` carries
`{"axis": "z", "op": "integrate", "weight": "rho_c"}`.

**NaN handling.** Under `nan_policy="omit"` (default) the weighted
path uses joint masking: cells where the field *or* weight is NaN
contribute zero to both numerator and denominator, skipping them
consistently. Under `"propagate"`, NaN flows through naturally. Under
`"raise"`, both field and weight are pre-checked for NaN.

## Selection composition

`selection=` runs first, then the reduction operates on the
selected region:

```python
# Mean over a sub-volume
box = pypic.BoxSelection(ranges={"x": (10, 30), "y": (10, 30)})
slab = pypic.reduce(ds, "z", reduction="mean", selection=box)

# Line of sight through a sphere — NaN-mask outside, integrate inside
ball = pypic.SphereSelection(center=(0.0, 0.0, 0.0), radius=5.0,
                             keep="inside")
los = pypic.reduce(ds, "z", reduction="integrate", selection=ball)
```

`SphereSelection` pairs naturally with `nan_policy="omit"`: outside
cells become NaN and are dropped from the reduction.

## Plot composition pattern

A reduced `FieldDataset` is just a smaller-dimension `FieldDataset` —
all existing plotting helpers work unchanged:

```python
fig, ax = pypic.plot_field_slice(
    ds.reduce("z", reduction="integrate"),
    "rho_c",
    title="Column density",
)
```

`reduce → plot_field_slice` is the canonical pattern for column
density / LOS imaging.

## Time-axis reduction

`reduce` accepts any dimension on the underlying xarray dataset —
not just the spatial `grid.surviving_axis_names`. For time-series
stores written by `to_zarr_timeseries`, every field has `time` as
the leading dim, so a time-mean / time-integrate works directly:

```python
ts = pypic.from_zarr("run.zarr")     # fields shaped (nt, nx, ny, nz)
B_mean = pypic.reduce(ts, "time", reduction="mean")
```

Pure non-spatial reductions bypass the Cartesian-grid gate — the
Jacobian only matters when integrating over a spatial axis. A
mixed `("time", "r")` reduction on a spherical grid still raises
`NotImplementedError` until Step 43b lands.

## Worked examples

Three first-line plasma diagnostics expressible in the current API.

### EDR localization via the Zenitani localizer

The electron-frame dissipation $D_e$ (Zenitani EDR localizer) peaks
inside the electron diffusion region. Per-column z-position of the
peak gives an EDR centroid map:

```python
ds_with_De = ds.compute("D_e")
# argmax returns the *coordinate value* of the maximum along z,
# not an integer index — the result is one z-position per (x, y).
z_peak = pypic.reduce(ds_with_De, "z", reduction="argmax", fields=["D_e"])
# z_peak.field_info("D_e").quantity_type == "length"
```

For the *value* at the peak (rather than the location), swap to
`reduction="max"`.

### Density-weighted column temperature (synthetic LOS observable)

The line-of-sight temperature that an external observer would
measure is the emission/density-weighted average — the yt
"`weight_field`" convention:

```python
T_col = pypic.reduce(ds, "z", reduction="integrate",
                     weight="rho_c", fields=["T_s0"])
# T_col["T_s0"] == ∫ T_s0 ρ_c dz / ∫ ρ_c dz
```

The provenance attr records the weight:
`T_col.xr["T_s0"].attrs["reduction"]` →
`{"axis": "z", "op": "integrate", "weight": "rho_c"}`.

### Energy-budget conservation

For each timestep, integrate the energy densities over the full
volume to get total magnetic, kinetic, and thermal energy:

```python
totals_by_step: list[tuple[float, dict[str, float]]] = []
for step in sim.steps:
    ds = sim.read(step).compute("e_B", "e_k", "e_th")
    totals = pypic.reduce(ds, ("x", "y", "z"), reduction="integrate")
    totals_by_step.append(
        (sim.time_for(step), {
            "E_B": float(totals["e_B"]),
            "E_k": float(totals["e_k"]),
            "E_th": float(totals["e_th"]),
        })
    )
# Plot E_B(t) + E_k(t) + E_th(t) — should be conserved in ideal MHD
# runs and slowly evolve in dissipative ones.
```

## CLI

```bash
pypic reduce apply <sim-dir> --output out.zarr --axis z \
    --reduction mean --weight rho_c --fields T_s0
```

Supports multi-step batch writes (`--step all` → time-series Zarr),
selection composition (`--box`, `--plane`), Icechunk versioning
(`--backend icechunk --tag v1`), and `--dry-run` for plan preview.

## API reference

::: pypic.reductions
    options:
      members:
        - reduce
        - Reduction
