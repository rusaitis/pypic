# Core Containers

The data structures every other part of pypic is written against.
`FieldDataset` is the central type: readers produce it, derived
quantities and selections consume it, and the I/O layer serializes it.

Dependency direction is one-way — `grid` ← `containers` ← `dataset` —
so the geometry description never has to know about the data it
describes.

| Type | Module | Role |
|------|--------|------|
| [`FieldDataset`][pypic.dataset.FieldDataset] | `pypic.dataset` | Field arrays plus grid, normalization, and species metadata. Wraps `xarray.Dataset`; computation happens on raw NumPy. |
| [`GridInfo`][pypic.grid.GridInfo] | `pypic.grid` | Dimensions, spacing, extent, and geometry of the co-located grid. |
| [`SimulationConfig`][pypic.containers.SimulationConfig] | `pypic.containers` | Validated `simulation.toml` contents in internal form. |
| [`TabularData`][pypic.containers.TabularData] | `pypic.containers` | Time-series and other column-oriented output (probes, energy histories). |
| [`ParticleData`][pypic.containers.ParticleData] | `pypic.containers` | Per-macroparticle positions, velocities, and weights. |
| [`StaggerInfo`][pypic.containers.StaggerInfo] | `pypic.containers` | Provenance record of the source code's mesh convention. Readers destagger on load; this documents what they destaggered *from*. |

## FieldDataset

::: pypic.dataset

## GridInfo

::: pypic.grid

## Simulation, particle, and tabular containers

::: pypic.containers
