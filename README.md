<p align="center">
  <img src="https://raw.githubusercontent.com/rusaitis/pypic/main/docs/assets/pypic-logo.png" alt="pypic" width="440">
</p>

<p align="center"><em>Python for Plasma In Cells</em></p>

<p align="center">
  <a href="https://github.com/rusaitis/pypic/actions/workflows/ci.yml"><img src="https://github.com/rusaitis/pypic/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://rusaitis.github.io/pypic/"><img src="https://img.shields.io/badge/docs-mkdocs--material-blue" alt="Documentation"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python 3.13+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

A Python toolkit for reading, analyzing, and plotting plasma simulation output
from particle-in-cell (PIC) and magnetohydrodynamic (MHD) codes.

Every simulation code invents its own file layout, field names, and
normalization. pypic maps them all onto one canonical schema, so an analysis
written against iPIC3D output runs unchanged against BATSRUS or OpenGGCM.
Computation happens in normalized code units using pure NumPy functions —
xarray is the container, not the compute engine — and SI conversion is applied
only at I/O and display boundaries.

## Installation

Requires Python 3.13+.

```sh
uv add pypic-plasma        # or: pip install pypic-plasma
```

The distribution is named `pypic-plasma`; the import name is `pypic`. Optional
extras cover the heavier dependencies:

```sh
uv add "pypic-plasma[plot]"    # matplotlib — 2D field plots
uv add "pypic-plasma[3d]"      # pyvista — 3D rendering and field lines
uv add "pypic-plasma[zarr]"    # Zarr v3 / Icechunk I/O
uv add "pypic-plasma[cli]"     # the `pypic` command-line tool
```

To work from a checkout instead:

```sh
git clone https://github.com/rusaitis/pypic.git && cd pypic
uv sync --all-extras --all-groups
```

## Quick start

```python
from pypic import open_simulation, PlaneSelection

# Format is auto-detected — iPIC3D, BATSRUS, OpenGGCM, or generic HDF5.
sim = open_simulation("path/to/output")
print(sim.describe())              # code, grid, species
print(sim.steps)                   # available timesteps

# Vector shorthand: "B" loads B_1, B_2, B_3.
data = sim.read(step=0, fields=["B", "E", "P_s0"])

# Derived quantities dispatch through the field registry.
b_mag = data.compute("|B|")        # magnetic field magnitude
beta = data.compute("beta")        # plasma beta, 2P/B²
v_a = data.compute("v_A")          # Alfvén speed

# Code units internally; convert at the display boundary.
b_nt = data.in_units("B_1", "nT")
v_kms = data.in_units("v_A", "km/s")

# Selections describe regions and return an ordinary FieldDataset.
midplane = PlaneSelection(normal="z").apply(data)
```

Unmatched field names raise `KeyError` rather than warning — a typo fails at
the call site instead of surfacing as missing data three steps downstream.

## Features

- **Multi-code readers** — iPIC3D (parallel HDF5, serial HDF5, H5hut), BATSRUS
  (IDL cell + HDF5 BATL with AMR regridding), OpenGGCM (Fortran binary 3df), and
  a generic HDF5 reader. Auto-detection via confidence-based probing.
- **Derived quantities** — field magnitudes, plasma beta, Alfvén speed, Mach
  numbers, Poynting flux, energy densities, pressure tensor decomposition,
  characteristic scales (skin depths, gyroradii, frequencies), entropy,
  reconnection diagnostics, and more. All pure functions: arrays in, arrays out.
- **Unit system** — PIC (electron- or ion-referenced), MHD (Alfvén-speed-based),
  SI, or custom normalization. Round-trip `normalize()` / `to_si()` with display
  unit conversion (`"nT"`, `"km/s"`, `"eV"`, ...).
- **Geometry-aware operators** — divergence, curl, gradient with coordinate
  metric factors. Cartesian implemented; spherical/cylindrical planned.
- **Selections** — `PlaneSelection`, `BoxSelection`, and `SphereSelection` slice
  3D data into lower-dimensional views or masked subregions.
- **Reductions** — `pypic.reduce(ds, axis, reduction=...)` collapses fields along
  one or more axes (trapezoidal `integrate`, `mean`/`median`/`sum`,
  `argmax`/`argmin` returning coordinate positions). Pairs with selections for
  column densities, slab averages, and density-weighted line averages.
- **Field-line tracing** — adaptive Dormand-Prince 5(4) tracer with PI step
  control, batched and scalar paths, plus Poincaré sections.
- **Modern I/O** — Zarr v3 export/import (single-step and time-series),
  Icechunk versioned storage, VirtualiZarr views over legacy HDF5, and
  Parquet/Arrow for particle data with Morton-ordered spatial pushdown.
- **Field registry** — `compute("beta")`, `compute("|B|")`, `compute("v_A")`
  dispatch to the right derived function. Extensible via `register_field()`.
- **Command line** — `pypic info`, `fields`, `stats`, `compare`, `plot`,
  `convert`, `reduce`, and `schema validate` for quick inspection without
  writing a script.

## Documentation

Full documentation, including the physics reference, lives at
**[rusaitis.github.io/pypic](https://rusaitis.github.io/pypic/)**.

| Page | Contents |
|------|----------|
| [Getting Started](https://rusaitis.github.io/pypic/getting-started/) | Installation, loading data, first derived quantities |
| [Tutorial](https://rusaitis.github.io/pypic/tutorial/) | End-to-end analysis walkthrough |
| [Equations](https://rusaitis.github.io/pypic/equations/) | Every derived quantity with its LaTeX form and SI conversion |
| [Conventions](https://rusaitis.github.io/pypic/conventions/) | Thermal speed, γ, temperature-in-energy-units, and the other choices that differ between textbooks |
| [Schema](https://rusaitis.github.io/pypic/schema/) | The `simulation.toml` contract and canonical field names |

## Status

pypic is **0.1.0 research software under active development**. The core is in
daily use — load data, compute derived quantities, compare runs, select
subregions, convert units, make figures — and is covered by ~2750 tests
including Hypothesis property tests, hand-calculated physics values, and
NRL Formulary cross-checks.

The public API may still change before 1.0. Non-Cartesian operators, several
additional readers (Vlasiator, VPIC, ARMS, openPMD), and the field-line mapping
module are planned rather than implemented — see [TASKS.md](TASKS.md) for the
roadmap and what is already done.

## Citing

If pypic contributes to work you publish, please cite it. Metadata lives in
[CITATION.cff](CITATION.cff); GitHub renders it as a ready-to-paste citation
via the *Cite this repository* button.

## Contributing

Bug reports, reader contributions for new simulation codes, and physics
corrections are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, test commands, and code conventions.

## License

MIT — see [LICENSE](LICENSE).
