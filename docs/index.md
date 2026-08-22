---
title: pypic
---

<p align="center">
  <img src="assets/pypic-logo.png" alt="pypic" width="440">
</p>

<p align="center"><em>Python for Plasma In Cells</em></p>

A Python toolkit for reading, analyzing, and plotting plasma simulation output from PIC and MHD codes.

pypic provides a unified interface to multiple simulation formats (iPIC3D, BATSRUS, OpenGGCM) and maps their native output to a canonical field schema. All computation happens in normalized code units using pure NumPy functions. xarray serves as the data container, not the computation engine. SI conversion is applied only at I/O and display boundaries.

## Features

- **Multi-code readers** -- iPIC3D (parallel HDF5, serial HDF5, H5hut), BATSRUS (IDL cell + HDF5 BATL with AMR regridding), OpenGGCM (Fortran binary 3df), and a generic HDF5 reader. Auto-detection via confidence-based probing.
- **Derived quantities** -- field magnitudes, plasma beta, Alfvén speed, Mach numbers, Poynting flux, energy densities, pressure tensor decomposition, characteristic scales, entropy, reconnection diagnostics, and more. All as pure functions: arrays in, arrays out.
- **Unit system** -- PIC (electron- or ion-referenced), MHD (Alfvén-speed-based), SI, or custom normalization. Round-trip `normalize()` / `to_si()` with display unit conversion.
- **Geometry-aware operators** -- divergence, curl, gradient with coordinate metric factors. Cartesian implemented; spherical/cylindrical planned.
- **Selections** -- `PlaneSelection`, `BoxSelection`, and `SphereSelection` slice 3D data into lower-dimensional views or masked subregions.
- **Reductions** -- `pypic.reduce(ds, axis, reduction=...)` collapses fields along one or more axes: column densities, slab averages, density-weighted line averages, projected-peak maps.
- **Field-line tracing** -- adaptive Dormand-Prince 5(4) tracer with PI step control, plus Poincaré sections.
- **Modern I/O** -- Zarr v3 export/import, Icechunk versioned storage, VirtualiZarr views over legacy HDF5, and Parquet/Arrow for particle data.
- **Field registry** -- `compute("beta")`, `compute("|B|")`, `compute("v_A")` dispatches to the right derived function. Extensible via `register_field()`.

## Quick start

```python
from pypic import open_simulation, PlaneSelection

# Load simulation data
sim = open_simulation("path/to/output")
fds = sim.read(step=0)

# Compute derived quantities
beta = fds.compute("beta")
v_a = fds.compute("v_A")

# Select a plane
plane = PlaneSelection(normal="z")
fds_2d = plane.apply(fds)

# Convert to SI for display
b_si = fds.in_si("B_1")
b_nT = fds.in_units("B_1", "nT")
```

## Requirements

Python 3.13+. Core dependencies: NumPy, SciPy, xarray, h5py.
