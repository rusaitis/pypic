# pypic

A Python toolkit for reading, analyzing, and plotting plasma simulation output from PIC and MHD codes.

pypic provides a unified interface to multiple simulation formats (iPIC3D, BATSRUS, OpenGGCM) and maps their native output to a canonical field schema. All computation happens in normalized code units using pure NumPy functions. xarray serves as the data container, not the computation engine. SI conversion is applied only at I/O and display boundaries.

## Features

- **Multi-code readers** -- iPIC3D (parallel HDF5, serial HDF5, H5hut), BATSRUS (IDL + HDF5 with AMR regridding), OpenGGCM (Fortran binary), and a generic HDF5 reader. Auto-detection via confidence-based probing.
- **Derived quantities** -- field magnitudes, plasma beta, Alfven speed, Mach numbers, Poynting flux, energy densities, pressure tensor decomposition, characteristic scales, entropy, and more. All as pure functions: arrays in, arrays out.
- **Unit system** -- PIC (electron- or ion-referenced), MHD (Alfven-speed-based), SI, or custom normalization. Round-trip `normalize()` / `to_si()`.
- **Geometry-aware operators** -- divergence, curl, gradient with coordinate metric factors.
- **Selections** -- `PlaneSelection` and `BoxSelection` for slicing 3D data into lower-dimensional views.
- **Field registry** -- `compute("beta")`, `compute("|B|")`, `compute("v_A")` dispatches to the right derived function.

## Quick start

```python
from pypic import open_simulation, PlaneSelection

# Load simulation data
sim = open_simulation("path/to/output")
fds = sim.read_step(0)

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
