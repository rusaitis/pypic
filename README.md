<p align="center">
  <img src="docs/assets/pypic-logo.png" alt="pypic" width="440">
</p>

<p align="center"><em>Python for Plasma In Cells</em></p>

A Python toolkit for reading, analyzing, and plotting plasma simulation output from PIC and MHD codes.

pypic provides a unified interface to multiple simulation formats (iPIC3D, BATSRUS, OpenGGCM) and maps their native output to a canonical field schema. All computation happens in normalized code units using pure NumPy functions — xarray serves as the data container, not the computation engine. SI conversion is applied only at I/O and display boundaries.

## Features

- **Multi-code readers** — iPIC3D (parallel HDF5, serial HDF5, H5hut), BATSRUS (IDL cell + HDF5 BATL with AMR regridding), OpenGGCM (Fortran binary 3df), and a generic HDF5 reader. Auto-detection via confidence-based probing.
- **Derived quantities** — field magnitudes, plasma beta, Alfvén speed, Mach numbers, Poynting flux, energy densities, pressure tensor decomposition, characteristic scales (skin depths, gyroradii, frequencies), entropy, and more. All as pure functions: arrays in, arrays out.
- **Unit system** — PIC (electron- or ion-referenced), MHD (Alfvén-speed-based), SI, or custom normalization. Round-trip `normalize()` / `to_si()` with display unit conversion (`"nT"`, `"km/s"`, etc.).
- **Geometry-aware operators** — divergence, curl, gradient with coordinate metric factors. Cartesian implemented; spherical/cylindrical planned.
- **Selections** — `PlaneSelection` and `BoxSelection` for slicing 3D data into lower-dimensional views.
- **Reductions** — `pypic.reduce(ds, axis, reduction=...)` collapses fields along one or more axes (trapezoidal `integrate`, `mean`/`median`/`sum`, `argmax`/`argmin` returning coordinate positions, etc.). Pairs with selections for column densities, slab averages, density-weighted line averages (`weight=`), and projected-peak diagnostics.
- **Field registry** — `compute("beta")`, `compute("|B|")`, `compute("v_A")` dispatches to the right derived function. Extensible via `register_field()`.

## Requirements

Python 3.13+. Core dependencies: NumPy, SciPy, xarray, h5py.

## Dev commands

```sh
uv run pytest -v                      # full suite (tests + doctests)
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run mypy src                       # type check
```

Generate 2D visual test plots (all themes, output in `tests/output/`):
```sh
uv run python tests/visual_plots.py
uv run python tests/visual_plots.py --theme dark  # single theme
```

Interactive 3D dipole with field line tracing and Bz slice:
```sh
uv run python tests/visual_dipole_3d.py
```

Optional dead code test using vulture:
```
uv run vulture src/ tests/ vulture_whitelist.py
```
Optional Sloppy code test:

```
uvx sloppylint src/ tests/
```

## Benchmarks

Manual performance benchmarks live in `benchmarks/`. Not run by CI — use
them to verify the batched/vectorized paths still pay after a kernel
change. Each script is self-contained, reproducible (seeded RNG), and
prints a small table:

```sh
uv run python benchmarks/bench_batched_tracer.py  # batched vs scalar adaptive tracer
```
