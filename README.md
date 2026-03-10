# pypic

A Python toolkit for reading, analyzing, and plotting plasma simulation output.

## What's implemented

- **`coordinates`** — Coordinate geometry definitions (Cartesian, cylindrical, spherical) with scale factors and Jacobians
- **`units`** — Normalization system, physical constants, and species info for converting between code and SI units
- **`readers`** — Base reader infrastructure: `FieldDataset` (xarray wrapper), `GridInfo`, and `SimulationReader` ABC

## Planned

- **Derived quantities** — Pure-function physics computations (Alfven speed, plasma beta, current density, etc.)
- **Diagnostics** — Energy budgets, conservation checks, spectral analysis
- **Selections** — Region descriptors (`PlaneSelection`, `BoxSelection`) that slice `FieldDataset`
- **Concrete readers** — One module per simulation code (FLEKS, OpenGGCM, etc.)
- **Plotting** — matplotlib-based visualization utilities

## Dev commands

```sh
uv run pytest -v                      # full suite (tests + doctests)
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run mypy src                       # type check
```
