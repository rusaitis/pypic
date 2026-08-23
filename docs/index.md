---
title: pypic
---

<p align="center">
  <img src="assets/pypic-logo.png" alt="pypic" width="440">
</p>

<p align="center"><em>Python for Plasma In Cells</em></p>

A Python toolkit for reading, analyzing, and plotting plasma simulation output from PIC and MHD codes.

Every simulation code invents its own file layout, field names, and
normalization. pypic maps them all onto one canonical schema, so an analysis
written against iPIC3D output runs unchanged against BATSRUS or OpenGGCM.
Computation happens in normalized code units using pure NumPy functions —
xarray is the container, not the compute engine — and SI conversion is applied
only at I/O and display boundaries.

## Install

```sh
uv add "pypic-plasma[plot,cli]"
```

Python 3.13+. The core install pulls in NumPy, SciPy, xarray, h5py, and
pydantic; everything heavier sits behind an extra. See
[Getting Started](getting-started.md#installation) for the full table of the
eight extras and what each one enables.

## Quick start

```python
from pypic import open_simulation, PlaneSelection

# Format is auto-detected — iPIC3D, BATSRUS, OpenGGCM, or generic HDF5.
sim = open_simulation("path/to/output")
data = sim.read(step=0, fields=["B", "E", "P_s0"])

# Derived quantities dispatch through the field registry.
beta = data.compute("beta")
v_a = data.compute("v_A")

# Code units internally; convert at the display boundary.
b_nt = data.in_units("B_1", "nT")

# Selections describe regions and return an ordinary FieldDataset.
midplane = PlaneSelection(normal="z").apply(data)
```

## Where to go next

| Page | Contents |
|------|----------|
| [Getting Started](getting-started.md) | Installation, a zero-data on-ramp, loading data, first derived quantities |
| [Tutorial](tutorial.md) | End-to-end analysis walkthrough, from load to publication figure |
| [Architecture](architecture.md) | The design rules behind the code, and the conventions a contribution follows |
| [Equations](equations.md) | Every derived quantity with its LaTeX form and SI conversion |
| [Conventions](conventions.md) | Thermal speed, $\gamma$, temperature-in-energy-units, and the other choices that differ between textbooks |
| [Schema](schema.md) | The `simulation.toml` contract and canonical field names |
| [Command Line](api/cli.md) | The `pypic` command, subcommand by subcommand |

The [API reference](api/containers.md) covers each public module in turn —
readers, derived quantities, selections, reductions, I/O, plotting, tracing,
and the Arrow IPC server.

## Ecosystem

pypic is the Python half of a three-part toolchain built around the shared
[`simulation.toml` schema](schema.md): **rustpic** (a Rust PIC/MHD solver) writes
the schema, pypic reads and analyzes it, and **webpic** (Three.js/WebGPU) renders
it in the browser over the Arrow IPC server in [`pypic.server`](api/server.md).
Both siblings are in development and not yet public — you will see them named in
a few docstrings and in the `[webpic]` block of the bundled plot themes. pypic is
fully usable on its own; nothing here depends on either of them.

## Status

pypic is early-stage research software (0.1.x) under active development. The
core is in daily use and is covered by ~2800 tests, including Hypothesis
property tests, hand-calculated physics values, and NRL Formulary cross-checks.
The public API may still change before 1.0.
