# Getting Started

## Installation

pypic requires Python 3.13+. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv add pypic-plasma
```

The distribution is named `pypic-plasma`; the import name is `pypic`.

Core dependencies (NumPy, SciPy, xarray, h5py, pydantic) are installed
automatically. Everything heavier sits behind an extra, and extras compose:

```sh
uv add "pypic-plasma[plot,cli]"        # what most installs want
uv add "pypic-plasma[plot,zarr,cli]"   # ... plus modern I/O
```

| Extra | Pulls in | Enables |
|---|---|---|
| `plot` | matplotlib | Field slices, comparisons, line plots, kymographs, quiver/streamlines, spectra, and the theme system |
| `3d` | pyvista | 3D rendering and field-line visualization |
| `cli` | typer, rich | the `pypic` command |
| `zarr` | zarr, numcodecs, virtualizarr, icechunk | Zarr v3 export/import, VirtualiZarr views over legacy HDF5, Icechunk storage |
| `icechunk` | icechunk, zarr, numcodecs | Icechunk versioned storage without the VirtualiZarr dependency |
| `arrow` | pyarrow | Parquet/Arrow particle I/O |
| `duckdb` | duckdb, pyarrow | SQL queries over particle Parquet |
| `server` | fastapi, uvicorn, pyarrow, websockets | the Arrow IPC server behind `pypic serve` |

Working from a checkout instead:

```sh
git clone https://github.com/rusaitis/pypic.git && cd pypic
uv sync --all-extras --all-groups
```

## Without simulation data

Everything below works on a dataset you build yourself, so you can try
pypic before you have output from a code. `FieldDataset.from_arrays`
takes a dict of NumPy arrays plus a `GridInfo`:

```python
import numpy as np
from pypic import CARTESIAN, FieldDataset, GridInfo

nx, ny, nz = 32, 32, 1
grid = GridInfo(
    dimensions=(nx, ny, nz),
    spacing=(0.5, 0.5, 1.0),
    origin=(-8.0, -8.0, 0.0),
    geometry=CARTESIAN,
)

# A Harris current sheet: B_x reverses across y, pressure balances it.
y = grid.origin[1] + (np.arange(ny) + 0.5) * grid.spacing[1]
bx = np.tanh(y / 2.0)[None, :, None] * np.ones((nx, ny, nz))

data = FieldDataset.from_arrays(
    {
        "B_1": bx,
        "B_2": np.zeros((nx, ny, nz)),
        "B_3": np.zeros((nx, ny, nz)),
        "rho_m": np.ones((nx, ny, nz)),
        "P": 0.5 * (1.0 - bx**2) + 0.1,
    },
    grid,
)

print(data.compute("|B|").max())   # -> 0.999, saturating at the edges
print(data.compute("beta").max())  # -> 76.6, pressure-dominated at the centre
```

Field names must resolve through the registry — see
[Schema § 3](schema.md#3-canonical-field-names) for the canonical set.
Pass `strict_fields=False` to allow unregistered names through.

This is also the entry point a new reader uses:
[`examples/custom_reader_example.py`](https://github.com/rusaitis/pypic/blob/main/examples/custom_reader_example.py)
is a self-contained script that generates a synthetic HDF5 file, maps its
native names onto canonical ones, and reads it back through
`open_simulation`.

## Loading simulation data

pypic auto-detects simulation formats (iPIC3D, BATSRUS, OpenGGCM):

```python
from pypic import open_simulation

sim = open_simulation("path/to/output")
print(sim.describe())           # metadata, grid, species
print(sim.steps)                # available timesteps

data = sim.read(step=0)         # load fields for timestep 0
print(data.field_names())       # canonical field names
```

For selective loading (faster for large datasets):

```python
data = sim.read(step=0, fields=["B", "rho_m", "P"])
```

Vector shorthand expands automatically: `"B"` loads `B_1`, `B_2`, `B_3`.

## Computing derived quantities

All derived quantities are computed from the canonical fields:

```python
beta = data.compute("beta")       # plasma beta (2P/B²)
v_a = data.compute("v_A")         # Alfvén speed
mach = data.compute("M_A")        # Alfvén Mach number

# Attach to the dataset for reuse
data = data.with_derived("|B|", "beta", "v_A")
```

See the [field registry](api/fields.md) for the full list of
computable quantities.

## Unit conversion

All computation uses normalized code units. Convert at display boundaries:

```python
from pypic import field_info

# SI conversion
b_si = data.in_si("B_1")          # Tesla
v_si = data.in_si("v_A")         # m/s

# Display units
b_nt = data.in_units("B_1", "nT")
v_kms = data.in_units("v_A", "km/s")

# Check metadata
info = field_info("|B|")
print(info.long_name, info.si_unit)  # "Magnetic field magnitude" "T"
```

## Selecting regions

Slice 3D data into lower-dimensional views:

```python
from pypic import PlaneSelection, BoxSelection

# XY plane at the midpoint
plane = PlaneSelection(normal="z")
slice_2d = plane.apply(data)

# Subvolume
box = BoxSelection(ranges={"x": (10, 50), "y": (20, 40)})
subregion = box.apply(data)
```

## Plotting

Basic 2D field visualization:

```python
from pypic.plotting import plot_field_slice

plot_field_slice(data, field="|B|", plane="xy")
```

For comparison of two simulations:

```python
from pypic.plotting import plot_comparison

plot_comparison(data_a, data_b, field="beta", plane="xz")
```
