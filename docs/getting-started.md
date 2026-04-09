# Getting Started

## Installation

pypic requires Python 3.13+. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv add pypic
```

Core dependencies (NumPy, SciPy, xarray, h5py) are installed automatically.
For plotting, add matplotlib:

```sh
uv add pypic matplotlib
```

## Loading simulation data

pypic auto-detects simulation formats (iPIC3D, BATSRUS, OpenGGCM):

```python
from pypic import open_simulation

sim = open_simulation("path/to/output")
print(sim.describe())           # metadata, grid, species
print(sim.timesteps)            # available timesteps

data = sim.read(step=0)         # load fields for timestep 0
print(data.field_names())       # canonical field names
```

For selective loading (faster for large datasets):

```python
data = sim.read(step=0, fields=["B", "rho_m", "P"])
```

Vector shorthand expands automatically: `"B"` loads `B1`, `B2`, `B3`.

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
b_si = data.in_si("B1")          # Tesla
v_si = data.in_si("v_A")         # m/s

# Display units
b_nt = data.in_units("B1", "nT")
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
