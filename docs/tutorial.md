# Tutorial: Analysis Workflow

A complete walkthrough: load simulation data, compute derived fields,
select regions, convert units, and make publication figures.

## 1. Load and inspect

```python
from pypic import open_simulation

sim = open_simulation("runs/reconnection/")
print(sim.describe())
# Simulation: iPIC3D (PIC)
#   Path:    runs/reconnection
#   Grid:    256 x 128 x 1 (cartesian)
#   Spacing: 0.10 x 0.10 x 1.00
#   Species: electrons (q/m=-256.0), ions (q/m=1.0)
#   Steps:   11 [0..1000]

data = sim.read(step=500, fields=["B", "E", "rho_c", "P"])
```

## 2. Compute derived quantities

```python
# Attach several derived fields at once
data = data.with_derived("|B|", "beta", "v_A", "J_dot_E")

# Per-species quantities
data = data.with_derived("omega_pe", "d_e", "Te")
```

The `with_derived` method returns a new dataset with the computed
fields attached. Dependencies are resolved automatically — requesting
`"beta"` loads `"|B|"` and `"P"` if not already present.

## 3. Select a 2D plane

```python
from pypic import PlaneSelection

# XY midplane (default index = middle of z-axis)
xy_plane = PlaneSelection(normal="z")
slice_2d = xy_plane.apply(data)

# Specific z-index
xz_slice = PlaneSelection(normal="y", index=64).apply(data)
```

## 4. Convert to SI for display

```python
# Magnetic field in nanoTesla
b_nt = slice_2d.in_units("|B|", "nT")

# Electron temperature: in_si gives joules, in_units gives eV
te_joules = slice_2d.in_si("Te")
te_ev = slice_2d.in_units("Te", "eV")
```

## 5. Plot a field slice

```python
from pypic.plotting import plot_field_slice

fig, ax = plot_field_slice(slice_2d, field="J_dot_E")
fig.savefig("j_dot_e.png", dpi=150)
```

The colormap is chosen automatically: diverging for signed fields
(like $\mathbf{J} \cdot \mathbf{E}$), sequential for positive-definite
fields (like $|\mathbf{B}|$).

## 6. Compare two runs

```python
from pypic.plotting import plot_comparison

sim_a = open_simulation("runs/low_resistivity/")
sim_b = open_simulation("runs/high_resistivity/")

data_a = sim_a.read(step=500, fields=["B", "P"])
data_b = sim_b.read(step=500, fields=["B", "P"])

data_a = data_a.with_derived("beta")
data_b = data_b.with_derived("beta")

plane = PlaneSelection(normal="z")

plot_comparison(
    plane.apply(data_a),
    plane.apply(data_b),
    field="beta",
)
```

This produces a three-panel figure: run A, run B, and the pointwise
difference.

## 7. Frame transforms

If `simulation.toml` defines coordinate transforms:

```python
# Transform from simulation frame to GSM
data_gsm = data.transform_to("GSM")
```

## 8. Relativistic workflows

For relativistic PIC codes (TRISTAN-MP, Zeltron, OSIRIS):

```python
data = data.with_derived("gamma_L", "sigma")

# Relativistic Alfvén speed (bounded by c)
from pypic import alfven_speed
v_a_rel = alfven_speed(data["|B|"], data["rho_m"], c=1.0)

# Lorentz factor from four-velocity (numerically stable).
# There is no "|u|" recipe — take the magnitude of the components.
import numpy as np

from pypic import lorentz_factor_from_four_velocity

u_mag = np.sqrt(sum(data[f"u_{i}"] ** 2 for i in (1, 2, 3)))
gamma = lorentz_factor_from_four_velocity(u_mag, c=1.0)
```

## Next steps

- [Equations reference](equations.md) — every formula with LaTeX
- [API reference](api/derived.md) — full function signatures
- [Schema](schema.md) — canonical field names and data contract
