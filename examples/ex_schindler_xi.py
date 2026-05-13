"""Example: Schindler 3D reconnection criterion on a Harris current sheet.

The Schindler-Hesse-Birn 1988 criterion defines reconnection as
$\\Xi(\\mathbf{x}_0) = \\int_{\\mathcal{L}} E_\\parallel \\, d\\ell \\neq 0$
along a magnetic field line $\\mathcal{L}$ threading the seed point
$\\mathbf{x}_0$. Unlike the 2D X-point definition, it does not require
a magnetic null and works in fully 3D geometries.

This example builds a 2.5D Harris-sheet-like configuration analytically,
seeds a grid of $\\Xi$ probes across the current sheet, and traces field
lines through ``schindler_xi`` to produce a 2D map of the reconnection
diagnostic. The peak Xi accumulates where field lines spend the most
time in the region of non-zero $E_\\parallel$ — i.e. near the X-point
in the guide-field current sheet.

Run::

    uv run python examples/ex_schindler_xi.py
"""

from __future__ import annotations

import numpy as np

from pypic import FieldDataset, GridInfo
from pypic.reconnection import schindler_xi

# ----------------------------------------------------------------------
# Build a 2.5D Harris-sheet-like field on a uniform grid.
# Domain x ∈ [-8, 8] (sheet-normal), y ∈ [-4, 4] (along-sheet),
# z ∈ [-4, 4] (out-of-plane). The sheet lies in the x = 0 plane.
# ----------------------------------------------------------------------
NX, NY, NZ = 48, 32, 16
DX, DY, DZ = 16.0 / NX, 8.0 / NY, 8.0 / NZ  # cell sizes
ORIGIN = (-8.0, -4.0, -4.0)
LAMBDA = 1.0  # sheet half-thickness (ion skin depths)
B0 = 1.0  # asymptotic in-plane field
BG = 0.1  # guide field — keeps |B| > 0 at the sheet center
L_LOC = 2.0  # localization length of E_par in the along-sheet (y) direction
E0 = 0.05  # peak non-ideal E

grid = GridInfo(dimensions=(NX, NY, NZ), spacing=(DX, DY, DZ), origin=ORIGIN)
xx, yy, _zz = (
    a.astype(np.float64) for a in np.meshgrid(*grid.coordinate_arrays(), indexing="ij")
)

# Harris sheet: B_y(x) reverses across x = 0, with weak guide B_z.
# (We use x as the sheet-normal so traces can be visualised in x-y.)
b1 = np.zeros_like(xx)  # B_x = 0
b2 = B0 * np.tanh(xx / LAMBDA)  # B_y(x) reversal
b3 = np.full_like(xx, BG)  # B_z (guide)

# Non-ideal field E_z peaked at the X-point (origin), compact in (x, y).
e1 = np.zeros_like(xx)
e2 = np.zeros_like(xx)
e3 = E0 / (np.cosh(xx / LAMBDA) ** 2 * np.cosh(yy / L_LOC) ** 2)

fields = {
    "B_1": b1,
    "B_2": b2,
    "B_3": b3,
    "E_1": e1,
    "E_2": e2,
    "E_3": e3,
}
data = FieldDataset.from_arrays(fields, grid)

# ----------------------------------------------------------------------
# Seed a 2D grid of Xi probes in the z = 0 midplane around the X-point.
# Each seed launches an RK4 field-line trace; schindler_xi integrates
# E_par = E · b̂ along arc length and returns one Xi value per seed.
# ----------------------------------------------------------------------
seed_x = np.linspace(-6.0, 6.0, 25)
seed_y = np.linspace(-3.0, 3.0, 13)
sx, sy = np.meshgrid(seed_x, seed_y, indexing="ij")
seeds = np.stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)], axis=1)

xi = schindler_xi(data, seeds, step_size=0.25, max_steps=4000)
xi_map = xi.reshape(sx.shape)

# ----------------------------------------------------------------------
# Report and (optionally) plot.
# ----------------------------------------------------------------------
finite = np.isfinite(xi_map)
peak_idx = np.unravel_index(
    np.argmax(np.where(finite, np.abs(xi_map), -1)), xi_map.shape
)
peak_x = sx[peak_idx]
peak_y = sy[peak_idx]
peak_xi = xi_map[peak_idx]

print(f"Grid: {NX}×{NY}×{NZ}, seeds: {sx.size}")
print(f"Finite Xi values: {int(finite.sum())} / {sx.size}")
print(
    f"|Xi|_max = {np.nanmax(np.abs(xi_map)):.4f} at (x, y) = ({peak_x:.2f}, {peak_y:.2f})"
)
print(f"Xi at peak = {peak_xi:+.4f}")
print(f"Xi mean (finite cells) = {np.nanmean(xi_map):+.4f}")

# Quick sanity check: Xi should be positive and peak near the X-point (origin),
# since E_z is positive there and the guide field steers traces through it.
assert peak_xi > 0, "Xi should be positive — E_z is positive at the X-point"
assert abs(peak_x) < 2.0 and abs(peak_y) < 2.0, "Peak should sit near the X-point"

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("(matplotlib not available — skipping contour plot)")
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = np.linspace(0.0, np.nanmax(xi_map), 12)
    cs = ax.contourf(sx, sy, xi_map, levels=levels, cmap="viridis")
    ax.contour(sx, sy, xi_map, levels=levels, colors="k", linewidths=0.3, alpha=0.6)
    ax.plot(peak_x, peak_y, "rx", markersize=10, label=f"peak Xi = {peak_xi:.3f}")
    ax.set_xlabel("x  (sheet-normal)")
    ax.set_ylabel("y  (along-sheet)")
    ax.set_title(r"Schindler $\Xi = \int E_\parallel \, d\ell$  —  Harris sheet")
    ax.legend(loc="upper right")
    fig.colorbar(cs, ax=ax, label=r"$\Xi$")
    fig.tight_layout()
    out = "ex_schindler_xi.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")

print("OK")
