r"""Example: Schindler 3D reconnection criterion on a Harris current sheet.

The Schindler-Hesse-Birn 1988 criterion defines reconnection as
$\Xi(\mathbf{x}_0) = \int_{\mathcal{L}} E_\parallel \, d\ell \neq 0$
along a magnetic field line $\mathcal{L}$ threading the seed point
$\mathbf{x}_0$. Unlike the 2D X-point definition, it does **not**
require a magnetic null — making it the correct diagnostic for the
guide-field reconnection regime that dominates collisionless plasmas
in nature.

This example builds a 2.5D guide-field Harris-sheet-like configuration
analytically and seeds a 2D grid of $\Xi$ probes across the current
sheet. The rendered figure shows three panels — source, integrand,
integral — that together tell the Schindler story:

1. The non-ideal source $E_z(x, y, 0)$ — compactly localized at the
   would-be X-point.
2. The Schindler integrand $E_\parallel = E_z\, B_z/|\mathbf{B}|$
   at the midplane — compressed in $x$ relative to $E_z$ because
   $\hat{b}_z = B_z/|\mathbf{B}|$ falls off rapidly outside the
   sheet, where $|B_y|$ grows.
3. The Schindler $\Xi$ map, which propagates the localized
   $E_\parallel$ information *along* field lines and accumulates a
   sharp peak at the X-line — even though there is no magnetic null
   anywhere ($|\mathbf{B}| \ge B_g = 0.1$ throughout the domain).

Run::

    uv run python examples/ex_schindler_xi.py

The figure is written next to this script. Set
``PYPIC_EXAMPLE_OUTPUT_DIR`` to send it elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pypic import FieldDataset, GridInfo
from pypic.reconnection import schindler_xi

# ----------------------------------------------------------------------
# Build a 2.5D Harris-sheet-like field on a uniform grid.
# Domain x ∈ [-8, 8] (sheet-normal), y ∈ [-4, 4] (along-sheet),
# z ∈ [-4, 4] (guide). The sheet lies in the x = 0 plane and the
# X-line points along +z.
# ----------------------------------------------------------------------
NX, NY, NZ = 48, 32, 16
DX, DY, DZ = 16.0 / NX, 8.0 / NY, 8.0 / NZ  # cell sizes
ORIGIN = (-8.0, -4.0, -4.0)
LAMBDA = 1.0  # sheet half-thickness (ion skin depths)
B0 = 1.0  # asymptotic in-plane field
BG = 0.1  # guide field — keeps |B| ≥ 0.1 everywhere, so the domain has
# *no* 3D magnetic null. Schindler Ξ still localizes the X-line; the
# 2D X-point definition would fail here because there is no null to
# find.
L_LOC = 2.0  # localization length of E_par in the along-sheet (y) direction
E0 = 0.05  # peak non-ideal E

grid = GridInfo(dimensions=(NX, NY, NZ), spacing=(DX, DY, DZ), origin=ORIGIN)
xx, yy, _ = (
    a.astype(np.float64) for a in np.meshgrid(*grid.coordinate_arrays(), indexing="ij")
)

# Harris sheet: B_y(x) reverses across x = 0, with weak guide B_z.
b1 = np.zeros_like(xx)  # B_x = 0
b2 = B0 * np.tanh(xx / LAMBDA)  # B_y(x) reversal
b3 = np.full_like(xx, BG)  # B_z (guide)

# Non-ideal field E_z peaked at the X-point (origin), compact in (x, y).
e1 = np.zeros_like(xx)
e2 = np.zeros_like(xx)
e3 = E0 / (np.cosh(xx / LAMBDA) ** 2 * np.cosh(yy / L_LOC) ** 2)

data = FieldDataset.from_arrays(
    {"B_1": b1, "B_2": b2, "B_3": b3, "E_1": e1, "E_2": e2, "E_3": e3},
    grid,
)

# ----------------------------------------------------------------------
# Seed a 2D grid of Xi probes in the z = 0 midplane around the X-point.
# Tightened to x ∈ [-2, 2] — the Xi structure lives within ~1λ of the
# sheet.  Each seed launches an RK4 field-line trace; schindler_xi
# integrates E_par = E · b̂ along arc length and returns one Xi value
# per seed.
# ----------------------------------------------------------------------
seed_x = np.linspace(-2.0, 2.0, 41)
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
    f"|Xi|_max = {np.nanmax(np.abs(xi_map)):.4f} "
    f"at (x, y) = ({peak_x:.2f}, {peak_y:.2f})"
)
print(f"Xi at peak = {peak_xi:+.4f}")
print(f"Xi mean (finite cells) = {np.nanmean(xi_map):+.4f}")

# Quick sanity check: Xi should be positive and peak at the X-line (origin),
# since E_z is positive there and the guide field steers traces through it.
assert peak_xi > 0, "Xi should be positive — E_z is positive at the X-point"
assert abs(peak_x) < 0.2, "Peak should sit on the X-line (x = 0)"
assert abs(peak_y) < 0.2, "Peak should sit on the X-line (y = 0)"

try:
    import matplotlib.pyplot as plt

    from pypic.plotting import add_colorbar, get_theme, set_theme, use_theme
except ImportError:
    print("(matplotlib not available — skipping multi-panel plot)")
else:
    # Midplane slices for the source / integrand panels.
    iz_mid = NZ // 2
    xx_mid = xx[:, :, iz_mid]
    yy_mid = yy[:, :, iz_mid]
    ez_mid = e3[:, :, iz_mid]
    # Schindler integrand at z = 0: E_par = E · b̂ = E_z * B_z / |B|
    # since E_x = E_y = 0 and B_x = 0 by construction.
    b_mag_mid = np.sqrt(b2[:, :, iz_mid] ** 2 + b3[:, :, iz_mid] ** 2)
    e_par_mid = ez_mid * b3[:, :, iz_mid] / b_mag_mid

    # Pick a pypic theme. Switch to "light", "synthwave", "lcars", etc.
    THEME = "dark"
    set_theme(THEME)
    theme = get_theme()
    seq_cmap = theme.sequential_cmap

    with use_theme(theme):
        # Each panel gets its own colorbar attached via make_axes_locatable
        # (inside add_colorbar), so sharey + constrained_layout would fight
        # the divider's child axes. Use tight_layout and pin ylim per panel.
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

        # Panel 1 — E_z (non-ideal source)
        ax = axes[0]
        ez_levels = np.linspace(0.0, E0, 11)
        cs = ax.contourf(xx_mid, yy_mid, ez_mid, levels=ez_levels, cmap=seq_cmap)
        ax.set_title(r"$E_z(x, y, 0)$  —  non-ideal source")
        ax.set_xlabel("x  (sheet-normal)")
        ax.set_ylabel("y  (along-sheet)")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-3.0, 3.0)
        cb = add_colorbar(fig, ax, cs, r"$E_z$", extend="neither", extremes=None)
        cb.set_ticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

        # Panel 2 — Schindler integrand E_par(x, y, 0).  Same sequential
        # scale as panel 1 so the eye can compare directly: the integrand
        # is the source weighted by b̂_z = B_z/|B|, which collapses
        # rapidly off the sheet as |B_y| grows.
        ax = axes[1]
        cs = ax.contourf(xx_mid, yy_mid, e_par_mid, levels=ez_levels, cmap=seq_cmap)
        ax.contour(
            xx_mid,
            yy_mid,
            ez_mid,
            levels=ez_levels[1::2],
            colors="white",
            linewidths=0.4,
            alpha=0.45,
        )
        ax.plot(
            0.0,
            0.0,
            "wx",
            markersize=11,
            markeredgewidth=2.0,
            label=r"X-line $\perp$ page",
        )
        ax.set_title(r"$E_\parallel = E_z\,B_z/|B|$  —  Schindler integrand")
        ax.set_xlabel("x  (sheet-normal)")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-3.0, 3.0)
        cb = add_colorbar(
            fig, ax, cs, r"$E_\parallel$", extend="neither", extremes=None
        )
        cb.set_ticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # Panel 3 — Schindler Xi
        ax = axes[2]
        xi_levels = np.linspace(0.0, 0.4, 9)
        cs = ax.contourf(sx, sy, xi_map, levels=xi_levels, cmap=seq_cmap, extend="max")
        ax.contour(
            sx,
            sy,
            xi_map,
            levels=xi_levels,
            colors=theme.secondary_text_color[:3],
            linewidths=0.3,
            alpha=0.5,
        )
        ax.plot(
            peak_x,
            peak_y,
            "x",
            color=theme.accent_color,
            markersize=11,
            markeredgewidth=2.0,
            label=rf"peak $\Xi = {peak_xi:.3f}$",
        )
        ax.set_title(r"Schindler $\Xi = \int E_\parallel \, d\ell$")
        ax.set_xlabel("x  (sheet-normal)")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-3.0, 3.0)
        cb = add_colorbar(fig, ax, cs, r"$\Xi$", extend="max")
        cb.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        fig.suptitle(
            r"Schindler $\Xi$ in guide-field reconnection  —  "
            r"no 3D null, yet the X-line is localized",
            fontsize=11,
        )
        fig.tight_layout()

        out_dir = Path(
            os.environ.get("PYPIC_EXAMPLE_OUTPUT_DIR", Path(__file__).parent)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ex_schindler_xi.png"
        fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
        print(f"Wrote {out}")

print("OK")
