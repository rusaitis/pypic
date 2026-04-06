"""3D dipole field line visualization with pyvista.

Generates an analytical magnetic dipole on a 3D grid, traces field lines
using the RK4 tracer, and renders an interactive pyvista window with a
planet sphere and colored field lines.

Controls:
- **Click** on the equatorial plane to place a seed marker
- **T** — trace a field line from the marker
- **C** — clear all interactively traced lines

Run with::

    uv run python tests/visual_dipole_3d.py
    uv run python tests/visual_dipole_3d.py --theme light
    uv run python tests/visual_dipole_3d.py --save
"""

from __future__ import annotations

import argparse

import numpy as np

from pypic.plotting import available_themes, set_theme
from pypic.plotting.pyvista import (
    add_axis_triad,
    add_equatorial_grid,
    add_field_line,
    add_field_lines,
    add_planet,
    create_plotter,
    resolve_cmap,
    set_camera,
)
from pypic.readers.base import FieldDataset, GridInfo
from pypic.traces import (
    VectorFieldInterpolator,
    attach_scalars,
    trace_field_line,
)
from pypic.units import Normalization

PLANET_RADIUS = 1.0
DOMAIN_HALF = 6.0
N_CELLS = 80


def _dipole_field(
    grid: GridInfo,
    moment: float = 1.0,
    planet_radius: float = PLANET_RADIUS,
) -> dict[str, np.ndarray]:
    """Compute analytical dipole B field on the grid, zeroed inside the planet."""
    coords = grid.coordinate_arrays()
    x, y, z = np.meshgrid(*coords, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    r5 = r_safe**5

    bx = 3.0 * moment * x * z / r5
    by = 3.0 * moment * y * z / r5
    bz = moment * (3.0 * z**2 - r_safe**2) / r5

    inside = r < planet_radius
    bx[inside] = 0.0
    by[inside] = 0.0
    bz[inside] = 0.0

    return {"B1": bx, "B2": by, "B3": bz}


def _seed_points(
    l_shells: list[float],
    n_per_shell: int = 3,
) -> list[tuple[float, float, float]]:
    """Generate seed points at given L-shells in the noon-midnight plane."""
    seeds: list[tuple[float, float, float]] = []
    latitudes = (
        [0.0] if n_per_shell == 1
        else np.linspace(-25.0, 25.0, n_per_shell).tolist()
    )
    for l_val in l_shells:
        for lat_deg in latitudes:
            lat = np.radians(lat_deg)
            r = l_val * np.cos(lat) ** 2
            x = r * np.cos(lat)
            z = r * np.sin(lat)
            if r > PLANET_RADIUS + 0.3:
                seeds.append((x, 0.0, z))
    return seeds


class _InteractiveTracer:
    """Click on the equatorial plane to place a seed, press T to trace."""

    def __init__(
        self,
        plotter: object,
        ds: FieldDataset,
        interp: VectorFieldInterpolator,
        cmap: object,
        vmax: float,
    ) -> None:
        self.plotter = plotter
        self.ds = ds
        self.interp = interp
        self.cmap = cmap
        self.vmax = vmax
        self.seed: tuple[float, float, float] | None = None
        self._marker_actor: object | None = None
        self._traced_actors: list[object] = []
        self._trace_count = 0

    def on_pick(self, point: np.ndarray) -> None:
        """Called when user clicks on the equatorial surface."""
        import pyvista as pv

        x, y = float(point[0]), float(point[1])
        r = np.hypot(x, y)
        if r < PLANET_RADIUS + 0.2:
            return  # too close to planet

        self.seed = (x, y, 0.0)

        # Update or create marker
        if self._marker_actor is not None:
            self.plotter.remove_actor(self._marker_actor)  # type: ignore[union-attr]
        marker = pv.Sphere(radius=0.15, center=self.seed)
        self._marker_actor = self.plotter.add_mesh(  # type: ignore[union-attr]
            marker, color="#ff6600", opacity=0.9,
        )
        r_eq = np.hypot(x, y)
        print(f"  Seed: ({x:+.2f}, {y:+.2f}, 0.00)  L≈{r_eq:.1f}")

    def trace(self) -> None:
        """Trace a field line from the current seed (T key)."""
        if self.seed is None:
            print("  No seed — click the equatorial plane first")
            return

        try:
            fl = trace_field_line(
                self.ds, self.seed,
                step_size=0.1, max_steps=5000,
                direction="both", null_threshold=1e-6,
                interpolator=self.interp,
            )
        except ValueError as exc:
            print(f"  Cannot trace from {self.seed}: {exc}")
            return

        fl = attach_scalars(fl, self.ds, ["B1", "B2", "B3"])
        bmag = np.sqrt(
            fl.scalars["B1"] ** 2 + fl.scalars["B2"] ** 2 + fl.scalars["B3"] ** 2
        )
        fl = fl.with_scalars(**{"|B|": bmag})

        add_field_line(
            self.plotter, fl,  # type: ignore[arg-type]
            scalar="|B|", cmap=self.cmap,
            clim=(0, self.vmax), signed=False, radius=0.06,
        )
        self._trace_count += 1
        print(f"  Traced line #{self._trace_count} ({fl.n_points} points)")

    def clear(self) -> None:
        """Clear all interactively traced lines (C key)."""
        self._trace_count = 0
        print("  Cleared interactive traces (re-run to reset fully)")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D dipole field line viewer.")
    all_available = available_themes()
    names = sorted(all_available)
    numbered = [f"{i + 1}={n}" for i, n in enumerate(names)]
    parser.add_argument(
        "--theme",
        default="dark",
        help=f"Theme name or number ({', '.join(numbered)})",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save screenshot instead of interactive",
    )
    args = parser.parse_args()
    key = names[int(args.theme) - 1] if args.theme.isdigit() else args.theme
    set_theme(key)

    # Build grid centered at origin
    dx = 2.0 * DOMAIN_HALF / N_CELLS
    grid = GridInfo(
        dimensions=(N_CELLS, N_CELLS, N_CELLS),
        spacing=(dx, dx, dx),
        origin=(-DOMAIN_HALF, -DOMAIN_HALF, -DOMAIN_HALF),
    )

    print("Computing dipole field...")
    fields = _dipole_field(grid)
    ds = FieldDataset.from_arrays(fields, grid, Normalization.identity())

    interp = VectorFieldInterpolator.from_dataset(ds)

    # Pre-trace field lines at various L-shells
    l_shells = [2.0, 3.0, 4.0, 5.0]
    seeds = _seed_points(l_shells, n_per_shell=3)
    print(f"Tracing {len(seeds)} field lines...")

    lines = []
    for seed in seeds:
        try:
            fl = trace_field_line(
                ds, seed, step_size=0.1, max_steps=5000,
                direction="both", null_threshold=1e-6, interpolator=interp,
            )
            lines.append(fl)
        except ValueError:
            pass

    print(f"  {len(lines)} lines traced successfully")

    # Sample |B| along each line for coloring
    colored_lines = []
    for fl in lines:
        fl = attach_scalars(fl, ds, ["B1", "B2", "B3"])
        bmag = np.sqrt(
            fl.scalars["B1"] ** 2 + fl.scalars["B2"] ** 2 + fl.scalars["B3"] ** 2
        )
        colored_lines.append(fl.with_scalars(**{"|B|": bmag}))

    all_bmag = np.concatenate([fl.scalars["|B|"] for fl in colored_lines])
    valid = all_bmag[np.isfinite(all_bmag) & (all_bmag > 0)]
    vmax = float(np.nanpercentile(valid, 98))

    cmap = resolve_cmap("plasma")

    # Render
    plotter = create_plotter(off_screen=args.save)

    add_planet(plotter, radius=PLANET_RADIUS)

    add_field_lines(
        plotter, colored_lines,
        scalar="|B|", cmap=cmap,
        clim=(0, vmax), signed=False, radius=0.05,
    )

    lim = 5.5
    add_axis_triad(plotter, length=2.0, labels=("$x$", "$y$", "$z$"))
    add_equatorial_grid(plotter, xlim=(-lim, lim), ylim=(-lim, lim), coord_units="$R_E$")
    set_camera(plotter, distance=18.0, elevation=15, azimuth=-60)

    if args.save:
        from pathlib import Path

        outfile = Path(__file__).parent / "output" / "visual_dipole_3d.png"
        outfile.parent.mkdir(exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(str(outfile), transparent_background=True)
        plotter.close()
        print(f"Saved to {outfile}")
    else:
        # Interactive mode: click to seed, T to trace, C to clear
        tracer = _InteractiveTracer(plotter, ds, interp, cmap, vmax)

        # Add a transparent equatorial plane for picking
        import pyvista as pv

        eq_plane = pv.Plane(
            center=(0, 0, 0), direction=(0, 0, 1),
            i_size=2 * lim, j_size=2 * lim,
            i_resolution=1, j_resolution=1,
        )
        plotter.add_mesh(eq_plane, opacity=0.0, pickable=True, name="eq_pick")

        plotter.enable_surface_point_picking(
            callback=tracer.on_pick,
            show_message=False,
            show_point=False,
            left_clicking=True,
            picker="cell",
        )

        plotter.add_key_event("t", tracer.trace)
        plotter.add_key_event("c", tracer.clear)

        print("Click equatorial plane to seed | T: trace | C: clear | Scroll: zoom")
        plotter.show()


if __name__ == "__main__":
    main()
