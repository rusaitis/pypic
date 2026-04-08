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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Allow running as `python tests/visual_dipole_3d.py` from any cwd by adding
# the project root to sys.path so `tests._helpers` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from pypic.traces import (
    VectorFieldInterpolator,
    attach_scalars,
    trace_field_line,
)
from tests._helpers import make_dipole_dataset

if TYPE_CHECKING:
    from pypic.readers.base import FieldDataset

PLANET_RADIUS = 1.0
DOMAIN_HALF = 6.0
N_CELLS = 80


def _seed_points(
    l_shells: list[float],
    n_per_shell: int = 3,
) -> list[tuple[float, float, float]]:
    """Generate seed points at given L-shells in the noon-midnight plane."""
    seeds: list[tuple[float, float, float]] = []
    latitudes = (
        [0.0] if n_per_shell == 1 else np.linspace(-25.0, 25.0, n_per_shell).tolist()
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
    """Drag the sphere widget to position a seed, press T to trace."""

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
        self.seed: tuple[float, float, float] = (4.0, 0.0, 0.0)
        self._trace_count = 0
        self._free_3d = False
        self._widget: object | None = None
        self._crosshair_len = DOMAIN_HALF * 0.4

    def _update_crosshairs(self, x: float, y: float, z: float) -> None:
        """Draw subtle dashed crosshair lines at the seed position."""
        import pyvista as pv

        L = self._crosshair_len
        color = "#ff6600"
        opacity = 0.3
        width = 1.0

        # X-line and Y-line on the equatorial plane (z=0)
        x_line = pv.Line((x - L, y, 0), (x + L, y, 0))
        self.plotter.add_mesh(  # type: ignore[union-attr]
            x_line,
            color=color,
            opacity=opacity,
            line_width=width,
            name="_cross_x",
            render_lines_as_tubes=False,
        )
        y_line = pv.Line((x, y - L, 0), (x, y + L, 0))
        self.plotter.add_mesh(  # type: ignore[union-attr]
            y_line,
            color=color,
            opacity=opacity,
            line_width=width,
            name="_cross_y",
            render_lines_as_tubes=False,
        )
        # Vertical z-line from plane to sphere (visible when off-plane)
        if abs(z) > 0.01:
            z_line = pv.Line((x, y, 0), (x, y, z))
            self.plotter.add_mesh(  # type: ignore[union-attr]
                z_line,
                color=color,
                opacity=opacity * 1.5,
                line_width=width,
                name="_cross_z",
                render_lines_as_tubes=False,
            )
            # Small dot on the equatorial plane beneath the sphere
            dot = pv.Sphere(radius=0.06, center=(x, y, 0))
            self.plotter.add_mesh(  # type: ignore[union-attr]
                dot,
                color=color,
                opacity=opacity,
                name="_cross_dot",
            )
        else:
            # Remove z-line and dot when on the plane
            self.plotter.remove_actor("_cross_z", render=False)  # type: ignore[union-attr]
            self.plotter.remove_actor("_cross_dot", render=False)  # type: ignore[union-attr]

    def on_move(self, center: np.ndarray) -> None:
        """Called when the sphere widget is released. Snaps to z=0 unless 3D."""
        x, y, z = float(center[0]), float(center[1]), float(center[2])

        if not self._free_3d:
            z = 0.0
            if self._widget is not None:
                self._widget.SetCenter(x, y, 0.0)  # type: ignore[union-attr]

        self.seed = (x, y, z)
        self._update_crosshairs(x, y, z)
        r_eq = np.hypot(x, y)
        print(
            f"  Seed: ({x:+.2f}, {y:+.2f}, {z:+.2f})  L≈{r_eq:.1f}",
            end="\r",
        )

    def _nudge(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Move seed by a small offset and update the widget."""
        x, y, z = self.seed
        x += dx
        y += dy
        z += dz if self._free_3d else 0.0
        self.seed = (x, y, z)
        if self._widget is not None:
            self._widget.SetCenter(x, y, z)  # type: ignore[union-attr]
        self._update_crosshairs(x, y, z)
        r_eq = np.hypot(x, y)
        print(f"  Seed: ({x:+.2f}, {y:+.2f}, {z:+.2f})  L≈{r_eq:.1f}", end="\r")
        self.plotter.render()  # type: ignore[union-attr]

    _STEP = 0.5  # arrow key step size in R_E

    def nudge_left(self) -> None:
        self._nudge(dx=-self._STEP)

    def nudge_right(self) -> None:
        self._nudge(dx=self._STEP)

    def nudge_up(self) -> None:
        self._nudge(dy=self._STEP)

    def nudge_down(self) -> None:
        self._nudge(dy=-self._STEP)

    def nudge_z_up(self) -> None:
        self._nudge(dz=self._STEP)

    def nudge_z_down(self) -> None:
        self._nudge(dz=-self._STEP)

    def toggle_3d(self) -> None:
        """Toggle between equatorial-plane-locked and free 3D movement."""
        self._free_3d = not self._free_3d
        mode = "3D free" if self._free_3d else "equatorial plane"
        print(f"\n  Mode: {mode}")

    def trace(self) -> None:
        """Trace a field line from the current seed (T key)."""
        try:
            fl = trace_field_line(
                self.ds,
                self.seed,
                step_size=0.1,
                max_steps=5000,
                direction="both",
                null_threshold=1e-6,
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
            self.plotter,
            fl,  # type: ignore[arg-type]
            scalar="|B|",
            cmap=self.cmap,
            clim=(0, self.vmax),
            signed=False,
            radius=0.06,
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
        "--save",
        action="store_true",
        help="Save screenshot instead of interactive",
    )
    args = parser.parse_args()
    key = names[int(args.theme) - 1] if args.theme.isdigit() else args.theme
    set_theme(key)

    print("Computing dipole field...")
    ds = make_dipole_dataset(
        n_cells=N_CELLS, domain_half=DOMAIN_HALF, planet_radius=PLANET_RADIUS
    )

    interp = VectorFieldInterpolator.from_dataset(ds)

    # Pre-trace field lines at various L-shells
    l_shells = [2.0, 3.0, 4.0, 5.0]
    seeds = _seed_points(l_shells, n_per_shell=3)
    print(f"Tracing {len(seeds)} field lines...")

    lines = []
    for seed in seeds:
        try:
            fl = trace_field_line(
                ds,
                seed,
                step_size=0.1,
                max_steps=5000,
                direction="both",
                null_threshold=1e-6,
                interpolator=interp,
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
        plotter,
        colored_lines,
        scalar="|B|",
        cmap=cmap,
        clim=(0, vmax),
        signed=False,
        radius=0.05,
        show_scalar_bar=True,
        scalar_bar_position="lower_right",
    )

    lim = 5.5
    add_axis_triad(plotter, length=2.0, labels=("$x$", "$y$", "$z$"))
    add_equatorial_grid(plotter, xlim=(-lim, lim), ylim=(-lim, lim))
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
        # Interactive mode: drag sphere to position, T to trace
        tracer = _InteractiveTracer(plotter, ds, interp, cmap, vmax)

        def _on_widget_move(center: np.ndarray, widget: object) -> None:
            tracer._widget = widget
            tracer.on_move(center)

        plotter.add_sphere_widget(
            callback=_on_widget_move,
            center=(4.0, 0.0, 0.0),
            radius=0.25,
            color="#ff6600",
            style="surface",
            selected_color="#ffcc66",
            theta_resolution=20,
            phi_resolution=20,
            interaction_event="end",
            pass_widget=True,
        )

        plotter.add_key_event("t", tracer.trace)
        plotter.add_key_event("c", tracer.clear)
        plotter.add_key_event("z", tracer.toggle_3d)
        plotter.add_key_event("Left", tracer.nudge_left)
        plotter.add_key_event("Right", tracer.nudge_right)
        plotter.add_key_event("Up", tracer.nudge_up)
        plotter.add_key_event("Down", tracer.nudge_down)
        plotter.add_key_event("Prior", tracer.nudge_z_up)  # Page Up
        plotter.add_key_event("Next", tracer.nudge_z_down)  # Page Down

        print(
            "Arrows: move seed | PgUp/PgDn: move z (3D mode) | "
            "T: trace | Z: toggle 3D | C: clear"
        )
        plotter.show()


if __name__ == "__main__":
    main()
