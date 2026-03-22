"""3D dipole Earth field line visualization with interactive seed dragging.

Generates an analytical magnetic dipole on a 3D grid, traces field lines
using the RK4 tracer, and renders an interactive 3D matplotlib figure
with a wireframe planet and colored field lines.

Controls:
- **Drag** the orange marker in the equatorial plane (z=0)
- **T** — trace a field line from the marker
- **S** — toggle a Bz color-map slice
- **+/-** — move the slice up/down through the volume

Run with::

    uv run python tests/visual_dipole_3d.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_tools import Cursors
from mpl_toolkits.mplot3d import proj3d

from pypic.plotting import DARK, use_theme
from pypic.readers.base import FieldDataset, GridInfo
from pypic.selections import PlaneSelection
from pypic.traces import VectorFieldInterpolator, trace_field_line
from pypic.units import Normalization

PLANET_RADIUS = 1.0
DOMAIN_HALF = 6.0
N_CELLS = 80


def _dipole_field(
    grid: GridInfo,
    moment: float = 1.0,
    planet_radius: float = PLANET_RADIUS,
) -> dict[str, np.ndarray]:
    """Compute analytical dipole B field on the grid, zeroed inside the planet.

    Dipole aligned with z-axis, moment M:
        Bx = 3 M x z / r^5
        By = 3 M y z / r^5
        Bz = M (3 z^2 - r^2) / r^5
    """
    coords = grid.coordinate_arrays()
    x, y, z = np.meshgrid(*coords, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)

    # Avoid division by zero at origin
    r_safe = np.where(r > 0, r, 1.0)
    r5 = r_safe**5

    bx = 3.0 * moment * x * z / r5
    by = 3.0 * moment * y * z / r5
    bz = moment * (3.0 * z**2 - r_safe**2) / r5

    # Zero out inside the planet — tracer will terminate via null detection
    inside = r < planet_radius
    bx[inside] = 0.0
    by[inside] = 0.0
    bz[inside] = 0.0

    return {"B1": bx, "B2": by, "B3": bz}


def _planet_mesh(
    radius: float = PLANET_RADIUS,
    n: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV sphere mesh for plot_surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _seed_points(
    l_shells: list[float],
    n_per_shell: int = 3,
) -> list[tuple[float, float, float]]:
    """Generate seed points at given L-shells in the noon-midnight plane (y=0).

    Seeds are placed at the magnetic equator (z=0) and at symmetric
    latitudes above/below. L-shell gives the equatorial crossing distance
    in R_E, so equatorial seeds sit at x = L, y = 0, z = 0.
    Off-equator seeds use the dipole field line equation r = L cos^2(lambda).
    """
    seeds: list[tuple[float, float, float]] = []
    if n_per_shell == 1:
        latitudes = [0.0]
    else:
        latitudes = np.linspace(-25.0, 25.0, n_per_shell).tolist()

    for l_val in l_shells:
        for lat_deg in latitudes:
            lat = np.radians(lat_deg)
            r = l_val * np.cos(lat) ** 2
            x = r * np.cos(lat)
            z = r * np.sin(lat)
            if r > PLANET_RADIUS + 0.3:
                seeds.append((x, 0.0, z))
    return seeds


def _style_3d_axes(ax: plt.Axes) -> None:  # type: ignore[type-arg]
    """Apply dark styling to 3D axes (Axes3D ignores most rcParams)."""
    bg = "#1e1e1e"
    text_color = "#e0e0e0"
    dim_color = "#444444"

    ax.set_facecolor(bg)
    ax.xaxis.pane.fill = False  # type: ignore[attr-defined]
    ax.yaxis.pane.fill = False  # type: ignore[attr-defined]
    ax.zaxis.pane.fill = False  # type: ignore[attr-defined]
    ax.xaxis.pane.set_edgecolor(dim_color)  # type: ignore[attr-defined]
    ax.yaxis.pane.set_edgecolor(dim_color)  # type: ignore[attr-defined]
    ax.zaxis.pane.set_edgecolor(dim_color)  # type: ignore[attr-defined]

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(text_color)  # type: ignore[attr-defined]
        axis.set_tick_params(colors=dim_color, labelsize=8)  # type: ignore[attr-defined]
        axis._axinfo["grid"]["color"] = dim_color  # type: ignore[attr-defined]
        axis._axinfo["grid"]["linewidth"] = 0.3  # type: ignore[attr-defined]

    ax.set_xlabel("$x$ [$R_E$]")
    ax.set_ylabel("$y$ [$R_E$]")
    ax.set_zlabel("$z$ [$R_E$]")  # type: ignore[attr-defined]


def _slice_bz(
    ds: FieldDataset, z_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract Bz on a z-plane, return (X, Y, values, z_position)."""
    plane = PlaneSelection(normal="z", index=z_index)
    sliced = plane.apply(ds)
    values = sliced["B3"]
    coords = sliced.grid.coordinate_arrays()
    x_1d, y_1d = coords[0], coords[1]
    x_2d, y_2d = np.meshgrid(x_1d, y_1d, indexing="ij")
    z_pos = ds.grid.coordinate_arrays()[2][z_index]
    return x_2d, y_2d, values, float(z_pos)


class SeedDragger:
    """Draggable seed marker on z=0, press T to trace a field line."""

    def __init__(
        self,
        ax: plt.Axes,  # type: ignore[type-arg]
        fig: plt.Figure,
        ds: FieldDataset,
        interp: VectorFieldInterpolator,
    ) -> None:
        self.ax = ax
        self.fig = fig
        self.ds = ds
        self.interp = interp
        self.pos = np.array([3.0, 0.0, 0.0])
        self.dragging = False
        self.pick_radius = 15  # pixels
        self.traced_lines: list[object] = []
        self._hovering = False

        # Equatorial slice state
        self._slice_visible = False
        self._slice_z_index = N_CELLS // 2
        self._slice_contour: object | None = None
        self._cbar: plt.colorbar.Colorbar | None = None  # type: ignore[name-defined]

        (self.marker,) = ax.plot(
            [self.pos[0]],
            [self.pos[1]],
            [self.pos[2]],
            "o",
            markersize=12,
            color="#ff6600",
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=10,
        )
        self.status = fig.text(
            0.5,
            0.02,
            self._status_text(),
            ha="center",
            color="#e0e0e0",
            fontsize=9,
            family="monospace",
        )
        # Unbind 's' from matplotlib's default save-figure so it reaches _on_key
        plt.rcParams["keymap.save"] = []

        connect = fig.canvas.mpl_connect
        self._cids = [
            connect("button_press_event", self._on_press),
            connect("motion_notify_event", self._on_motion),
            connect("button_release_event", self._on_release),
            connect("key_press_event", self._on_key),
        ]

    def _marker_pixel_pos(self) -> np.ndarray:
        """Forward-project marker 3D position to pixel coordinates."""
        xv, yv, _ = proj3d.proj_transform(*self.pos, self.ax.get_proj())
        return np.asarray(self.ax.transData.transform((xv, yv)))

    def _screen_to_equatorial(self, xv: float, yv: float) -> np.ndarray | None:
        """Ray-plane intersection: view coords to z=0 world coords."""
        focal = self.ax._focal_length  # type: ignore[attr-defined]
        zv = 1.0 if focal == np.inf else -1.0 / focal
        p1 = np.array(
            proj3d.inv_transform(xv, yv, zv, self.ax.invM)  # type: ignore[attr-defined]
        ).ravel()
        cam = np.asarray(self.ax._get_camera_loc())  # type: ignore[attr-defined]
        direction = cam - p1
        if direction[2] == 0:
            return None
        scale = p1[2] / direction[2]
        return p1 - scale * direction

    def _on_press(self, event: object) -> None:
        """Start drag if left-click near marker."""
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        mpx = self._marker_pixel_pos()
        ex, ey = getattr(event, "x", 0), getattr(event, "y", 0)
        if np.hypot(ex - mpx[0], ey - mpx[1]) < self.pick_radius:
            self.dragging = True
            self.ax.disable_mouse_rotation()  # type: ignore[attr-defined]

    def _is_near_marker(self, event: object) -> bool:
        """Check if the mouse event is within pick_radius of the marker."""
        ex, ey = getattr(event, "x", None), getattr(event, "y", None)
        if ex is None or ey is None:
            return False
        mpx = self._marker_pixel_pos()
        return float(np.hypot(ex - mpx[0], ey - mpx[1])) < self.pick_radius

    def _set_hover(self, hovering: bool) -> None:
        """Toggle hover highlight: hand cursor + enlarged marker."""
        if hovering == self._hovering:
            return
        self._hovering = hovering
        canvas = self.fig.canvas
        if hovering:
            canvas.set_cursor(Cursors.HAND)
            self.marker.set_markersize(16)
            self.marker.set_markeredgewidth(2.5)
        else:
            canvas.set_cursor(Cursors.POINTER)
            self.marker.set_markersize(12)
            self.marker.set_markeredgewidth(1.5)
        canvas.draw_idle()

    def _on_motion(self, event: object) -> None:
        """Update marker on z=0 plane while dragging, hover highlight otherwise."""
        if self.dragging:
            xdata = getattr(event, "xdata", None)
            ydata = getattr(event, "ydata", None)
            if xdata is None or ydata is None:
                return
            p = self._screen_to_equatorial(xdata, ydata)
            if p is None:
                return
            x = float(np.clip(p[0], -DOMAIN_HALF + 0.5, DOMAIN_HALF - 0.5))
            y = float(np.clip(p[1], -DOMAIN_HALF + 0.5, DOMAIN_HALF - 0.5))
            if np.hypot(x, y) < PLANET_RADIUS + 0.2:
                return  # reject positions inside the planet
            self.pos[:] = [x, y, 0.0]
            self.marker.set_data_3d([x], [y], [0.0])  # type: ignore[attr-defined]
            self.status.set_text(self._status_text())
            self.fig.canvas.draw_idle()
        else:
            self._set_hover(self._is_near_marker(event))

    def _on_release(self, event: object) -> None:
        if self.dragging:
            self.dragging = False
            self.ax.mouse_init()  # type: ignore[attr-defined]

    def _on_key(self, event: object) -> None:
        """Key handler: T=trace, S=toggle slice, +/-=move slice."""
        key = getattr(event, "key", None)
        if key in ("t", "T"):
            self._trace_from_marker()
        elif key in ("s", "S"):
            self._toggle_slice()
        elif key in ("+", "="):
            self._move_slice(1)
        elif key == "-":
            self._move_slice(-1)

    def _trace_from_marker(self) -> None:
        """Trace a field line from the current marker position."""
        seed = (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        try:
            fl = trace_field_line(
                self.ds,
                seed,
                step_size=0.1,
                max_steps=5000,
                direction="both",
                null_threshold=1e-6,
                interpolator=self.interp,
            )
        except ValueError as exc:
            print(f"  Cannot trace from {seed}: {exc}")
            return
        pts = fl.points
        color = plt.get_cmap("cool")(len(self.traced_lines) % 8 / 7)
        self.ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )
        self.traced_lines.append(fl)
        r_eq = np.hypot(self.pos[0], self.pos[1])
        print(
            f"  Traced line #{len(self.traced_lines)} from "
            f"({self.pos[0]:.2f}, {self.pos[1]:.2f}), "
            f"L\u2248{r_eq:.1f}, {fl.n_points} points"
        )
        self.fig.canvas.draw_idle()

    def _toggle_slice(self) -> None:
        """Toggle Bz color-map slice visibility."""
        self._slice_visible = not self._slice_visible
        if self._slice_visible:
            self._draw_slice()
        else:
            # Remove colorbar first (while its mappable still exists)
            if self._cbar is not None:
                self._cbar.remove()
                self._cbar = None
            self._remove_slice()
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _draw_slice(self) -> None:
        """Draw (or redraw) the Bz contourf at the current z-index."""
        # Remove old colorbar before its backing contour disappears
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self._remove_slice()
        x_2d, y_2d, values, z_pos = _slice_bz(self.ds, self._slice_z_index)
        cs = self.ax.contourf(
            x_2d, y_2d, values,
            levels=32, cmap="RdBu_r", alpha=0.7,
            zdir="z", offset=z_pos,
        )
        self._slice_contour = cs
        self._cbar = self.fig.colorbar(
            cs, ax=self.ax, shrink=0.5, pad=0.08, label="$B_z$",
        )

    def _remove_slice(self) -> None:
        """Remove existing slice contour from the axes."""
        if self._slice_contour is not None:
            self._slice_contour.remove()  # type: ignore[union-attr]
            self._slice_contour = None

    def _move_slice(self, delta: int) -> None:
        """Move the slice by delta grid cells in z, clamped to bounds."""
        new_idx = self._slice_z_index + delta
        nz = self.ds.grid.dimensions[2]
        if 0 <= new_idx < nz:
            self._slice_z_index = new_idx
            if self._slice_visible:
                self._draw_slice()
                self.status.set_text(self._status_text())
                self.fig.canvas.draw_idle()

    def _status_text(self) -> str:
        r = np.hypot(self.pos[0], self.pos[1])
        parts = [
            f"Seed: ({self.pos[0]:+.2f}, {self.pos[1]:+.2f}, 0.00) L\u2248{r:.1f}",
        ]
        if self._slice_visible:
            z_pos = self.ds.grid.coordinate_arrays()[2][self._slice_z_index]
            parts.append(f"z={z_pos:+.2f} Bz slice")
        parts.append("T: trace  S: slice  +/-: move")
        return "  |  ".join(parts)


def main() -> None:
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

    # Pre-build interpolator for all traces
    interp = VectorFieldInterpolator.from_dataset(ds)

    # Seed points at various L-shells
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
            pass  # seed at null or outside domain

    print(f"  {len(lines)} lines traced successfully")

    # Color palette for L-shells
    cmap = plt.get_cmap("plasma")
    l_colors = {
        l_val: cmap(i / (len(l_shells) - 1)) for i, l_val in enumerate(l_shells)
    }

    with use_theme(DARK):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        _style_3d_axes(ax)

        # Draw the planet
        px, py, pz = _planet_mesh(PLANET_RADIUS, n=25)
        ax.plot_surface(  # type: ignore[attr-defined]
            px,
            py,
            pz,
            color="#3a6ea5",
            alpha=0.3,
            edgecolor="#5a8ec5",
            linewidth=0.2,
        )

        # Draw field lines, colored by L-shell
        seed_idx = 0
        n_per_shell = 3
        for l_val in l_shells:
            color = l_colors[l_val]
            for _ in range(n_per_shell):
                if seed_idx < len(lines):
                    pts = lines[seed_idx].points
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color=color,
                        linewidth=1.2,
                        alpha=0.9,
                    )
                    seed_idx += 1

        # Axis limits and viewing angle
        lim = 5.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)  # type: ignore[attr-defined]
        ax.view_init(elev=15, azim=-60)  # type: ignore[attr-defined]
        ax.set_aspect("equal")

        # Legend via dummy lines
        for l_val in l_shells:
            ax.plot(
                [],
                [],
                [],
                color=l_colors[l_val],
                linewidth=2,
                label=f"L = {l_val}",
            )
        ax.legend(
            loc="upper left",
            fontsize=9,
            facecolor=(0.1, 0.1, 0.1, 0.7),
            edgecolor="none",
            labelcolor="#e0e0e0",
        )

        ax.set_title(
            "Magnetic Dipole Field Lines",
            color="#e0e0e0",
            fontsize=14,
            pad=10,
        )
        fig.set_facecolor("#1e1e1e")

        plt.tight_layout()

        dragger = SeedDragger(ax, fig, ds, interp)  # noqa: F841
        print("Drag marker + T to trace | S to toggle Bz slice | +/- to move slice")
        plt.show()


if __name__ == "__main__":
    main()
