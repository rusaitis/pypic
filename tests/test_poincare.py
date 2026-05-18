"""Tests for the Poincaré section diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from pypic import (
    PoincareSection,
    PoincareSurface,
    poincare_section,
)
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.traces import plane_crossings
from pypic.units import Normalization


def _closed_circle_data(n: int = 32, extent: float = 1.5) -> FieldDataset:
    r"""$\mathbf{B} = (-y, x, 0)$ on $[-L, L]^3$ — pure closed orbits.

    Used for the canonical magnetic-island test case.
    """
    coords = np.linspace(-extent, extent, n)
    X, Y, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    return FieldDataset.from_arrays(
        {
            "B_1": -Y,
            "B_2": X,
            "B_3": np.zeros_like(X),
        },
        GridInfo(
            dimensions=(n, n, n),
            spacing=(2 * extent / (n - 1),) * 3,
            origin=(-extent, -extent, -extent),
        ),
        Normalization.identity(),
    )


def _vortex_data(n: int = 32, extent: float = 1.5, drift: float = 0.1) -> FieldDataset:
    r"""$\mathbf{B} = (-y, x, \text{drift})$ — helix with axial drift."""
    coords = np.linspace(-extent, extent, n)
    X, Y, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    return FieldDataset.from_arrays(
        {
            "B_1": -Y,
            "B_2": X,
            "B_3": drift * np.ones_like(X),
        },
        GridInfo(
            dimensions=(n, n, n),
            spacing=(2 * extent / (n - 1),) * 3,
            origin=(-extent, -extent, -extent),
        ),
        Normalization.identity(),
    )


def _uniform_data(n: int = 16, extent: float = 1.0) -> FieldDataset:
    r"""$\mathbf{B} = (1, 0, 0)$ — straight lines along x."""
    shape = (n, n, n)
    return FieldDataset.from_arrays(
        {
            "B_1": np.ones(shape),
            "B_2": np.zeros(shape),
            "B_3": np.zeros(shape),
        },
        GridInfo(
            dimensions=shape,
            spacing=(2 * extent / (n - 1),) * 3,
            origin=(-extent, -extent, -extent),
        ),
        Normalization.identity(),
    )


class TestPoincareSurface:
    """PoincareSurface dataclass: constructors, projection, basis."""

    @pytest.mark.parametrize(
        ("axis", "expected_normal"),
        [
            ("x", (1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
            ("z", (0.0, 0.0, 1.0)),
        ],
    )
    def test_from_axis_round_trip(
        self, axis: str, expected_normal: tuple[float, float, float]
    ) -> None:
        surf = PoincareSurface.from_axis(axis, 1.5)  # type: ignore[arg-type]
        assert surf.normal == expected_normal
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        # Offset = n · p; for axis-aligned, equals the value
        assert surf.offset == pytest.approx(1.5)
        # Basis vectors are orthonormal and orthogonal to normal
        u, v = surf.basis_2d
        assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-15)
        assert float(np.linalg.norm(u)) == pytest.approx(1.0)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0)
        assert float(np.dot(surf._normal_arr, u)) == pytest.approx(0.0, abs=1e-15)
        assert float(np.dot(surf._normal_arr, v)) == pytest.approx(0.0, abs=1e-15)
        # The basis spans the (n-1) other axes
        assert abs(u[idx]) < 1e-15
        assert abs(v[idx]) < 1e-15

    def test_arbitrary_normal_orthonormal_basis(self) -> None:
        """Gram-Schmidt against the least-parallel world axis."""
        surf = PoincareSurface(normal=(1.0, 1.0, 1.0), point=(0.0, 0.0, 0.0))
        u, v = surf.basis_2d
        n = surf._normal_arr
        assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-15)
        assert float(np.linalg.norm(u)) == pytest.approx(1.0)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0)
        assert float(np.dot(n, u)) == pytest.approx(0.0, abs=1e-15)
        assert float(np.dot(n, v)) == pytest.approx(0.0, abs=1e-15)

    def test_project_round_trip(self) -> None:
        """Projecting a point in Σ recovers its in-plane offset from p."""
        surf = PoincareSurface.from_axis("z", 0.5)
        pts = np.array([[1.0, 2.0, 0.5], [-3.0, 4.0, 0.5]])
        uv = surf.project(pts)
        # For z-axis basis: u = (0, 1, 0), v = (-1, 0, 0) (per current
        # Gram-Schmidt convention picking x as the perpendicular seed)
        u, v = surf.basis_2d
        # u-coordinate = u · (p - point)
        assert uv[0, 0] == pytest.approx(float(np.dot(u, pts[0] - surf._point_arr)))
        assert uv[0, 1] == pytest.approx(float(np.dot(v, pts[0] - surf._point_arr)))
        # Distance in plane preserved
        in_plane_distance = float(np.linalg.norm(pts[0, :2] - pts[1, :2]))
        out_distance = float(np.linalg.norm(uv[0] - uv[1]))
        assert out_distance == pytest.approx(in_plane_distance)

    def test_project_empty_input(self) -> None:
        surf = PoincareSurface.from_axis("y", 0.0)
        out = surf.project(np.empty((0, 3)))
        assert out.shape == (0, 2)

    def test_zero_normal_raises(self) -> None:
        surf = PoincareSurface(normal=(0.0, 0.0, 0.0), point=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="non-zero"):
            _ = surf.offset

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="axis must be one of"):
            PoincareSurface.from_axis("q", 0.0)  # type: ignore[arg-type]


class TestPoincareSection:
    """End-to-end: trace + puncture + project."""

    def test_closed_circle_island_topology(self) -> None:
        r"""$\mathbf{B} = (-y, x, 0)$: punctures form a 2-cluster ring at $\pm r$.

        Tests topology, not numerical precision — grid interpolation of
        the analytical field introduces ~5% radial drift per orbit that
        no amount of tolerance tightening can remove without a finer
        grid (the tracer integrates the discretized field, not the
        analytical one). Asserts:

        - z is preserved exactly (B_3 = 0 ⇒ trace stays in z = 0)
        - punctures split into two clusters at ``v = ±r`` (the two
          intersections of the closed circle with the y = 0 plane)
        - each seed produces a clearly distinguishable ring radius
        """
        data = _closed_circle_data(n=128, extent=1.5)
        surf = PoincareSurface.from_axis("y", 0.0)
        radii = [0.3, 0.5, 0.8]
        seeds = np.array([[r, 0.0, 0.0] for r in radii])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=2000,
            direction="forward",
            atol=1e-10,
            rtol=1e-10,
            step_size_init=0.005,
            max_step=0.02,
        )

        # The y=0 plane's basis_2d gives u = -z, v = -x. Since B_3=0,
        # the trace stays in z = 0 and u should be exactly 0.
        for k, r in enumerate(radii):
            pts = section.punctures_2d[k]
            assert pts.shape[0] >= 4, (
                f"seed {k} (r={r}) only got {pts.shape[0]} punctures"
            )
            # z exactly preserved
            np.testing.assert_allclose(pts[:, 0], 0.0, atol=1e-12)
            # |v| is approximately r — punctures fall into the two clusters
            # at v = +r and v = -r. Allow 10% radial drift for grid-resolution
            # accumulated error.
            np.testing.assert_allclose(np.abs(pts[:, 1]), r, rtol=0.1)
            # Clusters separate: more than one sign of v
            has_positive = bool(np.any(pts[:, 1] > 0))
            has_negative = bool(np.any(pts[:, 1] < 0))
            assert has_positive, f"seed {k} (r={r}) lacks positive-v puncture"
            assert has_negative, f"seed {k} (r={r}) lacks negative-v puncture"

        # Different seeds produce clearly separated rings
        r0_max = np.abs(section.punctures_2d[0][:, 1]).max()
        r2_min = np.abs(section.punctures_2d[2][:, 1]).min()
        assert r0_max < r2_min, "seed-0 and seed-2 rings overlap"

        # Metadata bookkeeping: n_steps_per_seed mirrors the per-seed
        # trace length. Pins the schema documented on PoincareSection.
        for k in range(len(radii)):
            expected = section.field_lines[k].n_points - 1
            assert int(section.metadata["n_steps_per_seed"][k]) == expected, (
                f"seed {k}: metadata n_steps={section.metadata['n_steps_per_seed'][k]} "
                f"!= n_points-1={expected}"
            )

    def test_vortex_axial_drift(self) -> None:
        r"""Helical $\mathbf{B} = (-y, x, 0.1)$: punctures cluster around $|v| = r$.

        Same caveat as ``test_closed_circle_island_topology`` — grid
        interpolation introduces ~10% radial drift per orbit. We
        assert the topology: a monotone axial drift in the u-coord
        and clustered |v|.
        """
        data = _vortex_data(n=128, extent=2.0, drift=0.1)
        surf = PoincareSurface.from_axis("y", 0.0)
        seeds = np.array([[0.5, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=5000,
            direction="forward",
            atol=1e-10,
            rtol=1e-10,
            step_size_init=0.005,
            max_step=0.02,
        )
        pts = section.punctures_2d[0]
        assert pts.shape[0] >= 2
        # Plane coords for y=0 surface are u = -z, v = -x. Each puncture
        # is at x = ±r, so |v| ≈ r within 10%.
        np.testing.assert_allclose(np.abs(pts[:, 1]), 0.5, rtol=0.1)
        # The u-coordinate (= -z) drifts monotonically with axial flow.
        # Since z increases with arclength, u decreases monotonically.
        u_coords = pts[:, 0]
        assert np.all(np.diff(u_coords) < 0), (
            f"u (= -z) should decrease monotonically, got {u_coords}"
        )

    def test_tilted_plane_closed_circle_punctures(self) -> None:
        r"""Tilted plane $x + y = 0$ on closed circles: punctures at $(\pm r, 0)$.

        End-to-end exercise of a non-axis-aligned surface — the basis-only
        test pins Gram--Schmidt, but the full ``poincare_section`` path
        had no analytic prediction on a tilted plane.

        Geometry: $\mathbf{B} = (-y, x, 0)$ gives closed circles in
        constant-$z$ planes. The surface ``normal=(1,1,0)/√2``,
        ``point=origin`` has equation $x + y = 0$. A circle of radius
        $r$ intersects at $t = 3\pi/4, 7\pi/4$, giving 3D punctures
        $(\mp r/\sqrt{2}, \pm r/\sqrt{2}, 0)$.

        The Gram--Schmidt basis from the implementation is $\hat u =
        (1, -1, 0)/\sqrt{2}$ (least-parallel seed = $\hat z$) and
        $\hat v = (0, 0, -1)$. So $u = (x - y)/\sqrt{2}$ and $v = -z$;
        the two predicted punctures project to exactly $(\pm r, 0)$.
        """
        data = _closed_circle_data(n=128, extent=1.5)
        surf = PoincareSurface(normal=(1.0, 1.0, 0.0), point=(0.0, 0.0, 0.0))
        r = 0.5
        seeds = np.array([[r, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=2000,
            direction="forward",
            atol=1e-10,
            rtol=1e-10,
            step_size_init=0.005,
            max_step=0.02,
        )
        pts = section.punctures_2d[0]
        assert pts.shape[0] >= 4, (
            f"tilted plane got only {pts.shape[0]} punctures; need both signs"
        )
        # v ≈ 0 exactly (B_3 = 0 preserves z, and v = -z by construction)
        np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-10)
        # |u| ≈ r within ~10% grid-interp drift (same caveat as the
        # axis-aligned closed-circle test).
        np.testing.assert_allclose(np.abs(pts[:, 0]), r, rtol=0.1)
        # Punctures split into ±u clusters
        assert np.any(pts[:, 0] > 0), "missing positive-u puncture"
        assert np.any(pts[:, 0] < 0), "missing negative-u puncture"

    def test_uniform_field_single_puncture(self) -> None:
        """B = x̂: a trace from x<0 crosses x=x₀ exactly once."""
        data = _uniform_data(n=16, extent=1.0)
        surf = PoincareSurface.from_axis("x", 0.0)
        seeds = np.array([[-0.5, 0.2, 0.3]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=200,
            direction="forward",
            step_size_init=0.05,
            max_step=0.1,
        )
        pts3 = section.punctures_3d[0]
        # Exactly one puncture along the forward trace
        assert pts3.shape[0] == 1
        # The puncture lies on x = 0 (within FP tolerance)
        assert pts3[0, 0] == pytest.approx(0.0, abs=1e-6)
        # y, z unchanged
        assert pts3[0, 1] == pytest.approx(0.2, abs=1e-6)
        assert pts3[0, 2] == pytest.approx(0.3, abs=1e-6)

    def test_loop_tol_is_forced_off(self) -> None:
        """Closed circle: even with the auto detector on by default in
        trace_field_lines_adaptive, poincare_section forces loop_tol=None
        so closed orbits aren't terminated mid-sweep."""
        data = _closed_circle_data(n=64, extent=1.5)
        surf = PoincareSurface.from_axis("y", 0.0)
        seeds = np.array([[0.6, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=500,
            direction="forward",
            atol=1e-8,
            rtol=1e-8,
            step_size_init=0.05,
            max_step=0.1,
        )
        # If loop_tol had been enabled, termination would be CLOSED_LOOP
        # after ~1 orbit; we ran to max_steps instead.
        assert section.metadata["termination_reasons"][0] == "max_steps"
        # And we got many punctures — many orbits of the closed circle
        assert section.metadata["n_crossings_per_seed"][0] > 20

    def test_all_punctures_concatenation(self) -> None:
        data = _closed_circle_data(n=32, extent=1.5)
        surf = PoincareSurface.from_axis("y", 0.0)
        seeds = np.array([[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=500,
            direction="forward",
        )
        total_2d = section.all_punctures_2d
        total_3d = section.all_punctures_3d
        expected_2d = np.concatenate(section.punctures_2d, axis=0)
        expected_3d = np.concatenate(section.punctures_3d, axis=0)
        np.testing.assert_array_equal(total_2d, expected_2d)
        np.testing.assert_array_equal(total_3d, expected_3d)
        # Shape consistency
        assert total_2d.shape[0] == total_3d.shape[0]
        assert total_2d.shape[1] == 2
        assert total_3d.shape[1] == 3

    def test_repuncture_without_retracing(self) -> None:
        """Reusing section.field_lines on a different surface matches a fresh call."""
        data = _vortex_data(n=32, extent=1.5, drift=0.1)
        surf_a = PoincareSurface.from_axis("y", 0.0)
        seeds = np.array([[0.5, 0.0, 0.0]])
        section_a = poincare_section(
            data,
            seeds,
            surf_a,
            max_steps=2000,
            direction="forward",
            atol=1e-8,
            rtol=1e-8,
        )

        # Re-puncture the same trace on a different surface
        surf_b = PoincareSurface.from_axis("y", 0.1)
        offset_b = surf_b.offset
        repuncture_3d = plane_crossings(
            section_a.field_lines[0].points, surf_b.normal, offset_b
        )

        # Compare against a fresh poincare_section call with the same
        # tracer params on the new surface
        section_b = poincare_section(
            data,
            seeds,
            surf_b,
            max_steps=2000,
            direction="forward",
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(repuncture_3d, section_b.punctures_3d[0], atol=1e-10)

    def test_empty_puncture_seed(self) -> None:
        """Seed whose trace never crosses Σ produces an empty (0, 3) array."""
        # B = x̂, surface at x = 0; seed at x = +0.5 only moves forward
        # (i.e., away from the plane), so direction="forward" gives no crossings.
        data = _uniform_data(n=16, extent=1.0)
        surf = PoincareSurface.from_axis("x", 0.0)
        seeds = np.array([[0.5, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=200,
            direction="forward",
        )
        assert section.punctures_3d[0].shape == (0, 3)
        assert section.punctures_2d[0].shape == (0, 2)
        # all_punctures_* still work
        assert section.all_punctures_3d.shape == (0, 3)
        assert section.all_punctures_2d.shape == (0, 2)

    def test_section_is_immutable(self) -> None:
        """Frozen dataclass: metadata exposed as MappingProxyType."""
        data = _uniform_data(n=16, extent=1.0)
        surf = PoincareSurface.from_axis("x", 0.0)
        seeds = np.array([[-0.5, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=200,
            direction="forward",
        )
        with pytest.raises(TypeError):
            section.metadata["surface_name"] = "modified"  # type: ignore[index]

    def test_section_surface_name_propagates(self) -> None:
        data = _uniform_data(n=16, extent=1.0)
        surf = PoincareSurface.from_axis("x", 0.0, name="midplane")
        seeds = np.array([[-0.5, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=200,
            direction="forward",
        )
        assert section.metadata["surface_name"] == "midplane"
        assert section.surface.name == "midplane"

    def test_returns_poincare_section_instance(self) -> None:
        data = _uniform_data(n=16, extent=1.0)
        surf = PoincareSurface.from_axis("x", 0.0)
        seeds = np.array([[-0.5, 0.0, 0.0]])
        section = poincare_section(
            data,
            seeds,
            surf,
            max_steps=200,
            direction="forward",
        )
        assert isinstance(section, PoincareSection)
        assert section.n_seeds == 1
        assert section.direction == "forward"
