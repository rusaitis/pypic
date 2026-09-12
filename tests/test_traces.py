"""Tests for the pypic.traces module: containers, analysis, and sampling."""

from __future__ import annotations

import copy
from types import MappingProxyType

import numpy as np
import pytest

from pypic.dataset import FieldDataset
from pypic.traces import (
    FieldLine,
    ParticleTrace,
    arc_length_cumulative,
    arc_length_total,
    attach_scalars,
    attach_scalars_to_trace,
    closest_approach,
    curvature,
    displacement,
    drift_velocity,
    equatorial_crossings,
    gyroradius_estimate,
    kinetic_energy,
    mirror_points,
    plane_crossings,
    resample_by_arc_length,
    sample_field,
    sample_fields,
    speed,
    tangent_vectors,
)
from pypic.units import Normalization


def _identity() -> Normalization:
    return Normalization.identity()


def _straight_line(n: int = 10) -> np.ndarray:
    """Straight line along x-axis from 0 to n-1."""
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])


def _circle_points(n: int = 100, radius: float = 1.0) -> np.ndarray:
    """Points on a circle in the xy-plane."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, np.zeros(n)])


class TestFieldLine:
    def test_construction(self) -> None:
        pts = _straight_line(5)
        fl = FieldLine(
            points=pts,
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
        )
        assert fl.n_points == 5
        assert fl.field_name == "B"

    def test_wrong_shape_rejects(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            FieldLine(
                points=np.zeros((5,)),
                field_name="B",
                seed_point=(0.0, 0.0, 0.0),
                normalization=_identity(),
            )

    def test_too_few_points_rejects(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            FieldLine(
                points=np.zeros((1, 3)),
                field_name="B",
                seed_point=(0.0, 0.0, 0.0),
                normalization=_identity(),
            )

    def test_invalid_direction_rejects(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            FieldLine(
                points=_straight_line(3),
                field_name="B",
                seed_point=(0.0, 0.0, 0.0),
                normalization=_identity(),
                direction="up",
            )

    def test_scalar_shape_mismatch_rejects(self) -> None:
        with pytest.raises(ValueError, match="scalar"):
            FieldLine(
                points=_straight_line(5),
                field_name="B",
                seed_point=(0.0, 0.0, 0.0),
                normalization=_identity(),
                scalars={"|B|": np.ones(3)},
            )

    def test_start_end_points(self) -> None:
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fl = FieldLine(
            points=pts,
            field_name="B",
            seed_point=(1.0, 2.0, 3.0),
            normalization=_identity(),
        )
        assert fl.start_point == (1.0, 2.0, 3.0)
        assert fl.end_point == (4.0, 5.0, 6.0)

    def test_len(self) -> None:
        fl = FieldLine(
            points=_straight_line(7),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
        )
        assert len(fl) == 7

    def test_scalars_are_readonly(self) -> None:
        fl = FieldLine(
            points=_straight_line(3),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
            scalars={"|B|": np.ones(3)},
        )
        assert isinstance(fl.scalars, MappingProxyType)
        with pytest.raises(TypeError):
            fl.scalars["new"] = np.zeros(3)  # type: ignore[index]

    def test_metadata_is_readonly(self) -> None:
        fl = FieldLine(
            points=_straight_line(3),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
            metadata={"source": "test"},
        )
        assert isinstance(fl.metadata, MappingProxyType)

    def test_copy_replace(self) -> None:
        fl = FieldLine(
            points=_straight_line(3),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
        )
        fl2 = copy.replace(fl, field_name="E")
        assert fl2.field_name == "E"
        assert fl.field_name == "B"

    def test_with_scalars(self) -> None:
        fl = FieldLine(
            points=_straight_line(3),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
        )
        fl2 = fl.with_scalars(rho=np.ones(3))
        assert "rho" in fl2.scalars
        assert "rho" not in fl.scalars

    def test_repr(self) -> None:
        fl = FieldLine(
            points=_straight_line(3),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
            time=1.5,
        )
        r = repr(fl)
        assert "FieldLine" in r
        assert "'B'" in r
        assert "t=1.5" in r


class TestParticleTrace:
    def _make_trace(self, n: int = 5) -> ParticleTrace:
        pts = _straight_line(n)
        t = np.arange(n, dtype=float) * 0.1
        vel = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
        return ParticleTrace(
            points=pts,
            time=t,
            velocity=vel,
            species_name="electrons",
            normalization=_identity(),
        )

    def test_construction(self) -> None:
        tr = self._make_trace()
        assert tr.n_points == 5
        assert tr.species_name == "electrons"

    def test_wrong_time_shape_rejects(self) -> None:
        pts = _straight_line(5)
        with pytest.raises(ValueError, match="time must have shape"):
            ParticleTrace(
                points=pts,
                time=np.arange(3, dtype=float),
                velocity=np.ones((5, 3)),
                species_name="e",
                normalization=_identity(),
            )

    def test_non_monotonic_time_rejects(self) -> None:
        pts = _straight_line(3)
        with pytest.raises(ValueError, match="monotonically increasing"):
            ParticleTrace(
                points=pts,
                time=np.array([0.0, 0.5, 0.3]),
                velocity=np.ones((3, 3)),
                species_name="e",
                normalization=_identity(),
            )

    def test_wrong_velocity_shape_rejects(self) -> None:
        pts = _straight_line(5)
        with pytest.raises(ValueError, match="velocity must have shape"):
            ParticleTrace(
                points=pts,
                time=np.arange(5, dtype=float),
                velocity=np.ones((3, 3)),
                species_name="e",
                normalization=_identity(),
            )

    def test_duration(self) -> None:
        tr = self._make_trace(10)
        np.testing.assert_allclose(tr.duration, 0.9, rtol=1e-12)

    def test_start_end_time(self) -> None:
        tr = self._make_trace(5)
        assert tr.start_time == 0.0
        np.testing.assert_allclose(tr.end_time, 0.4, rtol=1e-12)

    def test_start_end_point(self) -> None:
        tr = self._make_trace(5)
        assert tr.start_point == (0.0, 0.0, 0.0)
        assert tr.end_point == (4.0, 0.0, 0.0)

    def test_len(self) -> None:
        assert len(self._make_trace(7)) == 7

    def test_scalars_readonly(self) -> None:
        tr = self._make_trace()
        assert isinstance(tr.scalars, MappingProxyType)

    def test_copy_replace(self) -> None:
        tr = self._make_trace()
        tr2 = copy.replace(tr, species_name="ions")
        assert tr2.species_name == "ions"

    def test_with_scalars(self) -> None:
        tr = self._make_trace(3)
        tr2 = tr.with_scalars(energy=np.ones(3))
        assert "energy" in tr2.scalars
        assert "energy" not in tr.scalars

    def test_repr(self) -> None:
        tr = self._make_trace()
        r = repr(tr)
        assert "ParticleTrace" in r
        assert "'electrons'" in r


class TestArcLength:
    def test_straight_line(self) -> None:
        pts = _straight_line(5)
        s = arc_length_cumulative(pts)
        np.testing.assert_array_equal(s, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_total_equals_last_cumulative(self) -> None:
        pts = _straight_line(10)
        assert arc_length_total(pts) == arc_length_cumulative(pts)[-1]

    def test_diagonal(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        np.testing.assert_array_equal(arc_length_total(pts), 5.0)

    def test_circle_arc_length(self) -> None:
        n = 1000
        r = 2.0
        # Closed circle: add first point at end
        theta = np.linspace(0, 2 * np.pi, n + 1)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n + 1)])
        np.testing.assert_allclose(arc_length_total(pts), 2 * np.pi * r, rtol=1e-4)


class TestTangentVectors:
    def test_straight_line_tangents(self) -> None:
        pts = _straight_line(5)
        t = tangent_vectors(pts)
        expected = np.column_stack([np.ones(5), np.zeros(5), np.zeros(5)])
        np.testing.assert_allclose(t, expected, atol=1e-15)

    def test_unit_length(self) -> None:
        pts = _circle_points(50)
        t = tangent_vectors(pts)
        norms = np.linalg.norm(t, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


class TestCurvature:
    def test_straight_line_zero_curvature(self) -> None:
        pts = _straight_line(10)
        kappa = curvature(pts)
        np.testing.assert_allclose(kappa, 0.0, atol=1e-12)

    def test_circle_curvature(self) -> None:
        n = 500
        r = 3.0
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n)])
        kappa = curvature(pts)
        # Interior points should approximate 1/R
        np.testing.assert_allclose(kappa[10:-10], 1.0 / r, rtol=0.02)


class TestDisplacement:
    def test_straight_line(self) -> None:
        pts = _straight_line(5)
        assert displacement(pts) == 4.0

    def test_closed_curve_zero(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 100)
        pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(100)])
        np.testing.assert_allclose(displacement(pts), 0.0, atol=1e-12)


class TestClosestApproach:
    def test_exact_point(self) -> None:
        pts = _straight_line(5)
        idx, dist = closest_approach(pts, (2.0, 0.0, 0.0))
        assert idx == 2
        np.testing.assert_allclose(dist, 0.0, atol=1e-15)

    def test_between_points(self) -> None:
        pts = _straight_line(5)
        idx, dist = closest_approach(pts, (1.3, 0.0, 0.0))
        assert idx == 1
        np.testing.assert_allclose(dist, 0.3, atol=1e-15)


class TestPlaneCrossings:
    def test_single_crossing(self) -> None:
        pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        crossings = plane_crossings(pts, (1.0, 0.0, 0.0), 0.0)
        assert crossings.shape == (1, 3)
        np.testing.assert_array_equal(crossings[0], [0.0, 0.0, 0.0])

    def test_no_crossing(self) -> None:
        pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        crossings = plane_crossings(pts, (1.0, 0.0, 0.0), 0.0)
        assert crossings.shape == (0, 3)

    def test_sine_curve_z_crossings(self) -> None:
        t = np.linspace(0, 4 * np.pi, 1000)
        pts = np.column_stack([t, np.zeros_like(t), np.sin(t)])
        crossings = equatorial_crossings(pts)
        # sin has sign changes at pi, 2pi, 3pi (not at endpoints 0, 4pi)
        assert crossings.shape[0] == 3

    def test_equatorial_equals_z_plane(self) -> None:
        pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -0.5]])
        eq = equatorial_crossings(pts)
        manual = plane_crossings(pts, (0.0, 0.0, 1.0), 0.0)
        np.testing.assert_array_equal(eq, manual)


class TestResample:
    def test_output_count(self) -> None:
        pts = _straight_line(10)
        new_pts, _ = resample_by_arc_length(pts, 20)
        assert new_pts.shape == (20, 3)

    def test_uniform_spacing(self) -> None:
        pts = _straight_line(10)
        new_pts, _ = resample_by_arc_length(pts, 5)
        # x should be uniformly spaced
        dx = np.diff(new_pts[:, 0])
        np.testing.assert_allclose(dx, dx[0], rtol=1e-12)

    def test_scalars_resampled(self) -> None:
        pts = _straight_line(5)
        scalars = {"rho": np.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        _, new_scalars = resample_by_arc_length(pts, 9, scalars=scalars)
        assert "rho" in new_scalars
        assert new_scalars["rho"].shape == (9,)
        np.testing.assert_array_equal(new_scalars["rho"][0], 0.0)
        np.testing.assert_array_equal(new_scalars["rho"][-1], 4.0)

    def test_too_few_rejects(self) -> None:
        with pytest.raises(ValueError, match="n_out must be >= 2"):
            resample_by_arc_length(_straight_line(5), 1)


class TestSpeed:
    def test_constant_velocity(self) -> None:
        v = np.full((5, 3), [3.0, 4.0, 0.0])
        s = speed(v)
        np.testing.assert_array_equal(s, 5.0)


class TestKineticEnergy:
    def test_basic(self) -> None:
        v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ke = kinetic_energy(v, mass=2.0)
        np.testing.assert_array_equal(ke, [1.0, 4.0])


class TestMirrorPoints:
    def test_known_maxima(self) -> None:
        b = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        idx = mirror_points(b)
        np.testing.assert_array_equal(idx, [1, 3])

    def test_no_maxima(self) -> None:
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = mirror_points(b)
        assert len(idx) == 0

    def test_too_short(self) -> None:
        idx = mirror_points(np.array([1.0, 2.0]))
        assert len(idx) == 0


class TestGyroradiusEstimate:
    def test_uniform_field(self) -> None:
        n = 50
        pts = _circle_points(n, radius=1.0)
        v_mag = 2.0
        tangents = tangent_vectors(pts)
        # Velocity purely perpendicular to the curve tangent (which the
        # estimator uses as the B-direction proxy): v_perp = v_mag.
        # r_g = m * v_perp / (|q| * B) = 2 * 2 / (1 * 4) = 1.0
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0], np.zeros(n)])
        vel = v_mag * normals
        b_mag = np.full(n, 4.0)
        rg = gyroradius_estimate(pts, vel, b_mag, charge=1.0, mass=2.0)
        assert rg.shape == (n,)
        np.testing.assert_allclose(rg, 1.0, rtol=0.05)

    def test_zero_b_gives_nan(self) -> None:
        pts = _straight_line(3)
        vel = np.ones((3, 3))
        b_mag = np.array([1.0, 0.0, 1.0])
        rg = gyroradius_estimate(pts, vel, b_mag, charge=1.0, mass=1.0)
        assert np.isnan(rg[1])


class TestDriftVelocity:
    def test_constant_motion(self) -> None:
        n = 20
        t = np.linspace(0, 1, n)
        pts = np.column_stack([2.0 * t, np.zeros(n), np.zeros(n)])
        vd = drift_velocity(pts, t, window=1)
        np.testing.assert_allclose(vd[:, 0], 2.0, rtol=1e-10)

    def test_smoothing_reduces_oscillation(self) -> None:
        n = 100
        t = np.linspace(0, 10, n)
        gyro = 0.5 * np.sin(20 * t)
        pts = np.column_stack([t, gyro, np.zeros(n)])
        vd_raw = drift_velocity(pts, t, window=1)
        vd_smooth = drift_velocity(pts, t, window=11)
        # Smoothed y-drift should have less variance
        assert np.std(vd_smooth[:, 1]) < np.std(vd_raw[:, 1])


class TestSampling:
    @pytest.fixture
    def field_dataset(self) -> object:
        """3D FieldDataset with a simple scalar field."""
        from pypic.dataset import FieldDataset
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(4, 4, 4),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        # Field value = x coordinate at each cell center
        x = np.arange(4) * 1.0 + 0.5
        field = np.broadcast_to(x[:, None, None], (4, 4, 4)).copy().astype(np.float64)
        return FieldDataset.from_arrays({"rho": field}, grid, strict_fields=False)

    def test_nearest_at_grid_nodes(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="nearest")
        np.testing.assert_array_equal(values, [0.5, 1.5, 2.5])

    def test_outside_domain_returns_nan(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[-10.0, 0.5, 0.5], [100.0, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="nearest")
        assert np.all(np.isnan(values))

    def test_linear_interpolation(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        # Point between two cell centers
        pts = np.array([[1.0, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="linear")
        np.testing.assert_allclose(values, [1.0], atol=0.1)

    def test_attach_scalars_to_fieldline(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        fl = FieldLine(
            points=pts,
            field_name="B",
            seed_point=(0.5, 0.5, 0.5),
            normalization=_identity(),
        )
        fl2 = attach_scalars(fl, data, ["rho"])
        assert "rho" in fl2.scalars
        np.testing.assert_array_equal(fl2.scalars["rho"], [0.5, 1.5, 2.5])

    def test_attach_scalars_to_particletrace(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        tr = ParticleTrace(
            points=pts,
            time=np.array([0.0, 1.0, 2.0]),
            velocity=np.ones((3, 3)),
            species_name="e",
            normalization=_identity(),
        )
        tr2 = attach_scalars_to_trace(tr, data, ["rho"])
        assert "rho" in tr2.scalars
        np.testing.assert_array_equal(tr2.scalars["rho"], [0.5, 1.5, 2.5])

    def test_invalid_method_rejects(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="Unknown interpolation"):
            sample_field(data, pts, "rho", method="cubic")


class TestSampleFieldsBatching:
    """Batched ``sample_fields`` matches per-field ``sample_field`` results.

    Regression guard for the P3b refactor that shares the nearest-index
    computation (``method="nearest"``) and interpolator construction
    (``method="linear"``) across all requested fields in one call.
    """

    @pytest.fixture
    def multi_field_dataset(self) -> FieldDataset:
        """3D dataset with three distinct fields on the same grid."""
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(5, 4, 3),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        x, y, z = (
            a.astype(np.float64)
            for a in np.meshgrid(
                np.arange(5) + 0.5,
                np.arange(4) + 0.5,
                np.arange(3) + 0.5,
                indexing="ij",
            )
        )
        return FieldDataset.from_arrays(
            {"B_1": x, "B_2": 2.0 * y, "B_3": x + y + z},
            grid,
            _identity(),
        )

    def test_empty_fields_returns_empty_dict(
        self, multi_field_dataset: FieldDataset
    ) -> None:
        pts = np.array([[0.5, 0.5, 0.5]])
        assert sample_fields(multi_field_dataset, pts, [], method="nearest") == {}

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_batched_matches_per_field(
        self, multi_field_dataset: FieldDataset, method: str
    ) -> None:
        # Mix of in-bounds and out-of-bounds points to exercise the NaN mask.
        pts = np.array(
            [
                [0.5, 0.5, 0.5],
                [2.3, 1.7, 1.1],
                [4.4, 3.4, 2.4],
                [-1.0, 0.5, 0.5],
                [100.0, 0.5, 0.5],
            ]
        )
        names = ["B_1", "B_2", "B_3"]
        batched = sample_fields(multi_field_dataset, pts, names, method=method)
        for name in names:
            per_field = sample_field(multi_field_dataset, pts, name, method=method)
            np.testing.assert_array_equal(batched[name], per_field)


class TestSamplingEdgeCases:
    """Edge cases: degenerate grid axes, NaN coords."""

    def test_single_node_axis_does_not_crash(self) -> None:
        """A FieldDataset with one node along z should sample cleanly.

        Regression for the ``np.clip(0, len-2)`` underflow path in
        ``_nearest_indices`` (Unit 14b).
        """
        from pypic.dataset import FieldDataset
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(4, 4, 1),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        # Field varies along x; constant along y, z
        x = np.arange(4) * 1.0 + 0.5
        field = np.broadcast_to(x[:, None, None], (4, 4, 1)).copy().astype(np.float64)
        data = FieldDataset.from_arrays({"rho": field}, grid, strict_fields=False)
        # Sample at y=0.5, z=0.5 (the only valid z slice)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="nearest")
        np.testing.assert_array_equal(values, [0.5, 1.5, 2.5])

    def test_single_node_axis_far_z_out_of_bounds(self) -> None:
        """Querying far outside the (single) z node returns NaN, not garbage."""
        from pypic.dataset import FieldDataset
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(4, 4, 1),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        field = np.zeros((4, 4, 1), dtype=np.float64)
        data = FieldDataset.from_arrays({"rho": field}, grid, strict_fields=False)
        pts = np.array([[0.5, 0.5, 100.0]])  # z way outside the single node
        values = sample_field(data, pts, "rho", method="nearest")
        assert np.all(np.isnan(values))


@pytest.fixture
def uniform_field_data() -> FieldDataset:
    """3D uniform B=(1,0,0) field on a 20x20x20 grid for tracing tests."""
    from pypic.grid import GridInfo

    grid = GridInfo(
        dimensions=(20, 20, 20),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    ones = np.ones((20, 20, 20), dtype=np.float64)
    zeros = np.zeros((20, 20, 20), dtype=np.float64)
    return FieldDataset.from_arrays(
        {"B_1": ones, "B_2": zeros, "B_3": zeros},
        grid,
        Normalization.identity(),
    )


class TestVectorFieldInterpolator:
    def test_from_dataset(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import VectorFieldInterpolator

        interp = VectorFieldInterpolator.from_dataset(uniform_field_data)
        # Verify construction succeeded AND evaluates correctly at the center
        result = interp(np.array([10.0, 10.0, 10.0]))
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_call_inside_domain(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import VectorFieldInterpolator

        interp = VectorFieldInterpolator.from_dataset(uniform_field_data)
        result = interp(np.array([10.0, 10.0, 10.0]))
        assert result.shape == (3,)
        assert not np.any(np.isnan(result))
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_call_outside_domain(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import VectorFieldInterpolator

        interp = VectorFieldInterpolator.from_dataset(uniform_field_data)
        result = interp(np.array([1e6, 1e6, 1e6]))
        assert np.all(np.isnan(result))


class TestTraceFieldLine:
    def test_uniform_field_straight_line(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_line

        seed = (10.0, 10.0, 10.0)
        fl = trace_field_line(
            uniform_field_data, seed, step_size=0.5, max_steps=10, direction="forward"
        )
        # In a uniform B=(1,0,0) field, the trace should move along +x
        assert fl.n_points > 2
        np.testing.assert_allclose(fl.points[:, 1], 10.0, atol=1e-10)
        np.testing.assert_allclose(fl.points[:, 2], 10.0, atol=1e-10)
        assert fl.points[-1, 0] > fl.points[0, 0]

    def test_forward_only(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=5,
            direction="forward",
        )
        assert fl.direction == "forward"
        assert fl.points[0, 0] == pytest.approx(10.0)
        diffs = np.diff(fl.points[:, 0])
        assert np.all(diffs >= 0)

    def test_backward_only(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=5,
            direction="backward",
        )
        assert fl.direction == "backward"
        # Backward trace should go in -x direction; points stored start-to-end
        assert fl.points[0, 0] < fl.points[-1, 0]

    def test_both_directions(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=5,
            direction="both",
        )
        assert fl.direction == "both"
        assert fl.n_points > 3  # at least fwd + bwd + seed

    def test_domain_exit_terminates(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import TerminationReason, trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (18.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=100,
            direction="forward",
        )
        assert fl.metadata["reason"] == str(TerminationReason.DOMAIN_EXIT)

    def test_terminate_callback(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import TerminationReason, trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=100,
            direction="forward",
            terminate=lambda pt: pt[0] > 12.0,
        )
        assert fl.metadata["reason"] == str(TerminationReason.CALLBACK)
        assert fl.points[-1, 0] > 12.0

    def test_invalid_direction_rejects(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_line

        with pytest.raises(ValueError, match="direction must be"):
            trace_field_line(uniform_field_data, (10.0, 10.0, 10.0), direction="up")

    def test_seed_outside_domain_rejects(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_line

        with pytest.raises(ValueError, match="outside"):
            trace_field_line(uniform_field_data, (1e6, 1e6, 1e6))

    def test_callback_terminates_after_first_step(
        self,
        uniform_field_data: FieldDataset,
    ) -> None:
        """A callback that returns True after one step yields a 2-point line."""
        from pypic.traces import TerminationReason, trace_field_line

        calls = {"n": 0}

        def stop_after_one(_pt: np.ndarray) -> bool:
            calls["n"] += 1
            return calls["n"] >= 1

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=0.5,
            max_steps=100,
            direction="forward",
            terminate=stop_after_one,
        )
        assert fl.metadata["reason"] == str(TerminationReason.CALLBACK)
        # Seed + one accepted step = 2 points
        assert fl.n_points == 2


class TestTraceFieldLineAdaptive:
    def test_uniform_field_straight_line(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_line_adaptive

        seed = (10.0, 10.0, 10.0)
        fl = trace_field_line_adaptive(
            uniform_field_data,
            seed,
            step_size_init=0.5,
            max_steps=10,
            direction="forward",
        )
        # In a uniform B=(1,0,0) field, trace should be a straight line along x
        np.testing.assert_allclose(fl.points[:, 1], 10.0, atol=1e-10)
        np.testing.assert_allclose(fl.points[:, 2], 10.0, atol=1e-10)
        assert fl.points[-1, 0] > fl.points[0, 0]

    def test_adaptive_stores_max_local_error(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_line_adaptive

        fl = trace_field_line_adaptive(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            max_steps=10,
            direction="forward",
        )
        assert "max_local_error" in fl.metadata
        assert fl.metadata["method"] == "rk45_dopri"


class TestTraceFieldLinesAdaptive:
    """Batched adaptive tracer over N seeds.

    Equivalence with the single-seed adaptive tracer is the core
    contract: tracing N seeds in the batch must produce the same
    FieldLine endpoints (up to floating-point noise from the kernel's
    different evaluation order) as N independent calls to
    :func:`trace_field_line_adaptive`.
    """

    def test_n_lines_returned_in_seed_order(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_lines_adaptive

        seeds = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [15.0, 15.0, 15.0]])
        lines = trace_field_lines_adaptive(
            uniform_field_data, seeds, max_steps=10, direction="forward"
        )
        assert len(lines) == 3
        # Seed-order preserved: each line starts at its seed.
        for i, line in enumerate(lines):
            np.testing.assert_array_equal(line.points[0], seeds[i])

    def test_equivalence_to_per_seed_loop(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """Endpoints match a per-seed scalar adaptive trace."""
        from pypic.traces import trace_field_line_adaptive, trace_field_lines_adaptive

        seeds = np.array([[5.0, 8.0, 7.0], [10.0, 10.0, 10.0], [12.0, 6.0, 14.0]])
        kw: dict = dict(  # type: ignore[type-arg]
            atol=1e-8, rtol=1e-8, step_size_init=0.5, max_steps=20, direction="forward"
        )
        batched = trace_field_lines_adaptive(uniform_field_data, seeds, **kw)
        for i in range(len(seeds)):
            scalar = trace_field_line_adaptive(
                uniform_field_data, tuple(seeds[i].tolist()), **kw
            )
            np.testing.assert_allclose(
                batched[i].points[-1], scalar.points[-1], atol=1e-9
            )
            # Same termination reason recorded.
            assert batched[i].metadata["reason"] == scalar.metadata["reason"]

    def test_mixed_termination_per_seed(self, uniform_field_data: FieldDataset) -> None:
        """One seed near the +x edge exits domain; others run to MAX_STEPS."""
        from pypic.traces import TerminationReason, trace_field_lines_adaptive

        # Domain is [0, 19]^3, B = (1,0,0). With max_step=0.5 and
        # max_steps=8, interior seeds advance at most 4.0 in +x. Seed
        # near the +x boundary exits before consuming all steps.
        seeds = np.array([[18.5, 10.0, 10.0], [5.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
        lines = trace_field_lines_adaptive(
            uniform_field_data,
            seeds,
            step_size_init=0.3,
            min_step=1e-3,
            max_step=0.5,
            max_steps=8,
            direction="forward",
        )
        # Seed 0 exits the +x boundary before consuming all steps.
        assert lines[0].metadata["reason"] == str(TerminationReason.DOMAIN_EXIT)
        # Seeds 1 and 2 run to MAX_STEPS (still inside domain).
        assert lines[1].metadata["reason"] == str(TerminationReason.MAX_STEPS)
        assert lines[2].metadata["reason"] == str(TerminationReason.MAX_STEPS)

    def test_terminate_callback_fires_per_seed(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """``terminate=`` fires asymmetrically: one seed stops, others run."""
        from pypic.traces import TerminationReason, trace_field_lines_adaptive

        # B = (1,0,0). With x > 12 the callback only ever triggers on the
        # seed that started at x=10 (which crosses 12 quickly); the seed
        # at x=5 doesn't reach 12 within the step budget.
        seeds = np.array([[10.0, 10.0, 10.0], [5.0, 10.0, 10.0]])
        lines = trace_field_lines_adaptive(
            uniform_field_data,
            seeds,
            step_size_init=0.3,
            min_step=1e-3,
            max_step=0.5,
            max_steps=8,
            direction="forward",
            terminate=lambda pt: pt[0] > 12.0,
        )
        assert lines[0].metadata["reason"] == str(TerminationReason.CALLBACK)
        assert lines[1].metadata["reason"] == str(TerminationReason.MAX_STEPS)

    def test_invalid_seed_shape_raises(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_lines_adaptive

        with pytest.raises(ValueError, match=r"seeds must have shape \(N, 3\)"):
            trace_field_lines_adaptive(
                uniform_field_data, np.array([1.0, 2.0, 3.0]), max_steps=4
            )

    def test_seed_outside_domain_raises(self, uniform_field_data: FieldDataset) -> None:
        """One bad seed in the batch fails the whole call (matches scalar contract)."""
        from pypic.traces import trace_field_lines_adaptive

        seeds = np.array([[10.0, 10.0, 10.0], [-1e6, -1e6, -1e6]])
        with pytest.raises(ValueError, match=r"outside the interpolation domain"):
            trace_field_lines_adaptive(uniform_field_data, seeds, max_steps=4)

    def test_both_directions(self, uniform_field_data: FieldDataset) -> None:
        """direction='both' produces lines longer than either single direction."""
        from pypic.traces import trace_field_lines_adaptive

        seeds = np.array([[10.0, 10.0, 10.0], [9.0, 10.0, 10.0]])
        fwd = trace_field_lines_adaptive(
            uniform_field_data, seeds, max_steps=8, direction="forward"
        )
        both = trace_field_lines_adaptive(
            uniform_field_data, seeds, max_steps=8, direction="both"
        )
        for i in range(len(seeds)):
            assert both[i].n_points >= fwd[i].n_points
            # Bidirectional trace passes through the seed point.
            assert any(np.allclose(p, seeds[i]) for p in both[i].points)


class TestClosedLoopDetection:
    """Sliding-window proximity detector for trapped / closed orbits.

    Mirror-mode magnetic holes and O-type islands have closed
    field-line topology — a tracer entering one orbits indefinitely
    and otherwise burns through ``max_steps``. The detector terminates
    with :attr:`TerminationReason.CLOSED_LOOP` when the trace re-enters
    a ``loop_tol``-radius ball around a past point separated by more
    than ``loop_min_arclen`` of arc length.
    """

    @pytest.fixture
    def closed_loop_field_data(self) -> FieldDataset:
        """Closed circular field $\\mathbf{B} = (-(y-cy), (x-cx), 0)$.

        Centered at (10, 10, 10) on a 20³ domain. Field-line topology
        is concentric circles in the z=const plane; the centre is a
        magnetic null (|B|=0) so seeds must lie off-axis.
        """
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(20, 20, 20),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        cx, cy = 10.0, 10.0
        coords = np.arange(20.0)
        x, y, _z = np.meshgrid(coords, coords, coords, indexing="ij")
        b1 = -(y - cy)
        b2 = x - cx
        b3 = np.zeros_like(x)
        return FieldDataset.from_arrays(
            {"B_1": b1, "B_2": b2, "B_3": b3},
            grid,
            Normalization.identity(),
        )

    @pytest.fixture
    def vortex_field_data(self) -> FieldDataset:
        """Helical field $\\mathbf{B} = (-(y-cy), (x-cx), 0.2)$.

        Same vortex pattern as :func:`closed_loop_field_data`, plus a
        constant axial drift. Field lines are open helices that exit
        the +z boundary; closed-loop detection should NOT trigger
        because each turn is offset in z by $2\\pi \\cdot 0.2 \\approx
        1.26$ — much larger than any reasonable ``loop_tol``.
        """
        from pypic.grid import GridInfo

        grid = GridInfo(
            dimensions=(20, 20, 20),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        cx, cy = 10.0, 10.0
        coords = np.arange(20.0)
        x, y, _z = np.meshgrid(coords, coords, coords, indexing="ij")
        b1 = -(y - cy)
        b2 = x - cx
        b3 = 0.2 * np.ones_like(x)
        return FieldDataset.from_arrays(
            {"B_1": b1, "B_2": b2, "B_3": b3},
            grid,
            Normalization.identity(),
        )

    def test_closed_circle_triggers_well_before_max_steps(
        self, closed_loop_field_data: FieldDataset
    ) -> None:
        """A trace on a closed-circle field terminates with CLOSED_LOOP."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        # Seed at radius 2 from the centre — one orbit is ~4π ≈ 12.6 of
        # arc length. With loop_tol=0.1 and loop_min_arclen=2.0, the
        # detector should fire well within a single orbit.
        fl = trace_field_line_adaptive(
            closed_loop_field_data,
            (12.0, 10.0, 10.0),
            step_size_init=0.1,
            max_step=0.2,
            max_steps=500,
            direction="forward",
            loop_tol=0.1,
            loop_min_arclen=2.0,
        )
        assert fl.metadata["reason"] == str(TerminationReason.CLOSED_LOOP)
        # Should terminate before consuming half the step budget.
        assert fl.n_points < 250

    def test_uniform_field_does_not_trigger(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """Straight-line trace: detection must NOT fire."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        fl = trace_field_line_adaptive(
            uniform_field_data,
            (5.0, 10.0, 10.0),
            step_size_init=0.5,
            max_step=0.5,
            max_steps=20,
            direction="forward",
            loop_tol=0.1,
            loop_min_arclen=2.0,
        )
        assert fl.metadata["reason"] != str(TerminationReason.CLOSED_LOOP)

    def test_open_helix_does_not_false_trigger(
        self, vortex_field_data: FieldDataset
    ) -> None:
        """Helical trace with axial pitch >> loop_tol must NOT fire."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        # Pitch per turn ≈ 2π * 0.2 / sqrt(1 + 0.2² normalization) ≈
        # 1.23 in arc length. With loop_tol=0.2 (much less than pitch)
        # and a couple of orbits of step budget, the detector must not
        # mistake near-passes for a closed loop.
        fl = trace_field_line_adaptive(
            vortex_field_data,
            (12.0, 10.0, 10.0),
            step_size_init=0.1,
            max_step=0.2,
            max_steps=400,
            direction="forward",
            loop_tol=0.2,
            loop_min_arclen=2.0,
        )
        assert fl.metadata["reason"] != str(TerminationReason.CLOSED_LOOP)

    def test_loop_min_arclen_guards_against_self_trigger(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """A loose ``loop_tol`` with too-small arclen guard self-triggers.

        Concrete arithmetic, so the test reads obviously-correct:

        - $\\mathbf{B} = \\hat{x}$ on a uniform grid → unit-magnitude
          tangent, so arc length per step equals the step size.
        - ``step_size_init=0.5``, ``max_step=0.5`` → every accepted step
          contributes 0.5 of arc length, and the past sample one step
          back is exactly 0.5 away in 3-space.
        - ``loop_tol=1.0`` is *larger* than that gap → distance check
          would admit the past sample as a "loop closure".
        - The ``loop_min_arclen`` guard is the only thing that excludes
          it. With 0.01, every past sample qualifies → detector fires
          on the first scan. With 100.0, no past sample is old enough
          → scan finds no candidates and the trace runs to its natural
          terminus.

        Also pins (B3) that the self-trigger fires within the first few
        accepted steps — a future buffer-layout change that shifts
        *which* past sample is matched will move ``n_points`` and the
        assertion catches the drift.
        """
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        seed = (5.0, 10.0, 10.0)
        # loop_min_arclen smaller than the first step's arc length →
        # the (n-1)-th sample sneaks past the guard.
        fl_self_trigger = trace_field_line_adaptive(
            uniform_field_data,
            seed,
            step_size_init=0.5,
            max_step=0.5,
            max_steps=20,
            direction="forward",
            loop_tol=1.0,
            loop_min_arclen=0.01,
        )
        assert fl_self_trigger.metadata["reason"] == str(TerminationReason.CLOSED_LOOP)
        # Detector must fire within the first few accepted steps — not
        # late in the trace. Without this, a buffer-layout regression
        # that pushes the match to step ~10 would pass silently.
        assert fl_self_trigger.n_points <= 5, (
            f"self-trigger fired late ({fl_self_trigger.n_points} points); "
            "expected detection within first few accepted steps"
        )

        # loop_min_arclen larger than any arc length reached →
        # detector never has candidates to test.
        fl_guarded = trace_field_line_adaptive(
            uniform_field_data,
            seed,
            step_size_init=0.5,
            max_step=0.5,
            max_steps=20,
            direction="forward",
            loop_tol=1.0,
            loop_min_arclen=100.0,
        )
        assert fl_guarded.metadata["reason"] != str(TerminationReason.CLOSED_LOOP)

    def test_scalar_and_batched_detectors_agree_on_one_seed(
        self, closed_loop_field_data: FieldDataset
    ) -> None:
        """One seed through both paths stops at the same step, same reason.

        The two detectors express the same predicate differently — the
        scalar path bounds the scan with ``searchsorted``, the batched
        path masks a rectangular window — so only a parity check keeps
        them from drifting apart.
        """
        from pypic.traces import trace_field_line_adaptive, trace_field_lines_adaptive

        seed = (12.0, 10.0, 10.0)
        kwargs = {
            "step_size_init": 0.1,
            "max_step": 0.2,
            "max_steps": 200,
            "direction": "forward",
            "loop_tol": 0.1,
            "loop_min_arclen": 2.0,
        }
        scalar = trace_field_line_adaptive(closed_loop_field_data, seed, **kwargs)
        (batched,) = trace_field_lines_adaptive(
            closed_loop_field_data, np.array([seed]), **kwargs
        )
        assert scalar.metadata["reason"] == batched.metadata["reason"]
        np.testing.assert_allclose(scalar.points, batched.points, rtol=1e-12)

    def test_batched_mixed_seed_termination(
        self,
        closed_loop_field_data: FieldDataset,
    ) -> None:
        """Batched: per-seed CLOSED_LOOP vs MAX_STEPS.

        Both seeds orbit (it's a vortex field — there's no escape),
        but only the small-radius orbit completes within the step
        budget. The large-radius orbit is truncated at MAX_STEPS, so
        the detector never sees the trace return to the seed.
        """
        from pypic.traces import TerminationReason, trace_field_lines_adaptive

        # Seed 0: radius 2 → circumference 4π ≈ 12.6, completes one
        #         orbit in ~126 steps at step_size 0.1.
        # Seed 1: radius 8 → circumference 16π ≈ 50.3, well beyond
        #         the 200-step × 0.2-max-step = 40 arclen budget.
        seeds = np.array([[12.0, 10.0, 10.0], [18.0, 10.0, 10.0]])
        lines = trace_field_lines_adaptive(
            closed_loop_field_data,
            seeds,
            step_size_init=0.1,
            max_step=0.2,
            max_steps=200,
            direction="forward",
            loop_tol=0.1,
            loop_min_arclen=2.0,
        )
        assert lines[0].metadata["reason"] == str(TerminationReason.CLOSED_LOOP)
        assert lines[1].metadata["reason"] == str(TerminationReason.MAX_STEPS)

    def test_explicit_none_disables(self, closed_loop_field_data: FieldDataset) -> None:
        """``loop_tol=None`` is the explicit opt-out — runs to MAX_STEPS."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        fl = trace_field_line_adaptive(
            closed_loop_field_data,
            (12.0, 10.0, 10.0),
            step_size_init=0.1,
            max_step=0.2,
            max_steps=200,
            direction="forward",
            loop_tol=None,
        )
        assert fl.metadata["reason"] == str(TerminationReason.MAX_STEPS)

    def test_auto_default_triggers_on_closed_circle(
        self, closed_loop_field_data: FieldDataset
    ) -> None:
        """Without any loop kwargs the default ``"auto"`` catches closed orbits."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        fl = trace_field_line_adaptive(
            closed_loop_field_data,
            (12.0, 10.0, 10.0),
            step_size_init=0.1,
            max_step=0.2,
            max_steps=500,
            direction="forward",
        )
        assert fl.metadata["reason"] == str(TerminationReason.CLOSED_LOOP)

    def test_auto_default_does_not_trigger_on_uniform(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """The default detector must not fire on a smooth open trace."""
        from pypic.traces import TerminationReason, trace_field_line_adaptive

        fl = trace_field_line_adaptive(
            uniform_field_data,
            (5.0, 10.0, 10.0),
            step_size_init=0.5,
            max_step=0.5,
            max_steps=20,
            direction="forward",
        )
        assert fl.metadata["reason"] != str(TerminationReason.CLOSED_LOOP)

    def test_auto_equals_half_min_spacing(
        self, closed_loop_field_data: FieldDataset
    ) -> None:
        """``"auto"`` derives ``0.5 * min(grid.spacing)`` — endpoints match.

        Sanity-check the formula doesn't drift: an explicit float equal
        to the auto-derived value must produce the same trace as the
        ``"auto"`` sentinel.
        """
        from pypic.traces import trace_field_line_adaptive

        spacing = closed_loop_field_data.grid.spacing
        explicit_tol = 0.5 * min(spacing)
        kw: dict = dict(  # type: ignore[type-arg]
            step_size_init=0.1,
            max_step=0.2,
            max_steps=500,
            direction="forward",
        )
        fl_auto = trace_field_line_adaptive(
            closed_loop_field_data, (12.0, 10.0, 10.0), **kw
        )
        fl_explicit = trace_field_line_adaptive(
            closed_loop_field_data,
            (12.0, 10.0, 10.0),
            loop_tol=explicit_tol,
            **kw,
        )
        np.testing.assert_array_equal(fl_auto.points, fl_explicit.points)
        assert fl_auto.metadata["reason"] == fl_explicit.metadata["reason"]

    def test_loop_min_arclen_default_equals_ten_step_init(
        self, closed_loop_field_data: FieldDataset
    ) -> None:
        """Default ``loop_min_arclen = 10 * step_size_init`` — endpoints match.

        Companion to ``test_auto_equals_half_min_spacing``: pins the
        second of the two grid/step-derived defaults in
        ``_resolve_loop_kwargs``. A future retune (10 → 8, say) is
        caught by element-wise equality of the trace points.
        """
        from pypic.traces import trace_field_line_adaptive

        kw: dict = dict(  # type: ignore[type-arg]
            step_size_init=0.1,
            max_step=0.2,
            max_steps=500,
            direction="forward",
        )
        fl_default = trace_field_line_adaptive(
            closed_loop_field_data, (12.0, 10.0, 10.0), **kw
        )
        fl_explicit = trace_field_line_adaptive(
            closed_loop_field_data,
            (12.0, 10.0, 10.0),
            loop_min_arclen=10.0 * kw["step_size_init"],
            **kw,
        )
        np.testing.assert_array_equal(fl_default.points, fl_explicit.points)
        assert fl_default.metadata["reason"] == fl_explicit.metadata["reason"]

    def test_loop_min_arclen_with_explicit_none_raises(
        self, uniform_field_data: FieldDataset
    ) -> None:
        """``loop_min_arclen`` without an active ``loop_tol`` is a misuse.

        With the auto default in effect, this is reachable only when the
        caller has explicitly disabled detection via ``loop_tol=None``.
        """
        from pypic.traces import trace_field_line_adaptive

        with pytest.raises(ValueError, match="loop_min_arclen requires loop_tol"):
            trace_field_line_adaptive(
                uniform_field_data,
                (10.0, 10.0, 10.0),
                max_steps=4,
                loop_tol=None,
                loop_min_arclen=1.0,
            )

    def test_negative_loop_tol_raises(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import trace_field_line_adaptive

        with pytest.raises(ValueError, match="loop_tol must be positive"):
            trace_field_line_adaptive(
                uniform_field_data,
                (10.0, 10.0, 10.0),
                max_steps=4,
                loop_tol=-0.1,
            )

    def test_negative_loop_min_arclen_raises(
        self, uniform_field_data: FieldDataset
    ) -> None:
        from pypic.traces import trace_field_line_adaptive

        with pytest.raises(ValueError, match="loop_min_arclen must be positive"):
            trace_field_line_adaptive(
                uniform_field_data,
                (10.0, 10.0, 10.0),
                max_steps=4,
                loop_tol=0.1,
                loop_min_arclen=-1.0,
            )


class TestEstimateTracingError:
    def test_error_estimate(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import estimate_tracing_error, trace_field_line

        fl = trace_field_line(
            uniform_field_data,
            (10.0, 10.0, 10.0),
            step_size=1.0,
            max_steps=5,
            direction="forward",
        )
        err = estimate_tracing_error(fl, uniform_field_data)
        # For a uniform field, RK4 is exact — error should be near zero
        assert err < 1e-6

    def test_missing_metadata_raises(self) -> None:
        from pypic.traces import estimate_tracing_error

        fl = FieldLine(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            field_name="B",
            seed_point=(0.0, 0.0, 0.0),
            normalization=_identity(),
        )
        with pytest.raises(ValueError, match="metadata missing"):
            estimate_tracing_error(fl, None)  # type: ignore[arg-type]


class TestFieldNameFromComponents:
    def test_valid(self) -> None:
        from pypic.traces._tracing import _field_name_from_components

        assert _field_name_from_components(("B_1", "B_2", "B_3")) == "B"
        assert _field_name_from_components(("Ve_1", "Ve_2", "Ve_3")) == "Ve"

    def test_mismatched_rejects(self) -> None:
        from pypic.traces._tracing import _field_name_from_components

        with pytest.raises(ValueError, match="same field"):
            _field_name_from_components(("B_1", "E_2", "B_3"))
