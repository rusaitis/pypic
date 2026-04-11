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
        np.testing.assert_allclose(s, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_total_equals_last_cumulative(self) -> None:
        pts = _straight_line(10)
        assert arc_length_total(pts) == arc_length_cumulative(pts)[-1]

    def test_diagonal(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        np.testing.assert_allclose(arc_length_total(pts), 5.0)

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
        np.testing.assert_allclose(crossings[0], [0.0, 0.0, 0.0])

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
        np.testing.assert_allclose(eq, manual)


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
        np.testing.assert_allclose(new_scalars["rho"][0], 0.0)
        np.testing.assert_allclose(new_scalars["rho"][-1], 4.0)

    def test_too_few_rejects(self) -> None:
        with pytest.raises(ValueError, match="n_out must be >= 2"):
            resample_by_arc_length(_straight_line(5), 1)


class TestSpeed:
    def test_constant_velocity(self) -> None:
        v = np.full((5, 3), [3.0, 4.0, 0.0])
        s = speed(v)
        np.testing.assert_allclose(s, 5.0)


class TestKineticEnergy:
    def test_basic(self) -> None:
        v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ke = kinetic_energy(v, mass=2.0)
        np.testing.assert_allclose(ke, [1.0, 4.0])


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
        vel = v_mag * tangents
        b_mag = np.full(n, 4.0)
        rg = gyroradius_estimate(pts, vel, b_mag, charge=1.0, mass=2.0)
        # For pure perpendicular motion: r_g = m * v_perp / (|q| * B) = 2*2/(1*4) = 1
        # The estimate is approximate since tangent ≠ B direction
        assert rg.shape == (n,)
        assert np.all(np.isfinite(rg))

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
        return FieldDataset.from_arrays({"rho": field}, grid)

    def test_nearest_at_grid_nodes(self, field_dataset: object) -> None:
        from pypic.dataset import FieldDataset

        data = field_dataset
        assert isinstance(data, FieldDataset)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="nearest")
        np.testing.assert_allclose(values, [0.5, 1.5, 2.5])

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
        np.testing.assert_allclose(fl2.scalars["rho"], [0.5, 1.5, 2.5])

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
            {"B1": x, "B2": 2.0 * y, "B3": x + y + z},
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
        names = ["B1", "B2", "B3"]
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
        data = FieldDataset.from_arrays({"rho": field}, grid)
        # Sample at y=0.5, z=0.5 (the only valid z slice)
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
        values = sample_field(data, pts, "rho", method="nearest")
        np.testing.assert_allclose(values, [0.5, 1.5, 2.5])

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
        data = FieldDataset.from_arrays({"rho": field}, grid)
        pts = np.array([[0.5, 0.5, 100.0]])  # z way outside the single node
        values = sample_field(data, pts, "rho", method="nearest")
        assert np.all(np.isnan(values))


# --- Tracing algorithm tests ---


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
        {"B1": ones, "B2": zeros, "B3": zeros},
        grid,
        Normalization.identity(),
    )


class TestVectorFieldInterpolator:
    def test_from_dataset(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import VectorFieldInterpolator

        interp = VectorFieldInterpolator.from_dataset(uniform_field_data)
        assert interp is not None

    def test_call_inside_domain(self, uniform_field_data: FieldDataset) -> None:
        from pypic.traces import VectorFieldInterpolator

        interp = VectorFieldInterpolator.from_dataset(uniform_field_data)
        result = interp(np.array([10.0, 10.0, 10.0]))
        assert result.shape == (3,)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

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

        assert _field_name_from_components(("B1", "B2", "B3")) == "B"
        assert _field_name_from_components(("Ve1", "Ve2", "Ve3")) == "Ve"

    def test_mismatched_rejects(self) -> None:
        from pypic.traces._tracing import _field_name_from_components

        with pytest.raises(ValueError, match="same field"):
            _field_name_from_components(("B1", "E2", "B3"))
