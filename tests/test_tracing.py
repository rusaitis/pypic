"""Tests for the pypic.traces._tracing module: RK4 field line tracing."""

from __future__ import annotations

import numpy as np
import pytest

from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.traces import (
    FieldLine,
    VectorFieldInterpolator,
    estimate_tracing_error,
    trace_field_line,
    trace_field_line_adaptive,
)


def _make_uniform_field(
    n: int = 20,
    extent: float = 10.0,
    b_vec: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> FieldDataset:
    """Uniform B = b_vec on an n^3 grid spanning [0, extent]."""
    dx = extent / n
    grid = GridInfo(
        dimensions=(n, n, n),
        spacing=(dx, dx, dx),
        origin=(0.0, 0.0, 0.0),
    )
    shape = (n, n, n)
    return FieldDataset.from_arrays(
        {
            "B_1": np.full(shape, b_vec[0]),
            "B_2": np.full(shape, b_vec[1]),
            "B_3": np.full(shape, b_vec[2]),
        },
        grid,
    )


def _make_circular_field(
    n: int = 40,
    extent: float = 10.0,
) -> FieldDataset:
    """B = (-y, x, 0): circular field lines centered at domain center."""
    dx = extent / n
    grid = GridInfo(
        dimensions=(n, n, n),
        spacing=(dx, dx, dx),
        origin=(0.0, 0.0, 0.0),
    )
    center = extent / 2.0
    x1d = np.arange(n) * dx + 0.5 * dx
    xx, yy, _ = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    bx = -(yy - center)
    by = xx - center
    bz = np.zeros_like(bx)
    return FieldDataset.from_arrays(
        {"B_1": bx, "B_2": by, "B_3": bz},
        grid,
    )


def _make_helical_field(
    n: int = 40,
    extent: float = 10.0,
) -> FieldDataset:
    """B = (-y, x, 1): helical field lines."""
    dx = extent / n
    grid = GridInfo(
        dimensions=(n, n, n),
        spacing=(dx, dx, dx),
        origin=(0.0, 0.0, 0.0),
    )
    center = extent / 2.0
    x1d = np.arange(n) * dx + 0.5 * dx
    xx, yy, _ = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    bx = -(yy - center)
    by = xx - center
    bz = np.ones_like(bx)
    return FieldDataset.from_arrays(
        {"B_1": bx, "B_2": by, "B_3": bz},
        grid,
    )


def _make_null_field(
    n: int = 20,
    extent: float = 10.0,
) -> FieldDataset:
    """B pointing inward: null at domain center."""
    dx = extent / n
    grid = GridInfo(
        dimensions=(n, n, n),
        spacing=(dx, dx, dx),
        origin=(0.0, 0.0, 0.0),
    )
    center = extent / 2.0
    x1d = np.arange(n) * dx + 0.5 * dx
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    bx = -(xx - center)
    by = -(yy - center)
    bz = -(zz - center)
    return FieldDataset.from_arrays(
        {"B_1": bx, "B_2": by, "B_3": bz},
        grid,
    )


class TestVectorFieldInterpolator:
    def test_evaluate_at_grid_center(self) -> None:
        data = _make_uniform_field(b_vec=(2.0, 3.0, 4.0))
        interp = VectorFieldInterpolator.from_dataset(data)
        result = interp(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0], atol=1e-10)

    def test_outside_domain_gives_nan(self) -> None:
        data = _make_uniform_field(extent=10.0)
        interp = VectorFieldInterpolator.from_dataset(data)
        result = interp(np.array([-100.0, 5.0, 5.0]))
        assert np.all(np.isnan(result))

    def test_linear_interpolation_accuracy(self) -> None:
        """Linear field B = (x, 0, 0) interpolated exactly."""
        n = 20
        extent = 10.0
        dx = extent / n
        grid = GridInfo(
            dimensions=(n, n, n),
            spacing=(dx, dx, dx),
            origin=(0.0, 0.0, 0.0),
        )
        x1d = np.arange(n) * dx + 0.5 * dx
        xx, _, _ = np.meshgrid(x1d, x1d, x1d, indexing="ij")
        data = FieldDataset.from_arrays(
            {
                "B_1": xx,
                "B_2": np.zeros_like(xx),
                "B_3": np.zeros_like(xx),
            },
            grid,
        )
        interp = VectorFieldInterpolator.from_dataset(data)
        test_x = 3.7
        result = interp(np.array([test_x, 5.0, 5.0]))
        np.testing.assert_allclose(result[0], test_x, atol=1e-10)

    def test_stacked_matches_per_component(self) -> None:
        """Stacked VectorFieldInterpolator matches three independent interps.

        Regression for the refactor that collapses three per-component
        ``RegularGridInterpolator`` calls into one call over a stacked
        ``(..., 3)`` value array. Values at every sampled point must
        equal the per-component result to machine precision.
        """
        from scipy.interpolate import RegularGridInterpolator

        data = _make_helical_field()
        coord_arrays = data.grid.coordinate_arrays()
        solo = [
            RegularGridInterpolator(
                coord_arrays,
                np.asarray(data[c], dtype=np.float64),
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            for c in ("B_1", "B_2", "B_3")
        ]
        interp = VectorFieldInterpolator.from_dataset(data)

        rng = np.random.default_rng(7)
        interior_points = rng.uniform(1.0, 9.0, size=(30, 3))
        out_of_bounds = np.array([[-5.0, 5.0, 5.0], [5.0, 5.0, 100.0]])
        points = np.vstack([interior_points, out_of_bounds])

        for pt in points:
            expected = np.array([float(s(pt.reshape(1, 3))[0]) for s in solo])
            result = interp(pt)
            np.testing.assert_array_equal(result, expected)


class TestTraceFixedUniform:
    def test_straight_line_forward(self) -> None:
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))
        seed = (5.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=10,
            direction="forward",
        )
        assert isinstance(fl, FieldLine)
        assert fl.direction == "forward"
        np.testing.assert_allclose(fl.points[:, 1], 5.0, atol=1e-10)
        np.testing.assert_allclose(fl.points[:, 2], 5.0, atol=1e-10)
        assert np.all(np.diff(fl.points[:, 0]) > 0)

    def test_straight_line_backward(self) -> None:
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))
        fl = trace_field_line(
            data,
            (5.0, 5.0, 5.0),
            step_size=0.5,
            max_steps=10,
            direction="backward",
        )
        assert fl.direction == "backward"
        assert np.all(np.diff(fl.points[:, 0]) > 0)

    def test_straight_line_both(self) -> None:
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))
        seed = (5.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=10,
            direction="both",
        )
        assert fl.direction == "both"
        distances = np.linalg.norm(
            fl.points - np.array(seed),
            axis=1,
        )
        seed_idx = int(np.argmin(distances))
        assert 0 < seed_idx < fl.n_points - 1

    def test_seed_in_result(self) -> None:
        data = _make_uniform_field()
        seed = (5.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=5,
            direction="forward",
        )
        np.testing.assert_allclose(
            fl.points[0],
            list(seed),
            atol=1e-12,
        )

    def test_zero_transverse_error(self) -> None:
        data = _make_uniform_field(b_vec=(0.0, 1.0, 0.0))
        fl = trace_field_line(
            data,
            (5.0, 5.0, 5.0),
            step_size=0.2,
            max_steps=20,
            direction="forward",
        )
        np.testing.assert_allclose(fl.points[:, 0], 5.0, atol=1e-10)
        np.testing.assert_allclose(fl.points[:, 2], 5.0, atol=1e-10)

    def test_metadata_populated(self) -> None:
        data = _make_uniform_field()
        fl = trace_field_line(
            data,
            (5.0, 5.0, 5.0),
            step_size=0.3,
            max_steps=5,
        )
        assert fl.metadata["method"] == "rk4"
        assert fl.metadata["step_size"] == 0.3
        assert "n_steps" in fl.metadata
        assert "reason" in fl.metadata


class TestTraceFixedCircular:
    def test_endpoint_returns_near_seed(self) -> None:
        """After one full circle, trace returns close to seed."""
        data = _make_circular_field(n=60, extent=10.0)
        center = 5.0
        radius = 2.0
        seed = (center + radius, center, 5.0)
        circumference = 2 * np.pi * radius
        n_steps = int(circumference / 0.05) + 100
        fl = trace_field_line(
            data,
            seed,
            step_size=0.05,
            max_steps=n_steps,
            direction="forward",
        )
        distances = np.linalg.norm(
            fl.points[1:] - np.array(seed),
            axis=1,
        )
        min_return_dist = float(np.min(distances))
        assert min_return_dist < 0.1

    def test_radius_preserved(self) -> None:
        """Radius from center stays constant along the trace."""
        data = _make_circular_field(n=60, extent=10.0)
        center = 5.0
        radius = 2.0
        seed = (center + radius, center, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.05,
            max_steps=200,
            direction="forward",
        )
        radii = np.sqrt(
            (fl.points[:, 0] - center) ** 2 + (fl.points[:, 1] - center) ** 2,
        )
        np.testing.assert_allclose(radii, radius, rtol=0.02)

    def test_fourth_order_convergence(self) -> None:
        """Error ratio when halving h should be ~16 (4th order).

        Uses n=200 grid so interpolation error is negligible.
        """
        data = _make_circular_field(n=200, extent=10.0)
        center = 5.0
        radius = 2.0
        seed = (center + radius, center, 5.0)
        interp = VectorFieldInterpolator.from_dataset(data)
        n_steps = 20

        errors = []
        for h in [0.4, 0.2, 0.1]:
            fl = trace_field_line(
                data,
                seed,
                step_size=h,
                max_steps=n_steps,
                direction="forward",
                interpolator=interp,
            )
            arc = n_steps * h
            theta = arc / radius
            exact = np.array(
                [
                    center + radius * np.cos(theta),
                    center + radius * np.sin(theta),
                    5.0,
                ]
            )
            errors.append(
                float(np.linalg.norm(fl.points[-1] - exact)),
            )

        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 10, f"Ratio {ratio_1:.1f} too low"
        assert ratio_2 > 10, f"Ratio {ratio_2:.1f} too low"


class TestTraceAdaptive:
    def test_adaptive_closes_circle(self) -> None:
        """Adaptive tracer closes the circle within tolerance.

        Disables the auto closed-loop detector so the trace can sweep a
        full orbit and be checked for proximity to its own seed; the
        detector's correctness is exercised by ``TestClosedLoopDetection``
        in ``test_traces.py``.
        """
        data = _make_circular_field(n=60, extent=10.0)
        center = 5.0
        radius = 2.0
        seed = (center + radius, center, 5.0)
        fl = trace_field_line_adaptive(
            data,
            seed,
            atol=1e-6,
            rtol=1e-4,
            max_steps=5000,
            direction="forward",
            loop_tol=None,
        )
        distances = np.linalg.norm(
            fl.points[1:] - np.array(seed),
            axis=1,
        )
        assert float(np.min(distances)) < 0.1

    def test_max_local_error_in_metadata(self) -> None:
        data = _make_uniform_field()
        fl = trace_field_line_adaptive(
            data,
            (5.0, 5.0, 5.0),
            max_steps=20,
            direction="forward",
        )
        assert "max_local_error" in fl.metadata
        assert fl.metadata["method"] == "rk45_dopri"

    def test_adaptive_fewer_points_in_uniform(self) -> None:
        """Adaptive uses fewer points in a smooth field."""
        data = _make_uniform_field(n=30, extent=10.0)
        seed = (5.0, 5.0, 5.0)
        fl_fixed = trace_field_line(
            data,
            seed,
            step_size=0.1,
            max_steps=20,
            direction="forward",
        )
        fl_adaptive = trace_field_line_adaptive(
            data,
            seed,
            step_size_init=0.1,
            max_step=2.0,
            max_steps=20,
            direction="forward",
        )
        assert fl_adaptive.n_points <= fl_fixed.n_points


class TestTermination:
    def test_domain_exit(self) -> None:
        data = _make_uniform_field(
            n=20,
            extent=10.0,
            b_vec=(1.0, 0.0, 0.0),
        )
        fl = trace_field_line(
            data,
            (9.0, 5.0, 5.0),
            step_size=0.5,
            max_steps=1000,
            direction="forward",
        )
        assert fl.metadata["reason"] == "domain_exit"

    def test_null_point(self) -> None:
        data = _make_null_field(n=20, extent=10.0)
        fl = trace_field_line(
            data,
            (5.5, 5.5, 5.5),
            step_size=0.1,
            max_steps=1000,
            direction="forward",
            null_threshold=0.1,
        )
        assert fl.metadata["reason"] == "null_point"

    def test_callback(self) -> None:
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))

        def stop_at_x_7(r: np.ndarray) -> bool:
            return bool(r[0] > 7.0)

        fl = trace_field_line(
            data,
            (5.0, 5.0, 5.0),
            step_size=0.5,
            max_steps=1000,
            direction="forward",
            terminate=stop_at_x_7,
        )
        assert fl.metadata["reason"] == "callback"
        assert fl.points[-1, 0] > 7.0

    def test_max_steps(self) -> None:
        data = _make_uniform_field()
        fl = trace_field_line(
            data,
            (5.0, 5.0, 5.0),
            step_size=0.1,
            max_steps=5,
            direction="forward",
        )
        assert fl.metadata["reason"] == "max_steps"
        assert fl.n_points == 6  # seed + 5 steps


class TestEstimateTracingError:
    def test_uniform_field_near_zero(self) -> None:
        """In a uniform field, RK4 is exact."""
        data = _make_uniform_field(n=30, extent=15.0)
        fl = trace_field_line(
            data,
            (7.0, 7.0, 7.0),
            step_size=0.5,
            max_steps=5,
            direction="forward",
        )
        err = estimate_tracing_error(fl, data)
        np.testing.assert_allclose(err, 0.0, atol=1e-10)

    def test_circular_error_positive(self) -> None:
        """In a curved field, the tracing error is nonzero."""
        data = _make_circular_field(n=60, extent=10.0)
        seed = (7.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.2,
            max_steps=20,
            direction="forward",
        )
        err = estimate_tracing_error(fl, data)
        assert err > 0


class TestDirectionHandling:
    def test_both_contains_seed(self) -> None:
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))
        seed = (5.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=5,
            direction="both",
        )
        distances = np.linalg.norm(
            fl.points - np.array(seed),
            axis=1,
        )
        assert float(np.min(distances)) < 1e-12

    def test_backward_ends_at_seed(self) -> None:
        """Backward trace reversed: last point is the seed."""
        data = _make_uniform_field(b_vec=(1.0, 0.0, 0.0))
        seed = (5.0, 5.0, 5.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=5,
            direction="backward",
        )
        np.testing.assert_allclose(
            fl.points[-1],
            list(seed),
            atol=1e-12,
        )

    def test_invalid_direction_raises(self) -> None:
        data = _make_uniform_field()
        with pytest.raises(ValueError, match="direction"):
            trace_field_line(
                data,
                (5.0, 5.0, 5.0),
                direction="sideways",
            )


class TestInterpolatorReuse:
    def test_prebuilt_matches_auto(self) -> None:
        data = _make_uniform_field(b_vec=(0.0, 1.0, 0.0))
        seed = (5.0, 5.0, 5.0)
        interp = VectorFieldInterpolator.from_dataset(data)
        fl_auto = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=10,
            direction="forward",
        )
        fl_pre = trace_field_line(
            data,
            seed,
            step_size=0.5,
            max_steps=10,
            direction="forward",
            interpolator=interp,
        )
        np.testing.assert_array_equal(
            fl_auto.points,
            fl_pre.points,
        )


class TestSeedValidation:
    def test_outside_domain_raises(self) -> None:
        data = _make_uniform_field(extent=10.0)
        with pytest.raises(ValueError, match="outside"):
            trace_field_line(data, (-100.0, 5.0, 5.0))

    def test_null_point_seed_raises(self) -> None:
        data = _make_null_field()
        with pytest.raises(ValueError, match="null"):
            trace_field_line(
                data,
                (5.0, 5.0, 5.0),
                null_threshold=1.0,
            )


class TestHelicalField:
    def test_z_advances(self) -> None:
        """In helical B=(-y,x,1), z increases along the trace."""
        data = _make_helical_field(n=60, extent=10.0)
        seed = (7.0, 5.0, 3.0)
        fl = trace_field_line(
            data,
            seed,
            step_size=0.05,
            max_steps=200,
            direction="forward",
        )
        assert fl.points[-1, 2] > fl.points[0, 2]
