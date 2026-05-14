"""Tests for pypic.reconnection — X-point detection, reconnection rate,
Schindler $\\Xi$ integral."""

from __future__ import annotations

import numpy as np
import pytest

from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.reconnection import find_saddle_points, reconnection_rate, schindler_xi
from pypic.units import Normalization


class TestFindSaddlePoints:
    def test_hyperbolic_saddle(self) -> None:
        """psi = x^2 - y^2 has a single saddle at the origin."""
        x = np.linspace(-5, 5, 101)
        dx = x[1] - x[0]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = xx**2 - yy**2
        saddles = find_saddle_points(psi, dx, dx)
        assert len(saddles) >= 1
        si, sj = saddles[0]
        assert abs(si - 50) <= 2
        assert abs(sj - 50) <= 2

    def test_no_saddle_in_parabola(self) -> None:
        """psi = x^2 + y^2 has no saddle points (all positive definite)."""
        x = np.linspace(-5, 5, 51)
        dx = x[1] - x[0]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = xx**2 + yy**2
        saddles = find_saddle_points(psi, dx, dx)
        assert len(saddles) == 0

    def test_multiple_saddles(self) -> None:
        """psi = sin(x)*sin(y) on [0, 4pi] has multiple interior saddles."""
        x = np.linspace(0, 4 * np.pi, 257)
        dx = x[1] - x[0]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = np.sin(xx) * np.sin(yy)
        saddles = find_saddle_points(psi, dx, dx, min_separation=10)
        assert len(saddles) >= 2

    def test_invalid_shapes_all_raise(self) -> None:
        """Every non-2D input must raise — guards against a truthy ndim check.

        Aggregated rather than parametrized: the structural invariant is
        ``find_saddle_points`` rejects any ``ndim != 2``, and the failure
        list reports every shape that slipped through.
        """
        bad_shapes: list[tuple[int, ...]] = [(10,), (4, 4, 4), (2, 3, 4, 5)]
        failures: list[str] = []
        for shape in bad_shapes:
            try:
                find_saddle_points(np.ones(shape), 1.0, 1.0)
            except ValueError as e:
                if "2D" not in str(e):
                    failures.append(
                        f"shape={shape}: ValueError without '2D' in message ({e!r})"
                    )
                continue
            except Exception as e:
                failures.append(
                    f"shape={shape}: raised {type(e).__name__} (expected ValueError)"
                )
                continue
            failures.append(f"shape={shape}: no exception raised")
        assert not failures, "Non-2D shape acceptance:\n  - " + "\n  - ".join(failures)

    def test_too_small_grids_all_raise(self) -> None:
        """Both axes must satisfy ``nx>=5`` AND ``ny>=5``; one-sided failure
        is enough.

        The 2-cell margin guard inside ``find_saddle_points`` requires
        ``nx>=5`` and ``ny>=5`` independently. A regression to
        ``nx<5 and ny<5`` (AND instead of OR) would let 4x10 slip
        through and emit garbage from the one-sided stencil. Aggregated
        invariant — every offending shape lists in a single failure.
        """
        small_shapes: list[tuple[int, int]] = [
            (3, 3),  # square_3
            (4, 5),  # nx<5
            (5, 4),  # ny<5
            (2, 10),  # nx_tiny
            (10, 2),  # ny_tiny
        ]
        failures: list[str] = []
        for shape in small_shapes:
            try:
                find_saddle_points(np.ones(shape), 1.0, 1.0)
            except ValueError as e:
                if "too small" not in str(e):
                    failures.append(
                        f"shape={shape}: ValueError without 'too small' ({e!r})"
                    )
                continue
            except Exception as e:
                failures.append(
                    f"shape={shape}: raised {type(e).__name__} (expected ValueError)"
                )
                continue
            failures.append(f"shape={shape}: no exception raised")
        assert not failures, "Too-small shape acceptance:\n  - " + "\n  - ".join(
            failures
        )


class TestReconnectionRate:
    @pytest.mark.parametrize(
        ("psi_val", "psi_prev_val", "dt", "x_point", "expected"),
        [
            # Negative rate: psi decreasing
            (0.0, 1.0, 0.5, (5, 5), -2.0),
            # Positive rate: psi increasing
            (3.0, 1.0, 1.0, (3, 3), 2.0),
            # dt scales the rate linearly (guards against missing division)
            (2.0, 0.0, 4.0, (5, 5), 0.5),
            # X-point not at the grid centre — uses correct indices
            (7.0, 3.0, 2.0, (0, 9), 2.0),
        ],
        ids=["negative", "positive", "dt_scaling", "off_center_xpoint"],
    )
    def test_finite_difference(
        self,
        psi_val: float,
        psi_prev_val: float,
        dt: float,
        x_point: tuple[int, int],
        expected: float,
    ) -> None:
        """Pins sign, magnitude, dt division, and x_point indexing."""
        psi = np.full((10, 10), psi_val)
        psi_prev = np.full((10, 10), psi_prev_val)
        rate = reconnection_rate(psi, psi_prev, dt=dt, x_point=x_point)
        np.testing.assert_allclose(rate, expected)


class TestReconnectionEdgeCases:
    """NaN handling and X-point near margin."""

    def test_nan_in_psi_no_crash(self) -> None:
        """A NaN in psi must not crash detection.

        Saddles in NaN regions are either skipped or NaN-tagged; the
        function must remain callable.
        """
        x = np.linspace(-5, 5, 51)
        dx = x[1] - x[0]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = xx**2 - yy**2
        psi[10, 10] = np.nan  # spike well away from the saddle at (25, 25)
        # Should not raise; the saddle at the centre is still findable
        saddles = find_saddle_points(psi, dx, dx)
        assert isinstance(saddles, list)

    def test_xpoint_just_inside_margin(self) -> None:
        """A saddle near (but inside) the margin still gets reported.

        With ``meshgrid(..., indexing='ij')`` the i-index of the returned
        saddle tracks ``x`` and the j-index tracks ``y`` (see the existing
        ``test_hyperbolic_saddle``). The shifted saddle below sits at
        ``(x=3.5, y=0)``, i.e. ``(i=34, j=20)`` on a 41x41 grid spanning
        ``[-5, 5]`` — only 6 cells from the upper-i edge but still inside
        the 2-cell margin guard.
        """
        x = np.linspace(-5, 5, 41)
        dx = x[1] - x[0]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = (xx - 3.5) ** 2 - yy**2
        saddles = find_saddle_points(psi, dx, dx)
        assert len(saddles) >= 1
        rows_i = [r for r, _ in saddles]  # x-axis index
        cols_j = [c for _, c in saddles]  # y-axis index
        assert any(abs(r - 34) <= 2 for r in rows_i)
        assert any(abs(c - 20) <= 2 for c in cols_j)

    def test_reconnection_rate_preserves_nan(self) -> None:
        """If psi has NaN at the X-point, the rate is NaN (not silently 0)."""
        psi = np.zeros((10, 10))
        psi_prev = np.zeros((10, 10))
        psi[5, 5] = np.nan
        rate = reconnection_rate(psi, psi_prev, dt=1.0, x_point=(5, 5))
        assert np.isnan(rate)


def _uniform_field_dataset(
    *,
    shape: tuple[int, int, int] = (32, 8, 8),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    b: tuple[float, float, float] = (1.0, 0.0, 0.0),
    e: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FieldDataset:
    """Build a domain with uniform B and E for Schindler integral tests."""
    return FieldDataset.from_arrays(
        {
            "B_1": np.full(shape, b[0]),
            "B_2": np.full(shape, b[1]),
            "B_3": np.full(shape, b[2]),
            "E_1": np.full(shape, e[0]),
            "E_2": np.full(shape, e[1]),
            "E_3": np.full(shape, e[2]),
        },
        GridInfo(dimensions=shape, spacing=spacing),
        Normalization.identity(),
    )


def _uniform_b_cos_e_dataset(
    *, n: int, domain: float, k: float, e0: float = 1.0
) -> FieldDataset:
    r"""Cubic [0, L]^3 with B = ẑ and E_z = E_0 cos(k z).

    Used by the Schindler-$\Xi$ convergence tests.  RK4 is exact on a
    constant $\hat{b}$, so the only error sources are trapezoid arc-
    length integration ($O(\Delta s^2)$) and trilinear field sampling
    ($O(\Delta x^2)$).
    """
    spacing = (domain / n, domain / n, domain / n)
    shape = (n, n, n)
    grid = GridInfo(dimensions=shape, spacing=spacing)
    _, _, zz = (
        a.astype(np.float64)
        for a in np.meshgrid(*grid.coordinate_arrays(), indexing="ij")
    )
    return FieldDataset.from_arrays(
        {
            "B_1": np.zeros(shape),
            "B_2": np.zeros(shape),
            "B_3": np.ones(shape),
            "E_1": np.zeros(shape),
            "E_2": np.zeros(shape),
            "E_3": e0 * np.cos(k * zz),
        },
        grid,
        Normalization.identity(),
    )


class TestSchindlerXi:
    """Schindler 1988 $\\Xi = \\int E_\\parallel \\, d\\ell$ along $\\mathbf{B}$."""

    def test_zero_when_e_perp_to_b(self) -> None:
        """E ⟂ B everywhere → E_par = 0 → Xi = 0 along any trace."""
        data = _uniform_field_dataset(b=(1.0, 0.0, 0.0), e=(0.0, 0.5, 0.0))
        seeds = np.array([[16.0, 4.0, 4.0]])
        xi = schindler_xi(data, seeds, step_size=0.5, max_steps=200)
        assert xi.shape == (1,)
        np.testing.assert_allclose(xi, 0.0, atol=1e-13)

    def test_uniform_parallel_e_field(self) -> None:
        """Uniform E ∥ B → Xi = E0 · (traced arc length).

        With direction='both' the trace spans the domain symmetrically;
        for a uniform unit B-field along x at the domain center, the
        traced length is just under 2 * (domain_half - seed_offset).
        Check exact agreement against the computed arc length rather
        than predicting it from max_steps.
        """
        from pypic.traces import trace_field_line

        e0 = 0.07
        data = _uniform_field_dataset(
            shape=(32, 8, 8), b=(1.0, 0.0, 0.0), e=(e0, 0.0, 0.0)
        )
        seed = (15.5, 3.5, 3.5)
        # First compute the actual traced arc length so the expected
        # answer matches whatever the integrator delivers (avoids
        # over-fitting to one set of max_steps / step_size knobs).
        fl = trace_field_line(
            data, seed, step_size=0.5, max_steps=200, direction="both"
        )
        deltas = np.diff(fl.points, axis=0)
        traced_length = float(np.sum(np.sqrt(np.sum(deltas**2, axis=1))))
        xi = schindler_xi(data, np.array([seed]), step_size=0.5, max_steps=200)
        np.testing.assert_allclose(xi[0], e0 * traced_length, rtol=1e-10)

    def test_out_of_domain_seed_returns_nan(self) -> None:
        """A seed outside the grid yields NaN, not a wrong value."""
        data = _uniform_field_dataset()
        seeds = np.array([[16.0, 4.0, 4.0], [1000.0, 0.0, 0.0]])
        xi = schindler_xi(data, seeds, step_size=0.5, max_steps=64)
        assert np.isfinite(xi[0])
        assert np.isnan(xi[1])

    def test_single_seed_shape(self) -> None:
        """A 1D seed (3,) returns a length-1 array, matching the doc contract."""
        data = _uniform_field_dataset()
        xi = schindler_xi(data, np.array([16.0, 4.0, 4.0]), step_size=0.5, max_steps=64)
        assert xi.shape == (1,)

    def test_invalid_seed_shape_raises(self) -> None:
        """Seed array with wrong shape raises ValueError."""
        data = _uniform_field_dataset()
        with pytest.raises(ValueError, match="seeds must have shape"):
            schindler_xi(data, np.array([[1.0, 2.0]]))  # (1, 2), not (1, 3)

    def test_curved_field_line(self) -> None:
        r"""Curved B with E ∥ B everywhere → $\Xi = E_0 \cdot \ell_{\rm traced}$.

        Builds a rotational field $\mathbf{B} = (-y, x, \varepsilon)$
        whose field lines are helical (circular in (x, y) with a slow
        axial drift) and sets $\mathbf{E} = E_0\,\hat{\mathbf{B}}$, so
        $E_\parallel = E_0$ at every point.  The existing
        ``test_uniform_parallel_e_field`` only exercises straight field
        lines — this test stresses the trapezoidal arc-length sum on
        non-collinear ``deltas``, where ``d\ell = |\Delta\mathbf{r}|``
        is strictly less than the sum of component magnitudes.
        """
        from pypic.traces import trace_field_line

        e0 = 0.1
        epsilon = 0.05
        nx, ny, nz = 64, 64, 16
        dx = 0.25
        origin = (-(nx * dx) / 2.0, -(ny * dx) / 2.0, -(nz * dx) / 2.0)

        grid = GridInfo(
            dimensions=(nx, ny, nz),
            spacing=(dx, dx, dx),
            origin=origin,
        )
        coord_arrays = grid.coordinate_arrays()
        xx, yy, _ = (
            a.astype(np.float64) for a in np.meshgrid(*coord_arrays, indexing="ij")
        )
        b1 = -yy
        b2 = xx
        b3 = np.full_like(b1, epsilon)
        b_mag = np.sqrt(b1**2 + b2**2 + b3**2)

        data = FieldDataset.from_arrays(
            {
                "B_1": b1,
                "B_2": b2,
                "B_3": b3,
                "E_1": e0 * b1 / b_mag,
                "E_2": e0 * b2 / b_mag,
                "E_3": e0 * b3 / b_mag,
            },
            grid,
            Normalization.identity(),
        )

        seed = (2.0, 0.0, 0.0)
        fl = trace_field_line(
            data, seed, step_size=0.1, max_steps=400, direction="forward"
        )
        deltas = np.diff(fl.points, axis=0)
        traced_length = float(np.sum(np.sqrt(np.sum(deltas**2, axis=1))))
        # Sanity check: a straight-line seed-to-end distance underestimates
        # the curved arc length, so the trace really is curved.
        chord = float(np.linalg.norm(fl.points[-1] - fl.points[0]))
        assert traced_length > 1.1 * chord

        xi = schindler_xi(
            data,
            np.array([seed]),
            step_size=0.1,
            max_steps=400,
            direction="forward",
        )
        # Tolerance set by trilinear interpolation error on the rotational
        # field — at grid points E∥ = E0 exactly, but at trace points
        # |B_interp| ≠ |B_grid|, so the integrand drifts by O((dx/r)²).
        # Observed deviation ≈ 0.13 % on this configuration; rtol=5e-3
        # gives ~4x margin while still pinning the proportionality.
        np.testing.assert_allclose(xi[0], e0 * traced_length, rtol=5e-3)

    def test_xi_converges_second_order_joint(self) -> None:
        r"""$\Xi$ converges as $O(h^2)$ when grid and step_size halve together.

        Uniform $\mathbf{B} = \hat{z}$ and $E_z = \cos(k z)$ on $[0, L]^3$.
        RK4 is exact on constant $\hat{b}$, so the only error sources are
        trapezoid arc-length integration ($O(\Delta s^2)$) and trilinear
        field sampling ($O(\Delta x^2)$).  Seed is placed at a cell center
        with $\Delta s = 2\,\Delta x$, so every trace point lands on a
        grid node and trilinear interpolation is exact — leaving only the
        trapezoid contribution.  Halving both knobs halves the error by
        $\approx 4\times$; we pin ratio > 3.9 to catch first-order
        regressions while tolerating pre-asymptotic sag.
        """
        domain = 8.0
        k = np.pi / 4.0
        trace_length = 7.0
        seed_xy = domain / 2.0

        errors: list[float] = []
        for n, step in [(32, 0.5), (64, 0.25), (128, 0.125)]:
            dx = domain / n
            z0 = 1.5 * dx  # second cell center — safely inside the interp domain
            z_end = z0 + trace_length
            xi_exact = (np.sin(k * z_end) - np.sin(k * z0)) / k
            data = _uniform_b_cos_e_dataset(n=n, domain=domain, k=k)
            max_steps = round(trace_length / step)
            xi = schindler_xi(
                data,
                np.array([[seed_xy, seed_xy, z0]]),
                step_size=step,
                max_steps=max_steps,
                direction="forward",
            )
            errors.append(float(abs(xi[0] - xi_exact)))

        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.9, (
            f"coarse→medium ratio = {ratio_1:.3f}, errors = {errors!r}"
        )
        assert ratio_2 > 3.9, f"medium→fine ratio = {ratio_2:.3f}, errors = {errors!r}"

    def test_xi_trapezoid_converges_second_order(self) -> None:
        r"""$\Xi$ converges as $O(\Delta s^2)$ with step_size halved alone.

        Grid pinned at $N = 128$.  Seed at a cell center with every
        $\Delta s$ an even multiple of $\Delta x$, so trace points land
        on grid nodes (trilinear interpolation exact).  Pure trapezoid
        regime; ratio > 3.9 catches a drop to first order.
        """
        domain = 8.0
        k = np.pi / 4.0
        n = 128
        dx = domain / n
        z0 = 1.5 * dx
        trace_length = 7.0
        z_end = z0 + trace_length
        xi_exact = (np.sin(k * z_end) - np.sin(k * z0)) / k
        seed_xy = domain / 2.0
        seeds = np.array([[seed_xy, seed_xy, z0]])
        data = _uniform_b_cos_e_dataset(n=n, domain=domain, k=k)

        errors: list[float] = []
        for step in [0.5, 0.25, 0.125]:
            max_steps = round(trace_length / step)
            xi = schindler_xi(
                data,
                seeds,
                step_size=step,
                max_steps=max_steps,
                direction="forward",
            )
            errors.append(float(abs(xi[0] - xi_exact)))

        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.9, (
            f"coarse→medium ratio = {ratio_1:.3f}, errors = {errors!r}"
        )
        assert ratio_2 > 3.9, f"medium→fine ratio = {ratio_2:.3f}, errors = {errors!r}"
