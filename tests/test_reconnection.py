"""Tests for pypic.reconnection — X-point detection and reconnection rate."""

from __future__ import annotations

import numpy as np
import pytest

from pypic.reconnection import find_saddle_points, reconnection_rate


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

    @pytest.mark.parametrize(
        "bad_shape",
        [(10,), (4, 4, 4), (2, 3, 4, 5)],
        ids=["1d", "3d", "4d"],
    )
    def test_invalid_shape(self, bad_shape: tuple[int, ...]) -> None:
        """Any non-2D input must raise — guards against a truthy ndim check."""
        with pytest.raises(ValueError, match="2D"):
            find_saddle_points(np.ones(bad_shape), 1.0, 1.0)

    @pytest.mark.parametrize(
        "shape",
        [(3, 3), (4, 5), (5, 4), (2, 10), (10, 2)],
        ids=["square_3", "nx_lt_5", "ny_lt_5", "nx_tiny", "ny_tiny"],
    )
    def test_grid_too_small(self, shape: tuple[int, int]) -> None:
        """Both axes must satisfy nx>=5 AND ny>=5; one-sided failure is enough.

        The 2-cell margin guard inside find_saddle_points requires
        ``nx>=5`` and ``ny>=5`` independently. A regression to
        ``nx<5 and ny<5`` (AND instead of OR) would let 4x10 slip
        through and emit garbage from the one-sided stencil.
        """
        with pytest.raises(ValueError, match="too small"):
            find_saddle_points(np.ones(shape), 1.0, 1.0)


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
