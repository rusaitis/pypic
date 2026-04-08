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

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            find_saddle_points(np.ones(10), 1.0, 1.0)

    def test_grid_too_small(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            find_saddle_points(np.ones((3, 3)), 1.0, 1.0)


class TestReconnectionRate:
    def test_basic(self) -> None:
        psi = np.zeros((10, 10))
        psi_prev = np.ones((10, 10))
        rate = reconnection_rate(psi, psi_prev, dt=0.5, x_point=(5, 5))
        np.testing.assert_allclose(rate, -2.0)

    def test_positive_rate(self) -> None:
        psi = np.full((10, 10), 3.0)
        psi_prev = np.full((10, 10), 1.0)
        rate = reconnection_rate(psi, psi_prev, dt=1.0, x_point=(3, 3))
        np.testing.assert_allclose(rate, 2.0)


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
