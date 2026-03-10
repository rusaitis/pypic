"""Tests for coordinate geometry definitions and metric scale factors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.coordinates import (
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    GeometryType,
)


class TestGeometryType:
    def test_values(self):
        assert GeometryType.CARTESIAN == "cartesian"
        assert GeometryType.SPHERICAL == "spherical"
        assert GeometryType.CYLINDRICAL == "cylindrical"

    def test_from_string(self):
        assert GeometryType("cartesian") is GeometryType.CARTESIAN
        assert GeometryType("spherical") is GeometryType.SPHERICAL
        assert GeometryType("cylindrical") is GeometryType.CYLINDRICAL


class TestAxisNames:
    def test_cartesian(self):
        assert CARTESIAN.axis_names == ("x", "y", "z")
        assert CARTESIAN.axis_units == ("length", "length", "length")

    def test_spherical(self):
        assert SPHERICAL.axis_names == ("r", "θ", "φ")
        assert SPHERICAL.axis_units == ("length", "angle", "angle")

    def test_cylindrical(self):
        assert CYLINDRICAL.axis_names == ("r", "φ", "z")
        assert CYLINDRICAL.axis_units == ("length", "angle", "length")


class TestCartesianMetricFactors:
    def test_scalar(self):
        h1, h2, h3 = CARTESIAN.metric_factors(1.0, 2.0, 3.0)
        assert h1 == 1.0
        assert h2 == 1.0
        assert h3 == 1.0

    def test_array(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 5, 5)
        z = np.linspace(-1, 1, 5)
        h1, h2, h3 = CARTESIAN.metric_factors(x, y, z)
        assert_allclose(np.broadcast_to(h1, x.shape), np.ones(5))
        assert_allclose(np.broadcast_to(h2, x.shape), np.ones(5))
        assert_allclose(np.broadcast_to(h3, x.shape), np.ones(5))


class TestSphericalMetricFactors:
    def test_unit_sphere_equator(self):
        """r=1, θ=π/2 → h1=1, h2=1, h3=1."""
        h1, h2, h3 = SPHERICAL.metric_factors(1.0, np.pi / 2, 0.0)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, 1.0)
        assert_allclose(h3, 1.0)

    def test_pole(self):
        """θ=0 → h3=r*sin(0)=0."""
        h1, h2, h3 = SPHERICAL.metric_factors(5.0, 0.0, 0.0)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, 5.0)
        assert_allclose(h3, 0.0, atol=1e-15)

    def test_general_point(self):
        """r=2, θ=π/6 → h2=2, h3=2*sin(π/6)=1."""
        h1, h2, h3 = SPHERICAL.metric_factors(2.0, np.pi / 6, 0.0)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, 2.0)
        assert_allclose(h3, 1.0, atol=1e-15)

    def test_array(self):
        r = np.array([1.0, 2.0, 3.0])
        theta = np.full(3, np.pi / 2)
        phi = np.zeros(3)
        h1, h2, h3 = SPHERICAL.metric_factors(r, theta, phi)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, r)
        assert_allclose(h3, r, atol=1e-15)


class TestCylindricalMetricFactors:
    def test_scalar(self):
        h1, h2, h3 = CYLINDRICAL.metric_factors(3.0, 0.0, 1.0)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, 3.0)
        assert_allclose(h3, 1.0)

    def test_array(self):
        r = np.array([1.0, 2.0, 5.0])
        phi = np.zeros(3)
        z = np.ones(3)
        h1, h2, h3 = CYLINDRICAL.metric_factors(r, phi, z)
        assert_allclose(h1, 1.0)
        assert_allclose(h2, r)
        assert_allclose(np.broadcast_to(h3, r.shape), np.ones(3))


class TestBroadcasting:
    """metric_factors works with both scalars and arrays, and broadcasts."""

    def test_scalar_input(self):
        _h1, h2, _h3 = SPHERICAL.metric_factors(2.0, np.pi / 4, 0.0)
        assert isinstance(h2, np.floating | np.ndarray)

    def test_2d_grid(self):
        r = np.array([[1.0, 2.0], [3.0, 4.0]])
        theta = np.full((2, 2), np.pi / 2)
        phi = np.zeros((2, 2))
        _h1, h2, h3 = SPHERICAL.metric_factors(r, theta, phi)
        assert_allclose(h2, r)
        assert_allclose(h3, r, atol=1e-15)
        assert h2.shape == (2, 2)


class TestFrozenDataclass:
    def test_immutable(self):
        with pytest.raises(AttributeError):
            CARTESIAN.type = GeometryType.SPHERICAL  # type: ignore[misc]
