import numpy as np
import pytest

from pypic.coordinates.geometry import GeometryType
from pypic.coordinates.operators import curl, divergence, gradient


class TestDivergenceCartesian:
    def test_uniform_field_zero_divergence(self):
        """Constant vector field has zero divergence."""
        shape = (8, 8, 8)
        f1 = 3.0 * np.ones(shape)
        f2 = -1.0 * np.ones(shape)
        f3 = 2.0 * np.ones(shape)
        result = divergence(f1, f2, f3, 0.1, 0.1, 0.1)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_linear_field_exact(self):
        r"""F = (x, y, z) has $\nabla \cdot F = 3$, exact for central diffs."""
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        f1, f2, f3 = np.meshgrid(x, y, z, indexing="ij")
        result = divergence(f1, f2, f3, dx, dy, dz)
        np.testing.assert_allclose(result, 3.0, rtol=1e-14)

    def test_div_curl_is_zero(self):
        r"""Vector identity: $\nabla \cdot (\nabla \times \mathbf{A}) = 0$."""
        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 0.1
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        a1 = np.sin(yy)
        a2 = np.sin(zz)
        a3 = np.sin(xx)

        c1, c2, c3 = curl(a1, a2, a3, dx, dy, dz)
        result = divergence(c1, c2, c3, dx, dy, dz)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_nan_propagation(self):
        shape = (4, 4, 4)
        f1 = np.ones(shape)
        f1[2, 2, 2] = np.nan
        f2 = np.zeros(shape)
        f3 = np.zeros(shape)
        result = divergence(f1, f2, f3, 1.0, 1.0, 1.0)
        assert np.any(np.isnan(result))


class TestCurlCartesian:
    def test_uniform_field_zero_curl(self):
        """Constant vector field has zero curl."""
        shape = (8, 8, 8)
        f1 = 5.0 * np.ones(shape)
        f2 = -2.0 * np.ones(shape)
        f3 = 7.0 * np.ones(shape)
        c1, c2, c3 = curl(f1, f2, f3, 0.1, 0.1, 0.1)
        np.testing.assert_allclose(c1, 0.0, atol=1e-14)
        np.testing.assert_allclose(c2, 0.0, atol=1e-14)
        np.testing.assert_allclose(c3, 0.0, atol=1e-14)

    def test_rigid_rotation(self):
        r"""F = (-y, x, 0) has $\nabla \times F = (0, 0, 2)$, exact for linear."""
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, _zz = np.meshgrid(x, y, z, indexing="ij")

        f1 = -yy
        f2 = xx
        f3 = np.zeros_like(xx)

        c1, c2, c3 = curl(f1, f2, f3, dx, dy, dz)
        np.testing.assert_allclose(c1, 0.0, atol=1e-14)
        np.testing.assert_allclose(c2, 0.0, atol=1e-14)
        np.testing.assert_allclose(c3, 2.0, rtol=1e-14)

    def test_curl_grad_is_zero(self):
        r"""Vector identity: $\nabla \times (\nabla f) = 0$."""
        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 0.1
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        f = np.sin(xx) * np.cos(yy) * np.exp(-zz * 0.1)
        g1, g2, g3 = gradient(f, dx, dy, dz)
        c1, c2, c3 = curl(g1, g2, g3, dx, dy, dz)

        np.testing.assert_allclose(c1, 0.0, atol=1e-10)
        np.testing.assert_allclose(c2, 0.0, atol=1e-10)
        np.testing.assert_allclose(c3, 0.0, atol=1e-10)

    def test_nan_propagation(self):
        shape = (4, 4, 4)
        f1 = np.ones(shape)
        f1[2, 2, 2] = np.nan
        f2 = np.zeros(shape)
        f3 = np.zeros(shape)
        _c1, c2, c3 = curl(f1, f2, f3, 1.0, 1.0, 1.0)
        assert np.any(np.isnan(c2)) or np.any(np.isnan(c3))


class TestGradientCartesian:
    def test_constant_field_zero_gradient(self):
        """Constant scalar field has zero gradient."""
        shape = (8, 8, 8)
        f = 42.0 * np.ones(shape)
        g1, g2, g3 = gradient(f, 0.1, 0.1, 0.1)
        np.testing.assert_allclose(g1, 0.0, atol=1e-14)
        np.testing.assert_allclose(g2, 0.0, atol=1e-14)
        np.testing.assert_allclose(g3, 0.0, atol=1e-14)

    def test_linear_field_exact(self):
        r"""f = 2x + 3y + 5z has $\nabla f = (2, 3, 5)$, exact for central diffs."""
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        f = 2.0 * xx + 3.0 * yy + 5.0 * zz
        g1, g2, g3 = gradient(f, dx, dy, dz)
        np.testing.assert_allclose(g1, 2.0, rtol=1e-14)
        np.testing.assert_allclose(g2, 3.0, rtol=1e-14)
        np.testing.assert_allclose(g3, 5.0, rtol=1e-14)

    def test_quadratic_field_interior(self):
        r"""f = x^2 + y^2 has $\nabla f = (2x, 2y, 0)$, exact in interior."""
        nx, ny, nz = 16, 16, 8
        dx = dy = dz = 0.5
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, _zz = np.meshgrid(x, y, z, indexing="ij")

        f = xx**2 + yy**2
        g1, g2, g3 = gradient(f, dx, dy, dz)

        # Central differences are exact for quadratics in interior
        interior = slice(1, -1)
        np.testing.assert_allclose(
            g1[interior, interior, interior],
            2.0 * xx[interior, interior, interior],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            g2[interior, interior, interior],
            2.0 * yy[interior, interior, interior],
            rtol=1e-14,
        )
        np.testing.assert_allclose(g3, 0.0, atol=1e-14)

    def test_nan_propagation(self):
        shape = (4, 4, 4)
        f = np.ones(shape)
        f[2, 2, 2] = np.nan
        g1, g2, g3 = gradient(f, 1.0, 1.0, 1.0)
        assert np.any(np.isnan(g1)) or np.any(np.isnan(g2)) or np.any(np.isnan(g3))


class TestConvergence:
    @staticmethod
    def _sinusoidal_divergence_error(n: int) -> float:
        r"""L-inf error of divergence for $F = (\sin x, \sin y, \sin z)$."""
        dx = 2.0 * np.pi / n
        x = np.arange(n) * dx
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        f1 = np.sin(xx)
        f2 = np.sin(yy)
        f3 = np.sin(zz)
        exact = np.cos(xx) + np.cos(yy) + np.cos(zz)
        numerical = divergence(f1, f2, f3, dx, dx, dx)
        interior = slice(2, -2)
        return float(
            np.max(
                np.abs(
                    numerical[interior, interior, interior]
                    - exact[interior, interior, interior]
                )
            )
        )

    @staticmethod
    def _sinusoidal_curl_error(n: int) -> float:
        r"""L-inf error of curl for $F = (\sin y, \sin z, \sin x)$."""
        dx = 2.0 * np.pi / n
        x = np.arange(n) * dx
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        f1 = np.sin(yy)
        f2 = np.sin(zz)
        f3 = np.sin(xx)
        # Exact curl: (cos(x) - cos(z), cos(y) - cos(x), cos(z) - cos(y))
        # curl_1 = df3/dy - df2/dz = 0 - cos(z)
        # curl_2 = df1/dz - df3/dx = 0 - cos(x)
        # curl_3 = df2/dx - df1/dy = 0 - cos(y)
        exact_1 = -np.cos(zz)
        exact_2 = -np.cos(xx)
        exact_3 = -np.cos(yy)
        c1, c2, c3 = curl(f1, f2, f3, dx, dx, dx)
        interior = slice(2, -2)
        s = (interior, interior, interior)
        error = max(
            float(np.max(np.abs(c1[s] - exact_1[s]))),
            float(np.max(np.abs(c2[s] - exact_2[s]))),
            float(np.max(np.abs(c3[s] - exact_3[s]))),
        )
        return error

    @staticmethod
    def _sinusoidal_gradient_error(n: int) -> float:
        r"""L-inf error of gradient for $f = \sin x \sin y \sin z$."""
        dx = 2.0 * np.pi / n
        x = np.arange(n) * dx
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        f = np.sin(xx) * np.sin(yy) * np.sin(zz)
        exact_1 = np.cos(xx) * np.sin(yy) * np.sin(zz)
        exact_2 = np.sin(xx) * np.cos(yy) * np.sin(zz)
        exact_3 = np.sin(xx) * np.sin(yy) * np.cos(zz)
        g1, g2, g3 = gradient(f, dx, dx, dx)
        interior = slice(2, -2)
        s = (interior, interior, interior)
        error = max(
            float(np.max(np.abs(g1[s] - exact_1[s]))),
            float(np.max(np.abs(g2[s] - exact_2[s]))),
            float(np.max(np.abs(g3[s] - exact_3[s]))),
        )
        return error

    def test_divergence_second_order(self):
        """Doubling resolution should reduce error by ~4x (second order)."""
        errors = [self._sinusoidal_divergence_error(n) for n in [16, 32, 64]]
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.5, f"First ratio {ratio_1:.2f} too low"
        assert ratio_2 > 3.5, f"Second ratio {ratio_2:.2f} too low"

    def test_curl_second_order(self):
        """Doubling resolution should reduce error by ~4x (second order)."""
        errors = [self._sinusoidal_curl_error(n) for n in [16, 32, 64]]
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.5, f"First ratio {ratio_1:.2f} too low"
        assert ratio_2 > 3.5, f"Second ratio {ratio_2:.2f} too low"

    def test_gradient_second_order(self):
        """Doubling resolution should reduce error by ~4x (second order)."""
        errors = [self._sinusoidal_gradient_error(n) for n in [16, 32, 64]]
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.5, f"First ratio {ratio_1:.2f} too low"
        assert ratio_2 > 3.5, f"Second ratio {ratio_2:.2f} too low"


class TestNonCartesianNotImplemented:
    @pytest.mark.parametrize("geom", [GeometryType.SPHERICAL, GeometryType.CYLINDRICAL])
    def test_divergence_raises(self, geom: GeometryType):
        f = np.zeros((4, 4, 4))
        with pytest.raises(NotImplementedError, match=geom.value):
            divergence(f, f, f, 1.0, 1.0, 1.0, geometry=geom)

    @pytest.mark.parametrize("geom", [GeometryType.SPHERICAL, GeometryType.CYLINDRICAL])
    def test_curl_raises(self, geom: GeometryType):
        f = np.zeros((4, 4, 4))
        with pytest.raises(NotImplementedError, match=geom.value):
            curl(f, f, f, 1.0, 1.0, 1.0, geometry=geom)

    @pytest.mark.parametrize("geom", [GeometryType.SPHERICAL, GeometryType.CYLINDRICAL])
    def test_gradient_raises(self, geom: GeometryType):
        f = np.zeros((4, 4, 4))
        with pytest.raises(NotImplementedError, match=geom.value):
            gradient(f, 1.0, 1.0, 1.0, geometry=geom)
