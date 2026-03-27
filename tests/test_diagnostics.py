import numpy as np
import pytest

from pypic.coordinates.geometry import GeometryType
from pypic.diagnostics import (
    div_b,
    div_e,
    field_difference,
    field_energy,
    l2_relative_error,
    linf_error,
    max_div_b,
)


class TestL2RelativeError:
    def test_hand_calculation(self):
        # computed = [3, 4], reference = [1, 2]
        # diff = [2, 2], ||diff|| = sqrt(8), ||ref|| = sqrt(5)
        computed = np.array([3.0, 4.0])
        reference = np.array([1.0, 2.0])
        expected = np.sqrt(8.0) / np.sqrt(5.0)
        np.testing.assert_allclose(
            l2_relative_error(computed, reference), expected, rtol=1e-15
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_zero_reference_gives_inf(self):
        result = l2_relative_error(np.array([1.0]), np.array([0.0]))
        assert np.isinf(result)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_both_zero_gives_nan(self):
        result = l2_relative_error(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_multidimensional(self):
        a = np.ones((3, 4))
        b = np.zeros((3, 4))
        np.testing.assert_allclose(l2_relative_error(a, a), 0.0, atol=1e-15)
        assert np.isinf(l2_relative_error(a, b))

    def test_nan_propagation(self):
        result = l2_relative_error(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))
        assert np.isnan(result)


class TestLinfError:
    def test_hand_calculation(self):
        computed = np.array([1.0, 5.0, 3.0])
        reference = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(linf_error(computed, reference), 3.0, rtol=1e-15)

    def test_nan_propagation(self):
        result = linf_error(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))
        assert np.isnan(result)


class TestFieldDifference:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            field_difference(np.array([1.0, 2.0]), np.array([1.0]))

    def test_preserves_shape(self):
        a = np.ones((2, 3, 4))
        b = np.zeros((2, 3, 4))
        result = field_difference(a, b)
        assert result.shape == (2, 3, 4)

    def test_nan_propagation(self):
        result = field_difference(np.array([np.nan]), np.array([1.0]))
        assert np.isnan(result[0])


class TestFieldEnergy:
    def test_uniform_field_3d(self):
        # 4x4x4 grid, spacing 0.5 in each direction, density 1.0
        # Volume = 4*0.5 * 4*0.5 * 4*0.5 = 8.0, energy = 64 * 0.125 = 8.0
        f = np.ones((4, 4, 4))
        np.testing.assert_allclose(field_energy(f, (0.5, 0.5, 0.5)), 8.0, rtol=1e-15)

    def test_uniform_field_value(self):
        # 2x2x2 grid, spacing 1.0, density 3.0
        # energy = 8 * 3.0 * 1.0 = 24.0
        f = 3.0 * np.ones((2, 2, 2))
        np.testing.assert_allclose(field_energy(f, (1.0, 1.0, 1.0)), 24.0, rtol=1e-15)

    def test_1d(self):
        f = np.array([1.0, 2.0, 3.0, 4.0])
        # sum = 10, dx = 0.5 -> 5.0
        np.testing.assert_allclose(field_energy(f, (0.5,)), 5.0, rtol=1e-15)

    def test_2d(self):
        f = np.ones((3, 3))
        # sum = 9, dA = 2*2 = 4 -> 36
        np.testing.assert_allclose(field_energy(f, (2.0, 2.0)), 36.0, rtol=1e-15)

    def test_spacing_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="spacing has 2 elements"):
            field_energy(np.ones((4, 4, 4)), (1.0, 1.0))

    def test_nan_propagation(self):
        f = np.array([[[1.0, np.nan]]])
        result = field_energy(f, (1.0, 1.0, 1.0))
        assert np.isnan(result)


class TestDivB:
    def test_div_curl_is_zero(self):
        r"""$\nabla \cdot (\nabla \times \mathbf{A}) = 0$ to machine precision.

        Construct B = curl(A) for a smooth vector potential, then verify
        div(B) ≈ 0. Uses second-order finite differences for both curl
        and div, so the identity holds to truncation-error level.
        """
        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 0.1
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # A = (sin(y), sin(z), sin(x))
        a1 = np.sin(yy)
        a2 = np.sin(zz)
        a3 = np.sin(xx)

        # B = curl(A): B1 = dA3/dy - dA2/dz, etc.
        b1 = np.gradient(a3, dy, axis=1) - np.gradient(a2, dz, axis=2)
        b2 = np.gradient(a1, dz, axis=2) - np.gradient(a3, dx, axis=0)
        b3 = np.gradient(a2, dx, axis=0) - np.gradient(a1, dy, axis=1)

        result = div_b(b1, b2, b3, dx, dy, dz)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_known_divergence(self):
        r"""Field with analytically known divergence.

        B = (x, y, z) has $\nabla \cdot \mathbf{B} = 3$.
        Central differences are exact for linear fields.
        """
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        b1, b2, b3 = np.meshgrid(x, y, z, indexing="ij")
        result = div_b(b1, b2, b3, dx, dy, dz)
        np.testing.assert_allclose(result, 3.0, rtol=1e-14)
        assert result.shape == (nx, ny, nz)

    def test_nan_propagation(self):
        shape = (4, 4, 4)
        b1 = np.ones(shape)
        b1[2, 2, 2] = np.nan
        b2 = np.zeros(shape)
        b3 = np.zeros(shape)
        result = div_b(b1, b2, b3, 1.0, 1.0, 1.0)
        assert np.any(np.isnan(result))


class TestMaxDivB:
    def test_wraps_div_b(self):
        """max_div_b should equal max(|div_b|)."""
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        b1, b2, b3 = np.meshgrid(x, y, z, indexing="ij")
        expected = np.max(np.abs(div_b(b1, b2, b3, dx, dy, dz)))
        result = max_div_b(b1, b2, b3, dx, dy, dz)
        np.testing.assert_allclose(result, expected, rtol=1e-15)


class TestDivE:
    def test_gauss_law_point_charge(self):
        r"""Verify Gauss's law: $\nabla \cdot \mathbf{E} = \rho_c$.

        For E = (x, 0, 0), div E = 1 everywhere. Central differences
        are exact for linear fields.
        """
        nx, ny, nz = 8, 8, 8
        dx = dy = dz = 1.0
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, _yy, _zz = np.meshgrid(x, y, z, indexing="ij")
        e1 = xx
        e2 = np.zeros_like(xx)
        e3 = np.zeros_like(xx)
        result = div_e(e1, e2, e3, dx, dy, dz)
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)
        assert result.shape == (nx, ny, nz)


class TestConvergence:
    def test_divergence_second_order(self):
        r"""Verify second-order convergence of the divergence stencil.

        For $\mathbf{B} = (\sin(x), \sin(y), \sin(z))$, the exact
        divergence is $\cos(x) + \cos(y) + \cos(z)$. Doubling the
        resolution should reduce the error by ~4x.
        """
        errors = []
        for n in [16, 32, 64]:
            dx = 2.0 * np.pi / n
            x = np.arange(n) * dx
            xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
            b1 = np.sin(xx)
            b2 = np.sin(yy)
            b3 = np.sin(zz)
            exact = np.cos(xx) + np.cos(yy) + np.cos(zz)
            numerical = div_b(b1, b2, b3, dx, dx, dx)
            # Exclude boundaries (one-sided stencil, lower order)
            interior = slice(2, -2)
            error = np.max(
                np.abs(
                    numerical[interior, interior, interior]
                    - exact[interior, interior, interior]
                )
            )
            errors.append(error)

        # Check convergence rate: error ratio ≈ 4 for second order
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 3.5, f"First ratio {ratio_1:.2f} too low for 2nd order"
        assert ratio_2 > 3.5, f"Second ratio {ratio_2:.2f} too low for 2nd order"


class TestGeometryForwarding:
    @pytest.mark.parametrize("func", [div_b, max_div_b, div_e])
    def test_rejects_spherical(self, func):
        f = np.ones((4, 4, 4))
        with pytest.raises(NotImplementedError, match="spherical"):
            func(f, f, f, 1.0, 1.0, 1.0, geometry=GeometryType.SPHERICAL)


class TestSpatialStatistics:
    def test_mean(self) -> None:
        from pypic.diagnostics import spatial_mean

        np.testing.assert_allclose(spatial_mean(np.array([1.0, 2.0, 3.0])), 2.0)

    def test_mean_with_nan(self) -> None:
        from pypic.diagnostics import spatial_mean

        np.testing.assert_allclose(
            spatial_mean(np.array([1.0, np.nan, 3.0])), 2.0,
        )

    def test_rms(self) -> None:
        from pypic.diagnostics import spatial_rms

        np.testing.assert_allclose(
            spatial_rms(np.array([3.0, 4.0])), np.sqrt(12.5),
        )

    def test_rms_with_nan(self) -> None:
        from pypic.diagnostics import spatial_rms

        np.testing.assert_allclose(
            spatial_rms(np.array([3.0, np.nan, 4.0])), np.sqrt(12.5),
        )

    def test_extrema(self) -> None:
        from pypic.diagnostics import field_extrema

        lo, hi = field_extrema(np.array([3.0, -1.0, 7.0]))
        assert lo == -1.0
        assert hi == 7.0

    def test_extrema_with_nan(self) -> None:
        from pypic.diagnostics import field_extrema

        lo, hi = field_extrema(np.array([3.0, -1.0, 7.0, np.nan]))
        assert lo == -1.0
        assert hi == 7.0

    def test_2d_array(self) -> None:
        from pypic.diagnostics import spatial_mean

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(spatial_mean(arr), 2.5)
