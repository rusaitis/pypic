"""Tests for pypic.spectral — power spectrum functions."""

from __future__ import annotations

import numpy as np
import pytest

from pypic.spectral import power_spectrum_1d, power_spectrum_2d, power_spectrum_3d


class TestPowerSpectrum1D:
    def test_single_mode_peak(self) -> None:
        """A pure sine at wavenumber k0 should peak at k0."""
        n, dx = 256, 0.1
        x = np.arange(n) * dx
        k0 = 2.0 * np.pi / (n * dx / 4)  # 4 wavelengths in the domain
        field = np.sin(k0 * x)
        k, power = power_spectrum_1d(field, dx)
        peak_k = k[np.argmax(power)]
        np.testing.assert_allclose(peak_k, k0, rtol=1e-12)

    def test_parseval(self) -> None:
        """Total power approximates field variance (Parseval's theorem)."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(512)
        dx = 1.0
        k, power = power_spectrum_1d(field, dx, window="boxcar")
        dk = k[1] - k[0]
        total_power = float(np.sum(power) * dk)
        np.testing.assert_allclose(total_power, np.var(field), rtol=1e-10)

    def test_2d_input_averages(self) -> None:
        """2D input: spectrum is averaged over non-FFT axis."""
        rng = np.random.default_rng(7)
        field = rng.standard_normal((64, 8))
        k, power = power_spectrum_1d(field, 1.0, axis=0)
        assert k.ndim == 1
        assert power.ndim == 1
        assert len(k) == len(power)

    def test_window_reduces_leakage(self) -> None:
        """Hann window should have lower sidelobes than boxcar."""
        n, dx = 256, 1.0
        x = np.arange(n) * dx
        field = np.sin(2 * np.pi * x / 50)
        _, p_box = power_spectrum_1d(field, dx, window="boxcar")
        _, p_hann = power_spectrum_1d(field, dx, window="hann")
        assert np.median(p_hann) < np.median(p_box)

    def test_dc_excluded(self) -> None:
        """k=0 (DC component) should not appear in output."""
        k, _power = power_spectrum_1d(np.ones(64), 1.0)
        assert k[0] > 0

    def test_k_units_radians(self) -> None:
        """Wavenumber should be in radians per unit length."""
        n, dx = 128, 0.5
        k, _ = power_spectrum_1d(np.ones(n), dx)
        dk_expected = 2.0 * np.pi / (n * dx)
        np.testing.assert_allclose(k[0], dk_expected, rtol=1e-12)


class TestPowerSpectrum2D:
    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        k, power = power_spectrum_2d(field, 1.0, 1.0)
        assert k.ndim == 1
        assert power.ndim == 1
        assert len(k) == len(power)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            power_spectrum_2d(np.ones(10), 1.0, 1.0)

    def test_isotropic_noise_roughly_flat(self) -> None:
        """White noise should have a roughly flat spectrum."""
        rng = np.random.default_rng(99)
        field = rng.standard_normal((128, 128))
        _k, power = power_spectrum_2d(field, 1.0, 1.0, n_bins=20)
        cv = float(np.std(power) / np.mean(power))
        assert cv < 0.5

    def test_positive_wavenumbers(self) -> None:
        rng = np.random.default_rng(1)
        k, _ = power_spectrum_2d(rng.standard_normal((32, 32)), 1.0, 1.0)
        assert np.all(k > 0)


class TestPowerSpectrum3D:
    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(42)
        field = rng.standard_normal((32, 32, 32))
        k, power = power_spectrum_3d(field, 1.0, 1.0, 1.0)
        assert k.ndim == 1
        assert power.ndim == 1
        assert len(k) == len(power)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            power_spectrum_3d(np.ones((10, 10)), 1.0, 1.0, 1.0)

    def test_isotropic_noise_roughly_flat(self) -> None:
        """White noise should have a roughly flat 3D spectrum."""
        rng = np.random.default_rng(99)
        field = rng.standard_normal((48, 48, 48))
        _k, power = power_spectrum_3d(field, 1.0, 1.0, 1.0, n_bins=15)
        cv = float(np.std(power) / np.mean(power))
        assert cv < 0.5

    def test_positive_wavenumbers(self) -> None:
        rng = np.random.default_rng(1)
        k, _ = power_spectrum_3d(rng.standard_normal((16, 16, 16)), 1.0, 1.0, 1.0)
        assert np.all(k > 0)

    def test_anisotropic_spacing(self) -> None:
        """Non-uniform spacing should still produce valid output."""
        rng = np.random.default_rng(7)
        field = rng.standard_normal((24, 24, 24))
        k, power = power_spectrum_3d(field, 0.5, 1.0, 2.0, n_bins=10)
        assert len(k) > 0
        assert np.all(np.isfinite(power))
        assert np.all(np.diff(k) > 0), "k bins must be monotonically increasing"
        assert np.all(power >= 0), "PSD must be non-negative"


class TestSpectralEdgeCases:
    """Edge cases: NaN, zeros, even/odd Nyquist, invalid windows."""

    def test_all_zero_field_1d(self) -> None:
        """All-zero field has zero power everywhere — no NaN, no inf."""
        k, power = power_spectrum_1d(np.zeros(64), 1.0)
        assert len(k) == len(power)
        assert np.all(np.isfinite(power))
        assert np.all(power == 0.0)

    def test_nan_propagates_1d(self) -> None:
        """NaN in input propagates to output (no silent dropping)."""
        field = np.zeros(64)
        field[10] = np.nan
        _k, power = power_spectrum_1d(field, 1.0)
        assert np.any(np.isnan(power))

    def test_invalid_window_raises(self) -> None:
        """Unknown window names raise (caller bug, surface immediately)."""
        with pytest.raises(ValueError, match=r"not_a_real_window|Unknown window"):
            power_spectrum_1d(np.ones(32), 1.0, window="not_a_real_window")

    @pytest.mark.parametrize("n", [64, 65])
    def test_nyquist_handling_parity(self, n: int) -> None:
        """Even and odd N both produce monotonically increasing k."""
        field = np.random.default_rng(0).standard_normal(n)
        k, power = power_spectrum_1d(field, 1.0)
        assert np.all(np.diff(k) > 0)
        assert np.all(np.isfinite(power))

    def test_all_zero_field_2d(self) -> None:
        """2D zero field — empty bins are NaN, valid bins are 0."""
        _k, power = power_spectrum_2d(np.zeros((32, 32)), 1.0, 1.0, n_bins=10)
        # Output excludes empty bins; surviving entries must all be 0
        assert np.all(power == 0.0)

    def test_all_zero_field_3d(self) -> None:
        """3D zero field — same property as 2D."""
        _k, power = power_spectrum_3d(
            np.zeros((16, 16, 16)),
            1.0,
            1.0,
            1.0,
            n_bins=8,
        )
        assert np.all(power == 0.0)
