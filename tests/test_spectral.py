"""Tests for pypic.spectral — power spectrum functions."""

from __future__ import annotations

import numpy as np
import pytest

from pypic.spectral import power_spectrum_1d, power_spectrum_2d, power_spectrum_3d


def _radial_k_max(*axes: tuple[int, float]) -> float:
    """Largest $|k|$ on the FFT grid — the upper edge of the last bin."""
    per_axis = [2.0 * np.pi * np.abs(np.fft.fftfreq(n, d=d)).max() for n, d in axes]
    return float(np.sqrt(sum(k**2 for k in per_axis)))


def _histogram_shell_average(
    k_radial: np.ndarray, cell_power: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shell-average *cell_power* via `np.histogram` — an independent oracle.

    Deliberately not the `digitize` + `np.add.at` path under test: numpy's
    own binning fixes the edges, the right-inclusive last bin and the
    non-empty-bin filtering, so an off-by-one in the implementation shows
    up as a shifted spectrum rather than passing unnoticed.
    """
    keep = k_radial.ravel() > 0
    k_flat = k_radial.ravel()[keep]
    p_flat = cell_power.ravel()[keep]
    span = (0.0, float(k_flat.max()))
    counts, edges = np.histogram(k_flat, bins=n_bins, range=span)
    totals, _ = np.histogram(k_flat, bins=n_bins, range=span, weights=p_flat)
    valid = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[valid], totals[valid] / counts[valid]


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

    def test_single_mode_peaks_in_the_bin_holding_its_wavenumber(self) -> None:
        """An $x$-aligned sinusoid peaks within half a bin of its own $k_0$."""
        nx, ny, dx, dy, cycles, n_bins = 64, 64, 0.5, 0.5, 8, 32
        k0 = 2.0 * np.pi * cycles / (nx * dx)
        phase = 2.0 * np.pi * cycles * np.arange(nx)[:, None] / nx
        field = np.broadcast_to(np.sin(phase), (nx, ny))
        k, power = power_spectrum_2d(field, dx, dy, window="boxcar", n_bins=n_bins)
        half_bin = 0.5 * _radial_k_max((nx, dx), (ny, dy)) / n_bins
        assert abs(float(k[np.argmax(power)]) - k0) < half_bin

    def test_radial_average_matches_a_histogram_oracle(self) -> None:
        """Bin centers and shell averages match `np.histogram` cell-for-cell."""
        rng = np.random.default_rng(5)
        nx, ny, dx, dy, n_bins = 48, 32, 0.5, 2.0, 16
        field = rng.standard_normal((nx, ny))
        k, power = power_spectrum_2d(field, dx, dy, window="boxcar", n_bins=n_bins)
        fk = np.fft.fft2(field)
        cell_power = np.abs(fk) ** 2 * (dx * dy / (nx * ny))
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :]
        expected_k, expected_power = _histogram_shell_average(
            np.sqrt(kx**2 + ky**2), cell_power, n_bins
        )
        np.testing.assert_allclose(k, expected_k, rtol=1e-12)
        np.testing.assert_allclose(power, expected_power, rtol=1e-12)


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

    def test_shell_average_matches_a_histogram_oracle(self) -> None:
        """Shell centers and averages match `np.histogram` cell-for-cell."""
        rng = np.random.default_rng(5)
        nx, ny, nz, n_bins = 16, 12, 20, 10
        dx, dy, dz = 0.5, 1.0, 2.0
        field = rng.standard_normal((nx, ny, nz))
        k, power = power_spectrum_3d(field, dx, dy, dz, window="boxcar", n_bins=n_bins)
        fk = np.fft.fftn(field)
        cell_power = np.abs(fk) ** 2 * (dx * dy * dz / (nx * ny * nz))
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None, None]
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :, None]
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)[None, None, :]
        expected_k, expected_power = _histogram_shell_average(
            np.sqrt(kx**2 + ky**2 + kz**2), cell_power, n_bins
        )
        np.testing.assert_allclose(k, expected_k, rtol=1e-12)
        np.testing.assert_allclose(power, expected_power, rtol=1e-12)

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
