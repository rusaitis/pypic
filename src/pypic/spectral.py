r"""Spectral analysis: power spectra for turbulence studies.

Pure functions operating on NumPy arrays. No FieldDataset dependency.
Wavenumbers are in radians per unit length: $k = 2\pi / \lambda$.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.types import FloatArray


def _radial_bin(
    power: FloatArray,
    k_radial: FloatArray,
    n_bins: int,
) -> tuple[FloatArray, FloatArray]:
    r"""Average *power* over equal-width $|k|$ bins, dropping DC.

    Shared by the 2-D (annulus) and 3-D (spherical shell) spectra: the
    geometric weighting lives in the isotropy of the $k$-grid sampling,
    so both reduce to the same count-average over $|k|$. Bin edges span
    $[0, \max |k|]$; the returned centers are bin midpoints, and empty
    bins are dropped rather than returned as NaN.
    """
    k_flat = k_radial.ravel()
    p_flat = power.ravel()
    nonzero = k_flat > 0
    k_flat = k_flat[nonzero]
    p_flat = p_flat[nonzero]

    k_max = float(np.max(k_flat))
    bin_edges = np.linspace(0, k_max, n_bins + 1)
    bin_indices = np.digitize(k_flat, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    power_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(power_binned, bin_indices, p_flat)
    np.add.at(counts, bin_indices, 1)

    valid = counts > 0
    power_binned[valid] /= counts[valid]
    power_binned[~valid] = np.nan

    k_centers: FloatArray = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return k_centers[valid], power_binned[valid]


def power_spectrum_1d(
    field: FloatArray,
    dx: float,
    *,
    axis: int = 0,
    window: str = "hann",
) -> tuple[FloatArray, FloatArray]:
    r"""Compute 1D power spectral density along one axis.

    $$P(k) = \frac{|\hat{f}(k)|^2 \, \Delta x}{2\pi N}$$

    With $k$ in radians per unit length, $dk = 2\pi/(N\Delta x)$, so
    the $2\pi$ factor in the denominator makes $\int P(k)\, dk$
    Parseval-consistent with the variance of the windowed signal.

    Uses ``np.fft.rfft`` (real input, positive frequencies only).
    For multi-dimensional input, the spectrum is averaged over the
    remaining axes.

    Parameters
    ----------
    field : FloatArray
        Input field (1D, 2D, or 3D). The FFT is computed along *axis*.
    dx : float
        Grid spacing along the FFT axis.
    axis : int
        Axis along which to compute the FFT.
    window : str
        Window function name (``"hann"``, ``"hamming"``, ``"blackman"``,
        ``"boxcar"`` for no windowing). Any name accepted by
        ``scipy.signal.windows.get_window``.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(k, power)`` where *k* is in radians per unit length and
        *power* is the spectral density. The DC component (k=0) is
        excluded.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 256, endpoint=False)
    >>> field = np.sin(2 * np.pi * x / 5)
    >>> k, power = power_spectrum_1d(field, x[1] - x[0])
    >>> k[np.argmax(power)]  # peak near k = 2*pi/5
    np.float64(1.2566370614359172)
    """
    from scipy.signal.windows import get_window

    n = field.shape[axis]

    # Build and apply window along the FFT axis
    win = get_window(window, n)
    correction = 1.0 / np.mean(win**2)
    shape = [1] * field.ndim
    shape[axis] = n
    win_nd = win.reshape(shape)
    windowed = field * win_nd

    # FFT (real input → positive frequencies only)
    # Normalization: P(k) dk integrates to variance.
    # With k in radians: dk = 2*pi/(N*dx), so P = |F|^2 * dx / (2*pi*N).
    fk = np.fft.rfft(windowed, axis=axis)
    power = np.abs(fk) ** 2 * (dx / (2.0 * np.pi * n)) * correction

    # Average over non-FFT axes
    other_axes = tuple(i for i in range(field.ndim) if i != axis)
    if other_axes:
        power = np.mean(power, axis=other_axes)

    # One-sided spectrum: double power for non-DC, non-Nyquist bins
    # to account for the negative-frequency mirror
    scale = np.full(len(power), 2.0)
    scale[0] = 1.0  # DC
    if n % 2 == 0:
        scale[-1] = 1.0  # Nyquist (even N only)
    power = power * scale

    # Wavenumber array (radians per unit length)
    freq = np.fft.rfftfreq(n, d=dx)
    k: FloatArray = 2.0 * np.pi * freq

    # Exclude DC component
    return k[1:], power[1:]


def power_spectrum_2d(
    field: FloatArray,
    dx: float,
    dy: float,
    *,
    window: str = "hann",
    n_bins: int = 50,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute radially averaged 2D power spectrum.

    $$P(k) = \langle |\hat{f}(k_x, k_y)|^2 \rangle_{|k|=k}$$

    The 2D FFT is binned by radial wavenumber
    $k = \sqrt{k_x^2 + k_y^2}$ and averaged within each bin.

    Parameters
    ----------
    field : FloatArray
        2D input field of shape ``(nx, ny)``.
    dx : float
        Grid spacing along axis 0.
    dy : float
        Grid spacing along axis 1.
    window : str
        Window function applied independently to each axis.
    n_bins : int
        Number of radial wavenumber bins.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(k, power)`` — bin-center wavenumbers and azimuthally
        averaged power spectral density.

    Raises
    ------
    ValueError
        If *field* is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> field = rng.standard_normal((64, 64))
    >>> k, power = power_spectrum_2d(field, 1.0, 1.0, n_bins=20)
    >>> k.shape == power.shape
    True
    """
    if field.ndim != 2:
        msg = f"power_spectrum_2d requires 2D input, got {field.ndim}D"
        raise ValueError(msg)

    from scipy.signal.windows import get_window

    nx, ny = field.shape

    # Separable 2D window
    win_x = get_window(window, nx)
    win_y = get_window(window, ny)
    win_2d = np.outer(win_x, win_y)
    correction = 1.0 / np.mean(win_2d**2)
    windowed = field * win_2d

    # 2D FFT
    fk = np.fft.fft2(windowed)
    power_2d = np.abs(fk) ** 2 * (dx * dy / (nx * ny)) * correction

    # Wavenumber grids
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    return _radial_bin(power_2d, k_radial, n_bins)


def power_spectrum_3d(
    field: FloatArray,
    dx: float,
    dy: float,
    dz: float,
    *,
    window: str = "hann",
    n_bins: int = 50,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute spherically averaged 3D power spectrum.

    $$P(k) = \langle |\hat{f}(k_x, k_y, k_z)|^2 \rangle_{|k|=k}$$

    The 3D FFT is binned by radial wavenumber
    $k = \sqrt{k_x^2 + k_y^2 + k_z^2}$ and averaged within each
    spherical shell.

    Parameters
    ----------
    field : FloatArray
        3D input field of shape ``(nx, ny, nz)``.
    dx : float
        Grid spacing along axis 0.
    dy : float
        Grid spacing along axis 1.
    dz : float
        Grid spacing along axis 2.
    window : str
        Window function applied independently to each axis.
    n_bins : int
        Number of radial wavenumber bins.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(k, power)`` — shell-center wavenumbers and spherically
        averaged power spectral density.

    Raises
    ------
    ValueError
        If *field* is not 3D.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> field = rng.standard_normal((32, 32, 32))
    >>> k, power = power_spectrum_3d(field, 1.0, 1.0, 1.0, n_bins=15)
    >>> k.shape == power.shape
    True
    """
    if field.ndim != 3:
        msg = f"power_spectrum_3d requires 3D input, got {field.ndim}D"
        raise ValueError(msg)

    from scipy.signal.windows import get_window

    nx, ny, nz = field.shape

    # Separable 3D window via broadcasting
    win_x = get_window(window, nx)
    win_y = get_window(window, ny)
    win_z = get_window(window, nz)
    win_3d = win_x[:, None, None] * win_y[None, :, None] * win_z[None, None, :]
    correction = 1.0 / np.mean(win_3d**2)
    windowed = field * win_3d

    # 3D FFT
    fk = np.fft.fftn(windowed)
    power_3d = np.abs(fk) ** 2 * (dx * dy * dz / (nx * ny * nz)) * correction

    # Wavenumber grids
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    return _radial_bin(power_3d, k_radial, n_bins)


__all__ = [
    "power_spectrum_1d",
    "power_spectrum_2d",
    "power_spectrum_3d",
]
