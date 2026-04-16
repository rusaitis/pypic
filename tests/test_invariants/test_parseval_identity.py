# Source: docs/equations.md § 10 footnote [^15]:
#         "The PSD uses Parseval-consistent normalization:
#          ∫ P(k) dk equals the variance of the windowed signal."
#         src/pypic/spectral.py:26-30 docstring:
#         "$P(k) = |FFT(f)|^2 dx / (2π N)$, so that $\int P(k) dk$
#          is Parseval-consistent with the variance of the windowed
#          signal."
# With the boxcar window (correction = 1, no tapering), the identity
# simplifies to ``sum(power) * dk == var(field)`` bit-exactly for
# zero-mean signals. Verified numerically to machine precision.
# Tapered windows (Hann/Hamming/Blackman) apply a ``1/mean(win²)``
# correction, but for discrete finite signals the integral only
# *approximately* matches the unwindowed variance — signals whose
# energy concentrates near the boundary get zeroed by the window taper,
# and no correction factor can recover that. Tested only with boxcar.
# Fresh invariant #9 after backlog exhaustion.
"""Parseval identity for ``power_spectrum_1d``."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.spectral import power_spectrum_1d


def _signal(n: int) -> st.SearchStrategy[np.ndarray]:
    """1D signals of length *n* with bounded magnitude."""
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@given(
    n=st.sampled_from([64, 128, 256, 512]),
    dx=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    data=st.data(),
)
@settings(max_examples=30, deadline=None)
def test_parseval_boxcar_integral_equals_variance(
    n: int, dx: float, data: st.DataObject
) -> None:
    r"""With ``window="boxcar"``: $\sum P(k) \cdot dk = \operatorname{var}(f)$
    bit-exact for a zero-mean signal.

    The zero-mean shift removes the DC term that ``power_spectrum_1d``
    drops from the returned array (spectral.py:101), so the integrated
    PSD accounts for the full variance.
    """
    field = data.draw(_signal(n))
    # Zero-mean: align with the DC-excluded PSD.
    field = field - field.mean()
    _, power = power_spectrum_1d(field, dx, window="boxcar")
    # Uniform k-grid from rfftfreq: dk = k[1] - k[0] = 2π/(N·dx).
    dk = 2.0 * np.pi / (n * dx)
    integral = float(np.sum(power) * dk)
    variance = float(np.mean(field**2))
    assert_allclose(integral, variance, rtol=1e-12, atol=1e-12)


@given(
    n=st.sampled_from([64, 128, 256]),
    amplitude=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    period_cells=st.integers(min_value=4, max_value=16),
)
@settings(max_examples=30, deadline=None)
def test_parseval_boxcar_pure_sinusoid(
    n: int, amplitude: float, period_cells: int
) -> None:
    r"""A pure sinusoid $A \sin(2\pi x / \lambda)$ with integer cycles
    has variance $A^2/2$; the Parseval-integrated PSD must recover it.

    Requires integer cycles so the DFT has no spectral leakage — the
    entire power sits in one bin, and the Parseval identity holds at
    float64 roundoff rather than at the bin-quantization level.
    """
    dx = 1.0
    cycles = n // period_cells  # integer number of periods in the window
    if cycles < 1:
        return
    x = np.arange(n) * dx
    field = amplitude * np.sin(2.0 * np.pi * cycles * x / n)
    # Zero-mean by construction (integer cycles of sin over [0, N)).
    _, power = power_spectrum_1d(field, dx, window="boxcar")
    dk = 2.0 * np.pi / (n * dx)
    integral = float(np.sum(power) * dk)
    assert_allclose(integral, amplitude**2 / 2.0, rtol=1e-12, atol=1e-12)


# Note: a tapered-window Parseval test is intentionally omitted.
# Hypothesis found counterexamples like ``field = [0, 1, 1, ..., 1]``
# where all the post-zero-mean variance sits near the boundary that
# Hann/Hamming/Blackman windows taper to zero — the integrated PSD
# under-reports the true variance regardless of the ``1/mean(win²)``
# correction. The stated Parseval consistency holds strictly only for
# the boxcar window; tapered-window correction is approximate.
