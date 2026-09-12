# Source: src/pypic/diagnostics.py:230 (field_energy,
#         E = Σ f_{ijk}·ΔV); :431 (spatial_mean, np.nanmean);
#         :457 (spatial_rms, sqrt(nanmean(f²))); docs/equations.md § 7
#         (reductions table); docs/conventions.md § "Field energy as
#         volume integral" ("computes Σ f·ΔV, not Σ f²·ΔV").
# Claim:
#  (a) field_energy(α·f, dx) == α · field_energy(f, dx) — linearity
#      in f. Sign-preserving: a negative f gives a negative energy
#      (catches any accidental square in the sum that would yield a
#      non-negative L2 norm² instead of the documented volume integral).
#  (b) field_energy(f, α·dx) == α^ndim · field_energy(f, dx) —
#      homogeneous of degree ndim in the spacing. Guards the
#      math.prod(spacing) product.
#  (c) spatial_mean(α·f + β·g) == α·spatial_mean(f) + β·spatial_mean(g)
#      — linearity of the unweighted NaN-ignoring mean.
#  (d) spatial_rms(α·f) == |α|·spatial_rms(f) — homogeneity of degree 1
#      in magnitude.
#  (e) spatial_rms(f)² == spatial_mean(f²) — the rms is literally
#      sqrt of mean-of-squares, so squaring the rms must recover the
#      mean-square exactly to float64.
# Fresh invariant #20 after backlog exhaustion.
"""Linearity + homogeneity of field_energy, spatial_mean, spatial_rms."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from pypic.diagnostics import field_energy, spatial_mean, spatial_rms

SHAPE = (3, 4, 2)


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    """Signed finite array. NaNs excluded to keep spatial_mean / rms
    answers deterministic (NaN handling is its own invariant family,
    covered by test_diagnostics.py)."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _positive_spacing() -> st.SearchStrategy[tuple[float, float, float]]:
    """Strictly positive spacing per axis — unphysical to have zero dx."""
    return st.tuples(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    )


def _nonzero_scalar() -> st.SearchStrategy[float]:
    """Signed scalar α bounded away from zero, for linear scaling tests."""
    return st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    ).flatmap(lambda x: st.sampled_from([x, -x]))


@given(
    f=_bounded_array(),
    spacing=_positive_spacing(),
    alpha=_nonzero_scalar(),
)
@settings(max_examples=40)
def test_field_energy_is_linear_in_field(
    f: np.ndarray, spacing: tuple[float, float, float], alpha: float
) -> None:
    r"""$E(\alpha f, \Delta x) = \alpha \cdot E(f, \Delta x)$.

    field_energy is a *volume integral* — linear in the integrand.
    If someone refactors ``np.sum(energy_density)`` to
    ``np.sum(energy_density**2)`` (an L2 norm²), this test catches
    it: $E(\alpha f) = \alpha^2 E(f)$ would fail at any $\alpha \ne \pm 1$,
    and negative $\alpha$ with the L2 form would give a positive
    result instead of negating.
    """
    baseline = field_energy(f, spacing)
    scaled = field_energy(alpha * f, spacing)
    # Summation order differs between the two sides, so the roundoff
    # scales with the sum of magnitudes, not with the (possibly cancelling)
    # result: an rtol on the result alone flakes whenever the integrand
    # nearly sums to zero.
    roundoff = 8 * np.finfo(float).eps * abs(alpha) * field_energy(np.abs(f), spacing)
    assert_allclose(scaled, alpha * baseline, rtol=1e-13, atol=roundoff)


@given(
    f=_bounded_array(),
    spacing=_positive_spacing(),
    alpha=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
)
@settings(max_examples=40)
def test_field_energy_scales_as_alpha_cubed_in_spacing(
    f: np.ndarray, spacing: tuple[float, float, float], alpha: float
) -> None:
    r"""$E(f, \alpha \Delta x) = \alpha^{n_{dim}} \cdot E(f, \Delta x)$.

    $\Delta V = \prod_k \Delta x_k$ scales as $\alpha^{n_{dim}}$ when
    every axis is rescaled uniformly. For 3D fields: α³. A bug that
    swapped math.prod for math.sum would give $n_{dim} \cdot \alpha$
    and fail here.
    """
    baseline = field_energy(f, spacing)
    scaled = field_energy(f, tuple(alpha * s for s in spacing))
    ndim = f.ndim
    assert_allclose(scaled, alpha**ndim * baseline, rtol=1e-13, atol=1e-13)


@given(
    f=_bounded_array(),
    g=_bounded_array(),
    alpha=_nonzero_scalar(),
    beta=_nonzero_scalar(),
)
@settings(max_examples=40)
def test_spatial_mean_is_linear(
    f: np.ndarray, g: np.ndarray, alpha: float, beta: float
) -> None:
    r"""$\langle \alpha f + \beta g \rangle = \alpha \langle f \rangle
    + \beta \langle g \rangle$.

    Pure linearity of the unweighted mean. nanmean is implemented via
    nansum/count, so the linearity holds exactly when no NaN entries
    are present (strategy excludes NaN).
    """
    lhs = spatial_mean(alpha * f + beta * g)
    rhs = alpha * spatial_mean(f) + beta * spatial_mean(g)
    assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)


@given(
    f=_bounded_array(),
    alpha=_nonzero_scalar(),
)
@settings(max_examples=40)
def test_spatial_rms_is_homogeneous_in_magnitude(f: np.ndarray, alpha: float) -> None:
    r"""$f_{rms}(\alpha f) = |\alpha| \cdot f_{rms}(f)$.

    rms is built from $f^2$: $f_{rms}(\alpha f) = \sqrt{\langle
    \alpha^2 f^2 \rangle} = |\alpha| f_{rms}(f)$. The absolute value
    is a built-in consequence of the square — a sign leak
    (``alpha * spatial_rms`` instead of ``abs(alpha) * spatial_rms``)
    would break this for $\alpha < 0$.
    """
    baseline = spatial_rms(f)
    scaled = spatial_rms(alpha * f)
    assert_allclose(scaled, abs(alpha) * baseline, rtol=1e-13, atol=1e-13)


@given(f=_bounded_array())
@settings(max_examples=40)
def test_spatial_rms_squared_equals_mean_of_squares(f: np.ndarray) -> None:
    r"""$f_{rms}^2 = \langle f^2 \rangle$ to float64 precision.

    Literal definition: ``spatial_rms(f) = sqrt(nanmean(f**2))``, so
    squaring the rms must recover the mean of squares. One round-trip
    through ``sqrt`` incurs ≤ 1 ulp on the value, dominated here by
    sum-of-squares magnitude. Not bit-exact (sqrt inherent roundoff)
    but tight.
    """
    lhs = spatial_rms(f) ** 2
    rhs = spatial_mean(f * f)
    assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)
