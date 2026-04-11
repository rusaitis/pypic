"""Comparison and validation diagnostics for simulation output.

Error norms, field comparison, and constraint monitoring (div B, div E).
All functions are pure: arrays in, scalars or arrays out. No FieldDataset
dependency. Divergence delegates to ``pypic.coordinates.operators``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from pypic.coordinates.geometry import GeometryType
from pypic.coordinates.operators import divergence

if TYPE_CHECKING:
    from typing import Any

    from pypic.types import FloatArray


type NanPolicy = Literal["omit", "propagate", "raise"]

# Anchor for ``warnings.warn(skip_file_prefixes=...)`` so NaN warnings
# point at the user's call site regardless of how deeply pypic itself
# wraps the diagnostic (direct, via ``compare_fields``, via CLI
# commands, ...). Python walks up the stack until it exits the pypic
# package prefix. Computed once at import time from this module's own
# file location.
_PYPIC_PREFIX = (str(Path(__file__).parent),)


def _apply_nan_policy(
    computed: FloatArray,
    reference: FloatArray,
    *,
    nan_policy: NanPolicy,
    function_name: str,
) -> tuple[FloatArray, FloatArray] | None:
    """Apply *nan_policy* to a (computed, reference) pair.

    Returns the (possibly masked) pair to feed into the metric, or
    ``None`` to signal "all cells masked, return NaN".

    For ``"omit"``, NaN cells in either input are dropped from both
    arrays via a joint mask, so downstream relative norms compare the
    same set of points in numerator and denominator. Emits a
    :class:`UserWarning` reporting the dropped count when masking
    occurs (no warning when both inputs are NaN-free).
    """
    if nan_policy not in ("omit", "propagate", "raise"):
        msg = f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        raise ValueError(msg)

    if nan_policy == "propagate":
        return computed, reference

    nan_mask = np.isnan(computed) | np.isnan(reference)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return computed, reference

    if nan_policy == "raise":
        msg = f"{function_name}: input contains {n_nan} NaN cell(s)"
        raise ValueError(msg)

    valid = ~nan_mask
    if not valid.any():
        warnings.warn(
            f"{function_name}: all {n_nan} cell(s) are NaN, result undefined",
            UserWarning,
            skip_file_prefixes=_PYPIC_PREFIX,
        )
        return None
    fraction = 100.0 * n_nan / nan_mask.size
    warnings.warn(
        f"{function_name}: ignored {n_nan} NaN cell(s) ({fraction:.2f}% of input)",
        UserWarning,
        skip_file_prefixes=_PYPIC_PREFIX,
    )
    return computed[valid], reference[valid]


def l2_relative_error(
    computed: FloatArray,
    reference: FloatArray,
    *,
    nan_policy: NanPolicy = "omit",
) -> np.floating[Any]:
    r"""Compute the discrete relative L2 error norm.

    $$\varepsilon_{L_2} = \frac{\sqrt{\sum_i (a_i - b_i)^2}}
    {\sqrt{\sum_i b_i^2}}$$

    Unweighted (no volume factor) — on the same grid, $\Delta V$
    cancels between numerator and denominator.

    Parameters
    ----------
    computed : NDArray
        Computed field values.
    reference : NDArray
        Reference (exact or baseline) field values.
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells in either input. ``"omit"`` masks them
        out (both numerator and denominator restricted to the same valid
        set, so the relative error stays mathematically coherent) and
        emits a :class:`UserWarning` reporting the dropped count.
        ``"propagate"`` is the unaltered NumPy reduction — any NaN
        poisons the result. ``"raise"`` errors on any NaN.

    Returns
    -------
    np.floating
        Relative L2 error. ``inf`` if the reference is all zeros over
        the valid cells, ``nan`` if both are all zeros or no valid
        cells remain.

    Examples
    --------
    >>> import numpy as np
    >>> l2_relative_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    np.float64(0.0)
    """
    masked = _apply_nan_policy(
        computed,
        reference,
        nan_policy=nan_policy,
        function_name="l2_relative_error",
    )
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    c, r = masked
    diff_norm = np.sqrt(np.sum((c - r) ** 2))
    ref_norm = np.sqrt(np.sum(r**2))
    return diff_norm / ref_norm  # type: ignore[no-any-return]  # inf or nan when ref_norm == 0


def linf_error(
    computed: FloatArray,
    reference: FloatArray,
    *,
    nan_policy: NanPolicy = "omit",
) -> np.floating[Any]:
    r"""Compute the absolute L-infinity (max-norm) error.

    $$\varepsilon_{L_\infty} = \max_i |a_i - b_i|$$

    Absolute, not relative — relative $L_\infty$ is misleading near
    field nulls.

    Parameters
    ----------
    computed : NDArray
        Computed field values.
    reference : NDArray
        Reference (exact or baseline) field values.
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells in either input. See
        :func:`l2_relative_error` for the full semantics.

    Returns
    -------
    np.floating
        Maximum absolute pointwise error. ``nan`` if no valid cells
        remain (``"omit"`` with all-NaN input).

    Examples
    --------
    >>> import numpy as np
    >>> linf_error(np.array([1.0, 3.0]), np.array([1.0, 2.0]))
    np.float64(1.0)
    """
    masked = _apply_nan_policy(
        computed,
        reference,
        nan_policy=nan_policy,
        function_name="linf_error",
    )
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    c, r = masked
    return np.max(np.abs(c - r))


def field_difference(
    a: FloatArray,
    b: FloatArray,
) -> FloatArray:
    r"""Compute the pointwise signed difference between two fields.

    Waste no time arguing what a good field should be. Compute one.

    $$\Delta f_i = a_i - b_i$$

    Parameters
    ----------
    a : NDArray
        First field.
    b : NDArray
        Second field (subtracted from ``a``).

    Returns
    -------
    NDArray
        Signed difference array, same shape as input.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> field_difference(np.array([3.0, 5.0]), np.array([1.0, 2.0]))
    array([2., 3.])
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    result: FloatArray = a - b
    return result


def field_energy(
    energy_density: FloatArray,
    spacing: tuple[float, ...],
) -> np.floating[Any]:
    r"""Compute the volume integral of a scalar field.

    $$E = \sum_{i,j,k} f_{i,j,k} \cdot \Delta V, \quad
    \Delta V = \prod_k \Delta x_k$$

    Pass an energy density (e.g. from ``magnetic_energy_density``) to
    get total energy, or a mass density to get total mass.

    Parameters
    ----------
    energy_density : NDArray
        Scalar field to integrate (1D, 2D, or 3D).
    spacing : tuple[float, ...]
        Grid spacing along each axis. Length must match the number of
        array dimensions.

    Returns
    -------
    np.floating
        Volume-integrated quantity.

    Raises
    ------
    ValueError
        If ``len(spacing)`` does not match the array dimensionality.

    Examples
    --------
    >>> import numpy as np
    >>> field_energy(np.ones((4, 4, 4)), (0.5, 0.5, 0.5))
    np.float64(8.0)
    """
    if len(spacing) != energy_density.ndim:
        msg = (
            f"spacing has {len(spacing)} elements but array "
            f"has {energy_density.ndim} dimensions"
        )
        raise ValueError(msg)
    dv = math.prod(spacing)
    return np.sum(energy_density) * dv


def div_b(
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> FloatArray:
    r"""Compute the divergence of the magnetic field.

    $$(\nabla \cdot \mathbf{B})_{i,j,k} =
    \frac{\partial B_1}{\partial x} +
    \frac{\partial B_2}{\partial y} +
    \frac{\partial B_3}{\partial z}$$

    Uses second-order central differences (interior) with second-order
    one-sided stencils at boundaries. Should be close to zero for
    physically valid magnetic fields.

    Parameters
    ----------
    b1 : NDArray
        First component of the magnetic field, shape ``(nx, ny, nz)``.
    b2 : NDArray
        Second component of the magnetic field, shape ``(nx, ny, nz)``.
    b3 : NDArray
        Third component of the magnetic field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only Cartesian is implemented.

    Returns
    -------
    NDArray
        Divergence of B, same shape as the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> b1 = np.ones((4, 4, 4))
    >>> b2 = np.ones((4, 4, 4))
    >>> b3 = np.ones((4, 4, 4))
    >>> np.max(np.abs(div_b(b1, b2, b3, 1.0, 1.0, 1.0)))
    np.float64(0.0)
    """
    return divergence(b1, b2, b3, d1, d2, d3, geometry=geometry)


def max_div_b(
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> np.floating[Any]:
    r"""Compute the maximum absolute divergence of B.

    $$\max |\nabla \cdot \mathbf{B}|$$

    The one-number quality metric reported in MHD code verification.

    Parameters
    ----------
    b1 : NDArray
        First component of the magnetic field, shape ``(nx, ny, nz)``.
    b2 : NDArray
        Second component of the magnetic field, shape ``(nx, ny, nz)``.
    b3 : NDArray
        Third component of the magnetic field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only Cartesian is implemented.

    Returns
    -------
    np.floating
        Maximum absolute value of div B.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.ones((4, 4, 4))
    >>> max_div_b(b, b, b, 1.0, 1.0, 1.0)
    np.float64(0.0)
    """
    return np.max(np.abs(div_b(b1, b2, b3, d1, d2, d3, geometry=geometry)))


def div_e(
    e1: FloatArray,
    e2: FloatArray,
    e3: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> FloatArray:
    r"""Compute the divergence of the electric field.

    $$(\nabla \cdot \mathbf{E})_{i,j,k} =
    \frac{\partial E_1}{\partial x} +
    \frac{\partial E_2}{\partial y} +
    \frac{\partial E_3}{\partial z}$$

    In normalized units ($\epsilon_0 = 1$), $\nabla \cdot \mathbf{E}
    = \rho_c$ (Gauss's law).

    Parameters
    ----------
    e1 : NDArray
        First component of the electric field, shape ``(nx, ny, nz)``.
    e2 : NDArray
        Second component of the electric field, shape ``(nx, ny, nz)``.
    e3 : NDArray
        Third component of the electric field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only Cartesian is implemented.

    Returns
    -------
    NDArray
        Divergence of E, same shape as the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> e = np.ones((4, 4, 4))
    >>> np.max(np.abs(div_e(e, e, e, 1.0, 1.0, 1.0)))
    np.float64(0.0)
    """
    return divergence(e1, e2, e3, d1, d2, d3, geometry=geometry)


def spatial_mean(field: FloatArray) -> np.floating[Any]:
    r"""Compute the spatial mean of a field, ignoring NaN.

    Uses unweighted averaging, which is correct for uniform Cartesian grids
    where all cells have equal volume. For non-Cartesian geometries, a
    volume-weighted average ($\int f\, dV / \int dV$) would be needed.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).

    Returns
    -------
    np.floating
        Spatial mean value.

    Examples
    --------
    >>> import numpy as np
    >>> spatial_mean(np.array([1.0, 2.0, 3.0]))
    np.float64(2.0)
    """
    return np.nanmean(field)


def spatial_rms(field: FloatArray) -> np.floating[Any]:
    r"""Compute the root-mean-square of a field, ignoring NaN.

    $$f_{rms} = \sqrt{\langle f^2 \rangle}$$

    Uses unweighted averaging, correct for uniform Cartesian grids.
    Non-Cartesian geometries require volume-weighted RMS.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).

    Returns
    -------
    np.floating
        RMS value.

    Examples
    --------
    >>> import numpy as np
    >>> spatial_rms(np.array([3.0, 4.0]))
    np.float64(3.5355339059327378)
    """
    return cast("np.floating[Any]", np.sqrt(np.nanmean(field**2)))


def field_extrema(
    field: FloatArray,
) -> tuple[np.floating[Any], np.floating[Any]]:
    r"""Return the minimum and maximum of a field, ignoring NaN.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).

    Returns
    -------
    tuple[np.floating, np.floating]
        ``(min, max)`` values.

    Examples
    --------
    >>> import numpy as np
    >>> field_extrema(np.array([3.0, -1.0, 7.0]))
    (np.float64(-1.0), np.float64(7.0))
    """
    return np.nanmin(field), np.nanmax(field)


__all__ = [
    "div_b",
    "div_e",
    "field_difference",
    "field_energy",
    "field_extrema",
    "l2_relative_error",
    "linf_error",
    "max_div_b",
    "spatial_mean",
    "spatial_rms",
]
