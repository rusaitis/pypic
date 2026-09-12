"""Comparison and validation diagnostics for simulation output.

Error norms, field comparison, and constraint monitoring (div B, div E).
All functions are pure: arrays in, scalars or arrays out. No FieldDataset
dependency. Divergence delegates to ``pypic.coordinates.operators``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args, overload

import numpy as np

from pypic.coordinates.geometry import GeometryType
from pypic.coordinates.operators import divergence

if TYPE_CHECKING:
    from typing import Any

    from pypic.types import FloatArray


type NanPolicy = Literal["omit", "propagate", "raise"]

# Single source for the policy vocabulary: `pypic.reductions` and
# `pypic.comparison` validate against this tuple too, so adding a policy
# means editing the alias above and nothing else.
_VALID_NAN_POLICIES: tuple[str, ...] = get_args(NanPolicy.__value__)

# Anchor for ``warnings.warn(skip_file_prefixes=...)``: Python walks up the
# stack until it leaves the pypic package, so a NaN warning points at the
# user's call site however deeply pypic wrapped the diagnostic.
_PYPIC_PREFIX = (str(Path(__file__).parent),)


@overload
def _apply_nan_policy(
    field: FloatArray,
    /,
    *,
    nan_policy: NanPolicy,
    function_name: str,
) -> tuple[FloatArray] | None: ...


@overload
def _apply_nan_policy(
    computed: FloatArray,
    reference: FloatArray,
    /,
    *,
    nan_policy: NanPolicy,
    function_name: str,
) -> tuple[FloatArray, FloatArray] | None: ...


def _apply_nan_policy(
    *arrays: FloatArray,
    nan_policy: NanPolicy,
    function_name: str,
) -> tuple[FloatArray, ...] | None:
    """Apply *nan_policy* to the arrays a diagnostic reduces over.

    Returns the (possibly masked) arrays to feed into the metric, or
    ``None`` to signal "all cells masked, result undefined".

    For ``"omit"``, NaN cells in *any* input are dropped from *every*
    input via one joint mask, so a relative norm compares the same set
    of points in numerator and denominator. Emits a `UserWarning`
    reporting the dropped count when masking occurs (no warning when
    every input is NaN-free).
    """
    if nan_policy not in _VALID_NAN_POLICIES:
        msg = f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        raise ValueError(msg)

    if nan_policy == "propagate":
        return arrays

    nan_mask = np.logical_or.reduce([np.isnan(array) for array in arrays])
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return arrays

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
    return tuple(array[valid] for array in arrays)


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
        emits a `UserWarning` reporting the dropped count.
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
        `l2_relative_error` for the full semantics.

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
    *,
    nan_policy: NanPolicy = "omit",
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
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells.  ``"omit"`` integrates over the valid
        region and emits a `UserWarning` reporting the dropped
        count; ``"propagate"`` is the unaltered ``np.sum`` (NaN
        poisons the integral); ``"raise"`` errors on any NaN.

    Returns
    -------
    np.floating
        Volume-integrated quantity. ``nan`` if all cells are NaN
        under ``"omit"``.

    Raises
    ------
    ValueError
        If ``len(spacing)`` does not match the array dimensionality,
        or if ``nan_policy="raise"`` and any NaN is present.

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
    masked = _apply_nan_policy(
        energy_density, nan_policy=nan_policy, function_name="field_energy"
    )
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    (valid,) = masked
    return np.sum(valid) * dv


def div_b(
    b1: FloatArray,
    b2: FloatArray,
    b3: FloatArray,
    d1: float,
    d2: float,
    d3: float | None = None,
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
    d3 : float or None
        Grid spacing along the third axis, or ``None`` for 2D data.
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
    d3: float | None = None,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
    nan_policy: NanPolicy = "omit",
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
    d3 : float or None
        Grid spacing along the third axis, or ``None`` for 2D data.
    geometry : GeometryType
        Coordinate geometry. Only Cartesian is implemented.
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells in the divergence array (typically
        from upstream NaN-masked input — e.g. ``SphereSelection``).
        See `field_energy` for full semantics.

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
    div = np.abs(div_b(b1, b2, b3, d1, d2, d3, geometry=geometry))
    masked = _apply_nan_policy(div, nan_policy=nan_policy, function_name="max_div_b")
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    (valid,) = masked
    return np.max(valid)


def div_e(
    e1: FloatArray,
    e2: FloatArray,
    e3: FloatArray,
    d1: float,
    d2: float,
    d3: float | None = None,
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
    d3 : float or None
        Grid spacing along the third axis, or ``None`` for 2D data.
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


def spatial_mean(
    field: FloatArray,
    *,
    nan_policy: NanPolicy = "omit",
) -> np.floating[Any]:
    r"""Compute the spatial mean of a field.

    Uses unweighted averaging, which is correct for uniform Cartesian grids
    where all cells have equal volume. For non-Cartesian geometries, a
    volume-weighted average ($\int f\, dV / \int dV$) would be needed.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells. ``"omit"`` averages over the valid
        region and emits a `UserWarning` reporting the dropped
        count; ``"propagate"`` is the unaltered ``np.mean`` (any NaN
        poisons the result); ``"raise"`` errors on any NaN.

    Returns
    -------
    np.floating
        Spatial mean value. ``nan`` if all cells are NaN under ``"omit"``.

    Examples
    --------
    >>> import numpy as np
    >>> spatial_mean(np.array([1.0, 2.0, 3.0]))
    np.float64(2.0)
    """
    masked = _apply_nan_policy(
        field, nan_policy=nan_policy, function_name="spatial_mean"
    )
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    (valid,) = masked
    return np.mean(valid)


def spatial_rms(
    field: FloatArray,
    *,
    nan_policy: NanPolicy = "omit",
) -> np.floating[Any]:
    r"""Compute the root-mean-square of a field.

    $$f_{rms} = \sqrt{\langle f^2 \rangle}$$

    Uses unweighted averaging, correct for uniform Cartesian grids.
    Non-Cartesian geometries require volume-weighted RMS.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells. See `spatial_mean` for full
        semantics.

    Returns
    -------
    np.floating
        RMS value. ``nan`` if all cells are NaN under ``"omit"``.

    Examples
    --------
    >>> import numpy as np
    >>> spatial_rms(np.array([3.0, 4.0]))
    np.float64(3.5355339059327378)
    """
    masked = _apply_nan_policy(
        field, nan_policy=nan_policy, function_name="spatial_rms"
    )
    if masked is None:
        return cast("np.floating[Any]", np.float64(np.nan))
    (valid,) = masked
    return cast("np.floating[Any]", np.sqrt(np.mean(valid**2)))


def field_extrema(
    field: FloatArray,
    *,
    nan_policy: NanPolicy = "omit",
) -> tuple[np.floating[Any], np.floating[Any]]:
    r"""Return the minimum and maximum of a field.

    Parameters
    ----------
    field : NDArray
        Scalar field (any dimensionality).
    nan_policy : {"omit", "propagate", "raise"}, default "omit"
        How to handle NaN cells. See `spatial_mean` for full
        semantics. Under ``"propagate"``, any NaN yields
        ``(nan, nan)``.

    Returns
    -------
    tuple[np.floating, np.floating]
        ``(min, max)`` values. ``(nan, nan)`` if all cells are NaN
        under ``"omit"``.

    Examples
    --------
    >>> import numpy as np
    >>> field_extrema(np.array([3.0, -1.0, 7.0]))
    (np.float64(-1.0), np.float64(7.0))
    """
    masked = _apply_nan_policy(
        field, nan_policy=nan_policy, function_name="field_extrema"
    )
    if masked is None:
        nan = np.float64(np.nan)
        return (
            cast("np.floating[Any]", nan),
            cast("np.floating[Any]", nan),
        )
    (valid,) = masked
    return np.min(valid), np.max(valid)


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
