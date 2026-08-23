"""Discrete differential operators for structured grids.

Geometry-aware ``divergence``, ``curl``, and ``gradient`` using second-order
central finite differences (interior) with second-order one-sided stencils
at boundaries (``np.gradient`` convention). Cartesian geometry is fully
implemented; spherical and cylindrical raise
[`GeometryUnsupportedError`][pypic.exceptions.GeometryUnsupportedError] (a subclass of
``NotImplementedError``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import numpy as np

from pypic.coordinates.geometry import GeometryType
from pypic.exceptions import GeometryUnsupportedError

if TYPE_CHECKING:
    from pypic.types import FloatArray


def _require_positive_spacing(d1: float, d2: float, d3: float) -> None:
    """Raise if any grid spacing is not positive."""
    if d1 <= 0 or d2 <= 0 or d3 <= 0:
        msg = f"Grid spacing must be positive, got ({d1}, {d2}, {d3})"
        raise ValueError(msg)


def _require_cartesian(geometry: GeometryType, operation: str) -> None:
    """Raise if *geometry* is not Cartesian."""
    match geometry:
        case GeometryType.CARTESIAN:
            return
        case GeometryType.SPHERICAL | GeometryType.CYLINDRICAL:
            msg = f"{operation} not implemented for {geometry.value} geometry"
            raise GeometryUnsupportedError(msg)
        case _ as unreachable:
            assert_never(unreachable)


def divergence(
    f1: FloatArray,
    f2: FloatArray,
    f3: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> FloatArray:
    r"""Compute the divergence of a 3D vector field.

    $$\nabla \cdot \mathbf{F} = \frac{\partial F_1}{\partial x}
    + \frac{\partial F_2}{\partial y}
    + \frac{\partial F_3}{\partial z}$$

    Parameters
    ----------
    f1 : NDArray
        First component of the vector field, shape ``(nx, ny, nz)``.
    f2 : NDArray
        Second component of the vector field, shape ``(nx, ny, nz)``.
    f3 : NDArray
        Third component of the vector field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only ``CARTESIAN`` is currently supported.

    Returns
    -------
    NDArray
        Divergence field, same shape as the input arrays.

    Raises
    ------
    GeometryUnsupportedError
        If ``geometry`` is spherical or cylindrical.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.ones((4, 4, 4))
    >>> np.max(np.abs(divergence(f, f, f, 1.0, 1.0, 1.0)))
    np.float64(0.0)
    """
    _require_cartesian(geometry, "divergence")
    _require_positive_spacing(d1, d2, d3)
    df1_d1: FloatArray = np.gradient(f1, d1, axis=0)
    df2_d2: FloatArray = np.gradient(f2, d2, axis=1)
    df3_d3: FloatArray = np.gradient(f3, d3, axis=2)
    result: FloatArray = df1_d1 + df2_d2 + df3_d3
    return result


def curl(
    f1: FloatArray,
    f2: FloatArray,
    f3: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the curl of a 3D vector field.

    $$(\nabla \times \mathbf{F})_1
    = \frac{\partial F_3}{\partial y} - \frac{\partial F_2}{\partial z}$$

    $$(\nabla \times \mathbf{F})_2
    = \frac{\partial F_1}{\partial z} - \frac{\partial F_3}{\partial x}$$

    $$(\nabla \times \mathbf{F})_3
    = \frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y}$$

    Parameters
    ----------
    f1 : NDArray
        First component of the vector field, shape ``(nx, ny, nz)``.
    f2 : NDArray
        Second component of the vector field, shape ``(nx, ny, nz)``.
    f3 : NDArray
        Third component of the vector field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only ``CARTESIAN`` is currently supported.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Curl components ``(curl_1, curl_2, curl_3)``, each with the same
        shape as the input arrays.

    Raises
    ------
    GeometryUnsupportedError
        If ``geometry`` is spherical or cylindrical.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.ones((4, 4, 4))
    >>> c1, c2, c3 = curl(f, f, f, 1.0, 1.0, 1.0)
    >>> np.max(np.abs(c1))
    np.float64(0.0)
    """
    _require_cartesian(geometry, "curl")
    _require_positive_spacing(d1, d2, d3)
    curl_1: FloatArray = np.gradient(f3, d2, axis=1) - np.gradient(f2, d3, axis=2)
    curl_2: FloatArray = np.gradient(f1, d3, axis=2) - np.gradient(f3, d1, axis=0)
    curl_3: FloatArray = np.gradient(f2, d1, axis=0) - np.gradient(f1, d2, axis=1)
    return (curl_1, curl_2, curl_3)


def gradient(
    f: FloatArray,
    d1: float,
    d2: float,
    d3: float,
    *,
    geometry: GeometryType = GeometryType.CARTESIAN,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the gradient of a 3D scalar field.

    $$(\nabla f)_i = \frac{\partial f}{\partial x_i}
    \quad \text{for } i = 1, 2, 3$$

    Parameters
    ----------
    f : NDArray
        Scalar field, shape ``(nx, ny, nz)``.
    d1 : float
        Grid spacing along the first axis.
    d2 : float
        Grid spacing along the second axis.
    d3 : float
        Grid spacing along the third axis.
    geometry : GeometryType
        Coordinate geometry. Only ``CARTESIAN`` is currently supported.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Gradient components ``(df_d1, df_d2, df_d3)``, each with the same
        shape as the input array.

    Raises
    ------
    GeometryUnsupportedError
        If ``geometry`` is spherical or cylindrical.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.ones((4, 4, 4))
    >>> g1, g2, g3 = gradient(f, 1.0, 1.0, 1.0)
    >>> np.max(np.abs(g1))
    np.float64(0.0)
    """
    _require_cartesian(geometry, "gradient")
    _require_positive_spacing(d1, d2, d3)
    df_d1: FloatArray = np.gradient(f, d1, axis=0)
    df_d2: FloatArray = np.gradient(f, d2, axis=1)
    df_d3: FloatArray = np.gradient(f, d3, axis=2)
    return (df_d1, df_d2, df_d3)


__all__ = [
    "curl",
    "divergence",
    "gradient",
]
