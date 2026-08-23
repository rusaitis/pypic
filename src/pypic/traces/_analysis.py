"""Pure analysis functions for curves in 3D space.

All functions take and return NumPy arrays — no FieldLine or ParticleTrace
dependency. Follows the same pattern as ``pypic.derived`` and
``pypic.diagnostics``.
"""

from __future__ import annotations

__all__ = [
    "arc_length_cumulative",
    "arc_length_total",
    "closest_approach",
    "curvature",
    "displacement",
    "drift_velocity",
    "equatorial_crossings",
    "gyroradius_estimate",
    "kinetic_energy",
    "mirror_points",
    "plane_crossings",
    "resample_by_arc_length",
    "speed",
    "tangent_vectors",
]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pypic.types import FloatArray, Vector3


def arc_length_cumulative(points: FloatArray) -> FloatArray:
    r"""Cumulative arc length along a curve.

    $$s_i = \sum_{k=1}^{i} \|\mathbf{r}_k - \mathbf{r}_{k-1}\|$$

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    FloatArray
        Cumulative arc length, shape ``(N,)``. First element is 0.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> arc_length_cumulative(pts)
    array([0., 1., 2.])
    """
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segments)))


def arc_length_total(points: FloatArray) -> float:
    r"""Total arc length of a curve.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    float
        Sum of segment lengths.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    >>> arc_length_total(pts)
    5.0
    """
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def tangent_vectors(points: FloatArray) -> FloatArray:
    r"""Compute unit tangent vectors along a curve via central differences.

    Uses ``np.gradient`` for second-order central differences at interior
    points and one-sided differences at endpoints.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    FloatArray
        Unit tangent vectors, shape ``(N, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> tangent_vectors(pts)
    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])
    """
    dr = np.gradient(points, axis=0)
    norms = np.linalg.norm(dr, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(dr.dtype).tiny)
    return dr / norms  # type: ignore[no-any-return]


def curvature(points: FloatArray) -> FloatArray:
    r"""Curvature $\kappa = \|d\hat{T}/ds\|$ along a curve.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    FloatArray
        Curvature at each point, shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> np.testing.assert_allclose(curvature(pts), [0.0, 0.0, 0.0], atol=1e-15)
    """
    dr = np.gradient(points, axis=0)
    ds = np.linalg.norm(dr, axis=1, keepdims=True)
    ds = np.maximum(ds, np.finfo(dr.dtype).tiny)
    t_hat = dr / ds
    dt = np.gradient(t_hat, axis=0)
    # dt/ds, where ds is the arc-length increment per index step
    return np.linalg.norm(dt / ds, axis=1)  # type: ignore[no-any-return]


def displacement(points: FloatArray) -> float:
    r"""End-to-end displacement $\|\mathbf{r}_N - \mathbf{r}_0\|$.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 4.0, 0.0]])
    >>> displacement(pts)
    5.0
    """
    return float(np.linalg.norm(points[-1] - points[0]))


def closest_approach(points: FloatArray, target: Vector3) -> tuple[int, float]:
    """Find the point on the curve nearest to *target*.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.
    target : Vector3
        Reference point.

    Returns
    -------
    tuple[int, float]
        ``(index, distance)`` of the closest point.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> closest_approach(pts, (0.9, 0.0, 0.0))
    (1, 0.09999999999999998)
    """
    distances = np.linalg.norm(points - np.asarray(target), axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def plane_crossings(
    points: FloatArray,
    normal: Vector3,
    offset: float = 0.0,
) -> FloatArray:
    r"""Find positions where a curve crosses a plane.

    The plane is defined by $\mathbf{n} \cdot \mathbf{r} = d$ where
    $\mathbf{n}$ is the normal and $d$ is the offset. Crossing positions
    are linearly interpolated between consecutive points. Segments
    tangent to the plane (same signed distance at both endpoints)
    are not counted as crossings.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.
    normal : Vector3
        Plane normal vector (will be normalized internally).
    offset : float
        Signed distance from origin along the normal direction.

    Returns
    -------
    FloatArray
        Crossing positions, shape ``(M, 3)``. Empty ``(0, 3)`` if none.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> plane_crossings(pts, (1.0, 0.0, 0.0), 0.0)
    array([[0., 0., 0.]])
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    signed_dist = points @ n - offset
    crossings = []
    for i in range(len(signed_dist) - 1):
        d0, d1 = signed_dist[i], signed_dist[i + 1]
        if d0 * d1 < 0:
            t = d0 / (d0 - d1)
            crossings.append(points[i] + t * (points[i + 1] - points[i]))
    if not crossings:
        return np.empty((0, 3), dtype=points.dtype)
    return np.array(crossings)


def equatorial_crossings(points: FloatArray) -> FloatArray:
    r"""Find positions where a curve crosses the $z = 0$ plane.

    Shortcut for ``plane_crossings(points, (0, 0, 1), 0.0)``.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.

    Returns
    -------
    FloatArray
        Crossing positions, shape ``(M, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    >>> equatorial_crossings(pts)
    array([[0., 0., 0.]])
    """
    return plane_crossings(points, (0.0, 0.0, 1.0), 0.0)


def resample_by_arc_length(
    points: FloatArray,
    n_out: int,
    *,
    scalars: dict[str, FloatArray] | None = None,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Resample a curve to uniform arc-length spacing via linear interpolation.

    Parameters
    ----------
    points : FloatArray
        Ordered positions, shape ``(N, 3)``.
    n_out : int
        Number of output points (must be >= 2).
    scalars : dict[str, FloatArray] | None
        Optional scalar arrays to resample, each shape ``(N,)``.

    Returns
    -------
    tuple[FloatArray, dict[str, FloatArray]]
        ``(resampled_points, resampled_scalars)`` where ``resampled_points``
        has shape ``(n_out, 3)`` and each scalar has shape ``(n_out,)``.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    >>> new_pts, _ = resample_by_arc_length(pts, 5)
    >>> new_pts[:, 0]
    array([0., 1., 2., 3., 4.])
    """
    if n_out < 2:
        msg = f"n_out must be >= 2, got {n_out}"
        raise ValueError(msg)
    s = arc_length_cumulative(points)
    s_new = np.linspace(s[0], s[-1], n_out)
    resampled = np.column_stack([np.interp(s_new, s, points[:, i]) for i in range(3)])
    resampled_scalars: dict[str, FloatArray] = {}
    if scalars:
        for name, arr in scalars.items():
            resampled_scalars[name] = np.interp(s_new, s, arr)
    return resampled, resampled_scalars


def speed(velocity: FloatArray) -> FloatArray:
    r"""Speed (magnitude of velocity) at each point.

    Parameters
    ----------
    velocity : FloatArray
        Velocity vectors, shape ``(N, 3)``.

    Returns
    -------
    FloatArray
        Speed at each point, shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    >>> speed(v)
    array([5., 5.])
    """
    return np.linalg.norm(velocity, axis=1)  # type: ignore[no-any-return]


def kinetic_energy(velocity: FloatArray, mass: float) -> FloatArray:
    r"""Non-relativistic kinetic energy $\frac{1}{2} m v^2$ at each point.

    Parameters
    ----------
    velocity : FloatArray
        Velocity vectors, shape ``(N, 3)``.
    mass : float
        Particle mass in code units.

    Returns
    -------
    FloatArray
        Kinetic energy at each point, shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    >>> kinetic_energy(v, 2.0)
    array([1., 4.])
    """
    return 0.5 * mass * np.sum(velocity**2, axis=1)  # type: ignore[no-any-return]


def mirror_points(b_magnitude: FloatArray) -> NDArray[np.intp]:
    r"""Find mirror points (local maxima in $|\mathbf{B}|$) along a path.

    A mirror point is where a trapped particle reverses direction due to
    the magnetic mirror force, occurring at local maxima of the field
    magnitude along the particle's guiding-center path.

    Parameters
    ----------
    b_magnitude : FloatArray
        Magnetic field magnitude along the path, shape ``(N,)``.

    Returns
    -------
    NDArray[np.intp]
        Integer indices of local maxima. Empty if none found.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.array([1.0, 3.0, 1.0, 4.0, 2.0])
    >>> mirror_points(b)
    array([1, 3])
    """
    if len(b_magnitude) < 3:
        return np.array([], dtype=np.intp)
    left = b_magnitude[:-2]
    center = b_magnitude[1:-1]
    right = b_magnitude[2:]
    is_max = (center > left) & (center > right)
    return np.where(is_max)[0] + 1


def gyroradius_estimate(
    points: FloatArray,
    velocity: FloatArray,
    b_magnitude: FloatArray,
    charge: float,
    mass: float,
) -> FloatArray:
    r"""Estimate local gyroradius $r_g = m v_\perp / (|q| B)$ along a path.

    The perpendicular velocity is estimated by subtracting the field-aligned
    component using the local tangent direction as a proxy for the field
    direction.

    Parameters
    ----------
    points : FloatArray
        Positions along the path, shape ``(N, 3)``.
    velocity : FloatArray
        Velocity at each point, shape ``(N, 3)``.
    b_magnitude : FloatArray
        Magnetic field magnitude at each point, shape ``(N,)``.
    charge : float
        Particle charge (absolute value used).
    mass : float
        Particle mass.

    Returns
    -------
    FloatArray
        Estimated gyroradius at each point, shape ``(N,)``.
        NaN where $B = 0$.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> velocity = np.tile([0.0, 3.0, 0.0], (3, 1))
    >>> b = np.full(3, 2.0)
    >>> gyroradius_estimate(points, velocity, b, charge=1.0, mass=1.0)
    array([1.5, 1.5, 1.5])
    """
    t_hat = tangent_vectors(points)
    v_par = np.sum(velocity * t_hat, axis=1, keepdims=True) * t_hat
    v_perp_mag = np.linalg.norm(velocity - v_par, axis=1)
    abs_q = abs(charge)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = mass * v_perp_mag / (abs_q * b_magnitude)
    rg[b_magnitude == 0] = np.nan
    return rg  # type: ignore[no-any-return]


def drift_velocity(
    points: FloatArray,
    time: FloatArray,
    *,
    window: int = 5,
) -> FloatArray:
    r"""Running-average guiding-center drift velocity.

    Smooths the instantaneous velocity $d\mathbf{r}/dt$ with a uniform
    window to approximate the guiding-center drift, filtering out the
    gyromotion.

    Parameters
    ----------
    points : FloatArray
        Positions, shape ``(N, 3)``.
    time : FloatArray
        Time at each point, shape ``(N,)``.
    window : int
        Averaging window size (must be odd and >= 1).

    Returns
    -------
    FloatArray
        Smoothed velocity, shape ``(N, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.zeros((5, 3))
    >>> points[:, 0] = np.arange(5.0)
    >>> drift_velocity(points, np.arange(5.0), window=1)[0]
    array([1., 0., 0.])
    """
    if window < 1:
        msg = f"window must be >= 1, got {window}"
        raise ValueError(msg)
    dt = np.gradient(time)
    dt = np.maximum(dt, np.finfo(dt.dtype).tiny)
    v_inst = np.gradient(points, axis=0) / dt[:, np.newaxis]
    if window == 1:
        return v_inst  # type: ignore[no-any-return]
    kernel = np.ones(window) / window
    smoothed = np.column_stack(
        [np.convolve(v_inst[:, i], kernel, mode="same") for i in range(3)]
    )
    return smoothed
