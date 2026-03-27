r"""Reconnection analysis: X-point detection and reconnection rate.

Pure functions for 2D magnetic reconnection diagnostics. Operates on
the magnetic flux function $\psi$ computed by
:func:`~pypic.derived.magnetic_flux_function`.

.. note::

   These functions assume **Cartesian geometry** with uniform grid spacing.
   Non-Cartesian coordinates would require metric-factor corrections in the
   Hessian computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.types import FloatArray


def find_saddle_points(
    psi: FloatArray,
    dx: float,
    dy: float,
    *,
    min_separation: int = 3,
) -> list[tuple[int, int]]:
    r"""Find saddle points (X-points) in a 2D flux function.

    A saddle point satisfies $\det(H) < 0$ where $H$ is the Hessian:

    $$\det(H) = \frac{\partial^2 \psi}{\partial x^2}
    \frac{\partial^2 \psi}{\partial y^2}
    - \left(\frac{\partial^2 \psi}{\partial x \partial y}\right)^2$$

    Second derivatives are computed via second-order central finite
    differences. Boundary points (2 cells from edges) are excluded
    because one-sided stencils reduce accuracy there.

    Parameters
    ----------
    psi : FloatArray
        2D flux function (from ``magnetic_flux_function``).
    dx : float
        Grid spacing along axis 0.
    dy : float
        Grid spacing along axis 1.
    min_separation : int
        Minimum grid-point separation between detected saddle points.
        Suppresses duplicate detections from the same physical X-point.

    Returns
    -------
    list[tuple[int, int]]
        Grid indices ``(i, j)`` of saddle points, sorted by
        $|\det(H)|$ descending (strongest first).

    Raises
    ------
    ValueError
        If *psi* is not 2D or grid is too small.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 101)
    >>> xx, yy = np.meshgrid(x, x, indexing="ij")
    >>> psi = xx**2 - yy**2
    >>> saddles = find_saddle_points(psi, x[1] - x[0], x[1] - x[0])
    >>> len(saddles) >= 1
    True
    >>> abs(saddles[0][0] - 50) <= 2
    True
    """
    if psi.ndim != 2:
        msg = f"find_saddle_points requires 2D input, got {psi.ndim}D"
        raise ValueError(msg)
    nx, ny = psi.shape
    if nx < 5 or ny < 5:
        msg = f"Grid too small for saddle detection: {psi.shape}"
        raise ValueError(msg)

    from scipy.ndimage import minimum_filter

    # First derivatives (gradient of psi)
    psi_x = np.gradient(psi, dx, axis=0)
    psi_y = np.gradient(psi, dy, axis=1)

    # Second derivatives via central differences
    psi_xx = np.gradient(psi_x, dx, axis=0)
    psi_yy = np.gradient(psi_y, dy, axis=1)
    psi_xy = np.gradient(psi_x, dy, axis=1)

    # Hessian determinant: negative at saddle points
    det_h = psi_xx * psi_yy - psi_xy**2

    # Gradient magnitude: critical points have |grad psi| ~ 0
    grad_mag = np.sqrt(psi_x**2 + psi_y**2)

    # Score: saddle points have det_H < 0 AND small gradient
    # Use grad_mag as a secondary criterion to locate the true X-point
    # within a region of uniform det_H
    saddle_mask = det_h < 0

    # Exclude boundary (2 cells from edges where stencils are one-sided)
    margin = 2
    interior = np.zeros_like(det_h, dtype=bool)
    interior[margin : nx - margin, margin : ny - margin] = True
    saddle_mask &= interior

    if not np.any(saddle_mask):
        return []

    # Find local minima of grad_mag within saddle regions (true critical points)
    size = 2 * min_separation + 1
    local_min = minimum_filter(grad_mag, size=size)
    is_local_min = grad_mag == local_min

    candidates = saddle_mask & is_local_min

    indices = np.argwhere(candidates)
    if len(indices) == 0:
        return []

    # Sort by gradient magnitude ascending (closest to true critical point)
    grad_at_candidates = grad_mag[candidates]
    order = np.argsort(grad_at_candidates)
    return [(int(indices[k, 0]), int(indices[k, 1])) for k in order]


def reconnection_rate(
    psi: FloatArray,
    psi_prev: FloatArray,
    dt: float,
    x_point: tuple[int, int],
) -> float:
    r"""Compute the reconnection rate at an X-point.

    $$R = \frac{\partial\psi}{\partial t}\bigg|_X \approx
    \frac{\psi(t) - \psi(t - \Delta t)}{\Delta t}$$

    Equal to the out-of-plane electric field $E_z$ at the X-point.
    [Biskamp 2000] §3.1.

    Parameters
    ----------
    psi : FloatArray
        Flux function at current time.
    psi_prev : FloatArray
        Flux function at previous time.
    dt : float
        Time interval between snapshots.
    x_point : tuple[int, int]
        Grid indices ``(i, j)`` of the X-point.

    Returns
    -------
    float
        Reconnection rate (time derivative of $\psi$ at the X-point).

    Examples
    --------
    >>> import numpy as np
    >>> psi = np.zeros((10, 10))
    >>> psi_prev = np.ones((10, 10))
    >>> reconnection_rate(psi, psi_prev, dt=0.5, x_point=(5, 5))
    -2.0
    """
    return float((psi[x_point] - psi_prev[x_point]) / dt)


__all__ = [
    "find_saddle_points",
    "reconnection_rate",
]
