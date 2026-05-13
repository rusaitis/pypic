r"""X-point detection, 2D reconnection rate, and 3D Schindler integral.

Pure functions for 2D X-point diagnostics built on the flux function
$\psi$ (``find_saddle_points``, ``reconnection_rate``), and a 3D
general-magnetic-reconnection criterion (``schindler_xi``) that
integrates $E_\parallel$ along magnetic field lines.

.. note::

   The 2D functions assume **Cartesian geometry** with uniform grid
   spacing. Non-Cartesian coordinates would require metric-factor
   corrections in the Hessian computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypic.dataset import FieldDataset
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
        gradient magnitude ascending (closest to true critical point first).

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


def schindler_xi(
    data: FieldDataset,
    seeds: FloatArray,
    *,
    step_size: float = 0.5,
    max_steps: int = 10_000,
    direction: str = "both",
    method: str = "linear",
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    e_components: tuple[str, str, str] = ("E_1", "E_2", "E_3"),
) -> FloatArray:
    r"""Compute the Schindler $\Xi$ integral of $E_\parallel$ along $\mathbf{B}$.

    $$\Xi(\mathbf{x}_0) = \int_{\mathcal{L}(\mathbf{x}_0)}
    E_\parallel \, d\ell$$

    The 3D general-magnetic-reconnection criterion: $\Xi \neq 0$ along
    a magnetic field line $\mathcal{L}$ defines reconnection, without
    requiring a 3D null. Vanishes identically in ideal MHD (where
    $\mathbf{E} = -\mathbf{V}\times\mathbf{B}$ is purely perpendicular
    to $\mathbf{B}$) and is non-zero only where the non-ideal terms in
    Ohm's law produce a parallel electric field (Schindler, Hesse, Birn,
    J. Geophys. Res. 93, 5547, 1988).

    For each seed point, a magnetic field line is traced using the
    classical RK4 integrator (``trace_field_line``), $\mathbf{E}$ and
    $\hat{b}$ are interpolated along the trace, and $E_\parallel
    = \mathbf{E}\cdot\hat{b}$ is integrated by the trapezoidal rule over
    arc length. Termination follows the tracer's conventions (null hit,
    domain exit, ``max_steps``).

    Parameters
    ----------
    data : FieldDataset
        Gridded $\mathbf{B}$ and $\mathbf{E}$ fields.
    seeds : FloatArray
        Seed positions, shape ``(M, 3)`` for ``M`` seeds or ``(3,)``
        for a single seed (returned as a length-1 array).
    step_size : float
        Arc-length step for the RK4 field-line integrator.
    max_steps : int
        Maximum integration steps per direction.
    direction : str
        ``"both"`` (default), ``"forward"``, or ``"backward"``.
    method : str
        Sampling interpolation: ``"linear"`` (default) or ``"nearest"``.
    field_components, e_components : tuple of str
        Canonical names of the $\mathbf{B}$ and $\mathbf{E}$ components
        in *data*. The defaults match pypic's Tier-3 naming.

    Returns
    -------
    FloatArray
        $\Xi$ for each seed, shape ``(M,)``. Seeds whose trace fails
        (out-of-domain, at-null) return NaN. Trace points that sample
        outside the domain ($|B| = 0$, NaN E) contribute zero to the
        integral — the diagnostic integrates over the valid portion of
        the field line.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> from pypic.units import Normalization
    >>> # Uniform B along x with a uniform parallel E: Xi = E * (path length).
    >>> shape = (32, 8, 8)
    >>> data = FieldDataset.from_arrays(
    ...     {
    ...         "B_1": np.ones(shape),
    ...         "B_2": np.zeros(shape),
    ...         "B_3": np.zeros(shape),
    ...         "E_1": 0.05 * np.ones(shape),
    ...         "E_2": np.zeros(shape),
    ...         "E_3": np.zeros(shape),
    ...     },
    ...     GridInfo(dimensions=shape, spacing=(1.0, 1.0, 1.0)),
    ...     Normalization.identity(),
    ... )
    >>> xi = schindler_xi(
    ...     data,
    ...     np.array([[15.0, 4.0, 4.0]]),
    ...     step_size=0.5,
    ...     max_steps=200,
    ... )
    >>> # E_par = 0.05; path length ≈ 2*100*0.5 + 2*16*0.5 ≈ ?
    >>> # With max_steps=200 in each direction, the trace integrates
    >>> # up to the domain edges. Just check it's positive and finite.
    >>> bool(xi[0] > 0 and np.isfinite(xi[0]))
    True
    """
    from pypic.traces import (
        VectorFieldInterpolator,
        sample_fields,
        trace_field_line,
    )

    seeds_arr = np.asarray(seeds, dtype=np.float64)
    if seeds_arr.ndim == 1:
        seeds_arr = seeds_arr[np.newaxis, :]
    if seeds_arr.ndim != 2 or seeds_arr.shape[1] != 3:
        msg = f"seeds must have shape (M, 3) or (3,), got {np.asarray(seeds).shape}"
        raise ValueError(msg)

    result = np.full(seeds_arr.shape[0], np.nan, dtype=np.float64)
    sample_keys = (*field_components, *e_components)
    # Build the B interpolator once and reuse across seeds.
    interpolator = VectorFieldInterpolator.from_dataset(data, field_components)

    for k in range(seeds_arr.shape[0]):
        seed = (
            float(seeds_arr[k, 0]),
            float(seeds_arr[k, 1]),
            float(seeds_arr[k, 2]),
        )
        try:
            field_line = trace_field_line(
                data,
                seed,
                step_size=step_size,
                max_steps=max_steps,
                direction=direction,
                field_components=field_components,
                interpolator=interpolator,
            )
        except ValueError:
            # Out of domain or at a null — leave NaN.
            continue

        points = field_line.points
        if points.shape[0] < 2:
            continue

        sampled = sample_fields(data, points, list(sample_keys), method=method)
        b1, b2, b3 = (sampled[k] for k in field_components)
        e1, e2, e3 = (sampled[k] for k in e_components)
        b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
        # E_parallel = E · b̂ — NaN where |B|=0 or any sampled value is NaN
        # (out-of-domain).  Treat those segments as having zero
        # contribution: the diagnostic is the integral over the valid
        # portion of the trace.
        e_par_raw = (e1 * b1 + e2 * b2 + e3 * b3) / np.where(b_mag > 0, b_mag, np.nan)
        e_par = np.where(np.isfinite(e_par_raw), e_par_raw, 0.0)

        deltas = np.diff(points, axis=0)
        d_ell = np.sqrt(np.sum(deltas**2, axis=1))
        result[k] = float(np.sum(0.5 * (e_par[:-1] + e_par[1:]) * d_ell))

    return result


__all__ = [
    "find_saddle_points",
    "reconnection_rate",
    "schindler_xi",
]
