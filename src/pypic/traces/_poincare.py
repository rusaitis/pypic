r"""Poincaré surface-of-section diagnostic for field-line topology.

Given an adaptive trace of a 3D vector field, project each line onto a
fixed transverse surface $\Sigma$ and record the puncture points. The
resulting 2D scatter pattern makes topology visually obvious — closed
curves are islands / O-points, dense 1D fills are KAM surfaces, blobs
are chaotic regions. Standard tool for fusion poloidal sections and
magnetotail X-line geometry — the FLARE 3D boundary code uses the
same construction for stellarator / divertor footprint analysis
[@Frerichs2024].

This module is a thin orchestrator on top of two existing primitives:
:func:`pypic.traces.trace_field_lines_adaptive` (integration) and
:func:`pypic.traces.plane_crossings` (sign-change + linear interp on
the plane). Punctures are extracted post-hoc from full traces;
``loop_tol=None`` is forced so closed orbits don't self-terminate
before they can be sampled.
"""

from __future__ import annotations

__all__ = [
    "PoincareSection",
    "PoincareSurface",
    "poincare_section",
]

from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from pypic.traces._analysis import plane_crossings
from pypic.traces._tracing import (
    VectorFieldInterpolator,
    trace_field_lines_adaptive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pypic.dataset import FieldDataset
    from pypic.traces._fieldline import FieldLine
    from pypic.types import FloatArray, Vector3


_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


# slots=False: cached_property writes through __dict__, which slotted
# dataclasses don't expose. Switching to slots would require a manual
# cache slot per cached attribute — not worth the line-count tradeoff
# for a result dataclass with three cached lookups.
@dataclass(frozen=True)
class PoincareSurface:
    r"""Transverse plane $\Sigma$ for a Poincaré section.

    $\Sigma = \{\mathbf{x} : \hat{\mathbf{n}} \cdot
    (\mathbf{x} - \mathbf{p}) = 0\}$, parameterized by an outward
    normal $\mathbf{n}$ and an in-plane reference point $\mathbf{p}$.
    The 2D plane coordinates $(u, v)$ produced by :meth:`project` are
    measured relative to $\mathbf{p}$ in an orthonormal basis
    :attr:`basis_2d` spanning $\Sigma$.

    Parameters
    ----------
    normal : Vector3
        Plane normal. Normalized internally; need not be unit length.
    point : Vector3
        A point on the plane (origin of the 2D $(u, v)$ frame).
    name : str | None
        Optional label propagated into output metadata and plots.

    Examples
    --------
    >>> surf = PoincareSurface.from_axis("y", 0.0, name="meridional")
    >>> surf.offset
    0.0
    >>> import numpy as np
    >>> u, v = surf.basis_2d
    >>> float(np.dot(u, v))
    0.0
    """

    normal: Vector3
    point: Vector3
    name: str | None = None

    @classmethod
    def from_axis(
        cls,
        axis: Literal["x", "y", "z"],
        value: float,
        *,
        name: str | None = None,
    ) -> Self:
        r"""Build an axis-aligned surface $x_i = \text{value}$.

        Covers tokamak poloidal sections (``from_axis("y", 0.0)`` for a
        $\phi = 0$ cut, with the toroidal direction along $\hat y$) and
        the standard magnetotail meridional plane
        (``from_axis("y", 0.0)`` in GSM).

        Examples
        --------
        >>> surf = PoincareSurface.from_axis("z", 1.5)
        >>> surf.normal
        (0.0, 0.0, 1.0)
        >>> surf.point
        (0.0, 0.0, 1.5)
        """
        if axis not in _AXIS_INDEX:
            msg = f"axis must be one of {sorted(_AXIS_INDEX)}, got {axis!r}"
            raise ValueError(msg)
        i = _AXIS_INDEX[axis]
        normal = [0.0, 0.0, 0.0]
        normal[i] = 1.0
        point = [0.0, 0.0, 0.0]
        point[i] = float(value)
        return cls(
            normal=(normal[0], normal[1], normal[2]),
            point=(point[0], point[1], point[2]),
            name=name,
        )

    @cached_property
    def _normal_arr(self) -> FloatArray:
        n = np.asarray(self.normal, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm == 0.0:
            msg = "PoincareSurface.normal must be non-zero"
            raise ValueError(msg)
        return n / norm

    @cached_property
    def _point_arr(self) -> FloatArray:
        return np.asarray(self.point, dtype=np.float64)

    @cached_property
    def offset(self) -> float:
        r"""Signed scalar $d = \hat{\mathbf{n}} \cdot \mathbf{p}$.

        The form consumed by :func:`pypic.traces.plane_crossings`.
        """
        return float(self._normal_arr @ self._point_arr)

    @cached_property
    def basis_2d(self) -> tuple[FloatArray, FloatArray]:
        r"""Orthonormal $(\hat{\mathbf{u}}, \hat{\mathbf{v}})$ spanning $\Sigma$.

        Constructed via Gram--Schmidt against the world axis least
        aligned with $\hat{\mathbf{n}}$ for numerical stability — the
        standard oblique-section basis shared by FLARE
        [@Frerichs2024] and most field-mapping tools.
        """
        n = self._normal_arr
        # Pick the world axis least parallel to n
        axis = int(np.argmin(np.abs(n)))
        seed = np.zeros(3, dtype=np.float64)
        seed[axis] = 1.0
        u = np.cross(n, seed)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        # cross of two unit-orthogonal vectors is already unit-length;
        # renormalize defensively against accumulated FP error.
        v = v / np.linalg.norm(v)
        return u, v

    def project(self, points_3d: FloatArray) -> FloatArray:
        r"""Project 3D points onto plane coordinates $(u, v)$.

        For each input point $\mathbf{x}$:
        $u = \hat{\mathbf{u}} \cdot (\mathbf{x} - \mathbf{p})$,
        $v = \hat{\mathbf{v}} \cdot (\mathbf{x} - \mathbf{p})$.

        Parameters
        ----------
        points_3d : FloatArray
            Input points, shape ``(M, 3)``. Empty ``(0, 3)`` allowed.

        Returns
        -------
        FloatArray
            Plane coordinates, shape ``(M, 2)``.

        Examples
        --------
        Plane coordinates are oriented by the Gram--Schmidt basis; for the
        z-axis surface the basis is $\hat{\mathbf{u}} = (0, 1, 0)$,
        $\hat{\mathbf{v}} = (-1, 0, 0)$ (right-handed about $\hat{\mathbf{n}}$):

        >>> import numpy as np
        >>> surf = PoincareSurface.from_axis("z", 0.0)
        >>> pts = np.array([[1.0, 2.0, 0.0], [-1.0, 3.0, 0.0]])
        >>> surf.project(pts)
        array([[ 2., -1.],
               [ 3.,  1.]])
        """
        if points_3d.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        u, v = self.basis_2d
        rel = points_3d - self._point_arr
        return np.column_stack([rel @ u, rel @ v])


# slots=False: same reason as PoincareSurface — cached_property needs
# the instance __dict__.
@dataclass(frozen=True)
class PoincareSection:
    r"""Result of :func:`poincare_section`: per-seed punctures and provenance.

    Holds both the 3D crossing positions (for re-projection onto a
    different surface) and the projected 2D coordinates (for plotting).
    The underlying :class:`FieldLine` traces are retained so the same
    trajectories can be re-punctured against a different surface without
    re-integrating.

    Parameters
    ----------
    surface : PoincareSurface
        The plane the punctures were extracted on.
    seeds : FloatArray
        Original seed positions, shape ``(M, 3)``.
    direction : str
        ``"forward"``, ``"backward"``, or ``"both"`` — propagated from
        the trace call.
    punctures_3d : tuple[FloatArray, ...]
        Per-seed crossings in 3D, each shape ``(n_k, 3)``.
    punctures_2d : tuple[FloatArray, ...]
        Per-seed crossings in plane coordinates, each ``(n_k, 2)``.
    field_lines : tuple[FieldLine, ...]
        Underlying trace per seed.
    metadata : dict[str, Any]
        Provenance: ``n_crossings_per_seed``, ``n_steps_per_seed``,
        ``termination_reasons``, ``surface_name``.

    Examples
    --------
    >>> import numpy as np
    >>> from pypic.units import Normalization
    >>> from pypic.dataset import FieldDataset
    >>> from pypic.grid import GridInfo
    >>> grid = GridInfo(dimensions=(16, 16, 16), spacing=(0.2, 0.2, 0.2),
    ...                 origin=(-1.5, -1.5, -1.5))
    >>> x = np.linspace(-1.5, 1.5, 16)
    >>> y = np.linspace(-1.5, 1.5, 16)
    >>> z = np.linspace(-1.5, 1.5, 16)
    >>> X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    >>> data = FieldDataset.from_arrays(
    ...     {"B_1": -Y, "B_2": X, "B_3": 0.1 * np.ones_like(X)},
    ...     grid, Normalization.identity(),
    ... )
    >>> surf = PoincareSurface.from_axis("y", 0.0)
    >>> section = poincare_section(
    ...     data, np.array([[0.6, 0.0, 0.0]]), surf, max_steps=2000,
    ...     direction="forward",
    ... )
    >>> bool(section.metadata["n_crossings_per_seed"][0] > 1)
    True
    """

    surface: PoincareSurface
    seeds: FloatArray
    direction: str
    punctures_3d: tuple[FloatArray, ...]
    punctures_2d: tuple[FloatArray, ...]
    field_lines: tuple[FieldLine, ...]
    # Read-only after __post_init__ rewraps to MappingProxyType. Accept
    # a plain dict on construction for callers' convenience.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @cached_property
    def n_seeds(self) -> int:
        """Number of seeds in this section."""
        return int(self.seeds.shape[0])

    @cached_property
    def all_punctures_2d(self) -> FloatArray:
        r"""Concatenated $(\sum_k n_k, 2)$ cloud across all seeds."""
        if not self.punctures_2d:
            return np.empty((0, 2), dtype=np.float64)
        return np.concatenate(self.punctures_2d, axis=0)

    @cached_property
    def all_punctures_3d(self) -> FloatArray:
        r"""Concatenated $(\sum_k n_k, 3)$ cloud across all seeds."""
        if not self.punctures_3d:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(self.punctures_3d, axis=0)


def poincare_section(
    data: FieldDataset,
    seeds: FloatArray | Sequence[Vector3],
    surface: PoincareSurface,
    *,
    direction: Literal["forward", "backward", "both"] = "forward",
    max_steps: int = 50_000,
    field_components: tuple[str, str, str] = ("B_1", "B_2", "B_3"),
    atol: float = 1e-6,
    rtol: float = 1e-4,
    step_size_init: float = 0.5,
    min_step: float = 1e-8,
    max_step: float = 2.0,
    null_threshold: float = 1e-12,
    interpolator: VectorFieldInterpolator | None = None,
) -> PoincareSection:
    r"""Build a Poincaré section by tracing seeds and puncturing on $\Sigma$.

    Each seed is integrated forward (and/or backward) with the adaptive
    Dormand-Prince tracer; every accepted-step segment that crosses
    $\Sigma$ contributes one puncture, located via linear interpolation
    on the signed-distance field $\mathbf{n}\cdot\mathbf{r} - d$.

    Internally forces ``loop_tol=None`` on the tracer — the auto
    closed-loop detector would otherwise terminate the very orbits we
    want to sample on the second puncture.

    Parameters
    ----------
    data : FieldDataset
        Gridded vector field data.
    seeds : FloatArray or Sequence[Vector3]
        Seed positions, shape ``(M, 3)`` (or any iterable that
        ``np.asarray(..., dtype=float64).reshape(-1, 3)`` accepts).
    surface : PoincareSurface
        Transverse plane to puncture on.
    direction : ``"forward"`` | ``"backward"`` | ``"both"``
        Trace direction. Forward only is typical for tokamak/stellarator
        cuts; ``"both"`` doubles puncture density for fixed ``max_steps``.
    max_steps : int
        Per-direction step budget for the adaptive tracer. Memory cost
        scales as ``M * max_steps * 3 * 8`` bytes; 50_000 × 100 seeds
        is ~120 MB.
    field_components : tuple[str, str, str]
        Field component names. Default ``("B_1", "B_2", "B_3")``.
    atol, rtol : float
        Adaptive error tolerances.
    step_size_init, min_step, max_step : float
        Step-size controller knobs.
    null_threshold : float
        Field-magnitude threshold below which a point is a null.
    interpolator : VectorFieldInterpolator | None
        Pre-built interpolator; constructed internally if ``None``.

    Returns
    -------
    PoincareSection

    Raises
    ------
    ValueError
        Propagated from :func:`trace_field_lines_adaptive` (bad seed,
        invalid direction, bad tolerances).

    Notes
    -----
    For fusion poloidal sections, set ``direction="forward"`` and pick
    ``max_steps`` to span ~$N$ poloidal transits per seed; ``N = 100``
    is typical for a quick island survey. For magnetotail X-line
    geometry, ``direction="both"`` captures the separatrix from both
    inflow regions.

    Examples
    --------
    See :class:`PoincareSection` for a closed-circle example.
    """
    seeds_arr = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)

    field_lines = trace_field_lines_adaptive(
        data,
        seeds_arr,
        atol=atol,
        rtol=rtol,
        step_size_init=step_size_init,
        min_step=min_step,
        max_step=max_step,
        max_steps=max_steps,
        direction=direction,
        field_components=field_components,
        null_threshold=null_threshold,
        loop_tol=None,
        interpolator=interpolator,
    )

    offset = surface.offset

    punctures_3d_list: list[FloatArray] = []
    punctures_2d_list: list[FloatArray] = []
    for fl in field_lines:
        pts3 = plane_crossings(fl.points, surface.normal, offset)
        punctures_3d_list.append(pts3)
        punctures_2d_list.append(surface.project(pts3))

    return PoincareSection(
        surface=surface,
        seeds=seeds_arr,
        direction=direction,
        punctures_3d=tuple(punctures_3d_list),
        punctures_2d=tuple(punctures_2d_list),
        field_lines=tuple(field_lines),
        metadata={
            "n_crossings_per_seed": np.array(
                [p.shape[0] for p in punctures_3d_list], dtype=np.intp
            ),
            "n_steps_per_seed": np.array(
                [fl.metadata.get("n_steps", fl.n_points - 1) for fl in field_lines],
                dtype=np.intp,
            ),
            "termination_reasons": tuple(
                fl.metadata.get("reason", "") for fl in field_lines
            ),
            "surface_name": surface.name,
        },
    )
