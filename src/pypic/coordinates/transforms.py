r"""Coordinate frame transforms: affine mappings between reference frames.

A `FrameTransform` describes how to convert coordinates and vector fields
from one named reference frame to another via an affine transformation:

$$\mathbf{x}_{target} = s \cdot R \cdot (\mathbf{x}_{source} - \mathbf{o})$$

where $\mathbf{o}$ is the origin offset (subtracted before rotation),
$R$ is the 3×3 rotation matrix, and $s$ is the uniform scale factor.
"""

from __future__ import annotations

__all__ = [
    "FrameTransform",
    "compose_transforms",
    "find_pressure_tensor_groups",
    "find_vector_triplets",
    "identity_transform",
    "resolve_transform",
    "rotate_pressure_tensor",
    "rotate_vector_components",
]

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pypic.types import FloatArray

_IDENTITY_3X3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

type Rotation3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _to_rotation(arr: np.ndarray) -> Rotation3x3:
    """Convert a (3, 3) array to an immutable nested tuple."""
    return (
        (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
        (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
    )


def _to_vec3(arr: np.ndarray) -> tuple[float, float, float]:
    """Convert a length-3 array to a float tuple."""
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True, slots=True)
class FrameTransform:
    r"""Affine transformation between coordinate reference frames.

    Transforms a point $\mathbf{x}$ from the source frame to the target:

    $$\mathbf{x}_{target} = s \cdot R \cdot (\mathbf{x}_{source} - \mathbf{o})$$

    Parameters
    ----------
    source_frame : str
        Name of the source frame (e.g. ``"simulation"``).
    target_frame : str
        Name of the target frame (e.g. ``"GSM"``).
    origin : tuple[float, float, float]
        Source-frame coordinates of the target-frame origin.
        Subtracted before rotation.
    rotation : Rotation3x3
        3×3 orthogonal rotation matrix as nested tuples.
    scale : float
        Converts code length units to target-frame units.
        ``scale=0.25`` with target in R_E means 1 d_i = 0.25 R_E.
        Default 1.0 (code and target use the same length unit).
        Auto-computed from ``physical_extent`` when available.
    target_axis_names : tuple[str, str, str] | None
        Axis names in the target frame. ``None`` keeps source names.

    Examples
    --------
    >>> t = FrameTransform("sim", "GSM", origin=(52.0, 26.0, 64.0))
    >>> t.source_frame
    'sim'
    >>> t.is_identity
    False
    >>> FrameTransform("a", "a").is_identity
    True
    """

    source_frame: str
    target_frame: str
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Rotation3x3 = _IDENTITY_3X3
    scale: float = 1.0
    target_axis_names: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if len(self.rotation) != 3 or any(len(row) != 3 for row in self.rotation):
            msg = f"rotation must be 3×3, got shape {len(self.rotation)}×..."
            raise ValueError(msg)
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        r = self.rotation_matrix
        err = np.max(np.abs(r.T @ r - np.eye(3)))
        if err > 1e-6:
            raise ValueError(
                f"rotation matrix is not orthogonal (max |R^T R - I| = {err:.2e})"
            )

    @property
    def rotation_matrix(self) -> FloatArray:
        """Return the rotation as a (3, 3) NumPy array.

        Examples
        --------
        >>> FrameTransform("a", "b").rotation_matrix.shape
        (3, 3)
        """
        return np.array(self.rotation, dtype=np.float64)

    @property
    def is_identity(self) -> bool:
        """True if this transform is a no-op.

        Examples
        --------
        >>> FrameTransform("a", "a").is_identity
        True
        >>> FrameTransform("a", "b", origin=(1.0, 0.0, 0.0)).is_identity
        False
        """
        return (
            self.origin == (0.0, 0.0, 0.0)
            and self.rotation == _IDENTITY_3X3
            and self.scale == 1.0
        )

    def inverse(self) -> FrameTransform:
        r"""Return the inverse transform (target → source).

        For $\mathbf{x}_t = s R (\mathbf{x}_s - \mathbf{o})$, the inverse
        is $\mathbf{x}_s = R^T \mathbf{x}_t / s + \mathbf{o}$.

        Examples
        --------
        >>> t = FrameTransform("a", "b", origin=(1.0, 2.0, 3.0), scale=2.0)
        >>> inv = t.inverse()
        >>> inv.source_frame, inv.target_frame
        ('b', 'a')
        """
        r = self.rotation_matrix
        o = np.array(self.origin, dtype=np.float64)
        # origin_inv = -s R o (from inverting the affine map)
        return FrameTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            origin=_to_vec3(-self.scale * r @ o),
            rotation=_to_rotation(r.T),
            scale=1.0 / self.scale,
            target_axis_names=None,
        )


def identity_transform(frame: str) -> FrameTransform:
    """Return a no-op transform within a single frame.

    Examples
    --------
    >>> identity_transform("sim").is_identity
    True
    """
    return FrameTransform(source_frame=frame, target_frame=frame)


def compose_transforms(first: FrameTransform, second: FrameTransform) -> FrameTransform:
    r"""Compose two transforms: apply *first* then *second*.

    If *first* maps A→B and *second* maps B→C, the result maps A→C.

    The composition follows from:

    $$\mathbf{x}_C = s_2 R_2 (s_1 R_1 (\mathbf{x}_A - \mathbf{o}_1) - \mathbf{o}_2)$$

    Parameters
    ----------
    first : FrameTransform
        Transform applied first (A→B).
    second : FrameTransform
        Transform applied second (B→C).

    Returns
    -------
    FrameTransform
        Composed transform (A→C).

    Raises
    ------
    ValueError
        If ``first.target_frame != second.source_frame``.

    Examples
    --------
    >>> a_to_b = FrameTransform("A", "B", origin=(1.0, 0.0, 0.0))
    >>> b_to_c = FrameTransform("B", "C", origin=(0.0, 2.0, 0.0))
    >>> a_to_c = compose_transforms(a_to_b, b_to_c)
    >>> a_to_c.source_frame, a_to_c.target_frame
    ('A', 'C')
    """
    if first.target_frame != second.source_frame:
        msg = (
            f"Cannot compose: first maps to {first.target_frame!r} "
            f"but second starts from {second.source_frame!r}"
        )
        raise ValueError(msg)

    r1 = first.rotation_matrix
    r2 = second.rotation_matrix
    o1 = np.array(first.origin, dtype=np.float64)
    o2 = np.array(second.origin, dtype=np.float64)

    # x_C = s2 R2 (s1 R1 (x_A - o1) - o2)
    #      = s2 s1 R2 R1 (x_A - o1) - s2 R2 o2
    #      = s_c R_c (x_A - o_c)
    # where R_c = R2 R1, s_c = s1 s2
    # s_c R_c (x_A - o_c) = s_c R_c x_A - s_c R_c o_c
    # must equal: s_c R_c x_A - s_c R_c o1 - s2 R2 o2
    # => o_c = o1 + (1/s1) R1^T o2
    combined_rotation = r2 @ r1
    combined_scale = first.scale * second.scale
    combined_origin = o1 + (1.0 / first.scale) * (r1.T @ o2)

    return FrameTransform(
        source_frame=first.source_frame,
        target_frame=second.target_frame,
        origin=_to_vec3(combined_origin),
        rotation=_to_rotation(combined_rotation),
        scale=combined_scale,
        target_axis_names=second.target_axis_names or first.target_axis_names,
    )


def resolve_transform(
    source_frame: str,
    target_frame: str,
    transforms: dict[str, FrameTransform],
) -> FrameTransform:
    """Find or compose a transform from *source_frame* to *target_frame*.

    Tries direct lookup first, then searches for a one-hop chain
    (A→B→C) through intermediate frames. Also checks inverse
    transforms (if A→B is registered, B→A is available via inverse).

    Parameters
    ----------
    source_frame : str
        Current frame name.
    target_frame : str
        Desired frame name.
    transforms : dict[str, FrameTransform]
        Registry mapping target frame names to transforms.

    Returns
    -------
    FrameTransform

    Raises
    ------
    ValueError
        If no transform path exists.

    Examples
    --------
    >>> t = FrameTransform("sim", "GSM", origin=(1.0, 0.0, 0.0))
    >>> resolve_transform("sim", "GSM", {"GSM": t}).target_frame
    'GSM'
    """
    if source_frame == target_frame:
        return identity_transform(source_frame)

    # Build a full map of all available directed edges
    edges: dict[tuple[str, str], FrameTransform] = {}
    for t in transforms.values():
        edges[(t.source_frame, t.target_frame)] = t
        edges[(t.target_frame, t.source_frame)] = t.inverse()

    # Direct lookup
    if (source_frame, target_frame) in edges:
        return edges[(source_frame, target_frame)]

    # One-hop chain: source → intermediate → target
    for (src, mid), first in list(edges.items()):
        if src != source_frame:
            continue
        if (mid, target_frame) in edges:
            return compose_transforms(first, edges[(mid, target_frame)])

    available = sorted(
        {t.source_frame for t in transforms.values()}
        | {t.target_frame for t in transforms.values()}
    )
    msg = (
        f"No transform path from {source_frame!r} to {target_frame!r}. "
        f"Available frames: {available}"
    )
    raise ValueError(msg)


def _build_vector_triplet_regex() -> re.Pattern[str]:
    """Build regex from the canonical vector prefixes in readers.base.

    Tier-3 canonical: ``<prefix>[_s<N>]_<component>`` — every semantic
    boundary is an underscore, with optional species qualifier between
    prefix and component (``B_1``, ``V_s0_1``, ``B0_1``, ``B0_s0_1``).
    """
    from pypic.grid import _FIELD_PREFIX_PAIRS

    prefixes = sorted(
        {canon for _, canon in _FIELD_PREFIX_PAIRS},
        key=len,
        reverse=True,
    )
    parts = [re.escape(p) for p in prefixes]
    return re.compile(
        rf"^({'|'.join(parts)})(?:_s(\d+))?_([123])$"
    )


_VECTOR_TRIPLET_RE = _build_vector_triplet_regex()


def find_vector_triplets(
    field_names: Iterable[str],
) -> list[tuple[str, str, str]]:
    """Group field names into vector triplets needing rotation.

    Returns a list of ``(name1, name2, name3)`` tuples for each
    complete vector field. Handles both total fields (``B_1, B_2, B_3``)
    and per-species fields (``J_s0_1, J_s0_2, J_s0_3``).

    Parameters
    ----------
    field_names : Iterable[str]
        All field names in the dataset.

    Returns
    -------
    list[tuple[str, str, str]]
        Complete vector triplets.

    Examples
    --------
    >>> find_vector_triplets(["B_1", "B_2", "B_3", "rho_c"])
    [('B_1', 'B_2', 'B_3')]
    >>> find_vector_triplets(["J_s0_1", "J_s0_2", "J_s0_3", "J_s1_1"])
    [('J_s0_1', 'J_s0_2', 'J_s0_3')]
    """
    groups: dict[str, dict[int, str]] = {}
    for name in field_names:
        m = _VECTOR_TRIPLET_RE.match(name)
        if m:
            prefix, species, component = m.groups()
            key = f"{prefix}_s{species}" if species else prefix
            groups.setdefault(key, {})[int(component)] = name
    return [(g[1], g[2], g[3]) for g in groups.values() if 1 in g and 2 in g and 3 in g]


# Tier-3 pressure tensor: ``P_<ij>`` (bare) or ``P_s<N>_<ij>`` (per-species).
_PRESSURE_RE = re.compile(r"^P(?:_s(\d+))?_(\d)(\d)$")


def find_pressure_tensor_groups(
    field_names: Iterable[str],
) -> list[tuple[str, str, str, str, str, str]]:
    """Group pressure tensor fields into complete symmetric tensors.

    Returns ``(P_11, P_22, P_33, P_12, P_13, P_23)`` tuples for each complete
    set. Handles per-species tensors (``P_s0_11`` etc.).

    Parameters
    ----------
    field_names : Iterable[str]
        All field names in the dataset.

    Returns
    -------
    list[tuple[str, str, str, str, str, str]]

    Examples
    --------
    >>> fields = ["P_11", "P_22", "P_33", "P_12", "P_13", "P_23"]
    >>> find_pressure_tensor_groups(fields)
    [('P_11', 'P_22', 'P_33', 'P_12', 'P_13', 'P_23')]
    """
    groups: dict[str, dict[str, str]] = {}
    for name in field_names:
        m = _PRESSURE_RE.match(name)
        if m:
            species, i, j = m.groups()
            key = f"P_s{species}" if species else "P"
            groups.setdefault(key, {})[f"{i}{j}"] = name
    required = {"11", "22", "33", "12", "13", "23"}
    return [
        (g["11"], g["22"], g["33"], g["12"], g["13"], g["23"])
        for g in groups.values()
        if required <= set(g)
    ]


def rotate_vector_components(
    v1: FloatArray,
    v2: FloatArray,
    v3: FloatArray,
    rotation: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Rotate vector field components by a 3×3 rotation matrix.

    $$v'_i = \sum_j R_{ij} \, v_j$$

    Parameters
    ----------
    v1, v2, v3 : FloatArray
        Vector field components (arbitrary shape, must match).
    rotation : FloatArray
        Shape ``(3, 3)`` rotation matrix.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]

    Examples
    --------
    >>> import numpy as np
    >>> v1, v2, v3 = np.array([1.0]), np.array([0.0]), np.array([0.0])
    >>> R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    >>> r1, r2, r3 = rotate_vector_components(v1, v2, v3, R)
    >>> float(r1[0]), float(r2[0]), float(r3[0])
    (0.0, 1.0, 0.0)
    """
    r = rotation
    v1_new = r[0, 0] * v1 + r[0, 1] * v2 + r[0, 2] * v3
    v2_new = r[1, 0] * v1 + r[1, 1] * v2 + r[1, 2] * v3
    v3_new = r[2, 0] * v1 + r[2, 1] * v2 + r[2, 2] * v3
    return v1_new, v2_new, v3_new


def rotate_pressure_tensor(
    p11: FloatArray,
    p22: FloatArray,
    p33: FloatArray,
    p12: FloatArray,
    p13: FloatArray,
    p23: FloatArray,
    rotation: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Rotate a symmetric pressure tensor by a rotation matrix.

    $$P'_{ij} = \sum_{k,l} R_{ik} \, R_{jl} \, P_{kl}$$

    The trace $P_{11} + P_{22} + P_{33}$ is invariant.

    Parameters
    ----------
    p11, p22, p33, p12, p13, p23 : FloatArray
        Independent components of the symmetric tensor.
    rotation : FloatArray
        Shape ``(3, 3)`` rotation matrix.

    Returns
    -------
    tuple[FloatArray, ...]
        ``(P_11', P_22', P_33', P_12', P_13', P_23')`` in the rotated frame.

    Examples
    --------
    >>> import numpy as np
    >>> I = np.eye(3)
    >>> p = rotate_pressure_tensor(
    ...     np.array([1.0]), np.array([2.0]), np.array([3.0]),
    ...     np.array([0.0]), np.array([0.0]), np.array([0.0]), I,
    ... )
    >>> [float(x[0]) for x in p]
    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    """
    r = rotation
    # Build the full 3×3 symmetric tensor per grid point, then rotate.
    # P'_ij = sum_kl R_ik R_jl P_kl
    # Expand all 9 components (P is symmetric: P_21=P_12, P_31=P_13, P_32=P_23)
    p = [[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]]

    def _component(i: int, j: int) -> FloatArray:
        result = r[i, 0] * r[j, 0] * p[0][0]
        for k in range(3):
            for el in range(3):
                if k == 0 and el == 0:
                    continue
                result = result + r[i, k] * r[j, el] * p[k][el]
        return cast("FloatArray", result)

    return (
        _component(0, 0),
        _component(1, 1),
        _component(2, 2),
        _component(0, 1),
        _component(0, 2),
        _component(1, 2),
    )
