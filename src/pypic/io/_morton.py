"""Morton (Z-order) curve encoding for 3D spatial sorting.

Pure NumPy implementation — no external dependencies.  Used by the
Parquet writer to sort particles so that Parquet row-group statistics
on x/y/z form tight spatial bounding boxes, enabling predicate pushdown
to skip 95%+ of row groups for spatial box queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _spread_bits_by3(v: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Spread the low 21 bits of *v* into every-third bit position.

    After spreading, bit *k* of the input occupies bit *3k* of the
    output.  Five rounds of shift-and-mask ("magic bits" technique).
    """
    v = v & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def _quantize(
    arr: NDArray[np.floating[Any]],
    bits: int,
) -> NDArray[np.uint64]:
    """Map floating-point values to [0, 2**bits - 1] unsigned integers."""
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if span == 0.0:
        return np.zeros(len(arr), dtype=np.uint64)
    max_val = np.uint64((1 << bits) - 1)
    scaled = (arr - lo) / span
    return np.minimum(
        (scaled * float(1 << bits)).astype(np.uint64),
        max_val,
    )


def morton_encode_3d(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    *,
    bits: int = 21,
) -> NDArray[np.uint64]:
    r"""Compute 3D Morton (Z-order) codes by bit-interleaving.

    Quantizes each coordinate to *bits*-bit unsigned integers (mapping
    the coordinate range $[\min, \max]$ to $[0, 2^{bits} - 1]$), then
    interleaves the bits of the three integer coordinates into a single
    64-bit code.

    Parameters
    ----------
    x, y, z : NDArray
        Coordinate arrays, each shape ``(N,)``.
    bits : int
        Bits per axis (max 21 for 63-bit result fitting uint64).

    Returns
    -------
    NDArray[np.uint64]
        Morton codes, shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> codes = morton_encode_3d(
    ...     np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]),
    ...     bits=2,
    ... )
    >>> codes
    array([ 0, 63], dtype=uint64)
    """
    if bits < 1 or bits > 21:
        msg = f"bits must be in [1, 21], got {bits}"
        raise ValueError(msg)
    xi = _quantize(x, bits)
    yi = _quantize(y, bits)
    zi = _quantize(z, bits)
    return (  # type: ignore[no-any-return]
        _spread_bits_by3(xi)
        | (_spread_bits_by3(yi) << np.uint64(1))
        | (_spread_bits_by3(zi) << np.uint64(2))
    )


def morton_sort_indices(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    *,
    bits: int = 21,
) -> NDArray[np.intp]:
    r"""Return permutation indices that sort particles by Morton code.

    Parameters
    ----------
    x, y, z : NDArray
        Coordinate arrays, each shape ``(N,)``.
    bits : int
        Bits per axis (max 21).

    Returns
    -------
    NDArray[np.intp]
        Permutation ``idx`` such that ``x[idx], y[idx], z[idx]`` are
        Morton-sorted.

    Examples
    --------
    >>> import numpy as np
    >>> idx = morton_sort_indices(
    ...     np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]),
    ... )
    >>> idx
    array([1, 0])
    """
    codes = morton_encode_3d(x, y, z, bits=bits)
    return np.argsort(codes)
