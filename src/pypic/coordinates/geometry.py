"""Coordinate geometry definitions with metric scale factors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pypic.types import Numeric

type ScaleFactor = np.floating[Any] | NDArray[np.floating[Any]]
type ScaleFactors = tuple[ScaleFactor, ScaleFactor, ScaleFactor]


class GeometryType(StrEnum):
    r"""Coordinate geometry type.

    Values match the ``[coordinates] geometry`` key in SCHEMA.md.

    Examples
    --------
    >>> GeometryType.CARTESIAN
    <GeometryType.CARTESIAN: 'cartesian'>
    >>> GeometryType("spherical")
    <GeometryType.SPHERICAL: 'spherical'>
    """

    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"


@dataclass(frozen=True, slots=True)
class CoordinateGeometry:
    r"""Coordinate system with axis metadata and metric scale factors.

    Parameters
    ----------
    type : GeometryType
        The coordinate geometry type.
    axis_names : tuple[str, str, str]
        Human-readable axis labels, e.g. ``("x", "y", "z")``.
    axis_units : tuple[str, str, str]
        Dimension kind per axis: ``"length"`` or ``"angle"``.

    Examples
    --------
    >>> CARTESIAN.axis_names
    ('x', 'y', 'z')
    >>> SPHERICAL.axis_names
    ('r', 'θ', 'φ')
    >>> CYLINDRICAL.axis_units
    ('length', 'angle', 'length')
    """

    type: GeometryType
    axis_names: tuple[str, str, str]
    axis_units: tuple[str, str, str]

    def metric_factors(self, x1: Numeric, x2: Numeric, x3: Numeric) -> ScaleFactors:
        r"""Compute metric scale factors $(h_1, h_2, h_3)$ for this geometry.

        The line element is $ds^2 = h_1^2 dx_1^2 + h_2^2 dx_2^2 + h_3^2 dx_3^2$.

        Parameters
        ----------
        x1 : Numeric
            First coordinate: $x$ (Cartesian), $r$ (spherical/cylindrical).
        x2 : Numeric
            Second coordinate: $y$ (Cartesian), $θ$ (spherical), $φ$ (cylindrical).
        x3 : Numeric
            Third coordinate: $z$ (Cartesian), $φ$ (spherical), $z$ (cylindrical).

        Returns
        -------
        tuple[ScaleFactor, ScaleFactor, ScaleFactor]
            Scale factors $(h_1, h_2, h_3)$. Constant factors are returned as
            ``np.float64(1.0)`` which broadcasts with any array shape.

        Examples
        --------
        >>> CARTESIAN.metric_factors(1.0, 2.0, 3.0)
        (np.float64(1.0), np.float64(1.0), np.float64(1.0))

        >>> h1, h2, h3 = SPHERICAL.metric_factors(2.0, np.pi / 2, 0.0)
        >>> float(h1), float(h2), float(h3)
        (1.0, 2.0, 2.0)

        >>> h1, h2, h3 = CYLINDRICAL.metric_factors(3.0, 0.0, 1.0)
        >>> float(h1), float(h2), float(h3)
        (1.0, 3.0, 1.0)
        """
        _one = np.float64(1.0)
        match self.type:
            case GeometryType.CARTESIAN:
                return (_one, _one, _one)
            case GeometryType.SPHERICAL:
                r = np.asarray(x1, dtype=np.float64)
                theta = np.asarray(x2, dtype=np.float64)
                return (_one, r, r * np.sin(theta))
            case GeometryType.CYLINDRICAL:
                r = np.asarray(x1, dtype=np.float64)
                return (_one, r, _one)
            case _:
                raise ValueError(f"Unknown geometry type: {self.type}")


CARTESIAN = CoordinateGeometry(
    type=GeometryType.CARTESIAN,
    axis_names=("x", "y", "z"),
    axis_units=("length", "length", "length"),
)

SPHERICAL = CoordinateGeometry(
    type=GeometryType.SPHERICAL,
    axis_names=("r", "θ", "φ"),
    axis_units=("length", "angle", "angle"),
)

CYLINDRICAL = CoordinateGeometry(
    type=GeometryType.CYLINDRICAL,
    axis_names=("r", "φ", "z"),
    axis_units=("length", "angle", "length"),
)
