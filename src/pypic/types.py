"""Shared type aliases for the pypic package."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type Numeric = float | np.floating[Any] | NDArray[np.floating[Any]]
type Vector3 = tuple[float, float, float]
