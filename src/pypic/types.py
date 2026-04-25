"""Shared type aliases for the pypic package."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type Numeric = float | np.floating[Any] | NDArray[np.floating[Any]]
type Vector3 = tuple[float, float, float]
type FloatArray = NDArray[np.floating[Any]]
type ModelType = Literal["PIC", "MHD", "hybrid"]
