"""Fixtures for the plotting suite; skips the whole directory without matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pypic.containers import TabularData
from pypic.dataset import FieldDataset
from pypic.units import Normalization
from tests._helpers import make_uniform_grid

if TYPE_CHECKING:
    from pypic.grid import GridInfo

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture
def grid_2d() -> GridInfo:
    return make_uniform_grid(10, 8)


@pytest.fixture
def grid_3d() -> GridInfo:
    return make_uniform_grid(10, 8, 6)


@pytest.fixture
def ds_2d(grid_2d: GridInfo) -> FieldDataset:
    rng = np.random.default_rng(42)
    return FieldDataset.from_arrays(
        {
            "B_1": rng.standard_normal((10, 8)),
            "B_2": rng.standard_normal((10, 8)),
            "B_3": rng.standard_normal((10, 8)),
            "rho_m": np.abs(rng.standard_normal((10, 8))) + 0.1,
            "P": np.abs(rng.standard_normal((10, 8))) + 0.1,
        },
        grid_2d,
        Normalization.identity(),
    )


@pytest.fixture
def ds_3d(grid_3d: GridInfo) -> FieldDataset:
    rng = np.random.default_rng(42)
    return FieldDataset.from_arrays(
        {
            "B_1": rng.standard_normal((10, 8, 6)),
            "B_2": rng.standard_normal((10, 8, 6)),
            "B_3": rng.standard_normal((10, 8, 6)),
            "rho_m": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
            "P": np.abs(rng.standard_normal((10, 8, 6))) + 0.1,
        },
        grid_3d,
        Normalization.identity(),
    )


@pytest.fixture
def ds_1d() -> FieldDataset:
    rng = np.random.default_rng(99)
    return FieldDataset.from_arrays(
        {"B_1": rng.standard_normal(20)},
        make_uniform_grid(20, spacing=0.5),
        Normalization.identity(),
    )


@pytest.fixture
def tabular() -> TabularData:
    cycles = np.arange(50, dtype=np.float64)
    return TabularData(
        name="diagnostics",
        columns={
            "cycle": cycles,
            "total_energy": np.exp(-cycles / 20.0),
            "kinetic_energy": 0.5 * np.exp(-cycles / 20.0),
        },
        index_column="cycle",
    )
