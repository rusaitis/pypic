"""Pytest configuration and fixtures shared across test modules."""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import settings

from pypic import (
    CARTESIAN,
    CoordinateGeometry,
    FieldDataset,
    GeometryType,
    GridInfo,
    Normalization,
)

# Typer force-enables rich colour when GITHUB_ACTIONS is set, regardless of
# NO_COLOR or TERM. Colour makes CLI error text unassertable: rich highlights
# option names, so "Invalid --plane" renders as "Invalid " plus three
# separately-escaped fragments and a plain substring check fails. Tests that
# read human-facing output must not depend on the terminal's colour support.
# Set before typer.rich_utils is imported — it reads this at module scope.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

# Every property test in tests/test_invariants/ needs deadline=None: the
# first example of a NumPy-heavy test pays import and JIT warmup that the
# rest do not, so a per-example deadline flakes on a loaded machine. A
# profile says it once instead of 83 times.
settings.register_profile("pypic", deadline=None)
settings.load_profile("pypic")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--sim-data`` command-line option."""
    parser.addoption(
        "--sim-data",
        nargs="?",
        const="examples",
        default=None,
        help=(
            "Path to simulation data directory. "
            "Without argument: scan examples/. "
            "With path: scan that directory."
        ),
    )


@pytest.fixture
def cartesian_3d() -> FieldDataset:
    """8x6x4 Cartesian dataset with B_1, B_2, B_3.

    Shared by the selection and reduction suites so composition tests
    (``BoxSelection`` then ``reduce``) line up on one shape convention.
    """
    grid = GridInfo(
        dimensions=(8, 6, 4),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    rng = np.random.default_rng(42)
    fields = {
        "B_1": rng.standard_normal((8, 6, 4)),
        "B_2": rng.standard_normal((8, 6, 4)),
        "B_3": rng.standard_normal((8, 6, 4)),
    }
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


@pytest.fixture
def spherical_3d() -> FieldDataset:
    """4x3x2 spherical dataset with B_1."""
    geom = CoordinateGeometry(
        type=GeometryType.SPHERICAL,
        axis_names=("r", "θ", "φ"),
        axis_units=("length", "angle", "angle"),
    )
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(0.5, 0.1, 0.2),
        origin=(1.0, 0.0, 0.0),
        geometry=geom,
    )
    fields = {"B_1": np.arange(24, dtype=float).reshape(4, 3, 2)}
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())
