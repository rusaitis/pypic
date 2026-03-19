"""Pytest configuration: ``--sim-data`` option for real-data smoke tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


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
