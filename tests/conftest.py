"""Pytest configuration: ``--sim-data`` option for real-data smoke tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

# Typer force-enables rich colour when GITHUB_ACTIONS is set, regardless of
# NO_COLOR or TERM. Colour makes CLI error text unassertable: rich highlights
# option names, so "Invalid --plane" renders as "Invalid " plus three
# separately-escaped fragments and a plain substring check fails. Tests that
# read human-facing output must not depend on the terminal's colour support.
# Set before typer.rich_utils is imported — it reads this at module scope.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"


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
