"""Tests that example scripts remain runnable.

The examples are the zero-data on-ramp — the only pypic code a new user
can run before they have simulation output — so a broken one is a broken
front door. Both are self-contained: they generate their own inputs.
"""

from __future__ import annotations

import runpy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_custom_reader_example() -> None:
    """The reader walkthrough runs end to end."""
    from examples.custom_reader_example import main

    main()


def test_schindler_example(tmp_path: Path, monkeypatch) -> None:
    """The Schindler figure example runs and writes exactly one figure.

    Redirecting the output also pins that the script honours
    ``PYPIC_EXAMPLE_OUTPUT_DIR`` rather than writing to the working
    directory, which used to drop a 76 KB PNG in the repository root on
    every documented invocation.
    """
    from pypic.plotting import get_theme, set_theme

    monkeypatch.setenv("PYPIC_EXAMPLE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    # The script calls set_theme() at module scope, which is global state.
    # Restore it so running this test cannot change what later tests see.
    original_theme = get_theme()
    try:
        runpy.run_path("examples/ex_schindler_xi.py", run_name="__not_main__")
    finally:
        set_theme(original_theme)

    assert [p.name for p in tmp_path.iterdir()] == ["ex_schindler_xi.png"]
