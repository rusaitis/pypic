"""Tests that example scripts remain runnable.

The examples are the zero-data on-ramp — the only pypic code a new user
can run before they have simulation output — so a broken one is a broken
front door. Each is self-contained: it generates its own inputs and
asserts its own results, so running it *is* the test.

They rot silently otherwise: four of these were hidden from the repo by
an over-broad ``.gitignore`` rule and had drifted onto pre-``B_1`` field
names and the pre-v1.0 schema by the time anyone looked.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# The numbered on-ramp plus the standalone snippets, in reading order.
# Every one prints to stdout and exits non-zero on a failed assertion.
_SELF_CHECKING_SCRIPTS = [
    "ex1_minimal_fields.py",
    "ex2_simple_hdf5.py",
    "ex3_field_mapping.py",
    "ex4_toml_config.py",
    "advanced_calculations.py",
]


@pytest.mark.parametrize("script", _SELF_CHECKING_SCRIPTS)
def test_self_checking_example_runs(script: str) -> None:
    """Each self-contained example runs green.

    ``run_name`` is deliberately not ``"__main__"``: these scripts do
    their work at module scope, so importing them under any name
    executes and self-checks them.
    """
    runpy.run_path(str(_EXAMPLES / script), run_name="__not_main__")


def test_every_committed_example_is_covered() -> None:
    """No example script escapes this module.

    A new example added to ``examples/`` without a line here would go
    untested — which is exactly how the numbered series rotted.
    """
    on_disk = {p.name for p in _EXAMPLES.glob("*.py")}
    covered = set(_SELF_CHECKING_SCRIPTS) | {
        "custom_reader_example.py",  # test_custom_reader_example
        "ex_schindler_xi.py",  # test_schindler_example
    }
    assert on_disk == covered, (
        f"examples/ and tests/test_examples.py disagree. Only on disk: "
        f"{sorted(on_disk - covered)}. Only in the test: "
        f"{sorted(covered - on_disk)}."
    )


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
        runpy.run_path(str(_EXAMPLES / "ex_schindler_xi.py"), run_name="__not_main__")
    finally:
        set_theme(original_theme)

    assert [p.name for p in tmp_path.iterdir()] == ["ex_schindler_xi.png"]
