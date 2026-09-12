"""Conventions the test suite itself has to hold to.

These read the suite's own source rather than exercising pypic. Each one
guards a convention that had already drifted by the time it was written,
and that no linter expresses.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).parent
INVARIANTS_ROOT = TESTS_ROOT / "test_invariants"


def _called_name(call: ast.Call) -> str:
    """Return the bare function name of a call, however it was reached."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def test_every_assert_allclose_states_a_tolerance() -> None:
    """A bare assert_allclose silently runs at numpy's 1e-7 default.

    That is four orders looser than float64 work deserves, and against an
    expected 0.0 it asserts nothing at all — rtol scales the *desired*
    value, so only an exact zero can pass. Either the comparison is
    approximate, and the tolerance belongs in the call where a reader can
    weigh it, or it is exact, and ``assert_array_equal`` says so.

    Parsed rather than grepped: a multi-line call carries its ``rtol`` on
    a later line, which a line-oriented search reports as bare.
    """
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{path.relative_to(TESTS_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _called_name(node) == "assert_allclose"
            and not {kw.arg for kw in node.keywords} & {"rtol", "atol"}
        )
    assert not offenders, (
        "assert_allclose without rtol or atol — pass an explicit tolerance, "
        "or use assert_array_equal if the claim is exactness:\n  "
        + "\n  ".join(offenders)
    )


def test_every_invariant_module_cites_a_source_and_a_claim() -> None:
    """Invariant tests carry their provenance in a header, or they rot.

    ``# Source:`` says where the invariant is written down — a docs
    section, a formula, a line of pypic. ``# Claim:`` says what this
    module asserts and why it follows. Without both, a property test that
    quietly stops matching the documentation looks exactly like one that
    still does.
    """
    missing: list[str] = []
    for path in sorted(INVARIANTS_ROOT.glob("test_*.py")):
        header = path.read_text(encoding="utf-8")
        absent = [
            label
            for label in ("# Source:", "# Claim:")
            if not any(line.startswith(label) for line in header.splitlines()[:60])
        ]
        if absent:
            missing.append(f"{path.name}: missing {', '.join(absent)}")
    assert not missing, "invariant modules without a provenance header:\n  " + (
        "\n  ".join(missing)
    )
