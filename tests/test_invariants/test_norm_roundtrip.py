# Source: docs/schema.md § 2 (Normalization must round-trip SI ↔ code).
# Extends tests/test_units.py:65 from {pic_electron, mhd_standard} to all
# four standard constructors × all six base quantities.
"""Hypothesis round-trip properties for ``Normalization.normalize`` / ``to_si``."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pypic.units import Normalization
from tests.strategies import (
    BASE_QUANTITIES,
    COMPOUND_QUANTITIES,
    base_quantities,
    finite_physical_floats,
    normalizations,
)


@given(norm=normalizations(), quantity=base_quantities(), x=finite_physical_floats())
@settings(max_examples=400, deadline=None)
def test_base_quantity_round_trip(norm: Normalization, quantity: str, x: float) -> None:
    """``to_si(q, normalize(q, x)) ≈ x`` for every base quantity and system."""
    round_tripped = norm.to_si(quantity, norm.normalize(quantity, x))
    assert math.isclose(round_tripped, x, rel_tol=1e-13, abs_tol=0.0)


@pytest.mark.parametrize(
    ("system", "norm"),
    [
        ("pic_standard", Normalization.pic_standard(1e18, 9.109e-31, 1.602e-19)),
        ("pic_electron", Normalization.pic_electron(1e18)),
        ("mhd_standard", Normalization.mhd_standard(1e6, 1e-12, 1e-9)),
        ("identity", Normalization.identity()),
    ],
)
def test_every_system_covers_every_base_quantity(
    system: str, norm: Normalization
) -> None:
    """Each constructor produces a Normalization that accepts all six base
    quantities without raising — guards against a constructor silently
    leaving a ``*_ref`` field unset. Uses deterministic fixed triples
    rather than Hypothesis strategies: this is an API-coverage check,
    not a property test.
    """
    del system  # used only for the test ID
    for quantity in BASE_QUANTITIES:
        norm.normalize(quantity, 1.0)
        norm.to_si(quantity, 1.0)


@given(
    norm=normalizations(),
    quantity=st.sampled_from(COMPOUND_QUANTITIES),
    x=finite_physical_floats(),
)
@settings(max_examples=400, deadline=None)
def test_compound_quantity_si_factor_round_trip(
    norm: Normalization, quantity: str, x: float
) -> None:
    """``x * si_factor / si_factor ≈ x`` — compound factors are positive
    finite, so division is always well-defined.
    """
    factor = norm.si_factor(quantity)
    assert math.isfinite(factor), f"si_factor({quantity!r}) not finite: {factor}"
    assert factor > 0.0, f"si_factor({quantity!r}) not positive: {factor}"
    round_tripped = (x * factor) / factor
    assert math.isclose(round_tripped, x, rel_tol=1e-13, abs_tol=0.0)
