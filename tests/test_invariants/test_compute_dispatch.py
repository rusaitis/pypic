# Source: docs/schema.md § 3 ("Other derived quantities" table — magnitudes)
#         + docs/equations.md § 6 ("Magnitudes and Differential Operators").
# Claim: compute("|X|")² == X1² + X2² + X3² for every registered magnitude.
# Covers compute-dispatch parity for the five magnitude recipes in _REGISTRY.
"""Algebraic identity for the magnitude recipes in ``compute._REGISTRY``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypic.compute import _REGISTRY
from tests._helpers import make_test_dataset

# Static-registry magnitudes (totals; no species). Per-species
# magnitudes (``|V_s0|``, ``|V_s1|``, ...) are template-synthesized
# and live outside this list. ``|Ve|`` collapsed into an alias for
# ``|V_s0|`` at Stage E — it's no longer in the static registry.
#
# Two recipe shapes:
#  * DIRECT — ``derived.*_magnitude`` over an explicit ``(X_1, X_2, X_3)``
#    triple; the registry directly declares the three squared inputs.
#  * PYTHAGOREAN — ``derived.perpendicular_magnitude`` over
#    ``(X_1, X_2, X_3, B_1, B_2, B_3)``, computing $\sqrt{|X|^2 - X_\parallel^2}$.
#    Coverage lives in ``tests/test_compute.py``
#    (``TestFieldAlignedPythagoreanIdentity``), which exercises the same
#    algebraic identity through the recipe's actual inputs.
DIRECT_MAGNITUDE_RECIPES: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("|B|", ("B_1", "B_2", "B_3")),
    ("|E|", ("E_1", "E_2", "E_3")),
    ("|J|", ("J_1", "J_2", "J_3")),
    ("|V|", ("V_1", "V_2", "V_3")),
    ("|vort|", ("vort_1", "vort_2", "vort_3")),
)
PYTHAGOREAN_MAGNITUDE_NAMES: frozenset[str] = frozenset(
    {
        "|J_perp|",
        "|V_perp|",
        "|E_perp|",
        "|E_prime_perp|",
        "|E_ideal_perp|",
        "|E_Hall_perp|",
    }
)


def test_registry_magnitudes_match_expected_set() -> None:
    """The hand-curated set above matches the registry — guards against
    silent registry drift (new magnitudes added without test coverage).
    """
    registry_mags = {
        name for name in _REGISTRY if name.startswith("|") and name.endswith("|")
    }
    expected = {
        name for name, _ in DIRECT_MAGNITUDE_RECIPES
    } | PYTHAGOREAN_MAGNITUDE_NAMES
    assert registry_mags == expected, (
        f"magnitude registry drifted: registry={registry_mags}, expected={expected}"
    )


SHAPE = (3, 4, 2)


def _finite_array() -> st.SearchStrategy[np.ndarray]:
    """Random float64 arrays of fixed shape, bounded so squared sums stay finite."""
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@pytest.mark.parametrize(
    ("name", "components"),
    [pytest.param(n, c, id=n) for n, c in DIRECT_MAGNITUDE_RECIPES],
)
@given(x1=_finite_array(), x2=_finite_array(), x3=_finite_array())
@settings(max_examples=50, deadline=None)
def test_magnitude_algebraic_identity(
    name: str,
    components: tuple[str, str, str],
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
) -> None:
    """``compute(name)² == sum(components²)`` to float64 precision.

    The magnitude recipe dispatches through ``_execute_recipe`` →
    ``derived.*_magnitude``; this asserts that the dispatch picks up the
    correct three components declared in the registry.
    """
    ds = make_test_dataset(
        {components[0]: x1, components[1]: x2, components[2]: x3},
        shape=SHAPE,
    )
    mag = ds.compute(name)
    expected_sq = x1 * x1 + x2 * x2 + x3 * x3
    np.testing.assert_allclose(mag * mag, expected_sq, rtol=1e-13, atol=1e-300)
