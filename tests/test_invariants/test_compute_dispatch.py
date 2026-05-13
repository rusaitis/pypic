# Source: docs/schema.md § 3 ("Other derived quantities" table — magnitudes)
#         + docs/equations.md § 6 ("Magnitudes and Differential Operators").
# Claim: compute("|X|")² == X1² + X2² + X3² for every registered magnitude.
# Covers sub-item 2a of the autoresearcher-pypic.md backlog (compute dispatch
# parity, split by recipe category): the five magnitude recipes in _REGISTRY.
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
# Field-aligned-decomposition magnitudes (``|J_perp|``, ``|V_perp|``,
# ``|E_perp|``, ``|E_prime_perp|``) use ``velocity_magnitude`` over the
# three perpendicular components, so the same algebraic identity holds.
MAGNITUDE_RECIPES: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("|B|", ("B_1", "B_2", "B_3")),
    ("|E|", ("E_1", "E_2", "E_3")),
    ("|J|", ("J_1", "J_2", "J_3")),
    ("|V|", ("V_1", "V_2", "V_3")),
    ("|vort|", ("vort_1", "vort_2", "vort_3")),
    ("|J_perp|", ("J_perp_1", "J_perp_2", "J_perp_3")),
    ("|V_perp|", ("V_perp_1", "V_perp_2", "V_perp_3")),
    ("|E_perp|", ("E_perp_1", "E_perp_2", "E_perp_3")),
    ("|E_prime_perp|", ("E_prime_perp_1", "E_prime_perp_2", "E_prime_perp_3")),
    ("|E_ideal_perp|", ("E_ideal_perp_1", "E_ideal_perp_2", "E_ideal_perp_3")),
    ("|E_Hall_perp|", ("E_Hall_perp_1", "E_Hall_perp_2", "E_Hall_perp_3")),
)


def test_registry_magnitudes_match_expected_set() -> None:
    """The hand-curated set above matches the registry — guards against
    silent registry drift (new magnitudes added without test coverage).
    """
    registry_mags = {
        name for name in _REGISTRY if name.startswith("|") and name.endswith("|")
    }
    expected = {name for name, _ in MAGNITUDE_RECIPES}
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
    [pytest.param(n, c, id=n) for n, c in MAGNITUDE_RECIPES],
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
